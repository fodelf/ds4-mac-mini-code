#!/bin/bash
# smoke-mtp-replica.sh — 【另一台】(8 GiB MacBook)上的 MTP drafter (ds4-mtp-replica)。
#
# 仿 smoke-decode-fast.sh: 小 ctx / 低内存 env / Metal only。
# 唯一区别: 不用 exec, 改为【后台启动 + 内存看门狗】—— replica 进程常驻 RSS 一旦
# 超过 8 GiB 立刻 SIGKILL, 绝不让这台 8G 机爆内存。
#
# 角色: 这台只跑 MTP drafter, listen 等本机 host(smoke-mtp-host.sh)连入。draft 只用
# base 的 token_embd + output head(不碰 43 层 routed experts), 配 view-shrink 常驻 ~5 GiB。
# 若 view-shrink 回归去 wire 整块 86 GiB base, 看门狗会在 8 GiB 处把它杀掉。
#
# Usage (在【另一台】上, 在放 ds4-mtp-replica + GGUF 的目录里):
#   ./smoke-mtp-replica.sh [port]
#   BASE=./ds4flash.gguf MTP=./ds4flash-mtp.gguf ./smoke-mtp-replica.sh 17502
#   CEIL_MIB=7168 ./smoke-mtp-replica.sh   # 想给 8G 机多留 margin, 把上限调到 7 GiB
#
# 绝不在 macOS 上跑 CPU 推理(会把内核搞崩); 仅 Metal。
set -u

BASE="${BASE:-./ds4flash.gguf}"                                # 完整 base(只用 embd+output)
MTP="${MTP:-./gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf}"   # MTP GGUF
PORT="${1:-17502}"
BIND="${BIND:-0.0.0.0}"
CTX="${DS4_CTX:-2048}"
CEIL_MIB="${CEIL_MIB:-8192}"                                   # 这台 8 GiB 机硬上限 → 超即杀

BIN=./ds4-mtp-replica
if [ ! -x "$BIN" ]; then echo "smoke-mtp-replica: $BIN 未构建, 先 make ds4-mtp-replica" >&2; exit 1; fi
if [ ! -f "$BASE" ]; then echo "smoke-mtp-replica: base 不存在 $BASE" >&2; exit 1; fi
if [ ! -f "$MTP" ];  then echo "smoke-mtp-replica: mtp 不存在 $MTP"  >&2; exit 1; fi

# 低内存 env(参考 smoke-decode-fast.sh)。view cap 跟 host 一样用【小】(2GiB):
#   - IOGPU 绑定一个 model view 时 wire 整个 MTLBuffer(不是只 wire 访问到的张量范围)。
#     所以 4GiB view 一绑就 wire 4GiB; replica draft 要 touch embd-view + output-view,
#     ~8GiB wired → 顶到 M1 Pro GPU recommendedMaxWorkingSet(~10.6G)/swap thrash(实测飙)。
#   - 2GiB view → 每个绑定 view 只 wire 2GiB, embd+output ~4GiB, 和 host 同档安全。
#   - 2GiB cap 在 81G base 上算 ~80 view; 需 ds4_metal.m:250 DS4_METAL_MAX_MODEL_VIEWS=128
#     (FIX A; 旧 64 上限会 "needs more mapped views" abort)→ replica 二进制必须是 rebuild 后的。
#   - NO_MODEL_WARMUP / NO_PREFILL_KERNEL_WARMUP: 防 warmup 去 touch 43 层 experts 把整块 base wire 进来。
export DS4_METAL_EXPERT_OFFLOAD=1
export DS4_METAL_MODEL_MAX_VIEW_BYTES="${VIEW_CAP:-2147483648}"   # 2 GiB (和 host 一致; 原因见上)
export DS4_METAL_NO_RESIDENCY=1
export DS4_METAL_NO_MODEL_WARMUP=1
export DS4_METAL_NO_PREFILL_KERNEL_WARMUP=1
export DS4_METAL_PREFILL_SPLIT=1

LOG=/tmp/smoke-mtp-replica.log; : > "$LOG"
echo "smoke-mtp-replica: base=$BASE mtp=$MTP port=$PORT ctx=$CTX  内存看门狗 ceiling=${CEIL_MIB} MiB"
echo "smoke-mtp-replica: 等本机 host 连入(看到 'listening on' 即就绪)"
echo

# 后台启动 replica(常驻 listen), 输出落 LOG; tail 到终端实时看加载/连接。
"$BIN" listen "$PORT" "$BIND" -m "$BASE" --mtp "$MTP" -c "$CTX" > "$LOG" 2>&1 &
pid=$!
tail -f "$LOG" & tailpid=$!

cleanup() { kill "$tailpid" 2>/dev/null; kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null; }
trap cleanup EXIT INT TERM

# 内存看门狗: 每 0.3s 看 RSS(KiB→MiB); 超 CEIL_MIB 立刻 SIGKILL, 抢在 swap thrash / panic 前。
while kill -0 "$pid" 2>/dev/null; do
    rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -z "$rss" ] && break
    mib=$(( rss / 1024 ))
    if [ "$mib" -gt "$CEIL_MIB" ]; then
        echo; echo "!! 内存看门狗: replica RSS ${mib} MiB > ${CEIL_MIB} MiB — SIGKILL (防这台 8G 机爆内存)"
        kill -9 "$pid" 2>/dev/null; kill "$tailpid" 2>/dev/null
        exit 2
    fi
    sleep 0.3
done

kill "$tailpid" 2>/dev/null
echo; echo "smoke-mtp-replica: replica 退出 (峰值未超 ${CEIL_MIB} MiB)"
