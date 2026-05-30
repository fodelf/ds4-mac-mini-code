#!/bin/bash
# smoke-mtp-host.sh — 本机(16 GiB Mac Mini M4)跑 target + 远程 MTP drafter 冒烟。
#
# 仿 smoke-decode-fast.sh: 小 ctx / 短 prompt / 跳 warmup / 低内存 env / Metal only。
# 唯一区别: 不用 exec, 改为【后台启动 + 内存看门狗】—— target 进程常驻 RSS 一旦
# 超过 12 GiB 立刻 SIGKILL, 绝不让本机爆内存(之前爆过两次)。
#
# 角色: 本机是 target。它通过 --mtp-remote 把"提议 token"这步交给【另一台】上
# 跑的 ds4-mtp-replica(见 smoke-mtp-replica.sh), target 自己只做验证。
#
# Usage (在本机):
#   ./smoke-mtp-host.sh [tokens] [prompt]
#   REMOTE=192.168.1.2:17502 ./smoke-mtp-host.sh 16 "1+1="
#   DS4_MODEL=./ds4flash.gguf CEIL_MIB=12288 ./smoke-mtp-host.sh   # 想跑完整 base, 看门狗兜底
#
# 绝不在 macOS 上跑 86 GiB base 的 CPU 推理(会把内核搞崩); 仅 Metal。
set -u

MODEL="${DS4_MODEL:-./ds4flash.gguf}"            # 完整 base(模板证明配 A3 expert offload 在 16G 不挂)
TOKENS="${1:-16}"
PROMPT="${2:-1+1=}"
CTX="${DS4_CTX:-2048}"
REMOTE="${REMOTE:-192.168.1.2:17502}"            # 另一台 replica 的 host:port
DRAFT="${DRAFT:-2}"                              # >1 才触发 speculative(配 --temp 0)
CEIL_MIB="${CEIL_MIB:-12288}"                    # 本机内存硬上限 12 GiB → 超即杀

if [ ! -x ./ds4 ]; then echo "smoke-mtp-host: ./ds4 未构建, 先 make" >&2; exit 1; fi
if [ ! -f "$MODEL" ]; then echo "smoke-mtp-host: 模型不存在 $MODEL" >&2; exit 1; fi

# 低内存 env。关键纠正(2026-05-29, 被用户点醒):
#   - 之前满模型能跑、内存不高、就是慢 —— 用的是【默认】view cap(设备 maxBufferLength ~8.88GiB)
#     → 81G base 只切 ~11 个 view(<64 上限), map 成功; + EXPERT_OFFLOAD 让 routed-expert 走
#     resident scratch 不 wire model view, 所以内存不高。
#   - 我之前照搬 smoke-decode-fast 塞了 DS4_METAL_MODEL_MAX_VIEW_BYTES=2GiB → step=cap-最大张量
#     ≈1GiB → 81G 切成 ~80 个 view > DS4_METAL_MAX_MODEL_VIEWS=64 → "needs more mapped views"
#     启动失败。这是【配置】不是代码(view 逻辑是 initial release 原样, git 已核实)。
#     小 view cap 是早期还没 expert offload 时压 wire 的老方案, 现在多余且有害 → 默认【不设】。
#   - 想试小 view 压每层 wire 峰值可 VIEW_CAP=3221225472(3GiB)等, 但别低于 ~2.3GiB(否则 view>64)。
export DS4_METAL_EXPERT_OFFLOAD=1
[ -n "${VIEW_CAP:-}" ] && export DS4_METAL_MODEL_MAX_VIEW_BYTES="$VIEW_CAP"
export DS4_METAL_NO_RESIDENCY=1
export DS4_METAL_NO_MODEL_WARMUP=1
export DS4_METAL_NO_PREFILL_KERNEL_WARMUP=1
export DS4_METAL_PREFILL_SPLIT=1

LOG=/tmp/smoke-mtp-host.log; : > "$LOG"
echo "smoke-mtp-host: model=$MODEL ctx=$CTX tokens=$TOKENS prompt=\"$PROMPT\""
echo "smoke-mtp-host: remote-mtp=$REMOTE draft=$DRAFT  内存看门狗 ceiling=${CEIL_MIB} MiB"
echo

# 后台启动 target(连远程 MTP), 输出落 LOG; tail 到终端实时看生成。
./ds4 -m "$MODEL" -c "$CTX" -n "$TOKENS" --temp 0 \
      --mtp-remote "$REMOTE" --mtp-draft "$DRAFT" -p "$PROMPT" > "$LOG" 2>&1 &
pid=$!
tail -f "$LOG" & tailpid=$!

cleanup() { kill "$tailpid" 2>/dev/null; kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null; }
trap cleanup EXIT INT TERM

# 内存看门狗: 每 0.5s 看 RSS; 超 CEIL_MIB 立刻 SIGKILL(抢在 swap thrash/panic 前); 每 ~3s 打一次 RSS 进度便于实时盯。
wd_i=0
while kill -0 "$pid" 2>/dev/null; do
    rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -z "$rss" ] && break
    mib=$(( rss / 1024 ))
    wd_i=$((wd_i+1)); [ $((wd_i % 6)) -eq 1 ] && echo "[watchdog] target RSS ${mib} MiB / ${CEIL_MIB} MiB"
    if [ "$mib" -gt "$CEIL_MIB" ]; then
        echo; echo "!! 内存看门狗: target RSS ${mib} MiB > ${CEIL_MIB} MiB — SIGKILL (防本机爆内存)"
        kill -9 "$pid" 2>/dev/null; kill "$tailpid" 2>/dev/null
        exit 2
    fi
    sleep 0.5
done

kill "$tailpid" 2>/dev/null
wait "$pid" 2>/dev/null; rc=$?
echo; echo "smoke-mtp-host: target 退出 rc=$rc (峰值未超 ${CEIL_MIB} MiB)"
exit "$rc"
