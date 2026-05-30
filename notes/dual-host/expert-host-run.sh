#!/bin/bash
# expert-host-run.sh — 本机(16 GiB Mac Mini M4)跑 target + 远程 routed-expert 冷层。
#
# 仿【已验证安全的 smoke-mtp-host.sh】低内存模板(默认 view cap + EXPERT_OFFLOAD 让
# routed 走 resident scratch 不 wire model view + 关 residency/warmup + prefill split),
# 把 --mtp-remote 换成 --expert-remote, 并加一个【小】expert cache(让 cache-miss 走
# 远程拉取回调)。
#
# !! 必须配 DS4_DENSE_RESIDENT=1 !! —— expert-remote 设计上要钉 dense(见 ds4_cli.c:97
#    帮助文本 "pair with DS4_DENSE_RESIDENT=1")。把 dense(~7.21G)memcpy 进真 RAM 池,
#    routed 从远程拉不在本地 fault; 缺它则全 base mmap 随机 fault 把 wired 怼到 ~10.9G
#    (= 首跑爆内存真因)。开 DENSE_RESIDENT 后预算 @ctx2048: ds4 RSS ~9.2-9.5 GiB,
#    看门狗 ceiling 收到 10.5 GiB 给真余量, 抢在 thrash 前。
#
# !! 本脚本在本机加载模型。先确认从机 replica 已 listening, 再【经用户授权】运行。!!
#
# Usage (在本机):
#   ./notes/dual-host/expert-host-run.sh <从机IP:端口> [tokens] [prompt]
#   REMOTE=10.0.0.2:17600 ./notes/dual-host/expert-host-run.sh
#
# 绝不在 macOS 上跑 CPU 推理(会把内核搞崩); 仅 Metal。
set -u

MODEL="${DS4_MODEL:-./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf}"
REMOTE="${REMOTE:-${1:-}}"
TOKENS="${2:-16}"
PROMPT="${3:-1+1=}"
CTX="${DS4_CTX:-2048}"
CEIL_MIB="${CEIL_MIB:-10752}"                    # 本机硬上限 10.5 GiB → 超即 SIGKILL(抢在 thrash 前,
                                                 # 正常峰 ~9.2-9.5G 留 ~1G 头)
EXPERT_CACHE="${EXPERT_CACHE:-268435456}"        # 256 MiB 锁定值(首测取小, 不要加大)
                                                 # 这是 budget-bounded 真分配, 实测 = slots×per_slot
                                                 # 叠在已验证安全的 smoke-mtp-host 基线上, 远低于 12G

if [ -z "$REMOTE" ]; then echo "用法: $0 <从机IP:端口> [tokens] [prompt]" >&2; exit 1; fi
if [ ! -x ./ds4 ]; then echo "expert-host-run: ./ds4 未构建, 先 make" >&2; exit 1; fi
if [ ! -f "$MODEL" ]; then echo "expert-host-run: 模型不存在 $MODEL" >&2; exit 1; fi

# precheck: 机器干净 + 无竞争进程(否则叠加爆内存)
SW0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}')
if [ "$SW0" -gt 300 ]; then echo "ABORT: swap_used=${SW0} MiB > 300, 先重启清干净"; exit 1; fi
if pgrep -f 'ds4 -m' >/dev/null; then echo "ABORT: 已有 ds4 推理进程在跑"; exit 1; fi

# 低内存 env —— 与 smoke-mtp-host.sh 安全模板一致(view cap 默认不设; 想压每层 wire 峰值
# 可 VIEW_CAP=3221225472(3GiB)等, 但别低于 ~2.3GiB 否则 view>64 启动失败)。
export DS4_METAL_EXPERT_OFFLOAD=1
[ -n "${VIEW_CAP:-}" ] && export DS4_METAL_MODEL_MAX_VIEW_BYTES="$VIEW_CAP"
export DS4_METAL_NO_RESIDENCY=1
export DS4_METAL_NO_MODEL_WARMUP=1
export DS4_METAL_NO_PREFILL_KERNEL_WARMUP=1
export DS4_METAL_PREFILL_SPLIT=1
# expert-remote 必需: 钉 dense(~7.21G)进真 RAM 池, routed 从远程拉不在本地 fault。
# 缺它 = 首跑 wired 怼 ~10.9G 爆内存真因(见 ds4_cli.c:97 帮助文本)。
export DS4_DENSE_RESIDENT=1
# 持久 expert cache(必须设, cache-miss 才会走远程拉取回调; 256 MiB 小, 全 miss 全拉)
export DS4_EXPERT_CACHE_BYTES="$EXPERT_CACHE"
export DS4_DIAG=1
export DS4_TOKEN_TIMING=1

LOG=/tmp/expert-host-run.log; : > "$LOG"
echo "expert-host-run: model=$MODEL ctx=$CTX tokens=$TOKENS prompt=\"$PROMPT\""
echo "expert-host-run: expert-remote=$REMOTE  cache=$((EXPERT_CACHE/1048576)) MiB  看门狗 ceiling=${CEIL_MIB} MiB  swap0=${SW0}"
echo

./ds4 -m "$MODEL" -c "$CTX" -n "$TOKENS" --temp 0 \
      --expert-remote "$REMOTE" -p "$PROMPT" > "$LOG" 2>&1 &
pid=$!
tail -f "$LOG" & tailpid=$!

cleanup() { kill "$tailpid" 2>/dev/null; kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null; }
trap cleanup EXIT INT TERM

# RSS 看门狗: 每 0.5s 看 RSS; 超 CEIL_MIB 立刻 SIGKILL(抢在 swap thrash/panic 前); 每 ~3s 打一次进度。
wd_i=0
while kill -0 "$pid" 2>/dev/null; do
    rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -z "$rss" ] && break
    mib=$(( rss / 1024 ))
    wd_i=$((wd_i+1)); [ $((wd_i % 6)) -eq 1 ] && echo "[watchdog] host RSS ${mib} MiB / ${CEIL_MIB} MiB"
    if [ "$mib" -gt "$CEIL_MIB" ]; then
        echo; echo "!! 看门狗: host RSS ${mib} MiB > ${CEIL_MIB} MiB — SIGKILL (防本机爆内存)"
        kill -9 "$pid" 2>/dev/null; kill "$tailpid" 2>/dev/null
        exit 2
    fi
    sleep 0.5
done

kill "$tailpid" 2>/dev/null
wait "$pid" 2>/dev/null; rc=$?
echo; echo "expert-host-run: host 退出 rc=$rc (峰值未超 ${CEIL_MIB} MiB)"
echo "== 关键诊断 =="
grep -E 'expert tier|connected|remote=|ms/token' "$LOG" | tail -20
exit "$rc"
