#!/bin/zsh
# 单进程、小上下文、少 token 的保守评估。
# 目的: 确认 rebuild 后的 ds4 默认路径(不带 --expert-remote)还能正常出 token,
#       且我的双机改动对默认路径零影响。不开 dense 池、不开 expert cache、不连 replica。
# 铁律: 硬内存看门狗 —— swap 涨过阈值或 RSS 越预算, 立刻 SIGKILL, 绝不拖垮系统。
set -u
cd "$(dirname "$0")/../.."

MODEL=./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
CTX=${CTX:-1024}
NGEN=${NGEN:-4}
PROMPT=${PROMPT:-"1+1=?"}
LOG=/tmp/ds4_small.log

# --- 看门狗阈值 (保守) ---
SWAP_DELTA_KILL_MiB=800     # swap 比基线涨过 800 MiB => anon 压力, 杀
RSS_KILL_MiB=12000          # 单进程 RSS 越 12 GiB => 接近 16 GiB 红线, 杀
WALL_CAP_S=240              # 硬墙: 最多跑 240s

swap_used() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}'; }
SWAP0=$(swap_used)
echo "watchdog: baseline swap=${SWAP0} MiB  kill@swap+${SWAP_DELTA_KILL_MiB} / rss>${RSS_KILL_MiB}MiB / wall>${WALL_CAP_S}s"

: > "$LOG"
DS4_TOKEN_TIMING=1 DS4_DIAG=1 ./ds4 -m "$MODEL" -c "$CTX" -n "$NGEN" -p "$PROMPT" >>"$LOG" 2>&1 &
PID=$!
echo "ds4 pid=$PID ctx=$CTX ngen=$NGEN prompt='$PROMPT'"

t0=$(date +%s)
while kill -0 "$PID" 2>/dev/null; do
  sleep 1
  sw=$(swap_used); dsw=$((sw - SWAP0))
  rss=$(ps -o rss= -p "$PID" 2>/dev/null | awk '{print int($1/1024)}'); rss=${rss:-0}
  now=$(date +%s); el=$((now - t0))
  if [ "$dsw" -gt "$SWAP_DELTA_KILL_MiB" ]; then
    echo "WATCHDOG KILL: swap +${dsw} MiB > ${SWAP_DELTA_KILL_MiB}"; kill -9 "$PID" 2>/dev/null; echo "KILLED_SWAP"; exit 9
  fi
  if [ "$rss" -gt "$RSS_KILL_MiB" ]; then
    echo "WATCHDOG KILL: rss ${rss} MiB > ${RSS_KILL_MiB}"; kill -9 "$PID" 2>/dev/null; echo "KILLED_RSS"; exit 9
  fi
  if [ "$el" -gt "$WALL_CAP_S" ]; then
    echo "WATCHDOG KILL: wall ${el}s > ${WALL_CAP_S}"; kill -9 "$PID" 2>/dev/null; echo "KILLED_WALL"; exit 9
  fi
done
wait "$PID"; rc=$?
echo "ds4 exited rc=$rc  swap_now=$(swap_used) MiB"
echo "DONE rc=$rc"
