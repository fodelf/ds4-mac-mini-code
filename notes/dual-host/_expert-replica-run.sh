#!/bin/bash
# _expert-replica-run.sh — 从机【后台/nohup 专用】启动器: 起 ds4-expert-replica + 双阈值看门狗。
# 与 expert-deploy-replica.sh 区别: 不 make、不 tail(适合远程 nohup), 看门狗加 swap-delta。
# 看门狗: RSS 超 CEIL_MIB(8G 机红线) 或 swap 比基线涨过 SWAPD_MIB(真 thrash) => 立刻 SIGKILL。
# 日志: /tmp/expert-replica.log(服务端) /tmp/expert-replica.wd(看门狗) /tmp/expert-replica.pid。
set -u
cd "$(dirname "$0")/../.."

PORT="${1:-17600}"
BIND="${2:-0.0.0.0}"
MODEL="${MODEL:-./ds4flash.gguf}"
CEIL_MIB="${CEIL_MIB:-7168}"          # 8G 机 RSS 硬上限 7 GiB
SWAPD_MIB="${SWAPD_MIB:-1500}"        # swap 比基线涨过 1.5 GiB => 杀(它在拖垮别的 app)

LOG=/tmp/expert-replica.log; WD=/tmp/expert-replica.wd; PIDF=/tmp/expert-replica.pid
: > "$LOG"; : > "$WD"
sw(){ sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}'; }
SW0=$(sw)

[ -f "$MODEL" ] || { echo "ABORT: 模型不存在 $MODEL" >> "$WD"; exit 1; }
./ds4-expert-replica listen "$PORT" "$BIND" -m "$MODEL" > "$LOG" 2>&1 &
pid=$!; echo "$pid" > "$PIDF"
echo "$(date +%H:%M:%S) started pid=$pid swap0=${SW0}MiB ceil=${CEIL_MIB} swapd_kill=${SWAPD_MIB}" >> "$WD"

while kill -0 "$pid" 2>/dev/null; do
  sleep 1
  rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' '); [ -z "$rss" ] && break
  mib=$((rss/1024)); d=$(( $(sw) - SW0 ))
  echo "$(date +%H:%M:%S) rss=${mib}MiB swapd=${d}MiB" >> "$WD"
  if [ "$mib" -gt "$CEIL_MIB" ] || [ "$d" -gt "$SWAPD_MIB" ]; then
    echo "$(date +%H:%M:%S) !! KILL rss=${mib} swapd=${d}" >> "$WD"
    kill -9 "$pid" 2>/dev/null; break
  fi
done
echo "$(date +%H:%M:%S) replica exited rc=$? swap_now=$(sw)MiB" >> "$WD"
