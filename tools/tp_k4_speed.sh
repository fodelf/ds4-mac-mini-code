#!/usr/bin/env bash
# 双机张量并行(TP) k4 最小上下文测速脚本 (reverse-connect 版)。
#
# 网络方向: 本机=coordinator 主动 connect, M1=worker listen。
#   起因: M1 那台机器 ds4 进程的"出站 connect"会诡异地返回 No route to host
#   (同机 nc/最小C/Metal 程序却都正常), 而本机 connect M1 完全 OK。所以让本机
#   主动连。DS4_TP_REVERSE_CONNECT=1 在 ds4 里翻转 listen/connect 角色;
#   TP all-reduce 是对称求和, 方向不影响结果。
#
# 流程: 同步源码→M1 → 两边 make clean && make ds4 → 清 5599 端口 →
#   先起 M1 worker(listen) → 等加载完 → 起本机 coordinator(connect) →
#   打印生成文本 + prefill/decode tok/s。
#
# 安全闸 (任一触发"两边同时杀进程, 只杀进程不删文件"):
#   1. 本终端 Ctrl+C。
#   2. 本机 ds4 RSS 超 LOCAL_MAX_GB。
#   3. M1   ds4 RSS 超 REMOTE_MAX_GB (k4 backbone 8.2G 是硬底, 默认 12G)。
set -uo pipefail

# ---------------- 配置 (均可 env 覆盖) ----------------
REMOTE=${REMOTE:-192.168.1.2}                  # M1 (worker), ssh 目标
REMOTE_DIR=${REMOTE_DIR:-/Users/fodelf/ds4-main}
LOCAL_DIR=${LOCAL_DIR:-/Users/fodelf/git/ds4-main}
RENDEZVOUS_IP=${RENDEZVOUS_IP:-192.168.1.2}    # 会合地址 = M1 雷电 IP (worker 在此 listen, coordinator connect 此)
PORT=${PORT:-5599}
MODEL=${MODEL:-gguf/ds4flash-k4.gguf}
CTX=${CTX:-32}
NPRED=${NPRED:-16}
PROMPT=${PROMPT:-hi}
TP_LAYERS=${TP_LAYERS:-3}
LOCAL_MAX_GB=${LOCAL_MAX_GB:-12}
REMOTE_MAX_GB=${REMOTE_MAX_GB:-12}             # k4 backbone 8.2G, 需 >8.2
LOCAL_BUDGET_MB=${LOCAL_BUDGET_MB:-13500}      # L1-gate planned-resident 预算 (与看门狗解耦)
REMOTE_BUDGET_MB=${REMOTE_BUDGET_MB:-13500}
RUN_ENV=${RUN_ENV:-"DS4_TP_REVERSE_CONNECT=1 DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_NO_RESIDENCY=1"}

LEADER_LOG=/tmp/tp_k4_leader.log
FOLLOWER_LOG=/tmp/tp_k4_follower.log
LEADER_PID=""

log(){ echo "[tp-speed] $*"; }

# ---------------- 清理: 两边同杀 (幂等, 只杀进程) ----------------
cleanup(){
  trap - INT TERM EXIT
  echo
  log "cleanup: 杀两边 ds4 进程 (只杀进程, 不删任何文件)"
  [ -n "$LEADER_PID" ] && kill "$LEADER_PID" 2>/dev/null || true
  pkill -f 'ds4 -m' 2>/dev/null || true
  ssh "$REMOTE" 'pkill -f "ds4 -m" 2>/dev/null' 2>/dev/null || true
  log "done."
}
trap cleanup INT TERM EXIT

# ---------------- RSS (GiB) ----------------
rss_gb_local(){ local kb; kb=$(ps -o rss= -p "$1" 2>/dev/null | tr -d ' '); [ -n "$kb" ] && awk "BEGIN{printf \"%.2f\",$kb/1048576}" || echo 0; }
rss_gb_remote(){
  local kb; kb=$(ssh "$REMOTE" "pid=\$(pgrep -f 'ds4 -m' | head -1); [ -n \"\$pid\" ] && ps -o rss= -p \$pid 2>/dev/null | tr -d ' '" 2>/dev/null)
  [ -n "$kb" ] && awk "BEGIN{printf \"%.2f\",$kb/1048576}" || echo 0
}
over(){ awk "BEGIN{a=$1+0;b=$2+0;exit !(a>b)}"; }

# ---------------- 1. 同步代码 ----------------
log "同步源码 → $REMOTE:$REMOTE_DIR"
rsync -a --exclude '.git' --exclude '*.o' --exclude '*.gguf' --exclude 'gguf/' \
  --exclude '*.bin' --exclude 'ds4' --exclude 'ds4-server' --exclude 'ds4-bench' \
  --exclude 'ds4-eval' --exclude 'ds4-agent' --exclude 'e0-pingpong' --exclude 'ds4_test' \
  "$LOCAL_DIR"/ "$REMOTE:$REMOTE_DIR"/ || { log "rsync 失败"; exit 1; }

# ---------------- 2. 两边 clean + build ----------------
log "本机: make clean && make ds4"
( cd "$LOCAL_DIR" && make clean >/dev/null 2>&1 && make ds4 >/tmp/tp_k4_build_local.log 2>&1 ) \
  || { log "本机编译失败:"; tail -8 /tmp/tp_k4_build_local.log; exit 1; }
log "M1:   make clean && make ds4"
ssh "$REMOTE" "cd '$REMOTE_DIR' && make clean >/dev/null 2>&1 && make ds4 >/tmp/tp_k4_build_remote.log 2>&1" \
  || { log "M1 编译失败:"; ssh "$REMOTE" "tail -8 /tmp/tp_k4_build_remote.log"; exit 1; }

# ---------------- 3. 清旧进程 + 清 5599 端口 ----------------
log "清理两边旧 ds4 进程与 $PORT 端口占用"
pkill -f 'ds4 -m' 2>/dev/null || true
ssh "$REMOTE" "pkill -f 'ds4 -m' 2>/dev/null; lsof -nP -iTCP:$PORT -t 2>/dev/null | xargs -r kill -9 2>/dev/null; true" 2>/dev/null || true
sleep 1

# ---------------- 4. 先起 M1 worker (listen) ----------------
log "启动 M1 worker: --listen $RENDEZVOUS_IP:$PORT (reverse)"
ssh "$REMOTE" "cd '$REMOTE_DIR' && rm -f '$FOLLOWER_LOG'; \
  $RUN_ENV DS4_MEM_BUDGET_MB=$REMOTE_BUDGET_MB \
  nohup ./ds4 -m '$MODEL' --role worker --listen '$RENDEZVOUS_IP' '$PORT' \
  --tp --tp-layers '$TP_LAYERS' -c '$CTX' -n '$NPRED' --temp 0 --nothink \
  -p '$PROMPT' > '$FOLLOWER_LOG' 2>&1 & echo launched" 2>/dev/null

log "等 M1 worker 加载完并进入 listen…"
wok=0
for _ in $(seq 1 60); do
  if ssh "$REMOTE" "grep -q 'backend initialized' '$FOLLOWER_LOG'" 2>/dev/null; then wok=1; break; fi
  if ssh "$REMOTE" "grep -qiE 'refusing to load|failed|Address already' '$FOLLOWER_LOG'" 2>/dev/null; then
    log "M1 worker 启动失败, 日志尾:"; ssh "$REMOTE" "tail -6 '$FOLLOWER_LOG'"; cleanup; exit 1
  fi
  rg=$(rss_gb_remote); over "$rg" "$REMOTE_MAX_GB" && { log "M1 加载阶段 RSS ${rg}G 超 ${REMOTE_MAX_GB}G → 杀"; cleanup; exit 2; }
  sleep 1
done
[ "$wok" = 1 ] || { log "M1 worker 60s 未就绪 → 放弃"; cleanup; exit 1; }
sleep 1   # worker 从 backend-init 到 listen/accept 的余量

# ---------------- 5. 起本机 coordinator (connect M1) ----------------
cd "$LOCAL_DIR"
log "启动本机 coordinator: --coordinator $RENDEZVOUS_IP:$PORT (connect, reverse)"
rm -f "$LEADER_LOG"
env $RUN_ENV DS4_MEM_BUDGET_MB=$LOCAL_BUDGET_MB \
  ./ds4 -m "$MODEL" --role coordinator --coordinator "$RENDEZVOUS_IP" "$PORT" \
  --tp --tp-layers "$TP_LAYERS" -c "$CTX" -n "$NPRED" --temp 0 --nothink \
  -p "$PROMPT" > "$LEADER_LOG" 2>&1 &
LEADER_PID=$!

# ---------------- 6. 看门狗主循环 ----------------
log "运行中… (Ctrl+C 两边同杀; 本机>${LOCAL_MAX_GB}G 或 M1>${REMOTE_MAX_GB}G 也两边同杀)"
while kill -0 "$LEADER_PID" 2>/dev/null; do
  lg=$(rss_gb_local "$LEADER_PID"); rg=$(rss_gb_remote)
  printf "\r[tp-speed] RSS 本机=%sG/%dG  M1=%sG/%dG    " "$lg" "$LOCAL_MAX_GB" "$rg" "$REMOTE_MAX_GB"
  over "$lg" "$LOCAL_MAX_GB" && { echo; log "本机 RSS ${lg}G 超限 → 两边同杀"; cleanup; exit 2; }
  over "$rg" "$REMOTE_MAX_GB" && { echo; log "M1 RSS ${rg}G 超限 → 两边同杀"; cleanup; exit 2; }
  sleep 1
done
echo

# ---------------- 7. 结果 ----------------
log "coordinator 结束。生成文本:"
grep -v -E '^ds4:|^ds4_profile' "$LEADER_LOG" | tail -5
echo "----------------------------------------"
if grep -q "tok/s" "$LEADER_LOG"; then
  log "速度:"; grep "tok/s" "$LEADER_LOG"
else
  log "未拿到计时行, leader 日志尾:"; tail -10 "$LEADER_LOG"
fi
cleanup
