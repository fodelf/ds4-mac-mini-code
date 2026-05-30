#!/bin/bash
# expert-deploy-replica.sh — 【8 GiB 从机】上的 routed-expert 冷层服务端(ds4-expert-replica)。
#
# 仿【已验证安全的 smoke-mtp-replica.sh】的后台+看门狗结构。但有本质区别(必须懂):
#   - MTP replica 用 Metal backend、wire model view, 所以靠 2GiB view cap 压 wired 峰值。
#   - 本 expert-replica 用【CPU backend + expert_server_mode】: 不建图、不 wire 任何 Metal
#     view、不跑推理。它只在收到 EXPERT_REQ 时从 mmap memcpy 几个 expert 字节发回。
#     faulted 的是 clean 只读页(MAP_PRIVATE, 永不写)→ OS 可即时回收、不走 swap。
#   - expert_server_mode 已在代码层【关掉】CPU-backend open 的 82 GiB WILLNEED 全量预读
#     (那正是之前把机器干爆的根因)。所以增长是【按请求逐个 fault】, 平缓可控, 不洪水。
#   - 所以这里【不设任何 DS4_METAL_* env】(对 CPU backend 无意义)。
#
# 红线 8 GiB: clean 缓存填满 RAM 本身是安全的(可回收), 但看门狗仍在 RSS 7 GiB 处兜底
#   SIGKILL —— 万一工作集异常或代码回归, 抢在 thrash 前停掉(干净退出, 不崩机)。
#
# -mcpu=native: 必须【在本机(从机)编译】, M4 上编出的二进制在 M1 会 SIGILL。
#
# Usage (在从机, 先 rsync 源码过来):
#   ./notes/dual-host/expert-deploy-replica.sh [port] [bind]
#   CEIL_MIB=6144 ./notes/dual-host/expert-deploy-replica.sh 17600 0.0.0.0
set -u
cd "$(dirname "$0")/../.."

PORT="${1:-17600}"
BIND="${2:-0.0.0.0}"                  # 直连 TB 时填从机 TB 网卡 IP 更安全
MODEL="${MODEL:-./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf}"
CEIL_MIB="${CEIL_MIB:-7168}"          # 8G 机硬上限 7 GiB 兜底 → 超即 SIGKILL

if [ ! -f "$MODEL" ]; then echo "expert-deploy-replica: 模型不存在 $MODEL" >&2; exit 1; fi

echo "== 在本机(从机)重编 ds4-expert-replica (-mcpu=native) =="
make ds4-expert-replica 2>&1 | tail -3 || { echo "ABORT: 编译失败" >&2; exit 1; }
if [ ! -x ./ds4-expert-replica ]; then echo "ABORT: 二进制未生成" >&2; exit 1; fi

SW0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}')
if [ "$SW0" -gt 300 ]; then echo "ABORT: swap_used=${SW0} MiB > 300, 先重启清干净"; exit 1; fi

LOG=/tmp/expert-deploy-replica.log; : > "$LOG"
echo "expert-deploy-replica: base=$MODEL port=$PORT bind=$BIND  看门狗 ceiling=${CEIL_MIB} MiB  swap0=${SW0}"
echo "expert-deploy-replica: 懒加载(无预读), 等本机 host 连入(看到 'listening on' 即就绪)"
echo

./ds4-expert-replica listen "$PORT" "$BIND" -m "$MODEL" > "$LOG" 2>&1 &
pid=$!
tail -f "$LOG" & tailpid=$!

cleanup() { kill "$tailpid" 2>/dev/null; kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null; }
trap cleanup EXIT INT TERM

# RSS 看门狗: 每 0.5s 看 RSS; 超 CEIL_MIB 立刻 SIGKILL(抢在 thrash 前); 每 ~5s 打一次进度。
wd_i=0
while kill -0 "$pid" 2>/dev/null; do
    rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -z "$rss" ] && break
    mib=$(( rss / 1024 ))
    wd_i=$((wd_i+1)); [ $((wd_i % 10)) -eq 1 ] && echo "[watchdog] replica RSS ${mib} MiB / ${CEIL_MIB} MiB"
    if [ "$mib" -gt "$CEIL_MIB" ]; then
        echo; echo "!! 看门狗: replica RSS ${mib} MiB > ${CEIL_MIB} MiB — SIGKILL (防 8G 机爆内存)"
        kill -9 "$pid" 2>/dev/null; kill "$tailpid" 2>/dev/null
        exit 2
    fi
    sleep 0.5
done

kill "$tailpid" 2>/dev/null
echo; echo "expert-deploy-replica: replica 退出 (峰值未超 ${CEIL_MIB} MiB)"
