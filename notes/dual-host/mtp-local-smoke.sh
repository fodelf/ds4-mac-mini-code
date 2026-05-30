#!/usr/bin/env bash
# mtp-local-smoke.sh — MEMORY-SAFE local probe of the ds4-mtp-replica footprint.
#
# Purpose: verify the #1 risk on THIS 16 GB box without risking it — does Metal
# view-shrink keep the replica near its ~5 GiB embd+output+MTP working set, or does
# it regress and try to wire the whole 86 GiB base? It launches ONLY the replica
# (no target), watches RSS, and SIGKILLs the instant resident crosses the ceiling.
#
# What it deliberately does NOT do: the full "1+1=?" end-to-end. That needs the
# 86 GiB base loaded TWICE on one machine (target ds4 + replica) — unsafe on 16 GB
# and BLOCKED by the memory-safety rule [[feedback_memory_safety_gate]]. End-to-end
# must run across two machines (notes/dual-host/mtp-deploy-replica.sh).
#
# By construction this cannot OOM the box: hard RAM preflight + RSS watchdog +
# auto-stop after a fixed probe window.
#
# Usage:
#   notes/dual-host/mtp-local-smoke.sh
#   PORT=17599 PROBE_SECS=40 notes/dual-host/mtp-local-smoke.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="${LOCAL_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

BIN="$LOCAL_DIR/ds4-mtp-replica"
BASE="${BASE:-$LOCAL_DIR/ds4flash.gguf}"
MTP="${MTP:-$LOCAL_DIR/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf}"
PORT="${PORT:-17599}"
BIND="${BIND:-127.0.0.1}"
CTX="${CTX:-4096}"

MEM_NEED_MIB="${MEM_NEED_MIB:-5222}"   # expected resident ~5.1 GiB
MEM_CEIL_MIB="${MEM_CEIL_MIB:-6656}"   # hard kill ceiling 6.5 GiB (safe on a 16 GB box)
PROBE_SECS="${PROBE_SECS:-40}"

echo "== ds4-mtp-replica LOCAL memory probe (NOT end-to-end) =="
echo "  bin:  $BIN"
echo "  base: $BASE"
echo "  mtp:  $MTP"
echo "  expect ~$MEM_NEED_MIB MiB | kill ceiling $MEM_CEIL_MIB MiB | probe ${PROBE_SECS}s"
echo

# --- preflight: files, binary sanity, this box's RAM ---
[ -x "$BIN" ] || { echo "!! $BIN missing/not executable — run: make ds4-mtp-replica" >&2; exit 1; }
# Stub guard: a real engine link pulls in the model loader; a CLI shim or empty
# stub does not. Size alone is a weak signal (the real ds4-mtp-replica is ~728 KB,
# while ds4-replica/ds4-kv-server are genuine ~37 KB shims), so gate on BOTH a low
# size floor AND the presence of an engine symbol — that is what makes it the
# drafter, not a stub. (256 KB floor excludes shims/empty stubs without tripping
# on the real link.)
binsz=$(stat -f%z "$BIN" 2>/dev/null || echo 0)
if [ "$binsz" -lt 262144 ]; then
  echo "!! $BIN is only $binsz bytes — too small to be a real engine link. Rebuild before probing." >&2
  exit 1
fi
if command -v nm >/dev/null 2>&1; then
  if ! nm "$BIN" 2>/dev/null | grep -q "ds4_engine_mtp_draft_burst"; then
    echo "!! $BIN does not contain ds4_engine_mtp_draft_burst — not a real MTP replica link. Rebuild." >&2
    exit 1
  fi
fi
[ -f "$BASE" ] || { echo "!! base gguf missing: $BASE" >&2; exit 1; }
[ -f "$MTP" ]  || { echo "!! mtp gguf missing: $MTP" >&2; exit 1; }
total_mib=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1048576 ))
if [ "$total_mib" -gt 0 ] && [ "$total_mib" -lt $(( MEM_CEIL_MIB + 2048 )) ]; then
  echo "!! this box RAM ${total_mib} MiB < ceiling $MEM_CEIL_MIB + 2048 margin — refusing (memory-safety rule)" >&2
  exit 3
fi
echo "   this box RAM: ${total_mib} MiB"

logf="$(mktemp -t mtp-local-smoke.XXXXXX.log)"
echo "   replica log: $logf"
echo

# --- launch replica only (view-shrink forced) ---
DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_MODEL_MAX_VIEW_BYTES=2147483648 \
  DS4_METAL_NO_RESIDENCY=1 DS4_METAL_NO_MODEL_WARMUP=1 DS4_METAL_NO_PREFILL_KERNEL_WARMUP=1 \
  "$BIN" listen "$PORT" "$BIND" -m "$BASE" --mtp "$MTP" -c "$CTX" > "$logf" 2>&1 &
rpid=$!
echo "   replica pid $rpid"

cleanup() { kill -0 "$rpid" 2>/dev/null && kill "$rpid" 2>/dev/null; }
trap cleanup EXIT INT TERM

peak_mib=0
killed=0
for (( i=0; i<PROBE_SECS*10; i++ )); do
  if ! kill -0 "$rpid" 2>/dev/null; then echo "!! replica exited early — see $logf"; break; fi
  rss_kib=$(ps -o rss= -p "$rpid" 2>/dev/null | tr -d ' ')
  if [ -n "$rss_kib" ]; then
    rss_mib=$(( rss_kib / 1024 ))
    (( rss_mib > peak_mib )) && peak_mib=$rss_mib
    if [ "$rss_mib" -gt "$MEM_CEIL_MIB" ]; then
      echo "!! MEM WATCHDOG: RSS ${rss_mib} MiB > $MEM_CEIL_MIB MiB — SIGKILL (view-shrink regressed)"
      kill -9 "$rpid" 2>/dev/null
      killed=1
      break
    fi
  fi
  sleep 0.1
done

# stop the replica cleanly if still up (probe window over)
kill -0 "$rpid" 2>/dev/null && kill "$rpid" 2>/dev/null

echo
echo "-- result --"
echo "   peak RSS: ${peak_mib} MiB (expected ~$MEM_NEED_MIB MiB)"
if grep -q "listening on" "$logf" 2>/dev/null; then
  echo "   replica reached listen state (model load completed)"; loaded=1
else
  echo "   WARN: replica never printed 'listening on' — load incomplete or crashed (see $logf)"; loaded=0
fi
if [ "$killed" = 1 ]; then
  echo "   VERDICT: FAIL — view-shrink regressed; replica tried to wire too much. DO NOT deploy."
  exit 1
elif [ "$loaded" = 1 ] && [ "$peak_mib" -le "$MEM_CEIL_MIB" ] && [ "$peak_mib" -ge 1024 ]; then
  echo "   VERDICT: PASS — footprint within budget; safe to deploy to the 8 GiB box."
else
  echo "   VERDICT: INCONCLUSIVE — check $logf."
  exit 2
fi
