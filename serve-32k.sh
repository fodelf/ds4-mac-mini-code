#!/bin/bash
# serve-32k.sh — start ds4-server on 16 GiB Mac Mini at the validated Stage 3
# ceiling (ctx=32768) with all the env vars that smoke-watch.sh proved stable.
#
# Why these env vars?  See notes/execution-log.md #47-#56:
# - DS4_METAL_NO_RESIDENCY      :  skip the 86 GiB residency-set request that
#                                  would otherwise be issued at model load.
# - DS4_METAL_NO_MODEL_WARMUP   :  skip the full-GGUF page-fault warmup that
#                                  triggered the 2026-05-24 kernel panic.
# - DS4_METAL_NO_PREFILL_KERNEL_WARMUP :
#                                  skip the HC-attn warmup matmul that wires
#                                  ~9 GiB of model view that never releases.
# - DS4_METAL_PREFILL_SPLIT=1   :  per-layer CB during short-prompt prefill
#                                  so wireable peak is bounded by one layer.
# - DS4_METAL_DECODE_SPLIT_EVERY=1 :
#                                  flush after every decode layer (this is
#                                  what bounds decode wireable but caps t/s
#                                  at ~0.09 — relax later for speed).
# - DS4_METAL_MODEL_MAX_VIEW_BYTES=2 GiB :
#                                  cap each mmap shared-buffer view so IOGPU
#                                  touched-pages-per-resource is bounded.
# - DS4_METAL_EXPERT_OFFLOAD=1  :  #55 A3 — CPU-side memcpy active experts
#                                  into resident scratch (1.728 GiB).  Per-CB
#                                  wireable is bounded by scratch size, not
#                                  by however many view buffers a layer's
#                                  tensors happen to straddle.
#
# Trace goes to /tmp/ds4-server-trace-*.txt so you can post-mortem prompts,
# cache decisions, tool-parser events without re-reading the whole stderr.
#
# Usage: ./serve-32k.sh [PORT]  (default 8000)
# Stop:  Ctrl+C (forwarded to ds4-server which saves any live KV checkpoint).

set -u

PORT="${1:-8000}"
MODEL="${DS4_MODEL:-./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf}"
CTX="${DS4_CTX:-32768}"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${DS4_SERVE_OUT_DIR:-/tmp}"
TRACE_LOG="$OUT_DIR/ds4-server-trace-$STAMP.txt"
STDERR_LOG="$OUT_DIR/ds4-server-$STAMP.stderr.log"

# Disk KV cache lets the second turn skip prefill of the matching prefix.
# Critical at 0.09 t/s decode — without it every Claude Code turn re-prefills
# the 10-15 k token system/tools header from scratch.
KV_DIR="${DS4_KV_DIR:-$OUT_DIR/ds4-kv}"
mkdir -p "$KV_DIR"

echo "serve: model=$MODEL"
echo "serve: ctx=$CTX  port=$PORT  bind=127.0.0.1"
echo "serve: trace -> $TRACE_LOG"
echo "serve: stderr -> $STDERR_LOG"
echo "serve: kv-disk -> $KV_DIR (8 GiB budget)"
echo "serve: Ctrl+C 干净中断 (SIGTERM -> 等 5 s flush KV -> SIGKILL)"

# Per-request expectations at this ceiling:
#   - first turn prefill 8-12k tok system+tools header: ~20-30 min
#   - first turn decode 200-500 tok: ~30-90 min
#   - second-turn cached-prefix hit: prefill only the new suffix
echo "serve: WARN — decode = 0.09 t/s.  500 token reply = ~92 min.  This is"
echo "serve:        for plumbing validation, not daily use.  See execution-log #56."

DS4_PID=""
on_interrupt() {
    echo
    echo "=== INTERRUPT $(date +%T) — SIGTERM to ds4-server (PID=$DS4_PID) ==="
    if [ -n "$DS4_PID" ]; then
        kill -TERM "$DS4_PID" 2>/dev/null
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            kill -0 "$DS4_PID" 2>/dev/null || break
            sleep 0.5
        done
        if kill -0 "$DS4_PID" 2>/dev/null; then
            echo "serve: ds4-server didn't respond in 5 s — SIGKILL"
            kill -KILL "$DS4_PID" 2>/dev/null
        fi
    fi
}
trap on_interrupt INT TERM

DS4_METAL_NO_RESIDENCY=1 \
DS4_METAL_NO_MODEL_WARMUP=1 \
DS4_METAL_NO_PREFILL_KERNEL_WARMUP=1 \
DS4_METAL_PREFILL_SPLIT=1 \
DS4_METAL_DECODE_SPLIT_EVERY=1 \
DS4_METAL_MODEL_MAX_VIEW_BYTES=2147483648 \
DS4_METAL_EXPERT_OFFLOAD=1 \
./ds4-server \
    -m "$MODEL" \
    -c "$CTX" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --kv-disk-dir "$KV_DIR" \
    --kv-disk-space-mb 8192 \
    --trace "$TRACE_LOG" \
    2> >(tee "$STDERR_LOG" >&2) &
DS4_PID=$!

wait "$DS4_PID" 2>/dev/null
rc=$?
while kill -0 "$DS4_PID" 2>/dev/null; do
    wait "$DS4_PID" 2>/dev/null
    rc=$?
done
echo "=== END $(date +%T) rc=$rc ==="
exit $rc
