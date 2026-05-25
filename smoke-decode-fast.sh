#!/bin/bash
# smoke-decode-fast.sh — Lvl 1 decode-speed experiment.
#
# Hypothesis (notes/execution-log.md #58): DS4_METAL_DECODE_SPLIT_EVERY=1 is
# now redundant after #55 A3 because A3's routed-MoE decode path already forces
# a per-layer CB boundary via `end_commands` (sync router) → memcpy scratch →
# `begin_commands` (re-bound expert dispatch).  The SPLIT_EVERY=1 flush is an
# additional second sync per layer.  Removing it should drop ~43 syncs / token
# = ~4.3 s / token saved at observed ~100 ms/sync, taking decode from 0.09 to
# roughly 0.15-0.18 t/s if the cost model is right.
#
# Diff vs smoke-watch.sh: ONLY removes DS4_METAL_DECODE_SPLIT_EVERY=1 from the
# ds4 env block.  Everything else (PREFILL_SPLIT, view cap, A3 EXPERT_OFFLOAD,
# NO_RESIDENCY, NO_MODEL_WARMUP, NO_PREFILL_KERNEL_WARMUP) is identical so the
# delta is attributable to that one variable.
#
# Risks:
# - Removing the per-layer flush means a decode CB may try to wire more model
#   views in one shot.  After #55 A3, routed-expert tensors live in resident
#   scratch and do NOT bind a view, so per-layer wireable delta is only attn
#   (~3 MiB FP4) + indexer (~1 MiB Q8) + a few MiB shared-expert / norm = ~10
#   MiB/layer extra.  43 layers worst case = ~430 MiB extra wireable, well
#   inside the ~3 GiB headroom validated at Stage 3.  OOM very unlikely.
# - If OOM does fire (e.g. some path still binds full views per layer), the
#   smoke prints `command batch failed` and `IOGPUCommandBufferCallbackError
#   OutOfMemory` — back off by re-enabling SPLIT_EVERY=1 or trying
#   SPLIT_EVERY=4/8 as middle ground.
#
# Usage: ./smoke-decode-fast.sh [CTX] [TOKENS] [PROMPT]
# Default: ctx=4096, tokens=64, prompt-file=notes/smoke-prompts/stage1.txt
#
# IMPORTANT: kill any running ds4-server first — single ds4-engine
# instance lock will refuse the second process.

set -u

CTX="${1:-4096}"
TOKENS="${2:-64}"
PROMPT="${3:-Hi}"
MODEL="${DS4_MODEL:-./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf}"

# Default to stage1 prompt for apples-to-apples decode-rate comparison with
# #56 Stage 1's 0.09 t/s baseline.
: "${DS4_PROMPT_FILE:=notes/smoke-prompts/stage1.txt}"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${DS4_SMOKE_OUT_DIR:-/tmp}"
STDERR_LOG="$OUT_DIR/ds4-decode-fast-$STAMP.stderr.log"
STDOUT_LOG="$OUT_DIR/ds4-decode-fast-$STAMP.stdout.log"

: "${DS4_METAL_MODEL_MAX_VIEW_BYTES_OVERRIDE:=2147483648}"  # 2 GiB; see #54-patch2
: "${DS4_METAL_EXPERT_OFFLOAD:=1}"

# Determine prompt source — file (default) vs inline string.
if [ -n "${DS4_PROMPT_FILE:-}" ] && [ "$DS4_PROMPT_FILE" != "" ]; then
    if [ ! -f "$DS4_PROMPT_FILE" ]; then
        echo "smoke-decode-fast: DS4_PROMPT_FILE=$DS4_PROMPT_FILE does not exist" >&2
        exit 2
    fi
    PROMPT_ARG=(--prompt-file "$DS4_PROMPT_FILE")
    PROMPT_DESC="file=$DS4_PROMPT_FILE ($(wc -c < "$DS4_PROMPT_FILE") bytes)"
else
    PROMPT_ARG=(-p "$PROMPT")
    PROMPT_DESC="prompt=\"$PROMPT\""
fi

echo "smoke-decode-fast: ctx=$CTX tokens=$TOKENS $PROMPT_DESC model=$MODEL"
echo "smoke-decode-fast: full stderr -> $STDERR_LOG"
echo "smoke-decode-fast: full stdout -> $STDOUT_LOG"
echo "smoke-decode-fast: A3 expert offload = $DS4_METAL_EXPERT_OFFLOAD"
echo "smoke-decode-fast: model view cap    = $DS4_METAL_MODEL_MAX_VIEW_BYTES_OVERRIDE bytes"
echo "smoke-decode-fast: ** DS4_METAL_DECODE_SPLIT_EVERY = UNSET (Lvl 1 experiment) **"
echo "smoke-decode-fast: Ctrl+C 干净中断 (SIGTERM→3s→SIGKILL)"
echo "=== START $(date +%T) ==="

# Filter matches smoke-watch.sh so the live tail looks the same.
LIVE_FILTER='gpu prefill layer|gpu decode layer|command batch failed|OutOfMemory|finish\[|metal\[post-end-commit\]|metal\[end_commands\]|prefill:.*generation:|residency requested|model warmup|prefill kernel warmup|split-loop entering|prefill_layer_major|ds4: Metal|DS4_METAL_EXPERT_OFFLOAD|DS4_METAL_DECODE_SPLIT_EVERY|cannot open model'

: > "$STDOUT_LOG"
tail -f "$STDOUT_LOG" | awk '{ printf "[stdout] %s\n", $0; fflush(); }' &
TAIL_PID=$!

DS4_PID=""
INTERRUPTED=0

on_exit() {
    if [ -n "$TAIL_PID" ]; then
        kill "$TAIL_PID" 2>/dev/null
        wait "$TAIL_PID" 2>/dev/null
    fi
}
trap on_exit EXIT

on_interrupt() {
    INTERRUPTED=1
    echo
    echo "=== INTERRUPT $(date +%T) — SIGTERM ds4 (PID=$DS4_PID) ==="
    if [ -n "$DS4_PID" ]; then
        kill -TERM "$DS4_PID" 2>/dev/null
        for _ in 1 2 3 4 5 6; do
            kill -0 "$DS4_PID" 2>/dev/null || break
            sleep 0.5
        done
        if kill -0 "$DS4_PID" 2>/dev/null; then
            echo "smoke-decode-fast: SIGTERM ignored, SIGKILL"
            kill -KILL "$DS4_PID" 2>/dev/null
        fi
    fi
}
trap on_interrupt INT TERM

# NOTE the missing DS4_METAL_DECODE_SPLIT_EVERY line vs smoke-watch.sh — this
# IS the experiment.  Everything else is held constant.
DS4_METAL_NO_RESIDENCY=1 \
DS4_METAL_NO_MODEL_WARMUP=1 \
DS4_METAL_NO_PREFILL_KERNEL_WARMUP=1 \
DS4_METAL_PREFILL_SPLIT=1 \
DS4_METAL_MODEL_MAX_VIEW_BYTES="$DS4_METAL_MODEL_MAX_VIEW_BYTES_OVERRIDE" \
DS4_METAL_EXPERT_OFFLOAD="$DS4_METAL_EXPERT_OFFLOAD" \
DS4_DIAG=1 \
./ds4 -m "$MODEL" \
      -c "$CTX" \
      -n "$TOKENS" \
      --temp 0 \
      "${PROMPT_ARG[@]}" \
  > "$STDOUT_LOG" \
  2> >(tee "$STDERR_LOG" | grep --line-buffered -E "$LIVE_FILTER" | awk '{ printf "[ds4]    %s\n", $0; fflush(); }' >&2) &
DS4_PID=$!

wait "$DS4_PID" 2>/dev/null
rc=$?
while kill -0 "$DS4_PID" 2>/dev/null; do
    wait "$DS4_PID" 2>/dev/null
    rc=$?
done
if [ "$INTERRUPTED" -eq 1 ]; then
    rc=130
fi

sleep 0.3
echo
echo "=== END   $(date +%T) rc=$rc ==="

# Quick comparison summary — focus on the decode t/s line vs #56 baseline 0.09.
echo
echo "----- decode rate (compare to #56 Stage 1 baseline 0.09 t/s) -----"
grep -E "prefill:.*generation:" "$STDERR_LOG" || echo "WARN: no prefill/generation summary line found"

echo
echo "----- any OOM / command-batch failures (should be EMPTY) -----"
grep -E "OutOfMemory|command batch failed" "$STDERR_LOG" | tail -10 || echo "(clean — none)"

echo
echo "----- last 8 metal[post-end-commit] for wireable / view trend -----"
grep -E "ds4_diag: metal\[post-end-commit\]" "$STDERR_LOG" | tail -8

echo
echo "----- generated stdout (raw bytes via od) -----"
od -c "$STDOUT_LOG" | head -20

exit $rc
