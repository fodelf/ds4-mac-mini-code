#!/bin/bash
# smoke-watch.sh — live-streaming wrapper around the smoke-low-mem env.
#
# Difference vs smoke-low-mem.sh: that script buffers everything to a file
# and prints a summary only after rc=?.  This one streams stderr (filtered)
# AND stdout (raw tokens) live to your terminal as the model runs, so you
# can watch prefill layers tick by, see OOMs the moment they fire, and see
# the first generated character the moment Metal hands it back.
#
# Full unfiltered stderr is still saved to disk for post-mortem.
#
# Usage: ./smoke-watch.sh [CTX] [TOKENS] [PROMPT]
# Defaults: ctx=1024, tokens=32, prompt="Hi" — note tokens defaults higher
# than smoke-low-mem.sh (1) so you actually see decode streaming.

set -u

CTX="${1:-1024}"
TOKENS="${2:-32}"
PROMPT="${3:-Hi}"
MODEL="${DS4_MODEL:-./gguf/ds4flash-k16.gguf}"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${DS4_SMOKE_OUT_DIR:-/tmp}"
STDERR_LOG="$OUT_DIR/ds4-watch-$STAMP.stderr.log"
STDOUT_LOG="$OUT_DIR/ds4-watch-$STAMP.stdout.log"

: "${DS4_METAL_MODEL_MAX_VIEW_BYTES_OVERRIDE:=3758096384}"  # 3.5 GiB; see #52

echo "smoke-watch: ctx=$CTX tokens=$TOKENS prompt=\"$PROMPT\" model=$MODEL"
echo "smoke-watch: full stderr -> $STDERR_LOG"
echo "smoke-watch: full stdout -> $STDOUT_LOG"
echo "smoke-watch: live stderr  = layer progress | OOM | finish[] | post-end metal[] | summary"
echo "smoke-watch: live stdout  = raw generated tokens (prefix [stdout])"
echo "smoke-watch: model view cap = $DS4_METAL_MODEL_MAX_VIEW_BYTES_OVERRIDE bytes"
echo "=== START $(date +%T) ==="

# Filter for the live-tail stderr stream.  Drops the high-frequency
# vmstat[pre-end-commit] / metal[begin_commands] / metal[pre-end-commit]
# lines (3 per layer × 43 layers = 129 noisy lines per prefill) and keeps:
#   - layer progression
#   - command-batch / OOM failures
#   - finish[] (per-CB transient drop count + alive after)
#   - metal[post-end-commit] (the line after a CB resolves; drv/pipelines)
#   - metal[end_commands] (final cleanup print)
#   - prefill/generation t/s summary
#   - residency / warmup / view-cap env verification on startup
LIVE_FILTER='gpu prefill layer|gpu decode layer|command batch failed|OutOfMemory|finish\[|metal\[post-end-commit\]|metal\[end_commands\]|prefill:.*generation:|residency requested|model warmup|prefill kernel warmup|split-loop entering|prefill_layer_major|ds4: Metal'

# Live-tail stdout in the background so generated tokens appear as the
# model emits them.  Touch first so tail -f doesn't error on missing file.
: > "$STDOUT_LOG"
tail -f "$STDOUT_LOG" | awk '{ printf "[stdout] %s\n", $0; fflush(); }' &
TAIL_PID=$!

# Make sure we tear the background tail down on any exit path.
cleanup() {
    kill "$TAIL_PID" 2>/dev/null
    wait "$TAIL_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

# All the env vars from smoke-low-mem.sh; comments there explain each one.
DS4_METAL_NO_RESIDENCY=1 \
DS4_METAL_NO_MODEL_WARMUP=1 \
DS4_METAL_NO_PREFILL_KERNEL_WARMUP=1 \
DS4_METAL_PREFILL_SPLIT=1 \
DS4_METAL_DECODE_SPLIT_EVERY=1 \
DS4_METAL_MODEL_MAX_VIEW_BYTES="$DS4_METAL_MODEL_MAX_VIEW_BYTES_OVERRIDE" \
DS4_DIAG=1 \
./ds4 -m "$MODEL" \
      -c "$CTX" \
      -n "$TOKENS" \
      --temp 0 \
      -p "$PROMPT" \
  > "$STDOUT_LOG" \
  2> >(tee "$STDERR_LOG" | grep --line-buffered -E "$LIVE_FILTER" | awk '{ printf "[ds4]    %s\n", $0; fflush(); }' >&2)
rc=$?

# Give the background stdout tail a tick to flush trailing tokens.
sleep 0.3
cleanup

echo
echo "=== END   $(date +%T) rc=$rc ==="

# Quick post-mortem when something went sideways.  When rc=0 and the stream
# already showed the summary line, this is just confirmation.
if [ "$rc" -ne 0 ] || ! grep -q "prefill:.*generation:" "$STDERR_LOG"; then
    echo
    echo "----- non-clean exit; last 30 metal[] lines from full log -----"
    grep -E "ds4_diag: metal\[" "$STDERR_LOG" | tail -30
    echo "----- last 5 finish[] lines -----"
    grep -E "ds4_diag: finish\[" "$STDERR_LOG" | tail -5
    echo "----- any OOM / command-batch lines -----"
    grep -E "OutOfMemory|command batch failed" "$STDERR_LOG" | tail -10
fi

echo
echo "----- generated stdout (raw bytes via od) -----"
od -c "$STDOUT_LOG" | head -20

exit $rc
