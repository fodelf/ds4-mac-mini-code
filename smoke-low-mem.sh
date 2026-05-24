#!/bin/bash
# smoke-low-mem.sh — minimal-risk smoke for 16 GiB Mac Mini M4.
#
# Skips the 86 GiB residency-set request and the warmup kernel that
# faults the whole GGUF into memory.  Without those env vars, model
# load + first-token decode triggers the macOS watchdog kernel panic
# observed on 2026-05-24 (see notes/execution-log.md).
#
# Usage: ./smoke-low-mem.sh [CTX] [TOKENS] [PROMPT]
# Defaults are the safest known values: ctx=1024, tokens=1, prompt="Hi".

set -u

CTX="${1:-1024}"
TOKENS="${2:-1}"
PROMPT="${3:-Hi}"
# Default to the full DeepSeek V4 Flash GGUF (81 GB).  K=16/K=24 shrunken
# variants were deleted after #54 confirmed A1 PoC works end-to-end with all
# 256 experts intact.  Going forward this script + smoke-watch.sh validate
# the Path C / #55 A3 pre-pack scratch architecture against the real model.
MODEL="${DS4_MODEL:-./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf}"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${DS4_SMOKE_OUT_DIR:-/tmp}"
STDERR_LOG="$OUT_DIR/ds4-smoke-$STAMP.stderr.log"
STDOUT_LOG="$OUT_DIR/ds4-smoke-$STAMP.stdout.log"
VMSTAT_BEFORE="$OUT_DIR/ds4-smoke-$STAMP.vmstat-before.txt"
VMSTAT_AFTER="$OUT_DIR/ds4-smoke-$STAMP.vmstat-after.txt"

# Optional prompt-file override (same convention as smoke-watch.sh).
if [ -n "${DS4_PROMPT_FILE:-}" ]; then
    if [ ! -f "$DS4_PROMPT_FILE" ]; then
        echo "smoke: DS4_PROMPT_FILE=$DS4_PROMPT_FILE does not exist" >&2
        exit 2
    fi
    PROMPT_ARG=(--prompt-file "$DS4_PROMPT_FILE")
    PROMPT_DESC="file=$DS4_PROMPT_FILE ($(wc -c < "$DS4_PROMPT_FILE") bytes)"
else
    PROMPT_ARG=(-p "$PROMPT")
    PROMPT_DESC="prompt=\"$PROMPT\""
fi

echo "smoke: ctx=$CTX tokens=$TOKENS $PROMPT_DESC model=$MODEL"
echo "smoke: stderr -> $STDERR_LOG"
echo "smoke: stdout -> $STDOUT_LOG"
echo "smoke: Ctrl+C 干净中断 (会 SIGTERM ds4，最多等 3 s 再 SIGKILL，然后跑 summary)"

vm_stat > "$VMSTAT_BEFORE"

echo "=== START $(date +%T) ==="
# DS4_METAL_PREFILL_SPLIT routes short-prompt prefill through the per-layer CB
# path so the commit does not have to wire all 43 layers at once.
# DS4_METAL_DECODE_SPLIT_EVERY=1 flushes after every decode layer so the first
# decode token's CB only wires one layer at a time.
# DS4_METAL_NO_PREFILL_KERNEL_WARMUP=1 skips the HC-attention warmup matmul
# that fires for n_tokens>8.  On 16 GiB Mac Mini that warmup wires ~9 GiB of
# model-view buffer that never releases — leaving no headroom for the actual
# per-layer prefill CBs (observed at smoke run 2026-05-24 19:01).
# DS4_METAL_MODEL_MAX_VIEW_BYTES caps each mmap shared-buffer view so IOGPU
# touched-pages-per-resource is bounded.  K=16 / 2 views @ 6.5 GiB OOMs at
# layer 21 (first layer whose tensors are in view 1).  3.5 GiB ≈ 5 views,
# expected OOM around layer ~34; if that's a clean win we shrink further.
# Set DS4_METAL_MODEL_MAX_VIEW_BYTES_OVERRIDE to disable, or to a different
# byte cap (min 256 MiB).
: "${DS4_METAL_MODEL_MAX_VIEW_BYTES_OVERRIDE:=2147483648}"  # 2 GiB; see #54-patch2
# #55 A3 default-on: CPU-side memcpy active experts into resident scratch so
# per-CB wireable peak is bounded by scratch size (~1.728 GiB) rather than
# however many mmap-view buffers a layer's tensors happen to straddle.
# Override with DS4_METAL_EXPERT_OFFLOAD=0 to A/B test the original path.
: "${DS4_METAL_EXPERT_OFFLOAD:=1}"
DS4_PID=""
INTERRUPTED=0

on_interrupt() {
    INTERRUPTED=1
    echo
    echo "=== INTERRUPT $(date +%T) — forwarding SIGTERM to ds4 (PID=$DS4_PID) ==="
    if [ -n "$DS4_PID" ]; then
        kill -TERM "$DS4_PID" 2>/dev/null
        for _ in 1 2 3 4 5 6; do
            kill -0 "$DS4_PID" 2>/dev/null || break
            sleep 0.5
        done
        if kill -0 "$DS4_PID" 2>/dev/null; then
            echo "smoke: ds4 didn't respond to SIGTERM in 3 s — SIGKILL"
            kill -KILL "$DS4_PID" 2>/dev/null
        fi
    fi
}
trap on_interrupt INT TERM

# Background ds4 so the wait is signal-interruptible; otherwise bash defers
# trap delivery until the foreground child exits on its own.
DS4_METAL_NO_RESIDENCY=1 \
DS4_METAL_NO_MODEL_WARMUP=1 \
DS4_METAL_NO_PREFILL_KERNEL_WARMUP=1 \
DS4_METAL_PREFILL_SPLIT=1 \
DS4_METAL_DECODE_SPLIT_EVERY=1 \
DS4_METAL_MODEL_MAX_VIEW_BYTES="$DS4_METAL_MODEL_MAX_VIEW_BYTES_OVERRIDE" \
DS4_METAL_EXPERT_OFFLOAD="$DS4_METAL_EXPERT_OFFLOAD" \
DS4_DIAG=1 \
./ds4 -m "$MODEL" \
      -c "$CTX" \
      -n "$TOKENS" \
      --temp 0 \
      "${PROMPT_ARG[@]}" \
  > "$STDOUT_LOG" \
  2> >(tee "$STDERR_LOG" >&2) &
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
echo "=== END   $(date +%T) rc=$rc ==="

vm_stat > "$VMSTAT_AFTER"

# ---- summary ----
echo
echo "----- generated token(s) -----"
cat "$STDOUT_LOG"
echo
echo "----- wired / free / file-backed pages (16 KiB each) -----"
printf "before: "; awk '/wired down/{w=$NF} /Pages free/{f=$NF} /File-backed/{fb=$NF} END{print "wired="w" free="f" file_backed="fb}' "$VMSTAT_BEFORE"
printf "after : "; awk '/wired down/{w=$NF} /Pages free/{f=$NF} /File-backed/{fb=$NF} END{print "wired="w" free="f" file_backed="fb}' "$VMSTAT_AFTER"
echo
echo "----- env-var verification (residency/warmup MUST both be ~0 ms) -----"
grep -E "residency requested|warmup" "$STDERR_LOG" || echo "WARN: env vars may not have taken effect"
echo "----- prefill commit pattern (look for 43 separate 'gpu prefill layer' prints, no command-batch OOM) -----"
grep -E "gpu prefill layer|command batch failed" "$STDERR_LOG" | tail -5
echo "----- ds4_diag setup-phase trace (alloc / steering / upload / warmup / prefill_layer_major / split-loop entry) -----"
grep -E "ds4_diag: (alloc_raw_cap|load_directional_steering|upload_prompt_(tokens|embeddings_hc)|warmup_prefill_kernels|prefill_layer_major|split-loop entering)" "$STDERR_LOG"
echo "----- ds4_diag CB commit trace (each begin/commit and the vmstat around it) -----"
grep -E "ds4_diag: (begin_commands|flush_commands|end_commands|vmstat\[)" "$STDERR_LOG" | head -60
echo "----- #51 Metal driver-side state (cb_total / cb_alive / drv MiB / transient high / pipelines / model_views) -----"
# Show first 6 (init + first few CBs) and last 12 (right before OOM) so the
# growth pattern is visible without dumping the whole log.
echo "  -- first 6 metal[...] lines:"
grep -E "ds4_diag: metal\[" "$STDERR_LOG" | head -6
echo "  -- last 12 metal[...] lines (incl. OOM neighborhood):"
grep -E "ds4_diag: metal\[" "$STDERR_LOG" | tail -12
echo "----- #51 finish[...] (per-CB transient drop count + alive after) -----"
grep -E "ds4_diag: finish\[" "$STDERR_LOG" | tail -8

exit $rc
