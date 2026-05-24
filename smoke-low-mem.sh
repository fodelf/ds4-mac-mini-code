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
# Default to the K=16 shrunken GGUF (~13.7 GiB, expected 2 mmap views @ ~6.85
# GiB each).  K=48 (22 GiB / 3 views) reached layer 15/16 then OOM'd because the
# wired model-view union + file_backed pages saturated the 16 GiB.  K=16 trades
# routing quality for headroom: top-6 router picks from 16 kept experts instead
# of 256.  Quality regression is intentional and temporary — first goal is to
# prove a token can emit at all on this hardware.  See notes/execution-log.md #50.
MODEL="${DS4_MODEL:-./gguf/ds4flash-k16.gguf}"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${DS4_SMOKE_OUT_DIR:-/tmp}"
STDERR_LOG="$OUT_DIR/ds4-smoke-$STAMP.stderr.log"
STDOUT_LOG="$OUT_DIR/ds4-smoke-$STAMP.stdout.log"
VMSTAT_BEFORE="$OUT_DIR/ds4-smoke-$STAMP.vmstat-before.txt"
VMSTAT_AFTER="$OUT_DIR/ds4-smoke-$STAMP.vmstat-after.txt"

echo "smoke: ctx=$CTX tokens=$TOKENS prompt=\"$PROMPT\" model=$MODEL"
echo "smoke: stderr -> $STDERR_LOG"
echo "smoke: stdout -> $STDOUT_LOG"

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
: "${DS4_METAL_MODEL_MAX_VIEW_BYTES_OVERRIDE:=3758096384}"  # 3.5 GiB
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
  2> >(tee "$STDERR_LOG" >&2)
rc=$?
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
