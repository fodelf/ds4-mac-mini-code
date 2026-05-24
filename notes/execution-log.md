# DS4 execution log

Append-only chronological record of significant DS4 work — design docs, atomic patches, validation steps, scope decisions, and any material blocker. Each entry links to the detailed artifact (`notes/`, source file + line, memory file). Skim §Most recent for current state.

**Rules for this file:**
- Append-only. Do not rewrite past entries.
- Each entry: date + headline + bullet points (what / why / where / status).
- Link to artifacts, do not duplicate their content. The log points; the artifacts contain.
- Status flags: `landed` / `in-progress` / `pending-authorization` / `deferred` / `blocked`.

---

## Most recent

- **2026-05-24** — #34 Lever A validation gate 1+2 passed (clean build green, `--metal-kernels` OK on Apple M4). Smoke + ctx_probe pending explicit user authorization.

## 2026-05-24 — #34 Lever A validation step 1+2 (build + metal-kernels)
- **Step 1 — clean build:** `make clean && make` produced all 5 binaries (ds4, ds4-server, ds4-bench, ds4-eval, ds4-agent) and `ds4_test` without errors. One unused-function warning: `ds4_gpu_encode_cpy_f16_f16_1d` in `ds4_metal.m:8952` — this is the post-#17 F16-to-F16 1D copy that the static-mixed / decode-mixed-batch FA paths used to dequant-copy compressed rows; Lever A replaces those call sites with `ds4_gpu_encode_dsv4_fp8_rows_to_f16`, so the helper is dead code. Cleanup deferred; not a correctness signal.
  - **Fix landed during validation:** the `DSV4_FP8_ATTN_ROW_BYTES` macro was originally placed beside the payload-version block at `ds4.c:16228` but its earliest uses are at `ds4.c:9033` (graph allocation), `9729` / `11961` / `12052` (commit calls), and `14500` (host-side decode). C is single-pass; the macro was undeclared at the use sites and the compile failed with 5 errors. Macro moved to right after the `DS4_N_*` enum block (~line 111), with the duplicate definition removed and replaced by a one-line cross-reference comment. Build then went green.
- **Step 2 — Metal kernel numerics:** `./ds4_test --metal-kernels` reported `metal-kernels: OK`. This isolated-kernel suite exercises the new `kernel_dsv4_f32_to_fp8_store_rows` (commit) and `kernel_dsv4_fp8_rows_to_f16` (FA pack-in dequant-copy) numerics against CPU reference values; green means the FP8 round-trip is consistent with the host-side decode helper bit-by-bit on Apple M4.
- **Hardware:** Apple M4, 16 GiB unified memory. Model: not loaded for `--metal-kernels` (it uses tiny synthetic tensors).
- **Status:** validation steps 1+2 landed; steps 3 (small-ctx smoke) and 4 (ctx_probe) are gated on explicit user authorization.

## 2026-05-24 — #34 Lever A atomic patch landed (code-complete, validation gated)

---

## 2026-05-24 — #34 Lever A atomic patch landed (code-complete, validation gated)
- **Scope:** Metal `attn_comp_kv` storage flipped from F16 (`head_dim * 2` = 1024 B/row) to FP8 E4M3FN + per-64-block scale + F16 RoPE tail = **608 B/row**. CPU reference path untouched. Indexer `index_comp_kv` untouched (Lever B is the indexer sibling).
- **What changed (code):**
  - New Metal kernels in `metal/dsv4_kv.metal`: `kernel_dsv4_f32_to_fp8_store_rows` (commit) and `kernel_dsv4_fp8_rows_to_f16` (FA pack-in dequant-copy).
  - New wrappers in `ds4_metal.m`: `ds4_gpu_dsv4_f32_to_fp8_store_rows_tensor` (~line 6085) and `ds4_gpu_dsv4_fp8_attn_row_bytes` (~6075); declared in `ds4_gpu.h`.
  - Heads8 attention kernels in `metal/dsv4_attn.metal` read FP8 rows inline (in-kernel dequant); static-mixed / gathered / decode-mixed-batch FA paths get a per-batch `dsv4_fp8_rows_to_f16` pack-in pass that materialises the flat F16 working buffer the FA kernels already consume.
  - `ds4.c` storage alloc, all 4 commit sites (1 decode + 3 prefill: aligned-chunk prefill, zero-prefix prefill, ratio4 replay), and per-token unaligned path: dropped the legacy in-scratch FP8 quantize → swap to `f32_to_fp8_store_rows_tensor` with `DSV4_FP8_ATTN_ROW_BYTES` stride (`ds4.c:8995, 9727, 11835, 11959, 12050`).
  - Snapshot save/load + budget aligned to the new row size — `session_payload_live_tensor_bytes` (16332-16348) + GPU save (~16743) + GPU load (~17050) all use `DSV4_FP8_ATTN_ROW_BYTES`. Side-effect: pre-existing `sizeof(float)` vs `sizeof(uint16_t)` mismatch in the GPU budget for attn comp (and indexer) is now resolved; v4 GPU budget matches v4 GPU writes exactly.
  - Host-side decoder `dsv4_fp8_attn_row_decode_cpu` (`ds4.c:1755-1786`) added so `tensor_read` diagnostic can interpret 608-byte rows as F32.
  - Debug dump (`ds4.c:11831-11838`) redirected to F32 scratch so existing F32-typed dump doesn't misread FP8 bytes.
- **Bit-identicality argument (why this is quality-neutral):** scale = `exp2(integer)`; FP8 normal magnitudes fit exactly in F16; therefore `F16(FP8_value × scale) == FP8_value × scale` exactly. GPU stores `FP8(x/scale)` + scale; CPU diagnostic stores `F16(FP8(x/scale) × scale)`; both decode to bit-identical floats for any input. RoPE remains F16 (no quantisation introduced).
- **Payload version bump:** `DS4_SESSION_PAYLOAD_VERSION` v3 → v4. v3 GPU snapshots intentionally fail-fast on a v4 runtime (existing header check at `ds4.c:16806`). CPU snapshots unaffected (CPU reference cache is still F16).
- **Saving (vs post-#17):** ATTN comp 1024 → 608 bytes/row = 41% reduction on that tensor (not the 50% from the v1 budget memo — RoPE F16 retention is what makes the row 608 not 512). Indexer comp_kv unchanged in Lever A → see Lever B.
- **Files:** `metal/dsv4_kv.metal`, `metal/dsv4_attn.metal`, `metal/dsv4_misc.metal`, `ds4_metal.m`, `ds4.c`, `ds4_gpu.h`.
- **Status:** code-complete; no build / no smoke / no ctx_probe (standing rule "没有我明确的指示不要私自验证"). Validation pending explicit user authorization.

## 2026-05-24 — Execution log created (this file)
- **Why:** user requested project-local execution log for post-hoc issue diagnosis ("把重要执行日志本地化当前项目，方便后续定位问题").
- **Scope going forward:** every design memo / atomic patch / validation step / material scope decision / blocker gets a dated entry here.
- **Cross-session enforcement:** feedback memory `execution-log-requirement` saved.

## 2026-05-24 — #34 Lever A design memo (in-progress)
- **Goal:** design memo for atomic patch that flips Metal attn_comp_kv storage from F16 (post-#17) to FP8 + per-row scale. Mirror template of `notes/17-f16-kv-metal.md`.
- **Survey data gathered (post-#17 state):**
  - GPU FP8 quantize kernel `kernel_dsv4_fp8_kv_quantize_f32` — `metal/dsv4_kv.metal:113`. Currently writes back F32 in scratch (after #17 redesign).
  - Alt store kernel `kernel_dsv4_kv_fp8_store_f32` — `metal/dsv4_kv.metal:217`. Takes explicit `fp8_scale` arg; may be reusable as the new commit kernel.
  - GPU quantize wrapper `ds4_gpu_dsv4_fp8_kv_quantize_tensor` — `ds4_metal.m:5940`. Two producer call sites: `ds4_metal.m:7620`, `7893`.
  - CPU side `dsv4_fp8_kv_quantize_row_inplace_cpu` — `ds4.c:1727`. 7 caller sites across compress paths.
  - #17 commit kernel `kernel_dsv4_f32_to_f16_store_rows` — `metal/dsv4_kv.metal:319`; wrapper `ds4_metal.m:6033`. Lever A needs a sibling `_f32_to_fp8_store_rows` kernel.
  - 9 consumer-stride sites at `sizeof(uint16_t)` (Lever A flips them to `sizeof(uint8_t) + scale`): `ds4_metal.m:4637, 4782, 9199, 9435, 10110, 10629, 11091/11168, 11367`.
  - Storage allocations: `ds4.c:8995` (attn comp), `ds4.c:9018` (index comp, **untouched by A — that's Lever B**).
- **Open design questions to settle in the memo:**
  - Per-row vs per-block scale layout (cache locality vs producer simplicity).
  - Where in the row to embed the scale (head vs side buffer) — Apple SIMD-group alignment constraints.
  - Snapshot v3 → v4 byte layout; v3 GPU snapshots intentionally fail-fast on v4 load.
- **Status:** survey complete, memo writing next.

## 2026-05-24 — Roadmap v2 quality-first finalized (#33)
- **File:** `notes/optimization-roadmap.md` (replaces previous Phase 1-5 framework; v1 paper citations preserved in §8).
- **Memory:** `optimization_roadmap_framework.md` updated.
- **Constraint hierarchy:** 1M ctx + 输出质量 are hard; speed deferred per user ("其他项可以再后续优化").
- **Three quality-neutral DS4-native levers:**
  - A: FP8 real-storage attn_comp_kv (skip dequant) — saves 2.71 GiB
  - B: FP4 real-storage index_comp_kv (skip dequant) — saves 0.98 GiB
  - C: Indexer-gated disk-tier ratio-4 attn_comp_kv — saves 2.58 GiB resident, costs ~5.4 MiB/token SSD
- **After A+B+C:** KV 6.80→0.50 GiB; routed/resident ratio 37×→10×; estimated 1 tok/s decode floor (cold expert paging dominant).
- **Deferred lossy levers** (require explicit user re-authorization): Phase 2 (REAP), Phase 4 (MoBiLE), Phase 5 (vocab trim).
- **Implementation order:** #34 design A → #35 patch A → validate → #36/#37 B → #38/#39 C.
- **Status:** approved by user ("开始实施"); #34 underway.

## 2026-05-24 — Resource budget evaluated post-#17 @ 1M ctx
- **File:** `notes/17-f16-kv-metal.md §6`.
- **Result:** forced-resident ≈ 13.7 GiB on 16 GiB; free for routed pool ≈ 2.3 GiB; routed pool 86 GiB. Ratio 37× = SSD-thrash zone.
- **Conclusion:** 1M ctx not viable without further work — motivated the optimization roadmap.

## 2026-05-24 — #17 F16 KV Metal storage flip landed (atomic)
- **File:** `notes/17-f16-kv-metal.md` (full atomic-patch record + post-#17 budget).
- **What changed:** Metal `attn_comp_kv` + `index_comp_kv` storage F32 → F16; producer pipeline redesigned with F32 scratch + half-precision commit kernel; 5 consumer kernels read `half*` with widen-at-load; snapshot bytes halved; payload version v2 → v3 (v2 GPU snapshots fail-fast).
- **Saves:** KV cache 13.6 → 6.80 GiB @ 1M ctx.
- **Status:** code landed; runtime validation pending explicit user authorization per standing rule.

---

## Pending atomic actions (require explicit user authorization)

- Build + smoke for #17 (Mac Mini M4).
- 1M ctx ctx_probe validation.
- Any phase implementation past design-memo stage.
