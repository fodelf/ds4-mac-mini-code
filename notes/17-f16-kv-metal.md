# #17 F16 KV Metal storage flip — atomic patch landed

**Status:** landed (one commit, per design contract).  
**Goal:** halve compressed-KV Metal cache from F32 to F16, freeing ~6.8 GiB at 1M ctx on 16 GiB Mac Mini M4 so the 1M-context deliverable becomes physically possible.  
**Counterpart:** CPU side already stored compressed KV as F16 (`uint16_t`) in #16. After #17, Metal matches.

This file is the **one place** to read for: why #17 had to land atomically, the producer-pipeline redesign that made it tractable, every modified call site, and the post-#17 resource budget that motivates the next move.

---

## 1. Why atomic, not incremental

The compressor → quantize chain on Metal previously ran **in-place on a view of `g->layer_*_comp_cache[il]` typed as F32**:

```
compressor_update_tensor(cache_view)   // pool + rms_norm + rope_tail, F32 in cache
fp8_kv_quantize_tensor(cache_view)     // FP8 round-trip, F32 in cache
// or for indexer ratio=4:
indexer_qat_tensor(cache_view)         // Hadamard FP4, F32 in cache
```

If we had just halved the storage to F16 without rewiring those producers first, every one of those kernels would write 4-byte floats into 2-byte slots and read past the end. The build would still compile and run; tokens would silently drift. Per the standing constraint **"没有我明确的指示不要私自验证"** there is no validation window, so a "land storage now, fix kernels next" plan is unsafe — it cannot be caught before being shipped.

Conclusion: storage flip + producer redirect + consumer cast + stride flip + snapshot bytes + payload version bump = **one commit**.

---

## 2. Producer pipeline redesign

**Insight:** none of `pool / rms_norm / rope_tail / ratio4_shift / FP8 quantize / indexer QAT` needed kernel changes. They all take a dst buffer and operate in-place. The only thing that changed is *which* buffer the chain writes into — and the addition of a final F16 commit step.

### New pieces

1. **Per-graph F32 scratch tensor `g->comp_scratch_f32`** — sized at  
   `max_prefill_chunk × max(DS4_N_HEAD_DIM, DS4_N_INDEXER_HEAD_DIM) × sizeof(float)` ≈ 4 MiB worst case (2048 × 512 × 4).
2. **New Metal kernel `kernel_dsv4_f32_to_f16_store_rows`** in `metal/dsv4_kv.metal` — ~15 lines, just `dst[gid] = half(src[gid])`. This is the *single point* in DS4 where the half() round happens for compressed-KV cache writes.
3. **New host wrapper `ds4_gpu_dsv4_f32_to_f16_store_rows_tensor`** in `ds4_metal.m`.
4. **New Metal template instantiation `kernel_cpy_f16_f16`** in `metal/cpy.metal` (one line on top of the existing generic `kernel_cpy_t_t<T0,T1>` template) — needed by FlashAttention pack-in copies that now read an F16 source instead of F32.
5. **New host wrapper `ds4_gpu_encode_cpy_f16_f16_1d`** in `ds4_metal.m` + pipeline global `g_cpy_f16_f16_pipeline` + init/shutdown wiring.

### Caller pattern (was → is)

```
WAS:  compressor_*(cache_view, ...)
      fp8_kv_quantize_tensor(cache_view)   // or indexer_qat_tensor(cache_view)
                                            // cache_view writes F32 into F32 cache

IS:   compressor_*(scratch_view, ...)
      fp8_kv_quantize_tensor(scratch_view) // or indexer_qat_tensor(scratch_view)
      f32_to_f16_store_rows_tensor(cache, dst_row_offset, scratch_view, 0, n_rows, head_dim)
                                            // scratch is F32; cache is F16; commit kernel rounds
```

The three compressor entrypoints (`compressor_update_tensor`, `compressor_prefill_tensor`, `compressor_prefill_ratio4_replay_tensor`) now take a `scratch_dst` parameter instead of `comp_cache`. The `quantize_fp8` flag was dropped from `compressor_prefill_tensor` because callers now drive FP8 themselves on scratch.

### Producer call sites updated (6 total: 3 attn × 3 indexer)

Attn:
- decode (`ds4.c:9637` then `9669`)
- prefill zero_prefix (`ds4.c:11733`)
- prefill aligned ratio-4 replay (`ds4.c:11806`)
- prefill fallback per-token (`ds4.c:11919` + `11941`)

Indexer (ratio=4 layers, 20 of 43):
- decode (`ds4.c:9717` then `9749`)
- prefill zero_prefix (`ds4.c:12042` + `12069`)
- prefill aligned ratio-4 replay (`ds4.c:12127`)

---

## 3. Consumer side — read half\*, widen on use

5 consumer kernels in `metal/dsv4_misc.metal` were changed to read `half*` / `half4*` from comp/index cache and widen to float at the use site:

| Kernel | Was | Is |
|---|---|---|
| `kernel_dsv4_indexer_score_one_direct` | `device const float *krow` | `device const half *krow`, `float(krow[tid])` |
| `kernel_dsv4_indexed_mixed_attention_heads8` | `device const float4 *src` | `device const half4 *src`, `kv_shared[tid] = float4(src[tid])` |
| `kernel_dsv4_indexed_mixed_attention_heads8_rb4` | same shape | same |
| `kernel_dsv4_indexer_scores_tiled_f32` | `device const float *row` | `device const half *row`, `v = float(row[d])` |
| `kernel_dsv4_indexer_scores_tiled` | `device const float *row` | `device const half *row`, `v = half(row[d])` |

The widen happens at the load instruction — there is no extra cache-resident F32 copy.

---

## 4. Storage allocation + stride flips

### Allocations (ds4.c)

| Site | Was | Is |
|---|---|---|
| `ds4.c:8987` raw cache | `sizeof(float)` | unchanged (raw stays F32; SWA window of 128 rows is small) |
| `ds4.c:8995` `layer_attn_comp_cache[il]` | `sizeof(float)` | **`sizeof(uint16_t)`** |
| `ds4.c:9018` `layer_index_comp_cache[il]` | `sizeof(float)` | **`sizeof(uint16_t)`** |

`comp_scratch_f32` was newly added to the graph struct + allocator + free path; sized to cover the worst-case prefill row chunk.

### Strides (ds4_metal.m)

Every host-side `head_dim * sizeof(float)` that feeds a kernel's `comp_row_stride` / `index_row_stride` was flipped to `sizeof(uint16_t)`. Sites (the consumer entrypoints):

- `4637` `indexer_score_one_tensor` — comp_bytes + index_row_stride
- `4782` `indexer_scores_batch_tensor` — comp_bytes + index_row_stride
- `9199` `static_mixed_heads_nonvec`
- `9435` `static_mixed_heads_vec`
- `10110` `gathered_heads`
- `10629` `decode_mixed_batch_heads`
- `11091` `attention_indexed_mixed` (local `row_bytes_f16 = head_dim * sizeof(uint16_t)`)
- `11168` `comp_row_stride`
- `11367` `attention_decode_heads_tensor`

**raw_kv stride stays F32 — only comp/index stride flipped.**

### Producer-internal `comp_bytes` — intentionally stays `sizeof(float)`

Sites at `ds4_metal.m:7320`, `7673`, `8081` (compressor_update/prefill/replay_tensor) **must remain `sizeof(float)`** because the `comp_cache` parameter at those sites now receives a **scratch F32 view** (per the redesign in §2). This is correct; not a bug.

### FlashAttention pack-in copies — 4 sites switched

The FA prefill paths that pull comp_kv into a separate working buffer used `ds4_gpu_encode_cpy_f32_f16_1d`. With the source now F16, those become `ds4_gpu_encode_cpy_f16_f16_1d`:

- `ds4_metal.m:9254` `static_mixed_heads_nonvec`
- `ds4_metal.m:9489` `static_mixed_heads_vec`
- `ds4_metal.m:10215` `gathered_heads`
- `ds4_metal.m:10714` `decode_mixed_batch_heads`

---

## 5. Snapshot + diagnostic round-trip

### Snapshot bytes (ds4.c)

Four GPU save/load sites halved:

| Site | What |
|---|---|
| `~16657` save `attn_comp_cache` | `sizeof(uint16_t)` |
| `~16682` save `index_comp_cache` | `sizeof(uint16_t)` |
| `~16964` load `attn_comp_cache` | `sizeof(uint16_t)` |
| `~16992` load `index_comp_cache` | `sizeof(uint16_t)` |

### Payload version bump

`DS4_SESSION_PAYLOAD_VERSION` bumped `2 → 3` at `ds4.c:16142` with comment noting that v2 GPU snapshot files were written with twice as many bytes per row and **cannot** be loaded by a v3 runtime. v2 CPU snapshots are unaffected.

### Diagnostic `tensor_read` (ds4.c:14447, 14465)

The Metal-vs-CPU comp/index diff trace previously read F32 from the GPU cache. Now reads `uint16_t`, widens on host via `f16_to_f32` before the `max_abs_diff` / `rms_abs_diff` calls.

---

## 6. Resource budget @ 1,048,576 ctx, post-#17

### Invariants (ds4.c)

| Constant | Value | Source |
|---|---|---|
| `DS4_N_LAYER` | 43 | ds4.c:87 |
| `DS4_N_HEAD_DIM` | 512 | ds4.c:92 |
| `DS4_N_INDEXER_HEAD_DIM` | 128 | ds4.c:105 |
| `DS4_N_SWA` (raw window) | 128 | ds4.c:103 |
| `compress_ratio(il)` | 0:dense / even:4 / odd:128 | ds4.c:411 |
| `comp_cap` formula | `ctx/ratio + 2` | ds4.c:6401 |
| `raw_cap` | `min(SWA, ctx) = 128` always | ds4.c:6367 |
| GGUF on disk (imatrix IQ2XXS) | **80.77 GiB** | `ds4flash.gguf` symlink |

Layer breakdown:
- 2 dense (0, 1) — no comp cache
- 21 ratio=4 (even 2..42) — attn comp + index comp
- 20 ratio=128 (odd 3..41) — attn comp only

### KV cache total @ 1M ctx (post-#17)

| Component | Per-layer | Layers | Total |
|---|---|---|---|
| `raw_kv` F32 (128 × 512 × 4) | 256 KiB | 43 | **11 MiB** |
| `attn_comp_kv` F16 ratio=4 (262146 × 512 × 2) | 256.0 MiB | 21 | **5,376 MiB ≈ 5.25 GiB** |
| `attn_comp_kv` F16 ratio=128 (8194 × 512 × 2) | 8.00 MiB | 20 | **160 MiB** |
| `index_comp_kv` F16 (262146 × 128 × 2) | 64.0 MiB | 21 | **1,344 MiB ≈ 1.31 GiB** |
| `attn_state_kv/score`, `index_state_kv/score` | — | 41 | ~10 MiB |
| `comp_scratch_f32` | 4 MiB | 1 | 4 MiB |
| **KV subtotal** | | | **≈ 6.80 GiB** |

Pre-#17 the same row count in F32 would have been ≈ **13.6 GiB**. **#17 saves ≈ 6.8 GiB.** (Prior memory note had estimated ~3.4 GiB because it counted only `attn_comp_kv`; `index_comp_kv` got the F16 flip too in CPU side, and now Metal matches.)

### Always-resident model weights

| Class | Size | Why must stay |
|---|---|---|
| `token_embd` F16 (4096 × 129280 × 2) | 1009 MiB | input lookup every token |
| `output` Q8_0 (same dim × 1.0625) | 537 MiB | logit projection every token |
| 43 × attn (q_a/q_b/kv/out_a/out_b Q8_0) | ≈ 2.5 GiB | every token, all layers |
| 43 × shared expert (gate+up+down Q8_0 ≈ 26.6 MiB/layer) | ≈ 1.1 GiB | every token, all layers |
| 43 × (attn/indexer compressor_kv + gate weight F16) | ≈ 0.7 GiB | every step in compressed layers |
| `ffn_gate_tid2eid` (layers 0..2) | 9 MiB | per token in hash-routed layers |
| **Hot-weight subtotal** | **≈ 5.85 GiB** | |

### Routed experts (the bulk)

43 layers × 256 experts × (IQ2_XXS gate/up + Q2_K down) ≈ **86 GiB total on disk**. mmap'd, paged in on demand. Per-token activation is only 6/256 routed experts per layer, but routing distribution over a session covers a much wider footprint.

### 16 GiB balance sheet (resident)

| | GiB |
|---|---|
| Always-hot weights (above) | 5.85 |
| KV cache (above) | 6.80 |
| Activations / scratch / spec_logits | ~0.05 |
| System / runtime overhead (estimate) | ~0.5–1.0 |
| **Forced-resident subtotal** | **≈ 12.7 GiB** |
| **Remaining for routed-expert pages** | **≈ 3.3 GiB** |

Routed pool is 86 GiB; ~3.3 GiB can stay resident → expect persistent SSD page-in during decode in MoE-heavy layers.

---

## 7. What this unlocks vs. what's still needed

**Unlocked by #17:** the 1M-ctx KV cache went from "would not fit alongside any model state" to "fits in ~7 GiB, leaving the other half of RAM for weights + paging headroom."

**Still required for actual 1M-ctx decode:**

1. **Expert keep_map activation.** Without it, 86 GiB of routed weights chasing 3.3 GiB of resident space will thrash SSD. Target: keep ≤32–64 experts per layer (routed pool drops to ~10–21 GiB), letting the OS keep most of it resident.
2. **Optional: vocab trim** (see [vocab-trim-feasibility memory]). 96k → 64k saves ~500 MiB on `token_embd` + `output` + `ffn_gate_tid2eid`. Fragile (BPE merges, hash routing in layers 0–2); only after #17 + keep_map proves insufficient.
3. **Operational discipline:** batch=1, bounded `--max-tokens`, accept slower throughput when paging dominates. The CPU backend is reference-only on macOS (has kernel-panicked the box twice).

---

## 8. Files touched in #17

```
ds4.c
  +  16142  PAYLOAD_VERSION 2 → 3 (with comment about v2 GPU files)
  ±  14420–14478  diagnostic tensor_read: uint16_t buffers + host f16→f32 widen
  ±  8985/8993/9016  cache allocation sizeof(float) → sizeof(uint16_t)
  +  comp_scratch_f32 graph struct field + alloc/free
  ±  16657/16682/16964/16992  snapshot save/load bytes halved
  ±  6 producer caller sites rewired to scratch (decode + prefill, attn + indexer)

ds4_metal.m
  +  g_cpy_f16_f16_pipeline global + init/shutdown wiring (~49, ~2950, ~4128)
  +  ds4_gpu_encode_cpy_f16_f16_1d (forward decl ~1543, definition ~8803)
  +  ds4_gpu_dsv4_f32_to_f16_store_rows_tensor (wrapper for new kernel)
  ±  3 compressor wrappers: comp_cache param → scratch_dst; quantize_fp8 dropped from prefill
  ±  9 consumer entrypoint comp_bytes/index_bytes flipped to sizeof(uint16_t)
  ±  3 row-stride struct fields flipped to sizeof(uint16_t)
  ±  4 FA pack-in copies switched f32_f16 → f16_f16

metal/cpy.metal
  +  template instantiation: kernel_cpy_f16_f16

metal/dsv4_kv.metal
  +  ds4_metal_args_dsv4_f32_to_f16_store_rows struct
  +  kernel_dsv4_f32_to_f16_store_rows

metal/dsv4_misc.metal
  ±  5 consumer kernels: float*/float4* → half*/half4* with widen at load
```

---

## 9. Validation gating (standing constraint)

No build, no `ds4-bench`, no `./ds4` run was performed for #17. Per the user's standing rule **"没有我明确的指示不要私自验证，所有验证之前必须计算可行性"**, the next runtime touch waits for an explicit user go-ahead. When that opens, the natural smoke sequence is:

1. `make clean && make` — confirms cpy.metal template + new kernel + wrapper + 5 consumer kernels + ds4.c version bump all compile.
2. Single short prompt at small ctx — confirms the producer-pipeline scratch redirect doesn't drift tokens.
3. `--dump-logprobs` at ~2k ctx vs. a pre-#17 baseline run — confirms F16 quantize round-trip stays within FP8/QAT noise floor.
4. `ctx_probe` (already done in #18) re-run to confirm the new 6.8 GiB number matches what the OS reports.

Old v2 GPU snapshot files will fail-fast on load (intentional: payload version bump). CPU-side v2 snapshots are unaffected because CPU was already F16.
