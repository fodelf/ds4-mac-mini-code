# Push-limit roadmap v2 — quality-first @ 1M ctx, 16 GiB Mac Mini M4

**Status:** plan only. No runtime, no patches landed for any phase in this file.
**Hard constraints (user):**
1. **1M context window** — non-negotiable
2. **Output quality** — non-negotiable (no lossy levers unless explicitly authorized)
3. Speed — soft, optimize later

**Standing rule:** "没有我明确的指示不要私自验证，所有验证之前必须计算可行性，当前系统不会崩溃"

**Sibling docs:**
- [17-f16-kv-metal.md](17-f16-kv-metal.md) — already-landed #17 atomic patch (F16 Metal KV) and post-#17 budget. This file's starting point.
- Prior version of this file (Phase 1-5 framework) is superseded — paper citations preserved in §8.

---

## 0. Starting point (post-#17)

| Resident class | GiB |
|---|---|
| KV cache @ 1M ctx (F16 attn_comp + F16 index_comp) | 6.80 |
| Hot weights (attn + shared + embed/out + compressor) | 5.85 |
| Activations + scratch + system overhead | ~1.0 |
| **Forced-resident subtotal** | **~13.7** |
| Physical RAM | 16.0 |
| **Free for routed-expert pages** | **~2.3** |
| Routed pool (43 layers × 256 experts) | **86 (on disk)** |

Routed pool / resident ratio = **37×**. Without further work this is in the SSD-thrash zone where a 1M-ctx decode cannot make forward progress.

---

## 1. DS4-native KV mechanism inventory

The previous roadmap missed three DS4 architectural features that enable quality-neutral savings. Inventory:

| Mechanism | Location | What it does | Already exploited? |
|---|---|---|---|
| **MLA** | Architecture (DS4) | 28× KV compression via low-rank latent + RoPE-tail | ✓ baseline |
| **Ratio-4 / 128 compressors** | ds4.c:411 | 21 layers compress 4:1, 20 layers compress 128:1 | ✓ baseline |
| **SWA raw window** | DS4_N_SWA=128 | Recent 128 raw tokens kept per layer (only 11 MiB total) | ✓ baseline |
| **Indexer top-K=512 gather** | ds4.c:106, metal/dsv4_misc.metal:561 | `kernel_dsv4_indexed_mixed_attention_heads8` reads ONLY 512 selected rows / 262146 per ratio-4 layer per token (0.2% sparsity) | ✗ **not exploited for storage** |
| **FP8 E4M3FN QAT (attn)** | metal/dsv4_kv.metal | Producer quantizes to FP8 then dequantizes back to F16 for storage | ✗ **bytes wasted** |
| **FP4 E2M1FN Hadamard QAT (indexer)** | metal/dsv4_kv.metal:165 | Producer applies Hadamard + FP4 then dequantizes back to F16 for storage | ✗ **bytes wasted** |
| **#17 scratch+commit producer** | ds4_metal.m | F32 scratch → quantize → half-precision commit kernel | ✓ in place, ready for FP8/FP4 reuse |
| **mmap-backed MTLBuffer** | ds4_metal.m:479 | `newBufferWithBytesNoCopy` wraps mmap'd GGUF directly | ✓ for weights, ✗ **not for KV** |
| **ds4_kvstore** | ds4_kvstore.c | Session-persistence disk cache (full snapshots) | not applicable to live tiering |

**Key insight:** DS4's `kernel_dsv4_indexed_mixed_attention_heads8` is a **gather** (reads selected rows via indices) not a mask (reads all rows, ignores masked). This means the storage for unselected rows is dead weight on every decode step — they only matter at the moment they're produced (write side) and on rare topk hits later. That asymmetry is what unlocks the disk-tier scheme below.

---

## 2. Three quality-preserving levers

All three are bit-identical to the current runtime's actual computed values (no new lossy step). The savings come from removing waste, not adding loss.

### Lever A — FP8 real-storage `attn_comp_kv`

**What:** Producer already runs `fp8_kv_quantize` (E4M3FN) on the compressed-attention row, then immediately dequantizes back to F16 for the commit kernel. Skip the dequant, store the FP8 byte + per-64-element block scale, dequant at consumer read. The 64-element RoPE tail stays F16 (RoPE is precision-sensitive and is *not* what the FP8 round-trip was acting on in the pre-A pipeline).

**Row layout (DS4 head_dim=512, n_rot=64):** 28 B per-block scales + 4 B pad + 448 B FP8 (E4M3FN) + 128 B F16 RoPE = **608 B/row** (vs 1024 B at F16). See `metal/dsv4_kv.metal:54`.

**Why quality-neutral:** the actual numeric values seen by every downstream attention head are already the FP8-quantized-then-dequantized values for the non-RoPE prefix; RoPE values are unchanged. Storing FP8 directly cannot change them. The only change is byte width.

**Saves (corrected against v1 memo — RoPE F16 carve-out matters):** ratio=4 5.25 GiB → **3.12 GiB**; ratio=128 160 MiB → **97 MiB**. Total **−2.19 GiB** (v1 memo said −2.71 GiB and implicitly assumed a 512 B/row FP8 layout; the implementation keeps RoPE in F16, so the row is 608 B not 512 B → ~41% per-row saving rather than 50%).

### Lever B — FP4 real-storage `index_comp_kv`

**What:** Indexer producer runs Hadamard-128 rotation + FP4 (E2M1FN) activation simulation, then dequantizes to F16. Skip the dequant, store FP4 packed (2 elements/byte) + per-block scale.

**Why quality-neutral:** same argument as Lever A — indexer values already in FP4 regime. Storing them as FP4 is lossless relative to current behavior.

**Saves:** 1.31 GiB → 0.33 GiB. Total **−0.98 GiB**.

### Lever C — Indexer-gated disk-tier ratio-4 `attn_comp_kv`

**What:** The dominant KV mass is the 21 ratio-4 layers' `attn_comp_kv` (after Lever A, still 2.63 GiB). Per decode token, only 512/262146 rows per layer (0.2%) are actually read by the gather kernel. Keep the full cache on a disk-backed file; maintain a small RAM hot-row cache; on each token, page in (or hit hot cache) the 512 selected rows per layer into a per-layer Metal staging buffer.

**Why quality-neutral:** the bytes computed and the bytes consumed are identical; only their storage location changes. No re-quantization, no row dropping, no sparsity approximation.

**Saves:** 2.63 GiB → ~50 MiB resident (5000-row LRU per layer × 21 layers × 512 byte/row at FP8 ≈ 53 MiB). Total **−2.58 GiB**.

**I/O cost (this is the price for the save):**
- Per token: 21 layers × 512 rows × 512 byte/row (FP8) ≈ **5.4 MiB** read
- Recent rows hit OS page cache (just-written) → most cost is on cold/distant rows
- M4 SSD: ~3 GB/s sequential, ~100-200 MB/s 4 KB random — cold-row scattered reads at staging granularity ≈ **30-100 ms/token KV latency**

---

## 3. Resource budget — per-phase

**Corrigendum (2026-05-24, post-#34 landing):** the +A column reflects the implemented row layout (608 B/row including F16 RoPE tail), not the v1 memo's 512 B/row assumption. Forced-resident at +A is ~11.5 GiB, not 10.9 GiB. 1M-ctx-with-only-A is meaningfully tighter than the v1 memo claimed — Levers B and C are still load-bearing for the routed pool headroom.

### 3.1 KV cache detail @ 1M ctx (forced-resident component)

| Component | Layers | Per-layer | Now (post-#17) | +A | +A+B | +A+B+C |
|---|---|---|---|---|---|---|
| `raw_kv` F32 (SWA=128 × 512 × 4) | 43 | 256 KiB | 11 MiB | 11 MiB | 11 MiB | 11 MiB |
| `attn_comp_kv` ratio=4 (262146 × 608 B/row at +A) | 21 | 256 MiB @F16 → 152 MiB @FP8+RoPE-F16 | 5.25 GiB | **3.12 GiB** | 3.12 GiB | **~61 MiB hot LRU** |
| `attn_comp_kv` ratio=128 (8194 × 608 B/row at +A) | 20 | 8.00 MiB @F16 → 4.87 MiB @FP8+RoPE-F16 | 160 MiB | **97 MiB** | 97 MiB | 97 MiB |
| `index_comp_kv` (262146 × 128) | 21 | 64 MiB @F16 → 16 MiB @FP4 | 1.31 GiB | 1.31 GiB | **0.33 GiB** | 0.33 GiB |
| Indexer/attn state + scratch | — | — | ~14 MiB | ~14 MiB | ~14 MiB | ~14 MiB |
| FP8/FP4 scale tables (block scales now in-row for ATTN) | — | — | 0 | (in-row) | ~2 MiB | ~2 MiB |
| **KV resident subtotal** | | | **6.80 GiB** | **4.56 GiB** | **3.58 GiB** | **~0.52 GiB** |

### 3.2 16 GiB balance sheet

| Bucket | Now | +A | +A+B | +A+B+C |
|---|---|---|---|---|
| Hot weights (token_embd + output + attn + shared + compressor + gate F16) | 5.85 | 5.85 | 5.85 | 5.85 |
| KV resident (above) | 6.80 | 4.56 | 3.58 | 0.52 |
| Activations + scratch + spec_logits | ~0.05 | ~0.05 | ~0.05 | ~0.05 |
| System / runtime overhead | ~1.0 | ~1.0 | ~1.0 | ~1.0 |
| **Forced-resident subtotal** | **~13.7** | **~11.5** | **~10.5** | **~7.4** |
| Free for routed-expert pages | **~2.3** | **~4.5** | **~5.5** | **~8.6** |
| Routed pool (mmap'd on disk) | 86 | 86 | 86 | 86 |
| **Routed / resident ratio** | **37×** | **19×** | **16×** | **10×** |

### 3.3 Verdict per the user's two hard constraints

| Constraint | Now | +A | +A+B | +A+B+C |
|---|---|---|---|---|
| 1M ctx physically fits | ✗ (over) | ✓ tight (~4.5 GiB headroom) | ✓ (~5.5 GiB headroom) | ✓ comfortable |
| Output quality preserved | ✓ | ✓ (FP8 already paid) | ✓ (FP4 already paid) | ✓ (bytes move, value bit-identical) |
| Decode forward progress (≥1 tok/s) | ✗ (thrash) | thrashy | borderline | **yes** (paging localized; cold expert SSD is dominant cost) |

After A+B+C: **both hard constraints satisfied**; speed is the residual variable. After A only, the routed-pool/free ratio is ~19× — Levers B and C are still required for the speed-floor target.

---

## 4. Speed-floor analysis (honest)

Two SSD-paging sources after A+B+C:

**Source 1 — Routed expert cold reads.** Per token: 43 layers × 6 routed experts ≈ 258 expert activations. Each expert ≈ 8 MiB (IQ2_XXS + Q2_K). At 8.6 GiB / 86 GiB ≈ 10% hot residency, ~232 reads/token miss = ~1.85 GiB/token. SSD ≈ 2 GB/s effective for 8 MiB sequential-ish reads → **~900 ms/token from routed expert paging.**

**Source 2 — Lever C disk-tier KV.** ~5.4 MiB/token, scattered across hot cache + cold disk. With page-cache warmth on recent rows, effective ≈ 1-2 MiB/token from SSD → **~30-100 ms/token.**

**Combined decode rate: ~1 tok/s** at full quality. Below 1 tok/s if routing is unusually broad; above 2 tok/s if routing concentrates on hot experts.

Prefill is much worse per-token but tokens-per-step amortizes — not separately modeled here.

This is the speed floor under the user's quality constraint. **If the user later allows lossy levers, the floor moves up substantially** (see §7).

---

## 5. Engineering risks per phase

### Phase A (FP8 real-storage attn_comp_kv) — atomic patch, same shape as #17

| Risk | Mitigation |
|---|---|
| Scale-table layout determines consumer access pattern. Per-row scales add a second load per attention step. | Pack scale at start of each row: 512 byte data + 4 byte scale, 16-byte align row → 528 byte/row → minor overhead. Consumer kernels read scale once into shared mem then reuse across head dim. |
| Producer pipeline already uses scratch+commit (#17). Need new commit kernel that takes F32 + emits FP8 byte + per-row scale. | Mirror existing `kernel_dsv4_f32_to_f16_store_rows` — new `_f32_to_fp8_store_rows`. ~30 lines Metal. |
| 5 consumer kernels (per dsv4_misc.metal) read F16/half4. Switch to FP8 read + scale-mul + widen. | Each kernel gets ~3 added lines. Same shape as #17 changes. |
| Snapshot bytes + payload version bump (v3 → v4). | Same pattern as #17. v3 GPU snapshots fail-fast on v4 load (intentional). |
| Indexer side (Lever B) does NOT block A — they're independent storage flips. | Land A first, validate, then B. |

**Scope estimate:** 1 atomic commit, ~600 LOC (similar magnitude to #17).

### Phase B (FP4 real-storage index_comp_kv) — atomic patch, same shape

| Risk | Mitigation |
|---|---|
| FP4 packing: 2 elements/byte. Indexer head dim = 128 → 64 bytes/row + scale. Need careful even-only access. | Indexer head dim is even (128), guaranteed packable. |
| FP4 dequant kernel already exists (`kernel_dsv4_indexer_hadamard_fp4_f32`) but currently only writes to F16 scratch. Need FP4 commit kernel + consumer-side dequant inline. | Add `_fp4_packed_store_rows` commit + change 2 indexer score consumer kernels to dequant inline. |
| Hadamard rotation must remain in producer (it's part of QAT). Storing FP4 only changes the bytes after rotation. | No change to Hadamard kernel itself. |

**Scope estimate:** 1 atomic commit, ~400 LOC.

### Phase C (Indexer-gated disk-tier ratio-4 attn_comp_kv) — **largest design surface**

This is the load-bearing lever. Engineering risk is real and design must be settled before any patch.

| Risk | Mitigation / decision needed |
|---|---|
| **Apple M4 page size is 16 KiB**, FP8 row size is 512 bytes. Direct `newBufferWithBytesNoCopy` on a per-row mmap would 32× the on-disk footprint (5.25 GiB → 168 GiB). | **Use staging-buffer scheme, not direct mmap:** keep disk as a flat host-side file (`pread()`-driven), maintain per-layer GPU staging buffer (512 rows × 528 byte ≈ 270 KiB × 21 layers ≈ 5.7 MiB), populate via Metal Shared buffer write. |
| **Random pread() of scattered 528-byte rows** at SSD page granularity (16 KiB) wastes bandwidth (~30× amplification). | Two-tier: (1) RAM hot-row LRU (~50 MiB sized to ~50k recent rows) absorbs most reads; (2) cold rows pread'd at 16 KiB block, neighbors cached. Profile-driven. |
| **Write path:** rows are produced sequentially (append at `n_comp`). Disk file write order = natural. | Write path is append-only sequential, low risk. Use `pwrite()` from commit kernel landing buffer or accumulate and flush per N rows. |
| **Snapshot/restore:** v4 (post-A+B) snapshot is large; disk-tier file IS effectively the snapshot for ratio=4 attn_comp_kv. | Snapshot logic for these rows changes from "read MTLBuffer + write" to "fsync disk file + write only LRU portion." v5 payload version. |
| **SSD wear:** 1M ctx generates 262K rows × 21 layers × 528 byte ≈ 2.8 GiB writes for a full-context prefill. Long sessions accumulate. | Within Mac Mini M4 SSD TBW budget (≈ 600 TB) → 200k full prefills before reaching budget. Not a practical concern at single-user pace. |
| **`madvise()` policy** on the disk file matters: `MADV_RANDOM` to disable readahead. | Standard advice. Apply at mmap setup. |
| **Failure mode:** if disk read fails mid-decode, can't recover. | Same SSD-failure mode as model GGUF mmap — accept as system failure. |

**Scope estimate:** 1 commit, ~1500 LOC + dedicated design memo (#34 mirroring [[f16-kv-metal-blocker]] template). **DO NOT start implementation without design-stage user signoff.**

### Mmap hygiene (no phase number, runs alongside A/B/C)

Low risk OS hint pass — `madvise(WILLNEED)` for hot weight regions, `MADV_RANDOM` for routed expert region, `MADV_DONTNEED` on evicted expert pages. Pure systems hygiene, no kernel changes.

---

## 6. Implementation ordering

```
[done] #17 F16 KV Metal (atomic, landed)
   │
   ▼
[done] #33 (this doc) — design approved by user
   │
   ▼
[done] #34 Lever A — FP8 real-storage attn_comp_kv, code-complete
   │   (see notes/execution-log.md "#34 Lever A atomic patch landed")
   ▼
[user authorization required] A validation (build + ctx_probe + small-ctx smoke)
   │
   ▼
[B] #36 design memo — FP4 real-storage index_comp_kv
   │
   ▼
[B] #37 implementation — atomic commit
   │
   ▼
[user authorization required] B validation
   │
   ▼
[C] #38 design memo — indexer-gated disk-tier  (most thorough survey; covers staging buffer, LRU, snapshot)
   │
   ▼
[C] #39 implementation
   │
   ▼
[user authorization required] C validation + 1M ctx end-to-end
   │
   ▼
[optional] madvise hygiene pass — bundleable with any patch above
```

**Why A → B → C, not C → A → B:**
- A and B are stationary storage flips, isomorphic to #17 — low novelty, high confidence
- C introduces a new memory hierarchy (disk-tier KV) — high novelty, needs the easier wins landed first to isolate debug signal
- C's RAM savings only matter once A and B have already brought KV under 4 GiB (otherwise routed-expert headroom is still limiting)

---

## 7. Deferred lossy levers (require explicit user authorization)

These are intentionally NOT in the quality-first plan. They change the actual numeric outputs the model produces. Listed here so they're available if the user later decides to trade quality for speed.

### Deferred — Expert pruning (was Phase 2, REAP-style keep_map)

**What:** Use calibration data + REAP saliency (router_weight × expert_output_norm) to select keep_map per layer; pre-shrink GGUF.

**Quality cost:** at keep=32 layer-wise, ~1-5% generic quality loss; **long-tail / specialized tasks may degrade more severely** (literature consensus). REAP minimizes this vs pure frequency pruning but does not eliminate it.

**Speed gain:** routed pool 86 → ~10.7 GiB. Combined with A+B+C, eliminates virtually all SSD paging → decode rate jumps to compute-bound (~5-10 tok/s on M4).

**Papers:** EEP (arXiv 2407.00945), REAP (arXiv 2510.13999), Not All Experts are Equal (arXiv 2402.14800).

### Deferred — MoBiLE big-little expert (was Phase 4)

**What:** Cold experts get small-rank approximations via offline training. Active small-expert when full expert not paged in.

**Quality cost:** explicit, paper-quantified ~1-3% generic.

**Engineering cost:** offline training pipeline — large lift beyond pure deployment.

**Paper:** MoBiLE (arXiv 2510.12357).

### Deferred — Vocab trim (was Phase 5)

**What:** 96k → 64k vocab via frequency-based pruning + BPE merge rewrite + hash routing patch.

**Quality cost:** unpredictable on rare-token tasks (code identifiers, multilingual). Fragile due to hash routing in layers 0-2.

**Saves:** ~0.5-1.2 GiB hot weight.

**See:** [[vocab-trim-feasibility]] memory.

### Deferred — Workload-aware routed expert cache replacement (was sub-part of Phase 3)

**What:** Replace OS page-cache LRU with custom MoE-Infinity-style score-based residency.

**Quality cost:** 0 — pure paging strategy.

**Speed gain:** modest under quality-first plan (just better hot-residency hit rate). Worth doing if A+B+C don't deliver target tok/s and user still won't authorize expert pruning.

---

## 8. Sources (paper citations preserved for deferred levers)

KV quantization
- [KVQuant (arXiv 2401.18079)](https://arxiv.org/pdf/2401.18079)
- [KIVI (arXiv 2402.02750)](https://arxiv.org/pdf/2402.02750)
- [CommVQ (arXiv 2506.18879)](https://arxiv.org/pdf/2506.18879)
- [KVTuner (arXiv 2502.04420)](https://arxiv.org/pdf/2502.04420)
- [KVmix (arXiv 2506.08018)](https://arxiv.org/pdf/2506.08018)
- [MiniKV (ACL 2025 Findings)](https://aclanthology.org/2025.findings-acl.952.pdf)

DeepSeek architecture
- [DeepSeek-V3 / MLA / Towards Economical Inference (arXiv 2502.14837)](https://arxiv.org/pdf/2502.14837)

MoE pruning (deferred levers)
- [EEP (arXiv 2407.00945)](https://arxiv.org/abs/2407.00945)
- [Not All Experts are Equal (arXiv 2402.14800)](https://arxiv.org/html/2402.14800v1)
- [SEER-MoE (arXiv 2404.05089)](https://arxiv.org/pdf/2404.05089)
- [REAP (arXiv 2510.13999)](https://www.arxiv.org/pdf/2510.13999)
- [MoE-Pruner (arXiv 2410.12013)](https://arxiv.org/pdf/2410.12013)
- [EAC-MoE (arXiv 2508.01625)](https://arxiv.org/pdf/2508.01625)
- [MoNE (arXiv 2507.00390)](https://arxiv.org/pdf/2507.00390)

MoE offloading + workload-aware caching
- [MoE-Infinity (arXiv 2401.14361)](https://arxiv.org/html/2401.14361v3)
- [MoBiLE (arXiv 2510.12357)](https://arxiv.org/pdf/2510.12357)
- [MoE-Gen (arXiv 2503.09716)](https://arxiv.org/html/2503.09716)
- [DALI (arXiv 2602.03495)](https://arxiv.org/html/2602.03495v1)

Apple Silicon
- [Native LLM Inference at Scale on Apple Silicon (arXiv 2601.19139)](https://arxiv.org/abs/2601.19139)
- [Production-Grade Local LLM Inference on Apple Silicon (arXiv 2511.05502)](https://arxiv.org/pdf/2511.05502)
- [Apple Silicon LLM Optimization Guide](https://blog.starmorph.com/blog/apple-silicon-llm-inference-optimization-guide)

---

## 9. What this roadmap does NOT do (standing rules)

- Does not change the 1M ctx target.
- Does not introduce any lossy computation step relative to current runtime values.
- Does not authorize implementation, validation, or runtime smoke tests — those wait on explicit user instruction per the standing constraint.
- Does not start Phase B implementation before A is validated; does not start C implementation before B is validated. Cross-phase debugging signal contamination is unacceptable on a system where validation requires authorization.
- Does not adopt LRU as the routed expert cache policy (literature consensus against; defer to workload-aware variant if/when routed paging dominates).
- Does not run CPU baseline (would kernel-panic the box per prior incidents — see [[goal-and-constraints]]).
