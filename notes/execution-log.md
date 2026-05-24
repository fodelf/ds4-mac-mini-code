# DS4 execution log

Append-only chronological record of significant DS4 work — design docs, atomic patches, validation steps, scope decisions, and any material blocker. Each entry links to the detailed artifact (`notes/`, source file + line, memory file). Skim §Most recent for current state.

**Rules for this file:**
- Append-only. Do not rewrite past entries.
- Each entry: date + headline + bullet points (what / why / where / status).
- Link to artifacts, do not duplicate their content. The log points; the artifacts contain.
- Status flags: `landed` / `in-progress` / `pending-authorization` / `deferred` / `blocked`.

---

## Most recent

- **2026-05-24** — #52 view-shrink env var landed + 理论待验证. K=16 smoke 跑出关键数据：cb_alive=0, transient=0, drv=13800 MiB, pipelines=20 全程平稳——userland 和 Metal 驱动报告侧零增长，但 IOGPU CB#22 (layer 21) `kIOGPUCommandBufferCallbackErrorOutOfMemory`. autoreleasepool 假说被证伪。新假说有数据支撑：OOM 层数 ≈ 总层数 / view 数（80GB/10view→layer 7, 22GB/3view→layer 15, 13GB/2view→layer 21），暗示**IOGPU 维护 per-resource 累计 touched-pages map，view 第一次被 CB 引用时整 view 加进 IOGPU wireable 预算**，2 个 6.5 GiB view 加起来超 Mac M4/16GiB 的 ~12 GiB 内核 wireable 上限。补丁：`DS4_METAL_MODEL_MAX_VIEW_BYTES` 强制 view 上限（默认仍为 `[g_device maxBufferLength]`，约 8 GiB；最小 256 MiB；smoke 脚本默认 3.5 GiB → 5 view）。预测：5 view 应推到 layer ~34；如果推得动，再 shrink 到 2 GiB 看能否走完 43 层。代码改动仅 `ds4_metal.m:478-525` 一段 + `smoke-low-mem.sh` 多一个 env-var 透传。Build 绿，待用户跑。
- **2026-05-24** — #51 加 Metal 内核态诊断探针 (DS4_DIAG=1 gated; build green). 两次 K=16 smoke 都 OOM 在 layer 21、system vmstat 全程冻结（wired=1430 MiB、file_backed=11069 MiB 不动）—— 这意味着 OOM 来自 Metal 驱动内核侧、跟系统 VM 解耦。`ds4_diag_vmstat` 看不到那里。新增 `ds4_diag_metal_state(tag)` 打印 `[g_device currentAllocatedSize]`、`g_transient_buffers.count`、`g_pending_cbs.count`、`g_pipeline_cache.count`、`g_model_buffer_cache.count`、CB 累计/活跃计数、`g_batch_cb/g_batch_enc/residency_set` 三个 nil 标志。CB 生命周期计数在 4 个创建点（`begin_commands` / `flush_commands` 新 CB / `command_buffer` 一次性 / `synchronize` 后备）和 2 个回收点（`finish_command_buffer` / `wait_pending_command_buffers`）配对维护，alive 数任何时刻反映"userland 仍持有强引用的 MTLCommandBuffer 数"。`finish_command_buffer` 也额外打 `dropped_transients` 数值，确认 transient 数组每次提交真的清零。`ds4_cuda.cu` 加空 stub，CPU 路径无 caller 故免改。下一步：用户跑 `./smoke-low-mem.sh` 看新 `metal[…]` 行随 21 层是涨什么。
- **2026-05-24** — #50 K=16 GGUF 生成 + smoke 默认 MODEL 切换. K=48 (22.3 GiB) smoke 推进到 layer 15/16 仍 OOM —— per-layer CB split + warmup-skip 都生效，但 macOS Metal `newBufferWithBytesNoCopy` 的 ~8 GiB per-buffer cap 强制把 22.3 GiB 模型切成 3 个 ~7.4 GiB shared buffer，单 CB 触一个 buffer 即全 wire，wired ~10.5 GiB + file_backed ~4.2 GiB + system ≈ 15.8 GiB 顶满 16 GiB ceiling。用户决策"先保证第一个 token 为第一要义" → 暂时让步质量改用 K=16（256 中保留 16，6.2% routed expert slot）。三步离线流水线：`router_norms.py`（1.5s）→ `make_expert_mask.py --keep-top-k 16`（即时）→ `shrink_gguf.py --do-it`（41s，写 13.68 GB）。`smoke-low-mem.sh:20` 默认 MODEL 改 `./gguf/ds4flash-k16.gguf`。新文件 13 GB → 预期 2 个 buffer view @ ~6.85 GiB。
- **2026-05-24** — #49 smoke 默认值校正 (script-only). `smoke-low-mem.sh` 默认 MODEL 从 `./ds4flash.gguf`（80 GiB / 10 个 mmap view）切到 `./gguf/ds4flash-k48.gguf`（22.3 GiB / 3 个 view），并新增 `DS4_METAL_NO_PREFILL_KERNEL_WARMUP=1`。诊断 trace（#48）显示 19:01 那次 smoke 实际跑的是 80 GiB 模型，且 prompt "Hi" 经 BOS+chat template 变 10 token 触发了 `metal_graph_warmup_prefill_kernels` 的 HC-attn 预热 matmul — 单个 CB 提交后 wired +9.1 GiB 永不释放，叠加 model-view 10 buffer 后 layer 7 必 OOM。无代码改动，env var 早已在 `ds4.c:11381` 就位。
- **2026-05-24** — #48 诊断日志注入 patch landed (DS4_DIAG=1 gated; build green). vmstat snapshots + per-CB commit prints around prefill setup (alloc/steering/upload/warmup/prefill_layer_major/split-loop) so the next smoke pinpoints exactly which step OOMs. `smoke-low-mem.sh` now exports `DS4_DIAG=1` and post-runs greps for the new traces.
- **2026-05-24** — #47 CB-split env-var patch landed (code-only, validation gated). Two opt-in env vars: `DS4_METAL_PREFILL_SPLIT=1` (forces per-layer CB during short-prompt prefill) + `DS4_METAL_DECODE_SPLIT_EVERY=1` (flushes after every N decode layers). Both default-off; no behavior change without opt-in. `smoke-low-mem.sh` updated to set both. Build + smoke pending explicit user authorization.

## 2026-05-24 — #53 第一 token 跑通 + smoke-watch.sh live monitor 落地
- **里程碑：** 16 GiB Mac Mini M4 在 K=16 + 3.5 GiB view-cap 配置下，**首个 token 成功 emit**。
- **证据（`/tmp/ds4-smoke-20260524-213935.{stdout,stderr}.log`）：**
  - stdout 3 字节：`<space>(\n`（temp=0，prompt="Hi"，K=16 路由打残后质量随机）
  - stderr 终止行：`ds4: prefill: 2.17 t/s, generation: 2444.99 t/s`（rc=0）
  - prefill 43/43 全跑完，无 OOM；CB#44（decode）`cb_alive=0 ok=1`；pipelines 20→22（decode 新 shader 编译）
  - 整条 trace `drv=15816.4 MiB` 平稳，`cb_alive` 每 commit 后归零（autoreleasepool 干净，与 #51 数据一致）
- **理论确认（IOGPU per-resource touched-pages）：**
  - K=48 / 3 view @ 7.4 GiB → layer 15/16 OOM
  - K=16 / 2 view @ 6.85 GiB → layer 21 OOM（首次跨 view）
  - K=16 / 5 view @ 3.5 GiB → layer 43 ✅
  - 模式吻合：单 view 越小，单层 prefill 触发的 wireable 增量越小，能撑到的层数越深 → IOGPU 是按 per-resource 累计 touched pages 计入 wireable 预算。view-shrink 是绕开这个 budget 的关键杠杆。
- **新工具 `smoke-watch.sh`（落地）：** smoke-low-mem.sh 的 live-stream 版本。
  - stderr 经 grep 过滤后实时打印（保留 layer 进度、OOM、finish[]、post-end metal[]、t/s 汇总；丢弃高频 vmstat[pre]/metal[begin]）
  - stdout 后台 `tail -f` 实时打印生成 token（带 `[stdout]` 前缀），不必等进程退出
  - 失败时自动 dump 末尾 30 行 metal[] + 5 行 finish[] + OOM 行
  - 全量 stderr/stdout 仍写盘到 `/tmp/ds4-watch-$STAMP.{stdout,stderr}.log`
  - 默认 tokens=32（vs smoke-low-mem.sh 的 1），便于看 decode 流。其他默认值与 smoke-low-mem.sh 一致。
- **下一步候选（非自动执行）：**
  - 阶梯回升 K=24 / K=32 找质量上限（每升一档先看 IOGPU 是否仍在 3.5 GiB view-cap 下不 OOM）
  - 拉长 prompt 看 prefill t/s 随 ctx 的曲线
  - decode 多 token 看是否稳定流出（当前只验了 1 token，rc=0 但没看到长度 >1 的输出）
- **Status:** 首 token 验证通过，监控脚本就位；阶梯回升 K / 拉长生成由用户决定何时跑。

## 2026-05-24 — #50 K=16 GGUF 生成 + smoke 默认 MODEL 切换
- **触发：** K=48 smoke (#49 重跑) trace 显示 per-layer CB split + warmup-skip 都已生效（推进到 layer 15/16，比 K=48 + 80 GiB 模型的 layer 7 多 8 层），但仍 OOM。新 trace 数据：
  - 模型 mmap `22325.67 MiB` 切成 3 个 shared buffer view（~7.4 GiB 每个）
  - 第一层 layer CB commit 后 wired 从 1558 → 10598 MiB（+9 GiB），file_backed 635 → 3935 MiB（+3.3 GiB）—— 单 CB 触 1-2 个 buffer 即整 buffer 被 wire
  - 后续 15 层 CB 提交，wired 维持 ~10.5 GiB 振荡，file_backed 从 3935 慢爬到 4226 MiB
  - 第 16 层时新 chunk page-in 需更多 wire，free 仅 100 MiB → IOGPU OOM
- **根因（架构性）：** macOS `newBufferWithBytesNoCopy + MTLResourceStorageModeShared` 每 buffer 上限 ~8 GiB（Apple cap）。Metal 把这种 buffer wire 是 buffer-level 颗粒度，不是 page-level —— 哪怕 CB 只 reference buffer 里一个 byte，整 buffer 7.4 GiB 必 wire。22.3 GiB 模型必须 3 buffer，至少 1 wired + 至少 1 在 file_backed cache 中 = 14+ GiB 内存常驻，加 system 顶满 16 GiB。
- **决策路径：** 用户拍板"先保证第一个 token 为第一要义" → 接受质量临时让步、用更小 K。三个候选 (K=32/K=24/K=16) 中选 K=16，理由：(a) 13.68 GiB 落在 2 buffer @ ~6.85 GiB，绕开 buffer-cap 触发的最坏情况；(b) 失败成本低，可阶梯回升 K=24/K=32 找质量上限。
- **离线流水线（全部不加载模型到 Metal，无崩机风险）：**
  - Step 1: `python3 gguf-tools/router_norms.py gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --out /tmp/router_norms.json` — 1.5s，仅顺序读 43 个 router 张量 ~130 MB
  - Step 2: `python3 gguf-tools/make_expert_mask.py /tmp/router_norms.json --keep-top-k 16 --out gguf/mask-k16.bin` — 即时，输出 1396 字节 mask（688 kept = 43 × 16）
  - Step 3: `python3 gguf-tools/shrink_gguf.py --in <src> --mask gguf/mask-k16.bin --out gguf/ds4flash-k16.gguf --do-it` — 41s（Apple NVMe 高速读写），输出 13.68 GB
- **`shrink_gguf.py` 注意点：** 默认 dry-run（feasibility 预估），必须显式 `--do-it` 才真写。第一次没传 flag，看到 dry-run 报告 "est output 13.68 GB / 6.2% experts kept" → 加 flag 重跑成功。
- **dry-run 给出的关键数字：** routed-expert bytes in 77.91 GB → out 4.87 GB，disk savings 73.04 GB。验证了"模型 90% 是 routed expert，10% 是固定开销 ~8 GB attn/shared/embedding"的反推（原 86.72 GB / K=48 21.81 GB 二点拟合）。
- **`smoke-low-mem.sh:20` 改动：** 默认 MODEL 从 `./gguf/ds4flash-k48.gguf` 改 `./gguf/ds4flash-k16.gguf`，注释更新解释 K=16 是质量让步换头部空间。
- **文件：** `gguf/mask-k16.bin`（新），`gguf/ds4flash-k16.gguf`（新，13 GB），`smoke-low-mem.sh`（默认值）。
- **无代码改动；无 build；无 smoke 运行。** 验证由用户手动 `./smoke-low-mem.sh`。
- **Status:** K=16 GGUF 就位 + 脚本默认改完；smoke pending 用户运行。

## 2026-05-24 — #49 smoke-low-mem.sh 默认值校正 (script-only, no code change)
- **触发：** #48 诊断 trace（`/tmp/ds4-smoke-20260524-190106.stderr.log`）暴露两个超出 #45-#47 OOM 分析假设的事实。
- **事实 1 — 跑错了模型：** 第 3 行 `mapped 82697.67 MiB`，第 4 行 `10 overlapping shared buffers`。`./ds4flash.gguf` symlink 指向 80 GiB 原始 GGUF；smoke 没显式 `DS4_MODEL=` 就走了默认路径。22.3 GiB 的 K=48 在 `./gguf/ds4flash-k48.gguf`（已验证存在，23.4 GB on-disk）。
- **事实 2 — warmup 触发：** 第 16 行 `warmup_prefill_kernels enter n_tokens=10 warmed=0`。Prompt "Hi" 经 BOS + chat template 后是 10 token，超过 `n_tokens<=8` early-return 阈值（`ds4.c:11395`），warmup 实际跑了 HC-attention 投影 matmul。CB #1（cb=0x102e2b070）commit 后第 20 行 wired 从 1693 → 10838 MiB（+9.1 GiB），这部分在后续 7 个 layer CB 期间从未释放（第 31/37/43/49/55/61/67 行 wired 一直 ≥ 10.5 GiB）。
- **改动：** `smoke-low-mem.sh:14-20` 默认 MODEL 改 `./gguf/ds4flash-k48.gguf`（DS4_MODEL env var 仍可覆盖）；`smoke-low-mem.sh:37-49` env-var 块新增 `DS4_METAL_NO_PREFILL_KERNEL_WARMUP=1`。该 env var 在 `ds4.c:11381` 处理（`if (warmed || getenv("DS4_METAL_NO_PREFILL_KERNEL_WARMUP") != NULL) return true;`），#48 trace 早已能打 "early-return (already warm or disabled)" 消息确认 hit。
- **预期效果：** (a) 模型 mmap 从 80 GiB / 10 view 降到 22.3 GiB / 3 view，每个 CB 触及的 model-view buffer 数量约 1/3；(b) 跳过 CB #1，省 9.1 GiB sticky wired；(c) 起始 `free_count ≈ 10.8 GiB`，per-layer CB 有充分余量推进 43 层。
- **无代码改动；无 build；无 smoke 运行。** 验证由用户手动 `./smoke-low-mem.sh` 执行。
- **风险：** 改默认值不影响任何被显式 `DS4_MODEL=` 调用的旧路径；env var 增加是 superset，不影响已通过测试的现有用法。
- **文件：** `smoke-low-mem.sh`.
- **Status:** script change landed; smoke pending user run.

## 2026-05-24 — #48 诊断日志注入 patch landed (DS4_DIAG=1 gated; build green)
- **Why:** post-#47 smoke still OOMs but with ZERO `gpu prefill layer` prints — meaning failure is somewhere between the `using GPU graph generation` banner (`ds4.c:15958`) and the first layer's CB submit. Need data to disambiguate, not more guessing (user: "你有什么问题可以再加日志再分析再解决，不要盲猜").
- **Mechanism:** new helpers `ds4_diag_enabled()` + `ds4_diag_vmstat(tag)` in `ds4_metal.m` (CUDA mirror in `ds4_cuda.cu`), declared in `ds4_gpu.h`. `ds4_diag_enabled()` caches `getenv("DS4_DIAG")` on first call; `ds4_diag_vmstat()` reads `host_statistics64(HOST_VM_INFO64)` → prints free/wired/file_backed/compressed/anon in MiB.
- **Injection points (all no-op without `DS4_DIAG=1`):**
  - `ds4_metal.m` `ds4_gpu_begin_commands`: print CB pointer
  - `ds4_metal.m` `ds4_gpu_flush_commands`: vmstat pre/post + CB pointer
  - `ds4_metal.m` `ds4_gpu_end_commands`: vmstat pre/post + CB pointer + commit ok flag
  - `ds4.c:metal_graph_alloc_raw_cap` enter/exit: vmstat + raw_cap/ctx_size/prefill_cap
  - `ds4.c:metal_graph_load_directional_steering` enter + early-return path: attn/ffn scales + path
  - `ds4.c:metal_graph_upload_prompt_embeddings_hc`: branch choice (CPU vs GPU) + n_tokens + exit code
  - `ds4.c:metal_graph_warmup_prefill_kernels`: enter + each early-return reason (already-warm / disabled / n<=8)
  - `ds4.c:metal_graph_prefill_layer_major` enter: vmstat + n_tokens + prefill_cap + imatrix flag
  - `ds4.c:metal_graph_prefill_layer_major` after split_commands compute: which inputs decided the branch
  - `ds4.c:metal_graph_prefill_layer_major` split-loop pre-entry: vmstat + DS4_N_LAYER
  - `ds4.c:metal_graph_prefill_layer_major` per-layer pre-begin: il
- **`smoke-low-mem.sh` updates:** export `DS4_DIAG=1`; two new grep summaries after the run — setup-phase trace (alloc/steering/upload/warmup/prefill/split-loop) and CB commit trace (begin/end/flush + vmstat lines).
- **Build:** `make` green. One pre-existing unrelated `-Wunused-function` warning for `ds4_gpu_encode_cpy_f16_f16_1d` (carry-over from #34 Lever A, documented).
- **Files:** `ds4_gpu.h`, `ds4_metal.m`, `ds4_cuda.cu`, `ds4.c`, `smoke-low-mem.sh`.
- **Status:** code-complete + builds. **No smoke run.** Validation = next manual `./smoke-low-mem.sh` by user.

## 2026-05-24 — #47 CB-split env-var patch landed (code-complete, validation gated)
- **Scope:** add two opt-in env vars to existing well-tested CB-split paths so a 16 GiB Mac Mini M4 can route short-prompt prefill + first decode token through small per-CB working sets instead of single 14 GiB unions. No new code paths — only new ways to reach paths already exercised by long-prompt prefill and the decode pipelining split.
- **What changed (code):**
  - `ds4.c:13572-13588` (prefill): new `split_env = getenv("DS4_METAL_PREFILL_SPLIT")` ORed into `split_commands`. When set, n_tokens ≤ 2048 prompts also take the per-layer CB branch at `ds4.c:13727-13749`. Multi-line comment added inline explaining the M4 motivation (single-CB union ≈ 14 GiB → kIOGPUCommandBufferCallbackErrorOutOfMemory).
  - `ds4.c:11115-11160` (decode `metal_graph_encode_token_raw_swa`): new `split_every = getenv("DS4_METAL_DECODE_SPLIT_EVERY")` parsed alongside the existing `split_after_layers` (`DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS`). Loop now flushes when `single_split_hit || every_split_hit`. `every_split_hit` excludes `layers_done == DS4_N_LAYER` so the final flush still happens via `ds4_gpu_end_commands` in the caller (avoids a spurious empty CB).
  - `smoke-low-mem.sh:33-37`: env-var block extended with `DS4_METAL_PREFILL_SPLIT=1 DS4_METAL_DECODE_SPLIT_EVERY=1`. New post-run grep prints prefill-layer trace and any `command batch failed` line for quick triage.
- **Why two patches not one:** prefill OOM was the proven failure. Decode has the same shape of working-set union per token but already had a one-shot split mechanism (default split at layer 4). With prefill split, the decode CB2 still holds 39 layers' dispatches — its working set could fit only because prefill warmed the relevant pages into `file_backed` and Metal can re-wire warm pages cheaply. That's a guess about Metal/Mach interactions, not a guarantee. The decode-side env var is cheap insurance: per-layer flushes cap the decode CB working set at ~one layer, same as prefill.
- **Behavior without env vars:** zero change. `split_env` defaults to false; `split_every` defaults to 0; the existing `split_after_layers = 4` pipelining split is untouched.
- **Files:** `ds4.c`, `smoke-low-mem.sh`.
- **Risk assessment:** very low. The per-layer split prefill path is exercised on every prompt > 2048 tokens today. The decode per-layer flush adds a `ds4_gpu_flush_commands` call inside the same loop that already has one — semantically identical, just more often. Inter-layer state passing (`g->cur_hc`/`after_ffn_hc` swap at 11148-11150) is local to the loop body and unaffected by where the CB boundary sits.
- **Status:** code-complete. No build, no smoke. Validation gated on explicit user authorization (standing rule "没有我明确的指示不要私自验证").

## 2026-05-24 — #46 decode-entry OOM diagnosis: root cause is single-CB prefill, not decode

## 2026-05-24 — #46 decode-entry OOM diagnosis: root cause is single-CB prefill, not decode
- **Scope:** read-only code-level investigation of where `Metal command batch failed: Insufficient Memory` is fired (`ds4_metal.m:281`). No code changes, no machine runs.
- **Findings:**
  - `Metal command batch failed: ...` is emitted by `ds4_gpu_wait_command_buffer` (`ds4_metal.m:281`) when called with label="command batch". The two callers are `ds4_gpu_end_commands` (`ds4_metal.m:4132`) and `ds4_gpu_flush_commands` (`ds4_metal.m:4120`).
  - `ds4_session_sync` short-prompt prefill path: `prompt->len=1` ("Hi") never trips `s->prefill_cap < prompt->len`, so we call `metal_graph_prefill_raw_swa` (`ds4.c:18059`) → `metal_graph_prefill_layer_major` (`ds4.c:13825` → `13556`).
  - `metal_graph_prefill_layer_major` computes `split_commands = split_profile || n_tokens > 2048 || imatrix != NULL` at `ds4.c:13579`. For our smoke all three are false → single-CB branch at `ds4.c:13585-13657`.
  - Inside that branch: `ds4_gpu_begin_commands()` at 13593 opens ONE CB; loop 13594-13605 calls `metal_graph_encode_layer_batch` for il=0..42 — but the `fprintf "gpu prefill layer N/43\r"` at 13602 prints as each layer is **encoded into the CB**, not when it executes. Output head is encoded into the same CB at 13620-13628. Single commit at `ds4_gpu_end_commands()` line 13631 is where it fails.
  - Metal command-buffer commit has to make every GPU buffer referenced by any encoded dispatch resident for execution. Union of embed + 43 × {attn proj + indexer + router + 6 routed expert MLPs} + output head ≈ 14 GiB. Observed: file_backed 211 MiB → 12.91 GiB during prefill, free 11.86 GiB → 125 MiB, then OOM. Working set matches.
  - Per-token decode (`metal_graph_eval_token_raw_swa` @ `ds4.c:13108`) wraps each token in its own `begin_commands`/`end_commands` pair → per-token CB working set ≈ one layer's union, ~300 MiB. Decode would succeed if prefill commits.
  - The split path at `ds4.c:13727-13749` (split_commands=true, split_profile=false) already issues one CB per layer (`begin_commands` → `encode_layer_batch` → `end_commands` per il). This is what we want. It is currently enabled only when `n_tokens > 2048` or `imatrix != NULL`; no opt-in env var exists for short prompts.
- **Proposed minimal patch (1 line, pending user authorization):**
  ```c
  /* ds4.c:13579 — add env-var opt-in to the existing well-tested split path */
  const bool split_commands = getenv("DS4_METAL_PREFILL_SPLIT") != NULL ||
                              split_profile || n_tokens > 2048 || imatrix != NULL;
  ```
  Setting `DS4_METAL_PREFILL_SPLIT=1` in `smoke-low-mem.sh` would route short-prompt prefill through the per-layer-CB path. Per-CB working set ≈ 300 MiB; between commits Metal releases wiring so Mach can reclaim file-backed pages on demand.
- **Why pruning K further was a red herring:** for a 1-token "Hi" prompt, every layer routes top-6 of K experts. Reducing K (48→24→16) shrinks the on-disk pool but NOT the per-token touch set — the same 6×43=258 experts get encoded into the CB. K only affects the inactive pool size on disk. The OOM is a Metal wiring failure, not a model-size problem.
- **Status:** read-only investigation complete. Patch is gated on explicit user authorization per standing rule. No build, no smoke.

## 2026-05-24 — #45 K=48 shrunken-GGUF smoke result (FAILED, decode-entry GPU OOM)
- **Setup:** `sudo purge` → `DS4_MODEL=./gguf/ds4flash-k48.gguf ./smoke-low-mem.sh` (env vars `DS4_METAL_NO_RESIDENCY=1 DS4_METAL_NO_MODEL_WARMUP=1`, ctx=1024, tokens=1, prompt="Hi"). Logs at `/tmp/ds4-smoke-20260524-180351.{stderr,stdout,vmstat-before,vmstat-after}`.
- **Loader behaviour confirmed correct:** `ds4: expert mask synthesized from shrunken-GGUF keep_map kept=2064/11008 (18.8%)` — DSXM mask + `expert_keep_map.kept_counts/original_ids` metadata round-trip works. `mapped 22325.67 MiB from offset 5.09 MiB` (3 overlapping shared buffers, down from 10/82697 MiB for original). Residency/warmup both ~0 ms — env vars active.
- **Failure mode:** all 43 layers of `gpu prefill` print, then `Metal command batch failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)`. Zero tokens generated. Failure is at the decode-entry command-buffer submit, not during prefill.
- **Memory accounting (vm_stat, 16 KiB pages):**
  - before purge ran clean: free=776 743 (11.86 GiB), file_backed=13 500 (211 MiB), wired=80 107 (1.22 GiB)
  - after decode-entry OOM: free=7 984 (125 MiB), file_backed=846 289 (12.91 GiB), wired=72 266 (1.10 GiB), swapouts +22 268 (348 MiB compressed out during run)
- **Interpretation:** the hot weights (token embed + 43 × attention/norm/router) are touched every forward pass independent of K and account for the bulk of the 12.91 GiB file_backed; routed-expert pruning K=48 reduces only the inactive-expert footprint. Working set ≈14 GiB leaves no headroom for the decode command-buffer alloc on a 16 GiB system minus kernel reserve.
- **Comparable earlier (wrong-model) run:** `/tmp/ds4-step1-stderr.log` mapped 82697.67 MiB / 10 buffers — that was original `./ds4flash.gguf` (80 GiB), included here only to confirm the 22.3 GiB / 3-buffer figure above is the K=48 path.
- **Status:** `blocked`. Need user direction on next step (more aggressive K, decode-alloc investigation, or different lever).
- **Memory not invalidated:** `[[goal_and_constraints]]` (no autonomous validation, no CPU inference) and `[[optimization_roadmap_framework]]` (Levers A/B/C are quality-first storage redesigns, independent of K-pruning) remain accurate.



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
