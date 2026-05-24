# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Scope

DwarfStar 4 (`ds4`) is a **DeepSeek V4 Flash specific** C inference engine — not a generic GGUF runner. The narrow bet is intentional: tokenizer, prompt rendering, KV cache layout, tool-call DSML, quantization mix, and graph kernels are all wired to one model. Code that "would also work for other models" is out of scope unless explicitly requested.

The production target is **Metal on macOS** (M3 Max / M3 Ultra class) and **CUDA on Linux** (DGX Spark / GB10 specifically tuned). The CPU backend exists for correctness/diagnostics only; **do not run large CPU inference on macOS — it has crashed the kernel via the VM subsystem** and is not fixable from userspace.

`ds4.c` does not link GGML/llama.cpp but borrows quant tables, dot kernels, and GGUF layout from them. Keep the GGML copyright in `LICENSE`.

## Build

Metal is the default on Darwin; on Linux, plain `make` prints help instead of guessing a CUDA arch.

```sh
make                  # macOS Metal: ds4, ds4-server, ds4-bench, ds4-eval, ds4-agent
make cuda-spark       # Linux CUDA tuned for DGX Spark / GB10 (no -arch flag — fastest there)
make cuda-generic     # Linux CUDA, sm_native
make cuda CUDA_ARCH=sm_120   # explicit arch
make cpu              # CPU-only reference build of all binaries (adds -DDS4_NO_GPU)
make clean
```

CPU objects (`ds4_cpu.o`, `ds4_cli_cpu.o`, ...) are compiled with `-DDS4_NO_GPU` from the same `.c` files. When touching a file that has both a GPU and CPU object, expect to rebuild both.

## Test

The C test runner `ds4_test` covers everything; `make test` runs `--all`. Tests need `./ds4flash.gguf` (or `DS4_TEST_MODEL=`).

```sh
make test
./ds4_test --server              # request parsing, chat rendering, streaming, tool calls, KV bookkeeping
./ds4_test --logprob-vectors     # local tokens vs official DeepSeek V4 Flash continuation vectors
./ds4_test --long-context        # fact recall from tests/long_context_story_prompt.txt
./ds4_test --tool-call-quality   # live DSML emission, fast + exact paths
./ds4_test --metal-kernels       # isolated kernel numerics

# CUDA-only smoke (on a CUDA box):
make cuda-regression
```

Overrides: `DS4_TEST_MODEL`, `DS4_TEST_VECTOR_FILE`, `DS4_TEST_LONG_PROMPT`.

## Speed regression

Speed changes require a before/after CSV on the same machine, backend, quant, thermal state. Use `ds4-bench` (instantaneous prefill/gen at context frontiers, not whole-run averages):

```sh
./ds4-bench -m ds4flash.gguf --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 65536 --step-incr 2048 --gen-tokens 128 --csv /tmp/out.csv
```

The only acceptable speed regression is one paying for a correctness fix.

## Architecture

### Engine boundary (`ds4.h`)

CLI/server/agent code talks to the engine through this header and must not reach into tensor internals. Two opaque objects:

- `ds4_engine` — loaded model (mmap-backed weights, tokenizer, kernel handles). One per process; there is an intentional instance lock so do not run multiple huge model processes concurrently.
- `ds4_session` — one mutable inference timeline. Owns the live KV cache + next-token logits. Callers always provide a **full token prefix** and let `ds4_session_sync()` decide whether to extend, rewrite in place (`ds4_session_rewrite_from_common`), or rebuild from scratch. The "common prefix → rewrite vs rebuild" path is the engine's contract with stateless HTTP APIs — don't bypass it.

Session snapshots (`ds4_session_save_payload` / `ds4_session_save_snapshot`) serialize the DS4-specific KV/session state. The outer file format (header, hashing, eviction) belongs to `ds4_kvstore`, not the engine.

### Files

- `ds4.c` (~800 KB) — model loading, tokenizer, CPU reference graph, Metal/CUDA graph scheduling, sessions, payload serialization. The big one. Comments live next to the code; don't add separate design docs.
- `ds4_metal.m` + `metal/*.metal` — Objective-C Metal runtime + compute kernels. Objective-C is allowed **only** here.
- `ds4_cuda.cu` + `ds4_iq2_tables_cuda.inc` + `ds4_gpu.h` — CUDA path.
- `ds4_cli.c` — argument parsing, linenoise REPL, multi-turn transcript, slash commands.
- `ds4_server.c` (~560 KB) — OpenAI / Anthropic / Responses HTTP API, SSE streaming, tool-call mapping, worker queue, disk KV policy.
- `ds4_agent.c` — native coding agent (alpha). Drives inference in-process; the on-disk KV cache *is* the session.
- `ds4_kvstore.{c,h}` — disk KV cache: SHA1-of-rendered-text-prefix filenames, fixed 48-byte header + extension flags + DS4 payload + optional tool-id replay map. Save reasons: cold / continued / evict / shutdown / agent_system / agent_session.
- `ds4_bench.c`, `ds4_eval.c` — throughput bench and 92-item capability eval (GPQA Diamond / SuperGPQA / AIME 2025 / COMPSEC).
- `rax.c`, `linenoise.c` — vendored radix tree and line editor.
- `gguf-tools/` — offline tooling (quantization, imatrix collection, quality scoring). Separate Makefile.
- `dir-steering/` — single-direction activation steering vectors.

### Server inference model

Request parsing and sockets run in client threads, but **inference is serialized through one graph worker**. There is no batching of independent requests. The server keeps **one live in-memory KV checkpoint** for the active session; everything else lives on disk if `--kv-disk-dir` is enabled.

For stateless clients (chat completions, Responses, Anthropic Messages): the server first tries an exact token-prefix hit, then a rendered byte-prefix hit, falling back to scanning the disk KV cache and tokenizing only the new suffix.

### Tool calls (DSML)

The model emits tool calls as DSML text. Agents send back **normalized JSON**, not the original DSML. Two reconciliation paths:

1. **Exact replay (primary).** Every tool call gets an unguessable API id; a bounded radix-tree map (`--tool-memory-max-ids`, default 100000) remembers `id → exact sampled DSML bytes`. On the next turn the renderer reuses those bytes so the prompt byte-prefix still matches the live KV. The map is persisted in the disk KV file (extension flag bit 0).
2. **Canonicalization (backup).** Used when exact replay is missing or `--disable-exact-dsml-tool-replay`. Renders deterministic DSML from the JSON; if it diverges from the live sampled stream, the server rewrites the live checkpoint or falls back to an older disk snapshot and replays the suffix.

During generation, DSML *structure* (tags, headers, JSON punctuation, closing markers) is sampled at `temperature=0` for parseability. **Argument payloads (`string=true` bodies, JSON string values, file contents, edits) use the request's normal sampling** — forced-greedy on long payloads produces repeated text.

### Thinking modes

Three modes: `DS4_THINK_NONE`, `DS4_THINK_HIGH`, `DS4_THINK_MAX`. The server defaults to thinking; `reasoning_effort=max` maps to Think Max **only** when the context is large enough (`ds4_think_max_min_context()`), otherwise falls back to normal thinking. In thinking mode, sampling defaults are fixed (`temperature=1, top_p=1, min_p=0.05`) and client sampling knobs are ignored — that matches DeepSeek's API behavior; don't "fix" it.

## Quality rules (from AGENT.md)

- No C++. C99 + Objective-C (Metal only) + CUDA.
- Comment the **why** of inference mechanics, cache lifetimes, memory policy, API orchestration. Skip narration of what the code obviously does.
- Keep public APIs narrow — CLI/server code must not depend on tensor internals.
- No permanent semantic variants behind flags. Diagnostic switches for validating the one release path are fine.
- Preserve correctness before speed. Don't ship a faster path with unexplained attention / KV / logits drift.
- Keep model loading mmap-backed; never eagerly copy the full GGUF.

## Debugging

Three first-line tools:

```sh
./ds4 --dump-tokens -p "..."                                  # tokenize and exit; shows DS4 specials
./ds4 --dump-logprobs /tmp/out.json --logprobs-top-k 20 --temp 0 -p "..."
./ds4-server --trace /tmp/ds4-trace.txt ...                   # prompts, cache decisions, tool-parser events
```

When opening issues for failing sessions, always include `--trace` output.

## Model files

The engine only runs **the DeepSeek V4 Flash GGUFs published for this project** (asymmetric 2-bit: routed MoE experts at IQ2_XXS up/gate + Q2_K down; shared experts and projections untouched). `./ds4flash.gguf` is the default path; `./download_model.sh q2-imatrix|q4-imatrix|q2|q4|mtp` fetches and symlinks. Prefer imatrix variants.
