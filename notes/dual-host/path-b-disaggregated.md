# 路径 B 设计 memo — Disaggregated dual-host inference

**Date:** 2026-05-27
**Status:** design + 协议层骨架开工.

## ⚠ 2026-05-27 X9 退出修正 (用户 verdict)

X9 验证 NO-GO (见 [[x9-routing-concentration-data]] 顶部修正). 本 memo 中所有引用 X9 hot pool 的段落 (§3.1 X9 pool ~4.0 GiB / §6 cold expert tier 预读 / §7 X9 单机版作为前置 / §11 与单机 X9 关系) **作废**.

**替代机制**: 编程域 REAP K=8 (3% expert) 静态 prune, routed pool 86 → ~2.7 GiB 全主机静态驻留, 无 LRU, 无 miss path. KV cold-tier 推从机 (主机活跃 KV 4.56 → ~2.0 GiB). 可选 N_EXPERT_USED 6→4 + MTP 2×.

**修正后 per-token (主机本地 hot path, 110 GiB/s)**:
- 4.84 dense + 1.19 routed (REAP K=8 + 6→4) + 2.0 KV (cold-tier 推从机后) + 0.34 idx = **8.37 GiB / token**
- 76 ms → **13.2 t/s** + MTP 2× = **~26 t/s** ✓ 余量比 X9 方案大

**修正后主机静态驻留**:
- 2.0 OS + 4.84 dense + 2.7 routed (REAP K=8) + 2.0 KV active + 0.34 idx ≈ **11.9 GiB** ✓ 在 13 GiB usable 内
- 从机: 2.0 OS + 4.84 dense + 2.56 KV cold-tier + 2.0 prefill working + (可选 REAP 剔除的 expert 兜底) ≈ **9.4 GiB** ✓ 在 ~8 GiB 切片内 (紧, 但可调)

**工程动作变化**:
- 砍掉 X9 LRU / miss handler / hot pool 管理 (M2 整段不要了)
- 加 REAP K=8 离线 GGUF 生成工具 (在 gguf-tools/, 不动 ds4 hot path)
- 加编程域 fine-tune 配套 (离线工作, 独立 milestone)
- KV streaming 协议 / ds4-replica 进程 / disaggregated 拓扑 **不变** — 与 X9 解耦的部分仍是 path-B 的基础设施, 立即可开工

**新 milestone (替换 §9)**:

| ID | 名称 | 备注 |
|---|---|---|
| M0 | Lever A runtime smoke | 不变, X9 退出不影响 |
| M1 | RDMA-over-TB spike | 不变 |
| M2' | ds4-replica 进程 + IPC 协议层 | 替换原 M2 (X9), **可立即开工** |
| M3 | KV streaming RPC 落地 | 不变 |
| M4 | REAP K=8 GGUF 生成 + 加载兼容 | 新 milestone, 离线 + ds4 loader 改 |
| M5 | 主机 KV cold-tier push to replica | 新 milestone, 主机 hot path 改, 需 M0 |
| M6 | MTP dual-host | 不变 |
| M7 | Claude Code 端到端 | 不变 |

---


**Author context:** 接 `notes/execution-log.md` 2026-05-27 双机可行性研究条目, 与 [[decision-gate-20tps]] / [[goal-and-constraints]] / [[lever-a-landed-state]] / [[x9-routing-concentration-data]] 衔接.

---

## 1. 目的与边界

让 DS4 在 **Mac Mini M4 16G (主机) + MacBook 16G 切 8G (从机) + Thunderbolt 直连 (上限 5 GB/s 单向)** 双机拓扑下满足 ≥ 20 t/s decode, **不改 V4 Flash IQ2XXS 模型**, 不缩 1M ctx, 让 Claude Code 本地客户端能通过主机的 ds4-server endpoint 真实可用.

**Hard locks (来自 [[decision-gate-20tps]]):**
- 模型锁: DeepSeek V4 Flash IQ2XXS 86 GiB, 不允许 prune / 换模型作为达 20 t/s 的手段
- 速度门: ≥ 20 t/s decode (物理推算必须先过)
- 1M ctx 锁: 不允许缩 ctx 作为达 20 t/s 的手段
- 网桥锁: TB 5 GB/s 单向, hot-path 不许过网桥

## 2. 拓扑

```
┌─────────────────────────────┐         TB 5 GB/s          ┌──────────────────────────┐
│  Host: Mac Mini M4 16 GiB   │ ◄────────────────────────► │  Replica: MacBook 16 GiB │
│  (12-13 GiB usable)         │   RDMA preferred, TCP OK   │  (8 GiB carved)          │
│                             │                            │                          │
│  Role: DECODE                │                            │  Role: PREFILL + KV     │
│  - ds4-server endpoint       │                            │  - prefill engine inst.  │
│  - full graph eval per token │                            │  - KV writer (FP8 row)   │
│  - X9 hot expert RAM pool    │                            │  - persistent KV store   │
│  - sampling + tool-call DSML │                            │  - cold expert tier      │
│                             │                            │                          │
│  Owns:                      │                            │  Owns:                   │
│    Dense hot (4.84 GiB)     │                            │    Dense hot (4.84 GiB)  │
│    X9 expert pool (~4 GiB)  │                            │    KV cold-store (FS)    │
│    KV live (4.56 GiB @ 1M)  │                            │    Cold expert tier      │
│    Indexer (0.34 GiB)       │                            │      (~3 GiB hot subset) │
└─────────────────────────────┘                            └──────────────────────────┘
        Claude Code client                                            (no client)
        OpenAI/Anthropic API
```

**主机在 hot path 上是单点** — decode 每 token 不跨机. 从机只在两种时刻参与:
1. **Prefill 阶段**: 接 prompt, 跑全模型, 算出 layer-by-layer KV, 流给主机 (一次性, cold start)
2. **Cold expert miss**: 主机 X9 pool miss 时, 从从机 expert tier 拉 (~5% rate, off hot path)

dense weights (4.84 GiB) 在双机各自存一份 — **是冗余, 不是浪费**. 跨机传 dense 不可行 (TB 5 GB/s / 120 GB/s = 24× slowdown 直接砍掉 20 t/s).

## 3. 字节预算

### 3.1 静态驻留 (RAM 占用)

| 组件 | 主机 (GiB) | 从机 (GiB) | 说明 |
|---|---|---|---|
| OS lean baseline | 2.0 | 2.0 | macOS 26.2 实测 1.8-2.5 |
| Dense hot (output proj + 43×attn Q8 + shared expert + compressor/indexer/gate/norm) | 4.84 | 4.84 | mmap 共享, 各自一份 |
| X9 hot expert pool (1024 expert × ~4 MiB avg) | ~4.0 | — | 主机独占, K=16 命中 95.6% |
| KV live FP8 @ 1M ctx (Lever A landed, `ds4.c:121` row=608B) | 4.56 | — | 主机活跃 KV |
| Indexer KV @ 1M | 0.34 | — | 主机 |
| Cold expert tier (replica subset, ~768 expert × ~4 MiB) | — | ~3.0 | 从机 LRU 二级 |
| KV cold-store (persistent disk-backed) | — | <0.5 RAM | mmap, 主体在 SSD |
| Prefill engine working set (transient) | — | ~2.0 | prefill 时短暂峰值 |
| **合计 (decode 稳态)** | **~15.7** | **~9.8** | 主机超 13 GiB 红线 ~2.7 GiB |

**主机 ~2.7 GiB 超预算** — 必须收缩。两个落地杠杆:
- 把 X9 pool 从 4.0 GiB → ~2.5 GiB (K=12 命中率约 88-92%, 见 [[x9-routing-concentration-data]])
- 把 KV 在 decode 稳态下从 4.56 GiB → 把 disk-tier ratio-4 (lever C) 拿出 ~1.5 GiB cold KV 推到从机

实战取舍待 §9 后再定; 物理上限不依赖具体取哪种.

### 3.2 Per-token decode 读量 (主机 hot path, 不过网桥)

| 成分 | 字节/token | 注 |
|---|---|---|
| Dense weights | 4.84 GiB | 必读全量 |
| Routed experts (43 × N_EXPERT_USED=6 × ~7 MiB) | 1.78 GiB | 95.6% 在主机 X9 pool 命中 |
| KV attn @ 1M (FP8 608B/row, ratio-4 indexed) | ~7 MiB | indexed gather, 不是全扫 |
| Indexer KV scan @ 1M | 0.34 GiB | 全扫 (lever B 后可缩) |
| **合计** | **~6.97 GiB/token** | 主机 RAM 路径 |

参 [[decode-bandwidth-ceiling-correction]] 公式, 数字一致 (旧 memory 单机假设也是这个 per-token 量, 因为读量本身与拓扑无关; 区别在 cold expert miss 是否走 SSD).

### 3.3 跨网桥流量 (off hot path)

| 事件 | 大小 | 网桥耗时 (5 GB/s + 50 µs RDMA / + 300 µs TCP) |
|---|---|---|
| KV cold-start stream (4.56 GiB layer-by-layer) | 4.56 GiB | ~912 ms 一次性 |
| Cold expert miss (~5% × 7 MiB) | 0.35 MiB/token avg | 70 µs RDMA / 370 µs TCP |
| Prefill 完成 → 主机 handover ack | <1 KB | <1 µs |
| Decode tool-call DSML 给主机 (无 KV 同步) | <8 KB | <2 µs |

## 4. 物理上限推算

**主机 RAM 带宽**: M4 base 120 GB/s 标称, 实测 ~80-110 GiB/s (取保守值 90 GiB/s = 96.6 GiB/s 换算).

**Decode per-token time (hot path 全主机):**
- Pure memory: 6.97 GiB / 96.6 GiB/s = **72 ms**
- Cold expert miss: 5% × 70 µs (RDMA) = 3.5 µs/token ⇒ 忽略
- GPU compute overhead (Q2 dot kernel ~60-80 GiB/s 等效): ~20-30 ms
- CB scheduling (43 层, post-A3): ~5 ms
- **合计**: ~100 ms/token ⇒ **~10 t/s**

**加 MTP 2x (80% accept)**: ~20 t/s ✓ **过 20 t/s 铁律**

**加 N_EXPERT_USED 6→4 编程域 (可选)**: per-token dense+routed 缩 ~7%, ~93 ms ⇒ ~10.7 t/s, +MTP ~21.5 t/s, 余量更宽

**Prefill (在从机):**
- 从机做 compute-bound 阶段, 不堵主机 decode
- 1k prompt prefill 从机带宽 90 GiB/s × 全模型读 ≈ 同主机量级
- TTFB = 从机 prefill + 4.56 GiB KV stream (~912 ms) = **~2-5s @ 32k ctx** (估)
- 首 token 后 decode 全主机, decode rate 不受影响

**关键结论**: 物理上限 ~20-22 t/s decode, **过铁律**, hot path 不依赖网桥, 不依赖 REAP.

## 5. KV streaming 协议

接口与 [[lever-a-landed-state]] FP8 row (608 B) 兼容. 不重新设计 KV 字节布局.

### 5.1 协议形状

从机算完一层 attn KV, 立刻通过网桥发给主机:

```
struct kv_stream_msg {
  uint32_t layer_idx;       // 0..42
  uint32_t row_start;       // 该 layer 的 row 起点 (token 偏移)
  uint32_t row_count;       // 本 chunk 的 row 数
  uint32_t flags;           // bit0=is_last_chunk_of_layer, bit1=is_last_layer
  uint8_t  rows[row_count * 608];  // FP8 attn_comp_kv rows (Lever A 格式)
};
```

每条消息 ≤ ~1 MiB (chunk size 调优). 主机收到立即 `memcpy` 进 `g->layer_attn_comp_cache[il]` (`ds4.c:9058`), 不解码, 不转换 — 这是 FP8 实存储 (Lever A) 的红利, **跨机字节流就是主机内存字节布局, 零拷贝路径**.

### 5.2 RDMA vs TCP

- **RDMA (preferred)**: 从机 RDMA-write 直接落主机内存 `g->layer_attn_comp_cache[il] + row_start*608`. 主机零参与, 内存到位即可读. Latency < 50 µs/chunk.
- **TCP fallback**: 从机 send → 主机 recv → memcpy. Latency 300+ µs/chunk, 但 chunk 大 (~1 MiB) 时占比小. 1 MiB / 5 GB/s = 200 µs transfer dominates, latency 加 50% open.

### 5.3 1M ctx 实际工作语义

主机维护 `ds4_session` 的 KV 切片. 从机 prefill 完成后:
1. 从机 layer 0 算完 KV ⇒ stream → 主机
2. 主机收到 layer 0 ⇒ ack, 从机开始 layer 1
3. 串行 43 层, 流水重叠 (从机算 N+1 时主机正在收 N)
4. 全部完成 ⇒ 从机 `handover_ack` ⇒ 主机开始 decode

**不允许的语义**:
- decode 期间从机继续 push KV (主机已经在改 KV) — 协议简化只支持 cold start handover, 不支持运行时双写
- 主机直接 PR rewrite KV 同时让从机做 prefill 续接 — 会破坏 ds4-server 现有 "common prefix → rewrite vs rebuild" 决策 (CLAUDE.md "Engine boundary" 段)

session-level common prefix 复用仍走主机本地 `ds4_kvstore` (mmap, sha1-of-rendered-prefix), 不走双机. 双机只在 "全 prompt 是新的 / 无共同前缀命中" 时介入.

## 6. 工作分配 / Layer 分片决策

**简化的边界**: 从机做 **完整的 prefill 全 43 层**, 不做 layer-cross-machine 切分.

为什么不切 layer:
- Layer 切会让 decode 每 token 跨机 ≥ 2 次 (路径 A), 我们已经排除了
- 拓扑 B 的核心思想是 phase split, 不是 layer split
- 主机 dense 已全量驻留 ⇒ decode 完全本地

**唯一切分**: prefill 期 cold expert tier — 从机预读热的 ~768 expert (3 GiB) 主动 push 给主机更新 X9 pool (基于 prompt-time router 统计, 比 decode-time miss-on-demand 命中率更高).

## 7. 与现有 enabling 的衔接

按依赖顺序:

1. **[[lever-a-landed-state]] FP8 KV** ⇒ patch #34 已 land, kernel test 绿. **runtime smoke 是 B 路径前置门** — 因为 KV 字节布局是双机协议的基础.
2. **[[x9-routing-concentration-data]] X9 GO 信号** ⇒ K=16 95.6% / K=10 ~85% 已实测. **X9 单机版必须先跑通**, 再做双机版的 cold tier (从机 expert server).
3. **DSML tool-call 协议不变** ⇒ ds4-server 现有的 exact-replay map (CLAUDE.md "Tool calls" 段) 完全在主机, 不跨机. 双机对 tool use 透明.
4. **disk KV store** ⇒ 主机仍写 `ds4_kvstore` (sha1 + 48B header + ext flags + payload v4 + tool id map), 不动协议. 从机的 KV 持久化层是**可选的二级 cache**, 不是 source of truth.

**没有任何已落地代码需要回退**. 路径 B 是叠加, 不是 fork.

## 8. RDMA over TB 可用性前置门

**问题**: Apple 公开的 RDMA-over-TB demo (macOS 26.2) 集中在 **Mac Studio M3 Ultra + TB5**. 主机 Mac Mini M4 base + 从机 MacBook (大概率 TB4 级) 上是否支持是公开未知.

**前置门 (M1 验证, 单独 spike, 不动 DS4 代码)**:
1. macOS 26.2 双机, TB 直连
2. `dt_send` / `mlx-distributed` / EXO 1.0 任一 RDMA 路径跑通
3. 实测 1 MiB chunk RTT, 期望 RDMA < 200 µs, TCP 300-500 µs
4. 若 RDMA 不可用, 评估 TCP 在 KV streaming 场景下的实际占比 (~5-15%, 仍可接受)

**Go / No-go**:
- RDMA 可用 ⇒ 优先 RDMA 路径, KV cold-start 接近 4.56 GiB / 5 GB/s 物理底 (~912 ms)
- 仅 TCP 可用 ⇒ 仍走路径 B, KV stream chunk 大小调优为 ≥4 MiB 摊薄 latency
- 都不可用 (TB 直连未识别) ⇒ no-go, 退回单机 X9 (本路径作废, 不是"勉强降速")

## 9. 工程量分解 (milestones)

| ID | 名称 | 依赖 | 工作 | 验证 |
|---|---|---|---|---|
| M0 | Lever A runtime smoke | 无 | 现有代码 smoke | KL ≤ 1e-3 vs pre-#34, KV 实测 4.56 GiB @ 1M |
| M1 | RDMA-over-TB spike | 无 | 双机 macOS 26.2 + TB + EXO 1.0 / mlx-distributed 之一 | RDMA 1 MiB chunk RTT < 200 µs 实测 |
| M2 | X9 单机 landed | M0 | 现有 X9 设计落代码 | 95% 编程域 prompt decode ≥ 8 t/s |
| M3 | dual-host runtime skeleton | M1, M2 | 引入 ds4-replica 进程 + KV stream RPC + cold tier server | 1k ctx 双机 decode ≥ M2 单机持平 (不退化) |
| M4 | KV streaming 落地 | M3 | M3 协议跑通 + 1M ctx KV cold-start 实测 < 1.5s | 1M ctx TTFB < 5s, decode ≥ M2 |
| M5 | Cold expert tier server | M3 | 从机 expert server + 主机 X9 miss 拉取路径 | 5% miss 走从机 + cold rate < 8% on long-session 编程域 |
| M6 | MTP dual-host | M4 | MTP 2x 在 disaggregated 拓扑下不退化 | decode ≥ 18 t/s @ 1k ctx, ≥ 15 t/s @ 32k ctx |
| M7 | Claude Code 端到端 | M5, M6 | ds4-server 主机起 endpoint, Claude Code 客户端配 base-url | 真 Claude Code session, tool use + 长 ctx 不退化, ≥ 20 t/s decode 实测 |

每个 milestone 独立可验, 失败可回退到上一个状态.

## 10. 风险表

| 风险 | 影响 | 缓解 |
|---|---|---|
| M4 base 内存带宽实测 < 90 GiB/s | decode 上限 < 20 t/s | 测真实带宽, 若 < 80 GiB/s 触发 N_EXPERT_USED 6→4 编程域 |
| RDMA over TB 在 M4 base + TB4 不可用 | KV streaming 慢 ~1.5×, microbatch 路径全废 | TCP fallback (M1 评估), chunk size 调优 |
| 从机 MacBook 是 M2/M3 (100 GB/s) 而非 M4 | prefill 慢 20% | 不影响 decode 路径; TTFB 增 ~1s |
| X9 pool 在长会话漂移, 命中率 < 80% | hot path 退化, cold tier miss 飙升 | M5 实测 ≥200 token 长会话, 命中率 gate, 不达回退到方案 C 简化版 |
| MTP 2× accept rate 在双机拓扑退化 | 实际加速 < 1.6× | M6 实测对比单机 MTP accept rate; 退化则降到 MTP 1.5× 仍过 20 |
| KV cold-start TTFB > 5s | 用户感知差 | M4 测实际 TTFB, > 5s 触发 chunk size + RDMA 调优 |
| 双机 process crash 一致性 | 主机重启需要从机重 prefill | 主机 KV 持久化到 `ds4_kvstore` 仍主拷; 从机重连只重做未持久化的尾部 |
| ds4-replica 引入新进程 / IPC bug 面 | 测试矩阵翻倍 | replica 进程范围小 (prefill + KV stream + expert serve), 单元测试 + integration 实测 |

## 11. 与现有单机路径的关系

**单机 X9 路径**: 路径 B 的子集 (M2 完成即得到的单机产物). 单机 X9 自身 decode 上限被 [[cb-floor-truth]] 修正为 ~1.9 t/s (因 routed pool SSD bound), **过不了 20 t/s 铁律**, 但作为 disaggregated 拓扑的 hot pool 实现是必备前置.

**优先级**: M0 → M1 (并行) → M2 (单机 X9) → M3..M7 (双机) — 这条线让"先有可用的单机 fallback (M2 完后即可单机跑 8 t/s), 再叠加双机过 20 t/s". 任何 milestone 失败不影响已落地的产物.

**不动单机路径**: M2 完成时主机仍可纯单机跑, 双机是 opt-in 拓扑.

## 12. 不动代码前可做的物理 spike

(用户授权前我都不会主动跑, 仅列出可做项以便决策)

| Spike | 工作 | 输出 | 工作量 |
|---|---|---|---|
| S1 | M4 base RAM 实测带宽 (memcpy / membench) | 实测 GiB/s, 验证 90 GiB/s 假设 | 30 分钟 |
| S2 | 测从机型号 + RAM 带宽 | 确认 M2/M3/M4, 算从机 prefill 速度 | 10 分钟 |
| S3 | macOS 26.2 RDMA over TB 双机连通性 (EXO 1.0 或 mlx-distributed hello-world) | RDMA 1 MiB chunk RTT 实测 | 1-2 小时 |
| S4 | M0 Lever A runtime smoke (KL + KV 字节实测) | Lever A 是否可作为 B 路径基础 | 1 小时 |
| S5 | 编程域 cross-layer expert Jaccard 实测延长到 ≥ 200 token | X9 长会话命中率假设验证 | 2-3 小时 (现有工具) |

**推荐顺序**: S4 (前置门, [[lever-a-landed-state]] 已等很久) → S1 + S2 (10-40 分钟即得真实数字) → S3 (决定 RDMA 路径)。S5 是 X9 落地后的事, 现在做属早.

## 13. 待用户决策项

1. memo 范围是否够用, 还是要展开某节 (比如 KV streaming RPC 协议字节级 / cold expert tier 的 LRU 策略)
2. 是否先跑 S4 (Lever A smoke) 把主线 enabling 闭环, 还是先跑 S3 (RDMA) 把网桥可用性兜底
3. 是否需要把 [[cb-floor-truth]] / [[decode-bandwidth-ceiling-correction]] 现在回填路径 B 字节预算, 还是等 S1/S4 实测后再统一回填
4. ds4-replica 这个新进程的名字 / 边界 (现在 memo 里假设它是 ds4 binary 复用, 加 `--role=replica` flag) 是否要单独立项设计

---

## 引用

- 现有 memory: [[decision-gate-20tps]] / [[goal-and-constraints]] / [[lever-a-landed-state]] / [[x9-routing-concentration-data]] / [[cb-floor-truth]] / [[decode-bandwidth-ceiling-correction]] / [[metal-wait-anukari-precedent]] / [[metal-buffer-residency-per-buffer-granularity]]
- 引用源码:
  - `ds4.c:121` DSV4_FP8_ATTN_ROW_BYTES (Lever A row size)
  - `ds4.c:9058` g->layer_attn_comp_cache[il] (KV stream 目的内存)
  - `ds4.c:1739` / `ds4.c:1767` FP8 quantize / decode (Lever A producer/decoder)
  - `ds4.c:16343` snapshot payload v4 (含 FP8 attn_comp)
  - `ds4_metal.m:13525-13538` ds4_gpu_router_dump_enabled (X9 Step 0 helper)
  - `ds4_metal.m:13738-13801` A3 sync block dump 注入点
  - `ds4_server.c` Engine boundary (CLAUDE.md "Engine boundary" 段)
  - `ds4_kvstore.c` disk KV cache (CLAUDE.md "Files" 段)
- 外部参考:
  - EXO Day 1 benchmarks (https://blog.exolabs.net/day-1/)
  - DGX Spark + M3 Ultra disaggregated inference (Tom's Hardware)
  - macOS 26.2 RDMA over TB (Jeff Geerling, 2025)
  - EdgeShard (arXiv 2405.14371) — 异构边缘 LLM 分片 DP
  - Speculative decoding in decentralized inference (arXiv 2511.11733)
