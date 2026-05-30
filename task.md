# task.md — 当前待办

> 目标: 让完整 DeepSeek V4 Flash (80.8 GiB base) 在 16 GiB Mac Mini M4 上跑。
> 铁律: **平稳运行 > 物理极限**, 绝不把内存怼红线; 没授权不跑模型; 改完两端要同步重编+rsync。
> 最后更新: 2026-05-30。

## 当前状态快照

- **双机 routed-expert cold tier 代码已落地、功能已验证通**:
  - 协议 EXPERT_REQ/RESP + helpers ✓ (ds4_replica.c/.h)
  - 从机 expert-server (ds4-expert-replica, expert_server_mode 懒加载) ✓
  - 主机拉取路径 + Metal 回调注入 (--expert-remote) ✓
  - 端到端首跑: host 连上从机、握手 OK、`session done: experts_served=12`, 跨 TB 拉 expert 正确。
- **从机 expert_server_mode 懒加载已坐实安全**: idle RSS 22 MiB, 跑时 ≤144 MiB (修复了之前 CPU-backend 82 GiB WILLNEED 全量预读把机器干爆的根因)。
- **本机单机最优 = DS4_DENSE_RESIDENT=1 dense-only, 0.207 t/s (2.3x), 内存安全已验证**。

## 已知结论 / 边界 (不要再重复踩)

- 首跑 host 爆 10.9 GiB 的**真因 = 我在 expert-host-run.sh 里关错了 `DS4_DENSE_RESIDENT`**, 不是物理墙。
  expert-remote tier 设计上**必须配 `DS4_DENSE_RESIDENT=1`** (见 ds4_cli.c:97 帮助文本)。
- 开 DENSE_RESIDENT 后预算 (ctx=2048): ds4 RSS ~9.2-9.5 GiB + macOS ~3-3.5 = 系统 ~12.5-13 → **free ~3 GiB** (紧但单机验证可跑)。
- 速度诚实上限 ~0.85 t/s (routed 从 SSD 4.3s 换 TB 0.7s), **过不了 20 t/s 铁闸** —— 纯为速度不该跑。
- 跨机 MTP 路径已判 NO-GO (0.10 t/s, 比本机直跑还慢, 内存也勉强)。锁仓本机 FIX A + DENSE_RESIDENT。

## 待办 (按优先级)

### P0 — 修脚本 (代码层, 不跑模型, 安全) ✅ 已完成 2026-05-30
- [x] **修 `notes/dual-host/expert-host-run.sh`**: 加回 `export DS4_DENSE_RESIDENT=1` (当前缺失 = 首跑爆内存真因)。
- [x] 同脚本: host 看门狗 ceiling 从 12288 收到 **10752 (10.5 GiB)**, 给真余量, 抢在 thrash 前。
- [x] 复核脚本与 smoke-mtp-host.sh 安全模板一致 (默认 view cap / EXPERT_OFFLOAD / 关 residency+warmup / prefill split) — 顺手补齐缺失的 `VIEW_CAP` 钩子。
- [x] (额外) 纠正头部第 9-10 行反向注释 "禁止 DENSE_RESIDENT" → "必须配", 消除脚本与 task.md 自相矛盾。`bash -n` 绿。

### P1 — 等用户授权才做 (会加载模型)
- [ ] 双机 expert-remote 端到端复跑 (开 DENSE_RESIDENT + 收窄看门狗), 验: free 站稳 ~3 GiB、无 swap thrash、输出连贯、命中率/速度实测。
- [ ] 出结论: t/s、experts_served、host/replica 峰值 RSS、是否守住红线。

### P2 — 收尾 (代码层)
- [ ] 决定双机 expert tier 代码去留: 功能正确但速度过不了 20 t/s 闸。是否合入 / 留作实验分支。
- [ ] 工作区一堆新增文件 (ds4_replica.*, ds4-kv-server, ds4_kv_server.c, ds4_mtp_replica_main.c 等) 与 notes/dual-host/ 脚本的提交/整理策略。

## 不做 / 已否决
- ❌ 关 DENSE_RESIDENT 跑 expert-remote (= 首跑爆内存配置)。
- ❌ 把 expert cache 从 256 MiB 往上加到 2.5G "提命中提速" (怼红线, 已否决)。
- ❌ 跨机 MTP (NO-GO)。
- ❌ 任何"确认安全后再调大提速"的渐进式怼红线动作。
- ❌ macOS 上跑 CPU 推理 (会把内核搞崩)。
