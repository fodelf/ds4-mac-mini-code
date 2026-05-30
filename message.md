# 双机 MTP 部署环境速查 (message.md)

> 两台 Mac 跨 Thunderbolt 网桥做 MTP 调度分离。本机跑 target，另一台跑 MTP drafter (replica)。
> 本文件记录网桥 / SSH / 机器 / 传输 / 部署的实测信息，避免每次重新探查。
> 最后更新: 2026-05-29。

## 两台机器

| | 本机 (host / target) | 另一台 (replica / MTP drafter) |
|---|---|---|
| hostname | `192.168.2.90` | `fodelf.local` |
| 芯片 | Apple M4 (Mac Mini) | **Apple M1 Pro** |
| 内存 | 16 GiB | 16 GiB |
| HOME | `/Users/ea34y0` | `/Users/ea34y0` (user `fodelf`) |
| 磁盘可用 | — | 44 GiB (460Gi 总, 90% used) |
| 角色 | 跑完整 base 的 target decode | 只用 base 的 embd+output + MTP，做 draft |

## 网络 / 网桥

- **用 `bridge0` = Thunderbolt 网桥** 互联 (member en2/en3/en4，status active)。
  - 本机 bridge0 = `192.168.1.3`
  - 另一台 bridge0 = `192.168.1.2`  ← **ssh/rsync 都走这个 IP**
  - `route -n get 192.168.1.2` → interface `bridge0` (已确认走网桥)
- **勿用** 本机 `en5 = 169.254.90.76`，media 是 `100baseTX`(**100 Mbit ≈ 12 MB/s**，极慢)。
- 其它接口(参考)：本机 en1=192.168.2.90；另一台 en0=192.168.2.18, en12=169.254.156.249。

## SSH

```sh
ssh fodelf@192.168.1.2          # 免密已配 (BatchMode=yes 可用)
```

## 文件传输 (rsync over ssh)

- 实测 **204 MB/s**。瓶颈是 **ssh 单核加密**，不是 TB 网桥带宽 (40 Gbps)。
- **提速**(将来重传大文件)：Apple Silicon 有 AES 硬件加速，用
  ```sh
  rsync -e 'ssh -c aes128-gcm@openssh.com' ...   # 默认 chacha20 走软件，慢
  ```
- 大文件(GGUF)用 `--inplace --partial`(断点续传 + 省临时空间，另一台磁盘紧)。
- **铁律：绝不用 `--delete`**(只增不删，遵守 no-remote-delete)。

## 另一台已部署 (`~/ds4-main/`)

| 文件 | 状态 |
|---|---|
| `ds4flash.gguf` → `gguf/...IQ2XXS...gguf` | 81G base (symlink 已建) |
| `gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` | 3.5G MTP |
| `ds4-mtp-replica` | 728K binary (本机 M4 编译，arm64) |
| `smoke-mtp-replica.sh` / `smoke-mtp-host.sh` | 已传 |
| 全套源码 + Makefile + `metal/`(19 kernels) + notes + tests | 已传 |
| `gguf/ds4flash-k16.gguf`(13G) / `k48.gguf`(22G) | **未传**(replica 用不上) |

## 跑法

```sh
# 另一台 (replica，先起):
cd ~/ds4-main && ./smoke-mtp-replica.sh 17502        # 8G 看门狗，listen 等连入

# 本机 (host，后起):
REMOTE=192.168.1.2:17502 ./smoke-mtp-host.sh 16 "1+1="   # 12G 看门狗
```

## 当前状态 / 待解决 (2026-05-29)

- ✅ 代码 + base + MTP 已传到另一台，全套就位。
- ❌ **replica 启动失败 (非内存爆)**：另一台日志
  ```
  ds4: Metal device Apple M1 Pro, 16.00 GiB RAM
  ds4: Metal model view max bytes overridden from 8.00 GiB to 2.00 GiB via DS4_METAL_MODEL_MAX_VIEW_BYTES
  ds4: Metal model needs more mapped views than expected
  ds4: metal failed to map model views; aborting startup.
  [ds4-mtp-replica] failed to open engine
  ```
  进程干净 abort(没爆内存)。
- **根因(已查源码)**：`ds4_metal.m:594` `step = max_buffer − overlap`，`overlap ≈ 最大单张量(~1GiB)+1页`，view 数 ≈ `model_size / step`。`smoke-mtp-replica.sh` 照搬了 host 的 `DS4_METAL_MODEL_MAX_VIEW_BYTES=2GiB` → step≈1GiB → 81G 算出 **~80+ 个 view**，超过 `DS4_METAL_MAX_MODEL_VIEWS=64`(行250) → map abort。
- **修复(已改)**：host 跑 43 层 experts 才需要小 view(2GiB)压每层 wire；replica 只 touch embd+output，要【大】view 减少 view 数。`smoke-mtp-replica.sh` 改为 `${VIEW_CAP:-4294967296}`(**4GiB**)→ step≈3GiB → ~28 view(<64)。`VIEW_CAP=` 可覆盖再调。
- ✅ **replica 已起来**(4GiB view cap)：base map 成 `27 overlapping shared buffers`(<64) + `listening on 0.0.0.0:17502 hc_floats=16384`；**idle RSS 仅 44 MiB**(map 不 wire 物理内存)。在另一台 16G 上监听等连入。
- **待**：本机 host 连入(`REMOTE=192.168.1.2:17502 ./smoke-mtp-host.sh 16 "1+1="`)；draft 时 replica 才 wire embd+output，那时看真实 RSS(另一台 16G，看门狗可从 8G 放宽)。

## 关键经验 (踩过的坑)

1. `192.168.1.2` 是**第二台真机**(M1 Pro)，不是回环；本机 bridge0 是 `192.168.1.3`。
2. env 名：`DS4_METAL_MODEL_MAX_VIEW_BYTES` 才对；`DS4_METAL_MODEL_MAX_TENSOR_BYTES`/`DS4_METAL_MAX_MODEL_VIEWS` 是错的(后者是源码 #define 非 env)。
3. host 与 replica 的 view cap **方向相反**：host 小(2G 压 wire)、replica 大(4G 减 view 数)。
4. 传输瓶颈是 ssh 加密非网桥；大文件加 `-e 'ssh -c aes128-gcm@openssh.com'`。
5. 两台都 16G(M4 + M1 Pro)；之前以为另一台 8G 是错的 —— 看门狗阈值可相应放宽。
