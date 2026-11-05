# A 完成后的 H1／响应／全 Hessian 交接

状态：**PREPARATION_ONLY**。本目录只准备输入依赖与容量清单，没有创建 enabled grant、启动 Q、读取 NPY、导入数值库或执行 Git。未来 A/B/响应结果的路径和 SHA 均未物化；不得把本目录当成实际计算完成证据。

冻结消费者为 `scripts/pyscf_staged_consumer_20261105_v7.py`，SHA `6e9b835f4b49329cd52162b5862da14b3dcd87c80390c1e09b3cf62e14c07dbe`。沿用原 V6b guard 和累计账本。已实际通过的控制 release 为 `actual-water-ethanol-numerical-control-release-001.json`，SHA `82d2138c8e7e514b20dd574b411a1093b8506f29f2698be0474c1c5aebe9b029`；这不代替未来 A 的完整健康闭合。

## 顺序与完成条件

| 步骤 | 必须先有的真实依赖 | 新成功状态／schema | 内容 |
|---|---|---|---|
| B：`h1ao` | 完整 A 健康 `EXITED/0`、`STAGE_COMPLETE`、实际 result SHA；`qualify_partial` 通过 | result `research4-staged-full-rks-hessian-result-v7` / `STAGE_COMPLETE`；manifest `research4-complete-full-rks-stage-cache-v7` | 75 个完整 `[3,762,762]` H1AO |
| C1：10 个 `response-block` | 同一个 A、完整健康 B、原 snapshot/G/release/物理状态不变 | 同 V7 result / `STAGE_COMPLETE`；manifest `research4-complete-full-rks-response-block-cache-v7` | 每块原生全 AO/MO 电子响应，完整原子索引 |
| C2：`global-assemble` | 全部 10 个健康响应结果，按原 B8 顺序形成实际 index/SHA | 同 V7 result / `FREQUENCIES_COMPUTED`；capture `research4-v7-full-global-assembly-capture-v1` | 原生完整 `hess_elec` 全交叉项＋核项一次，219 个有符号模式 |
| 独立分析 | 所有 Q 阶段已闭合、实际 global result SHA | 新 v2 分析器的真实 audit | 完整正常模式复算、虚频和极小点检查；IR 仍未计算 |

响应块的 `block-index` 依次为 0–9：`0–7, 8–15, 16–23, 24–31, 32–39, 40–47, 48–55, 56–63, 64–71, 72–74`。只分核位移 RHS，不截断 AO、占据／虚轨道、诱导势或原子间耦合。每块有效请求容差保持 `1e-8 × 原子数`，50 次迭代上限、level_shift=0。正常返回不冒充独立测得的 CPHF 残差。

所有依赖引用严格只有 `{"path": 实际路径, "sha256": 实际SHA}` 两键。文件大小属于 artifact 清单，不得混入 dependency ref。index 为 `{"schema":"research4-v7-closed-response-results-index-v1","results":[10个真实两键引用]}`；必须在全部 10 个响应真正健康闭合后才写入并计算其 SHA。

## CLI 物化说明

以下 `ACTUAL_...` 全是待 root 填入的参数，不是有效的未来哈希。每一步均用新 phase、output、log，先按**当时**累计 CPU／墙钟余量和磁盘情况收紧额度。

```bash
PYTHONPATH=.runtime/quantum-pilot-site:.runtime/quantum-upgrade-site \
.runtime/bin/python scripts/run_pyscf_continuation_budget_20261105_v6b.py run \
  --ledger reports/pyscf-spinosin-staged-20261105-v6b/cpu_budget.json \
  --phase ACTUAL_FRESH_PHASE --cpu-seconds ACTUAL_BOUNDED_CPU \
  --wall-seconds ACTUAL_BOUNDED_WALL --threads 2 --memory-gib 24 \
  --log ACTUAL_FRESH_LOG -- \
  .runtime/bin/python scripts/pyscf_staged_consumer_20261105_v7.py \
  --execute --threads 2 \
  --snapshot reports/pyscf-spinosin-resumable-20261104-v5/v5-spinosin-C001-prepare-001/exact-mf-snapshot \
  --snapshot-manifest-sha256 3089d6014ba2d2283540eb13cc7dbcdbbf284721042e7250b624197fc15c3311 \
  --gradient-check-result reports/pyscf-spinosin-resumable-20261104-v5/v5-spinosin-C001-restore-gradient-001/result.json \
  --gradient-check-result-sha256 6b143defc238471bc815857df08fc4034a9d8454a00567f3b084907a54a5c403 \
  --control-release reports/pyscf-spinosin-grid-checkpoint-20261105-v7/resource-protocol/actual-water-ethanol-numerical-control-release-001.json \
  --control-release-sha256 82d2138c8e7e514b20dd574b411a1093b8506f29f2698be0474c1c5aebe9b029 \
  --partial-result ACTUAL_HEALTHY_A_RESULT --partial-result-sha256 ACTUAL_CLOSED_A_SHA256 \
  --stage h1ao --output ACTUAL_FRESH_B_OUTPUT
```

响应沿用所有公共参数，将最后的 stage/output 部分替换为：

```text
--stage response-block --block-index ACTUAL_INDEX_0_TO_9
--h1-result ACTUAL_HEALTHY_B_RESULT --h1-result-sha256 ACTUAL_CLOSED_B_SHA256
--output ACTUAL_FRESH_RESPONSE_OUTPUT
```

全局组装同样沿用公共参数，最后替换为：

```text
--stage global-assemble
--h1-result ACTUAL_HEALTHY_B_RESULT --h1-result-sha256 ACTUAL_CLOSED_B_SHA256
--response-index ACTUAL_COMPLETE_INDEX --response-index-sha256 ACTUAL_INDEX_SHA256
--output ACTUAL_FRESH_GLOBAL_OUTPUT
```

禁止给 Spinosin 加 `--control-validation`。guard 累计 cap 仍为 216000 CPU 秒，截止 `2026-11-05T23:20:50.391742Z`；本准备没有增加预算。两线程既有 CPU 边界余量为 6 秒，另有 guard 清理保护，均不是绝对内核保证。`--memory-gib 24` 是虚拟地址上限，不能当作 24 GiB 可用物理内存；实际 RSS／cgroup guard 不变。

## 磁盘与内存

以下为原始 float64 payload，不含 NPY header、manifest、日志、分析 CSV／PNG／PDF／XYZ 和文件系统开销。

| 内容 | 字节 | 十进制 MB |
|---|---:|---:|
| 75 个 H1AO | 1,045,159,200 | 1045.1592 |
| 全部 MO1 | 219,456,000 | 219.456 |
| 全部 MO_E1 | 46,080,000 | 46.08 |
| 响应合计 | 265,536,000 | 265.536 |
| global 主要小数组合计 | 1,209,552 | 1.209552 |
| 以上新增 payload 总量 | 1,311,904,752 | 1311.904752 |

每个 8 原子响应块约 28,323,840 字节，最后 3 原子块约 10,621,440 字节。global 新增原始 Hessian、分析 Hessian 各 405,000 字节，模式 394,200 字节；不生成重复 H1 SAVE_ONLY 副本。

准备时实际空闲为约 **2.569 十进制 GB＝2.393 GiB**；扣除上述已知 payload 后剩约 **1.257 十进制 GB**，尚未扣额外开销。此值只在 JSON 记录的观察时刻有效，不能当成未来峰值磁盘保证。开始 B/C 前必须重新观察，不通过删除历史原件或修改冻结科学代码来腾空间。

6e9 的每个响应 phase 都读取完整 75 个 H1（1.045 GB），不是只读该 RHS 块；10 次逻辑载入合计约 10.452 GB，另外还有真实性 SHA 检查 IO。global 同时载入全 H1 与全 MO1/MO_E1，已知数组 payload 约 1.311 GB。partial provider 只额外复制约 405 KB 的小完整电子 Hessian，没有显式复制第二份全 H1。但 NumPy 匿名数组与源文件 page cache 可同时被计入 cgroup，加上 AO／FXC／Krylov／BLAS scratch 与分配器保留，实际峰值不能仅由 payload 相加推断。root 应继续串行 Q，使用真实 guard 和闭合阶段之间的资源观察；不能将文件缓存当成零内存成本。

实际 global 完成后使用冻结 v2：`scripts/analyze_pyscf_staged_modes_20261105_v7_v2.py`（SHA `76e2edde…`）。完整 219 模式与全正极小点是两项检查，保留负／零模式；理论与实验对照须等独立审计允许，IR 维持 `NOT_COMPUTED`。
