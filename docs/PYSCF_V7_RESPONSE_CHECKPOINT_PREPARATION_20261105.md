# V7 H1、响应块和完整 Hessian 的续算准备

本文件是代码准备记录。实际 H1、CPHF 响应、Hessian、频率和控制计算均为 **NOT_RUN**。21 项纯标准库测试和 metadata preflight 已执行通过；这不构成量子化学结果。A worker 尚未冻结时，`PARTIAL_WORKER_SHA=NOT_FROZEN`，实际入口在数值导入前拒绝执行。

新代码是 `scripts/pyscf_staged_consumer_20261105_v7.py`。它不修改冻结 V6b worker、guard 或历史分析器，不将 V7 产物伪装为 V6b。数学状态沿用真实相同的 `research4-full-rks-physical-binding-v6b`；生产者、结果、H1 cache 和响应 cache 使用新的 V7 schema/source。

仍只允许 root 使用原 `run_pyscf_continuation_budget_20261105_v6b.py`、原累计账本和原截止时刻启动。固定两线程、PBE/def2-SVP、原 grid1 坐标及权重、完整 AO/MO 空间、无 DF、无色散。没有增加任何预算或启动新的 72 小时时钟。

## 各阶段的实际含义

| stage | 计算 | 健康闭合后的产物 | 不代表 |
|---|---|---|---|
| `h1ao` | 原生完整 `make_h1`，所有原子 | 75 个全 AO H1 数组；控制分子按其实际原子数 | Hessian、响应或频率完成 |
| `response-block` | 原生 `solve_mo1`，一个原 B8 核位移 RHS 块 | 该块的 MO1/MO_E1；保留完整原子索引和全 AO/MO 电子响应 | 独立子体系、完整响应或 Hessian |
| `global-assemble` | 原生 `hess_elec` 显式使用完整 H1/MO1/MO_E1，再加一次原生核项 | 完整 Hessian 和 `3N−6` 个有符号模式 | 已验证全正极小点、已确认化学归属或 IR 强度 |

响应分块只分 RHS。斯皮诺素原块为 `[0..7], [8..15], …, [72..74]`，电子哈密顿量、全部 AO/MO、占据/虚轨道和全体系诱导势不截断。每个 phase 重建同一数学全体系 `rhf.gen_vind`，不是跨进程共享一个 Python 对象。原生有效 CPHF 请求容差为 `conv_tol_cpscf × len(block)`，50 次上限、level_shift=0；正常返回不等于独立测得残差达标，`residual_norm=NOT_COMPUTED`。

最终组装向 `hess_elec` 传入全部原子、完整 H1 和所有健康响应块的 MO1/MO_E1，因此仍执行全部原子间交叉收缩。原生 `make_h1` 与 `solve_mo1` 在这个最终 phase 内被禁止重新调用，调用计数应为 `{partial:1,h1:0,response_solver:0,nuclear:1}`。H1 通过已验证的源 manifest 引用，不再建立旧 SAVE_ONLY 的重复约 1 GB 捕获。这里是新的真实 V7 组装流程，不声称执行冻结 V8 SAVE_ONLY kernel。

## 输入链与恢复条件

1. V7 A 必须由真实 `pyscf_xc_chunk_worker_20261105_v7.py` 生产，`qualify_partial` 验证 RHF 与全部有序 XC 块的实际健康闭合、网格覆盖、完整小矩阵和物理状态。
2. 每个 H1/响应结果都核对真实父 proof、源代码 SHA、实际 ledger 健康行、input.json SHA 和 actual argv；失败、RUNNING、源码改变或数组缺失均不能作为 producer。
3. `response-index` 使用 `research4-v7-closed-response-results-index-v1`，`results` 为按原 B8 顺序排列的 `{path,sha256}` 列表。缺块、重复、乱序、不同 A/B、不同物理状态或不同维数均拒绝。
4. `global_assembly_capture.json` 记录真实 A/B/全部响应来源与调用计数。新独立 V7 分析器必须再次验证这条链，之后才允许 NumPy 读取数组和完整矩阵/正常模式复算。
5. 控制分子先做真实水/乙醇多网格块、完整矩阵及全部有符号频率对照。Spinosin 入口另需 root 创建并哈希绑定的实际控制 release；本准备记录不是 release。

建议新输出根为 `reports/pyscf-spinosin-grid-checkpoint-20261105-v7/`，scratch 为 `outputs/pyscf-spinosin-grid-checkpoint-20261105-v7/`。所有重试使用新 phase/output，不能覆盖历史结果。阶段预算按原累计余量收紧，失败成本保留。IR 始终 `NOT_COMPUTED`；理论结果不进入既有分类器。

纯标准库检查：

```bash
python -B -m unittest discover -s tests -p 'test_pyscf_staged_consumer_v7_stdlib.py' -v
```

准备产物保存在 `reports/pyscf-v7-consumer-preparation-20261105-v1/`。source_status.json 是当时草稿的哈希，不是最终冻结发布；后续 pin 和审查须新增记录，不倒改本次 NOT_RUN 快照。

随后源冻结记录位于 `reports/pyscf-v7-consumer-preparation-20261105-v2/SOURCE_FREEZE_RECEIPT.json`。A worker 已固定为 `00e710eb…`，XC core 为 `da1e2b02…`；消费者最终 SHA 为 `6e9b835f4b49329cd52162b5862da14b3dcd87c80390c1e09b3cf62e14c07dbe`。25 项标准库测试及独立实际模块来源／数学审查通过，独立收据见 `reports/pyscf-v7-consumer-independent-review-20261105-v1/`。这些是源码验证证据，实际量子结果另由 root 的真实阶段记录与独立数值审计确定；本文不将源码审查作为量子结果。
