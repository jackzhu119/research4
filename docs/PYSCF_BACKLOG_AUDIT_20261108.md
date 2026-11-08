# PySCF 已完成结果、未完成项与本轮后台优先队列

审计日期：2026-11-08（北京时间）。本文件是对已保存证据的整理和本轮队列依据，**不是新的量子计算结果或后台存活证明**。本轮用户另行授权总计 90,000 CPU 秒；旧授权、旧成本、旧失败与新账本分开保存，不重复发放旧余额，不改历史结果。

## 1. 三个版本入口必须分开

| 入口 | 本次审计固定提交 | 用途 |
| --- | --- | --- |
| `main` | `806584a5571c6a90ef37c6bfa915671b680ba48b` | 当前光谱有效清单与种类、产地标签 |
| `research/pyscf-spinosin-mechanism-20261031`（A） | `cd4b50c37c1e26064f9f15f340d30e20ac386458` | Spinosin、完整响应、后续 SCF 与积分网格诊断 |
| `research/parallel-worker-b-20261106-B20261106-01`（B） | `71d16678d831b65206fd355d14b9a7f5462ca23b` | 原儿茶酸、Calycosin 及后续数值方向、子空间验证 |

提交后的分支可能继续变化；下述证据均指上述固定提交，不以旧 README 的“最新”标题替代实际结果。

当前有效数据版本为 `2026-11-08-confirmed-origin-kind-v1`：18 份工作簿、1672 条谱记录，其中枣仁 364、酸枣仁 500、黄芪 403、天麻 405。默认入口是 [active_workbooks.json](../data/active_workbooks.json)，不得递归收集旧归档和汇总表当新增样品。原 ZY01—ZY07 的 855 条历史记录现已知混合枣仁和酸枣仁；历史 96.0234% 或 90.41% 仍保留原数据、划分与指标，不能称为当前 18 表上的实测结果或纯酸枣仁产地性能。本轮分子计算也不训练分类器、不修改光谱或标签。

## 2. Spinosin：完整计算已做，最低点资格仍未通过

证据入口：A 提交中的 [完整振动报告](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/PYSCF_SPINOSIN_COMPLETION_20261106.md)、[A004 实际闭合汇总](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/cloud-followup-20261106-v1/queue-host-a-004/science/postrun-analysis-001/README.md)和 [72 小时首批复核](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/RESEARCH4_72H_BATCH_REVIEW_20261108.md)。

| 项目 | 已保存状态 | 实际范围 |
| --- | --- | --- |
| 可信结构与原 C001 优化、fresh 梯度 | 已做 | C28H32O15、75 原子；PBE/def2-SVP/grid1/direct，明确无 D3；计算模型构象，不是实验晶体坐标 |
| 完整 Hessian、219 有符号模式及独立数学重放 | 已做 | 194 正、25 虚；虚频约 −312.34 至 −87.69 cm⁻¹；最低点与机制资格均为 false |
| 保存响应的 fresh 方程残差 | **225/225 RHS 已测完** | A004 补齐后已不再是 11 月 6 日早期报告中的 6/225；只核对给定响应方程，不证明 H1、A 或全部核二阶导正确；没有事后新增通过阈值 |
| 模式 1 双步长、模式 26 的 SCF/梯度方向诊断 | 已做 | A003 的 6 个实际位移点仍保存；模式 1 的旧解析负曲率与数值梯度正曲率不一致，不能直接归因于某一个实现错误 |
| 电子轨道稳定性 | 部分尝试、失败闭合 | A004 `spinosin-orbital-stability` 为 `FAILED_CLOSED`，成本 8881.818772 CPU 秒；不能称为未尝试，也不能称稳定性通过 |
| 新完整势能每轮重算的零点 | 已做 | 首批新零点电子 SCF 认证通过；电子收敛不等于核驻点或最低点 |
| 更严格方向能量曲率 | 失败保留 | 首批 mode 1 / mode 26 首个位移最终轨道残差分别约 2.5335e-9 / 3.1141e-9，大于 1e-9；非资源强杀 |
| 重建积分网格的新控制 | 第二批已通过 | `autonomous-recovery-002` 中 10 个水/乙醇控制点通过，实际 82.682657 CPU 秒 |
| Spinosin 重建 grid1 新零点与梯度 | 失败闭合 | 实际 5842.250239 CPU 秒；两次 SCF 的 final-extra 能量漂移分别 3.638e-12 / 5.457e-12 Eh，大于原 1e-12，虽然轨道残差约 5.60e-11 / 9.50e-11；未获得合格新零点，未进入梯度阶段 |
| Spinosin 重建 grid3 新零点与梯度 | 没有完成证据 | 固定 A 提交中的 recovery state 尚无该 job 条目；配置存在不代表实际执行或通过 |

第二批的确切证据是 A 提交中的 `reports/research72h-20261107-v1/autonomous-recovery-002/state.json` 与 `jobs/recovery-spino-grid-1-zero-002/attempt-001/{result.json,fresh-zero/native_result.json}`。该 state 更新时间为 UTC 2026-11-08 03:38，晚于说明文件首次写入的“准备完成、尚未运行”，应以结果和闭合账本为准。

仍缺的是：合格的新方法核驻点、完整导数不一致的受控解释、新完整并经验证的频率、可靠最低点资格。IR 强度仍 `NOT_COMPUTED`；化学峰归属与标准品 THz/HPLC/LC-MS 实测均没有取得，不能靠删虚频、取绝对值或仅模式图补齐。

## 3. Worker B：两种药材小分子的原 Opt/Freq 已完成

B 固定提交中共同前缀为 `reports/parallel-cloud-20261106/worker-b/B20261106-01/`，下称 `B/`。

| 项目 | 已完成证据 | 结果与缺口 |
| --- | --- | --- |
| 原儿茶酸初始 Opt/Freq/模式重放 | `B/science/protocatechuic_acid-{optimize,frequency,analyze}-attempt001/result.json` | 17 原子、C7H6O4、184 AO；45 模式中 4 虚；原优化约 1199.96、频率约 1059.00 CPU 秒；最低点 false |
| Calycosin 初始 Opt/Freq/模式重放 | `B/science/calycosin-{optimize,frequency,analyze}-attempt001/result.json` | 33 原子、C16H12O5、354 AO；93 模式中 7 虚；原优化约 10340.89、频率约 10684.16 CPU 秒；最低点 false |
| 更严格网格、SCF、固定/重建网格方向验证 | `B/followup-validation-recovery-v6/science/all-validation-points-comparison-attempt001/` | grid3/grid5、有限步长和网格响应控制已做；不能再当全未运行 |
| 原全部虚频子空间的数值验证 | `B/followup-negative-subspace-v7/science/{acid-g3-negative-subspace,caly-g3-negative-subspace}-attempt001/result.json` | 原儿茶酸全部 4 方向、Calycosin 全部 7 方向及耦合已测；True 梯度子空间最小诊断特征值约 0.002692606 / 0.002721932 Eh/Bohr²，均正 |
| 子空间组合方向独立位移复核 | `B/followup-negative-subspace-v7/science/all-negative-subspace-comparison-attempt001/analysis_conclusion_zh.txt` | 组合最小方向另做正负位移及 0.001/0.0005 Bohr 双步长；有限方向验证有正结果，但没有完整 Hessian 重算或全部方向步长收敛证明 |

最近一轮 B 量子队列在 UTC 2026-11-08 05:22（北京时间 13:22）实际闭合。核查范围内尚未发现新 grid3 严格优化或完整数值 Hessian；后续 `yanan-cause-followup-v2` 是光谱机器学习工作，不能算分子计算。

原儿茶酸原优化使用 grid1、`gradient_grid_response=false`、SCF 容差 1e-8，13 个实际优化循环后收敛；fresh 原子梯度 RMS 8.6750e-5、最大 1.8362e-4 Eh/Bohr。后续 grid3 center 使用原核坐标，求更严格 SCF 和 True/False 梯度，**没有改变或重新优化这套旧几何**。本次下载的旧 frequency 几何与 grid3 center 几何在 Bohr 数组上逐元素完全相同；旧优化 JSON 的 Å 坐标换算后最大差仅 4.44e-16 Å。

## 4. 本轮 90,000 CPU 秒优先补什么

优先级以新增可验证证据、避免重复昂贵失败为依据，具体启动参数和门禁以本轮独立冻结配置为准：

1. 新水分子控制：检查当前软件、严格 SCF、含网格响应梯度及双步长能量差分一致性；矩阵组装与平转投影另有实现测试。控制失败则阻止药材计算。
2. 从认证的原儿茶酸历史模型几何开始，在单独输出目录执行 PBE/def2-SVP/direct、明确无 D3、grid3 的新优化和 fresh 核梯度。新方法与旧 grid1 方法分开命名，保留旧结果。
3. 仅在新核驻点门禁通过后，逐坐标执行完整中央差分核 Hessian，预设两个步长 0.001 与 0.0005 Bohr。17 原子共有 51 个核坐标，每步长需要 102 个实际位移点；每点独立 SCF/梯度、来源和状态落盘。
4. 独立组装质量加权矩阵、平转投影、45 个有符号模式，核对矩阵反对称误差、两步长差和最低点门禁；不通过仍保存完整负结果。可复用的只是本轮已认证的中心和成功位移点，不重复执行已完成点。

B 旧同类位移点约 210 CPU 秒/点，因此两步长粗估约 44,000 CPU 秒，尚需优化、失败、测试和监督余量；这只是资源规划，不是运行时保证或通过承诺。Calycosin 完整数值双步长和 Spinosin 完整数值 Hessian 成本更大，暂不作为本轮必达目标；本轮不直接重复旧低精度 Opt/Freq，也不无限重试已闭合的 Spinosin SCF。

总额按本轮所有受监督 worker 的实际 user+system CPU 计数，线程 CPU 同样计入，不能把 90,000 当每个任务预算或墙钟时长。任务有阶段上限、日志、检查点、唯一身份和总额停止；历史账本、失败开销不抹除。独立后台进程可脱离本次聊天继续，但仍依赖云实例存活，不能承诺实例销毁后自动计算。

## 5. 本轮认证输入位置与真实性边界

认证输入位于 [`data/research/pyscf_autonomous_90000_20261108/inputs/`](../data/research/pyscf_autonomous_90000_20261108/inputs/)。主要文件：

- `protocatechuic_acid.xyz`：从旧优化 JSON 原子顺序与全精度 Å 坐标生成，无新构象搜索或量子计算；标为历史 DFT 模型几何，**不是实验晶体结构**。
- `historical_optimized_geometry.json` 与 `historical_optimization_result.json`：按 B 固定提交逐字节保存，优化结果中引用的结构 SHA-256 与下载结构一致。
- `historical_cases.json`、`historical_graph_validation.json`、`historical_source_evidence.json`：保留原可信连接关系、ChEBI 来源及历史图核验记录；原儿茶酸 C7H6O4、17 原子、80 电子、电荷 0、单重态、历史图记录 0 自由基电子。
- `historical_frequency_geometry_bohr.npy`、`historical_grid3_center_geometry_bohr.npy` 和 `historical_grid3_center_result.json`：证明 grid3 center 的核坐标未变。
- `chemical_case.json` 与 `source_receipt.json`：本轮提取的 case 和下载/派生收据；每项保存固定分支、提交、路径、Git blob SHA-1、实际下载内容 SHA-256、大小、单位及派生 XYZ SHA-256。

旧优化结构 SHA-256 为 `5899917276101687df3484c7610499c11d9ff8881f041210fc2571c1e5a76f0d`（1796 字节）。XYZ SHA-256 为 `5bc771e84703a6ce8ebc28f042c3bab45b903cc6ebe60f939cf96cb7ba3e975f`；解析 XYZ 后坐标与源 JSON 的 binary64 数值逐元素完全相同。

本次输入整理只验证来源一致性、元素/电荷、有限坐标、单位和历史图记录，不重新宣称实验结构鉴定、全局最低构象或药材成分实测。大检查点仅有远端哈希时不能视为文件可恢复；新计算从本次确实保存的模型坐标重新开始。
