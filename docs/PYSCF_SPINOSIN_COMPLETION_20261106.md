# Spinosin 完整振动计算、独立复核及残差抽查报告

本轮已经获得并保存 **219 个真实有符号正常模式，完整 Hessian 组装及独立数值复核均已完成**。但是，其中存在 **25 个大虚频，约 −312.34 至 −87.69 cm⁻¹**，因此当前构象的正频最低点资格和化学机制解释资格均未通过。不能把本轮结果称为“已确认稳定分子的理论吸收谱”，也不能把计算完成等同于物理化学验证成功。

本报告对应研究分支 `research/pyscf-spinosin-mechanism-20261031`。所有人类可读时间均为**北京时间（UTC+8）**；原始 JSON 保留 UTC 时间。完整组装于 **2026-11-06 14:20:44** 完成，独立模式复核于 **14:20:51** 完成，最后一项实际残差抽查于 **14:40:34** 在资源账本中闭合。本报告不代表此时间之后的新计算或实时进程状态。

## 1. 状态与结论边界

| 项目 | 状态 | 实际证据与限制 |
|---|---|---|
| Spinosin 结构、分子式、立体化学与来源核验 | DONE | C₂₈H₃₂O₁₅，75 原子；可信结构图生成的计算构象，非实验晶体结构 |
| 原 C001 构象 direct PBE 几何优化及后续 fresh SCF/梯度 | DONE | 真实优化收敛；相同方法的新梯度满足既定一阶阈值；不等于最低点证明 |
| 完整 RKS 二阶导数与 219 模式 | DONE | 31 个已闭合 XC 单元、完整 A、H1、10 个完整 AO/MO 响应块及全局组装 |
| 正常模式独立数值复核 | DONE | 从保存的完整 Hessian 独立执行平移/转动投影、特征值和模子空间复核；不是全新 DFT/Hessian 重算 |
| 水/乙醇新实现控制 | DONE | 全部 3/21 模式为正，与认证的原生完整参考矩阵和模式对照通过 |
| 四项 fresh CPHF 方程残差测量 | DONE | 水、乙醇各 3 RHS；Spinosin 原子 0 和 O36 各 3 RHS；仅测量，未设置新的 PASS/FAIL 阈值 |
| Spinosin 全 225 RHS 的 fresh 残差复核 | PARTIAL | 目前只抽查 6/225 RHS；完整响应计算完成不等于全部 RHS 已独立测残差 |
| 当前构象的正频局部最低点及机制资格 | BLOCKED | 25 个大虚频；`stationary_minimum_verified=false`、`qualified_for_mechanism=false` |
| 轨道稳定性检验 | NOT_RUN | MF 不变性标志不是轨道稳定性检验 |
| IR 强度 | NOT_RUN | 所有相关产物保持 `IR_intensities_status=NOT_COMPUTED` |
| v8 结构基团参与分析与化学峰归属 | NOT_RUN / BLOCKED | v8 基团适配器未运行；无合格最低点或标准品归属证据 |
| 标准品 THz、HPLC/LC-MS、真正独立新批次 | NOT_RUN | 尚无本轮实验室实测证据 |
| 使用理论结果调整分类器或产生新准确率 | NOT_RUN | 本轮未训练分类器、未修改光谱或标签、未把理论频率送入模型 |
| 本轮 Git 备份 | DONE | 实际科学提交 `39866b269d0c6fe17907e3132dab3578ccfe05c7` 已推送研究分支，远端分支指针及关键已提交文件哈希核验通过；见[备份回执](../reports/pyscf-spinosin-completion-20261106-v1/backup_receipt.json) |

`DONE` 表示表中限定范围的实际工作完成；`PARTIAL` 表示只完成其中一部分；`NOT_RUN` 表示没有执行；`BLOCKED` 表示当前证据或条件不足。流程终态“健康闭合”只表示受限工程流程实际完成，不会覆盖最低点和化学机制资格失败。

## 2. 真实计算方法与冻结实现

| 设置 | 本轮实际值 |
|---|---|
| 分子 | Spinosin，PubChem CID 155692，ChEBI 81360 |
| 分子式、原子数、电子数 | C₂₈H₃₂O₁₅；75；320 |
| 电荷与自旋 | 0；闭壳层单重态，`spin_2S=0` |
| 泛函 / 基组 | PBE / def2-SVP |
| 数值积分网格 | PySCF grid level 1 |
| 色散 | 明确未使用 D3；`NONE_EXPLICIT_PBE_PILOT_NOT_D3` |
| 积分路线 | direct RKS；本轮导数未使用 density fitting |
| AO / MO / 占据空间 | 完整 762 AO、完整方阵 MO、160 占据轨道；没有截断 AO/MO |
| SCF 收敛设置 | `conv_tol=1e-8`，`max_cycle=80`；direct SCF 阈值 `1e-13` |
| 响应设置 | `conv_tol_cpscf=1e-8`、`max_cycle=50`、`level_shift=0` |
| 分块 | B8；前 9 块各 8 原子，最后 1 块 3 原子，总计 75 原子 / 225 核位移 RHS |
| 梯度积分网格响应 | `gradient_grid_response=false` |
| 本轮工作线程 | 2；全局组装并不意味着所有时段均使用两条活跃 CPU 线程 |
| 能量 | −2210.445521964438 Eh，为恢复的相同 SCF 状态能量 |
| 软件 | Python 3.12.14、PySCF 2.14.0、NumPy 2.3.5、SciPy 1.17.0、geomeTRIC 1.1.1 |

冻结实现及其实际 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `scripts/run_pyscf_continuation_budget_20261105_v8.py` | `7bc773076ecc5e0f90fb58b81bf0641948dd09c14b2816138a0469f16e511c65` |
| `scripts/pyscf_staged_consumer_20261105_v8.py` | `3a107671d9c8a5a47c25f631fd09ed70148dc82b529448d9bafc67179801a585` |
| `scripts/analyze_pyscf_staged_modes_20261105_v8.py` | `da2a53d85015e0ced6445e7766e18ae4fbe58e098e0eb077719759be7b18cab6` |
| `scripts/audit_pyscf_saved_cphf_residual_20261105.py` | `6ff1df9930257f33637a34cfc58ba01b0fe38c246458ab704891663efbef800c` |

v8 显式核验旧 A/H1 生产者及其旧监督证明，随后使用新 v8 响应/全局生产者；旧结果没有被改写成新 schema。完整物理绑定 SHA 为 `a370cc3972b87be8f6f5359600ef4d2a77d9f9a858c768a0420682dae5139134`。

## 3. 构象来源、实际三维性与优化证据

当前 C001 的连接关系和立体化学来自经核验的 ChEBI/PubChem 结构，InChIKey 为 `VGGSULWDCMWZPO-ODEMIOGVSA-N`。构象谱系是 **核验结构图 → ETKDGv3/MMFF94s 模型构象 → 已记录 DFT 几何 → direct DFT 再优化**。这不是下载的实验晶体坐标，也不是已证明的全局最低能构象。

真实 direct 优化结果保存在：

- `reports/pyscf-spinosin-continuation-20261031-v3/spinosin-native4-optimization-001/result.json`
- 结果 SHA：`299539676308f3bac4bc01d3a0c1554a9b636228d91f7acd05e9fa08c0b7adeb`
- geomeTRIC 真实执行并收敛，记录最后几何循环为 5；PBE/def2-SVP/grid1/direct，与最终导数路线一致。其祖先使用 DF-J，并不意味着最终 direct Hessian 混用了 DF-J 导数。
- 旧几何坐标哈希：`cc3c3283c83cac02686457110a65a48652e33cf564a366c8654fa75c1260b159`。

后续 v5 `prepare-state` 没有重新优化，明确记录 `new_optimizer_run=false`，而是对原优化几何执行 fresh SCF/梯度，并保存精确状态。其后恢复梯度与源梯度最大差约 `9.9476e-14 Eh/Bohr`；源原子梯度 RMS 为 `6.2640408604e-5 Eh/Bohr`，原子最大梯度为 `2.0390345451e-4 Eh/Bohr`，通过原既定阈值。这证明达到所设一阶收敛条件，不证明 Hessian 正定。

用户后来上传的 CID155692 SDF 具有不同原子顺序；来源记录明确表示它**没有替换原 C001 坐标或标签**。不能跨两种顺序直接复用 AO、H1 或 Hessian 索引。

实际几何诊断给出的中心化坐标奇异值为 `[35.1819767079, 21.8921930048, 14.9573590298] Å`，最佳平面 RMS 距离 `1.7271270524 Å`，最小/最大奇异值比 `0.42514`。因此可以排除“整体二维平面坐标误当三维构象”这一特定问题；这不证明构象是最低点。

## 4. 已完成的完整 Hessian 流程与资源恢复

本轮没有通过删除色散标签、改变物理方法、缩减 AO/MO 或丢弃未完成原子来缩短计算。完整流水线为：

1. 复用已核验的同一精确 MF 快照和 fresh 梯度门槛。
2. 完成真实 XC 积分单元：20 个轻对角单元和 11 个重二阶导单元，共 31 个健康闭合单元；另保留失败尝试。
3. 从真实 RHF 辅助项和完整 XC 单元形成完整 RKS A。
4. 形成完整 H1AO。
5. 新 v8 顺序完成 10 个原生完整 AO/MO B8 响应块。
6. 显式提供全部 H1AO、MO1、MOE1，执行全局电子 Hessian 收缩及一次核 Hessian 相加，保留跨原子耦合。
7. 保存完整 Hessian，得到 219 有符号模式，再独立进行正常模式数学复核。

全局 capture 确认：完整响应块均健康闭合；显式全 H1AO/MO1/MOE1 输入；跨原子收缩保留；核项仅加一次；全局阶段没有重新求解 CPHF。capture 的原始 `measured_CPHF_residual_norm=NOT_COMPUTED` 保持不变，之后的抽查作为新增独立产物保存。

按实际累计账本中的闭合成本统计：

| 已完成阶段或独立尝试 | 真实 CPU 秒 | 说明 |
|---|---:|---|
| 31 个健康 XC 单元 | 32,319.639863 | 不含失败 XC 尝试；属于沿用的旧阶段成本 |
| 原生 RHF 辅助部分 | 361.923704 | 完整 RKS A 的真实组成之一 |
| 完整 RKS A 组装 | 1.806781 | 输入已有真实 XC/RHF，不是从零再算全部积分 |
| 完整 H1AO | 13,111.570415 | 旧健康生产者，经证明和哈希核验复用 |
| 新 v8 水/乙醇 5 个量子控制阶段 | 26.399542 | 真实新 response/global；另保留未执行量子 payload 的启动失败记录，未单独计量的启动开销不宣称为零 |
| Spinosin 10 个响应块 | **109,491.757191** | 225 RHS 的实际完整响应成本 |
| 第一次 v8 全局组装尝试 | 1,680.901295 | 触发实际 CPU guard，失败成本保留 |
| 独立 recovery-002 全局组装 | 4,780.385060 | 新 attempt，复用已核验响应；未覆盖失败目录 |
| 水/乙醇及 Spinosin 两原子的残差抽查 | 729.981699 | 四项新受监督测量 |
| 独立正常模式分析 | 6.203332 | 分析器自身 CPU；不是量子预算账本中的新 DFT 阶段 |

成本口径以监督器闭合账本的 `actual_cpu_seconds` 为准，包含该阶段启动等实际开销；因此可能略高于 `result.json` 中工作进程自计 CPU。表中不同范围的历史阶段和本轮阶段不能再次相加成一个新的全项目累计值。

汇总目录的 `resource_costs.csv` 只列新增的 22 个 v8 阶段；旧 133 个阶段仍完整保留在原账本，并通过 `legacy_closed_CPU_carried` 计入累计值。22 行不是全部历史计算次数。

历史失败包括：v6b A 尝试真实消耗 `24,863.36 CPU 秒`，观察到 SIGKILL，但原监督器未以 wait4 完整收尾，具体终止来源仍未确定；该失败成本经取证保留。v7 首个响应尝试触发实际 CPU guard，消耗 `7,080.874229 CPU 秒`。它们都没有被健康结果覆盖，也没有被免费重置。

第一次 v8 全局组装于北京时间 **2026-11-06 12:53:26** 因阶段 CPU 上限触发监督停止，随后完成闭合取证、保持原科学源及累计池上限，以新 `global-assemble-recovery-002` 配置启动全局阶段；第二次于 **14:20:44** 成功。新 attempt 没有重算已健康完成的 10 个响应块。

当前账本：`reports/pyscf-spinosin-96h-continuation-20261105-v8/cpu_budget.json`。它从旧账本真实已花费 `210,160.830836 CPU 秒` 开始，新增固定 `180,000 CPU 秒`，总上限 `390,160.830836 CPU 秒`；旧未用余额没有重复再分配。本文读取时累计 `326,870.255623 CPU 秒`，剩余 **63,290.575213 CPU 秒**。当前新授权阶段实际消耗 `116,709.424787 CPU 秒`。新期限是北京时间 **2026-11-09 20:30:00**；不会自动扩额。

内存协议为 24 GiB 地址空间上限、10 GiB RSS 尽力监督阈值和 12 GiB cgroup 监控停止阈值，并非“24 GiB 可用内存”或硬 RSS 保证。成功全局工作进程记录峰值 RSS `1,481,208 KiB`。详细阶段成本和失败记录继续保留在原账本及日志中。

## 5. 正常模式、独立数学复核及资格失败

全局 `result.json` 的状态是 `FREQUENCIES_COMPUTED`，预期模式数与实际模式数均为 `3×75−6=219`。其中 25 个负频、194 个正频、0 个零频。负号是虚频的有符号表示，不是实测的负吸收频率。

| 结果 | 值 |
|---|---:|
| 最负模式 1 | −312.3398803141 cm⁻¹ |
| 最后一个负模式 25 | −87.6904236998 cm⁻¹ |
| 0.1–1.5 THz 内正模式数 | 1 |
| 模式 26 | 37.0273696239 cm⁻¹ = 1.1100526153 THz |
| 全频最低点验证 | false |
| 可用于机制峰匹配 | false |

独立分析读取真实 Hessian 及核验过的谱系，重新构造质量加权、外部平移/转动投影和特征分解，并比较全部模式、特征向量子空间：

| 数值复核项 | 实际结果 |
|---|---:|
| 原始 Hessian 最大不对称 | `5.4756728041e-9 Eh/Bohr²` |
| 独立模式最大频率差 | `6.1390892370e-12 cm⁻¹` |
| 质量加权正交误差 | `2.5535129566e-15` |
| 外部空间投影误差 | `8.8817841970e-16` |
| 投影特征方程残差 | `4.1633363423e-16 Eh/(amu Bohr²)` |
| 数值重放状态 | `PASS_INDEPENDENT_NUMPY_TR_REPLAY` |

这些结果排除了保存 Hessian 到正常模式转换中的若干数学错误，并不证明 Hessian 的每一个物理导数都正确；特征方程残差也不是 CPHF 残差。独立分析没有运行新的 SCF、梯度或完整 Hessian，明确记录 `new_quantum_evaluations=0`。

`source_MF_stability_status=VERIFIED_UNCHANGED` 的范围是**恢复的 MF 数据没有被修改**，不是电子轨道稳定性分析。当前数据没有证明 25 个负模的原因。真实高阶非最低点构象、电子稳定性、网格/导数处理或大体系响应等可能性仍需受控区分；不能选择其中一种作为既定结论。25 个大负模在通过外部模式投影后仍存在，不能用删除模式或取绝对值把它变成合格最低点。

模式 26 的 1.11005 THz 只是当前非最低点构象下计算得到的一个正模式。即使它靠近此前酸枣仁候选频段，也不能把它列为已确认的斯皮诺素实验吸收峰。当前分析没有生成合格理论—实验归属表。

## 6. 新增 fresh CPHF 方程残差抽查

四项结果均独立保存于 `reports/cphf-residual-audit-actual-20261106-v1/`，由真实 v8 guard 监督并以 EXIT0 闭合。它们的科学状态统一为 `SAVED_RESPONSE_RESIDUALS_MEASURED_NOT_CONVERGENCE_GATED`，不是新增的“CPHF PASS”。原 response/global 结果中的 `residual_norm=NOT_COMPUTED` 不被回写。

测量从保存的 AO 响应 `W` 重建 `U=CᵀSW`，用完整分子 AO/MO 空间及冻结同一算子重新评价一次 `V(U)`，核对虚空间方程

`R_ai = (ε_a−ε_i) U_ai + H_ai − ε_i Q_ai + fresh V_ai(U)`。

虚空间残差单位为 Eh/Bohr；能隙预条件残差和占据规范残差单位为 1/Bohr。原生 Krylov 的试探向量停止量不是这个实际方程残差，不能拿两者的数值直接作为同一阈值比较。本轮没有增设新的通过阈值。

| 实际测量 | RHS / 全分子 RHS | 虚空间最大绝对残差 (Eh/Bohr) | 预条件残差最大值 (1/Bohr) | 监督器真实 CPU 秒 |
|---|---:|---:|---:|---:|
| 水，原子 0 的 x/y/z | 3 / 9 | `1.6821274966e-6` | `1.4681202760e-6` | 1.611299 |
| 乙醇，原子 0 的 x/y/z | 3 / 27 | `3.3214735194e-9` | `7.1320898887e-9` | 1.947004 |
| Spinosin，原子 0 的 x/y/z | 3 / 225 | `2.0186131015e-10` | `7.8825819711e-10` | 345.635102 |
| Spinosin，O36 的 x/y/z | 3 / 225 | `3.4524380113e-10` | `1.1583559631e-9` | 380.788294 |

原子编号从 0 开始。O36 是当前原子顺序下的氧，选择理由是它在最负模式的位移范数参与较大；这种参与不是 IR 强度，也不是振动化学归属。Spinosin 两项合计只覆盖 **6/225 RHS（约 2.67%）**，不等于完整响应的独立残差全检。

四项均只重新评价保存解的同一全算子，没有重新求解 CPHF，也没有运行 SCF、梯度、优化或 Hessian。结果中同时记录 AO 系数重建、占据规范、归一化和能量一阶响应差。抽查支持被测方向的响应方程一致性；不能据此确认全部 Hessian 正确、解释全部 25 个负模或称为独立完整 DFT 重算。水的最大测量值也如实保留，未挑选较小方向代替最大值。

`residual_summary.csv` 的 `source_result_sha256` 指原始 response 结果的 SHA；新增残差测量结果自身的 SHA 分别保存在 `summary.json` 的 `residual_audits[].source` 中，二者不能互换。

北京时间 **14:49:20** 的独立小文件审核保存在 `reports/cphf-residual-audit-actual-20261106-v1/actual-auxiliary-audit-independent-review.json`；它核验四项实际闭合、源/输入/输出哈希和真实成本，结论为 `PASS_ACTUAL_MEASUREMENT_PROVENANCE_AND_SCOPE_NOT_A_CONVERGENCE_PASS`。这个审核通过仍不是对残差收敛阈值的通过判定。

## 7. 实际结果、图表、结构与可追溯入口

下述 `B96` 是 `reports/pyscf-spinosin-96h-continuation-20261105-v8/`；`G2` 是 `B96/global-assemble-recovery-002/`，仅为本节简化路径显示。

可直接打开[汇总 JSON](../reports/pyscf-spinosin-completion-20261106-v1/summary.json)、[资源成本表](../reports/pyscf-spinosin-completion-20261106-v1/resource_costs.csv)、[残差表](../reports/pyscf-spinosin-completion-20261106-v1/residual_summary.csv)和[几何诊断](../reports/pyscf-spinosin-completion-20261106-v1/geometry_diagnostic.json)。全部 219 模式见[有符号频率 CSV](../reports/pyscf-spinosin-96h-continuation-20261105-v8/global-assemble-recovery-002/independent-modes/all_frequencies_cm1_THz.csv)，而非低频子集表。

| 产物 | 实际路径（相对仓库根目录） |
|---|---|
| 完整全局结果 | `G2/science/global-assemble/result.json` |
| 全局调用及完整组装 capture | `G2/science/global-assemble/global_assembly_capture.json` |
| 原生产者频率 CSV | `G2/science/global-assemble/frequencies.csv` |
| 计算几何 XYZ | `G2/science/global-assemble/normal_mode_geometry.xyz` |
| 独立复核 JSON | `G2/independent-modes/checks.json` |
| 全部 219 个有符号模式表 | `G2/independent-modes/all_frequencies_cm1_THz.csv` |
| 0.1–1.5 THz 正模式表 | `G2/independent-modes/normal_modes_0p1_to_1p5_THz.csv` |
| 全部有符号频率图 | `G2/independent-modes/signed_mode_frequencies.png` 及 `.pdf` |
| 低频位置图 | `G2/independent-modes/low_frequency_positions_not_IR.png` 及 `.pdf` |
| 模式动画 XYZ | `G2/independent-modes/animations/Spinosin_mode_001.xyz`、`002`、`003`、`026` |
| 文件哈希索引 | `G2/independent-modes/artifact_index.json` |
| 完整 10 响应结果 | `B96/response-pipeline-001/science/response-00/` 至 `response-09/` |
| 相同精确 MF 快照清单 | `reports/pyscf-spinosin-resumable-20261104-v5/v5-spinosin-C001-prepare-001/exact-mf-snapshot/manifest.json` |
| 同源恢复梯度 | `reports/pyscf-spinosin-resumable-20261104-v5/v5-spinosin-C001-restore-gradient-001/result.json` |
| 四项新残差结果 | `reports/cphf-residual-audit-actual-20261106-v1/{water-atom000-001,ethanol-atom000-001,spinosin-atom000-001,spinosin-atom036-001}/result.json` |
| 汇总与分项成本、残差表、三维诊断 | `reports/pyscf-spinosin-completion-20261106-v1/` 中 `summary.json`、`resource_costs.csv`、`residual_summary.csv`、`geometry_diagnostic.json` |

主要结果 SHA：

- 完整全局结果：`253c10a92c083f54aaeb3f1e407e095790649737f5f876bd3cecfd35bfb4f9b7`。
- 独立 checks：`b5edbd486f8e248d83efa556859f01cde080c849fddabd047649f7f5551601d0`。
- 全 219 模式 CSV：`6a4d41a99a919899a632355615b2d3df78e1169a97ebadc9d008da4b5b65419b`。
- 全局 capture：`c7065565e971562f47a7b8816eefbbad9a542ef2e20f6d5cd6d539ee159c2c5d`。
- 精确 MF 快照清单：`3089d6014ba2d2283540eb13cc7dbcdbbf284721042e7250b624197fc15c3311`。

PNG/PDF 的等高线段仅表示频率位置，**不表示吸收强度**。动画的位移振幅是人为设置的显示幅度，不是分子动力学轨迹；负模动画保留用于诊断。正常模式向量保存在实际结果链中，不能把任意向量符号、显示振幅或简并子空间旋转解释成新的物理结论。

## 8. 与酸枣仁主论文及实验峰的关系

本论文主线仍是酸枣仁 THz 来源分类及模型优化。历史 **MC-UVE＋SVM 96.0234%、Macro-F1 约 0.9197** 保持原有内部交叉验证语义；它既不是本轮 DFT 结果，也不是新训练或独立新批次结果。本轮没有产生任何新的分类准确率，不改原光谱、样本标签、数据划分、模型权重或 main。

当前机制部分只能报告“完成了一套可追溯的孤立分子初步计算，并发现当前构象未通过最低点验证”。它不能作为模型准确率提高的证据。孤立分子模式本身也不包含真实固体药材的晶格、复杂分子间氢键和集体振动；理论与实验频率接近不足以确认成分。标准品 THz、化学含量测量和真正独立新批次仍须由实际实验补足。

v8 `group_participation_adapter` 为 `NOT_RUN_NOT_IMPLEMENTED_FOR_V8`；旧 v5 基团分析准备文件不代表已分析此 v8 全频结果。IR 强度、谱峰归属、理论引导特征、标准品、HPLC/LC-MS 和新批次验证均未被虚构为完成。

## 9. 下一步及资源门槛

1. 本轮真实结果、完整有符号模式、失败成本及独立复核证据已备份到研究分支。root 在无量子活动时完成实际推送，并于北京时间 **2026-11-06 15:05:14** 核对远端研究分支指向上述科学提交，main 仍为 `709dc9a90f4f7ba3ae104503e612e5228de5f20d`。回执记录本地已提交文件与科学封存哈希的比较；未另行克隆远端，不把分支指针核验冒充独立内容下载核验。巨型 H1/MO1/MOE1 缓存仍仅本地保存，来源清单与哈希已提交。
2. 为负模原因建立新的独立诊断协议。可以分别考虑轨道稳定性、同方法特定位移的能量/梯度曲率核对或受控网格检查；任何未执行检查都继续标记 NOT_RUN，不能从当前证据直接认定原因。
3. 如果需要重新优化，必须作为新几何、新构象/状态谱系保存。几何变化后，旧 A、H1、MO1、XC 单元和 Hessian 不再是新几何的导数，**不能直接复用**。
4. 本轮完整 10 响应已真实消耗 **109,491.757191 CPU 秒**，大于当前剩余 **63,290.575213 CPU 秒**；尚未计入新几何的 SCF、梯度、XC/H1 和全局成本。因此不能承诺在原剩余池内完成一轮新优化后的完整频率，也不自动扩大 CPU 预算。新轮必须先制定独立协议、准确评估资源，再明确哪些阶段可执行。
5. 继续保留酸枣仁分类为研究主体。未合格的理论峰不进入当前分类器；待有合格最低点、必要控制及实验标准品证据后，才评估机制描述和预先固定的理论辅助对照。

本报告的终点是“真实计算及复核完成，最低点/机制资格未通过且原因未定”，并不是“已完成稳定分子机制解释”或“所有论文实验完成”。
