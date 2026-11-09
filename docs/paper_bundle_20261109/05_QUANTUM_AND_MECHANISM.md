# 05　量子化学实验、数值验证与机制结论

证据截止：2026-11-09T01:16:02.297194+00:00（UTC；北京时间为UTC+8）。整理时只读源结果，没有中断或重启两个后台任务。

## 摘要

完整分子频率计算已经真实执行，但“计算完”“优化收敛”“数值局部最低点”“严格残差认证”“实验化学峰归属”是不同结论。Spinosin已得到219个有符号模式，其中25个大虚频，最低点与机制资格仍未通过；原儿茶酸新grid3完整双步长数值Hessian各得到45个正模式，但严格认证仍需后续修复。毛蕊异黄酮的新完整数值Hessian和更严格原儿茶酸分支正在运行，均不写成已完成。

全部计算为PySCF；没有Gaussian引擎实算结果。IR强度均未计算，没有新增标准品THz或HPLC/LC-MS实测，也没有把理论峰加入分类器。本模块不把历史分类准确率作为量子计算的成果。

## 首先区分方法和软件版本

| 系列 | 物理/数值设置 | 结果应怎样命名 |
| --- | --- | --- |
| 早期pilot | PySCF2.14.0，PBE/def2-SVP/grid1，DF-J与direct软件控制，gradient_grid_response=False，无D3 | 初步优化可行性；Spinosin当时未完成频率 |
| A最终v8 / B初始完整频率 | PySCF2.14.0，完整direct AO/MO，PBE/def2-SVP/grid1、response=False、无D3 | 完整有符号模式；虚频/失败如实保留 |
| B后续方向/子空间 | grid3/grid5，固定/重建网格，response=True/False及更严格SCF | 有限方向或旧负模子空间诊断，不是完整新Hessian |
| 本次v3完整数值Hessian | PySCF2.13.0，PBE/def2-SVP/grid3，full-native VEFF，response=True；h=.001/.0005 Bohr | 新优化几何上的完整中央差分矩阵；与旧路线差异不能单因归咎 |
| 当前最低点修复 | 独立新源；SCF1e-10、轨道目标1e-10、direct筛选1e-15；几何grms3e-6/gmax1e-5；再计算新几何双步长矩阵 | 进行中；旧几何Hessian不能代替新几何Hessian |

## 已完成、失败和进行中总表

| 编号 | 实验 | 已保存状态 | 论文用途 |
| --- | --- | --- | --- |
| QC-001 | [Gaussian可行性、候选成分登记与PySCF技术smoke](#qc-001) | CANDIDATE_DFT_NOT_RUN_SOFTWARE_SMOKE_COMPLETED | 方法准备/补充材料；不能写为Gaussian DFT结果。 |
| QC-002 | [Spinosin初始pilot：12构象与4个真实几何评估](#qc-002) | PARTIAL_OPTIMIZATION_NO_SPINOSIN_FREQUENCIES | 结构来源、可恢复计算方法和失败透明性；非最终振动结果。 |
| QC-003 | [Spinosin direct几何续算与C001优化](#qc-003) | GEOMETRY_OPTIMIZATION_COMPLETED_FREQUENCIES_NOT_RUN_AT_THIS_STAGE | 方法谱系及初始构象说明。 |
| QC-004 | [v4/v5精确状态与跨进程Hessian恢复验证](#qc-004) | SOFTWARE_CONTROL_COMPLETED_SPINOSIN_INTERMEDIATES_PARTIAL | 可复现实现/检查点设计的补充材料。 |
| QC-005 | [v6/v7可恢复XC分块与完整导数控制](#qc-005) | ENGINEERING_RECOVERY_WITH_HISTORICAL_FAILURES | 工程方法附录；主文通常只需简述最终计算路线。 |
| QC-006 | [Spinosin v8完整Hessian、219有符号模式与独立重放](#qc-006) | COMPLETED_WITH_25_IMAGINARY_MODES_NOT_A_VERIFIED_MINIMUM | 可报告为计算诊断/负结果；不能作为已经确认的理论吸收谱或成分峰归属。 |
| QC-007 | [A003/A004：全部响应残差、方向有限差分及轨道稳定性尝试](#qc-007) | 225_RHS_MEASURED_DIRECTIONAL_DISAGREEMENT_ORBITAL_TEST_PARTIAL | 数值可靠性/限制与诊断附录。 |
| QC-008 | [固定网格/完整native势能的严格收敛诊断](#qc-008) | WATER_CONTROL_COMPLETED_LARGE_DIRECTIONAL_CASES_FAILED | 支持方法可靠性讨论，不能填补最低点或化学机制资格。 |
| QC-009 | [A的72h首批及recovery-002重建网格零点](#qc-009) | MAIN_QUEUE_CLOSED_WITH_FAILURES_RECOVERY_GRID1_FAILED_GRID3_NOT_EVIDENCED | 完整保留正/负结果，解释何以改变后续执行策略。 |
| QC-010 | [Worker B 原儿茶酸初始完整Opt/Freq/模式独立重放](#qc-010) | COMPLETED_WITH_IMAGINARY_MODES | 计算诊断及负结果，不作为合格理论峰谱。 |
| QC-011 | [Worker B 毛蕊异黄酮初始完整Opt/Freq/模式独立重放](#qc-011) | COMPLETED_WITH_IMAGINARY_MODES | 计算诊断及负结果，不作为合格理论峰谱。 |
| QC-012 | [Worker B网格/梯度响应及完整旧虚频子空间诊断](#qc-012) | DIRECTIONAL_AND_SUBSPACE_DIAGNOSTICS_COMPLETED_NO_FULL_NEW_HESSIAN | 可作为补充方法诊断，须说明新旧grid/SCF/梯度设置不相同。 |
| QC-013 | [本次v1/v2严格SCF认证队列的真实失败](#qc-013) | CLOSED_FAILED_OPTIMIZATION_HESSIANS_SKIPPED | 执行/验证限制与方法演化说明，成本及失败保留。 |
| QC-014 | [v3原儿茶酸grid3含网格响应的完整双步长数值Hessian](#qc-014) | FULL_NUMERICAL_HESSIANS_AND_MODES_COMPLETED_STRICT_CERTIFICATE_FALSE | 可报告孤立模型的数值曲率与方法验证，严格资格须按下一项分层解释。 |
| QC-015 | [原儿茶酸既有结果的最低点分层独立审计](#qc-015) | POSTRUN_AUDIT_NATIVE_PROTOCOL_EVIDENCE_TRUE_STRICT_FALSE | 用于解释验证级别及旧false与新证据的表面矛盾；主文措辞须注明精度和模型范围。 |
| QC-016 | [v3/Calycosin grid3新优化、完整数值Hessian断点及续算](#qc-016) | OPTIMIZATION_COMPLETED_FULL_HESSIAN_IN_PROGRESS | 当前待更新项；最终论文只引用冻结完成产物，或明确写进行中/未完成。 |
| QC-017 | [更严格原儿茶酸最低点修复后台分支](#qc-017) | RUNNING_AT_OBSERVATION | 论文准备清单中的进行中工作，不进入已完成结果表。 |
| QC-018 | [毛蕊异黄酮独立断点续算后台分支](#qc-018) | RUNNING_AT_OBSERVATION | 论文准备清单中的进行中工作，不进入已完成结果表。 |
| QC-019 | [标准品、成分定量、实验峰归属与独立批次验证缺口](#qc-019) | NOT_RUN_NO_CHEMICAL_ASSIGNMENT | 论文限制及后续验证方向；现阶段不能写成已完成化学机制验证。 |

## 最低点证书的三个层次不可混称

原儿茶酸既有两套矩阵没有负模式；原生优化与204点SCF也都返回收敛。然而PySCF2.13的extra循环允许放宽后的能差或轨道条件，native true不保证最后fresh轨道残差小于原配置目标。中心fresh残差1.6751455e-7，148/204点超过1e-7、最大2.6369975e-7；它们仍低于原生extra范围3e-7。中心核梯度最大分量3.6279491e-5亦略高于旧严格3e-5。

| 层次 | 当前既有证据 | 可以写什么 |
| --- | --- | --- |
| 原生算法接受范围的近似局部最低点证据 | true；原生优化/SCF、完整正曲率和双步长一致性支持 | 必须注明孤立模型、原生求解器及精度；不是精确驻点/全局最低构象 |
| fresh残差实际达到配置1e-7的独立证书 | false | 不得把native成功改述为全部fresh残差≤1e-7 |
| 原更严核驻点及1e-9残差证书 | false；独立修复计算进行中 | 不得只删除warning或修改布尔值使其通过 |

独立审计中`protocol_minimum_verified=true`的文字定义是第一层；修复程序用`native_protocol_minimum_verified=true`表达同层、用`configured_targets_minimum_verified=false`表达第二层。旧`minimum_verified=false`保持不变。引用时应写数值与语义，不能只比较字段名字。

## 分实验设计、结果与证据

<a id="qc-001"></a>

### QC-001　Gaussian可行性、候选成分登记与PySCF技术smoke

**对象与设计。** 水技术控制；Spinosin/Jujuboside A/B结构候选。引擎/许可与结构来源审计；水PBE/STO-3G/grid3单点、梯度、解析Hessian。候选Gaussian输入仅模板。

**已保存结果。** 证明当前开源导数路径可运行；没有取得候选分子理论吸收谱。Gaussian未实际执行。

| 指标 | 已核实值 |
| --- | --- |
| Gaussian_engine_calculations | 0 |
| water_atoms | 3 |
| water_energy_hartree | -75.22550440958054 |
| water_geometry_optimized | false |
| water_frequency_validation_performed | false |
| candidate_expected_modes | {"Spinosin":219,"Jujuboside_A":528,"Jujuboside_B":465} |

**论文用途。** 方法准备/补充材料；不能写为Gaussian DFT结果。

**边界。** 孤立分子模型，不是药材固态THz实测或新增独立样品；IR强度未计算；不据此确认实验峰归属、化学含量或分类性能机制；皂苷A身份/立体化学歧义及大分子资源问题保留；期望3N−6不是已算频率。

**来源。** [QUANTUM_CHEMISTRY_FEASIBILITY.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/QUANTUM_CHEMISTRY_FEASIBILITY.md)；[result.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/upgrade-20261031-v1/quantum/water-smoke-v2/result.json)

<a id="qc-002"></a>

### QC-002　Spinosin初始pilot：12构象与4个真实几何评估

**对象与设计。** Spinosin C28H32O15；75原子/320电子；水软件控制。可信ChEBI/PubChem立体图→ETKDGv3/MMFF94s12构象；选C001进行PBE/def2-SVP/grid1、DF-J、无D3优化。

**已保存结果。** 水Opt/Freq通过，Spinosin当时几何未收敛，按原资源限额停止；该NOT_RUN快照不能被后续完成情况回写。

| 指标 | 已核实值 |
| --- | --- |
| candidate_conformers | 12 |
| complete_spinosin_energy_gradient_pairs | 4 |
| actual_quantum_cpu_seconds | 2666.234148 |
| water_normal_modes | 3 |
| Spinosin_Hessian_computed | false |

**论文用途。** 结构来源、可恢复计算方法和失败透明性；非最终振动结果。

**边界。** 孤立分子模型，不是药材固态THz实测或新增独立样品；IR强度未计算；不据此确认实验峰归属、化学含量或分类性能机制；12个力场构象中的最低者不是DFT全局最低构象。

**来源。** [PYSCF_SPINOSIN_PILOT_20261031.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/PYSCF_SPINOSIN_PILOT_20261031.md)；[cpu_budget.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/pyscf-spinosin-pilot-20261031/cpu_budget.json)

<a id="qc-003"></a>

### QC-003　Spinosin direct几何续算与C001优化

**对象与设计。** 同一C001全精度原子顺序，非新SDF顺序。从已记录几何转入direct完整AO/MO PBE/def2-SVP/grid1优化；保留DF-J祖先与direct后代的区别。

**已保存结果。** 达到当时一阶优化要求；核几何优化收敛与Hessian正定是不同状态。

| 指标 | 已核实值 |
| --- | --- |
| optimizer_converged | true |
| last_geometry_cycle | 5 |
| atoms | 75 |
| basis_functions | 762 |
| energy_hartree | -2210.4455219704696 |
| component_gradient_rms_hartree_per_bohr | 3.609730214516183e-05 |
| atom_vector_gradient_max_hartree_per_bohr | 0.00020415338597535757 |
| frequency_status | "NOT_RUN" |

**论文用途。** 方法谱系及初始构象说明。

**边界。** 孤立分子模型，不是药材固态THz实测或新增独立样品；IR强度未计算；不据此确认实验峰归属、化学含量或分类性能机制；用户新SDF的原子顺序不同，不能按索引混用旧轨道或Hessian。

**来源。** [result.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/pyscf-spinosin-continuation-20261031-v3/spinosin-native4-optimization-001/result.json)；[PYSCF_CONTINUATION_STATUS_20261103.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/PYSCF_CONTINUATION_STATUS_20261103.md)

<a id="qc-004"></a>

### QC-004　v4/v5精确状态与跨进程Hessian恢复验证

**对象与设计。** 水/乙醇控制；Spinosin相同MF状态。精确保存MF、完整积分网格、分段RHF贡献与恢复梯度；对水/乙醇与stock全Hessian逐元素及全部模式比较。

**已保存结果。** 恢复算法小分子控制通过；21/150是RHF中间单元，不是21个模式或整个研究完成率。

| 指标 | 已核实值 |
| --- | --- |
| v4_closed_phases | 13 |
| v4_healthy_phases | 10 |
| v4_failed_phases | 3 |
| v4_actual_cpu_seconds | 3605.273693 |
| water_modes | 3 |
| ethanol_modes | 21 |
| water_max_Hessian_difference_Eh_per_bohr2 | 1.0880185641326534e-13 |
| ethanol_max_Hessian_difference_Eh_per_bohr2 | 8.881784197001252e-16 |
| spinosin_RHF_units_at_v5_snapshot | 21 |
| spinosin_RHF_total_units | 150 |

**论文用途。** 可复现实现/检查点设计的补充材料。

**边界。** 孤立分子模型，不是药材固态THz实测或新增独立样品；IR强度未计算；不据此确认实验峰归属、化学含量或分类性能机制；旧进程中断与退出边界异常成本保留，不能以缺结果当成本为0。

**来源。** [PYSCF_V5_RESUMPTION_20261104.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/PYSCF_V5_RESUMPTION_20261104.md)；[PYSCF_CONTINUATION_STATUS_20261103.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/PYSCF_CONTINUATION_STATUS_20261103.md)

<a id="qc-005"></a>

### QC-005　v6/v7可恢复XC分块与完整导数控制

**对象与设计。** Spinosin完整762 AO；水/乙醇分块控制。完整RKS A拆成一份RHF辅助项及diag/deriv2 XC块，完整H1、响应与核项保留；限定PBE/GGA适用范围。

**已保存结果。** 分块便于恢复和保存，不是新泛函或药材机制创新；必须以真实完整控制和最终组装结果判定，不能把代码测试当分子结果。

| 指标 | 已核实值 |
| --- | --- |
| v6b_failed_A_observed_cpu_seconds | 24863.36 |
| v7_failed_first_response_cpu_seconds | 7080.874229 |
| final_v8_healthy_XC_units | 31 |
| final_v8_response_blocks | 10 |

**论文用途。** 工程方法附录；主文通常只需简述最终计算路线。

**边界。** 孤立分子模型，不是药材固态THz实测或新增独立样品；IR强度未计算；不据此确认实验峰归属、化学含量或分类性能机制；求和顺序和版本更改需要数值控制；旧失败与旧producer不能冒充新成功。

**来源。** [PYSCF_V7_XC_CHUNK_RECOVERY_20261105.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/PYSCF_V7_XC_CHUNK_RECOVERY_20261105.md)；[PYSCF_V7_FULL_CONTROL_RESULTS_20261105.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/PYSCF_V7_FULL_CONTROL_RESULTS_20261105.md)；[PYSCF_SPINOSIN_COMPLETION_20261106.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/PYSCF_SPINOSIN_COMPLETION_20261106.md)

<a id="qc-006"></a>

### QC-006　Spinosin v8完整Hessian、219有符号模式与独立重放

**对象与设计。** C001；75原子/762 AO/160占据轨道。PBE/def2-SVP/grid1/direct、无D3、gradient_grid_response=False；完整A/H1/225 RHS响应及跨原子组装；独立平转投影/特征值重放。

**已保存结果。** 计算与保存完整，但当前构象/导数结果未通过正频最低点资格；25个大虚频原因未确定。

| 指标 | 已核实值 |
| --- | --- |
| signed_modes | 219 |
| positive_modes | 194 |
| imaginary_modes | 25 |
| most_negative_cm1 | -312.3398803141 |
| least_negative_cm1 | -87.6904236998 |
| energy_hartree | -2210.445521964438 |
| independent_replay_max_frequency_difference_cm1 | 6.139089236967266e-12 |
| stationary_minimum_verified | false |
| qualified_for_mechanism | false |
| positive_mode_26_THz | 1.1100526153 |

**论文用途。** 可报告为计算诊断/负结果；不能作为已经确认的理论吸收谱或成分峰归属。

**边界。** 孤立分子模型，不是药材固态THz实测或新增独立样品；IR强度未计算；不据此确认实验峰归属、化学含量或分类性能机制；模式26接近1.11THz不能用于确认Spinosin实验峰；独立模式重放排除的是矩阵到频率的若干数学错误，不能独立证明每个物理导数正确。

**来源。** [summary.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/pyscf-spinosin-completion-20261106-v1/summary.json)；[PYSCF_SPINOSIN_COMPLETION_20261106.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/PYSCF_SPINOSIN_COMPLETION_20261106.md)

<a id="qc-007"></a>

### QC-007　A003/A004：全部响应残差、方向有限差分及轨道稳定性尝试

**对象与设计。** 同一Spinosin；水/乙醇软件控制。保存响应全算子fresh残差；模式1 h=.01/.005与模式26 h=.01独立正负位移；轨道稳定性实际尝试。

**已保存结果。** 给定响应方程的225方向已测完；解析负曲率与梯度差分正曲率矛盾存在，不能据此单因归咎。轨道稳定性未通过。

| 指标 | 已核实值 |
| --- | --- |
| measured_unique_RHS | 225 |
| total_RHS | 225 |
| Spinosin_finite_displacement_points | 6 |
| mode1_analytic_curvature_Eh_per_bohr2 | -0.0375223030225 |
| mode1_h005_gradient_FD_curvature_Eh_per_bohr2 | 0.00489447762515 |
| orbital_stability_failed_cpu_seconds | 8881.818772 |

**论文用途。** 数值可靠性/限制与诊断附录。

**边界。** 孤立分子模型，不是药材固态THz实测或新增独立样品；IR强度未计算；不据此确认实验峰归属、化学含量或分类性能机制；旧完成报告早期6/225已被本后续证据补齐，但旧报告原字节不改；重建原子网格且grid_response=False的差分不是严格固定网格的同一导数问题。

**来源。** [result.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/cloud-followup-20261106-v1/queue-host-a-004/science/postrun-analysis-001/result.json)；[README.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/cloud-followup-20261106-v1/queue-host-a-004/science/postrun-analysis-001/README.md)

<a id="qc-008"></a>

### QC-008　固定网格/完整native势能的严格收敛诊断

**对象与设计。** 固定原积分网格水31点；Spinosin新零点和模式1/26。五个较小步长、31个fresh控制点；对比增量J与每循环完整VEFF；保留实际extra-cycle和独立Fock检查。

**已保存结果。** 小分子控制与Spinosin新零点通过不等于全部219模式重算；严格位移点未达标，旧25虚频仍未解决。

| 指标 | 已核实值 |
| --- | --- |
| water_prescribed_points | 31 |
| water_final_suite_all_points_completed | true |
| water_finest_pair_passed | true |
| water_same_spinosin_step_energy_passed | true |
| main72h_mode1_failed_final_orbital_norm | 2.5335e-09 |
| main72h_mode26_failed_final_orbital_norm | 3.1141e-09 |
| main72h_final_orbital_bound | 1e-09 |

**论文用途。** 支持方法可靠性讨论，不能填补最低点或化学机制资格。

**边界。** 孤立分子模型，不是药材固态THz实测或新增独立样品；IR强度未计算；不据此确认实验峰归属、化学含量或分类性能机制；增量势能漂移是受检验假设，不是全部虚频的既定原因。

**来源。** [FIXED_QUADRATURE_DIAGNOSIS_20261107.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/FIXED_QUADRATURE_DIAGNOSIS_20261107.md)；[result.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/fixed-quadrature-convergence-20261107-v2/queue-host-a-006/science/water-full-audit-convergence-attempt001/result.json)；[RESEARCH4_72H_BATCH_REVIEW_20261108.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/RESEARCH4_72H_BATCH_REVIEW_20261108.md)

<a id="qc-009"></a>

### QC-009　A的72h首批及recovery-002重建网格零点

**对象与设计。** Spinosin；水/乙醇控制。首批11个有限阶段包括ML独立任务；量子恢复先做10个小分子控制，再计划grid1/grid3 fresh零点与True/False梯度。

**已保存结果。** 首批有限队列结束不是科学问题解决；recovery控制成功、grid1电子零点失败，未进入梯度；固定A提交无grid3已完成证据。

| 指标 | 已核实值 |
| --- | --- |
| main_queue_success | 6 |
| main_queue_failed | 3 |
| main_queue_skipped_dependency | 2 |
| recovery_small_controls_points | 10 |
| recovery_controls_actual_cpu_seconds | 82.682657 |
| recovery_spino_grid1_actual_cpu_seconds | 5842.250239 |
| grid1_extra_energy_drifts_Eh | [3.638e-12,5.457e-12] |
| energy_bound_Eh | 1e-12 |
| grid3_actual_completed_job_present | false |

**论文用途。** 完整保留正/负结果，解释何以改变后续执行策略。

**边界。** 孤立分子模型，不是药材固态THz实测或新增独立样品；IR强度未计算；不据此确认实验峰归属、化学含量或分类性能机制；首批失败有真实闭合，不能归因于其后才发生的宿主机新boot。

**来源。** [RESEARCH4_72H_BATCH_REVIEW_20261108.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/RESEARCH4_72H_BATCH_REVIEW_20261108.md)；[state.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/research72h-20261107-v1/autonomous-main-001/state.json)；[state.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/research72h-20261107-v1/autonomous-recovery-002/state.json)

<a id="qc-010"></a>

### QC-010　Worker B 原儿茶酸初始完整Opt/Freq/模式独立重放

**对象与设计。** C7H6O4；17原子/80电子。PBE/def2-SVP/grid1/direct、无D3、gradient_grid_response=False；独立Opt、解析Freq、保存矩阵的模式数学重放。

**已保存结果。** 完整振动计算已做；虚频保留，最低点未通过。重放只核对数学变换，不是新DFT运行。

| 指标 | 已核实值 |
| --- | --- |
| normal_modes | 45 |
| positive_modes | 41 |
| imaginary_modes | 4 |
| optimizer_converged | true |
| stationary_minimum_verified | false |
| independent_mode_replay_max_difference_cm1 | 5.201172825763933e-12 |
| PySCF_version | "2.14.0" |

**论文用途。** 计算诊断及负结果，不作为合格理论峰谱。

**边界。** 孤立分子模型，不是药材固态THz实测或新增独立样品；IR强度未计算；不据此确认实验峰归属、化学含量或分类性能机制；ChEBI CC BY4.0

**来源。** [result.json](https://github.com/jackzhu119/research4/blob/71d16678d831b65206fd355d14b9a7f5462ca23b/reports/parallel-cloud-20261106/worker-b/B20261106-01/science/protocatechuic_acid-analyze-attempt001/result.json)；[result.json](https://github.com/jackzhu119/research4/blob/71d16678d831b65206fd355d14b9a7f5462ca23b/reports/parallel-cloud-20261106/worker-b/B20261106-01/science/protocatechuic_acid-frequency-attempt001/result.json)

<a id="qc-011"></a>

### QC-011　Worker B 毛蕊异黄酮初始完整Opt/Freq/模式独立重放

**对象与设计。** C16H12O5；33原子/148电子。PBE/def2-SVP/grid1/direct、无D3、gradient_grid_response=False；独立Opt、解析Freq、保存矩阵的模式数学重放。

**已保存结果。** 完整振动计算已做；虚频保留，最低点未通过。重放只核对数学变换，不是新DFT运行。

| 指标 | 已核实值 |
| --- | --- |
| normal_modes | 93 |
| positive_modes | 86 |
| imaginary_modes | 7 |
| optimizer_converged | true |
| stationary_minimum_verified | false |
| independent_mode_replay_max_difference_cm1 | 5.172751116333529e-12 |
| PySCF_version | "2.14.0" |

**论文用途。** 计算诊断及负结果，不作为合格理论峰谱。

**边界。** 孤立分子模型，不是药材固态THz实测或新增独立样品；IR强度未计算；不据此确认实验峰归属、化学含量或分类性能机制；ChEBI CC BY4.0; supplied SDF from user

**来源。** [result.json](https://github.com/jackzhu119/research4/blob/71d16678d831b65206fd355d14b9a7f5462ca23b/reports/parallel-cloud-20261106/worker-b/B20261106-01/science/calycosin-analyze-attempt001/result.json)；[result.json](https://github.com/jackzhu119/research4/blob/71d16678d831b65206fd355d14b9a7f5462ca23b/reports/parallel-cloud-20261106/worker-b/B20261106-01/science/calycosin-frequency-attempt001/result.json)

<a id="qc-012"></a>

### QC-012　Worker B网格/梯度响应及完整旧虚频子空间诊断

**对象与设计。** 旧原儿茶酸4个负模；旧Calycosin7个负模。grid3/grid5、固定/重建网格、True/False核网格响应；旧负子空间全部方向及耦合；组合最小方向再做双步长正负位移。

**已保存结果。** 所测旧负子空间在新条件下为正，支持导数/数值路线有待辨析；有限子空间不等于全分子最低点或完整频率验证。

| 指标 | 已核实值 |
| --- | --- |
| acid_old_negative_directions | 4 |
| caly_old_negative_directions | 7 |
| acid_true_branch_min_subspace_curvature_Eh_per_bohr2 | 0.002692605921458366 |
| caly_true_branch_min_subspace_curvature_Eh_per_bohr2 | 0.002721932369333398 |
| old_geometries_reoptimized_in_this_stage | false |
| full_new_Hessian_computed | false |

**论文用途。** 可作为补充方法诊断，须说明新旧grid/SCF/梯度设置不相同。

**边界。** 孤立分子模型，不是药材固态THz实测或新增独立样品；IR强度未计算；不据此确认实验峰归属、化学含量或分类性能机制；网格响应改变和几何/软件改变不能混称单一原因的因果证明。

**来源。** [result.json](https://github.com/jackzhu119/research4/blob/71d16678d831b65206fd355d14b9a7f5462ca23b/reports/parallel-cloud-20261106/worker-b/B20261106-01/followup-negative-subspace-v7/science/acid-g3-negative-subspace-attempt001/result.json)；[result.json](https://github.com/jackzhu119/research4/blob/71d16678d831b65206fd355d14b9a7f5462ca23b/reports/parallel-cloud-20261106/worker-b/B20261106-01/followup-negative-subspace-v7/science/caly-g3-negative-subspace-attempt001/result.json)；[analysis_conclusion_zh.txt](https://github.com/jackzhu119/research4/blob/71d16678d831b65206fd355d14b9a7f5462ca23b/reports/parallel-cloud-20261106/worker-b/B20261106-01/followup-negative-subspace-v7/science/all-negative-subspace-comparison-attempt001/analysis_conclusion_zh.txt)

<a id="qc-013"></a>

### QC-013　本次v1/v2严格SCF认证队列的真实失败

**对象与设计。** 原儿茶酸；水软件控制。v1普通SCF100周期；v2先Newton猜测再最多两次严格普通SCF，保留主循环/额外循环与fresh Fock；没有改写旧false。

**已保存结果。** v2第8点因实际额外循环能差未达标而失败；最终Fock残差小不能替代真实合格循环，也不能称几何已完成。

| 指标 | 已核实值 |
| --- | --- |
| v1_acid_failed_cpu_seconds | 280.987992 |
| v2_acid_failed_cpu_seconds | 5311.625816 |
| v2_successful_geometry_points_before_failure | 7 |
| v2_step007_extra_energy_changes_Eh | [-1.8189894035458565e-12,2.7284841053187847e-12] |
| strict_extra_energy_bound_Eh | 1e-12 |
| v2_step007_final_fresh_orbital_norm | 1.0264125192636708e-11 |
| new_Hessians_in_v1_v2 | 0 |

**论文用途。** 执行/验证限制与方法演化说明，成本及失败保留。

**边界。** 孤立分子模型，不是药材固态THz实测或新增独立样品；IR强度未计算；不据此确认实验峰归属、化学含量或分类性能机制；准备预留不是逐项精确实测CPU；账本cap、actual和live采样不可混用。

**来源。** [ledger.json](https://github.com/jackzhu119/research4/blob/225195e7f655d9571617f0d358c52e9ed8e13853/reports/pyscf-autonomous-90000-20261108/ledger.json)；[ledger.json](https://github.com/jackzhu119/research4/blob/225195e7f655d9571617f0d358c52e9ed8e13853/reports/pyscf-autonomous-90000-20261108/newton-recovery-v2/ledger.json)；[result.json](https://github.com/jackzhu119/research4/blob/225195e7f655d9571617f0d358c52e9ed8e13853/reports/pyscf-autonomous-90000-20261108/newton-recovery-v2/jobs/acid-optimize/optimization_steps/step-007/result.json)

<a id="qc-014"></a>

### QC-014　v3原儿茶酸grid3含网格响应的完整双步长数值Hessian

**对象与设计。** 同一新优化的原儿茶酸17原子模型。PySCF2.13.0；PBE/def2-SVP/grid3/direct、每轮完整native VEFF、grid_response=True；SCF1e-10/轨道目标1e-7；新几何每个步长全51坐标正负梯度独立起始，h=.001/.0005 Bohr。

**已保存结果。** 已完成完整双步长矩阵，全部45模式为正且两步长相近；原严格最低点标志仍false。

| 指标 | 已核实值 |
| --- | --- |
| finite_displacement_points_each | [102,102] |
| total_finite_displacement_points | 204 |
| native_unconverged_points_each | [0,0] |
| modes_each | 45 |
| imaginary_modes_each | [0,0] |
| frequency_min_max_cm1 | [[73.64628718814203,3723.9939644458873],[73.64642991600157,3723.9821583427597]] |
| relative_step_difference | 0.0001388876689707157 |
| raw_relative_asymmetry | [8.607577160853177e-05,0.00023494107572155024] |
| optimizer_converged | true |
| fresh_component_nuclear_rms | 1.1421842722830104e-05 |
| fresh_component_nuclear_max | 3.627949137569031e-05 |
| original_minimum_verified | false |

**论文用途。** 可报告孤立模型的数值曲率与方法验证，严格资格须按下一项分层解释。

**边界。** 孤立分子模型，不是药材固态THz实测或新增独立样品；IR强度未计算；不据此确认实验峰归属、化学含量或分类性能机制；与旧grid1/response=False解析Hessian相比，软件版本、几何、grid和导数路线均有差异；不能单独归因“修复PySCF错误”。

**来源。** [result.json](https://github.com/jackzhu119/research4/blob/225195e7f655d9571617f0d358c52e9ed8e13853/reports/pyscf-continuous-20261108-v3/jobs/acid-g3-optimize/result.json)；[result.json](https://github.com/jackzhu119/research4/blob/225195e7f655d9571617f0d358c52e9ed8e13853/reports/pyscf-continuous-20261108-v3/jobs/acid-g3-h001/result.json)；[result.json](https://github.com/jackzhu119/research4/blob/225195e7f655d9571617f0d358c52e9ed8e13853/reports/pyscf-continuous-20261108-v3/jobs/acid-g3-h0005/result.json)；[result.json](https://github.com/jackzhu119/research4/blob/225195e7f655d9571617f0d358c52e9ed8e13853/reports/pyscf-continuous-20261108-v3/jobs/acid-g3-analysis/result.json)

<a id="qc-015"></a>

### QC-015　原儿茶酸既有结果的最低点分层独立审计

**对象与设计。** 同一v3普通精度几何及204点，不是新量子计算。核对原生PySCF2.13主/extra循环语义，逐点来源/梯度/坐标，独立重组两完整Hessian并核平转投影及步长误差裕量。

**已保存结果。** 可在明确原生求解器接受语义下报告“近似局部最低点数值证据”；不能说全部fresh残差达到1e-7，更不能写严格驻点认证通过。

| 指标 | 已核实值 |
| --- | --- |
| quantum_calculations_performed | 0 |
| authentic_displacement_points | 204 |
| native_protocol_local_minimum_evidence | true |
| fresh_residual_configured_1e_7_certificate | false |
| strict_minimum_verified | false |
| center_fresh_orbital_gradient_norm | 1.675145517520958e-07 |
| point_count_above_configured_1e_7 | 148 |
| maximum_point_fresh_orbital_gradient_norm | 2.6369975074e-07 |
| native_extra_cycle_orbital_acceptance_bound | 3e-07 |
| original_minimum_verified_preserved | false |

**论文用途。** 用于解释验证级别及旧false与新证据的表面矛盾；主文措辞须注明精度和模型范围。

**边界。** 孤立分子模型，不是药材固态THz实测或新增独立样品；IR强度未计算；不据此确认实验峰归属、化学含量或分类性能机制；这是事后命名的证书层次，不是把事后标准冒充预注册；经验步长裕量不是精确物理Hessian的严格误差上界。

**来源。** [independent_acid_existing_audit.json](https://github.com/jackzhu119/research4/blob/225195e7f655d9571617f0d358c52e9ed8e13853/reports/pyscf-minimum-certification-20261109/independent_acid_existing_audit.json)；[result.json](https://github.com/jackzhu119/research4/blob/225195e7f655d9571617f0d358c52e9ed8e13853/reports/pyscf-resume-20261109-v4/acid-minimum/jobs/audit-existing/result.json)

<a id="qc-016"></a>

### QC-016　v3/Calycosin grid3新优化、完整数值Hessian断点及续算

**对象与设计。** 毛蕊异黄酮33原子/148电子，第一套完整矩阵预期198位移点。普通精度PBE/def2-SVP/grid3、完整native/direct、True核网格响应；先优化，h=.001矩阵逐点保存。旧主机重启后复用28点，新parent attempt继续，不重算缓存。

**已保存结果。** 新优化已完成；Hessian仍进行中，不能从部分矩阵报告93个新模式、无虚频或最低点通过。

| 指标 | 已核实值 |
| --- | --- |
| optimizer_converged | true |
| fresh_component_nuclear_rms | 9.608060765017846e-06 |
| fresh_component_nuclear_max | 3.399029478368121e-05 |
| first_Hessian_expected_points | 198 |
| first_Hessian_finite_points_at_cutoff | 28 |
| native_unconverged_points_at_cutoff | 0 |
| old_restart_checkpoint_points_reused | 28 |
| new_full_mode_count_computed | null |

**论文用途。** 当前待更新项；最终论文只引用冻结完成产物，或明确写进行中/未完成。

**边界。** 孤立分子模型，不是药材固态THz实测或新增独立样品；IR强度未计算；不据此确认实验峰归属、化学含量或分类性能机制；原B的93模/7虚频仍是旧grid1结果，不能用当前部分新矩阵替换。

**来源。** [result.json](https://github.com/jackzhu119/research4/blob/225195e7f655d9571617f0d358c52e9ed8e13853/reports/pyscf-continuous-20261108-v3/jobs/caly-g3-optimize/result.json)；[progress.json](../../evidence/paper_bundle_20261109/live_snapshots/quantum/30e5555c1d1234a4_progress.json)；[checkpoint_reuse_verification.json](https://github.com/jackzhu119/research4/blob/225195e7f655d9571617f0d358c52e9ed8e13853/reports/pyscf-resume-20261109-v4/checkpoint_reuse_verification.json)

<a id="qc-017"></a>

### QC-017　更严格原儿茶酸最低点修复后台分支

**对象与设计。** 原儿茶酸新更严几何及其后新Hessian。同一新续算窗口分成两独立allocation并行，保留旧账本/未知尾部。严格数值不达标只影响标签，不终止另一支；未完成不当作完成。

**已保存结果。** 本行仅是带时间戳的执行快照；后续结果应另冻结、核对并更新，不以RUNNING字样代替真实进程/科学完成。

| 指标 | 已核实值 |
| --- | --- |
| allocation_cpu_cap_seconds | 18000 |
| charged_at_observation_including_preparation_reservation | 2733.667363 |
| driver_live_verified | true |
| observed_job_states | {"audit-existing":"COMPLETED","refine-geometry":"RUNNING"} |
| old_final_CPU_cost_exact | null |
| current_run_is_new_prospective_allocation | true |
| strict_minimum_result_available | false |

**论文用途。** 论文准备清单中的进行中工作，不进入已完成结果表。

**边界。** 孤立分子模型，不是药材固态THz实测或新增独立样品；IR强度未计算；不据此确认实验峰归属、化学含量或分类性能机制；主机生存不是科学结论；结束聊天不会主动停止，云实例关机仍可中断；两支共28,000CPU秒为操作窗口，不表示旧90,000秒成本已精确结算。

**来源。** [ledger.json](../../evidence/paper_bundle_20261109/live_snapshots/quantum/4f0bd5e09c8f4af7_ledger.json)；[config.json](https://github.com/jackzhu119/research4/blob/225195e7f655d9571617f0d358c52e9ed8e13853/reports/pyscf-resume-20261109-v4/acid-minimum/config.json)；[continuation_authorization.json](https://github.com/jackzhu119/research4/blob/225195e7f655d9571617f0d358c52e9ed8e13853/reports/pyscf-resume-20261109-v4/continuation_authorization.json)

<a id="qc-018"></a>

### QC-018　毛蕊异黄酮独立断点续算后台分支

**对象与设计。** Calycosin第一套未完成198点矩阵及后续队列。同一新续算窗口分成两独立allocation并行，保留旧账本/未知尾部。严格数值不达标只影响标签，不终止另一支；未完成不当作完成。

**已保存结果。** 本行仅是带时间戳的执行快照；后续结果应另冻结、核对并更新，不以RUNNING字样代替真实进程/科学完成。

| 指标 | 已核实值 |
| --- | --- |
| allocation_cpu_cap_seconds | 10000 |
| charged_at_observation_including_preparation_reservation | 2974.804005 |
| driver_live_verified | true |
| observed_job_states | {"caly-g3-h001":"RUNNING"} |
| old_final_CPU_cost_exact | null |
| current_run_is_new_prospective_allocation | true |
| strict_minimum_result_available | false |

**论文用途。** 论文准备清单中的进行中工作，不进入已完成结果表。

**边界。** 孤立分子模型，不是药材固态THz实测或新增独立样品；IR强度未计算；不据此确认实验峰归属、化学含量或分类性能机制；主机生存不是科学结论；结束聊天不会主动停止，云实例关机仍可中断；两支共28,000CPU秒为操作窗口，不表示旧90,000秒成本已精确结算。

**来源。** [ledger.json](../../evidence/paper_bundle_20261109/live_snapshots/quantum/1ff15f1ddb404f69_ledger.json)；[config.json](https://github.com/jackzhu119/research4/blob/225195e7f655d9571617f0d358c52e9ed8e13853/reports/pyscf-resume-20261109-v4/caly-resume/config.json)；[continuation_authorization.json](https://github.com/jackzhu119/research4/blob/225195e7f655d9571617f0d358c52e9ed8e13853/reports/pyscf-resume-20261109-v4/continuation_authorization.json)

<a id="qc-019"></a>

### QC-019　标准品、成分定量、实验峰归属与独立批次验证缺口

**对象与设计。** 药材真实样品与孤立分子模型之间的外部证据。现有候选登记/文献、计算模型与实验需求分别保留；没有新增标准品THz、HPLC/LC-MS或真正新独立批次结果。

**已保存结果。** 当前量子证据不能证明产地分类由特定化合物/含量驱动，也不能取代固态实验吸收或定量检测。

| 指标 | 已核实值 |
| --- | --- |
| IR_intensities_computed | false |
| chemical_peak_assignment_verified | false |
| theoretical_modes_used_to_train_classifier | false |
| new_standard_THZ_HPLC_LCMS_results | false |

**论文用途。** 论文限制及后续验证方向；现阶段不能写成已完成化学机制验证。

**边界。** 孤立分子模型，不是药材固态THz实测或新增独立样品；IR强度未计算；不据此确认实验峰归属、化学含量或分类性能机制。

**来源。** [RESEARCH4_CHEMICAL_EVIDENCE_20261104.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/RESEARCH4_CHEMICAL_EVIDENCE_20261104.md)；[RESEARCH4_LAB_AND_BLIND_VALIDATION_20261105.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/RESEARCH4_LAB_AND_BLIND_VALIDATION_20261105.md)；[QUANTUM_CHEMISTRY_FEASIBILITY.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/QUANTUM_CHEMISTRY_FEASIBILITY.md)

## 写入论文前的结论口径

- 可以写：在给定孤立分子、泛函/基组、网格、构象和导数实现下，完成了哪些完整矩阵、多少个有符号模式及哪些有限差分/残差验证；正负结果和未完成状态分别报告。
- 不能写：Gaussian已计算；无D3的方法写成PBE-D3；虚频删除/取绝对值后称稳定；部分子空间正定替代全分子最低点；未算偶极导数却给IR强度；孤立分子模式直接确认药材固态峰或HPLC含量。
- 不能写：本轮DFT提升了分类准确率，或证明历史855条混合枣仁/酸枣仁记录的分类由特定成分引起；没有此类实验设计和验证证据。
- 两个当前后台任务应单列“进行中”。正式写结果时重新核对其完整产物、真实闭合和冻结提交，不把本文RUNNING快照当作未来完成保证。

## 来源与可追溯性

机器可读台账见[quantum.json](../../evidence/paper_bundle_20261109/quantum.json)。历史A、B结果使用固定Git提交和逐文件SHA-256。动态账本/进度使用[本次字节快照](../../evidence/paper_bundle_20261109/live_snapshots/quantum/)，保留原路径、观察时间和SHA；复制快照没有修改活跃源。源版本之间的旧NOT_RUN、失败和旧6/225表述不被回写，后续证据单独指出更新。
