# 多表示太赫兹神经网络：2026-11-04真实结果

**酸枣仁主任务没有出现足以替代传统强基线的神经网络增益。** 7组任务的α＋折射率n＋时域time融合M为82.4561%，低于相同参数三分支α控制C的82.6511%，Macro-F1与Balanced Accuracy也下降。黄芪M比C高8.5366个百分点，但仍低于同拆分LDA；天麻M比C高0.8032个百分点，与α单分支准确率相同，低于同拆分MLP。全部正、负结果保留。

本报告时点为NN及独立权重重放完成后、新Spinosin V6b大分子阶段启动前。所有指标来自实际预测与真值评分；当前query均为已曝光开发数据，不是全新独立确认。

## 数据、任务与可比范围

| 任务 | source | 未使用support池 | query | 类别 |
| --- | ---: | ---: | ---: | ---: |
| 酸枣仁原7组 | 513 | 171 | 171 | 7 |
| 黄芪 | 241 | 80 | 82 | 4 |
| 天麻 | 242 | 80 | 83 | 4 |
| 酸枣仁扩展10组 | 516 | 174 | 174 | 10 |

本次新增酸枣仁9条样品记录（ZY08—ZY10各3），黄芪／天麻没有新增独立样品。补充文件恢复HQ03的100条n表示、TM02／TM03的203条time表示，共303条既有样品表示；原855份酸枣仁四表示原本齐全。另2个受阻文件保留隔离。不能把表示数或重复预测算作新增独立样品，独立性仍依据用户声明，批次、仪器与单位核验仍不足。

[表示可用性审计](../reports/supplemental-intake-summary-20261104-v2-modality-availability/availability_comparison.json)与[固定角色表](../data/splits/supplemental_spectra_20261104_v1/all_tasks_split_manifest.csv)给出数据依据。各药材分开分类，不混合来源标签。

**96.0234%仍是原855份、5外折／2内折的MC-UVE＋SVM开发策略，Macro-F1为0.9197。** 本轮171／174／82／83固定query不能与该855 OOF成绩直接相减，不能拼接成连续提升阶梯。

## 实际执行与防泄漏

[v2 driver](../scripts/run_supplemental_multirepresentation_20261104_v2.py)、[leaf](../scripts/draft_supplemental_multirepresentation_20261104_v2.py)及[实际注册协议](../reports/supplemental-multirepresentation-20261104-v2/protocol.json)固定三臂：

- α单分支：128点α，compact ResNet。
- C：独立α128／α128／α1024三分支，长α直接来自原canonical612。
- M：独立α128／n128／time1024三分支，分别编码后融合；不同物理轴不当作同坐标通道。

C与M每分支长度、参数量、初始化、batch顺序和轮数相同，主要对比为M−C。C−单分支同时含容量与α分辨率变化，不能称纯容量因果贡献。α逐行SNV，n保留样品均值，time反混叠后L2；可学习标准化仅fit当前source。

先运行三任务×三source折×三seed＝27次α源内拟合，仅在24／48／96轮保存OOF，并按预定Macro-F1／BA／Accuracy规则选轮。三任务均选96，扩展10组继承96，不对singleton做CV。再完成4任务×3臂×3seed＝36次最终拟合；全部模型冻结后预测无标签query，全部预测冻结后才连接真值。42／43／44全部报告，没有query调参、再训练或挑最好种子。

[独立审计](../reports/supplemental-multirepresentation-independent-audit-20261104-v2/summary.json)为`PASS_INDEPENDENT_POST_Q_NUMERICAL_AUDIT`：27源模型、36最终模型元数据、36实际权重重放、54逐seed评分及18汇总全部核验；logit和派生softmax重放最大差均0，审计新增fit为0。另复核逐类、混淆、54组改对／伤害、source选轮和完整来源锁。数据访问门禁通过不等于排除批次混杂或达到独立确认。

## 全部主要结果

以下均为三个固定seed均值；Accuracy的±为样本SD，N为每seed实际query数。

| 任务 | 模型 | query N／seed | Accuracy ± 样本SD（%） | Macro-F1 | BA（%） |
| --- | --- | ---: | ---: | ---: | ---: |
| 酸枣仁7组 | α单分支 | 171 | 76.6082 ± 3.0944 | 0.672558 | 72.2789 |
| 酸枣仁7组 | C：三分支α控制 | 171 | 82.6511 ± 3.7139 | 0.699761 | 75.3061 |
| 酸枣仁7组 | M：α＋n＋time | 171 | 82.4561 ± 2.5491 | 0.668225 | 70.8844 |
| 黄芪4组 | α单分支 | 82 | 69.1057 ± 1.8628 | 0.689680 | 69.0530 |
| 黄芪4组 | C：三分支α控制 | 82 | 69.9187 ± 1.8628 | 0.695727 | 69.8106 |
| 黄芪4组 | M：α＋n＋time | 82 | 78.4553 ± 1.4082 | 0.784122 | 78.4470 |
| 天麻4组 | α单分支 | 83 | 94.3775 ± 3.0321 | 0.943664 | 94.5346 |
| 天麻4组 | C：三分支α控制 | 83 | 93.5743 ± 0.6956 | 0.936395 | 93.7951 |
| 天麻4组 | M：α＋n＋time | 83 | 94.3775 ± 0.6956 | 0.944100 | 94.5509 |
| 酸枣仁扩展10组 | α单分支 | 174 | 69.3487 ± 5.3400 | 0.517659 | 56.7143 |
| 酸枣仁扩展10组 | C：三分支α控制 | 174 | 79.8851 ± 4.1443 | 0.591545 | 65.2857 |
| 酸枣仁扩展10组 | M：α＋n＋time | 174 | 81.8008 ± 4.0774 | 0.591931 | 63.6667 |

[18行汇总](../reports/supplemental-multirepresentation-20261104-v2/score/model_comparison.csv)、[54行逐seed指标](../reports/supplemental-multirepresentation-20261104-v2/score/per_seed_metrics.csv)、[逐类P/R/F1](../reports/supplemental-multirepresentation-20261104-v2/score/per_class_metrics.csv)与[混淆矩阵计数](../reports/supplemental-multirepresentation-20261104-v2/score/confusion_matrix_counts.csv)完整保留。

| 任务 | M−C Accuracy（百分点） | M−C Macro-F1 | M−C BA（百分点） |
| --- | ---: | ---: | ---: |
| 酸枣仁7组 | -0.1949 | -0.031536 | -4.4218 |
| 黄芪4组 | +8.5366 | +0.088395 | +8.6364 |
| 天麻4组 | +0.8032 | +0.007705 | +0.7558 |
| 酸枣仁扩展10组 | +1.9157 | +0.000386 | -1.6190 |

M相对C平均改对／伤害：酸枣仁7组13.33／13.67，黄芪10.33／3.33，天麻2.00／1.33，扩展10组15.67／12.33。[逐seed配对整数](../reports/supplemental-multirepresentation-20261104-v2/score/paired_corrected_harmed.csv)可追溯；均值不是增加独立N或显著性证据。

酸枣仁7组M虽比α单分支Accuracy高，Macro-F1却从0.672558降至0.668225，BA从72.2789%降至70.8844%；不能只展示Accuracy声称全面优化。扩展10组M−C的Macro-F1仅+0.000386，BA反降1.6190个百分点。

## 同拆分传统基线及新类别边界

下表取[传统真实结果](../reports/supplemental-spectra-benchmark-20261104-v1/scoring/model_calibration_comparison.csv)的`method=none, shot=0, stratum=all`三seed均值。角色及query相同，但输入表示、内层候选与计算预算不同；这是同拆分完整pipeline对照，不是等计算或仅隔离分类器贡献。

| 任务 | 无校准pipeline | Accuracy（%） | Macro-F1 | BA（%） |
| --- | --- | ---: | ---: | ---: |
| 酸枣仁7组 | mcuve_svm | 93.1774 | 0.915255 | 91.6667 |
| 酸枣仁7组 | shrinkage_lda | 93.5673 | 0.920707 | 92.1429 |
| 黄芪4组 | shrinkage_lda | 84.1463 | 0.844067 | 84.0909 |
| 天麻4组 | mlp | 99.1968 | 0.991870 | 99.2063 |
| 酸枣仁扩展10组 | mcuve_svm | 94.8276 | 0.730505 | 76.5000 |
| 酸枣仁扩展10组 | shrinkage_lda | 92.5287 | 0.743086 | 74.5000 |

酸枣仁主证据继续保留MC-UVE＋SVM优化，同时报告固定7组协议下LDA的强对照，不能称MC在所有协议／指标均第一。黄芪M可写为本次神经对照内部的表示收益，尚未超过LDA；天麻M未超过MLP。

扩展10组的同一旧171子集：M82.4561%，C80.7018%，传统MC-UVE＋SVM95.9064%。NN内部旧171的Macro-F1从C的0.705905降至M的0.680228、BA从78.9796%降至71.9048%。不同输出类别集合的子集Macro-F1口径不可混比，与传统模型此处仅比较Accuracy。

新query仅ZY08／09／10各1份，共3份。M平均44.4444%，C33.3333%，α单分支44.4444%，MLP也为44.4444%；三seed不增加独立样品，不能证明新类别泛化、稳定少样本收益或真实跨批次校准。[新3预测](../reports/supplemental-multirepresentation-independent-audit-20261104-v2/new3_predictions.csv)和[旧171配对](../reports/supplemental-multirepresentation-independent-audit-20261104-v2/old171_A_vs_B_paired.csv)单列。

## 既有负迁移全部保留

此前跨植物α迁移使用同855队列的五外折／两内折与三seed，不是本轮固定query协议。[两预算六行表](../reports/cross-plant-conservative-pilot-20261104-v2/two_budget_model_comparison.csv)与[完整报告](CONSERVATIVE_CROSS_PLANT_BUDGET_ABLATION_20261104.md)记录：48轮上限时从头A80.3509%、外部黄芪／天麻预训练B76.1404%；96轮上限时A87.0565%、额外计算Aextra86.3548%、B85.3801%。第二轮45权重原生重放通过。

扩大source选轮预算提高了配方成绩，但B仍低于同预算A，不能归因于外部数据。不同协议不并成同一优化阶梯。公开新数值植物THz数据本轮新增下载0；SDF、文献成分均值和DFT频率未作为逐样品化学特征输入分类器。

## 真实失败、时间轴修复与成本

v1在0拟合时因原6750点time坐标非严格均匀而FAILED_CLOSED，[原失败账本](../reports/supplemental-multirepresentation-20261104-v1/resource_ledger.json)保留。仅轴审计发现最大坐标偏差约0.000494个采样间隔，与导出舍入相容；v2明确先在原端点范围内逐行线性插值到6750均匀点，再沿用polyphase512／3375、Kaiser5、line边界和L2。不删样品、改标签或把抖动轴直接当均匀轴。[时间轴独立审计](../reports/supplemental-multirepresentation-independent-audit-20261104-v2/time_axis_repair_audit.json)及[修复说明](SUPPLEMENTAL_MULTIREPRESENTATION_V2_TIME_REPAIR_20261104.md)保留原／均匀／输出轴SHA与source/query同合同。时间单位仍unknown，不作THz FFT解释。

v2实际child CPU430.602959秒，监督进程观测1.353656048秒，合计431.956615048秒；独立审计含启动的最终观测3.988732564秒。加既有失败／诊断及明确非实测的10秒保守reserve，总envelope charge448.294784878秒，低于原3600上限；原23:38:58.123369 UTC deadline未重置。[运行完成](../reports/supplemental-multirepresentation-20261104-v2/completion.json)、[审计闭合](../reports/supplemental-multirepresentation-independent-audit-20261104-v2/audit_stage_closed.json)及[参数／CPU／权重大小](../reports/supplemental-multirepresentation-20261104-v2/score/model_costs.csv)给出分项。含reserve的charge不是全部实际计算CPU，也不据共享CPU墙钟声称加速。

## 图表与物理解释状态

实际[模型PNG](../reports/supplemental-multirepresentation-20261104-v2/score/model_comparison.png)／[PDF](../reports/supplemental-multirepresentation-20261104-v2/score/model_comparison.pdf)、[source选轮PNG](../reports/supplemental-multirepresentation-20261104-v2/score/source_epoch_selection.png)／[PDF](../reports/supplemental-multirepresentation-20261104-v2/score/source_epoch_selection.pdf)可用于论文。原预测、权重及来源锁不覆盖。

传统4管线＋神经3管线的同拆分[28行联合表](../reports/supplemental-combined-models-20261104-v1/model_comparison.csv)、[84行逐seed指标](../reports/supplemental-combined-models-20261104-v1/per_seed_metrics.csv)、[Accuracy图](../reports/supplemental-combined-models-20261104-v1/accuracy_comparison.pdf)、[Macro-F1图](../reports/supplemental-combined-models-20261104-v1/macro_f1_comparison.pdf)及[旧171配对Accuracy图](../reports/supplemental-combined-models-20261104-v1/paired_old171_accuracy.pdf)已经真实生成。联合图误差条统一由三个实际seed计算样本SD，未把传统报告的population SD直接混用；历史96.0234%不作为同拆分横线。旧171只比较Accuracy及改对／伤害，避免不同输出类别口径的子集Macro-F1混淆。[联合图说明](../reports/supplemental-combined-models-20261104-v1/README.md)列明不同预算／表示及新增类别的限制。

PySCF的真实V6b水／乙醇各A/B/global三阶段已健康完成，独立v2分析获得3／21个全正模式。与同源旧stock完整raw Hessian最大差为2.928768338961163e−13／3.4083846855992306e−13 Eh/Bohr²；全部有符号频率最大差8.094502845779061e−11／1.0732037480920553e−10 cm⁻¹。证据位于量子研究工作树`reports/pyscf-spinosin-staged-20261105-v6b/control-{water,ethanol}-independent-analysis-002/checks.json`。这仅是小分子软件验证；IR仍NOT_COMPUTED，mechanism qualification为false。原分析器元数据接口失败保留，新v2修复未重算Hessian。

**本报告时点新Spinosin V6b大分子阶段仍NOT_RUN，不声称已取得其219个可信模式或药材峰归属。** DFT仅作机制辅助，不进入分类器、不提高或改写96.02%；孤立分子、药材固体集体振动与纯品化学归属须区分。

## 论文取舍

当前可写证据为酸枣仁传统可复核优化、source/query隔离下系统对照、表示消融与真实负迁移／少数类分析。黄芪辅助收益不能替代酸枣仁核心；参数寻优、标准ResNet与MC-UVE不能改名为全新算法。最重要缺口仍是ZY02扩样、真实溯源、同类别多独立批次与封存确认集。三条新类别query不足以替代这些工作，后续不得反复用当前query筛最有利配方。
