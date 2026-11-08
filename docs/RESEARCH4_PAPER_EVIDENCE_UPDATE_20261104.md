# Research4论文证据增补：主模型、迁移负结果与分子机制边界

记录日期：2026-11-04，America/Los_Angeles。证据读取时点为03:46 PST
（11:46 UTC）。本文件只整理已经存在的预测、审计和来源记录，不新增训练、
量子计算或数据清洗。它是[主计划](RESEARCH4_PAPER_C_MASTER_PLAN.md)、
[提纲](RESEARCH4_PAPER_OUTLINE.md)和[实验审计](RESEARCH4_EXPERIMENT_AUDIT.md)
的日期化增补；旧文中“迁移未运行”“Gaussian排除”“候选优化未运行”等段落
保留其当时语义，不能覆盖后续获授权的真实探索。本次分子计算使用PySCF，
没有Gaussian运行结论。

**主结果仍为SNV＋MC-UVE＋RBF-SVM：821/855＝96.0234%，Macro-F1
0.919685，Balanced Accuracy 89.9184%。** 该结果有标签、分区、选择锁、
原权重重放及同折重新拟合证据，是回顾性内部开发结果。新增两轮受控迁移
没有提高对应酸枣仁从头训练的Accuracy或Macro-F1；新增公开数值谱为0。
斯皮诺素SCF与梯度已实际完成，完整Hessian和正常模式仍在计算；药材分子
振动的机制结果在此时点为**NOT_RUN**，IR强度为**NOT_COMPUTED**。

## 1. 可供审稿核对的证据矩阵

| 研究问题 | 已有实证与分母 | 目前可支持的表述 | 资格边界与证据入口 |
| --- | --- | --- | --- |
| 酸枣仁模型是否有可重复改善？ | R0 773/855；R2 821/855；相同原5外折/2内折、seed42，pooled OOF | 在这批已曝光匿名组数据上，R2比R0净多48份正确，Accuracy增加5.6140个百分点，同时Macro-F1及Balanced Accuracy提高 | 项目E2，非E3；每组一来源批次，不能区分来源与来源批次效应。[历史逐行台账](../reports/paper-c/2026-10-31/history-audit/main_results.csv) |
| MC-UVE和SVM参数选择是否有作用？ | R2去MC、随机同K、固定C/gamma、去权重、固定MC20等预定条件消融均已重训 | 支持所选特征和参数在继承设置下有条件收益 | 各因素有交互；R0还内调Raw/SNV及C，R1−R0不能全部归于MC次数。[同折比较和消融](../reports/paper-c/2026-10-31/fair-benchmark/README.md) |
| 深度学习是否优于强SVM？ | 原升级120个外层管线；四表示主策略seed42为727/855、85.0292%、F1 0.7449；三seed平均83.7037% | 当前已执行的最小多表示网络没有超过R2；多表示值得作为完整正负结果的系统对照 | 12/24轮耦合profile改变多项设置，不能作为纯融合贡献或证明所有深度学习无效。[升级报告](MULTIREPRESENTATION_UPGRADE_REPORT.md) |
| 方法是否在其他药材上可用？ | 黄芪403、天麻405，各四匿名组，分别训练与评价；MC-SVM的seed42分别349/403、398/405 | 相同方法家族在两个独立植物任务中可执行，传统模型在本轮较强 | 原料祖先、独立批次、仪器及绝对α单位仍有缺口；不是冻结酸枣仁模型的零训练识别，也非全国来源或跨仪器确认。[辅助任务证据](../reports/upgrade-20261031-v1/auxiliary/README.md) |
| 外部预训练是否有净收益？ | 新48/96上限试验各30条source训练轨迹和45个outer管线；每arm每seed均评价相同855份 | 两预算的黄芪＋天麻预训练均低于同预算从头训练及额外目标训练控制的Accuracy/Macro-F1 | 两次均为曝光队列上的开发性谱形实验，不是两个确认队列。[两预算报告](CONSERVATIVE_CROSS_PLANT_BUDGET_ABLATION_20261104.md) |
| 保守少样本校准是否提准？ | 原513/171/171独立协议；100次重叠1-shot支持抽样，源内门禁回退identity | 保守回退避免本协议传统λ1的大幅伤害，但没有分类增益 | 171 query不等于855 OOF；七类3/5/10-shot不可用，ZY02每次同一份支持，未完成真实跨批次校准。[原校准证据](../reports/paper-c/2026-10-31/calibration/README.md) |
| 是否获得新公开训练谱？ | 本轮14条来源资格记录、四次机器GET失败记录；新增数值谱0 | 有来源和阻塞台账，可决定下一次合法获取的优先次序 | 网页元数据、文件数、压片数或像素数不能变成新增独立植物数。[外部来源报告](RESEARCH4_EXTERNAL_EVIDENCE_20261104.md) |
| 化学资料是否支持候选选择？ | 两上传SDF分别核为Spinosin C28H32O15/75原子及Calycosin C16H12O5/33原子；ChEBI图与立体Key匹配 | 支持计算输入身份及分子候选的文献依据 | 计算构象不是晶体实测结构；没有855份样品逐一配对的含量数据。[化学证据](RESEARCH4_CHEMICAL_EVIDENCE_20261104.md) |
| 是否已有酸枣仁真实理论振动解释？ | C001优化证据、fresh SCF和独立恢复梯度存在；V5直接续跑仍活跃，读取时71/150贡献单元已健康闭合 | 电子结构及数值流程有真实进展，完整振动解释尚未完成 | 贡献单元不是频率或完成百分比；不能提前填入理论峰或配对结论。[V5说明](PYSCF_V5_RESUMPTION_20261104.md)、[本次续跑登记](../reports/pyscf-spinosin-resumable-20261104-v5/resource-protocol/direct-resume-001.json) |

本表中的用户独立样品声明与实验室溯源审计不同。匿名ZY01—ZY07仍可作为
当前分类标签；若正式论文宣称特定生长产地，应另补真实名称及溯源。当前
独立确认队列为0，重放、更多种子和更多外折运行都不会增加独立样品数。

## 2. 主论文优先保留的五阶段结果

以下数值从[审计主表](../reports/paper-c/2026-10-31/history-audit/main_results.csv)
读取；每行分母855、每份样品恰有一次OOF预测。百分比与百分点分开表达。

| 阶段 | 正确数 | Accuracy | Macro-F1 | Balanced Accuracy | 较R0的Accuracy变化 | 较R0相对准确率增长 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 原MC-UVE＋SVM | 773 | 90.4094% | 0.809185 | 83.6735% | 0 | 0 |
| R1 MC参数选择 | 791 | 92.5146% | 0.827691 | 86.6531% | ＋2.1053个百分点 | ＋2.3286% |
| R2 SVM参数选择 | 821 | 96.0234% | 0.919685 | 89.9184% | ＋5.6140个百分点 | ＋6.2096% |
| R3 预处理选择 | 816 | 95.4386% | 0.887720 | 86.3878% | ＋5.0292个百分点 | ＋5.5627% |
| R4 组合家族选择 | 812 | 94.9708% | 0.882042 | 85.8163% | ＋4.5614个百分点 | ＋5.0453% |

R2有34份错误，ZY02只对3/5。R2五折特征数92/62/123/123/92、平均98.4，
均使用SNV；SNV仍使用完整612点，不能把较低分类器维度写成相同比例的
仪器采集成本节约。R2是逐折训练内选择的策略和五个折模型，不是一组全局
唯一参数，也不能直接把96.0234%赋给全855训练的部署模型。

下降应保留：R2→R3纠正5份、伤害10份；R3→R4纠正9份、伤害13份。
这说明扩展候选不保证外层提升；没有证据把下降全归于某一种SG、MSC或
稳定筛选失效。主模型选择保留R2，而不是把四轮写成单调增长过程。

同一855分区的新对照可以放入另一主表：全谱SNV-SVM 753/855＝88.0702%、
F1 0.7963；收缩LDA 786/855＝91.9298%、F1 0.8300；单隐层sklearn MLP
759/855＝88.7719%、F1 0.7900。历史64–32强MLP seed42的200轮基础训练为
768/855，追加100次源监督更新后777/855＝90.8772%、F1 0.8319；追加的是
100次更新，并非再训练100轮。全部三seed仍须附表，不能只挑seed42的收益。
这些模型具有相同分区和源内选择边界，候选数、表示及预算不同，适合任务
层面的开发比较，不是严格等计算量或只更换一个组件的实验。

## 3. 本轮两预算迁移结果必须完整分表

来源为[六行CSV](../reports/cross-plant-conservative-pilot-20261104-v2/two_budget_model_comparison.csv)。
A为酸枣仁从头训练；Aextra在目标source先训练再以同一L2-SP规则微调；
B使用黄芪＋天麻预训练再微调。所有外部植物保持各自分类头，未把外部类别
解释为ZY标签。下表均为seed42/43/44的均值，每seed分母均为同一855份。

| 轮数上限 | 方法 | Accuracy | Macro-F1 | Balanced Accuracy | 较同预算A变化 |
| --- | --- | ---: | ---: | ---: | ---: |
| 48 | A | 80.3509% | 0.678626 | 73.9592% | 0 |
| 48 | Aextra＋L2-SP | 78.3236% | 0.651480 | 70.1701% | −2.0273个百分点 |
| 48 | B黄芪＋天麻＋L2-SP | 76.1404% | 0.630414 | 68.3333% | −4.2105个百分点 |
| 96 | A | 87.0565% | 0.760988 | 79.1837% | 0 |
| 96 | Aextra＋L2-SP | 86.3548% | 0.747579 | 77.3878% | −0.7018个百分点 |
| 96 | B黄芪＋天麻＋L2-SP | 85.3801% | 0.737772 | 77.4082% | −1.6764个百分点 |

128点0.21–1.49THz的α、SNV、source-only scaler、小ResNet、L2-SP系数0.001
及120步预训练均固定。共同目标轮数只由A的两内折source OOF选择，选择锁
早于outer预测，预测锁早于独立真值评分。v2的五折选择91/86/92/89/92轮；
与v1前48轮的30条source轨迹逐元素一致，完整45个外层权重重放7695条预测
的logit最大差0。两轮训练/评分实际CPU共720.283753秒，完整失败/负结果保留。

48→96时，A提高6.7057、Aextra提高8.0312、B提高9.2398个百分点。这是相同
观测上的候选轮数预算消融；外部池和预训练步数未变，不能称新数据增益。
v2在首轮开发曝光后追加登记，也不能写成起初封存的独立确认。

96上限下B相对Aextra仍低0.9747个百分点，F1低0.009807，但Balanced
Accuracy高0.0204个百分点。应按三项指标分别报告，不能写成“所有指标
均下降”，也不能只摘这个很小的正差称迁移成功。seed42/43/44的B相对A
纠正/伤害分别32/37、47/62、35/58，三seed正确数分别754、714、722。
局部seed的指标改善不改变完整配对结果。

## 4. 哪些比较可以放在同一结论中

| 比较 | 可作的比较 | 不能据此推出的结论 |
| --- | --- | --- |
| R0—R4 | 相同855/五二折/seed42的pooled OOF开发变化；同时列F1和BA | 所有阶段单调提升；每个增量都是独立算法因果贡献；全新外部确认 |
| R2与同折SVM/LDA/MLP | 相同评价成员、源内学习边界下的方法比较，附预算/表示差异 | 严格等计算预算或只改变分类器；旧171成绩可直接挪入本表 |
| 每预算A/Aextra/B | 相同目标成员、主干、选轮规则下的受控迁移配对 | 跨批次或跨仪器已验证；三seed等于三组独立植物样品 |
| 同一arm的48/96 | 扩大候选训练轮数的开发性敏感性/预算消融 | 更换外部数据带来的改善；新确认队列 |
| 新128点三seed网络与历史612点seed42 R2 | 可分面展示已探索模型的性能及协议，说明R2仍是强候选 | 相减后称同一MC优化阶梯的增量或外部数据净贡献 |
| 黄芪、天麻、人参和酸枣仁 | 分别定义的任务内评价与方法可执行性 | 同号类别共享标签；分数高即说明某植物本征更易；不同模态拼接成同类THz |
| 171 query校准与855 OOF | 各自分表讨论校准与模型识别，明确各自分母 | 把171的改善/下降解释为96.02%模型校准的净变化 |
| 文献/SDF/理论模式与分类器 | 分子输入、机制动机和将来独立固定的先验研究 | 充当新药材谱；按真实产地填文献平均含量；理论数据已提高分类准确率 |

## 5. 可直接引用的论文表图与原始证据

下列均为现存产物，不是新生成的示意指标。种子均值和单seed图在图注中分开。

| 建议论文位置 | 表/预测依据 | 已有图 |
| --- | --- | --- |
| 主表1：R0—R4 | [历史审计CSV](../reports/paper-c/2026-10-31/history-audit/main_results.csv) | [Accuracy/F1 PNG](../reports/paper-c/2026-10-31/paper-assets/historical_stage_accuracy_macro_f1.png)、[PDF](../reports/paper-c/2026-10-31/paper-assets/historical_stage_accuracy_macro_f1.pdf) |
| 主表2：同折分类器与条件消融 | [真实重训比较CSV](../reports/paper-c/2026-10-31/fair-benchmark/model_comparison.csv) | [分类器PNG](../reports/paper-c/2026-10-31/paper-assets/unified_classifier_comparison.png)、[消融PNG](../reports/paper-c/2026-10-31/paper-assets/r2_conditional_ablation.png) |
| 主图：R2少数类与频点 | [候选索引](../reports/paper-c/2026-10-31/best-candidate/candidate_index.json) | [混淆矩阵](../reports/paper-c/2026-10-31/paper-assets/r2_confusion_counts_and_recall.png)、[K与描述性稳定性](../reports/paper-c/2026-10-31/paper-assets/r2_selected_frequency_count_and_stability.png) |
| 辅助表：多表示及其他药材 | [升级证据表Excel](../reports/upgrade-20261031-v1/paper-assets/Research4升级_多表示神经与防泄漏证据.xlsx)、[辅助任务审计](../reports/upgrade-20261031-v1/auxiliary/README.md) | [升级图表索引](../reports/upgrade-20261031-v1/paper-assets/README.md) |
| 迁移表及预算图 | [六行模型表](../reports/cross-plant-conservative-pilot-20261104-v2/two_budget_model_comparison.csv)、[预算变化CSV](../reports/cross-plant-conservative-pilot-20261104-v2/epoch_budget_changes.csv) | [预算对照PNG](../reports/cross-plant-conservative-pilot-20261104-v2/epoch_budget_ablation.png)、[PDF](../reports/cross-plant-conservative-pilot-20261104-v2/epoch_budget_ablation.pdf)、[source选轮曲线](../reports/cross-plant-conservative-pilot-20261104-v2/source_epoch_curves.png) |
| 迁移逐类/错误分析 | [全部seed P/R/F1](../reports/cross-plant-conservative-pilot-20261104-v2/per_class_metrics.csv)、[全部混淆计数](../reports/cross-plant-conservative-pilot-20261104-v2/confusion_matrix_counts.csv)、[改对/伤害](../reports/cross-plant-conservative-pilot-20261104-v2/paired_corrected_harmed.csv) | [v2 seed42混淆图](../reports/cross-plant-conservative-pilot-20261104-v2/confusion_seed42.png)；v1应使用[已修正布局图](../reports/cross-plant-conservative-pilot-20261104-v1/confusion_seed42_layout2.png)，两图都仅代表seed42 |
| 可复核的迁移预测 | [v1全部预测](../reports/cross-plant-conservative-pilot-20261104-v1/all_outer_predictions.csv.gz)、[v2全部预测](../reports/cross-plant-conservative-pilot-20261104-v2/all_outer_predictions.csv.gz)、[v2原生权重重放](../reports/cross-plant-conservative-pilot-20261104-v2/native_model_replay.json) | 单条预测用于配对核验；三seed合并7695行不代表7695个独立样品 |
| 来源/许可/下载阻塞附表 | [14条谱来源](../reports/external-evidence-20261104-v1/spectra/source_registry.csv)、[四次下载实测](../reports/external-evidence-20261104-v1/spectra/network_attempts_20261104.json)、[化学候选表](../reports/external-evidence-20261104-v1/chemistry/chemical_candidates.csv) | 本轮来源目录没有新光谱或化学结果图，不补画假谱 |
| 分子机制准备附录 | [20条源内候选峰](../reports/upgrade-20261031-v1/source-peaks/experimental_candidates_by_fold.csv)、[75行原子映射](../reports/external-evidence-20261104-v1/mode-interpretation-preparation/C001_atom_map.csv)、[分子分组](../reports/external-evidence-20261104-v1/mode-interpretation-preparation/atom_groups_C001.json) | 静态分组和实验候选峰已有；真实Spinosin频率及振型图此时NOT_RUN |

不能在一张无协议说明的排行榜中混排以上所有数值。主结果、预算消融、
其他植物和分子机制分别成表；图注说明样品粒度、种子聚合方式、历史曝光
和表示/预算。共享CPU下的墙钟与当前PySCF并发，不能据此宣称普适提速。

## 6. 化学与PySCF部分目前可写到哪里

结构核验可写：上传Spinosin的完整InChIKey为
`VGGSULWDCMWZPO-ODEMIOGVSA-N`，中性Calycosin为
`ZZAJQOPSWWVMBI-UHFFFAOYSA-N`，均有可信图来源。上传Spinosin构象与
当前C001的原子顺序不同；本轮保持旧C001坐标、轨道及原子编号，未混用。
75行原子对应和图分组只为后续正常模式描述准备，不能写成已观察到分子运动。

数值流程可写：已优化水/乙醇的完整频率与跨进程Hessian对照分别得到3/21个
全正模式，最大Hessian差约1.09×10⁻¹³/8.88×10⁻¹⁶ Eh/Bohr²；这是计算
流程控制，不是药材结果，也没有证明大分子加速。当前Spinosin使用显式的
PBE/def2-SVP/grid1、中性单重态、762 AO/320电子，不含DF或D3，IR强度
未计算；fresh SCF、同方法梯度及恢复梯度真实完成。还没有独立CPHF残差
范数，不能将正常返回冒称已测量该残差。[详细数值控制](PYSCF_V5_RESUMPTION_20261104.md)
保留实际版本、阈值、日志和资源边界。

截至本快照，`direct-resume-001`的进度事件为`SUPERVISOR_RUNNING`，其
阶段性`status=READY`不是最终状态。旧自动备份曾因GitHub内部错误停止，
51个单元已保全并补推；当前直接控制器沿用同一预算和科学输入续跑，旧
`BLOCKED_PARTIAL_JOB`报告保留失败历史，不代表当前仍未恢复。最终论文
应在完整闭合result、150个贡献单元、完整Hessian、219个正常模式及独立
复算通过后再更新本节；计算完成与全正频极小点资格分别记载，虚频不删除。

实验候选峰为各外折684条source均值按固定规则提取，每折4个，共20行。
五个训练集重叠，20行不能算20个独立化学峰。无纯品THz或同一批药材的
HPLC/LC-MS测定时，即使将来理论频率接近1.08、1.13–1.14、1.18–1.19或
1.28–1.30THz，也只能讨论候选关系。孤立分子缺少晶格、氢键网络和复杂
基质；文献SERS的1076/1614cm⁻¹对应32.26/48.39THz，不在目标区间。

本轮没有把SDF、理论峰或文献地区均值放入来源分类器。一个化合物的固定
结构描述符不能为855个未知样品提供不同实测含量；按来源回填文献均值还
会携带标签。将来若研究理论频段先验，需另立source内固定协议，并设置
无先验和匹配数量随机频段对照，不回溯选择当前测试上的有利峰位。

## 7. 论文结果顺序与真正缺口

正文先回答酸枣仁问题：样品/源组及失衡→R0/R2改善→同折分类器与条件
消融→逐类和错误分析。随后呈现多表示网络的完整结果、黄芪/天麻独立
任务及受控负迁移，最后放少样本保守回退和已完成范围内的分子机制探索。
本顺序保留酸枣仁优化为主要贡献，不要求按工作量比例凑文字。

可以使用的结果概述是：“在855份、七个匿名来源组的回顾性内部OOF评价中，
训练内MC-UVE与RBF-SVM选择策略取得96.0234%的Accuracy和0.9197的
Macro-F1，较参考流程提高5.6140个百分点。受控神经网络与多植物预训练
实验未超过该强候选，负结果与少数类限制均保留。”不能把这段扩写成新
算法、真实跨批次泛化、少样本稳定提准或已由化学成分证明来源差异。

| 仍缺的证据 | 为什么当前模型分数不能补足 | 满足后才可新增的结论 |
| --- | --- | --- |
| 覆盖相同类别的真正新来源批次，尤其ZY02；独立support/query及溯源日志 | 当前各组一批、ZY02仅5份，重复抽样不能产生新批次 | 在预先冻结模型和规则下的独立确认、真实跨批次校准 |
| 仪器、α长度单位、厚度/水分、参考谱和测量日期 | 同轴数值和插值不证明物理兼容，处理后的谱形可保留来源混杂 | 有条件的跨仪器/批次方法泛化，物理参数一致性讨论 |
| 公开原始字节、许可和原料/重复层级 | 来源页面可读或文件多不代表独立训练样品，当前新增下载为0 | 新外部任务及另行冻结的迁移试验 |
| 完整Spinosin频率及极小点核验；适用时更稳健方法/构象检查 | 当前SCF和部分导数不等于Opt/Freq全流程完成 | 有边界的孤立分子低频运动描述；仍不能单凭近峰确认成分 |
| 纯品谱或与本项目ID逐一配对的HPLC/LC-MS | 文献成分和全局分子描述符不提供当前样品的实际含量 | 可检验的化学关联及真正样品级多模态融合 |
| 明确的有限复杂度新方法及独立验证 | MC-UVE/SVM/LDA/ResNet/L2-SP均为已有方法；参数搜索本身不构成原创 | 按真实增量确定方法贡献和投稿范围，不预先保证SCI新颖性 |

论文可重复证据已经比单一最高分完整，但独立新批次、来源/物理溯源以及
机制资格仍是实质缺口。后续实际振动计算完成时，应新增有结果SHA的最终
状态报告再引用；本文件保留03:46 PST这一尚未完成的快照，不回写历史结果。
