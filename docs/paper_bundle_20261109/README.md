# Research4 论文写作资料总入口

整理日期：2026-11-09，北京时间。本文汇总既有实验与回传证据；不是重新训练或新增确认实验。运行中的两项量子计算继续使用原代码和账本。

## 摘要

现在可以开始组织论文的方法、内部验证结果、消融与局限。主线建议采用**种类修正后的纯酸枣仁来源组识别**，以当前实验的样品角色、训练内选择和完整正负对照为依据。当前18表共1672条记录，中文种类和独立实物关系均有用户声明；这不等于新增批次、独立植物学鉴定或未曝光确认集。[CUR-001—002](03_CURRENT_DATA_AND_SPECIES.md)

旧855条数据的R2准确率96.0234%属于后来确认混有枣仁、酸枣仁的历史七来源任务。它是保留的方法开发证据，不能改写成当前纯酸枣仁成绩。最新等K主矩阵在300条纯酸枣仁TRAIN记录上，SNV吸收系数的MC20与MC100主方案均为93.4444%平均准确率；这是三次分区重复的内部评估，不是新增900个实物，也不是独立新批次性能。[历史链](02_HISTORICAL_ML.md)｜[当前矩阵](03_CURRENT_DATA_AND_SPECIES.md)

校准、迁移、模态融合有条件性正结果，也有明确负结果和不变结果，不能合并成“全面提升”。量子计算目前支持的是孤立分子数值导数与振动可靠性探索；IR强度、标准品实测和药材峰归属仍不能写成已完成。[迁移/稳健性](04_EXTERNAL_TRANSFER_AND_ROBUSTNESS.md)｜[来源诊断](06_WORKER_B_DIAGNOSTICS.md)｜[量子](05_QUANTUM_AND_MECHANISM.md)

## 开始写作时按这个顺序阅读

1. [论文定位、可写结论与图表计划](01_PAPER_READINESS_AND_CLAIMS.md)：先确定主任务、分母和结论等级。
2. [当前数据与修正种类后的全部实验](03_CURRENT_DATA_AND_SPECIES.md)：主文数据、角色划分、传统方法、神经网络与校准结果。
3. [全部实验族台账](08_EXPERIMENT_REGISTER.md)：按协议/实验族找设计、状态、指标和来源。附[Excel可读CSV](../../evidence/paper_bundle_20261109/experiment_registry.csv)与[完整JSON](../../evidence/paper_bundle_20261109/experiment_registry.json)。
4. [历史方法开发](02_HISTORICAL_ML.md)、[外部/迁移/稳健性](04_EXTERNAL_TRANSFER_AND_ROBUSTNESS.md)、[Worker B来源机制](06_WORKER_B_DIAGNOSTICS.md)：选择需要进入补充材料的对照，不遗漏负结果。
5. [量子与机制证据](05_QUANTUM_AND_MECHANISM.md)：区分已完成矩阵、虚频、最低点资格和还在运行的计算。
6. [复现与分支地图](07_REPRODUCIBILITY_AND_SOURCE_MAP.md)、[核验过的原始来源](09_REVIEWED_SOURCES.md)：打开固定提交，或检查本分支保存的原样小型报告。

## 六个容易混淆的数据范围

| 范围 | 分母/划分 | 应怎样称呼 |
| --- | --- | --- |
| 历史ZY01—ZY07 | 855，七来源，后来确认含两种中文类别 | 历史混合种类来源组开发研究 |
| 当前有效全库 | 18表，1672；枣仁364、酸枣仁500、黄芪403、天麻405 | 用户声明类别/独立实物的数据集合 |
| 当前master角色 | TRAIN999 / CAL334 / PRED339 | 当前协议中的固定角色；不是全部历史均未见的新确认集 |
| 纯酸枣仁 | 共500；TRAIN300 / CAL100 / PRED100 | 五来源组任务；内部验证与当前固定留出分别报告 |
| 枣仁/酸枣仁来源留出诊断 | TRAIN池516及其各轮来源留出/固定query | 已曝光训练池内的种类/来源覆盖机制诊断 |
| 分子计算 | 每个指定模型构象及其方法、网格、核坐标 | 孤立分子计算，不是增加药材谱样品 |

独立实物声明来自原855的2026-10-31声明及后续817的2026-11-08补充声明。没有把外部公开数据的人参根数、水果重复日期或模型训练行数自动变成新的独立实物数。数据来源与角色详见[CUR-001—002](03_CURRENT_DATA_AND_SPECIES.md)。

## 主要交付文件

| 文件 | 用途 |
| --- | --- |
| [实验台账CSV](../../evidence/paper_bundle_20261109/experiment_registry.csv) | 筛选实验族、设计、结果、结论、用途和限制 |
| [指标长表CSV](../../evidence/paper_bundle_20261109/metric_values.csv) | 保留各层条件和分母的数值查询；不混合排名 |
| [来源CSV](../../evidence/paper_bundle_20261109/sources.csv) | 固定提交、原路径、SHA-256、引用实验ID |
| [来源目录JSON](../../evidence/paper_bundle_20261109/source_catalog.json) | 跨分支去重导航，包含目录和大文件的可用位置 |
| [核验记录](../../evidence/paper_bundle_20261109/build_validation.json) | 汇总生成时逐项来源哈希核验与模块数量 |
| [原样来源副本](../../evidence/paper_bundle_20261109/reviewed_sources/) | 小型设计文档、汇总JSON/CSV的离线检查；不复制大型模型或原始数据档 |
| [独立审查](../../evidence/paper_bundle_20261109/independent_review.json) | 分母、版本、核心指标和链接的检查范围与限制 |

实验CSV避免把完整多层结果塞进超长单元格；按`experiment_id`连接[指标长表](../../evidence/paper_bundle_20261109/metric_values.csv)，或直接读取完整JSON。[当前纯酸枣仁主模型表](../../evidence/paper_bundle_20261109/tables/current_acid_primary_models.csv)、[全部修正任务固定留出表](../../evidence/paper_bundle_20261109/tables/corrected_fixed_holdout_all_results.csv)及[角色表](../../evidence/paper_bundle_20261109/tables/current_task_roles.csv)可单独打开。

## 与正在运行的任务如何衔接

最低点修复与毛蕊异黄酮续算在后台独立运行。本资料把观察到的运行状态保存成带时间的副本，不修改其配置、进程、账本或科学结论。后续新完成结果应追加新版本证据，不能覆盖本次截点。运行中的原始状态入口：[最低点修复](https://github.com/jackzhu119/research4/blob/research/pyscf-autonomous-90000-20261108/reports/pyscf-resume-20261109-v4/acid-minimum/ledger.json)｜[毛蕊异黄酮续算](https://github.com/jackzhu119/research4/blob/research/pyscf-autonomous-90000-20261108/reports/pyscf-resume-20261109-v4/caly-resume/ledger.json)。

论文中尚不能写成已完成的项目包括：真实独立批次/年份/仪器确认、独立植物学鉴定、HPLC/LC-MS成分定量、药材标准品THz峰归属、IR强度，以及仍在运行的严格最低点修复或完整毛蕊异黄酮新Hessian。详见[结论边界](01_PAPER_READINESS_AND_CLAIMS.md)。
