# 当前数据与种类修正实验：设计、回传结果与论文边界

整理日期：2026-11-09（Asia/Shanghai）。本页只整理固定提交中的已保存证据，不训练新模型、不接管后台计算。机器可读台账为[当前物种实验JSON](../../evidence/paper_bundle_20261109/current_species.json)，其中保存每项来源的完整提交和SHA-256。

## 执行摘要

当前有效材料为18份工作簿、1672条用户声明独立实物样品记录；酸枣仁500条、五个来源组是当前同种来源任务。后续用户声明补齐了原855条之外817条记录的独立实物依据，但没有新增批次或拉丁分类鉴定证据。原855条混合枣仁与酸枣仁，旧96.0234%必须保留为混合种类七来源组成绩。

纠正标签后的固定100份酸枣仁PRED上，R2为87/100，加入全部100份CAL重训为93/100；同场R0为94/100、完整重训96/100。之后在300份master TRAIN内部的三次五折开发验证中，等特征预算方案alpha-SNV/MC20/SVM的inner-selected结果为93.44%。这些数字回答不同问题，不能拼成一条模型性能提升曲线。

目前两大新矩阵均有真实回传结果：原物种重部署1305个科学作业已完成（225经典＋1080神经；剩余480由缓存续算完成），另一个等预算证据矩阵330/330已完成。条件性的正迁移和负迁移均存在，冻结模型校准也有明显负结果；它们都是已曝光TRAIN内部或固定内部留出研究，未构成新批次确认。

## 1. 当前18表数据与声明应一起引用（CUR-001—002）

| 中文种类 | 总数 | TRAIN | CAL | PRED | 来源组与限制 |
| --- | ---: | ---: | ---: | ---: | --- |
| 酸枣仁 | 500 | 300 | 100 | 100 | ZY03—ZY07，每组100份；来源标记不是五个已核实互斥行政区域 |
| 枣仁 | 364 | 216 | 74 | 74 | ZY01=350、ZY02=5，另三个ZR组各3份；高度不平衡 |
| 黄芪 | 403 | 241 | 80 | 82 | 4来源组；后续独立性声明不增造批次 |
| 天麻 | 405 | 242 | 80 | 83 | 4来源组；中文标签不是独立植物学鉴定 |
| 合计 | 1672 | 999 | 334 | 339 | 原855角色逐ID继承，其他817补齐角色 |

来源：[固定划分协议](https://github.com/jackzhu119/research4/blob/0156e2189d13456d1faf05ebfd53d86eeeb4f5e7/data/research/species_revision_20261108/protocol.json)、[逐样品固定名单](https://github.com/jackzhu119/research4/blob/0156e2189d13456d1faf05ebfd53d86eeeb4f5e7/data/research/species_revision_20261108/split_manifest.csv)、[新增817条独立实物声明](https://github.com/jackzhu119/research4/blob/0156e2189d13456d1faf05ebfd53d86eeeb4f5e7/data/annotations/specimen_independence_user_declaration_20261108.json)。

相对原855条ZY主任务，另外817条为黄芪403＋天麻405＋新枣仁9；这不意味着这些谱均于同一天新采集。原件、更名件、四种信号表和归档不得重复计为独立样品。`species`继续使用用户确认的中文种类，不代填拉丁种、药用部位鉴定或化学含量。

`main@806584a...`与标签修正分支虽使用相同version字符串，登记表SHA不同：main为`d7717138...`，修正分支为`dd9ec772...`；差异主要是后续独立性声明字段，谱文件路径、内容hash、数量和标签未变。论文应引用标签修正分支及两次声明，不能用main较早“新材料未声明独立”的scope覆盖后续确认。原ZY每来源只有一批的声明仍限定于原855条；后续没有实际新批次ID。

预处理/特征入口使用alpha主区间0.1—1.5保留612个原始频点；迁移用128点0.21—1.49网格。该区间属于工程设定，不能据此宣称仪器信噪有效区或化学最优范围已验证。

## 2. 标签修正后的固定留出结果已经完成（CUR-003—007）

### 2.1 旧预测的重解释不等于新训练

旧R0七分类773/855、R2为821/855。按中文种类折叠原预测，分别得到853/855和852/855；这只是同一旧预测的回顾性计数。旧七分类正确数在真实酸枣仁500条子集上由428升至470，贡献了全部48个净新增正确中的42个，说明任务标签误用不等于全部算法差异自动归零。不可把470/500写成新训练的纯酸枣仁五分类成绩。来源：[历史重标签审计](https://github.com/jackzhu119/research4/blob/0156e2189d13456d1faf05ebfd53d86eeeb4f5e7/data/research/species_revision_20261108/historical_relabel_audit.json)。

### 2.2 六项任务与全部酸枣仁分支

| 任务及预设路线 | 源模型正确数/PRED | Accuracy | Macro-F1 | Balanced accuracy | 加入全部CAL完整重训 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 枣仁/酸枣仁，R2 | 174/174 | 100.00% | 1.0000 | 100.00% | 174/174 |
| 酸枣仁五来源，R2 | 87/100 | 87.00% | 0.8693 | 87.00% | 93/100 |
| 四药材，固定全谱SVM | 331/339 | 97.64% | 0.9751 | 97.50% | 330/339 |
| 黄芪四来源，固定全谱SVM | 69/82 | 84.15% | 0.8386 | 84.09% | 69/82 |
| 天麻四来源，固定全谱SVM | 81/83 | 97.59% | 0.9756 | 97.62% | 82/83 |
| 枣仁五来源诊断，固定全谱SVM | 72/74 | 97.30% | 0.5972 | 60.00% | 72/74 |

| 酸枣仁算法分支 | 只用300 TRAIN | 加入100 CAL完整重训 |
| --- | ---: | ---: |
| 固定全谱SVM | 85/100 | 84/100 |
| 固定MC20＋SVM | 87/100 | 82/100 |
| R0训练内重新选参 | 94/100 | 96/100 |
| R2训练内重新选参 | 87/100 | 93/100 |

来源：[分类全量JSON](https://github.com/jackzhu119/research4/blob/0156e2189d13456d1faf05ebfd53d86eeeb4f5e7/data/research/species_revision_20261108/classification/summary.json)、[冻结配置](https://github.com/jackzhu119/research4/blob/0156e2189d13456d1faf05ebfd53d86eeeb4f5e7/data/research/species_revision_20261108/classification/frozen_config.json)、[已完成执行报告](https://github.com/jackzhu119/research4/blob/0156e2189d13456d1faf05ebfd53d86eeeb4f5e7/docs/SPECIES_REVISION_RESULTS_20261108.md)。R2训练内两折宏F1高于R0，但固定PRED上反而较低；不能看完PRED再把R0换成“事前指定的最优最终模型”。完整重训还使用了额外100个标签并重新拟合Pipeline，不是冻结模型K-shot校准。

枣仁74份PRED中延安占70份，永远预测延安就有94.59%。辽宁和内蒙古各仅1份PRED且均错，两个来源召回为0；97.30%不能单独支持可靠枣仁来源识别。山西与山西灵石也没有被确认成不重叠地理总体。

### 2.3 冻结分数校准没有稳定正增益

固定酸枣仁R2、同100份PRED，每类K份支持从独立CAL池抽取，20次嵌套抽样：

| K/类 | 不校准 | 原lambda=1锚定均值 | lambda=1000、混合0.05弱校准 |
| --- | ---: | ---: | ---: |
| 1 | 87.00% | 75.90% | 87.00% |
| 3 | 87.00% | 81.50% | 87.00% |
| 5 | 87.00% | 84.05% | 87.00% |
| 10 | 87.00% | 86.30% | 87.00% |

分数argmax和SVC投票另有零支持匹配控制，不能把决策规则变化算校准收益。更多支持抽样不增加独立query或批次数。来源同上分类JSON。

### 2.4 修正物种后的首次迁移是匹配负结果

| 固定三种子对照 | Accuracy均值 | Macro-F1均值 |
| --- | ---: | ---: |
| A：只训练目标 | 81.00% | 0.8085 |
| Aextra：额外目标预训练 | 80.33% | 0.8041 |
| B：黄芪＋天麻预训练 | 79.00% | 0.7881 |

辅助只用241份HQ TRAIN＋242份TM TRAIN，目标为300份酸枣仁TRAIN、100份固定PRED；128点表示，120步预训练、96轮目标训练，种子42/43/44。B−A平均−2.00pp、B−Aextra−1.33pp。只否定该固定设置的稳定正增益，不否定所有迁移。来源：[迁移全量结果](https://github.com/jackzhu119/research4/blob/0156e2189d13456d1faf05ebfd53d86eeeb4f5e7/data/research/species_revision_20261108/transfer/result.json)、[配对逐种子结果](https://github.com/jackzhu119/research4/blob/0156e2189d13456d1faf05ebfd53d86eeeb4f5e7/data/research/species_revision_20261108/transfer/paired_source_transfer.csv)。

初次结果已有独立整数混淆复算，14,638项检查通过、最大指标差0；9个神经模型logit重载复现。1366个分类设置与1449个迁移设置不是新增样品或独立研究。来源：[独立验证](https://github.com/jackzhu119/research4/blob/0156e2189d13456d1faf05ebfd53d86eeeb4f5e7/data/research/species_revision_20261108/independent_results_validation.json)。

## 3. 原1305作业矩阵已经续算完整，不再停留在575/825快照（CUR-008—010）

这一轮只导出999 master TRAIN，主CAL/PRED没有进入新搜索。经典部分225作业；神经部分1080作业。酸枣仁经典以300份TRAIN作三次划分重复（20261031/20261108/42），每次五外折、内层两折，比较alpha、n、n+alpha及R0/R2/fullSVM/PLS-DA。三重复的同一300份样品不是900个独立样品。来源：[科学协议](https://github.com/jackzhu119/research4/blob/8a388b8b4d8a5baf9f2e4321e76455c2c0b49eca/docs/SPECIES_RELAUNCH_SCIENTIFIC_REVIEW_20261108.md)。

| 原经典流程，酸枣仁开发OOF | R0 Accuracy | R2 Accuracy |
| --- | ---: | ---: |
| alpha | 85.44% | 91.33% |
| n | 81.00% | 64.33% |
| n＋alpha | 91.56% | 92.33% |

融合R2的15折中13折没有n特征，全部2943次入选频点中n仅28次；融合平均196.2特征而alpha为82.2。因此92.33%不能直接解释成物理互补收益，特征预算是竞争解释。n的R2固定SNV，15个R0折却都选择raw，提示绝对折射率水平的丢失也是待检因素。该动机及数字见[后续证据协议](https://github.com/jackzhu119/research4/blob/71a303e089c7c7b7b2557d62372aaceda44e2f91/docs/EVIDENCE_DRIVEN_VALIDATION_20261108.md)。

缓存续算保留原825项（225经典＋600神经），逐字复制尚未完成480个神经描述符继续执行；最终`reports/00480`记录1080个神经闭合作业，288个基础条件均完整，覆盖3目标×8profile×3预训练预算×4arm，各条件五折与三种子。不能再把“825、480、1080、1305”相加计为更多实验。1000 cache hits/80 new scans只是报告读取缓存，不是1000个新模型。来源：[缓存续算README](https://github.com/jackzhu119/research4/blob/df3e12656e2177cfcf0a298e66bdedc5c524cb74/README.md)、[剩余矩阵锁](https://github.com/jackzhu119/research4/blob/df3e12656e2177cfcf0a298e66bdedc5c524cb74/evidence/remaining_matrix_selection.json)、[最终00480汇总](https://github.com/jackzhu119/research4/blob/df3e12656e2177cfcf0a298e66bdedc5c524cb74/reports/00480/summary.json)。

酸枣仁`p24/e192/full/pre480`中A=72.00%、Aextra=73.5556%、B=77.2222%、匹配SVM=70.3333%，对应B−A=+5.2222pp、B−Aextra=+3.6667pp，允许报告**该匹配条件下**正迁移。其他预算、冻结层和种类有负结果，完整矩阵必须保留；与第2节81/79%的不同source/query协议不能直接做前后提升。

该最终精简readout只汇总`source_only/K0`；支持校准原记录保留，但没有在这一版重新汇总。不得将source-only基础矩阵的完成解释为所有校准结论均已汇总。不同目标上的事后最佳B配置也不相同，不能当预先锁定最终模型。

补回传分支另封存了00575时点：17个连续压缩分卷，2193文件、源字节约3.007GB、压缩约548.19MB，按manifest和round-trip收据恢复。它是**已有闭合结果的运输**，不是另一批575次新拟合，而且不含此后全部1305任务。来源：[分卷说明](https://github.com/jackzhu119/research4/blob/f38113e51072f0a99c864b2e6dae79a9169ff93c/README.md)、[最终补传收据](https://github.com/jackzhu119/research4/blob/f38113e51072f0a99c864b2e6dae79a9169ff93c/backfill_result.json)、[逐文件清单](https://github.com/jackzhu119/research4/blob/f38113e51072f0a99c864b2e6dae79a9169ff93c/manifest.json)。

## 4. 等预算证据矩阵330/330已完成（CUR-013—016）

v3先完成51个阶段模型后遇到sklearn1.8早停MLP字符串标签问题；v4仅修MLP内部标签编码并继承原失败与成本，不是换新算法或重领预算。最终330作业为15个经典匹配作业＋315个表示/迁移作业，122个汇总组均保留。来源：[协议及恢复说明](https://github.com/jackzhu119/research4/blob/71a303e089c7c7b7b2557d62372aaceda44e2f91/docs/EVIDENCE_DRIVEN_VALIDATION_20261108.md)、[最终00330结果](https://github.com/jackzhu119/research4/blob/71a303e089c7c7b7b2557d62372aaceda44e2f91/data/research/evidence_validation_20261108_v4/snapshots/00330/summary.json)。

### 4.1 内层选K的主结果与固定K消融分开

所有经典行均为同300份酸枣仁TRAIN、三次五折开发OOF的均值；source每折240、query60。K候选31/62/123/184/245，C/gamma及必要模态权重只在source内两折选择。

| 方法family | primary/inner-selected Accuracy | Macro-F1 |
| --- | ---: | ---: |
| alpha raw，MC20 | 93.22% | 0.9325 |
| alpha SNV，MC20 | 93.44% | 0.9344 |
| alpha SNV，MC100 | 93.44% | 0.9345 |
| alpha SNV，5×MC20聚合 | 92.78% | 0.9280 |
| n raw，MC20 | 83.67% | 0.8348 |
| n SNV，MC20 | 62.56% | 0.6220 |
| n＋alpha均SNV | 91.67% | 0.9169 |
| raw n＋SNV alpha全局筛选 | 87.11% | 0.8711 |
| 分块各半保留 | 88.78% | 0.8875 |
| 后缩放模态加权 | 89.22% | 0.8923 |
| 全alpha shrinkage LDA | 91.78% | 0.9176 |
| 全alpha MLP | 73.22% | 0.7286 |

固定K=245的alpha-SNV/MC100为93.89%，**只是消融行**，不能用外层分数挑K替代inner-selected程序。MC100相对MC20的准确率均差为0；5×MC20并未提高。n的raw明显好于SNV，受控fusion在本设置也没有稳定超过alpha。所有50条固定K与12条主结果及11项配对效应均在[指标CSV](https://github.com/jackzhu119/research4/blob/71a303e089c7c7b7b2557d62372aaceda44e2f91/data/research/evidence_validation_20261108_v4/snapshots/00330/metrics.csv)和JSON中。

配对不确定性按300个physical_sample_id聚类重抽，保留每份样品的三次预测，500次bootstrap；这是固定开发结果条件下的探索性区间，不是训练/选择全过程的不确定性或跨批次置信区间，不能因比较数量多而只挑有利区间。

### 4.2 神经表示密度、source量、辅助药材和计算量对照

alpha128/alpha512共用0.21—1.49区间；512点不是新的独立测量。source190保留50支持、query60；source240把支持池还给训练，不进行该支持校准。预训练480步，辅助各自保留来源头。source240/e144与source190/e192各576目标优化步，source240/e192为768步；相同步数仍非相同样品曝光或FLOPs。

| 无目标支持校准条件 | A：仅目标 | Aextra | HQ辅助 | TM辅助 | HQ＋TM |
| --- | ---: | ---: | ---: | ---: | ---: |
| alpha128/source240/e192 | 75.44% | 77.11% | 76.00% | 71.11% | 79.78% |
| alpha512/source240/e192 | 67.22% | 67.22% | 67.33% | 59.11% | 63.44% |
| alpha512/source190/e192 | 62.33% | 58.11% | 64.11% | 54.67% | 60.78% |

表中三种子42/43/44及全部五折完整，均为300个既有样品的开发OOF。128/source240在这个配方下有条件性正迁移；512的HQ＋TM却不优于target-only。不能说更多频点或辅助药材必然提高，也不能把这些神经分数与source240/612点SVM直接相减当纯架构效应。

alpha512/source190/HQ＋TM还保留K=1/3/5/10、20次嵌套支持的identity/传统/弱校准对照；每种条件要相对同source190的零支持规则比较，不能用source240分数充当校准基线。完整正负数值见同一00330 JSON，机器台账CUR-016保留全部相关行。

## 5. 独立来源组与组件系统诊断已完成（CUR-011—012）

另一主机B使用516个master TRAIN，固定SNV/Scaler/SVM，五个来源组配对留出：真实控制＋三个fit-label置乱，共20次实际拟合，没有再选参、没有提取CAL/PRED谱。来源：[真实结果](https://github.com/jackzhu119/research4/blob/c8945acc0a57d75aae438fd9de81d05b422cc1f3/reports/parallel-cloud-20261106/worker-b/B20261106-01/species-misuse-audit-v1/science/audit-attempt001/result.json)、[TRAIN-only控制](https://github.com/jackzhu119/research4/blob/c8945acc0a57d75aae438fd9de81d05b422cc1f3/reports/parallel-cloud-20261106/worker-b/B20261106-01/species-misuse-audit-v1/science/audit-attempt001/TRAIN_only_group_control.json)。

| 控制 | 正确数/516 | Accuracy | Macro-F1 | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| 真实标签 | 493/516 | 95.54% | 0.9534 | 94.68% |
| 训练标签置乱42 | 167/516 | 32.36% | 0.2487 | 27.96% |
| 训练标签置乱43 | 211/516 | 40.89% | 0.2955 | 35.30% |
| 训练标签置乱44 | 189/516 | 36.63% | 0.2705 | 31.56% |

真实控制23个错均为ZY01枣仁→酸枣仁，酸枣仁300份均对。来源组规模极不平衡，ZY01被留出时其余枣仁训练池极小。来源组不是核实的新批次，三置乱seed不支持正式p值；不得把95.54%与固定174/174当相同协议的算法提升或下降。

在相同339个PRED ID上，固定种类gate和正确route来源head同时正确304/339=89.6755%，误路由8份。逐真实种类：TM81/83、枣仁70/74、酸枣仁87/100、HQ66/82。它是组件逻辑与，未计算误route后的实际来源输出，完整joint macro-F1仍NOT_COMPUTED；不是一个新训练且完整评价的端到端层级系统。来源：[同ID联合正确性](https://github.com/jackzhu119/research4/blob/c8945acc0a57d75aae438fd9de81d05b422cc1f3/reports/parallel-cloud-20261106/worker-b/B20261106-01/species-misuse-audit-v1/science/audit-attempt001/joint_components_exact_correctness.json)。

## 6. 论文写作可用结论与不可跨越的边界

| 可以据此写 | 不能由当前证据推出 |
| --- | --- |
| 用户确认中文种类后，种类鉴别与同种来源组任务应分开 | 独立拉丁物种或药用部位鉴定、药效/化学含量、真伪标签已验证 |
| 酸枣仁内部留出/训练内开发存在可识别信号，来源与批次混淆需披露 | 外部新批次、跨年份、跨仪器泛化已经证实 |
| 旧R2提升中仍有酸枣仁子集贡献，但旧任务命名应纠正 | 把旧96.02%、固定PRED87/93/96%和开发93.44%拼成提升曲线 |
| 在匹配配方下迁移可以正，也可以负，预算和辅助种类改变结论 | 搜索中最好某配置证明所有迁移普遍优于强经典模型 |
| 同批支持校准有无收益都应保留，整池重训单独计标签量 | 93%整池重训代表1-shot/10-shot成功，或支持重抽新增独立验证 |
| 固定K/MC100/稳定聚合/融合等消融揭示竞争解释 | 依外层query挑固定K作为事前主方法，或把标准构造命名为新算法 |

数据来源组与某一批来源样品效应仍无法分离；新增独立批次、标准品THz、HPLC/LC-MS等仍不是本模块已完成实验。后续论文可以先写方法、可复核内部结果和限制；结果表必须标清群体、角色、分母、参数选择层级与实际source提交。

## 7. 机器台账与覆盖检查

本模块有16个CUR实验/证据family：数据及固定角色、旧预测重解释、六任务留出、校准、初次迁移、复算、225经典、分卷回传、原1305续算、B来源组控制、组件联合诊断、v3失败修复、等K经典、315神经条件、支持校准。它们是文档组织单位，不是16个独立受试队列。

完整JSON保留初次所有source/retrain行、原1305的288个神经基础组和配对差、最终330矩阵的全部经典/神经/校准汇总行、11项配对经典效应，以及每项固定来源SHA-256。大模型/压缩预测文件是否已完整回传，应按原始分支各自manifest核实；最终汇总已回传不自动等于所有大型artifact均公开。
