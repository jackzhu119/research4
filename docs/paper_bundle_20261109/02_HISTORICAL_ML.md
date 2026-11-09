# 历史855条记录：算法、校准、嵌套优化与部署实验汇编

整理日期：2026-11-09（北京时间）；证据固定提交：`cd4b50c37c1e26064f9f15f340d30e20ac386458`。

## 摘要

历史主结果是混合ZY匿名组任务的R2：821/855，Accuracy **96.0234%**、Macro-F1 **0.9196846**、Balanced Accuracy **89.9184%**。相对773/855的R0净增48份正确（纠正62、引入14），为同一旧分区中的可重放回顾性结果。R3/R4、固定预算CNN和多表示神经策略未进一步超过它。

**数据纠正优先于旧报告标题。** 855个历史ZY记录已知含枣仁（ZY01/02）和酸枣仁（ZY03—07）；旧文件标题中的“酸枣仁”是当时口径。本汇编不把这些分数写成纯酸枣仁产地分类，也不转移到当前18表1672条记录。后者的单种类实验另见本汇编当前数据模块。

多数小样本校准主方案回退为零支持；弱神经FO-MAML曾改善自身，但其绝对表现远低于SVM，后续匹配监督控制也未支持MAML特有收益。固定171留出、855五折OOF和单上下文342内验证有不同分母与用途，**不拼成一条“逐轮提高”的准确率阶梯**。

本文件覆盖22个历史实验/工程家族。外部植物、跨植物迁移、11月4日补充多表示、11月7日固定OOF/扰动/增强由本汇编外部与补充模块登记，不在此重复计数。这里的“家族”不是22个独立研究或样品队列。

## 先分清三套评价口径

| 口径 | 固定样品/分区 | 对论文允许的解释 |
| --- | --- | --- |
| 855样品级嵌套OOF | 5外折/2内折，split seed20261031，常用model seed42；每方法每样品一次外测 | 已曝光混合ZY任务的内部对照；非跨批次、非当前单物种任务 |
| 513/171/171留出 | seed20261101；513源训练、171支持池、171固定query；七类仅1-shot | 同批次回顾性校准/表征/概率比较，不能称未见确认 |
| 342/342源内pilot | 固定outer00.inner00，342source和342inner-validation；不读取外层171 | 学习轮数与优化诊断，无新855OOF或独立确认成绩 |

所有训练侧预处理、缩放、选点与参数选择须保留原实现的source范围；哈希/分离接口是审计证据，不能消除研究者多轮看过测试分数的全流程曝光，也不能证明所有生物学样品/批次已由实验室核实。

## 实验设计、结果和论文用途

| ID | 实验家族 | 已保存结论 | 推荐用途 |
| --- | --- | --- | --- |
| HML-001 | 老师代码静态审读与方法语义核对 | 定位CARS报告误差除2.8、MC-UVE有符号排序差异、SSA复用寻优折评分、PLS联合归一化、CNN人工2D重排/验证集复用等问题。标准方法和现成包不能因组合或改名变为原创算法。 | 方法实现与可复现性背景；保留原作者/许可引用，不能作本研究新算法性能证据。 |
| HML-002 | Octave/Python核心兼容性与工程实测 | 核心PLS/SVM/MC-UVE可复用；92次尝试的pass包含成功复现缺陷。原SSA示例调参和报告同折，示例CNN存在跨集相同特征行；其高分均不作药材泛化成绩。 | 软件可用性和缺陷修复说明，科学实验表中应与药材结果分开。 |
| HML-003 | 批准主矩阵：传统分类与变量筛选 | MC20优于本矩阵普通全点SVM；提高MC次数、重复稳定筛选没有进一步增益。CARS-style为Python多分类适配，不是原MATLAB精度复现。 | 历史基线与负对照；可引用设计/训练内选择思路，分数仅当时混合ZY任务。 |
| HML-004 | 批准矩阵中的模态与完整导出范围敏感性 | 直接拼接未超过同类α单支。全导出高分不能证明该范围物理有效或不存在来源/制样混杂；负α及有效频段仍需仪器记录。 | 表示消融与高分敏感性诊断；不能以100%替换预定主区间。 |
| HML-005 | 固定三用途留出与首次1-shot分数校准 | SVM88.8889%→74.9415%，MC20 87.7193%→72.8947%；固定弱样本校正平均明显下降，不是默认改善步骤。 | 可写为同批次回顾性校准负结果，不是跨批次校准。 |
| HML-006 | 116配置校准搜索与六类辅助K-shot | 七类正式主方案均回退不调整。强正则弱校正SVM89.0994%(+0.2105pp)仅预选诊断，开发增益未过规则，不能事后改主方法；六类10shot开发通过后最终仍微降0.0353pp。 | 保留全部正负结果、matched-zero结构控制与开发/最终差异。 |
| HML-007 | 54源表示/分类器搜索 | 开发选出的LDA最终未胜SVM；较高平衡准确率不等于所有指标改善。 | 表示/读出模型对照；不得只挑有利指标。 |
| HML-008 | 60潜空间适配及全范围敏感性 | 三个主适配均回退不调整；PLS局部记忆给支持标签反而低于相同零支持结构，不能将读出变化当校准收益。 | 匹配零支持消融和负结果；100%另列敏感性。 |
| HML-009 | 论文启发的监督ResNet/ProtoNet/FO-MAML分类适配 | FO-MAML自身64.7173%→67.5458%，宏F1约.3922→.4338，仍低于88.8889%SVM；头微调/全网微调/ProtoNet未通过开发门禁，诊断负结果保留。 | 可作为弱神经起点的探索性自身收益，不是强主模型优势或原论文严格复现。 |
| HML-010 | 四轮85候选：强MLP、支持加权SVM、回放锚定与元训练控制 | 正式90.2534%来自零目标支持的源训练；同任务监督继续训练逐样品预测与FO相同，未显示MAML特有增益。500更新高分仅预选诊断，校准反降。 | 可报告基础训练改善和校准失败；不能写“7样本校准达到90.25%”。 |
| HML-011 | 图片引导特征/分类40设置 | LDA158/171=92.3977%但宏F1.8352<原SVM.8616及强MLP.8787；变量筛选候选没有自动提准。 | 多指标冲突与稀少类假阳性负对照；不可只报最高accuracy。 |
| HML-012 | 图片引导概率/判别21设置 | SVM温度缩放NLL .541355→.409750、Brier .264373→.199344、ECE .229348→.093669，准确率保持88.8889%；硬分类改动未过开发。 | 概率可靠性与分类准确率分开报告，不把temperature解释为提高分类。 |
| HML-013 | 图片引导SVM/MLP融合20设置 | 正式回退SVM权重0的较强单MLP，不能称融合收益；真值oracle91.8129%不可部署，不是实测模型成绩。 | 融合负结果和错误互补诊断；oracle只用于解释可能空间。 |
| HML-014 | R0—R3嵌套优化与90.41%精确重现 | R2 821/85596.0234%、宏F1.9196846、BA89.9184%；更复杂R3反降5个正确。R1相对固定锚定与R0的对比含义不同，不能全归因单个因素。 | 历史混合匿名组可重放强候选及方法选择过程；不得转成当前纯酸枣仁主成绩。 |
| HML-015 | R4组合家族、稳定选择与CNN受控对照 | R4内选策略94.9708%低于R3和R2；预算MC、稳定MC/CARS、CNN路线均未胜较强传统路线。 | 完整负结果与复杂度/选择方差讨论；不能推断所有CNN、SG或稳定方法普遍无效。 |
| HML-016 | 统一855协议基线复现与条件消融 | R2精确重现；全谱SVM88.0702%、LDA91.9298%、单隐层MLP88.7719%；去MC、随机同K、去C/gamma搜索、去类权等条件消融均下降。 | 可重放的历史对照和贡献分解；不同模型网格/成本不同，不称所有方法严格等预算。 |
| HML-017 | 原64—32强MLP的三seed统一OOF与源更新控制 | 追加训练的三seed均值略升，seed43反降；未超过R2，不能只挑seed42多9份作为所有种子改善。 | 神经强对照和种子敏感性；模型架构本来自旧513开发，历史选择曝光需明列。 |
| HML-018 | 统一研究审计中的保守校准独立复核 | v1限幅使满足翻类必要条件的11120对实际均不翻类；正式回退无增益，不能作为提准新方法。旧λ1负结果保留，抽样均值不覆盖先前20draw。 | 保守规则的机制负诊断，区分“稳定不坏”与“带来新收益”。 |
| HML-019 | 多表示/深度学习升级：8家族小预算嵌套实测 | 四表示主策略seed42 85.0292%、三seed均值83.7037%，未超过R2；12/24epoch小预算不能否定所有深度网络，多表示profile耦合不能称纯因素融合消融。 | 神经/融合辅助对照与真实负结果；不能把同批重复模态当新增独立样品。 |
| HML-020 | 480条历史训练损失曲线只读诊断 | 全部配置末段训练loss仍下降，仅说明该损失定义下优化仍有进展；缺逐epoch验证曲线，不能据此判定欠拟合或延长训练必提准。 | 学习过程诊断，不能当新外层模型成绩。 |
| HML-021 | 仅源内的96epoch学习曲线pilot | 24→96轮αCNN内验证64.9123→75.9259%，四表示76.6082→83.8207%；四表示BA48→96却74.9320→69.4388。优化进展和少数类/泛化单调改善不是一回事。 | 源内学习曲线及负变化；不能和855OOF96.02%直接相减，也未据此改部署。 |
| HML-022 | 匿名七分类完整部署与未来输入工具 | 工具和完整模型已交付；最终MC50/PLS1/20%=123点，SVM C1000/gamma0.1×scale/balanced。真新样品准确率未测，不能给部署模型填历史96.02%。 | 工程应用入口与可复现性附件，不替代未见样本验证；匿名类号不自动映射真实物种/产地。 |

## 可直接核对的历史主表

| 方法/角色 | 正确/855 | Accuracy | Macro-F1 | Balanced Accuracy |
| --- | ---: | ---: | ---: | ---: |
| R0 | 773/855 | 90.4094% | 0.8091848 | 83.6735% |
| R1 | 791/855 | 92.5146% | 0.8276913 | 86.6531% |
| R2 | 821/855 | 96.0234% | 0.9196846 | 89.9184% |
| R3 | 816/855 | 95.4386% | 0.8877204 | 86.3878% |
| R4 | 812/855 | 94.9708% | 0.8820416 | 85.8163% |

[固定提交原始指标CSV](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/paper-c/2026-10-31/history-audit/all_method_metrics.csv)；[四轮嵌套报告](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/STAGED_NESTED_OPTIMIZATION_REPORT.md)。R1还应与固定SNV/C10/MC20锚定88.1871%分别解释，不能把与R0的变化全归因MC参数。

R2的ZY02仅3/5识别正确（60%召回）；总体96.02%不表示稀少类也有96%准确率。旧五折模型和全855重新拟合部署模型是不同产物。

## 每个家族的设计和证据入口

### HML-001 老师代码静态审读与方法语义核对

**设计与范围：** 8个算法压缩包、147个可读文本成员；核对入口、预处理、变量筛选、PLS核心/CV、SSA适应度、CNN/BP；未声称逐行审计所有LIBSVM/二进制。

**实际结论：** 定位CARS报告误差除2.8、MC-UVE有符号排序差异、SSA复用寻优折评分、PLS联合归一化、CNN人工2D重排/验证集复用等问题。标准方法和现成包不能因组合或改名变为原创算法。

**论文定位：** 方法实现与可复现性背景；保留原作者/许可引用，不能作本研究新算法性能证据。

**限制：** 静态发现不等于全部原程序均已修复或所有MATLAB环境必失败。

**固定证据：** [docs/ALGORITHM_REVIEW.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/ALGORITHM_REVIEW.md)。

### HML-002 Octave/Python核心兼容性与工程实测

**设计与范围：** GNU Octave9.4/Python CPU隔离环境；29原脚本入口、原核心、声明兼容适配、逻辑/性能/形状夹具；保存每次命令/输入/源码hash。

**实际结论：** 核心PLS/SVM/MC-UVE可复用；92次尝试的pass包含成功复现缺陷。原SSA示例调参和报告同折，示例CNN存在跨集相同特征行；其高分均不作药材泛化成绩。

**论文定位：** 软件可用性和缺陷修复说明，科学实验表中应与药材结果分开。

**限制：** Octave兼容性不是MATLAB原生复现；工程夹具和原包演示不增加药材样本。

**固定证据：** [docs/RUNTIME_USABILITY_REPORT.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/RUNTIME_USABILITY_REPORT.md)；[docs/INSTALL_AND_RUN.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/INSTALL_AND_RUN.md)。

### HML-003 批准主矩阵：传统分类与变量筛选

**设计与范围：** 7表855；α612点候选区0.1—1.5；分层物理ID互斥5外/2内/CARS2；split_seed20261031、model_seed42。Raw/SNV、缩放、变量选择与C在当前训练区拟合。13路径65外折、11115条OOF。

**实际结论：** MC20优于本矩阵普通全点SVM；提高MC次数、重复稳定筛选没有进一步增益。CARS-style为Python多分类适配，不是原MATLAB精度复现。

**论文定位：** 历史基线与负对照；可引用设计/训练内选择思路，分数仅当时混合ZY任务。

**限制：** 沿用本章列明的混合种类、单来源批次、历史曝光与少数类样本量限制。

**固定证据：** [docs/ZAOREN_APPROVED_EXECUTION_REPORT.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/ZAOREN_APPROVED_EXECUTION_REPORT.md)；[data/research/approved_zaoren_experiment_plan_20261031.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/data/research/approved_zaoren_experiment_plan_20261031.json)；[reports/approved-zaoren/2026-10-31/independent-audit/pooled_metrics.csv](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/approved-zaoren/2026-10-31/independent-audit/pooled_metrics.csv)。

### HML-004 批准矩阵中的模态与完整导出范围敏感性

**设计与范围：** 与HML-003同855/同折；n、n+α直接拼接；另固定全导出α2578点SVM/PLS-DA敏感性。不是另一批独立样本。

**实际结论：** 直接拼接未超过同类α单支。全导出高分不能证明该范围物理有效或不存在来源/制样混杂；负α及有效频段仍需仪器记录。

**论文定位：** 表示消融与高分敏感性诊断；不能以100%替换预定主区间。

**限制：** 沿用本章列明的混合种类、单来源批次、历史曝光与少数类样本量限制。 全频敏感性与候选区主矩阵不得混排名或据外层分数反选频段。

**固定证据：** [docs/ZAOREN_APPROVED_EXECUTION_REPORT.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/ZAOREN_APPROVED_EXECUTION_REPORT.md)；[reports/approved-zaoren/2026-10-31/experiments/modality_controls/summary.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/approved-zaoren/2026-10-31/experiments/modality_controls/summary.json)；[reports/approved-zaoren/2026-10-31/experiments/full_export_sensitivity/summary.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/approved-zaoren/2026-10-31/experiments/full_export_sensitivity/summary.json)。

### HML-005 固定三用途留出与首次1-shot分数校准

**设计与范围：** seed20261101固定513训练/171支持池/171query；513内3折选Raw/SNV、C，冻结全点SVM与MC20SVM。每类1份共7支持、20次抽样、λ=1；query始终171，不将未用support回填。

**实际结论：** SVM88.8889%→74.9415%，MC20 87.7193%→72.8947%；固定弱样本校正平均明显下降，不是默认改善步骤。

**论文定位：** 可写为同批次回顾性校准负结果，不是跨批次校准。

**限制：** 沿用本章列明的混合种类、单来源批次、历史曝光与少数类样本量限制。 七类3/5/10shot不可行：ZY02支持池与query各仅1份。

**固定证据：** [docs/ZAOREN_THREEWAY_REPORT.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/ZAOREN_THREEWAY_REPORT.md)；[data/research/zaoren_threeway_protocol_20261031.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/data/research/zaoren_threeway_protocol_20261031.json)；[reports/zaoren-threeway/2026-10-31/summary.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/zaoren-threeway/2026-10-31/summary.json)。

### HML-006 116配置校准搜索与六类辅助K-shot

**设计与范围：** 原513/171/171不动；30源内开发任务=24选择+6门禁；候选必须优于源模型及匹配零支持控制；七类1shot，另六类辅助1/3/5/10shot；100次支持抽样。

**实际结论：** 七类正式主方案均回退不调整。强正则弱校正SVM89.0994%(+0.2105pp)仅预选诊断，开发增益未过规则，不能事后改主方法；六类10shot开发通过后最终仍微降0.0353pp。

**论文定位：** 保留全部正负结果、matched-zero结构控制与开发/最终差异。

**限制：** 沿用本章列明的混合种类、单来源批次、历史曝光与少数类样本量限制。 六类辅助排除ZY02不替代七类主任务；重复support情景不是独立批次。

**固定证据：** [docs/CALIBRATION_OPTIMIZATION_REPORT.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/CALIBRATION_OPTIMIZATION_REPORT.md)；[data/research/calibration_optimization_protocol_20261031.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/data/research/calibration_optimization_protocol_20261031.json)；[reports/calibration-optimization/2026-10-31/summary.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/calibration-optimization/2026-10-31/summary.json)。

### HML-007 54源表示/分类器搜索

**设计与范围：** 原513训练内30任务，24选择/6检验，6预处理×9分类设置；最终171不选参。1620次开发源拟合。

**实际结论：** 开发选出的LDA最终未胜SVM；较高平衡准确率不等于所有指标改善。

**论文定位：** 表示/读出模型对照；不得只挑有利指标。

**限制：** 沿用本章列明的混合种类、单来源批次、历史曝光与少数类样本量限制。

**固定证据：** [docs/REPRESENTATION_AND_LATENT_REPORT.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/REPRESENTATION_AND_LATENT_REPORT.md)；[data/research/representation_calibration_protocol_20261031.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/data/research/representation_calibration_protocol_20261031.json)；[reports/representation-optimization/2026-10-31/source-search/source_metrics.csv](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/representation-optimization/2026-10-31/source-search/source_metrics.csv)。

### HML-008 60潜空间适配及全范围敏感性

**设计与范围：** 3个锁定基础源×60适配，源拟合投影/尺度/头/协方差不动；30任务×20draws=108000开发配置评价；最终171×100支持抽样。另6条全导出敏感性不使用支持标签。

**实际结论：** 三个主适配均回退不调整；PLS局部记忆给支持标签反而低于相同零支持结构，不能将读出变化当校准收益。

**论文定位：** 匹配零支持消融和负结果；100%另列敏感性。

**限制：** 沿用本章列明的混合种类、单来源批次、历史曝光与少数类样本量限制。 部分四舍五入来自报告表；精确全表见固定源JSON。

**固定证据：** [docs/REPRESENTATION_AND_LATENT_REPORT.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/REPRESENTATION_AND_LATENT_REPORT.md)；[reports/representation-optimization/2026-10-31/latent-search/summary.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/representation-optimization/2026-10-31/latent-search/summary.json)。

### HML-009 论文启发的监督ResNet/ProtoNet/FO-MAML分类适配

**设计与范围：** 紧凑16/32通道骨干，先40epoch监督，再相同100源任务额外更新；31预设候选，6源开发任务4选/2检，seed42/43/44；每seed100次1shot，共用171query。来源论文为回归，本实现为分类适配。

**实际结论：** FO-MAML自身64.7173%→67.5458%，宏F1约.3922→.4338，仍低于88.8889%SVM；头微调/全网微调/ProtoNet未通过开发门禁，诊断负结果保留。

**论文定位：** 可作为弱神经起点的探索性自身收益，不是强主模型优势或原论文严格复现。

**限制：** 沿用本章列明的混合种类、单来源批次、历史曝光与少数类样本量限制。 源meta任务3类1support/1query；样品记录互斥不等于源与目标类别词表不同。

**固定证据：** [docs/PAPER_FEWSHOT_CLASSIFICATION_REPORT.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/PAPER_FEWSHOT_CLASSIFICATION_REPORT.md)；[data/research/paper_neural_classification_protocol_20261031.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/data/research/paper_neural_classification_protocol_20261031.json)；[reports/paper-fewshot-classification/2026-10-31/summary.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/paper-fewshot-classification/2026-10-31/summary.json)。

### HML-010 四轮85候选：强MLP、支持加权SVM、回放锚定与元训练控制

**设计与范围：** 24神经基础+21支持加权SVM+24回放/锚定+16元训练配置；全部4锁统一放行后才评分171；配对源监督与FO共享任务和更新次数，3seeds。

**实际结论：** 正式90.2534%来自零目标支持的源训练；同任务监督继续训练逐样品预测与FO相同，未显示MAML特有增益。500更新高分仅预选诊断，校准反降。

**论文定位：** 可报告基础训练改善和校准失败；不能写“7样本校准达到90.25%”。

**限制：** 沿用本章列明的混合种类、单来源批次、历史曝光与少数类样本量限制。 额外更新不是额外epoch；神经/元策略未作FLOPs严格匹配。

**固定证据：** [docs/MULTIROUND_OPTIMIZATION_REPORT.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/MULTIROUND_OPTIMIZATION_REPORT.md)；[data/research/multiround_optimization_plan_20261031.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/data/research/multiround_optimization_plan_20261031.json)；[reports/multiround-optimization/2026-10-31/meta-refinement/summary.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/multiround-optimization/2026-10-31/meta-refinement/summary.json)。

### HML-011 图片引导特征/分类40设置

**设计与范围：** SNV/MSC、CARS/MC-UVE/噪声UVE/SPA、LDA/PCA-LDA，源内拟合；513/171/171不变，0目标支持标签，3seeds、开发锁先于query。

**实际结论：** LDA158/171=92.3977%但宏F1.8352<原SVM.8616及强MLP.8787；变量筛选候选没有自动提准。

**论文定位：** 多指标冲突与稀少类假阳性负对照；不可只报最高accuracy。

**限制：** 沿用本章列明的混合种类、单来源批次、历史曝光与少数类样本量限制。 LDA对ZY02召回1/1但精确率1/5；不能称全部类别改善。

**固定证据：** [docs/IMAGE_GUIDED_OPTIMIZATION_REPORT.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/IMAGE_GUIDED_OPTIMIZATION_REPORT.md)；[data/research/image_guided_plan_20261031.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/data/research/image_guided_plan_20261031.json)；[reports/image-guided-optimization/2026-10-31/features/summary.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/image-guided-optimization/2026-10-31/features/summary.json)。

### HML-012 图片引导概率/判别21设置

**设计与范围：** 源内部3折OOF分数拟合temperature/类别bias；训练/校准分开；不访问171query作温度或判别选择，0目标支持。

**实际结论：** SVM温度缩放NLL .541355→.409750、Brier .264373→.199344、ECE .229348→.093669，准确率保持88.8889%；硬分类改动未过开发。

**论文定位：** 概率可靠性与分类准确率分开报告，不把temperature解释为提高分类。

**限制：** 沿用本章列明的混合种类、单来源批次、历史曝光与少数类样本量限制。 SVMmargin不是概率，softmax未校准参照须标明；仅当前同批数据可靠性。

**固定证据：** [docs/IMAGE_GUIDED_OPTIMIZATION_REPORT.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/IMAGE_GUIDED_OPTIMIZATION_REPORT.md)；[reports/image-guided-optimization/2026-10-31/probability/summary.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/image-guided-optimization/2026-10-31/probability/summary.json)。

### HML-013 图片引导SVM/MLP融合20设置

**设计与范围：** 源内选权重/温度，所有lock先于query评分；0目标支持，3seeds。

**实际结论：** 正式回退SVM权重0的较强单MLP，不能称融合收益；真值oracle91.8129%不可部署，不是实测模型成绩。

**论文定位：** 融合负结果和错误互补诊断；oracle只用于解释可能空间。

**限制：** 沿用本章列明的混合种类、单来源批次、历史曝光与少数类样本量限制。

**固定证据：** [docs/IMAGE_GUIDED_OPTIMIZATION_REPORT.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/IMAGE_GUIDED_OPTIMIZATION_REPORT.md)；[reports/image-guided-optimization/2026-10-31/fusion/summary.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/image-guided-optimization/2026-10-31/fusion/summary.json)。

### HML-014 R0—R3嵌套优化与90.41%精确重现

**设计与范围：** 855每方法一次5外折OOF，2内折，源内每折独立选择：R1 MC36、R2 SVM96、R3预处理7；所有内层锁齐才统一外层放行；原标准SVC投票不换argmax。

**实际结论：** R2 821/85596.0234%、宏F1.9196846、BA89.9184%；更复杂R3反降5个正确。R1相对固定锚定与R0的对比含义不同，不能全归因单个因素。

**论文定位：** 历史混合匿名组可重放强候选及方法选择过程；不得转成当前纯酸枣仁主成绩。

**限制：** 沿用本章列明的混合种类、单来源批次、历史曝光与少数类样本量限制。

**固定证据：** [docs/STAGED_NESTED_OPTIMIZATION_REPORT.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/STAGED_NESTED_OPTIMIZATION_REPORT.md)；[data/research/staged_nested_plan_20261031.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/data/research/staged_nested_plan_20261031.json)；[reports/paper-c/2026-10-31/history-audit/all_method_metrics.csv](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/paper-c/2026-10-31/history-audit/all_method_metrics.csv)。

### HML-015 R4组合家族、稳定选择与CNN受控对照

**设计与范围：** 继承对应外折R1—3，内层选7家族，另保留各家族诊断；CNN15个新编码器200epoch/9223参数/32维表示，与SVM共享编码器。

**实际结论：** R4内选策略94.9708%低于R3和R2；预算MC、稳定MC/CARS、CNN路线均未胜较强传统路线。

**论文定位：** 完整负结果与复杂度/选择方差讨论；不能推断所有CNN、SG或稳定方法普遍无效。

**限制：** 沿用本章列明的混合种类、单来源批次、历史曝光与少数类样本量限制。

**固定证据：** [docs/STAGED_NESTED_OPTIMIZATION_REPORT.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/STAGED_NESTED_OPTIMIZATION_REPORT.md)；[data/research/staged_nested_combinations_protocol_20261031.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/data/research/staged_nested_combinations_protocol_20261031.json)；[reports/staged-nested-optimization/2026-10-31/combinations/summary.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/staged-nested-optimization/2026-10-31/combinations/summary.json)。

### HML-016 统一855协议基线复现与条件消融

**设计与范围：** 同855同外/内折：SVM96、LDA10、sklearnMLP8候选内调；R2已选参数单独重放/重拟合；条件消融继承R2其他设置，非重新联合调参。

**实际结论：** R2精确重现；全谱SVM88.0702%、LDA91.9298%、单隐层MLP88.7719%；去MC、随机同K、去C/gamma搜索、去类权等条件消融均下降。

**论文定位：** 可重放的历史对照和贡献分解；不同模型网格/成本不同，不称所有方法严格等预算。

**限制：** 沿用本章列明的混合种类、单来源批次、历史曝光与少数类样本量限制。 条件消融不严格分离参数交互因果；F1和BA不得由accuracy推算。

**固定证据：** [docs/RESEARCH4_EXPERIMENT_AUDIT.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/RESEARCH4_EXPERIMENT_AUDIT.md)；[reports/paper-c/2026-10-31/fair-benchmark/model_comparison.csv](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/paper-c/2026-10-31/fair-benchmark/model_comparison.csv)；[reports/paper-c/2026-10-31/final-audit/same855_model_metrics.csv](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/paper-c/2026-10-31/final-audit/same855_model_metrics.csv)。

### HML-017 原64—32强MLP的三seed统一OOF与源更新控制

**设计与范围：** 精确复用原实现200epoch，再加100次源监督更新，不是300epoch；同855旧5/2折，各42/43/44均保存。

**实际结论：** 追加训练的三seed均值略升，seed43反降；未超过R2，不能只挑seed42多9份作为所有种子改善。

**论文定位：** 神经强对照和种子敏感性；模型架构本来自旧513开发，历史选择曝光需明列。

**限制：** 沿用本章列明的混合种类、单来源批次、历史曝光与少数类样本量限制。

**固定证据：** [docs/RESEARCH4_EXPERIMENT_AUDIT.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/RESEARCH4_EXPERIMENT_AUDIT.md)；[reports/paper-c/2026-10-31/strong-mlp/continuation_paired_summary.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/paper-c/2026-10-31/strong-mlp/continuation_paired_summary.json)。

### HML-018 统一研究审计中的保守校准独立复核

**设计与范围：** 仍为513/171/171，源内30开发任务和新100支持抽样seed20261104；冻结源、限幅校准与匹配零支持控制。

**实际结论：** v1限幅使满足翻类必要条件的11120对实际均不翻类；正式回退无增益，不能作为提准新方法。旧λ1负结果保留，抽样均值不覆盖先前20draw。

**论文定位：** 保守规则的机制负诊断，区分“稳定不坏”与“带来新收益”。

**限制：** 沿用本章列明的混合种类、单来源批次、历史曝光与少数类样本量限制。

**固定证据：** [docs/RESEARCH4_EXPERIMENT_AUDIT.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/RESEARCH4_EXPERIMENT_AUDIT.md)；[data/research/paper_c_calibration_protocol_20261031.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/data/research/paper_c_calibration_protocol_20261031.json)；[reports/paper-c/2026-10-31/calibration/run/summary.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/paper-c/2026-10-31/calibration/run/summary.json)。

### HML-019 多表示/深度学习升级：8家族小预算嵌套实测

**设计与范围：** 同855/5外2内/seed42、43、44；8家族×2事前耦合profile，12/24epoch；源内Macro-F1主选，所有学习操作记当前source，外层独立评分。

**实际结论：** 四表示主策略seed42 85.0292%、三seed均值83.7037%，未超过R2；12/24epoch小预算不能否定所有深度网络，多表示profile耦合不能称纯因素融合消融。

**论文定位：** 神经/融合辅助对照与真实负结果；不能把同批重复模态当新增独立样品。

**限制：** 沿用本章列明的混合种类、单来源批次、历史曝光与少数类样本量限制。 本模块只涵盖历史855主任务；HQ/TM辅助、补充多表示与外部迁移见本汇编外部/补充模块。

**固定证据：** [docs/MULTIREPRESENTATION_UPGRADE_REPORT.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/MULTIREPRESENTATION_UPGRADE_REPORT.md)；[data/research/upgrade_neural_protocol_20261031.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/data/research/upgrade_neural_protocol_20261031.json)；[reports/upgrade-20261031-v1/neural/model_comparison.csv](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/upgrade-20261031-v1/neural/model_comparison.csv)。

### HML-020 480条历史训练损失曲线只读诊断

**设计与范围：** 只读取8家族×2profile×5外×2内×3seed=480个源training.json；逐tar/member SHA核对，导出8640epoch记录，不读外层score/truth，不新增fit。

**实际结论：** 全部配置末段训练loss仍下降，仅说明该损失定义下优化仍有进展；缺逐epoch验证曲线，不能据此判定欠拟合或延长训练必提准。

**论文定位：** 学习过程诊断，不能当新外层模型成绩。

**限制：** 沿用本章列明的混合种类、单来源批次、历史曝光与少数类样本量限制。 源loss-only读取与随后真实96epoch pilot分开登记。

**固定证据：** [docs/DEPLOYMENT_NEURAL_DIAGNOSTIC_20261031.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/DEPLOYMENT_NEURAL_DIAGNOSTIC_20261031.md)；[reports/zaoren-neural-source-diagnostic-20261031-v1/group_summary.csv](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/zaoren-neural-source-diagnostic-20261031-v1/group_summary.csv)。

### HML-021 仅源内的96epoch学习曲线pilot

**设计与范围：** 固定原outer00.inner00：342source/342inner-validation，未访问外层171；αCNN/四表示×3seed六条fresh96epoch轨迹，预定12/24/48/96四快照，所有学习统计仅342source。

**实际结论：** 24→96轮αCNN内验证64.9123→75.9259%，四表示76.6082→83.8207%；四表示BA48→96却74.9320→69.4388。优化进展和少数类/泛化单调改善不是一回事。

**论文定位：** 源内学习曲线及负变化；不能和855OOF96.02%直接相减，也未据此改部署。

**限制：** 沿用本章列明的混合种类、单来源批次、历史曝光与少数类样本量限制。 684条仍是已曝光开发记录，仅单个源内上下文，不是新确认队列。

**固定证据：** [reports/zaoren-neural-learning-pilot-20261031-v1/README.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/zaoren-neural-learning-pilot-20261031-v1/README.md)；[reports/zaoren-neural-learning-pilot-20261031-v1/protocol.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/zaoren-neural-learning-pilot-20261031-v1/protocol.json)；[reports/zaoren-neural-learning-pilot-20261031-v1/checkpoint_seed_summary.csv](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/zaoren-neural-learning-pilot-20261031-v1/checkpoint_seed_summary.csv)。

### HML-022 匿名七分类完整部署与未来输入工具

**设计与范围：** 全部855视为已曝光开发数据；五个历史R2源内配置×5旧开发折×3seed共75候选拟合；按15case源内宏F1等选一个配置，再全855拟合seed42/43/44三模型。未来预测拒已知ID/旧谱重命名，真值独立评分。

**实际结论：** 工具和完整模型已交付；最终MC50/PLS1/20%=123点，SVM C1000/gamma0.1×scale/balanced。真新样品准确率未测，不能给部署模型填历史96.02%。

**论文定位：** 工程应用入口与可复现性附件，不替代未见样本验证；匿名类号不自动映射真实物种/产地。

**限制：** 沿用本章列明的混合种类、单来源批次、历史曝光与少数类样本量限制。 同855全源训练后的回代诊断accuracy=null；三个seed只是分歧参考，不是已验证集成。

**固定证据：** [docs/ZAOREN_ANONYMOUS_DEPLOYMENT.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/ZAOREN_ANONYMOUS_DEPLOYMENT.md)；[reports/zaoren-deployment-20261031-v1/deployment/model_seed42.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/zaoren-deployment-20261031-v1/deployment/model_seed42.json)；[reports/zaoren-deployment-20261031-v1/deployment/bundle_manifest.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/zaoren-deployment-20261031-v1/deployment/bundle_manifest.json)；[reports/zaoren-deployment-20261031-v1/feature-stability/summary.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/zaoren-deployment-20261031-v1/feature-stability/summary.json)。

## 论文写作时可用与不可转述的边界

- 可用：既有方法的透明实现、训练内选择、同口径比较、条件消融、少数类错误分析、真实负结果和可重放工程证据。
- 不能直接写成：已验证纯酸枣仁产地准确率96.02%、新批次小样本校准有效、所有深度学习均不适用、MAML特有机制获益、现成算法组合构成原创方法、部署模型的新样品准确率已知。
- 全导出100%只保留为敏感性/混杂诊断；概率校准改善NLL/Brier/ECE不能当分类准确率提高；真值oracle不能当可部署模型。
- 所有历史正负结果保留。新种类纠正和新数据产生的结果必须另表，不改旧标签、旧分母、旧协议或旧结果以维持表面一致。

## 本次整理的验证范围

本次读取固定Git提交的协议、报告和小型机器指标表，生成逐来源SHA-256；核对R2的821/855及原始指标CSV，完整保留多个方案及负对照。未训练模型，未重算所有原始逐样品预测，未修改任何正在运行的科学程序或后台账本。旧报告所述独立审计次数是当时留下的证据，不表示本次又执行了对应全量审计。

完整机器台账、精确数值摘要及每个来源的固定分支/提交/路径/SHA-256：[`historical_ml.json`](../../evidence/paper_bundle_20261109/historical_ml.json)。
