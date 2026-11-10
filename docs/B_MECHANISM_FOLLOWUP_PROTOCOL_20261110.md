# B方案第二轮机制补充：冻结协议

基础提交 `76966534df314e5bcb108a1fdccd611d1ad52910`，来源分支 `research/b-mechanism-gaps-20261110`；新增分支 `research/b-mechanism-followup-20261110`。原105任务的协议、标签、模型、结果和main均不修改，不自动合并。

## 问题及授权

用户在阅读第一轮结果后要求“进一步测试，多设计一点，多得到一些结果，依旧是后台自己运行”。沿用其先前后台自主推进、不因准确率/耗时/CPU阈值自动中断的要求。本轮执行有限、事先冻结的完整矩阵；不根据Query成绩增加种子、筛选方法或修改指标。科学一致性与哈希检查继续保留。

第一轮五来源固定SVM源模型平均Accuracy为80.00%，K10完整重训83.89%，Anchor为78.78%；二分类源模型98.44%，来源覆盖效应很小。第二轮分别检查校准自由度/分数单位、支持抽样、重训时训练统计量及gamma、类别权重与学习样本量，保留正、负、零效应。

## 数据与共同规则

只使用当前有效登记表和固定 master TRAIN。1,672份用户声明独立样品、999 TRAIN/334 CAL/339 PRED，种类数量及ZY01/02枣仁、ZY03—07酸枣仁语义沿用现行清单。原CAL/PRED不进入本轮拟合或选择。

首先对第一轮最终封存收据列出的全部工件核验SHA-256；每次模型复用再次检查其任务和直接工件哈希。重用字节完全一致的alpha612缓存，并按record_id对齐。固定SNV(ddof=1)、训练侧StandardScaler、C10 RBF-SVM；除明确的类别权重、Scaler、gamma对照外不新增搜索。Source/Support/Query按physical_sample_id互斥；同一比较使用匹配查询。

F01/F02选择全部15个旧E03上下文，以及全部30个旧E01比例B上下文（五目标来源×三种子×两种覆盖）。这是按设计覆盖选择，未依据其结果挑选上下文。

## F01：校准自由度、分数单位与支持样品敏感性（45任务）

直接复用冻结源模型。锁定原每类10份Support候选池；canonical draw0沿用原顺序，新重排种子101—110在每类候选池内确定性重排，K1/3/5/10嵌套抽样，所有方法同ID。

比较原分数Anchor lambda1、Bias-only、弱Anchor lambda1000/blend0.05、每类仅自身分数斜率/截距的对角Anchor lambda1、分数全局乘0.1/10后的Anchor，以及逐样品分数行中心化后用Source分数全局RMS归一化的Anchor。归一化参数只由Source产生，逐行中心化与正比例缩放保持零支持argmax；单独保留Identity与SVC.predict对照。

乱标控制仅对Support标签的显式副本作确定性置乱，维持标签直方图，不改变样品ID、元数据或历史真值；记录实际改变标签数，不为保证退化而重抽乱标。

K10真标签使用同一个完整候选池，因此只保留canonical draw0，其他重排不作为新实验。K10乱标副本仍单列各固定随机控制。每任务284个端点，其中12个canonical原校准直接复用；新增270次校准拟合，45任务合计12,150次，新增光谱分类器拟合为0。

12,150指实际校准拟合调用，不是12,150项独立实验。随机支持抽样可能恰好选到相同ID，乱标控制也可能恰好保持标签不变；按预设抽样保留其权重与实际改变标签数，不为增加差异重抽，不将这些重复当成新实物证据。

保存支持ID、分数变换、校准参数、设计矩阵秩与正则条件数、修正范数、跨类别耦合、决策改变、二分类仿射a/b、排序/AUROC和阈值等价核验。参数秩或条件数用于解释，不作为自动停止或筛选成绩的阈值。

## F02：完整重训的Scaler与gamma配对分解（45任务）

每个上下文固定Source、Support池、Query，K1/3/5/10。对同一Source+K Support做2×2对照：Scaler冻结原Source统计量/在Source+Support重新拟合；gamma冻结原模型实际数值/在当前训练矩阵按`scale`重新计算。C10、类别权重balanced和其他设置固定。

仅冻结Scaler却仍用`gamma='scale'`会同时改变gamma，不能单独归因于统计量。本实验显式记录有效gamma和Scaler哈希并提供差中之差。

原“重拟合Scaler+重新计算gamma”就是已保存的完整重训模型，180个端点直接复用。额外审计发现172/180个“重拟合Scaler+冻结gamma”端点的有效数值gamma也与该旧模型**精确相等**，这些端点继续复用；只有8个例外按不同精确参数拟合。浮点ULP差异不解释为生物学机制，不使用容差、准确率或预测相似性决定复用。每个任务的精确复用K清单冻结在JSON。原计划540次新分类器拟合减去172次重复，实际新增368次。

冻结Scaler下重新计算gamma的180个条件均未与原gamma精确相等；仍执行完整配对对照。所有新统计量仅在Source+当前Support上拟合，不读取Query标签或Query分布。

## F03：类别权重与来源覆盖稳定性（240任务）

五目标酸枣仁来源、种子42—49、三比例A/B/C、两覆盖条件，共240上下文。42—44完全继承旧名单；45—49为新增、预先固定的抽样。每上下文源样本总数240，查询20+20、候选支持10+10，来源覆盖以独立目标20份等量替换，类别比例沿用60/180、120/120、180/60。

每上下文比较`class_weight=balanced`与None，报告原生SVC.predict和分数argmax诊断。旧90个balanced模型复用。比例B的120/120使balanced的两个实际权重均为1，与None为同一有效训练目标；80个B上下文共用同一模型，同时保留两个具明确等价依据的评价端点，不再重复拟合。新增分类器310次。

报告Accuracy、Macro-F1、Balanced Accuracy、逐类Recall、混淆矩阵、margin、支持向量数；分析权重×类别比例×来源覆盖和各来源的配对转移。新旧种子分层展示，避免把重复抽样当成新实物或外部确认。

## F04：五来源标签预算学习曲线（15任务）

保持原15个E03外折的60份Query不变。从原每类38份Source的固定顺序取嵌套均衡前缀：Source50/100/150/190，最后240使用原Source190+Support50。未根据难度或成绩挑选新增标签。

Source190源模型及Source240旧K10完整重训模型直接复用；每任务只新增Source50/100/150三个模型，共45次。明确比较的是标签预算变化，240条件已经使用50份候选支持标签，不能称零新增标签收益或新独立样品。

## 矩阵、计数及统计

共345任务包：F01=45、F02=45、F03=240、F04=15；14,580个评价端点。经有效目标精确去重后，新增分类器拟合723次（368+310+45），新增校准拟合12,150次。任务/端点/拟合/预测行/唯一实物分别计数，复用不增加训练次数。

所有配对变化先按任务、方法、K、draw及实物ID对齐。描述性95%区间以physical_sample_id聚类，自助抽样1,000次并保留该实物的重复预测；同时保存方法对比、Recall代价、纠错/误伤、来源分层与非单调曲线。支持抽样的变异与查询实物不确定性分开报告，不把675,000条预期预测记录当独立样本。

这轮是在已看第一轮结果后设计的内部探索，不是事前确认性研究。现有五来源与真实批次混杂、缺失真实批次编号、历史样品曝光和长期开发选择仍限制解释。所有方法/尺度的完整结果保留，不在Query上选最优配置后声称普遍提升。

## 后台、恢复及交付

先运行独立测试与四任务pilot，每个家族各一项。pilot完整封存后由全队列复用，不以成绩决定是否推进。数值库单线程，无准确率、CPU、耗时或重试次数截止；单项失败留证、继续独立任务，完整工件不重复训练和计数。输入/源码/角色/工件完整性不匹配拒绝错误恢复。

独立会话监督器在聊天结束后继续运行，自动汇总、独立指标复算、工件核验、限定新文件提交及研究分支推送。云实例暂停或销毁无法持续计算，已有工件可恢复。旧分支、旧账本及main不修改、不合并。

输出在 `reports/b_mechanism_followup_20261110/`，中文结果文档为 `docs/B_MECHANISM_FOLLOWUP_RESULTS_20261110.md`。真实新批次依旧 `NOT_RUN_NEEDS_NEW_BATCH_DATA`，沿用已有采样模板与验证器，不虚构新采集。

```bash
python scripts/run_b_mechanism_followup_20261110.py --dry-run
python scripts/run_b_mechanism_followup_20261110.py --prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p test_b_mechanism_followup_20261110.py -v
python scripts/run_b_mechanism_followup_20261110.py --pilot
python scripts/background_b_mechanism_followup_20261110.py --launch
python scripts/background_b_mechanism_followup_20261110.py --status
python scripts/run_b_mechanism_followup_20261110.py --resume
python scripts/summarize_b_mechanism_followup_20261110.py --resume
python scripts/summarize_b_mechanism_followup_20261110.py --verify
python scripts/run_b_mechanism_followup_20261110.py --verify
```
