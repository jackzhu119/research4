# B方案第三轮：正则化、标签信息与分类约束机制

基础提交 `ad21deecbb53371f9b4547646a2aa985d8f617fb`，来源分支 `research/b-mechanism-followup-20261110`；新增分支 `research/b-mechanism-round3-20261110`。结果写入 `reports/b_mechanism_round3_20261110/`，不覆盖前两轮工件，不合并main。

## 科学问题和授权

用户在第二轮完成后明确要求继续设计更多补充实验并让后台自行运行。本轮仍遵循其先前取消成绩、CPU、耗时自动中断阈值的要求，执行一次有限冻结矩阵；身份、角色、标签副本和哈希完整性继续检查。

第二轮观察到纯酸五来源强Anchor仍退化、弱校准接近Identity；完整重训在冻结Scaler时仍有大部分收益；类别权重改变未改变二分类预测。第三轮检验四个具体问题：正则化相对Support样本量及截距惩罚的影响；新增标签与无标签统计更新的贡献；已知人工标签污染的剂量效应；类别权重零变化是否与SVC约束未生效一致。

这是在看过前两轮结果之后设计的**内部探索**。冻结后不根据新Query成绩改变样品、方法、污染掩码、K、种子或指标；完整保存正、负和零结果，不从本轮多重比较中挑选确认性赢家。

## 数据和前序证据

当前有效清单18份工作簿、1,672份用户声明独立实物；种类为枣仁364、酸枣仁500、黄芪403、天麻405。固定角色999 TRAIN/334 CAL/339 PRED；本轮仅使用TRAIN，原CAL/PRED不进入拟合或选择。ZY01/02属于枣仁，ZY03—07属于酸枣仁。

先核验前两轮最终封存收据列出的全部工件、源模型和结果哈希。沿用与历史NPZ字节一致的612点alpha及原始坐标，按record_id对齐数值行。所有拟合仅接收数值谱，不把ID、标签或来源元数据作为特征。

G01—G03共同覆盖全部15个旧纯酸五来源上下文以及全部30个旧二分类比例B上下文（5来源×3种子×2覆盖），依设计覆盖选择而非依成绩选择。每一比较严格固定Source、Support候选、Query，逐physical_sample_id互斥。G04覆盖第二轮全部240个F03上下文，含种子42—49、三比例、两覆盖。

## G01：损失归一化 × 截距惩罚（45任务）

冻结源模型分数和Support池。draw0沿用原顺序，draw101/102沿用第二轮已锁定重排；K1/3/5/10每类嵌套，真标签K10仅保留draw0。每个上下文共10个合法支持回合。

在相同分数单位和相同ID上做2×2：sum-loss lambda=1 / mean-loss lambda=1（等价sum-loss lambda=N，N为总Support标签数），以及截距惩罚lambda / 截距不惩罚。斜率修正始终惩罚。原sum-loss且截距惩罚的10个回合直接复用第二轮Anchor，新增三个条件×10回合=30次校准拟合/任务，共1,350次。单独保留argmax Identity和原生SVC.predict的零支持对照。

使用标准锚定残差岭回归：`Z=[scores,1]`，拟合到`onehot-support_scores`；惩罚矩阵为斜率lambda、截距lambda或0。记录N、实际sum lambda、目标函数、设计秩、正则条件数、有效自由度`trace(Z (Z'Z+P)^(-1) Z')`、修正及跨类耦合范数。二分类另存a/b、排序、AUROC及阈值等价诊断。不按Query选择lambda。

## G02：新增标签信息与无标签统计量分解（45任务）

K1/3/5/10时，共享同一Source+Support谱值来拟合SNV之后的StandardScaler，所有条件冻结原源模型的实际数值gamma、C10、balanced权重。

比较三个适配臂及原始源模型：

- `unlabeled_statistics`：Scaler看Source+Support谱，SVC仅用Source谱与Source标签重新拟合，Support标签不进入分类器。
- `true_support`：相同Scaler、Source+Support真标签拟合，完全复用第二轮对应的固定gamma模型。
- `cyclic_support`：相同谱、Scaler和gamma，Source标签保留，仅将Support标签的副本按固定类别循环映射到下一类，保持标签直方图。

不把新Scaler直接喂给旧冻结SVC作为合理适配；所有分类器在各自Scaler坐标下由对应训练样品拟合。每任务新增8个分类器，合计360次，旧真标签端点不重训。

原Support池是按已知类别分层锁定的，因此“无标签统计”只表示此臂不拟合Support标签值，不是从完全未知类别的新采集池开展无监督新批次验证。Query谱及标签均不用于统计量估计。

真标签相对无标签统计臂同时增加了SVC损失项的样品数；固定每样品C时，总监督损失权重也随之增加。因此该差值称“增量监督训练收益”，不能唯一归因为抽象的标签信息。真标签与循环复制标签臂具有相同谱、样品数、Scaler、gamma和标签直方图，是更直接的标签正确性对照。

## G03：受控Support标签污染剂量（45任务）

固定K10完整Support池。每类10份中确定性选择1/2/4份，形成10%/20%/40%的**Support标签**污染，固定掩码种子201—205；同一掩码种子内1⊂2⊂4嵌套。选择样品循环改为下一类别，均衡每类改变数量，维持完整Support标签直方图。100%污染为全部10份/类，只保存单个canonical201端点。

每个剂量比较原Anchor lambda1、弱Anchor lambda1000/blend0.05与固定gamma完整重训；原0%校准和真标签模型复用。100%完整重训复用同上下文G02的`cyclic_support_k10`，不重复拟合。其余噪声目标按有序Support ID及复制标签向量精确去重，保留所有预设采样端点及其权重；掩码与去重在任何新Query评分之前冻结。本轮所有45个上下文的部分污染掩码均不重复，新增675次分类器和1,440次校准拟合。

每个任务分别保存匹配的argmax Identity和原生SVC.predict，避免接口变化被误认为标签收益。报告每份Support的原标签、复制标签、掩码ID、实际改变数、Support污染比例及全训练集污染比例。Source标签不改变。纯酸40% Support污染仅为20/240≈8.33%的全部已标记训练样品；二分类40%为8/260≈3.08%，不得混写为全训练集40%错误。

这种固定、均衡循环污染是人工机制对照，不是iid现场标注错误，不是发现现有标签有问题；不因此修改、删除或降权真实样品。

## G04：类别权重的分数、margin和有效C约束审计（240任务）

不拟合新模型，复用第二轮已保存的balanced/None原生及argmax四个预测端点，并从已封存模型计算Source侧目标诊断。对齐每个Source支持向量的双对偶系数绝对值alpha，与其有效`C_i=C×class_weight`比较，保存最大alpha/C、连续边界间隙、hinge slack、RKHS范数及原/对偶目标等数值描述。

记录两个模型的Scaler、gamma、支持向量ID、对齐双系数和截距是否相同，以及Query分数、margin、AUROC、排序和类别变化。若相同核和双系数且两组box约束均未生效，可解释这些冻结上下文对类别权重不敏感；不推广为类别权重普遍无效。

边界接近计数的rtol=1e-7、atol=1e-9仅是预设数值描述规则，保留连续量，不用于终止、筛选训练或判定生物学效果。Source训练内分数只用于目标审计，不用于“无泄漏”的参数选择。

## 总矩阵、统计与计算

共375任务包：G01=45、G02=45、G03=45、G04=240；5,820个评价端点、预期265,200条相关预测记录。新增分类器1,035次（360+675），新增校准拟合调用2,790次（1,350+1,440）。任务、拟合、端点、预测行与唯一实物分别计数，不把重复预测当新样本。

由保存预测独立复算Accuracy、Macro-F1、Balanced Accuracy、逐类Recall和混淆矩阵；报告配对纠错/误伤、正则化因子交互、真标签相对无标签统计量增量、污染曲线及来源分层。描述性95%区间以physical_sample_id聚类、自助重采样1,000次，保留同实物的相关预测，不以行数代替样本量。

固定数值库单线程、独立新成本账本；无成绩、CPU、墙钟或重试次数截止条件。每任务先写临时目录，完整后原子封存并校验结果/工件哈希；恢复不重复训练或重复计数完整任务。G03依赖同上下文G02封存的全污染模型，调度先G02后G03。失败尝试保存依据，继续其余独立任务；不安全的输入/角色/哈希恢复拒绝执行。

先做独立测试和四家族各一项真实pilot，pilot完整工件进入全队列复用，成绩不作为推进门槛。随后独立监督器运行、自动汇总、独立指标复算、核验、限定新文件提交及研究分支推送；聊天结束后继续。云实例暂停或销毁时无法持续计算，恢复后通过完整工件续跑。

## 未识别机制与新数据

来源仍与批次混杂，只有五个酸枣仁来源，真实batch_id缺失保持缺失。历史样品已曝光、研究方向受前序结果启发，多重比较和长期开发偏倚仍存在；不能写作独立确认、真实跨批次泛化或新生物学机制。

真实新批次评价继续`NOT_RUN_NEEDS_NEW_BATCH_DATA`；沿用已有真实批次模板、导入验证器和采样起点，本轮不虚构采集或批次字段。二分类与五分类的样品、类别及标签预算不同，不直接相减准确率解释类别数作用。

```bash
python scripts/run_b_mechanism_round3_20261110.py --dry-run
python scripts/run_b_mechanism_round3_20261110.py --prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p test_b_mechanism_round3_20261110.py -v
python scripts/run_b_mechanism_round3_20261110.py --pilot
python scripts/background_b_mechanism_round3_20261110.py --launch
python scripts/background_b_mechanism_round3_20261110.py --status
python scripts/run_b_mechanism_round3_20261110.py --resume
python scripts/summarize_b_mechanism_round3_20261110.py --resume
python scripts/summarize_b_mechanism_round3_20261110.py --verify
python scripts/run_b_mechanism_round3_20261110.py --verify
```
