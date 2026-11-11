# B方案第五轮：Source内策略选择、Support获取与分数几何

基础提交 `bbab5a9de3f3e79fe8b12dd5e06c29547e3b9778`，基础分支 `research/b-mechanism-round4-20261111`；新增分支 `research/b-mechanism-round5-20261111`。新结果保存在 `reports/b_mechanism_round5_20261111/`，中文结果文档为 `docs/B_MECHANISM_ROUND5_RESULTS_20261111.md`。历史工件不覆盖，不自动合并main。

## 授权和问题

用户要求“继续扩展训练，设计下一轮”，沿用已经明确的后台方式：完成预检、独立测试与pilot后启动脱离聊天的监督器，确认进度增长后结束聊天；不因准确率、CPU、耗时或重试次数阈值提前终止固定队列。输入/角色/协议/源码哈希不一致仍拒绝错误恢复。

第四轮说明新增样品收益不能由增加Source损失质量单独解释；低C下权重开始影响部分决策；弱校准的大部分修正不足以跨过margin；稀缺类加权可缓解部分退化，却不保证胜过Identity。第五轮从单个条件推进到预先定义的完整流程：仅用Source选择适配策略，标签盲法排列候选Support，比较OVO信息与OVR聚合，以及对高权重低ESS提供固定额外收缩。

所有候选方法、角色、获取排序、预算、复用条件和规则在本轮外层Query评分前冻结。实验依前序已曝光结果提出，仍是探索性内部开发，不将选择后的最好内层分数或多重比较正结果写成独立确认。

## 数据及共同限制

数据入口仍为有效登记表和固定角色清单：18份工作簿、1,672份用户声明独立实物；枣仁364、酸枣仁500、黄芪403、天麻405。只在999份master TRAIN中构造角色，原334 CAL/339 PRED不参与拟合或选择。ZY01/02为枣仁，ZY03—07为酸枣仁。

先对前四轮最终收据所列全部工件核验SHA-256。复用与历史字节一致的alpha612缓存及原始频率轴，按record_id对齐。固定45个上下文：全部15个纯酸五来源旧外折，以及全部30个旧二分类比例B上下文；这按设计覆盖决定，不按其准确率挑选。

同一任务的外层Source/Support/Query按physical_sample_id互斥，各方法共享角色和预算。每个Source二分类各120份，五来源各38份。查询实物数按实际工件统计，不能沿用包含更多种子的旧422分母。旧Support、Query和内部重复分割不构成新实物或新数据采集。

## I01：仅在Source内部选择适配策略（45任务）

固定三个内层留出种子501/502/503，在每个外层Source内按类独立排列并分配：五来源每类20 innerSource＋10 innerSupport候选＋8 innerValidation；二分类每类90＋10＋20。三种角色在每个内层回合互斥，全部属于外层Source，外层Support/Query不进入选择。

每个回合重新拟合SNV之后的StandardScaler与C10/RBF/gamma=scale源模型，不使用包含innerValidation的外层Source统计量。内层Support按每类K=1/3/5/10嵌套，比较四个固定候选：Identity、Bias lambda1、mean-Anchor（sum lambda=N）及冻结内层Source坐标/gamma的Source+Support完整分类器重训。

按每个K的三回合平均innerValidation Macro-F1选择；数值比较舍入12位，平局依次优先Identity、Bias、mean-Anchor、完整重训。不改变K、种子、参数或候选集。先保存选择决定，再评价/读取外层候选预测；外层Query值不参与策略选择。执行器可能预先物化Query特征，不能把该工程事实误写为从未读取其行。

最终外层Identity、Bias、mean-Anchor和冻结坐标重训是精确历史端点，直接复用；只增加Source内选择过程。每任务3个内层源模型＋12个内层重训模型=15次新分类器拟合，24次内层校准；共675次分类器、1,080次校准。每任务保存18个外层端点及完整内层角色、验证预测/指标、选择次数和成本。

内层训练量更少、验证来自Source域，其偏好未必迁移到目标Support域。重复留出相关，选择后的最好内验证均值有选择偏差；不能当无偏外部成绩。选择Identity的策略不需要拟合外层Support标签，实际使用量与分配预算分别记录。

## I02：固定总预算的标签盲法Support获取（45任务）

每个任务使用原10份/类候选池，二分类总20、五来源总50。五种预设排序：随机优先级601/602/603、最小Source top2 margin、不含候选真值的Source质心锚定max-min覆盖。预算为类别数L×1/3/5/10，即二分类2/6/10/20、五分类5/15/25/50，均取同一排序的嵌套前缀。

排序器只接收Source拟合坐标下的候选谱、Source分数、Source质心和不含来源语义的哈希键；不接收候选标签或Query。质心锚定max-min始终保留Source质心作为已有参考中心，逐步选择到已有中心/已选点最小距离最大的候选；它不是仅对已选候选点的无锚max-min。所有排序固定之后才揭示真实标签，保存实际类别覆盖和观察成本。

候选池最初按已知类别构造，因此只称“池内标签盲排名”，不称真实采样过程完全无需标签；候选ID前缀含来源语义，排序仅使用不可解释哈希，避免以字母前缀暗中排序。

每个获取条件比较：

- 冻结Source Scaler/实际gamma、C10、固定逐点权重1的Source+已选Support分类器重训。Source本来均衡；选后Support可能失衡，不重新估计balanced类别权重，避免同时改变原Source点的C_i。
- mean-Anchor（sum lambda=已选标签总数）；如果Support缺少任何Source类别，按预先指定规则回退Identity，不重抽直到满足覆盖，也不终止其他任务。用于判断覆盖的标签也已观察，不能把回退说成零标签成本。

模型拟合采用原候选池顺序过滤得到的规范ID顺序，排名顺序另存。完整池、恰好与历史标准K相同的集合以及不同策略得到的完全相同目标直接复用。前置计划已冻结实际集合和资格：本轮新增674次分类器和511次校准，其余端点为历史/本任务缓存或缺类回退。每任务42个端点；预算使用k=-1表示并非每类K相同，真实总量和逐类数量单列。

获取策略对比同时改变入选实物、类别覆盖和不确定性分布，不把差异唯一归因为多样性。Source margin是决策值启发式，不是已校准概率。这里的标签预算是对已有数据的回顾性模拟，不是新增采集或新标注数量。

## I03：保留输出基线的OVO信息校准（45任务）

输出始终锚定旧OVR源分数S0：`Snew=S0+[features,1]delta`，拟合Support残差`onehot-S0`。K0不改变原分数，不把改用OVO argmax的接口变化当校准收益。

比较三种只在Source拟合的残差输入：中心化并按Source平均行平方范数归一化的OVR；同样处理的原始OVO成对margin；Source-only OVO PCA保留min(L,P)维后进行相同归一化（L类别数，P成对数）。统一输入协方差trace质量与sum lambda=N，全部参数惩罚。PCA、中心和归一化均不读Support/Query统计量。

通过深拷贝SVC并使用公共decision_function提取OVO，不修改冻结源模型。仍用draw0/101/102、每类K1/3/5/10的10个合法回合，完整K10仅一次。原raw mean-Anchor端点精确复用，每任务新增3×10=30次校准，共1,350次，新增光谱分类器为0；Source特征基统计和PCA拟合另列，不计为新SVM。

二分类OVR、OVO及其完整一维PCA在此归一化下的Gram在实数算术中等价，实际求解并记录浮点差作为负对照，不把三个坐标实现称为独立新算法。五分类OVO保留更多成对置信信息，PCA维度对照用于区分信息与自由度；所有条件均报告，不按Query挑选表示。

## I04：同Support权重下的ESS额外收缩（45任务）

沿用第四轮H04的全部构成、draw和精确Support ID，类别均衡权重w=N/(L×n_class)，总权重N不变。比较既有weighted lambda1与两个固定条件：mean-loss lambda=N，及ESS启发式lambda=N²/ESS，其中ESS=(sum w)²/sum(w²)。

该规则是预先指定的收缩启发式，不是经Query选出的最优lambda或普遍有效的新算法。ESS只描述权重集中度，不等于独立实物数，也不解决来源/批次相关性。

均衡时所有权重1、ESS=N，两个新条件均精确等于历史mean-Anchor K5，直接复用。非均衡二分类新增24次/任务、五来源30次/任务，共1,170次校准，新增分类器0。保留历史lambda1、Identity及原生SVC对照，并保存修正幅度、决策移动、纠错/误伤，避免把退回Identity描述为获得适配效用。

任务中继承H04计划的旧成本字段仅供溯源；本轮成本以当前feasibility、执行器及封存结果为准。构成端点仍用k=-1及真实逐类标签数量，原始标签不改写。

## 完整规模、统计与后台

共180任务包，每组45个；6,840个外层评价端点，预期321,000条相关外层预测。Source内策略验证另有86,400条相关预测，不与外层查询混算。新增分类器1,349次（675内层策略＋674获取策略），新增校准拟合调用4,111次（1,080内层＋511获取＋1,350分数几何＋1,170ESS）。Source基统计/PCA拟合及所有复用另列。

由保存的逐实物真值/预测独立复算Accuracy、Macro-F1、Balanced Accuracy、逐类Recall及混淆矩阵；对齐相同Query计算纠错/误伤、策略选择频率、标签资格/覆盖、全部预算和方法差值。描述性95%区间按physical_sample_id聚类、固定1,000次重采样，保留同实物的全部相关预测，不把内层重复、策略和表示坐标当独立样品。

先独立测试和四组各一项真实pilot，pilot完整工件直接进入全队列复用，不按成绩决定是否继续。数值库单线程，独立新CPU账本；临时目录完整后原子封存，逐工件哈希，恢复不重复拟合或计数完整任务，失败尝试另存。无准确率、CPU、墙钟或重试次数的截止条件。

独立会话监督器完成训练、汇总、独立复算、核验、限定本轮文件提交和研究分支推送；确认后台进度增长后结束聊天。云实例暂停/销毁不能继续计算，恢复后从完整工件续跑。

## 解释限制与未运行项目

全部样品已有历史曝光，四轮结果影响了本轮问题选择；Source-only选择和标签盲排序只避免特定流程内泄漏，不清除长期开发/协议选择偏倚。当前来源与批次混杂，少量来源组不支持总体因果或真实跨批次主张。

Source、Support与Query只是当前TRAIN内的角色，原CAL/PRED未参与本轮搜索。中文种类/来源沿用用户确认，不等同独立植物鉴定。二分类和五分类的类别数、样品、训练量及标签总预算不同，准确率差不解释为类别数单因子效应。

真实新批次评价继续`NOT_RUN_NEEDS_NEW_BATCH_DATA`；沿用既有导入验证器、元数据模板和采样建议，不推断或补造batch_id、仪器或实测属性。

```bash
python scripts/run_b_mechanism_round5_20261111.py --dry-run
python scripts/run_b_mechanism_round5_20261111.py --prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p test_b_mechanism_round5_20261111.py -v
python scripts/run_b_mechanism_round5_20261111.py --pilot
python scripts/background_b_mechanism_round5_20261111.py --launch
python scripts/background_b_mechanism_round5_20261111.py --status
python scripts/run_b_mechanism_round5_20261111.py --resume
python scripts/summarize_b_mechanism_round5_20261111.py --resume
python scripts/summarize_b_mechanism_round5_20261111.py --verify
python scripts/run_b_mechanism_round5_20261111.py --verify
```
