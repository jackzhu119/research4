# 深层机制验证：完整对照与写作边界（v3）

本页是写作核对材料，范围为 `deep-mechanisms-v3`，不是已经验证的完整论文。971项科学任务已封存，16个预定汇总块完整。原114批回传与随后12个纯补传阶段均已通过真实远端核验，原13项待回传逻辑范围已全部覆盖；这些是已完成实验的上传恢复，没有重新训练。核验时远端提交为 `c8945acc0a57d75aae438fd9de81d05b422cc1f3`。本页及配套指标是之后整理的写作资料，提交与发布由ROOT统一处理，不把本页整理完成冒充本页已经发布。

所有表来自已保存的结果JSON／CSV。本文不重新训练、不改标签、不重新筛选最佳参数；百分比为准确率，`pp`为百分点。完整聚合、配对、随机种子分解及预设未做项见[配套指标CSV](deep_mechanisms_key_metrics.csv)，原始路径与SHA见[来源清单](deep_mechanisms_sources.json)。

## 1. 数据、评估单位与完成范围

- 本轮数值只使用原999份历史TRAIN：二分类516份（沿用标签的枣仁216、酸枣仁300）及黄芪241、天麻242；原334 CAL和339 PRED光谱未进入本轮。
- 经典模型使用原612个频率点（约0.100708–1.499176 THz），α／n分别作逐谱SNV后再拼接fusion；神经模型使用α在0.21–1.49 THz的128点插值、禁止外推。因此两条路径的差异不能归为纯网络架构差异。
- 本轮没有增加生物样本。编码器种子、目标训练种子、支持集抽样和同ID重复预测均为算法重复；唯一历史ID数也不自动等于经实验室审计的独立植物数。
- 陕西延安材料的植物身份未获得实物鉴定。这里保留当前种类标签，分类一致性不能确认其到底是枣仁还是酸枣仁；来源与批次混淆仍存在。
- 971是任务数，不是独立生物实验数。42个旧pure预训练产物经封存与来源核验后继承，不继承旧目标微调预测，不退还旧计算成本。

| 任务类型 | 已完成／预定 |
| --- | --- |
| source_scores | 15 / 15 |
| support_mechanisms | 60 / 60 |
| donor_mechanisms | 75 / 75 |
| aux_pretrain | 20 / 20 |
| target_pretrain | 50 / 50 |
| source_A_cross | 240 / 240 |
| target_eval_cross | 480 / 480 |
| bn_probe | 30 / 30 |
| summary | 1 / 1 |

三个汇总块仅核对辅助预训练、目标源预训练和A基线产物的来源覆盖，不为它们虚构独立分类成绩；其余13个指标块有独立index。

| 评估家族 | 汇总重复预测行／每设置 | 唯一历史查询ID | 重复来源 |
| --- | --- | --- | --- |
| source score | 1,350 | 510 | 5个held酸枣仁组，每组270查询 |
| 支持校准 | 13,500 | 255 | 5个来源任务 × 20支持draw × 固定135查询 |
| 供体因素主网格 | 6,750 | 510 | 5个来源任务 × 5供体draw × 固定270查询 |
| 供体单来源配对 | 5,400 | 450 | 该单来源可用的4个held任务 × 5draw × 270查询 |
| 神经主对照 | 12,384 | 516 | 同516个ID × 8编码器种子 × 3目标训练种子 |
| BN机制探针 | 1,548 | 516 | 同516个ID × 3编码器种子，目标种子固定101 |

二分类516 ID在神经任务中沿用已锁定的五个源／查询任务；支持和供体分析使用不同的查询子集，因此不能将这些表的总准确率直接排成统一算法榜。

## 2. 支持校准：完整K曲线与完整重训分开报告

每个来源任务固定6个非ZY01枣仁源样本及240个其他来源酸枣仁源样本；模型为沿用的逐谱SNV（ddof=1）＋仅源StandardScaler＋SVC（C=10，gamma=scale，balanced）。支持集与查询集不重叠，K是每类支持样本数。每个K单元为13,500次重复评估、255个唯一历史查询ID；不是13,500份样本。

完整重训重新拟合完整管线，使用支持标签的方式与源模型固定后的分数校准不同。它只预设K=1/5/20；K=3/10标为NOT_RUN，K=0没有新增重训。所有K和负结果保留，不能按查询成绩挑选“最佳K”作为事先确定的方法。

| 模态 | K／类 | 源预测 % | anchor λ1 % | bias-only λ1 % | 完整重训 % | 评估行／唯一ID |
| --- | --- | --- | --- | --- | --- | --- |
| α | 0 | 89.9259 | 89.9259 | 89.9259 | —（未新增拟合） | 13,500 / 255 |
| α | 1 | 89.9259 | 93.3333 | 93.3333 | 92.1556 | 13,500 / 255 |
| α | 3 | 89.9259 | 94.3556 | 94.3556 | NOT_RUN | 13,500 / 255 |
| α | 5 | 89.9259 | 94.2000 | 94.2000 | 97.0741 | 13,500 / 255 |
| α | 10 | 89.9259 | 94.4741 | 94.4741 | NOT_RUN | 13,500 / 255 |
| α | 20 | 89.9259 | 94.4519 | 94.4519 | 98.5185 | 13,500 / 255 |
| 折射率 n | 0 | 54.5185 | 54.5185 | 54.5185 | —（未新增拟合） | 13,500 / 255 |
| 折射率 n | 1 | 54.5185 | 67.6889 | 67.6889 | 62.6296 | 13,500 / 255 |
| 折射率 n | 3 | 54.5185 | 70.5778 | 70.5778 | NOT_RUN | 13,500 / 255 |
| 折射率 n | 5 | 54.5185 | 73.3704 | 73.3704 | 85.5037 | 13,500 / 255 |
| 折射率 n | 10 | 54.5185 | 74.1778 | 74.1778 | NOT_RUN | 13,500 / 255 |
| 折射率 n | 20 | 54.5185 | 74.0963 | 74.0963 | 94.5037 | 13,500 / 255 |
| α+n fusion | 0 | 68.2963 | 68.2963 | 68.2963 | —（未新增拟合） | 13,500 / 255 |
| α+n fusion | 1 | 68.2963 | 84.9037 | 84.9037 | 77.7481 | 13,500 / 255 |
| α+n fusion | 3 | 68.2963 | 90.9926 | 90.9926 | NOT_RUN | 13,500 / 255 |
| α+n fusion | 5 | 68.2963 | 93.7185 | 93.7185 | 95.4889 | 13,500 / 255 |
| α+n fusion | 10 | 68.2963 | 95.0889 | 95.0889 | NOT_RUN | 13,500 / 255 |
| α+n fusion | 20 | 68.2963 | 95.7037 | 95.7037 | 98.5185 | 13,500 / 255 |

### 2.1 已验证的阈值机制，以及实际出现的例外

对二分类源分数 `[-s,s]`，记校准后的两类分数差为 `a·s+b`。当每类K个支持、参数和偏置采用同一岭惩罚λ时，平衡标签和为0，正规方程给出 `b=−a·Σs/(2K+λ)`。当 `a≠0` 时，零分数差的位置为 `τ=Σs/(2K+λ)`；只有斜率 `a>0` 时，完整决策规则才与对应bias-only的正向分类规则等价。

- 三模态、K=1/3/5/10/20的15个anchor−bias配对均为两侧独有正确数0，每条件100个来源／draw单元。这里预测相同的结论依据逐样本二分类配对，不仅是平均准确率相等。
- 该条件式说明：保持正斜率时，平衡标签排列不会改变阈值位置；正分数单位变换也会同向缩放阈值。但标签和分数单位可以改变斜率符号，不能写成“标签完全无关”或“任意单位缩放都不影响预测”。
- **保留实际反例：** n模态、held酸枣仁组ZY05、draw52、K=1，将分数乘10后 `a=−0.44552732226069924`，方向反向。该135条查询中原anchor为96正确、乘10为39正确；全13,500条汇总从67.6889%降至67.2667%，净降0.4222 pp，其余99个来源／draw单元相同。阈值公式本身仍满足（残差约−2×10⁻¹⁵），失败的是正方向前提。原证据为 [support_n_rows_020.csv](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/science/summary-attempt001/support_n_rows_020.csv) 的threshold_mechanism及 [support_n_rows_022.csv](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/science/summary-attempt001/support_n_rows_022.csv) 的对应配对记录（job `support_ZY05_n_d50`，draw52）；其SHA同时保存在来源清单。
- K=5三模态的正单位缩放对照均预测一致；全K范围除上述n/K1/×10例外外，其余已保存的单位缩放配对一致。标签完全翻转对照可反转边界，属于必须报告的负结果。
- bias-only仍使用按真实标签构造的平衡支持集；不能改称“完全无标签校准”。均匀无分层居中使用另一组支持样本，并预设等类先验，不能与平衡采样规则混为同一实验。
- 不平衡支持、a≤0、与identity混合、不同偏置惩罚和数值并列不在上述简单等价命题范围内。

### 2.2 固定正则量与支持数的关系

固定λ=1时，阈值对支持均值的收缩因子为 `2K/(2K+1)`，随K变化。另设 `λ_K=K/5`，使λ／样本数恒定、收缩因子为10/11。该控制不能消除支持样本选择本身的随机性。

| K／类 | λ_K | 固定λ1的收缩因子 | 归一化λ α % | 归一化λ n % | 归一化λ fusion % | 评估行／唯一ID |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.2 | 0.6667 | 94.1556 | 73.7037 | 88.3259 | 13,500 / 255 |
| 3 | 0.6 | 0.8571 | 94.6296 | 71.7852 | 91.9556 | 13,500 / 255 |
| 5 | 1 | 0.9091 | 94.2000 | 73.3704 | 93.7185 | 13,500 / 255 |
| 10 | 2 | 0.9524 | 94.2074 | 72.8370 | 94.5111 | 13,500 / 255 |
| 20 | 4 | 0.9756 | 94.0593 | 72.3259 | 94.7778 | 13,500 / 255 |

固定λ的α、n曲线出现非单调；不能把某个K的下降直接解释为“多加标签有害”。极弱λ1000、blend=.05对照在本轮未改变源预测；完整CSV同时保留支持ERM阈值、置乱／翻转、单位缩放、源标准差归一化等所有设置。

## 3. 供体数量、缩放器、gamma与惩罚强度

固定保留6个非ZY01枣仁源样本，改变酸枣仁供体数n=6/24/60。每个n执行2（源子集／全246源缩放器）×2（子集scale／全源reference gamma）×2（C=10／归一化C）的网格。归一化 `C_n=10×246/(6+n)`，使balanced权重下每类总惩罚C·N/2保持基准值；n=6/24/60时C_n分别205、82、37.2727。n=240使用源producer中的共享完整基线，不额外重复八次拟合。

full缩放器／full-reference gamma即使分类器只用少量标签，仍使用全部246个允许源向量；不能把这项控制写成“整个算法只看12／30份数据”。

下表固定full源缩放器、full-reference gamma、C=10，展示预定的全部供体数量。每设置6,750次重复评估、510个唯一历史ID。

| 酸枣仁供体n | 分类器源总数 | α % | n % | fusion % | 评估行／唯一ID |
| --- | --- | --- | --- | --- | --- |
| 6 | 12 | 97.7778 | 64.6815 | 96.6222 | 6,750 / 510 |
| 24 | 30 | 94.7259 | 59.4667 | 87.1259 | 6,750 / 510 |
| 60 | 66 | 93.2148 | 57.5259 | 75.2000 | 6,750 / 510 |
| 240 | 246 | 91.2593 | 54.0741 | 69.2593 | 6,750 / 510 |

### 3.1 n=24的完整2×2×2因素切片

下表保留八个预设单元。全供体网格详见CSV；本表每格仍为6,750次重复评估、510个唯一历史ID。

| 源缩放器 | gamma | C | α % | n % | fusion % | 评估行／唯一ID |
| --- | --- | --- | --- | --- | --- | --- |
| full | full_reference | 10 | 94.7259 | 59.4667 | 87.1259 | 6,750 / 510 |
| full | full_reference | 82 | 94.7259 | 59.4667 | 87.1259 | 6,750 / 510 |
| full | subset_scale | 10 | 97.7185 | 59.3926 | 96.3852 | 6,750 / 510 |
| full | subset_scale | 82 | 97.7185 | 59.3926 | 96.3852 | 6,750 / 510 |
| subset | full_reference | 10 | 97.1704 | 60.0593 | 84.5926 | 6,750 / 510 |
| subset | full_reference | 82 | 97.1704 | 60.0593 | 84.5926 | 6,750 / 510 |
| subset | subset_scale | 10 | 97.1704 | 60.0593 | 84.5926 | 6,750 / 510 |
| subset | subset_scale | 82 | 97.1704 | 60.0593 | 84.5926 | 6,750 / 510 |

本轮全网格中，C归一化造成的准确率最大变化为α／fusion 0 pp、n 0.0889 pp；这只是此有限网格的结果，不表示任意C都无作用，也不能仅由准确率相等推出每个预测相等。fusion在n=24、full缩放器下改变gamma，87.1259%→96.3852%，差9.2593 pp，说明源表征与核几何对当前边界有实际影响。subset缩放器下两个gamma选择可能代数性接近／重合，不应当作完全独立的新机制。

### 3.2 同n=24的来源覆盖对照

将酸枣仁24个供体取自单一来源，或取自四个允许来源各6个；使用相同full源缩放器、full-reference gamma及归一化C。每个单来源只对其可用的四个held任务比较，配对是5,400行、450个唯一ID。下表是**同cohort的单来源减mixed配对差**，不能用单来源5,400行准确率直接减mixed全局6,750行准确率。

| 单一酸枣仁供体来源 | α差 pp | n差 pp | fusion差 pp | 配对行／唯一ID |
| --- | --- | --- | --- | --- |
| ZY03 | +2.5741 | +1.0000 | +6.3148 | 5,400 / 450 |
| ZY04 | +2.5556 | +6.3148 | +9.6111 | 5,400 / 450 |
| ZY05 | +0.6481 | -1.5185 | +2.7407 | 5,400 / 450 |
| ZY06 | +2.0926 | +0.5370 | +6.0000 | 5,400 / 450 |
| ZY07 | +2.3889 | +2.2222 | +8.3333 | 5,400 / 450 |

这些结果不支持“来源越多必然更好”，也不允许在查询结果出来后选一个最好的单来源作为预定方案。数量、源表示与覆盖效应仍受当前少数非ZY01枣仁、来源／批次混淆和已暴露数据范围约束。

## 4. 深度迁移：四种主条件完整报告

固定8个编码器／预训练种子42–49，与3个独立目标训练种子101/202/303交叉；先按每个组合合并原五个查询任务，再比较同516个ID。每个条件12,384次重复预测，只有516个唯一历史查询ID。

- **A：** 从头训练。
- **Aextra：** 仅目标任务源数据的额外来源预训练，再进行相同目标训练。
- **B：** 黄芪／天麻预训练，再进行相同目标训练。
- **Bshuffle：** 同B的数据与步骤，但在各植物内置乱预训练来源标签。
- 120/480是预训练优化步数；full更新整个编码器，frozen保留编码器及其native BN、只训目标头。各臂目标头初始化、数据顺序及dropout RNG按目标种子匹配，目标训练96 epochs。相等优化步数不等于相同样本曝光或FLOPs。
- A在120/480条件复用同一目标拟合的预测；相同A数字不是两次新的独立验证。

| 预训练步数／适配 | A % | Aextra % | B % | Bshuffle % | 评估行／唯一ID |
| --- | --- | --- | --- | --- | --- |
| 120 / full | 98.7888 | 98.8049 | 98.8776 | 98.8211 | 12,384 / 516 |
| 120 / frozen | 89.1069 | 96.3824 | 97.2384 | 89.7852 | 12,384 / 516 |
| 480 / full | 98.7888 | 98.7645 | 99.0391 | 98.7726 | 12,384 / 516 |
| 480 / frozen | 89.1069 | 94.9774 | 96.0271 | 92.1269 | 12,384 / 516 |

配套CSV保留macro-F1、balanced accuracy和正确数；各来源召回保存在对应原index中，不能只看总准确率忽略少数来源。下表按完整24种子组合报告B相对两个必要对照，正／零／负不是生物样本胜负次数。

| 条件 | 比较 | 平均差 pp | 24组合正／零／负 | 单组合差范围 pp | 评估行／唯一ID |
| --- | --- | --- | --- | --- | --- |
| 120/full | B MINUS Aextra | +0.0727 | 14/1/9 | -1.5504 ～ +0.9690 | 12,384 / 516 |
| 120/full | B MINUS Bshuffle | +0.0565 | 11/5/8 | -0.9690 ～ +1.1628 | 12,384 / 516 |
| 120/frozen | B MINUS Aextra | +0.8559 | 11/0/13 | -5.4264 ～ +7.1705 | 12,384 / 516 |
| 120/frozen | B MINUS Bshuffle | +7.4532 | 20/1/3 | -3.2946 ～ +47.0930 | 12,384 / 516 |
| 480/full | B MINUS Aextra | +0.2745 | 11/7/6 | -0.9690 ～ +1.7442 | 12,384 / 516 |
| 480/full | B MINUS Bshuffle | +0.2665 | 15/2/7 | -1.3566 ～ +1.7442 | 12,384 / 516 |
| 480/frozen | B MINUS Aextra | +1.0497 | 15/0/9 | -8.5271 ～ +6.9767 | 12,384 / 516 |
| 480/frozen | B MINUS Bshuffle | +3.9002 | 16/0/8 | -6.0078 ～ +34.8837 | 12,384 / 516 |

B在这些主对照的总体均值高于Aextra，但120/frozen有13／24组合为负，480/full也有负组合。可以报告“本历史数据设计下的探索性平均增益及其不稳定性”；不能改写为“已建立稳定跨物种正迁移”或“真实跨批次迁移有效”。

### 4.1 随机种子方差来源

对每个完整8×3矩阵，使用编码器主效应、目标训练主效应及剩余交互的描述性平方和分解；下表为其占总平方和的百分比。它不是因果方差归因、显著性检验或生物抽样置信区间。CSV另保留每个A/Aextra/B/Bshuffle的绝对准确率随机化分解。

| 条件 | 配对差 | 编码器份额 % | 目标种子份额 % | 交互份额 % | 组合／评估行／唯一ID |
| --- | --- | --- | --- | --- | --- |
| 120/full | B MINUS Aextra | 28.35 | 8.04 | 63.61 | 24组合；12,384 / 516 |
| 120/full | B MINUS Bshuffle | 26.37 | 5.02 | 68.62 | 24组合；12,384 / 516 |
| 120/frozen | B MINUS Aextra | 63.55 | 4.78 | 31.67 | 24组合；12,384 / 516 |
| 120/frozen | B MINUS Bshuffle | 40.38 | 23.73 | 35.89 | 24组合；12,384 / 516 |
| 480/full | B MINUS Aextra | 41.50 | 3.71 | 54.80 | 24组合；12,384 / 516 |
| 480/full | B MINUS Bshuffle | 42.97 | 8.24 | 48.79 | 24组合；12,384 / 516 |
| 480/frozen | B MINUS Aextra | 43.40 | 4.45 | 52.15 | 24组合；12,384 / 516 |
| 480/frozen | B MINUS Bshuffle | 74.87 | 4.44 | 20.69 | 24组合；12,384 / 516 |

公式为 `d_ij=μ+e_i+t_j+r_ij`；`SS_E=3Σe_i²`、`SS_T=8Σt_j²`、`SS_I=Σr_ij²`，三项和为总平方和。编码器种子包含初始化／预训练随机性，目标种子包含头初始化、顺序和dropout；因此不能把编码器份额直接写成“植物知识贡献”。

## 5. BN机制探针：只与同cohort native比较

这两项探针固定480步、frozen、编码器种子42/43/44、目标种子101。每臂1,548次重复预测、516个唯一ID，**不能直接拿它与上节8×3主块的均值相减**。

- neutral：BN running mean=0、variance=1、计数=0，保留仿射参数；对应A的中性状态可经完整身份检查后复用。
- source_recal：只用当前允许源数据作一次完整批BN重估，关闭dropout、不求梯度、momentum=1，BN重估阶段的编码器参数哈希不变；之后恢复eval／frozen，按同一预定96 epochs训练目标头。除经身份核验复用的neutral A外，干预后的目标头重新训练；这不是对同一个已训分类头只切换BN的一次前向比较。没有使用支持或查询数据更新BN。
- 下表native正确数由各探针的配对记录重建：`native_correct=intervention_correct−(left_only_correct−right_only_correct)`，两种策略重建一致，避免从24组合主均值借用不匹配基线。

| 臂 | 同cohort native | neutral | neutral差 pp | source_recal | 重估差 pp | 唯一ID |
| --- | --- | --- | --- | --- | --- | --- |
| A | 1425/1548 (92.0543%) | 1425/1548 (92.0543%) | +0.0000 | 1467/1548 (94.7674%) | +2.7132 | 516 |
| Aextra | 1487/1548 (96.0594%) | 1442/1548 (93.1525%) | -2.9070 | 1484/1548 (95.8656%) | -0.1938 | 516 |
| B | 1497/1548 (96.7054%) | 1473/1548 (95.1550%) | -1.5504 | 1497/1548 (96.7054%) | +0.0000 | 516 |
| Bshuffle | 1483/1548 (95.8010%) | 1472/1548 (95.0904%) | -0.7106 | 1473/1548 (95.1550%) | -0.6460 | 516 |

neutral下B比Bshuffle仅多1／1548正确（0.0646 pp）；source_recal使A净增42／1548（+2.7132 pp），B净正确数不变，但有3条由错变对、3条由对变错。说明BN统计能改变一部分结果，不能把它当作已经定位的唯一原因。两策略各60项pointwise／frozen检查通过；主full/frozen块各480项pointwise检查通过，frozen块亦记录480项冻结状态不变。

## 6. 可用于论文的结论，与不能使用的说法

| 可据实写 | 不能据此写 |
| --- | --- |
| 当前平衡支持、统一岭惩罚、正斜率的二分类校准表现为阈值移动，并有完整bias-only对照。 | 校准学习了新的植物化学机制；标签／分数单位永远无关。 |
| 支持标签完整重训与固定源分数校准是不同操作，应分别报告完整K曲线。 | 将完整重训成绩作为少样本分数校准成绩，或查询后挑K宣称预定最好方法。 |
| 当前供体构成下，缩放器、gamma和来源覆盖会明显影响边界。 | 更多数据普遍有害，或总惩罚C是唯一原因。 |
| B有探索性平均增益，需同时给Aextra、Bshuffle及所有种子正负结果。 | 已证明稳定跨物种／跨仪器／跨批次迁移有效。 |
| BN干预可改变部分预测，源重估改善A且对B总准确率净0。 | BN解释全部迁移收益，或BN干预净0说明每个预测未变。 |
| SNV对逐谱正仿射幅度变换不变，是方法性质和实现控制。 | 已证明厚度、含水量或仪器偏差就是陕西延安材料异常的真实原因。 |
| 在保留当前标签的历史TRAIN中观察到可复核的分类规律。 | 仅凭模型认定陕西延安材料必为枣仁／酸枣仁，或确认独立样本与产地批次真实性。 |

本轮没有做HPLC／LC-MS、标准品实验、真实厚度／含水量干预、实物植物学鉴定、独立新批次验证、跨实验室验证或量子新计算。原量子结果和IR状态不由这些分类实验更改。

## 7. 仍需要的验证：均为未做，不能写成现有证据

1. **身份与批次。** 优先补陕西延安及其他关键来源的实物凭证、供货／采集记录，必要时由有资质机构作植物学鉴定；按物种×产地记录多个独立批次。样本量先按目标性能和功效设计，预留整个未见批次作最终预测。
2. **真实少样本校准。** 在新批次中预先固定支持／查询个体，支持与查询实物不重叠；事前锁定K、算法、无支持控制、bias-only及完整重训对照。原CAL/PRED即使未进入本轮，也不能被包装成全研究从未暴露的外部确认数据。
3. **采集与物理干预。** 记录仪器、日期、操作者、制样厚度、含水量和重复测量结构；用同一实物的受控厚度／湿度改变分离物理效应，不能以简单光谱乘法代替。
4. **公平迁移。** 对外部预训练保留目标源额外训练和植物内标签置乱对照，同时记录样本曝光、优化步数、墙钟／CPU或FLOPs；在新物种／新批次／新仪器上分别检验，避免仅在一个已暴露集合上继续调参。
5. **应用与化学解释。** 开放集拒识、概率校准、置信度可靠性和化学峰归属如需进入论文主张，须另作预先设计的独立验证；SVC决策分数不能直接当概率，理论频率接近也不等于实验组分归属。

以上是后续验证需求清单，本页没有启动这些新实验，也不补造实验数据。

## 8. 可核对的来源与SHA256

全部来源均为仓库相对路径，链接在GitHub中指向同一分支文件；`.npz/.pt`等重产物和原始大日志仍为本地保留，本文只引用许可的小型结果。

| 指标块 | index文件 | SHA256 |
| --- | --- | --- |
| source_scores | [source_scores_index.json](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/science/summary-attempt001/source_scores_index.json) | `8bdee972fdd2fd73c6b9d3b03e6db8359d4892fcafb875de04e2b3a06497edf4` |
| support_alpha | [support_alpha_index.json](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/science/summary-attempt001/support_alpha_index.json) | `4a8411a640cc574707607590fe69762f7286fde2415b52172c26be54588e521f` |
| donor_alpha | [donor_alpha_index.json](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/science/summary-attempt001/donor_alpha_index.json) | `39708c8863b6058cb8c99c3529982a73b138f51c85bec51227a2f06326650a6d` |
| support_n | [support_n_index.json](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/science/summary-attempt001/support_n_index.json) | `95904a472a01411a6cda659f1465973735edbf9dda8389cb8d7617c35714b837` |
| donor_n | [donor_n_index.json](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/science/summary-attempt001/donor_n_index.json) | `624637f160bb3b0231094101846fafdba4766801962e72716b6555b03ab92e67` |
| support_fusion | [support_fusion_index.json](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/science/summary-attempt001/support_fusion_index.json) | `c948dc4ad85b65be050efa3b475d19c20fd03ad9900c9f76a5f65e84da20b396` |
| donor_fusion | [donor_fusion_index.json](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/science/summary-attempt001/donor_fusion_index.json) | `7fc95d6a17aaaea8c7870907ea9948190eb641f4ce143d52ba905f145fb70134` |
| neural_120_full | [neural_120_full_index.json](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/science/summary-attempt001/neural_120_full_index.json) | `0c4252a2765d7682a854c79eb7464cc5ae47ae563e6c256aaa9e26f2628c2723` |
| neural_120_frozen | [neural_120_frozen_index.json](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/science/summary-attempt001/neural_120_frozen_index.json) | `2fa1830c1c6ca13a5f7a740aad6f4ee861be9e71ce90ea17f7e881e0618bfcf6` |
| neural_480_full | [neural_480_full_index.json](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/science/summary-attempt001/neural_480_full_index.json) | `d2fba906d50b784e376bf6bd1f2a8833d50284e819705d4b46855580873c2ba2` |
| neural_480_frozen | [neural_480_frozen_index.json](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/science/summary-attempt001/neural_480_frozen_index.json) | `b2e9eddab33e545f7b27243427055495fe82234aff056600c0a1678cbac979f1` |
| bn_neutral | [bn_neutral_index.json](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/science/summary-attempt001/bn_neutral_index.json) | `0767b0f6875822317854e53e6465bb100dba8c3b6f8377412f889e07f0e0a203` |
| bn_source_recal | [bn_source_recal_index.json](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/science/summary-attempt001/bn_source_recal_index.json) | `0ddad03674cf4db24bb93f922e57b717bddcad233bcef923d514b2f9ab2ee8f9` |

基础来源与完整性证据：

- [manifest_ROOT_AUTHORIZED.json](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/manifest_ROOT_AUTHORIZED.json)；SHA256 `8defa363852ff3f05bf3c7635fbd79dde3781a6832d565d0ee4b5bd874cf7c29`。
- [scientific_protocol_ROOT.json](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/preparation/scientific_protocol_ROOT.json)；SHA256 `60d33b662410f1e7729a28cf585ec255616a47b51be151fa10f4ef0278264ba1`。
- [summary_index.json](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/science/summary-attempt001/summary_index.json)；SHA256 `60a97fddfd5d7bc37a56a0c18ba19818a79bc552a416d77815480740e020d8ce`。
- [independent_completion_science_audit_20261109_001.json](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/analysis/independent_completion_science_audit_20261109_001.json)；SHA256 `63d533d6ebbe0ea871f52e923d8ecfb5f07b94bf126737f018be303064afc86a`。
- [independent_original_publication_proof_20261109_001.json](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/analysis/independent_original_publication_proof_20261109_001.json)；SHA256 `351324f39fdb2b1b64c246fc327f3b85544ce938fe43e9944b8ff18624288606`。

原114批回传核验是此前时间点的快照；后续12阶段纯补传和原13项待回传范围的最终核验见 [final_publication_verification_001.json](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/publication-recovery-v1/analysis/final_publication_verification_001.json)，SHA256 `ac00a83907e2dd6737b2c25ae2eac7c0240c08cb298c3e54ef5ff04328298ff1`。该证明确认科学小结果与最终回执位于真实远端；本页新增写作文件的发布记录与之分开。
