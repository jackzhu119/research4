# 模型实验结果、协议边界与论文可用结论

整理日期：2026-11-09（北京时间）。这是已保存结果的证据汇编，不是新一轮训练。所有数值对应下文固定提交；分支后续变化不自动纳入本汇编。DONE表示该有界实验实际完成，不表示其假设已经成立或获得外部验证。

## 首先确定论文中哪个准确率可以写在哪里

当前纯酸枣仁任务的主要开发候选为**吸收系数α＋SNV＋MC-UVE20＋RBF-SVM，Accuracy=93.4444%，Macro-F1=0.934401，Balanced Accuracy=93.4444%**。该数值来自300份酸枣仁TRAIN、3次划分重复×5外折／2内折。三次完整OOF分别282、280、279／300，即841／900条重复预测正确；900不是900份独立样品。

旧**96.0234%**属于855份ZY01—ZY07混合枣仁／酸枣仁来源组识别，应移入历史方法开发或标签纠正说明。固定100份酸枣仁预测池的R0最高94%、加入100份校准池完整重训96%，也不是上面的嵌套CV或全新独立确认。不能将这些数字串成同一协议的提升阶梯。

当前基于128点α的最佳迁移网络79.8889%／Macro-F1=0.799491，属于受控神经网络开发矩阵的最佳观察配置；不应替换强传统模型或写成深度学习超过MC-UVE＋SVM。下面按协议分表，避免不同样品量与搜索预算混排。

## P-CV-MATCH：纯酸枣仁统一嵌套比较（DONE）

数据为当前版本`2026-11-08-confirmed-origin-kind-v1`中的ZY03—ZY07：300份master TRAIN，各组60份；334份master CAL和339份master PRED均没有进入这些训练。划分seed=20261031／20261108／42，模型seed=42。每外折240训练／60验证，训练内2折，选择器和预处理只在相应训练折拟合。原始α有612频点；K候选31／62／123／184／245，C候选0.1／1／10／100，gamma候选scale／0.1×scale／10×scale，SVM使用balanced权重。MC-UVE PLS潜变量数固定2，MC次数和表示因素按家族预定。

主模型的K与超参数由内层平均Macro-F1优先选择；然后BA、Accuracy及固定平局顺序。实际MC20主候选15个外折全部内选C=100、gamma=0.1×scale；K184占6折、K245占9折，平均K=220.6。这是折内选择记录，不是一份在全部数据上已拟合的最终部署模型。全谱LDA／MLP也在相同source/query划分内选择预处理或网络配置。MLP内部early-stop只用source，sklearn1.8字符串标签兼容性修复仅编码估计器内部目标再映回原标签，未改样品真值。

| 主选择流程／家族 | Accuracy（三重复均值） | Macro-F1（三重复均值） | Balanced Accuracy |
|---|---:|---:|---:|
| alpha_snv100 | 93.4444% | 0.934503 | 93.4444% |
| alpha_snv20 | 93.4444% | 0.934401 | 93.4444% |
| alpha_raw20 | 93.2222% | 0.932543 | 93.2222% |
| alpha_snv_stability5x20 | 92.7778% | 0.927969 | 92.7778% |
| fullalpha_shrinkLDA | 91.7778% | 0.917591 | 91.7778% |
| concat_both_snv20 | 91.6667% | 0.916904 | 91.6667% |
| weighted_concat_nraw_asnv20 | 89.2222% | 0.892297 | 89.2222% |
| block_half_nraw_asnv20 | 88.7778% | 0.887508 | 88.7778% |
| concat_nraw_asnv20 | 87.1111% | 0.871076 | 87.1111% |
| n_raw20 | 83.6667% | 0.834790 | 83.6667% |
| fullalpha_MLP | 73.2222% | 0.728649 | 73.2222% |
| n_snv20 | 62.5556% | 0.622045 | 62.5556% |

**解释：** MC100与MC20准确率相同，增加迭代次数没有证实Accuracy收益；两者Macro-F1略有差别。5×MC20排序稳定化为标准聚合构造，未超过MC20，不宜改名宣称新算法。收缩LDA是有竞争力的简洁对照；MLP在此配置较弱，不应删掉该负结果。以上均属于反复开发后的内部研究比较，不是外部确认。

[最终全部122个聚合设置及逐重复混淆矩阵](https://github.com/jackzhu119/research4/blob/71a303e089c7c7b7b2557d62372aaceda44e2f91/data/research/evidence_validation_20261108_v4/snapshots/00330/summary.json) · [可导入表格](https://github.com/jackzhu119/research4/blob/71a303e089c7c7b7b2557d62372aaceda44e2f91/data/research/evidence_validation_20261108_v4/snapshots/00330/metrics.csv) · [冻结协议](https://github.com/jackzhu119/research4/blob/71a303e089c7c7b7b2557d62372aaceda44e2f91/data/research/evidence_validation_20261108_v4/protocol.json) · [V4实现](https://github.com/jackzhu119/research4/blob/71a303e089c7c7b7b2557d62372aaceda44e2f91/scripts/research4_matched_feature_validation_20261108_v4.py)。

## 特征数量、融合、稳定性及成本消融（DONE）

- 固定K曲线是消融，不是用外折挑K。比如K245的MC100=93.8889%不能事后晋升主模型；主流程仍是内层选K后的93.4444%。
- 在K123时α-SNV=89.6667%、双SNV融合=91.0000%；在K245时分别93.7778%、92.4444%。融合收益依赖特征数量，不能写成普遍优于单谱。
- n原谱与SNV差异明显：K62时81.0000%对43.3333%；内层选K主流程83.6667%对62.5556%。这支持“折射率预处理必须按物理量验证”的探索性判断，不能把性能差异当成特定化学物的证明。
- 旧融合R2的15外折中13折没有选中任何n变量，n仅占28／2943次入选，融合平均K196.2、α平均K82.2；因此旧融合优势存在特征预算混杂。不同K的Jaccard也不能直接用于声称选择更稳定。
- MC20与MC100配对重复预测改对5／改错5；按300个physical_sample_id聚类重采样、保留每个样品三次预测，500次bootstrap的ΔAccuracy95%区间为[−0.6667，＋0.5556]个百分点。这里的区间是探索性不确定性估计，没有多重选择校正。
- 旧融合R2=92.3333%、Macro-F1=0.923547，与当前MC20的相同300ID和外折配对净增10／900（改对30、改错20），＋1.1111个百分点；先前独立配对检查的区间[−0.67，＋3.22]个百分点跨0。搜索K和方法也变化，不能声称一个因素造成显著提升。
- 运行时完整成本在账本，单个模型／选择器成本在各fold的summary/checkpoint中；不能把局部阶段CPU当整个嵌套搜索或部署延迟。MC20/100记录的选择器及候选拟合阶段约17.688／32.929CPU秒，仅是该范围内的成本对照。

[预算混杂审查](https://github.com/jackzhu119/research4/blob/71a303e089c7c7b7b2557d62372aaceda44e2f91/data/research/evidence_validation_20261108_v4/audits/previous_feature_budget_confound_001.json) · [首折独立核查](https://github.com/jackzhu119/research4/blob/71a303e089c7c7b7b2557d62372aaceda44e2f91/data/research/evidence_validation_20261108_v4/audits/first_fold_independent_001.json) · [旧完整开发参考表](https://github.com/jackzhu119/research4/blob/71a303e089c7c7b7b2557d62372aaceda44e2f91/data/research/evidence_validation_20261108_v4/reference_previous_development.csv)。

普通SVM、PLS-DA及R0/R2在早先225项经典实验完成。对于相同300份酸枣仁的三重复OOF，α全谱SVM为83.4444%、F1=0.834170；PLS-DA为76.6667%、F1=0.766561；R0为85.4444%、R2为91.3333%；n＋α R2为92.3333%。其具体搜索流程与本轮P-CV-MATCH不同，作为前一代流程对照，不能把差异归因于单一新增模块。默认应使用estimator_vote行，score_argmax是另一个零支持语义对照，不可混用挑高分。

## P-NN：跨植物预训练、表示与优化步数对照（DONE）

原有限矩阵完成**1080个神经科学任务**：酸枣仁、黄芪、天麻各为独立来源任务，8训练profile×3预训练预算120／480／1920×5折×3模型seed。各任务训练样品只来自master TRAIN，其他两种药材用独立来源头辅助预训练。相同A／SVM缓存命中不算新拟合；1080是任务数，不是1080个全部独立训练模型。

最终旧矩阵无校准读数中：酸枣仁最好B/p24_e192_full/pre480=77.2222%／F1=0.771980；匹配A=72.0000%、Aextra=73.5556%。黄芪最好B/p24_e384_full/pre480=73.3057%／F1=0.733470；天麻最好Aextra/p24_e96_full/pre120=94.7658%／F1=0.947813。目标样品量、source/support分配不同，应按各任务单独评价。

[旧矩阵288个完整source_only聚合行](https://github.com/jackzhu119/research4/blob/df3e12656e2177cfcf0a298e66bdedc5c524cb74/reports/00480/summary.json) · [表格](https://github.com/jackzhu119/research4/blob/df3e12656e2177cfcf0a298e66bdedc5c524cb74/reports/00480/metrics.csv) · [旧实验协议](https://github.com/jackzhu119/research4/blob/8a388b8b4d8a5baf9f2e4321e76455c2c0b49eca/data/research/species_relaunch_20261108_v2/protocol.json)。

本轮又完成**315个神经任务**（21预定条件×5折×3seed）。128／512点均使用0.21—1.49THz，不改变频段；190source＋50reserved TRAIN support＋60query与240source＋0support＋60query分开比较。标准192epochs在source190/240下分别为576／768优化步；新144epochs的240source条件是576步，与190source192epochs匹配。相同步数不等于相同样品暴露量或FLOPs。

| 表示／source／epochs | 目标从头A | 目标额外训练Aextra | 黄芪预训练 | 天麻预训练 | 黄芪＋天麻B |
|---|---:|---:|---:|---:|---:|
| α128／240／192 | 75.4444% | 77.1111% | 76.0000% | 71.1111% | 79.7778% |
| α128／240／144 | 77.3333% | 未设计此条件 | 未设计此条件 | 未设计此条件 | **79.8889%** |
| α512／240／192 | 67.2222% | 67.2222% | 67.3333% | 59.1111% | 63.4444% |
| α512／240／144 | 68.2222% | 未设计此条件 | 未设计此条件 | 未设计此条件 | 67.0000% |

128点240source144epochs下，B−A=＋2.5556个百分点，Macro-F1从0.773523到0.799491。512点则没有相同正迁移。不能由最佳单配置推出跨物种普遍正迁移或深度模型优于传统筛选。315任务里没有已执行Attention或DFT先验分支；本轮网络profile的attention=false、residual=true。

神经网同表示同source的固定SVM也保留：例如α128/source240为73.3333%，α512/source240为76.3333%。它不是612点MC-UVE调优SVM，应与匹配神经条件对照；不能把匹配SVM的73.33%偷换成论文传统方法的最优水平。

## 少样本校准及零支持对照（DONE，但稳定改善未成立）

本轮α512/source190/B的原模型60.7778%、F1=0.607249。每类K=1／3／5／10，20次支持抽样，先保证完整5折覆盖，再按seed平均。支持来自每折TRAIN保留的50份，不能称真实新批次；每次query始终同一300物理样品覆盖。

| 每类K | 传统λ1 Accuracy | 传统Macro-F1 | 弱校准λ1000／blend0.05 Accuracy | 弱校准Macro-F1 |
|---|---:|---:|---:|---:|
| 1 | 47.0833% | 0.468520 | 60.7944% | 0.607403 |
| 3 | 56.1222% | 0.557182 | 60.7500% | 0.606962 |
| 5 | 59.5722% | 0.592018 | 60.7667% | 0.607147 |
| 10 | 61.3333% | 0.609991 | 60.7778% | 0.607249 |

identity_zero_support与source_only完全相同。传统K10的均值增益只在seed44明显，其他seed下降，不能当稳定校准成功。弱校准多数保持原预测，近零变化不等于产生性能提升。逐设置改对／改错和真值在fold-levelsummary/predictions，最终compact包含逐重复指标，但不应把20×3次抽样当60个独立实验样本。

旧1080NN的全部校准压缩预测保留，最终runtime compact明确`calibration_policy=NOT_SUMMARIZED`，只汇总无校准，因此**全旧矩阵校准最终聚合表为PARTIAL**，不可用无校准报告代替。固定预测池的独立校准试验见下一节。

## P-HOLDOUT：已曝光固定339预测角色（DONE，单列历史内部验证）

master名单固定999TRAIN／334CAL／339PRED。酸枣仁300／100／100、枣仁216／74／74、黄芪241／80／82、天麻242／80／83。历史855角色继承未洗牌。此预测池早已曝光，后续模型探索未以它选参，但它不能重新包装为盲测。

| 任务／原source模型 | 正确／预测数 | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|---:|
| 枣仁／酸枣仁二分类R2 | 174／174 | 100.0000% | 1.0000 | 100.0000% |
| 纯酸枣仁五来源R2 | 87／100 | 87.0000% | 0.8693 | 87.0000% |
| 四种药材全谱SVM | 331／339 | 97.6401% | 0.9751 | 97.5000% |
| 黄芪四来源全谱SVM | 69／82 | 84.1463% | 0.8386 | 84.0909% |
| 天麻四来源全谱SVM | 81／83 | 97.5904% | 0.9756 | 97.6190% |
| 枣仁五来源诊断 | 72／74 | 97.2973% | 0.5972 | 60.0000% |

枣仁97.30%的少数类很差：预测74份中延安70份，辽宁与内蒙古各1份均错。不能只报告Accuracy。酸枣仁R2加全部100CAL完整重训为93／100，R0源模型94／100、加CAL96／100；完整重训与冻结模型K-shot修正应分别命名，不能事后按PRED最优选择模型。R2的传统锚定校准K1/3/5/10分别75.90/81.50/84.05/86.30%，弱校准均87.00%，没有增益。

固定酸枣仁100PRED的旧跨植物对照（三seed、120预训练步、96目标epoch）：A81.00%、Aextra80.33%、B79.00%，F1分别0.8085、0.8041、0.7881，保留该负迁移。此协议不能与P-NN最佳79.89%直接相减。

[固定分类、校准完整记录](https://github.com/jackzhu119/research4/blob/71a303e089c7c7b7b2557d62372aaceda44e2f91/data/research/species_revision_20261108/classification/summary.json) · [逐设置CSV](https://github.com/jackzhu119/research4/blob/71a303e089c7c7b7b2557d62372aaceda44e2f91/data/research/species_revision_20261108/classification/metrics_overview.csv) · [固定迁移结果](https://github.com/jackzhu119/research4/blob/71a303e089c7c7b7b2557d62372aaceda44e2f91/data/research/species_revision_20261108/transfer/result.json) · [逐样品名单](https://github.com/jackzhu119/research4/blob/71a303e089c7c7b7b2557d62372aaceda44e2f91/data/research/species_revision_20261108/split_manifest.csv) · [独立指标复核](https://github.com/jackzhu119/research4/blob/71a303e089c7c7b7b2557d62372aaceda44e2f91/data/research/species_revision_20261108/independent_results_validation.json)。

## 来源留出和层级系统：附加诊断，不能替换主任务

另一主机B固定SNV＋SVM在516份TRAIN上进行预定来源组留出，20次真实拟合（真标签五折＋三个fit-label置乱五折）：真实493／516=95.5426%、F1=0.953424、BA=94.6759%；置乱准确率32.36／40.89／36.63%。这不是随机样品折，也不是已核实新批次；3次置乱不足以估计正式p值。

对固定339个ID的已冻结组件逻辑与为304／339=89.6755%，误路由8份。该结果仅是组件共同正确诊断；误路由后的真实来源输出和完整系统Macro-F1为NOT_COMPUTED。不能以四药材97.64%代替“先种类再来源”的系统性能。

[跨分支固定证据路径及解释](https://github.com/jackzhu119/research4/blob/71a303e089c7c7b7b2557d62372aaceda44e2f91/docs/CROSS_BRANCH_EVIDENCE_AUDIT_20261108.md)，另一主机真实结果绑定`503de339ed5d42967f5dbf0502e9d0b5ffc1115b`。

## 逐类指标、模型与重现材料在哪里

- 最新最终`summary.json`每个设置的`repeats[].metrics`包含完整confusion_matrix和per_class。每个300样品重复应单独展示混淆矩阵；可以报告三重复均值，但不得把叠加混淆矩阵的900当独立N。
- 经典每折`runs/matched_r<seed>_f<fold>/summary.json`包含内选K/参数、单折指标及模型hash；`validation_predictions.csv.gz`、`selected_features.csv.gz`、`inner_search.csv.gz`和`selection_lock.json`给出逐预测、频点及选择轨迹。变换器及模型路径在`complete.json`封存，巨型可重建缓存不等于已全部远端上传。
- 神经每折`runs/representation_<...>/summary.json`记录网络结构、实际优化步、source/auxiliaryID、重用cache、CPU/wall；`predictions.csv.gz`记录类别和校准变化；`complete.json`绑定模型收据。
- 旧最终无校准表包含`mean_per_class`，不包含完整逐seed混淆矩阵；详细混淆应查原单任务summary及压缩预测。固定339各任务summary和压缩预测也保留逐类指标。
- 复现入口、依赖和冻结源在各协议及提交；写作阶段默认只读取封存产物。不要把重新运行训练输出覆盖到本次既有目录。

## 可写与不能写的结论

可以写：纠正种类定义后纯酸枣仁仍有较强内部来源分类信号；MC-UVE/SVM是本数据最有价值主模型；控制特征数量可改变融合收益判断；n预处理和神经表示/步数对迁移结果影响大；在纯酸枣仁五来源任务中，保守校准可以限制破坏，但尚未稳定提高准确率。

另一主机B的枣仁／酸枣仁二分类来源域适配有不同结论：传统λ1锚定在其重复查询控制中存在改善，不能把本节纯酸五来源负结果推广为所有校准无效。B的二分类、来源留出与重复样品分母见[B专项报告](PARALLEL_HOST_B_RESULTS.md)，不把其高二分类准确率替换为本论文五来源主成绩。

不能写：已确认外部泛化或真实跨批次校准成功；DL一定优于传统模型；所有植物都能迁移；斯皮诺素峰已确认或DFT提高分类准确率；模型鉴定药效/化学含量；当前候选是已验证原创算法。原始种类/来源由用户声明，尚非独立植物学与采样溯源鉴定。
