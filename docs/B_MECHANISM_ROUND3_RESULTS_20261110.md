# B方案第三轮：正则化、增量监督与类别权重边界

状态：**PARTIAL**；已封存 **4/375** 个任务包。
本报告只读取完整封存工件，不拟合模型，不修改前两轮代码、模型、数据、标签或结果。

新增分类器拟合23次、校准拟合调用62次；科研任务CPU合计1.489615秒，控制/核验/推送成本另见后台账本。
已保存4480条预测，对应40个唯一Query实物；重复记录不是独立样本。

## 完成范围

|块|问题|完成/计划|状态|
|---|---|---:|---|
|G01|损失归一化 × 截距惩罚|1/45|PARTIAL|
|G02|增量监督训练与无标签统计量分解|1/45|PARTIAL|
|G03|人工支持标签污染剂量|1/45|PARTIAL|
|G04|类别权重、margin与对偶约束的零拟合审计|1/240|PARTIAL|

## 冻结设计与解释边界

G01在同一源分数、支持ID和Query上比较sum-loss λ1与mean-loss λ1（等价sum λ=N），交叉截距惩罚λ或0。原sum/惩罚截距端点复用第二轮Anchor；新增参数不依据Query选择。draw0/101/102在每类10份池内嵌套K1/3/5/10，真标签K10只保留一次。保存有效自由度、秩、条件数、跨类耦合和修正范数。

G02保持数值gamma、C及支持预算固定。无标签统计条件只用Source+Support的谱拟合Scaler，再只用Source标签重训SVC；真实支持与循环错标条件使用完全相同的Source+Support谱和Scaler，只有Support标签副本不同。没有把新Scaler直接接到旧SVC上冒充一致的适配。真实标签重训端点优先复用。true_support相对unlabeled_statistics同时加入监督训练样品和损失项；固定C下样本数变化也改变总损失与正则项的相对关系。因此该差值描述增量监督训练，不唯一识别标签信息的单一因果作用。true_support与cyclic_support保持训练样本数、Scaler、gamma和标签直方图相同，是更直接的支持标签正确性对照。Support候选池原本由已知标签分层构造，所有条件共用该池；无标签统计条件指额外Support标签值不进入拟合，不能据此称实际采样流程完全无需标签或证明外部无监督适配。

G03仅在每类10份Support上人工改变1/2/4/10份，实现精确10/20/40/100%污染。循环类标签保留直方图，原元数据及真值不变；100%完整污染端点只保留一次并复用G02模型，完全相同的掩码目标共享拟合。保留随机掩码的抽样权重，但不把重复当成新实物。它不是独立同分布现场误标，也不证明现有标签错误。

污染百分比的分母仅为Support；Source标签保持正确。每类改变4/10份时，五来源为20/50份Support受扰，但占完整190+50训练集的20/240=8.33%；二分类为8/20份Support受扰，占240+20训练集的8/260=3.08%。各端点另存总有标签训练集污染比例，不能把40%Support污染表述为全部训练数据40%污染。

G04只复用第二轮240个来源上下文的模型、原始预测和分数。类别决策一致时仍检查分数、排序、AUROC、支持向量和α/C约束。接近边界的数值容差仅是诊断，既不终止训练，也不作为通用类别权重无效的证明。

## 全部方法的描述性结果

以下端点按任务及合法draw等权汇总；K10、零污染及100%污染的单一端点不会被伪装为多次新实验。

|块|原任务|方法|K/类|Support污染比例|Accuracy|Macro-F1|Balanced Accuracy|端点数|
|---|---|---|---:|---:|---:|---:|---:|---:|
|G01|E01|identity|0||1.000000|1.000000|1.000000|1|
|G01|E01|mean_free_intercept|1||1.000000|1.000000|1.000000|3|
|G01|E01|mean_free_intercept|3||1.000000|1.000000|1.000000|3|
|G01|E01|mean_free_intercept|5||1.000000|1.000000|1.000000|3|
|G01|E01|mean_free_intercept|10||1.000000|1.000000|1.000000|1|
|G01|E01|mean_penalized|1||1.000000|1.000000|1.000000|3|
|G01|E01|mean_penalized|3||1.000000|1.000000|1.000000|3|
|G01|E01|mean_penalized|5||1.000000|1.000000|1.000000|3|
|G01|E01|mean_penalized|10||1.000000|1.000000|1.000000|1|
|G01|E01|sum_free_intercept|1||1.000000|1.000000|1.000000|3|
|G01|E01|sum_free_intercept|3||1.000000|1.000000|1.000000|3|
|G01|E01|sum_free_intercept|5||1.000000|1.000000|1.000000|3|
|G01|E01|sum_free_intercept|10||1.000000|1.000000|1.000000|1|
|G01|E01|sum_penalized|1||1.000000|1.000000|1.000000|3|
|G01|E01|sum_penalized|3||1.000000|1.000000|1.000000|3|
|G01|E01|sum_penalized|5||1.000000|1.000000|1.000000|3|
|G01|E01|sum_penalized|10||1.000000|1.000000|1.000000|1|
|G01|E01|svc_predict|0||1.000000|1.000000|1.000000|1|
|G02|E01|cyclic_support|1||1.000000|1.000000|1.000000|1|
|G02|E01|cyclic_support|3||1.000000|1.000000|1.000000|1|
|G02|E01|cyclic_support|5||1.000000|1.000000|1.000000|1|
|G02|E01|cyclic_support|10||0.975000|0.974984|0.975000|1|
|G02|E01|identity|0||1.000000|1.000000|1.000000|1|
|G02|E01|true_support|1||1.000000|1.000000|1.000000|1|
|G02|E01|true_support|3||1.000000|1.000000|1.000000|1|
|G02|E01|true_support|5||1.000000|1.000000|1.000000|1|
|G02|E01|true_support|10||1.000000|1.000000|1.000000|1|
|G02|E01|unlabeled_statistics|1||1.000000|1.000000|1.000000|1|
|G02|E01|unlabeled_statistics|3||1.000000|1.000000|1.000000|1|
|G02|E01|unlabeled_statistics|5||1.000000|1.000000|1.000000|1|
|G02|E01|unlabeled_statistics|10||1.000000|1.000000|1.000000|1|
|G03|E01|full_pipeline_fixedgamma|10|0%|1.000000|1.000000|1.000000|1|
|G03|E01|full_pipeline_fixedgamma|10|10%|1.000000|1.000000|1.000000|5|
|G03|E01|full_pipeline_fixedgamma|10|20%|1.000000|1.000000|1.000000|5|
|G03|E01|full_pipeline_fixedgamma|10|40%|1.000000|1.000000|1.000000|5|
|G03|E01|full_pipeline_fixedgamma|10|100%|0.975000|0.974984|0.975000|1|
|G03|E01|identity|0|0%|1.000000|1.000000|1.000000|1|
|G03|E01|raw_anchor_lambda1|10|0%|1.000000|1.000000|1.000000|1|
|G03|E01|raw_anchor_lambda1|10|10%|1.000000|1.000000|1.000000|5|
|G03|E01|raw_anchor_lambda1|10|20%|1.000000|1.000000|1.000000|5|
|G03|E01|raw_anchor_lambda1|10|40%|1.000000|1.000000|1.000000|5|
|G03|E01|raw_anchor_lambda1|10|100%|0.000000|0.000000|0.000000|1|
|G03|E01|svc_predict|0|0%|1.000000|1.000000|1.000000|1|
|G03|E01|weak_lambda1000_blend005|10|0%|1.000000|1.000000|1.000000|1|
|G03|E01|weak_lambda1000_blend005|10|10%|1.000000|1.000000|1.000000|5|
|G03|E01|weak_lambda1000_blend005|10|20%|1.000000|1.000000|1.000000|5|
|G03|E01|weak_lambda1000_blend005|10|40%|1.000000|1.000000|1.000000|5|
|G03|E01|weak_lambda1000_blend005|10|100%|1.000000|1.000000|1.000000|1|
|G04|E01|class_weight_balanced|0||1.000000|1.000000|1.000000|1|
|G04|E01|class_weight_balanced_argmax|0||1.000000|1.000000|1.000000|1|
|G04|E01|class_weight_none|0||1.000000|1.000000|1.000000|1|
|G04|E01|class_weight_none_argmax|0||1.000000|1.000000|1.000000|1|

## 配对变化与描述性区间

全部方法、剂量、交互、逐类Recall变化及纠错/误伤在CSV中保留。下面列全体匹配Query的对照，按physical_sample_id聚类，不把支持draw、污染掩码或重复外折行当独立观察。

|块|原任务|比较|K/类|Support污染比例|ΔAccuracy百分点及95%区间|纠错/误伤|预测行/实物|
|---|---|---|---:|---:|---:|---:|---:|
|G01|E01|sum_penalized_minus_identity|1||+0.000 [+0.000,+0.000]|0/0|120/40|
|G01|E01|mean_penalized_minus_identity|1||+0.000 [+0.000,+0.000]|0/0|120/40|
|G01|E01|sum_free_intercept_minus_identity|1||+0.000 [+0.000,+0.000]|0/0|120/40|
|G01|E01|mean_free_intercept_minus_identity|1||+0.000 [+0.000,+0.000]|0/0|120/40|
|G01|E01|mean_minus_sum_penalized_intercept|1||+0.000 [+0.000,+0.000]|0/0|120/40|
|G01|E01|mean_minus_sum_free_intercept|1||+0.000 [+0.000,+0.000]|0/0|120/40|
|G01|E01|sum_free_minus_penalized_intercept|1||+0.000 [+0.000,+0.000]|0/0|120/40|
|G01|E01|mean_free_minus_penalized_intercept|1||+0.000 [+0.000,+0.000]|0/0|120/40|
|G01|E01|sum_penalized_minus_identity|3||+0.000 [+0.000,+0.000]|0/0|120/40|
|G01|E01|mean_penalized_minus_identity|3||+0.000 [+0.000,+0.000]|0/0|120/40|
|G01|E01|sum_free_intercept_minus_identity|3||+0.000 [+0.000,+0.000]|0/0|120/40|
|G01|E01|mean_free_intercept_minus_identity|3||+0.000 [+0.000,+0.000]|0/0|120/40|
|G01|E01|mean_minus_sum_penalized_intercept|3||+0.000 [+0.000,+0.000]|0/0|120/40|
|G01|E01|mean_minus_sum_free_intercept|3||+0.000 [+0.000,+0.000]|0/0|120/40|
|G01|E01|sum_free_minus_penalized_intercept|3||+0.000 [+0.000,+0.000]|0/0|120/40|
|G01|E01|mean_free_minus_penalized_intercept|3||+0.000 [+0.000,+0.000]|0/0|120/40|
|G01|E01|sum_penalized_minus_identity|5||+0.000 [+0.000,+0.000]|0/0|120/40|
|G01|E01|mean_penalized_minus_identity|5||+0.000 [+0.000,+0.000]|0/0|120/40|
|G01|E01|sum_free_intercept_minus_identity|5||+0.000 [+0.000,+0.000]|0/0|120/40|
|G01|E01|mean_free_intercept_minus_identity|5||+0.000 [+0.000,+0.000]|0/0|120/40|
|G01|E01|mean_minus_sum_penalized_intercept|5||+0.000 [+0.000,+0.000]|0/0|120/40|
|G01|E01|mean_minus_sum_free_intercept|5||+0.000 [+0.000,+0.000]|0/0|120/40|
|G01|E01|sum_free_minus_penalized_intercept|5||+0.000 [+0.000,+0.000]|0/0|120/40|
|G01|E01|mean_free_minus_penalized_intercept|5||+0.000 [+0.000,+0.000]|0/0|120/40|
|G01|E01|sum_penalized_minus_identity|10||+0.000 [+0.000,+0.000]|0/0|40/40|
|G01|E01|mean_penalized_minus_identity|10||+0.000 [+0.000,+0.000]|0/0|40/40|
|G01|E01|sum_free_intercept_minus_identity|10||+0.000 [+0.000,+0.000]|0/0|40/40|
|G01|E01|mean_free_intercept_minus_identity|10||+0.000 [+0.000,+0.000]|0/0|40/40|
|G01|E01|mean_minus_sum_penalized_intercept|10||+0.000 [+0.000,+0.000]|0/0|40/40|
|G01|E01|mean_minus_sum_free_intercept|10||+0.000 [+0.000,+0.000]|0/0|40/40|
|G01|E01|sum_free_minus_penalized_intercept|10||+0.000 [+0.000,+0.000]|0/0|40/40|
|G01|E01|mean_free_minus_penalized_intercept|10||+0.000 [+0.000,+0.000]|0/0|40/40|
|G01|E01|score_argmax_minus_native_svc_zero_support|0||+0.000 [+0.000,+0.000]|0/0|40/40|
|G02|E01|unlabeled_statistics_minus_source_identity|1||+0.000 [+0.000,+0.000]|0/0|40/40|
|G02|E01|true_support_minus_source_identity|1||+0.000 [+0.000,+0.000]|0/0|40/40|
|G02|E01|cyclic_support_minus_source_identity|1||+0.000 [+0.000,+0.000]|0/0|40/40|
|G02|E01|true_labels_minus_unlabeled_statistics|1||+0.000 [+0.000,+0.000]|0/0|40/40|
|G02|E01|cyclic_labels_minus_true_labels|1||+0.000 [+0.000,+0.000]|0/0|40/40|
|G02|E01|cyclic_labels_minus_unlabeled_statistics|1||+0.000 [+0.000,+0.000]|0/0|40/40|
|G02|E01|unlabeled_statistics_minus_source_identity|3||+0.000 [+0.000,+0.000]|0/0|40/40|
|G02|E01|true_support_minus_source_identity|3||+0.000 [+0.000,+0.000]|0/0|40/40|
|G02|E01|cyclic_support_minus_source_identity|3||+0.000 [+0.000,+0.000]|0/0|40/40|
|G02|E01|true_labels_minus_unlabeled_statistics|3||+0.000 [+0.000,+0.000]|0/0|40/40|
|G02|E01|cyclic_labels_minus_true_labels|3||+0.000 [+0.000,+0.000]|0/0|40/40|
|G02|E01|cyclic_labels_minus_unlabeled_statistics|3||+0.000 [+0.000,+0.000]|0/0|40/40|
|G02|E01|unlabeled_statistics_minus_source_identity|5||+0.000 [+0.000,+0.000]|0/0|40/40|
|G02|E01|true_support_minus_source_identity|5||+0.000 [+0.000,+0.000]|0/0|40/40|
|G02|E01|cyclic_support_minus_source_identity|5||+0.000 [+0.000,+0.000]|0/0|40/40|
|G02|E01|true_labels_minus_unlabeled_statistics|5||+0.000 [+0.000,+0.000]|0/0|40/40|
|G02|E01|cyclic_labels_minus_true_labels|5||+0.000 [+0.000,+0.000]|0/0|40/40|
|G02|E01|cyclic_labels_minus_unlabeled_statistics|5||+0.000 [+0.000,+0.000]|0/0|40/40|
|G02|E01|unlabeled_statistics_minus_source_identity|10||+0.000 [+0.000,+0.000]|0/0|40/40|
|G02|E01|true_support_minus_source_identity|10||+0.000 [+0.000,+0.000]|0/0|40/40|
|G02|E01|cyclic_support_minus_source_identity|10||-2.500 [-7.500,+0.000]|0/1|40/40|
|G02|E01|true_labels_minus_unlabeled_statistics|10||+0.000 [+0.000,+0.000]|0/0|40/40|
|G02|E01|cyclic_labels_minus_true_labels|10||-2.500 [-7.500,+0.000]|0/1|40/40|
|G02|E01|cyclic_labels_minus_unlabeled_statistics|10||-2.500 [-7.500,+0.000]|0/1|40/40|
|G03|E01|raw_anchor_lambda1_minus_matched_source_control|10|0%|+0.000 [+0.000,+0.000]|0/0|40/40|
|G03|E01|raw_anchor_lambda1_noise_minus_true_support|10|0%|+0.000 [+0.000,+0.000]|0/0|40/40|
|G03|E01|raw_anchor_lambda1_minus_matched_source_control|10|10%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|raw_anchor_lambda1_noise_minus_true_support|10|10%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|raw_anchor_lambda1_minus_matched_source_control|10|20%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|raw_anchor_lambda1_noise_minus_true_support|10|20%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|raw_anchor_lambda1_minus_matched_source_control|10|40%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|raw_anchor_lambda1_noise_minus_true_support|10|40%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|raw_anchor_lambda1_minus_matched_source_control|10|100%|-100.000 [-100.000,-100.000]|0/40|40/40|
|G03|E01|raw_anchor_lambda1_noise_minus_true_support|10|100%|-100.000 [-100.000,-100.000]|0/40|40/40|
|G03|E01|weak_lambda1000_blend005_minus_matched_source_control|10|0%|+0.000 [+0.000,+0.000]|0/0|40/40|
|G03|E01|weak_lambda1000_blend005_noise_minus_true_support|10|0%|+0.000 [+0.000,+0.000]|0/0|40/40|
|G03|E01|weak_lambda1000_blend005_minus_matched_source_control|10|10%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|weak_lambda1000_blend005_noise_minus_true_support|10|10%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|weak_lambda1000_blend005_minus_matched_source_control|10|20%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|weak_lambda1000_blend005_noise_minus_true_support|10|20%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|weak_lambda1000_blend005_minus_matched_source_control|10|40%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|weak_lambda1000_blend005_noise_minus_true_support|10|40%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|weak_lambda1000_blend005_minus_matched_source_control|10|100%|+0.000 [+0.000,+0.000]|0/0|40/40|
|G03|E01|weak_lambda1000_blend005_noise_minus_true_support|10|100%|+0.000 [+0.000,+0.000]|0/0|40/40|
|G03|E01|full_pipeline_fixedgamma_minus_matched_source_control|10|0%|+0.000 [+0.000,+0.000]|0/0|40/40|
|G03|E01|full_pipeline_fixedgamma_noise_minus_true_support|10|0%|+0.000 [+0.000,+0.000]|0/0|40/40|
|G03|E01|full_pipeline_fixedgamma_minus_matched_source_control|10|10%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|full_pipeline_fixedgamma_noise_minus_true_support|10|10%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|full_pipeline_fixedgamma_minus_matched_source_control|10|20%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|full_pipeline_fixedgamma_noise_minus_true_support|10|20%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|full_pipeline_fixedgamma_minus_matched_source_control|10|40%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|full_pipeline_fixedgamma_noise_minus_true_support|10|40%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|full_pipeline_fixedgamma_minus_matched_source_control|10|100%|-2.500 [-7.500,+0.000]|0/1|40/40|
|G03|E01|full_pipeline_fixedgamma_noise_minus_true_support|10|100%|-2.500 [-7.500,+0.000]|0/1|40/40|
|G03|E01|weak_lambda1000_blend005_minus_raw_anchor_lambda1|10|0%|+0.000 [+0.000,+0.000]|0/0|40/40|
|G03|E01|full_pipeline_fixedgamma_minus_raw_anchor_lambda1|10|0%|+0.000 [+0.000,+0.000]|0/0|40/40|
|G03|E01|full_pipeline_fixedgamma_minus_weak_lambda1000_blend005|10|0%|+0.000 [+0.000,+0.000]|0/0|40/40|
|G03|E01|weak_lambda1000_blend005_minus_raw_anchor_lambda1|10|10%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|full_pipeline_fixedgamma_minus_raw_anchor_lambda1|10|10%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|full_pipeline_fixedgamma_minus_weak_lambda1000_blend005|10|10%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|weak_lambda1000_blend005_minus_raw_anchor_lambda1|10|20%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|full_pipeline_fixedgamma_minus_raw_anchor_lambda1|10|20%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|full_pipeline_fixedgamma_minus_weak_lambda1000_blend005|10|20%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|weak_lambda1000_blend005_minus_raw_anchor_lambda1|10|40%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|full_pipeline_fixedgamma_minus_raw_anchor_lambda1|10|40%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|full_pipeline_fixedgamma_minus_weak_lambda1000_blend005|10|40%|+0.000 [+0.000,+0.000]|0/0|200/40|
|G03|E01|weak_lambda1000_blend005_minus_raw_anchor_lambda1|10|100%|+100.000 [+100.000,+100.000]|40/0|40/40|
|G03|E01|full_pipeline_fixedgamma_minus_raw_anchor_lambda1|10|100%|+97.500 [+92.500,+100.000]|39/0|40/40|
|G03|E01|full_pipeline_fixedgamma_minus_weak_lambda1000_blend005|10|100%|-2.500 [-7.500,+0.000]|0/1|40/40|
|G03|E01|score_argmax_minus_native_svc_zero_support|0||+0.000 [+0.000,+0.000]|0/0|40/40|
|G04|E01|unweighted_minus_balanced|0||+0.000 [+0.000,+0.000]|0/0|40/40|
|G04|E01|class_weight_balanced_argmax_minus_native|0||+0.000 [+0.000,+0.000]|0/0|40/40|
|G04|E01|class_weight_none_argmax_minus_native|0||+0.000 [+0.000,+0.000]|0/0|40/40|

## 权重与margin审计

逐模型box约束、Source hinge loss、支持向量对应及成对分数结果见mechanism_analysis.csv与每任务margin_mechanism.csv。若两种权重下类别预测相同，只能先说明这些已保存Query上的决策一致；若对偶系数和截距一致，且两组箱约束均未激活，才支持当前Source上下文未激活权重差异的有限解释。A/C比例下箱约束上限C_i按设计不同，不能将它们说成约束相同；也不能据此概括所有来源或未来批次。

## 局限与未运行事项

本轮是依据已曝光内部结果设计的探索性机制补充，仅使用master TRAIN。重复划分、支持抽样和污染掩码不增加独立实物数；描述性95%区间按physical_sample_id聚类，保留同一实物全部相关预测，不覆盖来源/批次总体、协议选择和长期开发偏倚。当前来源与批次混杂，不能据此宣称真实跨批次泛化或确认性因果机制。

二分类与五来源的样本组成、类别数、训练量及支持标签预算不同，不能直接相减解释为类别数的单因素作用。Source训练内诊断不被用于选择正则化、污染掩码或最终方法。所有正、负、零效应保留，不从多重比较挑单个正结果作确认性结论。

真实新批次评价继续为 **NOT_RUN_NEEDS_NEW_BATCH_DATA**。尚未采集的批次、样品、仪器或物理属性不推断补填。

## 复现

```bash
python scripts/run_b_mechanism_round3_20261110.py --resume
python scripts/summarize_b_mechanism_round3_20261110.py --resume
python scripts/summarize_b_mechanism_round3_20261110.py --verify
python scripts/run_b_mechanism_round3_20261110.py --verify
```

训练中途PARTIAL仅表示已封存子集。summary_verification.json保存独立混淆计数复算和工件核验依据。
