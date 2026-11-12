# B方案第三轮：正则化、增量监督与类别权重边界

状态：**DONE**；已封存 **375/375** 个任务包。
本报告只读取完整封存工件，不拟合模型，不修改前两轮代码、模型、数据、标签或结果。

新增分类器拟合1035次、校准拟合调用2790次；科研任务CPU合计108.035185秒，控制/核验/推送成本另见后台账本。
已保存265200条预测，对应422个唯一Query实物；重复记录不是独立样本。

## 完成范围

|块|问题|完成/计划|状态|
|---|---|---:|---|
|G01|损失归一化 × 截距惩罚|45/45|DONE|
|G02|增量监督训练与无标签统计量分解|45/45|DONE|
|G03|人工支持标签污染剂量|45/45|DONE|
|G04|类别权重、margin与对偶约束的零拟合审计|240/240|DONE|

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
|G01|E01|identity|0||0.983333|0.983323|0.983333|30|
|G01|E01|mean_free_intercept|1||0.983889|0.983879|0.983889|90|
|G01|E01|mean_free_intercept|3||0.983333|0.983323|0.983333|90|
|G01|E01|mean_free_intercept|5||0.983333|0.983323|0.983333|90|
|G01|E01|mean_free_intercept|10||0.983333|0.983323|0.983333|30|
|G01|E01|mean_penalized|1||0.983611|0.983601|0.983611|90|
|G01|E01|mean_penalized|3||0.983333|0.983323|0.983333|90|
|G01|E01|mean_penalized|5||0.983333|0.983323|0.983333|90|
|G01|E01|mean_penalized|10||0.983333|0.983323|0.983333|30|
|G01|E01|sum_free_intercept|1||0.983889|0.983879|0.983889|90|
|G01|E01|sum_free_intercept|3||0.983333|0.983323|0.983333|90|
|G01|E01|sum_free_intercept|5||0.983333|0.983323|0.983333|90|
|G01|E01|sum_free_intercept|10||0.983333|0.983323|0.983333|30|
|G01|E01|sum_penalized|1||0.983889|0.983879|0.983889|90|
|G01|E01|sum_penalized|3||0.983333|0.983323|0.983333|90|
|G01|E01|sum_penalized|5||0.983333|0.983323|0.983333|90|
|G01|E01|sum_penalized|10||0.983333|0.983323|0.983333|30|
|G01|E01|svc_predict|0||0.983333|0.983323|0.983333|30|
|G01|E03|identity|0||0.800000|0.798475|0.800000|15|
|G01|E03|mean_free_intercept|1||0.741111|0.724128|0.741111|45|
|G01|E03|mean_free_intercept|3||0.791111|0.788364|0.791111|45|
|G01|E03|mean_free_intercept|5||0.798519|0.796316|0.798519|45|
|G01|E03|mean_free_intercept|10||0.800000|0.798023|0.800000|15|
|G01|E03|mean_penalized|1||0.750741|0.735301|0.750741|45|
|G01|E03|mean_penalized|3||0.796667|0.794947|0.796667|45|
|G01|E03|mean_penalized|5||0.799630|0.797547|0.799630|45|
|G01|E03|mean_penalized|10||0.800000|0.798023|0.800000|15|
|G01|E03|sum_free_intercept|1||0.693704|0.670820|0.693704|45|
|G01|E03|sum_free_intercept|3||0.745185|0.734677|0.745185|45|
|G01|E03|sum_free_intercept|5||0.776296|0.770550|0.776296|45|
|G01|E03|sum_free_intercept|10||0.788889|0.784699|0.788889|15|
|G01|E03|sum_penalized|1||0.695926|0.673379|0.695926|45|
|G01|E03|sum_penalized|3||0.746296|0.735754|0.746296|45|
|G01|E03|sum_penalized|5||0.776667|0.770945|0.776667|45|
|G01|E03|sum_penalized|10||0.787778|0.783517|0.787778|15|
|G01|E03|svc_predict|0||0.800000|0.798475|0.800000|15|
|G02|E01|cyclic_support|1||0.983333|0.983295|0.983333|30|
|G02|E01|cyclic_support|3||0.978333|0.978320|0.978333|30|
|G02|E01|cyclic_support|5||0.965833|0.965750|0.965833|30|
|G02|E01|cyclic_support|10||0.925833|0.924779|0.925833|30|
|G02|E01|identity|0||0.983333|0.983323|0.983333|30|
|G02|E01|true_support|1||0.983333|0.983323|0.983333|30|
|G02|E01|true_support|3||0.983333|0.983323|0.983333|30|
|G02|E01|true_support|5||0.983333|0.983323|0.983333|30|
|G02|E01|true_support|10||0.983333|0.983323|0.983333|30|
|G02|E01|unlabeled_statistics|1||0.983333|0.983323|0.983333|30|
|G02|E01|unlabeled_statistics|3||0.983333|0.983323|0.983333|30|
|G02|E01|unlabeled_statistics|5||0.983333|0.983323|0.983333|30|
|G02|E01|unlabeled_statistics|10||0.983333|0.983323|0.983333|30|
|G02|E03|cyclic_support|1||0.801111|0.799036|0.801111|15|
|G02|E03|cyclic_support|3||0.790000|0.789076|0.790000|15|
|G02|E03|cyclic_support|5||0.770000|0.770008|0.770000|15|
|G02|E03|cyclic_support|10||0.733333|0.731808|0.733333|15|
|G02|E03|identity|0||0.800000|0.798475|0.800000|15|
|G02|E03|true_support|1||0.800000|0.798441|0.800000|15|
|G02|E03|true_support|3||0.806667|0.805168|0.806667|15|
|G02|E03|true_support|5||0.821111|0.819859|0.821111|15|
|G02|E03|true_support|10||0.838889|0.837811|0.838889|15|
|G02|E03|unlabeled_statistics|1||0.800000|0.798647|0.800000|15|
|G02|E03|unlabeled_statistics|3||0.800000|0.798647|0.800000|15|
|G02|E03|unlabeled_statistics|5||0.801111|0.799661|0.801111|15|
|G02|E03|unlabeled_statistics|10||0.801111|0.799565|0.801111|15|
|G03|E01|full_pipeline_fixedgamma|10|0%|0.983333|0.983323|0.983333|30|
|G03|E01|full_pipeline_fixedgamma|10|10%|0.986167|0.986158|0.986167|150|
|G03|E01|full_pipeline_fixedgamma|10|20%|0.985333|0.985320|0.985333|150|
|G03|E01|full_pipeline_fixedgamma|10|40%|0.984500|0.984468|0.984500|150|
|G03|E01|full_pipeline_fixedgamma|10|100%|0.925833|0.924779|0.925833|30|
|G03|E01|identity|0|0%|0.983333|0.983323|0.983333|30|
|G03|E01|raw_anchor_lambda1|10|0%|0.983333|0.983323|0.983333|30|
|G03|E01|raw_anchor_lambda1|10|10%|0.983333|0.983323|0.983333|150|
|G03|E01|raw_anchor_lambda1|10|20%|0.983333|0.983323|0.983333|150|
|G03|E01|raw_anchor_lambda1|10|40%|0.983333|0.983323|0.983333|150|
|G03|E01|raw_anchor_lambda1|10|100%|0.016667|0.016260|0.016667|30|
|G03|E01|svc_predict|0|0%|0.983333|0.983323|0.983333|30|
|G03|E01|weak_lambda1000_blend005|10|0%|0.983333|0.983323|0.983333|30|
|G03|E01|weak_lambda1000_blend005|10|10%|0.983333|0.983323|0.983333|150|
|G03|E01|weak_lambda1000_blend005|10|20%|0.983333|0.983323|0.983333|150|
|G03|E01|weak_lambda1000_blend005|10|40%|0.983333|0.983323|0.983333|150|
|G03|E01|weak_lambda1000_blend005|10|100%|0.983333|0.983323|0.983333|30|
|G03|E03|full_pipeline_fixedgamma|10|0%|0.838889|0.837811|0.838889|15|
|G03|E03|full_pipeline_fixedgamma|10|10%|0.826444|0.825694|0.826444|75|
|G03|E03|full_pipeline_fixedgamma|10|20%|0.819333|0.818479|0.819333|75|
|G03|E03|full_pipeline_fixedgamma|10|40%|0.800889|0.799924|0.800889|75|
|G03|E03|full_pipeline_fixedgamma|10|100%|0.733333|0.731808|0.733333|15|
|G03|E03|identity|0|0%|0.800000|0.798475|0.800000|15|
|G03|E03|raw_anchor_lambda1|10|0%|0.787778|0.783517|0.787778|15|
|G03|E03|raw_anchor_lambda1|10|10%|0.788889|0.784903|0.788889|75|
|G03|E03|raw_anchor_lambda1|10|20%|0.775111|0.771690|0.775111|75|
|G03|E03|raw_anchor_lambda1|10|40%|0.662667|0.648516|0.662667|75|
|G03|E03|raw_anchor_lambda1|10|100%|0.064444|0.059684|0.064444|15|
|G03|E03|svc_predict|0|0%|0.800000|0.798475|0.800000|15|
|G03|E03|weak_lambda1000_blend005|10|0%|0.801111|0.799554|0.801111|15|
|G03|E03|weak_lambda1000_blend005|10|10%|0.801111|0.799554|0.801111|75|
|G03|E03|weak_lambda1000_blend005|10|20%|0.801111|0.799554|0.801111|75|
|G03|E03|weak_lambda1000_blend005|10|40%|0.801111|0.799554|0.801111|75|
|G03|E03|weak_lambda1000_blend005|10|100%|0.802222|0.800823|0.802222|15|
|G04|E01|class_weight_balanced|0||0.987917|0.987897|0.987917|240|
|G04|E01|class_weight_balanced_argmax|0||0.987917|0.987897|0.987917|240|
|G04|E01|class_weight_none|0||0.987917|0.987897|0.987917|240|
|G04|E01|class_weight_none_argmax|0||0.987917|0.987897|0.987917|240|

## 配对变化与描述性区间

全部方法、剂量、交互、逐类Recall变化及纠错/误伤在CSV中保留。下面列全体匹配Query的对照，按physical_sample_id聚类，不把支持draw、污染掩码或重复外折行当独立观察。

|块|原任务|比较|K/类|Support污染比例|ΔAccuracy百分点及95%区间|纠错/误伤|预测行/实物|
|---|---|---|---:|---:|---:|---:|---:|
|G01|E01|sum_penalized_minus_identity|1||+0.056 [+0.000,+0.217]|2/0|3600/261|
|G01|E01|mean_penalized_minus_identity|1||+0.028 [+0.000,+0.108]|1/0|3600/261|
|G01|E01|sum_free_intercept_minus_identity|1||+0.056 [+0.000,+0.217]|2/0|3600/261|
|G01|E01|mean_free_intercept_minus_identity|1||+0.056 [+0.000,+0.217]|2/0|3600/261|
|G01|E01|mean_minus_sum_penalized_intercept|1||-0.028 [-0.108,+0.000]|0/1|3600/261|
|G01|E01|mean_minus_sum_free_intercept|1||+0.000 [+0.000,+0.000]|0/0|3600/261|
|G01|E01|sum_free_minus_penalized_intercept|1||+0.000 [+0.000,+0.000]|0/0|3600/261|
|G01|E01|mean_free_minus_penalized_intercept|1||+0.028 [+0.000,+0.108]|1/0|3600/261|
|G01|E01|sum_penalized_minus_identity|3||+0.000 [+0.000,+0.000]|0/0|3600/261|
|G01|E01|mean_penalized_minus_identity|3||+0.000 [+0.000,+0.000]|0/0|3600/261|
|G01|E01|sum_free_intercept_minus_identity|3||+0.000 [+0.000,+0.000]|0/0|3600/261|
|G01|E01|mean_free_intercept_minus_identity|3||+0.000 [+0.000,+0.000]|0/0|3600/261|
|G01|E01|mean_minus_sum_penalized_intercept|3||+0.000 [+0.000,+0.000]|0/0|3600/261|
|G01|E01|mean_minus_sum_free_intercept|3||+0.000 [+0.000,+0.000]|0/0|3600/261|
|G01|E01|sum_free_minus_penalized_intercept|3||+0.000 [+0.000,+0.000]|0/0|3600/261|
|G01|E01|mean_free_minus_penalized_intercept|3||+0.000 [+0.000,+0.000]|0/0|3600/261|
|G01|E01|sum_penalized_minus_identity|5||+0.000 [+0.000,+0.000]|0/0|3600/261|
|G01|E01|mean_penalized_minus_identity|5||+0.000 [+0.000,+0.000]|0/0|3600/261|
|G01|E01|sum_free_intercept_minus_identity|5||+0.000 [+0.000,+0.000]|0/0|3600/261|
|G01|E01|mean_free_intercept_minus_identity|5||+0.000 [+0.000,+0.000]|0/0|3600/261|
|G01|E01|mean_minus_sum_penalized_intercept|5||+0.000 [+0.000,+0.000]|0/0|3600/261|
|G01|E01|mean_minus_sum_free_intercept|5||+0.000 [+0.000,+0.000]|0/0|3600/261|
|G01|E01|sum_free_minus_penalized_intercept|5||+0.000 [+0.000,+0.000]|0/0|3600/261|
|G01|E01|mean_free_minus_penalized_intercept|5||+0.000 [+0.000,+0.000]|0/0|3600/261|
|G01|E01|sum_penalized_minus_identity|10||+0.000 [+0.000,+0.000]|0/0|1200/261|
|G01|E01|mean_penalized_minus_identity|10||+0.000 [+0.000,+0.000]|0/0|1200/261|
|G01|E01|sum_free_intercept_minus_identity|10||+0.000 [+0.000,+0.000]|0/0|1200/261|
|G01|E01|mean_free_intercept_minus_identity|10||+0.000 [+0.000,+0.000]|0/0|1200/261|
|G01|E01|mean_minus_sum_penalized_intercept|10||+0.000 [+0.000,+0.000]|0/0|1200/261|
|G01|E01|mean_minus_sum_free_intercept|10||+0.000 [+0.000,+0.000]|0/0|1200/261|
|G01|E01|sum_free_minus_penalized_intercept|10||+0.000 [+0.000,+0.000]|0/0|1200/261|
|G01|E01|mean_free_minus_penalized_intercept|10||+0.000 [+0.000,+0.000]|0/0|1200/261|
|G01|E01|score_argmax_minus_native_svc_zero_support|0||+0.000 [+0.000,+0.000]|0/0|1200/261|
|G01|E03|sum_penalized_minus_identity|1||-10.407 [-12.444,-8.296]|73/354|2700/300|
|G01|E03|mean_penalized_minus_identity|1||-4.926 [-6.482,-3.259]|63/196|2700/300|
|G01|E03|sum_free_intercept_minus_identity|1||-10.630 [-12.594,-8.518]|72/359|2700/300|
|G01|E03|mean_free_intercept_minus_identity|1||-5.889 [-7.519,-4.073]|69/228|2700/300|
|G01|E03|mean_minus_sum_penalized_intercept|1||+5.481 [+4.369,+6.593]|178/30|2700/300|
|G01|E03|mean_minus_sum_free_intercept|1||+4.741 [+3.741,+5.778]|154/26|2700/300|
|G01|E03|sum_free_minus_penalized_intercept|1||-0.222 [-0.481,+0.037]|4/10|2700/300|
|G01|E03|mean_free_minus_penalized_intercept|1||-0.963 [-1.444,-0.444]|7/33|2700/300|
|G01|E03|sum_penalized_minus_identity|3||-5.370 [-7.074,-3.592]|59/204|2700/300|
|G01|E03|mean_penalized_minus_identity|3||-0.333 [-1.000,+0.333]|19/28|2700/300|
|G01|E03|sum_free_intercept_minus_identity|3||-5.481 [-7.186,-3.704]|57/205|2700/300|
|G01|E03|mean_free_intercept_minus_identity|3||-0.889 [-1.594,-0.148]|20/44|2700/300|
|G01|E03|mean_minus_sum_penalized_intercept|3||+5.037 [+3.519,+6.630]|181/45|2700/300|
|G01|E03|mean_minus_sum_free_intercept|3||+4.593 [+3.111,+6.149]|174/50|2700/300|
|G01|E03|sum_free_minus_penalized_intercept|3||-0.111 [-0.259,+0.037]|1/4|2700/300|
|G01|E03|mean_free_minus_penalized_intercept|3||-0.556 [-0.926,-0.222]|1/16|2700/300|
|G01|E03|sum_penalized_minus_identity|5||-2.333 [-3.556,-1.148]|41/104|2700/300|
|G01|E03|mean_penalized_minus_identity|5||-0.037 [-0.630,+0.556]|15/16|2700/300|
|G01|E03|sum_free_intercept_minus_identity|5||-2.370 [-3.556,-1.185]|40/104|2700/300|
|G01|E03|mean_free_intercept_minus_identity|5||-0.148 [-0.815,+0.481]|16/20|2700/300|
|G01|E03|mean_minus_sum_penalized_intercept|5||+2.296 [+1.296,+3.371]|92/30|2700/300|
|G01|E03|mean_minus_sum_free_intercept|5||+2.222 [+1.259,+3.296]|90/30|2700/300|
|G01|E03|sum_free_minus_penalized_intercept|5||-0.037 [-0.111,+0.000]|0/1|2700/300|
|G01|E03|mean_free_minus_penalized_intercept|5||-0.111 [-0.296,+0.037]|1/4|2700/300|
|G01|E03|sum_penalized_minus_identity|10||-1.222 [-2.556,+0.000]|9/20|900/300|
|G01|E03|mean_penalized_minus_identity|10||+0.000 [-0.667,+0.667]|4/4|900/300|
|G01|E03|sum_free_intercept_minus_identity|10||-1.111 [-2.444,+0.114]|10/20|900/300|
|G01|E03|mean_free_intercept_minus_identity|10||+0.000 [-0.667,+0.667]|4/4|900/300|
|G01|E03|mean_minus_sum_penalized_intercept|10||+1.222 [+0.222,+2.225]|16/5|900/300|
|G01|E03|mean_minus_sum_free_intercept|10||+1.111 [+0.111,+2.111]|16/6|900/300|
|G01|E03|sum_free_minus_penalized_intercept|10||+0.111 [+0.000,+0.333]|1/0|900/300|
|G01|E03|mean_free_minus_penalized_intercept|10||+0.000 [+0.000,+0.000]|0/0|900/300|
|G01|E03|score_argmax_minus_native_svc_zero_support|0||+0.000 [+0.000,+0.000]|0/0|900/300|
|G02|E01|unlabeled_statistics_minus_source_identity|1||+0.000 [+0.000,+0.000]|0/0|1200/261|
|G02|E01|true_support_minus_source_identity|1||+0.000 [+0.000,+0.000]|0/0|1200/261|
|G02|E01|cyclic_support_minus_source_identity|1||+0.000 [-2.265,+2.457]|10/10|1200/261|
|G02|E01|true_labels_minus_unlabeled_statistics|1||+0.000 [+0.000,+0.000]|0/0|1200/261|
|G02|E01|cyclic_labels_minus_true_labels|1||+0.000 [-2.265,+2.457]|10/10|1200/261|
|G02|E01|cyclic_labels_minus_unlabeled_statistics|1||+0.000 [-2.265,+2.457]|10/10|1200/261|
|G02|E01|unlabeled_statistics_minus_source_identity|3||+0.000 [+0.000,+0.000]|0/0|1200/261|
|G02|E01|true_support_minus_source_identity|3||+0.000 [+0.000,+0.000]|0/0|1200/261|
|G02|E01|cyclic_support_minus_source_identity|3||-0.500 [-1.913,+1.683]|9/15|1200/261|
|G02|E01|true_labels_minus_unlabeled_statistics|3||+0.000 [+0.000,+0.000]|0/0|1200/261|
|G02|E01|cyclic_labels_minus_true_labels|3||-0.500 [-1.913,+1.683]|9/15|1200/261|
|G02|E01|cyclic_labels_minus_unlabeled_statistics|3||-0.500 [-1.913,+1.683]|9/15|1200/261|
|G02|E01|unlabeled_statistics_minus_source_identity|5||+0.000 [+0.000,+0.000]|0/0|1200/261|
|G02|E01|true_support_minus_source_identity|5||+0.000 [+0.000,+0.000]|0/0|1200/261|
|G02|E01|cyclic_support_minus_source_identity|5||-1.750 [-3.192,-0.159]|6/27|1200/261|
|G02|E01|true_labels_minus_unlabeled_statistics|5||+0.000 [+0.000,+0.000]|0/0|1200/261|
|G02|E01|cyclic_labels_minus_true_labels|5||-1.750 [-3.192,-0.159]|6/27|1200/261|
|G02|E01|cyclic_labels_minus_unlabeled_statistics|5||-1.750 [-3.192,-0.159]|6/27|1200/261|
|G02|E01|unlabeled_statistics_minus_source_identity|10||+0.000 [+0.000,+0.000]|0/0|1200/261|
|G02|E01|true_support_minus_source_identity|10||+0.000 [+0.000,+0.000]|0/0|1200/261|
|G02|E01|cyclic_support_minus_source_identity|10||-5.750 [-8.193,-3.609]|6/75|1200/261|
|G02|E01|true_labels_minus_unlabeled_statistics|10||+0.000 [+0.000,+0.000]|0/0|1200/261|
|G02|E01|cyclic_labels_minus_true_labels|10||-5.750 [-8.193,-3.609]|6/75|1200/261|
|G02|E01|cyclic_labels_minus_unlabeled_statistics|10||-5.750 [-8.193,-3.609]|6/75|1200/261|
|G02|E03|unlabeled_statistics_minus_source_identity|1||+0.000 [-0.333,+0.333]|1/1|900/300|
|G02|E03|true_support_minus_source_identity|1||+0.000 [-0.889,+0.889]|8/8|900/300|
|G02|E03|cyclic_support_minus_source_identity|1||+0.111 [-0.889,+1.111]|11/10|900/300|
|G02|E03|true_labels_minus_unlabeled_statistics|1||+0.000 [-0.889,+0.889]|8/8|900/300|
|G02|E03|cyclic_labels_minus_true_labels|1||+0.111 [-1.000,+1.333]|16/15|900/300|
|G02|E03|cyclic_labels_minus_unlabeled_statistics|1||+0.111 [-0.781,+1.111]|11/10|900/300|
|G02|E03|unlabeled_statistics_minus_source_identity|3||+0.000 [-0.333,+0.333]|1/1|900/300|
|G02|E03|true_support_minus_source_identity|3||+0.667 [-0.556,+1.781]|20/14|900/300|
|G02|E03|cyclic_support_minus_source_identity|3||-1.000 [-2.667,+0.667]|19/28|900/300|
|G02|E03|true_labels_minus_unlabeled_statistics|3||+0.667 [-0.444,+1.778]|20/14|900/300|
|G02|E03|cyclic_labels_minus_true_labels|3||-1.667 [-3.778,+0.444]|28/43|900/300|
|G02|E03|cyclic_labels_minus_unlabeled_statistics|3||-1.000 [-2.778,+0.667]|19/28|900/300|
|G02|E03|unlabeled_statistics_minus_source_identity|5||+0.111 [+0.000,+0.336]|1/0|900/300|
|G02|E03|true_support_minus_source_identity|5||+2.111 [+0.889,+3.556]|28/9|900/300|
|G02|E03|cyclic_support_minus_source_identity|5||-3.000 [-4.892,-1.108]|22/49|900/300|
|G02|E03|true_labels_minus_unlabeled_statistics|5||+2.000 [+0.778,+3.333]|27/9|900/300|
|G02|E03|cyclic_labels_minus_true_labels|5||-5.111 [-7.333,-2.889]|20/66|900/300|
|G02|E03|cyclic_labels_minus_unlabeled_statistics|5||-3.111 [-5.111,-1.111]|22/50|900/300|
|G02|E03|unlabeled_statistics_minus_source_identity|10||+0.111 [-0.222,+0.444]|2/1|900/300|
|G02|E03|true_support_minus_source_identity|10||+3.889 [+2.219,+6.000]|49/14|900/300|
|G02|E03|cyclic_support_minus_source_identity|10||-6.667 [-9.222,-4.108]|25/85|900/300|
|G02|E03|true_labels_minus_unlabeled_statistics|10||+3.778 [+2.111,+5.889]|49/15|900/300|
|G02|E03|cyclic_labels_minus_true_labels|10||-10.556 [-13.667,-7.667]|25/120|900/300|
|G02|E03|cyclic_labels_minus_unlabeled_statistics|10||-6.778 [-9.333,-4.222]|24/85|900/300|
|G03|E01|raw_anchor_lambda1_minus_matched_source_control|10|0%|+0.000 [+0.000,+0.000]|0/0|1200/261|
|G03|E01|raw_anchor_lambda1_noise_minus_true_support|10|0%|+0.000 [+0.000,+0.000]|0/0|1200/261|
|G03|E01|raw_anchor_lambda1_minus_matched_source_control|10|10%|+0.000 [+0.000,+0.000]|0/0|6000/261|
|G03|E01|raw_anchor_lambda1_noise_minus_true_support|10|10%|+0.000 [+0.000,+0.000]|0/0|6000/261|
|G03|E01|raw_anchor_lambda1_minus_matched_source_control|10|20%|+0.000 [+0.000,+0.000]|0/0|6000/261|
|G03|E01|raw_anchor_lambda1_noise_minus_true_support|10|20%|+0.000 [+0.000,+0.000]|0/0|6000/261|
|G03|E01|raw_anchor_lambda1_minus_matched_source_control|10|40%|+0.000 [+0.000,+0.000]|0/0|6000/261|
|G03|E01|raw_anchor_lambda1_noise_minus_true_support|10|40%|+0.000 [+0.000,+0.000]|0/0|6000/261|
|G03|E01|raw_anchor_lambda1_minus_matched_source_control|10|100%|-96.667 [-100.000,-91.363]|20/1180|1200/261|
|G03|E01|raw_anchor_lambda1_noise_minus_true_support|10|100%|-96.667 [-100.000,-91.363]|20/1180|1200/261|
|G03|E01|weak_lambda1000_blend005_minus_matched_source_control|10|0%|+0.000 [+0.000,+0.000]|0/0|1200/261|
|G03|E01|weak_lambda1000_blend005_noise_minus_true_support|10|0%|+0.000 [+0.000,+0.000]|0/0|1200/261|
|G03|E01|weak_lambda1000_blend005_minus_matched_source_control|10|10%|+0.000 [+0.000,+0.000]|0/0|6000/261|
|G03|E01|weak_lambda1000_blend005_noise_minus_true_support|10|10%|+0.000 [+0.000,+0.000]|0/0|6000/261|
|G03|E01|weak_lambda1000_blend005_minus_matched_source_control|10|20%|+0.000 [+0.000,+0.000]|0/0|6000/261|
|G03|E01|weak_lambda1000_blend005_noise_minus_true_support|10|20%|+0.000 [+0.000,+0.000]|0/0|6000/261|
|G03|E01|weak_lambda1000_blend005_minus_matched_source_control|10|40%|+0.000 [+0.000,+0.000]|0/0|6000/261|
|G03|E01|weak_lambda1000_blend005_noise_minus_true_support|10|40%|+0.000 [+0.000,+0.000]|0/0|6000/261|
|G03|E01|weak_lambda1000_blend005_minus_matched_source_control|10|100%|+0.000 [+0.000,+0.000]|0/0|1200/261|
|G03|E01|weak_lambda1000_blend005_noise_minus_true_support|10|100%|+0.000 [+0.000,+0.000]|0/0|1200/261|
|G03|E01|full_pipeline_fixedgamma_minus_matched_source_control|10|0%|+0.000 [+0.000,+0.000]|0/0|1200/261|
|G03|E01|full_pipeline_fixedgamma_noise_minus_true_support|10|0%|+0.000 [+0.000,+0.000]|0/0|1200/261|
|G03|E01|full_pipeline_fixedgamma_minus_matched_source_control|10|10%|+0.283 [-0.100,+1.246]|20/3|6000/261|
|G03|E01|full_pipeline_fixedgamma_noise_minus_true_support|10|10%|+0.283 [-0.100,+1.246]|20/3|6000/261|
|G03|E01|full_pipeline_fixedgamma_minus_matched_source_control|10|20%|+0.200 [-0.480,+1.436]|27/15|6000/261|
|G03|E01|full_pipeline_fixedgamma_noise_minus_true_support|10|20%|+0.200 [-0.480,+1.436]|27/15|6000/261|
|G03|E01|full_pipeline_fixedgamma_minus_matched_source_control|10|40%|+0.117 [-0.763,+1.730]|38/31|6000/261|
|G03|E01|full_pipeline_fixedgamma_noise_minus_true_support|10|40%|+0.117 [-0.763,+1.730]|38/31|6000/261|
|G03|E01|full_pipeline_fixedgamma_minus_matched_source_control|10|100%|-5.750 [-8.193,-3.609]|6/75|1200/261|
|G03|E01|full_pipeline_fixedgamma_noise_minus_true_support|10|100%|-5.750 [-8.193,-3.609]|6/75|1200/261|
|G03|E01|weak_lambda1000_blend005_minus_raw_anchor_lambda1|10|0%|+0.000 [+0.000,+0.000]|0/0|1200/261|
|G03|E01|full_pipeline_fixedgamma_minus_raw_anchor_lambda1|10|0%|+0.000 [+0.000,+0.000]|0/0|1200/261|
|G03|E01|full_pipeline_fixedgamma_minus_weak_lambda1000_blend005|10|0%|+0.000 [+0.000,+0.000]|0/0|1200/261|
|G03|E01|weak_lambda1000_blend005_minus_raw_anchor_lambda1|10|10%|+0.000 [+0.000,+0.000]|0/0|6000/261|
|G03|E01|full_pipeline_fixedgamma_minus_raw_anchor_lambda1|10|10%|+0.283 [-0.100,+1.246]|20/3|6000/261|
|G03|E01|full_pipeline_fixedgamma_minus_weak_lambda1000_blend005|10|10%|+0.283 [-0.100,+1.246]|20/3|6000/261|
|G03|E01|weak_lambda1000_blend005_minus_raw_anchor_lambda1|10|20%|+0.000 [+0.000,+0.000]|0/0|6000/261|
|G03|E01|full_pipeline_fixedgamma_minus_raw_anchor_lambda1|10|20%|+0.200 [-0.480,+1.436]|27/15|6000/261|
|G03|E01|full_pipeline_fixedgamma_minus_weak_lambda1000_blend005|10|20%|+0.200 [-0.480,+1.436]|27/15|6000/261|
|G03|E01|weak_lambda1000_blend005_minus_raw_anchor_lambda1|10|40%|+0.000 [+0.000,+0.000]|0/0|6000/261|
|G03|E01|full_pipeline_fixedgamma_minus_raw_anchor_lambda1|10|40%|+0.117 [-0.763,+1.730]|38/31|6000/261|
|G03|E01|full_pipeline_fixedgamma_minus_weak_lambda1000_blend005|10|40%|+0.117 [-0.763,+1.730]|38/31|6000/261|
|G03|E01|weak_lambda1000_blend005_minus_raw_anchor_lambda1|10|100%|+96.667 [+91.363,+100.000]|1180/20|1200/261|
|G03|E01|full_pipeline_fixedgamma_minus_raw_anchor_lambda1|10|100%|+90.917 [+85.866,+94.509]|1105/14|1200/261|
|G03|E01|full_pipeline_fixedgamma_minus_weak_lambda1000_blend005|10|100%|-5.750 [-8.193,-3.609]|6/75|1200/261|
|G03|E01|score_argmax_minus_native_svc_zero_support|0||+0.000 [+0.000,+0.000]|0/0|1200/261|
|G03|E03|raw_anchor_lambda1_minus_matched_source_control|10|0%|-1.222 [-2.556,+0.000]|9/20|900/300|
|G03|E03|raw_anchor_lambda1_noise_minus_true_support|10|0%|+0.000 [+0.000,+0.000]|0/0|900/300|
|G03|E03|raw_anchor_lambda1_minus_matched_source_control|10|10%|-1.111 [-2.201,-0.044]|46/96|4500/300|
|G03|E03|raw_anchor_lambda1_noise_minus_true_support|10|10%|+0.111 [-0.556,+0.867]|53/48|4500/300|
|G03|E03|raw_anchor_lambda1_minus_matched_source_control|10|20%|-2.489 [-3.778,-1.289]|55/167|4500/300|
|G03|E03|raw_anchor_lambda1_noise_minus_true_support|10|20%|-1.267 [-2.356,-0.044]|80/137|4500/300|
|G03|E03|raw_anchor_lambda1_minus_matched_source_control|10|40%|-13.733 [-16.578,-10.866]|101/719|4500/300|
|G03|E03|raw_anchor_lambda1_noise_minus_true_support|10|40%|-12.511 [-15.333,-9.600]|149/712|4500/300|
|G03|E03|raw_anchor_lambda1_minus_matched_source_control|10|100%|-73.556 [-78.333,-68.667]|37/699|900/300|
|G03|E03|raw_anchor_lambda1_noise_minus_true_support|10|100%|-72.333 [-77.333,-67.331]|51/702|900/300|
|G03|E03|weak_lambda1000_blend005_minus_matched_source_control|10|0%|+0.111 [+0.000,+0.333]|1/0|900/300|
|G03|E03|weak_lambda1000_blend005_noise_minus_true_support|10|0%|+0.000 [+0.000,+0.000]|0/0|900/300|
|G03|E03|weak_lambda1000_blend005_minus_matched_source_control|10|10%|+0.111 [+0.000,+0.333]|5/0|4500/300|
|G03|E03|weak_lambda1000_blend005_noise_minus_true_support|10|10%|+0.000 [+0.000,+0.000]|0/0|4500/300|
|G03|E03|weak_lambda1000_blend005_minus_matched_source_control|10|20%|+0.111 [+0.000,+0.333]|5/0|4500/300|
|G03|E03|weak_lambda1000_blend005_noise_minus_true_support|10|20%|+0.000 [+0.000,+0.000]|0/0|4500/300|
|G03|E03|weak_lambda1000_blend005_minus_matched_source_control|10|40%|+0.111 [+0.000,+0.333]|5/0|4500/300|
|G03|E03|weak_lambda1000_blend005_noise_minus_true_support|10|40%|+0.000 [+0.000,+0.000]|0/0|4500/300|
|G03|E03|weak_lambda1000_blend005_minus_matched_source_control|10|100%|+0.222 [+0.000,+0.556]|2/0|900/300|
|G03|E03|weak_lambda1000_blend005_noise_minus_true_support|10|100%|+0.111 [+0.000,+0.336]|1/0|900/300|
|G03|E03|full_pipeline_fixedgamma_minus_matched_source_control|10|0%|+3.889 [+2.219,+6.000]|49/14|900/300|
|G03|E03|full_pipeline_fixedgamma_noise_minus_true_support|10|0%|+0.000 [+0.000,+0.000]|0/0|900/300|
|G03|E03|full_pipeline_fixedgamma_minus_matched_source_control|10|10%|+2.644 [+1.222,+4.201]|209/90|4500/300|
|G03|E03|full_pipeline_fixedgamma_noise_minus_true_support|10|10%|-1.244 [-2.178,-0.355]|44/100|4500/300|
|G03|E03|full_pipeline_fixedgamma_minus_matched_source_control|10|20%|+1.933 [+0.600,+3.467]|204/117|4500/300|
|G03|E03|full_pipeline_fixedgamma_noise_minus_true_support|10|20%|-1.956 [-3.311,-0.711]|70/158|4500/300|
|G03|E03|full_pipeline_fixedgamma_minus_matched_source_control|10|40%|+0.089 [-1.334,+1.489]|184/180|4500/300|
|G03|E03|full_pipeline_fixedgamma_noise_minus_true_support|10|40%|-3.800 [-5.556,-2.288]|86/257|4500/300|
|G03|E03|full_pipeline_fixedgamma_minus_matched_source_control|10|100%|-6.667 [-9.222,-4.108]|25/85|900/300|
|G03|E03|full_pipeline_fixedgamma_noise_minus_true_support|10|100%|-10.556 [-13.667,-7.667]|25/120|900/300|
|G03|E03|weak_lambda1000_blend005_minus_raw_anchor_lambda1|10|0%|+1.333 [+0.111,+2.556]|20/8|900/300|
|G03|E03|full_pipeline_fixedgamma_minus_raw_anchor_lambda1|10|0%|+5.111 [+3.000,+7.444]|64/18|900/300|
|G03|E03|full_pipeline_fixedgamma_minus_weak_lambda1000_blend005|10|0%|+3.778 [+2.111,+5.669]|48/14|900/300|
|G03|E03|weak_lambda1000_blend005_minus_raw_anchor_lambda1|10|10%|+1.222 [+0.178,+2.289]|96/41|4500/300|
|G03|E03|full_pipeline_fixedgamma_minus_raw_anchor_lambda1|10|10%|+3.756 [+2.156,+5.511]|278/109|4500/300|
|G03|E03|full_pipeline_fixedgamma_minus_weak_lambda1000_blend005|10|10%|+2.533 [+1.133,+4.133]|207/93|4500/300|
|G03|E03|weak_lambda1000_blend005_minus_raw_anchor_lambda1|10|20%|+2.600 [+1.422,+3.867]|167/50|4500/300|
|G03|E03|full_pipeline_fixedgamma_minus_raw_anchor_lambda1|10|20%|+4.422 [+2.844,+6.156]|333/134|4500/300|
|G03|E03|full_pipeline_fixedgamma_minus_weak_lambda1000_blend005|10|20%|+1.822 [+0.489,+3.313]|202/120|4500/300|
|G03|E03|weak_lambda1000_blend005_minus_raw_anchor_lambda1|10|40%|+13.844 [+11.022,+16.646]|722/99|4500/300|
|G03|E03|full_pipeline_fixedgamma_minus_raw_anchor_lambda1|10|40%|+13.822 [+10.977,+16.422]|835/213|4500/300|
|G03|E03|full_pipeline_fixedgamma_minus_weak_lambda1000_blend005|10|40%|-0.022 [-1.489,+1.423]|183/184|4500/300|
|G03|E03|weak_lambda1000_blend005_minus_raw_anchor_lambda1|10|100%|+73.778 [+69.000,+78.556]|700/36|900/300|
|G03|E03|full_pipeline_fixedgamma_minus_raw_anchor_lambda1|10|100%|+66.889 [+62.111,+71.225]|626/24|900/300|
|G03|E03|full_pipeline_fixedgamma_minus_weak_lambda1000_blend005|10|100%|-6.889 [-9.556,-4.219]|25/87|900/300|
|G03|E03|score_argmax_minus_native_svc_zero_support|0||+0.000 [+0.000,+0.000]|0/0|900/300|
|G04|E01|unweighted_minus_balanced|0||+0.000 [+0.000,+0.000]|0/0|1600/402|
|G04|E01|unweighted_minus_balanced|0||+0.000 [+0.000,+0.000]|0/0|1600/402|
|G04|E01|unweighted_minus_balanced|0||+0.000 [+0.000,+0.000]|0/0|1600/402|
|G04|E01|unweighted_minus_balanced|0||+0.000 [+0.000,+0.000]|0/0|1600/402|
|G04|E01|unweighted_minus_balanced|0||+0.000 [+0.000,+0.000]|0/0|1600/402|
|G04|E01|unweighted_minus_balanced|0||+0.000 [+0.000,+0.000]|0/0|1600/402|
|G04|E01|class_weight_balanced_argmax_minus_native|0||+0.000 [+0.000,+0.000]|0/0|9600/402|
|G04|E01|class_weight_none_argmax_minus_native|0||+0.000 [+0.000,+0.000]|0/0|9600/402|

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
