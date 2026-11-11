# B方案第五轮：Source内策略、Support获取与分数信息

状态：**PARTIAL**；已封存 **4/180** 个任务包。
本报告只读取完整封存工件，不拟合或选择新模型，不修改前四轮输入、标签、协议、模型及结果。

新增分类器拟合30次，校准拟合调用90次；科研任务CPU合计2.052478秒。
保存5960条相关外层Query预测，涉及40个唯一Query实物；控制、核验、失败及推送成本另见账本。

Source内部验证预测另计1920/86400条（已封存/冻结计划），不计入外层Query预测数，也不增加独立验证实物。

## 完成范围

|块|问题|完成/计划|状态|
|---|---|---:|---|
|I01|仅用Source内部验证选择校准策略|1/45|PARTIAL|
|I02|隐藏候选标签的Support获取顺序|1/45|PARTIAL|
|I03|保留OVR输出锚的OVO信息校准|1/45|PARTIAL|
|I04|权重ESS与预设收缩|1/45|PARTIAL|

以下角色数直接来自冻结任务中的ID与source_class_counts，未由目标比例反推。

|块|原任务|Source数|Source类别计数|候选Support池数|Query数|冻结上下文数|
|---|---|---:|---|---:|---:|---:|
|I01|E01|240|{"枣仁": 120, "酸枣仁": 120}|20|40|30|
|I01|E03|190|{"ZY03": 38, "ZY04": 38, "ZY05": 38, "ZY06": 38, "ZY07": 38}|50|60|15|
|I02|E01|240|{"枣仁": 120, "酸枣仁": 120}|20|40|30|
|I02|E03|190|{"ZY03": 38, "ZY04": 38, "ZY05": 38, "ZY06": 38, "ZY07": 38}|50|60|15|
|I03|E01|240|{"枣仁": 120, "酸枣仁": 120}|20|40|30|
|I03|E03|190|{"ZY03": 38, "ZY04": 38, "ZY05": 38, "ZY06": 38, "ZY07": 38}|50|60|15|
|I04|E01|240|{"枣仁": 120, "酸枣仁": 120}|20|40|30|
|I04|E03|190|{"ZY03": 38, "ZY04": 38, "ZY05": 38, "ZY06": 38, "ZY07": 38}|50|60|15|

## 设计与解释边界

I01仅在每个外层Source内部构造三次角色划分，每次inner Source、Support和验证样品互斥，外层Support/Query值均不进入选择，外层结果在选择完成后才载入。预设Identity、Bias λ1、mean-Anchor λ=N和冻结坐标完整重训四个候选，按三次内验证平均Macro-F1及预先固定的简洁优先平局规则选择；随后一次应用到原冻结Source和外层Support。不能使用Source模型的训练内预测冒充内验证。最大内验证成绩是选择统计量，不是无偏效果；内层训练量更小，Source内域与外层Support来源也可能不同，策略迁移不保证成功。所有候选和选择后外层结果均报告。

I02固定总标签预算，比较三个固定随机顺序、Source-margin不确定性和在Source坐标中持续保留Source质心锚的max-min覆盖。距离更新同时考虑这一预先存在的Source锚与已选候选点。排序接口不接收候选真标签、语义ID前缀或Query，只用候选谱、Source预测及不含语义的固定hash次序。候选池原本已按类别分层，故这是该池内的标签盲排序实验，并非完全无需标签的实际采集流程。选后揭示标签并记录覆盖；完整SVM仍包含全部Source类别，校准缺类时按预设规则回退Identity，不丢弃条件、不重抽至满意。不同政策同时改变入选实物、类别覆盖和谱位置，不能将差值唯一归因为多样性。回退前已观察的标签仍计入预算，不把回退称零标签成本。

I03保持原OVR分数为输出锚；只把拟合残差的输入特征换为Source中心化并按全局row-L2均方根规范化的OVR、OVO margins和Source-only OVO PCA。所有基与归一化仅由Source得到，输出为原分数加拟合修正，K0决策保持不变；不对原始OVO列直接argmax。固定mean-loss单位先验（sum λ=N），不根据Query选强度。二分类规范化OVR/OVO Gram代数等价，三种嵌入的数值结果是负对照，不算三项独立方法证据；五分类PCA控制维度也改变保留子空间，不能唯一识别维度的单因素效应。

I04完全复用上一轮Support构成与ID，将相同类别均衡权重的残差惩罚从λ1比较到λ=N及预设λ=N²/ESS。ESS=(Σw)²/Σw²是权重集中度，不是新增实物数；ESS公式是冻结的启发式收缩，不是由Query选出的最优λ或新算法。均衡时ESS=N，两个新条件等价并复用G01的mean-Anchor。减少误伤可能只是修正更小、更多保持Source，必须同时看分数运动、类别变化、纠错及净收益。

## 全部冻结条件的描述性结果

|块|原任务|方法|K/类|政策/预算/构成|Accuracy|Macro-F1|Balanced Accuracy|端点数|
|---|---|---|---:|---|---:|---:|---:|---:|
|I01|E01|bias|1||1.000000|1.000000|1.000000|1|
|I01|E01|bias|3||1.000000|1.000000|1.000000|1|
|I01|E01|bias|5||1.000000|1.000000|1.000000|1|
|I01|E01|bias|10||1.000000|1.000000|1.000000|1|
|I01|E01|full_refit|1||1.000000|1.000000|1.000000|1|
|I01|E01|full_refit|3||1.000000|1.000000|1.000000|1|
|I01|E01|full_refit|5||1.000000|1.000000|1.000000|1|
|I01|E01|full_refit|10||1.000000|1.000000|1.000000|1|
|I01|E01|identity|0||1.000000|1.000000|1.000000|1|
|I01|E01|mean_anchor|1||1.000000|1.000000|1.000000|1|
|I01|E01|mean_anchor|3||1.000000|1.000000|1.000000|1|
|I01|E01|mean_anchor|5||1.000000|1.000000|1.000000|1|
|I01|E01|mean_anchor|10||1.000000|1.000000|1.000000|1|
|I01|E01|source_selected|1||1.000000|1.000000|1.000000|1|
|I01|E01|source_selected|3||1.000000|1.000000|1.000000|1|
|I01|E01|source_selected|5||1.000000|1.000000|1.000000|1|
|I01|E01|source_selected|10||1.000000|1.000000|1.000000|1|
|I01|E01|svc_predict|0||1.000000|1.000000|1.000000|1|
|I02|E01|full_refit|不适用|policy=diversity, budget=2.0|1.000000|1.000000|1.000000|1|
|I02|E01|full_refit|不适用|policy=diversity, budget=6.0|1.000000|1.000000|1.000000|1|
|I02|E01|full_refit|不适用|policy=diversity, budget=10.0|1.000000|1.000000|1.000000|1|
|I02|E01|full_refit|不适用|policy=diversity, budget=20.0|1.000000|1.000000|1.000000|1|
|I02|E01|full_refit|不适用|policy=random_601, budget=2.0|1.000000|1.000000|1.000000|1|
|I02|E01|full_refit|不适用|policy=random_601, budget=6.0|1.000000|1.000000|1.000000|1|
|I02|E01|full_refit|不适用|policy=random_601, budget=10.0|1.000000|1.000000|1.000000|1|
|I02|E01|full_refit|不适用|policy=random_601, budget=20.0|1.000000|1.000000|1.000000|1|
|I02|E01|full_refit|不适用|policy=random_602, budget=2.0|1.000000|1.000000|1.000000|1|
|I02|E01|full_refit|不适用|policy=random_602, budget=6.0|1.000000|1.000000|1.000000|1|
|I02|E01|full_refit|不适用|policy=random_602, budget=10.0|1.000000|1.000000|1.000000|1|
|I02|E01|full_refit|不适用|policy=random_602, budget=20.0|1.000000|1.000000|1.000000|1|
|I02|E01|full_refit|不适用|policy=random_603, budget=2.0|1.000000|1.000000|1.000000|1|
|I02|E01|full_refit|不适用|policy=random_603, budget=6.0|1.000000|1.000000|1.000000|1|
|I02|E01|full_refit|不适用|policy=random_603, budget=10.0|1.000000|1.000000|1.000000|1|
|I02|E01|full_refit|不适用|policy=random_603, budget=20.0|1.000000|1.000000|1.000000|1|
|I02|E01|full_refit|不适用|policy=uncertainty, budget=2.0|1.000000|1.000000|1.000000|1|
|I02|E01|full_refit|不适用|policy=uncertainty, budget=6.0|1.000000|1.000000|1.000000|1|
|I02|E01|full_refit|不适用|policy=uncertainty, budget=10.0|1.000000|1.000000|1.000000|1|
|I02|E01|full_refit|不适用|policy=uncertainty, budget=20.0|1.000000|1.000000|1.000000|1|
|I02|E01|guarded_mean_anchor|不适用|policy=diversity, budget=2.0|1.000000|1.000000|1.000000|1|
|I02|E01|guarded_mean_anchor|不适用|policy=diversity, budget=6.0|1.000000|1.000000|1.000000|1|
|I02|E01|guarded_mean_anchor|不适用|policy=diversity, budget=10.0|1.000000|1.000000|1.000000|1|
|I02|E01|guarded_mean_anchor|不适用|policy=diversity, budget=20.0|1.000000|1.000000|1.000000|1|
|I02|E01|guarded_mean_anchor|不适用|policy=random_601, budget=2.0|1.000000|1.000000|1.000000|1|
|I02|E01|guarded_mean_anchor|不适用|policy=random_601, budget=6.0|1.000000|1.000000|1.000000|1|
|I02|E01|guarded_mean_anchor|不适用|policy=random_601, budget=10.0|1.000000|1.000000|1.000000|1|
|I02|E01|guarded_mean_anchor|不适用|policy=random_601, budget=20.0|1.000000|1.000000|1.000000|1|
|I02|E01|guarded_mean_anchor|不适用|policy=random_602, budget=2.0|1.000000|1.000000|1.000000|1|
|I02|E01|guarded_mean_anchor|不适用|policy=random_602, budget=6.0|1.000000|1.000000|1.000000|1|
|I02|E01|guarded_mean_anchor|不适用|policy=random_602, budget=10.0|1.000000|1.000000|1.000000|1|
|I02|E01|guarded_mean_anchor|不适用|policy=random_602, budget=20.0|1.000000|1.000000|1.000000|1|
|I02|E01|guarded_mean_anchor|不适用|policy=random_603, budget=2.0|1.000000|1.000000|1.000000|1|
|I02|E01|guarded_mean_anchor|不适用|policy=random_603, budget=6.0|1.000000|1.000000|1.000000|1|
|I02|E01|guarded_mean_anchor|不适用|policy=random_603, budget=10.0|1.000000|1.000000|1.000000|1|
|I02|E01|guarded_mean_anchor|不适用|policy=random_603, budget=20.0|1.000000|1.000000|1.000000|1|
|I02|E01|guarded_mean_anchor|不适用|policy=uncertainty, budget=2.0|1.000000|1.000000|1.000000|1|
|I02|E01|guarded_mean_anchor|不适用|policy=uncertainty, budget=6.0|1.000000|1.000000|1.000000|1|
|I02|E01|guarded_mean_anchor|不适用|policy=uncertainty, budget=10.0|1.000000|1.000000|1.000000|1|
|I02|E01|guarded_mean_anchor|不适用|policy=uncertainty, budget=20.0|1.000000|1.000000|1.000000|1|
|I02|E01|identity|0|budget=None|1.000000|1.000000|1.000000|1|
|I02|E01|svc_predict|0|budget=None|1.000000|1.000000|1.000000|1|
|I03|E01|identity|0||1.000000|1.000000|1.000000|1|
|I03|E01|ovo_pca_trace1|1||1.000000|1.000000|1.000000|3|
|I03|E01|ovo_pca_trace1|3||1.000000|1.000000|1.000000|3|
|I03|E01|ovo_pca_trace1|5||1.000000|1.000000|1.000000|3|
|I03|E01|ovo_pca_trace1|10||1.000000|1.000000|1.000000|1|
|I03|E01|ovo_trace1|1||1.000000|1.000000|1.000000|3|
|I03|E01|ovo_trace1|3||1.000000|1.000000|1.000000|3|
|I03|E01|ovo_trace1|5||1.000000|1.000000|1.000000|3|
|I03|E01|ovo_trace1|10||1.000000|1.000000|1.000000|1|
|I03|E01|ovr_trace1|1||1.000000|1.000000|1.000000|3|
|I03|E01|ovr_trace1|3||1.000000|1.000000|1.000000|3|
|I03|E01|ovr_trace1|5||1.000000|1.000000|1.000000|3|
|I03|E01|ovr_trace1|10||1.000000|1.000000|1.000000|1|
|I03|E01|raw_mean_anchor|1||1.000000|1.000000|1.000000|3|
|I03|E01|raw_mean_anchor|3||1.000000|1.000000|1.000000|3|
|I03|E01|raw_mean_anchor|5||1.000000|1.000000|1.000000|3|
|I03|E01|raw_mean_anchor|10||1.000000|1.000000|1.000000|1|
|I03|E01|svc_predict|0||1.000000|1.000000|1.000000|1|
|I04|E01|identity|0||1.000000|1.000000|1.000000|1|
|I04|E01|svc_predict|0||1.000000|1.000000|1.000000|1|
|I04|E01|weighted_ess|不适用|composition_id=balanced|1.000000|1.000000|1.000000|3|
|I04|E01|weighted_ess|不适用|composition_id=binary_1_9|1.000000|1.000000|1.000000|3|
|I04|E01|weighted_ess|不适用|composition_id=binary_3_7|1.000000|1.000000|1.000000|3|
|I04|E01|weighted_ess|不适用|composition_id=binary_7_3|1.000000|1.000000|1.000000|3|
|I04|E01|weighted_ess|不适用|composition_id=binary_9_1|1.000000|1.000000|1.000000|3|
|I04|E01|weighted_lambda1|不适用|composition_id=balanced|1.000000|1.000000|1.000000|3|
|I04|E01|weighted_lambda1|不适用|composition_id=binary_1_9|1.000000|1.000000|1.000000|3|
|I04|E01|weighted_lambda1|不适用|composition_id=binary_3_7|1.000000|1.000000|1.000000|3|
|I04|E01|weighted_lambda1|不适用|composition_id=binary_7_3|1.000000|1.000000|1.000000|3|
|I04|E01|weighted_lambda1|不适用|composition_id=binary_9_1|1.000000|1.000000|1.000000|3|
|I04|E01|weighted_mean|不适用|composition_id=balanced|1.000000|1.000000|1.000000|3|
|I04|E01|weighted_mean|不适用|composition_id=binary_1_9|1.000000|1.000000|1.000000|3|
|I04|E01|weighted_mean|不适用|composition_id=binary_3_7|1.000000|1.000000|1.000000|3|
|I04|E01|weighted_mean|不适用|composition_id=binary_7_3|1.000000|1.000000|1.000000|3|
|I04|E01|weighted_mean|不适用|composition_id=binary_9_1|1.000000|1.000000|1.000000|3|

## 配对变化与描述性不确定性

所有正、负和零结果保留。配对以相同外层Query的physical_sample_id对齐，区间保留同实物相关预测；完整来源分层、Recall代价及纠错/误伤另见CSV。

|块|原任务|比较|预算/政策/构成|ΔAccuracy百分点及95%区间|纠错/误伤|预测行/实物|
|---|---|---|---|---:|---:|---:|
|I01|E01|bias_minus_matched_source_control|k=1.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|mean_anchor_minus_matched_source_control|k=1.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|full_refit_minus_matched_source_control|k=1.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|source_selected_minus_matched_source_control|k=1.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|source_selected_minus_bias|k=1.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|source_selected_minus_mean_anchor|k=1.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|source_selected_minus_full_refit|k=1.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|bias_minus_matched_source_control|k=3.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|mean_anchor_minus_matched_source_control|k=3.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|full_refit_minus_matched_source_control|k=3.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|source_selected_minus_matched_source_control|k=3.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|source_selected_minus_bias|k=3.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|source_selected_minus_mean_anchor|k=3.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|source_selected_minus_full_refit|k=3.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|bias_minus_matched_source_control|k=5.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|mean_anchor_minus_matched_source_control|k=5.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|full_refit_minus_matched_source_control|k=5.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|source_selected_minus_matched_source_control|k=5.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|source_selected_minus_bias|k=5.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|source_selected_minus_mean_anchor|k=5.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|source_selected_minus_full_refit|k=5.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|bias_minus_matched_source_control|k=10.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|mean_anchor_minus_matched_source_control|k=10.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|full_refit_minus_matched_source_control|k=10.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|source_selected_minus_matched_source_control|k=10.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|source_selected_minus_bias|k=10.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|source_selected_minus_mean_anchor|k=10.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I01|E01|source_selected_minus_full_refit|k=10.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=2.0, policy=diversity|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=2.0, policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=2.0, policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=2.0, policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=2.0, policy=uncertainty|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=6.0, policy=diversity|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=6.0, policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=6.0, policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=6.0, policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=6.0, policy=uncertainty|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=10.0, policy=diversity|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=10.0, policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=10.0, policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=10.0, policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=10.0, policy=uncertainty|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=20.0, policy=diversity|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=20.0, policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=20.0, policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=20.0, policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=20.0, policy=uncertainty|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=2.0, policy=diversity|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=2.0, policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=2.0, policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=2.0, policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=2.0, policy=uncertainty|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=6.0, policy=diversity|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=6.0, policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=6.0, policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=6.0, policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=6.0, policy=uncertainty|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=10.0, policy=diversity|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=10.0, policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=10.0, policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=10.0, policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=10.0, policy=uncertainty|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=20.0, policy=diversity|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=20.0, policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=20.0, policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=20.0, policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=20.0, policy=uncertainty|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=2.0, policy=diversity|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=2.0, policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=2.0, policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=2.0, policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=2.0, policy=uncertainty|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|uncertainty_minus_random_601|budget=2.0, policy=uncertainty, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|uncertainty_minus_random_602|budget=2.0, policy=uncertainty, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|uncertainty_minus_random_603|budget=2.0, policy=uncertainty, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|diversity_minus_random_601|budget=2.0, policy=diversity, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|diversity_minus_random_602|budget=2.0, policy=diversity, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|diversity_minus_random_603|budget=2.0, policy=diversity, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|uncertainty_minus_random_601|budget=2.0, policy=uncertainty, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|uncertainty_minus_random_602|budget=2.0, policy=uncertainty, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|uncertainty_minus_random_603|budget=2.0, policy=uncertainty, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|diversity_minus_random_601|budget=2.0, policy=diversity, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|diversity_minus_random_602|budget=2.0, policy=diversity, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|diversity_minus_random_603|budget=2.0, policy=diversity, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=6.0, policy=diversity|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=6.0, policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=6.0, policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=6.0, policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=6.0, policy=uncertainty|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|uncertainty_minus_random_601|budget=6.0, policy=uncertainty, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|uncertainty_minus_random_602|budget=6.0, policy=uncertainty, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|uncertainty_minus_random_603|budget=6.0, policy=uncertainty, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|diversity_minus_random_601|budget=6.0, policy=diversity, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|diversity_minus_random_602|budget=6.0, policy=diversity, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|diversity_minus_random_603|budget=6.0, policy=diversity, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|uncertainty_minus_random_601|budget=6.0, policy=uncertainty, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|uncertainty_minus_random_602|budget=6.0, policy=uncertainty, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|uncertainty_minus_random_603|budget=6.0, policy=uncertainty, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|diversity_minus_random_601|budget=6.0, policy=diversity, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|diversity_minus_random_602|budget=6.0, policy=diversity, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|diversity_minus_random_603|budget=6.0, policy=diversity, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=10.0, policy=diversity|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=10.0, policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=10.0, policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=10.0, policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=10.0, policy=uncertainty|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|uncertainty_minus_random_601|budget=10.0, policy=uncertainty, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|uncertainty_minus_random_602|budget=10.0, policy=uncertainty, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|uncertainty_minus_random_603|budget=10.0, policy=uncertainty, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|diversity_minus_random_601|budget=10.0, policy=diversity, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|diversity_minus_random_602|budget=10.0, policy=diversity, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|diversity_minus_random_603|budget=10.0, policy=diversity, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|uncertainty_minus_random_601|budget=10.0, policy=uncertainty, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|uncertainty_minus_random_602|budget=10.0, policy=uncertainty, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|uncertainty_minus_random_603|budget=10.0, policy=uncertainty, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|diversity_minus_random_601|budget=10.0, policy=diversity, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|diversity_minus_random_602|budget=10.0, policy=diversity, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|diversity_minus_random_603|budget=10.0, policy=diversity, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=20.0, policy=diversity|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=20.0, policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=20.0, policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=20.0, policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=20.0, policy=uncertainty|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|uncertainty_minus_random_601|budget=20.0, policy=uncertainty, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|uncertainty_minus_random_602|budget=20.0, policy=uncertainty, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|uncertainty_minus_random_603|budget=20.0, policy=uncertainty, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|diversity_minus_random_601|budget=20.0, policy=diversity, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|diversity_minus_random_602|budget=20.0, policy=diversity, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|diversity_minus_random_603|budget=20.0, policy=diversity, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|uncertainty_minus_random_601|budget=20.0, policy=uncertainty, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|uncertainty_minus_random_602|budget=20.0, policy=uncertainty, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|uncertainty_minus_random_603|budget=20.0, policy=uncertainty, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|diversity_minus_random_601|budget=20.0, policy=diversity, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|diversity_minus_random_602|budget=20.0, policy=diversity, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|40/40|
|I02|E01|diversity_minus_random_603|budget=20.0, policy=diversity, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|40/40|
|I03|E01|raw_mean_anchor_minus_identity|k=1.0|+0.000 [+0.000,+0.000]|0/0|120/40|
|I03|E01|ovr_trace1_minus_identity|k=1.0|+0.000 [+0.000,+0.000]|0/0|120/40|
|I03|E01|ovo_trace1_minus_identity|k=1.0|+0.000 [+0.000,+0.000]|0/0|120/40|
|I03|E01|ovo_pca_trace1_minus_identity|k=1.0|+0.000 [+0.000,+0.000]|0/0|120/40|
|I03|E01|ovr_trace1_minus_raw_mean_anchor|k=1.0|+0.000 [+0.000,+0.000]|0/0|120/40|
|I03|E01|ovo_trace1_minus_ovr_trace1|k=1.0|+0.000 [+0.000,+0.000]|0/0|120/40|
|I03|E01|ovo_pca_trace1_minus_ovo_trace1|k=1.0|+0.000 [+0.000,+0.000]|0/0|120/40|
|I03|E01|raw_mean_anchor_minus_identity|k=3.0|+0.000 [+0.000,+0.000]|0/0|120/40|
|I03|E01|ovr_trace1_minus_identity|k=3.0|+0.000 [+0.000,+0.000]|0/0|120/40|
|I03|E01|ovo_trace1_minus_identity|k=3.0|+0.000 [+0.000,+0.000]|0/0|120/40|
|I03|E01|ovo_pca_trace1_minus_identity|k=3.0|+0.000 [+0.000,+0.000]|0/0|120/40|
|I03|E01|ovr_trace1_minus_raw_mean_anchor|k=3.0|+0.000 [+0.000,+0.000]|0/0|120/40|
|I03|E01|ovo_trace1_minus_ovr_trace1|k=3.0|+0.000 [+0.000,+0.000]|0/0|120/40|
|I03|E01|ovo_pca_trace1_minus_ovo_trace1|k=3.0|+0.000 [+0.000,+0.000]|0/0|120/40|
|I03|E01|raw_mean_anchor_minus_identity|k=5.0|+0.000 [+0.000,+0.000]|0/0|120/40|
|I03|E01|ovr_trace1_minus_identity|k=5.0|+0.000 [+0.000,+0.000]|0/0|120/40|
|I03|E01|ovo_trace1_minus_identity|k=5.0|+0.000 [+0.000,+0.000]|0/0|120/40|
|I03|E01|ovo_pca_trace1_minus_identity|k=5.0|+0.000 [+0.000,+0.000]|0/0|120/40|
|I03|E01|ovr_trace1_minus_raw_mean_anchor|k=5.0|+0.000 [+0.000,+0.000]|0/0|120/40|
|I03|E01|ovo_trace1_minus_ovr_trace1|k=5.0|+0.000 [+0.000,+0.000]|0/0|120/40|
|I03|E01|ovo_pca_trace1_minus_ovo_trace1|k=5.0|+0.000 [+0.000,+0.000]|0/0|120/40|
|I03|E01|raw_mean_anchor_minus_identity|k=10.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I03|E01|ovr_trace1_minus_identity|k=10.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I03|E01|ovo_trace1_minus_identity|k=10.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I03|E01|ovo_pca_trace1_minus_identity|k=10.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I03|E01|ovr_trace1_minus_raw_mean_anchor|k=10.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I03|E01|ovo_trace1_minus_ovr_trace1|k=10.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I03|E01|ovo_pca_trace1_minus_ovo_trace1|k=10.0|+0.000 [+0.000,+0.000]|0/0|40/40|
|I04|E01|weighted_lambda1_minus_identity|k=-1.0, composition_id=balanced|+0.000 [+0.000,+0.000]|0/0|120/40|
|I04|E01|weighted_mean_minus_identity|k=-1.0, composition_id=balanced|+0.000 [+0.000,+0.000]|0/0|120/40|
|I04|E01|weighted_ess_minus_identity|k=-1.0, composition_id=balanced|+0.000 [+0.000,+0.000]|0/0|120/40|
|I04|E01|weighted_mean_minus_weighted_lambda1|k=-1.0, composition_id=balanced|+0.000 [+0.000,+0.000]|0/0|120/40|
|I04|E01|weighted_ess_minus_weighted_mean|k=-1.0, composition_id=balanced|+0.000 [+0.000,+0.000]|0/0|120/40|
|I04|E01|weighted_lambda1_minus_identity|k=-1.0, composition_id=binary_1_9|+0.000 [+0.000,+0.000]|0/0|120/40|
|I04|E01|weighted_mean_minus_identity|k=-1.0, composition_id=binary_1_9|+0.000 [+0.000,+0.000]|0/0|120/40|
|I04|E01|weighted_ess_minus_identity|k=-1.0, composition_id=binary_1_9|+0.000 [+0.000,+0.000]|0/0|120/40|
|I04|E01|weighted_mean_minus_weighted_lambda1|k=-1.0, composition_id=binary_1_9|+0.000 [+0.000,+0.000]|0/0|120/40|
|I04|E01|weighted_ess_minus_weighted_mean|k=-1.0, composition_id=binary_1_9|+0.000 [+0.000,+0.000]|0/0|120/40|
|I04|E01|weighted_lambda1_minus_identity|k=-1.0, composition_id=binary_3_7|+0.000 [+0.000,+0.000]|0/0|120/40|
|I04|E01|weighted_mean_minus_identity|k=-1.0, composition_id=binary_3_7|+0.000 [+0.000,+0.000]|0/0|120/40|
|I04|E01|weighted_ess_minus_identity|k=-1.0, composition_id=binary_3_7|+0.000 [+0.000,+0.000]|0/0|120/40|
|I04|E01|weighted_mean_minus_weighted_lambda1|k=-1.0, composition_id=binary_3_7|+0.000 [+0.000,+0.000]|0/0|120/40|
|I04|E01|weighted_ess_minus_weighted_mean|k=-1.0, composition_id=binary_3_7|+0.000 [+0.000,+0.000]|0/0|120/40|
|I04|E01|weighted_lambda1_minus_identity|k=-1.0, composition_id=binary_7_3|+0.000 [+0.000,+0.000]|0/0|120/40|
|I04|E01|weighted_mean_minus_identity|k=-1.0, composition_id=binary_7_3|+0.000 [+0.000,+0.000]|0/0|120/40|
|I04|E01|weighted_ess_minus_identity|k=-1.0, composition_id=binary_7_3|+0.000 [+0.000,+0.000]|0/0|120/40|
|I04|E01|weighted_mean_minus_weighted_lambda1|k=-1.0, composition_id=binary_7_3|+0.000 [+0.000,+0.000]|0/0|120/40|
|I04|E01|weighted_ess_minus_weighted_mean|k=-1.0, composition_id=binary_7_3|+0.000 [+0.000,+0.000]|0/0|120/40|
|I04|E01|weighted_lambda1_minus_identity|k=-1.0, composition_id=binary_9_1|+0.000 [+0.000,+0.000]|0/0|120/40|
|I04|E01|weighted_mean_minus_identity|k=-1.0, composition_id=binary_9_1|+0.000 [+0.000,+0.000]|0/0|120/40|
|I04|E01|weighted_ess_minus_identity|k=-1.0, composition_id=binary_9_1|+0.000 [+0.000,+0.000]|0/0|120/40|
|I04|E01|weighted_mean_minus_weighted_lambda1|k=-1.0, composition_id=binary_9_1|+0.000 [+0.000,+0.000]|0/0|120/40|
|I04|E01|weighted_ess_minus_weighted_mean|k=-1.0, composition_id=binary_9_1|+0.000 [+0.000,+0.000]|0/0|120/40|

## Source策略与获取成本审计

source_policy.json、inner_plan.json与inner_validation_predictions.csv.gz记录每个Source内选择的依据；汇总再次核验内层角色与指标。Source选择器训练成本与选择后外层端点复用分开计数。acquisition_plan.json及机制表记录实际覆盖、资格回退、观察标签量与拟合用标签量；全池政策端点不重复训练，也不当作多个独立获取结果。

三次Source内划分的实物重叠数、交叉角色复用数与逐次分类器拟合成本保存于mechanism_analysis.csv；它们不构成新增独立验证样品。

|原任务|K/类|Source内选中方法|上下文数|平均最大内验证Macro-F1（选择统计量）|
|---|---:|---|---:|---:|
|E01|1|identity|1|0.974984|
|E01|3|identity|1|0.974984|
|E01|5|identity|1|0.974984|
|E01|10|identity|1|0.974984|

## 局限与未运行事项

本轮根据前四轮已曝光结果设计，所有实验仅使用master TRAIN，仍属探索性内部研究。内层Source重复划分、Support获取顺序、输入几何及收缩条件不增加独立实物数。描述性95%区间按physical_sample_id聚类并保留相关预测，不覆盖来源/批次总体、协议选择和长期开发偏倚；来源与批次混杂，不能称真实新批次验证或确认性因果结论。

二分类和五来源的类别数、样品组成及标签预算不同，不能直接相减Accuracy解释类别数效应。构成、获取顺序、校准几何或收缩启发式均不依据本轮外层Query选择最终赢家。真实新批次评价继续 **NOT_RUN_NEEDS_NEW_BATCH_DATA**；真实batch_id、样品属性及独立验证不补造。

## 复现

```bash
python scripts/run_b_mechanism_round5_20261111.py --resume
python scripts/summarize_b_mechanism_round5_20261111.py --resume
python scripts/summarize_b_mechanism_round5_20261111.py --verify
python scripts/run_b_mechanism_round5_20261111.py --verify
```

PARTIAL只描述已完整封存子集；核验通过不会把未运行任务升级为DONE。
