# B方案第五轮：Source内策略、Support获取与分数信息

状态：**DONE**；已封存 **180/180** 个任务包。
本报告只读取完整封存工件，不拟合或选择新模型，不修改前四轮输入、标签、协议、模型及结果。

新增分类器拟合1349次，校准拟合调用4111次；科研任务CPU合计115.754809秒。
保存321000条相关外层Query预测，涉及356个唯一Query实物；控制、核验、失败及推送成本另见账本。

Source内部验证预测另计86400/86400条（已封存/冻结计划），不计入外层Query预测数，也不增加独立验证实物。

## 完成范围

|块|问题|完成/计划|状态|
|---|---|---:|---|
|I01|仅用Source内部验证选择校准策略|45/45|DONE|
|I02|隐藏候选标签的Support获取顺序|45/45|DONE|
|I03|保留OVR输出锚的OVO信息校准|45/45|DONE|
|I04|权重ESS与预设收缩|45/45|DONE|

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
|I01|E01|bias|1||0.983333|0.983323|0.983333|30|
|I01|E01|bias|3||0.983333|0.983323|0.983333|30|
|I01|E01|bias|5||0.983333|0.983323|0.983333|30|
|I01|E01|bias|10||0.983333|0.983323|0.983333|30|
|I01|E01|full_refit|1||0.983333|0.983323|0.983333|30|
|I01|E01|full_refit|3||0.983333|0.983323|0.983333|30|
|I01|E01|full_refit|5||0.983333|0.983323|0.983333|30|
|I01|E01|full_refit|10||0.983333|0.983323|0.983333|30|
|I01|E01|identity|0||0.983333|0.983323|0.983333|30|
|I01|E01|mean_anchor|1||0.983333|0.983323|0.983333|30|
|I01|E01|mean_anchor|3||0.983333|0.983323|0.983333|30|
|I01|E01|mean_anchor|5||0.983333|0.983323|0.983333|30|
|I01|E01|mean_anchor|10||0.983333|0.983323|0.983333|30|
|I01|E01|source_selected|1||0.983333|0.983323|0.983333|30|
|I01|E01|source_selected|3||0.983333|0.983323|0.983333|30|
|I01|E01|source_selected|5||0.983333|0.983323|0.983333|30|
|I01|E01|source_selected|10||0.983333|0.983323|0.983333|30|
|I01|E01|svc_predict|0||0.983333|0.983323|0.983333|30|
|I01|E03|bias|1||0.773333|0.764809|0.773333|15|
|I01|E03|bias|3||0.785556|0.779839|0.785556|15|
|I01|E03|bias|5||0.788889|0.784470|0.788889|15|
|I01|E03|bias|10||0.797778|0.793716|0.797778|15|
|I01|E03|full_refit|1||0.801111|0.799543|0.801111|15|
|I01|E03|full_refit|3||0.802222|0.800661|0.802222|15|
|I01|E03|full_refit|5||0.821111|0.820229|0.821111|15|
|I01|E03|full_refit|10||0.837778|0.836784|0.837778|15|
|I01|E03|identity|0||0.800000|0.798475|0.800000|15|
|I01|E03|mean_anchor|1||0.752222|0.738630|0.752222|15|
|I01|E03|mean_anchor|3||0.794444|0.793237|0.794444|15|
|I01|E03|mean_anchor|5||0.801111|0.799488|0.801111|15|
|I01|E03|mean_anchor|10||0.800000|0.798023|0.800000|15|
|I01|E03|source_selected|1||0.792222|0.789319|0.792222|15|
|I01|E03|source_selected|3||0.792222|0.789805|0.792222|15|
|I01|E03|source_selected|5||0.818889|0.817987|0.818889|15|
|I01|E03|source_selected|10||0.836667|0.835706|0.836667|15|
|I01|E03|svc_predict|0||0.800000|0.798475|0.800000|15|
|I02|E01|full_refit|不适用|policy=diversity, budget=2.0|0.983333|0.983323|0.983333|30|
|I02|E01|full_refit|不适用|policy=diversity, budget=6.0|0.983333|0.983323|0.983333|30|
|I02|E01|full_refit|不适用|policy=diversity, budget=10.0|0.983333|0.983323|0.983333|30|
|I02|E01|full_refit|不适用|policy=diversity, budget=20.0|0.983333|0.983323|0.983333|30|
|I02|E01|full_refit|不适用|policy=random_601, budget=2.0|0.983333|0.983323|0.983333|30|
|I02|E01|full_refit|不适用|policy=random_601, budget=6.0|0.983333|0.983323|0.983333|30|
|I02|E01|full_refit|不适用|policy=random_601, budget=10.0|0.983333|0.983323|0.983333|30|
|I02|E01|full_refit|不适用|policy=random_601, budget=20.0|0.983333|0.983323|0.983333|30|
|I02|E01|full_refit|不适用|policy=random_602, budget=2.0|0.983333|0.983323|0.983333|30|
|I02|E01|full_refit|不适用|policy=random_602, budget=6.0|0.983333|0.983323|0.983333|30|
|I02|E01|full_refit|不适用|policy=random_602, budget=10.0|0.983333|0.983323|0.983333|30|
|I02|E01|full_refit|不适用|policy=random_602, budget=20.0|0.983333|0.983323|0.983333|30|
|I02|E01|full_refit|不适用|policy=random_603, budget=2.0|0.983333|0.983323|0.983333|30|
|I02|E01|full_refit|不适用|policy=random_603, budget=6.0|0.983333|0.983323|0.983333|30|
|I02|E01|full_refit|不适用|policy=random_603, budget=10.0|0.983333|0.983323|0.983333|30|
|I02|E01|full_refit|不适用|policy=random_603, budget=20.0|0.983333|0.983323|0.983333|30|
|I02|E01|full_refit|不适用|policy=uncertainty, budget=2.0|0.983333|0.983323|0.983333|30|
|I02|E01|full_refit|不适用|policy=uncertainty, budget=6.0|0.983333|0.983323|0.983333|30|
|I02|E01|full_refit|不适用|policy=uncertainty, budget=10.0|0.983333|0.983323|0.983333|30|
|I02|E01|full_refit|不适用|policy=uncertainty, budget=20.0|0.983333|0.983323|0.983333|30|
|I02|E01|guarded_mean_anchor|不适用|policy=diversity, budget=2.0|0.983333|0.983323|0.983333|30|
|I02|E01|guarded_mean_anchor|不适用|policy=diversity, budget=6.0|0.986667|0.986658|0.986667|30|
|I02|E01|guarded_mean_anchor|不适用|policy=diversity, budget=10.0|0.983333|0.983323|0.983333|30|
|I02|E01|guarded_mean_anchor|不适用|policy=diversity, budget=20.0|0.983333|0.983323|0.983333|30|
|I02|E01|guarded_mean_anchor|不适用|policy=random_601, budget=2.0|0.983333|0.983323|0.983333|30|
|I02|E01|guarded_mean_anchor|不适用|policy=random_601, budget=6.0|0.984167|0.984157|0.984167|30|
|I02|E01|guarded_mean_anchor|不适用|policy=random_601, budget=10.0|0.983333|0.983323|0.983333|30|
|I02|E01|guarded_mean_anchor|不适用|policy=random_601, budget=20.0|0.983333|0.983323|0.983333|30|
|I02|E01|guarded_mean_anchor|不适用|policy=random_602, budget=2.0|0.983333|0.983323|0.983333|30|
|I02|E01|guarded_mean_anchor|不适用|policy=random_602, budget=6.0|0.984167|0.984157|0.984167|30|
|I02|E01|guarded_mean_anchor|不适用|policy=random_602, budget=10.0|0.983333|0.983323|0.983333|30|
|I02|E01|guarded_mean_anchor|不适用|policy=random_602, budget=20.0|0.983333|0.983323|0.983333|30|
|I02|E01|guarded_mean_anchor|不适用|policy=random_603, budget=2.0|0.983333|0.983323|0.983333|30|
|I02|E01|guarded_mean_anchor|不适用|policy=random_603, budget=6.0|0.983333|0.983323|0.983333|30|
|I02|E01|guarded_mean_anchor|不适用|policy=random_603, budget=10.0|0.983333|0.983323|0.983333|30|
|I02|E01|guarded_mean_anchor|不适用|policy=random_603, budget=20.0|0.983333|0.983323|0.983333|30|
|I02|E01|guarded_mean_anchor|不适用|policy=uncertainty, budget=2.0|0.983333|0.983323|0.983333|30|
|I02|E01|guarded_mean_anchor|不适用|policy=uncertainty, budget=6.0|0.986667|0.986658|0.986667|30|
|I02|E01|guarded_mean_anchor|不适用|policy=uncertainty, budget=10.0|0.991667|0.991661|0.991667|30|
|I02|E01|guarded_mean_anchor|不适用|policy=uncertainty, budget=20.0|0.983333|0.983323|0.983333|30|
|I02|E01|identity|0|budget=None|0.983333|0.983323|0.983333|30|
|I02|E01|svc_predict|0|budget=None|0.983333|0.983323|0.983333|30|
|I02|E03|full_refit|不适用|policy=diversity, budget=5.0|0.803333|0.801924|0.803333|15|
|I02|E03|full_refit|不适用|policy=diversity, budget=15.0|0.807778|0.806349|0.807778|15|
|I02|E03|full_refit|不适用|policy=diversity, budget=25.0|0.818889|0.818165|0.818889|15|
|I02|E03|full_refit|不适用|policy=diversity, budget=50.0|0.837778|0.836784|0.837778|15|
|I02|E03|full_refit|不适用|policy=random_601, budget=5.0|0.802222|0.801195|0.802222|15|
|I02|E03|full_refit|不适用|policy=random_601, budget=15.0|0.815556|0.814403|0.815556|15|
|I02|E03|full_refit|不适用|policy=random_601, budget=25.0|0.818889|0.817625|0.818889|15|
|I02|E03|full_refit|不适用|policy=random_601, budget=50.0|0.837778|0.836784|0.837778|15|
|I02|E03|full_refit|不适用|policy=random_602, budget=5.0|0.800000|0.798602|0.800000|15|
|I02|E03|full_refit|不适用|policy=random_602, budget=15.0|0.808889|0.807602|0.808889|15|
|I02|E03|full_refit|不适用|policy=random_602, budget=25.0|0.820000|0.819098|0.820000|15|
|I02|E03|full_refit|不适用|policy=random_602, budget=50.0|0.837778|0.836784|0.837778|15|
|I02|E03|full_refit|不适用|policy=random_603, budget=5.0|0.802222|0.800813|0.802222|15|
|I02|E03|full_refit|不适用|policy=random_603, budget=15.0|0.805556|0.804117|0.805556|15|
|I02|E03|full_refit|不适用|policy=random_603, budget=25.0|0.821111|0.820076|0.821111|15|
|I02|E03|full_refit|不适用|policy=random_603, budget=50.0|0.837778|0.836784|0.837778|15|
|I02|E03|full_refit|不适用|policy=uncertainty, budget=5.0|0.816667|0.815416|0.816667|15|
|I02|E03|full_refit|不适用|policy=uncertainty, budget=15.0|0.826667|0.825671|0.826667|15|
|I02|E03|full_refit|不适用|policy=uncertainty, budget=25.0|0.831111|0.830275|0.831111|15|
|I02|E03|full_refit|不适用|policy=uncertainty, budget=50.0|0.837778|0.836784|0.837778|15|
|I02|E03|guarded_mean_anchor|不适用|policy=diversity, budget=5.0|0.800000|0.798475|0.800000|15|
|I02|E03|guarded_mean_anchor|不适用|policy=diversity, budget=15.0|0.795556|0.794069|0.795556|15|
|I02|E03|guarded_mean_anchor|不适用|policy=diversity, budget=25.0|0.796667|0.795712|0.796667|15|
|I02|E03|guarded_mean_anchor|不适用|policy=diversity, budget=50.0|0.800000|0.798023|0.800000|15|
|I02|E03|guarded_mean_anchor|不适用|policy=random_601, budget=5.0|0.800000|0.798475|0.800000|15|
|I02|E03|guarded_mean_anchor|不适用|policy=random_601, budget=15.0|0.793333|0.789488|0.793333|15|
|I02|E03|guarded_mean_anchor|不适用|policy=random_601, budget=25.0|0.796667|0.794197|0.796667|15|
|I02|E03|guarded_mean_anchor|不适用|policy=random_601, budget=50.0|0.800000|0.798023|0.800000|15|
|I02|E03|guarded_mean_anchor|不适用|policy=random_602, budget=5.0|0.800000|0.798475|0.800000|15|
|I02|E03|guarded_mean_anchor|不适用|policy=random_602, budget=15.0|0.796667|0.794471|0.796667|15|
|I02|E03|guarded_mean_anchor|不适用|policy=random_602, budget=25.0|0.801111|0.799331|0.801111|15|
|I02|E03|guarded_mean_anchor|不适用|policy=random_602, budget=50.0|0.800000|0.798023|0.800000|15|
|I02|E03|guarded_mean_anchor|不适用|policy=random_603, budget=5.0|0.794444|0.792058|0.794444|15|
|I02|E03|guarded_mean_anchor|不适用|policy=random_603, budget=15.0|0.795556|0.792825|0.795556|15|
|I02|E03|guarded_mean_anchor|不适用|policy=random_603, budget=25.0|0.800000|0.798045|0.800000|15|
|I02|E03|guarded_mean_anchor|不适用|policy=random_603, budget=50.0|0.800000|0.798023|0.800000|15|
|I02|E03|guarded_mean_anchor|不适用|policy=uncertainty, budget=5.0|0.800000|0.798475|0.800000|15|
|I02|E03|guarded_mean_anchor|不适用|policy=uncertainty, budget=15.0|0.781111|0.776063|0.781111|15|
|I02|E03|guarded_mean_anchor|不适用|policy=uncertainty, budget=25.0|0.795556|0.793110|0.795556|15|
|I02|E03|guarded_mean_anchor|不适用|policy=uncertainty, budget=50.0|0.800000|0.798023|0.800000|15|
|I02|E03|identity|0|budget=None|0.800000|0.798475|0.800000|15|
|I02|E03|svc_predict|0|budget=None|0.800000|0.798475|0.800000|15|
|I03|E01|identity|0||0.983333|0.983323|0.983333|30|
|I03|E01|ovo_pca_trace1|1||0.984444|0.984435|0.984444|90|
|I03|E01|ovo_pca_trace1|3||0.983333|0.983323|0.983333|90|
|I03|E01|ovo_pca_trace1|5||0.983333|0.983323|0.983333|90|
|I03|E01|ovo_pca_trace1|10||0.983333|0.983323|0.983333|30|
|I03|E01|ovo_trace1|1||0.984444|0.984435|0.984444|90|
|I03|E01|ovo_trace1|3||0.983333|0.983323|0.983333|90|
|I03|E01|ovo_trace1|5||0.983333|0.983323|0.983333|90|
|I03|E01|ovo_trace1|10||0.983333|0.983323|0.983333|30|
|I03|E01|ovr_trace1|1||0.984444|0.984435|0.984444|90|
|I03|E01|ovr_trace1|3||0.983333|0.983323|0.983333|90|
|I03|E01|ovr_trace1|5||0.983333|0.983323|0.983333|90|
|I03|E01|ovr_trace1|10||0.983333|0.983323|0.983333|30|
|I03|E01|raw_mean_anchor|1||0.983611|0.983601|0.983611|90|
|I03|E01|raw_mean_anchor|3||0.983333|0.983323|0.983333|90|
|I03|E01|raw_mean_anchor|5||0.983333|0.983323|0.983333|90|
|I03|E01|raw_mean_anchor|10||0.983333|0.983323|0.983333|30|
|I03|E01|svc_predict|0||0.983333|0.983323|0.983333|30|
|I03|E03|identity|0||0.800000|0.798475|0.800000|15|
|I03|E03|ovo_pca_trace1|1||0.798519|0.796127|0.798519|45|
|I03|E03|ovo_pca_trace1|3||0.802593|0.801021|0.802593|45|
|I03|E03|ovo_pca_trace1|5||0.802963|0.801493|0.802963|45|
|I03|E03|ovo_pca_trace1|10||0.803333|0.801949|0.803333|15|
|I03|E03|ovo_trace1|1||0.798519|0.796127|0.798519|45|
|I03|E03|ovo_trace1|3||0.802593|0.801021|0.802593|45|
|I03|E03|ovo_trace1|5||0.803333|0.801829|0.803333|45|
|I03|E03|ovo_trace1|10||0.804444|0.802958|0.804444|15|
|I03|E03|ovr_trace1|1||0.799630|0.797315|0.799630|45|
|I03|E03|ovr_trace1|3||0.801481|0.799934|0.801481|45|
|I03|E03|ovr_trace1|5||0.801852|0.800319|0.801852|45|
|I03|E03|ovr_trace1|10||0.800000|0.798441|0.800000|15|
|I03|E03|raw_mean_anchor|1||0.750741|0.735301|0.750741|45|
|I03|E03|raw_mean_anchor|3||0.796667|0.794947|0.796667|45|
|I03|E03|raw_mean_anchor|5||0.799630|0.797547|0.799630|45|
|I03|E03|raw_mean_anchor|10||0.800000|0.798023|0.800000|15|
|I03|E03|svc_predict|0||0.800000|0.798475|0.800000|15|
|I04|E01|identity|0||0.983333|0.983323|0.983333|30|
|I04|E01|svc_predict|0||0.983333|0.983323|0.983333|30|
|I04|E01|weighted_ess|不适用|composition_id=balanced|0.983333|0.983323|0.983333|90|
|I04|E01|weighted_ess|不适用|composition_id=binary_1_9|0.983333|0.983323|0.983333|90|
|I04|E01|weighted_ess|不适用|composition_id=binary_3_7|0.983333|0.983323|0.983333|90|
|I04|E01|weighted_ess|不适用|composition_id=binary_7_3|0.983333|0.983323|0.983333|90|
|I04|E01|weighted_ess|不适用|composition_id=binary_9_1|0.983333|0.983323|0.983333|90|
|I04|E01|weighted_lambda1|不适用|composition_id=balanced|0.983333|0.983323|0.983333|90|
|I04|E01|weighted_lambda1|不适用|composition_id=binary_1_9|0.983333|0.983323|0.983333|90|
|I04|E01|weighted_lambda1|不适用|composition_id=binary_3_7|0.983333|0.983323|0.983333|90|
|I04|E01|weighted_lambda1|不适用|composition_id=binary_7_3|0.983333|0.983323|0.983333|90|
|I04|E01|weighted_lambda1|不适用|composition_id=binary_9_1|0.983333|0.983323|0.983333|90|
|I04|E01|weighted_mean|不适用|composition_id=balanced|0.983333|0.983323|0.983333|90|
|I04|E01|weighted_mean|不适用|composition_id=binary_1_9|0.983333|0.983323|0.983333|90|
|I04|E01|weighted_mean|不适用|composition_id=binary_3_7|0.983333|0.983323|0.983333|90|
|I04|E01|weighted_mean|不适用|composition_id=binary_7_3|0.983333|0.983323|0.983333|90|
|I04|E01|weighted_mean|不适用|composition_id=binary_9_1|0.983333|0.983323|0.983333|90|
|I04|E03|identity|0||0.800000|0.798475|0.800000|15|
|I04|E03|svc_predict|0||0.800000|0.798475|0.800000|15|
|I04|E03|weighted_ess|不适用|composition_id=balanced|0.799630|0.797547|0.799630|45|
|I04|E03|weighted_ess|不适用|composition_id=scarce_ZY03|0.792963|0.789265|0.792963|45|
|I04|E03|weighted_ess|不适用|composition_id=scarce_ZY04|0.792963|0.789001|0.792963|45|
|I04|E03|weighted_ess|不适用|composition_id=scarce_ZY05|0.798889|0.797248|0.798889|45|
|I04|E03|weighted_ess|不适用|composition_id=scarce_ZY06|0.785556|0.781538|0.785556|45|
|I04|E03|weighted_ess|不适用|composition_id=scarce_ZY07|0.785926|0.782197|0.785926|45|
|I04|E03|weighted_lambda1|不适用|composition_id=balanced|0.776667|0.770945|0.776667|45|
|I04|E03|weighted_lambda1|不适用|composition_id=scarce_ZY03|0.721111|0.701152|0.721111|45|
|I04|E03|weighted_lambda1|不适用|composition_id=scarce_ZY04|0.743704|0.728146|0.743704|45|
|I04|E03|weighted_lambda1|不适用|composition_id=scarce_ZY05|0.760370|0.752891|0.760370|45|
|I04|E03|weighted_lambda1|不适用|composition_id=scarce_ZY06|0.725926|0.706392|0.725926|45|
|I04|E03|weighted_lambda1|不适用|composition_id=scarce_ZY07|0.726667|0.713834|0.726667|45|
|I04|E03|weighted_mean|不适用|composition_id=balanced|0.799630|0.797547|0.799630|45|
|I04|E03|weighted_mean|不适用|composition_id=scarce_ZY03|0.783704|0.778797|0.783704|45|
|I04|E03|weighted_mean|不适用|composition_id=scarce_ZY04|0.791111|0.786885|0.791111|45|
|I04|E03|weighted_mean|不适用|composition_id=scarce_ZY05|0.797778|0.795735|0.797778|45|
|I04|E03|weighted_mean|不适用|composition_id=scarce_ZY06|0.784074|0.779504|0.784074|45|
|I04|E03|weighted_mean|不适用|composition_id=scarce_ZY07|0.784074|0.779826|0.784074|45|

## 配对变化与描述性不确定性

所有正、负和零结果保留。配对以相同外层Query的physical_sample_id对齐，区间保留同实物相关预测；完整来源分层、Recall代价及纠错/误伤另见CSV。

|块|原任务|比较|预算/政策/构成|ΔAccuracy百分点及95%区间|纠错/误伤|预测行/实物|
|---|---|---|---|---:|---:|---:|
|I01|E01|bias_minus_matched_source_control|k=1.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|mean_anchor_minus_matched_source_control|k=1.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|full_refit_minus_matched_source_control|k=1.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|source_selected_minus_matched_source_control|k=1.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|source_selected_minus_bias|k=1.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|source_selected_minus_mean_anchor|k=1.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|source_selected_minus_full_refit|k=1.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|bias_minus_matched_source_control|k=3.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|mean_anchor_minus_matched_source_control|k=3.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|full_refit_minus_matched_source_control|k=3.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|source_selected_minus_matched_source_control|k=3.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|source_selected_minus_bias|k=3.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|source_selected_minus_mean_anchor|k=3.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|source_selected_minus_full_refit|k=3.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|bias_minus_matched_source_control|k=5.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|mean_anchor_minus_matched_source_control|k=5.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|full_refit_minus_matched_source_control|k=5.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|source_selected_minus_matched_source_control|k=5.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|source_selected_minus_bias|k=5.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|source_selected_minus_mean_anchor|k=5.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|source_selected_minus_full_refit|k=5.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|bias_minus_matched_source_control|k=10.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|mean_anchor_minus_matched_source_control|k=10.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|full_refit_minus_matched_source_control|k=10.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|source_selected_minus_matched_source_control|k=10.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|source_selected_minus_bias|k=10.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|source_selected_minus_mean_anchor|k=10.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E01|source_selected_minus_full_refit|k=10.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I01|E03|bias_minus_matched_source_control|k=1.0|-2.667 [-4.333,-1.111]|12/36|900/300|
|I01|E03|mean_anchor_minus_matched_source_control|k=1.0|-4.778 [-6.778,-2.664]|19/62|900/300|
|I01|E03|full_refit_minus_matched_source_control|k=1.0|+0.111 [-0.778,+1.000]|8/7|900/300|
|I01|E03|source_selected_minus_matched_source_control|k=1.0|-0.778 [-1.778,+0.444]|8/15|900/300|
|I01|E03|source_selected_minus_bias|k=1.0|+1.889 [+0.333,+3.444]|30/13|900/300|
|I01|E03|source_selected_minus_mean_anchor|k=1.0|+4.000 [+2.000,+5.892]|58/22|900/300|
|I01|E03|source_selected_minus_full_refit|k=1.0|-0.889 [-1.778,+0.003]|5/13|900/300|
|I01|E03|bias_minus_matched_source_control|k=3.0|-1.444 [-3.000,+0.000]|14/27|900/300|
|I01|E03|mean_anchor_minus_matched_source_control|k=3.0|-0.556 [-1.444,+0.333]|5/10|900/300|
|I01|E03|full_refit_minus_matched_source_control|k=3.0|+0.222 [-1.000,+1.333]|16/14|900/300|
|I01|E03|source_selected_minus_matched_source_control|k=3.0|-0.778 [-2.000,+0.444]|13/20|900/300|
|I01|E03|source_selected_minus_bias|k=3.0|+0.667 [-0.889,+2.333]|26/20|900/300|
|I01|E03|source_selected_minus_mean_anchor|k=3.0|-0.222 [-1.556,+1.000]|18/20|900/300|
|I01|E03|source_selected_minus_full_refit|k=3.0|-1.000 [-2.000,-0.108]|7/16|900/300|
|I01|E03|bias_minus_matched_source_control|k=5.0|-1.111 [-2.444,+0.333]|13/23|900/300|
|I01|E03|mean_anchor_minus_matched_source_control|k=5.0|+0.111 [-0.333,+0.667]|3/2|900/300|
|I01|E03|full_refit_minus_matched_source_control|k=5.0|+2.111 [+0.889,+3.444]|26/7|900/300|
|I01|E03|source_selected_minus_matched_source_control|k=5.0|+1.889 [+0.667,+3.111]|24/7|900/300|
|I01|E03|source_selected_minus_bias|k=5.0|+3.000 [+1.219,+4.667]|44/17|900/300|
|I01|E03|source_selected_minus_mean_anchor|k=5.0|+1.778 [+0.556,+3.111]|26/10|900/300|
|I01|E03|source_selected_minus_full_refit|k=5.0|-0.222 [-0.556,+0.000]|0/2|900/300|
|I01|E03|bias_minus_matched_source_control|k=10.0|-0.222 [-1.556,+1.111]|14/16|900/300|
|I01|E03|mean_anchor_minus_matched_source_control|k=10.0|+0.000 [-0.667,+0.667]|4/4|900/300|
|I01|E03|full_refit_minus_matched_source_control|k=10.0|+3.778 [+2.111,+5.669]|48/14|900/300|
|I01|E03|source_selected_minus_matched_source_control|k=10.0|+3.667 [+2.000,+5.556]|46/13|900/300|
|I01|E03|source_selected_minus_bias|k=10.0|+3.889 [+1.778,+6.111]|53/18|900/300|
|I01|E03|source_selected_minus_mean_anchor|k=10.0|+3.667 [+1.778,+5.558]|48/15|900/300|
|I01|E03|source_selected_minus_full_refit|k=10.0|-0.111 [-0.556,+0.222]|1/2|900/300|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=2.0, policy=diversity|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=2.0, policy=random_601|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=2.0, policy=random_602|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=2.0, policy=random_603|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=2.0, policy=uncertainty|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=6.0, policy=diversity|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=6.0, policy=random_601|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=6.0, policy=random_602|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=6.0, policy=random_603|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=6.0, policy=uncertainty|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=10.0, policy=diversity|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=10.0, policy=random_601|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=10.0, policy=random_602|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=10.0, policy=random_603|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=10.0, policy=uncertainty|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=20.0, policy=diversity|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=20.0, policy=random_601|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=20.0, policy=random_602|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=20.0, policy=random_603|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_matched_source_control|k=-1.0, budget=20.0, policy=uncertainty|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=2.0, policy=diversity|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=2.0, policy=random_601|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=2.0, policy=random_602|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=2.0, policy=random_603|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=2.0, policy=uncertainty|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=6.0, policy=diversity|+0.333 [+0.000,+1.301]|4/0|1200/261|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=6.0, policy=random_601|+0.083 [+0.000,+0.325]|1/0|1200/261|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=6.0, policy=random_602|+0.083 [+0.000,+0.325]|1/0|1200/261|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=6.0, policy=random_603|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=6.0, policy=uncertainty|+0.333 [+0.000,+1.301]|4/0|1200/261|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=10.0, policy=diversity|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=10.0, policy=random_601|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=10.0, policy=random_602|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=10.0, policy=random_603|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=10.0, policy=uncertainty|+0.833 [+0.000,+3.252]|10/0|1200/261|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=20.0, policy=diversity|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=20.0, policy=random_601|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=20.0, policy=random_602|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=20.0, policy=random_603|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=20.0, policy=uncertainty|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=2.0, policy=diversity|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=2.0, policy=random_601|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=2.0, policy=random_602|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=2.0, policy=random_603|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=2.0, policy=uncertainty|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|uncertainty_minus_random_601|budget=2.0, policy=uncertainty, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|uncertainty_minus_random_602|budget=2.0, policy=uncertainty, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|uncertainty_minus_random_603|budget=2.0, policy=uncertainty, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|diversity_minus_random_601|budget=2.0, policy=diversity, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|diversity_minus_random_602|budget=2.0, policy=diversity, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|diversity_minus_random_603|budget=2.0, policy=diversity, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|uncertainty_minus_random_601|budget=2.0, policy=uncertainty, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|uncertainty_minus_random_602|budget=2.0, policy=uncertainty, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|uncertainty_minus_random_603|budget=2.0, policy=uncertainty, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|diversity_minus_random_601|budget=2.0, policy=diversity, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|diversity_minus_random_602|budget=2.0, policy=diversity, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|diversity_minus_random_603|budget=2.0, policy=diversity, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=6.0, policy=diversity|-0.333 [-1.301,+0.000]|0/4|1200/261|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=6.0, policy=random_601|-0.083 [-0.325,+0.000]|0/1|1200/261|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=6.0, policy=random_602|-0.083 [-0.325,+0.000]|0/1|1200/261|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=6.0, policy=random_603|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=6.0, policy=uncertainty|-0.333 [-1.301,+0.000]|0/4|1200/261|
|I02|E01|uncertainty_minus_random_601|budget=6.0, policy=uncertainty, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|uncertainty_minus_random_602|budget=6.0, policy=uncertainty, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|uncertainty_minus_random_603|budget=6.0, policy=uncertainty, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|diversity_minus_random_601|budget=6.0, policy=diversity, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|diversity_minus_random_602|budget=6.0, policy=diversity, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|diversity_minus_random_603|budget=6.0, policy=diversity, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|uncertainty_minus_random_601|budget=6.0, policy=uncertainty, reference_policy=random_601|+0.250 [+0.000,+0.976]|4/1|1200/261|
|I02|E01|uncertainty_minus_random_602|budget=6.0, policy=uncertainty, reference_policy=random_602|+0.250 [+0.000,+0.976]|4/1|1200/261|
|I02|E01|uncertainty_minus_random_603|budget=6.0, policy=uncertainty, reference_policy=random_603|+0.333 [+0.000,+1.301]|4/0|1200/261|
|I02|E01|diversity_minus_random_601|budget=6.0, policy=diversity, reference_policy=random_601|+0.250 [+0.000,+0.976]|4/1|1200/261|
|I02|E01|diversity_minus_random_602|budget=6.0, policy=diversity, reference_policy=random_602|+0.250 [+0.000,+0.976]|4/1|1200/261|
|I02|E01|diversity_minus_random_603|budget=6.0, policy=diversity, reference_policy=random_603|+0.333 [+0.000,+1.301]|4/0|1200/261|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=10.0, policy=diversity|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=10.0, policy=random_601|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=10.0, policy=random_602|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=10.0, policy=random_603|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=10.0, policy=uncertainty|-0.833 [-3.252,+0.000]|0/10|1200/261|
|I02|E01|uncertainty_minus_random_601|budget=10.0, policy=uncertainty, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|uncertainty_minus_random_602|budget=10.0, policy=uncertainty, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|uncertainty_minus_random_603|budget=10.0, policy=uncertainty, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|diversity_minus_random_601|budget=10.0, policy=diversity, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|diversity_minus_random_602|budget=10.0, policy=diversity, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|diversity_minus_random_603|budget=10.0, policy=diversity, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|uncertainty_minus_random_601|budget=10.0, policy=uncertainty, reference_policy=random_601|+0.833 [+0.000,+3.252]|10/0|1200/261|
|I02|E01|uncertainty_minus_random_602|budget=10.0, policy=uncertainty, reference_policy=random_602|+0.833 [+0.000,+3.252]|10/0|1200/261|
|I02|E01|uncertainty_minus_random_603|budget=10.0, policy=uncertainty, reference_policy=random_603|+0.833 [+0.000,+3.252]|10/0|1200/261|
|I02|E01|diversity_minus_random_601|budget=10.0, policy=diversity, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|diversity_minus_random_602|budget=10.0, policy=diversity, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|diversity_minus_random_603|budget=10.0, policy=diversity, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=20.0, policy=diversity|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=20.0, policy=random_601|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=20.0, policy=random_602|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=20.0, policy=random_603|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|full_refit_minus_guarded_mean_anchor|budget=20.0, policy=uncertainty|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|uncertainty_minus_random_601|budget=20.0, policy=uncertainty, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|uncertainty_minus_random_602|budget=20.0, policy=uncertainty, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|uncertainty_minus_random_603|budget=20.0, policy=uncertainty, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|diversity_minus_random_601|budget=20.0, policy=diversity, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|diversity_minus_random_602|budget=20.0, policy=diversity, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|diversity_minus_random_603|budget=20.0, policy=diversity, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|uncertainty_minus_random_601|budget=20.0, policy=uncertainty, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|uncertainty_minus_random_602|budget=20.0, policy=uncertainty, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|uncertainty_minus_random_603|budget=20.0, policy=uncertainty, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|diversity_minus_random_601|budget=20.0, policy=diversity, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|diversity_minus_random_602|budget=20.0, policy=diversity, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E01|diversity_minus_random_603|budget=20.0, policy=diversity, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I02|E03|full_refit_minus_matched_source_control|k=-1.0, budget=5.0, policy=diversity|+0.333 [-0.333,+0.889]|6/3|900/300|
|I02|E03|full_refit_minus_matched_source_control|k=-1.0, budget=5.0, policy=random_601|+0.222 [-0.667,+1.111]|9/7|900/300|
|I02|E03|full_refit_minus_matched_source_control|k=-1.0, budget=5.0, policy=random_602|+0.000 [-0.778,+0.778]|6/6|900/300|
|I02|E03|full_refit_minus_matched_source_control|k=-1.0, budget=5.0, policy=random_603|+0.222 [-0.556,+1.000]|8/6|900/300|
|I02|E03|full_refit_minus_matched_source_control|k=-1.0, budget=5.0, policy=uncertainty|+1.667 [+0.778,+2.667]|18/3|900/300|
|I02|E03|full_refit_minus_matched_source_control|k=-1.0, budget=15.0, policy=diversity|+0.778 [-0.222,+1.889]|16/9|900/300|
|I02|E03|full_refit_minus_matched_source_control|k=-1.0, budget=15.0, policy=random_601|+1.556 [+0.333,+2.778]|23/9|900/300|
|I02|E03|full_refit_minus_matched_source_control|k=-1.0, budget=15.0, policy=random_602|+0.889 [-0.333,+2.111]|19/11|900/300|
|I02|E03|full_refit_minus_matched_source_control|k=-1.0, budget=15.0, policy=random_603|+0.556 [-0.556,+1.667]|15/10|900/300|
|I02|E03|full_refit_minus_matched_source_control|k=-1.0, budget=15.0, policy=uncertainty|+2.667 [+1.222,+4.222]|38/14|900/300|
|I02|E03|full_refit_minus_matched_source_control|k=-1.0, budget=25.0, policy=diversity|+1.889 [+0.444,+3.444]|31/14|900/300|
|I02|E03|full_refit_minus_matched_source_control|k=-1.0, budget=25.0, policy=random_601|+1.889 [+0.444,+3.556]|31/14|900/300|
|I02|E03|full_refit_minus_matched_source_control|k=-1.0, budget=25.0, policy=random_602|+2.000 [+0.556,+3.669]|34/16|900/300|
|I02|E03|full_refit_minus_matched_source_control|k=-1.0, budget=25.0, policy=random_603|+2.111 [+0.778,+3.447]|28/9|900/300|
|I02|E03|full_refit_minus_matched_source_control|k=-1.0, budget=25.0, policy=uncertainty|+3.111 [+1.556,+5.000]|45/17|900/300|
|I02|E03|full_refit_minus_matched_source_control|k=-1.0, budget=50.0, policy=diversity|+3.778 [+2.111,+5.669]|48/14|900/300|
|I02|E03|full_refit_minus_matched_source_control|k=-1.0, budget=50.0, policy=random_601|+3.778 [+2.111,+5.669]|48/14|900/300|
|I02|E03|full_refit_minus_matched_source_control|k=-1.0, budget=50.0, policy=random_602|+3.778 [+2.111,+5.669]|48/14|900/300|
|I02|E03|full_refit_minus_matched_source_control|k=-1.0, budget=50.0, policy=random_603|+3.778 [+2.111,+5.669]|48/14|900/300|
|I02|E03|full_refit_minus_matched_source_control|k=-1.0, budget=50.0, policy=uncertainty|+3.778 [+2.111,+5.669]|48/14|900/300|
|I02|E03|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=5.0, policy=diversity|+0.000 [+0.000,+0.000]|0/0|900/300|
|I02|E03|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=5.0, policy=random_601|+0.000 [+0.000,+0.000]|0/0|900/300|
|I02|E03|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=5.0, policy=random_602|+0.000 [+0.000,+0.000]|0/0|900/300|
|I02|E03|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=5.0, policy=random_603|-0.556 [-1.222,+0.111]|2/7|900/300|
|I02|E03|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=5.0, policy=uncertainty|+0.000 [+0.000,+0.000]|0/0|900/300|
|I02|E03|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=15.0, policy=diversity|-0.444 [-1.111,+0.111]|3/7|900/300|
|I02|E03|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=15.0, policy=random_601|-0.667 [-1.778,+0.336]|8/14|900/300|
|I02|E03|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=15.0, policy=random_602|-0.333 [-1.222,+0.444]|6/9|900/300|
|I02|E03|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=15.0, policy=random_603|-0.444 [-1.444,+0.556]|8/12|900/300|
|I02|E03|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=15.0, policy=uncertainty|-1.889 [-3.222,-0.556]|8/25|900/300|
|I02|E03|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=25.0, policy=diversity|-0.333 [-1.111,+0.336]|4/7|900/300|
|I02|E03|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=25.0, policy=random_601|-0.333 [-1.114,+0.444]|6/9|900/300|
|I02|E03|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=25.0, policy=random_602|+0.111 [-0.444,+0.667]|4/3|900/300|
|I02|E03|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=25.0, policy=random_603|+0.000 [-0.669,+0.667]|5/5|900/300|
|I02|E03|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=25.0, policy=uncertainty|-0.444 [-1.333,+0.333]|5/9|900/300|
|I02|E03|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=50.0, policy=diversity|+0.000 [-0.667,+0.667]|4/4|900/300|
|I02|E03|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=50.0, policy=random_601|+0.000 [-0.667,+0.667]|4/4|900/300|
|I02|E03|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=50.0, policy=random_602|+0.000 [-0.667,+0.667]|4/4|900/300|
|I02|E03|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=50.0, policy=random_603|+0.000 [-0.667,+0.667]|4/4|900/300|
|I02|E03|guarded_mean_anchor_minus_matched_source_control|k=-1.0, budget=50.0, policy=uncertainty|+0.000 [-0.667,+0.667]|4/4|900/300|
|I02|E03|full_refit_minus_guarded_mean_anchor|budget=5.0, policy=diversity|+0.333 [-0.333,+0.889]|6/3|900/300|
|I02|E03|full_refit_minus_guarded_mean_anchor|budget=5.0, policy=random_601|+0.222 [-0.667,+1.111]|9/7|900/300|
|I02|E03|full_refit_minus_guarded_mean_anchor|budget=5.0, policy=random_602|+0.000 [-0.778,+0.778]|6/6|900/300|
|I02|E03|full_refit_minus_guarded_mean_anchor|budget=5.0, policy=random_603|+0.778 [-0.222,+1.889]|14/7|900/300|
|I02|E03|full_refit_minus_guarded_mean_anchor|budget=5.0, policy=uncertainty|+1.667 [+0.778,+2.667]|18/3|900/300|
|I02|E03|uncertainty_minus_random_601|budget=5.0, policy=uncertainty, reference_policy=random_601|+1.444 [+0.222,+2.889]|23/10|900/300|
|I02|E03|uncertainty_minus_random_602|budget=5.0, policy=uncertainty, reference_policy=random_602|+1.667 [+0.556,+2.892]|20/5|900/300|
|I02|E03|uncertainty_minus_random_603|budget=5.0, policy=uncertainty, reference_policy=random_603|+1.444 [+0.444,+2.444]|18/5|900/300|
|I02|E03|diversity_minus_random_601|budget=5.0, policy=diversity, reference_policy=random_601|+0.111 [-1.000,+1.111]|12/11|900/300|
|I02|E03|diversity_minus_random_602|budget=5.0, policy=diversity, reference_policy=random_602|+0.333 [-0.556,+1.111]|8/5|900/300|
|I02|E03|diversity_minus_random_603|budget=5.0, policy=diversity, reference_policy=random_603|+0.111 [-0.889,+1.000]|9/8|900/300|
|I02|E03|uncertainty_minus_random_601|budget=5.0, policy=uncertainty, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|900/300|
|I02|E03|uncertainty_minus_random_602|budget=5.0, policy=uncertainty, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|900/300|
|I02|E03|uncertainty_minus_random_603|budget=5.0, policy=uncertainty, reference_policy=random_603|+0.556 [-0.111,+1.222]|7/2|900/300|
|I02|E03|diversity_minus_random_601|budget=5.0, policy=diversity, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|900/300|
|I02|E03|diversity_minus_random_602|budget=5.0, policy=diversity, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|900/300|
|I02|E03|diversity_minus_random_603|budget=5.0, policy=diversity, reference_policy=random_603|+0.556 [-0.111,+1.222]|7/2|900/300|
|I02|E03|full_refit_minus_guarded_mean_anchor|budget=15.0, policy=diversity|+1.222 [+0.000,+2.556]|23/12|900/300|
|I02|E03|full_refit_minus_guarded_mean_anchor|budget=15.0, policy=random_601|+2.222 [+0.778,+3.781]|35/15|900/300|
|I02|E03|full_refit_minus_guarded_mean_anchor|budget=15.0, policy=random_602|+1.222 [-0.111,+2.556]|27/16|900/300|
|I02|E03|full_refit_minus_guarded_mean_anchor|budget=15.0, policy=random_603|+1.000 [-0.333,+2.444]|24/15|900/300|
|I02|E03|full_refit_minus_guarded_mean_anchor|budget=15.0, policy=uncertainty|+4.556 [+2.667,+6.444]|60/19|900/300|
|I02|E03|uncertainty_minus_random_601|budget=15.0, policy=uncertainty, reference_policy=random_601|+1.111 [-0.111,+2.556]|27/17|900/300|
|I02|E03|uncertainty_minus_random_602|budget=15.0, policy=uncertainty, reference_policy=random_602|+1.778 [+0.222,+3.444]|33/17|900/300|
|I02|E03|uncertainty_minus_random_603|budget=15.0, policy=uncertainty, reference_policy=random_603|+2.111 [+0.667,+3.667]|32/13|900/300|
|I02|E03|diversity_minus_random_601|budget=15.0, policy=diversity, reference_policy=random_601|-0.778 [-1.889,+0.444]|14/21|900/300|
|I02|E03|diversity_minus_random_602|budget=15.0, policy=diversity, reference_policy=random_602|-0.111 [-1.225,+1.000]|15/16|900/300|
|I02|E03|diversity_minus_random_603|budget=15.0, policy=diversity, reference_policy=random_603|+0.222 [-0.778,+1.222]|12/10|900/300|
|I02|E03|uncertainty_minus_random_601|budget=15.0, policy=uncertainty, reference_policy=random_601|-1.222 [-2.667,+0.333]|18/29|900/300|
|I02|E03|uncertainty_minus_random_602|budget=15.0, policy=uncertainty, reference_policy=random_602|-1.556 [-2.889,-0.333]|9/23|900/300|
|I02|E03|uncertainty_minus_random_603|budget=15.0, policy=uncertainty, reference_policy=random_603|-1.444 [-2.778,-0.111]|13/26|900/300|
|I02|E03|diversity_minus_random_601|budget=15.0, policy=diversity, reference_policy=random_601|+0.222 [-0.889,+1.336]|14/12|900/300|
|I02|E03|diversity_minus_random_602|budget=15.0, policy=diversity, reference_policy=random_602|-0.111 [-0.889,+0.556]|7/8|900/300|
|I02|E03|diversity_minus_random_603|budget=15.0, policy=diversity, reference_policy=random_603|+0.000 [-1.111,+1.111]|12/12|900/300|
|I02|E03|full_refit_minus_guarded_mean_anchor|budget=25.0, policy=diversity|+2.222 [+0.667,+3.889]|35/15|900/300|
|I02|E03|full_refit_minus_guarded_mean_anchor|budget=25.0, policy=random_601|+2.222 [+0.667,+3.892]|37/17|900/300|
|I02|E03|full_refit_minus_guarded_mean_anchor|budget=25.0, policy=random_602|+1.889 [+0.333,+3.556]|34/17|900/300|
|I02|E03|full_refit_minus_guarded_mean_anchor|budget=25.0, policy=random_603|+2.111 [+0.889,+3.556]|29/10|900/300|
|I02|E03|full_refit_minus_guarded_mean_anchor|budget=25.0, policy=uncertainty|+3.556 [+1.889,+5.447]|49/17|900/300|
|I02|E03|uncertainty_minus_random_601|budget=25.0, policy=uncertainty, reference_policy=random_601|+1.222 [-0.111,+2.778]|26/15|900/300|
|I02|E03|uncertainty_minus_random_602|budget=25.0, policy=uncertainty, reference_policy=random_602|+1.111 [-0.111,+2.556]|24/14|900/300|
|I02|E03|uncertainty_minus_random_603|budget=25.0, policy=uncertainty, reference_policy=random_603|+1.000 [-0.333,+2.444]|25/16|900/300|
|I02|E03|diversity_minus_random_601|budget=25.0, policy=diversity, reference_policy=random_601|+0.000 [-1.333,+1.222]|19/19|900/300|
|I02|E03|diversity_minus_random_602|budget=25.0, policy=diversity, reference_policy=random_602|-0.111 [-1.222,+1.000]|16/17|900/300|
|I02|E03|diversity_minus_random_603|budget=25.0, policy=diversity, reference_policy=random_603|-0.222 [-1.556,+1.222]|19/21|900/300|
|I02|E03|uncertainty_minus_random_601|budget=25.0, policy=uncertainty, reference_policy=random_601|-0.111 [-1.000,+0.778]|11/12|900/300|
|I02|E03|uncertainty_minus_random_602|budget=25.0, policy=uncertainty, reference_policy=random_602|-0.556 [-1.222,+0.111]|3/8|900/300|
|I02|E03|uncertainty_minus_random_603|budget=25.0, policy=uncertainty, reference_policy=random_603|-0.444 [-1.222,+0.333]|5/9|900/300|
|I02|E03|diversity_minus_random_601|budget=25.0, policy=diversity, reference_policy=random_601|+0.000 [-0.781,+0.778]|8/8|900/300|
|I02|E03|diversity_minus_random_602|budget=25.0, policy=diversity, reference_policy=random_602|-0.444 [-1.111,+0.222]|3/7|900/300|
|I02|E03|diversity_minus_random_603|budget=25.0, policy=diversity, reference_policy=random_603|-0.333 [-1.000,+0.333]|3/6|900/300|
|I02|E03|full_refit_minus_guarded_mean_anchor|budget=50.0, policy=diversity|+3.778 [+1.889,+5.667]|50/16|900/300|
|I02|E03|full_refit_minus_guarded_mean_anchor|budget=50.0, policy=random_601|+3.778 [+1.889,+5.667]|50/16|900/300|
|I02|E03|full_refit_minus_guarded_mean_anchor|budget=50.0, policy=random_602|+3.778 [+1.889,+5.667]|50/16|900/300|
|I02|E03|full_refit_minus_guarded_mean_anchor|budget=50.0, policy=random_603|+3.778 [+1.889,+5.667]|50/16|900/300|
|I02|E03|full_refit_minus_guarded_mean_anchor|budget=50.0, policy=uncertainty|+3.778 [+1.889,+5.667]|50/16|900/300|
|I02|E03|uncertainty_minus_random_601|budget=50.0, policy=uncertainty, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|900/300|
|I02|E03|uncertainty_minus_random_602|budget=50.0, policy=uncertainty, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|900/300|
|I02|E03|uncertainty_minus_random_603|budget=50.0, policy=uncertainty, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|900/300|
|I02|E03|diversity_minus_random_601|budget=50.0, policy=diversity, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|900/300|
|I02|E03|diversity_minus_random_602|budget=50.0, policy=diversity, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|900/300|
|I02|E03|diversity_minus_random_603|budget=50.0, policy=diversity, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|900/300|
|I02|E03|uncertainty_minus_random_601|budget=50.0, policy=uncertainty, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|900/300|
|I02|E03|uncertainty_minus_random_602|budget=50.0, policy=uncertainty, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|900/300|
|I02|E03|uncertainty_minus_random_603|budget=50.0, policy=uncertainty, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|900/300|
|I02|E03|diversity_minus_random_601|budget=50.0, policy=diversity, reference_policy=random_601|+0.000 [+0.000,+0.000]|0/0|900/300|
|I02|E03|diversity_minus_random_602|budget=50.0, policy=diversity, reference_policy=random_602|+0.000 [+0.000,+0.000]|0/0|900/300|
|I02|E03|diversity_minus_random_603|budget=50.0, policy=diversity, reference_policy=random_603|+0.000 [+0.000,+0.000]|0/0|900/300|
|I03|E01|raw_mean_anchor_minus_identity|k=1.0|+0.028 [+0.000,+0.108]|1/0|3600/261|
|I03|E01|ovr_trace1_minus_identity|k=1.0|+0.111 [+0.000,+0.434]|4/0|3600/261|
|I03|E01|ovo_trace1_minus_identity|k=1.0|+0.111 [+0.000,+0.434]|4/0|3600/261|
|I03|E01|ovo_pca_trace1_minus_identity|k=1.0|+0.111 [+0.000,+0.434]|4/0|3600/261|
|I03|E01|ovr_trace1_minus_raw_mean_anchor|k=1.0|+0.083 [+0.000,+0.325]|3/0|3600/261|
|I03|E01|ovo_trace1_minus_ovr_trace1|k=1.0|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I03|E01|ovo_pca_trace1_minus_ovo_trace1|k=1.0|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I03|E01|raw_mean_anchor_minus_identity|k=3.0|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I03|E01|ovr_trace1_minus_identity|k=3.0|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I03|E01|ovo_trace1_minus_identity|k=3.0|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I03|E01|ovo_pca_trace1_minus_identity|k=3.0|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I03|E01|ovr_trace1_minus_raw_mean_anchor|k=3.0|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I03|E01|ovo_trace1_minus_ovr_trace1|k=3.0|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I03|E01|ovo_pca_trace1_minus_ovo_trace1|k=3.0|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I03|E01|raw_mean_anchor_minus_identity|k=5.0|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I03|E01|ovr_trace1_minus_identity|k=5.0|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I03|E01|ovo_trace1_minus_identity|k=5.0|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I03|E01|ovo_pca_trace1_minus_identity|k=5.0|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I03|E01|ovr_trace1_minus_raw_mean_anchor|k=5.0|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I03|E01|ovo_trace1_minus_ovr_trace1|k=5.0|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I03|E01|ovo_pca_trace1_minus_ovo_trace1|k=5.0|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I03|E01|raw_mean_anchor_minus_identity|k=10.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I03|E01|ovr_trace1_minus_identity|k=10.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I03|E01|ovo_trace1_minus_identity|k=10.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I03|E01|ovo_pca_trace1_minus_identity|k=10.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I03|E01|ovr_trace1_minus_raw_mean_anchor|k=10.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I03|E01|ovo_trace1_minus_ovr_trace1|k=10.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I03|E01|ovo_pca_trace1_minus_ovo_trace1|k=10.0|+0.000 [+0.000,+0.000]|0/0|1200/261|
|I03|E03|raw_mean_anchor_minus_identity|k=1.0|-4.926 [-6.482,-3.259]|63/196|2700/300|
|I03|E03|ovr_trace1_minus_identity|k=1.0|-0.037 [-0.519,+0.519]|13/14|2700/300|
|I03|E03|ovo_trace1_minus_identity|k=1.0|-0.148 [-0.630,+0.370]|12/16|2700/300|
|I03|E03|ovo_pca_trace1_minus_identity|k=1.0|-0.148 [-0.630,+0.370]|12/16|2700/300|
|I03|E03|ovr_trace1_minus_raw_mean_anchor|k=1.0|+4.889 [+3.259,+6.370]|182/50|2700/300|
|I03|E03|ovo_trace1_minus_ovr_trace1|k=1.0|-0.111 [-0.222,+0.000]|0/3|2700/300|
|I03|E03|ovo_pca_trace1_minus_ovo_trace1|k=1.0|+0.000 [+0.000,+0.000]|0/0|2700/300|
|I03|E03|raw_mean_anchor_minus_identity|k=3.0|-0.333 [-1.000,+0.333]|19/28|2700/300|
|I03|E03|ovr_trace1_minus_identity|k=3.0|+0.148 [-0.222,+0.556]|8/4|2700/300|
|I03|E03|ovo_trace1_minus_identity|k=3.0|+0.259 [-0.185,+0.741]|11/4|2700/300|
|I03|E03|ovo_pca_trace1_minus_identity|k=3.0|+0.259 [-0.185,+0.741]|11/4|2700/300|
|I03|E03|ovr_trace1_minus_raw_mean_anchor|k=3.0|+0.481 [-0.074,+0.926]|24/11|2700/300|
|I03|E03|ovo_trace1_minus_ovr_trace1|k=3.0|+0.111 [+0.000,+0.296]|3/0|2700/300|
|I03|E03|ovo_pca_trace1_minus_ovo_trace1|k=3.0|+0.000 [+0.000,+0.000]|0/0|2700/300|
|I03|E03|raw_mean_anchor_minus_identity|k=5.0|-0.037 [-0.630,+0.556]|15/16|2700/300|
|I03|E03|ovr_trace1_minus_identity|k=5.0|+0.185 [-0.259,+0.667]|10/5|2700/300|
|I03|E03|ovo_trace1_minus_identity|k=5.0|+0.333 [-0.037,+0.741]|11/2|2700/300|
|I03|E03|ovo_pca_trace1_minus_identity|k=5.0|+0.296 [-0.111,+0.741]|11/3|2700/300|
|I03|E03|ovr_trace1_minus_raw_mean_anchor|k=5.0|+0.222 [-0.111,+0.556]|12/6|2700/300|
|I03|E03|ovo_trace1_minus_ovr_trace1|k=5.0|+0.148 [+0.000,+0.333]|4/0|2700/300|
|I03|E03|ovo_pca_trace1_minus_ovo_trace1|k=5.0|-0.037 [-0.111,+0.000]|0/1|2700/300|
|I03|E03|raw_mean_anchor_minus_identity|k=10.0|+0.000 [-0.667,+0.667]|4/4|900/300|
|I03|E03|ovr_trace1_minus_identity|k=10.0|+0.000 [-0.444,+0.444]|2/2|900/300|
|I03|E03|ovo_trace1_minus_identity|k=10.0|+0.444 [+0.111,+0.889]|4/0|900/300|
|I03|E03|ovo_pca_trace1_minus_identity|k=10.0|+0.333 [-0.111,+0.778]|4/1|900/300|
|I03|E03|ovr_trace1_minus_raw_mean_anchor|k=10.0|+0.000 [-0.444,+0.444]|2/2|900/300|
|I03|E03|ovo_trace1_minus_ovr_trace1|k=10.0|+0.444 [+0.111,+0.889]|4/0|900/300|
|I03|E03|ovo_pca_trace1_minus_ovo_trace1|k=10.0|-0.111 [-0.333,+0.000]|0/1|900/300|
|I04|E01|weighted_lambda1_minus_identity|k=-1.0, composition_id=balanced|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I04|E01|weighted_mean_minus_identity|k=-1.0, composition_id=balanced|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I04|E01|weighted_ess_minus_identity|k=-1.0, composition_id=balanced|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I04|E01|weighted_mean_minus_weighted_lambda1|k=-1.0, composition_id=balanced|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I04|E01|weighted_ess_minus_weighted_mean|k=-1.0, composition_id=balanced|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I04|E01|weighted_lambda1_minus_identity|k=-1.0, composition_id=binary_1_9|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I04|E01|weighted_mean_minus_identity|k=-1.0, composition_id=binary_1_9|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I04|E01|weighted_ess_minus_identity|k=-1.0, composition_id=binary_1_9|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I04|E01|weighted_mean_minus_weighted_lambda1|k=-1.0, composition_id=binary_1_9|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I04|E01|weighted_ess_minus_weighted_mean|k=-1.0, composition_id=binary_1_9|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I04|E01|weighted_lambda1_minus_identity|k=-1.0, composition_id=binary_3_7|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I04|E01|weighted_mean_minus_identity|k=-1.0, composition_id=binary_3_7|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I04|E01|weighted_ess_minus_identity|k=-1.0, composition_id=binary_3_7|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I04|E01|weighted_mean_minus_weighted_lambda1|k=-1.0, composition_id=binary_3_7|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I04|E01|weighted_ess_minus_weighted_mean|k=-1.0, composition_id=binary_3_7|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I04|E01|weighted_lambda1_minus_identity|k=-1.0, composition_id=binary_7_3|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I04|E01|weighted_mean_minus_identity|k=-1.0, composition_id=binary_7_3|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I04|E01|weighted_ess_minus_identity|k=-1.0, composition_id=binary_7_3|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I04|E01|weighted_mean_minus_weighted_lambda1|k=-1.0, composition_id=binary_7_3|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I04|E01|weighted_ess_minus_weighted_mean|k=-1.0, composition_id=binary_7_3|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I04|E01|weighted_lambda1_minus_identity|k=-1.0, composition_id=binary_9_1|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I04|E01|weighted_mean_minus_identity|k=-1.0, composition_id=binary_9_1|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I04|E01|weighted_ess_minus_identity|k=-1.0, composition_id=binary_9_1|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I04|E01|weighted_mean_minus_weighted_lambda1|k=-1.0, composition_id=binary_9_1|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I04|E01|weighted_ess_minus_weighted_mean|k=-1.0, composition_id=binary_9_1|+0.000 [+0.000,+0.000]|0/0|3600/261|
|I04|E03|weighted_lambda1_minus_identity|k=-1.0, composition_id=balanced|-2.333 [-3.556,-1.148]|41/104|2700/300|
|I04|E03|weighted_mean_minus_identity|k=-1.0, composition_id=balanced|-0.037 [-0.630,+0.556]|15/16|2700/300|
|I04|E03|weighted_ess_minus_identity|k=-1.0, composition_id=balanced|-0.037 [-0.630,+0.556]|15/16|2700/300|
|I04|E03|weighted_mean_minus_weighted_lambda1|k=-1.0, composition_id=balanced|+2.296 [+1.296,+3.371]|92/30|2700/300|
|I04|E03|weighted_ess_minus_weighted_mean|k=-1.0, composition_id=balanced|+0.000 [+0.000,+0.000]|0/0|2700/300|
|I04|E03|weighted_lambda1_minus_identity|k=-1.0, composition_id=scarce_ZY03|-7.889 [-10.186,-5.665]|82/295|2700/300|
|I04|E03|weighted_mean_minus_identity|k=-1.0, composition_id=scarce_ZY03|-1.630 [-2.594,-0.704]|26/70|2700/300|
|I04|E03|weighted_ess_minus_identity|k=-1.0, composition_id=scarce_ZY03|-0.704 [-1.407,-0.074]|20/39|2700/300|
|I04|E03|weighted_mean_minus_weighted_lambda1|k=-1.0, composition_id=scarce_ZY03|+6.259 [+4.259,+8.370]|245/76|2700/300|
|I04|E03|weighted_ess_minus_weighted_mean|k=-1.0, composition_id=scarce_ZY03|+0.926 [+0.407,+1.481]|33/8|2700/300|
|I04|E03|weighted_lambda1_minus_identity|k=-1.0, composition_id=scarce_ZY04|-5.630 [-7.408,-3.702]|60/212|2700/300|
|I04|E03|weighted_mean_minus_identity|k=-1.0, composition_id=scarce_ZY04|-0.889 [-1.704,+0.000]|21/45|2700/300|
|I04|E03|weighted_ess_minus_identity|k=-1.0, composition_id=scarce_ZY04|-0.704 [-1.482,+0.148]|19/38|2700/300|
|I04|E03|weighted_mean_minus_weighted_lambda1|k=-1.0, composition_id=scarce_ZY04|+4.741 [+3.259,+6.148]|172/44|2700/300|
|I04|E03|weighted_ess_minus_weighted_mean|k=-1.0, composition_id=scarce_ZY04|+0.185 [-0.037,+0.444]|7/2|2700/300|
|I04|E03|weighted_lambda1_minus_identity|k=-1.0, composition_id=scarce_ZY05|-3.963 [-5.296,-2.593]|35/142|2700/300|
|I04|E03|weighted_mean_minus_identity|k=-1.0, composition_id=scarce_ZY05|-0.222 [-0.815,+0.371]|14/20|2700/300|
|I04|E03|weighted_ess_minus_identity|k=-1.0, composition_id=scarce_ZY05|-0.111 [-0.630,+0.408]|11/14|2700/300|
|I04|E03|weighted_mean_minus_weighted_lambda1|k=-1.0, composition_id=scarce_ZY05|+3.741 [+2.481,+5.075]|131/30|2700/300|
|I04|E03|weighted_ess_minus_weighted_mean|k=-1.0, composition_id=scarce_ZY05|+0.111 [-0.148,+0.370]|6/3|2700/300|
|I04|E03|weighted_lambda1_minus_identity|k=-1.0, composition_id=scarce_ZY06|-7.407 [-9.481,-5.333]|81/281|2700/300|
|I04|E03|weighted_mean_minus_identity|k=-1.0, composition_id=scarce_ZY06|-1.593 [-2.816,-0.370]|32/75|2700/300|
|I04|E03|weighted_ess_minus_identity|k=-1.0, composition_id=scarce_ZY06|-1.444 [-2.593,-0.296]|26/65|2700/300|
|I04|E03|weighted_mean_minus_weighted_lambda1|k=-1.0, composition_id=scarce_ZY06|+5.815 [+4.148,+7.481]|223/66|2700/300|
|I04|E03|weighted_ess_minus_weighted_mean|k=-1.0, composition_id=scarce_ZY06|+0.148 [-0.259,+0.519]|11/7|2700/300|
|I04|E03|weighted_lambda1_minus_identity|k=-1.0, composition_id=scarce_ZY07|-7.333 [-9.185,-5.444]|62/260|2700/300|
|I04|E03|weighted_mean_minus_identity|k=-1.0, composition_id=scarce_ZY07|-1.593 [-2.630,-0.556]|16/59|2700/300|
|I04|E03|weighted_ess_minus_identity|k=-1.0, composition_id=scarce_ZY07|-1.407 [-2.444,-0.333]|20/58|2700/300|
|I04|E03|weighted_mean_minus_weighted_lambda1|k=-1.0, composition_id=scarce_ZY07|+5.741 [+4.111,+7.519]|218/63|2700/300|
|I04|E03|weighted_ess_minus_weighted_mean|k=-1.0, composition_id=scarce_ZY07|+0.185 [-0.111,+0.519]|12/7|2700/300|

## Source策略与获取成本审计

source_policy.json、inner_plan.json与inner_validation_predictions.csv.gz记录每个Source内选择的依据；汇总再次核验内层角色与指标。Source选择器训练成本与选择后外层端点复用分开计数。acquisition_plan.json及机制表记录实际覆盖、资格回退、观察标签量与拟合用标签量；全池政策端点不重复训练，也不当作多个独立获取结果。

三次Source内划分的实物重叠数、交叉角色复用数与逐次分类器拟合成本保存于mechanism_analysis.csv；它们不构成新增独立验证样品。

|原任务|K/类|Source内选中方法|上下文数|平均最大内验证Macro-F1（选择统计量）|
|---|---:|---|---:|---:|
|E01|1|identity|30|0.993327|
|E01|3|identity|30|0.993327|
|E01|5|identity|30|0.993327|
|E01|10|identity|30|0.993327|
|E03|1|bias|3|0.718789|
|E03|1|full_refit|7|0.755038|
|E03|1|identity|5|0.727062|
|E03|3|bias|2|0.774026|
|E03|3|full_refit|7|0.726532|
|E03|3|identity|2|0.761473|
|E03|3|mean_anchor|4|0.778998|
|E03|5|full_refit|14|0.765115|
|E03|5|identity|1|0.744810|
|E03|10|full_refit|14|0.787733|
|E03|10|identity|1|0.744810|

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
