# B方案第二轮：校准退化、重训分解与来源覆盖稳定性

状态：**DONE**，已封存345/345个任务包。
本报告只读取完整封存预测；不拟合模型、不修改原105任务的代码、数据、标签或结果。

新增完整分类器拟合723次；已封存科研任务CPU 114.547649秒。重复使用的旧模型端点不计为新训练；控制、核验、失败尝试及推送成本另见后台账本。
逐样品预测行数675000，涉及唯一Query实物422；两者不得互换。

## 状态及设计

|实验块|内容|完成/计划|状态|
|---|---|---:|---|
|F01|分数几何、支持重排及乱标诊断|45/45|DONE|
|F02|Scaler × gamma 重训分解|45/45|DONE|
|F03|类别权重 × 来源覆盖 × 类别比例|240/240|DONE|
|F04|五来源固定嵌套学习曲线|15/15|DONE|

当前有效数据仍为18份工作簿、1672份用户声明独立实物。本轮仅在master TRAIN构造角色，不使用原CAL/PRED搜索模型。ZY01/ZY02为枣仁、ZY03—ZY07为酸枣仁；原来源/批次混杂不变。

## F01：支持抽样和分数几何

复用15个五来源以及预先指定30个类别平衡二分类模型。原始Support顺序作为draw0，再固定101—110十个重排种子；每类K嵌套取1/3/5/10。真标签K10完整池仅保留一次，重排不是新的实物或新真标签端点。乱标仅作用于显式诊断副本，保持样品ID与类别计数。

比较原始Anchor、Bias、弱Anchor、对角Anchor、0.1/10倍分数和逐样品行中心化＋Source全局RMS归一化。正比例与行中心化保留原始K0类别边界；归一化统计量只来自Source。完整Anchor包含跨类耦合，对角方法是降低自由度的明确对照。分数不是概率。

## 全部冻结方法的描述性结果

下表按每任务、每个合法draw的端点等权汇总；不从这些结果选择最优配置作为确认性结论。

|块|原任务|方法|K/类|平均Accuracy|平均Macro-F1|平均Balanced Accuracy|端点数|
|---|---|---|---:|---:|---:|---:|---:|
|F01|E01|diagonal_anchor_lambda1|1|0.983485|0.983475|0.983485|330|
|F01|E01|diagonal_anchor_lambda1|3|0.983333|0.983323|0.983333|330|
|F01|E01|diagonal_anchor_lambda1|5|0.983333|0.983323|0.983333|330|
|F01|E01|diagonal_anchor_lambda1|10|0.983333|0.983323|0.983333|30|
|F01|E01|identity|0|0.983333|0.983323|0.983333|30|
|F01|E01|raw_anchor_lambda1|1|0.983485|0.983475|0.983485|330|
|F01|E01|raw_anchor_lambda1|3|0.983333|0.983323|0.983333|330|
|F01|E01|raw_anchor_lambda1|5|0.983333|0.983323|0.983333|330|
|F01|E01|raw_anchor_lambda1|10|0.983333|0.983323|0.983333|30|
|F01|E01|raw_bias_only|1|0.983485|0.983475|0.983485|330|
|F01|E01|raw_bias_only|3|0.983333|0.983323|0.983333|330|
|F01|E01|raw_bias_only|5|0.983333|0.983323|0.983333|330|
|F01|E01|raw_bias_only|10|0.983333|0.983323|0.983333|30|
|F01|E01|raw_weak_lambda1000_blend005|1|0.983333|0.983323|0.983333|330|
|F01|E01|raw_weak_lambda1000_blend005|3|0.983333|0.983323|0.983333|330|
|F01|E01|raw_weak_lambda1000_blend005|5|0.983333|0.983323|0.983333|330|
|F01|E01|raw_weak_lambda1000_blend005|10|0.983333|0.983323|0.983333|30|
|F01|E01|rowcenter_source_rms_anchor|1|0.983485|0.983475|0.983485|330|
|F01|E01|rowcenter_source_rms_anchor|3|0.983333|0.983323|0.983333|330|
|F01|E01|rowcenter_source_rms_anchor|5|0.983333|0.983323|0.983333|330|
|F01|E01|rowcenter_source_rms_anchor|10|0.983333|0.983323|0.983333|30|
|F01|E01|scale01_anchor|1|0.983485|0.983475|0.983485|330|
|F01|E01|scale01_anchor|3|0.983333|0.983323|0.983333|330|
|F01|E01|scale01_anchor|5|0.983333|0.983323|0.983333|330|
|F01|E01|scale01_anchor|10|0.983333|0.983323|0.983333|30|
|F01|E01|scale10_anchor|1|0.983485|0.983475|0.983485|330|
|F01|E01|scale10_anchor|3|0.983333|0.983323|0.983333|330|
|F01|E01|scale10_anchor|5|0.983333|0.983323|0.983333|330|
|F01|E01|scale10_anchor|10|0.983333|0.983323|0.983333|30|
|F01|E01|shuffled_anchor_lambda1|1|0.631667|0.631516|0.631667|330|
|F01|E01|shuffled_anchor_lambda1|3|0.543939|0.543749|0.543939|330|
|F01|E01|shuffled_anchor_lambda1|5|0.631818|0.631664|0.631818|330|
|F01|E01|shuffled_anchor_lambda1|10|0.514545|0.514344|0.514545|330|
|F01|E01|svc_predict|0|0.983333|0.983323|0.983333|30|
|F01|E03|diagonal_anchor_lambda1|1|0.714545|0.685904|0.714545|165|
|F01|E03|diagonal_anchor_lambda1|3|0.767273|0.756258|0.767273|165|
|F01|E03|diagonal_anchor_lambda1|5|0.773737|0.765890|0.773737|165|
|F01|E03|diagonal_anchor_lambda1|10|0.774444|0.768751|0.774444|15|
|F01|E03|identity|0|0.800000|0.798475|0.800000|15|
|F01|E03|raw_anchor_lambda1|1|0.713535|0.696546|0.713535|165|
|F01|E03|raw_anchor_lambda1|3|0.752727|0.742470|0.752727|165|
|F01|E03|raw_anchor_lambda1|5|0.771414|0.764517|0.771414|165|
|F01|E03|raw_anchor_lambda1|10|0.787778|0.783517|0.787778|15|
|F01|E03|raw_bias_only|1|0.775253|0.767307|0.775253|165|
|F01|E03|raw_bias_only|3|0.784747|0.779376|0.784747|165|
|F01|E03|raw_bias_only|5|0.788283|0.783642|0.788283|165|
|F01|E03|raw_bias_only|10|0.797778|0.793716|0.797778|15|
|F01|E03|raw_weak_lambda1000_blend005|1|0.800404|0.798858|0.800404|165|
|F01|E03|raw_weak_lambda1000_blend005|3|0.800303|0.798724|0.800303|165|
|F01|E03|raw_weak_lambda1000_blend005|5|0.800404|0.798837|0.800404|165|
|F01|E03|raw_weak_lambda1000_blend005|10|0.801111|0.799554|0.801111|15|
|F01|E03|rowcenter_source_rms_anchor|1|0.747172|0.733917|0.747172|165|
|F01|E03|rowcenter_source_rms_anchor|3|0.767374|0.759321|0.767374|165|
|F01|E03|rowcenter_source_rms_anchor|5|0.778384|0.772524|0.778384|165|
|F01|E03|rowcenter_source_rms_anchor|10|0.792222|0.788332|0.792222|15|
|F01|E03|scale01_anchor|1|0.748990|0.735793|0.748990|165|
|F01|E03|scale01_anchor|3|0.751212|0.739116|0.751212|165|
|F01|E03|scale01_anchor|5|0.761010|0.751152|0.761010|165|
|F01|E03|scale01_anchor|10|0.775556|0.768460|0.775556|15|
|F01|E03|scale10_anchor|1|0.584040|0.557426|0.584040|165|
|F01|E03|scale10_anchor|3|0.729495|0.716240|0.729495|165|
|F01|E03|scale10_anchor|5|0.759697|0.750189|0.759697|165|
|F01|E03|scale10_anchor|10|0.784444|0.779769|0.784444|15|
|F01|E03|shuffled_anchor_lambda1|1|0.260606|0.226341|0.260606|165|
|F01|E03|shuffled_anchor_lambda1|3|0.312828|0.279594|0.312828|165|
|F01|E03|shuffled_anchor_lambda1|5|0.225253|0.207048|0.225253|165|
|F01|E03|shuffled_anchor_lambda1|10|0.247071|0.220763|0.247071|165|
|F01|E03|svc_predict|0|0.800000|0.798475|0.800000|15|
|F02|E01|identity|0|0.983333|0.983323|0.983333|30|
|F02|E01|scaler_frozen_gamma_fixed|1|0.983333|0.983323|0.983333|30|
|F02|E01|scaler_frozen_gamma_fixed|3|0.983333|0.983323|0.983333|30|
|F02|E01|scaler_frozen_gamma_fixed|5|0.983333|0.983323|0.983333|30|
|F02|E01|scaler_frozen_gamma_fixed|10|0.983333|0.983323|0.983333|30|
|F02|E01|scaler_frozen_gamma_scale|1|0.983333|0.983323|0.983333|30|
|F02|E01|scaler_frozen_gamma_scale|3|0.983333|0.983323|0.983333|30|
|F02|E01|scaler_frozen_gamma_scale|5|0.983333|0.983323|0.983333|30|
|F02|E01|scaler_frozen_gamma_scale|10|0.983333|0.983323|0.983333|30|
|F02|E01|scaler_refit_gamma_fixed|1|0.983333|0.983323|0.983333|30|
|F02|E01|scaler_refit_gamma_fixed|3|0.983333|0.983323|0.983333|30|
|F02|E01|scaler_refit_gamma_fixed|5|0.983333|0.983323|0.983333|30|
|F02|E01|scaler_refit_gamma_fixed|10|0.983333|0.983323|0.983333|30|
|F02|E01|scaler_refit_gamma_scale|1|0.983333|0.983323|0.983333|30|
|F02|E01|scaler_refit_gamma_scale|3|0.983333|0.983323|0.983333|30|
|F02|E01|scaler_refit_gamma_scale|5|0.983333|0.983323|0.983333|30|
|F02|E01|scaler_refit_gamma_scale|10|0.983333|0.983323|0.983333|30|
|F02|E03|identity|0|0.800000|0.798475|0.800000|15|
|F02|E03|scaler_frozen_gamma_fixed|1|0.801111|0.799543|0.801111|15|
|F02|E03|scaler_frozen_gamma_fixed|3|0.802222|0.800661|0.802222|15|
|F02|E03|scaler_frozen_gamma_fixed|5|0.821111|0.820229|0.821111|15|
|F02|E03|scaler_frozen_gamma_fixed|10|0.837778|0.836784|0.837778|15|
|F02|E03|scaler_frozen_gamma_scale|1|0.801111|0.799543|0.801111|15|
|F02|E03|scaler_frozen_gamma_scale|3|0.802222|0.800661|0.802222|15|
|F02|E03|scaler_frozen_gamma_scale|5|0.818889|0.817898|0.818889|15|
|F02|E03|scaler_frozen_gamma_scale|10|0.837778|0.836813|0.837778|15|
|F02|E03|scaler_refit_gamma_fixed|1|0.800000|0.798441|0.800000|15|
|F02|E03|scaler_refit_gamma_fixed|3|0.806667|0.805168|0.806667|15|
|F02|E03|scaler_refit_gamma_fixed|5|0.821111|0.819859|0.821111|15|
|F02|E03|scaler_refit_gamma_fixed|10|0.838889|0.837811|0.838889|15|
|F02|E03|scaler_refit_gamma_scale|1|0.800000|0.798441|0.800000|15|
|F02|E03|scaler_refit_gamma_scale|3|0.806667|0.805168|0.806667|15|
|F02|E03|scaler_refit_gamma_scale|5|0.821111|0.819859|0.821111|15|
|F02|E03|scaler_refit_gamma_scale|10|0.838889|0.837811|0.838889|15|
|F03|E01|class_weight_balanced|0|0.987917|0.987897|0.987917|240|
|F03|E01|class_weight_balanced_argmax|0|0.987917|0.987897|0.987917|240|
|F03|E01|class_weight_none|0|0.987917|0.987897|0.987917|240|
|F03|E01|class_weight_none_argmax|0|0.987917|0.987897|0.987917|240|
|F04|E03|source_n050|0|0.687778|0.684228|0.687778|15|
|F04|E03|source_n100|0|0.744444|0.742911|0.744444|15|
|F04|E03|source_n150|0|0.785556|0.782761|0.785556|15|
|F04|E03|source_n190|0|0.800000|0.798475|0.800000|15|
|F04|E03|source_n240|0|0.838889|0.837811|0.838889|15|

## F02：重训收益的Scaler与gamma分解

保持Source、Support、Query和支持标签预算完全相同，分别冻结/重估StandardScaler、冻结原源模型数值gamma/按新增训练侧重新计算gamma。C固定10，类别权重balanced。原Scaler重估＋gamma重算的端点与上一轮完整重训完全相同，优先复用，不重复训练。

即使冻结Scaler，gamma='scale'也会随训练集合改变，因此不能把两者混为同一因素。逐任务保存有效gamma与Scaler哈希，四个条件的全部配对差值见paired_transitions.csv。对于重新逐列StandardScaler后的完整612点输入，scale gamma通常等于1/612；严格相同的数值与等价目标端点直接复用，不构成独立干预。审计中的ULP级差异也不能解释为生物学或稳健的核宽度机制。

## 配对效应及描述性不确定性

以下先列每块全体Query对齐的主要对照；每来源、逐类Recall代价、全部方法差值与区间保存在CSV。

|块|原任务|比较|K/类|ΔAccuracy百分点及95%区间|纠错/误伤|重复预测/实物|
|---|---|---|---:|---:|---:|---:|
|F01|E01|raw_anchor_lambda1_minus_identity|1|+0.015 [+0.000,+0.059]|2/0|13200/261|
|F01|E01|raw_anchor_lambda1_minus_identity|3|+0.000 [+0.000,+0.000]|0/0|13200/261|
|F01|E01|raw_anchor_lambda1_minus_identity|5|+0.000 [+0.000,+0.000]|0/0|13200/261|
|F01|E01|raw_anchor_lambda1_minus_identity|10|+0.000 [+0.000,+0.000]|0/0|1200/261|
|F01|E01|raw_bias_only_minus_identity|1|+0.015 [+0.000,+0.059]|2/0|13200/261|
|F01|E01|raw_bias_only_minus_identity|3|+0.000 [+0.000,+0.000]|0/0|13200/261|
|F01|E01|raw_bias_only_minus_identity|5|+0.000 [+0.000,+0.000]|0/0|13200/261|
|F01|E01|raw_bias_only_minus_identity|10|+0.000 [+0.000,+0.000]|0/0|1200/261|
|F01|E01|raw_weak_lambda1000_blend005_minus_identity|1|+0.000 [+0.000,+0.000]|0/0|13200/261|
|F01|E01|raw_weak_lambda1000_blend005_minus_identity|3|+0.000 [+0.000,+0.000]|0/0|13200/261|
|F01|E01|raw_weak_lambda1000_blend005_minus_identity|5|+0.000 [+0.000,+0.000]|0/0|13200/261|
|F01|E01|raw_weak_lambda1000_blend005_minus_identity|10|+0.000 [+0.000,+0.000]|0/0|1200/261|
|F01|E01|diagonal_anchor_lambda1_minus_identity|1|+0.015 [+0.000,+0.059]|2/0|13200/261|
|F01|E01|diagonal_anchor_lambda1_minus_identity|3|+0.000 [+0.000,+0.000]|0/0|13200/261|
|F01|E01|diagonal_anchor_lambda1_minus_identity|5|+0.000 [+0.000,+0.000]|0/0|13200/261|
|F01|E01|diagonal_anchor_lambda1_minus_identity|10|+0.000 [+0.000,+0.000]|0/0|1200/261|
|F01|E01|scale01_anchor_minus_identity|1|+0.015 [+0.000,+0.059]|2/0|13200/261|
|F01|E01|scale01_anchor_minus_identity|3|+0.000 [+0.000,+0.000]|0/0|13200/261|
|F01|E01|scale01_anchor_minus_identity|5|+0.000 [+0.000,+0.000]|0/0|13200/261|
|F01|E01|scale01_anchor_minus_identity|10|+0.000 [+0.000,+0.000]|0/0|1200/261|
|F01|E01|scale10_anchor_minus_identity|1|+0.015 [+0.000,+0.059]|2/0|13200/261|
|F01|E01|scale10_anchor_minus_identity|3|+0.000 [+0.000,+0.000]|0/0|13200/261|
|F01|E01|scale10_anchor_minus_identity|5|+0.000 [+0.000,+0.000]|0/0|13200/261|
|F01|E01|scale10_anchor_minus_identity|10|+0.000 [+0.000,+0.000]|0/0|1200/261|
|F01|E01|rowcenter_source_rms_anchor_minus_identity|1|+0.015 [+0.000,+0.059]|2/0|13200/261|
|F01|E01|rowcenter_source_rms_anchor_minus_identity|3|+0.000 [+0.000,+0.000]|0/0|13200/261|
|F01|E01|rowcenter_source_rms_anchor_minus_identity|5|+0.000 [+0.000,+0.000]|0/0|13200/261|
|F01|E01|rowcenter_source_rms_anchor_minus_identity|10|+0.000 [+0.000,+0.000]|0/0|1200/261|
|F01|E01|shuffled_anchor_lambda1_minus_identity|1|-35.167 [-36.364,-33.270]|78/4720|13200/261|
|F01|E01|shuffled_anchor_lambda1_minus_identity|3|-43.939 [-45.455,-41.529]|100/5900|13200/261|
|F01|E01|shuffled_anchor_lambda1_minus_identity|5|-35.152 [-36.364,-33.223]|80/4720|13200/261|
|F01|E01|shuffled_anchor_lambda1_minus_identity|10|-46.879 [-48.610,-44.293]|106/6294|13200/261|
|F01|E03|raw_anchor_lambda1_minus_identity|1|-8.646 [-9.960,-7.152]|220/1076|9900/300|
|F01|E03|raw_anchor_lambda1_minus_identity|3|-4.727 [-6.172,-3.242]|187/655|9900/300|
|F01|E03|raw_anchor_lambda1_minus_identity|5|-2.859 [-4.041,-1.585]|168/451|9900/300|
|F01|E03|raw_anchor_lambda1_minus_identity|10|-1.222 [-2.556,+0.000]|9/20|900/300|
|F01|E03|raw_bias_only_minus_identity|1|-2.475 [-3.566,-1.252]|151/396|9900/300|
|F01|E03|raw_bias_only_minus_identity|3|-1.525 [-2.788,-0.181]|152/303|9900/300|
|F01|E03|raw_bias_only_minus_identity|5|-1.172 [-2.374,+0.172]|149/265|9900/300|
|F01|E03|raw_bias_only_minus_identity|10|-0.222 [-1.556,+1.111]|14/16|900/300|
|F01|E03|raw_weak_lambda1000_blend005_minus_identity|1|+0.040 [+0.000,+0.122]|4/0|9900/300|
|F01|E03|raw_weak_lambda1000_blend005_minus_identity|3|+0.030 [+0.000,+0.092]|3/0|9900/300|
|F01|E03|raw_weak_lambda1000_blend005_minus_identity|5|+0.040 [+0.000,+0.121]|4/0|9900/300|
|F01|E03|raw_weak_lambda1000_blend005_minus_identity|10|+0.111 [+0.000,+0.333]|1/0|900/300|
|F01|E03|diagonal_anchor_lambda1_minus_identity|1|-8.545 [-10.385,-6.666]|275/1121|9900/300|
|F01|E03|diagonal_anchor_lambda1_minus_identity|3|-3.273 [-4.769,-1.879]|134/458|9900/300|
|F01|E03|diagonal_anchor_lambda1_minus_identity|5|-2.626 [-3.930,-1.424]|98/358|9900/300|
|F01|E03|diagonal_anchor_lambda1_minus_identity|10|-2.556 [-3.778,-1.333]|4/27|900/300|
|F01|E03|scale01_anchor_minus_identity|1|-5.101 [-6.616,-3.505]|196/701|9900/300|
|F01|E03|scale01_anchor_minus_identity|3|-4.879 [-6.788,-3.020]|192/675|9900/300|
|F01|E03|scale01_anchor_minus_identity|5|-3.899 [-5.698,-2.111]|193/579|9900/300|
|F01|E03|scale01_anchor_minus_identity|10|-2.444 [-4.111,-1.000]|10/32|900/300|
|F01|E03|scale10_anchor_minus_identity|1|-21.596 [-23.849,-19.171]|311/2449|9900/300|
|F01|E03|scale10_anchor_minus_identity|3|-7.051 [-8.758,-5.313]|255/953|9900/300|
|F01|E03|scale10_anchor_minus_identity|5|-4.030 [-5.435,-2.585]|205/604|9900/300|
|F01|E03|scale10_anchor_minus_identity|10|-1.556 [-2.889,-0.222]|9/23|900/300|
|F01|E03|rowcenter_source_rms_anchor_minus_identity|1|-5.283 [-6.313,-4.080]|179/702|9900/300|
|F01|E03|rowcenter_source_rms_anchor_minus_identity|3|-3.263 [-4.404,-2.030]|148/471|9900/300|
|F01|E03|rowcenter_source_rms_anchor_minus_identity|5|-2.162 [-3.202,-1.121]|139/353|9900/300|
|F01|E03|rowcenter_source_rms_anchor_minus_identity|10|-0.778 [-2.000,+0.333]|9/16|900/300|
|F01|E03|shuffled_anchor_lambda1_minus_identity|1|-53.939 [-57.587,-50.433]|301/5641|9900/300|
|F01|E03|shuffled_anchor_lambda1_minus_identity|3|-48.717 [-52.162,-45.191]|341/5164|9900/300|
|F01|E03|shuffled_anchor_lambda1_minus_identity|5|-57.475 [-60.850,-53.816]|321/6011|9900/300|
|F01|E03|shuffled_anchor_lambda1_minus_identity|10|-55.293 [-58.869,-51.808]|353/5827|9900/300|
|F02|E01|scaler_frozen_gamma_fixed_minus_source_baseline|1|+0.000 [+0.000,+0.000]|0/0|1200/261|
|F02|E01|scaler_frozen_gamma_scale_minus_source_baseline|1|+0.000 [+0.000,+0.000]|0/0|1200/261|
|F02|E01|scaler_refit_gamma_fixed_minus_source_baseline|1|+0.000 [+0.000,+0.000]|0/0|1200/261|
|F02|E01|scaler_refit_gamma_scale_minus_source_baseline|1|+0.000 [+0.000,+0.000]|0/0|1200/261|
|F02|E01|scaler_frozen_gamma_fixed_minus_source_baseline|3|+0.000 [+0.000,+0.000]|0/0|1200/261|
|F02|E01|scaler_frozen_gamma_scale_minus_source_baseline|3|+0.000 [+0.000,+0.000]|0/0|1200/261|
|F02|E01|scaler_refit_gamma_fixed_minus_source_baseline|3|+0.000 [+0.000,+0.000]|0/0|1200/261|
|F02|E01|scaler_refit_gamma_scale_minus_source_baseline|3|+0.000 [+0.000,+0.000]|0/0|1200/261|
|F02|E01|scaler_frozen_gamma_fixed_minus_source_baseline|5|+0.000 [+0.000,+0.000]|0/0|1200/261|
|F02|E01|scaler_frozen_gamma_scale_minus_source_baseline|5|+0.000 [+0.000,+0.000]|0/0|1200/261|
|F02|E01|scaler_refit_gamma_fixed_minus_source_baseline|5|+0.000 [+0.000,+0.000]|0/0|1200/261|
|F02|E01|scaler_refit_gamma_scale_minus_source_baseline|5|+0.000 [+0.000,+0.000]|0/0|1200/261|
|F02|E01|scaler_frozen_gamma_fixed_minus_source_baseline|10|+0.000 [+0.000,+0.000]|0/0|1200/261|
|F02|E01|scaler_frozen_gamma_scale_minus_source_baseline|10|+0.000 [+0.000,+0.000]|0/0|1200/261|
|F02|E01|scaler_refit_gamma_fixed_minus_source_baseline|10|+0.000 [+0.000,+0.000]|0/0|1200/261|
|F02|E01|scaler_refit_gamma_scale_minus_source_baseline|10|+0.000 [+0.000,+0.000]|0/0|1200/261|
|F02|E03|scaler_frozen_gamma_fixed_minus_source_baseline|1|+0.111 [-0.778,+1.000]|8/7|900/300|
|F02|E03|scaler_frozen_gamma_scale_minus_source_baseline|1|+0.111 [-0.778,+1.000]|8/7|900/300|
|F02|E03|scaler_refit_gamma_fixed_minus_source_baseline|1|+0.000 [-0.889,+0.889]|8/8|900/300|
|F02|E03|scaler_refit_gamma_scale_minus_source_baseline|1|+0.000 [-0.889,+0.889]|8/8|900/300|
|F02|E03|scaler_frozen_gamma_fixed_minus_source_baseline|3|+0.222 [-1.000,+1.333]|16/14|900/300|
|F02|E03|scaler_frozen_gamma_scale_minus_source_baseline|3|+0.222 [-1.000,+1.333]|16/14|900/300|
|F02|E03|scaler_refit_gamma_fixed_minus_source_baseline|3|+0.667 [-0.556,+1.781]|20/14|900/300|
|F02|E03|scaler_refit_gamma_scale_minus_source_baseline|3|+0.667 [-0.556,+1.781]|20/14|900/300|
|F02|E03|scaler_frozen_gamma_fixed_minus_source_baseline|5|+2.111 [+0.889,+3.444]|26/7|900/300|
|F02|E03|scaler_frozen_gamma_scale_minus_source_baseline|5|+1.889 [+0.667,+3.222]|25/8|900/300|
|F02|E03|scaler_refit_gamma_fixed_minus_source_baseline|5|+2.111 [+0.889,+3.556]|28/9|900/300|
|F02|E03|scaler_refit_gamma_scale_minus_source_baseline|5|+2.111 [+0.889,+3.556]|28/9|900/300|
|F02|E03|scaler_frozen_gamma_fixed_minus_source_baseline|10|+3.778 [+2.111,+5.669]|48/14|900/300|
|F02|E03|scaler_frozen_gamma_scale_minus_source_baseline|10|+3.778 [+2.111,+5.778]|48/14|900/300|
|F02|E03|scaler_refit_gamma_fixed_minus_source_baseline|10|+3.889 [+2.219,+6.000]|49/14|900/300|
|F02|E03|scaler_refit_gamma_scale_minus_source_baseline|10|+3.889 [+2.219,+6.000]|49/14|900/300|
|F03|E01|unweighted_minus_balanced ratio=A coverage=0.0|0|+0.000 [+0.000,+0.000]|0/0|1000/335|
|F03|E01|unweighted_minus_balanced ratio=A coverage=1.0|0|+0.000 [+0.000,+0.000]|0/0|1000/335|
|F03|E01|unweighted_minus_balanced ratio=B coverage=0.0|0|+0.000 [+0.000,+0.000]|0/0|1000/335|
|F03|E01|unweighted_minus_balanced ratio=B coverage=1.0|0|+0.000 [+0.000,+0.000]|0/0|1000/335|
|F03|E01|unweighted_minus_balanced ratio=C coverage=0.0|0|+0.000 [+0.000,+0.000]|0/0|1000/335|
|F03|E01|unweighted_minus_balanced ratio=C coverage=1.0|0|+0.000 [+0.000,+0.000]|0/0|1000/335|
|F04|E03|source_n050_minus_source190|0|-11.222 [-14.222,-8.333]|37/138|900/300|
|F04|E03|source_n100_minus_source190|0|-5.556 [-7.778,-3.333]|24/74|900/300|
|F04|E03|source_n150_minus_source190|0|-1.444 [-2.889,+0.111]|19/32|900/300|
|F04|E03|source_n190_minus_source190|0|+0.000 [+0.000,+0.000]|0/0|900/300|
|F04|E03|source_n240_minus_source190|0|+3.889 [+2.219,+6.000]|49/14|900/300|

## F03：权重、来源覆盖与比例

五个目标酸枣仁来源×种子42—49×三种比例×两种覆盖，每个条件固定240个Source与20+20 Query，比对balanced与无类别权重。42—44中已有balanced源模型直接复用；45—49是预先指定的新划分重复，并非新样品。原生SVC预测用于因子效应，score argmax接口诊断单列。

分别报告旧种子42—44、新种子45—49以及全部42—49的对照，并保留每来源结果。覆盖样品始终等量替换其他酸枣仁来源；不能称跨批次或任意未知来源泛化。

## F04：标签预算与学习曲线

原15个五来源外折中固定Query，在原Source每来源38份的锁定顺序中取10/20/30份，再比较原190份以及Source190＋全部Support50＝240份。190与240端点分别复用上一轮源模型和K10完整重训。所有预算平衡且按每类嵌套抽样；固定同一SVM参数，不依据Query选训练量。

学习曲线各折是否单调和每步纠错/误伤全部保存；预算变化同时改变标签数量与训练统计，不据此主张独立因果机制。

## 局限与未运行项目

本轮为依据上一轮结果预先冻结的探索性补充。仅使用master TRAIN；样品已有历史曝光。支持重排和新增划分种子不增加独立实物数。描述性95%区间按physical_sample_id聚类，不覆盖来源总体、批次总体、协议选择或长期开发偏倚。来源和批次混杂，不能据此宣称真实跨批次泛化或确认性因果机制。

二分类与五来源的样品组成、类别数和标签预算不同，准确率不得直接相减解释为类别数的单因素作用。新增实验依据已曝光结果提出，本轮不是独立确认集。所有正、负、零效应均保留，不因成绩低而更换Query、种子、方法或指标。

真实新批次评价继续为**NOT_RUN_NEEDS_NEW_BATCH_DATA**。上一轮已交付元数据模板、导入验证器及五来源优先450份新样品的采集起点；目前未采集，不生成真实batch_id或补造评价。

## 复现

```bash
python scripts/run_b_mechanism_followup_20261110.py --resume
python scripts/summarize_b_mechanism_followup_20261110.py --resume
python scripts/summarize_b_mechanism_followup_20261110.py --verify
```

summary_verification.json记录独立混淆计数复算、任务与工件哈希核验。训练中途的PARTIAL报告只反映已封存子集。
