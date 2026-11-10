# B方案第二轮：校准退化、重训分解与来源覆盖稳定性

状态：**PARTIAL**，已封存4/345个任务包。
本报告只读取完整封存预测；不拟合模型、不修改原105任务的代码、数据、标签或结果。

新增完整分类器拟合12次；已封存科研任务CPU 2.303835秒。重复使用的旧模型端点不计为新训练；控制、核验、失败尝试及推送成本另见后台账本。
逐样品预测行数12500，涉及唯一Query实物95；两者不得互换。

## 状态及设计

|实验块|内容|完成/计划|状态|
|---|---|---:|---|
|F01|分数几何、支持重排及乱标诊断|1/45|PARTIAL|
|F02|Scaler × gamma 重训分解|1/45|PARTIAL|
|F03|类别权重 × 来源覆盖 × 类别比例|1/240|PARTIAL|
|F04|五来源固定嵌套学习曲线|1/15|PARTIAL|

当前有效数据仍为18份工作簿、1672份用户声明独立实物。本轮仅在master TRAIN构造角色，不使用原CAL/PRED搜索模型。ZY01/ZY02为枣仁、ZY03—ZY07为酸枣仁；原来源/批次混杂不变。

## F01：支持抽样和分数几何

复用15个五来源以及预先指定30个类别平衡二分类模型。原始Support顺序作为draw0，再固定101—110十个重排种子；每类K嵌套取1/3/5/10。真标签K10完整池仅保留一次，重排不是新的实物或新真标签端点。乱标仅作用于显式诊断副本，保持样品ID与类别计数。

比较原始Anchor、Bias、弱Anchor、对角Anchor、0.1/10倍分数和逐样品行中心化＋Source全局RMS归一化。正比例与行中心化保留原始K0类别边界；归一化统计量只来自Source。完整Anchor包含跨类耦合，对角方法是降低自由度的明确对照。分数不是概率。

## 全部冻结方法的描述性结果

下表按每任务、每个合法draw的端点等权汇总；不从这些结果选择最优配置作为确认性结论。

|块|原任务|方法|K/类|平均Accuracy|平均Macro-F1|平均Balanced Accuracy|端点数|
|---|---|---|---:|---:|---:|---:|---:|
|F01|E01|diagonal_anchor_lambda1|1|1.000000|1.000000|1.000000|11|
|F01|E01|diagonal_anchor_lambda1|3|1.000000|1.000000|1.000000|11|
|F01|E01|diagonal_anchor_lambda1|5|1.000000|1.000000|1.000000|11|
|F01|E01|diagonal_anchor_lambda1|10|1.000000|1.000000|1.000000|1|
|F01|E01|identity|0|1.000000|1.000000|1.000000|1|
|F01|E01|raw_anchor_lambda1|1|1.000000|1.000000|1.000000|11|
|F01|E01|raw_anchor_lambda1|3|1.000000|1.000000|1.000000|11|
|F01|E01|raw_anchor_lambda1|5|1.000000|1.000000|1.000000|11|
|F01|E01|raw_anchor_lambda1|10|1.000000|1.000000|1.000000|1|
|F01|E01|raw_bias_only|1|1.000000|1.000000|1.000000|11|
|F01|E01|raw_bias_only|3|1.000000|1.000000|1.000000|11|
|F01|E01|raw_bias_only|5|1.000000|1.000000|1.000000|11|
|F01|E01|raw_bias_only|10|1.000000|1.000000|1.000000|1|
|F01|E01|raw_weak_lambda1000_blend005|1|1.000000|1.000000|1.000000|11|
|F01|E01|raw_weak_lambda1000_blend005|3|1.000000|1.000000|1.000000|11|
|F01|E01|raw_weak_lambda1000_blend005|5|1.000000|1.000000|1.000000|11|
|F01|E01|raw_weak_lambda1000_blend005|10|1.000000|1.000000|1.000000|1|
|F01|E01|rowcenter_source_rms_anchor|1|1.000000|1.000000|1.000000|11|
|F01|E01|rowcenter_source_rms_anchor|3|1.000000|1.000000|1.000000|11|
|F01|E01|rowcenter_source_rms_anchor|5|1.000000|1.000000|1.000000|11|
|F01|E01|rowcenter_source_rms_anchor|10|1.000000|1.000000|1.000000|1|
|F01|E01|scale01_anchor|1|1.000000|1.000000|1.000000|11|
|F01|E01|scale01_anchor|3|1.000000|1.000000|1.000000|11|
|F01|E01|scale01_anchor|5|1.000000|1.000000|1.000000|11|
|F01|E01|scale01_anchor|10|1.000000|1.000000|1.000000|1|
|F01|E01|scale10_anchor|1|1.000000|1.000000|1.000000|11|
|F01|E01|scale10_anchor|3|1.000000|1.000000|1.000000|11|
|F01|E01|scale10_anchor|5|1.000000|1.000000|1.000000|11|
|F01|E01|scale10_anchor|10|1.000000|1.000000|1.000000|1|
|F01|E01|shuffled_anchor_lambda1|1|0.636364|0.636364|0.636364|11|
|F01|E01|shuffled_anchor_lambda1|3|0.545455|0.545455|0.545455|11|
|F01|E01|shuffled_anchor_lambda1|5|0.636364|0.636364|0.636364|11|
|F01|E01|shuffled_anchor_lambda1|10|0.545455|0.545455|0.545455|11|
|F01|E01|svc_predict|0|1.000000|1.000000|1.000000|1|
|F02|E01|identity|0|1.000000|1.000000|1.000000|1|
|F02|E01|scaler_frozen_gamma_fixed|1|1.000000|1.000000|1.000000|1|
|F02|E01|scaler_frozen_gamma_fixed|3|1.000000|1.000000|1.000000|1|
|F02|E01|scaler_frozen_gamma_fixed|5|1.000000|1.000000|1.000000|1|
|F02|E01|scaler_frozen_gamma_fixed|10|1.000000|1.000000|1.000000|1|
|F02|E01|scaler_frozen_gamma_scale|1|1.000000|1.000000|1.000000|1|
|F02|E01|scaler_frozen_gamma_scale|3|1.000000|1.000000|1.000000|1|
|F02|E01|scaler_frozen_gamma_scale|5|1.000000|1.000000|1.000000|1|
|F02|E01|scaler_frozen_gamma_scale|10|1.000000|1.000000|1.000000|1|
|F02|E01|scaler_refit_gamma_fixed|1|1.000000|1.000000|1.000000|1|
|F02|E01|scaler_refit_gamma_fixed|3|1.000000|1.000000|1.000000|1|
|F02|E01|scaler_refit_gamma_fixed|5|1.000000|1.000000|1.000000|1|
|F02|E01|scaler_refit_gamma_fixed|10|1.000000|1.000000|1.000000|1|
|F02|E01|scaler_refit_gamma_scale|1|1.000000|1.000000|1.000000|1|
|F02|E01|scaler_refit_gamma_scale|3|1.000000|1.000000|1.000000|1|
|F02|E01|scaler_refit_gamma_scale|5|1.000000|1.000000|1.000000|1|
|F02|E01|scaler_refit_gamma_scale|10|1.000000|1.000000|1.000000|1|
|F03|E01|class_weight_balanced|0|1.000000|1.000000|1.000000|1|
|F03|E01|class_weight_balanced_argmax|0|1.000000|1.000000|1.000000|1|
|F03|E01|class_weight_none|0|1.000000|1.000000|1.000000|1|
|F03|E01|class_weight_none_argmax|0|1.000000|1.000000|1.000000|1|
|F04|E03|source_n050|0|0.683333|0.675348|0.683333|1|
|F04|E03|source_n100|0|0.700000|0.693658|0.700000|1|
|F04|E03|source_n150|0|0.766667|0.762158|0.766667|1|
|F04|E03|source_n190|0|0.766667|0.762600|0.766667|1|
|F04|E03|source_n240|0|0.783333|0.779794|0.783333|1|

## F02：重训收益的Scaler与gamma分解

保持Source、Support、Query和支持标签预算完全相同，分别冻结/重估StandardScaler、冻结原源模型数值gamma/按新增训练侧重新计算gamma。C固定10，类别权重balanced。原Scaler重估＋gamma重算的端点与上一轮完整重训完全相同，优先复用，不重复训练。

即使冻结Scaler，gamma='scale'也会随训练集合改变，因此不能把两者混为同一因素。逐任务保存有效gamma与Scaler哈希，四个条件的全部配对差值见paired_transitions.csv。对于重新逐列StandardScaler后的完整612点输入，scale gamma通常等于1/612；严格相同的数值与等价目标端点直接复用，不构成独立干预。审计中的ULP级差异也不能解释为生物学或稳健的核宽度机制。

## 配对效应及描述性不确定性

以下先列每块全体Query对齐的主要对照；每来源、逐类Recall代价、全部方法差值与区间保存在CSV。

|块|原任务|比较|K/类|ΔAccuracy百分点及95%区间|纠错/误伤|重复预测/实物|
|---|---|---|---:|---:|---:|---:|
|F01|E01|raw_anchor_lambda1_minus_identity|1|+0.000 [+0.000,+0.000]|0/0|440/40|
|F01|E01|raw_anchor_lambda1_minus_identity|3|+0.000 [+0.000,+0.000]|0/0|440/40|
|F01|E01|raw_anchor_lambda1_minus_identity|5|+0.000 [+0.000,+0.000]|0/0|440/40|
|F01|E01|raw_anchor_lambda1_minus_identity|10|+0.000 [+0.000,+0.000]|0/0|40/40|
|F01|E01|raw_bias_only_minus_identity|1|+0.000 [+0.000,+0.000]|0/0|440/40|
|F01|E01|raw_bias_only_minus_identity|3|+0.000 [+0.000,+0.000]|0/0|440/40|
|F01|E01|raw_bias_only_minus_identity|5|+0.000 [+0.000,+0.000]|0/0|440/40|
|F01|E01|raw_bias_only_minus_identity|10|+0.000 [+0.000,+0.000]|0/0|40/40|
|F01|E01|raw_weak_lambda1000_blend005_minus_identity|1|+0.000 [+0.000,+0.000]|0/0|440/40|
|F01|E01|raw_weak_lambda1000_blend005_minus_identity|3|+0.000 [+0.000,+0.000]|0/0|440/40|
|F01|E01|raw_weak_lambda1000_blend005_minus_identity|5|+0.000 [+0.000,+0.000]|0/0|440/40|
|F01|E01|raw_weak_lambda1000_blend005_minus_identity|10|+0.000 [+0.000,+0.000]|0/0|40/40|
|F01|E01|diagonal_anchor_lambda1_minus_identity|1|+0.000 [+0.000,+0.000]|0/0|440/40|
|F01|E01|diagonal_anchor_lambda1_minus_identity|3|+0.000 [+0.000,+0.000]|0/0|440/40|
|F01|E01|diagonal_anchor_lambda1_minus_identity|5|+0.000 [+0.000,+0.000]|0/0|440/40|
|F01|E01|diagonal_anchor_lambda1_minus_identity|10|+0.000 [+0.000,+0.000]|0/0|40/40|
|F01|E01|scale01_anchor_minus_identity|1|+0.000 [+0.000,+0.000]|0/0|440/40|
|F01|E01|scale01_anchor_minus_identity|3|+0.000 [+0.000,+0.000]|0/0|440/40|
|F01|E01|scale01_anchor_minus_identity|5|+0.000 [+0.000,+0.000]|0/0|440/40|
|F01|E01|scale01_anchor_minus_identity|10|+0.000 [+0.000,+0.000]|0/0|40/40|
|F01|E01|scale10_anchor_minus_identity|1|+0.000 [+0.000,+0.000]|0/0|440/40|
|F01|E01|scale10_anchor_minus_identity|3|+0.000 [+0.000,+0.000]|0/0|440/40|
|F01|E01|scale10_anchor_minus_identity|5|+0.000 [+0.000,+0.000]|0/0|440/40|
|F01|E01|scale10_anchor_minus_identity|10|+0.000 [+0.000,+0.000]|0/0|40/40|
|F01|E01|rowcenter_source_rms_anchor_minus_identity|1|+0.000 [+0.000,+0.000]|0/0|440/40|
|F01|E01|rowcenter_source_rms_anchor_minus_identity|3|+0.000 [+0.000,+0.000]|0/0|440/40|
|F01|E01|rowcenter_source_rms_anchor_minus_identity|5|+0.000 [+0.000,+0.000]|0/0|440/40|
|F01|E01|rowcenter_source_rms_anchor_minus_identity|10|+0.000 [+0.000,+0.000]|0/0|40/40|
|F01|E01|shuffled_anchor_lambda1_minus_identity|1|-36.364 [-36.364,-36.364]|0/160|440/40|
|F01|E01|shuffled_anchor_lambda1_minus_identity|3|-45.455 [-45.455,-45.455]|0/200|440/40|
|F01|E01|shuffled_anchor_lambda1_minus_identity|5|-36.364 [-36.364,-36.364]|0/160|440/40|
|F01|E01|shuffled_anchor_lambda1_minus_identity|10|-45.455 [-45.455,-45.455]|0/200|440/40|
|F02|E01|scaler_frozen_gamma_fixed_minus_source_baseline|1|+0.000 [+0.000,+0.000]|0/0|40/40|
|F02|E01|scaler_frozen_gamma_scale_minus_source_baseline|1|+0.000 [+0.000,+0.000]|0/0|40/40|
|F02|E01|scaler_refit_gamma_fixed_minus_source_baseline|1|+0.000 [+0.000,+0.000]|0/0|40/40|
|F02|E01|scaler_refit_gamma_scale_minus_source_baseline|1|+0.000 [+0.000,+0.000]|0/0|40/40|
|F02|E01|scaler_frozen_gamma_fixed_minus_source_baseline|3|+0.000 [+0.000,+0.000]|0/0|40/40|
|F02|E01|scaler_frozen_gamma_scale_minus_source_baseline|3|+0.000 [+0.000,+0.000]|0/0|40/40|
|F02|E01|scaler_refit_gamma_fixed_minus_source_baseline|3|+0.000 [+0.000,+0.000]|0/0|40/40|
|F02|E01|scaler_refit_gamma_scale_minus_source_baseline|3|+0.000 [+0.000,+0.000]|0/0|40/40|
|F02|E01|scaler_frozen_gamma_fixed_minus_source_baseline|5|+0.000 [+0.000,+0.000]|0/0|40/40|
|F02|E01|scaler_frozen_gamma_scale_minus_source_baseline|5|+0.000 [+0.000,+0.000]|0/0|40/40|
|F02|E01|scaler_refit_gamma_fixed_minus_source_baseline|5|+0.000 [+0.000,+0.000]|0/0|40/40|
|F02|E01|scaler_refit_gamma_scale_minus_source_baseline|5|+0.000 [+0.000,+0.000]|0/0|40/40|
|F02|E01|scaler_frozen_gamma_fixed_minus_source_baseline|10|+0.000 [+0.000,+0.000]|0/0|40/40|
|F02|E01|scaler_frozen_gamma_scale_minus_source_baseline|10|+0.000 [+0.000,+0.000]|0/0|40/40|
|F02|E01|scaler_refit_gamma_fixed_minus_source_baseline|10|+0.000 [+0.000,+0.000]|0/0|40/40|
|F02|E01|scaler_refit_gamma_scale_minus_source_baseline|10|+0.000 [+0.000,+0.000]|0/0|40/40|
|F04|E03|source_n050_minus_source190|0|-8.333 [-20.000,+3.333]|4/9|60/60|
|F04|E03|source_n100_minus_source190|0|-6.667 [-15.000,+0.000]|1/5|60/60|
|F04|E03|source_n150_minus_source190|0|+0.000 [-5.000,+5.000]|1/1|60/60|
|F04|E03|source_n190_minus_source190|0|+0.000 [+0.000,+0.000]|0/0|60/60|
|F04|E03|source_n240_minus_source190|0|+1.667 [-3.333,+8.333]|2/1|60/60|

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
