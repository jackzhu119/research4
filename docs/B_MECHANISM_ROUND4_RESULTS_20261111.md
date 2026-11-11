# B方案第四轮：监督损失质量、C约束与Support构成

状态：**PARTIAL**；已封存 **4/375** 个任务包。
本报告只读取封存工件，不拟合或选择模型，不修改前三轮数据、标签、模型、协议和结果。

新增分类器拟合12次、校准拟合调用48次；科研任务CPU合计1.664843秒。
保存5560条相关预测，涉及40个唯一Query实物；控制、核验和推送成本另见账本。

## 完成范围

|块|问题|完成/计划|状态|
|---|---|---:|---|
|H01|监督损失质量与新增样品分解|1/45|PARTIAL|
|H02|C × 类别权重的约束激活|1/240|PARTIAL|
|H03|弱校准分数移动与margin激活审计|1/45|PARTIAL|
|H04|固定Support总数的类别构成对照|1/45|PARTIAL|

## 机制定义与配对限制

H01所有条件冻结原Source的Scaler和实际数值gamma。源模型C10为基线；仅提高Source的C到10×N_total/N_source，不加入Support谱或标签；新增Source+Support则比较固定C10和C=10×N_source/N_total。后者固定C×有标签样品数的总损失系数，旧固定C端点直接复用。libSVM实际训练one-vs-one子问题，即使输出OVR分数；本块Source各类均衡且每类添加相同K，因此全局N_total/N_source等于每个OVO类别对的样品数比例，损失质量对照在这个受限设计下成立。本轮描述固定损失质量下的增量监督收益，不能把新增谱值、标签、训练点和损失项的联合变化唯一归因于抽象标签信息。

H02固定Source、Scaler、gamma、Query，预设C=0.1/1/10交叉balanced/None，保存原生SVC与argmax接口。C10复用；Source类别均衡时两种权重的有效训练目标相同，共享同一模型。C降低属于约束激活的机制干预，不是搜索最佳超参数；所有三个C均报告，Source训练内约束诊断不用于模型选择。

H03完全不拟合：复用第二轮34个弱校准回合及第三轮16个正污染回合，排除已重复的零污染K10。每个保存点核验旧top2分数间隔g与实际修正无穷范数d，g>2d足以保持argmax；同时记录分数是否真的移动、类别是否改变。沿固定S(t)=S_source+t×Delta解析计算首次argmax边界，实际t=0.05，不做新blend网格，不依Query真值挑选t。首次边界≤0.05包含恰好相等的边界，argmax采用首索引平局规则，所以它与实际类别改变数分别保存，不能等同。t>1仅是数学外推，不作为适配效用；未受害不等于有收益，分数移动但不跨margin也不等于完全没有计算。

H04二分类Support总数固定10，类别构成为1:9/3:7/5:5/7:3/9:1；五分类总数25，比较5:5:5:5:5及各类轮流稀缺1份、其他类各6份。draw0/101/102在同一候选池取固定前缀；同一ID比较Anchor/Bias及普通/类别均衡残差拟合。类别权重N/(类别数×该类支持数)的总和为N，固定lambda1；均衡K5直接复用，权重等价端点共享结果。composition方法k=-1表示每类K不适用，不表示零支持，实际总量及类别数见support_counts_json。改变构成同时改变实际Support ID和各类覆盖量，不能称纯先验的单因素因果干预；同一构成内加权与普通拟合使用相同ID，是更直接的权重作用对照。

所有Query和Source均来自master TRAIN。Support候选池原本按已知标签分层，构成对照是采样设计诊断，不是已知真实部署先验；残差分数不称后验概率。类别构成改变同时影响每类信息量和有效权重ESS，不能据此宣称跨批次先验修正。

## 全部冻结条件的结果

|块|原任务|方法|K/类|Support构成|Accuracy|Macro-F1|Balanced Accuracy|端点数|
|---|---|---|---:|---|---:|---:|---:|---:|
|H01|E01|identity|0||1.000000|1.000000|1.000000|1|
|H01|E01|source_mass_only|1||1.000000|1.000000|1.000000|1|
|H01|E01|source_mass_only|3||1.000000|1.000000|1.000000|1|
|H01|E01|source_mass_only|5||1.000000|1.000000|1.000000|1|
|H01|E01|source_mass_only|10||1.000000|1.000000|1.000000|1|
|H01|E01|source_support_fixed_C|1||1.000000|1.000000|1.000000|1|
|H01|E01|source_support_fixed_C|3||1.000000|1.000000|1.000000|1|
|H01|E01|source_support_fixed_C|5||1.000000|1.000000|1.000000|1|
|H01|E01|source_support_fixed_C|10||1.000000|1.000000|1.000000|1|
|H01|E01|source_support_normalized_mass|1||1.000000|1.000000|1.000000|1|
|H01|E01|source_support_normalized_mass|3||1.000000|1.000000|1.000000|1|
|H01|E01|source_support_normalized_mass|5||1.000000|1.000000|1.000000|1|
|H01|E01|source_support_normalized_mass|10||1.000000|1.000000|1.000000|1|
|H02|E01|C_0p1_balanced|0||1.000000|1.000000|1.000000|1|
|H02|E01|C_0p1_balanced_argmax|0||1.000000|1.000000|1.000000|1|
|H02|E01|C_0p1_none|0||1.000000|1.000000|1.000000|1|
|H02|E01|C_0p1_none_argmax|0||1.000000|1.000000|1.000000|1|
|H02|E01|C_10_balanced|0||1.000000|1.000000|1.000000|1|
|H02|E01|C_10_balanced_argmax|0||1.000000|1.000000|1.000000|1|
|H02|E01|C_10_none|0||1.000000|1.000000|1.000000|1|
|H02|E01|C_10_none_argmax|0||1.000000|1.000000|1.000000|1|
|H02|E01|C_1_balanced|0||1.000000|1.000000|1.000000|1|
|H02|E01|C_1_balanced_argmax|0||1.000000|1.000000|1.000000|1|
|H02|E01|C_1_none|0||1.000000|1.000000|1.000000|1|
|H02|E01|C_1_none_argmax|0||1.000000|1.000000|1.000000|1|
|H03|E01|f01_weak|1||1.000000|1.000000|1.000000|11|
|H03|E01|f01_weak|3||1.000000|1.000000|1.000000|11|
|H03|E01|f01_weak|5||1.000000|1.000000|1.000000|11|
|H03|E01|f01_weak|10||1.000000|1.000000|1.000000|1|
|H03|E01|g03_weak|10||1.000000|1.000000|1.000000|16|
|H03|E01|identity|0||1.000000|1.000000|1.000000|1|
|H03|E01|svc_predict|0||1.000000|1.000000|1.000000|1|
|H04|E01|anchor_class_balanced|不适用|balanced|1.000000|1.000000|1.000000|3|
|H04|E01|anchor_class_balanced|不适用|binary_1_9|1.000000|1.000000|1.000000|3|
|H04|E01|anchor_class_balanced|不适用|binary_3_7|1.000000|1.000000|1.000000|3|
|H04|E01|anchor_class_balanced|不适用|binary_7_3|1.000000|1.000000|1.000000|3|
|H04|E01|anchor_class_balanced|不适用|binary_9_1|1.000000|1.000000|1.000000|3|
|H04|E01|anchor_unweighted|不适用|balanced|1.000000|1.000000|1.000000|3|
|H04|E01|anchor_unweighted|不适用|binary_1_9|1.000000|1.000000|1.000000|3|
|H04|E01|anchor_unweighted|不适用|binary_3_7|1.000000|1.000000|1.000000|3|
|H04|E01|anchor_unweighted|不适用|binary_7_3|1.000000|1.000000|1.000000|3|
|H04|E01|anchor_unweighted|不适用|binary_9_1|1.000000|1.000000|1.000000|3|
|H04|E01|bias_class_balanced|不适用|balanced|1.000000|1.000000|1.000000|3|
|H04|E01|bias_class_balanced|不适用|binary_1_9|1.000000|1.000000|1.000000|3|
|H04|E01|bias_class_balanced|不适用|binary_3_7|1.000000|1.000000|1.000000|3|
|H04|E01|bias_class_balanced|不适用|binary_7_3|1.000000|1.000000|1.000000|3|
|H04|E01|bias_class_balanced|不适用|binary_9_1|1.000000|1.000000|1.000000|3|
|H04|E01|bias_unweighted|不适用|balanced|1.000000|1.000000|1.000000|3|
|H04|E01|bias_unweighted|不适用|binary_1_9|1.000000|1.000000|1.000000|3|
|H04|E01|bias_unweighted|不适用|binary_3_7|1.000000|1.000000|1.000000|3|
|H04|E01|bias_unweighted|不适用|binary_7_3|1.000000|1.000000|1.000000|3|
|H04|E01|bias_unweighted|不适用|binary_9_1|1.000000|1.000000|1.000000|3|
|H04|E01|identity|0||1.000000|1.000000|1.000000|1|
|H04|E01|svc_predict|0||1.000000|1.000000|1.000000|1|

## 配对效应与描述性95%区间

保留全部正、负和零效应。以下区间以physical_sample_id聚类；每来源、每类Recall代价、纠错和误伤详见CSV。

|块|原任务|比较|K/类|C|比例/覆盖/构成/来源范围|ΔAccuracy百分点及95%区间|纠错/误伤|预测行/实物|
|---|---|---|---:|---:|---|---:|---:|---:|
|H01|E01|source_mass_only_minus_source_C10|1||scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H01|E01|source_support_fixed_C_minus_source_C10|1||scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H01|E01|source_support_normalized_mass_minus_source_C10|1||scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H01|E01|additional_samples_minus_loss_mass_only|1||scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H01|E01|normalized_mass_minus_fixed_C|1||scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H01|E01|normalized_additional_samples_minus_mass_only|1||scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H01|E01|source_mass_only_minus_source_C10|3||scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H01|E01|source_support_fixed_C_minus_source_C10|3||scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H01|E01|source_support_normalized_mass_minus_source_C10|3||scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H01|E01|additional_samples_minus_loss_mass_only|3||scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H01|E01|normalized_mass_minus_fixed_C|3||scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H01|E01|normalized_additional_samples_minus_mass_only|3||scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H01|E01|source_mass_only_minus_source_C10|5||scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H01|E01|source_support_fixed_C_minus_source_C10|5||scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H01|E01|source_support_normalized_mass_minus_source_C10|5||scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H01|E01|additional_samples_minus_loss_mass_only|5||scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H01|E01|normalized_mass_minus_fixed_C|5||scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H01|E01|normalized_additional_samples_minus_mass_only|5||scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H01|E01|source_mass_only_minus_source_C10|10||scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H01|E01|source_support_fixed_C_minus_source_C10|10||scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H01|E01|source_support_normalized_mass_minus_source_C10|10||scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H01|E01|additional_samples_minus_loss_mass_only|10||scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H01|E01|normalized_mass_minus_fixed_C|10||scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H01|E01|normalized_additional_samples_minus_mass_only|10||scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H02|E01|none_minus_balanced|0|0.1|ratio=A, coverage=0.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H02|E01|none_minus_balanced|0|0.1|ratio=A, coverage=0.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|40/40|
|H02|E01|none_minus_balanced|0|1|ratio=A, coverage=0.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H02|E01|none_minus_balanced|0|1|ratio=A, coverage=0.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|40/40|
|H02|E01|none_minus_balanced|0|10|ratio=A, coverage=0.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H02|E01|none_minus_balanced|0|10|ratio=A, coverage=0.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|40/40|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=0.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=0.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|40/40|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=0.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=0.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|40/40|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=0.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=0.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|40/40|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=0.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=0.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|40/40|
|H02|E01|score_argmax_minus_native|0|0.1|scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H02|E01|score_argmax_minus_native|0|0.1|scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H02|E01|score_argmax_minus_native|0|1|scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H02|E01|score_argmax_minus_native|0|1|scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H02|E01|score_argmax_minus_native|0|10|scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H02|E01|score_argmax_minus_native|0|10|scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H03|E01|saved_weak_minus_identity|1||scope=ALL|+0.000 [+0.000,+0.000]|0/0|440/40|
|H03|E01|saved_weak_minus_identity|3||scope=ALL|+0.000 [+0.000,+0.000]|0/0|440/40|
|H03|E01|saved_weak_minus_identity|5||scope=ALL|+0.000 [+0.000,+0.000]|0/0|440/40|
|H03|E01|saved_weak_minus_identity|10||scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H03|E01|saved_weak_minus_identity|10||scope=ALL|+0.000 [+0.000,+0.000]|0/0|640/40|
|H03|E01|score_argmax_minus_native_zero_support|0||scope=ALL|+0.000 [+0.000,+0.000]|0/0|40/40|
|H04|E01|anchor_unweighted_minus_identity|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|anchor_class_balanced_minus_identity|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|bias_unweighted_minus_identity|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|bias_class_balanced_minus_identity|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|anchor_class_balanced_minus_unweighted|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|bias_class_balanced_minus_unweighted|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|anchor_minus_bias_unweighted|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|anchor_minus_bias_class_balanced|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|anchor_unweighted_minus_identity|不适用||composition_id=binary_1_9, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|anchor_class_balanced_minus_identity|不适用||composition_id=binary_1_9, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|bias_unweighted_minus_identity|不适用||composition_id=binary_1_9, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|bias_class_balanced_minus_identity|不适用||composition_id=binary_1_9, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|anchor_class_balanced_minus_unweighted|不适用||composition_id=binary_1_9, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|bias_class_balanced_minus_unweighted|不适用||composition_id=binary_1_9, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|anchor_minus_bias_unweighted|不适用||composition_id=binary_1_9, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|anchor_minus_bias_class_balanced|不适用||composition_id=binary_1_9, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_1_9, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_1_9, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_1_9, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_1_9, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|anchor_unweighted_minus_identity|不适用||composition_id=binary_3_7, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|anchor_class_balanced_minus_identity|不适用||composition_id=binary_3_7, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|bias_unweighted_minus_identity|不适用||composition_id=binary_3_7, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|bias_class_balanced_minus_identity|不适用||composition_id=binary_3_7, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|anchor_class_balanced_minus_unweighted|不适用||composition_id=binary_3_7, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|bias_class_balanced_minus_unweighted|不适用||composition_id=binary_3_7, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|anchor_minus_bias_unweighted|不适用||composition_id=binary_3_7, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|anchor_minus_bias_class_balanced|不适用||composition_id=binary_3_7, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_3_7, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_3_7, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_3_7, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_3_7, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|anchor_unweighted_minus_identity|不适用||composition_id=binary_7_3, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|anchor_class_balanced_minus_identity|不适用||composition_id=binary_7_3, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|bias_unweighted_minus_identity|不适用||composition_id=binary_7_3, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|bias_class_balanced_minus_identity|不适用||composition_id=binary_7_3, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|anchor_class_balanced_minus_unweighted|不适用||composition_id=binary_7_3, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|bias_class_balanced_minus_unweighted|不适用||composition_id=binary_7_3, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|anchor_minus_bias_unweighted|不适用||composition_id=binary_7_3, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|anchor_minus_bias_class_balanced|不适用||composition_id=binary_7_3, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_7_3, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_7_3, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_7_3, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_7_3, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|anchor_unweighted_minus_identity|不适用||composition_id=binary_9_1, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|anchor_class_balanced_minus_identity|不适用||composition_id=binary_9_1, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|bias_unweighted_minus_identity|不适用||composition_id=binary_9_1, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|bias_class_balanced_minus_identity|不适用||composition_id=binary_9_1, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|anchor_class_balanced_minus_unweighted|不适用||composition_id=binary_9_1, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|bias_class_balanced_minus_unweighted|不适用||composition_id=binary_9_1, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|anchor_minus_bias_unweighted|不适用||composition_id=binary_9_1, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|anchor_minus_bias_class_balanced|不适用||composition_id=binary_9_1, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_9_1, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_9_1, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_9_1, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_9_1, scope=ALL|+0.000 [+0.000,+0.000]|0/0|120/40|

## 弱校准激活界限

逐点证据在每任务weak_certificates.csv，聚合在mechanism_analysis.csv。严格间隔证书仅证明这些保存分数的类别不变性；证书未覆盖的点也可能保持类别。解析首次交界不使用Query标签做选择，不代表推荐提高blend；只有随后对齐真实标签的纠错/误伤才能描述当前查询上的有限效用。

## 局限和未运行事项

本轮依据前三轮已曝光内部结果提出并冻结，仅使用master TRAIN，属于探索性机制补充。重复划分、Support重排、类别构成及参数条件不增加独立实物数。描述性95%区间以physical_sample_id聚类，保留同实物所有相关预测；不覆盖来源/批次总体、协议选择及长期开发偏倚。来源与批次仍混杂，不能据此宣称真实跨批次泛化、确认性因果机制或新的独立验证。

人工标签污染证据仅作为既有弱校准路径的诊断副本复用，不证明原始标签有错。二分类和五分类的类别数、样品组成及Support总预算不同，不直接相减解释类别数作用。本轮不根据Query选择最终C、Support构成或校准方法。真实新批次继续为 **NOT_RUN_NEEDS_NEW_BATCH_DATA**；未提供的真实batch_id、仪器及样品属性不推断或补造。

## 复现

```bash
python scripts/run_b_mechanism_round4_20261111.py --resume
python scripts/summarize_b_mechanism_round4_20261111.py --resume
python scripts/summarize_b_mechanism_round4_20261111.py --verify
python scripts/run_b_mechanism_round4_20261111.py --verify
```

PARTIAL报告仅反映当前完整封存子集；核验完成不把未运行任务升级为DONE。
