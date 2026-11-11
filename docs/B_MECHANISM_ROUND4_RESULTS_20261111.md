# B方案第四轮：监督损失质量、C约束与Support构成

状态：**DONE**；已封存 **375/375** 个任务包。
本报告只读取封存工件，不拟合或选择模型，不修改前三轮数据、标签、模型、协议和结果。

新增分类器拟合1160次、校准拟合调用2340次；科研任务CPU合计122.154240秒。
保存392700条相关预测，涉及422个唯一Query实物；控制、核验和推送成本另见账本。

## 完成范围

|块|问题|完成/计划|状态|
|---|---|---:|---|
|H01|监督损失质量与新增样品分解|45/45|DONE|
|H02|C × 类别权重的约束激活|240/240|DONE|
|H03|弱校准分数移动与margin激活审计|45/45|DONE|
|H04|固定Support总数的类别构成对照|45/45|DONE|

## 机制定义与配对限制

H01所有条件冻结原Source的Scaler和实际数值gamma。源模型C10为基线；仅提高Source的C到10×N_total/N_source，不加入Support谱或标签；新增Source+Support则比较固定C10和C=10×N_source/N_total。后者固定C×有标签样品数的总损失系数，旧固定C端点直接复用。libSVM实际训练one-vs-one子问题，即使输出OVR分数；本块Source各类均衡且每类添加相同K，因此全局N_total/N_source等于每个OVO类别对的样品数比例，损失质量对照在这个受限设计下成立。本轮描述固定损失质量下的增量监督收益，不能把新增谱值、标签、训练点和损失项的联合变化唯一归因于抽象标签信息。

H02固定Source、Scaler、gamma、Query，预设C=0.1/1/10交叉balanced/None，保存原生SVC与argmax接口。C10复用；Source类别均衡时两种权重的有效训练目标相同，共享同一模型。C降低属于约束激活的机制干预，不是搜索最佳超参数；所有三个C均报告，Source训练内约束诊断不用于模型选择。

H03完全不拟合：复用第二轮34个弱校准回合及第三轮16个正污染回合，排除已重复的零污染K10。每个保存点核验旧top2分数间隔g与实际修正无穷范数d，g>2d足以保持argmax；同时记录分数是否真的移动、类别是否改变。沿固定S(t)=S_source+t×Delta解析计算首次argmax边界，实际t=0.05，不做新blend网格，不依Query真值挑选t。首次边界≤0.05包含恰好相等的边界，argmax采用首索引平局规则，所以它与实际类别改变数分别保存，不能等同。t>1仅是数学外推，不作为适配效用；未受害不等于有收益，分数移动但不跨margin也不等于完全没有计算。

H04二分类Support总数固定10，类别构成为1:9/3:7/5:5/7:3/9:1；五分类总数25，比较5:5:5:5:5及各类轮流稀缺1份、其他类各6份。draw0/101/102在同一候选池取固定前缀；同一ID比较Anchor/Bias及普通/类别均衡残差拟合。类别权重N/(类别数×该类支持数)的总和为N，固定lambda1；均衡K5直接复用，权重等价端点共享结果。composition方法k=-1表示每类K不适用，不表示零支持，实际总量及类别数见support_counts_json。改变构成同时改变实际Support ID和各类覆盖量，不能称纯先验的单因素因果干预；同一构成内加权与普通拟合使用相同ID，是更直接的权重作用对照。

所有Query和Source均来自master TRAIN。Support候选池原本按已知标签分层，构成对照是采样设计诊断，不是已知真实部署先验；残差分数不称后验概率。类别构成改变同时影响每类信息量和有效权重ESS，不能据此宣称跨批次先验修正。

## 全部冻结条件的结果

|块|原任务|方法|K/类|Support构成|Accuracy|Macro-F1|Balanced Accuracy|端点数|
|---|---|---|---:|---|---:|---:|---:|---:|
|H01|E01|identity|0||0.983333|0.983323|0.983333|30|
|H01|E01|source_mass_only|1||0.983333|0.983323|0.983333|30|
|H01|E01|source_mass_only|3||0.983333|0.983323|0.983333|30|
|H01|E01|source_mass_only|5||0.983333|0.983323|0.983333|30|
|H01|E01|source_mass_only|10||0.983333|0.983323|0.983333|30|
|H01|E01|source_support_fixed_C|1||0.983333|0.983323|0.983333|30|
|H01|E01|source_support_fixed_C|3||0.983333|0.983323|0.983333|30|
|H01|E01|source_support_fixed_C|5||0.983333|0.983323|0.983333|30|
|H01|E01|source_support_fixed_C|10||0.983333|0.983323|0.983333|30|
|H01|E01|source_support_normalized_mass|1||0.983333|0.983323|0.983333|30|
|H01|E01|source_support_normalized_mass|3||0.983333|0.983323|0.983333|30|
|H01|E01|source_support_normalized_mass|5||0.983333|0.983323|0.983333|30|
|H01|E01|source_support_normalized_mass|10||0.983333|0.983323|0.983333|30|
|H01|E03|identity|0||0.800000|0.798475|0.800000|15|
|H01|E03|source_mass_only|1||0.800000|0.798475|0.800000|15|
|H01|E03|source_mass_only|3||0.800000|0.798475|0.800000|15|
|H01|E03|source_mass_only|5||0.800000|0.798475|0.800000|15|
|H01|E03|source_mass_only|10||0.800000|0.798475|0.800000|15|
|H01|E03|source_support_fixed_C|1||0.801111|0.799543|0.801111|15|
|H01|E03|source_support_fixed_C|3||0.802222|0.800661|0.802222|15|
|H01|E03|source_support_fixed_C|5||0.821111|0.820229|0.821111|15|
|H01|E03|source_support_fixed_C|10||0.837778|0.836784|0.837778|15|
|H01|E03|source_support_normalized_mass|1||0.801111|0.799543|0.801111|15|
|H01|E03|source_support_normalized_mass|3||0.802222|0.800661|0.802222|15|
|H01|E03|source_support_normalized_mass|5||0.821111|0.820229|0.821111|15|
|H01|E03|source_support_normalized_mass|10||0.837778|0.836784|0.837778|15|
|H02|E01|C_0p1_balanced|0||0.984375|0.984353|0.984375|240|
|H02|E01|C_0p1_balanced_argmax|0||0.984375|0.984353|0.984375|240|
|H02|E01|C_0p1_none|0||0.983437|0.983416|0.983437|240|
|H02|E01|C_0p1_none_argmax|0||0.983437|0.983416|0.983437|240|
|H02|E01|C_10_balanced|0||0.987917|0.987897|0.987917|240|
|H02|E01|C_10_balanced_argmax|0||0.987917|0.987897|0.987917|240|
|H02|E01|C_10_none|0||0.987917|0.987897|0.987917|240|
|H02|E01|C_10_none_argmax|0||0.987917|0.987897|0.987917|240|
|H02|E01|C_1_balanced|0||0.987500|0.987480|0.987500|240|
|H02|E01|C_1_balanced_argmax|0||0.987500|0.987480|0.987500|240|
|H02|E01|C_1_none|0||0.987708|0.987689|0.987708|240|
|H02|E01|C_1_none_argmax|0||0.987708|0.987689|0.987708|240|
|H03|E01|f01_weak|1||0.983333|0.983323|0.983333|330|
|H03|E01|f01_weak|3||0.983333|0.983323|0.983333|330|
|H03|E01|f01_weak|5||0.983333|0.983323|0.983333|330|
|H03|E01|f01_weak|10||0.983333|0.983323|0.983333|30|
|H03|E01|g03_weak|10||0.983333|0.983323|0.983333|480|
|H03|E01|identity|0||0.983333|0.983323|0.983333|30|
|H03|E01|svc_predict|0||0.983333|0.983323|0.983333|30|
|H03|E03|f01_weak|1||0.800404|0.798858|0.800404|165|
|H03|E03|f01_weak|3||0.800303|0.798724|0.800303|165|
|H03|E03|f01_weak|5||0.800404|0.798837|0.800404|165|
|H03|E03|f01_weak|10||0.801111|0.799554|0.801111|15|
|H03|E03|g03_weak|10||0.801181|0.799634|0.801181|240|
|H03|E03|identity|0||0.800000|0.798475|0.800000|15|
|H03|E03|svc_predict|0||0.800000|0.798475|0.800000|15|
|H04|E01|anchor_class_balanced|不适用|balanced|0.983333|0.983323|0.983333|90|
|H04|E01|anchor_class_balanced|不适用|binary_1_9|0.983333|0.983323|0.983333|90|
|H04|E01|anchor_class_balanced|不适用|binary_3_7|0.983333|0.983323|0.983333|90|
|H04|E01|anchor_class_balanced|不适用|binary_7_3|0.983333|0.983323|0.983333|90|
|H04|E01|anchor_class_balanced|不适用|binary_9_1|0.983333|0.983323|0.983333|90|
|H04|E01|anchor_unweighted|不适用|balanced|0.983333|0.983323|0.983333|90|
|H04|E01|anchor_unweighted|不适用|binary_1_9|0.984444|0.984435|0.984444|90|
|H04|E01|anchor_unweighted|不适用|binary_3_7|0.983333|0.983323|0.983333|90|
|H04|E01|anchor_unweighted|不适用|binary_7_3|0.983333|0.983323|0.983333|90|
|H04|E01|anchor_unweighted|不适用|binary_9_1|0.983333|0.983323|0.983333|90|
|H04|E01|bias_class_balanced|不适用|balanced|0.983333|0.983323|0.983333|90|
|H04|E01|bias_class_balanced|不适用|binary_1_9|0.983333|0.983323|0.983333|90|
|H04|E01|bias_class_balanced|不适用|binary_3_7|0.983333|0.983323|0.983333|90|
|H04|E01|bias_class_balanced|不适用|binary_7_3|0.983333|0.983323|0.983333|90|
|H04|E01|bias_class_balanced|不适用|binary_9_1|0.983333|0.983323|0.983333|90|
|H04|E01|bias_unweighted|不适用|balanced|0.983333|0.983323|0.983333|90|
|H04|E01|bias_unweighted|不适用|binary_1_9|0.990833|0.990829|0.990833|90|
|H04|E01|bias_unweighted|不适用|binary_3_7|0.991667|0.991661|0.991667|90|
|H04|E01|bias_unweighted|不适用|binary_7_3|0.983333|0.983323|0.983333|90|
|H04|E01|bias_unweighted|不适用|binary_9_1|0.981667|0.981649|0.981667|90|
|H04|E01|identity|0||0.983333|0.983323|0.983333|30|
|H04|E01|svc_predict|0||0.983333|0.983323|0.983333|30|
|H04|E03|anchor_class_balanced|不适用|balanced|0.776667|0.770945|0.776667|45|
|H04|E03|anchor_class_balanced|不适用|scarce_ZY03|0.721111|0.701152|0.721111|45|
|H04|E03|anchor_class_balanced|不适用|scarce_ZY04|0.743704|0.728146|0.743704|45|
|H04|E03|anchor_class_balanced|不适用|scarce_ZY05|0.760370|0.752891|0.760370|45|
|H04|E03|anchor_class_balanced|不适用|scarce_ZY06|0.725926|0.706392|0.725926|45|
|H04|E03|anchor_class_balanced|不适用|scarce_ZY07|0.726667|0.713834|0.726667|45|
|H04|E03|anchor_unweighted|不适用|balanced|0.776667|0.770945|0.776667|45|
|H04|E03|anchor_unweighted|不适用|scarce_ZY03|0.698889|0.654882|0.698889|45|
|H04|E03|anchor_unweighted|不适用|scarce_ZY04|0.664815|0.609516|0.664815|45|
|H04|E03|anchor_unweighted|不适用|scarce_ZY05|0.701481|0.673886|0.701481|45|
|H04|E03|anchor_unweighted|不适用|scarce_ZY06|0.692593|0.629389|0.692593|45|
|H04|E03|anchor_unweighted|不适用|scarce_ZY07|0.658889|0.603252|0.658889|45|
|H04|E03|bias_class_balanced|不适用|balanced|0.787407|0.783106|0.787407|45|
|H04|E03|bias_class_balanced|不适用|scarce_ZY03|0.781111|0.775712|0.781111|45|
|H04|E03|bias_class_balanced|不适用|scarce_ZY04|0.789259|0.783660|0.789259|45|
|H04|E03|bias_class_balanced|不适用|scarce_ZY05|0.785556|0.781108|0.785556|45|
|H04|E03|bias_class_balanced|不适用|scarce_ZY06|0.781481|0.774723|0.781481|45|
|H04|E03|bias_class_balanced|不适用|scarce_ZY07|0.772222|0.765538|0.772222|45|
|H04|E03|bias_unweighted|不适用|balanced|0.787407|0.783106|0.787407|45|
|H04|E03|bias_unweighted|不适用|scarce_ZY03|0.794815|0.791645|0.794815|45|
|H04|E03|bias_unweighted|不适用|scarce_ZY04|0.772593|0.766215|0.772593|45|
|H04|E03|bias_unweighted|不适用|scarce_ZY05|0.777407|0.771528|0.777407|45|
|H04|E03|bias_unweighted|不适用|scarce_ZY06|0.794815|0.790978|0.794815|45|
|H04|E03|bias_unweighted|不适用|scarce_ZY07|0.760000|0.751482|0.760000|45|
|H04|E03|identity|0||0.800000|0.798475|0.800000|15|
|H04|E03|svc_predict|0||0.800000|0.798475|0.800000|15|

## 配对效应与描述性95%区间

保留全部正、负和零效应。以下区间以physical_sample_id聚类；每来源、每类Recall代价、纠错和误伤详见CSV。

|块|原任务|比较|K/类|C|比例/覆盖/构成/来源范围|ΔAccuracy百分点及95%区间|纠错/误伤|预测行/实物|
|---|---|---|---:|---:|---|---:|---:|---:|
|H01|E01|source_mass_only_minus_source_C10|1||scope=ALL|+0.000 [+0.000,+0.000]|0/0|1200/261|
|H01|E01|source_support_fixed_C_minus_source_C10|1||scope=ALL|+0.000 [+0.000,+0.000]|0/0|1200/261|
|H01|E01|source_support_normalized_mass_minus_source_C10|1||scope=ALL|+0.000 [+0.000,+0.000]|0/0|1200/261|
|H01|E01|additional_samples_minus_loss_mass_only|1||scope=ALL|+0.000 [+0.000,+0.000]|0/0|1200/261|
|H01|E01|normalized_mass_minus_fixed_C|1||scope=ALL|+0.000 [+0.000,+0.000]|0/0|1200/261|
|H01|E01|normalized_additional_samples_minus_mass_only|1||scope=ALL|+0.000 [+0.000,+0.000]|0/0|1200/261|
|H01|E01|source_mass_only_minus_source_C10|3||scope=ALL|+0.000 [+0.000,+0.000]|0/0|1200/261|
|H01|E01|source_support_fixed_C_minus_source_C10|3||scope=ALL|+0.000 [+0.000,+0.000]|0/0|1200/261|
|H01|E01|source_support_normalized_mass_minus_source_C10|3||scope=ALL|+0.000 [+0.000,+0.000]|0/0|1200/261|
|H01|E01|additional_samples_minus_loss_mass_only|3||scope=ALL|+0.000 [+0.000,+0.000]|0/0|1200/261|
|H01|E01|normalized_mass_minus_fixed_C|3||scope=ALL|+0.000 [+0.000,+0.000]|0/0|1200/261|
|H01|E01|normalized_additional_samples_minus_mass_only|3||scope=ALL|+0.000 [+0.000,+0.000]|0/0|1200/261|
|H01|E01|source_mass_only_minus_source_C10|5||scope=ALL|+0.000 [+0.000,+0.000]|0/0|1200/261|
|H01|E01|source_support_fixed_C_minus_source_C10|5||scope=ALL|+0.000 [+0.000,+0.000]|0/0|1200/261|
|H01|E01|source_support_normalized_mass_minus_source_C10|5||scope=ALL|+0.000 [+0.000,+0.000]|0/0|1200/261|
|H01|E01|additional_samples_minus_loss_mass_only|5||scope=ALL|+0.000 [+0.000,+0.000]|0/0|1200/261|
|H01|E01|normalized_mass_minus_fixed_C|5||scope=ALL|+0.000 [+0.000,+0.000]|0/0|1200/261|
|H01|E01|normalized_additional_samples_minus_mass_only|5||scope=ALL|+0.000 [+0.000,+0.000]|0/0|1200/261|
|H01|E01|source_mass_only_minus_source_C10|10||scope=ALL|+0.000 [+0.000,+0.000]|0/0|1200/261|
|H01|E01|source_support_fixed_C_minus_source_C10|10||scope=ALL|+0.000 [+0.000,+0.000]|0/0|1200/261|
|H01|E01|source_support_normalized_mass_minus_source_C10|10||scope=ALL|+0.000 [+0.000,+0.000]|0/0|1200/261|
|H01|E01|additional_samples_minus_loss_mass_only|10||scope=ALL|+0.000 [+0.000,+0.000]|0/0|1200/261|
|H01|E01|normalized_mass_minus_fixed_C|10||scope=ALL|+0.000 [+0.000,+0.000]|0/0|1200/261|
|H01|E01|normalized_additional_samples_minus_mass_only|10||scope=ALL|+0.000 [+0.000,+0.000]|0/0|1200/261|
|H01|E03|source_mass_only_minus_source_C10|1||scope=ALL|+0.000 [+0.000,+0.000]|0/0|900/300|
|H01|E03|source_support_fixed_C_minus_source_C10|1||scope=ALL|+0.111 [-0.778,+1.000]|8/7|900/300|
|H01|E03|source_support_normalized_mass_minus_source_C10|1||scope=ALL|+0.111 [-0.778,+1.000]|8/7|900/300|
|H01|E03|additional_samples_minus_loss_mass_only|1||scope=ALL|+0.111 [-0.778,+1.000]|8/7|900/300|
|H01|E03|normalized_mass_minus_fixed_C|1||scope=ALL|+0.000 [+0.000,+0.000]|0/0|900/300|
|H01|E03|normalized_additional_samples_minus_mass_only|1||scope=ALL|+0.111 [-0.778,+1.000]|8/7|900/300|
|H01|E03|source_mass_only_minus_source_C10|3||scope=ALL|+0.000 [+0.000,+0.000]|0/0|900/300|
|H01|E03|source_support_fixed_C_minus_source_C10|3||scope=ALL|+0.222 [-1.000,+1.333]|16/14|900/300|
|H01|E03|source_support_normalized_mass_minus_source_C10|3||scope=ALL|+0.222 [-1.000,+1.333]|16/14|900/300|
|H01|E03|additional_samples_minus_loss_mass_only|3||scope=ALL|+0.222 [-1.000,+1.333]|16/14|900/300|
|H01|E03|normalized_mass_minus_fixed_C|3||scope=ALL|+0.000 [+0.000,+0.000]|0/0|900/300|
|H01|E03|normalized_additional_samples_minus_mass_only|3||scope=ALL|+0.222 [-1.000,+1.333]|16/14|900/300|
|H01|E03|source_mass_only_minus_source_C10|5||scope=ALL|+0.000 [+0.000,+0.000]|0/0|900/300|
|H01|E03|source_support_fixed_C_minus_source_C10|5||scope=ALL|+2.111 [+0.889,+3.444]|26/7|900/300|
|H01|E03|source_support_normalized_mass_minus_source_C10|5||scope=ALL|+2.111 [+0.889,+3.444]|26/7|900/300|
|H01|E03|additional_samples_minus_loss_mass_only|5||scope=ALL|+2.111 [+0.889,+3.444]|26/7|900/300|
|H01|E03|normalized_mass_minus_fixed_C|5||scope=ALL|+0.000 [+0.000,+0.000]|0/0|900/300|
|H01|E03|normalized_additional_samples_minus_mass_only|5||scope=ALL|+2.111 [+0.889,+3.444]|26/7|900/300|
|H01|E03|source_mass_only_minus_source_C10|10||scope=ALL|+0.000 [+0.000,+0.000]|0/0|900/300|
|H01|E03|source_support_fixed_C_minus_source_C10|10||scope=ALL|+3.778 [+2.111,+5.669]|48/14|900/300|
|H01|E03|source_support_normalized_mass_minus_source_C10|10||scope=ALL|+3.778 [+2.111,+5.669]|48/14|900/300|
|H01|E03|additional_samples_minus_loss_mass_only|10||scope=ALL|+3.778 [+2.111,+5.669]|48/14|900/300|
|H01|E03|normalized_mass_minus_fixed_C|10||scope=ALL|+0.000 [+0.000,+0.000]|0/0|900/300|
|H01|E03|normalized_additional_samples_minus_mass_only|10||scope=ALL|+3.778 [+2.111,+5.669]|48/14|900/300|
|H02|E01|none_minus_balanced|0|0.1|ratio=A, coverage=0.0, scope=ALL|-0.500 [-1.260,+0.000]|0/8|1600/402|
|H02|E01|none_minus_balanced|0|0.1|ratio=A, coverage=0.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=A, coverage=0.0, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=A, coverage=0.0, scope=ZY05|-2.500 [-6.136,+0.000]|0/8|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=A, coverage=0.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=A, coverage=0.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=A, coverage=0.0, scope=ALL|+0.188 [+0.000,+0.709]|3/0|1600/402|
|H02|E01|none_minus_balanced|0|1|ratio=A, coverage=0.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=A, coverage=0.0, scope=ZY04|+0.312 [+0.000,+0.997]|1/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=A, coverage=0.0, scope=ZY05|+0.312 [+0.000,+0.997]|1/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=A, coverage=0.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=A, coverage=0.0, scope=ZY07|+0.312 [+0.000,+0.997]|1/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=A, coverage=0.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|none_minus_balanced|0|10|ratio=A, coverage=0.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=A, coverage=0.0, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=A, coverage=0.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=A, coverage=0.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=A, coverage=0.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=0.0, scope=ALL|-0.500 [-1.398,+0.000]|0/8|1600/402|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=0.0, scope=ZY03|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=0.0, scope=ZY04|-0.625 [-1.608,+0.000]|0/2|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=0.0, scope=ZY05|-0.625 [-1.608,+0.000]|0/2|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=0.0, scope=ZY06|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=0.0, scope=ZY07|-0.625 [-1.608,+0.000]|0/2|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=0.0, scope=ALL|-0.188 [-0.709,+0.000]|0/3|1600/402|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=0.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=0.0, scope=ZY04|-0.312 [-0.997,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=0.0, scope=ZY05|-0.312 [-0.997,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=0.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=0.0, scope=ZY07|-0.312 [-0.997,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=0.0, scope=ALL|-1.000 [-2.173,-0.182]|0/16|1600/402|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=0.0, scope=ZY03|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=0.0, scope=ZY04|-0.625 [-1.608,+0.000]|0/2|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=0.0, scope=ZY05|-3.125 [-6.897,-0.604]|0/10|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=0.0, scope=ZY06|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=0.0, scope=ZY07|-0.625 [-1.608,+0.000]|0/2|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=0.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=0.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=0.0, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=0.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=0.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=0.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=A, coverage=1.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|none_minus_balanced|0|0.1|ratio=A, coverage=1.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=A, coverage=1.0, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=A, coverage=1.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=A, coverage=1.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=A, coverage=1.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=A, coverage=1.0, scope=ALL|+0.062 [+0.000,+0.236]|1/0|1600/402|
|H02|E01|none_minus_balanced|0|1|ratio=A, coverage=1.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=A, coverage=1.0, scope=ZY04|+0.312 [+0.000,+0.997]|1/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=A, coverage=1.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=A, coverage=1.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=A, coverage=1.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=A, coverage=1.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|none_minus_balanced|0|10|ratio=A, coverage=1.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=A, coverage=1.0, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=A, coverage=1.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=A, coverage=1.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=A, coverage=1.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=1.0, scope=ALL|-0.375 [-1.244,+0.000]|0/6|1600/402|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=1.0, scope=ZY03|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=1.0, scope=ZY04|-0.625 [-1.608,+0.000]|0/2|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=1.0, scope=ZY05|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=1.0, scope=ZY06|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=1.0, scope=ZY07|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=1.0, scope=ALL|-0.062 [-0.236,+0.000]|0/1|1600/402|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=1.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=1.0, scope=ZY04|-0.312 [-0.997,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=1.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=1.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=1.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=1.0, scope=ALL|-0.375 [-1.244,+0.000]|0/6|1600/402|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=1.0, scope=ZY03|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=1.0, scope=ZY04|-0.625 [-1.608,+0.000]|0/2|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=1.0, scope=ZY05|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=1.0, scope=ZY06|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=A, coverage=1.0, scope=ZY07|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=1.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=1.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=1.0, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=1.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=1.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=A, coverage=1.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=B, coverage=0.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|none_minus_balanced|0|0.1|ratio=B, coverage=0.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=B, coverage=0.0, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=B, coverage=0.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=B, coverage=0.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=B, coverage=0.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=B, coverage=0.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|none_minus_balanced|0|1|ratio=B, coverage=0.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=B, coverage=0.0, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=B, coverage=0.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=B, coverage=0.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=B, coverage=0.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=B, coverage=0.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|none_minus_balanced|0|10|ratio=B, coverage=0.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=B, coverage=0.0, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=B, coverage=0.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=B, coverage=0.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=B, coverage=0.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=B, coverage=0.0, scope=ALL|-0.312 [-1.203,+0.000]|0/5|1600/402|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=B, coverage=0.0, scope=ZY03|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=B, coverage=0.0, scope=ZY04|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=B, coverage=0.0, scope=ZY05|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=B, coverage=0.0, scope=ZY06|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=B, coverage=0.0, scope=ZY07|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=B, coverage=0.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|C_dose_minus_C10|0|1|ratio=B, coverage=0.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=B, coverage=0.0, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=B, coverage=0.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=B, coverage=0.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=B, coverage=0.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=B, coverage=0.0, scope=ALL|-0.312 [-1.203,+0.000]|0/5|1600/402|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=B, coverage=0.0, scope=ZY03|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=B, coverage=0.0, scope=ZY04|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=B, coverage=0.0, scope=ZY05|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=B, coverage=0.0, scope=ZY06|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=B, coverage=0.0, scope=ZY07|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=B, coverage=0.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|C_dose_minus_C10|0|1|ratio=B, coverage=0.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=B, coverage=0.0, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=B, coverage=0.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=B, coverage=0.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=B, coverage=0.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=B, coverage=1.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|none_minus_balanced|0|0.1|ratio=B, coverage=1.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=B, coverage=1.0, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=B, coverage=1.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=B, coverage=1.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=B, coverage=1.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=B, coverage=1.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|none_minus_balanced|0|1|ratio=B, coverage=1.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=B, coverage=1.0, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=B, coverage=1.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=B, coverage=1.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=B, coverage=1.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=B, coverage=1.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|none_minus_balanced|0|10|ratio=B, coverage=1.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=B, coverage=1.0, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=B, coverage=1.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=B, coverage=1.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=B, coverage=1.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=B, coverage=1.0, scope=ALL|-0.312 [-1.203,+0.000]|0/5|1600/402|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=B, coverage=1.0, scope=ZY03|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=B, coverage=1.0, scope=ZY04|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=B, coverage=1.0, scope=ZY05|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=B, coverage=1.0, scope=ZY06|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=B, coverage=1.0, scope=ZY07|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=B, coverage=1.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|C_dose_minus_C10|0|1|ratio=B, coverage=1.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=B, coverage=1.0, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=B, coverage=1.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=B, coverage=1.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=B, coverage=1.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=B, coverage=1.0, scope=ALL|-0.312 [-1.203,+0.000]|0/5|1600/402|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=B, coverage=1.0, scope=ZY03|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=B, coverage=1.0, scope=ZY04|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=B, coverage=1.0, scope=ZY05|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=B, coverage=1.0, scope=ZY06|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=B, coverage=1.0, scope=ZY07|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=B, coverage=1.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|C_dose_minus_C10|0|1|ratio=B, coverage=1.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=B, coverage=1.0, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=B, coverage=1.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=B, coverage=1.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=B, coverage=1.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=C, coverage=0.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|none_minus_balanced|0|0.1|ratio=C, coverage=0.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=C, coverage=0.0, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=C, coverage=0.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=C, coverage=0.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=C, coverage=0.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=C, coverage=0.0, scope=ALL|-0.125 [-0.481,+0.000]|0/2|1600/402|
|H02|E01|none_minus_balanced|0|1|ratio=C, coverage=0.0, scope=ZY03|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=C, coverage=0.0, scope=ZY04|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=C, coverage=0.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=C, coverage=0.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=C, coverage=0.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=C, coverage=0.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|none_minus_balanced|0|10|ratio=C, coverage=0.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=C, coverage=0.0, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=C, coverage=0.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=C, coverage=0.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=C, coverage=0.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=C, coverage=0.0, scope=ALL|-0.312 [-1.203,+0.000]|0/5|1600/402|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=C, coverage=0.0, scope=ZY03|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=C, coverage=0.0, scope=ZY04|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=C, coverage=0.0, scope=ZY05|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=C, coverage=0.0, scope=ZY06|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=C, coverage=0.0, scope=ZY07|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=C, coverage=0.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|C_dose_minus_C10|0|1|ratio=C, coverage=0.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=C, coverage=0.0, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=C, coverage=0.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=C, coverage=0.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=C, coverage=0.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=C, coverage=0.0, scope=ALL|-0.312 [-1.203,+0.000]|0/5|1600/402|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=C, coverage=0.0, scope=ZY03|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=C, coverage=0.0, scope=ZY04|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=C, coverage=0.0, scope=ZY05|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=C, coverage=0.0, scope=ZY06|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=C, coverage=0.0, scope=ZY07|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=C, coverage=0.0, scope=ALL|-0.125 [-0.481,+0.000]|0/2|1600/402|
|H02|E01|C_dose_minus_C10|0|1|ratio=C, coverage=0.0, scope=ZY03|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=C, coverage=0.0, scope=ZY04|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=C, coverage=0.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=C, coverage=0.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=C, coverage=0.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=C, coverage=1.0, scope=ALL|-0.062 [-0.243,+0.000]|0/1|1600/402|
|H02|E01|none_minus_balanced|0|0.1|ratio=C, coverage=1.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=C, coverage=1.0, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=C, coverage=1.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=C, coverage=1.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|0.1|ratio=C, coverage=1.0, scope=ZY07|-0.312 [-1.024,+0.000]|0/1|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=C, coverage=1.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|none_minus_balanced|0|1|ratio=C, coverage=1.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=C, coverage=1.0, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=C, coverage=1.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=C, coverage=1.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|1|ratio=C, coverage=1.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=C, coverage=1.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|none_minus_balanced|0|10|ratio=C, coverage=1.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=C, coverage=1.0, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=C, coverage=1.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=C, coverage=1.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|none_minus_balanced|0|10|ratio=C, coverage=1.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=C, coverage=1.0, scope=ALL|-0.312 [-1.203,+0.000]|0/5|1600/402|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=C, coverage=1.0, scope=ZY03|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=C, coverage=1.0, scope=ZY04|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=C, coverage=1.0, scope=ZY05|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=C, coverage=1.0, scope=ZY06|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=C, coverage=1.0, scope=ZY07|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=C, coverage=1.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|C_dose_minus_C10|0|1|ratio=C, coverage=1.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=C, coverage=1.0, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=C, coverage=1.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=C, coverage=1.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=C, coverage=1.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=C, coverage=1.0, scope=ALL|-0.375 [-1.216,+0.000]|0/6|1600/402|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=C, coverage=1.0, scope=ZY03|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=C, coverage=1.0, scope=ZY04|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=C, coverage=1.0, scope=ZY05|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=C, coverage=1.0, scope=ZY06|-0.312 [-1.007,+0.000]|0/1|320/178|
|H02|E01|C_dose_minus_C10|0|0.1|ratio=C, coverage=1.0, scope=ZY07|-0.625 [-1.656,+0.000]|0/2|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=C, coverage=1.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|C_dose_minus_C10|0|1|ratio=C, coverage=1.0, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=C, coverage=1.0, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=C, coverage=1.0, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=C, coverage=1.0, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|C_dose_minus_C10|0|1|ratio=C, coverage=1.0, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=A, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=A, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=A, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=A, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=A, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=A, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=B, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=B, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=B, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=B, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=B, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=B, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=C, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=C, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=C, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=C, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=C, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=C, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|ratio_B_minus_A|0|0.1|coverage=0.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|ratio_C_minus_A|0|0.1|coverage=0.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|ratio_C_minus_B|0|0.1|coverage=0.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|ratio_B_minus_A|0|0.1|coverage=1.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|ratio_C_minus_A|0|0.1|coverage=1.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|ratio_C_minus_B|0|0.1|coverage=1.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|score_argmax_minus_native|0|0.1|scope=ALL|+0.000 [+0.000,+0.000]|0/0|9600/402|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=A, scope=ALL|+0.500 [+0.000,+1.260]|8/0|1600/402|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=A, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=A, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=A, scope=ZY05|+2.500 [+0.000,+6.136]|8/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=A, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=A, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=B, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=B, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=B, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=B, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=B, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=B, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=C, scope=ALL|-0.062 [-0.243,+0.000]|0/1|1600/402|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=C, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=C, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=C, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=C, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|0.1|ratio=C, scope=ZY07|-0.312 [-1.024,+0.000]|0/1|320/178|
|H02|E01|ratio_B_minus_A|0|0.1|coverage=0.0, scope=ALL|+0.500 [+0.000,+1.260]|8/0|1600/402|
|H02|E01|ratio_C_minus_A|0|0.1|coverage=0.0, scope=ALL|+0.500 [+0.000,+1.260]|8/0|1600/402|
|H02|E01|ratio_C_minus_B|0|0.1|coverage=0.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|ratio_B_minus_A|0|0.1|coverage=1.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|ratio_C_minus_A|0|0.1|coverage=1.0, scope=ALL|-0.062 [-0.243,+0.000]|0/1|1600/402|
|H02|E01|ratio_C_minus_B|0|0.1|coverage=1.0, scope=ALL|-0.062 [-0.243,+0.000]|0/1|1600/402|
|H02|E01|score_argmax_minus_native|0|0.1|scope=ALL|+0.000 [+0.000,+0.000]|0/0|9600/402|
|H02|E01|coverage_1_minus_0|0|1|ratio=A, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|coverage_1_minus_0|0|1|ratio=A, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=A, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=A, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=A, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=A, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=B, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|coverage_1_minus_0|0|1|ratio=B, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=B, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=B, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=B, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=B, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=C, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|coverage_1_minus_0|0|1|ratio=C, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=C, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=C, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=C, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=C, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|ratio_B_minus_A|0|1|coverage=0.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|ratio_C_minus_A|0|1|coverage=0.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|ratio_C_minus_B|0|1|coverage=0.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|ratio_B_minus_A|0|1|coverage=1.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|ratio_C_minus_A|0|1|coverage=1.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|ratio_C_minus_B|0|1|coverage=1.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|score_argmax_minus_native|0|1|scope=ALL|+0.000 [+0.000,+0.000]|0/0|9600/402|
|H02|E01|coverage_1_minus_0|0|1|ratio=A, scope=ALL|-0.125 [-0.473,+0.000]|0/2|1600/402|
|H02|E01|coverage_1_minus_0|0|1|ratio=A, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=A, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=A, scope=ZY05|-0.312 [-0.997,+0.000]|0/1|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=A, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=A, scope=ZY07|-0.312 [-0.997,+0.000]|0/1|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=B, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|coverage_1_minus_0|0|1|ratio=B, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=B, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=B, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=B, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=B, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=C, scope=ALL|+0.125 [+0.000,+0.481]|2/0|1600/402|
|H02|E01|coverage_1_minus_0|0|1|ratio=C, scope=ZY03|+0.312 [+0.000,+1.007]|1/0|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=C, scope=ZY04|+0.312 [+0.000,+1.007]|1/0|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=C, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=C, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|1|ratio=C, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|ratio_B_minus_A|0|1|coverage=0.0, scope=ALL|-0.188 [-0.709,+0.000]|0/3|1600/402|
|H02|E01|ratio_C_minus_A|0|1|coverage=0.0, scope=ALL|-0.312 [-0.847,+0.000]|0/5|1600/402|
|H02|E01|ratio_C_minus_B|0|1|coverage=0.0, scope=ALL|-0.125 [-0.481,+0.000]|0/2|1600/402|
|H02|E01|ratio_B_minus_A|0|1|coverage=1.0, scope=ALL|-0.062 [-0.236,+0.000]|0/1|1600/402|
|H02|E01|ratio_C_minus_A|0|1|coverage=1.0, scope=ALL|-0.062 [-0.236,+0.000]|0/1|1600/402|
|H02|E01|ratio_C_minus_B|0|1|coverage=1.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|score_argmax_minus_native|0|1|scope=ALL|+0.000 [+0.000,+0.000]|0/0|9600/402|
|H02|E01|coverage_1_minus_0|0|10|ratio=A, scope=ALL|-0.125 [-0.473,+0.000]|0/2|1600/402|
|H02|E01|coverage_1_minus_0|0|10|ratio=A, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=A, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=A, scope=ZY05|-0.312 [-0.997,+0.000]|0/1|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=A, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=A, scope=ZY07|-0.312 [-0.997,+0.000]|0/1|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=B, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|coverage_1_minus_0|0|10|ratio=B, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=B, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=B, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=B, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=B, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=C, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|coverage_1_minus_0|0|10|ratio=C, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=C, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=C, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=C, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=C, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|ratio_B_minus_A|0|10|coverage=0.0, scope=ALL|-0.188 [-0.709,+0.000]|0/3|1600/402|
|H02|E01|ratio_C_minus_A|0|10|coverage=0.0, scope=ALL|-0.188 [-0.709,+0.000]|0/3|1600/402|
|H02|E01|ratio_C_minus_B|0|10|coverage=0.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|ratio_B_minus_A|0|10|coverage=1.0, scope=ALL|-0.062 [-0.236,+0.000]|0/1|1600/402|
|H02|E01|ratio_C_minus_A|0|10|coverage=1.0, scope=ALL|-0.062 [-0.236,+0.000]|0/1|1600/402|
|H02|E01|ratio_C_minus_B|0|10|coverage=1.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|score_argmax_minus_native|0|10|scope=ALL|+0.000 [+0.000,+0.000]|0/0|9600/402|
|H02|E01|coverage_1_minus_0|0|10|ratio=A, scope=ALL|-0.125 [-0.473,+0.000]|0/2|1600/402|
|H02|E01|coverage_1_minus_0|0|10|ratio=A, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=A, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=A, scope=ZY05|-0.312 [-0.997,+0.000]|0/1|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=A, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=A, scope=ZY07|-0.312 [-0.997,+0.000]|0/1|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=B, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|coverage_1_minus_0|0|10|ratio=B, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=B, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=B, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=B, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=B, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=C, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|coverage_1_minus_0|0|10|ratio=C, scope=ZY03|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=C, scope=ZY04|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=C, scope=ZY05|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=C, scope=ZY06|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|coverage_1_minus_0|0|10|ratio=C, scope=ZY07|+0.000 [+0.000,+0.000]|0/0|320/178|
|H02|E01|ratio_B_minus_A|0|10|coverage=0.0, scope=ALL|-0.188 [-0.709,+0.000]|0/3|1600/402|
|H02|E01|ratio_C_minus_A|0|10|coverage=0.0, scope=ALL|-0.188 [-0.709,+0.000]|0/3|1600/402|
|H02|E01|ratio_C_minus_B|0|10|coverage=0.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|ratio_B_minus_A|0|10|coverage=1.0, scope=ALL|-0.062 [-0.236,+0.000]|0/1|1600/402|
|H02|E01|ratio_C_minus_A|0|10|coverage=1.0, scope=ALL|-0.062 [-0.236,+0.000]|0/1|1600/402|
|H02|E01|ratio_C_minus_B|0|10|coverage=1.0, scope=ALL|+0.000 [+0.000,+0.000]|0/0|1600/402|
|H02|E01|score_argmax_minus_native|0|10|scope=ALL|+0.000 [+0.000,+0.000]|0/0|9600/402|
|H03|E01|saved_weak_minus_identity|1||scope=ALL|+0.000 [+0.000,+0.000]|0/0|13200/261|
|H03|E01|saved_weak_minus_identity|3||scope=ALL|+0.000 [+0.000,+0.000]|0/0|13200/261|
|H03|E01|saved_weak_minus_identity|5||scope=ALL|+0.000 [+0.000,+0.000]|0/0|13200/261|
|H03|E01|saved_weak_minus_identity|10||scope=ALL|+0.000 [+0.000,+0.000]|0/0|1200/261|
|H03|E01|saved_weak_minus_identity|10||scope=ALL|+0.000 [+0.000,+0.000]|0/0|19200/261|
|H03|E01|score_argmax_minus_native_zero_support|0||scope=ALL|+0.000 [+0.000,+0.000]|0/0|1200/261|
|H03|E03|saved_weak_minus_identity|1||scope=ALL|+0.040 [+0.000,+0.122]|4/0|9900/300|
|H03|E03|saved_weak_minus_identity|3||scope=ALL|+0.030 [+0.000,+0.092]|3/0|9900/300|
|H03|E03|saved_weak_minus_identity|5||scope=ALL|+0.040 [+0.000,+0.121]|4/0|9900/300|
|H03|E03|saved_weak_minus_identity|10||scope=ALL|+0.111 [+0.000,+0.333]|1/0|900/300|
|H03|E03|saved_weak_minus_identity|10||scope=ALL|+0.118 [+0.000,+0.347]|17/0|14400/300|
|H03|E03|score_argmax_minus_native_zero_support|0||scope=ALL|+0.000 [+0.000,+0.000]|0/0|900/300|
|H04|E01|anchor_unweighted_minus_identity|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|anchor_class_balanced_minus_identity|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|bias_unweighted_minus_identity|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|bias_class_balanced_minus_identity|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|anchor_class_balanced_minus_unweighted|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|bias_class_balanced_minus_unweighted|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|anchor_minus_bias_unweighted|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|anchor_minus_bias_class_balanced|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|anchor_unweighted_minus_identity|不适用||composition_id=binary_1_9, scope=ALL|+0.111 [+0.000,+0.434]|4/0|3600/261|
|H04|E01|anchor_class_balanced_minus_identity|不适用||composition_id=binary_1_9, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|bias_unweighted_minus_identity|不适用||composition_id=binary_1_9, scope=ALL|+0.750 [-0.246,+3.218]|30/3|3600/261|
|H04|E01|bias_class_balanced_minus_identity|不适用||composition_id=binary_1_9, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|anchor_class_balanced_minus_unweighted|不适用||composition_id=binary_1_9, scope=ALL|-0.111 [-0.434,+0.000]|0/4|3600/261|
|H04|E01|bias_class_balanced_minus_unweighted|不适用||composition_id=binary_1_9, scope=ALL|-0.750 [-3.218,+0.246]|3/30|3600/261|
|H04|E01|anchor_minus_bias_unweighted|不适用||composition_id=binary_1_9, scope=ALL|-0.639 [-2.778,+0.246]|3/26|3600/261|
|H04|E01|anchor_minus_bias_class_balanced|不适用||composition_id=binary_1_9, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_1_9, scope=ALL|+0.111 [+0.000,+0.434]|4/0|3600/261|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_1_9, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_1_9, scope=ALL|+0.750 [-0.246,+3.218]|30/3|3600/261|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_1_9, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|anchor_unweighted_minus_identity|不适用||composition_id=binary_3_7, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|anchor_class_balanced_minus_identity|不适用||composition_id=binary_3_7, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|bias_unweighted_minus_identity|不适用||composition_id=binary_3_7, scope=ALL|+0.833 [+0.000,+3.252]|30/0|3600/261|
|H04|E01|bias_class_balanced_minus_identity|不适用||composition_id=binary_3_7, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|anchor_class_balanced_minus_unweighted|不适用||composition_id=binary_3_7, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|bias_class_balanced_minus_unweighted|不适用||composition_id=binary_3_7, scope=ALL|-0.833 [-3.252,+0.000]|0/30|3600/261|
|H04|E01|anchor_minus_bias_unweighted|不适用||composition_id=binary_3_7, scope=ALL|-0.833 [-3.252,+0.000]|0/30|3600/261|
|H04|E01|anchor_minus_bias_class_balanced|不适用||composition_id=binary_3_7, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_3_7, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_3_7, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_3_7, scope=ALL|+0.833 [+0.000,+3.252]|30/0|3600/261|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_3_7, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|anchor_unweighted_minus_identity|不适用||composition_id=binary_7_3, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|anchor_class_balanced_minus_identity|不适用||composition_id=binary_7_3, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|bias_unweighted_minus_identity|不适用||composition_id=binary_7_3, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|bias_class_balanced_minus_identity|不适用||composition_id=binary_7_3, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|anchor_class_balanced_minus_unweighted|不适用||composition_id=binary_7_3, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|bias_class_balanced_minus_unweighted|不适用||composition_id=binary_7_3, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|anchor_minus_bias_unweighted|不适用||composition_id=binary_7_3, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|anchor_minus_bias_class_balanced|不适用||composition_id=binary_7_3, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_7_3, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_7_3, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_7_3, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_7_3, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|anchor_unweighted_minus_identity|不适用||composition_id=binary_9_1, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|anchor_class_balanced_minus_identity|不适用||composition_id=binary_9_1, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|bias_unweighted_minus_identity|不适用||composition_id=binary_9_1, scope=ALL|-0.167 [-0.540,+0.000]|0/6|3600/261|
|H04|E01|bias_class_balanced_minus_identity|不适用||composition_id=binary_9_1, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|anchor_class_balanced_minus_unweighted|不适用||composition_id=binary_9_1, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|bias_class_balanced_minus_unweighted|不适用||composition_id=binary_9_1, scope=ALL|+0.167 [+0.000,+0.540]|6/0|3600/261|
|H04|E01|anchor_minus_bias_unweighted|不适用||composition_id=binary_9_1, scope=ALL|+0.167 [+0.000,+0.540]|6/0|3600/261|
|H04|E01|anchor_minus_bias_class_balanced|不适用||composition_id=binary_9_1, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_9_1, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_9_1, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_9_1, scope=ALL|-0.167 [-0.540,+0.000]|0/6|3600/261|
|H04|E01|composition_minus_balanced_support|不适用||composition_id=binary_9_1, scope=ALL|+0.000 [+0.000,+0.000]|0/0|3600/261|
|H04|E03|anchor_unweighted_minus_identity|不适用||composition_id=balanced, scope=ALL|-2.333 [-3.556,-1.148]|41/104|2700/300|
|H04|E03|anchor_class_balanced_minus_identity|不适用||composition_id=balanced, scope=ALL|-2.333 [-3.556,-1.148]|41/104|2700/300|
|H04|E03|bias_unweighted_minus_identity|不适用||composition_id=balanced, scope=ALL|-1.259 [-2.519,+0.038]|39/73|2700/300|
|H04|E03|bias_class_balanced_minus_identity|不适用||composition_id=balanced, scope=ALL|-1.259 [-2.519,+0.038]|39/73|2700/300|
|H04|E03|anchor_class_balanced_minus_unweighted|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|2700/300|
|H04|E03|bias_class_balanced_minus_unweighted|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|2700/300|
|H04|E03|anchor_minus_bias_unweighted|不适用||composition_id=balanced, scope=ALL|-1.074 [-2.444,+0.296]|78/107|2700/300|
|H04|E03|anchor_minus_bias_class_balanced|不适用||composition_id=balanced, scope=ALL|-1.074 [-2.444,+0.296]|78/107|2700/300|
|H04|E03|composition_minus_balanced_support|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|2700/300|
|H04|E03|composition_minus_balanced_support|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|2700/300|
|H04|E03|composition_minus_balanced_support|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|2700/300|
|H04|E03|composition_minus_balanced_support|不适用||composition_id=balanced, scope=ALL|+0.000 [+0.000,+0.000]|0/0|2700/300|
|H04|E03|anchor_unweighted_minus_identity|不适用||composition_id=scarce_ZY03, scope=ALL|-10.111 [-13.370,-6.810]|81/354|2700/300|
|H04|E03|anchor_class_balanced_minus_identity|不适用||composition_id=scarce_ZY03, scope=ALL|-7.889 [-10.186,-5.665]|82/295|2700/300|
|H04|E03|bias_unweighted_minus_identity|不适用||composition_id=scarce_ZY03, scope=ALL|-0.519 [-1.407,+0.407]|27/41|2700/300|
|H04|E03|bias_class_balanced_minus_identity|不适用||composition_id=scarce_ZY03, scope=ALL|-1.889 [-3.259,-0.629]|34/85|2700/300|
|H04|E03|anchor_class_balanced_minus_unweighted|不适用||composition_id=scarce_ZY03, scope=ALL|+2.222 [+0.073,+4.296]|163/103|2700/300|
|H04|E03|bias_class_balanced_minus_unweighted|不适用||composition_id=scarce_ZY03, scope=ALL|-1.370 [-2.186,-0.630]|7/44|2700/300|
|H04|E03|anchor_minus_bias_unweighted|不适用||composition_id=scarce_ZY03, scope=ALL|-9.593 [-12.852,-6.370]|80/339|2700/300|
|H04|E03|anchor_minus_bias_class_balanced|不适用||composition_id=scarce_ZY03, scope=ALL|-6.000 [-8.185,-3.888]|102/264|2700/300|
|H04|E03|composition_minus_balanced_support|不适用||composition_id=scarce_ZY03, scope=ALL|-7.778 [-10.631,-4.852]|84/294|2700/300|
|H04|E03|composition_minus_balanced_support|不适用||composition_id=scarce_ZY03, scope=ALL|-5.556 [-7.296,-3.815]|78/228|2700/300|
|H04|E03|composition_minus_balanced_support|不适用||composition_id=scarce_ZY03, scope=ALL|+0.741 [+0.110,+1.444]|45/25|2700/300|
|H04|E03|composition_minus_balanced_support|不适用||composition_id=scarce_ZY03, scope=ALL|-0.630 [-1.223,-0.037]|23/40|2700/300|
|H04|E03|anchor_unweighted_minus_identity|不适用||composition_id=scarce_ZY04, scope=ALL|-13.519 [-17.704,-9.148]|135/500|2700/300|
|H04|E03|anchor_class_balanced_minus_identity|不适用||composition_id=scarce_ZY04, scope=ALL|-5.630 [-7.408,-3.702]|60/212|2700/300|
|H04|E03|bias_unweighted_minus_identity|不适用||composition_id=scarce_ZY04, scope=ALL|-2.741 [-4.371,-1.110]|32/106|2700/300|
|H04|E03|bias_class_balanced_minus_identity|不适用||composition_id=scarce_ZY04, scope=ALL|-1.074 [-2.334,+0.259]|45/74|2700/300|
|H04|E03|anchor_class_balanced_minus_unweighted|不适用||composition_id=scarce_ZY04, scope=ALL|+7.889 [+4.481,+11.519]|333/120|2700/300|
|H04|E03|bias_class_balanced_minus_unweighted|不适用||composition_id=scarce_ZY04, scope=ALL|+1.667 [+0.444,+2.963]|71/26|2700/300|
|H04|E03|anchor_minus_bias_unweighted|不适用||composition_id=scarce_ZY04, scope=ALL|-10.778 [-15.038,-6.369]|173/464|2700/300|
|H04|E03|anchor_minus_bias_class_balanced|不适用||composition_id=scarce_ZY04, scope=ALL|-4.556 [-5.889,-2.999]|60/183|2700/300|
|H04|E03|composition_minus_balanced_support|不适用||composition_id=scarce_ZY04, scope=ALL|-11.185 [-15.260,-7.111]|138/440|2700/300|
|H04|E03|composition_minus_balanced_support|不适用||composition_id=scarce_ZY04, scope=ALL|-3.296 [-4.593,-1.851]|61/150|2700/300|
|H04|E03|composition_minus_balanced_support|不适用||composition_id=scarce_ZY04, scope=ALL|-1.481 [-2.667,-0.407]|20/60|2700/300|
|H04|E03|composition_minus_balanced_support|不适用||composition_id=scarce_ZY04, scope=ALL|+0.185 [-0.481,+0.926]|38/33|2700/300|
|H04|E03|anchor_unweighted_minus_identity|不适用||composition_id=scarce_ZY05, scope=ALL|-9.852 [-12.408,-7.258]|41/307|2700/300|
|H04|E03|anchor_class_balanced_minus_identity|不适用||composition_id=scarce_ZY05, scope=ALL|-3.963 [-5.296,-2.593]|35/142|2700/300|
|H04|E03|bias_unweighted_minus_identity|不适用||composition_id=scarce_ZY05, scope=ALL|-2.259 [-4.074,-0.481]|54/115|2700/300|
|H04|E03|bias_class_balanced_minus_identity|不适用||composition_id=scarce_ZY05, scope=ALL|-1.444 [-2.741,-0.111]|35/74|2700/300|
|H04|E03|anchor_class_balanced_minus_unweighted|不适用||composition_id=scarce_ZY05, scope=ALL|+5.889 [+3.444,+8.222]|231/72|2700/300|
|H04|E03|bias_class_balanced_minus_unweighted|不适用||composition_id=scarce_ZY05, scope=ALL|+0.815 [-0.259,+1.889]|54/32|2700/300|
|H04|E03|anchor_minus_bias_unweighted|不适用||composition_id=scarce_ZY05, scope=ALL|-7.593 [-10.854,-4.368]|140/345|2700/300|
|H04|E03|anchor_minus_bias_class_balanced|不适用||composition_id=scarce_ZY05, scope=ALL|-2.519 [-4.112,-0.999]|78/146|2700/300|
|H04|E03|composition_minus_balanced_support|不适用||composition_id=scarce_ZY05, scope=ALL|-7.519 [-10.334,-4.919]|88/291|2700/300|
|H04|E03|composition_minus_balanced_support|不适用||composition_id=scarce_ZY05, scope=ALL|-1.630 [-2.593,-0.741]|51/95|2700/300|
|H04|E03|composition_minus_balanced_support|不适用||composition_id=scarce_ZY05, scope=ALL|-1.000 [-2.037,+0.000]|30/57|2700/300|
|H04|E03|composition_minus_balanced_support|不适用||composition_id=scarce_ZY05, scope=ALL|-0.185 [-0.667,+0.296]|17/22|2700/300|
|H04|E03|anchor_unweighted_minus_identity|不适用||composition_id=scarce_ZY06, scope=ALL|-10.741 [-14.890,-6.665]|121/411|2700/300|
|H04|E03|anchor_class_balanced_minus_identity|不适用||composition_id=scarce_ZY06, scope=ALL|-7.407 [-9.481,-5.333]|81/281|2700/300|
|H04|E03|bias_unweighted_minus_identity|不适用||composition_id=scarce_ZY06, scope=ALL|-0.519 [-1.815,+0.778]|40/54|2700/300|
|H04|E03|bias_class_balanced_minus_identity|不适用||composition_id=scarce_ZY06, scope=ALL|-1.852 [-3.333,-0.369]|44/94|2700/300|
|H04|E03|anchor_class_balanced_minus_unweighted|不适用||composition_id=scarce_ZY06, scope=ALL|+3.333 [+0.258,+6.333]|255/165|2700/300|
|H04|E03|bias_class_balanced_minus_unweighted|不适用||composition_id=scarce_ZY06, scope=ALL|-1.333 [-2.260,-0.481]|19/55|2700/300|
|H04|E03|anchor_minus_bias_unweighted|不适用||composition_id=scarce_ZY06, scope=ALL|-10.222 [-13.853,-6.481]|111/387|2700/300|
|H04|E03|anchor_minus_bias_class_balanced|不适用||composition_id=scarce_ZY06, scope=ALL|-5.556 [-7.333,-3.666]|98/248|2700/300|
|H04|E03|composition_minus_balanced_support|不适用||composition_id=scarce_ZY06, scope=ALL|-8.407 [-12.222,-4.775]|127/354|2700/300|
|H04|E03|composition_minus_balanced_support|不适用||composition_id=scarce_ZY06, scope=ALL|-5.074 [-6.816,-3.481]|67/204|2700/300|
|H04|E03|composition_minus_balanced_support|不适用||composition_id=scarce_ZY06, scope=ALL|+0.741 [+0.037,+1.481]|40/20|2700/300|
|H04|E03|composition_minus_balanced_support|不适用||composition_id=scarce_ZY06, scope=ALL|-0.593 [-1.259,+0.074]|31/47|2700/300|
|H04|E03|anchor_unweighted_minus_identity|不适用||composition_id=scarce_ZY07, scope=ALL|-14.111 [-18.111,-10.370]|87/468|2700/300|
|H04|E03|anchor_class_balanced_minus_identity|不适用||composition_id=scarce_ZY07, scope=ALL|-7.333 [-9.185,-5.444]|62/260|2700/300|
|H04|E03|bias_unweighted_minus_identity|不适用||composition_id=scarce_ZY07, scope=ALL|-4.000 [-5.890,-2.110]|38/146|2700/300|
|H04|E03|bias_class_balanced_minus_identity|不适用||composition_id=scarce_ZY07, scope=ALL|-2.778 [-4.259,-1.184]|41/116|2700/300|
|H04|E03|anchor_class_balanced_minus_unweighted|不适用||composition_id=scarce_ZY07, scope=ALL|+6.778 [+3.778,+10.075]|313/130|2700/300|
|H04|E03|bias_class_balanced_minus_unweighted|不适用||composition_id=scarce_ZY07, scope=ALL|+1.222 [+0.147,+2.556]|63/30|2700/300|
|H04|E03|anchor_minus_bias_unweighted|不适用||composition_id=scarce_ZY07, scope=ALL|-10.111 [-14.481,-6.111]|185/458|2700/300|
|H04|E03|anchor_minus_bias_class_balanced|不适用||composition_id=scarce_ZY07, scope=ALL|-4.556 [-6.260,-2.814]|109/232|2700/300|
|H04|E03|composition_minus_balanced_support|不适用||composition_id=scarce_ZY07, scope=ALL|-11.778 [-15.779,-8.185]|97/415|2700/300|
|H04|E03|composition_minus_balanced_support|不适用||composition_id=scarce_ZY07, scope=ALL|-5.000 [-6.297,-3.593]|61/196|2700/300|
|H04|E03|composition_minus_balanced_support|不适用||composition_id=scarce_ZY07, scope=ALL|-2.741 [-4.111,-1.519]|10/84|2700/300|
|H04|E03|composition_minus_balanced_support|不适用||composition_id=scarce_ZY07, scope=ALL|-1.519 [-2.222,-0.778]|10/51|2700/300|

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
