# 独立科学复核

复核对象：`manuscript_source.txt`当前稿；对照当前分类白名单、独立V4特征复核、`b_rounds_evidence.md`和`molecular_evidence.md`。未编辑论文源稿。

## 总体结果

未发现需要替换的主要数值。稿件已删除旧混种七类成绩、42/48误差分解及93族实验统计；主文没有复活撤回结果。六任务分母、93.44%主结果、K62的87.11%/81.00%、930/930吸收、Jaccard0.0111、B控制链80.00%/83.78%/83.89%、分子最低频率和IR数值均与现有证据吻合。未把三个重复分区当作新增样品，未声称跨批次验证或已完成分子峰归属。

## 建议优先修正的两处表述

### 摘要明确K62的SNV比较口径

当前摘要写“joint absorption–refractive-index pool… compared with absorption-only selection”，省略了两者的SNV条件。由于原始吸收K62为88.78%，高于联合SNV池87.11%，省略预处理条件容易产生“优于吸收单块方法”的宽泛印象。

建议替换为：

> At a fixed budget of 62 predictors, selection from a joint SNV absorption–refractive-index pool achieved 87.11%, compared with 81.00% for SNV absorption-only selection.

主文结果段已正确列出原始吸收88.78%，无需再加防御性段落。

### 结论收束为已观察到的吸收端点

当前结论“a joint ... pool improved prediction ... showing that candidate integration can act through the selected absorption subset”可能把准确率改善过度归因于子集改变。正文已正确指出两种流程各自inner-selected的C/gamma也可能影响准确率，当前设计没有完全拆开子集效应与SVM设置效应。

建议替换为：

> At K = 62, joint-SNV candidate selection exceeded SNV-absorption selection while retaining absorption variables exclusively, exposing a different absorption-only endpoint at matched predictor width.

它保留核心创新，同时避免把特征子集这一可观察解释写成已完成的单因素因果分解。

## 其余检查

- MC20到MC100的5纠正/5损伤是900条相关重复预测的转换；稿件“repeated predictions”用语正确。
- “均值保留220.6”指SNV吸收MC20内部选K的15外拟合均值，与93.44%是同一主协议；没有用K245的93.78%替换主分数。
- K62联合最终没有n变量的结论涵盖15个outer refit，不涵盖所有inner fit、所有模态或所有药材。正文已用“15 joint-pool outer refits”准确限定。
- 五来源B实验38/10/12每类、190/50/60总数与当前上下文一致。I02预算5/15/25/50是总标签数，稿件未误写成每类shot。
- loss-mass的C=7.9167与Source190、增加50支持后总240相符；当前五类Source与完整Support平衡，C×N作为总惩罚质量控制适用。
- 分子段把两个数值实现限定为同一孤立局部模型层次，未用直接积分与RI-J的绝对能量排序构象。最低74.0312/74.0062 cm−1及2.21940/2.21865 THz、最低IR2.14307/2.13888均正确。
- 引言[9]支持“判别不等于认证”这一认识，不直接支持固态/孤立分子的频率差异；若保留当前引用，最好让它仅附于“classification alone”相关句意，不把其当分子光谱方法出处。该项不影响现有数值结论。
- 主文仅把尚缺制备参数集中指向SI，没有补造压片比例、压力或吸收导出单位。不能将本版称为所有实验细节已经补齐的最终投稿认证稿。

## 叙事聚焦

最强主线已经建立：候选谱块 → 实际保留的吸收变量 → 额外有效标签如何更新决策 → 合格分子模型能解释的频率范围。B结果是新增的机制证据，分子计算是定量物理参照；不应再把未完成元学习或尚未合格的其他分子提为并列主创新。当前正文结构与这一取舍一致。

## 仓库清理注意

本复核没有发现主文引用撤回结果。上传原包属于用户提供的历史输入，内含历史混种及93族文件；最终论文证据包只应包含当前白名单、合格分子与本次干净派生统计，不能把整个上传原包重新打包为当前有效结果。当前清洁V4副本已保存在`analysis/verified_v4_sources/`，其62文件与允许来源固定提交一致，可在移走历史上传输入后单独复现。
