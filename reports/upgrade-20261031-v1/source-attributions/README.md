# 评分后的源内输入梯度描述

本目录是**source-only、事后可解释性描述**。仅使用旧outer0训练684中的21条（每类按recordID排序首3），分析冻结的`cnn_alpha`、`se_alpha`和内层主策略`fusion_all_four`，均seed42及各自内层选中profile。没有按外层最高分挑家族；不使用当前171测试数据，结果不反馈本版选择、输入或权重。

[三模型PNG](source_alpha_attributions.png) · [矢量PDF](source_alpha_attributions.pdf) · [21条源选择与曝光身份](balanced_source_selection_scoped.csv)

## 实际计算

对已有源scaler转换后的标准化α输入z计算：

`abs(z_alpha * d(logit_predicted_class) / d(z_alpha))`

目标为模型自己预测的类别logit，真实源组标签只用于每类取3的固定均衡采样，未作为梯度目标或数值特征。模型保持eval，调用torch.autograd.grad仅取输入导数；没有创建optimizer、调用backward更新权重或清零已有parameter.grad。所有参数、BN buffers、原有parameter.grad（含None状态）和磁盘模型SHA均未改变。

每模型目录包含：

- `sample_attribution_profiles.csv`：相同21个ID的612点绝对input×gradient；字段`abs_input_gradient_000..611`是缩写，实际值是上式，不是单纯的绝对梯度。
- `input_gradient_arrays.npz`：标准化α、实际带符号输入梯度、绝对input×gradient、逐谱L1归一化、预测类索引和logit。
- `mean_alpha_attribution_612.csv`：612点的原始平均归因及逐谱L1归一化后的平均位置分布。
- `attribution_state_audit.json`：源身份、原权重SHA、state前后及parameter.grad检查。

## 这些图不能证明什么

这是标准化输入空间的局部敏感度，不是SHAP或积分梯度，不是原始α经过SNV／scaler完整链式求导后的化学贡献，更不是浓度或因果归因。绝对值丢失方向，原带符号导数另存。不同模型logit尺度不同，不能凭raw归因总和给模型或药材成分排序。

图为每条source归因先做L1归一化，再平均21条，仅用于描述同模型内的位置。融合模型固定其余n／time／frequency输入，这个α条件导数不能当作四模态重要性排名。

`se_alpha`的SE模块是**通道注意力**；图中频率曲线来自输入导数，不是把SE通道权重当频率权重或含量。没有将归因图与实验局部峰自动匹配，更没有匹配三化合物的理论峰。Spinosin／Jujuboside A／B完整DFT仍NOT_RUN，化学assignment为0。

21条是相同旧source用于三个模型，共63条模型描述记录；不是63份新样品，不是独立query解释或确认集。当前数据仍同批次、已曝光、用户声明独立的EXPLORATORY开发数据。

## 元数据字段的范围澄清

原冻结`balanced_source_selection.csv`的`never_used_as_query=True`仅指同行`outer_fold=0`的本次角色，并不表示历史从未曝光。为避免宽泛解读，保留原脚本／原CSV且新增[明确作用域的展示CSV](balanced_source_selection_scoped.csv)及[澄清记录](identity_scope_clarification.json)，显式标记所有21条为historically_exposed_development_record，非独立确认；没有改ID、归因值或预测。

## 实际检查与可追溯性

[运行前协议锁](protocol_lock.json)绑定源pack、内层锁、代码、三个原模型SHA及采样规则。[独立直接autograd复核](independent_attribution_audit.json)共196676项检查、梯度最大差0；从冻结模型重新取21个输入导数，逐值复核数组、CSV及612点均值，原参数和parameter.grad仍不变。没有新分类器拟合。

[3项实际单元测试](source-attribution-tests.log)通过：线性模型解析梯度与原有非None parameter.grad、BN及None-grad保护、query ID在forward前拒绝。合成fixture只用于软件检查，不计真实样品。PNG已通过view_image实际目检，无裁字、图例遮挡或化学峰标注。

源选择允许读取source_labels；训练、测试分数、测试真值没有进入解释函数。这里是显式API及路径级只读流程，不声称操作系统隔离了研究者对所有历史数据的访问。

重现至新目录：

```bash
.runtime/bin/python scripts/analyze_upgrade_source_attributions.py --output outputs/source-attribution-reproduction-new
```

程序拒绝覆盖非空目录；需要冻结neural run与其原source pack。实际起止记录在父目录`command_invocations.jsonl`，stdout见[execution.log](execution.log)。
