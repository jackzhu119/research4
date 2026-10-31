# 匿名七分类部署：实际模型、模板与验证

本轮按用户要求只使用ZY01—ZY07。已完成75个源内开发比较模型和3个完整855重拟合模型，主模型seed42。模型与原始数据／旧96.02%结果均不互相替换。当前部署模型没有独立新样品准确率，外部Accuracy=null。

最终配置：SNV＋MC-UVE（50次、LV1、20%＝123点）＋RBF-SVM（C1000、gamma0.1×scale、balanced）。选择用五个原源内R2配置、原五fold成员×三seed的15case平均Macro-F1／BA／Accuracy；不是挑原外层最高分，也不是合并旧频点。

[完整使用说明](../../docs/ZAOREN_ANONYMOUS_DEPLOYMENT.md)、[模型与空模板ZIP](ZY七分类_冻结模型与空白模板.zip)、[模板Excel](deployment/templates/new_spectrum.xlsx)、[独立审查](../zaoren-deployment-review-20261031-v1/README.md)。

`deployment/`保存三模型／参数／612精确轴／123选点／训练历史数值匹配参考／环境和SHA。未来推理拒绝历史ID、原谱改名及1e-12舍入重导出；训练内诊断明确不作为准确率。单样本／批量支持Excel、CSV、NPZ，输入不能包含真实标签。真实标签之后由独立scorer连接，未提供新数据时不运行研究评分。

`examples/sample2_training_diagnostic/`是实际原样本2五条，全部已训练标志true、accuracy=null；三个实际负例和未填模板拒绝记录在`examples/`。第一次负例汇总脚本对异常文字的断言不同已原样留档，三次生产拒绝均按预期发生，不因断言措辞差异重新训练。

75模型独审221672项、3最终模型独审12493项通过；855条单点／batch／逆序输出与模型状态一致。21825源内＋1209完整训练共23034个真实学习操作经ID和数值指纹核验。`fit_scope_events.jsonl.gz`无损保留304417200字节原记录，SHA在`fit_scope_manifest.json`，原件移至outputs本地保存；如需运行原审计脚本可先在新工作副本解压恢复原JSONL，不覆盖历史目录。

399项全项目测试实际通过，见`project-tests.log`及`project-test-events.jsonl`。新模型及推理状态检查是有限技术证据，没有OS级沙箱或独立批次确认保证。数学source-only预测链不需要真实地名，仍需确认输入物理量／轴与训练数据相容。

额外描述：`feature-stability/`只读冻结源内选点，108对Jaccard独立复算差0；不产生新预测掩码。新的6条神经96轮轨迹另存[内验证pilot](../zaoren-neural-learning-pilot-20261031-v1/README.md)，不反馈本次SVM选择，也不与855旧OOF混算。

完整复现先prepare新目录并核SHA，再用新source gate运行select，经独立审查后另发full-fit release。旧gate不可授权不同源码／输入／分区；旧数据及结果不改，未合并main。
