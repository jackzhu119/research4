# Research4升级运行证据：2026-10-31

本目录为新运行`upgrade-20261031-v1`，保留原始材料、旧权重与历史成绩。主任务仍为酸枣仁THz来源组识别：已审计R2为821／855＝96.0234%、Macro-F1 0.9197。本次内层选择的四表示神经策略为727／855＝85.0292%、Macro-F1 0.7449，三seed准确率均值83.7037%。没有为使神经胜出改变标签、样品、候选预算或在外层后追加epoch。

这是历史曝光样品上的内部开发，以及独立植物任务上的重新训练探索；没有新增独立批次确认，不宣称绝对零泄漏、成功迁移或已确认分子机制。

## 阅读顺序

1. [完整中文报告](../../docs/MULTIREPRESENTATION_UPGRADE_REPORT.md)与[要求逐项对应](../../docs/UPGRADE_REQUIREMENTS_TRACEABILITY.md)。
2. [15表Excel、五组PNG／PDF及表图来源](paper-assets/README.md)。
3. [数据库与质量](../../docs/MULTIPLANT_DATA_AUDIT.md)、[数据曝光](../../docs/DATA_EXPOSURE_REGISTER.md)、[严格防泄漏协议](../../docs/DATA_LEAKAGE_PREVENTION_PROTOCOL.md)。
4. [八家族三seed神经流程与完整模型](neural/README.md)、[三阶段独立审计](neural-independent-audit/README.md)。
5. [代表性SVM真实C/D/H/I对抗运行](adversarial-audit/README.md)、`leakage-core/`有故意失败夹具与真实命令。
6. [黄芪／天麻各自四来源组试验](auxiliary/README.md)，最新用户声明另存覆盖层，批次数仍unknown。缺模态不删除类别凑融合。
7. [源内候选峰](source-peaks/README.md)、[冻结模型源内梯度解释](source-attributions/README.md)、[量子化学实际计算与NOT_RUN项](quantum/README.md)。

## 时间、锁与失败记录

`state-audit/`保存起始Git／PR状态；`command_invocations.jsonl`及各组件事件日志记录实际命令时间。主神经source与outer分两次root放行；辅助任务采用各自独立gate，不能借主神经授权越过锁。根`selection_lock.json`、`fit_scope_manifest.json`和`leakage_audit.json`是**组件完成后的汇总索引**，不冒称事前总预注册或外部执行认证。各组件的实际原锁、预测SHA与放行时间均保留。

`auxiliary/pre_fit_boundary_v1/`是0研究拟合时发现的评分锁缺口；`auxiliary/source_inference_failure_v2/`是首个SVM管线完成源scaler／SVC拟合后，在7条查询尾batch一致性检查触发的实现错误，尚未读inner真值或外测。各自封存后新版本重跑，不吞失败。图表导出的失败／被替代渲染版本另在`paper-assets-export-attempts/`，不是模型失败或新的实测样品。

## 复现和交付检查

各组件README给出新输出目录的复现命令与固定环境，不复用旧授权或覆盖目录。`scripts/finalize_upgrade_delivery.py`仅复核组件SHA、起始Git历史材料字节、实际OOF计数与指标，不拟合模型。`delivery_validation.json`记录最终结果，根`files_sha256.json`覆盖本目录交付文件并排除自身与Python字节码。

旧343项和随后358项项目测试日志均原样保留；最终冻结源码后的项目测试另存`project-tests-release.log`。Git只推研究分支，不合并main。发布后的Git HEAD与远程main核验会在最终聊天交付，不把递归包含自身提交号当内容SHA。
