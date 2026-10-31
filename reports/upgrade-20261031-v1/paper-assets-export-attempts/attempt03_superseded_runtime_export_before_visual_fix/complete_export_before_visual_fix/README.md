# Research4升级：真实表图与范围

[Excel证据表](Research4升级_多表示神经与防泄漏证据.xlsx)含15张表：9神经角色逐种子及均值/样本SD、135外折、锁定架构/profile、逐类P/R/F1、混淆、23085逐样品OOF、R2配对改对/伤害、成本和数据资格。

| 图 | 内容 |
| --- | --- |
| [Accuracy与Macro-F1](model_accuracy_macro_f1.png) | [PDF](model_accuracy_macro_f1.pdf)；柱为固定seed42，空点为42/43/44均值±样本SD |
| [R2与内层选择策略混淆](r2_vs_inner_selected_confusion.png) | [PDF](r2_vs_inner_selected_confusion.pdf)；各855份，ZY02仅5份突出显示 |
| [源标签置乱诊断](source_label_permutation_diagnostic.png) | [PDF](source_label_permutation_diagnostic.pdf)；10条件置乱、同171与多数类40.94%，不按1/7验收 |
| [四植物科学可用性](multiplant_scientific_readiness.png) | [PDF](multiplant_scientific_readiness.pdf)；可读不等于独立验证合格 |
| [受守卫研究流程](guarded_research_workflow.png) | [PDF](guarded_research_workflow.pdf)；DFT与外部阻塞范围明确 |

所有新指标从锁定预测和独立评分真值重算；导出不训练模型，不改变旧R2的821/855。`inner_selected_neural`是内层选择的主策略，各家族为预定义诊断；策略引用已有权重不增加真实样品或模型数。历史855份已曝光，每类一批，科学状态为EXPLORATORY，不是新独立确认。

三种子SD只反映初始化敏感性，不能当生物学置信区间。SE是通道注意力，不是化学浓度或频率峰归属；四表示属于相关测量链，不能称四个独立传感器。两profile联合改变多个因素，不是全因素消融。

黄芪403／天麻405已获后续用户独立性声明，四来源组内部分类为EXPLORATORY；地名／批次／单位仍未外核、跨植物迁移BLOCKED，小麦平均曲线任务BLOCKED；三候选分子的优化/频率/IR均NOT_RUN，水只作软件技术测试，没有拼接虚构化学峰。源内实验峰候选见[单独source-only峰分析](../source-peaks/README.md)，不能回填本版本模型。

再生需真实神经scoring完成，使用新输出目录：

```bash
.runtime/bin/python scripts/export_upgrade_research_assets.py --neural-dir reports/upgrade-20261031-v1/neural --output outputs/upgrade-paper-assets-reproduction-new
```

输入/输出SHA及数值核对见`export_validation.json`和`files_sha256.json`。
