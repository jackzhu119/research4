# 方案C论文图表证据

本目录仅从已保存的真实预测与真实标签导出图表，没有新增训练或重新选择参数。酸枣仁是主任务，外部植物及迁移是辅助。

[论文模型表Excel](paper_c_model_tables.xlsx)含历史五阶段、全部20角色、预处理与组合负对照、逐类指标、R2混淆计数及五折参数。另含同855样品和相同划分的新SVM／LDA／MLP统一对照及R2条件消融；不同搜索预算与表示不称严格单因素因果。强64–32 MLP按预定seed42列入主表，全部42／43／44种子及均值、样本标准差另列附表；不挑最高种子。强MLP旧架构来自已曝光的513源研究，当前30个内层场景仅作诊断，不声称完整架构选择的无偏嵌套确认；200轮加100次源更新的配对改对／改错另列。171份固定query的校准结果只放Calibration171_only独立工作表，未与855 OOF混排；100次支持抽样是同一query上的重叠重复。

| 图 | 内容 |
| --- | --- |
| [原五阶段准确率与Macro-F1](historical_stage_accuracy_macro_f1.png) | [PDF](historical_stage_accuracy_macro_f1.pdf) |
| [R2计数和行归一混淆矩阵；ZY02仅5份](r2_confusion_counts_and_recall.png) | [PDF](r2_confusion_counts_and_recall.pdf) |
| [R2每折K、频点和描述性稳定性](r2_selected_frequency_count_and_stability.png) | [PDF](r2_selected_frequency_count_and_stability.pdf) |
| [70%主任务、20%外部任务、10%迁移校准及证据门禁流程](paper_c_experimental_flow.png) | [PDF](paper_c_experimental_flow.pdf) |
| [统一分类器对照](unified_classifier_comparison.png) | [PDF](unified_classifier_comparison.pdf) |
| [R2条件消融](r2_conditional_ablation.png) | [PDF](r2_conditional_ablation.pdf) |

图中百分比均明确标为准确率，Macro-F1使用0—1尺度；不截断纵轴夸大差距。没有按外测最高值回填选择。所有855份与历史外测此前已曝光，且每产地只有一批，因此是回顾性批次内验证，不能代表真正跨批次或新独立确认。

流程图中的比例是工作量规划，不是证据权重或论文固定文字比例。公开THz数据及黄芪缺少的样品身份／标签／单位信息均是门禁；虚线不是已完成迁移实验。

频率轴按当前酸枣仁输入的原坐标标THz；频点选择不等于化学峰归属，也不推断其他植物或仪器的物理量兼容。

再生：`.runtime/bin/python scripts/export_paper_c_evidence.py`；有已完成新对照时增加`--benchmark-dir <completed-result-directory>`，强MLP增加`--strong-mlp-dir <completed-strong-mlp-directory>`，单列校准增加`--calibration-dir <completed-calibration-directory>`。输入SHA和独立重算记录见[验证记录](export_validation.json)。
