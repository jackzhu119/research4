# 获批酸枣仁实验执行证据（2026-10-31）

**执行后用户补充确认：每个产地目前只有一个来源批次。以下结果是用户声明独立样品上的匿名产地组单批次内部识别，产地与来源批次不可分离；没有真实跨批次验证。** 原执行文本保留当时状态，最新解释以[stage_summary.json](stage_summary.json)的`post_run_batch_declaration`为准。

本轮排除Gaussian。只使用[当前7份编号版](../../../data/CURRENT_DATASET.md)，共855份用户声明独立样品；真实地名、制样／测量日期及仪器映射仍未提供。

## 实际执行范围

- 13条真实分类路径、65个外层折模型、每路径855份OOF预测，共11,115份预测；所有方法共用5外折／2内折，CARS-style内部CV显式为2。
- 主比较范围固定为0.1—1.5 THz工程候选区（612点）；全导出范围（2578点）另列敏感性，不根据测试分数改称最优有效频段。
- ZY01=350、ZY02=5、ZY03—ZY07各100；ZY02每个外折测试仅1份，不能用复制或重复扫描增加独立样本量。
- 100项单元／集成工程检查通过；合成分类迁移另有21组评价、630份样品预测，真实药材样品数为0，不能作为真实校准收益。

## 全部真实路径结果

| 输入／方法 | 准确率（%） | 平衡准确率（%） | 宏F1 |
| --- | ---: | ---: | ---: |
| 候选区α／多数类 | 40.94 | 14.29 | 0.0830 |
| 候选区α／SVM | 88.54 | 78.16 | 0.7878 |
| 候选区α／PLS-DA | 81.87 | 64.39 | 0.6428 |
| 候选区α／MC20＋SVM | 90.41 | 83.67 | 0.8092 |
| 候选区α／CARS-style＋SVM | 79.53 | 70.69 | 0.6696 |
| 候选区n／SVM | 75.91 | 64.88 | 0.6124 |
| 候选区n／PLS-DA | 63.74 | 42.24 | 0.3904 |
| 候选区n＋α／SVM | 83.86 | 72.86 | 0.7136 |
| 候选区n＋α／PLS-DA | 78.95 | 60.92 | 0.6016 |
| 候选区α／MC100＋SVM | 88.07 | 81.02 | 0.7766 |
| 候选区α／重复MC＋SVM | 90.29 | 80.82 | 0.7923 |
| 全导出α／SVM（敏感性） | 99.77 | 94.29 | 0.9639 |
| 全导出α／PLS-DA（敏感性） | 100.00 | 100.00 | 1.0000 |

候选区MC20相对全谱SVM数值改善；重复稳定筛选未超过MC20，n＋α拼接也未超过α单独输入。全部正负结果均保留，这些差值未作显著性或置信区间推断。全导出α＋PLS-DA的100%来自实际保存预测，但不证明跨批次鉴别、化学因果或频段物理有效性。

完整α数据中有439,027个负值单元格，工程候选区负α单元格为0；原值保留，没有按结果删除样品或修正数值。ZY02还包含2份1.0 mm与3份1.5 mm压片，其他组为1.5 mm，应结合制样、仪器和来源批次记录解释。

## 图和可检查证据

- [13路径指标比较PNG](figures/metric_comparison.png)／[PDF](figures/metric_comparison.pdf)：主宏F1、辅平衡准确率，坐标固定0—1，橙色仅表示全导出敏感性。
- [预定α-SVM混淆矩阵PNG](figures/confusion_alpha_svm.png)／[PDF](figures/confusion_alpha_svm.pdf)。
- [预定n＋α-SVM混淆矩阵PNG](figures/confusion_concat_svm.png)／[PDF](figures/confusion_concat_svm.pdf)。两矩阵按预定模型展示，单元格为原计数及行百分比，明确ZY02分母5。
- [独立复算](independent-audit/independent_recalculation.json)、[总体CSV](independent-audit/pooled_metrics.csv)、[逐类召回CSV](independent-audit/class_recalls.csv)、[全部差值](independent-audit/comparison_deltas.csv)。复算与保存指标最大绝对差为0。
- [SVC平票策略诊断](independent-audit/tie_policy_diagnostics.json)：8条SVC路径的`predict`与OVR分数argmax共31处不同；主结果保留冻结的`predict`，未据此改模型重跑。
- [数值／配对质控](data-preparation/quality_report.json)、[逐记录清单](data-preparation/record_manifest.csv)、[候选区生成记录](datasets/generation.json)。
- [冻结协议](experiments/frozen_protocol.json)、[实际运行命令及耗时](experiments/runs_manifest.json)、源码快照`experiments/execution_code/`及原始日志`experiments/logs/`。
- [100项检查摘要](tests_summary.json)和[完整日志](all-tests.log)。
- `synthetic-transfer-results/`与[输入范围说明](synthetic-transfer-inputs/input_scope.json)仅为合成工程验证。原`status.json`中的protocol scope不表示使用了真实药材。

## 归档与复现边界

本目录没有归档`.cache`、joblib／pickle模型缓存或派生NPZ输入；约2.2 GB缓存留在被Git忽略的outputs中。派生输入hash、原始7表hash及元数据准备时hash均记录，原始7份编号版字节未改。`execution_helpers/`保留实际执行辅助脚本，其中绝对路径是原运行证据；可移植入口仍以仓库scripts中的正式命令为准。

`independent-audit/outer_fold_selection_frequency.csv`是5个重叠外层训练集合之间的选点频率，不能冒充重复筛选内部10次重采样频率、p值、FDR控制或独立批次稳定性。

本地冻结协议不属于外部预注册。后续需新增各产地独立批次及可核验采样／制样记录，才能检验跨批次泛化与小样本校准。
