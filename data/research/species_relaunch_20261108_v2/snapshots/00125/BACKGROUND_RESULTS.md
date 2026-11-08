# 后台训练内开发结果

本汇总只读取后台作业输出，不读取主校准集或主预测集。分数来自999份主训练样品内部的开发折；它们不是新的独立外部测试，也不能证明跨批次校准。

已完成并通过结果文件哈希核验：125个作业；未启动：1180；已启动未完成：0；错误：0。

每个重复先合并五个互斥开发折的逐样品预测，再重新计算正确数、总数、宏F1和平衡准确率。缺折结果只显示为未完成，不参与候选推荐。模型种子和支持集重抽样均不增加独立样品数。

## 训练内候选

| 路线 | 任务 | 候选 | 训练内OOF准确率 | 宏F1 | 唯一样品数 |
| --- | --- | --- | ---: | ---: | ---: |
| classic | acid_origins | n_alpha/0/R2/none/0 | 92.33% | 0.9235 | 300 |
| classic | huangqi_origins | alpha/0/R2/none/0 | 82.16% | 0.8215 | 241 |
| neural | acid_origins | alpha_transfer_128/p12_e96_full/120/Aextra/weak_lambda1000_blend005/10 | 70.22% | 0.7026 | 300 |

这些是当前已完成候选中的训练内结果；若预定矩阵尚有未运行或未完成部分，推荐是临时的。确认评价须另行锁定配置。不同数据量、频点数或划分的经典与神经路线不直接相减作为匹配提升；共享SVM缓存只是参考，不算新增独立重复。

## 跨物种配对对照

| 任务 | 配置 | 比较 | 平均准确率差（百分点） | 宏F1差 |
| --- | --- | --- | ---: | ---: |
| acid_origins | p12_e96_full/120步/legacy_lambda1/K1 | B-A | -0.161 | -0.0046 |
| acid_origins | p12_e96_full/120步/legacy_lambda1/K3 | B-A | -1.233 | -0.0120 |
| acid_origins | p12_e96_full/120步/legacy_lambda1/K5 | B-A | -1.689 | -0.0179 |
| acid_origins | p12_e96_full/120步/legacy_lambda1/K10 | B-A | -2.333 | -0.0247 |
| acid_origins | p12_e96_full/120步/source_only/K0 | B-A | -3.111 | -0.0301 |
| acid_origins | p12_e96_full/120步/weak_lambda1000_blend005/K1 | B-A | -3.128 | -0.0304 |
| acid_origins | p12_e96_full/120步/weak_lambda1000_blend005/K3 | B-A | -3.206 | -0.0312 |
| acid_origins | p12_e96_full/120步/weak_lambda1000_blend005/K5 | B-A | -3.250 | -0.0317 |
| acid_origins | p12_e96_full/120步/weak_lambda1000_blend005/K10 | B-A | -3.222 | -0.0314 |
| acid_origins | p12_e96_full/120步/legacy_lambda1/K1 | B-Aextra | -1.789 | -0.0206 |
| acid_origins | p12_e96_full/120步/legacy_lambda1/K3 | B-Aextra | -2.839 | -0.0300 |
| acid_origins | p12_e96_full/120步/legacy_lambda1/K5 | B-Aextra | -2.661 | -0.0291 |
| acid_origins | p12_e96_full/120步/legacy_lambda1/K10 | B-Aextra | -2.889 | -0.0311 |
| acid_origins | p12_e96_full/120步/source_only/K0 | B-Aextra | -5.000 | -0.0499 |
| acid_origins | p12_e96_full/120步/weak_lambda1000_blend005/K1 | B-Aextra | -5.011 | -0.0500 |
| acid_origins | p12_e96_full/120步/weak_lambda1000_blend005/K3 | B-Aextra | -5.039 | -0.0503 |
| acid_origins | p12_e96_full/120步/weak_lambda1000_blend005/K5 | B-Aextra | -5.072 | -0.0506 |
| acid_origins | p12_e96_full/120步/weak_lambda1000_blend005/K10 | B-Aextra | -5.111 | -0.0510 |
| acid_origins | p12_e96_full/120步/legacy_lambda1/K1 | B-matched_SVM | -1.311 | -0.0137 |
| acid_origins | p12_e96_full/120步/legacy_lambda1/K3 | B-matched_SVM | -0.633 | -0.0058 |
| acid_origins | p12_e96_full/120步/legacy_lambda1/K5 | B-matched_SVM | -0.589 | -0.0062 |
| acid_origins | p12_e96_full/120步/legacy_lambda1/K10 | B-matched_SVM | -1.222 | -0.0122 |
| acid_origins | p12_e96_full/120步/source_only/K0 | B-matched_SVM | -5.222 | -0.0496 |
| acid_origins | p12_e96_full/120步/weak_lambda1000_blend005/K1 | B-matched_SVM | -5.233 | -0.0497 |
| acid_origins | p12_e96_full/120步/weak_lambda1000_blend005/K3 | B-matched_SVM | -5.239 | -0.0498 |
| acid_origins | p12_e96_full/120步/weak_lambda1000_blend005/K5 | B-matched_SVM | -5.239 | -0.0498 |
| acid_origins | p12_e96_full/120步/weak_lambda1000_blend005/K10 | B-matched_SVM | -5.222 | -0.0496 |

所有完整及不完整分组、正负增益、逐类混淆矩阵和重复统计保留在 `results_summary.json`；精简分组数值见 `results_summary.csv`。

来源为各作业的 `job_manifest.json`、`complete.json`、`summary.json` 和 `validation_predictions.csv.gz`。本脚本不会改动工作器检查点或发布到网络。
