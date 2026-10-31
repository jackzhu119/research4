# 论文方法分类适配：运行证据

结论及使用边界见[完整中文报告](../../../docs/PAPER_FEWSHOT_CLASSIFICATION_REPORT.md)。一阶MAML从自身64.7173%提高到67.5458%，仍低于同一预测组SVM的88.8889%；其他神经方法未通过开发选择。全部为单批次回顾性探索，ZY01—ZY07是匿名组。

优先打开[预测与标签对照Excel](论文方法试验_预测与标签对照.xlsx)。其中720条随机展示记录含311条错误，未筛除；MAML逐样品表为3种子×171份，分别汇总100次支持抽样。

| 文件 | 用途 |
| --- | --- |
| `protocol_snapshot.json`、`candidate_grid.json` | 先固定的协议与31候选 |
| `run_manifest.json`、`selection_lock.json` | 输入／代码hash、开发选择与回退；先锁定后评分 |
| `all_development_metrics.json` | 2790次开发评价 |
| `final_metrics.json`、`summary.json` | 3600次最终评价及12行汇总，含零支持／正式选择／诊断候选 |
| `development/*/seed_*/training.json` | 18个开发上下文的源训练与任务记录 |
| `final/seed_*/training.json`、`predictions.csv.gz` | 3个最终上下文及615600条预测；模型hash和100次支持抽样均可追溯 |
| `per_seed_summary.json`、`fomaml_class_diagnostic.json` | 全部训练种子、逐类改善和下降 |
| `random_predictions.csv`、`random_query_ids.json` | 预先固定的展示编号，包含错误 |
| `random_examples.csv`、`fomaml_per_sample.csv`、`per_seed_summary.csv` | Excel对应的可读表 |
| `paper-fewshot-pilot-review-20261031.json` | 原论文逐页证据、实际分类ProtoNet／一阶MAML实现审读 |
| `independent_paper_pilot_audit.json`、同名`.py` | 独立重算指标、隔离关系、任务清单和权重hash；3806593项检查通过 |
| `arithmetic_and_lineage_audit.json`、`paper-fewshot-pilot-audit-20261031.py` | 运行方数值复核及脚本 |
| `table_export_validation.json` | 表格导出与数值核验 |
| `all-tests.log` | 本轮172项项目测试通过 |
| `figures/` | 由汇总JSON直接生成的PNG、PDF及绘图脚本 |
| `files_sha256.json` | 本目录逐文件校验；不包含该索引自身 |

源权重保存在[模型目录](../../../models/paper_fewshot_20261031/README.md)，不在证据目录重复保存。作者网络与本轮工程网络不同，不是原论文逐项复现。三个种子与100次重叠支持抽样不是300个独立批次，不据此计算独立批次置信区间。
