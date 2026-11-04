# 最大96轮的单变量预算消融 v2

状态：**DONE_DEVELOPMENT_ONLY**。30个source／45个outer管线、全部固定三种子
及独立评分完成。真实均值A87.0565%、Aextra＋L2-SP86.3548%、黄芪／天麻
预训练＋L2-SP85.3801%；外部B仍低于同预算A及额外计算控制Aextra的准确率。

[完整两预算报告](../../docs/CONSERVATIVE_CROSS_PLANT_BUDGET_ABLATION_20261104.md)。
只改变最大轮数48→96和独立资源上限，数据／种子／λ／预训练／选轮规则不变。
原因是首轮source选轮接近原上界，不是根据outer负分另调λ或挑种子。
30条source前48轮轨迹及scaler／ID／输入严格复核；source新选91／86／92／89／92。

`protocol.json`和`registration.json`先于训练；`source_prefix_budget_ablation_audit.json`
及`source_epoch_lock.json`先于outer；`outer_prediction_lock.json`先于单独评分。
最终状态以`completion.json`／`worker_result.json`为准，`progress.json`仅是最后
拟合事件快照，不代表当前仍运行。

主要产物：

- `two_budget_model_comparison.csv`及`epoch_budget_changes.csv`：两预算完整对照。
- `model_comparison.csv`、`metrics.json`、`per_class_metrics.csv`、`confusion_matrix_counts.csv`。
- `all_outer_predictions.csv.gz`：三种子、所有arm的原始预测与真实标签。
- `model_metrics.png/pdf`、`confusion_seed42.png/pdf`、`source_epoch_curves.png/pdf`、
  `epoch_budget_ablation.png/pdf`：实际数值图，非示意或合成结果。
- `native_model_replay.json`：45个保存权重＋scaler重放7695预测，最大logit差0。
- `resource_ledger.json`：456.892275实际CPU秒，900秒上限未触发。
- `source_models/`／`outer_models/`：完整原始模型、batch来源、scaler及过程日志。

855条是同一曝光开发队列，三种子及两预算不增加独立N。α单位、仪器、批次
信息仍有缺口；SDF和文献化学数据未加入模型。不得将本轮同历史96.0234%
MC-UVE＋SVM拼成同一优化阶梯或称新的独立确认。

复现到新目录：

```bash
.runtime/bin/python scripts/run_conservative_cross_plant_pilot_20261104_v2.py register --output outputs/conservative-budget96-reproduction
.runtime/bin/python scripts/run_conservative_cross_plant_pilot_20261104_v2.py supervise --output outputs/conservative-budget96-reproduction
```

需要保持v1封存source参考路径可用，prefix检查会按原SHA确认它们；若来自完整
归档，请先按附带索引恢复原路径。不要覆盖本轮目录或修改v1参考。
