# 已执行的保守跨植物开发试验 v1

状态：**DONE_DEVELOPMENT_ONLY**。真实30个source／45个outer管线及独立评分
完成，全部三种子、五外折、855条目标样品保留。本轮外部预训练未改善结果。

完整解释：[报告](../../docs/CONSERVATIVE_CROSS_PLANT_PILOT_20261104.md)。

| 方法 | Accuracy三seed均值 | Macro-F1 | Balanced Accuracy |
| --- | ---: | ---: | ---: |
| A scratch | 80.3509% | 0.678626 | 73.9592% |
| 目标source预训练＋L2-SP | 78.3236% | 0.651480 | 70.1701% |
| 黄芪＋天麻预训练＋L2-SP | 76.1404% | 0.630414 | 68.3333% |

这是同一曝光队列上的开发探索，不是外部独立确认，也不与历史96.0234%
MC-UVE＋SVM或旧96轮ResNet结果合并计算涨分。每种子的目标分母为855，
外部403＋405是复用既有观测，并非本轮新下载样品。

`protocol.json`／`registration.json`在训练前固定；`source_epoch_lock.json`
先于outer预测，`outer_prediction_lock.json`先于另进程真值评分。
`completion.json`和`worker_result.json`为最终状态；`progress.json`保留
最后一次拟合事件快照，其PARTIAL字段不是当前仍运行或失败的断言。

逐条预测、所有指标、种子、选轮数、源记录及批次轨迹、模型权重、scaler、
资源及依赖版本均保留。`native_model_replay.json`实际重放45个模型7695条
预测，logit最大差0，无新增fit。`source_models/`、`outer_models/`保存原始
过程证据，原始仓库数据和旧结果未改。

主要图为`model_metrics.png/pdf`及`confusion_seed42_layout2.png/pdf`。
最初`confusion_seed42.png/pdf`存在顶端标题相碰，已另画布局2；数值和原图
均保留，不覆盖原始文件。8项软件检查及独立审计见同级
`../cross-plant-conservative-pilot-20261104-preflight-v1/`。

复现使用新目录（禁止覆盖本轮）：

```bash
.runtime/bin/python scripts/run_conservative_cross_plant_pilot_20261104.py register --output outputs/conservative-transfer-reproduction
.runtime/bin/python scripts/run_conservative_cross_plant_pilot_20261104.py supervise --output outputs/conservative-transfer-reproduction
```

监督器设置单线程、累计600实际CPU秒、1800墙钟秒和1.5GiB best-effort RSS；
运行环境记录于`environment.json`。不使用GPU、DFT／分子输入或新的未标注谱。
