# 跨植物预训练冻结证据归档

已完成全部评分和独立完整重放后，root放行本次无损归档。3,624个原文件／272,334,058字节压缩为7个ZIP／107,771,275字节，最大17,995,363字节；原文件仍在本地原路径，没有移动、删除或改写。

范围仅包含`neural/source_models/`、`neural/outer_models/`、`empirical/prepared/fits/`，覆盖240条主要source管线、60条主要outer管线及240条小预算实证管线的全部权重、训练摘要、fit范围、批次日程／轨迹、推理审计与锁。输入数组、协议、主选择／预测／评分锁和汇总结果仍按原目录直接交付。

`archive_manifest.json`记录每个源路径、字节数和SHA256及对应归档成员，同时记录每个归档的字节数和SHA。manifest SHA：`e0e76d74f5b142f64c01fcd9301644354a0e99aee0be07671e28ffcfc10c8fb0`。

从仓库根目录运行：

```bash
python scripts/archive_cross_plant_transfer_evidence_20261031.py verify --archive-dir reports/cross-plant-deep-transfer-20261031-v1/evidence-archives-v1
python scripts/archive_cross_plant_transfer_evidence_20261031.py restore --archive-dir reports/cross-plant-deep-transfer-20261031-v1/evidence-archives-v1
```

已有同字节文件会跳过；不同文件、路径穿越、符号链接、范围／归档／解码摘要变化会拒绝，不覆盖原件。旧archive helper为固定SHA的不可变依赖，不需要重写历史脚本。

真实新建临时目录的3,624个文件已完整恢复并再次逐路径SHA复核，幂等恢复亦通过。`cross_plant_transfer.verify_run`、`verify_selected`及`verify_predictions_before_truth`全部通过；实证`verify_prediction_lock`通过，全部540条管线的权重、训练／批次证据及锁保持不变。详见[实际执行记录](../delivery-preflight/archive_execution.json)。本交付不新增训练或评分。

实证诊断的冻结manifest保留原工作区三个绝对主清单路径；本次原生检查仍验证这些原路径，另独立证明恢复位置的三个主清单字节相同。主实验验证直接使用恢复位置；不要把本次验证描述成已消除实证快照的所有绝对路径依赖。迁移工作区时应据冻结SHA定位相同主文件，不改写旧快照冒充原运行。

本轮只涉及用户提供的酸枣仁／黄芪／天麻与CC BY 4.0人参观测；水果、叶片及其未知许可衍生数据不在归档中。manifest可信SHA须由上级交付索引绑定，本工具不是数字签名或恶意并发环境的OS沙箱。
