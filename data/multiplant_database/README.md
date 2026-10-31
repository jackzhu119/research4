# 多药材光谱数据库

原始Excel不改写。本目录保存独立物种／表示的数值接口与血缘，不把四种表示计为四份样品。

- [完整数据审计](../../docs/MULTIPLANT_DATA_AUDIT.md)
- [统一清单](dataset_inventory.csv)
- [样品×表示血缘](metadata/sample_lineage.csv)：6399行不是6399独立样品
- [酸枣仁四表示855条配对](zaoren/alignment.json)及[原表数值行绑定](zaoren/modality_alignment.csv)
- [逐工作簿质量](data_quality/source_workbooks.json)、[上传别名／旧坏版](data_quality/incoming_aliases.json)、[小麦平均曲线](data_quality/wheat_workbooks.json)
- [科学资格](metadata/readiness.json)：酸枣仁EXPLORATORY；黄芪／天麻物理身份未知，独立验证BLOCKED；小麦平均曲线BLOCKED；独立确认集MISSING。

各模态NPZ的`X`是数值谱，`axis`是原坐标，`record_ids`只用于血缘；`labels`仅酸枣仁已有匿名组，`filename_groups`不是已核验生长产地。禁止把这些ID／路径／厚度拼入特征。

`zaoren/aligned_modalities.npz`按同一855个record顺序提供`X_alpha/X_n`各612点、`X_time`6750点、`X_frequency`1001点及各自axis；`physical_sample_ids`来自用户声明，不是采样日志认证。原全轴α/n及各表示另存。未知物种、仪器、批次和单位不填写为“已确认”。

参考曲线只留档、不作为新增样品；损坏／空模态不填补。不得用黄芪n的303条或天麻时域202条静默替代完整四分类、不得直接和不同样本集合相减解释融合收益。
