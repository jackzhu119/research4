# 外部植物预训练与酸枣仁识别：2026-10-31 v1

根据用户本次要求，检验新增植物训练信息是否帮助酸枣仁七分类。原96.0234%MC-UVE＋SVM及历史证据不变。本目录全部为新增产物；目标855条仍是历史曝光开发数据。

- [冻结研究设计与控制组](../../docs/CROSS_PLANT_DEEP_TRANSFER_PROTOCOL_20261031.md)
- [完整结果与论文含义](../../docs/CROSS_PLANT_DEEP_TRANSFER_REPORT_20261031.md)
- [15表主工作簿、2表补充工作簿、CSV及论文图](comparison/README.md)
- [实际数据资格及阻塞材料](data-audit/README.md)
- [独立数据／模型审计](independent-audit/)
- [真实C／D／H诊断及全部负结果](empirical/EXECUTED_RESULTS.md)
- [实际source训练成本](source-execution/README.md)及[outer成本](outer-execution/README.md)
- [实际预训练覆盖](pretraining-coverage/summary.json)
- [502项项目测试](project-validation/validation.json)
- [完整证据归档／恢复验证](delivery-preflight/EXECUTED_RESULTS.md)

实际240个source＋60个outer管线均已完成，另有240个小预算诊断管线；全部原生预测和实际优化输入独立重放通过。三种子平均Accuracy：从头训练87.21%，仅目标source额外训练87.37%，人参预训练86.43%，多植物预训练85.65%。后两者均未提高准确率或Macro-F1。预定source-policy为87.95%，只选Aextra／A，因此不能把这部分收益归因外部数据。

外部850条观测用于独立任务头，不添加为ZY标签。物理单位、采集仪器及样品祖先未核实，本轮仅为明确标记的工程形状迁移探索。酸枣仁855条已历史曝光、每来源仅一批；所有新神经结果仍低于历史MC-UVE＋SVM96.0234%，不替换强基线，也不称新批次确认。
