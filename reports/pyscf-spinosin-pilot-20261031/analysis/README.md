# PySCF实际结果分析与机制边界

本目录为只读后处理，新增SCF、梯度或Hessian计算均为0。所有图来自已结束阶段的真实日志或坐标；不会把水的频率当作斯皮诺素频率。

水分子实际通过检查的频率记录：6行（不同后端复测不是新分子）。斯皮诺素通过全部频率检查的记录：0行；其中0.1—1.5 THz：0个模式。

斯皮诺素频率解释状态：**BLOCKED_NO_VALIDATED_DFT_MODES**。未通过优化／驻点／完整模式／正频率检查时，比较表的理论频率和差值保持空值；不使用水或合成测试数值填空。

`water_validated_frequencies.csv`为实际技术测试；`phase_status_and_resources.csv`、`actual_optimization_steps.csv`分别保留DONE／PARTIAL阶段与实际梯度轨迹；`experiment_theory_pairs.csv`列出全部实际实验候选及可用的理论模式笛卡尔对照，不按峰位接近挑构象。频率位置图使用等高标记，**不代表IR强度**，IR始终NOT_COMPUTED。模式箭头／动画仅作任意幅度显示。

孤立分子的分子内振动不包含晶格、复杂基质、分子间氢键和集体振动。当前没有纯品THz和化学含量证据，任何频率接近都不能确认化学归属。本次不修改分类模型，也不声称提高96.0234%准确率。

`summary.json`绑定输入摘要和完整条件；`files_sha256.json`覆盖本目录其它产物。
