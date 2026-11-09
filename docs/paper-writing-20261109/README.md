# 论文证据目录

整理日期2026-11-09。总入口：[PAPER_WRITING_START_HERE.md](../../PAPER_WRITING_START_HERE.md)。

便捷阅读：[实验流程图](EXPERIMENT_FLOW.md) · [结果Excel工作簿](../../reports/paper-evidence-20261109/RESEARCH4_PAPER_RESULTS_20261109.xlsx)。

| 阶段 | 阅读文件 | 原始结果／机器索引 | 使用边界 |
|---|---|---|---|
| 数据、种类更正、固定角色 | [数据与方法](DATA_AND_METHODS.md) | [数据与固定划分副本](../../reports/paper-evidence-20261109/sources/data_and_fixed_split/) | 1672用户声明独立样品，非独立植物学鉴定 |
| 教师算法与旧855阶段 | [历史及未完成项](HISTORY_QUANTUM_AND_NOT_RUN.md) | [历史/量子登记](../../reports/paper-evidence-20261109/tables/historical_quantum_experiment_register.csv) | 混合枣仁／酸枣仁；不转移准确率 |
| 更正后固定339预测池 | [模型结果](MODEL_RESULTS_AND_CONCLUSIONS.md) | [原报告](../SPECIES_REVISION_RESULTS_20261108.md) | 内部已曝光预测角色；完整重训不等于K-shot |
| 225经典＋1080神经矩阵 | [模型结果](MODEL_RESULTS_AND_CONCLUSIONS.md) | [最终288行无校准读数](../../reports/paper-evidence-20261109/sources/legacy_final/summary.json) | 全校准原始记录另归档，不冒充此表已汇总 |
| V4匹配特征＋表示／预算矩阵 | [模型结果](MODEL_RESULTS_AND_CONCLUSIONS.md) | [122行当前最终设置](../../reports/paper-evidence-20261109/sources/current_v4/summary.json) | 15经典任务＋315神经任务，仅TRAIN开发 |
| 另一主机B新增研究 | [B结果](PARALLEL_HOST_B_RESULTS.md) | [B实验登记](../../reports/paper-evidence-20261109/tables/parallel_b_experiment_register.csv) | 按该分支实际样品／协议单列 |
| PySCF、小分子、Spinosin及皂苷 | [量子与未完成项](HISTORY_QUANTUM_AND_NOT_RUN.md) | [真实历史/量子副本](../../reports/paper-evidence-20261109/sources/history_quantum/) | 频率、虚频、曲率、最低点资格分别核查 |
| 云端收尾与可还原科研证据 | [实验审计](../RESEARCH4_EXPERIMENT_AUDIT.md) | [逐任务台账](../../reports/paper-evidence-20261109/tables/scientific_job_register.csv)、[无损归档](../../reports/paper-evidence-20261109/archives/README.md) | 不删失败记录，不把预算余量当重跑理由 |
| 论文组织 | [大纲](../RESEARCH4_PAPER_OUTLINE.md)、[主计划](../RESEARCH4_PAPER_C_MASTER_PLAN.md) | [图表与图注](../../reports/paper-evidence-20261109/README.md) | 主线纯酸枣仁，外部药材与机制为辅助 |

写作优先采用同协议表，不按跨协议最高分拼排名。`sources/`中的副本保留原始字节与来源哈希，
`tables/`是可导入Excel或统计软件的表，`figures/`提供PNG／PDF／SVG，`archives/`用于逐任务复核。
本索引不意味着所有设想都已执行；未完成事项在专项报告中明列。
