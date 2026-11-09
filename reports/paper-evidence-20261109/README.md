# 论文证据与图表包

本目录只汇编已完成实验，不产生新训练分数。人类阅读入口：[论文写作总入口](../../PAPER_WRITING_START_HERE.md)。

## 文件布局

| 目录／文件 | 内容 |
|---|---|
| `sources/current_v4/` | 新330任务协议、最终122设置、实际闭合状态与原Git备份回执 |
| `sources/legacy_final/` | 原1305矩阵、1080神经最终288行无校准读数、旧/续跑闭合状态 |
| `sources/data_and_fixed_split/` | 登记表、1672主角色、特征质量检查、固定预测与历史更正报告 |
| `sources/history_quantum/` | 固定提交的历史优化、元学习、量子结果及来源证明 |
| `sources/parallel_b/` | 另一主机B的专项结果及来源证明 |
| `tables/scientific_job_register.csv` | 全部1635成功科学任务、命令、状态、成本、输出与回执哈希 |
| `tables/all_final_model_settings.csv` | 122当前设置＋288旧神经source_only设置；协议不能混排 |
| `tables/model_protocol_register.csv` | 按家族登记的模型实验、分数、限制与固定来源 |
| `tables/historical_quantum_experiment_register.csv` | 历史、量子、未完成思路与外部数据可用性 |
| `tables/parallel_b_experiment_register.csv` | B分支新增实验登记 |
| `tables/current_per_class_by_repeat.csv` | 当前各设置逐重复逐类precision/recall/F1 |
| `tables/current_confusion_counts_by_repeat.csv` | 当前各设置逐重复混淆计数长表 |
| `tables/feature_and_parameter_selection.csv` | 逐外折K、C、γ、n/α入选数及局部计时 |
| `tables/paired_classical_effects.csv` | 配对改对/改错、物理ID聚类bootstrap区间 |
| `tables/branch_inventory.csv` | 本次检查远端分支与不可变提交 |
| `figures/` | 真实结果生成的PNG、PDF和SVG |
| `archives/` | 逐任务小结果与预测的无损分卷、清单、恢复工具 |
| `source_provenance.json` | 来源路径/提交/哈希；Git对象读取不冒充当前运行 |
| `RESEARCH4_PAPER_RESULTS_20261109.xlsx` | 多工作表便捷副本，保留CSV原文本，不混排不同协议 |
| `bundle_manifest.json` / `bundle_verification.json` | 本次汇编完整性与独立指标复核 |

## 图注，可在论文中按协议改写

1. **01_classical_primary_comparison**：当前300酸枣仁TRAIN中3次完整五折外层预测的Accuracy与Macro-F1均值。
   所有图中K均由内层选择；样品重现不是新增实物。MLP、LDA与MC家族都使用同一source/query划分。
2. **02_feature_budget_ablation**：预设K31/62/123/184/245的固定K消融。曲线用于分析变量预算与表示交互，
   不根据外折曲线挑K。折射率SNV的下降及融合随K改变的正负结果均保留。
3. **03_primary_confusion_by_repeat**：MC20主流程三次完整OOF的混淆矩阵，每面板300实物，
   类顺序ZY03—ZY07；不能把三幅图当900份独立样品。更易混淆的ZY07→ZY04需在讨论中说明。
4. **04_transfer_representation_budget**：相同240来源训练／60外折查询下，128/512点与144/192轮的
   目标从头和黄芪＋天麻预训练对照。144轮为576更新，192轮为768更新；同轮或同步数不等于同FLOPs。

没有理论IR强度图、没有新DFT峰归属图，也没有把不完整Hessian当完整谱。
量子原有实测计算表/图的科学资格另见专项报告，不从模型关注区域倒推化学成分。

## 核验和重建

仅核查本包，标准Python即可：

```bash
python scripts/build_paper_evidence_bundle_20261109.py --verify
```

重新生成本包表图需原有完成工作区与matplotlib：

```bash
python scripts/build_paper_evidence_bundle_20261109.py --build \
  --legacy /workspace/research4-relaunch \
  --runtime /workspace/research4-runtime-20261108
```

`--build`重新读取保存预测/指标，不重新拟合。完整训练的实际命令位于任务台账，
需按原固定提交的依赖、源路径映射和输入锁在新的独立结果目录重现，不能覆盖本次科学封存。
归档恢复命令及分卷校验以`archives/README.md`为准。
