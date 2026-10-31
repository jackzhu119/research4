# 历史64–32强MLP：统一855样品协议的实际复核

本轮精确复用历史较强PyTorch MLP实现，而非新增sklearn单隐层MLP。原b006架构是在旧513份源数据的内部开发中选出；这些样品与本轮855份各外折存在历史重叠。因此本轮是**历史固定架构在相同外折重新训练的回顾性对照**，不能称包含此前全部架构选择过程的无偏嵌套确认。30个内层源上下文只作诊断，不再选择架构、轮数或种子。

模型在同一855份真实编号酸枣仁、612点α候选区间及旧5外折／2内折上重新训练。源预处理只在每个当前训练分区拟合，不用目标校准标签；所有种子42／43／44均报告。主表预先指定seed42，与本轮SVM／LDA／sklearnMLP参照一致，未挑最好种子。

| 配对方法 | seed42正确数／855 | Accuracy | Macro-F1 | Balanced Accuracy |
| --- | ---: | ---: | ---: | ---: |
| 原64–32 MLP，200个epoch | 768 | 89.8246% | 0.8191 | 79.6327% |
| 同一起点＋100次源监督更新 | 777 | 90.8772% | 0.8319 | 80.9184% |

200个基础epoch之后的100次是**源任务优化更新，不是额外100个epoch，也不是少样本校准**。两条方法共享同一基础模型，追加训练只改变其复制品。R2历史MC-UVE＋SVM仍为821／855、96.0234%、Macro-F1 0.9197；本轮旧强MLP未胜出。不得把旧171份上的90.25%／0.8787直接放进本表。

## 精确复用的训练语义

- 原`neural_base_search.fit_preprocessor/fit_model`：逐谱SNV（ddof1）→当前源StandardScaler，无PCA；612→64→32→7，ReLU、41543参数；Adam lr0.001、weight_decay0.001、batch64、200epoch，无类别权重，无早停。
- 原`image_probability.supervised_continuation`：复制基础模型，新建Adam lr0.0001、默认weight_decay0，100次更新；每次无权源3-way query CE＋无权全源CE。source任务各类1份support／1份query互斥，support不参与该监督控制的episode梯度；full-source anchor保留七类。continuation seed固定base seed＋1600。
- 每个内层源342份、验证342份，其中ZY02源2份；每个外层源684份、留出171份，其中ZY02源4份。本轮所有7类均满足episode最低2份，因此没有排除实际ZY02；辅助单元测试另核验singleton从episode排除、仍保留于七类全源CE的边界语义。
- 所有预处理、新基础模型及继续训练都在当前源分区重建，未复用旧见过外折样品的权重。两角色主比较只是监督预算对照，没有目标标签贡献。

## 全部种子与负结果

| 方法 | seed | 正确数／855 | Accuracy | Macro-F1 | Balanced Accuracy | ZY02正确数／5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原64–32 MLP，200个epoch | 42 | 768 | 89.8246% | 0.8191 | 79.6327% | 2 |
| 原64–32 MLP，200个epoch | 43 | 775 | 90.6433% | 0.8294 | 80.5306% | 2 |
| 原64–32 MLP，200个epoch | 44 | 774 | 90.5263% | 0.7885 | 77.7755% | 1 |
| 同一起点＋100次源监督更新 | 42 | 777 | 90.8772% | 0.8319 | 80.9184% | 2 |
| 同一起点＋100次源监督更新 | 43 | 772 | 90.2924% | 0.8162 | 80.2041% | 2 |
| 同一起点＋100次源监督更新 | 44 | 775 | 90.6433% | 0.8198 | 80.6327% | 2 |

| 方法 | 三seed Accuracy均值±样本SD | Macro-F1均值±样本SD | Balanced Accuracy均值±样本SD |
| --- | ---: | ---: | ---: |
| 原64–32 MLP，200个epoch | 90.3314% ± 0.4428个百分点 | 0.8123 ± 0.0212 | 79.3129% ± 1.4051个百分点 |
| 同一起点＋100次源监督更新 | 90.6043% ± 0.2943个百分点 | 0.8226 ± 0.0082 | 80.5850% ± 0.3595个百分点 |

三seed重复使用同一855份独立样品，不增加样品数，SD只描述训练随机性，不是独立样品置信区间。

| seed | 追加训练纠正原错误 | 原正确变错 | 净增加正确 | Accuracy变化（百分点） |
| --- | ---: | ---: | ---: | ---: |
| 42 | 13 | 4 | +9 | +1.0526 |
| 43 | 2 | 5 | -3 | -0.3509 |
| 44 | 6 | 5 | +1 | +0.1170 |

seed43追加训练反而下降；该负结果和逐样品变动均保留。主表未根据这一结果选择其他种子或训练预算。

## 计算预算与复现

本轮内层30个基础训练上下文、60个配对方法评价、20520条内层预测，实际selection耗时34.1192秒；外层15个新基础训练上下文、30个角色、5130条OOF（每方法每seed恰855），evaluation耗时29.8962秒。CPU、单线程；各自源预处理／基础训练／追加训练／预测时间及环境版本见原始JSON，不能把共享基础模型成本重复算作两次。三seed总计45个基础上下文、9000个基础epoch和4500次源任务继续更新，不等于9000份新样品或独立批次。

```bash
.runtime/bin/python scripts/run_paper_c_strong_mlp.py --protocol data/research/paper_c_strong_mlp_protocol_20261031.json --output outputs/paper-c-strong-mlp-new --stage selection
# 外层需先独立审查并生成绑定此新run锁／源码／协议的release；不能复用旧release。
.runtime/bin/python scripts/run_paper_c_strong_mlp.py --protocol data/research/paper_c_strong_mlp_protocol_20261031.json --output outputs/paper-c-strong-mlp-new --stage evaluation --release outputs/new-strong-mlp-release.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .runtime/bin/python reports/paper-c/2026-10-31/strong-mlp/replay_audit.py
```

## 证据与检查

- `protocol_snapshot.json`、`run_manifest.json`、`selection_lock.json`及`source_snapshot/`：运行前冻结模型定义、源码／输入SHA、完整内层锁与三个种子。
- `selection/`：全部30内层上下文的基础／追加权重、源预处理器、源／验证编号、200epoch损失、100源任务及每次损失。
- `models/`：全部15外层上下文的两份配对权重、预处理器、训练记录和留出编号；都是684源训练的外折模型，不是一个全855部署模型。
- `inner_predictions.csv.gz`、`outer_predictions.csv.gz`与对应metrics：真实逐样品标签／预测／七维logits；logits不是已校准概率。
- `pooled_metrics_by_seed.json`、`primary_seed42_metrics.json`、`seed_sensitivity.json`：全部种子及预先主种子的Accuracy／Macro-F1／Balanced Accuracy、逐类P／R／F1、混淆矩阵及计算成本。
- `continuation_paired_changes.csv`、`continuation_paired_summary.json`：全部2565个seed×样品情景的改对／改错，目标支持标签使用数0。
- `archive_manifest.json`：208份原始运行文件的按字节归档索引，原路径字段保留实际运行目录，不改写历史内容。
- `replay_audit.py/json`：零拟合重放全部45上下文／90权重角色、25650条内外预测、4500个源任务，138032项检查通过，最大数值差0；源缩放、初始种子、标签／身份、标准化及配对基础模型不变均通过。
- 外测前独立审查见[第三协议review](../development-review/STRONG_MLP_REVIEW.md)；[统一外测放行](../evaluation_release.json)先于实际外层调用，[命令起止](../command_invocations.jsonl)保留真实UTC／argv／exit0。

每产地仍只有一个来源批次，且历史数据与开发结果此前已曝光。该结果不能包装为真实跨批次确认，不能以新增植物或校准替代酸枣仁主研究。
