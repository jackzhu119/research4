# Research4 论文写作总入口（2026-11-09）

这份汇编把此前设计、运行、失败、修正及尚未完成的实验放在同一个写作索引下。
本次工作是**保存证据和整理结论，没有重新训练模型，也没有执行新的量子计算**。
原始数据、标签、历史结果与main保持不变；本入口所在研究分支用于论文准备。

## 先阅读这五份文件

1. [完整证据目录](docs/paper-writing-20261109/README.md)：按研究阶段查实验、代码、结果和备份。
2. [数据与方法](docs/paper-writing-20261109/DATA_AND_METHODS.md)：现行种类、来源组、样品数、固定划分与泄漏限制。
3. [模型结果与可写结论](docs/paper-writing-20261109/MODEL_RESULTS_AND_CONCLUSIONS.md)：主模型、传统对照、融合、迁移、校准和负面结果。
4. [实验审计总表](docs/RESEARCH4_EXPERIMENT_AUDIT.md)：证据等级、完成范围、准确率适用条件。
5. [论文大纲](docs/RESEARCH4_PAPER_OUTLINE.md)：各章节应引用的证据、图表及未解决局限。

## 当前论文应使用的主结论

**纯酸枣仁主开发候选是α＋SNV＋MC-UVE20＋RBF-SVM：Accuracy 93.4444%，Macro-F1 0.934401，Balanced Accuracy 93.4444%。**
这是当前300份酸枣仁TRAIN中三次划分、五外折／二内折的内部开发结果。
每次都覆盖同一300份实物，三次正确数282、280、279；841/900指重复预测，不是900份实物。
K、C和γ在内折选择；15外折K184占6折、K245占9折，均K220.6。
这是经验证的选择流程，不是一份已在全部TRAIN上拟合并经过独立验证的部署模型。

| 同一V4内部开发协议 | Accuracy | Macro-F1 | 论文用途 |
|---|---:|---:|---|
| SNV＋MC-UVE20＋SVM | 93.4444% | 0.934401 | 简洁主候选 |
| SNV＋MC-UVE100＋SVM | 93.4444% | 0.934503 | 增加MC次数的消融，准确率未提高 |
| 稳定5×MC20＋SVM | 92.7778% | 0.927969 | 稳定性消融，保留负结果 |
| 收缩LDA | 91.7778% | 0.917591 | 同折传统强对照 |
| n＋α双SNV融合 | 91.6667% | 0.916904 | 多表示消融，收益依K而变化 |
| 全α MLP | 73.2222% | 0.728649 | 同折神经基线 |

校准结论须按任务分开：纯酸五来源暂未显示稳定改善；另一主机B的枣仁／酸枣仁二分类来源域适配
存在传统锚定收益，详见[B专项](docs/paper-writing-20261109/PARALLEL_HOST_B_RESULTS.md)，不能互相替代。

另一个受控神经协议下，最佳观察配置为α128／240source／144epochs／黄芪＋天麻预训练：
79.8889%、Macro-F1 0.799491；其匹配目标从头训练为77.3333%。
这不是与612点调优SVM完全相同的表示或搜索预算，也不支持深度学习已超过主候选。

**旧96.0234%不能作为纯酸枣仁的准确率。** 旧855份七组混合了枣仁与酸枣仁，
现保留为历史混合种类来源组识别。固定100份已曝光预测池的87%、94%、重训93%或96%
也必须单列，不能与93.4444%拼成提升阶梯。

## 数据与实验边界

当前18份工作簿共1672份用户声明独立实物：酸枣仁500、枣仁364、黄芪403、天麻405。
固定999训练／334校准／339预测。纯酸枣仁为300／100／100。
现行登记表为[data/active_workbooks.json](data/active_workbooks.json)，
固定角色见[逐样品名单](data/research/species_revision_20261108/split_manifest.csv)。
中文种类与来源是用户确认，不能代替独立植物学鉴定；现有结构仍有来源／批次混杂。
后续优化只接收TRAIN包，但样品已有历史曝光，不能称全新独立确认集。

## 所有设计与未完成项在哪里

- [历史算法、量子化学及未完成实验](docs/paper-writing-20261109/HISTORY_QUANTUM_AND_NOT_RUN.md)
- [另一云端B的独立实验](docs/paper-writing-20261109/PARALLEL_HOST_B_RESULTS.md)
- [后续论文研究定位](docs/RESEARCH4_PAPER_C_MASTER_PLAN.md)
- [机器可读结果包及图表](reports/paper-evidence-20261109/README.md)
- [可下载结果Excel工作簿](reports/paper-evidence-20261109/RESEARCH4_PAPER_RESULTS_20261109.xlsx)
- [实验流程图](docs/paper-writing-20261109/EXPERIMENT_FLOW.md)
- [1,635项任务逐条登记](reports/paper-evidence-20261109/tables/scientific_job_register.csv)
- [全部最终聚合设置](reports/paper-evidence-20261109/tables/all_final_model_settings.csv)
- [无损结果归档与恢复说明](reports/paper-evidence-20261109/archives/README.md)

`DONE`表示指定有界实验确实完成；`PARTIAL`表示只完成部分步骤；`NOT_RUN`表示没有执行；
`BLOCKED`表示有明确条件阻塞。得到219模式不等于无虚频最低点，也不等于实验吸收峰归属。
IR强度未计算时必须写NOT_COMPUTED。标准品THz、HPLC/LC-MS、独立新批次仍未实测。

## 开始写论文时的使用顺序

先以大纲写材料方法和主结果，再用CSV及图表填入确切数字；每一个数字保留协议和来源链接。
讨论应保留不成功的校准、负迁移及虚频诊断，不把标准方法改名作为原创，也不作成分含量或药效推断。
核查本次结果包可运行：

```bash
python scripts/build_paper_evidence_bundle_20261109.py --verify
```

该命令只读取保存证据并输出核验记录，不拟合模型、不恢复后台队列。
