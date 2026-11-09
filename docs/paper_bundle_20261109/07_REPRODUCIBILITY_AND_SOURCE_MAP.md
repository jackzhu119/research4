# 复现入口、跨分支来源与证据位置

编制：2026-11-09；源目录截止UTC 2026-11-09T01:13:24.282031+00:00。本章提供证据导航，不重新训练模型、不执行新量子计算，也不依据旧PID推断当前任务存活。

## 可以按固定来源找到设计、原始结果与解释边界

机器可读[源目录](../../evidence/paper_bundle_20261109/source_catalog.json)覆盖**17个远端分支和1个本地已提交HEAD快照**。按Git内容对象去重后，共有**2382份唯一来源内容**、**8223个跨分支/路径位置**。其中**2279份小文本/配置**实际读取计算SHA-256；**103份较大来源**仅登记Git blob SHA-1、大小和路径，未在此次目录工作中读取正文。大模型、工作簿数值、全部逐样品/逐核坐标原始文件和科学分卷不被重新复制或解压。

这些数量是文件/位置计数，不是实验数、独立研究数、样本数或真实完成工作数。相同文件被多个分支继承只计一份内容；同一`version`文字但字节不同的登记表仍分别保留。`canonical_location`只是索引展示优先级，不代表该版本最新、批准或科学上更可信。

| 分支／快照 | 固定提交 | 入选来源位置数 | 作用 |
| --- | --- | ---: | --- |
| `data/confirmed-origins-20261108` | [`8119d5635ea1`](https://github.com/jackzhu119/research4/commit/8119d5635ea1dd4eb5a0c5212893b7c0d7a37309) | 12 | 用户文件名与种类/产地确认 |
| `main` | [`806584a5571c`](https://github.com/jackzhu119/research4/commit/806584a5571c6a90ef37c6bfa915671b680ba48b) | 12 | 当前18表文件／标签基础版本；独立性声明字段可能较新研究分支旧 |
| `research/algorithm-review-and-baselines-20261030` | [`3fc996e99cce`](https://github.com/jackzhu119/research4/commit/3fc996e99cce8ff5799b5a2af9b565f982dfc899) | 302 | 历史855混合ZY任务、分类/校准/优化/部署 |
| `research/cache-aware-continuation-20261108` | [`df3e12656e21`](https://github.com/jackzhu119/research4/commit/df3e12656e2177cfcf0a298e66bdedc5c524cb74) | 504 | 正确种类重启矩阵剩余神经任务、压缩摘要及独立外层运行账本 |
| `research/evidence-driven-validation-90000-20261108` | [`71a303e089c7`](https://github.com/jackzhu119/research4/commit/71a303e089c7c7b7b2557d62372aaceda44e2f91) | 452 | 证据驱动验证330个科学任务及正式汇总 |
| `research/paper-evidence-compendium-20261109` | [`a861d28a1fa0`](https://github.com/jackzhu119/research4/commit/a861d28a1fa075f09f06b19c765098251d54d569) | 452 | 已存在的跨分支证据整理和大结果分卷，不能仅因名称当最新结论 |
| `research/parallel-worker-b-20261106` | [`c2284ad4755b`](https://github.com/jackzhu119/research4/commit/c2284ad4755b223830aeb2283d7491a3a3f35a28) | 772 | 可移植B包与资格检查 |
| `research/parallel-worker-b-20261106-B20261106-01` | [`c8945acc0a57`](https://github.com/jackzhu119/research4/commit/c8945acc0a57d75aae438fd9de81d05b422cc1f3) | 1117 | B返回的量子与后续机器学习机制诊断 |
| `research/postrun-analysis-20261106` | [`592a62aef75d`](https://github.com/jackzhu119/research4/commit/592a62aef75d25252b7513330f9944a9fa4c08dd) | 783 | 量子闭合/响应和曲率事后分析 |
| `research/pyscf-autonomous-90000-20261108` | [`5d71d0d99f69`](https://github.com/jackzhu119/research4/commit/5d71d0d99f697a004b3593e4bc0a6922c0ba52be) | 21 | 当前整合分支先前提交；动态运行另有时间戳快照 |
| `research/pyscf-spinosin-mechanism-20261031` | [`cd4b50c37c1e`](https://github.com/jackzhu119/research4/commit/cd4b50c37c1e26064f9f15f340d30e20ac386458) | 804 | 历史量子及多阶段光谱实验全集 |
| `research/species-label-correction-20261108` | [`0156e2189d13`](https://github.com/jackzhu119/research4/commit/0156e2189d13456d1faf05ebfd53d86eeeb4f5e7) | 25 | 纠正种类后的固定角色验证、注释更新 |
| `research/species-reconciliation-audit-20261108` | [`debb2402edf8`](https://github.com/jackzhu119/research4/commit/debb2402edf8eea6a132da124abb1411b68d3219) | 811 | 混种结论纠正及输出审计 |
| `research/species-relaunch-90000-20261108` | [`8a388b8b4d8a`](https://github.com/jackzhu119/research4/commit/8a388b8b4d8a5baf9f2e4321e76455c2c0b49eca) | 710 | 纠正种类后1305任务矩阵的冻结设计与首批返回 |
| `research/species-results-backfill-20261108` | [`f38113e51072`](https://github.com/jackzhu119/research4/commit/f38113e51072f0a99c864b2e6dae79a9169ff93c) | 1 | 已闭合结果回填收据 |
| `research/supplemental-spectra-20261104` | [`765742a10d51`](https://github.com/jackzhu119/research4/commit/765742a10d516aff66079f748c9ba3ffcd8ce933) | 652 | 多模态/其他植物/保守迁移及材料审计 |
| `research/worker-b-bootstrap-20261106` | [`294f0cfe6626`](https://github.com/jackzhu119/research4/commit/294f0cfe6626d9b95df3f7b7b8fdd3c40f2f205b) | 772 | 另一主机启动说明与依赖，不是独立新增实验 |
| `workspace_head_at_catalog_cutoff` | [`5d71d0d99f69`](https://github.com/jackzhu119/research4/commit/5d71d0d99f697a004b3593e4bc0a6922c0ba52be) | 21 | 生成源目录时的本地已提交HEAD；未提交动态文件不在此目录内 |

## 按论文工作顺序读取证据

本目录的[历史ML](02_HISTORICAL_ML.md)、[当前数据与种类](03_CURRENT_DATA_AND_SPECIES.md)、[外部数据/迁移/稳健性](04_EXTERNAL_TRANSFER_AND_ROBUSTNESS.md)、[量子计算](05_QUANTUM_AND_MECHANISM.md)与[B机制诊断](06_WORKER_B_DIAGNOSTICS.md)分别解释数字、正负结果和可写结论。不要直接以某一分支README标题里的“最新”代替下面的固定提交。

| 要找什么 | 固定版本入口 | 使用时核对 |
| --- | --- | --- |
| 当前数据身份 | [data/active_workbooks.json](https://github.com/jackzhu119/research4/blob/806584a5571c6a90ef37c6bfa915671b680ba48b/data/active_workbooks.json) | 只从有效清单选输入；18表/1672谱，四模态不乘成四倍样本 |
| 后续单独实物独立性声明 | [data/annotations/specimen_independence_user_declaration_20261108.json](https://github.com/jackzhu119/research4/blob/a861d28a1fa075f09f06b19c765098251d54d569/data/annotations/specimen_independence_user_declaration_20261108.json) | 核对是否新增声明；不把旧单批次声明外推给新数据 |
| 历史嵌套优化设计 | [docs/STAGED_NESTED_OPTIMIZATION_PROTOCOL.md](https://github.com/jackzhu119/research4/blob/3fc996e99cce8ff5799b5a2af9b565f982dfc899/docs/STAGED_NESTED_OPTIMIZATION_PROTOCOL.md) | 原855混合种类、原五外/二内折，不能转称当前纯酸枣仁成绩 |
| 历史优化执行 | [docs/STAGED_NESTED_OPTIMIZATION_REPORT.md](https://github.com/jackzhu119/research4/blob/3fc996e99cce8ff5799b5a2af9b565f982dfc899/docs/STAGED_NESTED_OPTIMIZATION_REPORT.md) | 与同目录原始OOF/参数/锁定记录配读 |
| 多模态执行 | [docs/SUPPLEMENTAL_MULTIREPRESENTATION_RESULTS_20261104.md](https://github.com/jackzhu119/research4/blob/765742a10d516aff66079f748c9ba3ffcd8ce933/docs/SUPPLEMENTAL_MULTIREPRESENTATION_RESULTS_20261104.md) | 保持原固定角色和对照，不把不同任务准确率连成提升阶梯 |
| 新种类实验设计 | [docs/SPECIES_RELAUNCH_90000_20261108.md](https://github.com/jackzhu119/research4/blob/8a388b8b4d8a5baf9f2e4321e76455c2c0b49eca/docs/SPECIES_RELAUNCH_90000_20261108.md) | 查本轮实际设计与角色，不继承旧855模型的成绩 |
| 证据驱动验证 | [docs/EVIDENCE_DRIVEN_VALIDATION_20261108.md](https://github.com/jackzhu119/research4/blob/71a303e089c7c7b7b2557d62372aaceda44e2f91/docs/EVIDENCE_DRIVEN_VALIDATION_20261108.md) | 读预设比较、源内选择与330任务真实完成摘要 |
| 剩余任务控制入口 | [CURRENT_RUNTIME_CONTROLLER.json](https://github.com/jackzhu119/research4/blob/df3e12656e2177cfcf0a298e66bdedc5c524cb74/CURRENT_RUNTIME_CONTROLLER.json) | 指向新的外层controller；旧PAUSED只作历史 |
| 剩余科学描述符 | [remaining_jobs.json](https://github.com/jackzhu119/research4/blob/df3e12656e2177cfcf0a298e66bdedc5c524cb74/remaining_jobs.json) | 原480未完成descriptor原样迁移；配置存在不等于已完成 |
| 完整剩余任务汇总 | [reports/00480/REPORT.md](https://github.com/jackzhu119/research4/blob/df3e12656e2177cfcf0a298e66bdedc5c524cb74/reports/00480/REPORT.md) | 固定提交下的有完整折/种子支持的返回汇总；不能把480当样本数 |
| B机器学习后续分析 | [reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/analysis/independent_final_science_analysis_001.json](https://github.com/jackzhu119/research4/blob/c8945acc0a57d75aae438fd9de81d05b422cc1f3/reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/analysis/independent_final_science_analysis_001.json) | 与本包06章核对来源、预算控制及有害/获益结论 |
| B深机制块索引 | [reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/science/summary-attempt001/summary_index.json](https://github.com/jackzhu119/research4/blob/c8945acc0a57d75aae438fd9de81d05b422cc1f3/reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/science/summary-attempt001/summary_index.json) | 块级科学任务汇总，不必读取所有原逐点结果来重复概述 |
| Spinosin量子完成/未认证边界 | [docs/PYSCF_SPINOSIN_COMPLETION_20261106.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/PYSCF_SPINOSIN_COMPLETION_20261106.md) | 已计算219模式不等于无虚频最低点，后续诊断另查QC章 |
| 参考资料归档 | [references/paper-materials/manifest.csv](https://github.com/jackzhu119/research4/blob/806584a5571c6a90ef37c6bfa915671b680ba48b/references/paper-materials/manifest.csv) | 51份写作/算法/文献资料；不是51次科学实验或新增观测 |

## 复现链条与最低交付材料

| 层次 | 应保存或打开的材料 | 不能混淆的状态 |
| --- | --- | --- |
| 输入身份 | `active_workbooks.json`、逐文件SHA-256、注释声明、当前/历史替换表 | 不递归扫描raw/archive/processed来决定当前输入；旧原件、编号版、更名版不算三份样品 |
| 协议 | 对应研究版本的protocol/plan/config、数据暴露记录、种子、角色/折划分与选择锁 | 用户批准/冻结配置不等于任务已执行；不同fixed query与OOF不能直接配对 |
| 实际运行 | 输入/源码hash、实际输出路径、日志和闭合账本、全部正负/失败输出 | 工程单测、原MATLAB兼容性或Python适配验证不等于新药材科学成绩 |
| 预测与模型 | 对应逐样品OOF或固定query预测、truth接口、模型/权重和全参数 | 同一模型在已曝光数据上的诊断不是新确认；决策分数不自动是概率 |
| 指标与统计 | 按原任务分母重算的Accuracy、Macro-F1、BA、逐类召回及预设配对 | 应池化预测后算率，不能随意平均折/种子；bootstrap条件与独立批次置信区间分开 |
| 量子证据 | 原子顺序、可信连接/构象来源、坐标单位、方法/网格、SCF/核梯度/Hessian/符号频率 | 计算完成、原生SCF接受、fresh残差目标、核驻点和最低点验证是不同字段；IR未算不填写 |
| 回传与复用 | 小结果快照、Git远端提交核验、被排除大cache的本地路径与hash | push尝试不等于成功；归档分卷需完整清单/哈希才能恢复，不能仅凭分支存在声称大数据均已备份 |

当前18工作簿/1672谱对应四个用户中文种类标签，真实batch信息与实验单位仍需按来源分别核对。后续新增独立性声明不能被较旧main注释覆盖；声明不等于外部实验室审计。历史855条混合ZY来源任务的分区和指标保持原样，只能带着其当时数据范围引用。

## 大结果分卷及参考资料

固定`paper-evidence-compendium`提交中的`reports/paper-evidence-20261109/archives/`保存科学结果tar.gz分卷，本次目录只盘点其位置和体积，不重新拼接/解压，也不把压缩体积当科学工作量。恢复应遵循该发布的原清单并验证所有分卷哈希；缺少某卷或未成功回传的本地大文件不能假称可恢复。

`references/paper-materials/manifest.csv`中的51份资料是参考文献、算法程序、写作/投稿材料、教程和RAR归档。原MATLAB代码与本研究Python独立适配实现需区分，兼容性运行不能写成原算法精确科学复现。文献或演示代码不替代原始药材标签、独立样品、批次或成分实测。

## 最小复现操作顺序

1. 用本章固定提交打开对应协议和源文件，核对输入清单、标签和已曝光范围；不在正在运行的工作树切换分支。
2. 在独立副本中安装该实验实际记录的依赖，核对源码/数据hash，读取已经固定的角色、外折和模型种子；不要从外层或最终query表现另选赢家。
3. 优先读取已回传的小汇总、预测和完成收据，检查覆盖了全部预设折/种子。缺失/失败不当0，不把计划任务数当完成数。
4. 需要重新计算时，从同实验协议的可执行入口启动新的、独立命名的运行，保留原始结果与失败；本文目录本身不触发重算。
5. 写作时引用固定提交与本包实验ID。把当前进行中量子任务引用为带观察时间的状态快照，完成并独立复核前不写进已经取得的正式结论。
