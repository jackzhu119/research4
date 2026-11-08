# Research4跨分支证据审查：下一轮训练的依据与限制

本审查于2026-11-08通过已连接GitHub工具只读取得各分支提交、代表性小型报告、
真实结果和完成记录。没有拟合、改变数据、接管计算或修改既有Git工作区。
指标是对已保存证据的读取，不代表本次重新训练或完整复算所有历史实验。
本文件只服务于新实验的假设形成，不能将不同任务或划分的数字拼接成提升阶梯。

## 已固定的分支版本

| 分支 | 核验提交SHA | 主要用途 |
|---|---|---|
| `research/algorithm-review-and-baselines-20261030` | `3fc996e99cce8ff5799b5a2af9b565f982dfc899` | 原855份、612频点、嵌套MC-UVE/SVM及组合对照 |
| `research/supplemental-spectra-20261104` | `765742a10d516aff66079f748c9ba3ffcd8ce933` | 固定角色多表示、黄芪/天麻及保守迁移预算消融 |
| `research/postrun-analysis-20261106` | `592a62aef75d25252b7513330f9944a9fa4c08dd` | Host A量子曲率与响应诊断、证据闭合 |
| `research/parallel-worker-b-20261106` | `c2284ad4755b223830aeb2283d7491a3a3f35a28` | 冻结的另一主机计算包及真实水控制 |
| `research/parallel-worker-b-20261106-B20261106-01` | `503de339ed5d42967f5dbf0502e9d0b5ffc1115b` | 另一主机真实量子诊断，以及已完成的种类误用审计 |
| `research/pyscf-spinosin-mechanism-20261031` | `cd4b50c37c1e26064f9f15f340d30e20ac386458` | 完整旧219模式和后续固定网格诊断记录 |
| `main` | `806584a5571c6a90ef37c6bfa915671b680ba48b` | 用户确认来源/种类的数据版本核查 |
| `data/confirmed-origins-20261108` | `8119d5635ea1dd4eb5a0c5212893b7c0d7a37309` | 同上，尚未补入全部独立性声明的登记表 |
| `research/species-label-correction-20261108` | `21d6edab24d198dc60b1ad4ef50e83602c9ecfa7` | 当前训练使用的数据、独立性声明和固定角色依据 |

分支后续可能继续增加提交。本审查结论绑定以上SHA，不把之后变化默认为已核验。
现行运行分支和备份分支的任务状态应另查实际账本；准备配置、分支存在或旧PID均不证明当前运行。

## 数据登记表的实际差异

核对文件均为`data/active_workbooks.json`。当前独立新工作区的canonical文件为：

`/workspace/research4-nextstage/data/active_workbooks.json`

其SHA-256为`dd9ec7728590dde2574f64b9664148465c85d7b618311494c613c32b4d879f95`。

| 固定来源 | SHA-256 | 与当前canonical关系 |
|---|---|---|
| `main@806584a5571c6a90ef37c6bfa915671b680ba48b` | `d771713855e56c8c47b5e96dc0530233d2cf0e5a3d15e2cd13d00aaed35eee54` | 字节和JSON语义不同；未替换canonical |
| `data/confirmed-origins-20261108@8119d5635ea1dd4eb5a0c5212893b7c0d7a37309` | `d771713855e56c8c47b5e96dc0530233d2cf0e5a3d15e2cd13d00aaed35eee54` | 与上述main相同 |
| `research/species-label-correction-20261108@21d6edab24d198dc60b1ad4ef50e83602c9ecfa7` | `dd9ec7728590dde2574f64b9664148465c85d7b618311494c613c32b4d879f95` | 字节完全相同，JSON语义完全相同 |

三个远端版本都声明`version=2026-11-08-confirmed-origin-kind-v1`、18工作簿、1672记录，
种类计数同为枣仁364、酸枣仁500、黄芪403、天麻405。递归JSON比较发现35处差异：
顶层`independence_confirmation_records`和`scope_note`，以及工作簿索引7—17的
`physical_independence_status`、`independence_basis`、`independence_confirmation_record`。
这些对应后来对817份样品“每列独立实物”的用户声明，未改变工作簿路径、文件哈希、
种类、来源或记录数。相同version字符串不意味着登记表字节相同。
不得用main较旧的独立性字段覆盖当前版本；用户独立性声明也不等于独立实验室鉴定、
多个已核实批次或真实跨批次验证。

## 原算法分支：可复用流程，不可转移混种准确率

已读：

- [`docs/STAGED_NESTED_OPTIMIZATION_REPORT.md`](https://github.com/jackzhu119/research4/blob/3fc996e99cce8ff5799b5a2af9b565f982dfc899/docs/STAGED_NESTED_OPTIMIZATION_REPORT.md)
- 该报告引用的原始目录为`reports/staged-nested-optimization/2026-10-31/`。

旧实验使用855份ZY记录、612频点、相同5外折/2内折和逐样品OOF：

| 旧策略 | 正确数/855 | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| R0 | 773 | 90.4094% | 0.8092 |
| R1 | 791 | 92.5146% | 0.8277 |
| R2 | 821 | 96.0234% | 0.9197 |
| R3 | 816 | 95.4386% | 0.8877 |
| R4 | 812 | 94.9708% | 0.8820 |

R2相对R0纠正62份旧错误、引入14份新错误，净增48份；后续复杂方案未继续提高。
同参数继承条件下SG一阶+SNV、稳定MC/CARS和CNN未超过原较强路线。这些负结果必须保留，
但它们也不能证明改正种类后的任务或其他联合优化必然无效。

ZY01/ZY02现在确认属于枣仁，其他五组属于酸枣仁。因此96.0234%只能保留为
历史混合种类匿名来源组识别结果。当前纯酸枣仁300TRAIN重复嵌套CV与855OOF、
固定171/100预测集均不是相同协议。可复用MC前缀计算、训练内参数选择、投票语义、
逐样品改对/改错及来源锁；不可把旧全855目标训练权重直接作为当前外折模型初始化。

## 多表示与旧迁移：复杂度和辅助数据需要匹配对照

已读固定版本`765742a10d516aff66079f748c9ba3ffcd8ce933`上的：

- [`docs/SUPPLEMENTAL_MULTIREPRESENTATION_RESULTS_20261104.md`](https://github.com/jackzhu119/research4/blob/765742a10d516aff66079f748c9ba3ffcd8ce933/docs/SUPPLEMENTAL_MULTIREPRESENTATION_RESULTS_20261104.md)
- [`docs/CONSERVATIVE_CROSS_PLANT_BUDGET_ABLATION_20261104.md`](https://github.com/jackzhu119/research4/blob/765742a10d516aff66079f748c9ba3ffcd8ce933/docs/CONSERVATIVE_CROSS_PLANT_BUDGET_ABLATION_20261104.md)
- 原始模型联合表：`reports/supplemental-combined-models-20261104-v1/model_comparison.csv`。
- 配对表：`reports/supplemental-multirepresentation-20261104-v2/score/paired_corrected_harmed.csv`。
- 两预算表：`reports/cross-plant-conservative-pilot-20261104-v2/two_budget_model_comparison.csv`。

| 旧固定query任务 | 匹配三分支α控制C | α+n+time融合M | 解释限制 |
|---|---:|---:|---|
| 旧7个ZY组，171query | 82.6511% | 82.4561% | M的Macro-F1和BA亦下降；仍为混种来源任务 |
| 黄芪，82query | 69.9187% | 78.4553% | M比C高8.5366个百分点，但仍低于同拆分LDA84.1463% |
| 天麻，83query | 93.5743% | 94.3775% | 小幅表示收益，仍低于同拆分MLP99.1968% |

旧855任务的跨植物预算消融为：

| 最大轮数 | A从头 | Aextra额外目标训练 | B黄芪+天麻预训练 |
|---|---:|---:|---:|
| 48 | 80.3509% | 78.3236% | 76.1404% |
| 96 | 87.0565% | 86.3548% | 85.3801% |

三种配方都因更大预算提高，B在各预算仍低于A。该证据支持保留Aextra成本控制，
不支持把增加迭代数视为外部知识收益，也不排斥当前纯酸枣仁其他配置出现正迁移。
黄芪/天麻来源标签与酸枣仁标签不能混成一个产地头。旧时域还经历非均匀导出轴的显式修复，
单位未核实；不能直接把这些时域坐标当作THz物理频率或据此声称FFT化学解释。

## 另一主机B：较新的实际结果优于旧等待摘要

以下文件绑定分支`research/parallel-worker-b-20261106-B20261106-01`、
提交`503de339ed5d42967f5dbf0502e9d0b5ffc1115b`。

完整目录前缀为：

`reports/parallel-cloud-20261106/worker-b/B20261106-01/species-misuse-audit-v1/`

`analysis/ROOT_final_summary_zh.json`在2026-11-08T04:52Z记录等待和0新拟合。
但更晚的下列实际记录证明该有界审计后来完成，不能继续拿旧摘要声称其尚未训练：

- `reports/parallel-cloud-20261106/worker-b/B20261106-01/species-misuse-audit-v1/science/audit-attempt001/result.json`：`SPECIES_AUDIT_COMPLETE`、20次真实拟合、完成于2026-11-08T05:23:14Z。
- `reports/parallel-cloud-20261106/worker-b/B20261106-01/species-misuse-audit-v1/execution/audit-attempt001.wait4.json`：exit0，用户CPU10.610038秒、系统CPU0.453137秒，合计11.063175秒。
- `reports/parallel-cloud-20261106/worker-b/B20261106-01/species-misuse-audit-v1/execution/audit-attempt001.proof.json`：实际监督来源、输入和600秒审计上限；该上限来自原B累计90000秒余量，不是本机新授予90000秒。
- `reports/parallel-cloud-20261106/worker-b/B20261106-01/species-misuse-audit-v1/science/audit-attempt001/TRAIN_only_group_control.json`。
- `reports/parallel-cloud-20261106/worker-b/B20261106-01/species-misuse-audit-v1/science/audit-attempt001/analysis_conclusion_zh.txt`。

它只使用516份枣仁/酸枣仁master TRAIN，按预先固定的五组来源配对留出，
固定SNV+Scaler+SVM；每fold分别真实标签及三个fit-label置乱，共20次拟合，不做参数选择：

| 控制 | 正确数/516 | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|---:|
| 真实标签 | 493 | 95.5426% | 0.953424 | 94.6759% |
| 训练标签置乱42 | 167 | 32.3643% | 0.248670 | 27.9630% |
| 训练标签置乱43 | 211 | 40.8915% | 0.295546 | 35.2963% |
| 训练标签置乱44 | 189 | 36.6279% | 0.270465 | 31.5648% |

真实控制的23个错误都是ZY01枣仁判成酸枣仁；酸枣仁300份均正确。
不同来源组大小差异很大，ZY01被留出时其余枣仁训练池极小。
该结果不等于随机样品折的约99.81%，也不能把二者差异归因为一种模型改进。
来源组不是已核实的新批次；三个置乱seed不能支持正式p值。
这些20次已完成拟合不应作为新任务重复执行或再计为本次新贡献。

另已核：

`reports/parallel-cloud-20261106/worker-b/B20261106-01/species-misuse-audit-v1/science/audit-attempt001/joint_components_exact_correctness.json`

其对冻结组件按相同339个ID逻辑与，种类+来源共同正确304/339=89.6755%，误路由8份。
逐真实种类为天麻81/83、枣仁70/74、酸枣仁87/100、黄芪66/82。
这不是完整新层级模型：误路由后的实际来源输出`NOT_COMPUTED`，
`full_joint_macro_f1=NOT_COMPUTED`。不能相乘边际准确率，或用四药材97.64%代替系统准确率。

## 量子分支：真实完成范围与未获得的证据

已读：

- [`docs/PYSCF_SPINOSIN_COMPLETION_20261106.md`](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/PYSCF_SPINOSIN_COMPLETION_20261106.md)
- [`docs/FIXED_GRID_STEP_CONVERGENCE_20261107.md`](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/FIXED_GRID_STEP_CONVERGENCE_20261107.md)
- `reports/fixed-quadrature-convergence-20261107-v2/queue-host-a-006/state.json`。
- `reports/fixed-quadrature-convergence-20261107-v2/queue-host-a-006/root-current-state-handoff-057.json`。
- B分支`503de339...`的`reports/parallel-cloud-20261106/worker-b/B20261106-01/followup-negative-subspace-v7/science/all-negative-subspace-comparison-attempt001/result.json`。

旧Spinosin完整219个有符号模式和独立数学模式复核已完成；194正、25虚。
这些结果不证明最低点，IR强度保持`NOT_COMPUTED`，不能据1.11005THz附近单个正模式
确认化学峰归属或将其作为可靠训练先验。

A006水v3的31/31端点与预设步长控制通过，是真实小分子控制。
但A006的较新state记录Spinosin单方向能量任务于2026-11-07T02:43:35Z
`FAILED_CLOSED`，停止原因`ACTUAL_CPU_GUARD`，真实7381.797739CPU秒，
`scientific_release=false`。更早01:43Z的handoff057还写正在计算，不能据它声称当前在运行。
该Git记录不是本机实时进程证明，也没有产生新的完整频率谱。

B的v7汇总于2026-11-08T05:22:18Z为`AGGREGATE_COMPLETE`，missing/excluded为空，
但`chemical_peak_assignment_verified=false`、`classifier_modified=false`、
`IR_intensities_status=NOT_COMPUTED`。其结论限定为诊断观察；固定绝对积分网格和
重建原子中心网格是不同离散化，不合并。部分网格加密/更小步长的曲率符号仍与旧解析值不同。
不能据此擅自认定某个实现缺陷、翻转虚频、宣告稳定最低点或加入所谓分子化学特征。

## 对新一轮有限训练的直接影响

1. 酸枣仁主线应先检验分块加权融合、MC-UVE稳定性和匹配特征预算。
   单独n有弱点不意味着n+α无价值；更多表示也不自动构成创新。
2. 在现行300酸枣仁TRAIN内部，同划分同选参范围加入LDA/MLP等传统强对照。
   旧固定query高分只是选择对照的依据，不能转移成新任务成绩。
3. 若增加系统实验，应真正重建TRAIN内部gate→各药材来源头，记录所有误路由后的
   输出与联合混淆、Macro-F1，而不重复既有304/339组件逻辑与。
4. 迁移保留A/Aextra/B匹配计算成本、相同source/support/query、完整seed/fold和负结果。
   外部只含辅助药材TRAIN，不能用旧全目标池训练权重污染当前外折。
5. 依照canonical独立性声明及固定999/334/339角色；CAL/PRED不参与新选择。
   这些材料已曝光，新的开发验证仍不是独立确认，不能声称真实跨批次校准。
6. 本轮量子机制先验`NOT_RUN`；没有可靠峰归属或逐样品化学参照，不把文献平均值、
   分子SDF或未通过最低点资格的频率填为样品训练特征。
