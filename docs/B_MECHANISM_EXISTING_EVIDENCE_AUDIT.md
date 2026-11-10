# B 方案既有证据与重复实验审计（2026-11-10）

本次审计只读取历史证据和实现定义；没有重训历史任务，没有修改历史标签、模型、账本、封存状态或原始工作簿。旧实验只能作为有条件的历史诊断，不替代本轮冻结 ID 下的新结果。

## 固定来源与内容哈希

| 来源 | 固定提交 / 分支 | SHA-256（文件原始 UTF-8 字节） |
|---|---|---|
| `docs/paper_bundle_20261109/06_WORKER_B_DIAGNOSTICS.md` | `01604532a1232620681a7fbece7698fbf678d216`，`docs/paper-writing-evidence-20261109` | `81f3cfc20394a381c43558a44e4b5caed0b4bdcb68a0963f6df3bcc29f1a399a` |
| `evidence/paper_bundle_20261109/worker_b_ml.json` | 同上，Git blob `62fb21b63144a9987e7f0438f1769908b85458cd`，1,886,956 字节 | `76b048992f1f42f0a28989c4ab2bd2f580602d4a18d8c406aaa2c82302addc9f` |
| `docs/paper-writing-20261109/PARALLEL_HOST_B_RESULTS.md` | 基础分支提交 `b051733989161517a68c4a60cce75766880c95f5` | `f627ac293b883da382ea34376139b283e7989270e2ee26aa0d68c8b4c24e1738` |
| `src/herb_spectra/calibration.py` | 同一基础提交 | `b05ea9ddbcf6d79909b5795f596fe8d940c7c597d1af9b46481dd7e75aa36ace` |
| `src/herb_spectra/conservative_calibration.py` | 同一基础提交 | `b69871f7379f6b5e2e930429a357478e8b89fd68940bfc35e7372d8077a28472` |

历史章节实际绑定 Worker B 源提交 `c8945acc0a57d75aae438fd9de81d05b422cc1f3`，属于 `research/parallel-worker-b-20261106-B20261106-01`。基础分支的 Host B 文档绑定较早的 `c9f2fcf2b57934597c45caea2c113b713810d117`。本轮不将整个历史分支合并到当前数据分支，避免覆盖种类纠正后的有效清单。

复核方式：读取固定提交的文件原字节再计算 SHA-256。机器证据文件超过 1 MiB，GitHub Contents API 的普通 `.content` 字段可能为空；应请求 `Accept: application/vnd.github.raw+json`，不能把空内容的哈希当作真实文件哈希。

## 历史完成状态与时间截点

基础分支旧 Host B 摘要在当时只证明 V3 的 970 项科学任务封存，最终 summary 尚未封存；不能用该旧快照推断后续进程未完成。指定的较新诊断章节截点为 `2026-11-09T01:20:29Z`，记录 V3 的 970 项科学任务及 1 项汇总全部封存、16 个块完整，历史科学 wait4 CPU 为 12,566.568807 秒。该成本属于旧研究，不计入本次新研究账本。

初轮 V2 为 219 项科学任务完成；原汇总子进程真实 exit 0，但 4,740,138 字节结果超过旧监督器 4 MiB 读取门槛，保留 `FAILED_RESULT_VALIDATION_CLOSED`。后来的独立读取器复核 13 块一致，并验证 144 行指标及 158 行配对表。正确表述是科学任务完成、原工程失败保留且外部技术复核补足，不能改写成原 220 项全部 `DONE_SEALED`。这里的“外部复核”不是外部生物样本验证。

## E01—E04 重复性判定

| 本轮实验 | 历史相关证据 | 是否完全重复 | 本轮处理 |
|---|---|---|---|
| E01：Source 固定 240，酸/枣 60/180、120/120、180/60，目标酸枣仁来源覆盖 0/20，40 Query，20 Support，三种子 | V2 同来源 ZY01 覆盖；V3 donor 五来源 × 三模态 × 五抽样，固定 6 份枣仁、酸枣仁 n6/24/60/240，交叉 scaler/gamma/C 控制 | 否；来源干预方向、Source 总量、类别比例、Query 分母、ID 与抽样结构均不同 | 冻结新 ID 与协议后执行；历史准确率不充当本轮配对结果 |
| E02：复用 E01 模型，K0/1/3/5/10，Identity/Bias/Anchor/弱修正/全 Pipeline 重训 | V2/V3 已研究二分类 K-shot；V3 固定 135 Query，20 个支持 draw45–64，K0/1/3/5/10/20，多种阈值与缩放对照 | 机制问题有明显重叠，但不是相同模型与相同 ID 的完全重复 | 复用经过验证的源码和分数定义；本轮仅运行新 E01 源模型对应端点，明确不是首次发现该代数性质 |
| E03：300 纯酸枣仁 TRAIN，三次 × 五折，190 Source/50 Support/60 Query，五种统一方法 | 既有神经脚本提供相同 190/50/60 的单划分种子协议和 20 支持抽样；V4 有 alpha128/512，参考 SVM；旧传统三重复使用 240 Source/60 Query | 未找到完全匹配的新全方法矩阵。单划分、128/512 表示、旧支持顺序、方法集合与本轮协议不同 | 可以复用明确相同的划分/校准定义；未证明 ID、频点、拟合设置完全匹配的旧模型不能直接当本轮缓存 |
| E04：从 E01—E03 预测分析阈值、错误转移、覆盖×比例、聚类区间 | V3 已验证正斜率阈值解释及错误配对，旧 donor 与当前新因子结构不同 | 代数解释有重复；新预测上的量化效应与交互不是完全重复 | 引用旧机制证据并对新预测独立复算，避免新增大规模拟合 |

审计结论是“未发现匹配完整协议的可直接复用结果”，不是穷尽所有远端未公开二进制后证明它不存在。若后续发现 Source/Support/Query ID、输入数组、频点和所有参数完全匹配的已验证模型，应引用并复用其冻结结果及哈希，不重复拟合。

## 可复用的校准定义与必要对照

`AnchorScoreCalibrator` 拟合支持侧 ridge 残差：`onehot(y_support) - source_scores`，设计矩阵为 `[source_scores, 1]`，权重与截距均受同一正则惩罚；校准后分数为 `scores + [scores,1] @ delta`。lambda 固定 1 时，支持数量增加会改变数据损失相对先验的权重。输入原始分数单位会改变拟合，禁止根据 Query 调整单位或正则。

既有弱方案为 `damped_anchor`，lambda=1000，blend=0.05；其输出严格是 `scores + .05 * (anchor_lambda1000(scores)-scores)`。无需引入候选搜索。旧脚本 `species_revision_background_neural.py` 的固定定义为 `legacy_lambda1` 和 `weak_lambda1000_blend005`；原 K 为 1/3/5/10，20 次支持抽样，`SPLIT_SEED=20261031`、`SUPPORT_SEED=20261108`。该旧参考 SVM 是 128 点 SNV/Scaler/C10/balanced RBF，不应误称当前主 612 点模型。

二分类分数定义为 `[-s,s]`，`s=decision_function`。任意二列线性 Anchor 在这一直线上可写为 `margin_new=a*s+b`。a>0 时阈值为 `-b/a`，排序和 AUROC 不变；该性质不是特征学习或概率校准。a<=0 必须单列，包括排序反转和退化情形。与 Anchor 匹配的阈值对照应清楚标记是否由同一个 Anchor 拟合导出；独立 bias-only 残差方案与 Anchor 导出的等价阈值不是通用的同一个方法。

必须分别保存 `SVC.predict`、原 decision_function、二列/多列 score argmax 和校准 score argmax。尤其多分类 SVC 的原预测由投票产生，OVR 分数 argmax 在票数打平时可能不等价；零支持时各校准方法应与其分数 argmax Identity 比较，不能将改接口的差异归为校准增益。

完整重训必须重新拟合 Source+Support 的 SNV 之后的训练侧 Scaler 与 SVM；它增加训练标签并改变边界，不能归入冻结分数修正。

## 历史已验证现象与不能外推的机制

V3 二分类 K5 的 100 个 cell 斜率均为正；三个模态的 legacy 与 bias-only、正分数重标度对照在配对预测中一致。旧 alpha 的 K 曲线不单调，弱适配预测不变，完整重训优于部分冻结校准，但存在负 cell。以上是旧协议条件下的已验证现象，不保证本轮每个 cell 重现。

V2 加入 ZY01 来源覆盖同时改变数量、Source 统计和类别覆盖；V3 donor 数量对照也不是本轮固定总量的纯类别比例干预。不得从旧准确率差直接宣称厚度、仪器或物理批次的唯一机制。

历史诊断在已曝光的 master TRAIN 内开展，重复预测行不是独立样品。本轮必须按 physical_sample_id 聚类描述不确定性，保持来源仅五组、每来源仅单批且无真实 batch_id 的限制。所有结论是当前用户声明种类/来源下的内部机制诊断；不构成独立植物鉴定、未知来源或真实新批次验证。
