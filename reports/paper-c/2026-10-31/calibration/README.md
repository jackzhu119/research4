# 保守少样本校准：辅助实验

本实验服务于酸枣仁主任务，属于约10%的适配辅助工作。**不能将其当作96.02%主模型的校准结果**：这里沿用另一个已存在的513训练／171校准池／171固定预测协议，基础源为旧SNV-SVM及MC20-SVM；主模型R2属于855份5外折／2内折OOF。

## 冻结设计与已完成的源开发

协议为[paper_c_calibration_protocol_20261031.json](../../../../data/research/paper_c_calibration_protocol_20261031.json)。全部855份及历史预测结果已用于既往研究，本轮是单批次回顾性探索，不是独立确认或真实跨批次实验。

- 基础模型：SNV、源标准化、可选MC20、balanced RBF SVM，C10、gamma scale、seed42。最终复用原513份源模型，权重不改。
- 原513份内部固定30个任务，每任务307源／103支持池／103开发查询；前24任务选择，后6任务门禁；每任务20次各类1-shot。
- 新机制使用既有λ1锚定Ridge校正，按源top-two间隔尺度裁剪，再乘固定0.25；仅低源间隔样品允许调整，支持错误证据不足时严格保持原预测。
- 12候选：cap比例0.05／0.1／0.25，源间隔门控分位0.5／0.75，最少支持错误1／2。所有阈值在源内计算，不拟合查询批均值或尺度。
- 按源选择Macro-F1配对增益排序；要求准确率增益>0.1个百分点、Macro-F1增益>0.001、Balanced Accuracy损失不超0.5个百分点，并通过最后6任务同门禁，否则identity。

实际完成60个新源模型拟合、14,400个候选／抽样评价、1,483,200条开发预测；合计23.765秒，包含预测及落盘，不能当作纯训练耗时。**两基础源均无候选产生分类增益，已在最终查询评分前按协议退回identity。**

选择锁SHA：`6298a118306bcaab5cb85951643982625ace100cae751f9c0e0cd98bd2119b7a`，实际锁定UTC为2026-10-31T08:18:05.577725+00:00。

## 为什么本轮没有产生预测改善

不是把零增益当成了成功。源内结构诊断表明，存在11,120个重叠的draw-sample对满足必要翻类界限`原top-two间隔 ≤ 2×cap×blend`并通过支持证据门控；但这11,120对的top1／top2校正都被裁成相同偏移，原顺序不变，实际flip为0。平均支持错误证据拒绝率为13.75%。

因此这一版对原OVR分数逐分量裁剪的控制过于保守；它验证了保持预测的行为，**没有验证有效提高分类率的校准方法**。该负结果不证明所有校准都无用，也不应被包装为新的成功算法。若研究中心化残差或其他限幅方式，应另立源内协议，不能利用本轮最终查询分数追参。

## 对照与评价边界

最终预设对照为无校准、传统λ1、保守门禁策略、同结构零支持、同结构无支持标签，另保存开发预选诊断。最终每源100次固定支持抽样，每次七类各1份；ZY02校准池只有1份，每次均重复使用该份，不能声称100个独立批次或七类3／5／10-shot。

源模型使用原固定171协议的OVR argmax；两个冻结模型均`break_ties=True`，已逐源513份核对`pipe.predict`与该argmax完全一致。它与旧R0默认SVC投票规则分别记录，不能改规则制造校准增益。

新最终171评价已经在[统一放行](../evaluation_release.json)后实际执行。放行UTC为08:32:33.089261，外部driver记录实际开始08:32:54.618054、完成08:32:57.674041，退出码0；均为2026-10-31。没有从诊断外测分数替换主策略。

## 最终171份实测结果

下表为100次预定支持抽样的均值；准确率SD是支持选择变化的描述，不是独立样品置信区间。

| 冻结基础源 | 方法 | Accuracy | Accuracy SD（百分点） | Macro-F1 | Balanced Accuracy | 平均改对／改错 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 全谱SNV-SVM | 无校准 | 88.8889% | 0 | 0.861624 | 86.4286% | 0／0 |
| 全谱SNV-SVM | 既有锚定Ridge校准λ1 | 74.0526% | 11.6794 | 0.694403 | 72.7612% | 3.23／28.60 |
| 全谱SNV-SVM | 源开发门禁后的保守策略 | **88.8889%** | **0** | **0.861624** | **86.4286%** | **0／0** |
| MC20-SVM | 无校准 | 87.7193% | 0 | 0.782745 | 89.0816% | 0／0 |
| MC20-SVM | 既有锚定Ridge校准λ1 | 74.0175% | 9.2336 | 0.647505 | 74.3847% | 1.73／25.16 |
| MC20-SVM | 源开发门禁后的保守策略 | **87.7193%** | **0** | **0.782745** | **89.0816%** | **0／0** |

同结构零支持、同结构无支持标签以及预先锁定的开发诊断，全部与各自无校准的171个标签逐项一致；保守主策略的全局开发门禁回退率为100%。预锁诊断的支持错误证据拒绝率分别为62%／83%，不能与全局回退率混淆。

全谱传统λ1的100次均下降，准确率范围30.9942%—86.5497%；MC20传统λ1为1次提高／99次下降，范围45.0292%—88.3041%。不从这1次有利抽样中挑选成功结论。本轮support seed固定20261104，与旧报告抽样规则分开，不能把旧λ1均值当本轮结果。

**结论是保守策略依照源开发证据保留了原预测，并未带来少样本提升。** 没有把传统λ1的损失恢复冒充超过源模型的收益。主论文仍应以酸枣仁R2模型优化为核心。

本轮实际评价1200个方法／抽样角色、205,200条预测；源模型新增拟合0。全部源内及最终预测独立复算403,689项核对通过，数值差0；另从两个冻结源权重重新生成分数，独立重解全部最终Ridge及限幅计算，逐项重放205,200条预测完全一致。两个原源模型的171个标准预测与OVR argmax均完全一致，核对不使用query真值。共5项方法测试通过，新增Excel 111,900单元格读回核验通过，数值差最大5.33e-15。

[下载完整Excel](少样本保守校准_完整指标.xlsx)，含抽样汇总、逐抽样指标、逐类均值、逐抽样逐类和平均混淆矩阵。逐类统计以每个171份查询任务计算，再对100抽样求均值，不能将重复查询当成17,100份独立样品。

## 证据与重现

| 文件 | 内容 |
| --- | --- |
| `run/protocol_snapshot.json`、`run/run_manifest.json` | 冻结配置、源代码／输入／划分／旧源模型SHA、软件版本 |
| `run/selection_lock.json`、`run/selection_events.json` | 24／6源内选择及实际锁定UTC |
| `run/development/` | 全部60组源／池／查询分数与ID，14,400个指标及全部开发预测 |
| `run/development_budget.json` | 60次真实源拟合、全部支持列表、指纹、耗时 |
| `recompute_audit.py/json` | 独立公式重解Ridge并复算全部开发及最终预测，403,689项核对、数值差0、无新模型拟合 |
| `cap_structural_diagnostic.py/csv/json` | 源间隔界限、候选／抽样门控和实际flip |
| `clipping_toptwo_diagnostic.py/json` | 相同截断偏移的结构原因 |
| `source_vote_consistency.json` | 原源模型投票与OVR argmax一致性 |
| `query_vote_consistency.json` | 171份查询标准预测与OVR argmax无真值一致性 |
| `final_replay_audit.py/json`、`*_final_source_scores.npz` | 从冻结旧源权重重新生成分数，不调用新适配器而独立重解205,200条最终预测 |
| `run/final_predictions.csv.gz`、`run/final_metrics.json`、`run/summary.json` | 全部205,200条最终预测、逐抽样指标、稳定性与改对／改错 |
| `逐类均值.csv`、`逐抽样逐类.csv`、`平均混淆矩阵.csv` | 每类Precision／Recall／F1及混淆，不省略ZY02 |
| `table_validation.json` | Excel源指标及111,900单元格读回核验 |
| `historical_context.json` | 旧校准主选择、弱校正及传统λ1的历史证据来源，不用于重选本轮外测策略 |

原始`final_metrics.json`的`candidate_id`记录配对的保守配置，具体行为由`method`区分；传统λ1的参数仍是λ1。其`support_evidence_rejected`字段是该配对保守诊断的证据状态，并不表示传统λ1也执行了拒绝。主策略是否全局回退请看`development_gate_fallback`。

源开发命令（仅在新目录运行）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .runtime/bin/python scripts/run_paper_c_calibration.py select --protocol data/research/paper_c_calibration_protocol_20261031.json --output outputs/paper-c-20261031/calibration/run
```

最终评价命令需要含本选择锁SHA的外部放行JSON：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .runtime/bin/python scripts/run_paper_c_calibration.py evaluate --protocol data/research/paper_c_calibration_protocol_20261031.json --output outputs/paper-c-20261031/calibration/run --release reports/paper-c/2026-10-31/evaluation-releases/calibration.json
```

输入NPZ可由现有规范化准备流程重建；频率区间0.1—1.5，612点。运行时校验输入及记录的冻结源码SHA。旧数据、旧源权重、旧校准与神经适配报告均保留。
