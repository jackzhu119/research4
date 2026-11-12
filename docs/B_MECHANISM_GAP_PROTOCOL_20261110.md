# B 方案补充研究冻结协议（2026-11-10）

基础提交：`b051733989161517a68c4a60cce75766880c95f5`，基础分支 `research/paper-evidence-compendium-20261109`。
本轮仅写 `research/b-mechanism-gaps-20261110`，不修改 main、不合并、不覆盖历史结果。

## 授权与计算安排

本轮依据附件研究说明及用户最新补充：“在训练开始之后确保能在后台自行训练不要中断，不要设置阈值导致哪里不符合就自动结束了训练……结束这个聊天让训练在后台自己推进”。因此原附件 10,000 累计 CPU 秒停止条件由最新指令覆盖：本轮没有准确率、耗时、CPU、试跑成绩或重试次数的自动终止阈值。记录独立 CPU 账本，不继承或重置旧 Worker B 预算。只执行下述固定矩阵，不新增大型搜索、神经网络、PySCF、Gaussian 或无关外部数据。

数据身份、实物 ID、角色互斥、输入/代码/工件哈希不属于成绩阈值，仍须核验。完整性错误拒绝不安全恢复并留下 BLOCKED 依据；单项可隔离失败先记录，再推进其余独立任务并重试缺失任务，不更改样品或协议获取正结果。

数值库固定单线程。E00 后先运行一个完整可复用任务作为 pilot；pilot 仅验证工程路径，不依据成绩挑选是否运行矩阵。随后独立会话监督器启动完整队列，完成后自动汇总、独立指标复算、核验、限定范围提交及推送。监督器脱离聊天/终端，但云实例暂停或销毁时没有持续执行保证；恢复环境后用 `--resume` 复用完整工件。

## E00：数据及历史核对

唯一入口为 `data/active_workbooks.json`，固定角色为 `data/research/species_revision_20261108/split_manifest.csv`。检查 18 份工作簿 SHA-256、1,672 个唯一实物 ID、种类数量 364/500/403/405、999 TRAIN/334 CAL/339 PRED，保留两次独立实物声明的不同依据。ZY01/ZY02 为枣仁；ZY03—ZY07 为酸枣仁。

仅按有效清单重建缓存到新目录 `reports/b_mechanism_20261110/.cache/`，不覆盖历史缓存审计。检查谱列映射、递增坐标、有限数值和40个独立读取器抽检单元格。记录 NPZ 字节哈希和每个数组内容哈希；字节哈希不同不直接推出数组相同/不同。E00 报告历史对照依据见 `B_MECHANISM_EXISTING_EVIDENCE_AUDIT.md`。

真实 batch_id 缺失保持缺失；不从来源名生成。所有新实验只从 master TRAIN 构造角色，原 CAL/PRED 不用于本轮训练或模型选择。

## E01：固定样本量来源覆盖 × 类别比例

目标为 ZY03—ZY07，抽样种子42/43/44。每个目标/种子先按排序后的 ID 做确定性 PCG64 排列，目标酸枣仁和 ZY01 分别固定20 Query、10 Support。所有6个条件复用相同角色 ID。

每次 Source240：酸枣仁/枣仁分别为60/180、120/120、180/60。枣仁仅用 ZY01；无覆盖时酸枣仁来自其他4组；有覆盖时等量用20份目标来源独立样品替换其他来源。其他来源和 ZY01 使用同一预先随机序列的嵌套前缀，覆盖替换只改变20份酸枣仁。

共90个源模型。模型为612原始频点 alpha、逐谱 SNV（ddof=1）、仅训练侧 StandardScaler、RBF SVC（C=10、gamma=scale、class_weight=balanced、固定模型种子42），复用仓库 `fit_source_candidate`。不调参、不做变量筛选。覆盖条件属于已覆盖来源的内部识别，不称未知来源或跨批次泛化。

## E02：固定源模型 K-shot

每个 E01 模型直接复用，K=0/1/3/5/10每类，从锁定10份/类 Support 的同一随机顺序取嵌套前缀。方法为 Identity、独立 Bias-only（onehot与源分数残差的惩罚截距，lambda=1）、Anchor lambda1、弱 Anchor lambda1000/blend0.05、Source+Support完整流水线重训。

Anchor 和弱校准复用 `AnchorScoreCalibrator`，原源模型保持不可变。完整重训每次重新拟合 SNV 之后的训练统计量和分类器。每个源模型新增4个非零K完整拟合，K0不重复拟合。

保留原生 `SVC.predict`、二分类 decision_function 符号、类别对齐分数 argmax、校准分数。冻结校准对照用匹配的 argmax Identity；重训对照用匹配原生预测。K0各自精确一致，接口变化单独报告。二分类检查 margin_new=a×margin_old+b；a>0时核验排序、AUROC、阈值等价；a<=0单列。独立 Bias-only 与由 Anchor 推导的等价阈值不混为同一估计器。

## E03：纯酸枣仁五来源

仅300份纯酸枣仁TRAIN，按 record_id 排序后，外层 StratifiedKFold5/shuffle，重复种子20261031、20261108、42。每折60Query（每类12），剩余每类48，确定性 SeedSequence([重复种子,折号,20261110]) 排列，前10作Support池，其余38作Source。合计190Source、50Support、60Query。

同一源模型/外折支持各方法使用相同嵌套K，采用与E02一致五种主要方法和固定SVM。15个源模型；总源模型105，总实际完整流水线拟合525（105源模型+420支持重训），K0重用。每类shot和总标签量分开记录。二分类与五分类预算、样品组成不同，准确率差不是类别数的单因素因果效应。

## E04：保存预测上的机制描述

逐实物对齐分数、真值及预测；报告覆盖/比例的配对差值和交互、每目标来源、错误转正确/正确转错误、K曲线非单调性、重训增量、逐类Recall代价以及margin与错误的描述关系。主要二分类因子效果采用原生源模型K0，校准收益采用匹配规则对照。

描述性区间以 physical_sample_id 为聚类单位，自助抽样保留每个实物的重复预测，固定抽样种子和1,000次重复；列明预测行数、唯一实物数、抽样结构。5来源及来源/批次混杂不足以识别来源总体或跨批次因果作用，重复开发和多重比较不产生确认性主结论。全部正、负、零效应保留。E04不额外训练。

## E05：新批次准备

仅交付模板、导入验证器和采样协议，状态固定 `NOT_RUN_NEEDS_NEW_BATCH_DATA`。具体字段、最低150份与优先450份独立新样品计划及非功效论证见 `reports/b_mechanism_20261110/new_batch_protocol.md`。不虚构新批次评价。

## 工件与复现

机器冻结协议包含完整逐任务ID和源码/输入/数组哈希；`split_locks.csv`逐ID登记角色。每任务写临时目录，完整后原子改名并保存结果/工件哈希与 COMPLETE 标记。恢复先核验，完整任务不重复训练或计数；失败尝试单独留在执行账本。

```bash
python scripts/run_b_mechanism_gap_20261110.py --dry-run
python scripts/run_b_mechanism_gap_20261110.py --prepare
python -m unittest discover -s tests -p test_b_mechanism_gap_20261110.py -v
python scripts/run_b_mechanism_gap_20261110.py --pilot
python scripts/background_b_mechanism_20261110.py --launch
python scripts/background_b_mechanism_20261110.py --status
python scripts/run_b_mechanism_gap_20261110.py --resume
python scripts/summarize_b_mechanism_gap_20261110.py --resume
python scripts/summarize_b_mechanism_gap_20261110.py --verify
python scripts/run_b_mechanism_gap_20261110.py --verify
```

`DONE`须有完整工件及核验；`PARTIAL`为仅完成部分；样品缺口写`INFEASIBLE`；完整性阻塞写`BLOCKED`；未采集真实新批次不升级为已运行。正式结果由后台读取保存预测生成，不在协议中预填结论。
