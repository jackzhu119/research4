# 各轮回传结果重验与后台接续

本轮日期为2026-11-12（北京时间），独立研究分支为
`research/all-round-validation-20261112`。用户初始授权990000 CPU秒，
用于盘点仓库各轮回传结果、重验已完成证据、补齐分子频率与独立IR强度验证。
这是本轮新授权，不重置、合并或挪用其他聊天的历史额度。
后续用户依次追加100000、900000、900000 CPU秒，累计授权2890000 CPU秒；幂等追加记录见
`data/research/all_round_validation_20261112/authorization_extensions.json`。

2026-11-16用户进一步指示终止独立IR偶极导数计算，将全部可用计算资源投入当前
毛蕊异黄酮（calycosin）的频率矩阵，并取消后续排队任务。当前唯一计算范围为
`calycosin/df-grid5/round000`的双步长396点频率矩阵及其矩阵装配、带符号频率分析。
斯皮诺素（spinosin）、后续分子、后续方法与构型、独立IR和其他验证队列全部取消。
此范围调整沿用原2890000 CPU秒授权及累计耗用账本，没有新增授权或清零耗用。

## 已核对的结果与当前范围

- 已盘点27个远端分支，固定提交与报告入口保存在
  `reports/all_round_validation_20261112/source_snapshot.json`。
- B机制第1—5轮、1305项历史保存预测、330项V4保存预测和初始元学习备份的本轮复核已完成，
  完成回执为`reports/all_round_validation_20261112/ml/summary.json`与`ml/adoption.json`。
  这项保存结果复核没有重新训练模型，也不授予已撤回结果新的论文资格。
  完整轮次台账见同目录`ml_round_audit.json`和`.md`。
- 原儿茶酸两套来源（RI-J/grid5与direct/grid5）已完成完整频率计算，
  本轮独立三步长偶极导数、逐模式SCF和刚体恒等式验收也已通过；
  两份独立IR结果均为`QUALIFIED`，保存在`quantum/ir/jobs/`对应来源的`round00000/result.json`。
  这些已完成结果保留，不随此次取消独立IR计算而撤销。
- 毛蕊异黄酮当前仅补齐上述396点矩阵；交接前350个已合格点原样保留并经SHA校验复用，
  回执为`frequency_only/handoff_completed_points_before.json`。
  斯皮诺素旧219模式中含25虚频，不能把计算结束当作已确认极小值；其后续计算现已取消。
  原始来源、提交、已完成量与限制见同目录`quantum_round_audit.json`和`.md`。
- 已找到旧1305项矩阵的591份预测补全归档，结合其他714份可恢复完整预测覆盖，
  逐预测指标复核现已完成。连续元学习仅回传初始检查点，
  不能凭启动报告宣称后续训练已经完成；指标复算也不等于全部模型重载。

## 后台执行约定

当前频率专用监督器只启动两个协作计算进程，每个进程保持原数值协议的2线程，
共使用全部4核；两者通过逐点锁分工计算同一396点矩阵，不重复合格点。
任务均从单独封印的频率专用代码加载；计算进程脱离调用终端，按逐点检查点恢复。
当前矩阵和频率分析完成后退出，不进入其他分子、独立IR或后续验证。
原多任务入口已由持久停止标记阻止复活，范围决策及新策略分别见
`data/research/all_round_validation_20261112/execution_scope_20261116.json`与
`data/research/all_round_validation_20261112/frequency_only_policy.json`。
只读消费上游回传时保存固定提交、来源SHA和导入记录；历史报告不原地改写。
位置迁移后的检查点属于本轮派生副本，不冒充原始字节。版本、几何、原子顺序、
方法或密度种子不兼容的缓存须重建或明确拒用。

不按准确率、运行时长、连续失败次数或无进展次数自动终止当前矩阵任务。
科学验收条件用于判定结果资格，保留不合格原因、带符号虚频及负结果。
当前2890000累计CPU秒是用户授权的资源边界；实际准备与后台消耗计入账本，
不预扣提前退出储备。资源边界保留检查点及尚未解决的问题。
环境重启后沿用原耗用账本，无法精确观测的旧环境尾部单列保守上界，
不把该上界写成实际CPU耗用。原冻结数值代码和输入保持原样；已完成ML只继承回执，
不重新执行或重建已撤回的论文材料。

监督器与独立guard记录Linux身份、心跳、代码封印及累计CPU；
状态和结果写入`reports/all_round_validation_20261112/`；统一状态入口仍为其中的
`status.json`，当前范围为`FREQUENCY_MATRIX_ONLY`且`queued_tasks=[]`。
频率专用状态、逐点进度与最终完成封印写入`frequency_only/`。
每约600秒仅对本轮明确路径保存非强制Git检查点，备份失败不会停止科学计算。
用户随后明确授权创建草稿PR，已恢复向`research/all-round-validation-20261112`
远端研究分支回传检查点，保持其他分支与main不变。计算进程没有因交付策略切换而重启。
此前的本地备份位于`/workspace/all-round-validation-checkpoints.git`。
当前交付策略及授权记录分别见`reports/all_round_validation_20261112/delivery_policy.json`
和`data/research/all_round_validation_20261112/publication_authorization.json`。
旧`extend_all_round_validation_20261112.py`不再用于启动研究队列；当前控制入口如下。

```bash
/workspace/validation-venv/bin/python scripts/background_frequency_only_20261116.py --root /workspace/research4 --start
/workspace/validation-venv/bin/python scripts/background_frequency_only_20261116.py --root /workspace/research4 --verify
/workspace/validation-venv/bin/python scripts/background_frequency_only_20261116.py --root /workspace/research4 --status
```

启动后须实际核对监督器、guard和计算进程身份、脱离会话、冻结代码、
CPU账本及科学进度，不能用历史PID或旧启动回执证明本轮已运行。

## 论文使用范围

当前论文引用资格遵循`PAPER_RESULTS_POLICY.md`和`main:paper/current`显式清单；
本轮数值复验通过不自动授予已撤回结果的论文使用许可。
不重新构建已撤回的论文包，不把历史保存预测的复核回执改作当前论文输入。

本次频率矩阵的`COMPLETE`仅表示396点、矩阵装配和频率分析已完成并通过工件封印核验。
它不自动证明极小值，极小值结论单列为`frequency_minimum_verified`；也不授予独立IR强度资格。
本次毛蕊异黄酮任务不新增独立偶极导数或逐模式IR计算，输出的`IR_status`为`CANCELLED_BY_USER_NOT_COMPUTED`。

只有实际通过原件重验、完整极小值验收及本轮独立IR验收的数据，
才能按其真实方法标为孤立分子局部谐振结果。水分子控制或软件测试不替代目标分子结果，
正频率也不证明全局最低构象、固体THz峰归属或实测成分。

分类结果按其当时样品、种类、划分与开发曝光范围报告。
真实跨批次、实验室成分或物种验证仍需要对应独立数据，不能用CPU补造。
云环境暂停或销毁会影响本地后台；本监督器不承诺平台之外的持续运行。
