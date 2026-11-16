# 最新结果与后台检查点

核验时间：2026-11-16T23:07:25.426142+08:00。毛蕊异黄酮频率矩阵已于北京时间2026-11-16T19:59:40.170079+08:00正常完成，当前没有科学计算进程，后续队列为空。按用户此前指令，完成该矩阵后不再启动独立IR、斯皮诺素或其他后续任务。累计授权仍为289万CPU秒，未新增授权或清零耗用。

| 项目 | 状态 | 已保存结果 |
|---|---|---|
| 毛蕊异黄酮完整频率 | 已完成并核验 | 396/396点；两个99×99 Hessian；每套93个正频、0虚频 |
| 毛蕊异黄酮局部最低点数值检查 | 当前模型下通过 | 细步长最低频率29.12898680 cm⁻¹（0.8732650551 THz）；不授予完整IR资格 |
| 毛蕊异黄酮独立IR导数预计算 | 用户取消，已停止执行 | 原件保留；停止前最近核验317/594点，未完成完整IR验收 |
| 斯皮诺素及后续IR、修复任务 | 用户取消 | 队列为空，未自动启动 |
| 原儿茶酸 RI-J/grid5 | 完整独立IR数值验收通过 | 270/270逐模式点合格，已完成结果保留 |
| 原儿茶酸 direct/grid5 | 完整独立IR数值验收通过 | 270/270逐模式点合格，已完成结果保留 |
| 分类数值复核 | 已完成 | B1—5及1305＋330项已保存预测复核通过，不重建撤回论文包 |

已核验完成封印、6个矩阵/分析工件及1980条原点/数组哈希；两套Hessian从原梯度重新组装逐值相同，分析重放186行频率误差为0。两个步长的最大频差为0.0134594561 cm⁻¹。交接前350点原样保留，两个worker均一次启动完成并返回0，没有触发后续任务。

完整科学结果已回传固定提交 `24cf1b017e2dc591cf1f6a558ffa952153d21eb4`，当前远端提交核验一致；摘要、封印、两个矩阵及4个分析文件共8项SHA与本地一致。见 [完成核验](frequency_only/completion_verification_20261116.json)、[最终摘要](frequency_only/summary.json)、[完成封印](frequency_only/COMPLETE.json)、[逐模式结果](frequency_only/frequency_analysis/all_modes.csv)。

当前恢复入口仍为 `scripts/background_frequency_only_20261116.py`。任务状态已为`COMPLETE`，本次检查没有重启计算。原始含后续队列的入口已被用户指令取代，不应再启动。

`frequency_minimum_verified=true`仅表示所注明模型下的孤立分子局部最低点数值检查通过。独立IR已取消，`IR_status=CANCELLED_BY_USER_NOT_COMPUTED`；`paper_ready_molecular_harmonic_data=false`保持原样，不把频率矩阵完成改写成完整IR论文资格。

## 已核验的历史上传快照

2026-11-15上传已确认远端提交 `37b0687e3761b2fd420c713f3c51ff59c97a078f`，核验 14905 个工件、98519891 字节。见 [最新上传回执](upload_receipt_20261115_latest.json) 与 [最新逐工件SHA清单](upload_manifest_20261115_latest.json)。清单绑定该不可变快照；后续运行状态会继续更新。

## 已完成IR结果原件

- RI-J/grid5：[逐模式CSV](quantum/ir/jobs/f626f77fcebb7b9e5bdf10c3725664b4754e01232361b3c209106da04b4206f6/round00000/ir_modes.csv)、[完整数组NPZ](quantum/ir/jobs/f626f77fcebb7b9e5bdf10c3725664b4754e01232361b3c209106da04b4206f6/round00000/ir_arrays.npz)、[验收结果](quantum/ir/jobs/f626f77fcebb7b9e5bdf10c3725664b4754e01232361b3c209106da04b4206f6/round00000/result.json)、[封印](quantum/ir/jobs/f626f77fcebb7b9e5bdf10c3725664b4754e01232361b3c209106da04b4206f6/round00000/COMPLETE.json)。
- direct/grid5：[逐模式CSV](quantum/ir/jobs/d2f61d359fb8e80c51d8ffdaca21289debc01c1eef6f398dea07b71e0b195f11/round00000/ir_modes.csv)、[完整数组NPZ](quantum/ir/jobs/d2f61d359fb8e80c51d8ffdaca21289debc01c1eef6f398dea07b71e0b195f11/round00000/ir_arrays.npz)、[验收结果](quantum/ir/jobs/d2f61d359fb8e80c51d8ffdaca21289debc01c1eef6f398dea07b71e0b195f11/round00000/result.json)、[封印](quantum/ir/jobs/d2f61d359fb8e80c51d8ffdaca21289debc01c1eef6f398dea07b71e0b195f11/round00000/COMPLETE.json)。

各点电子审计、导数/模式数组、方法与几何来源、三步长比较、逐模式独立SCF、刚体与质量归一检查均保留在上述科学目录。完整CSV/NPZ的SHA本次重验匹配。

## 上传与恢复

上传范围为本轮明确路径：完整科学原件、逐点审计、冻结代码/输入、已完成ML复算的无损压缩详细表、授权与CPU账本、未完成的有效断点。可再生密度缓存、进程锁、临时日志及未压缩的巨大明细留在本地，不作为新增科学结果重复上传。

可检查 [JSON快照](LATEST_PROGRESS.json)、[上传工件SHA清单](upload_manifest_20261115.json)、[CPU账本](cpu_accounting.json)与[状态](status.json)。发布保持非强制推送，不改main，不恢复用户撤回的内容。累计授权为2890000 CPU秒，原耗用不重置，平台重启间隔的不确定上界单列。

## 论文使用范围

通过的是所注明PBE/def2-SVP/noD3、direct或RI-J模型的孤立分子局部谐振IR数值验收。毛蕊异黄酮已完成当前频率矩阵及局部最低点数值检查，其独立IR及斯皮诺素后续任务已由用户取消；毛蕊异黄酮尚无完整独立IR资格。论文引用继续以 `PAPER_RESULTS_POLICY.md` 和 `main:paper/current` 显式清单为准；数值复核不自动授予已撤回成绩的使用许可。

## 本次上传核验与恢复

上传清单描述的是固定上传快照，快照提交号与逐工件核验结论见 [上传核验回执](upload_receipt_20261115.json)；后续实时进度文件更新时，应按该快照提交核验清单，避免把不同时间的文件版本混作一次验收。

当前完成状态由上述完成核验与最终摘要给出；旧上传回执只证明其固定历史快照，不证明当前进程仍在执行。
