# 最新结果与后台检查点

核验时间：2026-11-16T15:25:45.254993+08:00。按用户“终止导数计算全部算力投入到频率矩阵，取消后面的排队任务”的指令，已切换为当前毛蕊异黄酮频率矩阵专用后台。累计授权仍为289万CPU秒，原耗用账本沿用，没有新增授权或清零。

| 项目 | 状态 | 已保存进度 |
|---|---|---|
| 毛蕊异黄酮完整频率 | 唯一运行中的科学任务 | 350/396点（88.4%）；两个各2线程进程并行计算不同点 |
| 毛蕊异黄酮独立IR导数预计算 | 用户取消，已停止执行 | 已完成原件保留；停止前最近核验317/594点，不授予完整IR资格 |
| 斯皮诺素及后续IR、修复任务 | 用户取消 | 不再排队或自动启动 |
| 原儿茶酸 RI-J/grid5 | 完整独立IR数值验收通过 | 270/270逐模式点合格，已完成结果保留 |
| 原儿茶酸 direct/grid5 | 完整独立IR数值验收通过 | 270/270逐模式点合格，已完成结果保留 |
| 分类数值复核 | 已完成 | B1—5及1305＋330项已保存预测复核通过，不重建撤回论文包 |

交接前350个已完成频率点逐一SHA重验相同；原科学代码、几何、密度种子和每点2线程数值设置保持一致。旧控制进程均已停止执行，旧队列STOP保留；新监督器、独立guard及两个worker均已脱离调用终端运行，实测合计约4核。仅交接时尚未完成的点可能重算。

新的入口为 `scripts/background_frequency_only_20261116.py`，恢复和核验须使用该入口。最新启动实测见 [交接核验](frequency_only/launch_verification.json)，固定范围见 [频率矩阵协议](frequency_only/protocol.json) 与 [用户取消范围](frequency_only_user_scope_20261116.json)。原始含后续队列的入口已被此次用户指令取代，不应再启动。

频率矩阵自身需要核梯度差分；这部分仍会计算。396点完成后只组装两个步长的Hessian、分析带符号频率并重放验证，不启动独立IR、额外分子或后续科学计算。虚频及不合格结论如实保留，矩阵完成不自动授予极小值或完整IR资格。

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

通过的是所注明PBE/def2-SVP/noD3、direct或RI-J模型的孤立分子局部谐振IR数值验收。毛蕊异黄酮频率矩阵尚未完成最终验收，其独立IR及斯皮诺素后续任务已由用户取消。论文引用继续以 `PAPER_RESULTS_POLICY.md` 和 `main:paper/current` 显式清单为准；数值复核不自动授予已撤回成绩的使用许可。

## 本次上传核验与恢复

上传清单描述的是固定上传快照，快照提交号与逐工件核验结论见 [上传核验回执](upload_receipt_20261115.json)；后续实时进度文件更新时，应按该快照提交核验清单，避免把不同时间的文件版本混作一次验收。

当前运行状态由上述频率专用交接核验与实时状态给出；旧上传回执只证明其固定历史快照，不证明当前进程仍在执行。
