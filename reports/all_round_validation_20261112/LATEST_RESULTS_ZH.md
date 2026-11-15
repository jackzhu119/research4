# 最新结果与后台检查点

核验时间：2026-11-15 15:17:07（北京时间）。后台两个量子计算进程已核验持续运行，本次上传不重启计算。实时状态见 `status.json` 和 `latest_platform_resume_verification.json`。

| 项目 | 状态 | 已保存进度 |
|---|---|---|
| 原儿茶酸 RI-J/grid5 | 完整独立IR数值验收通过 | 270/270逐模式点合格，极小值与独立IR门禁通过 |
| 原儿茶酸 direct/grid5 | 完整独立IR数值验收通过 | 270/270逐模式点合格，极小值与独立IR门禁通过 |
| 毛蕊异黄酮完整频率 | 运行中，尚未最终验收 | 257/396点（64.9%），第258点运行 |
| 毛蕊异黄酮IR导数预计算 | 运行中，尚不授予完整IR资格 | 209/594点（35.2%）；第一组198点完成，第二组11/198点 |
| 斯皮诺素修复 | 排队 | 本轮尚未启动 |
| 分类数值复核 | 已完成，不重新训练 | B1—5及1305＋330项已保存预测复核通过；不重建撤回的旧论文包 |

## 已完成IR结果原件

- RI-J/grid5：[逐模式CSV](quantum/ir/jobs/f626f77fcebb7b9e5bdf10c3725664b4754e01232361b3c209106da04b4206f6/round00000/ir_modes.csv)、[完整数组NPZ](quantum/ir/jobs/f626f77fcebb7b9e5bdf10c3725664b4754e01232361b3c209106da04b4206f6/round00000/ir_arrays.npz)、[验收结果](quantum/ir/jobs/f626f77fcebb7b9e5bdf10c3725664b4754e01232361b3c209106da04b4206f6/round00000/result.json)、[封印](quantum/ir/jobs/f626f77fcebb7b9e5bdf10c3725664b4754e01232361b3c209106da04b4206f6/round00000/COMPLETE.json)。
- direct/grid5：[逐模式CSV](quantum/ir/jobs/d2f61d359fb8e80c51d8ffdaca21289debc01c1eef6f398dea07b71e0b195f11/round00000/ir_modes.csv)、[完整数组NPZ](quantum/ir/jobs/d2f61d359fb8e80c51d8ffdaca21289debc01c1eef6f398dea07b71e0b195f11/round00000/ir_arrays.npz)、[验收结果](quantum/ir/jobs/d2f61d359fb8e80c51d8ffdaca21289debc01c1eef6f398dea07b71e0b195f11/round00000/result.json)、[封印](quantum/ir/jobs/d2f61d359fb8e80c51d8ffdaca21289debc01c1eef6f398dea07b71e0b195f11/round00000/COMPLETE.json)。

各点电子审计、导数/模式数组、方法与几何来源、三步长比较、逐模式独立SCF、刚体与质量归一检查均保留在上述科学目录。完整CSV/NPZ的SHA本次重验匹配。

## 上传与恢复

上传范围为本轮明确路径：完整科学原件、逐点审计、冻结代码/输入、已完成ML复算的无损压缩详细表、授权与CPU账本、未完成的有效断点。可再生密度缓存、进程锁、临时日志及未压缩的巨大明细留在本地，不作为新增科学结果重复上传。

可检查 [JSON快照](LATEST_PROGRESS.json)、[上传工件SHA清单](upload_manifest_20261115.json)、[CPU账本](cpu_accounting.json)与[状态](status.json)。发布保持非强制推送，不改main，不恢复用户撤回的内容。累计授权为1990000 CPU秒，原耗用不重置，平台重启间隔的不确定上界单列。

## 论文使用范围

通过的是所注明PBE/def2-SVP/noD3、direct或RI-J模型的孤立分子局部谐振IR数值验收。毛蕊异黄酮与斯皮诺素尚未完成最终验收。论文引用继续以 `PAPER_RESULTS_POLICY.md` 和 `main:paper/current` 显式清单为准；数值复核不自动授予已撤回成绩的使用许可。

## 本次上传核验与恢复

上传清单描述的是固定上传快照，快照提交号与逐工件核验结论见 [上传核验回执](upload_receipt_20261115.json)；后续实时进度文件更新时，应按该快照提交核验清单，避免把不同时间的文件版本混作一次验收。

恢复核验时间：2026-11-15T15:14:07.695407+08:00。两个量子计算进程、监督器及guard均已实际核验运行，累计199万CPU秒授权继续生效；未新增授权或重算已完成的分类矩阵。见 [恢复后运行核验](latest_platform_resume_verification.json)。
