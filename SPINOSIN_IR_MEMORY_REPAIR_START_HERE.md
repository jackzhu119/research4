# 斯皮诺素网格响应内存修复（2026-11-12）

原C003／C008两路计算在北京时间20:28被系统OOM强制结束，自动重启后仍在重复初始SCF。检查时尚无已提交优化梯度、完整Hessian或IR结果。CPU时限均为unlimited，退出并非预算或运行时长阈值触发。

根因是PySCF2.13完整Becke网格响应一次分配`(75,75,3,42978)`数组，单份约5.40GiB；旧数组、下一份数组、AO导数及另一进程同时驻留时超过本机16GiB。此前内存评估没有覆盖这一分配路径。

两个当前电子态checkpoint已保全并核对原子次序、几何、762AO和320电子。它们只作为新的fresh SCF初猜，不继承梯度、驻点或IR资格。旧尝试、日志及原科学代码均保留；旧进程经保全后人工停止，以免再次重算和OOM。

修复在新目录、新代码指纹下执行：保持PBE／def2-SVP／原网格和完整响应，按网格块计算Becke导数与AO导数，不降低网格或放宽资格标准。C003先验证第一个真实75原子梯度；通过后C008自动放行，没有等待超时或CPU预算终止。

最新核查快照（北京时间2026-11-15）：C003／C008分别有26／27次完整合格梯度评估，较上次22／22次新增4／5次；本次宿主环境重启后，已从保留检查点恢复两路后台计算。两者仍在几何优化，正式Hessian与IR未开始。最新评估包含被拒试探步，不能将其当作已接受结构或单调改善。53次完整评估的科学数组及完整结果快照已校验；[最新进展、归档说明及恢复效率缺陷](docs/SPINOSIN_IR_PROGRESS_20261115.md)提供全部数据与日志入口。运行中的科学源码、方法与资格标准未改。

```bash
cd /workspace/research4
python scripts/manage_spinosin_ir_memory_repair_20261112.py status
python scripts/manage_spinosin_ir_memory_repair_20261112.py start
```

- 原始任务入口：[斯皮诺素与IR补位](SPINOSIN_IR_GAP_FILL_START_HERE.md)
- 原始科学设计：[协议](docs/SPINOSIN_IR_GAP_FILL_PROTOCOL_20261112.md)
- 恢复目录：`reports/spinosin_ir_memory_repair_20261112/`
- [系统OOM证据](data/research/spinosin_ir_memory_repair_20261112/oom_diagnosis.json)
- [数值对照与22项测试](data/research/spinosin_ir_memory_repair_20261112/engineering_validation.json)
- [恢复进程凭证](data/research/spinosin_ir_memory_repair_20261112/launch_receipt.json)
- [电子初猜保全包清单](data/research/spinosin_ir_memory_repair_20261112/recovery_archive_manifest.json)

实际运行状态以新管理器和最新原生日志为准；心跳、SCF迭代、完整核梯度、优化收敛及IR完成是不同阶段。

## 2026-11-13授权与续跑（历史快照）

用户追加100,000 CPU小时（360,000,000 CPU秒），已登记为授权而非执行倒计时。最近宿主环境重启后，两路已按原冻结协议及最近完整检查点恢复并行，科学源码、方法和资格标准未改。

C003已核验5次、C008已核验4次完整合格梯度评估；其中包含试探评估，不等同于已接受的优化步。首个75原子完整梯度门已通过，C008已自动放行。最新核梯度RMS分别约0.00273310／0.00202023 Eh/Bohr，几何仍未收敛，完整Hessian与生产IR尚未开始。

[本次授权与续跑凭证](data/research/spinosin_ir_memory_repair_20261112/continuations/100000_cpu_hours_01.json)。原启动段落是历史交接快照，实时状态以管理器为准。

## 2026-11-14新增授权与续跑

用户本轮追加900,000 CPU秒，即250 CPU小时，已在共享授权账本中单独登记；累计授权为1,191,024 CPU小时（4,287,686,400 CPU秒）。该记录不代表平台预留资源，也未引入CPU、墙钟、重试或指标自动停止条件。平台暂停／重启仍可使本机进程退出，不能保证不中断。

本次已确认旧进程不存在、检查点及首梯度门仍在，再启动两路独立后台监督进程；新的两路工作进程均存活、有新心跳且实际消耗CPU。[本轮授权、22／22梯度核验与续跑凭证](data/research/spinosin_ir_memory_repair_20261112/continuations/900000_cpu_seconds_20261114_01.json)。
