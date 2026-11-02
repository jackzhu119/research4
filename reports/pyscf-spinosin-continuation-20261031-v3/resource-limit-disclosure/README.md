# 资源限额披露：已结束的斯皮诺素优化阶段

这是独立追加的静态披露，依据[已封存的优化完成账本快照](../optimization-completion-snapshot/ledger_at_optimization_completion.json)及[优化结果](../spinosin-native4-optimization-001/result.json)；不修改冻结程序、原账本、结果或限额。生成时间：2026-11-01T12:18:03.828020+00:00。

**该阶段配置的21600秒CPU硬限额，未成为实际wait4用量的严格上界。** 科学优化及独立fresh SCF／梯度已完成，不等于CPU限额严格达标。

| 已结束阶段 `spinosin-native4-optimization-001` | CPU秒 |
|---|---:|
| 配置阶段额度／启动时RLIMIT_CPU hard | 21600 |
| 启动时RLIMIT_CPU soft | 21480 |
| 父进程wait4实测 user＋system | 21681.889694 |
| worker报告process CPU计时 | 21681.306838823002 |
| wait4超出配置hard | 81.889694 |
| wait4超出配置soft | 201.889694 |
| worker计时超出配置hard | 81.306838823002 |
| wait4与worker计时数值差 | 0.582855176998 |

超出hard的比例为 **0.379118954%**（`100 × (21681.889694 − 21600) / 21600`）；不是准确率变化。不同计时的统计范围／时钟不同，二者之差不能单独证明启动、收尾或某种内核机制贡献了多少时间。机器可读原值、公式与来源SHA见[disclosure.json](disclosure.json)。

父进程记录 `EXITED`、exit code `0`；归档标志 `wall_terminated`、`hard_killed`、`memory_terminated`、`monitor_failed` 均为false。真实21681.889694秒已完整计入快照，未改小或截断为21600。退出成功或没有stop标志均不能证明CPU用量未超配置。启动证明记录的soft／hard确为21480／21600。

超额的**准确根因仍未证实（UNPROVEN）**。[运行与普通CPU短测审查](../cpu-limit-overshoot-audit/closed_opt_cpu_audit.json)、[内核／时钟源码审查](../cpu-overshoot-kernel-audit/audit.json)保留了证据和限制。不同CPU时钟／计账机制属于源码支持的可能解释；没有该长阶段最终raw PROF计数及完整信号递送轨迹，不能据此确认因果。普通小进程短测不是六小时量子阶段严格限额的验证。

## 新有限预算与历史快照的范围

[冻结授权](../authorization.json)与[冻结资源协议](../../../docs/PYSCF_MONITORED_DIRECT_PROTOCOL_20261031.md)定义的是**另行授权的有限V3批次**：48 CPU小时（172800秒）、24墙钟小时，最多4线程；单阶段另有有限额度。并非无限运行，也未重置或抹去V2已发生用量。V2成本在原账本及来源链中单独保留。

本历史快照只包括已结束阶段前缀，其实测CPU合计为21869.3734439999999994秒，低于该新批次172800秒配置预算；快照所记剩余额度为150930.626556秒。**这不是当前或最终全批次成本，也不保证未来阶段不超配置。** 不把后续阶段申请额度当成实际消耗，也不把本附录准备完成当成量子计算完成。

## 地址空间、物理内存与尽力监测

历史启动配置的 **24 GiB RLIMIT_AS是虚拟地址空间上限，不是24 GiB物理RAM，也不是RSS限额**。原有16 GiB cgroup内存边界保持不变。父监视器按0.1秒采样，group RSS高于10 GiB请求终止、约1秒宽限内执行强制停止；cgroup用量高于12 GiB触发停止。其语义明确为 `BEST_EFFORT_RSS_GUARD_NOT_HARD_RSS_LIMIT`，不是内核强制的10 GiB驻留内存硬限额；快速原生分配可能在采样间隔内越界，不能保证固定空余内存。历史启动证明不是当前监视器健康状态。

已结束优化阶段记录的采样峰值group RSS为6501228544字节，cgroup为8912896000字节；这些是采样记录，不保证捕捉到所有瞬时峰值。这里只引用归档授权、协议、证明与历史快照，未读取当前系统或活动计算内存状态。

## 科学结论及本附录边界

[独立科学优化审查](../opt-native4-completion-audit/science-point-check/science_audit.json)证明真实优化器收敛并完成同方法独立fresh SCF／梯度，和上述资源超额同时如实保留。**本附录不评价频率阶段是否完成、是否满足资源限额或是否为正频极小点。** 未读取活动频率HDF、NPY、checkpoint或实时账本，未启动任何SCF、梯度或Hessian计算；没有更改阈值、信号或模型。非线性75原子分子预期219个振动模式不等于已经得到219个模式，水和乙醇的软件验证模式也不能计作斯皮诺素结果。

可复算命令（必须指定全新输出目录，已有输出拒绝覆盖）：

```bash
python reports/pyscf-spinosin-continuation-20261031-v3/resource-limit-disclosure/build_disclosure.py --output /tmp/research4-resource-disclosure-fresh
```

本目录[index.json](index.json)保存文件SHA-256；索引不包含自身哈希，其哈希由上级交付记录绑定。生成器只使用Python标准库及固定静态来源，不导入量子引擎。
