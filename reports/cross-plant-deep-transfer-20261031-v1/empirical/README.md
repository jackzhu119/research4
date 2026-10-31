# 酸枣仁跨植物神经迁移：有限实测隔离诊断

本诊断与主试验分开，固定原outer0：684条source、171条query、原二内折。种子42；四条路径A从头训练、Aextra当前酸枣仁source额外监督训练、B人参、C黄芪／天麻／人参。保留共同128频率网格、网络结构和预定源内选择规则，诊断只把目标训练预算改为2／4轮、预训练改为4步。

每次完整重跑16个内层与4个最终管线；真实原标签、外层真值置乱陷阱重跑以及10次source标签置乱共240个管线。含180段预训练、720次预训练优化器更新。每个目标训练batch、源标签及数值指纹、source分类头、预训练权重、BN重置和最终权重均留存。主实验的候选与预算事先固定，不根据本诊断分数改动。

- C：原始与置乱的query真值文件在训练前分别冻结，实际字节摘要不同；拟合中真实读取探针必须被拒绝。重新执行全部内层选择与最终训练，比较全部20个模型的源选择、权重、标准化器、BN、预训练头、优化器初始状态、批次顺序／数值指纹和最终预测。
- D：拟合后只改query第1行第1点＋0.01，验证模型状态和其余170条预测／分数不变。明确不冒充完整D源重训。
- H：10次在原两内层query块内部置乱当前source标签，保持每块类别计数／稀少类覆盖和固定成员；内验证标签也来自这份置乱source映射。外部植物标签不变，各次均重新执行完整选择与训练。
- 各次source与模型／预测锁先保存；全部12次完成后，独立`score`进程才解析原始query标签，报告五种角色、source众数对照及null经验尾概率（分辨率1／11）。

命名PythonI/O钩子覆盖builtins.open、io.open／Path.open及os.open，拟合／选择／推理中拒绝query真值读取，评分只读；不是操作系统沙箱，不拦截已有文件描述符或任意原生扩展。准备过程会读取标签以生成冻结文件，完整性检查可读取文件字节摘要，不把这些操作说成完全不可访问真值的沙箱。

## 冻结及执行

[预检记录](preflight_validation.json)与[10项合成测试完整日志](unit_tests.log)。测试包含真实命名文件读取／写入拒绝、稀少类安全置乱、共同训练轮数选择、策略回退，以及最后一个null的优化器轨迹损坏必须在任何query标签解析前拒绝。合成测试不等于240个真实模型已经执行。

`prepared/frozen_manifest.json`绑定主代码、诊断脚本／测试、主协议／准备／运行清单、source/query/辅助域数组和两份query真值文件。实际训练仍需root出具与其SHA绑定的release；所有历史数据／结果不变。

```bash
python scripts/audit_cross_plant_transfer_empirical.py execute --output reports/cross-plant-deep-transfer-20261031-v1/empirical/prepared --release <root_release.json>
python scripts/audit_cross_plant_transfer_empirical.py score --output reports/cross-plant-deep-transfer-20261031-v1/empirical/prepared
```

真实执行完成后另增结果记录和审计，不覆盖本次预检时的0fit状态。任何结果仅限本折、低预算、未知物理等价性的工程探索，不证明主试验所有模型绝对无泄漏或真实新批次改善。

## 实际执行完成

[完整已执行结果与限制](EXECUTED_RESULTS.md)。上述预检0fit快照保留；真实240管线、独立进程评分和逐ID复核另有证据，不改写预检状态。
