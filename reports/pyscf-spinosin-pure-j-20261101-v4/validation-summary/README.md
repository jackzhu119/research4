# 实际小分子配对对照

水与乙醇均使用同一MF、完整坐标/MO/网格、同一已验证B8/native4装配，仅比较stock partial与纯J partial。真实完整Hessian及全部3/21个模式的差异通过预先固定门槛；它们不是酸枣仁或斯皮诺素结果。

每个分子仅一次配对时间测量：水的纯J case CPU较低，乙醇case CPU略高、墙钟略低；不能据此宣称大分子加速或统计显著改善。case耗时不是整个phase成本，表中另列最终wait4总CPU（包含fresh SCF/gradient及记录开销）。旧V3失败耗时与未来V4完整计算耗时不可直接换算为提速。IR强度NOT_COMPUTED。
