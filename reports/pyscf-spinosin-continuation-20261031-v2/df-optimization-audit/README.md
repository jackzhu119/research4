# 已结束的密度拟合PBE优化独立核查

状态：**PASS_CLOSED_DF_OPTIMIZATION_STATIONARITY_AUDIT_NOT_A_FREQUENCY_RESULT**。

仅审查已退出的`spinosin_extended_optimization_001`，不要求整个计算批次结束，也不读取其它活动阶段的检查点。源阶段于客户端时间2026-10-31 23:58:49 America/Los_Angeles结束（日志UTC2026-11-01 06:58:49），exit0。新增SCF、积分、核梯度或Hessian调用均为0。

已核对20个完整几何评估点：每步75原子全精度Angstrom／Bohr坐标、坐标／梯度文件SHA、五项梯度重算、对应SCF收敛日志能量及3D立体身份均通过。全部保持可信斯皮诺素`VGGSULWDCMWZPO-ODEMIOGVSA-N`。geomeTRIC driver确有`Converged!`，与优化收敛标志一致。

最后回调几何点能量为−2210.4474787483687 Eh；其后的新SCF能量为**−2210.4474787341223 Eh**，新梯度原子RMS **0.00007407006626 Eh/Bohr**、最大原子范数 **0.00022965282016 Eh/Bohr**，分别低于0.000300／0.000450阈值。两次梯度独立保存，不能混用它们的数值。

优化后XYZ及全精度NPY与结果SHA一致。只在确认此源阶段结束后读取其`calculation.chk`；其SHA为`f7e03d489704a3e8a2b21e576bf0d131efe08ed147393c4faf02dae8b1915df7`，与结果一致。检查点几何与优化NPY匹配，75原子、320电子、电荷0／spin0、762×762 MO系数、有限轨道能、闭壳层占据总数均通过，SCF能量对应最终新SCF。检查前后文件哈希未改变。

20次评估加最终新梯度，共21次SCF缓存释放／完成梯度事件均核对通过。源阶段父进程实际CPU11089.683443秒，墙钟3561.682139秒；这些是该阶段计时，不是最终批次累计。

这证明在PBE／def2-SVP／grid1／无D3、密度拟合DF-J近似下获得了通过核查的收敛驻点几何。**尚未计算斯皮诺素Hessian或频率，不能证明正频率极小值、全局最低构象或化学峰归属。** 取消密度拟合的另一计算仍是DFT，但须独立核验其驻点，不能直接沿用这里的DF状态。

`completed_df_optimization_review.json`包含20步详细重算、源路径／哈希、检查点轨道数组指纹与已退出阶段证明。`ledger_snapshot_with_closed_source.json`只是审计时的批次快照，别的阶段可以仍在运行；本审查不声称整个批次已经完成。
