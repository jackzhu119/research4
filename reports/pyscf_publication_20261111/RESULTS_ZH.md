# PySCF 论文数据修复结果

计算执行状态与论文数据资格分别报告。原始实验、负频率和所有失败记录保留。
RI-J 近似、辅助基组、网格及步长按实际方法记录，不改称旧 direct 方法。

|分子|执行状态|分子谐振数据资格|实际方法|
|---|---|---|---|
|acid|QUALIFIED_LOCAL_HARMONIC_DATA|通过|df-grid5|
|calycosin|PENDING_OR_CHECKPOINTED|尚未通过|待核验|
|spinosin|PENDING_OR_CHECKPOINTED|尚未通过|待核验|

可用范围：通过核验的孤立分子局部极小值及谐振/IR 预测。
不能据此认定全局最低构象、药材固体 THz 化学峰归属或成分辅助分类收益。
云环境休眠或销毁会停止本地后台；逐点检查点可在恢复环境后续跑。
