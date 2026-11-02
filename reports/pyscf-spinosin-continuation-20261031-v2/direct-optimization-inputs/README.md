# 几何与来源转移准备

状态：READY_INPUTS_ONLY_NOT_EXECUTED。新增量子计算0，未启动worker，未读取检查点。

用途：optimization；原阶段状态：DONE。选择：optimized_fullprecision_with_fresh_gradient。

这是几何和来源转移，源能量／梯度只与选中的源几何配对，不能赋给另一方法。目标方法必须重新SCF与核梯度；优化用途使用新geomeTRIC，不恢复历史。PARTIAL仅取最后完整步，不能视为驻点。频率用途保持原优化XYZ及完整精度NPY的字节SHA，同时仍须目标worker的新SCF／梯度检查。
