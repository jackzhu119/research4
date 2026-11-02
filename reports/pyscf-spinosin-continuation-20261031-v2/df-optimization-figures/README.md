# 已完成DF优化的可复现图表

**DONE：仅对已结束的DF-PBE优化产物做后处理。新增量子计算0次，checkpoint读取0次。**

来源为`spinosin-extended-optimization-001/`，该进程退出码0，`result.json`为DONE，并由`df-optimization-audit/completed_df_optimization_review.json`审计。脚本只读取该已完成目录中的原始JSON／NPY／最终XYZ及已核实的初始SDF连接关系，不读取正在执行的direct任务、checkpoint或动态预算文件。

## 图表与表格

- `spinosin_df_optimization_progress.png`／`.pdf`：20次完整几何评估的能量、原子梯度RMS和最大原子梯度范数。点之间为原始序列连线，没有平滑、插值或删除非单调点。能量显示为相对第20次评估的mEh；绝对Hartree值保存在CSV。
- `df_optimization_evaluations.csv`：逐次能量、相邻评估能量差、独立重算的原子／分量梯度统计、原有门槛和源SHA。梯度统计来自每步75×3原始梯度NPY；能量来自匹配的原始step JSON。20个完整坐标数组各不相同，但不因此声称20个独立构象或实验样品。
- `final_geometry_fresh_validation.csv`：同一最终几何的新SCF／梯度复核，**仍标记为geometry20，不增加第21几何或构象**。图中空心菱形也位于横坐标20。
- `spinosin_df_optimized_geometry.png`／`.pdf`：实际最终75原子几何。显示用刚体旋转保持距离不变。键来自已核实的C001起始SDF连接关系，元素顺序及全部键长已检查；不是DFT键级分析。`display_topology.json`保存绘图连接表。
- `metadata.json`与`source_hashes.json`：方法、环境、定义、状态、来源哈希。
- `generation.log`：实际后处理输出。`files_sha256.json`封存本目录文件，自身除外。

## 数值与解释范围

首次评估能量为−2210.4403572045712 Eh；第20次原始回调能量为−2210.4474787483687 Eh。最终同几何fresh复核能量为−2210.4474787341223 Eh，原子梯度RMS为7.407006626162465×10⁻⁵ Eh/bohr，最大原子梯度范数为2.2965282016092038×10⁻⁴ Eh/bohr，低于原有3×10⁻⁴和4.5×10⁻⁴门槛。

原子梯度RMS定义为`sqrt(mean_i(sum_a(g[i,a]^2)))`，最大原子梯度为`max_i(sqrt(sum_a(g[i,a]^2)))`。它们与对225个笛卡尔分量直接求RMS／最大绝对值不同，CSV同时保留两套统计。CSV的相邻能量差门槛仅表示相邻完整评估之间的数值检查，不等同于geomeTRIC在接受／拒绝试探步过程中全部收敛条件。只有原有driver日志／审计确认最终优化收敛。

方法严格标注为**气相、density-fitted PBE/def2-SVP、grid level1、没有D3色散修正**。这是暂定驻点几何，未证明所有振动频率为正，也未证明全局最低构象。斯皮诺素频率为NOT_RUN，IR强度为NOT_COMPUTED；不绘制理论吸收谱、不作化学峰归属，也不把DF优化结果替称为no-DF结果。

## 重现

从仓库根目录运行到新的输出目录，避免覆盖此次产物：

```bash
PYTHONPATH=.runtime/quantum-upgrade-site:.runtime/quantum-pilot-site OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .runtime/bin/python reports/pyscf-spinosin-continuation-20261031-v2/df-optimization-figures/generate_df_optimization_figures.py --output /tmp/research4-closed-df-figures-replay
```

脚本验证已完成源审计状态和源文件SHA，独立重算梯度后才写CSV及绘图。准备／后处理完成不能表示Hessian、频率或酸枣仁THz机制研究已完成。
