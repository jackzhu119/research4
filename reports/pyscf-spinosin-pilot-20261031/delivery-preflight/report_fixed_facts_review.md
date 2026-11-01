# 过程报告固定事实只读复核

审阅对象：`docs/PYSCF_SPINOSIN_PILOT_20261031.md` 的计算运行中草稿。
本审计没有修改该文档，没有读取活动Spinosin检查点，也未启动量子计算。

固定事实已核对：两次水Opt/Freq确实完成，3个正频率、单位换算、
最终原子梯度范数、能量及Hessian非对称差与原始结果一致。
PBE/def2-SVP、DF-J辅助基、grid1、`grid_response=False`、无D3的描述
与代码及来源审计一致。A/B的原子数、AO数、缓存量及A未指定立体中心
的来源歧义均有相应记录；缓存体积不能解释为所有方法的最低内存要求。

已向主执行者提出以下修订，由其处理正文：

1. 原direct-J v1的PySCF内存参数为4000 MB；缓存v2才是8000 MB。
2. 梯度定义应明确为 `gRMS=||G||F/sqrt(Natom)`、
   `gmax=max_i||G_i||2`，区别于3N分量RMS和最大绝对分量。
3. 3300 CPU秒是量子阶段累计上限，另300秒是准备／测试预留；
   预留不等于逐项实测了全部非量子CPU。
4. XYZ重投影的小偏差不能归因为物理频率对极小坐标舍入敏感。
   新增只读矩阵诊断显示：原Hessian稍不对称，投影后调用默认
   `numpy.linalg.eigh` 的数值路径有影响；诊断性对称化后，
   XYZ与完整精度几何的差从约1.36e−4降至1.14e−12 cm⁻¹。
   原始Hessian、频率及全部封存结果保持原样。

数值诊断证据为同目录的
`diagnose_saved_water_projection.py`、
`water_projection_numerical_diagnostic.json` 和
`water_projection_diagnostic_command.log`。
对称化结果仅用于解释审核中的数值现象，不替换原水频率表，也不提供
DFT精度、网格收敛或Spinosin振动的证据。

Spinosin最终能量、梯度、收敛状态及累计CPU仍需等待运行阶段终止后复核。
