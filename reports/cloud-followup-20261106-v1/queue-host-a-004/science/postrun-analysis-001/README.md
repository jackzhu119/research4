# 实际闭合队列与ROOT审核复用结果的异常汇总

本报告只汇总已保存结果；未启动量子计算、训练、安装或Git，未修改原账本或历史结果。

原Spinosin结果仍为219模式＝194正频＋25虚频，稳定极小点及峰归属资格未通过。

| 项目 | 状态 | 实际结果状态 | 实际量子CPU秒（含该项目失败尝试） |
|---|---|---|---:|
| spinosin-residual-block01 | DONE | SAVED_RESPONSE_RESIDUALS_MEASURED_NOT_CONVERGENCE_GATED | 1437.415673 |
| spinosin-residual-block02 | DONE | SAVED_RESPONSE_RESIDUALS_MEASURED_NOT_CONVERGENCE_GATED | 1433.623072 |
| spinosin-residual-block03 | DONE | SAVED_RESPONSE_RESIDUALS_MEASURED_NOT_CONVERGENCE_GATED | 1358.2048559999998 |
| spinosin-residual-block04 | DONE | SAVED_RESPONSE_RESIDUALS_MEASURED_NOT_CONVERGENCE_GATED | 1263.7824269999999 |
| spinosin-residual-block05 | DONE | SAVED_RESPONSE_RESIDUALS_MEASURED_NOT_CONVERGENCE_GATED | 1266.2734619999999 |
| spinosin-residual-block06 | DONE | SAVED_RESPONSE_RESIDUALS_MEASURED_NOT_CONVERGENCE_GATED | 1267.143394 |
| spinosin-residual-block07 | DONE | SAVED_RESPONSE_RESIDUALS_MEASURED_NOT_CONVERGENCE_GATED | 1213.554378 |
| spinosin-residual-block08 | DONE | SAVED_RESPONSE_RESIDUALS_MEASURED_NOT_CONVERGENCE_GATED | 1261.891898 |
| spinosin-residual-block09 | DONE | SAVED_RESPONSE_RESIDUALS_MEASURED_NOT_CONVERGENCE_GATED | 620.3073989999999 |
| spinosin-orbital-stability | PARTIAL | FAILED_CLOSED | 8881.818771999999 |
| spinosin-m001-h010-plus | DONE | DISPLACED_SCF_GRADIENT_COMPUTED | 5400.307901 |
| spinosin-m001-h010-minus | DONE | DISPLACED_SCF_GRADIENT_COMPUTED | 5301.771959000001 |
| spinosin-m001-h010-fd | DONE | DIRECTIONAL_FINITE_DIFFERENCE_ANALYZED | 未计入量子账本，不当作0 |
| spinosin-m001-h005-plus | DONE | DISPLACED_SCF_GRADIENT_COMPUTED | 5455.584538 |
| spinosin-m001-h005-minus | DONE | DISPLACED_SCF_GRADIENT_COMPUTED | 5624.295814 |
| spinosin-m001-h005-fd | DONE | DIRECTIONAL_FINITE_DIFFERENCE_ANALYZED | 未计入量子账本，不当作0 |
| spinosin-m026-h010-plus | DONE | DISPLACED_SCF_GRADIENT_COMPUTED | 5628.118890000001 |
| spinosin-m026-h010-minus | DONE | DISPLACED_SCF_GRADIENT_COMPUTED | 5517.808869 |
| spinosin-m026-h010-fd | DONE | DIRECTIONAL_FINITE_DIFFERENCE_ANALYZED | 未计入量子账本，不当作0 |
| spinosin-residual-block00 | DONE | SAVED_RESPONSE_RESIDUALS_MEASURED_NOT_CONVERGENCE_GATED | 1379.9283249999999 |

## 实际方向曲率

单位均为Eh/Bohr²；位移按Cartesian欧氏单位向量、步长为Bohr。下表保留符号，没有事后通过阈值。

| 来源/项目 | 步长 | 解析曲率 | 梯度差分曲率 | 能量差分曲率 | 精确符号不同 |
|---|---:|---:|---:|---:|---|
| ROOT_AUDITED_REUSED_EVIDENCE/spinosin-m001-h010-fd | 0.01 | -0.0375223030225 | 0.00489378759125 | 0.00577995706408 | True |
| ROOT_AUDITED_REUSED_EVIDENCE/spinosin-m001-h005-fd | 0.005 | -0.0375223030225 | 0.00489447762515 | 0.00535546860192 | True |
| ROOT_AUDITED_REUSED_EVIDENCE/spinosin-m026-h010-fd | 0.01 | 0.000182408261888 | 0.00325451787973 | 0.00326338522427 | False |
| REUSED_A002_WATER_CONTROL/water-c0-fd | 0.005 | 0.549184330869 | 0.615037648674 | 0.613154946905 | False |
| REUSED_A002_WATER_CONTROL/water-c4-fd | 0.005 | 0.210191218132 | 0.209937473193 | 0.210194051533 | False |
| REUSED_A002_WATER_CONTROL/water-c8-fd | 0.005 | 7.02736745684e-06 | -4.90479906814e-05 | -0.00143767465488 | True |

Rebuilt atom-centred grids with gradient_grid_response=False do not form an exact common fixed-quadrature finite difference. A discrepancy alone does not identify its cause. Exact sign disagreements near zero are not a physical significance test. No new PASS/FAIL threshold is set.

水控制中的相对差及近零分量符号也如实保留。DONE表示计算完成；原stock完整Hessian对照通过，不等于本次FD数值通过。

## 响应与轨道检查

实际唯一核位移残差覆盖为 225/225；重复方向不重复计数。残差仅测量，未设置新的收敛PASS/FAIL阈值。
轨道根逐一报告native收敛标志。仅原始stable标志或正根不足以判定已收敛，内部电子稳定性也不证明核坐标极小点。

## 证据及边界

小文件逐个重算SHA。CHK/HDF等检查点和大数组只记录原生产者封存SHA及当前大小，未读其内容，不能称已独立重哈希。
所有失败成本保留。IR强度NOT_COMPUTED，无峰归属、无新分类准确率；96.0234%保持既有内部交叉验证语义。标准品、HPLC/LC-MS及独立新批次仍NOT_RUN。
