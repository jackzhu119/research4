# V5按核扰动分批求响应：只读API设计审查

状态：**DESIGN_SOURCE_REVIEW_PASS_PENDING_REAL_V4_V5_WATER_AND_TARGET_VALIDATION**。审查只读取PySCF2.14.0源码和函数默认值、计算数组整数尺寸；没有构建分子、调用SCF、积分、响应算子或Hessian，未读取活动检查点。设计可行性不等于V5已完成科学验证。

## 公共API与全部交叉耦合

`pyscf.hessian.rhf.solve_mo1`返回两个长度为`mol.natm`的列表。每个请求原子`ia`的MO响应写入`mo1s[ia]`，一阶轨道能响应写入`e1s[ia]`；未请求原子的列表元素是None。它接收的是**完整、全局索引的h1ao**，不能给两项h1ao子列表再用全局原子编号索引。

`atmlst`选择核坐标扰动的右端项，**没有截断电子子系统**。`gen_vind`始终用完整MO系数、全部占据轨道和全AO密度矩阵构建响应，再调用全体系`mf.gen_response`。因此可以将75个原子的225个扰动分成37个双原子块和1个单原子块，逐块调用原公开求解器，将返回值按全局原子编号合并为完整75项MO1／MOE1列表。

原`hess_elec`随后仍对所有`ia,ja`计算`h1ao[ia]`与`mo1[ja]`等交叉项，因而保留全部75×75原子对耦合。**不能分别执行小块`kernel(atmlst=...)`并拼成对角Hessian**；那会丢掉跨块项。缺失或重复响应块必须失败，不能补零冒充完整模式。

## 数值停止条件

保持`conv_tol_cpscf=1e-8`、`max_cycle=50`、`level_shift=0`。原`HessianBase.solve_mo1`把这些参数传到公开函数；分块wrapper应调用保存的原绑定方法或等价公开函数，避免递归调用自己的wrapper。

原生子批请求容差为`conv_tol_cpscf × 本批原子数`，所以V5双原子块名义请求2e−8，最后一个单原子块1e−8；完整75原子原生调用名义请求7.5e−7。按762个满秩MO／160个占据轨道估算，源码的2000 MB分块内存下限已经给出至少113原子一批，所以默认完整调用会把75原子一起处理。

但是`lib.krylov`停止条件是`max_innerprod < lindep`或`max_innerprod < tol²`，实际默认`lindep=1e-13`。这里的`max_innerprod`是Krylov内部试探向量范数条件，**不是已独立测得的最终响应方程残差**。正常返回只能记录native solver returned normally，不能声称实测残差已小于2e−8。最大迭代未达到停止条件时会抛RuntimeError，不能吞掉并标记成功。

改变右端项批量也改变了数值Krylov空间和停止路径。因此即使线性响应数学问题相同，仍必须实际运行V4／V5小分子对照，核验原始Hessian、频率、完整模式与最终几何，之后才可运行目标频率。V5保留SCREENED_DIRECT_RKS_V4物理积分后端，另记录响应分批实现标识；不改泛函、基组、网格、色散或筛选精度。

## 数组尺寸，不是峰值内存实测

在75原子、762 AO、762满秩MO、160占据轨道假设下，完整H1AO、全局AO-MO1、全局MOE1和partial Hessian的数值负载分别为：

- H1AO：1,045,159,200字节。
- MO1：219,456,000字节。
- MOE1：46,080,000字节。
- partial Hessian：405,000字节。

合计1,311,100,200字节，仅是这些必要结果的大小，不含其它工作区及运行时。保存H1列表时应逐原子写出，避免仅为保存再stack复制约1 GiB。

一个完整225右端项的原始Krylov向量批为219,456,000字节，双原子6右端项为5,852,160字节；单次全AO响应密度／势数组分别为1,045,159,200与27,870,912字节。若许多Krylov向量保留，分批可显著减小每个求解器的工作集，但QR降秩、收敛情况和其它积分工作区影响真实峰值；不能把这些算术当作已测资源或保证CPU更快。

## SAVE ONLY与来源绑定

partial Hessian、H1AO及每个真正完成响应块可以保存到ignored outputs，报告仅保留SHA及索引。每块必须绑定精确MO系数／能量／占据数组的形状、dtype、字节SHA，以及几何、原子顺序、计算设置、PySCF版本和worker SHA。仅能量相近或文件名一致不足以复用缓存，轨道旋转也可能改变MO表示。

默认跨进程恢复状态必须为**NOT_IMPLEMENTED_SAVE_ONLY**。保存成功不意味着有跨进程读取／恢复实现，不能提供虚构的Hessian断点恢复命令。partial Hessian和H1AO也不能被当作完整振动Hessian；只有原全体系装配完成、正常模式及驻点条件通过后才可解释频率。

`public_api_response_review.json`记录精确整数尺寸、实读默认常量、关键源码行与源文件SHA；`source_snapshots/`原样封存所审查的源码，未修改安装库或冻结worker。
