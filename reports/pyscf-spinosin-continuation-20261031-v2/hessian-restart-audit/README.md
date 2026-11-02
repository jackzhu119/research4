# PySCF2.14 Hessian公开接口与分阶段保存审计

**结论：完整H1AO及完成的CPHF原子响应块可经公开接口复用；标准kernel没有自动Hessian断点恢复。中断Krylov迭代的原生恢复不可用。** 本审计只读源码、导入API元数据并做整数尺寸计算，未构造波函数／求积分，也未读活跃检查点。下述数组均为规模推导，不是已生成的科研中间量。

## 接口事实与源码位置

- `pyscf/hessian/rks.py:2108—2120`：RKS Hessian的`hess_elec`直接绑定`rhf.hess_elec`，但`partial_hess_elec`和`make_h1`使用RKS版本。
- `pyscf/hessian/rhf.py:39—61`：`hess_elec(..., mo1=None, mo_e1=None, h1ao=None, ...)`接受完整缓存；H1缺省时才调用make_h1，mo1／mo_e1任何一个缺省都会重新求响应。**partial_hess_elec在52行总是先计算，没有partial参数。** 保存partial后避免重算需要另一个经过验证的工程hook，原生参数不能直接加载。
- `rhf.py:546—550,556—572`：类方法solve_mo1保留`max_cycle`／`level_shift`；kernel只接受MO数组和atmlst，不接受H1／响应缓存，且最终加核Hessian和可选色散。PBE无色散时无需添加不存在的项。
- `rks.py:120—176`：make_h1的`chkfile`是未使用的兼容形参，没有自动保存／加载H1；`scf/chkfile.py:28—42`只保存Mole和SCF能量／MO数组。
- `rhf.py:286—341`：solve_mo1返回两个长度为全局natm的列表，按全局atom index填值，其余为None。atmlst仅选择扰动RHS；`gen_vind`在343—360行始终使用全AO／MO电子响应。
- `rhf.py:68—86`：最终hess_elec对完整atmlst的所有ij交叉项组装。**不能将kernel(atmlst=互不相交小块)的子Hessian拼成全矩阵**，那会遗漏跨块耦合。可取完整global H1AO，分小atmlst调用响应求解，按全局ID合并mo1／mo_e1，再交回原全局hess_elec。
- `rhf.py:310—332`：内部max_memory至少2000 MB，给定75／762／160规模时blksize最小113，因此默认75原子仍一次全求。外层显式分响应原子块才有意义。每块实际CPHF阈值为`mf.conv_tol_cpscf * 块原子数`；原值1e−8时，2原子为2e−8，1原子为1e−8，均比全75的7.5e−7严格。数值路径变化必须实测水对照，不可宣称逐bit等价。
- `pyscf/scf/cphf.py:29—49,122—148`：不暴露x0／checkpoint／callback；虽然底层lib.krylov有x0，它没有由此公开传入。`lib/linalg_helper.py:1308—1345`未收敛会抛RuntimeError，不能将失败块保存为成功。底层私有_Xlist临时HDF也不是可用的检查点接口。

## 精确逻辑存储估计

假设75原子、762AO／全秩762MO、160双占轨道、float64；不含NPY／HDF头、工作副本或积分暂存。

| 对象 | 维数 | 数据字节 |
|---|---|---:|
| 完整H1AO | 75×3×762×762 | 1,045,159,200 |
| AO表示MO1 | 75×3×762×160 | 219,456,000 |
| 占据子空间MO_E1矩阵 | 75×3×160×160 | 46,080,000 |
| partial或完整Hessian | 75×75×3×3 | 405,000 |
| 上四项合计 | — | 1,311,100,200 |
| RKS partial内部XC vmat | 75×3×3×762×762 | 3,135,477,600 |

`rks.py:353`的XC数组仍按全部75原子分配，即使atmlst较小；`rks.py:1117`的H1 XC数组同样按全natm分配。直接后端避免DF数十GiB强制临时矩阵，不等于所有内存开销很小。

底层Krylov实际导入默认max_memory为4000 MB；全75的RHS219,456,000字节，4000MB／RHS>14，选择incore。极保守的50轮无降秩上界，仅xs+ax可达21,945,600,000字节；2原子块对应585,216,000字节。它们不是实测峰RSS，也不是必然达到的大小。实际秩、迭代数、工作副本和网格均影响资源；完整整数表见`api_and_storage_audit.json`。

## 保存／复用的必要绑定

所有真实缓存必须绑定完全相同的全精度几何、atom／AO顺序、Mole基组／球形约定、电荷／自旋、PBE／网格／筛选设置、PySCF版本及源码哈希、SCF已收敛能量和**实际MO_coeff／MO_energy／MO_occ字节哈希**。MO1／MO_E1依赖轨道相位／子空间规范，重新SCF可能改变轨道表示；不能只凭相同能量或几何复用。

还需保存实际网格坐标／权重摘要、CPHF原始阈值及每块有效阈值、max_cycle／level_shift、完整global atomIDs、shape／dtype／哈希、结束时间和原生求解正常返回证据。独立残差若未算应明确NOT_COMPUTED。只保存已完成块，部分写入文件不能列为合格缓存。

当前自动跨进程恢复仍为**BLOCKED_NOT_IMPLEMENTED_OR_VALIDATED**；没有可运行恢复命令，也没有伪造缓存。后续工程包装可先只SAVE真实数据，经过root实测水完整Hessian对照后，再决定是否实现严格输入绑定的LOAD；本审计不修改冻结V4或正在运行的优化。

`api_and_storage_audit.json`记录实际API签名／源行、4份源码SHA及精确尺寸；`audit_public_restart_api.py`可只读复算，`audit.log`保留本次执行。所有新量子任务为0。
