# PySCF方法、资源路径与保存结果核查

本目录是源码审查及保存结果的独立后处理。**本审计没有启动任何SCF、核梯度、Hessian积分或几何优化任务**，不重复消耗主量子化学预算。合成Hessian只检验投影／质量加权代码，不作为真实分子频率输出。

## 当前检查结果

| 项目 | 状态与证据 |
| --- | --- |
| PBE／def2-SVP及DF-RKS导数实现 | 源码存在；梯度辅助基响应True、Hessian辅助基响应2；见[method_audit.json](method_audit.json) |
| geomeTRIC成功判定 | 用`kernel`的布尔值；`optimize`只返回结构，不能据此认证收敛 |
| 核梯度范数 | geomeTRIC的RMS为每原子三维梯度向量范数的RMS，即`norm(G)/sqrt(Natoms)`；不是3N分量RMS；见[resource_path_audit.json](resource_path_audit.json) |
| D3(BJ) | PySCF需额外`pyscf-dispersion`插件；D3 Hessian源码为梯度中心差分，不是解析Hessian。本轮选定PBE明确无D3，不冒称已计算色散 |
| 频率／正常模式 | PySCF标准质量加权并投影平动转动；`imaginary_freq=False`保留有符号负频率。非线性Spinosin预期219模式，不是已完成219模式 |
| IR强度 | 标准`harmonic_analysis`源码保留TODO，故NOT_COMPUTED，不从振型伪造吸收强度 |
| 小分子真实结果 | root实际执行的水Opt/Freq已完成；审计只读取保存Hessian重新对角化，三频率与全精度检查点完全一致；见[保存水结果核查](water_saved_result_review.json) |
| Cached-DF执行策略 | 只改变同一PBE／Jfit计算的积分存取方式；见[资源路径源码核查](cached_df_execution_audit.json)和[两种水计算的比较](water_backend_comparison.json) |
| Worker实际源码 | 两份Worker的输入、成功门槛、资源与恢复边界、检查点及Hessian接口已审查；见[worker_source_review.json](worker_source_review.json) |

`method_audit.json`依赖项是最早只加入旧`quantum-upgrade-site`路径的时间点快照；此后root已把geomeTRIC1.1.1和RDKit装入独立`quantum-pilot-site`。最终安装与版本以主报告`environment/`记录为准，初始缺项不覆盖重写。

## 实际水结果与独立复算

direct-J水计算PBE／def2-SVP、grid1：优化器和最终SCF收敛，原子梯度RMS为1.3675974×10⁻⁴、最大原子梯度2.3104086×10⁻⁴ Hartree/Bohr，均满足已记录门槛。三个真实正常模式为1605.3559866、3688.8362991、3787.6695292 cm⁻¹，虚频0；IR为NOT_COMPUTED。

独立审计用SCF检查点中的全精度分子和保存的Hessian重新计算，频率差为0；正常模式按质量度量正交，允许本征向量任意正负号。最初审计器从12位小数XYZ重新构建分子后采用过严的频率容差，产生1.36×10⁻⁴ cm⁻¹的投影差异，原代码和失败日志保存在`water_review_attempt01_xyz_precision/`。随后改为使用权威检查点几何，未改原量子结果；XYZ坐标舍入误差小于10⁻¹² Bohr也记录在结果中。

cached-J水计算也完成Opt/Freq，同一方法的两种执行路径最大差：能量1.42×10⁻¹⁴ Hartree、核梯度1.02×10⁻¹⁴ Hartree/Bohr、Hessian1.50×10⁻¹¹ Hartree/Bohr²、投影频率9.64×10⁻⁵ cm⁻¹。这是数值实现smoke比较，不是Spinosin频率完成或实验物理精度证明。cached路径确有71×300、170400字节的内存张量。

## 大体系存储风险与限制

球谐def2-SVP实际基组定义给出H5、C14、O14个AO。因此Spinosin为762 AO，皂苷A／B为1646／1442 AO。这些是基组规模指标，不是已经运行的皂苷DFT成本测量。

Spinosin的Jfit辅助基2459函数，若存储完整秩压缩三中心张量需5,718,709,416字节；JKfit3831函数需8,909,465,544字节。pure-PBE原路径在`_cderi is None`时自动采用直接J，不保存大张量。cached V2用`mol.incore_anyway=True`进入库内缓存DF路径，逐几何重置，内存参数8000 MB，并在最终新SCF前释放上一缓存；仍需父进程硬内存、CPU及磁盘监督。此标志不能随意用于普通HF／混合泛函的精确K路径，否则可能触发不可承受的完整四中心积分。

PySCF默认`_compatible_format=False`的outcore blocked路径只写一份张量；兼容格式路径会另写全尺寸中间文件，在初始9.6GB可用盘上不可假定安全。缓存构建内存分析是源码公式推算，未替代真实峰值RSS。所有巨型派生积分都不应推送Git。

## 解释门槛

Hessian响应方程默认最多50步，`conv_tol_cpscf`为1e-8并乘当前原子块数。库Krylov迭代不收敛会抛RuntimeError，Worker会在频率完成前保留PARTIAL；本审计没有独立重算响应残差数值。

Worker的`DONE`可能表示频率生成完成。物理最小值还需检查优化器／最终SCF／最终梯度通过、完整模式数、有限值及**全部频率严格为正**；仅“虚频计数0”不能排除精确零模。当前水三个频率均严格正。Spinosin需要其自身优化和Hessian证据，不能由水测试替代。grid1为明确的低成本pilot，未做网格收敛或固态作用验证；禁止按实验候选峰调整频率或认定成分归属。

可复现审查脚本和真实命令日志都在本目录；它们仅后处理数据，不启动新量子任务。
