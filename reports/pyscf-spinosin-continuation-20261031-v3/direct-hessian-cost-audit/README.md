# 75原子／762 AO解析Hessian的静态计算成本审计

**状态：SOURCE_ANALYSIS_NOT_NEW_QUANTUM_RESULTS。当前计算应继续按既定监督器运行；本目录不判断其当前kernel、百分比或完成时间。**

审计范围是已安装PySCF2.14.0的`hessian/rhf.py`、`hessian/rks.py`、普通SCF／CPHF相关源码，以及仓库V8源码`pyscf_spinosin_parallel_hessian_worker_20261031.py`。V8源码SHA-256为`acfe7708748ac237c9912aa9b1d98e047d842794fd0c6f08b1a67b993f5b277d`。未导入PySCF，未创建分子或调用任何积分／量子程序，未读取活动运行目录、NPY数组、HDF、checkpoint、动态进度或进程状态。源码版本绑定不能替代监督器对活动进程实际执行身份的核验。

## 结论

解析Hessian包含二阶显式核坐标导数、一次H1构建、反复的电子轨道响应CPHF，以及全局组装。最终225×225矩阵很小，但形成它需要处理大量四中心导数积分、全局AO矩阵、XC网格导数及多组响应密度，不能按一次SCF或一次梯度的时间线性外推。

**当前安装版partial Hessian不会因为`with_k=False`自动省去K型收缩。** `rks.partial_hess_elec`把`hybrid`传入`rhf._partial_hess_ejk(..., with_k=hybrid)`，但后者的函数体根本没有读取`with_k`。静态AST核验其body中该名字的Load引用数为0。它仍请求J和K形状的收缩及`ek`，之后纯PBE以`hyb=0`使交换能贡献不进入RKS最终组合。这是实现中的额外工作，不表示纯PBE物理模型包含精确交换。

H1与CPHF的情况不同：RKS非hybrid H1分支确实只请求J脚本；CPHF的非hybrid响应也确实跳过K与远程交换分支。因此“纯PBE省K”只能逐阶段依据源码判断。

## 四中心调用地图

| 阶段 | 积分核／分量 | 全局或原子切片 | PBE下真正的J／K分支 | 显式VHFOpt |
| --- | --- | --- | --- | --- |
| partial的对角公共项 | `int2e_ipip1`，9分量 | 全局一次，四个壳层腿均为全分子范围 | 同时请求J、K型脚本 | 调用`_make_vhfopt`；所需原生符号不可用时允许返回None |
| partial逐原子项 | `int2e_ip1ip2`，9分量 | 每个原子一次；第一壳层腿为该原子，其余3腿全局 | 同时请求1个J及2个K型脚本 | 复用`ip1ip2_opt` |
| partial逐原子项 | `int2e_ipvip1`，9分量 | 每个原子一次；第一壳层腿为该原子，其余3腿全局 | 同时请求J、K型脚本 | 复用`ipvip1_opt` |
| H1AO | `int2e_ip1`，3分量 | 每原子一次；第一壳层腿切片，其余3腿全局 | 非hybrid分支仅2个J脚本；K分支确实不执行 | RKS调用处未传该参数，默认None |
| CPHF响应算子 | 普通无核坐标导数的`int2e_sph/cart`，1分量 | 全局AO响应密度；按扰动分批，不截断电子空间 | 非hybrid分支仅J；K与长／短程K确实跳过 | 普通SCF的缓存VHFOpt、J-only预筛选 |

对于完整75原子的**一次**partial函数，源码循环给出`1 + 2×75 = 151`次Python层`_get_jk`收缩调用，包含151个J脚本绑定和226个仍被请求的K形状脚本绑定。不同脚本可以共享同一轮积分计算，不能把绑定数当成独立全积分次数或速度倍数。筛选后实际计算多少壳层四元组、各调用耗时、CPHF迭代次数均未在此测量。

VHFOpt建立还有额外准备：`_make_vhfopt`使用`int2e_ipip1ipip2`、`int2e_ip1ip2`、`int2e_ipvip1ipvip2`等辅助积分与普通`int2e`建立q-conditions；每次传入新密度时，`direct_bindm`更新密度筛选条件。源码以原生符号是否存在为条件，本审计没有动态加载库以确认本轮各helper的实际执行次数。

## 筛选和“direct”需要分清

- partial显式传入导数专用VHFOpt；`rhf._get_jk`经`_vhf.direct_bindm`传到native driver。它不等同于自动沿用MF对象的普通SCF optimizer。
- 导数`_make_vhfopt`新建`_VHFOpt`时未传`mf.direct_scf_tol`，安装版构造函数默认值为`1e-14`；V8给普通MF设置的值是`1e-13`。不能把两个来源写成同一个筛选设置，也不据此修改它们。
- H1的非hybrid分支未传导数VHFOpt，`direct_bindm`因而传空CVHF optimizer；native入口仍可创建libcint integral optimizer。这不能简化成“全部底层筛选都关闭”。MF的`direct_scf=True`也不会自动给这个独立导数helper补上传参。
- CPHF经`mf.get_j`→V8 override→`hf.SCF.get_jk`，明确保留普通SCF的Schwarz／密度筛选，并使用J-only prescreen；每次响应密度会更新条件。V8清空并禁止全局`_eri`缓存，不走普通RHF可能采用的incore全ERI路径。
- direct避免存储整个AO四中心张量，仍需要按筛选壳层四元组反复产生积分与收缩；它不等于无四中心成本。

## CPHF为什么依然昂贵

`rhf.gen_vind`把每组MO响应转换为全局`(nset,nao,nao)`对称密度矩阵，然后调用RKS响应函数。纯PBE每轮仍需XC核网格积分`nr_rks_fxc`与普通Coulomb J响应。`cphf.solve_withs1`的Krylov迭代反复调用它，结束后还会再计算一次响应；静态源码不能给出本次迭代或总调用数。

V8创建一次共享的全局响应算子，按扰动原子分块调用原生solver；全部AO／MO空间都保留。B8只限制一批核扰动数量，不把分子或电子响应局域化。V8对partial和H1调用的是原始全局函数，**先完整返回，之后才保存**；partial没有按B8切片，H1逐原子保存文件也不代表H1已逐个返回或当前执行到哪个原子。

## 尺寸算术，不能当作RSS或耗时测量

以下只使用任务给定75原子、762 AO及320电子闭壳层160占据轨道；没有打开活动分子、轨道或Hessian数组。NMO=762的项明确假设完整秩MO空间。

| 数组或假设张量 | float64形状 | 名义大小 |
| --- | --- | --- |
| 最终笛卡尔Hessian | 75×3×75×3 | 405,000字节 |
| RKS二阶XC全局累积`vmat` | 75×3×3×762×762 | 2.9201 GiB |
| 所有原子的H1AO | 75×3×762×762 | 0.9734 GiB |
| 一个9分量AO矩阵 | 3×3×762×762 | 0.0389 GiB |
| 8原子响应批密度 | 24×762×762 | 0.1038 GiB |
| 8原子MO1批，假设NMO=762 | 24×762×160 | 0.0218 GiB |

这些量不一定同时存在；表中不含native scratch、网格AO导数、临时矩阵、Krylov数据及分配器开销，不能相加后声称测得峰值RAM。相反，完整未收缩762⁴四中心张量仅作反事实示例，约2511.94 GiB，**V8并不分配这个张量**，不能把它称为本次真实内存需求。

PBE属于GGA。`rks._get_vxc_diag`使用到三阶AO导数，`_get_vxc_deriv2`和`_get_vxc_deriv1`使用二阶AO导数及XC二阶核；这是额外的网格计算与大矩阵累积。V8的3000MB上下文只调整数值分块策略并保留相同网格，PySCF部分路径仍有2000MB分块下限；它不是所有全局数组或RSS的硬上限。

这些源码尺寸可帮助后续选择内存和经验证的计算路径，但不提供本次ETA，也不承诺增加线程、改变B8或某个新方法能带来特定加速。DF和direct路径的资源／数值差异需要分别验证，不能把一种结果重标成另一种。本次不建议热改、停止或替换正在运行的冻结V8。

## 行号与文件

- `hessian/rks.py:60–80`：`with_k=hybrid`及`de2 += ej - hyb*ek`。
- `hessian/rhf.py:99–178`：partial完整函数及所有J／K型脚本；`180–208`：导数VHFOpt准备；`261–283`：转发driver。
- `hessian/rks.py:141–172`：H1 hybrid与非hybrid真实分支。
- `hessian/rhf.py:286–359`、`scf/_response_functions.py:44–101`、`scf/cphf.py:122–137`：全局密度响应与反复CPHF。
- `scf/hf.py:2124–2164`、`scf/_vhf.py:370–410,482–550`：普通J-only direct与derivative direct的区别。
- V8源码`576–606,611–652,678–702`：partial/H1完整计算后保存、响应分块和原生全局组装；`930–944`：强制普通SCF direct路径；`767–817`：native thread guard保留原始参数。

`source_line_excerpts.txt`保存带精确行号的关键源码片段；完整来源SHA见`source_hashes.json`。`audit.json`包含AST检查、静态调用地图和限制；`four_center_call_map.csv`可直接用于方法讨论；`dimensional_cost_examples.json`保存未取整尺寸。它们都不是运行日志或新计算结果。

重现静态检查请使用新目录，命令不导入PySCF、不访问活动产物：

```bash
.runtime/bin/python reports/pyscf-spinosin-continuation-20261031-v3/direct-hessian-cost-audit/inspect_static_hessian_cost.py --output /tmp/research4-direct-hessian-source-audit
```

native guard只记录密度数／分量数等签名，且相同签名可能对应不同积分核，首次签名事件也不是每个原子的进度。不能用旧的`density_count`／`components`事件断言当前卡在某个kernel、核实进程死锁、计算百分比或倒推出剩余时间。
