# 直接积分与分块响应Hessian协议

本文件是当前延长计算批次的数值后端补充，沿用用户已授权的云端资源预算。原DF优化与历史结果保持原样。使用PySCF标准PBE/def2-SVP/grid1、无D3；本补充不是新泛函或新量子化学算法。

## 改变的原因与方法隔离

PySCF2.14默认DF Hessian在本分子上的一个阶段至少写入约43.525 GiB临时HDF5，本机空间不足。新`SCREENED_DIRECT_RKS_V4`采用标准屏蔽直接积分，明确`density_fitting=False`、`incore_anyway=False`、`direct_scf=True`及原screening容差1e−13，不存全四中心ERI。

这是去除RI近似的独立数值后端，不把DF的优化完成标志转授给直接积分频率。须先实际验证V4水完整Opt/Freq，再从可信几何启动新的V4斯皮诺素优化，通过优化器及最终新SCF／梯度检查后才允许频率。

## V5工程响应分块

`pyscf_spinosin_checkpointed_direct_worker_20261031.py`保留V4物理后端。原全体系`kernel`、`hess_elec`和核Hessian装配不改，仅为公开方法添加可还原的实例封装：

- `partial_hess_elec`及完整H1AO仍由原方法实际计算。
- `solve_mo1`按默认2原子的全局索引调用原公开接口，电子响应仍覆盖完整AO／MO体系。一次构建并共享同一`gen_vind`算子，不能复用上一右端项的响应。
- 将全部75个原子的MO1／MOE1填回全局列表，由原`hess_elec`装配全部交叉项。禁止把小块Hessian简单拼接。
- `conv_tol_cpscf=1e−8`、原生`lindep`策略、50循环及零level shift明确记录。名义块容差不等于独立测得的残差；残差为NOT_COMPUTED，原生失败异常必须保留并返回PARTIAL。

须实际比较V4与V5水的能量、梯度、原始／对称Hessian及3个正常模式，再用于斯皮诺素。静态测试和源码复核不能替代这项实测。

## 真实中间量与恢复边界

实际完成的partial Hessian、逐原子H1及逐块响应保存于ignored的`outputs/`唯一新目录；报告只记录manifest、SHA、尺寸和耗时。Spinosin所需这些数组合计约1.311 GB，开始前要求充足磁盘余量。它们不是SCF检查点，也不能单独当作已完成频率。

缓存绑定精确几何、基组、网格、MO系数／占据／能量及源码。非ECP目标仅在序列化副本中规范化PySCF未恢复的last-rinv-atom bookkeeping位，不修改实际Mole或积分；其它物理环境改变均拒绝。

当前模式为**SAVE_ONLY**，跨进程加载／恢复**NOT_IMPLEMENTED**。不得编造续算命令、补缺响应块或混用另一波函数缓存。异常保留已经完成的中间量；若未来增加恢复，必须另行验证来源与数值一致性。

最终频率需完整219模式、几何及新梯度检查通过、raw Hessian有限且对称误差合格。保存raw原件，正常模式分析采用明确的对称化副本；负／零频率不能通过极小值检查。IR强度NOT_COMPUTED，未计算数据留空，理论频率接近实验候选不等于成分确认。

## 实测失败后的V6原生积分线程保护

V5的4线程真实水测试在CPHF直接积分分配中原生退出，V4的4线程斯皮诺素首轮SCF也退出；失败原件及缺失worker结果保留。官方C通用驱动每OMP线程最多预留约3.2 GB缓冲，这个常量不受`mf.max_memory`控制；额外运行时映射会使4线程申请触及12 GiB地址空间。源码机制与实际分配字节相符，但小水曾成功的调度细节没有瞬时观测，不作唯一因果推断。

V6仅在独立worker进程内为整个`_vhf.nr_direct_drv`调用链添加局部OMP上限，完整转发参数与全部density，覆盖SCF、梯度、Hessian导数及CPHF。调用外还原原配置，最终还原原函数，原积分、筛选、电子体系及AS上限不变。`--native-direct-threads 1|2`默认1，不超过外层线程配置；记录的是配置最大值，不冒称真实活动团队数。2线程对斯皮诺素的资源余量仍须实测。

V6源码及15项纯防护测试通过；真实水验证是后续运行门槛，不能由源码审查替代。V4单线程水已实际通过，并用于当前直接积分几何复核。缺失`result.json`的原生退出阶段只能由已结束父进程及真实input/progress派生PARTIAL，禁止向原目录补造计算结果。
