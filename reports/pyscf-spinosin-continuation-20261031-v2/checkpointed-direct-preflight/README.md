# V5：实际Hessian中间量保存与分块响应的工程准备

**当前只完成源码、20项纯／Mole-only测试和独立方法源码审查，没有运行water或Spinosin量子计算。缓存功能为SAVE_ONLY，不是已经实现或验证了跨进程恢复。** 新源码为`scripts/pyscf_spinosin_checkpointed_direct_worker_20261031.py`，冻结V4及历史结果未修改。

## 保持原电子结构方法和全局Hessian

物理积分后端仍是`SCREENED_DIRECT_RKS_V4`：PBE／def2-SVP／grid1／无D3、中性单重态、无DF或全四中心ERI缓存。工程版本另记`CHECKPOINTED_DIRECT_V5_SAVE_ONLY`。优化阈值和SCF条件保持不变；V4已收敛优化可作为相同积分方法的候选父结果，仍须精确几何SHA及新SCF／梯度检查，不能用DF父优化冒充。

Hessian仍为真正`pyscf.hessian.rks.Hessian`实例。只在调用期间包装partial_hess_elec／make_h1／solve_mo1，保留原bound方法并调用；原kernel、hess_elec及核Hessian装配均不改变。结束或异常时恢复原实例属性，不伪造类module。

完整global H1AO保留在内存，逐atom写到ignored cache，不stack再复制整个GiB矩阵。响应默认每块2原子，CLI可选1—4；每块向原公开solve_mo1传入完整H1AO和global atomIDs，合并到全局列表后仍由原hess_elec装配所有跨原子耦合。没有用小块kernel Hessian拼接全矩阵。

缺省fx只由原公开`rhf.gen_vind`生成一次，之后同一全AO／MO响应算子供所有块复用；已有fx则原样保留。该闭包缓存固定基态XC核，对各块密度扰动仍重新响应，不截断电子体系。

## 数值条件与未测量内容

显式保持`conv_tol_cpscf=1e-8`、Hessian max_cycle50、level_shift0；每块记录请求的有效容差`1e-8 × 块原子数`及实际Krylov默认lindep。lindep也可能触发原生停止，因此**正常返回不代表独立实测残差低于请求容差**，残差标记NOT_COMPUTED。未收敛RuntimeError不会被吞成成功。

原始及对称化分析Hessian、完整精度几何、模式及频率结果契约沿V4，完整219非零实模式和全部正频率才可判minimum。IR仍NOT_COMPUTED。真实water V4／V5完整Opt/Freq比较必须先由root执行，纯测试和源码审查不能替代这个门槛。

## 真实缓存契约

运行时才会生成：电子partial Hessian、各atom H1AO、每个完整响应块的MO1／MO_E1；数组保留在仓库`outputs/`下唯一新目录，报告只存`hessian_cache_manifest.json`，记录真实文件／数组哈希、shape／dtype、字节、时间、global atomIDs、原生返回条件及停止原因。已完成块会保留，异常重抛并标PARTIAL，部分写入文件不列为合格缓存。

缓存绑定全精度坐标、完整Mole物理状态、实际SCF能量与MO_coeff／MO_energy／MO_occ字节、实际网格坐标／权重、设置、版本及worker／原生Hessian相关源码SHA。MO相位或占据表示变化即拒绝。完整Spinosin缓存逻辑约1,311,100,200字节，启动前要求至少两倍缓存加16 MiB磁盘余量；这不是Hessian峰RAM保证。

首次额外Mole-only边界测试揭示：原生with_rinv_at_nucleus退出后保留_env[17]（AS_RINV_ORIG_ATOM最后核索引），所以“raw Mole.dumps字节应不变”的新测试假设失败。实际负对照保存在`molecule_bookkeeping_diagnostic.*`：仅17位0→1，坐标不变。V5不修改Mole，只在身份哈希的序列化副本中对**非ECP**的这一个临时索引归一，其余env（包括range-omega）、基组和几何仍完整绑定；初始raw Mole哈希另存。ECP被明确拒绝而不猜测。修订后的实际Mole-only测试检查此上下文变化可接受，而物理env变化仍改变身份。

没有LOAD入口、自动Hessian恢复脚本或伪造中间量。之后若要跨进程复用，应另行实现并验证精确MF／缓存来源恢复，不能仅凭checkpoint可读或能量接近套用响应。

## 证据

`tests_final.log`和`validation.json`记录20项最终测试、实际测试CPU及源码SHA；测试仅使用清理后的微型ignored夹具，另构造一个Mole作上下文语义核验，没有积分／SCF／梯度／Hessian调用。独立方法代理已复核全局索引、响应算子复用、原装配、容差、失败传播和非ECP上下文marker身份语义。工程准备完成不等于科研计算完成。
