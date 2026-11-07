# V6：限制原生VHF线程的资源修复准备

新worker为`scripts/pyscf_spinosin_memory_bounded_direct_worker_20261031.py`。15项纯防护测试及独立方法源码复核通过，**尚未由此代理执行任何量子任务；真实water及Spinosin必须由root按预算串行验证。** V4／V5及V5失败现场均未修改。

## 源码级故障依据

已封存的[v5失败审查](../v5-failure-audit/native_memory_failure_review.json)记录父进程exit1、CPU4.322317秒及C层`malloc(533459336) failed in CVHFallocate_JKArray`。实际partial Hessian、3个atom的H1和共享响应算子建立已完成，首个CPHF响应尚未完成；没有Hessian／频率，也没有worker result.json。C层exit绕过Python finally，原capture manifest保留RUNNING快照；它不代表仍有活跃计算，应由已结束父phase加progress派生PARTIAL，不能补写假result。

PySCF2.14的`nr_direct.c:408`使用硬编码`(400000000 - di^4*ncomp - cache_size)/n_dm`，注释明确“up to3.2GB per thread”；414行OMP parallel中每线程为全部density分配JK数组，255行按`8*(size_limit+136²)`malloc。Python`_vhf.nr_direct_drv`没有max_memory形参，因此只降低mf.max_memory不会限制这些C缓冲。

失败单次533,459,336字节×6density＝每线程3,200,756,016字节；4线程为12,803,024,064字节。12 GiB地址空间总限为12,884,901,888字节，以上缓冲后仅约78 MiB留给其它映射。这解释了低RSS仍可能地址空间分配失败。V5实际缺少失败瞬间VmSize，不能把此算式当作已测完整峰值；V4小水为何偶然成功的调度／分配时序尚无直接观测证明。

源C随安装包提供，字节与官方v2.14.0标签一致，SHA为`9c181daa8ce689e312e548694f47a2935105094fab1e685e82b4bf485e0639c4`。C源、Python入口和实际libcvhf.so哈希分别保留；这不等于已经重编译并证明二进制构建可重现。

## 最小工程修复

V6仅在本worker进程运行期间包装`pyscf.scf._vhf.nr_direct_drv`。所有参数、完整density批次、operator脚本和输出对象原样转发；进入原驱动时用公共`lib.with_omp_threads`限制native线程，调用后finally恢复原OMP配置。退出worker时恢复原Python函数，未修改安装目录C／Python／so文件，也不影响其它进程。

统一入口覆盖`direct`／`direct_mapdm`／`direct_bindm`，因此SCF、标准梯度、Hessian导数及CPHF直接J均受到保护；不是只修复mf.get_jk而遗漏Hessian导数。调用外XC／BLAS仍使用外层线程配置。任何线程／函数还原故障都会使worker状态PARTIAL，原异常不被吞掉。

CLI增加`--native-direct-threads`（长别名`--native-direct-driver-threads`），允许1／2，默认1，不能超过当前外层线程数。1线程C缓冲约3.2GB，2线程约6.4GB；H1、分块Krylov、网格和库映射额外占内存。这只是源尺寸估计，**不是Spinosin峰内存或成功保证**；地址空间12 GiB和原CPU／墙钟限制均保持，2线程须先实际water控制再决定使用。

物理后端仍为SCREENED_DIRECT_RKS_V4；PBE／基组／网格／筛选、全电子响应、CPHF条件及V5全局Hessian装配均不变。工程版本另记MEMORY_BOUNDED_DIRECT_V6_SAVE_ONLY，仍无跨进程缓存LOAD／自动恢复声明。

## 新观测与验证边界

input保存实际Python入口、C源和binary SHA；每个新density_count／ncomp组合记录进入C前的OMP配置上限、`/proc/self/statm`虚拟空间／RSS及实际RLIMIT_AS。OMP getter观测的是配置上限，不冒充并行区实际活跃worker数。最终result汇总进入／正常返回次数与还原故障，不写虚构SCI结果。

测试验证完整6-density对象不切片、所有args／kwargs透传、1／2线程上下文、异常与KeyboardInterrupt还原、重复／嵌套安装拒绝、全部标准动态入口覆盖及真实PySCF OMP配置恢复（替身native函数，没有积分）。失败缓冲算式只做整数计算，没有申请GiB内存。`tests_final.log`与`validation.json`记录15项最终测试、CPU及冻结源码SHA。
