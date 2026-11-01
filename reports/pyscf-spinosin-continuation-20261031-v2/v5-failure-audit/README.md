# V5真实水测试的native内存失败审查

**V5实际水Opt/Freq验证失败，不能用20项纯测试通过代替数值验证。** 本目录只读取已退出的失败阶段、已结束的V4水对照及源码，新增SCF、积分、梯度、响应或Hessian求值均为0；不改失败现场、不补造`result.json`，不读取其它活动阶段的检查点。

## 已观察到的事实

- `water_direct_v5_block_validation`以C退出码1结束，实际CPU4.322317秒；地址空间上限12 GiB，配置4线程，记录峰值RSS167632 KiB。
- 原始driver两次报告`malloc(533459336) failed in CVHFallocate_JKArray`。没有Python异常文件或最终result，因为该native失败路径直接`exit(1)`，绕过Python的异常／finally。
- SAVE ONLY清单中的partial Hessian及3个原子的H1AO均已成功返回并保存，文件SHA、形状及有限值通过；共享`gen_vind`已创建。完成响应块仍为0，失败发生在首个CPHF响应阶段。清单原来的RUNNING_CAPTURE_ONLY状态保留，不冒称其仍在运行或已经完成。
- 已结束的水检查点为3原子、10电子、中性单重态、24×24有限MO矩阵，未发现畸形MO／H1形状证据。没有观测到C调用级的具体DM形状／线程团队或失败时完整VmSize，因此不把未记录的数据当实测。

## 版本源码支持的机制

官方PySCF v2.14.0的`nr_direct.c`在`CVHFnr_direct_drv`使用：

```c
// up to 3.2 GB per thread.
size_limit = (400000000 - di*di*di*di*ncomp - cache_size) / n_dm;
```

进入OpenMP区域后，每个线程对每个DM脚本调用`CVHFallocate_JKArray`，其中数据缓冲为`8 × (size_limit + 136²)`字节。Python `_vhf.nr_direct_drv`没有`max_memory`参数，所以仅减小`mf.max_memory`不控制这个C常量。

若首个双原子块是预期6个方向／DM，日志中的单次申请533,459,336字节对应每线程6份共3,200,756,016字节；4线程同时持有时共12,803,024,064字节。12 GiB上限为12,884,901,888字节，仅剩81,877,824字节（约78 MiB）容纳Python、动态库、栈和其它数组，已低于记录RSS峰本身。水的6份24×24响应密度本身仅27,648字节，大申请来自native缓冲策略，不能据此断言电子响应被错误截断。

这些是源码及整数推导，支持“native私有缓冲与当前地址空间上限不相容”的机制；未记录当时逐线程地址映射，仍需修复后的真实运行确认。malloc主要涉及虚拟地址保留，低RSS不代表没有碰到`RLIMIT_AS`。

V4同4线程水对照曾成功，不证明同线程大分子安全。源码使用`omp for nowait`后逐线程释放；24 AO水的小积分分块很少，空闲线程可能很快释放，调度、分配器碎片或实际活动团队可改变缓冲同时存活的峰值。这些是**源码支持的解释假设，尚非失败时序实测**，不把它们写成唯一已证实根因。

## 后续新版本的验证门槛

保留冻结V4／V5及失败日志，不扩大12 GiB地址空间、不改积分／筛选／CPHF精度、不截断电子响应。新版本可在进程内受控包装`pyscf.scf._vhf.nr_direct_drv`，以1个native OpenMP线程运行其原函数，完整转发参数并在finally恢复外层线程设置。这样同时覆盖SCF／CPHF和梯度／Hessian的`direct_bindm`、`direct_mapdm`路径；仅包`mf.get_jk`会漏掉导数调用。

此为新版本设计建议，**本审查没有执行修复或证明V6通过**。必须真实运行小分子Opt/Freq，记录native线程数／地址空间，再与原方法Hessian和完整模式对照，之后才进入目标计算。保存的partial Hessian／H1不是完整振动Hessian，也不是已经实现跨进程续算的证明。

## 溯源

- `native_source_snapshots/nr_direct.c/.h`取得自官方v2.14.0，并通过Git blob SHA验证；传输时多出的末尾空行已在封存前按官方blob精确校正。
- C源SHA-256：`9c181daa8ce689e312e548694f47a2935105094fab1e685e82b4bf485e0639c4`。
- 实际安装`libcvhf.so` SHA-256：`5d7461e3976fd40fce7c9ba571e45b01830dd018d75927e389c32a6f4d09df0c`。
- `native_memory_failure_review.json`保存已退出阶段证明、确切字节推导、源哈希和不确定性边界；源码版本与错误格式相符，但本审查没有重新编译二进制或取得C调用级追踪。
