# 独立附录：源码中的导数optimizer尝试与本构建实际可用性

状态：**DONE_STATIC_PLUS_ABI_AUDIT；NO_NUMERICAL_VALIDATION**。

本附录补充已封存`../direct-hessian-cost-audit/`，不修改其README、JSON、索引或冻结源。旧报告准确注明未动态加载库确认helper实际执行；本次增加了ABI证据，因此可以把条件描述收窄到当前构建的实际guard结果。

| 旧静态报告表述／字段 | 本次补充的准确解释 |
|---|---|
| “调用`_make_vhfopt`；所需原生符号不可用时允许返回None” | 当前精确库句柄对三个被检查裸名均返回False，helper在182—183行提前None。 |
| “复用`ip1ip2_opt`／`ipvip1_opt`” | 源码复用的是这两个变量；在当前构建中它们的值为None，不能说成已成功创建并复用了导数筛选对象。 |
| `explicit_vhfopt: true`、显式传入VHFOpt等静态标记 | 代表源码存在该参数／调用路径，不等于运行时必定有非None对象。 |
| “VHFOpt建立还有额外准备”、q-condition和density-condition描述 | 是成功构造时的分支说明。本构建的guard提前返回，不能把相应helper准备当作本轮已经执行的工作或耗时。 |
| `direct_bindm`更新density-condition | 仅非None分支会调用set_dm；这三个当前helper路径不进入该分支。普通SCF／CPHF路径和libcint内部机制另论。 |

实际探针只用`ctypes.CDLL`加载确切库并做`hasattr`，不调用任何导出数值函数；dlopen可能执行一般库初始化器。这不是对当前worker的C参数／调用位置／积分数量进行探测。

| `_make_vhfopt`收到的精确guard字符串 | bare `hasattr` | 对应`_sph` | 对应`_cart` |
|---|---|---|---|
| `int2e_ipip1ipip2` | False | True | True |
| `int2e_ip1ip2` | False | True | True |
| `int2e_ipvip1ipvip2` | False | True | True |

关键二进制SHA-256：

- `libcvhf.so`：`5d7461e3976fd40fce7c9ba571e45b01830dd018d75927e389c32a6f4d09df0c`
- 依赖`libcint.so`：`950bb0a297507369dce95b9013ef2c5b64b0fa36a0018c303c98e95efce7db7e`
- `libcgto.so`：`337d956486a156f1aa7ca636ac5a91c6277058ffedb6db2efe714fee65f726e0`

源代码`hessian/rhf.py` SHA：`2d986bfc01aac98447034a207df639b3d6897f67ce1ab2a013fe2d0853c85de5`。完整来源和探针索引见`../non-df-pure-j-abi-audit/`，其index SHA为`6f62bfa2cffc39abf435f0188ef96d4a76c241941ce4375428c49fff21951a83`。

该结论只适用于上述已核实二进制与源码组合。不泛化其它PySCF构建，不表示所有底层筛选关闭，也不证明当前某个native调用正处在哪一项。它不包含速度／内存改善或数值等价性结果。

后缀符号存在不足以授权“只改suffix就启用筛选”：原DM列表有全局与原子切片形状差异，而set_dm的C端要求完整密度。任何未来实现必须先审清这些约定，并经过真实水／乙醇完整Hessian对照；当前worker、源和限制均不改变。
