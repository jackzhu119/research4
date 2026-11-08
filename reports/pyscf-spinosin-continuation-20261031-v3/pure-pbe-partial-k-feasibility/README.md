# Pure-PBE partial Hessian：跳过未使用K收缩的静态预案

状态：**NOT_IMPLEMENTED_NOT_VALIDATED**。本目录只有源码／ABI证据分析及后续验证要求，没有新worker、库补丁、SCF、梯度、Hessian、当前C调用定位或速度试验；没有读取活动产物。当前计算保持不变，缓存恢复仍NOT_IMPLEMENTED。

## 当前确切行为

PySCF2.14 `hessian/rhf.py:99—178`的`_partial_hess_ejk`接收`with_k`，函数体没有读取它。RKS在`rks.py:66—69`传入`with_k=hybrid`，之后使用`e1 + ej - hyb*ek`。本项目纯PBE的`hybrid=False、hyb=0`，但安装实现仍计算三组K形状的收缩并填充`ek`。这些额外收缩不意味着PBE模型使用了精确交换。

仅删K的候选需要保持原J和一电子／重叠项完全相同，并保持默认`with_k=True`路径。它与“启用导数筛选”是两个独立工程实验，不能混成一次未经分解的改动。

## 最小候选编辑位置

以下是未来修改清单，**不是已实施补丁**。仅在已核实的纯PBE／`with_k=False`路径选择J输出；True路径可完整委派原实现，避免改变RHF／hybrid行为。

| 原代码位置 | False路径保留的原始J绑定 | 只在True路径保留的K工作 |
|---|---|---|
| 120—126，公共`int2e_ipip1` | `comp=9`、`aosym='s2kl'`、`'lk->s1ij', dm0`，取唯一返回项并仍reshape为`(3,3,nao,nao)` | `'jk->s1il', dm0`及`vk1_diag` reshape |
| 141—146，每原子`int2e_ip1ip2` | 原`shls_slice`、`comp=9`、`aosym='s1'`、`'ji->s1kl', dm0[:,p0:p1]` | `'li->s1kj', dm0[:,p0:p1]`、`'lj->s1ki', dm0`及`vk1[:,:,p0:p1] += vk2` |
| 148—155，每原子`int2e_ipvip1` | 原`shls_slice`、`comp=9`、`aosym='s2kl'`、`'lk->s1ij', dm0` | `'li->s1kj', dm0[:,p0:p1]`、`vk1 += vk2.transpose(0,2,1)`及K reshape |
| 159、166、175 | `e1`／`ej`所有原累加和转置始终执行 | `ek`对角／交叉累加及对称拷贝只在True执行 |

`_get_jk`收到单元素脚本列表时仍返回列表，候选需要正确取唯一J项，不能误解为原来的多返回tuple。

必须保留：

- `dm0=2*Cocc*Cocc.T`和能量加权`dme0`；完整`get_ovlp`与`hcore_generator`，不把一电子项当作可丢弃的K。
- `vj1[:,:,p0:p1] += vj2.transpose(0,2,1)*0.5`，再对完整`vj1`进行原reshape。`vj2`本身包含原子切片，不能随意改成全AO矩阵或去掉0.5。
- `ej`对角乘2（158行）、原子对乘4（165行），`e1`重叠项乘2（160、167行）、原core项（169—170行），以及原全局原子顺序和对称拷贝。
- 返回仍为三个float64数组`(e1,ej,ek)`，每个shape为`(len(atmlst),len(atmlst),3,3)`。False路径的`ek`保留同shape有限零数组，不能返回None，因为RKS仍会计算`hyb*ek`。
- RKS自身XC导数、H1、CPHF、完整全体系装配和一次核Hessian不变。无需改RKS返回约定，也不能顺便改网格、泛函、筛选阈值或收敛阈值。

75原子完整partial的源码调用数仍是`1+2×75=151`轮`_get_jk`。原151个J绑定和226个K绑定中的K可成为删除候选；这不是删除226次独立全积分。C代码先生成一个壳层积分，再循环各输出做收缩。积分遍历依然存在。

## 筛选与实际ABI边界

独立ABI探针已实际加载当前`libcvhf.so`句柄并只做符号查询，没有调用导出数值函数、构造Mole或执行积分。对`_make_vhfopt`检查的三个精确裸名称，`hasattr`均为False，后缀`_sph`／`_cart`均为True；库和源码SHA保存在`../non-df-pure-j-abi-audit/`。

因此，在该确切构建下，`rhf.py:180—183`的三次helper都会在构造前返回None。这是**实际ABI可用性＋源码控制流**结论，不是对当前活动C调用参数、位置或耗时的采样。与旧静态成本报告的措辞差异见[独立附录](DERIVATIVE_OPT_ABI_ADDENDUM.md)，旧文件保持原样。

第一版仅删K候选应保持这个已核实的筛选路径，不同时改变suffix guard。对于另一个会返回非None的构建，不能未经验证宣称删K不影响筛选：`_vhf.direct_bindm:491`会对传入的DM列表调用`vhfopt.set_dm`，改变输出列表也可能改变density-condition构造。

即使保留当前None路径，输出数量变化也会改变native每输出缓存、flush／归并顺序和OpenMP工作分配，不能预先保证J逐位不变。需要实际数值等价性门槛。普通SCF／CPHF的优化器及libcint内部筛选不能据此一概说成关闭。

## 仅修suffix guard是否安全

**尚不能认定安全。** 后缀符号存在仅证明可解析，不证明原来没有进入的分支已经满足全部数据形状和筛选约定。

`_VHFOpt.set_dm:224—244`把收到的列表做`numpy.asarray`并交给`CVHFnr_dm_cond1`。原`ip1ip2`绑定密度是“原子列切片、同切片、完整DM”，`ipvip1`是“完整DM、原子列切片”，通常不是规整的全局DM堆叠。仅删K后，`ip1ip2`仍只有`nao×nao_atom`的切片；C端`optimizer.c:472—492`却按完整`nao×nao`密度索引。静态约定显示必须先处理筛选密度与收缩切片的区别，不能让截断缓冲被当作全局DM；本审查没有执行该分支或把预测异常当作实测结果。

未来启用官方导数optimizer的实验应先明确完整全局screening DM、原J／K收缩slice、q-condition、density-condition和生命周期的约定，再进行真实小模型验证。导数optimizer构造器默认cutoff为1e-14，普通MF是1e-13；当前None路径并没有实际应用该导数optimizer cutoff，不能把两个设置混写。不要把suffix调整、density-condition修正和删K揉成一个声称“仅修名称”的改动。

## 后续真实验证要求

当前阶段结束、资源和独立协议明确后，才考虑实现新的工程版本，不修改冻结旧库／worker。至少先完成：

1. 纯语法／调用测试：False只请求原J脚本，True完整走原实现；原DM切片、因子、对称性、原子顺序和返回shape保持；任何未审ABI构建拒绝或明确回到原路径，不静默启用筛选。
2. 同一已收敛水和9原子乙醇、同一fresh MF／几何／MO／实际网格下，比较原实现与候选的`e1`、`ej`和最终完整PBE Hessian。False的`ek=0`是明确跳过未用输出，不要求它与原来多算的K相等。
3. 保持原native线程／内存、grid1、PBE／def2-SVP／无DF／无D3、筛选与CPHF参数。复用已事先固定的门槛：能量1e-8Ha、梯度1e-8Ha/Bohr、全raw Hessian最大差1e-6Ha/Bohr²、全部频率最大差0.01cm⁻¹；不得看结果后放宽。
4. 检查全原子对及跨块耦合、raw／sym矩阵、全部水3／乙醇21模式；正频极小点与数值等价性分别报告。记录每轮实际CPU、wall、RSS、调用绑定数、失败和负频，不以一次有利耗时作普遍加速结论。
5. “仅删K”“仅启用／修正导数筛选”“两者组合”分成独立对照。任何筛选启用还需证明完整DM约定、q/dm条件正确，不能拿删K实验替它担保。

移除K输出不会按输出比例降低native线程的约3.2GB scratch预算，因为其内部size_limit会按输出数重新分配。降低输出／收缩量有潜在价值，但在导数积分、XC或其它阶段主导时，实际节省可能有限。**没有已验证的加速比例、内存改善或新斯皮诺素频率结果。**

## 证据

- `static_plan.json`：逐处保留／候选删除清单、源码SHA、AST中with_k未读证明、ABI证据引用及旧报告hash。
- `../non-df-pure-j-abi-audit/symbols.json`：三个实际裸名False／后缀True、确切共享库SHA及零数值调用范围。
- `DERIVATIVE_OPT_ABI_ADDENDUM.md`：新增解释，不覆盖旧成本报告。

本预案不会触发新计算，不提供可立即运行的“优化worker”或恢复命令。后续是否实施取决于当前真实任务的结束状态及剩余资源。
