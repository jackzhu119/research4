# Hessian恢复可行性审查

状态：**NOT_IMPLEMENTED_NOT_VALIDATED**。

本目录只有冻结源码和已结束水／乙醇软件验证样例的只读审查。没有实现恢复器、生成恢复命令、执行SCF／梯度／Hessian／振动分析，也没有构造Mole或重建网格。**没有读取当前斯皮诺素检查点、活动HDF5或当前Hessian缓存。** 若本轮完整频率计算顺利结束，不需要开发恢复功能。

结论是**有条件可行，但现有SAVE_ONLY尚不能直接跨进程恢复**。已保存的电子partial Hessian、H1AO和完整响应块具有复用价值；必须恢复产生它们的同一收敛波函数及实际网格，逐项核对字节与原始证据后再用原全体系装配。不能重新跑SCF后混用旧MO响应，也不能把资源中断的原任务包装成DONE。

## 本轮实际检查

仅固定读取以下已经DONE的样例，未扫描目标计算目录：

| 样例 | 缓存 | 完整H1原子数 | 完整响应分块 | 实际校验数组文件数 |
|---|---|---:|---|---:|
| 水，3原子／24AO／5占据轨道 | native4，B2 | 3 | 2＋1 | 10 |
| 乙醇，9原子／72AO／13占据轨道 | native4，B4 | 9 | 4＋4＋1 | 28 |
| 同一乙醇 | native4，B8 | 9 | 8＋1 | 28 |

共核对66个实际数组文件、2,781,072字节数值负载的路径、文件SHA、dtype／shape、有限值及C顺序payload SHA。两个已关闭HDF检查点中的`mo_coeff`、`mo_energy`、`mo_occ`、能量与三个manifest对应的binding全部精确相同；检查点中的Bohr坐标和按冻结V8规则归一化的序列化Mole哈希也一致。这里只证明保存的数据可匹配，**没有执行实际加载后求导或恢复等价性测试**。

两个实际检查点都只有：

```text
mol
scf/e_tot
scf/mo_coeff
scf/mo_energy
scf/mo_occ
```

它们均不含独立的SCF收敛标志、实际grid数组、完整RKS方法设置或CPHF迭代状态。SCF收敛与fresh梯度证据来自已关闭计算的结果、日志／事件和原manifest谱系，不能由检查点存在单独推出。

## 必须恢复同一波函数

将检查点作为SCF初猜再调用`mf.kernel()`不是这里讨论的恢复。即使能量近似相同，新SCF也可能改变轨道相位、简并空间旋转、占据排序或本征值；旧MO1／MO_E1不能和不同轨道混用。

未来加载器需要：

1. 只处理已结束父进程和冻结的输入快照，保存原真实DONE／PARTIAL／缺worker结果状态、预算账、原日志／监督证明及文件SHA。不得读取活动文件。
2. 加载**实际产生这些Hessian缓存的频率阶段检查点**的完整Mole和同一MO三数组、能量，验证所有shape、float64、占据、有限值、连接／基组／AO顺序、Bohr坐标及payload SHA。不能仅因几何相同就改用更早优化阶段的检查点；其轨道规范未必相同。避免从舍入XYZ重建另一份几何。
3. 从冻结的原设置恢复PBE、def2-SVP、grid1、无DF／无D3、电荷／自旋、直接积分与筛选阈值、CPHF参数。检查点不保存完整RKS设置；默认构造RKS会使用另一默认泛函，不能依赖它猜测原方法。
4. 核对原实际SCF收敛及同一几何fresh梯度文件，重算保存梯度的统计并核对源hash和阈值。若原总体Hessian任务PARTIAL，但SCF／梯度已完成，应明确记录这些分阶段证据，而不是把原任务改为DONE。
5. 恢复的库对象可以依据原收敛证据带上需要的`converged`状态，报告必须写成“恢复已收敛波函数／复用已验证原fresh梯度”；不能说新执行了SCF、优化或fresh梯度。
6. 在原Hessian调用前保证完整MO已加载，阻止缺失`mo_energy`时基类kernel自动调用`base.run()`。不得用隐式新SCF填补缺失数据。

如果上述任何一项不匹配，恢复应BLOCKED，或另立完整重算；不能放宽hash、改轨道或只比较能量使旧缓存“通过”。

## 实际网格是关键缺口

V8 binding保存actual grid坐标和权重的精确fingerprint，但不保存其数组；标准SCF检查点也不保存。`grid_level=1`本身不等于完整网格定义，还涉及径向方法、角向prune、Becke分区、半径调整、原子网格、排列／padding及库配置。

`non0tab`／screening table等派生筛选数据也没有独立保存或fingerprint。未来需按同一基组、实际坐标、cutoff及实现重建并通过控制样例验证；不能把两个网格数组hash当作所有RKS内部状态已经证明相同。

本地PySCF2.14的`small_rho_cutoff`默认是0，已关闭验证日志也记录0。因此**不能声称本项目一定做了first-density剪裁**。源码在该选项大于1e-20且存在二维密度时才启用这类剪裁；配置启用时，最初密度还可能成为额外重建依赖。这与默认角向`nwchem_prune`不是同一件事。

未来可尝试从同一Mole、相同安装库和完整设置生成网格而不执行SCF；只有重建的坐标和权重**逐字节匹配原两个fingerprint**才允许继续。失败必须BLOCKED。本轮没有做该生成或声称它已经通过。以后独立版本可以在fresh SCF／梯度后保存实际网格和完整网格设置sidecar，当前活动worker和冻结代码保持不变。

## 哪些中间量可复用

| 缓存 | 复用条件与边界 |
|---|---|
| 电子partial Hessian | 必须是manifest中原函数正常返回后保存的`partial_hessian`，不是最终总Hessian。加载为独立可写副本；原`hess_elec`会原地增加响应项，不能修改旧文件／只读映射。 |
| 每原子H1AO | 完整全局原子顺序、所有原子、原`H1AO_complete`及各文件hash通过后可供全体系响应／装配。缺失集合不能当作完整H1；补算／合并方案需另行验证。 |
| 完整MO1／MO_E1响应块 | 只复用已写入manifest、原solver实际正常返回、该块全部文件通过hash／shape／finite检查的块。MO1保存的是AO×占据轨道表示，不能误作完整MO空间矩阵。 |
| 未完成块／孤立NPY | 即使某个原子文件已经落盘，没有完整block的manifest封存也不能据此认定块完成。保留原文件作为失败证据，未来重算整个原分块。 |
| CPHF Krylov中间迭代 | 没有保存，不能从该迭代位置续算；只能重算未完成块。 |
| 已完整保存的原始总Hessian | 若中断发生在后续分析阶段，可另行核对其完整文件和原波函数／梯度证据后开展只读模式分析；不能把它当电子partial再加一次响应或核项。该分支本轮也未实现。 |

继续缺失响应时必须保留原global atom分组、B值、max_cycle50、level_shift0、`conv_tol_cpscf=1e-8`、native lindep及每块请求容差`原块原子数×1e-8`。不能把剩余原子重新随意分组后，默称同一数值协议。“原solver正常返回”仍不代表独立测得残差已经低于请求值；原`NOT_COMPUTED`应保留。

原RKS使用RHF的`hess_elec`。它从完整partial开始，对全部原子对加入H1／MO1／MO_E1／重叠导数交叉项；原kernel随后添加**一次**核Hessian。未来可采用保存的原方法工程hooks返回已验证数据并计算缺失原块，保持原完整global装配。不能对`kernel(atmlst=局部块)`的子Hessian拼接成全体系矩阵，也不能漏掉跨块耦合。

## 双来源绑定应如何表达

现有binding含`worker_script_sha256`、完整settings／versions／input_evidence、原监督证明SHA和方法源码SHA。新恢复器文件的SHA必然不同，因此现有整字典相等检查不能原样充当新加载器；但也不能删除来源字段、整个settings或网格项来“解决”不匹配。

未来需要明确的只读源绑定与新消费者绑定：

- 原manifest、原full binding和binding SHA原封不动保存，注明producer run／phase／worker SHA。
- 新resume脚本SHA、root permit、新监督证明、输出路径和实际预算另行记录。
- 物理／数值不变量显式列出并逐项严格核对：Mole／坐标、MO三数组和能量、实际网格、PBE／基组／电荷／自旋／无DF／无D3、筛选、CPHF、分块、质量阈值和对应实现来源。先保持线程／内存等原数值执行设置；更改它们需要另立等价性验证。
- 每个复用数组记录其旧生产者；新补算数组记录新生产者，二者指向同一已恢复波函数和网格。不能让新工程版本字段覆盖旧生产者身份。
- 原fresh梯度与优化父证据保留原hash／原状态；新报告分别标明恢复、复用和真实新计算。IR强度仍为NOT_COMPUTED。

当前SAVE_ONLY没有源／消费者双schema、严格加载器、实际grid sidecar、缺失块合并器、自动恢复测试或成功的跨进程等价性实验。因此本报告不提供恢复命令。

现行V8还会在cache目录或manifest已存在时直接拒绝，保护历史文件；把旧目录再次传作新输出并不会恢复。未来应把旧缓存作为只读输入，新计算写入另一个新目录。

数值payload目前位于ignored `outputs/`；manifest中的hash不能代替这些实际文件。如果工作区或大缓存丢失，能追溯来源也不等于仍可复用。跨环境恢复还需要单独核实实际payload可获得性和合适的备份容量，不能擅自把约GiB级临时缓存塞入既有限额的Git交付。

## 任何未来实现前的验证门槛

应先在这些已关闭水／9原子乙醇控制样例进行独立受限实验：完整缓存重装配，以及故意保留完整若干响应块、其余块真正补算的跨进程测试。实际检查恢复的MO／几何／网格字节、新SCF调用数为0、全部跨块矩阵条目、raw／sym Hessian、全部模式及固定阈值；对错误波函数、网格、源hash、缺文件和半块必须拒绝。测试是软件验证，不得当作斯皮诺素机制证据。

若未来目标阶段资源中断，仍须等其父进程闭合并保存真实CPU、旧finisher完成实际末报告与交付，保留旧closed ledger／release／结果快照，再决定是否在同一V3剩余pool／原deadline内用新独立permit追加恢复。不能重置账本、热改当前限制或无限重试。若完整频率计算顺利完成，这些准备不应触发额外计算。

## 源码与证据

- `closed_fixture_observations.json`：66个实际缓存数组校验结果、HDF结构、精确source SHA及源码位置；没有当前目标缓存信息。
- 冻结V8：`hessian_cache_binding`约475行，partial／H1／response hooks约576／591／611行，原global调用约678—735行；几何／波函数门槛与输入checkpoint来源也保留于同文件。
- PySCF `scf/chkfile.py:25—42`：SCF检查点字段；`scf/hf.py:159—162,1923—1941`：Mole写入及SCF保存。
- `dft/rks.py:330—343,499—519`：方法／grid初始化和条件密度剪裁；`dft/gen_grid.py:649,715`：网格构造与剪裁入口。
- `hessian/rhf.py:42—89,286—360,546—572`：全体系电子装配、全局响应和核项；`hessian/rks.py:120—176,2108—2119`：RKS H1和共用装配。
- `execution_record.json`：首次全局Python缺少h5py、随后用现有运行环境成功只读检查的实际经过。失败发生于导入阶段，尚未读取任何输入。

只读检查的实际命令（**不是恢复命令**）：

```bash
PYTHONPATH=.runtime/quantum-upgrade-site .runtime/bin/python reports/pyscf-spinosin-continuation-20261031-v3/hessian-resume-feasibility/audit_closed_fixtures.py
```

脚本拒绝覆盖本次检查结果；复核者应在新的审查目录保存输出。没有提出新算法、没有声称恢复已经通过，也没有修改原始光谱或分类模型。
