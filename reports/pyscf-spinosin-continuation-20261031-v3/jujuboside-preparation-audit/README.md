# 酸枣仁皂苷A/B准备状态审计

**状态：JUJUBOSIDE_PREPARATION_AUDIT_METADATA_DONE_3D_AND_DFT_NOT_RUN。快照：2026-11-01 13:17:47 UTC。**

原任务要求的结构来源、立体分子图、构象搜索可行性与基组规模准备，在关闭的pilot结构报告中已有较完整记录；不能把这些准备记成完成了3D构象、力场优化、SCF或DFT。此次只核查关闭的元数据／源码，并补读官方ChEBI身份字段，没有运行PySCF、RDKit、嵌入、力场或量子计算，未读取活动斯皮诺素的数组、HDF、checkpoint或进程。

## 已核实与未运行

| 项目 | Jujuboside A | Jujuboside B |
| --- | --- | --- |
| 分子式／显式加氢后原子数 | C58H94O26／178 | C52H84O21／157 |
| 重原子／中性电子数 | 84／650 | 73／564 |
| 选定PubChem CID | 51346169 | 24721031 |
| 此次直接读取的官方ChEBI | [CHEBI:177582](https://www.ebi.ac.uk/chebi/CHEBI:177582) | [CHEBI:177644](https://www.ebi.ac.uk/chebi/CHEBI:177644) |
| InChIKey | `KVKRFLVYJLIZFD-OQVAHBRNSA-N` | `OAVAUZCEOWCYCC-QEOGCQCLSA-N` |
| 形式电荷／模型多重度 | 0／1 | 0／1 |
| 已指定／未指定手性中心 | 34／0 | 29／0 |
| 重原子图strict可转动键 | 13 | 10 |
| 原有最终MMFF参数覆盖查询 | PASS | PASS |
| 已有坐标 | RDKit生成的2D描绘 | RDKit生成的2D描绘 |
| 官方3D坐标／算法3D构象 | 未取得／NOT_RUN | 未取得／NOT_RUN |
| 嵌入、MMFF优化、多构象搜索 | NOT_RUN | NOT_RUN |
| SCF、梯度、DFT优化、Hessian／频率 | NOT_RUN | NOT_RUN |
| IR强度 | NOT_COMPUTED | NOT_COMPUTED |

此次两个ChEBI页面直接返回分子式、电荷、SMILES、完整InChI及Key；其InChI与原来选定数据库记录**逐字一致**。新字段保存在`official_chebi_metadata_supplement.json`，来源回执在`new_database_read_receipts.txt`；没有改写历史registry或源结构文件。ChEBI化学数据依据其官方许可说明以CC BY 4.0注明来源。

中性、无自由基价态图支持选择闭壳层单重态模型，但这不等于实际做过自旋能量比较，也不能证明购买标准品或当前855份样品的化学身份。纯品实测THz和样品含量证据依然未核实。

## 历史记录的层次与A的歧义

- 初始`candidate_registry.json`是早期快照，还保留“结构／电荷待核验”状态；后来的`structures/graph_verification/verification.json`已确认所选A/B分子图、全立体信息及MMFF参数覆盖。较早内联`graph_identity_verification.json`中的MMFF查询标志为false，不应把它误当成最终查询结果。历史文件全部保留。
- A的CID171446与选定CID51346169有相同分子式和连接Key前缀，但前者有两个未指定手性中心，Key为`KVKRFLVYJLIZFD-LQRLABOESA-N`，不能静默等同。新官方ChEBI177582的完整InChI／Key支持当前选定的51346169候选；实际参考标准品仍需核对。旧报告也保留FDA交叉引用171446却给出全指定InChI的情况。
- B保留CID24721031及完整Key，不替换成B1或同式异构立体记录。已有旧记录同时引用官方PubChem索引、大学维护的PubChemLite及JST。
- `.mol`文件头明确为`RDKit 2D`。它们是重原子连接关系描绘，不是可直接进入DFT的完整三维构象。官方PubChem对A的索引还显示“Conformer generation is disallowed since too many atoms”；这只是该服务的生成限制，不是算法生成分子模型在理论上不可能。

## 搜索准备与计算成本依据

现有`prepare_spinosin_conformers_20261031.py`硬绑定斯皮诺素源SHA、CID155692、C28H32O15、75原子及10个手性中心，不能不经修改和审查就喂入A/B。A/B的MMFF全参数覆盖仅表示力场参数工具上可用，**不代表ETKDG已成功嵌入、MMFF已收敛、构象搜索充分或找到全局最低构象**。13／10个重原子strict转动键之外还有OH相关取向，后续搜索完整性不能照搬斯皮诺素12个候选的结果。本审计没有新增候选、改动该脚本或启动搜索。

旧`basis_resource_proxies.json`按内置def2-SVP球形基组给出H=5、C/O=14个AO；此次纯算术复核A=1646、B=1442 AO。它们是基组规模，不是分子的实际SCF／梯度运行基准。

依据已封存的安装版标准非DF RKS源码，GGA Hessian的`_get_vxc_deriv2`包含全局float64数组`(natom,3,3,nao,nao)`：

| 名义中间数组 | A | B |
| --- | ---: | ---: |
| 二阶XC全局`vmat` | 32.3379 GiB | 21.8909 GiB |
| 全部H1AO数组 | 10.7793 GiB | 7.2970 GiB |
| 预期3N−6模式数，非已算频率 | 528 | 465 |

这仅是源码形状和基组数量的精确存储算术，没有分配这些数组，也不是测量到的峰值RSS。单个XC数组就超过本项目关闭的4线程优化阶段所记录的16 GiB cgroup上限（08:20 UTC静态证据，未重新轮询活动环境）；若仍采用同一限制和未改的标准RKS路径，不能启动该完整Hessian。B8只控制后面的响应扰动批次，不会缩小这个全局数组。更小分块参数也不是该数组或总RSS的硬上限。

本审计不从AO倍数预测小时数，不把假想完整ERI张量当作真实direct内存需求；完整分子的运行基准、收敛性、可行新算法路径或更大资源仍未验证。**A/B不启动大分子DFT或频率计算，维持NOT_RUN。**

## 交付证据

`audit.json`逐项保存身份、准备状态、限制和名义成本；`readiness_table.csv`用于盘点；`source_hashes.json`记录20份关闭／静态来源及新数据库回执哈希；`readback_checks.json`记录一致性检查；`files_sha256.json`封存本目录，自身除外。没有改原始光谱、分类器、论文主文档、科学源代码或旧计算产物，也没有Git提交。
