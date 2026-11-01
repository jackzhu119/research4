# 论文方法证据笔记（暂定快照）

**状态：PROVISIONAL_RECORDED_METHODS_NO_NEW_QUANTUM_DATA。快照：2026-11-01 11:56:34 UTC。**

截至本笔记快照，Hessian由主监督器报告仍在运行；本笔记未读取活动文件来独立轮询。下面把已完成的方法与待完成分析分开，不能直接整段作为最终论文结果使用。没有新增量子计算或改动历史分类器，96.0234%的既有分类指标保持原记录；这里没有重新计算该指标，也不声称DFT提高分类准确率或提出新DFT算法。

## 已有真实证据支持的方法

斯皮诺素连接关系与立体化学依据ChEBI CHEBI:81360，并参考PubChem CID155692；已核验InChIKey为`VGGSULWDCMWZPO-ODEMIOGVSA-N`。中性单重态模型为C₂₈H₃₂O₁₅、75原子、320电子。坐标由已核实的立体连接关系生成，并非实验或官方数据库下载的3D坐标。RDKit2026.09.1采用ETKDGv3生成12个候选，MMFF94s筛选均收敛，按该力场能量选择C001（种子20261032）进入此处报告的DFT计算路线；没有完成其余11个候选的DFT构象比较，不能认定全局最低构象。

先以气相DF-PBE/def2-SVP（辅助基组def2-universal-jfit）进行几何预优化，随后转为相同PBE泛函、def2-SVP基组和grid level1的非DF screened-direct模型重新优化与复核。已完成的DF扩展阶段记录20次完整几何评估，这不是该构象全部历史DFT计算次数。最终direct阶段记录5次完整评估，之后对同一最终几何重新SCF／梯度检查；fresh复核不算第6个构象。其输入坐标文件哈希与DF优化输出哈希一致。

这两个完成阶段均采用PySCF2.14.0、geomeTRIC1.1.1、SCF能量阈值10⁻⁸ Eh及4个CPU线程，无D3色散修正。direct路径使用`direct_scf_tol=10⁻¹³`并禁止完整四中心ERI缓存。原子梯度RMS及最大原子梯度范数门槛分别为3×10⁻⁴和4.5×10⁻⁴ Eh/bohr。最终direct fresh结果为−2210.4455219704696 Eh、RMS 6.252236133158532×10⁻⁵、最大值2.0415338597535757×10⁻⁴ Eh/bohr；优化进程正常退出且通过驻点门槛。**驻点通过不等于已经证明所有振动频率为正。**

## English methods draft: completed work only

Spinosin connectivity and stereochemistry were verified using ChEBI (CHEBI:81360), with PubChem CID 155692 as a corroborating reference (InChIKey: VGGSULWDCMWZPO-ODEMIOGVSA-N). A neutral singlet C28H32O15 model containing 75 atoms and 320 electrons was used. Twelve candidate geometries were generated using RDKit 2026.09.1 ETKDGv3 and screened by converged MMFF94s minimization. The candidate with the lowest MMFF energy, C001 (seed 20261032), was advanced along the DFT workflow reported here; an exhaustive DFT conformational search was not performed. The coordinates were generated from verified connectivity. A gas-phase density-fitted PBE/def2-SVP pre-optimization (def2-universal-jfit auxiliary basis) was followed by non-density-fitted screened-direct PBE/def2-SVP refinement and a fresh SCF/gradient check at the final geometry. Both completed stages used PySCF 2.14.0, geomeTRIC 1.1.1, quadrature grid level 1, an SCF energy convergence threshold of 10⁻⁸ Eh, and four CPU threads, without a D3 dispersion correction. The final direct refinement comprised five complete geometry evaluations; its subsequent fresh check did not constitute an additional conformer. Atomic gradient RMS and maximum norm thresholds were 3×10⁻⁴ and 4.5×10⁻⁴ Eh/bohr, respectively. The final geometry passed these stationarity criteria; its positive-frequency minimum status remained unverified at this snapshot.

## Pending: 不能写成已经完成的频率结果

- 解析Hessian仍为RUNNING（主监督器告知；未在本笔记读取活动产物）。当前采用最多8个原子的扰动响应批次，同时保留完整分子的AO／MO电子耦合；B8不是截断分子、局域电子模型或原子分块partial Hessian。
- 非线性75原子分子去掉整体平移和转动后，**预期**为3N−6＝219个振动模式；实际模式数量、虚频／正频、完整Hessian和正常模式有效性仍待真实产物检查。
- 计划采用`ν(THz) = 0.0299792458 × ν(cm⁻¹)`；这仅是光速定义下的单位换算，不是经验频率缩放、小样本校准或对实验峰的拟合。
- 标准PySCF `harmonic_analysis`没有IR强度输出；IR强度保持`NOT_COMPUTED`。没有理论强度曲线、预填峰位、峰匹配成功或化学归属结论。

**English pending statement — keep separate from completed methods/results:** An analytical Hessian calculation is currently in progress. Nuclear perturbation responses are batched for up to eight atoms while retaining full electronic coupling throughout the molecule. Following validation of the complete Hessian and normal modes, 219 vibrational modes are expected after removal of translations and rotations. Wavenumbers will be converted to THz using the exact unit factor 0.0299792458. IR intensities are not computed. The completed Spinosin vibrational spectrum, positive-frequency minimum assessment, and experimental peak assignment remain pending at this snapshot.

## 解释范围与已核引文

斯皮诺素作为酸枣仁代表性成分有Q-marker综述背景（S01：2023，DOI `10.1016/j.jep.2023.116204`）；它不证明本项目855份样品的斯皮诺素含量或组间成分差异。分子晶体THz可包含柔软分子内运动、分子间运动、氢键网络和集体位移（S02：2014，`10.3390/cryst4020074`；S03：2023，`10.1038/s41570-023-00487-w`）。因此，孤立分子的理论模式即使接近药材固体峰，也不能单凭接近确认归属；需要纯品、含量测定或适当固态证据。

单一DFT构象、grid1及无D3仍是局限，应保留后续构象／网格／色散敏感性验证。S04（2022，`10.1021/acs.jpclett.2c01278`）支持关注柔软低频模式的数值积分误差，**不提供斯皮诺素或本项目PBE/grid1的具体误差上界**。各引文的直接读取／官方索引核验等级、访问限制及短片段均见已封存`mechanism-literature-context/`；这里不把索引核验说成阅读全文。本笔记不采用评审带保留意见的软件稿作为斯皮诺素归属证据。

## 可追溯文件

`evidence_snapshot.json`记录结构、筛选、两个已完成优化阶段、方法参数、待完成项与UTC快照；`source_hashes.json`列出14份仅含已结束结果／冻结文献与源码审计的来源哈希；`files_sha256.json`封存本目录。direct完成证据来自`spinosin-native4-optimization-001/result.json`及其EXITED／exit0静态行，未把早期PARTIAL的单线程尝试误作成功结果。未打开任何checkpoint、HDF或活动Hessian数组。
