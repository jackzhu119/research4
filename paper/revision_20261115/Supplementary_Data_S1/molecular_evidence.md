# 分子计算新增证据审查（2026-11-15，北京时间）

## 可直接用于新稿的结论

原儿茶酸（protocatechuic acid, PCA；C7H6O4，17原子）的**两套完整局部极小值与谐振IR结果已合格**。这实质性更新了上传旧稿中“严格驻点未过、没有可信IR强度”的状态。新稿应使用现在的结果与数值，替换旧稿73.6463 cm−1（2.20786 THz）的非驻点grid3结果。不能把两者混合平均。

该进展提供了一项可复核的理论约束：在所计算的两个孤立PCA局部模型中，最低谐振基频约2.219 THz，**0.1–1.5 THz内没有谐振基频**。因此本研究的0.1–1.5 THz判别坐标没有获得孤立PCA基频解释。这一结论限定模型解释范围，既不证明药材不存在PCA，也不排除PCA的固体/分子间贡献。

## 两个合格结果的参数与实际数字

| 项目 | direct-grid5 | RI-J / df-grid5 |
|---|---:|---:|
| 软件 | PySCF 2.13.0 | PySCF 2.13.0 |
| XC / 基组 | PBE / def2-SVP | PBE / def2-SVP |
| 分子模型 | 中性、闭壳层、孤立分子；未加D3 | 同左 |
| 积分模型 | 原生直接电子排斥积分 | 库仑密度拟合，def2-universal-jkfit辅助基组 |
| 积分网格 | grid level 5，PySCF默认NWChem pruning | 同左 |
| 梯度响应 | grid_response=True | grid_response=True，auxbasis_response=True |
| 完整双步长Hessian | h=0.001 / 0.0005 Bohr，各102位移，合计204 | h=0.002 / 0.001 Bohr，各102位移，合计204 |
| 独立IR笛卡尔三步长 | 0.001 / 0.0005 / 0.00025 Bohr | 0.002 / 0.001 / 0.0005 Bohr |
| 独立逐模式正常坐标三步长 | 0.002 / 0.001 / 0.0005 Bohr√amu | 0.004 / 0.002 / 0.001 Bohr√amu |
| 逐模式独立位移计算数 | 270（45×2×3） | 270（45×2×3） |
| 合格振动模式 / 虚频数 | 45 / 0 | 45 / 0 |
| 最低频率 / cm−1 | 74.03118967563198 | 74.00616649967424 |
| 最低频率 / THz | 2.219399232152193 | 2.2186490562094594 |
| 最低模式IR强度 / km mol−1 | 2.1430710183885173 | 2.1388837008982984 |
| 优化中心能量 / Eh | −570.2261138802517 | −570.2262808707744 |
| 中心核梯度RMS / Eh Bohr−1 | 3.4088839779×10−7 | 8.0528701431×10−8 |
| 中心核梯度最大分量 / Eh Bohr−1 | 9.0119264406×10−7 | 2.2248563658×10−7 |
| 0.1–1.5 THz模式数 | 0 | 0 |
| 上游极小值验收 | 通过，无失败门槛 | 通过，无失败门槛 |
| 独立IR验收 | QUALIFIED / paper_ready=true | QUALIFIED / paper_ready=true |
| 独立IR完成时间（北京） | 2026-11-12 13:10:08 | 2026-11-12 19:37:35 |

**解释时注意**：不同积分近似的绝对能量不能用于推断两个构象的相对稳定性。这里不做全局最低构象、实验准确度或基组收敛的声明。

两套45个模式按升序一一作描述性比较，最大绝对频差0.4048637553360095 cm−1、RMS频差0.11069200026341006 cm−1；最大绝对强度差0.4653021293132724 km mol−1。它们反映两个数值模型给出的近似一致性，不能称为对实验谱的误差或化学归属验证。新SI如需简洁，保留最低模式及45/0即可，把全部45行频率/强度作为机器可读表。

## 本次实际复核的证据

文件位于 `outputs/revision_20261115/molecular_sources/acid_d2f61d35/`（direct）和 `acid_f626f77f/`（RI-J）。每个目录包含：`ir_modes.csv`、`ir_arrays.npz`、`result.json`、`COMPLETE.json`、`input.json`，以及上游中心、模式、资格与来源记录。

已执行 `molecular_sources/verify_molecular_tables.py`；结果为 `molecular_sources/molecular_table_verification.json`：

- `COMPLETE.json`中的result SHA、独立IR结果中的NPZ与CSV SHA均匹配。
- 输入来源记录的上游case结果SHA匹配；两个上游publication gate均无失败项，minimum_verified=true。
- IR NPZ中LᵀML与I的最大差分别3.55×10−15、2.44×10−15。
- NPZ频率与CSV最大差仅2.27×10−13、4.55×10−13 cm−1（文本读取浮点差）。
- 从投影总偶极导数用I=974.8801098×|∂μ/∂Q|²重建的IR与CSV最大差3.70×10−9 km mol−1。
- 各45个模式全正，坐标数组17×3、质量17、模式矩阵51×45、三步长Cartesian导数3×3×51、逐模式导数3×3×45，结构相符。

上述是对已归档结果的数值重算与源一致性核验，没有重新执行量子化学SCF、优化或Hessian作业。

## 来源固定点与定位

独立IR结果固定提交：`83df7a9ffa82c7045a22722382aeb78aed55db6c`（分支 `research/pyscf-ir-strength-repair-20261111`）。

- direct source ID：`d2f61d359fb8e80c51d8ffdaca21289debc01c1eef6f398dea07b71e0b195f11`
- RI-J source ID：`f626f77fcebb7b9e5bdf10c3725664b4754e01232361b3c209106da04b4206f6`
- 每套独立IR路径：`reports/pyscf_ir_20261111/jobs/<source_id>/round00000/{ir_modes.csv,ir_arrays.npz,result.json,COMPLETE.json}`。
- 每套上游原件镜像路径：`reports/pyscf_ir_20261111/sources/<source_id>/evidence/`。
- direct上游分支/提交：`research/pyscf-frequency-repair-20261111` / `ffc716671314904f683b1b94ea071d4678f3ea48`，case=`reports/pyscf_repair_20261111/jobs/acid`。
- RI-J上游分支/提交：`research/pyscf-publication-recovery-20261111` / `bbfd858e32776efa059fcd5788960d6c76b6e4ca`，case=`reports/pyscf_publication_20261111/jobs/acid/df-grid5/round000`。
- 独立IR综述入口：上述83df7a9f提交的 `reports/pyscf_ir_20261111/summary.json`，PCA两个模型target资格true，毛蕊异黄酮/斯皮诺素false。
- 上传旧稿对照：`extracted_text/03_Supplementary_information.txt` S7及 `01_Manuscript_Applied_Spectroscopy.txt`理论段。旧稿“没有已计算IR”及“严格PCA极小值未通过”现已过时。

## 仍属于进度/负结果的分子工作

| 对象 | 最新可查状态 | 新稿使用方式 |
|---|---|---|
| 毛蕊异黄酮 calycosin | publication提交e1787e6d…，完整双步长导数263/396（2026-11-15 15:35:49）；独立IR仍是PRECURSOR_ONLY，source_minimum_qualified=false | SI状态表标“进行中/部分导数”，不画最终光谱、不写完成93正模式、不能引用水控制来授予其资格 |
| 斯皮诺素 spinosin C003/C008 | 98a2eadd…的20261115_1513快照分别26/27次完整梯度评估；历史最低观测RMS分别0.000251016445/0.000324753531 Eh Bohr−1。仍OPTIMIZE，0个production Hessian点，无合格IR | 状态表标“优化尚未到严格驻点”，不把梯度评估数当接受优化步、独立构象或振动模式 |
| 斯皮诺素另一路C002/C004 | c189e896…完整包README：10条合格能量/核梯度记录对应8个不同几何点；无收敛极小值、Hessian或IR | 作为另一计算队列的进度，不与C003/8合并成完成模式或分子样本数 |
| 旧斯皮诺素219模式/25虚频；旧毛蕊异黄酮93模式/7虚频 | 历史非极小值诊断原件 | SI说明诊断与修复动机即可；不得删虚频或取绝对值作“理论峰” |
| 水控制 | 真实通过direct与RI-J频率/IR数值控制；独立解析IR对照存在 | 只支持算法数值实现的控制，不替代药材目标分子的验收 |

进度是上述固定Git快照，不保证作业目前仍在线。持续计算、CPU追加、备份/恢复或重新上传是运维进展，不会自动产生新的合格光谱。

## 建议正文与SI的用法

主线继续围绕“候选池组成如何改变吸收变量筛选及内部来源判别”，分子结果支持解释边界与后续验证方向；无需把一个计算完成事件改成第二篇机制论文的主线。正文用一段结果先行的理论发现，完整方法与模式表放SI。

可直接入正文的英文段落（需按最终章节编号改Fig./Table）：

> The molecular comparison provides a quantitative boundary for interpreting the selected coordinates. Two independently qualified PBE/def2-SVP calculations of isolated protocatechuic acid, using direct integrals and RI-J Coulomb fitting, each yielded 45 positive vibrational modes. Their lowest fundamentals were 2.21940 and 2.21865 THz, respectively, above the 0.1–1.5 THz analysis interval (Fig. Sx). Thus, these models supply no isolated-molecule fundamental within the interval used for discrimination. The agreement between the two numerical treatments supports this model-specific frequency separation; it does not establish the molecular origin of the herbal-solid response. The selected variables remain experimental fingerprint coordinates whose chemical interpretation requires matched compositional and solid-state measurements.

如需再精简，删除最后一句而把固态解释需求并入已有discussion，避免重复“不能”。不建议把尚未完成的毛蕊异黄酮和斯皮诺素工作写进摘要。

可用SI图：`figures/Fig_S_quantum_accepted_PCA.png`（另有PDF/SVG），由`molecular_sources/plot_accepted_quantum.py`直接绘制原始棒谱，未增加经验展宽。图a只放低频区与0.1–1.5THz区间，图b放全部45模式。图注建议：

> Qualified isolated-molecule IR predictions for protocatechuic acid at PBE/def2-SVP without D3, using direct electron-repulsion integrals and RI-J fitting. (a) Low-frequency comparison with the experimental analysis interval shaded. (b) All 45 positive modes. Stick positions and heights are the archived frequencies and dipole-derived intensities; no band broadening or fitted scaling is applied. Both profiles have their lowest fundamental above 2.218 THz. The calculation describes the stated local isolated-molecule models and is not a solid-state assignment.
