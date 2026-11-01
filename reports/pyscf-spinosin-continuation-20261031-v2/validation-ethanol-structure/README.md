# 乙醇软件／数值验证结构

**状态：DONE_STRUCTURE_PREPARATION_ONLY。SCF、DFT几何优化、Hessian、频率分析均为NOT_RUN。**

此目录只提供9原子乙醇输入结构，用于PySCF计算流程及Hessian响应块B4/B8的后续工程验证。乙醇不是本任务新增的植物数据，也不是对酸枣仁含有乙醇的鉴定；不计入任何独立药材样品数。制备结构不代表已经完成Opt/Freq验证。

## 连接关系与许可

- 主要来源：EMBL-EBI ChEBI，CHEBI:16236，ethanol：<https://www.ebi.ac.uk/chebi/CHEBI:16236>。
- 官方记录核实分子式C2H6O、净电荷0、SMILES `CCO`、Standard InChI `InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3`及InChIKey `LFQSCWFLJHTTHZ-UHFFFAOYSA-N`。两种连接描述分别解析后图一致。
- 官方许可说明：<https://www.ebi.ac.uk/chebi/about>，Licensing：CC BY 4.0，并遵守EMBL-EBI使用条款。结构信息归属ChEBI；许可证链接：<https://creativecommons.org/licenses/by/4.0/>。
- PubChem CID702作为参考标识保留。此次PubChem compound页面返回reCAPTCHA，PUG REST URL在web读取工具中不可访问；**没有声称成功下载PubChem结构**。失败回执也保存在来源快照中。
- `official_structure_web_snapshot.txt`保存官方结构页面的工具读取结果；`source_license_and_access_web_snapshot.txt`保存官方许可页面、检索回执和失败信息。它们是文本读取快照，不是原始HTML或官方3D文件；各SHA-256见身份登记表。

## 实际生成

RDKit ETKDGv3固定`randomSeed=20261031`、单线程生成一个构象，随后MMFF94s力场优化（上限750步，`forceTol=1e-4`，`energyTol=1e-6`）。实际状态0（收敛），最终力场能量为−1.517097577199456 kcal/mol。该值不是DFT电子能量，也没有多构象筛选或全局最小构象证明。

**全部坐标都是算法生成的模型坐标；不是实验测定坐标，也不是官方数据库下载的3D坐标。**

显式加氢后组成C2/H6/O1，总9原子、26电子、无自由基、单一连通分子、无未指定立体中心。采用净电荷0／自旋多重度1的闭壳层模型；没有实施自旋稳定性计算。

## 文件接口

- `ethanol_mmff94s.xyz`：用于后续验证的输入，元素与坐标单位Å，索引顺序与NPY及SDF相同。
- `ethanol_coordinates_angstrom.npy`：float64全精度9×3坐标。
- `ethanol_mmff94s.sdf`：含连接关系的生成模型；SDF格式小数位限制会使其坐标与NPY略有舍入差异。
- `ethanol_etkdgv3.xyz`：MMFF前的生成坐标。
- `atom_order_and_bonds.json`：零起始索引、元素、键及键长。
- `ethanol_verified_2d.json`：来源、身份、独立图校验与许可。
- `generation.json`：实际版本、参数、力场能量、状态和耗时。
- `provenance.json`：机器可读的验证用途、输入SHA、已完成和未运行状态。
- `readback_checks.json`：实际重新读取XYZ/NPY/SDF后的相互一致性检查。
- `files_sha256.json`：本目录各文件SHA-256封存索引，自身除外。

## 可重现命令

从仓库根目录执行，使用一个新的输出目录，脚本拒绝覆盖已有生成产物：

```bash
PYTHONPATH=.runtime/quantum-upgrade-site:.runtime/quantum-pilot-site OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .runtime/bin/python reports/pyscf-spinosin-continuation-20261031-v2/validation-ethanol-structure/prepare_ethanol_validation_fixture.py --output /tmp/research4-ethanol-validation-reproduce
```

生成脚本仅做RDKit连接关系处理、嵌入、MMFF94s及一致性检查，不导入PySCF、不运行任何量子计算。后续工程验证应在独立输出目录保存Opt、fresh SCF／梯度及standard/B4/B8 Hessian结果；不得将其解释为酸枣仁分子频率研究结果。
