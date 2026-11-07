# Spinosin结构核验、构象生成与初筛：实际完成

本阶段实际生成并初筛12个Spinosin计算构象；**没有在此阶段执行DFT、频率或IR强度计算**。这些坐标由可靠构象算法从已核实的完整立体化学图生成，不是实验坐标、官方PubChem3D记录或最低能量全局构象。

输入为[已核实的ChEBI／PubChem连接与立体记录](../structures/spinosin_verified_2d.json)，其SHA256为`9855fb57cac932c221ac561884562e38a97688b143bf8dbb7c5ad09333eea079`。RDKit实际计算的标准InChI和InChIKey与来源完全一致，十个四面体中心全部指定；显式加氢后C28H32O15、75原子、43重原子、总电荷0、零自由基电子、320电子。选择闭壳层单重态，未声称做过自旋态搜索。

实际方法为RDKit2026.09.1的ETKDGv3，种子20261031—20261042、单线程，每次生成一个不同起始构象；MMFF94s最多750次迭代，forceTol0.001、energyTol1e-6。12/12均嵌入成功、力场优化返回0。每个优化前后均由3D坐标重新指定立体化学，并核验完整InChIKey；SDF重读的连接和立体身份、XYZ原子顺序、75个有限坐标及化学键长再次核对通过。

实际构象搜索CPU时间2.5596秒、墙钟2.5591秒，设置了60秒CPU硬预算（软限60／硬限65）。真实命令、起止及退出码见[commands.jsonl](commands.jsonl)，完整日志见[generation.log](generation.log)。另11项结构／预算／几何防护测试实际通过，见[tests.log](tests.log)，没有量子化学计算。

最低MMFF94s筛选能量为C001的227.2838852 kcal/mol；全部12个结果范围227.2839—258.8864 kcal/mol。**这只是力场能量，不能称DFT电子能或直接用作理论振动峰。** C001是此次有限搜索的首个DFT起点；不能据此推断DFT最低构象或真实样品的构象分布。

- [全部初筛排序CSV](../conformers/conformer_ranking.csv)
- [C001完整75原子XYZ](../conformers/C001_mmff94s.xyz)，SHA256 `4567bbacd7a1881e20d9ef823e241757a2c479d49ee20c98cb206c4aeeddb588`
- [C001带连接和立体信息的SDF](../conformers/C001_mmff94s.sdf)
- [供DFT入口使用的来源／身份核验](selected_geometry_provenance.json)
- [所有构象XYZ／SDF／原子键长及立体复核](geometry_quality_recheck.json)

43个重原子按相同原子映射和正确旋转对齐后，66个构象对的RMSD为0.5479—3.3508 Å，没有低于0.35 Å的对。此指标只说明此次生成的几何差异，不证明构象搜索完整，也未包含周期晶体或分子间氢键网络。

原生成摘要中`rotatable_bonds_toolkit=16`是**显式加氢图**的RDKit描述符，不等于标准重原子图可旋转键数。标准重原子图为7，差异涉及末端羟基扭转计数约定；原文件和生成脚本字节保持不变，明确核查见[rotor_convention_review.json](rotor_convention_review.json)。不得在成本表中把16标成标准PubChem可旋转键数。

重新生成到新目录（不覆盖本次证据）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=.runtime/quantum-pilot-site:.runtime/quantum-upgrade-site \
.runtime/bin/python scripts/prepare_spinosin_conformers_20261031.py \
  --source reports/pyscf-spinosin-pilot-20261031/structures/spinosin_verified_2d.json \
  --output outputs/spinosin-conformers-reproduction \
  --site .runtime/quantum-pilot-site --count 12 --seed 20261031 \
  --max-iterations 750 --cpu-budget-seconds 60
```

理论DFT后续状态、梯度和频率必须以根实验目录的真实量子计算结果为准，不由本构象阶段的DONE状态代替。
