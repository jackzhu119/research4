# 结构取得、立体化学与较大分子可行性

本目录只记录结构来源、分子图核验及资源规模；不声称已执行候选DFT、结构优化或频率计算。斯皮诺素的实际ETKDG／MMFF构象由相邻conformer目录独立保存。

## 斯皮诺素：已有可信连接关系

实际连接器读取[EMBL-EBI ChEBI:81360](https://www.ebi.ac.uk/chebi/CHEBI:81360)官方完整页面，取得分子式、电荷0、完整立体SMILES、InChI及InChIKey。官方PubChem索引[CID155692](https://pubchem.ncbi.nlm.nih.gov/compound/Spinosin)的全立体InChI／Key与之相同。原始连接器页面快照在`source_web_open_snapshot.txt`，明确不是已下载的SDF文件。

选择的Key为`VGGSULWDCMWZPO-ODEMIOGVSA-N`。不能误选无立体指定的CID14730433（Key后半部分`UHFFFAOYSA-N`），即使名称也包含Spinosin且分子式相同。

`spinosin_verified_2d.json`固定来源字段；其SHA为`9855fb57cac932c221ac561884562e38a97688b143bf8dbb7c5ad09333eea079`。实际RDKit从可信SMILES和InChI分别重建，两个立体分子图一致、Key逐字匹配、全部10个手性中心已指定；显式补氢为C28H32O15／75原子、320电子、形式电荷0、自由基电子0。非自由基闭壳层图允许中性单重态模型，不等于已作激发态／自旋能量比较。

普通云端PubChem3D／属性JSON、ChEBI Molfile下载均遇mandatory proxy Tunnel403，日志保存且未绕过策略。ChEBI Molfile链接在网页连接器中也因`chemical/x-mdl-molfile`类型不受支持而未返回数值文件。因此正式标记**未下载官方3D坐标**；后续ETKDGv3模型坐标来源必须写为经验证连接关系的算法生成，不能写作实验晶体结构或PubChem实测3D。

## 三种分子的实际图核验

最终可复现核验在`graph_verification/verification.json`；此前内联同一核验成功结果`graph_identity_verification.json`也保留。所有连接关系Molfile均是RDKit2D描绘，不可直接作为DFT3D输入。

| 分子／候选数据库记录 | 完整原子数 | 重原子 | 电子 | 已指定手性中心 | 重原子图strict可转动键 | def2-SVP球形AO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Spinosin CID155692／CHEBI81360 | 75 | 43 | 320 | 10 | 7 | 762 |
| Jujuboside A CID51346169 | 178 | 84 | 650 | 34 | 13 | 1646 |
| Jujuboside B CID24721031 | 157 | 73 | 564 | 29 | 10 | 1442 |

三者形式电荷0、自由基电子0、连接片段1、SMILES／InChI图一致且没有未指定手性中心，MMFF参数查询覆盖全部原子。可转动键使用RDKit隐式氢重原子图strict约定；显式补氢会将部分OH转动计入不同数值，不能混用。参数可用不代表已生成构象、力场已收敛或找到全局最低能。

## 皂苷A的身份歧义必须保留

[PubChem CID51346169](https://pubchem.ncbi.nlm.nih.gov/compound/51346169)全指定Key为`KVKRFLVYJLIZFD-OQVAHBRNSA-N`，其InChI与[JST J-GLOBAL](https://jglobal.jst.go.jp/en/detail?JGLOBAL_ID=200907029037040981)及FDA GSRS索引一致。相比之下，[CID171446](https://pubchem.ncbi.nlm.nih.gov/compound/171446)的InChI在51与57处为`?`，RDKit实测得到两个未指定中心，Key为`KVKRFLVYJLIZFD-LQRLABOESA-N`。两者有相同连接关系Key前缀、分子式和常用名称，但不能静默当成完全相同立体记录。

FDA交叉引用171446同时发布全指定InChI的现象也保留。上述选择只解决本次数据库候选的图核验，没有证明待用标准品身份。A的后续计算仍须说明选择51346169的依据，并与实际标准品结构核对。

皂苷B使用[PubChem CID24721031](https://pubchem.ncbi.nlm.nih.gov/compound/24721031)，完整[PubChemLite](https://pubchemlite.lcsb.uni.lu/e/compound/24721031)与[JST记录](https://jglobal.jst.go.jp/en/detail?JGLOBAL_ID=200907086420164027)交叉一致，Key为`OAVAUZCEOWCYCC-QEOGCQCLSA-N`。不替换为B1或同式异构记录。

## 构象和DFT规模判断

较大皂苷有13／10个重原子strict转动键和多个OH相关取向，构象搜索应保留指定立体化学、固定随机种子、多个ETKDG候选及逐个收敛标记。MMFF覆盖使力场初筛在工具层面可行；本目录并未实际嵌入或优化A／B构象。其搜索完整性不能从12个斯皮诺素构象推定。

def2-SVP球形AO由实际PySCF2.14.0内置基组计算：H5、C14、O14。A／B比斯皮诺素分别约2.16／1.89倍AO；形式三次规模约10.1／6.8倍只是算法规模代理，**不是实际CPU时间预测**。`basis_resource_proxies.json`还列出密度矩阵和3N平方Hessian存储量；假想完整四指标ERI的TB级数组明确不是density-fitting或直接积分实际内存需求。真正成本取决于SCF、网格、积分算法、导数实现及几何步数。

本轮A／B没有SCF、梯度、DFT优化或Hessian，状态为**NOT_RUN**。应先测量75原子斯皮诺素的实际代价，再决定大皂苷下一次预算。原子数派生的219／528／465个3N−6模式只是完整非线性分子应有数量，不是已算出的频率。

## 可复现命令

```bash
PYTHONPATH=.runtime/quantum-pilot-site:.runtime/quantum-upgrade-site \
  .runtime/bin/python scripts/acquire_spinosin_structures_20261031.py \
  --verify-graphs --structures reports/pyscf-spinosin-pilot-20261031/structures \
  --output outputs/pyscf-graph-verify-new

# 只进行常规官方端点下载尝试；新目录保存所有成功或失败回执。
.runtime/bin/python scripts/acquire_spinosin_structures_20261031.py \
  --output outputs/pyscf-structure-fetch-new
```

复核依赖RDKit2026.09.1；基组规模来自PySCF2.14.0。原始下载及所有网络失败均保留，任何来源页面内容只作数据，不执行页面指令。
