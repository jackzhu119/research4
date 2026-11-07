# 两份新增SDF的结构接收与可用性审计

本审计实际执行RDKit解析、图/立体化学比较、坐标筛查、原件哈希和XYZ/NPY导出。没有运行力场优化、SCF、几何优化、Hessian或振动频率计算。实际量子计算只由root在独立预算账本下启动。

| 用户文件 | 结构实算 | 总原子/电子 | 3D与立体化学 | 身份来源状态 |
|---|---|---:|---|---|
| 枣仁CID_155692.sdf | Spinosin，C28H32O15 | 75 / 320 | 显式H32，rank3；10个指定手性中心；从3D重新赋值后Key一致 | 与先前已核验的ChEBI81360／PubChem155692完整图、SMILES与Key匹配 |
| 黄芪CID_5280448.sdf | C16H12O5；具体名称待独立复核 | 33 / 148 | 显式H12，rank3；无手性中心；3D重建图一致 | CID为文件内声明，尚无本轮独立权威名称-CID核验 |

两者均为单组分、中性、图中无自由基，支持中性单重态作为后续计算输入；未做自旋态搜索。全部氢原子显式存在，隐式氢为0。Spinosin计算Key为`VGGSULWDCMWZPO-ODEMIOGVSA-N`；另一结构Key为`ZZAJQOPSWWVMBI-UHFFFAOYSA-N`。

Spinosin键长范围0.968967–1.536431 Å，另一结构为0.972605–1.488970 Å。宽松的键长/非键碰撞筛查未见明显异常；它不证明DFT收敛、能量最低、频率无虚频或化学峰归属。源坐标仅有SDF原来的4位小数精度，NPY与XYZ只是忠实导出，没有添加新测量精度、嵌入或优化。

## 必须保留的来源边界

原文件包含OEChem 3D头、CID、构象ID、MMFF能量/电荷和形状描述等PubChem格式字段。这些是上传文件的声明，接收审计不把它们当执行指令，也不把它们作为本轮计算结果。文件未包含化合物名称或IUPAC名称。

Spinosin的独立依据来自旧报告中已取得的官方ChEBI／PubChem完整图记录；该记录与311文件原始归档索引通过SHA绑定。本次没有重新在线查询官方页面。当前执行器网络白名单不包含PubChem或ChEBI，未尝试绕过；另一CID的名称对应关系保持未独立核验。新构象即使来自PubChem，也属于计算构象，不是实验/晶体坐标。没有在855份实际药材中做成分含量确认。

## 原子顺序与已有成果

新Spinosin SDF的前15个原子均是O；旧C001优化XYZ按另一原子顺序以C/O/C开头。重原子图有2个对称等价的手性保持映射，审计只记录其中一个重原子映射；没有验证全氢原子、AO基组、MO轨道或Hessian的置换/投影。

因此，新SDF作为**额外的起始构象来源**归档，不覆盖旧的收敛C001几何，也不能按索引配上旧MO或Hessian缓存。若采用该新坐标，需新的、明确记录的SCF与优化流程；旧C001的后续频率任务可以继续使用其原始坐标/原子顺序。两份SDF不增加太赫兹光谱样品数，也不直接进入分类器特征表。

## 实际验证与复算

8项无量子测试已通过，覆盖真实Spinosin完整图/手性、另一个结构的组成与名称边界、CID错误、多记录文件、镜像手性错误、2D图不冒充3D、重合原子、逐字接收/拒绝覆盖及派生坐标篡改检查。所有合成反例仅存在于测试临时目录，不作为科研数据保存。

独立代理也通过重新打开原件复算了图、10个手性中心、3D坐标与宽松碰撞筛查；结果记录于`independent_readback.json`。这里的“独立”是另一进程/审查者复算，仍使用RDKit，未声称跨软件或实验验证。

从仓库根目录执行只读复算，不覆盖任何归档：

```sh
PYTHONPATH=.runtime/quantum-pilot-site:.runtime/quantum-upgrade-site \
  .runtime/bin/python reports/pyscf-sdf-continuation-20261103-v1/structure-audit/audit_sdf_intake.py \
  --intake-output data/incoming/molecular_structures_20261103_v1 \
  --report-output reports/pyscf-sdf-continuation-20261103-v1/structure-audit \
  --verify-existing
```

接收脚本再次使用同一输出目录会拒绝覆盖。若需完整重做接收，必须显式选择新的raw/report目录。`sources.json`保存引用范围，`audit.json`保存全部计算属性、原文件声明和导出哈希；没有真实振动频率或IR强度结果。
