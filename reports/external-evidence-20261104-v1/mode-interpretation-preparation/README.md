# C001正常模式解释前的静态原子索引准备

本目录完成结构图、原子顺序及来源链检查，**尚未读取振动模式或解释任何频率**。未读取活跃Hessian、NPY或CHK，也未运行量子计算、构象搜索、分类训练或Git操作。

当前优化C001可以使用原始`C001_mmff94s.sdf`的原子顺序进行后续模式标注。可信ChEBI完整立体SMILES经显式补氢后的图与C001按索引逐项相同；当前v5 prepared XYZ与v3优化XYZ的坐标逐项最大差为0.0 Å，重新从该坐标检查立体化学也保持完整InChIKey一致。独立代码审查支持优化及坐标传递保留原子顺序。相同元素序列单独不足以排除同元素互换，不能省略图和代码来源链证据。

原始C001是由可信连接图生成的计算构象，**不是来自后上传PubChem SDF的逐原子坐标继承**。后上传SDF的化学身份相同，但含芳环和附氢的图对称性：43个重原子存在2个保留立体化学的图匹配，75原子存在48个匹配。本目录没有任取第一个作为唯一历史映射。后续画图及模式标签统一采用旧C001索引。

## 索引和图分区

下表均为**零基文件索引**，不是IUPAC化学位次；CSV及JSON同时提供一基原子编号。C001原子索引与当前PySCF输入索引为0–74逐项对应。

| 互斥分区 | 重原子索引 | 附氢索引 |
|---|---|---|
| 黄酮苷元区，含甲氧基、酚OH | 0–20 | 43–53 |
| 与苷元C连接的近端糖区 | 21–30 | 54–63 |
| 糖间氧桥 | 31 | 无 |
| 经O31连接的末端糖区 | 32–42 | 64–74 |

四个分区恰好覆盖75个原子一次；分区的元素计数不是拆开的独立中性分子式。JSON还保存黄酮A环、含氧杂环、B苯环、两个糖环、甲氧基、羰基、C—C及C—O—C连接等重叠子组；这些子组不能当作互斥分区直接求和。

羟基来自原始连接图中的O—H及O—C键，不是频谱推断：酚OH为O11–H49、O19–H53；糖OH为O25–H58、O27–H60、O29–H62、O36–H68、O38–H70、O40–H72、O42–H74。这里只标记图结构中的共价OH，没有据此判断某一氢键或低频峰。

## 使用边界

- 后续必须先取得完整、经过模式数和虚频检查的正常模式，并核对其几何及原子顺序与本目录锚点一致，才能使用此映射。
- 可以依据动画和坐标位移描述“糖区相对骨架摆动”等分子内运动。群参与比例需先说明位移是否质量加权；几何振幅占比不能改称振动能量或实验IR强度。本目录尚未计算任何参与比例。
- 一个正常模式可同时涉及多个基团；可视化标签不证明实验峰的化学成分归属。孤立分子没有真实药材的晶格、完整氢键网络和分子间集体运动。
- 此次图标注不确认855份药材中斯皮诺素的含量，不修改分类模型或96.0234%历史指标，也不把后上传SDF作为新增THz样本。

## 文件与复核

- `atom_order_audit.json`：静态输入路径/SHA、图与坐标检查。
- `C001_atom_map.csv`：75行旧C001与当前索引对照。
- `atom_groups_C001.json`：互斥分区、重叠骨架及9组OH。
- `uploaded_SDF_mapping_ambiguity.json`：新上传SDF的多个合法匹配，禁止强行唯一化。
- `code_lineage_review.json`：独立只读代码/来源链审查。

```bash
python reports/external-evidence-20261104-v1/mode-interpretation-preparation/check_static_atom_audit.py
```

重建时使用独立新输出目录，脚本拒绝覆盖已有审计；它只进行RDKit图操作和静态文本读取，不启动QM：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=.runtime/quantum-pilot-site:.runtime/quantum-upgrade-site \
.runtime/bin/python reports/external-evidence-20261104-v1/mode-interpretation-preparation/build_static_atom_audit.py \
  --output /tmp/research4-C001-static-atom-audit-new
```
