# Gaussian／DFT升级：候选成分与真实运行可行性

2026-10-31，北京时间。按用户本轮新方向，量子化学重新进入机制研究的辅助部分；不改写此前排除Gaussian时的历史实验，也不替代酸枣仁MC-UVE＋SVM基线。

## 已完成与尚未完成

**已实际完成开源PySCF水分子环境测试：PBE／STO-3G单点、核梯度及解析Hessian。没有运行Gaussian引擎，没有优化或计算三种酸枣仁候选的振动频率，也没有得到可用于分类的理论峰。**

PySCF2.14.0、h5py3.16.0仅安装到`.runtime/quantum-upgrade-site/`，不改主ML环境的NumPy／SciPy。默认研究Python仍不能直接import这两个新增包；显式命令才加入独立路径。下载轮子SHA及许可元数据保存在[环境锁](../data/research/quantum_upgrade_20261031/environment.lock.json)。PySCF采用Apache2.0，实际wheel与[官方仓库许可](https://github.com/pyscf/pyscf/blob/master/LICENSE)一致。

本地GaussView6程序与私有启动器仍存在；本次只执行状态检查，没有重新进行GUI功能认证。已知私有部署路径与PATH均未发现g16／g09引擎，也没有新的可核验Gaussian授权安装介质。GaussView显示结构不等于Gaussian可做DFT；[旧安装记录](GAUSSIAN_INSTALLATION.md)保留原样。ORCA、Psi4、RDKit等在检查时未安装，未冒称它们已运行。

## 三个候选登记与结构门槛

第一份任务上传实际在`Jujub`处截断；后续用户已补充候选为Spinosin、Jujuboside A、Jujuboside B。完整登记见[JSON](../data/research/quantum_upgrade_20261031/candidate_registry.json)、[CSV](../data/research/quantum_upgrade_20261031/candidate_registry.csv)及[来源表](../data/research/quantum_upgrade_20261031/sources.json)。

| 候选 | 数据库分子式 | 由分子式计算总原子数 | 重原子数 | 非线性完整分子预期3N−6 | 本次完整分子DFT |
| --- | --- | ---: | ---: | ---: | --- |
| Spinosin，斯皮诺素 | C28H32O15 | 75 | 43 | 219 | NOT_RUN |
| Jujuboside A，酸枣仁皂苷A | C58H94O26 | 178 | 84 | 528 | NOT_RUN |
| Jujuboside B，酸枣仁皂苷B | C52H84O21 | 157 | 73 | 465 | NOT_RUN |

原子数、电子数和3N−6是公式派生的规模描述，不是已计算的原子坐标、正常模式或频率。分子含较多糖苷／羟基相关自由度；准确的可转动键数量、构象分布及最低构象尚未从已下载结构核验，均标记NOT_VERIFIED。完整候选资源开销未实测，不能根据三原子水的耗时外推75／178／157原子的DFT Hessian成本。没有截取片段替代大皂苷的完整计算。

三候选作为酸枣仁成分有已发表色谱文献依据：[2014年HPLC-ESI-MS/MS研究](https://pubmed.ncbi.nlm.nih.gov/25422554/)，DOI10.4103/0973-1296.141777；[2021年HPLC／UHPLC-MS研究](https://pubmed.ncbi.nlm.nih.gov/34486838/)，DOI10.3724/SP.J.1123.2021.06019。它们不证明当前855份样品的成分含量，也不证明产地分类由这些成分造成。

PubChem索引记录可核到[Spinosin CID155692](https://pubchem.ncbi.nlm.nih.gov/compound/155692)、[Jujuboside B CID24721031](https://pubchem.ncbi.nlm.nih.gov/compound/24721031)及其分子式。**Jujuboside A有需解决的身份细节：**[CID171446](https://pubchem.ncbi.nlm.nih.gov/compound/171446)的已索引InChI保留部分未确定立体标记，[CID51346169](https://pubchem.ncbi.nlm.nih.gov/compound/51346169)有不同的立体指定；[FDA GSRS](https://precision.fda.gov/ginas/app/ui/substances/23597106-2887-4b5d-902e-c0f6cbda734b)也有交叉映射。不能仅凭同名同式随便选一份结构，应与所用标准品及结构文献核对。

本次PubChem直接页面遇到CAPTCHA／JavaScript限制，PUG属性JSON通过网页工具不可访问；未绕过云端目的地限制。没有获得已审查的3D结构文件或构象集合，未从搜索文本拼接SMILES，也未猜测候选坐标、电荷或多重度。

## 实际水分子技术测试

| 项目 | 实际值 |
| --- | --- |
| 输入 | 仓库既有手工`examples/gaussian/water.xyz`，未修改 |
| 分子 | H2O，3原子、10电子，中性单重态 |
| 引擎 | PySCF2.14.0，单线程，PBE／STO-3G，grid level3 |
| SCF | 收敛，能量−75.22550440958054 Hartree |
| 核梯度范数 | 0.1647459504 Hartree／Bohr |
| 解析Hessian | 3×3×3×3，全部有限；最大对称差约9.54×10⁻¹² |
| 最新技术复测耗时 | 0.6592秒，包含小体系单点及导数；不代表大分子成本 |
| 几何优化／频率／IR强度 | 均NOT_RUN |
| 药材候选计算／新增真实样品 | 均0 |

梯度明显非零，因此该固定几何不是本方法下已验证驻点；未用其Hessian生成“可信水频率”或药材模拟谱。解析器防护收紧后重复了一次同一技术测试，两个实际结果、源码快照和命令起止均保留，能量差为0；不把两次计算算成两份实测样品。

[实际运行证据](../reports/upgrade-20261031-v1/quantum/README.md)含真实UTC、命令、SCF日志、梯度与Hessian数组、环境检查和两个源码快照。单点导数可用不代表已完成优化、正常模式或IR强度验证。

## 模板与失败关闭的输入／解析接口

[三个Gaussian模板](../examples/gaussian/upgrade_20261031/README.md)均为`.gjf.template`，含明确占位符、状态NOT_RUN，不是完整可执行输入。B3LYP／6-31G(d)、GD3BJ、Opt／Freq等方法行只是提案，未经过Gaussian原生执行验证，也未依据测试分数选择。

`scripts/prepare_quantum_upgrade.py`提供：

- `validated_xyz`／`gaussian_input`：要求审定结构来源、立体状态、坐标内容hash及完整组成；拒绝缺来源、改坐标、错分子式、非有限值与不相容电荷／自旋奇偶。这些格式／溯源检查不能证明正确基态或构象。
- `parse-log`：识别正常／错误终止、优化完成、频率和IR强度、虚频及固定cm⁻¹→THz换算；缺失／非有限数据不通过基础块检查。解析文本始终不直接认证物理解释，还需核模式完整性、几何、构象和标准品。
- `validate_theory_prior`：拒绝测试标签／得分选峰、按真实产地选不同分子、跨协议先验、训练范围外ID或未完成计算。固定外部理论先验必须在外测前锁定；数据参与的先验只能来自当前训练折。有限检查不证明绝对零泄漏。

7项实际自动化检查通过，包括故意错结构、改hash、错自旋、坏频率块、测试选峰／跨折先验及NOT_RUN数据被拒绝。测试中频率文本为明确的合成解析夹具，不是Gaussian输出，也未计入候选计算。

## THz机制可以论证到什么程度

相同分子不会因文件标为不同产地就自动获得不同理论化学结构。当前没有这些样品的HPLC含量、纯品THz或晶体结构，不能声称某个产地模型频点已被确认属于某成分。

孤立分子计算适合先研究指定构象的分子内振动；固体低频还受分子间相互作用、堆积、水合和集体运动影响。[L-抗坏血酸的原始计算／实验研究](https://pubmed.ncbi.nlm.nih.gov/37078189/)说明其0.2—1.6THz晶体谱不能简单由有限孤立团簇取代周期体系；[2023年THz材料综述](https://www.nature.com/articles/s41570-023-00487-w)也讨论了凝聚相集体低频动力学。这些文献是方法边界，不是酸枣仁峰归属的证据。

本轮不向分类器添加任何理论频点或掩码。后续若有审定计算，须预注册外部先验或在当前内层训练范围建立，窗口宽度／权重也只能源内选择；设置无理论先验及匹配预算控制。注意力、SHAP或测试分数的事后解释不得反馈到同版本选参。纯品／HPLC和必要的固态模型补足前，机制结论保持BLOCKED／EXPLORATORY。

## 复现技术测试

```bash
.runtime/bin/python -m pip download --no-deps --only-binary=:all: --dest outputs/quantum-wheels-new pyscf==2.14.0 h5py==3.16.0
# 安装前按environment.lock.json核对相应平台wheel的SHA；其它平台重新锁定轮子。
.runtime/bin/python -m pip install --no-index --no-deps --target .runtime/quantum-upgrade-site outputs/quantum-wheels-new/pyscf-2.14.0-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl outputs/quantum-wheels-new/h5py-3.16.0-cp312-cp312-manylinux_2_28_x86_64.whl
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .runtime/bin/python scripts/prepare_quantum_upgrade.py water-smoke --output outputs/water-smoke-new
.runtime/bin/python -m unittest discover -s tests -p test_quantum_upgrade.py -v
```

需已有匹配NumPy2.3.5／SciPy1.17.0及Python3.12环境。软件包二进制未提交到Git，真实下载文件名和hash已记录；已有非空smoke目录会被拒绝覆盖。未来Gaussian日志可用`parse-log --input <log> --output <新的JSON>`检查，不能将模板或合成测试文本当真实计算结果。
