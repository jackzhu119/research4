# Research4 公开植物THz来源接收总表

核对日期：2026-10-31（UTC）。本表把原16项来源目录、真实下载回执、2026人参重叠审查、额外中药线索与水果独立审查合并展示；不修改这些冻结证据。本表及配套`FINAL_INTAKE_SUMMARY.csv`共22项来源，包含6项补充来源／软件记录。

**原acquisition下载回执范围内实际取得20个文件、138,459,184字节，全部按原manifest重验大小及SHA-256通过。已有42条人参记录是复用此前用户上传原件，新增远程下载数为0；不能把它们宣传成本轮新下载样品。** 水果30个持久ID与120条选定时点观察也不能相加成150份样品；叶片68条绘图曲线不能计为68份独立叶片。

本轮未取得可确认是新增、独立中药产地队列的公开数值原件。已在手人参可用于其自身的观察级物种分类；水果可用于独立、受限的原始波形回归研究。均不能替代酸枣仁主线，也不能据此声称提高了酸枣仁准确率。**跨植物预训练、标签合并、频谱直接拼接与迁移实验均未在本接收流程中开展。**

## 已在手数值、观察单位与准入

| 来源 | 本轮新增下载 | 实际观察单位与标签 | 数据接口／准入 | 许可与物理边界 |
| --- | --- | --- | --- | --- |
| [人参／西洋参V2](https://data.mendeley.com/datasets/hvbp5mdtxx/2)，DOI `10.17632/hvbp5mdtxx.2` | **0文件、0字节；复用已有用户上传Dataset.rar** | 42片观察，22西洋参／20人参；6组是技术导出组，原植株、共同原料及批次关系未知 | 42条配对alpha/n记录；65点0.2–1.6THz。可以单独开展严格source-only、按导出组分割的探索性物种分类；不是酸枣仁产地标签 | 官方CC BY4.0；alpha为cm⁻¹、n无量纲。两种表示不是84份样品；不能用导出组证明跨生物批次能力 |
| [Princeton Meta-Sticker原始水果仓库](https://github.com/PrincetonUniversity/-Sub-Terahertz-Fruit-Sensing)，CITATION DOI `10.1145/3625687.3625815` | **1 ZIP，138,001,691字节** | 30个持久果实ID：芒果、梨、柿各10个；40个计划时点，Tag/NT配对复测。2525个MAT不是独立果实数 | 已准备固定Tag、时点索引0/10/20/30的**120×7000**接口，每种水果40观察／10ID。Brix分植物回归，DM仅芒果；外内层都须按果实ID分组 | 未找到LICENSE，原件及衍生NPZ仅本地忽略。幅值向量只有sample_index，不能假定0.1ps或构造校准THz轴；目标按原工作簿reading units描述 |
| [活体叶片作者数值绘图源码](https://github.com/hasantahir/Characterization-and-Water-Content-Estimation-Method-of-Living-Plant-Leaves-Using-Terahertz-Waves)，DOI `10.3390/app9142781` | **18份TikZ＋1份作者TeX，共19文件、457,493字节** | 9个作者目录名，Re n／Re ε合计68条日次绘图曲线；未建立独立叶片ID、物种身份逐样本表或结果映射 | 原作者数值坐标可读，频率0.75175–1.1THz；仅作物理与接口背景审查，**不准入独立样品分类** | 仓库未找到LICENSE；原件仅本地。不是图像数字化，也不是68份独立样品；频率点不能拆成样品 |

上述20个下载文件的逐项URL、版本、时间、大小、SHA见[原始下载manifest](download_manifest.json)。20文件／138,459,184字节的统计范围不含下面单独列出的两个软件ZIP，也不含本地生成的NPZ、模型或报告。

水果原件SHA-256：`902cca56a744187dd3eb3bbed41aa1ce28eb5e75e5b003946d29007a7fed603f`。其引用属于SenSys 2023 proceedings，官方出版记录为2024-04-26；不能仅凭相近标题，把它当成后续Nature Food 2025正式存储的同一数值队列。后者Figshare许可不能自动覆盖这个GitHub包。

水果逐文件审查发现60个首时点别名重复，以及40对同场次Background／Metal数值完全相同，原件保持不变；缺失目标与缺失波形均另有记录。不能把旧2021参考文件当成2023每条波形的已校准时间轴，也不能直接用相同Background／Metal作有效参考校正。详情见[水果可行性审查](../fruit-feasibility/README.md)及[固定接口manifest](../fruit-feasibility/prepared_interface.json)。

## 已锁定官方来源但本环境下载受阻

以下URLs是原始来源，不是已经取得的数据文件。8项普通端点请求均保留在[403实测记录](blocked_endpoint_checks.json)；2026人参另有独立请求日志。继承代理返回403只证明当前云端下载受阻，不能写成官方已下架、数据收费或互联网上无该数据。

| 数据来源与官方URL | 许可证据 | 样品／标签及当前阻塞 |
| --- | --- | --- |
| [水稻籽粒饱满度V3](https://data.mendeley.com/datasets/wj2h5hn7kc/3)，`10.17632/wj2h5hn7kc.3` | 官方CC BY4.0 | 官方V3宣称8.34MB谱／成像包；未读压缩包，独立种子数、标签编码、单位待核。V1–V3不能当三个新队列 |
| [转基因玉米DeepSpectra](https://zenodo.org/records/19150796)，`10.5281/zenodo.19150796` | 尚未验证具体数据许可 | 最终论文指向含谱／标签／划分仓库；文献1080片、1074保留不是本仓库已取得矩阵数。需核原料批次、含量标签及121点0.2–1.2THz实际文件 |
| [小麦粉苯甲酸含量](https://datadryad.org/dataset/doi:10.5061/dryad.945c410)，`10.5061/dryad.945c410` | Dryad数据CC0政策；待逐文件核实 | 原记录有data/X/Y/fz等MAT与MATLAB源码；未取得数组。属于含量回归，不能改成产地分类 |
| [白桦种子原始THz扫描](https://zenodo.org/records/17258169)，`10.5281/zenodo.17258169` | 页面未取得明确许可名称 | seed.zip公开说明1.1GB；列为Time/Amplitude/Frequency/Spectrum。独立种子数和分类标签未知；0.1mm扫描点不是独立种子 |
| [Nature Food正式水果处理数据](https://figshare.com/articles/dataset/Sub-terahertz_metamaterial_stickers_for_non-invasive_fruit_ripeness_sensing/27299859)，`10.6084/m9.figshare.27299859.v1` | 官方CC BY4.0 | 官方265.05kB包未下载。与已下GitHub原始水果包的队列对应、轴和目标测定方式未逐数值确认，不能算重复或新增队列 |
| [栎叶介电函数](https://doi.org/10.6084/m9.figshare.21647759.v1)，`10.6084/m9.figshare.21647759.v1` | 尚未验证 | 文献支持温度相关0.2–1THz测量；文件未读，独立叶片ID未知。不同温度不等于不同叶片／产地 |
| [木材THz／X射线／RGB](https://doi.org/10.57745/RITX8A)，`10.57745/RITX8A` | 官方Etalab Open License2.0；论文表述CC-BY2.0，取得文件后应保留具体条款 | 官方10.1GB，文献110样品／38木种；110–130GHz FMCW成像，不是药材粉末THz-TDS吸收谱。大小与模式不匹配，低优先级 |
| [水稻二化螟THz检测](https://data.mendeley.com/datasets/njdgtskp38/1)，`10.17632/njdgtskp38.1` | 官方CC BY4.0 | 图像／谱文件及真实标签未取得。不得将像素冒充独立虫／植物，仅潜在农业辅助任务 |
| [2026人参光学常数V1](https://data.mendeley.com/datasets/k4pjjjmgnr/1)，`10.17632/k4pjjjmgnr.1` | 官方CC BY4.0 | 数值原件0文件／0字节。4方案×24个CV导出文件=96文件，**不是96份样品**；与2025已有42片是否重叠尚未判定 |

2026作者稿有43片图示及42／43分母线索，未经原件对应不能推造第43份样品、合并成85片，或把其“不同单根”声明赋给既有2025包。跨版本精确／近值重叠检查仍未运行。见[2026人参审查](../ginseng2026/README.md)及[重叠边界记录](../ginseng2026/overlap_audit.json)。

## 相关文献不等于已经取得训练数据

| 线索 | 一手证据与限制 | 当前准入 |
| --- | --- | --- |
| [柴胡／陈皮ISEMCC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10950663/)，`10.1016/j.heliyon.2024.e27743` | 原文数据需合理申请。文献柴胡130片／13原料批次，陈皮288片／48批次；每批10或6片，须保持批次。原文文章CC BY-NC-ND4.0不自动等于数值数据许可 | 原谱、标签、批次映射及数据许可均缺；0新增数组、未联系作者 |
| [贝母产地THz](https://www.sciencedirect.com/science/article/abs/pii/S0009261425004920)，`10.1016/j.cplett.2025.142350` | 只核出版社预览，包含多个贝母物种；完整数据声明、批次、独立N与许可未确认，不能预设是纯产地区分 | 未验证公开数值仓库；不能把“未找到”写成作者确认不公开 |
| [CascadeSeed转基因水稻](https://spj.science.org/doi/full/10.34133/plantphenomics.0071)，`10.34133/plantphenomics.0071` | 原文数据和代码向通讯作者索取；文献3378粒、0.3–2THz93带。补充图与代谢物表不等于原始THz样品矩阵 | 原始数值队列未下；未联系作者 |
| [林下参年限THz/CNN](https://www.sciencedirect.com/science/article/pii/S0026265X26001542) | 摘要提5–20年参龄；原始独立样品数、数据许可及公开数值文件未核实 | 只能登记文献，不能凭摘要或论文图生成实验样品与准确率 |

额外中药线索详情见[独立中药来源审查](../herbal-additional/README.md)和[source_audit.json](../herbal-additional/source_audit.json)。本接收汇总没有执行外部源码或新增模型拟合。

## 单独记录的软件归档：不计为新数据队列

| 软件仓库 | 实际取得与核验 | 数据／许可结论 |
| --- | --- | --- |
| [jamilmai/Analysis-and-ANN-for-TCM-DL](https://github.com/jamilmai/Analysis-and-ANN-for-TCM-DL) | 本地`outputs/public_thz_20261031_v1/ginseng_author_code/author_repository.zip`，**10,951字节**，5成员：2个Python、2个Notebook、1个README。README说明是人参／西洋参光学常数与ANN代码；未执行。SHA-256 `36b1c4f5595b66b906e0d443994c9ce862c4f174270a6540ee35e3969f91bcf9` | ZIP无LICENSE、无随附独立数值谱／标签文件；这份main归档回执没有单独绑定commit，不虚构版本身份。仅本地保存，0新增样品 |
| [jiaoyiping630/spectrum-preprocessing](https://github.com/jiaoyiping630/spectrum-preprocessing)，commit `45c3d67b5479ba82fa972f43eac12c3b1d945c53` | 198,964字节ZIP、14成员：10个MATLAB、1个README、3张JPEG；逐文件核查已由独立审查完成，未执行 | 是NIR/octane通用预处理示例，无随附数值数据、无LICENSE；不是已验证贝母THz仓库，0新增样品 |

这两个软件归档属于另外的接收范围，不加入原20文件／138,459,184字节的回执统计，也不增加数值数据集或独立植物数量。

## 原目录中的排除项继续保留

| 来源 | 实际排除原因 |
| --- | --- |
| [fzbingmo/Nature_food_Machine-learning-based-non-destructive-terahertz](https://github.com/fzbingmo/Nature_food_Machine-learning-based-non-destructive-terahertz) | 源码树为12张补充JPG、视频及设备占位；未找到逐样品数值光谱／标签 |
| [roshni-mahtani/MSFAN](https://github.com/roshni-mahtani/MSFAN) | CSV对应12类聚合物，不能计为植物验证 |
| [EigenJames/THZ_TDS_code](https://github.com/EigenJames/THZ_TDS_code) | 工程说明是合成物理信号流程，不能冒充真实植物独立样品 |
| [Rice HSI hackathon](https://github.com/strath-ai/rice-hsi-hackathon-2025) | 光学高光谱不等于THz；不能重命名波长单位或混入药材THz任务 |

## 统计口径与研究主线

配套CSV逐行保留来源URL、DOI、接收范围、下载字节、观察单位、标签、轴、许可和审查路径，22项中完整保留原16项，不用“找到了论文”替代“拿到了原始数组”。文件数、表示数、CV折数、重复时点、扫描像素与生物学独立样品数分别记载。

当前可推进的是保留酸枣仁强基线与严格训练内部优化；在人参自身物种任务上检验同一方法，在水果自身分组回归任务上作有限工程验证。其他植物任务不与酸枣仁标签混合，不把方法跨任务可用写成已完成跨植物迁移，也不把外部验证成绩与酸枣仁既有96.02%直接相减。
