# 外部光谱候选：食品与药片

这里只完成公开文件下载、来源与结构检查，**没有训练模型或计算预测成绩**。本轮主实验优先采用另行找到的 UltraIR 中药数据；下列食品数据保留为备用工程基准。

| 数据集 | 实际文件结构 | 可研究任务 | 本轮状态 |
| --- | --- | --- | --- |
| OliveOil | 固定 TRAIN 30 / TEST 30，570 点，4 类 | 食品产地光谱分类 | 已下载；备用 |
| Coffee | 固定 TRAIN 28 / TEST 28，286 点，2 类 | Arabica / Robusta 品种分类，不是产地分类 | 已下载；备用 |
| Tecator | 240 行，100 吸光度列，22 预计算 PCA 列，moisture/fat/protein 3 个目标 | NIT 成分回归 | 已下载；备用；只能使用前100光谱列作为原始输入 |
| Pharma Instrument1/2 | 每仪器 655 行、650 个光谱列；Train 155 / Test 460 / Validate 40 | 药片定量与跨仪器迁移候选 | 已下载；来源说明与CSV结构冲突，尚未解决，不作为已确认研究数据 |

OliveOil、Coffee 来自 `tslearn-team/tslearn` 固定提交 `ee252503404ea947a5cae482540132c05eba4064` 的 `docs/datasets/`。每条谱已经按自身均值/标准差标准化；没有物理样品编号、批次编号或物理光谱坐标。跨 TRAIN/TEST 未发现完全相同的特征行，但这不能证明不存在同一样品的重复扫描。类别整数与具体国家/品种名称的映射没有在镜像说明中确认，不自行命名类别。

OliveOil 的原始论文链接由镜像说明提供：<http://www.ncbi.nlm.nih.gov/pubmed/14518931>。Coffee 说明引用 Briandet 等 1996 年文章 *Discrimination of Arabica and Robusta in Instant Coffee by Fourier Transform Infrared Spectroscopy and Chemometrics*，以及 Bagnall 等 SDM 2012 的档案使用。两个镜像说明没有显式的数据专属许可证；不将软件许可证自动解释为数据授权。

Tecator 和 Pharma 来自 `caseykneale/ChemometricsTools.jl` 固定提交 `0403cd4014af9b9d1dcac20c42ca3543ca56a9d5` 的 `data/`。`Readme.md` 保存 Tecator 原始许可说明：数据为 public domain，允许在附带该说明时再分发；发表时需提及 Tecator 仪器和公司，原说明还请求寄送论文预印本。本次没有联系或发送给任何人。

Pharma 的 README 声称653片、600–1398 nm；CSV实际为655行，光谱列名600–1898、步长2。两个仪器每行的 SampleType、weight、hardness、assay 完全一致，但文件没有显式样品ID。这些观察只支持后续核验配对关系；不能跳过来源冲突直接宣称确定的样本数量或测量范围。

逐文件下载URL、错误记录、结构观察和限制见 `provenance.json`；`downloads.json` 保留下载尝试；`files_sha256.json` 保存已下载内容校验值。主站 timeseriesclassification 的直接下载受到当前HTTP目的地策略限制，公开GitHub镜像可访问，未绕过网络策略。Meat镜像路径返回404，没有伪造数据或将失败计为下载成功。
