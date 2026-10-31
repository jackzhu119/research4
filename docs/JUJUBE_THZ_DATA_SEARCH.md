# 枣／酸枣仁太赫兹公开数据专项检索

核验日期：2026-10-31。目标是寻找可下载的逐样本数值光谱，优先中国来源，并判断能否补充当前酸枣仁产地识别及新批次小样本校准研究。

## 结论

**本轮没有核验到可公开下载、带产地标签的枣／酸枣仁太赫兹数据集。** 这不代表此类数据不存在；部分平台需登录或申请，搜索引擎也不覆盖全部研究数据。

直接相关的酸枣仁THz来源仍是全成分配方颗粒检测专利，未见逐样本数值包。另核实到公开的枣类可见／近红外高光谱数据入口，以及非枣类种子的THz数据记录。材料相近、测量技术相同和任务相同是不同条件，不能互相替代。

**本轮新增数据文件下载数、数值数组解析数、模型运行数均为0，未联系作者。** 下文“公开入口”仅表示发布者页面列出数据文件或下载链接，不表示已经下载并审查数据内容。

## 核实来源与获取方式

| 来源 | 已核实内容 | 获取状态及与主线的关系 |
| --- | --- | --- |
| [酸枣仁THz检测专利 CN105136730A](https://patents.google.com/patent/CN105136730A/zh) | 北京中医药大学东直门医院申请；商业全成分酸枣仁颗粒经研磨压片，描述0.2–5 THz测量、参考／样品时域及吸收系数；至少三次重复测量取平均 | 全文已打开。见方法及图2，未见逐样本CSV／XLSX／MAT数据、产地表或数据仓储链接。不是多产地原药材队列；重复扫描不能算成独立批次 |
| [阳明交通大学枣类高光谱开放数据](https://ebil.web.nycu.edu.tw/open-data/smart_agricultural_technology_202505/) | Ziziphus mauritiana、Tainung No.13 Shirley；205个果实形成2123个切片，每片有糖度°Brix。官网列有`400_1700`目录，train／val／test的data及label共6个`.pkl`文件名，预划分1482／319／322 | 官网全文及[Google Drive入口](https://drive.google.com/drive/folders/1S83roGwfiXbJQfeUylWlgcE35NDvMP0b?usp=sharing)已核实；Drive内容本工具无法打开，未下载。属于可见／近红外高光谱果实糖度任务，不是酸枣仁THz或产地标签。2123片不能当2123个独立果实；需检查果实ID及跨集合关系 |
| [白桦种子THz时域数据](https://zenodo.org/records/17258169)，DOI `10.5281/zenodo.17258169` | 2025-10-03发布，公开列出`seed.zip`，1.1 GB，MD5 `82717c2ae282eaca71b006fe6356b9ef`；说明列为Time、Amplitude、Frequency、Spectrum，扫描分辨率0.1 mm | 发布页及文件清单已打开，压缩包未下载、内部文件未检查。对象不是枣，页面未说明独立样品数、产地、批次标签或完整许可条款。仅登记为THz信号处理候选，不能据扫描点数推断独立样品量 |

大学页面关联论文为[Hyperspectral signature-band extraction and adaptation for sugar content prediction on Ziziphus mauritiana and Syzygium samarangense](https://doi.org/10.1016/j.atech.2025.101009)。本轮核验的是大学数据说明；未根据其旧参考文献中的DOI推断新论文数据内容。

仓库此前已解析的[人参／西洋参THz n／α数据](../data/external/ginseng_hvbp5mdtxx_v2/README.md)仍可作为同模态外部算法基准：42片压片，20个亚洲人参、22个西洋参标签。它不是本轮新增下载，不是枣，也没有产地标签；压片数不自动等于独立植株或来源批次数。

## 同药材、同任务的文献线索

以下文献适合借鉴或进一步核对数据获取方式，但**没有核实到公开THz原始数据包**，也不能因检索结果包含THz字样就算作THz研究：

- [近红外光谱和一维卷积神经网络的酸枣仁产地鉴别](https://www.spgykj.com/article/doi/10.13386/j.issn1002-0306.2024110205)：出版者检索条目说明河北、河南、山东、山西、陕西5产地。任务贴近当前课题，技术为近红外；全文本次访问超时，数据声明和原始数据获取方式未核实。
- [近红外高光谱成像结合分水岭算法鉴别酸枣仁药材产地](https://ywfx.nifdc.org.cn/CN/10.16155/j.0254-1793.2021.04.22)：官方条目可检索，本次正文访问502；未找到数值数据下载入口。
- [酸枣仁种仁饱满度：高光谱透射成像与改进Otsu](https://www.sciencedirect.com/science/article/abs/pii/S0168169922004458)：高光谱研究；搜索摘要里的THz文字来自相关推荐，不应据此改称THz枣仁数据。

未取得这些文献的逐样本标签、独立性记录和数据许可前，不将其录入当前有效数据清单。

## 检索范围与排除项

检索中英文材料名：酸枣仁、枣仁、酸枣、红枣、冬枣、jujube、sour jujube、Chinese date、Ziziphus／Zizyphus jujuba、Ziziphi Spinosae Semen／Semen Ziziphi；组合太赫兹、THz、THz-TDS、terahertz、数据、dataset、spectra、data availability。

检索覆盖公开网页及Zenodo、Mendeley Data、Figshare、ScienceDB、国家基础学科公共科学数据中心和中国科学数据相关域名的定向网页搜索；这不是已登录所有平台的完整内部检索。同步查看高校数据发布页、出版者条目及专利全文。

常见误命中：

- [JujubeBruiseNet](https://data.mendeley.com/datasets/3mtdhrwgfr/5)：手机照片健康／碰伤分类，JPG／PNG，不是THz。
- [Jujube Diseases Identification Image Dataset](https://data.mendeley.com/datasets/6vttpjr7xw/2)：发布说明包含“spectral”字样，但给出的数据组成是病害照片及其增强图，未提供THz测量证据。
- 枣花蜜THz研究的对象是蜂蜜；其他文章仅在参考文献、相关推荐或仪器型号中出现THz，也不能当作枣仁THz数据来源。

## 对后续实验的影响

当前公开检索尚不能解决酸枣仁只有每产地一批的问题。现有855份样品的匿名产地组内部验证结论保持原范围；不把非枣THz、枣果近红外、模拟谱、参考谱或重复扫描标成真实酸枣仁新批次。

公开数据可分两条辅助线：非枣THz用于读取、信号处理和适用任务的外部算法验证；枣类近红外用于跨光谱任务的方法复现。两者均不支持直接把外部样品投入现有7产地THz分类器并解释为酸枣仁产地识别率。

本轮结果只新增来源记录，不改变[data/CURRENT_DATASET.md](../data/CURRENT_DATASET.md)及有效工作簿清单。机器可读核验记录见[jujube_search_20261031.json](../data/external/thz_sources/jujube_search_20261031.json)。
