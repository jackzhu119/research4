# 后续科研数据目录：已核验来源与可用性

日期：2026-10-30。本轮登记34条数据来源/表示记录：24条药材相关记录、10条迁移候选。**不是34个独立队列，也不是34组都已下载或建模。** 同一队列的模态、表征和不同仓库镜像分别标记；已有UltraIR不重复算作新增独立来源。

药材目录有11条本地可读的数据表示；另9条的公开元数据已核验但当前执行环境域名不支持下载，剩余分别是作者云盘、访问受限、只有论文、只有索引。迁移目录包含3组新下载解析的数据家族（小麦、芒果、Melamine），2组既有资料（药片、玉米），以及5组尚未下载的候选。

## 当前用于实测的新增药材数据

| 表示 | 原谱行→精确去重后 | 点数 | 标签 | 关键限制 |
| --- | ---: | ---: | --- | --- |
| 黄芪Raman | 160→160 | 2090 | 四产地 | 无物理样品/批次映射；与UV同一源队列 |
| 黄芪UV | 160→160 | 601 | 同四产地 | 不是第二独立队列；源文档602维与CSV谱点数有差异 |
| 源标注Raspberry Raman | 250→249 | 1682 | 五产地 | 源写Rubus idaeus，未确认药典覆盆子物种 |
| 源标注Shihu3 Raman | 200→195 | 1681 | 三产地标签 | 整数至具体产地完整映射未明示，保留数字类别 |

下载自作者发布的[qsi-tk1.4.1](https://pypi.org/project/qsi-tk/1.4.1/)中数据成员，只读取CSV/文本，未安装或执行其代码/字节码。完整wheel及成员SHA已保存。LICENCE明确代码LGPL3.0；这不自动建立每个数据集的独立许可。黄芪原始公开数据另有[Mendeley记录](https://data.mendeley.com/datasets/c2gdkcwvnj/2)，但本轮未将包内数组与该原始文件逐字节核对，不能混称同一字节版本。原谱、wheel和工作簿不镜像进本仓库，保留下载器和来源清单。

## 迁移与定量数据的优先级

- IDRC2002药片：两仪器，源195记录、目标460记录，assay单位mg/tablet；按原划分和恢复记录ID隔离，未把同药片另一仪器谱当新独立样品。用于校准工程验证，不是中药产地证据。
- IDRC2016小麦：作者文档给248原样品、另150新样品/新仪器；目标A/B是相同150样品的不同新仪器测量。蛋白参考值单位未在所读文档中确认，报告原数值单位，不猜测百分比。
- 黄芪AMM高光谱：作者论文与表格一致，720叶行来自15植株，三化学目标单位mg/g且植株内标签重复。应以15植株分组评价，不能随机把720行分进训练/测试后宣称独立外测。本轮登记并审计，尚未纳入成分主模型。
- 芒果跨季数据：已下载v1原始11691条谱，有四季与温度/品种信息；每果重复扫描、部分多温测量而无明确fruitID，暂不以谱行做K-shot果实校准。最新v6的元数据用于识别家族限制，不冒充已下载v6。

## 全部记录

“本地”只代表已下载并完成结构检查，不能自动理解为标签/物种/独立样品均已核验。原始详细字段与证据链接见[药材JSON](../data/external/research_stage2_herbs.json)、[迁移JSON](../data/external/research_stage2_transfer.json)及[检索CSV](../data/external/research_stage2_catalog.csv)。

| 类别 | 数据记录 | 谱技术 | 当前状态 | 主来源 |
| --- | --- | --- | --- | --- |
| 药材 | Astragalus geographic origin Raman | Raman | 本地下载/已解析 | [来源](https://pypi.org/project/qsi-tk/1.4.1/) |
| 药材 | Astragalus geographic origin UV-Vis | UV-Vis | 本地下载/已解析 | [来源](https://pypi.org/project/qsi-tk/1.4.1/) |
| 药材 | Dendrobium three-origin Raman | Raman | 本地下载/已解析 | [来源](https://pypi.org/project/qsi-tk/1.4.1/) |
| 药材 | Five-origin raspberry Raman | Raman | 本地下载/已解析 | [来源](https://pypi.org/project/qsi-tk/1.4.1/) |
| 药材 | Dendrobium two-origin Raman subset | Raman | 本地下载/已解析 | [来源](https://pypi.org/project/qsi-tk/1.4.1/) |
| 药材 | Dendrobium region-and-brand Raman | Raman | 本地下载/已解析 | [来源](https://pypi.org/project/qsi-tk/1.4.1/) |
| 药材 | Fritillaria species-region Raman | Raman | 本地下载/已解析 | [来源](https://pypi.org/project/qsi-tk/1.4.1/) |
| 药材 | Bupleurum mixed origin/cultivation/processing Raman | Raman | 本地下载/已解析 | [来源](https://pypi.org/project/qsi-tk/1.4.1/) |
| 药材 | Polygonatum wild/cultivated Raman | Raman | 本地下载/已解析 | [来源](https://pypi.org/project/qsi-tk/1.4.1/) |
| 药材 | Polygonatum stem-type Raman | Raman | 本地下载/已解析 | [来源](https://pypi.org/project/qsi-tk/1.4.1/) |
| 药材 | Astragalus leaves hyperspectral to root bioactive compounds | VNIR hyperspectral mean spectra | 本地下载/已解析 | [来源](https://github.com/SSNssn123/AMM) |
| 药材 | AM-MHENet Astragalus multi-tissue HSI | hyperspectral imaging | 作者云盘；未下载 | [来源](https://github.com/zyl-mwy/MPAINet) |
| 药材 | Gongju geographical-origin HSI | VNIR+SWIR HSI | 仅元数据；当前下载受限 | [来源](https://data.mendeley.com/datasets/f3f8hh767x/1) |
| 药材 | Ginseng/American ginseng THz refractive indices | THz-TDS optical constants | 仅元数据；当前下载受限 | [来源](https://data.mendeley.com/datasets/hvbp5mdtxx/1) |
| 药材 | Optical Constants of Ginseng Tablets | THz optical constants | 仅元数据；当前下载受限 | [来源](https://data.mendeley.com/datasets/k4pjjjmgnr/1) |
| 药材 | NIR of Glehniae Radix and Its Adulterants | NIR | 仅元数据；当前下载受限 | [来源](https://data.mendeley.com/datasets/vf3njn9tdd/1) |
| 药材 | TCM-Herb8 / HerbFusion-8 | NIR absorbance | 仅元数据；当前下载受限 | [来源](https://data.mendeley.com/datasets/9sx7k9b4t9/1) |
| 药材 | Medicine-Food Homologous Herbs | NIR | 仅元数据；当前下载受限 | [来源](https://data.mendeley.com/datasets/f8r7cpgyjn/1) |
| 药材 | Dataset clasificación HM | NIR hyperspectral reflectance | 仅元数据；当前下载受限 | [来源](https://data.mendeley.com/datasets/kythyt9dt2/1) |
| 药材 | Spectral Repository of Medicinal Plants from Western Himalayan Regions | leaf/canopy spectral signatures (instrument/range unverified) | 仅元数据；当前下载受限 | [来源](https://data.mendeley.com/datasets/hm57zk2xvx/2) |
| 药材 | Latvian Herbal Medicines Under the Infrared Lens | ATR-FTIR | 文件访问受限 | [来源](https://zenodo.org/records/16892975) |
| 药材 | Metabolites from Ten Medicinal Trees Species Indigenous to Bowen University | FTIR | 仅元数据；当前下载受限 | [来源](https://data.mendeley.com/datasets/szvphh4hfb/1) |
| 药材 | Asparagus racemosus leaf ATR-FTIR location study | ATR-FTIR | 仅论文；无原始数据 | [来源](https://zenodo.org/records/18115642) |
| 药材 | Raman spectroscopy: a new high-performance method to detect TCM properties | Raman | 元数据/索引；未下载 | [来源](https://data.mendeley.com/datasets/v2mbrf92fd/2) |
| 迁移 | Anderson mango dry matter across harvest seasons | Vis/NIR; downloaded axis285–1200 nm; not MIR | 本地下载/已解析 | [来源](https://data.mendeley.com/datasets/46htwnp833/6) |
| 迁移 | Melamine formaldehyde FT-NIR transfer benchmark | NIR/MIR optical spectra | 本地下载/已解析 | [来源](https://github.com/RNL1/Melamine-Dataset) |
| 迁移 | IDRC2016 wheat:248 originals plus150 new samples/new instruments | NIR/MIR optical spectra | 本地下载/已解析 | [来源](https://github.com/HMzhu/CTVPdtw) |
| 迁移 | IDRC2002 pharmaceutical tablets two instruments | NIR/MIR optical spectra | 既有资料/已核验 | [来源](https://eigenvector.com/resources/data-sets/) |
| 迁移 | Cargill corn three instruments | NIR/MIR optical spectra | 既有资料/已核验 | [来源](https://eigenvector.com/resources/data-sets/) |
| 迁移 | KU Leuven raw-milk NIR eight-week temporal dataset | NIR/MIR optical spectra | 仅元数据；当前下载受限 | [来源](https://zenodo.org/records/8263430) |
| 迁移 | Butter quality/origin four-instrument NIR dataset | NIR/MIR optical spectra | 仅元数据；当前下载受限 | [来源](https://researchdata.cab.unipd.it/1890/) |
| 迁移 | Agroscope SSL 2024 two instruments and scanning setups | NIR/MIR optical spectra | 仅元数据；当前下载受限 | [来源](https://zenodo.org/records/11204174) |
| 迁移 | NeoSpectra handheld soil library | NIR/MIR optical spectra | 仅元数据；当前下载受限 | [来源](https://docs.soilspectroscopy.org/neospectra.html) |
| 迁移 | OSSL cross-laboratory MIR/VisNIR library collection | NIR/MIR optical spectra | 仅元数据；当前下载受限 | [来源](https://docs.soilspectroscopy.org/libraries.html) |
