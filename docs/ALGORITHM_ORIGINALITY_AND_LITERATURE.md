# 老师源码性质与论文主线文献对照

> **当前数据版本（2026-10-31）：酸枣仁仅使用[样本1—7编号版](../data/processed/zaoren_20261031/README.md)，以[有效工作簿清单](../data/active_workbooks.json)为准，共855条谱记录。** 旧未编号输入已移至`data/archive/zaoren_superseded_20261031/`供追溯，不能与新版合并或继续充当默认建模输入。下面文献及源码审查结果保留原核验范围；它们不是新版数据的模型实测。当前主线暂不加入Gaussian。

核验日期：2026-10-30。目标是回答两件事：提供的代码属于哪些既有方法、哪里出现了改动；哪些论文能帮助开展酸枣仁或其他中药的太赫兹产地鉴别、跨批次泛化及小样本校准。

## 对老师算法的判断

**现有材料以已有算法实现、教学示例及光谱任务适配为主。能够确认实现差异，尚不足以确认某个新算法由老师原创，或稿件中的全部改进已包含在当前程序中。** 这也不等于否定老师论文的研究贡献：实验设计、材料应用、测量方法、结果解释和外部验证可以构成应用研究贡献，应结合正式论文及对应代码版本评价。

“老师提供”说明资料来源；“算法最初由谁提出”“程序由谁编写或改动”“当前文件是否就是论文最终版本”是另外三个问题。文件名、修改时间或运行得到高分都不足以建立这些归属。我们的兼容修复、独立Python实现、1D-CNN及阶段二稳定筛选／校准模块，也不能反过来当作老师原件里的创新。

| 原件／算法 | 已核验的性质 | 具体证据与边界 |
| --- | --- | --- |
| CARS两份压缩包 | 已有方法源码，有版本分支及本地差异 | `carspls.m`头注明Hongdong Li及2009年CARS论文；0308包默认简化分支是源码已有版本选项，不能据此归为老师新发明 |
| UVE／MC-UVE | 已有变量筛选方法；不同版本在稳定性符号、接口等处有差异 | 噪声参照UVE、MC子采样PLS系数稳定性均有既有方法来源；两版分别使用有符号与绝对稳定性，需要明确而非混用 |
| IRIV／Random Frog | 已发表研究算法的工具箱实现 | 文件头分别有Yonghuan Yun、H.D.Li及已有文献信息，并非单纯凭文件夹名推断 |
| SVM／LS-SVM／PCA／PLS | 常见库调用、统计方法及任务封装 | `templateSVM/fitcecoc`、`trainlssvm/simlssvm`、`pca`、`plsregress`等；读Excel、划折、调核参数属于应用工作，不自动成为新学习原理 |
| BPNN | 同时有教学模板与光谱分类封装 | `BPNN.zip`原例是24维输入、四类语音信号；可用TXT调用`newff/train/sim`。不能把所有BP文件都当成小麦专用或中药专用模型 |
| SSA-SVM | 用已有SSA算法优化SVM的C、gamma；稿件有额外改进主张 | 当前程序确认二维参数寻优；小麦筋度稿件称引入自适应变异，但代码与这段描述的对应尚未建立，见下文 |
| CNN | 常见LeNet式层级与光谱输入适配 | 补零后32×32重排、卷积通道6／16／120及BN等属于可见结构。没有足够版本对照与消融证据确认新增机制，人工二维重排也不等于真实太赫兹成像 |

详细原件成员路径、行号、原方法链接及14个成员哈希核对见 [teacher_code.json](../data/research/literature_audit_20261030/teacher_code.json)，运行和接口问题另见 [算法审读](ALGORITHM_REVIEW.md)。这里没有根据后续修正版反推原件行为。

PLS文件还需单独区分：`PLS.txt`／`PLS-Prediction-shenyin.txt`实际采用X和y分别PCA再回归，不是标准PLS联合潜变量提取；这种算法语义差异也不自动构成有效创新。

### 两处尤其需要分清

**稿件的改进与手中的版本。** 《基于太赫兹光谱技术的小麦筋度鉴别I（中文）》第87个Word段落提出根据个体与平均适应度调整变异概率／幅度，第148段写种群10、迭代100、二维参数。`SSA_SVM_main.m`采用对应参数设置，但第117–188行主循环中尚未识别出该显式自适应机制。当前结论是“稿件有改进主张，源码版本未建立对应”，而不是“老师一定没有创新”或“当前包已经实现改进SSA”。

**实现变化与性能改进。** `CARS（sy）/carspls.m`存在`RMSEP/2.8`及`Rmin/2.8`的输出缩放，源码未见相应目标单位换算依据；它不能作为误差下降的模型贡献。当前SSA寻优和最终报告重复使用同一折分数，也不能用来证明独立泛化。修复这些问题是必要的复现工作，不能包装成一种新算法。

## 与主线最接近的文献应分三组阅读

第一组看相同中药和相同产地任务，第二组看相近THz测量与模型，第三组看跨域和小样本评价。三组能互相帮助，但不能把NIR／高光谱图像、FTIR和THz-TDS的数据或结论直接互换。论文给出数据可用性声明也不等于本仓库已经取得其原始数据。

建议优先精读的6篇及补充3篇见[阅读顺序记录](../data/research/literature_audit_20261030/selection.json)，其余论文用于补充方法边界和既有工作对照。

### 同为酸枣仁：先读研究问题与实验设计

| 论文 | 与本课题的对应 | 应如何使用 |
| --- | --- | --- |
| **Identification of geographical origin of semen ziziphi spinosae based on hyperspectral imaging combined with convolutional neural networks**，*Infrared Physics & Technology* 136（2024）104982。[DOI](https://doi.org/10.1016/j.infrared.2023.104982) | 同一药材、五个产地；比较图像CNN及一维光谱CNN、PLS-DA、SVM，讨论HPLC与成分解释 | 这是选题设计的直接参考。测量为高光谱，不能当作你的THz时域／n／α来源。本轮读到出版社摘要及Highlights，未核验独立批次划分或公开原始数据 |
| **Research on the identification of Ziziphi Spinosae semen and its adulterants based on the fusion of hyperspectral technology and transfer learning**，*Spectrochimica Acta A* 346（2026）126907；2025-09在线。[PubMed](https://pubmed.ncbi.nlm.nih.gov/40945004/)，[DOI](https://doi.org/10.1016/j.saa.2025.126907) | 同一药材，图像迁移学习、波长选择及图谱融合 | 任务是酸枣仁与掺伪品识别，不能当作产地识别或THz跨仪器校准结果。摘要级核验，完整划分及源目标样品关系未确认 |

### 中药THz：看测量、特征与已有算法组合

| 论文／DOI | 与本课题的关系 | 需要保留的区别 |
| --- | --- | --- |
| **Rapid determination of Panax notoginseng origin by terahertz spectroscopy combined with the machine learning method**，*Spectroscopy Letters* 55（2022）566–578，[10.1080/00387010.2022.2125017](https://doi.org/10.1080/00387010.2022.2125017) | 三七产地，改进鲸鱼算法IWOA优化SVM；与老师SSA-SVM思路近 | 摘要可核验，具体仪器／THz-TDS模式、独立样品数和跨批次验证未核验；不假定原始矩阵已公开 |
| **Rapid identification of Rhizoma Coptidis origin using terahertz spectroscopy and deep learning**，*Spectrochimica Acta A* 349（2026）127283，[10.1016/j.saa.2025.127283](https://doi.org/10.1016/j.saa.2025.127283) | 黄连四产地，THz-TDS；ICOA优化GRU参数并结合MSA注意力 | 与“优化器＋深模型＋药材产地”非常接近，说明这类组合已有先例。摘要未核验样本数及独立批次评价，不据此补造准确率 |
| **Identification of Pueraria lobata origin using terahertz precision spectroscopy and CNN-transformer hybrid network algorithm**，*Spectrochimica Acta A*（2026），[10.1016/j.saa.2025.127212](https://doi.org/10.1016/j.saa.2025.127212) | 葛根8产区，CNN与Transformer结合，并有HPLC解释 | 本轮未确认FTIR与TDS采集方式，验证细节以[记录中的阅读范围](../data/research/literature_audit_20261030/direct_thz.json)为限；不能把注意力／Transformer这一名称本身作为新颖性 |
| **A Label-Free and Data-Efficient Botanical Characterization Framework Using Terahertz Spectroscopy and Soft-Voting Ensembles**，*IEEE Access* 14（2026）90179–90190，[10.1109/ACCESS.2026.3692252](https://doi.org/10.1109/ACCESS.2026.3692252) | 人参n／α、神经网络软投票与4套特征消融；最贴近当前光学常数输入 | 物种二分类而非产地；对应2026仓储。当前用户上传42片是2025仓储，不能自动当同队列或把两库样本数相加。主来源为作者机构出版目录及仓储，详细结构尚未全面核验 |

2025人参仓储关联题名 **Discrimination between Panax Ginseng and Panax Quinquefolius with Deep Learning Terahertz Spectroscopy** 的正式期刊版本本轮未核验，不能直接称其为SCI期刊论文。2026文章题名中的“Label-Free”指传感检测无需外源标记，不表示监督分类没有类别标签；用户上传包中6个`label_*.csv`确实含物种标签。详见[原包实际检查](../data/external/ginseng_hvbp5mdtxx_v2/README.md)。

以下几篇也很相关，但应单独定位：

- **Identification of Panax notoginseng origin using terahertz precision spectroscopy and neural network algorithm**，*Talanta*（2024），[10.1016/j.talanta.2024.125968](https://doi.org/10.1016/j.talanta.2024.125968)。全文使用Bruker VERTEX 80v的FTIR、1.5–16 THz，不是你的原始THz-TDS波形；可参考特征模型及化学解释，不能直接声称同技术复现。
- **Categorizing Chinese Wolfberry of Different Geographical Origins by Terahertz Spectroscopy and Machine Learning Algorithms**，*ACS Food Science & Technology*（2025），[10.1021/acsfoodscitech.5c00390](https://doi.org/10.1021/acsfoodscitech.5c00390)。药食材料与产地任务很接近；但期刊官方当前列为**ESCI**，此处不计作已核验SCIE文献。
- **Determining the geographical origin of Fritillaria by terahertz spectroscopy and machine learning algorithms**，*Chemical Physics Letters*（2025），[10.1016/j.cplett.2025.142350](https://doi.org/10.1016/j.cplett.2025.142350)。摘要涉及不同贝母物种，需防物种与地理混杂，不能等同于同一种酸枣仁的多产地验证。
- **Classification of wheat grain varieties using terahertz spectroscopy and convolutional neural network**，*Journal of Food Composition and Analysis* 129（2024）106060，[10.1016/j.jfca.2024.106060](https://doi.org/10.1016/j.jfca.2024.106060)。模型家族与老师资料很近；原文说明CARS特征用于传统模型、CNN使用全谱，不能一概称CARS-CNN。作者及组合相近不足以证明它与上传稿件或代码一一对应。

### 稳定筛选、泛化与小样本：决定后续实验如何设计

| 必读方法文献 | 可以借鉴 | 不能直接推断 |
| --- | --- | --- |
| **Algorithm of Stability-Analysis-Based Feature Selection for NIR Calibration Transfer**，*Sensors*（2022），[10.3390/s22041659](https://doi.org/10.3390/s22041659) | SAFS以主从仪器间稳定频段筛选结合校准迁移，是“稳定筛选＋迁移”的直接先例 | 当前单源重采样入选频率不是该方法的原样实现；重复UVE再投票也已有EMCUVE先例，不能据组合宣称首次 |
| **A task-oriented few-shot spectroscopy benchmark: Empirical evaluation of meta-learning approaches**，*Engineering Applications of Artificial Intelligence* 167（2026）113593，[10.1016/j.engappai.2025.113593](https://doi.org/10.1016/j.engappai.2025.113593) | 按光谱任务建立support/query，比较微调、MAML、ProtoNet等；含按年份／地区设计的任务及[公开代码](https://github.com/ivgase/spectroscopy-fewshot-benchmark) | 是NIR／MIR基准，不是中药THz结果；小样本回归与每类K-shot分类需区分。不能只从同一批扫描随机拆小训练集就称跨域小样本 |
| **Domain-Invariant Partial-Least-Squares Regression**，*Analytical Chemistry*（2018），[10.1021/acs.analchem.8b00498](https://doi.org/10.1021/acs.analchem.8b00498) | diPLS提供域对齐与预测目标相结合的回归思路 | 原论文为回归；产地分类需要说明扩展，不能直接当PLS-DA同一算法 |
| **Deep calibration transfer: Transferring deep learning models between infrared spectroscopy instruments**，*Infrared Physics & Technology*（2021），[10.1016/j.infrared.2021.103863](https://doi.org/10.1016/j.infrared.2021.103863) | 若走CNN路线，参考已有模型如何跨仪器微调 | 不使用配对标准样本不等于不需要目标标签，也不等于已证明跨中药物种迁移 |

若使用DS／PDS作校准基线，再读 **Multivariate instrument standardization**，*Analytical Chemistry*（1991），[10.1021/ac00023a016](https://doi.org/10.1021/ac00023a016)，先确认是否有主从设备测过的配对标准样品。其余方法条件、阅读范围与代码／数据状态见 [transfer.json](../data/research/literature_audit_20261030/transfer.json)，既有新颖性审查见 [阶段二方法审查](../reports/research-stage2/2026-10-30/method_audit.md)。

## “去SCI看”的核验范围

本轮通过出版社页面、PubMed及机构仓储定向核对题名、DOI、摘要或可读全文，**未登录Web of Science，也没有逐篇核验WoS入藏号、JCR分区或影响因子**。不把有DOI、英文发表、PubMed收录自动当作SCIE证明。

官方来源本轮能够明确确认：

- [Spectroscopy Letters](https://www.tandfonline.com/journals/lstl20/about-this-journal)、[IEEE Access](https://ieeeaccess.ieee.org/about/bibliometrics/)、[Journal of Infrared, Millimeter, and Terahertz Waves](https://link.springer.com/journal/10762)列出SCIE。
- [ACS Food Science & Technology](https://pubs.acs.org/afsthl/pages/About)列出Web of Science ESCI，不应混写成SCIE。
- 部分Elsevier期刊的当前索引页不可读，本轮未取得官方当前证明；“未核验”不表示该刊不被SCIE收录。证据、尝试及限制逐条见 [journal_index.json](../data/research/literature_audit_20261030/journal_index.json)。

Springer的[JIMTW投稿范围](https://link.springer.com/journal/10762/aims-and-scope)还明确限制仅使用通用PCA／机器学习进行信号分类的稿件，以及小幅延伸或复制。这是该刊具体要求，不能泛化为所有SCI期刊的统一标准；它提醒本课题需要明确的科学问题和证据，而非只增加模型名称。

## 建议收紧成怎样的研究问题

优先研究：**同一种酸枣仁，在已知生长产地类别内，跨批次测量时，n与α的特征选择及融合能否维持识别能力；加入少量目标批次有标签样品后，能改善多少。** n和α来自同一次THz测量，是两种派生物理量，不能自动称两个独立传感模态。这个方向值得检验，尚未建立新颖性或优越性。

应先比较全谱SVM／PLS-DA、普通CARS／MC-UVE、简单特征拼接，再考察一种清楚定义的稳定筛选或融合候选。SSA与同预算随机／网格搜索作对照；轻量1D-CNN在有足够独立样本后加入。修参数时只访问训练区，不根据最终测试结果选择“最优改进”。

研究设计需要区分两件事：

1. **已知产地的新批次。** 同一组产地类别均在源训练中出现，按独立样品及批次留出测试；允许用少量目标批次support样品校准，固定query评价。报告宏平均F1、平衡准确率、各产地召回率、混淆矩阵、校准增益及波动。
2. **新增产地类别。** 源训练完全没有该产地，是新类别／开放集或小样本增类问题。普通SVM不会自动输出从未训练过的产地名，不能与前一问题混为同一泛化指标。

用户已确认相关地名是生长地，并说明每份当前工作簿对应一个产地；ZY01—ZY07据此表示文件产地组。当前855条谱记录（350／5／100／100／100／100／100）仍需真实地名、物理样品、重复扫描及批次映射，不能直接视为855个独立样品。当前人参公开包没有批次／原植物映射，只适合初步物种分类流程核对，不能支撑跨批次泛化结论。指定成分含量检测作为后续独立问题，需要化学参考值，产地标签不能替代浓度。

本轮是定向文献与源码证据审查，没有运行新模型，没有把论文报告的准确率当作本项目结果，也没有进行全面的新颖性证明。先把任务、数据独立性及对照设计做实，再决定是否需要一个新的算法机制。
