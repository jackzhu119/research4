# 2026人参THz数据：获取与队列重叠审查

2026-10-31，北京时间。本任务没有训练模型，没有修改2025原件。**2026数值原件下载0个、0字节；公开说明已核验，数值对应仍受阻。**

## 官方记录能确认什么

[官方Mendeley V1](https://data.mendeley.com/datasets/k4pjjjmgnr/1)，DOI10.17632/k4pjjjmgnr.1，发布2026-03-20，贡献者Jamil Mai，National Tsing Hua University（新竹），声明CC BY4.0。

官方说明的Scheme_1—4是四个输入特征消融方案；每个方案包含六折的6个训练谱、6个测试谱、6个训练标签、6个测试标签CSV。4×24＝96是CV／表征展开文件数，**不是96个独立人参样品**。跨折训练表本来重复涉及同队列，跨scheme亦不能重复计样。实际外层压缩包名、大小和数值文件ID未取得，不能猜造直链或宣称已下载。

同团队[官方发表目录](https://www.ee.nthu.edu.tw/shanghua/publications/index.html)列2026论文为IEEE Access14:90179–90190，DOI10.1109/ACCESS.2026.3692252。论文使用人参／西洋参物种分类，不能解释为酸枣仁产地分类。2026数值标签编码尚未实际读取。

## 与2025包的关系不能擅自确定

现有用户上传2025包hvbp5mdtxx V2明确42压片／6文件组；本次核了原件SHA，未把已有文件当新下载。2026来自同贡献者、相近任务和物理量，只能说明有潜在来源关系，不能据此证实同队列或新队列。

[2026作者稿](https://www.researchgate.net/publication/404746543_A_Label-Free_and_Data-Efficient_Botanical_Characterization_Framework_using_Terahertz_Spectroscopy_and_Soft-Voting_Ensembles)方法声称每片来自不同单根，Figure3却描述43压片。该身份声明属于2026作者稿，**不能在数组映射未核实前赋给2025的42片**。稿件表中的LDA38.10%与16/42相容，ANN97.67%与42/43相容；这里只登记分母核查线索，不推造第43个样品，也不将论文准确率当本仓库复现成绩。作者稿含未编辑模板字样，最终版本仍需与原件逐项核对。

没有2026数组，跨版本精确／近值比较均NOT_RUN；不能把2025与2026样本数相加。正式分析前须恢复真实样品ID、检查不同方案和折的重复及预处理范围，并在原始n／alpha共同轴上作可追溯对应。现有42片的数据接口与实验由另一个独立任务负责，本目录不改变它的unknown上游身份。

## 实际获取尝试和边界

`http_attempts.json`保留三个普通HTTPS实测：官方页面、public-api版本、public-api文件端点均URLError／Tunnel403。保持继承的代理和TLS，没有试图修改允许列表、绕过代理或拿截图补成数值。网页工具可读官方介绍，但未取得API文件ID；检索未定位到原作者的公开GitHub数值发布。这不是证明所有公开镜像不存在。

[获取回执](../../../data/external/public_thz_20261031_v1/ginseng2026_acquisition/acquisition_receipt.json)、`source_evidence.json`、`search_log.json`和`overlap_audit.json`分开记录元数据证据、请求、已知原件和未决问题。可审查原始公开数据来源，但不能把这份回执当数据文件或增加实测N。当前训练资格BLOCKED，模型运行0。
