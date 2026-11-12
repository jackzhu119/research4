# Research4：新增检索、已有外部数据与训练资格复核

本次核验日期：2026-11-04（北京时间）。**本次新增取得的外部数值光谱数据集为0，新增模型拟合为0。** 在线主源再核验、真实下载失败及本地来源核查分别保存，不能把可读网页写成已下载原始光谱。旧数据、旧结果和855份酸枣仁主任务均未更改。

## 对当前研究最有用的结论

现有黄芪403条、天麻405条及人参／西洋参42片仍是可以执行独立任务的现成资源。黄芪、天麻按用户独立样品声明及匿名组做探索性分类；人参按已知两物种做观测级分类。原料祖先、批次、仪器及部分纵轴单位尚未核清，结果不能扩写成真实跨批次、全国来源或跨仪器确认。来自同压片的折射率和吸收系数始终只计一份观测。

公开检索中较接近的Mendeley人参2026版、稻谷THz数据、玉米DeepSpectra数据及Dryad小麦含量数据，均已在仓库旧目录登记。本轮重新读了主源，**没有将它们计作新发现或新独立队列**。当前机器向Mendeley、Zenodo、Dryad、Cambridge的四次合法GET均被继承代理CONNECT403拒绝；网页工具可读部分元数据，但没有取得这些数据集的新数值字节。详见[下载实测](../reports/external-evidence-20261104-v1/spectra/network_attempts_20261104.json)。

## 可执行与受阻来源

| 来源 | 已核内容与粒度 | 当前资格／使用边界 |
| --- | --- | --- |
| 自有黄芪／天麻 | 403／405条谱，四个匿名来源组；用户声明为独立样品。完整导出2578坐标，现有主窗口接口612坐标 | `TRAINABLE_SEPARATE_TASK`：匿名组、谱形探索；不与酸枣仁标签混合 |
| [人参／西洋参V2](https://data.mendeley.com/datasets/hvbp5mdtxx/2) | 已有18CSV；42压片、65点、两个物种；官方页明确THz、n的RIU及α的cm⁻¹，CC BY 4.0 | `TRAINABLE_SEPARATE_TASK`：已有观测级物种任务；未知原植物身份，n／α不是84样品 |
| [人参2026 V1](https://data.mendeley.com/datasets/k4pjjjmgnr/1) | 四方案×六折的预切分数据／标签；与2025队列的关系未核实，官方许可CC BY 4.0 | `BLOCKED`：未下载；96个CSV不等于96独立样品，不能与42片直接相加 |
| [稻谷THz V3](https://data.mendeley.com/datasets/wj2h5hn7kc/3) | 仓储列THz光谱／成像、CC BY 4.0；内部独立种子／像素／重复关系尚未解析 | `BLOCKED`：不能随机按像素切分后声称独立种子验证 |
| [GA21玉米DeepSpectra论文](https://www.nature.com/articles/s41538-026-00866-9)及其[数据DOI](https://doi.org/10.5281/zenodo.19150796) | 论文报告六含量组、1080压片，分析区0.2—1.2THz、121点；不是已核验的独立植株数 | `BLOCKED`：数值原件、许可和原料关系未核。不能把论文指标当本项目复算 |
| [白桦种子THz](https://zenodo.org/records/17258169) | 页面列1.1GB时域包及Time／Amplitude／Frequency／Spectrum列；可读许可字段为空 | `BLOCKED`：独立种子、标签及单位未建立；未启动巨型下载 |
| [Dryad小麦粉苯甲酸](https://datadryad.org/dataset/doi:10.5061/dryad.945c410) | 官方列data.mat、X.mat、Y.mat、fz.mat及BPNN／GRNN程序 | `BLOCKED`：原件下载失败；将来是独立含量回归任务，非产地分类 |
| [Cambridge THz教程V2](https://doi.org/10.17863/CAM.120263.2) | 已校正的乳糖／PE参考波形与处理脚本，仓储注明CC BY 4.0 | `BLOCKED`：适合物理参数提取核查；不是植物分类队列 |
| 用户小麦表 | 50个平均曲线列，不是已证实的50独立样品；标签、物理量缺失且有3个非有限格 | `BLOCKED`：先补原始对应关系，不能造类别 |

完整14条资格记录及字段在[JSON](../reports/external-evidence-20261104-v1/spectra/source_registry.json)／[CSV](../reports/external-evidence-20261104-v1/spectra/source_registry.csv)。`TRAINABLE_SEPARATE_TASK`并不意味着生物学独立性或物理兼容性已获得外部认证；每条的`permitted_scope`保留具体限制。

## 酸枣仁近红外资料有价值，但不能改名为THz

[五来源NIR＋1DCNN论文](https://doi.org/10.13386/j.issn1002-0306.2024110205)直接关联酸枣仁来源、SG导数、CARS／SPA和CNN，可作为方法比较及湿度／成分混杂的参考。本轮没有取得其逐样品原始矩阵和许可，且NIR坐标与物理机制不能按名称替换成THz。

[FT-NIR掺伪研究](https://doi.org/10.1016/j.fochx.2023.101022)的正式数据声明将所用数据标为保密；附录存在不等于原始训练谱公开。它可支持方法讨论，不能登记为已下载新训练集。文献中测得的水分、脂肪、蛋白或化学成分不对应当前855份样品，不能补填到本项目的逐样品标签。

## DeepSpectra官方代码与下载的额外核查

玉米论文把数据与代码都指向Zenodo。现有限定检索没有找到由该论文作者确认的GitHub镜像：GitHub代码搜索DOI、`DeepSpectraV2`、`DeepSpectra maize`未返回匹配；裸数字19150796命中的是无关记录。相近仓名不构成论文归属：[Gananath/DeepSpectra](https://github.com/Gananath/DeepSpectra)的作者README描述2017年的XRD晶系分类；另一同名仓README仅标题，无法验证作者关系。本轮没有执行或下载这些同名仓的代码或训练数据。

该结论只描述本次可核验的检索范围，不证明作者不存在其他公开发布。不得通过第三方代理绕过Zenodo下载限制；也不把不能访问的DOI推断为不存在。

## 现有训练证据：已经观察到负迁移

错误混种分类成绩已按2026-11-12用户决定撤回；论文使用main的paper/current白名单。

[现有公开THz验证](PUBLIC_PLANT_THZ_VALIDATION_20261031.md)也已完成数轮人参模型比较，不能再根据该已曝光留出集反复挑参数或种子。新增试验应由训练内部确定规则；所有真实负结果继续保留。

## 本轮后的具体进入条件

1. 主研究继续855份酸枣仁，保留原MC-UVE＋SVM及全部历史指标；外部数据的任务头、标签和评价样品独立。
2. 黄芪／天麻可在既有匿名任务接口上开展预先固定的小预算配对探索。共同区插值只在实测支撑范围内，不能创造新分辨率或新增N。物理单位未明时明确限于谱形实验。
3. 新公开谱先取得真实字节、来源许可、样品／重复映射和频率轴，再决定分类／回归接口。文件数、像素数、重复扫描及多模态不能充当新独立样品数。
4. 若使用新玉米数据，保留原始排除记录，并重新建立训练内部的预处理／筛选；不照搬全数据异常剔除，不按当前测试错误删除样品。
5. PySCF计算和文献用于分子输入与机制解释。相同化合物的一组全局振动频率不是855条新的带产地光谱，也不能填补逐样品含量。若后续研究物理先验约束，先在训练／开发内锁定窗口，再与同协议无先验基线比较。

本次来源核验未取得新增公开训练谱，也未进行模型拟合，因此不报告新的准确率提升。其他并行受控试验的结果由各自协议与报告提供；新资料的真实作用与阻塞原因保留在本台账中。

## 可复核证据

- [本地既有数据只读核查](../reports/external-evidence-20261104-v1/spectra/existing-local-audit/audit.json)：四份静态NPZ字节SHA与冻结审计一致，未解析活跃PySCF数组。
- [主源资格台账](../reports/external-evidence-20261104-v1/spectra/source_registry.csv)：许可、单位、粒度、同队列风险及使用范围。
- [检索与排除记录](../reports/external-evidence-20261104-v1/spectra/search_log.json)：限定查询与官方链接、同名仓排除。
- [四次机器网络实测](../reports/external-evidence-20261104-v1/spectra/network_attempts_20261104.json)：全部403，无谱文件SHA被编造。
