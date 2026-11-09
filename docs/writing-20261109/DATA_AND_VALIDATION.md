# 数据版本、标签依据与验证范围

整理日期：2026-11-09。本页为论文写作资料，只整理现有证据；没有重训、修改标签、重新划分或新增样品。完整18组来源与工作簿hash见 [data_lineage.csv](data_lineage.csv)。后续机制结果见 [DEEP_MECHANISMS_RESULTS.md](DEEP_MECHANISMS_RESULTS.md)。

## 写新稿应引用哪个数据版本

**本B工作树根目录的 `data/CURRENT_DATASET.md` 仍保留2026-10-31旧855份说明，不能当成新稿当前版本。** 本轮实验绑定的是封存源提交 `21d6edab24d198dc60b1ad4ef50e83602c9ecfa7` 中的 `2026-11-08-confirmed-origin-kind-v1`，不是重新递归发现根目录或归档数据。下列不可变GitHub链接指向该源提交，保留跨分支来源边界。

| 版本或范围 | 记录数 | 含义及写作边界 |
| --- | ---: | --- |
| 原ZY01—ZY07 | 855 | 原匿名来源组。现标签解释为355枣仁＋500酸枣仁；历史七分类不能继续称纯酸枣仁产地识别。 |
| 旧元数据登记 | 1258 | 原855＋黄芪403；更早808行模板也只是历史版本，均不代表本轮1672范围。 |
| 2026-11-08确认版本 | 1672、18本 | 原855＋黄芪403＋天麻405＋新增枣仁9。四个中文类别按用户确认登记。 |
| 本B的Yanan v2/v3数值输入 | 999 TRAIN | 216枣仁＋300酸枣仁＋241黄芪＋242天麻；不是1672份全部参与新机制训练。 |

“新增817份分配角色”是相对于原855份历史划分而言，即黄芪403＋天麻405＋新枣仁9；不能把817全部称为新采集数据。现行登记相对旧1258份增加的是414份（天麻405＋新枣仁9）。数据复制、重新编号、文件改名及重复上传均不新增观察。

## 固定训练、校准和预测名单

| 用户确认中文类别 | TRAIN | CAL | PRED | 合计 |
| --- | ---: | ---: | ---: | ---: |
| 枣仁 | 216 | 74 | 74 | 364 |
| 酸枣仁 | 300 | 100 | 100 | 500 |
| 黄芪 | 241 | 80 | 82 | 403 |
| 天麻 | 242 | 80 | 83 | 405 |
| 总计 | 999 | 334 | 339 | 1672 |

| 来源组 | 用户中文类别 | 用户产地标签 | 全部 | TRAIN | CAL | PRED |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ZY01 | 枣仁 | 陕西延安 | 350 | 210 | 70 | 70 |
| ZY02 | 枣仁 | 云南 | 5 | 3 | 1 | 1 |
| ZY03 | 酸枣仁 | 内蒙古 | 100 | 60 | 20 | 20 |
| ZY04 | 酸枣仁 | 山西灵石 | 100 | 60 | 20 | 20 |
| ZY05 | 酸枣仁 | 安徽亳州 | 100 | 60 | 20 | 20 |
| ZY06 | 酸枣仁 | 河北石家庄 | 100 | 60 | 20 | 20 |
| ZY07 | 酸枣仁 | 山西 | 100 | 60 | 20 | 20 |
| HQ01 | 黄芪 | 曲靖 | 100 | 60 | 20 | 20 |
| HQ02 | 黄芪 | 内蒙古固阳 | 100 | 60 | 20 | 20 |
| HQ03 | 黄芪 | 甘肃 | 100 | 60 | 20 | 20 |
| HQ04 | 黄芪 | 山西大同 | 103 | 61 | 20 | 22 |
| ZR01 | 枣仁 | 陕西榆林 | 3 | 1 | 1 | 1 |
| ZR02 | 枣仁 | 辽宁 | 3 | 1 | 1 | 1 |
| ZR03 | 枣仁 | 内蒙古 | 3 | 1 | 1 | 1 |
| TM01 | 天麻 | 贵州雷山 | 100 | 60 | 20 | 20 |
| TM02 | 天麻 | 吉林长白山 | 102 | 61 | 20 | 21 |
| TM03 | 天麻 | 安徽毫州 | 103 | 61 | 20 | 22 |
| TM04 | 天麻 | 云南 | 100 | 60 | 20 | 20 |

表中的地名保持用户登记文本；天麻“安徽毫州”按原登记写法保留，没有擅改地名。山西与山西灵石是工作簿来源组，不据此宣称其地理范围互斥。ZR01—ZR03各只有3份，划分为1/1/1；ZY02仅5份，划分为3/1/1。单份预测的召回率只作诊断，不能支撑稳定的小产地识别或正常分层调参。

主划分seed为20261101。原855份逐样品继承已保存的513/171/171角色，没有重新随机；新增817份在组内按记录ID排序，以 `SHA256(seed:group)` 派生组级PCG64种子，按约60%/20%/20%补齐，未使用光谱或成绩分配角色。本次小表复核确认1672个ID唯一、855个历史角色逐项相同、999个TRAIN ID与封存数据生产器元数据完全相同。原855份已用于此前多轮内部CV和开发；保留新流程的互斥角色并不能消除历史暴露，固定PRED也不是外部盲测。

## 陕西延安两套RAR的重复关系

两套分卷外名分别为“陕西延安枣仁202605210834”和“陕西枣仁202605210834”。RAR容器本身hash不同，但各自完整解出的XLSX均为54,540,804字节，SHA256均为：

`5c569b7c6be91157ca2a0f738ac7a10d2fdd7b5644b7abbffd6ecdb74354ce90`

既有审查同时做了全字节比较，结果相同。这两份是同一份350记录数据的两次打包，登记中ZY01只出现一次，仍为210 TRAIN／70 CAL／70 PRED。四张表（时域、频域、折射率、吸收系数）是同350个记录的配对视图，不能计为1400；两套上传也不能计为700。

当前编号版/改名版“陕西延安枣仁.xlsx”的SHA256为：

`e9dd6398868979036ec98e7c722b17afa9319ac0289280233452f604407fe292`

它与原件hash不同来自1400个样品表头编号修改。既有XML逐成员核查确认：4个sheet去掉首行后字节相同，另外9个包成员完全相同，数值没有因此变成另一批数据。后来仅改磁盘文件名也未改变该编号版字节。因此本次重复上传新增独立记录0，现用1672及旧855计数均不变；原登记没有把两个上传副本重复纳入模型，不能把“去除重复上传”写成提高模型成绩的新实验。

## 标签依据和未确认事项

“枣仁、酸枣仁、黄芪、天麻”是用户确认的中文类别标签，不是经独立植物学鉴定的拉丁学名标签。产地和种类依据2026-11-08确认记录；其不可变源文件为 [用户产地/种类确认](https://github.com/jackzhu119/research4/blob/806584a5571c6a90ef37c6bfa915671b680ba48b/data/annotations/origin_kind_confirmation_20261108.json)，SHA256 `68fc2dfbbf0d30cb97835655b74791c42fa37ff184eace2c8c16f10b042ba682`。

原855个ZY记录依据2026-10-31用户声明“每个编号均是独立样品”；另817份依据2026-11-08答复“每列都是独立实物样品”。`physical_sample_id=record_id` 是这些声明的显式登记，不是从编号或文件列数推定。没有做采样日志/凭证核验。原855份每来源只有一批也是用户声明；该声明不能外推为新增各组批次数。逐样品主名单的1672个 `batch_id` 均为空，当前无法分离来源与来源批次效应。

对于ZY01，350个时域头均带“1.5mm”声明，但没有实测厚度字段，批次、仪器和采集日期也为空。文件名时间戳和样品序号不代表经验证的测量时序。既有控制中23个错分针对210个留出TRAIN（187/210正确），不能写成整本350份均为酸枣仁。该控制的其它来源枣仁训练供体只有6份、酸枣仁供体240份，类别加权已使用；来源覆盖与测量/表示差异仍是竞争解释。**光谱分类、阈值校正或跨种迁移的表现都不能据此把ZY01改标为酸枣仁。** 包装、供货/采样记录或独立鉴定目前未补齐。

黄芪甘肃当前采用补齐折射率表的完整版本，旧不完整版本保留为历史；写作/复现应按现行清单的工作簿SHA绑定输入，不混用相似文件名的旧版本。

## 各轮验证使用什么数据

| 实验范围 | 数值训练与评价边界 | 不应混淆的事项 |
| --- | --- | --- |
| 旧855七组CV | 当时原855匿名组内部CV；全部已历史暴露 | 旧96.0234%保留为混合种类来源组内部验证，不能移作纯酸枣仁或新独立验证成绩。 |
| 源提交21d6的标签纠正分类 | 所有任务共用999/334/339主名单；训练内选参，CAL仅指定适配，PRED评分；该轮实际主实验用α | 6项不同任务和R0/R2等分支不能事后按PRED挑赢家。整条“先种类、后产地”系统没有由单项成绩相乘得到。 |
| B的TRAIN-only来源控制与Yanan v2 | 只读取999 TRAIN光谱；二分类使用其中516，辅助HQ/TM为241/242 | source-held群组对照与主名单CAL/PRED是不同评价边界。 |
| B的v3阈值/供体机制 | 同一封存TRAIN；α、n及分别SNV后融合；每个ZY01/酸来源的初始source对照由6个其它枣仁＋240酸枣仁供体拟合（完整重训另加入允许的支持样品） | 支持和query都从该次留出的TRAIN群组分开；没有调用主名单334 CAL或339 PRED。 |
| B的v3神经/BN机制 | 同一516二分类TRAIN及483辅助TRAIN；编码器种子42—49×目标训练种子101/202/303，固定各折和对照 | BN重估仅用源供体；查询不更新参数、统计量或方法。种子方差不是生物样本不确定性。 |

Yanan v2封存数据生产器记录 `nontrain_numeric_cells_decoded=0`、`numeric_scope=master_train_only`、所有ID/种类/组/来源列/角色核查通过。主轴为共同原始612点（约0.100708—1.499176 THz），不插值；神经输入沿用128点0.21—1.49 THz线性网格，无外推。α、n按记录ID/列/轴严格配对；逐谱SNV的均值/标准差只用自身谱，StandardScaler及模型只拟合允许的source。访问接口、拟合收据、输入hash及封存证明是本轮工程审计证据，不等于未使用过的外部样品或实验室真实性鉴定。

v3每个支持集条件是5个酸来源×20次抽样×135次query，即13,500行预测，但只有255个唯一历史ID；每个神经主条件是516个ID×24种子组合，即12,384行预测，唯一ID仍为516。不能把这些重复行、971个任务、增强/缩放副本或同一记录四种信号当新增独立样本。v3具体结论与完整对照见 [机制结果](DEEP_MECHANISMS_RESULTS.md)。

## 用户此前给出的成绩对应哪轮

以下数字在源提交21d6的历史报告和逐样品预测中确实存在。本次写作核查读取小型预测CSV（非谱矩阵）按真值/预测重新整数计数，ID数均等于各自分母，`is_correct` 一致；它们不是本次971项v3的成绩。

| 历史任务/模型 | 源模型正确数/固定PRED数 | 加入全部CAL后完整Pipeline重训 | 解释 |
| --- | ---: | ---: | --- |
| 枣仁/酸枣仁二分类，R2 | 174/174 | 174/174 | 只说明此历史内部PRED上的分类表现，不等于植物身份鉴定。 |
| 酸枣仁五来源，R2 | 87/100 | 93/100 | 93%用额外100个CAL标签并重新拟合整个Pipeline，不能称K-shot冻结模型校准。 |
| 四种药材，固定全谱SVM | 331/339 | 330/339 | 重训未必提高；不得省去负向结果。 |
| 黄芪四来源，固定全谱SVM | 69/82 | 69/82 | 内部来源组识别，非独立批次迁移。 |
| 天麻四来源，固定全谱SVM | 81/83 | 82/83 | 分母83，不能拿准确率与其它分母任务直接比较成“更易识别”。 |
| 枣仁五来源诊断，固定全谱SVM | 72/74 | 72/74 | 70/74为ZY01；两小类各1份预测均错，宏F1仅约0.597、平衡准确率60%。 |

表中的R2指修订标签后、在TRAIN内重新选择的 `R2_reselected` 流程，不表示直接复用了旧混种模型的已训练权重。

历史报告还保留酸枣仁R0的94/100→96/100、固定MC20的87/100→82/100等分支；不能看完PRED后仅选择96%代表预先锁定方案。历史纯酸枣仁五类迁移B平均79%、A81%的结果，与最新B的二分类TRAIN机制实验任务和分母不同，不应直接相减解释为方法已因标签修正而进步。

## 可用于方法段的事实及待补验证

可据实表述：本研究保留各版本输入hash和逐记录ID；按用户确认的四个中文类别更正原匿名组任务，所有任务共享固定角色。原855份角色逐项继承，新增组以不使用光谱的确定性规则分配。初始训练只用预定source，完整重训明确加上允许的支持或CAL标签，支持与query互斥；报告样品ID数、重复预测数、宏F1、平衡准确率及少数来源召回。

结论应限定为现有历史数据中的内部识别与算法机制。当前没有经确认并用于外部验证的独立新批次、盲法植物学鉴定或已记录的跨仪器确认；不得写成“证明陕西延安材料实际为酸枣仁”“已建立普适跨物种校准迁移”或“100%外部鉴别”。光谱缩放/SNV不变性是表示层面的数学对照，不是制样厚度或真实仪器漂移的因果实验。

写作中应把以下事项标为未完成或下一阶段需要，而非已经跑出的实验：独立鉴定/凭证与材料留样；覆盖各来源的新批次及真实批次ID；厚度、含水率、仪器/日期/参考波形的采集记录；按实物样品和批次分组的外部确认；完整两级分类流程的端到端错误率；未见药材拒识。化学成分定量、药效、质量等级、掺伪比例还需要相应化学参照或预设混合样品，当前种类/产地标签不能提供这些结论。

## 可核查证据与SHA256

不可变上游源提交 `21d6edab24d198dc60b1ad4ef50e83602c9ecfa7`：

| 证据 | SHA256 |
| --- | --- |
| [现行18表清单](https://github.com/jackzhu119/research4/blob/21d6edab24d198dc60b1ad4ef50e83602c9ecfa7/data/active_workbooks.json) | `dd9ec7728590dde2574f64b9664148465c85d7b618311494c613c32b4d879f95` |
| [现行数据说明](https://github.com/jackzhu119/research4/blob/21d6edab24d198dc60b1ad4ef50e83602c9ecfa7/data/CURRENT_DATASET.md) | `a147144953566beed3fdbc2b58451102ec64902e1f5f18ddeb2618af6df2b4e7` |
| [逐样品主划分](https://github.com/jackzhu119/research4/blob/21d6edab24d198dc60b1ad4ef50e83602c9ecfa7/data/research/species_revision_20261108/split_manifest.csv) | `0767e3165984a4e0e4ae782d69fdea6f410bfe36bce05ec72f8eadca982317d4` |
| [主划分协议](https://github.com/jackzhu119/research4/blob/21d6edab24d198dc60b1ad4ef50e83602c9ecfa7/data/research/species_revision_20261108/protocol.json) | `a4d31c9a23711df97b2e3e883fd44065c7cbd5bbce3bd65c4efb46b6f6905f0c` |
| [855历史角色](https://github.com/jackzhu119/research4/blob/21d6edab24d198dc60b1ad4ef50e83602c9ecfa7/data/research/species_revision_20261108/historical_split_plan.json) | `17055eef5a4c544344f735032b55ef12f52bde341ffdff9378b25a835baa9594` |
| [817样品独立性声明](https://github.com/jackzhu119/research4/blob/21d6edab24d198dc60b1ad4ef50e83602c9ecfa7/data/annotations/specimen_independence_user_declaration_20261108.json) | `752c337d461b97304d9c27057afd473ff8ab7fe15fb5816e5130bed5f7c6cad6` |
| [历史标签纠正分类报告](https://github.com/jackzhu119/research4/blob/21d6edab24d198dc60b1ad4ef50e83602c9ecfa7/docs/SPECIES_REVISION_RESULTS_20261108.md) | `91ca7ace303fe6ec307207b60fc69ad6126d97725f4a1ee6867ae14a2f3d6738` |
| [历史分类指标](https://github.com/jackzhu119/research4/blob/21d6edab24d198dc60b1ad4ef50e83602c9ecfa7/data/research/species_revision_20261108/classification/metrics_overview.csv) | `c14f673461275cf1166ff59042c39785b20ae2689d9256231e8f2597eca5709b` |

本B既有审计和封存结果（保持原字节）：

| 证据 | SHA256 |
| --- | --- |
| [两套上传重复结论](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-upload-duplicate-audit-20261108/duplicate_verdict_zh.json) | `ea2247d36dd46b438f8dbf4625d4b32d6d90d48bfc120ddc86da6096d3c4b6d1` |
| [重复上传对登记计数的影响](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-upload-duplicate-audit-20261108/registry_count_impact_review.json) | `b3366eed302ef49bacc95863003ddfe0884ded85c039a006cfd4cf7565a9319e` |
| [原件与编号版逐成员核验](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-upload-duplicate-audit-20261108/original_vs_registered_numbered_content.json) | `af857bf99c1bbccf03c677cbc3339e0d428f7bdc463bcf7f35727ece2fe8485a` |
| [999 TRAIN封存元数据](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/science/prepare_data-attempt001/dataset_metadata.json) | `e1b13276e0eae291f68fd994549632801d68f3b162b56304f4106a6aa490d0bf` |
| [999 TRAIN准备结果](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/science/prepare_data-attempt001/result.json) | `cc5f55eadc5be74b9798fe64e71ca43689c9a5d21f63420477a067bacc9e2788` |
| [ZY01最初因果核查](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/analysis/initial_ZY01_causal_audit.json) | `75bae76b5c68309404ba570ef7126a7b982d4906d5a2d612d5e1e7fa239fdfa2` |
| [v3冻结机制协议](../../reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/preparation/scientific_protocol_ROOT.json) | `60d33b662410f1e7729a28cf585ec255616a47b51be151fa10f4ef0278264ba1` |

历史逐预测文件（这里只读取小型预测表，未读取原始谱矩阵或重载模型）：

| 文件 | SHA256 |
| --- | --- |
| [binary_kinds逐样品预测](https://github.com/jackzhu119/research4/blob/21d6edab24d198dc60b1ad4ef50e83602c9ecfa7/data/research/species_revision_20261108/classification/binary_kinds/predictions.csv.gz) | `11673bffcbb45066b96095ca040880bc27a4064554c89d256c843b3a4f1c352d` |
| [acid_origins逐样品预测](https://github.com/jackzhu119/research4/blob/21d6edab24d198dc60b1ad4ef50e83602c9ecfa7/data/research/species_revision_20261108/classification/acid_origins/predictions.csv.gz) | `74e6c86a5b60f3c27768b26d7228801c9c4c1391fea97472d316de539e061dfb` |
| [four_kinds逐样品预测](https://github.com/jackzhu119/research4/blob/21d6edab24d198dc60b1ad4ef50e83602c9ecfa7/data/research/species_revision_20261108/classification/four_kinds/predictions.csv.gz) | `233a62099c198db2b2487aa4f27f0f7b8ea776687fa119341f3187d380db1def` |
| [huangqi_origins逐样品预测](https://github.com/jackzhu119/research4/blob/21d6edab24d198dc60b1ad4ef50e83602c9ecfa7/data/research/species_revision_20261108/classification/huangqi_origins/predictions.csv.gz) | `3b6624ac797ba6be487858f84046392b5ae2f493caf56ca8a00f3698e0f52a60` |
| [tianma_origins逐样品预测](https://github.com/jackzhu119/research4/blob/21d6edab24d198dc60b1ad4ef50e83602c9ecfa7/data/research/species_revision_20261108/classification/tianma_origins/predictions.csv.gz) | `90df4a91c5043721a6aa055e968aae69a8c50ce5eca8135c48de92e9d47ff85d` |
| [jujube_origins逐样品预测](https://github.com/jackzhu119/research4/blob/21d6edab24d198dc60b1ad4ef50e83602c9ecfa7/data/research/species_revision_20261108/classification/jujube_origins/predictions.csv.gz) | `a062d99a5ca9aeba9fea07a74942e88563909b3136ad3cf34223c94f2f45b703` |

本页审查只核对小型注册/划分/元数据及已保存预测，不新解码工作簿谱值或NPZ/PT，不修改原始数据、角色、旧结果或物种标签。重新整理后的数据来源表SHA以 [data_lineage.csv](data_lineage.csv) 的实际发布版本为准。
