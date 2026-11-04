# 20261104补充数据：小预算多表示ResNet草案

状态：**DRAFT_METADATA_ONLY_NOT_EXECUTABLE**。本阶段仅小源码/元数据审查、
纯stdlib软件测试与新草案写入；没有导入Torch/NumPy、读光谱数组、准备数值
角色包、拟合或评分。PySCF独占恢复期间不运行任何数值阶段。本文件不改
活动数据registry、旧拆分、MC-UVE结果或历史模型。

## 实验要回答的问题

本批没有增加黄芪/天麻独立样品，但补齐黄芪100条n和天麻203条time。当前
可在**相同403/405个既有ID**上比较完整三表示，不能用旧缺模态子集分数
与新全集相减。酸枣仁旧855个样品四模态此前已经完整。全部旧样品和多次
query分数已曝光，新的网络和可读表示不会生成新的独立确认队列。

主比较固定三arm，使用旧compact `Encoder(width12, blocks2, dropout.35,
residual=True, attention=False)`：

| arm | 分支输入与结构 | 作用 |
| --- | --- | --- |
| alpha_single | 一个α分支，16维embedding与线性head | 普通小ResNet基线 |
| alpha_capacity3 | α128、同一α128、直接从原canonical612插值的α1024送入三个**独立、不共享权重**分支，拼接48维 | 控制分支容量和长度/计算，不增加样品、独立模态或新光谱信息 |
| alpha_n_time | α128、n128、time1024分别独立编码，拼接48维 | 在长度与容量控制之外检验新表示是否有增量 |

主融合效应看`alpha_n_time − alpha_capacity3`；另报两者各自相对单支的变化。
后两arm必须有相同初始参数形状与哈希、参数数目、seed、epoch和样品批次
顺序，并具有相同的分支输入长度[128,128,1024]。第三α分支必须直接来自
原canonical612，不能把α128再上采样1024。三个arm同轮数/更新次数不代表
单支与三支的FLOPs严格相等，计算成本
实际记录。没有预训练、L2-SP或支持校准，避免混淆新表示与额外标签收益。

## 输入准备契约：当前仍NOT_RUN

现有`data/splits/supplemental_spectra_20261104_v1`角色NPZ只有α的612点窗口，
不能当作多模态包。Q结束后另建独立compact角色包，不修改旧输入：

- 继承`all_tasks_split_manifest.csv`全部角色；用`incoming_role_map.csv`的
  sample→canonical record→entity映射显式配对每种表示，所有别名和父实体同角色。
- α保留旧角色包的canonical数值，不静默替换重导出的浮点值。n/time通过
  v2登记表的实际artifact引用补齐，旧坏表和格式等价匹配证据保留。
- incoming merged酸枣仁只有514条（旧505＋新9），缺ZY01的350条；完整855
  要从旧数据库补齐，不能把514当855。黄芪当前merged在v2，天麻部分引用
  仍在v1；必须跟registry引用而非根据目录名猜路径。
- source包可带`source_labels`；query包仅ID、实体别名、严格限定的数值表示
  及轴/单位信息，拒绝`y/labels/truth/origin_group_ids`。准备阶段全角色元数据
  的label只用于配对审计和输出独立真值，不进入模型。支持池本试验完全不用。
- 每模态保留自己的原轴指纹、指纹定义、物理量、单位状态、原source SHA及
  有序canonical IDs。原始字节轴SHA与shape/dtype＋bytes的array_fingerprint
  不是同一定义，不能直接互换。参考列不进入样品features。

频率表示与时域使用各自的采集坐标和处理方法，不因分支编码而假称同轴：

| 表示 | 固定pointwise转换 | 训练内学习 |
| --- | --- | --- |
| α | 自有频率轴0.21–1.49THz无外推线性插值，再按行SNV（总体std） | 当前fit-source的独立StandardScaler |
| n | 自有频率轴相同数值窗口、独立核验/插值；**不做行SNV或行去均值** | 当前fit-source的独立StandardScaler；保留样品间n均值差异 |
| time | 6750点均匀原时间轴先固定polyphase抗混叠重采样到1024，再每行除L2范数，不去均值 | 当前fit-source的独立StandardScaler |
| α1024控制 | 从原canonical612直接无外推线性插值到0.21–1.49THz的1024点，再行SNV；不创造新信息 | 当前fit-source的独立StandardScaler，仅用于容量/长度控制 |

time原登记6750点、约0–89.9867，单位并未明确声明，不能默认ps、换成THz
或做未经证明的物理FFT。不能把窄脉冲直接线性抽到128点。time固定
`resample_poly(up512,down3375,window=('kaiser',5.0),padtype='line')`，要求
先通过6750点单调/均匀原轴检查（rtol1e−8、atol1e−12）。边界线性延拓
仅是预定FIR边界条件，目标时间坐标保持原单位和采集范围，不变成THz。
此滤波目前未实际运行，数值反混叠/脉冲保持核验须在Q后执行。time窗口
必须在Q后从注册的**source采集轴**绑定，不由query值或类别选择。
同任务各角色的同一模态轴契约须一致；未知
单位继续标为谱形/表示工程探索。常数α、零time、非有限值、缺模态、错序
或祖先冲突都阻止该任务，不按预测正误删样本或自动造值。

compact包的每模态axis_metadata必须包含`native_axis_fingerprint`、
`native_axis_fingerprint_definition`、`points`（α/n为128、time/α长支为1024）、独立`target_axis`、
`axis_kind/axis_unit/quantity/value_unit`、固定`pointwise_transform`及
`pointwise_prepared`，以及显式`resampling/derived_from`。α长支必须绑定与
α128同一原canonical轴/原件；time的抗混叠参数必须匹配冻结配方。这是
准备器的契约，不是本草案已经读完数值的证明。

## 固定拆分、选轮与预测屏障

| 任务 | source/support/query | source选轮 |
| --- | ---: | --- |
| 酸枣仁旧7类 | 513/171/171 | 继承既有source三折，不搜拆分seed |
| 黄芪旧4类 | 241/80/82 | 继承既有source三折 |
| 天麻旧4类 | 242/80/83 | 继承既有source三折 |
| 酸枣仁扩10类 | 516/174/174 | 新类各source仅1，**无singleton CV**，继承旧7类共同epoch |

只让alpha_single在三个非扩展任务各自source三折、全部seed42/43/44上
训练至96轮，保存24/48/96的source-validation预测。每seed拼接三个valid
折，计算pooled Macro-F1/BA/Accuracy，再平均三seed，以该顺序最大化并
以更早epoch打破完全平局。任何seed、候选或OOF ID缺失均不能选赢家。
这计划27个source fit，之后四任务×三arm×三seed＝36个最终source fit；
当前实际fit数仍为0。

同任务三arm和三seed共享source选出的epoch。扩10类最终监督训练包含
全部10类，不能丢弃新单例；epoch继承旧7类锁。所有source选择锁形成后
才做最终source拟合；训练函数不接收真实query。全部36个模型及scaler
冻结后，再由独立label-free预测函数读取query；所有预测锁定后独立进程
才连接truth评分。ID只用于分区/配对，绝不编码进数值特征。

源内Scaler和每个实际神经batch都通过`FitScope`记录真实fit IDs与数值
指纹；运行前还需独立审查准备器、折ID、冻结context及输入/源码hash。
推理需eval/no_grad、点式/反序/不同batch以及BN状态不变检查。它们是API
和来源级技术检查，不是OS真值权限沙箱或“绝对零泄漏”认证。

## 交付范围与Q后的执行条件

- [草案JSON](../data/research/supplemental_multirepresentation_draft_20261104.json)：
  绑定和结果字段明确NOT_RUN/null；2400实际CPU秒、7200墙钟、单线程、
  RSS1.5GiB仅为待root冻结的资源提案，无GPU/付费服务。
- [小源码](../scripts/draft_supplemental_multirepresentation_20261104.py)：
  导入/`--plan`仅stdlib；已实现角色/别名/折门禁、source-only共同选轮、
  固定行转换、延迟加载的source训练叶函数与独立冻结模型预测叶函数。
  数值路径必须收到root的Q终态复核、数据/源码冻结、scope审查与外部实际
  CPU监督release，当前草案不能执行。没有完整数值准备器、总driver、
  source/model/prediction release binder或独立真实评分产物；这些不能省略。
- [纯软件测试](../tests/test_supplemental_multirepresentation_draft.py)：
  使用明确软件fixture检查隔离和公式，不保存假样品或假指标。Torch
  forward/反向、参数精确匹配、真实输入准备/axis/模态配对均尚未运行。

当前24项纯stdlib测试全部通过；独立小源码复核发现并修复了计划门禁未核对
分支长度/α长支派生配方的问题，新增相应负测试后独立重跑也全部通过。
`--plan`实际返回`fits_executed=0`、`numerical_inputs_read=0`、
`runtime_imported=false`。[静态预检收据](../reports/supplemental-multirepresentation-draft-20261104-v1/preflight.json)
只证明本草案的明确软件边界，不代表真实数值准备、模型训练或SCI结果已完成。

Q后由root先建立新输出目录和数值准备/审计，再冻结新协议副本、源码、
轴、全部fit-context ID与总资源预算。源训练、预测、独立评分分阶段执行，
原始数据/原633-fit实验和96.0234%历史强候选保留。不能因为本次query成绩
不理想再改epoch/seed/arm，不能声称用本NNpilot重现了原五二折96.0234%。

评分必须报全seed的Accuracy、Macro-F1、Balanced Accuracy、逐类P/R/F1、
混淆矩阵、配对改对/伤害、参数量/模型大小/耗时/实际CPU/RSS。扩10类同时
单列总174、旧171和新增3 query（每新类1份），新增3只作描述性入组探查。
技术可读、模态齐全、用户独立样品声明与真实跨批次生物独立性分别陈述，
不能把303条补齐表示或复制三分支算作新增样品。
