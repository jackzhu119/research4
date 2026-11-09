# 外部数据、迁移、补充表示与稳健性：设计—结果—论文使用边界

整理日期：2026-11-09（北京时间）；历史证据固定至 `cd4b50c37c1e26064f9f15f340d30e20ac386458`。本次只读归档、核对与汇总，没有新增训练或改动正在运行的量子任务。

## 阅读结论

外部植物预训练在已执行的配对神经网络试验中没有显示稳定平均增益。扩大训练预算可以改善该神经配方，但外部预训练臂在同预算内仍低于从头及额外目标训练控制。补充多表示神经网络在黄芪任务中有内部对照收益，在旧ZY主任务中没有；所有正负结果均保留。

冻结频轴扰动暴露了历史模型的敏感性；预定频移增强改善了模拟扰动表现，同时降低干净谱表现。这是开发数据上的工程权衡，不是新批次或真实仪器漂移验证。

**人群与任务必须先分开。** 源报告沿用当时的“855份酸枣仁”或“10类酸枣仁”名称；当前有效标签已经确认旧ZY队列混合枣仁与酸枣仁。下文统一称“历史ZY来源组/混合种类队列”。所有855/864、固定171/174 query的旧成绩，不是当前18表/1672谱的新结果，更不是纯酸枣仁产地识别的有效确认。最新有效清单与标签请以总入口和当前数据模块为准。

**独立性应区分当时记录与后来确认。** 历史运行时尚未建立的独立性状态保持原样，不回填旧run元数据；后来2026-11-08用户另明确确认黄芪403、天麻405及新增枣仁9条共817列“每列都是独立实物样品”，与原855条声明合并，当前1672条均有用户独立性声明。较早AGENTS段落未包含这项后续证据，应以该独立声明核对当前注释。这仍不是独立实验室采样审计、拉丁物种鉴定或真实批次/跨仪器确认；也不扩展到外部人参、果实或叶片。[S23](#s23)

## 1. 完整实验索引

| ID | 实验族与实际状态 | 主要设计 | 结果/结论 | 论文使用 |
| --- | --- | --- | --- | --- |
| EXT-001 | UltraIR外部FTIR来源分类；COMPLETED | 50个正式分类场景；作者5外折+建模池3内折 | PLS-DA在两个外部任务中数值最高；筛选器没有在两个队列上稳定胜出，山银花PLS-DA与CARS只差1条谱。 [S1](#s1) [S2](#s2) | 辅助外部方法基准；不能与本项目THz准确率拼接。 |
| EXT-002 | UltraIR相对丰度回归及归档Octave核心；COMPLETED | 250回归场景+30归档Octave调用；全部10目标 | 若干目标有有限可预测性，多数弱或负R²；程序执行成功不等于实用定量成功。 [S1](#s1) [S3](#s3) | 方法学补充；保留全部正负R²，不可称药典绝对含量测定。 |
| EXT-003 | 公开THz数据获取及资格核验；COMPLETED_WITH_BLOCKED_SOURCES | 来源/字节/许可/轴/独立单位分别核验 | 可执行的外部任务与受阻数据已分开，不能按文件数/曲线数/像素数扩充独立N。 [S4](#s4) [S5](#s5) | 数据来源/排除流程、局限性和补充材料索引。 |
| EXT-004 | 公开人参/西洋参四轮算法验证；COMPLETED_EXPLORATORY | 6导出组留出、3种子、R0—R4训练内选择 | 预先source选择的R4未超过R0，复杂筛选无稳定收益；n+LDA是事后开发候选，不是确认赢家。 [S4](#s4) | 辅助方法适用边界/负结果；不能将61.90%作为事前确认结论。 |
| EXT-005 | 公开水果按果实分组的工程回归；COMPLETED_EXPLORATORY | 四任务各10果实/40记录、5外3内、果实互斥 | PR/PM有有限工程回归收益，MA-Brix全部家族负R²，MA-DM学习模型不如均值基线。 [S4](#s4) | 外部流程验证及负结果补充，不作药材含量或THz物理验证。 |
| EXT-006 | 跨植物ResNet预训练配对试验；COMPLETED_NEGATIVE_TRANSFER | 原5外2内、3种子、A/Aextra/B/C配对 | 加入外部预训练无平均收益；source-policy从未选择B/C，其较好表现不能归因于外部数据。 [S6](#s6) [S7](#s7) | 辅助负迁移结果，限制此配方的收益主张；不否定所有迁移方法。 |
| EXT-007 | 保守跨植物预训练48轮上限试验；COMPLETED_NEGATIVE_TRANSFER | 同3arm、48轮上限、L2-SP=.001 | 固定保守配方没有获益，外部B比同额外计算Aextra仍低。 [S8](#s8) | 辅助负迁移，不称全新算法或真实域校准。 |
| EXT-008 | 同配方48→96轮上限的预算消融；COMPLETED_NEGATIVE_TRANSFER_WITH_BUDGET_GAIN | 仅候选epoch48→96，旧source前缀一致 | 扩大训练/选择预算有益，但96上限B仍低于A及Aextra准确率；不能把预算收益记为外部数据收益。 [S9](#s9) | 训练预算消融及负迁移补充；按同预算对照解释。 |
| EXT-009 | 补充光谱入库与多表示恢复；COMPLETED_WITH_BLOCKED_FILES | 9新增谱+303既有样品模态恢复；2文件仍BLOCKED | 补表示提高可运行性，不扩大独立N；6399条记录×表示血缘行不是6399份独立样品。 [S10](#s10) [S11](#s11) [S12](#s12) | 材料方法的数据血缘与质量控制；当前正式计数由2026-11-08 active_registry接管。 |
| EXT-010 | 补充固定三分角色的传统分类对照；COMPLETED | 固定A/B/HQ/TM三角色，21候选、3种子 | MC在旧A比同协议SVM提高4.2885pp并压缩89.87%变量，但LDA略优；B整体高分主要来自旧171，新3个query每seed仅对1个。 [S11](#s11) | 可作为已曝光固定角色对照、压缩和负面边界的补充；不可替代同855 OOF主表。 |
| EXT-011 | 补充角色少样本校准与保守no-op；COMPLETED_WITH_NEGATIVE_AND_NOOP_RESULTS | Anchor vs保守no-op；10支持重抽×3seed | 传统校准明显损害部分任务，保守规则避免损害但没有带来提升。 [S11](#s11) | 校准失败及no-op作为讨论/补充，不能写成成功域适配。 |
| EXT-012 | 补充多表示ResNet：匹配三分支控制；COMPLETED_WITH_MIXED_RESULTS | α单臂、matched C、α+n+time M；27+36拟合 | 主ZY任务没有神经多模态普遍增益；HQ有匹配神经内部收益但低于同拆分LDA，TM低于同拆分MLP。 [S12](#s12) [S13](#s13) | 表示消融的辅助结果，须保留全指标及同拆分传统对照。 |
| EXT-013 | 补充NN历史无拟合独立重放；COMPLETED_NO_NEW_TRAINING | 36预测复算/36权重重放，新增fit=0 | 后续实际重放验证原结果未被过期训练期限或合法量子账本追加破坏；没有新的科学性能增益。 [S14](#s14) | 可复现性补充；不能计作第二个独立实验或新训练样品。 |
| EXT-014 | 保存OOF的11方法配对统计与稀有类影响；COMPLETED_NO_NEW_FITS | 11保存OOF方法、11比较、2000条件bootstrap | 保存预测的条件统计支持历史R2在该曝光集合的比较，不能当嵌套CV显著性检验或独立批次总体置信保证。 [S15](#s15) [S16](#s16) [S17](#s17) | 方法比较/不确定性与ZY02敏感性补充；须准确命名条件区间。 |
| EXT-015 | 冻结R2模型18条件谱形稳健性；COMPLETED_EXPLORATORY | 18条件含基线，模型冻结、无重训 | 乘法增益和常数基线下标签不变；频轴±.002THz准确率降至89.9415%/92.1637%，提示该工程扰动敏感性。 [S18](#s18) [S19](#s19) | 稳健性诊断，不能称真实仪器漂移/批次泛化。 |
| EXT-016 | 预定半bin/整bin源内频移增强；COMPLETED_WITH_TRADEOFF | 两预定频移配方、10新SVC、6评价 | 两种配方提高模拟频移下准确率，同时损失原始谱准确率及Macro-F1；不能宣称提高96.0234%的主任务成绩。 [S20](#s20) [S21](#s21) [S22](#s22) | 干净精度与扰动稳健性的权衡补充，全部两配方/三场景不择优。 |
| EXT-017 | 2026-11-04公开来源再核验与下载阻塞；COMPLETED_ACQUISITION_REVIEW_NO_NEW_DATA | 14来源资格、4真实下载失败、新数值队列0 | 已有来源可用于独立任务探索，公开候选受阻/许可不明的保持BLOCKED，不能写成新采集成功。 [S5](#s5) | 数据来源/许可与未完成项清单。 |

机器可读完整台账包含各实验的设计、全部关键数值、状态、限制和来源SHA-256：[external_robustness.json](../../evidence/paper_bundle_20261109/external_robustness.json)。下列各表只在同一协议内比较，不能跨表拼成“连续提升阶梯”。

## 2. 外部FTIR基准：筛选器并非普遍最优

UltraIR固定提交下的FTIR是独立外部方法基准。原train+valid构成建模池、内3折选择，五个作者test折保留；Python的raw/SNV、scaler和选择器均在当前训练侧拟合。分类/回归集合有重叠，不能将表行人数相加为独立样品总量。[S1](#s1) [S2](#s2)

| 药材/谱行数 | SVM准确率 | PLS-DA准确率 | CARS-style+SVM | MC-UVE+SVM |
| --- | ---: | ---: | ---: | ---: |
| 金银花120/4类 | 72.50% | 76.67% | 71.67% | 74.17% |
| 山银花150/5类 | 82.00% | 84.67% | 84.00% | 77.33% |

PLS-DA总体数值最高，但山银花与CARS仅相差1条谱；不能据此宣称稳定显著优势。源资料缺物理样品/重复/批次ID，四组均为1868点FTIR百分透射率，不能改称THz、黄芪或独立物理样品验证。

全部相对丰度目标的总体OOF R²如下，列序固定，不挑选目标。原标签是LC-MS相对丰度a.u.，不是mg/g药典含量。[S1](#s1) [S3](#s3)

| 目标 | 均值基线 | PLS | SVR | CARS-style+PLS | MC-UVE+PLS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 金银花/4-Methylcoumarin | -0.0382 | 0.2315 | -0.1200 | 0.2097 | 0.0169 |
| 金银花/Milrinone | -0.0207 | 0.3888 | 0.4246 | 0.4948 | 0.4740 |
| 金银花/Geniposide | -0.0316 | 0.0094 | 0.1436 | 0.1118 | 0.0201 |
| 金银花/Phenylacetaldehyde | -0.0301 | 0.2222 | 0.1334 | 0.0915 | 0.0579 |
| 金银花/cis-Cinnamic acid | -0.0301 | 0.2262 | 0.1988 | 0.0991 | 0.0562 |
| 金银花/Lumazine | -0.0338 | 0.2753 | 0.1401 | 0.1130 | 0.0361 |
| 山银花/Coniferyl aldehyde | -0.0536 | 0.5762 | 0.6498 | 0.4096 | 0.5442 |
| 山银花/quercetin | -0.0321 | 0.4942 | 0.5892 | 0.3652 | 0.2377 |
| 山银花/kaempferol | -0.0356 | 0.4727 | 0.5582 | 0.3182 | 0.3729 |
| 山银花/Luteolin | -0.0154 | -0.1572 | 0.0489 | -0.1888 | -0.1158 |

归档Octave核心的30次调用全部完成，但金银花首目标PLS/MC-UVE/CARS总体R²分别−0.0777/−0.1044/−0.0191；执行通过不等于有效定量。Python和Octave在缩放、调参、保留数和PLS潜变量上不同，不作等设置优劣推断。数据数组许可未单独确认，汇总不重新发布原数组。

## 3. 公开THz植物：已取得内容与实际模型结果

| 来源 | 实际内容 | 可用范围与未完成事项 |
| --- | --- | --- |
| 人参/西洋参2025 V2 | 已有42压片、65点、α/n；CC BY 4.0；本次新增远程样品0 | 2物种观测级分类；原植物/共同原料未知，6导出组不等于生物批次 |
| 人参2026 V1 | 元数据96CSV=4方案×6折×4文件类型 | 未取得数值原件，不能称96独立样品或复算原论文97.67% |
| Princeton水果 | 下载约138MB；30可追踪果实；选定120唯一记录 | 4个工程回归任务，每任务10果实/40相关时点记录；不猜时间/化学单位 |
| 叶片资料 | 68条TikZ曲线 | 不等于68独立叶片，缺任务标签/样品映射，不训练 |
| 稻谷、玉米DeepSpectra、小麦、白桦等 | 元数据/许可资格和合法下载失败记录 | 未取得可用原件或粒度未厘清的保持BLOCKED，不搬论文分数 |

水果与叶片原GitHub队列未发现许可文件，原件、衍生逐条数值与水果模型留本地忽略目录；只归档自编代码、固定来源、协议和聚合结果。[S4](#s4) [S5](#s5) [S5](#s5)

### 人参：预先选择策略没有稳定提高

| 方案 | Accuracy | Macro-F1 | 较R0准确率变化 |
| --- | ---: | ---: | ---: |
| R0 raw α+SVM | 54.76% | 0.5411 | 0 |
| R1 SNV+MC-UVE选择 | 45.24% | 0.4507 | −9.52pp |
| R2 继续选择SVM | 43.65% | 0.4352 | −11.11pp |
| R3 继续选预处理 | 43.65% | 0.4352 | −11.11pp |
| R4源内整体策略 | 51.59% | 0.5117 | −3.17pp |
| n+收缩LDA，事后观察家族最高 | 61.90% | 0.6156 | +7.14pp |

每个种子仍是同42观测，三种子不是126独立样品。n+LDA不是事前确认赢家；事后多数类背景为22/42=52.38%，其优势只多4个观测。MC变量Jaccard约0.204—0.436，不能当已确认成分峰。[S4](#s4)

### 水果：分组回归保留负结果

| 预定source-policy任务 | RMSE | R² | 对均值预测RMSE减少 |
| --- | ---: | ---: | ---: |
| MA-Brix | 0.5171 | −0.1312 | +3.21% |
| PM-Brix | 0.3370 | 0.1995 | +13.06% |
| PR-Brix | 0.5144 | 0.3474 | +23.40% |
| MA-DM | 0.2652 | −0.1945 | −5.19% |

PR-Brix事后MLP家族R²0.5140、PM-Brix事后PLSR R²0.2514不能回填预定source-policy。MA-Brix全部家族R²负，MA-DM全部学习模型劣于固定均值对照。任务按果实分组但日期同时存在于source/query，不是跨时间或跨批次验证。[S4](#s4)

## 4. 负迁移与训练预算消融

所有目标为历史ZY855混合种类队列，沿用五外折/二内折及42/43/44全种子。源内选择锁先于outer预测，外部植物任务头独立；共同频轴插值只是谱形工程，不创造分辨率或物理兼容证据。

| 120步预训练试验 | Accuracy均值 | Macro-F1 | 对A准确率变化 |
| --- | ---: | ---: | ---: |
| A从头 | 87.21% | 0.7616 | 0 |
| Aextra目标source额外训练 | 87.37% | 0.7671 | +0.16pp |
| B人参预训练 | 86.43% | 0.7539 | −0.78pp |
| C黄芪+天麻+人参预训练 | 85.65% | 0.7333 | −1.56pp |
| source-policy | 87.95% | 0.7706 | +0.74pp |

source-policy在所有外折只选A/Aextra，从未选B/C，不能把其较高分数记为外部数据收益。B−Aextra为−0.94pp，C−Aextra为−1.72pp。[S6](#s6) [S7](#s7)

| 同一保守L2-SP配方 | 48轮上限Accuracy | 96轮上限Accuracy | 预算扩大的变化 |
| --- | ---: | ---: | ---: |
| A从头 | 80.3509% | 87.0565% | +6.7057pp |
| Aextra目标source预训练 | 78.3236% | 86.3548% | +8.0312pp |
| B黄芪/天麻预训练 | 76.1404% | 85.3801% | +9.2398pp |

第二轮只扩大候选epoch上限，初始化、外部池、120预训练步数及L2-SP等不变；source旧前48轮logit/loss完全重放一致，选轮91/86/92/89/92。预算增加改善各臂，但B仍低于同预算A/Aextra。B对Aextra的BA仅+0.0204pp，不能写成三指标都下降，也不能因此宣称整体迁移成功。[S8](#s8) [S9](#s9)

## 5. 补充光谱及多表示：同拆分精确比较

A旧7组为513训练/171支持池/171 query；B扩展10组为516/174/174，新增3来源各仅3条谱；黄芪241/80/82，天麻242/80/83。A与B共享旧855，不能相加。补回黄芪100条n及天麻203条time表示是恢复既有记录的模态，不是303份新独立样品。[S10](#s10) [S11](#s11) [S12](#s12) 后来的新增817条用户独立性声明改变当前注释依据，但不把这些已曝光历史预测变成新的独立确认，也不回写当时的run限制。[S23](#s23)

### 传统管线：压缩有用，MC并非全部第一

| 任务/query N每seed | MC-UVE+SVM Accuracy/Macro-F1 | 普通SVM Accuracy | LDA Accuracy | MLP Accuracy |
| --- | ---: | ---: | ---: | ---: |
| A旧7组/171 | 93.1774% / 0.915255 | 88.8889% | 93.5673% | 90.4483% |
| B扩展10组/174 | 94.8276% / 0.730505 | 87.9310% | 92.5287% | 88.8889% |
| 黄芪/82 | 81.7073% / 0.815596 | 82.9268% | 84.1463% | 82.9268% |
| 天麻/83 | 97.5904% / 0.976168 | 98.7952% | 98.7952% | 99.1968% |

A的MC保留62/612变量，减少89.87%，较同协议SVM准确率+4.2885pp，但LDA略优。加入新来源后，MC在同一旧171子集93.1774%→95.9064%，同时新增3query每seed仅1/3正确。新增训练量、类别、权重和选择器同时变化，不能把子集改善因果归于“数据新增”，也不能用95.9064%代表整个10类。[S11](#s11)

### 少样本校准：避免伤害不等于提高

| 任务/shot每类 | 未校准 | 传统Anchor | 保守机制 |
| --- | ---: | ---: | ---: |
| A/1 | 93.1774% | 59.6686% | 93.1774% |
| B/1 | 94.8276% | 55.0192% | 94.8276% |
| 黄芪/1 | 81.7073% | 66.2602% | 81.7073% |
| 黄芪/3 | 81.7073% | 71.7073% | 81.7073% |
| 黄芪/5 | 81.7073% | 78.0488% | 81.7073% |
| 天麻/1、3、5 | 97.5904% | 97.5904% | 97.5904% |

保守规则在源内门禁后全部拒绝启用，240份保守输出与原生预测/分数逐元素相同：保留no-op而非包装为域适配成功。旧A的ZY02和新增来源每类仅1支持记录，统一3/5-shot不可执行。[S11](#s11)

### 多表示神经网络：必须用匹配三分支C作为直接基线

| 任务 | α单分支Accuracy | C三分支α | M=α+n+time | M−C Accuracy | M−C Macro-F1 | M−C BA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 旧ZY7组 | 76.6082% | 82.6511% | 82.4561% | −0.1949pp | −0.031536 | −4.4218pp |
| 黄芪 | 69.1057% | 69.9187% | 78.4553% | +8.5366pp | +0.088395 | +8.6364pp |
| 天麻 | 94.3775% | 93.5743% | 94.3775% | +0.8032pp | +0.007705 | +0.7558pp |
| 扩展10组 | 69.3487% | 79.8851% | 81.8008% | +1.9157pp | +0.000386 | −1.6190pp |

C/M的分支长度、参数量、初始化、batch及epoch配对，α/n/time物理轴各自编码。C−α单支还混入容量和α分辨率改变。主ZY任务M无增益；黄芪M仍低于同拆分LDA84.1463%，天麻M低于同拆分MLP99.1968%。v1在0fit因time轴不均匀失败，v2明确端点内重采样后完成27源+36最终拟合，旧失败与轴SHA保留。[S12](#s12) [S13](#s13)

原36权重已完成独立重放；2026-11-05另外实际执行的历史无拟合复核再次审36预测/36权重，最大指标差1.4211e-14，新增fit=0。后者是复现证据，不是第二个独立性能实验。[S14](#s14)

## 6. 固定OOF统计、18条件扰动和频移增强

### 条件bootstrap不替代独立确认

11个保存OOF方法、11个预设配对、2000次分层bootstrap（seed20261107）；本任务新fit和新inference均为0。R2仍821/855，Accuracy96.0234%、Macro-F1 0.9196846、BA89.9184%。R2相对全谱SVM的Accuracy差7.9532pp，条件百分位范围约5.848—10.175pp；这只是固定预测/现有类数条件下的范围，不覆盖训练/选参不确定性，不能当独立批次总体CI或经验证的嵌套CV显著性检验。[S15](#s15) [S16](#s16) [S17](#s17)

完整11方法×3指标及全部配对范围已保存在JSON。strong MLP的预定主seed42与200epochs+100source updates都不应重命名为最优种子或300epochs；ZY02固定预测删一例诊断不等于重训或生物重复。

### 冻结R2：频轴模拟最敏感

| 条件 | Accuracy | Macro-F1 | BA | 正确/855 |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 96.0234% | 0.919685 | 89.9184% | 821/855 |
| `noise_0.001_seed42` | 96.1404% | 0.921079 | 90.0612% | 822/855 |
| `noise_0.001_seed43` | 96.0234% | 0.908968 | 90.0204% | 821/855 |
| `noise_0.001_seed44` | 96.1404% | 0.921079 | 90.0612% | 822/855 |
| `noise_0.005_seed42` | 95.4386% | 0.931122 | 91.9184% | 816/855 |
| `noise_0.005_seed43` | 95.4386% | 0.901786 | 89.3061% | 816/855 |
| `noise_0.005_seed44` | 95.5556% | 0.889668 | 86.6327% | 817/855 |
| `noise_0.01_seed42` | 95.3216% | 0.911073 | 89.0612% | 815/855 |
| `noise_0.01_seed43` | 95.0877% | 0.906130 | 91.6939% | 813/855 |
| `noise_0.01_seed44` | 95.0877% | 0.874966 | 86.1633% | 813/855 |
| `gain_0.9` | 96.0234% | 0.919685 | 89.9184% | 821/855 |
| `gain_1.1` | 96.0234% | 0.919685 | 89.9184% | 821/855 |
| `constant_baseline_-0.05` | 96.0234% | 0.919685 | 89.9184% | 821/855 |
| `constant_baseline_0.05` | 96.0234% | 0.919685 | 89.9184% | 821/855 |
| `linear_baseline_-0.05` | 96.0234% | 0.908968 | 90.0204% | 821/855 |
| `linear_baseline_0.05` | 95.7895% | 0.916954 | 89.6327% | 819/855 |
| `axis_shift_-0.002_THz` | 89.9415% | 0.789039 | 80.4898% | 769/855 |
| `axis_shift_0.002_THz` | 92.1637% | 0.832354 | 82.6939% | 788/855 |

18条件包含基线；15390条预测是同855记录的重复角色。增益/常数偏移对标签不变并非所有干扰都无效；频轴−/+0.002THz分别降至769/855、788/855。所有扰动是工程模拟，没有新的实测仪器漂移、批次或化学浓度证据。[S18](#s18) [S19](#s19)

### 预定频移增强：稳健性改善同时损失干净精度

| 评价条件 | 冻结R2 | 半bin增强 | 整bin增强 |
| --- | ---: | ---: | ---: |
| 干净谱Accuracy | 96.0234% | 95.2047% | 95.3216% |
| 干净谱Macro-F1 | 0.919685 | 0.899056 | 0.900489 |
| −0.002THz Accuracy | 89.9415% | 94.2690% | 94.0351% |
| +0.002THz Accuracy | 92.1637% | 94.1520% | 94.7368% |

半bin/整bin位移分别为0.0011444092/0.0022888184THz；每折原始训练684记录，2052行是依赖变体。只重新拟合10个SVC，原变换、MC选择、C、数值gamma及类权重冻结；6个场景全部保留，不用已评分query再挑赢家。该设计受已观察频轴敏感性启发，是曝光开发研究。原文件的source_labels_sha256字段实际为joblib指纹，真实SHA另有勘误/复核，不应误称新独立确认。[S20](#s20) [S21](#s21) [S22](#s22)

## 7. 论文可写与不可写

| 可支持的写法 | 不应使用的写法 |
| --- | --- |
| 在所执行的外部FTIR任务/固定预算下，筛选器没有普遍提高 | 筛选算法对所有药材或所有谱技术普遍最优 |
| 外部预训练在这些已曝光配对配方中无平均收益，增加训练预算改善神经表现 | 所有跨植物迁移无效；或把预算收益写成新增外部数据收益 |
| 同拆分旧ZY任务MC压缩特征并超过普通SVM，但LDA略优 | MC在所有协议/指标均第一；将171query与855OOF相减作统一提升 |
| 黄芪多表示M相对匹配神经C有收益 | 多模态普遍提高主ZY任务；已超过所有传统模型 |
| 保守校准本轮保持原预测，避免了传统Anchor损害 | no-op代表成功少样本域适配或真实新批次收益 |
| 固定预测条件bootstrap、模拟频移敏感性与增强权衡 | 全训练过程总体CI、真实仪器漂移、跨批次确认或化学机制证明 |
| 每条当前有效谱记录、物理独立性与来源声明分别说明 | 用重复seed、增强行、多表示、文件数或像素数扩大独立样本数 |

本模块不追加任何模型搜索或训练。用户当前两项量子任务的运行结果由独立量子模块按时间截点整理，不与这些冻结历史分类器混成提升证据。

## 8. 固定来源与可追溯性

所有链接固定到源提交，避免分支后续自动回传改写“当时”的证据。每份完整SHA-256及实验—来源映射均在机器可读台账；以下短SHA仅便于人工定位。

<a id="s1"></a>**S1.** [docs/EXTERNAL_BENCHMARK_REPORT.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/EXTERNAL_BENCHMARK_REPORT.md)；SHA-256 `3213eff7231ef85dc299cbd58dc7f2534f38450323ece6dd8c315ffb21f9439f`。

<a id="s2"></a>**S2.** [reports/external-benchmarks/2026-10-30/benchmark_summary.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/external-benchmarks/2026-10-30/benchmark_summary.json)；SHA-256 `27775f165b2df3f1e32b5b8c61025ff8f7f67dc1f64bc7ddf901e34de86595f1`。

<a id="s3"></a>**S3.** [reports/external-benchmarks/2026-10-30/pooled_metrics.csv](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/external-benchmarks/2026-10-30/pooled_metrics.csv)；SHA-256 `ee9ebcc83b347bacb8fb07a6968c06f280adb7d3ecfb21ff881382d1465fb424`。

<a id="s4"></a>**S4.** [docs/PUBLIC_PLANT_THZ_VALIDATION_20261031.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/PUBLIC_PLANT_THZ_VALIDATION_20261031.md)；SHA-256 `924cb45e36c2438abb3d774d72d632c5241c295063e6bc18696cd68b4dfc1030`。

<a id="s5"></a>**S5.** [docs/RESEARCH4_EXTERNAL_EVIDENCE_20261104.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/RESEARCH4_EXTERNAL_EVIDENCE_20261104.md)；SHA-256 `91a00639521697987383722a5828a5d06ed8611a2ed98103005c4bd46bc85a8e`。

<a id="s6"></a>**S6.** [docs/CROSS_PLANT_DEEP_TRANSFER_REPORT_20261031.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/CROSS_PLANT_DEEP_TRANSFER_REPORT_20261031.md)；SHA-256 `966b753229b43ad4465089d89da4e550c7636f6f9573382cd5a6427646487685`。

<a id="s7"></a>**S7.** [docs/CROSS_PLANT_DEEP_TRANSFER_PROTOCOL_20261031.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/CROSS_PLANT_DEEP_TRANSFER_PROTOCOL_20261031.md)；SHA-256 `c1e72157bf8b306c9013b03315e6e69bd5e43a4b7c8fe26fb1236f396d752d71`。

<a id="s8"></a>**S8.** [docs/CONSERVATIVE_CROSS_PLANT_PILOT_20261104.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/CONSERVATIVE_CROSS_PLANT_PILOT_20261104.md)；SHA-256 `9a0ef39e3555af3c2e182e0d7220d309312bbcb917897eebbaa28a19dc47141b`。

<a id="s9"></a>**S9.** [docs/CONSERVATIVE_CROSS_PLANT_BUDGET_ABLATION_20261104.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/CONSERVATIVE_CROSS_PLANT_BUDGET_ABLATION_20261104.md)；SHA-256 `9ed4049a85a7d377224e5e625df20f86f223150954d457cb11a3668024e17624`。

<a id="s10"></a>**S10.** [docs/MULTIPLANT_DATA_AUDIT.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/MULTIPLANT_DATA_AUDIT.md)；SHA-256 `de0ebed10d49bdd2d18ed8330162881159c16ac4e783576d97bf84494c19ee99`。

<a id="s11"></a>**S11.** [docs/SUPPLEMENTAL_SPECTRA_RESULTS_20261104.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/SUPPLEMENTAL_SPECTRA_RESULTS_20261104.md)；SHA-256 `422237072b5232124aef234e5f76f9fe2c8360f1375c8fd72ce6fc88718d6409`。

<a id="s12"></a>**S12.** [docs/SUPPLEMENTAL_MULTIREPRESENTATION_RESULTS_20261104.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/SUPPLEMENTAL_MULTIREPRESENTATION_RESULTS_20261104.md)；SHA-256 `b5f7cddfc0ec050dcacfb4d096a05d1651456e3449a0395ddb495f147adafca2`。

<a id="s13"></a>**S13.** [docs/SUPPLEMENTAL_MULTIREPRESENTATION_V2_TIME_REPAIR_20261104.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/SUPPLEMENTAL_MULTIREPRESENTATION_V2_TIME_REPAIR_20261104.md)；SHA-256 `11577a7e95881b492628de5da97e28880ab1b35bfb54ae482bf3f829f7d0dff4`。

<a id="s14"></a>**S14.** [docs/SUPPLEMENTAL_MULTIREPRESENTATION_HISTORICAL_AUDIT_20261104_V3.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/SUPPLEMENTAL_MULTIREPRESENTATION_HISTORICAL_AUDIT_20261104_V3.md)；SHA-256 `47d69f70f27267201d123d6259b8d24dfabedfabb78f6d5fbf054d2559e2adfb`。

<a id="s15"></a>**S15.** [reports/research72h-20261107-v1/initial-ml-evidence-001/summary.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/research72h-20261107-v1/initial-ml-evidence-001/summary.json)；SHA-256 `0d025486a3af090920763192ac1dfa2de5f66d48c0b789def97e0e7ec87ef130`。

<a id="s16"></a>**S16.** [reports/research72h-20261107-v1/initial-ml-evidence-001/method_metrics.csv](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/research72h-20261107-v1/initial-ml-evidence-001/method_metrics.csv)；SHA-256 `041a0dfa3af133fda798d2edf74ea382142e8982532b815086c6250e3d3f2479`。

<a id="s17"></a>**S17.** [reports/research72h-20261107-v1/initial-ml-evidence-001/paired_conditional_ranges.csv](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/research72h-20261107-v1/initial-ml-evidence-001/paired_conditional_ranges.csv)；SHA-256 `f4ff05eec65ff6b6919e9a608a05c282bd9f134b083de4e5aed2105346f11841`。

<a id="s18"></a>**S18.** [reports/research72h-20261107-v1/frozen-robustness/results-v1/summary.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/research72h-20261107-v1/frozen-robustness/results-v1/summary.json)；SHA-256 `b4d8425f5a65d05cbc8b3130cbfb879bf7ebad6206f6ccc14c5aeb833d8682e9`。

<a id="s19"></a>**S19.** [reports/research72h-20261107-v1/frozen-robustness/results-v1/metric_summary.csv](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/research72h-20261107-v1/frozen-robustness/results-v1/metric_summary.csv)；SHA-256 `dda7ba72cf7fac97baa5affc5e06f23d274f8f3221220f574b9bb652163bd365`。

<a id="s20"></a>**S20.** [reports/research72h-20261107-v1/autonomous-main-001/jobs/source-shift-augmentation/attempt-001/summary.json](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/research72h-20261107-v1/autonomous-main-001/jobs/source-shift-augmentation/attempt-001/summary.json)；SHA-256 `431f47cc75c5c646e5470c523bef06b741867c70c38a7e27b1c05cb19ce1564b`。

<a id="s21"></a>**S21.** [reports/research72h-20261107-v1/autonomous-main-001/jobs/source-shift-augmentation/attempt-001/metrics.csv](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/research72h-20261107-v1/autonomous-main-001/jobs/source-shift-augmentation/attempt-001/metrics.csv)；SHA-256 `12417fb64e518003e9c5d50ee728201f37f04c8dd6470ce7189a568afaa8dc2c`。

<a id="s22"></a>**S22.** [docs/RESEARCH4_72H_BATCH_REVIEW_20261108.md](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/RESEARCH4_72H_BATCH_REVIEW_20261108.md)；SHA-256 `652564f0ab567d6fbb515670c51803846fc06463b74a7fa52760502dc9a4a51d`。

<a id="s23"></a>**S23.** [data/annotations/specimen_independence_user_declaration_20261108.json](https://github.com/jackzhu119/research4/blob/0156e2189d13456d1faf05ebfd53d86eeeb4f5e7/data/annotations/specimen_independence_user_declaration_20261108.json)；SHA-256 `752c337d461b97304d9c27057afd473ff8ab7fe15fb5816e5130bed5f7c6cad6`。
