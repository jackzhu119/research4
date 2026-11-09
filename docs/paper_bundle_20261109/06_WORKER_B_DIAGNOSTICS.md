# Worker B：来源覆盖、支持适配与辅助预训练诊断

本章整理已返回的两轮 Worker B 机制实验，供论文方法、结果和局限性写作使用。初轮有 **219 项科学任务完成**，另有一个因读取大小限制而保留失败标记的汇总任务；后续 deep-v3 有 **970 项科学任务及 1 项汇总全部封存，16 个汇总块完整**。这两个数字是计算任务数，不是独立样品数。

固定来源为 `research/parallel-worker-b-20261106-B20261106-01` 的 `c8945acc0a57d75aae438fd9de81d05b422cc1f3`。本章截点为 `2026-11-09T01:20:29Z`；机器可读的全部端点聚合、配对统计、随机化描述量与SHA在[worker_b_ml.json](../../evidence/paper_bundle_20261109/worker_b_ml.json)，实验编号 **BML-001—019**。本次只读已有结果和Git对象，未重训、提取新原始光谱或修改计算进程。

## 证据边界与研究问题

这两轮针对历史 master TRAIN 范围内的枣仁/酸枣仁二分类，解释ZY01错误集中、来源供体覆盖、少量目标支持适配以及辅助预训练。deep-v3限定历史999条TRAIN；CAL334/PRED339数值未进入。结果属于已暴露开发数据上的诊断，不能替代未见来源/真实新批次的确认性性能。原标签均保留，模型与标签一致不独立证明植物身份。当前18工作簿/1672条总体口径与当前多任务实验见[03_CURRENT_DATA_AND_SPECIES.md](03_CURRENT_DATA_AND_SPECIES.md)；早期20fit结果由该章记录，本文不重复计数。

[初轮协议](https://github.com/jackzhu119/research4/blob/c8945acc0a57d75aae438fd9de81d05b422cc1f3/reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/preparation/scientific_protocol_ROOT.json)、[实现边界](https://github.com/jackzhu119/research4/blob/c8945acc0a57d75aae438fd9de81d05b422cc1f3/reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/preparation/implementation_boundaries_ROOT.json)、[deep-v3预先科学设计](https://github.com/jackzhu119/research4/blob/c8945acc0a57d75aae438fd9de81d05b422cc1f3/reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/preparation/independent_scientific_design_recommendations.json)与[实际授权manifest](https://github.com/jackzhu119/research4/blob/c8945acc0a57d75aae438fd9de81d05b422cc1f3/reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/manifest_ROOT_AUTHORIZED.json)定义范围与对照；最终结果以[初轮独立科学分析](https://github.com/jackzhu119/research4/blob/c8945acc0a57d75aae438fd9de81d05b422cc1f3/reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/analysis/independent_final_science_analysis_001.json)及[deep-v3完整索引](https://github.com/jackzhu119/research4/blob/c8945acc0a57d75aae438fd9de81d05b422cc1f3/reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/science/summary-attempt001/summary_index.json)为入口。预先设计记录不是另一个完成结果，实际任务规模以最终封存审计为准。

## 初轮219项：错误集中不能直接变成物种改标签

**BML-001。** 旧23错是ZY01留出210条中的23条，不是整本350条中的23条。ZY01占全部枣仁留出216条的210条（97.22%），source仅6条其它来源枣仁对240条酸枣仁。α/SNV/SVC旧错跨5个ZY01-held组合持续，但α/SNV/LR每组208/210，恢复21个旧错；18种模型/表示×5组合中，没有一个ZY01 ID在全部90次都错。因此“单模型稳定错”不足以鉴定样品为酸枣仁，也不足以自动删除旧错。

向source加入同来源ZY01的其它TRAIN样品后，B0/B1在相同ordinal查询510个ID上配对：

| 模态 | B0正确 | B1正确 | 由错转对 | 由对转错 |
| --- | --- | --- | --- | --- |
| alpha | 486 | 507 | 21 | 0 |
| n | 385 | 503 | 120 | 2 |
| fusion | 427 | 507 | 80 | 0 |

α和fusion仍错ZY01-007/009/013；n另有其它未恢复项。该干预同时改变数量、source统计和标签覆盖，支持来源覆盖影响当前算法的解释，尚不能唯一归因于厚度、仪器、真实批次或植物身份。随机查询seed42/43/44的510行仅439/443/445个唯一ID，不与ordinal510作逐行配对。

**BML-002。** 初轮支持适配每模态2025条重复评价，仅436个唯一ID（176枣仁、260酸枣仁），15个来源×seed cell；K固定为0/1/3/5/10/20。下表为正确数，完整各K配对正负cell见JSON。

| 模态 | legacy各K正确数 | 弱anchor各K | 全流程重拟合各K正确数 |
| --- | --- | --- | --- |
| alpha | 1826,1927,1946,1963,1946,1939 | 1826（均相同） | 1826,1856,1981,1993,1995,1995 |
| n | 1094,1394,1398,1512,1453,1431 | 1094（均相同） | 1094,1462,1709,1783,1867,1905 |
| fusion | 1401,1819,1919,1965,1968,1950 | 1401（均相同） | 1401,1703,1945,1961,1985,1995 |

legacy的K曲线并非单调。弱适配预测不变不代表分数完全不变；全流程重拟合改变学习方案，应单列。α全重拟合K1仍有一个cell从122降到121/135。旧23个ID仅18个进入过任一支持查询，不能写“23错全部修复”。

**BML-003。** 三种子初轮的A为从头训练、Aextra为目标任务额外预训练、B为辅助任务预训练、Bshuffle为打乱辅助标签。每设置1548行仅516个唯一ID；A跨步数复用。全部预设条件如下，最后一列为seed42/43/44的B−Aextra正确数差：

| 步数/适配 | A | Aextra | B | Bshuffle | B−Aextra逐seed |
| --- | --- | --- | --- | --- | --- |
| 120/full | 1521/1548 | 1531/1548 | 1527/1548 | 1525/1548 | +0,-3,-1 |
| 120/frozen | 1432/1548 | 1509/1548 | 1494/1548 | 1489/1548 | -24,+2,+7 |
| 480/full | 1521/1548 | 1532/1548 | 1527/1548 | 1532/1548 | +1,-1,-5 |
| 480/frozen | 1432/1548 | 1475/1548 | 1502/1548 | 1476/1548 | +23,-6,+10 |

初轮full条件下B低于Aextra均值；480-frozen平均有利但含负seed，不能据此选为最终赢家。Aextra匹配optimizer step，不完全匹配样本曝光/FLOPs。210个新target端点点式核查通过，其中105个冻结编码器的完整状态不变；最大绝对logit残差2.670288e−5仍满足声明的绝对+相对联合容差（atol=rtol=1e−5），不能将此解释为任意输入数学恒等。

**BML-004。** [汇总外部资格记录](https://github.com/jackzhu119/research4/blob/c8945acc0a57d75aae438fd9de81d05b422cc1f3/reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/analysis/summary_external_scientific_qualification_001.json)说明：汇总真实exit0且13块齐全，但结果4,740,138字节超过冻结监督器4MiB门槛，因此保留原`FAILED_RESULT_VALIDATION_CLOSED`。独立≤8MiB读取器重放13块逐结构一致，验证144行指标CSV和158行配对CSV。正确表述是“219项科学封存，汇总经外部核查补足资格”，不是“220项全部DONE_SEALED”。

## deep-v3：全部16块返回及完整性

**BML-005、013。** [最终独立完成审核](https://github.com/jackzhu119/research4/blob/c8945acc0a57d75aae438fd9de81d05b422cc1f3/reports/parallel-cloud-20261106/worker-b/B20261106-01/yanan-cause-followup-v2/deep-mechanisms-v3/analysis/independent_completion_science_audit_20261109_001.json)确认全部971项退出为0、DAG/封存/来源证明绑定及工件哈希，科学wait4 CPU为12,566.568807秒；审核不重新训练。任务分解为source分数15、支持机制60、供体机制75、aux预训练20、target预训练50、source-A交叉240、target交叉评价480、BN probe30、汇总1。其和为971。aux/target预训练与source-A三块只计工件/溯源覆盖，准确率统一在端点块中报告。

汇总有13个含指标的block index和3个工件块；本次重新校验13个index的SHA与summary_index相同。旧审核的“publication separate”描述审核时的发布状态；本章所引摘要现可由固定Git commit读取。模型、预测CSV与封存凭据的具体导航在各index的`details`、`score_replay_exports`及manifest内，本文不据“摘要已上GitHub”推断所有大模型二进制均可公开下载。

## deep-v3支持机制：主要是阈值/位置调整

**BML-006—009。** source分数基线每模态1350重复行、510唯一ID，α1232/1350（91.259%）、n730/1350（54.074%）、fusion935/1350（69.259%）。支持机制另外固定旧draw42 query，5个held-acid×20个新支持顺序draw45..64，每cell135行，合计13500行但只有255个唯一ID（105个ZY01加5×30酸枣仁）。支持K嵌套，source模型不重新训练；70种设置/模态全部在JSON保留。

legacy λ1的各K正确数（统一分母13500）：

| 模态 | K0 | K1 | K3 | K5 | K10 | K20 |
| --- | --- | --- | --- | --- | --- | --- |
| alpha | 12140 | 12600 | 12738 | 12717 | 12754 | 12751 |
| n | 7360 | 9138 | 9528 | 9905 | 10014 | 10003 |
| fusion | 9220 | 11462 | 12284 | 12652 | 12837 | 12920 |

对照正确数（同一分母13500）：

| 模态 | bias-only K5 | 排列标签 K5 | 翻转标签 K5 | 弱适配 K20 | 全重拟合 K20 |
| --- | --- | --- | --- | --- | --- |
| alpha | 12717 | 7356 | 783 | 12140 | 13300 |
| n | 9905 | 7864 | 3595 | 7360 | 12758 |
| fusion | 12652 | 9150 | 848 | 9220 | 13300 |

二分类SVC分数为`[-s,s]`，仿射ridge后的margin为`a*s+b`；当`a>0`，只移动阈值`−b/a`而不改变排序/AUROC。三模态K5的100个cell斜率全部为正；legacy与bias-only、正分数重标度0.1/10在配对记录中均无由错转对/由对转错差异，source标准化anchor的聚合也相同。该条件的增益适合称为标签支持下的阈值/位置修正，不称新特征学习或概率校准。斜率符号与阈值分布完整保留，错误标签翻转并不是经验随机零假设p值。排列控制可能在K1碰巧不改变标签，不能假定所有排列均错。

λ固定时每样品有效惩罚随K改变，λK=K/5另作控制；α/n在K10→20略降，禁止依据query挑选最佳K。全重拟合K20更高属于不同方案，不能归入同一种分数校准收益。

**SNV数值控制。** 15个source缓存×4个正缩放/常数偏移变换，共60项均未改变预测，最大特征差8.105e−14，原数组不变。这符合`SNV(a*x+b)=SNV(x)`（a>0），约束纯幅度/偏移解释；人工变换不能替代实测厚度、频率相关形变或物理材料鉴定。

## deep-v3供体因素：数量、统计、核尺度与惩罚须区分

**BML-010—012。** 75个job为5held-acid×3模态×5draw45..49。固定6个枣仁供体，酸枣仁n6/24/60；交叉subset/full246 source标准化、subset/fixed-full gamma、C10/`Cn=10*246/(6+n)`三因素。n240原全source基线只复用一次。各模态30端点完整保留，下面列固定full标准化/gamma并保持总惩罚，以及一项预先定义的subset参考；不是事后选择赢家。

| 模态 | n240基线 | n6/full/总惩罚固定 | n24/full/总惩罚固定 | n60/full/总惩罚固定 | n6/subset/subsetγ/C10 |
| --- | --- | --- | --- | --- | --- |
| alpha | 6160 | 6600 | 6394 | 6292 | 6605 |
| n | 3650 | 4366 | 4014 | 3881 | 4403 |
| fusion | 4675 | 6522 | 5881 | 5076 | 6183 |

以上分母均6750重复行、510唯一ID。较少酸枣仁供体的部分设置提高与声明标签一致性，但仍不证明纯类别不平衡因果。全source统计控制使用246个source向量，即使分类器只用小标签子集，须在部署数据需求中如实说明。

n24另比较单酸枣仁来源24条与四来源各6条，以固定source标准化、gamma和总惩罚探查多样性。单来源端点5400行/450唯一ID，常规四来源端点6750行/510唯一ID；两者覆盖不同held-source，禁止直接相减总体准确率，应读取同一index的匹配`paired_aggregates`。所有单来源、因素组合及配对损益在JSON保留。

## deep-v3交叉种子：平均正差与种子不稳定同时成立

**BML-014—017。** 8个encoder/pretrain seed42..49×独立target seed101/202/303×5固定fold，目标训练96epoch；A/Aextra/B/Bshuffle匹配初始encoder和目标head、dropout及顺序。每条件24个随机化cell，每cell516个历史ID，合计12384重复评价；A跨120/480复用。这与初轮3seed耦合设计不同，应并列报告。

| 设置 | A | Aextra | B | Bshuffle | B−Aextra pp | 24cell正/零/负 | cell差范围 pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 120_full | 12234/12384 (98.789%) | 12236/12384 (98.805%) | 12245/12384 (98.878%) | 12238/12384 (98.821%) | +0.0727 | 14/1/9 | [-1.550,+0.969] |
| 120_frozen | 11035/12384 (89.107%) | 11936/12384 (96.382%) | 12042/12384 (97.238%) | 11119/12384 (89.785%) | +0.8559 | 11/0/13 | [-5.426,+7.171] |
| 480_full | 12234/12384 (98.789%) | 12231/12384 (98.765%) | 12265/12384 (99.039%) | 12232/12384 (98.773%) | +0.2745 | 11/7/6 | [-0.969,+1.744] |
| 480_frozen | 11035/12384 (89.107%) | 11762/12384 (94.977%) | 11892/12384 (96.027%) | 11409/12384 (92.127%) | +1.0497 | 15/0/9 | [-8.527,+6.977] |

四设置pooled B−Aextra均为正，但120-frozen有13个负cell、11个正cell；其平均正差并不意味着多数seed组合获益。480-frozen虽然平均高1.0497个百分点，仍有9个负cell，单cell差可至−8.527个百分点。480-full平均差0.2745个百分点，11正/7零/6负；均不能称跨种子稳定优越。与Bshuffle的配对、来源类别召回和encoder/target/交互描述方差在JSON全部保留，不能以重复行当独立样本计算显著性。

初轮full均值负差与新版交叉设计pooled正差均保留；前者不能概括后者，后者也不能抹去前者。不同预算、适配策略与随机化设计分别有条件解释，不在四设置中事后选一个“最终最佳模型”。

## deep-v3 BN干预：条件性贡献，尚无唯一物理归因

**BML-018—019。** 仅480-frozen、target seed101、encoder seed42/43/44×5fold；各臂使用同一策略。neutral将BN running mean/var设0/1但保留学习权重；source-recal只用target-donor整批更新BN缓冲区后固定编码器。每设置1548重复行、516唯一ID。

| BN策略 | A | Aextra | B | Bshuffle | B−Aextra pp | B−Bshuffle pp |
| --- | --- | --- | --- | --- | --- | --- |
| bn_neutral | 1425/1548 (92.054%) | 1442/1548 (93.152%) | 1473/1548 (95.155%) | 1472/1548 (95.090%) | +2.0026 | +0.0646 |
| bn_source_recal | 1467/1548 (94.767%) | 1484/1548 (95.866%) | 1497/1548 (96.705%) | 1473/1548 (95.155%) | +0.8398 | +1.5504 |

neutral使B相对native少24/1548，且B与Bshuffle只差1/1548；source-recal的B相对native净差0（3得、3失），B与Bshuffle差24/1548。BN统计会改变比较；这些小范围干预仍不能证明辅助类别语义普遍有效、已校正真实物理批次，或识别植物身份。显式BN probe与原native frozen条件分开，不能把缓冲区变化藏在“冻结不变”表述中。

## 论文可用结论及仍缺的证据

- 来源覆盖、模型表示与二分类阈值调整能显著改变历史开发集错误分布；旧23错应作为可复核ID，不能转为自动改标签依据。
- 支持适配与完整重拟合属于不同学习方案。弱适配、错误标签、bias-only及分数尺度控制必须同时报告；不要只展示最有利K或seed。
- 辅助预训练的效果依预算、适配方式、encoder/target随机性及BN策略变化。交叉设计pooled正差与大量负cell应并列呈现，避免“稳定正迁移”或“普遍无迁移”的整体宣称。
- 真实来源/植物身份、仪器/采集日期/物理批次/实测厚度及未见批次确认性验证仍缺。历史TRAIN、重复seed与人工仿射控制均不能补足这些证据。

跨分支源头与复现实验入口见[07_REPRODUCIBILITY_AND_SOURCE_MAP.md](07_REPRODUCIBILITY_AND_SOURCE_MAP.md)。本章不修改标签、冻结状态、原失败标记或后台PySCF任务。
