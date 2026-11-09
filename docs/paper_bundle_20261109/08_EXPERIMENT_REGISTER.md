# 全部实验族台账

由已审查的分模块台账生成。每行代表一个实验族或冻结协议，不能把行数当独立研究数量。数据范围不同的成绩不在此横向排序。

完整参数与指标见[JSON台账](../../evidence/paper_bundle_20261109/experiment_registry.json)、[Excel可读CSV](../../evidence/paper_bundle_20261109/experiment_registry.csv)和[长表指标](../../evidence/paper_bundle_20261109/metric_values.csv)。

| ID | 实验族 | 状态 | 结论与用途 | 分模块记录 |
| --- | --- | --- | --- | --- |
| CUR-001 | 当前数据及独立实物声明的版本核对 | PROVENANCE_VERIFIED | 同一version字符串存在独立性字段不同的登记表；当前分析应连同后续声明引用，不能用main较早scope否定后续实际声明。 | [设计、指标与证据](03_CURRENT_DATA_AND_SPECIES.md) |
| CUR-002 | 固定角色继承与当前六任务协议 | COMPLETED_PROTOCOL_AND_SPLIT | 分类、校准和迁移共享固定逐ID名单，校准池与PRED互斥。原历史角色及已曝光事实继续保留。 | [设计、指标与证据](03_CURRENT_DATA_AND_SPECIES.md) |
| CUR-003 | 旧855来源OOF的种类重解释 | COMPLETED_RETROSPECTIVE_AUDIT_NO_NEW_FIT | 旧R2的96.0234%只对应混合种类七来源组。算法增益仍含纯酸枣仁子集中的42个新增正确，但子集/折叠分数不是新训练纯物种模型成绩。 | [设计、指标与证据](03_CURRENT_DATA_AND_SPECIES.md) |
| CUR-004 | 标签修正后的六项固定留出分类 | COMPLETED_SIX_FIXED_HOLDOUT_TASKS | 二分类174/174；纯酸枣仁R2源87/100，整池重训93/100；R0源94/100、整池96/100必须同时报告。四药材331/339、HQ69/82、TM81/83；枣仁72/74掩盖少数来源零召回。 | [设计、指标与证据](03_CURRENT_DATA_AND_SPECIES.md) |
| CUR-005 | 修正任务的冻结分数K-shot校准 | COMPLETED_NEGATIVE_OR_NEUTRAL | 原lambda1平均75.90/81.50/84.05/86.30%，弱校准维持87%。本设置没有稳定少样本增益；87→93%源自额外100标签整池重训，不是K-shot校准。 | [设计、指标与证据](03_CURRENT_DATA_AND_SPECIES.md) |
| CUR-006 | 修正种类后首次跨物种迁移 | COMPLETED_MATCHED_NEGATIVE_TRANSFER | A81.00%、Aextra80.33%、B79.00%；B−A均值−2pp，B−Aextra−1.33pp。仅说明固定设置没有正迁移，不证明所有迁移方法无效。 | [设计、指标与证据](03_CURRENT_DATA_AND_SPECIES.md) |
| CUR-007 | 修正任务结果独立重算验证 | COMPLETED_ENGINEERING_VALIDATION | 指标、样品名单、支持条件与预测一致；这提升复现可信度，不增加样品数或外部科学证据。 | [设计、指标与证据](03_CURRENT_DATA_AND_SPECIES.md) |
| CUR-008 | 重新部署的TRAIN-only经典主矩阵 | COMPLETED_RETAINED_CLASSICAL_MATRIX | 融合R2 92.33%仍是开发分数；其15折中13折没有n特征，融合与alpha的特征预算不同，不能直接宣称多模态物理互补机制。 | [设计、指标与证据](03_CURRENT_DATA_AND_SPECIES.md) |
| CUR-009 | 已闭合作业的分卷补回传 | VERIFIED_BACKFILL_NOT_NEW_EXPERIMENT | 补回传2193个源文件、原字节约3.007GB、压缩约548.19MB；这是已有结果的运输和可恢复性证据，不新增575次训练或样品。 | [设计、指标与证据](03_CURRENT_DATA_AND_SPECIES.md) |
| CUR-010 | 缓存续算完成剩余神经矩阵 | COMPLETED_1305_ORIGINAL_MATRIX_NO_DUPLICATE_COUNT | 原1305矩阵完成，不能把825+480+1080+1305相加。酸枣仁p24/e192/full/pre480 A72.00%、Aextra73.5556%、B77.2222%，是匹配条件下+5.2222/+3.6667pp正迁移；其他负设置保留，不能普遍化。 | [设计、指标与证据](03_CURRENT_DATA_AND_SPECIES.md) |
| CUR-011 | 来源组留出和训练标签置乱诊断 | COMPLETED_20_FITS | 真实493/516=95.5426%，置乱准确率32.36/40.89/36.63%；源组大小与枣仁小训练池限制明显。来源组不是已核实新批次，3置乱seed不支持正式p值。 | [设计、指标与证据](03_CURRENT_DATA_AND_SPECIES.md) |
| CUR-012 | 冻结种类门与来源头的同ID联合正确性 | COMPLETED_RETROSPECTIVE_COMPONENT_COMPOSITION | 组件联合304/339=89.6755%，误路由8；不是新训练完整层级模型，完整joint macroF1和误route来源输出仍NOT_COMPUTED。 | [设计、指标与证据](03_CURRENT_DATA_AND_SPECIES.md) |
| CUR-013 | 等特征预算证据验证v3失败及v4继承 | COMPLETED_ENGINEERING_RECOVERY | 实现兼容修复不是新算法贡献；不能把v3失败记成功或视为独立重试支持。 | [设计、指标与证据](03_CURRENT_DATA_AND_SPECIES.md) |
| CUR-014 | 等K、表示与MC预算的完整经典消融 | COMPLETED_15_JOBS_ALL_DECLARED_REPEATS | inner-selected alpha_snv20与alpha_snv100均93.4444%，LDA91.7778%、MLP73.2222%；both-SNV融合91.6667%，n_raw83.6667%、n_SNV62.5556%。固定K245的alpha_snv10093.8889%只是消融，不能替换主程序。 | [设计、指标与证据](03_CURRENT_DATA_AND_SPECIES.md) |
| CUR-015 | 表示密度、source量和辅助种类的完整神经对照 | COMPLETED_315_JOBS_WITH_ALL_FOLDS_SEEDS | 在alpha128/source240/e192，A75.4444%、Aextra77.1111%、B79.7778%；alpha512同source240/e192的B63.4444%低于A67.2222%。细化采样不是独立新测量，也没有普遍提升；辅助HQ/TM效果不同。 | [设计、指标与证据](03_CURRENT_DATA_AND_SPECIES.md) |
| CUR-016 | 新神经source190的固定支持校准 | COMPLETED_CALIBRATION_CONTROLS | 所有正负校准值已回传，需相对identity规则和同source190的matched SVM比较；支持抽样不是新增独立样品或批次。不得用source240无支持的成绩直接替代source190校准基线。 | [设计、指标与证据](03_CURRENT_DATA_AND_SPECIES.md) |
| HML-001 | 老师代码静态审读与方法语义核对 | COMPLETED_STATIC_AUDIT | 定位CARS报告误差除2.8、MC-UVE有符号排序差异、SSA复用寻优折评分、PLS联合归一化、CNN人工2D重排/验证集复用等问题。标准方法和现成包不能因组合或改名变为原创算法。 | [设计、指标与证据](02_HISTORICAL_ML.md) |
| HML-002 | Octave/Python核心兼容性与工程实测 | COMPLETED_ENGINEERING_VALIDATION | 核心PLS/SVM/MC-UVE可复用；92次尝试的pass包含成功复现缺陷。原SSA示例调参和报告同折，示例CNN存在跨集相同特征行；其高分均不作药材泛化成绩。 | [设计、指标与证据](02_HISTORICAL_ML.md) |
| HML-003 | 批准主矩阵：传统分类与变量筛选 | COMPLETED_HISTORICAL_EXPLORATORY | MC20优于本矩阵普通全点SVM；提高MC次数、重复稳定筛选没有进一步增益。CARS-style为Python多分类适配，不是原MATLAB精度复现。 | [设计、指标与证据](02_HISTORICAL_ML.md) |
| HML-004 | 批准矩阵中的模态与完整导出范围敏感性 | COMPLETED_HISTORICAL_EXPLORATORY | 直接拼接未超过同类α单支。全导出高分不能证明该范围物理有效或不存在来源/制样混杂；负α及有效频段仍需仪器记录。 | [设计、指标与证据](02_HISTORICAL_ML.md) |
| HML-005 | 固定三用途留出与首次1-shot分数校准 | COMPLETED_HISTORICAL_EXPLORATORY | SVM88.8889%→74.9415%，MC20 87.7193%→72.8947%；固定弱样本校正平均明显下降，不是默认改善步骤。 | [设计、指标与证据](02_HISTORICAL_ML.md) |
| HML-006 | 116配置校准搜索与六类辅助K-shot | COMPLETED_HISTORICAL_EXPLORATORY | 七类正式主方案均回退不调整。强正则弱校正SVM89.0994%(+0.2105pp)仅预选诊断，开发增益未过规则，不能事后改主方法；六类10shot开发通过后最终仍微降0.0353pp。 | [设计、指标与证据](02_HISTORICAL_ML.md) |
| HML-007 | 54源表示/分类器搜索 | COMPLETED_HISTORICAL_EXPLORATORY | 开发选出的LDA最终未胜SVM；较高平衡准确率不等于所有指标改善。 | [设计、指标与证据](02_HISTORICAL_ML.md) |
| HML-008 | 60潜空间适配及全范围敏感性 | COMPLETED_HISTORICAL_EXPLORATORY | 三个主适配均回退不调整；PLS局部记忆给支持标签反而低于相同零支持结构，不能将读出变化当校准收益。 | [设计、指标与证据](02_HISTORICAL_ML.md) |
| HML-009 | 论文启发的监督ResNet/ProtoNet/FO-MAML分类适配 | COMPLETED_HISTORICAL_EXPLORATORY | FO-MAML自身64.7173%→67.5458%，宏F1约.3922→.4338，仍低于88.8889%SVM；头微调/全网微调/ProtoNet未通过开发门禁，诊断负结果保留。 | [设计、指标与证据](02_HISTORICAL_ML.md) |
| HML-010 | 四轮85候选：强MLP、支持加权SVM、回放锚定与元训练控制 | COMPLETED_HISTORICAL_EXPLORATORY | 正式90.2534%来自零目标支持的源训练；同任务监督继续训练逐样品预测与FO相同，未显示MAML特有增益。500更新高分仅预选诊断，校准反降。 | [设计、指标与证据](02_HISTORICAL_ML.md) |
| HML-011 | 图片引导特征/分类40设置 | COMPLETED_HISTORICAL_EXPLORATORY | LDA158/171=92.3977%但宏F1.8352<原SVM.8616及强MLP.8787；变量筛选候选没有自动提准。 | [设计、指标与证据](02_HISTORICAL_ML.md) |
| HML-012 | 图片引导概率/判别21设置 | COMPLETED_HISTORICAL_EXPLORATORY | SVM温度缩放NLL .541355→.409750、Brier .264373→.199344、ECE .229348→.093669，准确率保持88.8889%；硬分类改动未过开发。 | [设计、指标与证据](02_HISTORICAL_ML.md) |
| HML-013 | 图片引导SVM/MLP融合20设置 | COMPLETED_HISTORICAL_EXPLORATORY | 正式回退SVM权重0的较强单MLP，不能称融合收益；真值oracle91.8129%不可部署，不是实测模型成绩。 | [设计、指标与证据](02_HISTORICAL_ML.md) |
| HML-014 | R0—R3嵌套优化与90.41%精确重现 | COMPLETED_HISTORICAL_EXPLORATORY | R2 821/85596.0234%、宏F1.9196846、BA89.9184%；更复杂R3反降5个正确。R1相对固定锚定与R0的对比含义不同，不能全归因单个因素。 | [设计、指标与证据](02_HISTORICAL_ML.md) |
| HML-015 | R4组合家族、稳定选择与CNN受控对照 | COMPLETED_HISTORICAL_EXPLORATORY | R4内选策略94.9708%低于R3和R2；预算MC、稳定MC/CARS、CNN路线均未胜较强传统路线。 | [设计、指标与证据](02_HISTORICAL_ML.md) |
| HML-016 | 统一855协议基线复现与条件消融 | COMPLETED_HISTORICAL_EXPLORATORY | R2精确重现；全谱SVM88.0702%、LDA91.9298%、单隐层MLP88.7719%；去MC、随机同K、去C/gamma搜索、去类权等条件消融均下降。 | [设计、指标与证据](02_HISTORICAL_ML.md) |
| HML-017 | 原64—32强MLP的三seed统一OOF与源更新控制 | COMPLETED_HISTORICAL_EXPLORATORY | 追加训练的三seed均值略升，seed43反降；未超过R2，不能只挑seed42多9份作为所有种子改善。 | [设计、指标与证据](02_HISTORICAL_ML.md) |
| HML-018 | 统一研究审计中的保守校准独立复核 | COMPLETED_HISTORICAL_EXPLORATORY | v1限幅使满足翻类必要条件的11120对实际均不翻类；正式回退无增益，不能作为提准新方法。旧λ1负结果保留，抽样均值不覆盖先前20draw。 | [设计、指标与证据](02_HISTORICAL_ML.md) |
| HML-019 | 多表示/深度学习升级：8家族小预算嵌套实测 | COMPLETED_HISTORICAL_EXPLORATORY | 四表示主策略seed42 85.0292%、三seed均值83.7037%，未超过R2；12/24epoch小预算不能否定所有深度网络，多表示profile耦合不能称纯因素融合消融。 | [设计、指标与证据](02_HISTORICAL_ML.md) |
| HML-020 | 480条历史训练损失曲线只读诊断 | COMPLETED_HISTORICAL_EXPLORATORY | 全部配置末段训练loss仍下降，仅说明该损失定义下优化仍有进展；缺逐epoch验证曲线，不能据此判定欠拟合或延长训练必提准。 | [设计、指标与证据](02_HISTORICAL_ML.md) |
| HML-021 | 仅源内的96epoch学习曲线pilot | COMPLETED_HISTORICAL_EXPLORATORY | 24→96轮αCNN内验证64.9123→75.9259%，四表示76.6082→83.8207%；四表示BA48→96却74.9320→69.4388。优化进展和少数类/泛化单调改善不是一回事。 | [设计、指标与证据](02_HISTORICAL_ML.md) |
| HML-022 | 匿名七分类完整部署与未来输入工具 | COMPLETED_DEPLOYMENT_FUTURE_VALIDATION_NOT_RUN | 工具和完整模型已交付；最终MC50/PLS1/20%=123点，SVM C1000/gamma0.1×scale/balanced。真新样品准确率未测，不能给部署模型填历史96.02%。 | [设计、指标与证据](02_HISTORICAL_ML.md) |
| EXT-001 | UltraIR外部FTIR来源分类 | COMPLETED | PLS-DA在两个外部任务中数值最高；筛选器没有在两个队列上稳定胜出，山银花PLS-DA与CARS只差1条谱。 | [设计、指标与证据](04_EXTERNAL_TRANSFER_AND_ROBUSTNESS.md) |
| EXT-002 | UltraIR相对丰度回归及归档Octave核心 | COMPLETED | 若干目标有有限可预测性，多数弱或负R²；程序执行成功不等于实用定量成功。 | [设计、指标与证据](04_EXTERNAL_TRANSFER_AND_ROBUSTNESS.md) |
| EXT-003 | 公开THz数据获取及资格核验 | COMPLETED_WITH_BLOCKED_SOURCES | 可执行的外部任务与受阻数据已分开，不能按文件数/曲线数/像素数扩充独立N。 | [设计、指标与证据](04_EXTERNAL_TRANSFER_AND_ROBUSTNESS.md) |
| EXT-004 | 公开人参/西洋参四轮算法验证 | COMPLETED_EXPLORATORY | 预先source选择的R4未超过R0，复杂筛选无稳定收益；n+LDA是事后开发候选，不是确认赢家。 | [设计、指标与证据](04_EXTERNAL_TRANSFER_AND_ROBUSTNESS.md) |
| EXT-005 | 公开水果按果实分组的工程回归 | COMPLETED_EXPLORATORY | PR/PM有有限工程回归收益，MA-Brix全部家族负R²，MA-DM学习模型不如均值基线。 | [设计、指标与证据](04_EXTERNAL_TRANSFER_AND_ROBUSTNESS.md) |
| EXT-006 | 跨植物ResNet预训练配对试验 | COMPLETED_NEGATIVE_TRANSFER | 加入外部预训练无平均收益；source-policy从未选择B/C，其较好表现不能归因于外部数据。 | [设计、指标与证据](04_EXTERNAL_TRANSFER_AND_ROBUSTNESS.md) |
| EXT-007 | 保守跨植物预训练48轮上限试验 | COMPLETED_NEGATIVE_TRANSFER | 固定保守配方没有获益，外部B比同额外计算Aextra仍低。 | [设计、指标与证据](04_EXTERNAL_TRANSFER_AND_ROBUSTNESS.md) |
| EXT-008 | 同配方48→96轮上限的预算消融 | COMPLETED_NEGATIVE_TRANSFER_WITH_BUDGET_GAIN | 扩大训练/选择预算有益，但96上限B仍低于A及Aextra准确率；不能把预算收益记为外部数据收益。 | [设计、指标与证据](04_EXTERNAL_TRANSFER_AND_ROBUSTNESS.md) |
| EXT-009 | 补充光谱入库与多表示恢复 | COMPLETED_WITH_BLOCKED_FILES | 补表示提高可运行性，不扩大独立N；6399条记录×表示血缘行不是6399份独立样品。 | [设计、指标与证据](04_EXTERNAL_TRANSFER_AND_ROBUSTNESS.md) |
| EXT-010 | 补充固定三分角色的传统分类对照 | COMPLETED | MC在旧A比同协议SVM提高4.2885pp并压缩89.87%变量，但LDA略优；B整体高分主要来自旧171，新3个query每seed仅对1个。 | [设计、指标与证据](04_EXTERNAL_TRANSFER_AND_ROBUSTNESS.md) |
| EXT-011 | 补充角色少样本校准与保守no-op | COMPLETED_WITH_NEGATIVE_AND_NOOP_RESULTS | 传统校准明显损害部分任务，保守规则避免损害但没有带来提升。 | [设计、指标与证据](04_EXTERNAL_TRANSFER_AND_ROBUSTNESS.md) |
| EXT-012 | 补充多表示ResNet：匹配三分支控制 | COMPLETED_WITH_MIXED_RESULTS | 主ZY任务没有神经多模态普遍增益；HQ有匹配神经内部收益但低于同拆分LDA，TM低于同拆分MLP。 | [设计、指标与证据](04_EXTERNAL_TRANSFER_AND_ROBUSTNESS.md) |
| EXT-013 | 补充NN历史无拟合独立重放 | COMPLETED_NO_NEW_TRAINING | 后续实际重放验证原结果未被过期训练期限或合法量子账本追加破坏；没有新的科学性能增益。 | [设计、指标与证据](04_EXTERNAL_TRANSFER_AND_ROBUSTNESS.md) |
| EXT-014 | 保存OOF的11方法配对统计与稀有类影响 | COMPLETED_NO_NEW_FITS | 保存预测的条件统计支持历史R2在该曝光集合的比较，不能当嵌套CV显著性检验或独立批次总体置信保证。 | [设计、指标与证据](04_EXTERNAL_TRANSFER_AND_ROBUSTNESS.md) |
| EXT-015 | 冻结R2模型18条件谱形稳健性 | COMPLETED_EXPLORATORY | 乘法增益和常数基线下标签不变；频轴±.002THz准确率降至89.9415%/92.1637%，提示该工程扰动敏感性。 | [设计、指标与证据](04_EXTERNAL_TRANSFER_AND_ROBUSTNESS.md) |
| EXT-016 | 预定半bin/整bin源内频移增强 | COMPLETED_WITH_TRADEOFF | 两种配方提高模拟频移下准确率，同时损失原始谱准确率及Macro-F1；不能宣称提高96.0234%的主任务成绩。 | [设计、指标与证据](04_EXTERNAL_TRANSFER_AND_ROBUSTNESS.md) |
| EXT-017 | 2026-11-04公开来源再核验与下载阻塞 | COMPLETED_ACQUISITION_REVIEW_NO_NEW_DATA | 已有来源可用于独立任务探索，公开候选受阻/许可不明的保持BLOCKED，不能写成新采集成功。 | [设计、指标与证据](04_EXTERNAL_TRANSFER_AND_ROBUSTNESS.md) |
| BML-001 | ZY01错误集中与同源覆盖干预（219项初轮） | ALL219_SCIENCE_SEALS_VERIFIED_FINAL_SUMMARY_EXTERNALLY_QUALIFIED | 错误集中同时受查询分母与来源覆盖影响；同源覆盖改善与声明标签的一致性。不同模型可恢复大多数旧错，因此不能将ZY01整本或旧23错自动改为酸枣仁。 | [设计、指标与证据](06_WORKER_B_DIAGNOSTICS.md) |
| BML-002 | 少量目标支持校准与全流程重拟合（219项初轮） | COMPLETE_EXTERNALLY_QUALIFIED | 支持适配通常提高当前来源查询一致性，但K并非处处单调；弱anchor预测与source相同。全流程重拟合为不同学习方案，应与分数适配分别报告。 | [设计、指标与证据](06_WORKER_B_DIAGNOSTICS.md) |
| BML-003 | 三种子辅助预训练匹配对照（219项初轮） | COMPLETE_EXTERNALLY_QUALIFIED | 三种子初轮的full条件下B低于Aextra平均值；480-frozen平均正收益依赖种子，不能指定为普遍赢家。后续独立交叉随机种子设计另列，不能混用分母或结论。 | [设计、指标与证据](06_WORKER_B_DIAGNOSTICS.md) |
| BML-004 | 初轮汇总程序的外部科学资格 | 219_DONE_SEALED_PLUS_ONE_FAILED_SUMMARY_EXTERNALLY_QUALIFIED | 219项科学任务完成。汇总失败来自结果超过4MiB读取门槛，外部核查补足结果科学资格；不可写为220项全部DONE_SEALED。 | [设计、指标与证据](06_WORKER_B_DIAGNOSTICS.md) |
| BML-005 | deep-v3全部科学任务与16块汇总完整性 | PASS_ALL_971_SCIENTIFIC_JOBS_AND_16_SUMMARY_BLOCKS_INDEPENDENTLY_VERIFIED | 970项科学计算及1项汇总均封存，16个块完整；其中3块只提供预训练/来源A工件与溯源覆盖，不单独推断准确率。 | [设计、指标与证据](06_WORKER_B_DIAGNOSTICS.md) |
| BML-006 | 固定source分数与SNV仿射变换控制 | COMPLETE_BLOCK | 60项人工正仿射输入变换均未改变预测；这符合row-SNV的数值不变性，单独的纯幅度/常数偏移不足以解释该SNV模型的错误。频率相关形状、散射/干涉或反演变化仍未被排除。 | [设计、指标与证据](06_WORKER_B_DIAGNOSTICS.md) |
| BML-007 | deep-v3 alpha支持适配、偏置及标签控制 | COMPLETE_BLOCK | legacy anchor由12140/13500增至K5的12717；K10为12754、K20为12751，存在非单调性。 K5正斜率100/100；与bias-only和正分数重标度的配对正确性逐行一致，支持将该二分类增益解释为阈值/位置调整。错误标签翻转显著恶化，不构成随机零假设检验。 | [设计、指标与证据](06_WORKER_B_DIAGNOSTICS.md) |
| BML-008 | deep-v3 n支持适配、偏置及标签控制 | COMPLETE_BLOCK | legacy anchor由7360/13500增至K5的9905；K10为10014、K20为10003，存在非单调性。 K5正斜率100/100；与bias-only和正分数重标度的配对正确性逐行一致，支持将该二分类增益解释为阈值/位置调整。错误标签翻转显著恶化，不构成随机零假设检验。 | [设计、指标与证据](06_WORKER_B_DIAGNOSTICS.md) |
| BML-009 | deep-v3 fusion支持适配、偏置及标签控制 | COMPLETE_BLOCK | legacy anchor由9220/13500增至K5的12652，K20为12920；该条件中增加支持量继续改善，但不是所有模态的普遍规律。 K5正斜率100/100；与bias-only和正分数重标度的配对正确性逐行一致，支持将该二分类增益解释为阈值/位置调整。错误标签翻转显著恶化，不构成随机零假设检验。 | [设计、指标与证据](06_WORKER_B_DIAGNOSTICS.md) |
| BML-010 | deep-v3 alpha供体数量/标准化/gamma/正则/来源多样性 | COMPLETE_BLOCK | 较少酸枣仁供体的某些设置改善当前查询，但改变标准化、核带宽与惩罚后效果仍依模态和来源而变；供体数量效应不能直接命名为纯类别不平衡因果。来源多样性须使用匹配held-source的配对行解释。 | [设计、指标与证据](06_WORKER_B_DIAGNOSTICS.md) |
| BML-011 | deep-v3 n供体数量/标准化/gamma/正则/来源多样性 | COMPLETE_BLOCK | 较少酸枣仁供体的某些设置改善当前查询，但改变标准化、核带宽与惩罚后效果仍依模态和来源而变；供体数量效应不能直接命名为纯类别不平衡因果。来源多样性须使用匹配held-source的配对行解释。 | [设计、指标与证据](06_WORKER_B_DIAGNOSTICS.md) |
| BML-012 | deep-v3 fusion供体数量/标准化/gamma/正则/来源多样性 | COMPLETE_BLOCK | 较少酸枣仁供体的某些设置改善当前查询，但改变标准化、核带宽与惩罚后效果仍依模态和来源而变；供体数量效应不能直接命名为纯类别不平衡因果。来源多样性须使用匹配held-source的配对行解释。 | [设计、指标与证据](06_WORKER_B_DIAGNOSTICS.md) |
| BML-013 | deep-v3纯预训练与source-A复用工件 | COMPLETE_ARTIFACT_PROVENANCE_BLOCKS | 三块完整，但不由预训练任务数推断准确率，也不将复用A预测记成额外拟合或额外生物样品。 | [设计、指标与证据](06_WORKER_B_DIAGNOSTICS.md) |
| BML-014 | deep-v3交叉随机种子 120_full | COMPLETE_BLOCK | B−Aextra pooled为+0.072674个百分点；24个随机化cell中14正/1零/9负。只能报告该预设条件的平均差与随机化敏感性，不支持所有种子稳定获益。 | [设计、指标与证据](06_WORKER_B_DIAGNOSTICS.md) |
| BML-015 | deep-v3交叉随机种子 120_frozen | COMPLETE_BLOCK | B−Aextra pooled为+0.855943个百分点；24个随机化cell中11正/0零/13负。只能报告该预设条件的平均差与随机化敏感性，不支持所有种子稳定获益。 | [设计、指标与证据](06_WORKER_B_DIAGNOSTICS.md) |
| BML-016 | deep-v3交叉随机种子 480_full | COMPLETE_BLOCK | B−Aextra pooled为+0.274548个百分点；24个随机化cell中11正/7零/6负。只能报告该预设条件的平均差与随机化敏感性，不支持所有种子稳定获益。 | [设计、指标与证据](06_WORKER_B_DIAGNOSTICS.md) |
| BML-017 | deep-v3交叉随机种子 480_frozen | COMPLETE_BLOCK | B−Aextra pooled为+1.049742个百分点；24个随机化cell中15正/0零/9负。只能报告该预设条件的平均差与随机化敏感性，不支持所有种子稳定获益。 | [设计、指标与证据](06_WORKER_B_DIAGNOSTICS.md) |
| BML-018 | deep-v3冻结编码器BN干预 bn_neutral | COMPLETE_BLOCK | neutral下B为1473/1548，与Bshuffle仅差1行；相对native B少24行。BN统计可影响比较，现有对照未确立唯一物理批次原因或普遍辅助语义优势。 | [设计、指标与证据](06_WORKER_B_DIAGNOSTICS.md) |
| BML-019 | deep-v3冻结编码器BN干预 bn_source_recal | COMPLETE_BLOCK | source-recal下B为1497/1548，与Aextra差13行、Bshuffle差24行；相对native B为3行由错转对、3行由对转错，净差0。BN统计可影响比较，现有对照未确立唯一物理批次原因或普遍辅助语义优势。 | [设计、指标与证据](06_WORKER_B_DIAGNOSTICS.md) |
| QC-001 | Gaussian可行性、候选成分登记与PySCF技术smoke | CANDIDATE_DFT_NOT_RUN_SOFTWARE_SMOKE_COMPLETED | 证明当前开源导数路径可运行；没有取得候选分子理论吸收谱。Gaussian未实际执行。 | [设计、指标与证据](05_QUANTUM_AND_MECHANISM.md) |
| QC-002 | Spinosin初始pilot：12构象与4个真实几何评估 | PARTIAL_OPTIMIZATION_NO_SPINOSIN_FREQUENCIES | 水Opt/Freq通过，Spinosin当时几何未收敛，按原资源限额停止；该NOT_RUN快照不能被后续完成情况回写。 | [设计、指标与证据](05_QUANTUM_AND_MECHANISM.md) |
| QC-003 | Spinosin direct几何续算与C001优化 | GEOMETRY_OPTIMIZATION_COMPLETED_FREQUENCIES_NOT_RUN_AT_THIS_STAGE | 达到当时一阶优化要求；核几何优化收敛与Hessian正定是不同状态。 | [设计、指标与证据](05_QUANTUM_AND_MECHANISM.md) |
| QC-004 | v4/v5精确状态与跨进程Hessian恢复验证 | SOFTWARE_CONTROL_COMPLETED_SPINOSIN_INTERMEDIATES_PARTIAL | 恢复算法小分子控制通过；21/150是RHF中间单元，不是21个模式或整个研究完成率。 | [设计、指标与证据](05_QUANTUM_AND_MECHANISM.md) |
| QC-005 | v6/v7可恢复XC分块与完整导数控制 | ENGINEERING_RECOVERY_WITH_HISTORICAL_FAILURES | 分块便于恢复和保存，不是新泛函或药材机制创新；必须以真实完整控制和最终组装结果判定，不能把代码测试当分子结果。 | [设计、指标与证据](05_QUANTUM_AND_MECHANISM.md) |
| QC-006 | Spinosin v8完整Hessian、219有符号模式与独立重放 | COMPLETED_WITH_25_IMAGINARY_MODES_NOT_A_VERIFIED_MINIMUM | 计算与保存完整，但当前构象/导数结果未通过正频最低点资格；25个大虚频原因未确定。 | [设计、指标与证据](05_QUANTUM_AND_MECHANISM.md) |
| QC-007 | A003/A004：全部响应残差、方向有限差分及轨道稳定性尝试 | 225_RHS_MEASURED_DIRECTIONAL_DISAGREEMENT_ORBITAL_TEST_PARTIAL | 给定响应方程的225方向已测完；解析负曲率与梯度差分正曲率矛盾存在，不能据此单因归咎。轨道稳定性未通过。 | [设计、指标与证据](05_QUANTUM_AND_MECHANISM.md) |
| QC-008 | 固定网格/完整native势能的严格收敛诊断 | WATER_CONTROL_COMPLETED_LARGE_DIRECTIONAL_CASES_FAILED | 小分子控制与Spinosin新零点通过不等于全部219模式重算；严格位移点未达标，旧25虚频仍未解决。 | [设计、指标与证据](05_QUANTUM_AND_MECHANISM.md) |
| QC-009 | A的72h首批及recovery-002重建网格零点 | MAIN_QUEUE_CLOSED_WITH_FAILURES_RECOVERY_GRID1_FAILED_GRID3_NOT_EVIDENCED | 首批有限队列结束不是科学问题解决；recovery控制成功、grid1电子零点失败，未进入梯度；固定A提交无grid3已完成证据。 | [设计、指标与证据](05_QUANTUM_AND_MECHANISM.md) |
| QC-010 | Worker B 原儿茶酸初始完整Opt/Freq/模式独立重放 | COMPLETED_WITH_IMAGINARY_MODES | 完整振动计算已做；虚频保留，最低点未通过。重放只核对数学变换，不是新DFT运行。 | [设计、指标与证据](05_QUANTUM_AND_MECHANISM.md) |
| QC-011 | Worker B 毛蕊异黄酮初始完整Opt/Freq/模式独立重放 | COMPLETED_WITH_IMAGINARY_MODES | 完整振动计算已做；虚频保留，最低点未通过。重放只核对数学变换，不是新DFT运行。 | [设计、指标与证据](05_QUANTUM_AND_MECHANISM.md) |
| QC-012 | Worker B网格/梯度响应及完整旧虚频子空间诊断 | DIRECTIONAL_AND_SUBSPACE_DIAGNOSTICS_COMPLETED_NO_FULL_NEW_HESSIAN | 所测旧负子空间在新条件下为正，支持导数/数值路线有待辨析；有限子空间不等于全分子最低点或完整频率验证。 | [设计、指标与证据](05_QUANTUM_AND_MECHANISM.md) |
| QC-013 | 本次v1/v2严格SCF认证队列的真实失败 | CLOSED_FAILED_OPTIMIZATION_HESSIANS_SKIPPED | v2第8点因实际额外循环能差未达标而失败；最终Fock残差小不能替代真实合格循环，也不能称几何已完成。 | [设计、指标与证据](05_QUANTUM_AND_MECHANISM.md) |
| QC-014 | v3原儿茶酸grid3含网格响应的完整双步长数值Hessian | FULL_NUMERICAL_HESSIANS_AND_MODES_COMPLETED_STRICT_CERTIFICATE_FALSE | 已完成完整双步长矩阵，全部45模式为正且两步长相近；原严格最低点标志仍false。 | [设计、指标与证据](05_QUANTUM_AND_MECHANISM.md) |
| QC-015 | 原儿茶酸既有结果的最低点分层独立审计 | POSTRUN_AUDIT_NATIVE_PROTOCOL_EVIDENCE_TRUE_STRICT_FALSE | 可在明确原生求解器接受语义下报告“近似局部最低点数值证据”；不能说全部fresh残差达到1e-7，更不能写严格驻点认证通过。 | [设计、指标与证据](05_QUANTUM_AND_MECHANISM.md) |
| QC-016 | v3/Calycosin grid3新优化、完整数值Hessian断点及续算 | OPTIMIZATION_COMPLETED_FULL_HESSIAN_IN_PROGRESS | 新优化已完成；Hessian仍进行中，不能从部分矩阵报告93个新模式、无虚频或最低点通过。 | [设计、指标与证据](05_QUANTUM_AND_MECHANISM.md) |
| QC-017 | 更严格原儿茶酸最低点修复后台分支 | RUNNING_AT_OBSERVATION | 本行仅是带时间戳的执行快照；后续结果应另冻结、核对并更新，不以RUNNING字样代替真实进程/科学完成。 | [设计、指标与证据](05_QUANTUM_AND_MECHANISM.md) |
| QC-018 | 毛蕊异黄酮独立断点续算后台分支 | RUNNING_AT_OBSERVATION | 本行仅是带时间戳的执行快照；后续结果应另冻结、核对并更新，不以RUNNING字样代替真实进程/科学完成。 | [设计、指标与证据](05_QUANTUM_AND_MECHANISM.md) |
| QC-019 | 标准品、成分定量、实验峰归属与独立批次验证缺口 | NOT_RUN_NO_CHEMICAL_ASSIGNMENT | 当前量子证据不能证明产地分类由特定化合物/含量驱动，也不能取代固态实验吸收或定量检测。 | [设计、指标与证据](05_QUANTUM_AND_MECHANISM.md) |
