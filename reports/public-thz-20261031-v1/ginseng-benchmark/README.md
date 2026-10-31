# 公开人参／西洋参THz：冻结多轮文件组留出实验

本目录由用户要求的公开植物THz扩展任务建立。数据是仓库既有、来自用户上传公开V2资料包的42条压片观测，不是本轮新下载42个独立植株；本次只新建算法实验。原18个CSV及旧论文／数据／酸枣仁96.02%基线不改。

- 来源：Mendeley DOI10.17632/hvbp5mdtxx.2，发布者CC BY4.0。
- 标签：0＝西洋参Panax quinquefolius（22条），1＝亚洲人参Panax ginseng（20条）；不是产地。
- 表示：n无量纲、alpha cm^-1，原65点0.2–1.6THz；141个负alpha值全部保留并标记。
- 六个导出文件组各7条。outer逐个留一组，inner逐个留其余五组；同记录n/alpha始终同角色。
- 上游植株、原料和批次身份仍未知；故仅EXPLORATORY_OBSERVATION_LEVEL_UNKNOWN_UPSTREAM，不称真实跨批次或独立植株确认。

## 事前冻结预算

R0固定Raw alpha＋训练源StandardScaler＋C1/scale/balanced SVM。
R1共36个SNV alpha MC-UVE配置（MC10/20/50×LV1/2/3×保留.1/.2/.4/.6），SVM固定C10/scale/balanced。
R2对当前outer源内选中的MC设置试48个SVM配置（C六档×gamma四档×两类别权重）。
R3保持当前源选MC/SVM设置，比较七种预处理。
R4每个alpha/n/paired表示各11个fullSVM／收缩LDA／PCA-LDA／PLSDA／MLP候选，共33；另保留R2/R3为已有源结果别名。
全部seed42/43/44保留；每个outer、每个候选按15个inner case平均Macro-F1→Balanced Accuracy→Accuracy→固定ID选择，不使用outer标签。

正式源候选分类器计划11160次（6×5×3×124），保存全部权重／失败。15个表示×分类器家族角色各保留源内winner，加R0–R3和R4总体policy共20角色；每角色每seed恰42条OOF，重复seed和角色不增加实物样品数。外层相同数值配置可作完全相同当前源权重别名，记录实际拟合数和别名数。

## 实际边界和缓存

12项实际正反测试已通过，含独立原MC各比例变换逐元素完全一致、每PLS源追踪、cache字节／内存扰动、allowed-source改变、二分类SVC原生票决、分模态SG、严格无真值推理包、最后一份outer truth篡改在第一份语义读取前拒绝。测试拟合是明确合成夹具，不是新增真实药材。

只在同source IDs／X／y／split／representation／preprocessor／MC runs/LV／code／seed下复用拟合transform，SVM超参数变化不会重用其他折权重。MC只从当前源估计完整排名；不同保留比例是确定性排序前缀再按原列号排序，未借用全42排名或历史折并集。PCA和PLSDA内部PLS有独立FitScope。

当前实际run对全部输入pack字节（含axis）已锁SHA；standalone transform_source的缓存identity未额外单列axis，只能在本已锁原65轴协议中使用，不作为任意变轴API。SG固定15点在该粗网格约跨0.30625THz，不能宣称与酸枣仁612点上的同15点滤波具有相同物理带宽。所有paired操作先分别处理alpha/n再拼接，不对两种物理量的拼接边界卷积。

## 执行状态

2026-10-31已完成12项实际边界测试、11,160次源内分类器拟合和全部六个outer的20角色选择锁。独立审计重放全部78,120条源预测及分数，检查全部权重、缓存和实际PLS源范围后PASS；root随后才释放outer。正式outer实际拟合306个分类器，54个完全相同源配置作为别名，无额外拟合；全部2,520条预测先锁，新的独立score进程核完全部truth字节后才首次解析外层真值。文件／API隔离不是OS沙箱，有限测试不证明绝对零泄漏。

本run在G1外层已有小预算对抗诊断曝光后开展，不能称全新独立确认；所有主预算和全六折选择锁在主run外层评分之前冻结，不根据此次outer结果追加配置。原料身份未知的限制仍在。

|事前定义角色|Accuracy，三种子均值±样本标准差|Macro-F1均值|Balanced Accuracy均值|
|---|---:|---:|---:|
|R0 Raw-alpha SVM|54.76% ± 0.00|0.5411|0.5432|
|R1 MC-UVE参数选择|45.24% ± 2.38|0.4507|0.4515|
|R2 SVM参数选择|43.65% ± 5.99|0.4352|0.4371|
|R3 预处理选择|43.65% ± 5.99|0.4352|0.4371|
|R4 源内总体选择policy|51.59% ± 1.37|0.5117|0.5129|

每个种子均为同一42条记录的6组OOF预测；重复种子不会增加样本数。外层描述性最高家族是n＋收缩LDA：每种子26/42（61.90%），Macro-F1 0.6156，Balanced Accuracy 0.6159。这是20个预定义角色中观察到的最高值，**不是事前选定的独立确认模型**；R4源内总体选择规则本次仅51.59%。保留MC-UVE/SVM优化在该小型外部任务下降的真实结果，不据此改写酸枣仁96.02%的历史结果，也不把两个不同任务的数字合并。

实际MC学习共1,368份当前源排名、32,760次系数标准化和65,520次底层one-vs-rest PLS拟合。源内被选配置保留7或13/65特征；15个源子集×种子之间两两Jaccard均值为0.204–0.436，稳定性有限。固定300epoch的MLP共1,134次均记录ConvergenceWarning，不能描述为已经收敛。源试验累计计时276.50秒，outer累计8.23秒，包含当前试验变换缓存检查和序列化，非纯训练时间；完整命令墙钟时间见driver events。

主要证据：`selection_lock.json`（8d40ced3…）、`prediction_lock.json`（f0a6001f…）、`scoring_manifest.json`（81ba5343…）、`source_selection_diagnostics.json`、`execution_costs.json`、`seed_summary.json`、全部种子逐记录预测与原权重。outer独立重放已PASS：`../independent-audit/evaluation_review.json`（46b1b58a…）重新加载全部306权重、54别名，复算全部2,520条原生预测及分数、60个pooled指标、360个fold指标，最大分数差约4e-15，独立审计零拟合。完整逐类Precision/Recall/F1及混淆矩阵见`../independent-audit/independently_recomputed_ginseng_metrics.json`（b6cf760f…）。不要把公开论文准确率或其他植物结果当本次复现结果。
