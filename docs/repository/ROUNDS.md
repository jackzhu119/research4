# 保留的任务与轮次

早期错标分类各R轮、混种增强调优和旧预测重贴标签均从有效台账删除。分类论文只使用[清洁结果包](../../paper/current/README.md)，以下还包括未完成任务、核验和量子工作，不等于独立训练次数。

|记录|用途／状态|说明|
|---|---|---|
|`S-fixed6` 纠正种类：六项固定留出分类|COMPLETE|六任务1366个分类设置；使用纠正后的种类，旧样品有历史开发曝光，不能作为全新确认集。|
|`S-transfer3seed` 纠正种类：三种子迁移|COMPLETE|1449迁移设置；与六项分类分开登记，不按继承分支重复累计。|
|`S-background1305` 早期后台矩阵启动记录|SUPERSEDED_PROGRESS_SNAPSHOT|启动与早期进度由S-1305最终证据接续，不能另算1305新任务。|
|`S-1305` 纠正种类原矩阵：重启、备份、续算合一|COMPLETE|225经典+1080神经=1305。825已结束+480剩余续算；575备份点是复制子集。|
|`S-validation330` V4新增验证矩阵|COMPLETE|15经典+315神经=330。V3标签编码失败和V4修复不重复计同一科学任务；原始失败保留。|
|`S-worker-B20fit` Worker B种类误用控制|COMPLETE|20次诊断性拟合，作用是审计种类混用；不把部署分支再算一轮。|
|`B1` B机制第1轮：E01：固定样本量来源覆盖 × 类别比例；E02：固定源模型 K-shot；E03：纯酸枣仁五来源；E04：保存预测上的机制描述|COMPLETE|New finite exploratory mechanism matrix with exact parent endpoint reuse; inherited report trees are identical snapshots, not new runs.|
|`B2` B机制第2轮：F01：校准自由度、分数单位与支持样品敏感性（45任务）；F02：完整重训的Scaler与gamma配对分解（45任务）；F03：类别权重与来源覆盖稳定性（240任务）；F04：五来源标签预算学习曲线（15任务）|COMPLETE|New finite exploratory mechanism matrix with exact parent endpoint reuse; inherited report trees are identical snapshots, not new runs.|
|`B3` B机制第3轮：G01：损失归一化 × 截距惩罚（45任务）；G02：新增标签信息与无标签统计量分解（45任务）；G03：受控Support标签污染剂量（45任务）；G04：类别权重的分数、margin和有效C约束审计（240任务）|COMPLETE|New finite exploratory mechanism matrix with exact parent endpoint reuse; inherited report trees are identical snapshots, not new runs.|
|`B4` B机制第4轮：H01：总损失质量与增量监督样品（45任务）；H02：C × 类别权重的约束激活实验（240任务）；H03：弱校准运动与解析决策激活审计（45任务，零拟合）；H04：固定总支持量的类别构成与残差加权（45任务）|COMPLETE|New finite exploratory mechanism matrix with exact parent endpoint reuse; inherited report trees are identical snapshots, not new runs.|
|`B5` B机制第5轮：I01：仅在Source内部选择适配策略（45任务）；I02：固定总预算的标签盲法Support获取（45任务）；I03：保留输出基线的OVO信息校准（45任务）；I04：同Support权重下的ESS额外收缩（45任务）|COMPLETE|New finite exploratory mechanism matrix with exact parent endpoint reuse; inherited report trees are identical snapshots, not new runs.|
|`M-continuous` 持续元学习|LIVE_STATUS_UNVERIFIED|11/11 16:40仅有45上下文初始检查点、51文件恢复验证；无全部后续评估快照，当前运行及最终完成均待核实。|
|`V-all-round` 全轮核验：ML已完成，量子继续|PARTIAL_COMPLETE_RUNNING_AT_LAST_HEARTBEAT|ML复核已完成；量子publication/IR心跳运行。原儿茶酸IR第16模式、毛蕊异黄酮导数35/396，与另外两条计算线重叠。|
|`Q-frequency-repair` PySCF虚频修复|LIVE_STATUS_UNVERIFIED|09:02北京时间仅证明云重启后恢复；progress.json仍是11/11水控制快照，当前存活待核实。|
|`Q-IR-repair` PySCF独立IR强度修复|RUNNING_AT_LAST_HEARTBEAT|原儿茶酸direct/grid5独立IR已合格；df-grid5逐模式偶极第32模式仍在计算，毛蕊异黄酮/斯皮诺素未全部合格。|
|`Q-publication-recovery` PySCF论文恢复|RUNNING_AT_LAST_HEARTBEAT|毛蕊异黄酮完整数值导数61/396；斯皮诺素在原顺序队列未全部合格；新并行线已立即补其几何计算。|
|`Q-spinosin219` 斯皮诺素完整219有符号模式|COMPUTATION_COMPLETE_SCIENTIFIC_QUALIFICATION_FAILED|完整Hessian组装及独立数学正常模式重放；219模式=194正+25虚；2026-11-06北京时间14:20:44组装、14:20:51重放；最低点/机制资格false，IR NOT_COMPUTED。|
|`Q-fixedgrid` A005/A006固定网格步长与能量诊断|PARTIAL_FAILED_CLOSED|水31点步长门禁后条件式Spinosin单方向3SCF；A005水13点完成但控制失败，大分子未启动；A006水31/31通过，Spinosin因ACTUAL_CPU_GUARD失败闭合；无新完整频谱。|
|`Q-72h` 72小时首批及recovery-002|QUEUE_CLOSED_WITH_FAILURES|ML增强稳健性+量子严格零点/方向检查和恢复控制；首批11项6成功/3失败/2依赖未运行，其中一次成功复用零点；recovery002水/乙醇10控制通过，Spinosin grid1严格零点失败；队列结束不等于科学问题解决。|
|`Q-host-A` Host A A003/A004响应与曲率诊断|HISTORICAL_DIAGNOSTICS_CLOSED|6位移点、225响应残差与轨道稳定性尝试；A003元数据封印故障旧RUNNING不是活进程；A004补齐225/225 RHS；轨道稳定性FAILED_CLOSED成本保留，不证明核Hessian正确。|
|`Q-autonomous` 自主90,000 CPU秒队列及grid3后续|HISTORICAL_PARTIAL_SUPERSEDED|原儿茶酸/Calycosin严格重优化、数值Hessian及恢复；原儿茶酸grid3双步长完整各102/102位移、45正模，但严格核驻点未过；Calycosin旧快照28/198位移为partial，不能称完整93模式；须看新repair分支后续。|
|`Q-worker-B-baseline` Worker B原12阶段基线|COMPUTATION_COMPLETE_SCIENTIFIC_QUALIFICATION_FAILED|原儿茶酸45模式/Calycosin93模式及控制；12阶段DONE_SEALED；原儿茶酸4虚、Calycosin7虚；计算结束不具最低点资格，IR NOT_COMPUTED。|
|`Q-worker-B-v7` Worker B全负模子空间双步长诊断|DIAGNOSTIC_COMPLETE|原儿茶酸4个/Calycosin7个原虚频方向数值子空间；AGGREGATE_COMPLETE，missing/excluded为空；曲率观察为正不证明完整Hessian正定；化学峰归属false，IR NOT_COMPUTED。|
|`Q-parallel-missing` 斯皮诺素与IR并行补算|RUNNING_WITH_WAITING_DEPENDENCY|斯皮诺素严格几何与毛蕊异黄酮前驱IR已启动；斯皮诺素IR进程等待驻点/模式来源，三路均未验收完成。|
|`Q-conformers-preparation` C002/C003构象补充准备|PREPARING_NOT_STARTED|仅有job_claim，明确尚未启动；不同起始构象作为补充，最终仍须按几何去重。|
|`Q-grid7-superseded` grid7初始方案启动前替代|SUPERSEDED_BEFORE_START|未启动且active_jobs为空；被C002/C003构象方案替代，900000 CPU秒授权转移而非另增。|

完整正确结果的固定来源与哈希见[论文白名单](../../paper/current/manifest.json)。量子原始子任务保留于rounds.json，但不得把子任务数量当成完整频谱数量或分类实验数。
