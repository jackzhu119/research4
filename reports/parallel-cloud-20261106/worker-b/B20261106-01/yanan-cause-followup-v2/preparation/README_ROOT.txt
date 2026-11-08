陕西延安来源异常：本轮执行与核验入口

研究目标：检验枣仁/酸枣仁误判集中的来源覆盖、表征、局部谱质量、少样本支持及跨种类预训练机制。保留原标签；现有数据不能完成植物身份鉴定。

此前350份延安记录中，仅210个master-TRAIN用于来源留出控制，187正确、23错误。其余留出枣仁只有6个，所以“23错误都来自延安”不能独立证明整份350数据标错。两套上传压缩包内XLSX字节相同，只对应同一来源一次观测。

正式状态以本目录state.json、cpu_budget.json、current-launch.json和execution下真实wait4/proof为准；准备与合成测试不等于正式训练启动。每个PID必须同时核对boot/start_ticks/UID/PGID/session及CPU/日志推进。

有限队列共220项：1数据准备、1描述诊断、25来源对留出、15编号块/随机对照、45少样本支持、12辅助预训练、30目标源预训练、30随机初始化目标模型、60迁移配对评价、1汇总。各任务包含的模型拟合数不同，不能把220称为220个独立样品或220个独立数据集。

数值输入仅固定999个master-TRAIN，CAL334/PRED339不导出数值到本轮。旧5个同条件OOF基线须核实有序IDs/标签/输入哈希/配方与封存链后复用，其他旧目标微调权重不得进入新留出折。

资源：本轮新授权最多90000累计CPU核秒；既有实际82727.759434、既有账面83728.759434保持。旧余6271.240566不滚入本轮。新轮准备命令永久保守扣1000，明确非实测；本轮实际父/worker/Git成本另加，尾耗预留单列且非实测。生产启动前余额89000，再扣尾耗及已观测启动成本。2数值线程、12GiB地址空间、8GiB RSS停止限、单个科学worker。原截止2026-11-09T08:25:17.730193Z。

每个真实退出先保存kernel wait4原始status/rusage与身份，再判进程树闭合。仅明确未进入任何fit/train的已闭合操作故障可有限重试；训练已开始的部分结果保留，不整job重做。未知成本或孤儿硬阻塞，不删除状态、marker或用新目录重获预算。已封存DONE任务复核后跳过。

科学worker闭合后分阶段回传许可小结果和源码精确字节TXT副本，NPZ/PT及原始XLSX留本地。已闭合的单纯Git失败标pending并继续科学任务，最后补齐其剩余有限上传尝试。失败成本与负结果保留，不force push，不合main。

主科学配方：preparation/scientific_protocol_ROOT.json。
实现边界：preparation/implementation_boundaries_ROOT.json；自足fit-ID/数值哈希守卫，非旧FitScope完整hook或OS文件沙箱。神经梯度入口仅接source tensors，预测先落盘再连真值评分。
已有证据与中文解释：analysis/ROOT_existing_evidence_and_next_tests_zh.json。
新汇总：science/summary-attempt*/内JSON/CSV；只有完整25对/完整5折×3种子块才汇总，缺失明确保留。同一样品的重复预测不增加独立N。

本说明不提供固定PID或完成声明；读取当前身份及实际封存结果核验。
