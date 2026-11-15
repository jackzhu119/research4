# 斯皮诺素与IR后台补位（2026-11-12）

后续更新：完整网格响应出现系统OOM，原进程已保全电子态后人工停止。当前恢复入口移至[网格响应分块内存修复](SPINOSIN_IR_MEMORY_REPAIR_START_HERE.md)，旧代码、输入和尝试日志保持不变；不要用本页旧启动器重启旧路径。

本轮核对云端最新回传后，只补充两个新的斯皮诺素建模构象C003和C008。其他主机已在处理原儿茶酸IR和毛蕊异黄酮数值频率，原C001起点也已在其串行队列中；本轮使用不同的、已经过立体身份核验的初始构象。

启动已核实：两份原生PySCF后台worker均已进入严格几何优化，CPU持续推进，随后自动接完整频率和IR阶段。32项测试、双构象生产维度恢复模拟及真实水分子整条计算链均已通过。此处是交接快照，实时状态使用管理器查看。

- [科学设计、任务范围与恢复规则](docs/SPINOSIN_IR_GAP_FILL_PROTOCOL_20261112.md)
- [运行和状态读取](docs/SPINOSIN_IR_GAP_FILL_RUNBOOK_20261112.md)
- [输入及来源](data/research/spinosin_ir_gap_fill_20261112/inputs/cases.json)
- [云端回传核对记录](data/research/spinosin_ir_gap_fill_20261112/remote_audit/publication.json)
- [IR主机回传核对记录](data/research/spinosin_ir_gap_fill_20261112/remote_audit/ir.json)
- [科学源码固定版本及SHA](data/research/spinosin_ir_gap_fill_20261112/vendor_provenance.json)
- [原生梯度与偶极工程验收](data/research/spinosin_ir_gap_fill_20261112/native_backend_gradient_smoke.json)
- [32项测试和真实水整链验收](data/research/spinosin_ir_gap_fill_20261112/engineering_validation.json)
- [启动、进程和资源凭证](data/research/spinosin_ir_gap_fill_20261112/launch_receipt.json)

两个构象各用一份独立worker，每路2线程。严格优化及电子稳定性通过后，自动进行900个双步长Cartesian梯度／总偶极点，再进行完整219模式和独立IR验证。两个构象的中心、密度、梯度、模式和缓存始终分开。

追加900,000 CPU秒（250 CPU小时）已登记。后台不使用CPU额度、运行时长、分数、失败次数或全局几何轮数提前退出。结果资格仍需真实核验，未通过不能写成最低点或IR完成；自然完成后待命，数值方案不足时明确等待精化。实际状态以管理器及最近科学进度为准，不能只看进程存活。

运行产物保存在`reports/spinosin_ir_gap_fill_20261112/`；原有分类结果和其他主机作业保持独立。后台与聊天会话分离，主机生命周期仍可能影响进程。
