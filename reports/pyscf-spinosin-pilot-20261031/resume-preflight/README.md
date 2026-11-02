# 检查点续算准备工具：实现与纯数据测试

状态：**DONE（工具与测试）；真实检查点恢复／续算命令准备NOT_RUN，等待最终计算阶段退出和root放行。没有执行额外SCF、梯度、优化或频率。**

`scripts/prepare_pyscf_pilot_resume_20261031.py`会在原CPU账本锁上取得只读共享锁，拒绝仍运行或账目不一致的阶段。它不创建新账本、不修改或扩大3300秒量子CPU／3600秒总CPU上限，不执行准备出来的命令。

后续仅在所有阶段结束后，从最后一个已记录的Spinosin优化阶段读取`calculation.chk`中的完整精度Mole；先核对最终worker结果或root显式绑定的检查点SHA。现存检查点损坏／错配时拒绝，不静默换几何。检查点不存在时，才允许从SCF已收敛且同时保存XYZ／75×3有限梯度的最后完整步骤恢复；明确保留原12位小数精度。

恢复后重新核对75原子组成、顺序、键长、总电荷0、320电子、零自由基图、十个立体中心及完整InChIKey；保存Angstrom／Bohr两份.npy及新的XYZ／来源SHA。检查点的分子在SCF开始前写入，可能是下一试探几何，不能冒充最后完成的梯度或驻点；未来继续运行必须重新SCF和梯度。

若预算允许，准备的命令使用同一个既有账本、唯一新phase名、fresh输出和日志、原检查点、PBE／def2-SVP／grid1／显式无D3、最多35步。CPU请求取`min(请求值, floor(原剩余秒数)−2)`，少于5秒则BLOCKED且不生成执行脚本。预算不足时，新的独立预算批次需要另行授权，工具不会自动初始化。命令执行前还会核对父文件、账本、恢复坐标、脚本和方法worker的字节摘要。

最终root release应包含：

- `allow_resume_preparation: true`及`created_at_utc`。
- 最终`cpu_budget_sha256`、`source_phase`及`source_result_sha256`。
- 若worker最终结果缺失，必须显式`allow_missing_worker_result: true`；如果检查点hash也不能由最终结果提供，则须`source_checkpoint_sha256`。

最终CLI形式为`prepare --run-root ... --source-phase <最终已结束phase> --output <新continuation目录> --release <root放行JSON> --resume-phase spinosin-optimize-resume-001 --requested-cpu-seconds 900 --maxsteps 35`。这里只描述接口，不是声称已经执行的继续计算命令。

16项实际测试通过，见[tests_checkpoint_bound.log](tests_checkpoint_bound.log)。覆盖原预算剩余额、运行锁、账目／授权／检查点／坐标／shell篡改、完整精度、缺检查点回退、错误立体与原子顺序、旧输出保护及无重置命令。测试包含一个实际PySCF Mole的HDF序列化／读取夹具，没有SCF或其他量子计算。较早14／15项成功测试及未执行的旧草稿源码快照保留，用于追溯新增检查点绑定防护。
