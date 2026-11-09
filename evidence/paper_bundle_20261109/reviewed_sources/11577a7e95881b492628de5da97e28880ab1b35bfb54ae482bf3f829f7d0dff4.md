# 多表示 NN v2：时间轴数值修复及原预算延续

v1 的真实 `prepare` 在任何拟合之前因原时间轴不满足严格均匀条件而 `FAILED_CLOSED`。已发生 child CPU 2.214606 s、监督进程观测 CPU 0.043965017 s，合计 2.258571017 s；拟合、预测与评分均为 0。`reports/supplemental-multirepresentation-20261104-v1/`、v1 driver `1b70e4...`、leaf `b27ca1...` 和原协议保留不变。

v2 只修复这一个数值输入条件，并继承原 CPU／绝对截止时间。架构、27 次 source OOF 选择、36 次最终拟合、三组种子、所有角色与标签、共同选轮规则及主要 M−C 比较均不改变。当前 **v2 真实 prepare/bind/fit/predict/score 全部 NOT_RUN**；这份修复不能当作准确率提高的证据。

## 已完成的仅轴诊断

`scripts/audit_supplemental_time_axis_20261104_v2.py` 使用纯标准库 zipfile，只解压四个原 NPZ 中各 54 KB 的 `axis.npy`；不读取 `X`、record IDs 或标签、不导入 NumPy／Torch。独立 wait4 监督限制为 1 CPU、20 child CPU 秒、60 wall 秒、256 MiB 地址空间。

真实回执：`reports/supplemental-multirepresentation-time-axis-wait4-diagnostic-20261104-v2/execution_receipt.json`。child 实测 0.09028 CPU s，监督进程观测 0.0005808009999999988 CPU s，总计 0.09086080099999999 s，EXITED/0；起止为 2026-11-04 19:42:56.510560—19:42:56.614236 UTC。最早一次纯标准库诊断没有 wait4 启动成本，不补造精确值；单独计 10 CPU s 保守 charge/reserve，明确不是测得的实际成本。

原酸枣仁、补充酸枣仁、黄芪、天麻的四条时间轴完全同指纹 `fbf09eefe1c1970452b8b33c4dd91400768e3dc1d0065a9ac6e394824fb95887`。6750 点覆盖 0—89.9866638183594，端点均匀步长 0.013333332911299364，原相邻间隔范围 0.013328552245994274—0.013336181640710265。最大原坐标到端点均匀网格的偏差为 6.582599539228795e−6，约 0.000493695 个采样间隔；最大间隔相对偏差约 0.00035855。坐标到最近 float32 值最大差仅 5.684341886080802e−14，与 float32 导出舍入一致；这不是物理时钟原本严格均匀的证明。时间单位仍未明确，不能进行 THz FFT 或物理时延归属。

## 固定的两步处理

v2 leaf 为 `scripts/draft_supplemental_multirepresentation_20261104_v2.py`，driver 为 `scripts/run_supplemental_multirepresentation_20261104_v2.py`，独立协议为 `data/research/supplemental_multirepresentation_draft_20261104_v2.json`。

1. 保留原6750点轴和全部原行，检查有限、严格递增、固定已诊断原轴 SHA。最大坐标偏差／步长、最大间隔相对偏差都必须 ≤0.001；明显不规则轴明确阻塞。
2. 根据同一冻结采集轴端点建立6750点均匀网格，逐行线性插值且无外推。均匀步长必须精确等于 `(end-start)/6749`，均匀轴固定 SHA 为 `fb8570076ac289d8362ffd8e518960fdfc7f5b549bf4ac71e5cd2c222e4b348e`。不把原抖动坐标直接宣称为均匀采样。
3. 沿用原 `resample_poly(up=512, down=3375, window=(kaiser,5), padtype=line)` 反混叠到1024点，再逐行 L2 归一化。滤波的 line 边界延拓与线性插值无外推是不同步骤，分别明确记录。

每个 time pack 的 `resampling.axis_audit` 保存原／均匀／输出轴 SHA、端点和步长、间隔 min/max/mean/median/SD、全部偏差、未知单位及无标签选择声明。source/query 必须有相同的完整轴契约。ID 显式映射、alias 同角色、source-only StandardScaler 和无标签 query pack 的门禁原样保留。

真实 bind 时额外检查：均匀输入与原固定滤波一致、抖动坐标下解析线性信号的插值、脉冲、行顺序／批量不变性、原高频抑制。**这些 NumPy／SciPy／Torch 运行门禁目前 NOT_RUN，不能用静态测试替代。**

## 不重置的执行预算

新 root authority 必须具备 `continuation_accounting.schema = research4-fusion-bounded-continuation-v2`。`prior_run` 是原 v1 `protocol.json` 的 path/SHA/bytes ref；`ledger_ref`、`registration_ref`、`original_authority_ref` 均绑定原实际文件。`diagnostic_ref` 必须指包含 wait4 成本的 **execution_receipt.json**，不能指仅含坐标统计的 axis_diagnostic.json。

成本字段为：`prior_actual_CPU=2.258571017`、`diagnostic_exact_wait4_CPU=0.09028`、`diagnostic_supervisor_CPU_observed=0.0005808009999999988`、`first_unmeasured_diagnostic_conservative_charge_CPU=10`。总 charge 12.349431818，原3600上限下向下取整剩余额度3587；root 拟给 v2 流程3400，另留180给独立审计，均不能扩大总范围。

原 v1 创建时间 19:38:58.123369 UTC、绝对 deadline 23:38:58.123369 UTC 均沿用；只有 `continuation_registered_at_utc` 记录新注册时刻。新 authority 的 `authorized_output_path` 绑定唯一新绝对目录，拒绝换目录、覆盖旧目录、未知／RUNNING 成本或先前已有拟合。成本和 deadline 在每个真实阶段重新核验；缺少剩余时间时如实 PARTIAL，不重设时钟。

当前只执行纯标准库测试及元数据 plan。由 root 在 Q controls 健康闭合后提供新 authority、注册并真实运行；本子任务不创建授权、不启动拟合或量子任务。
