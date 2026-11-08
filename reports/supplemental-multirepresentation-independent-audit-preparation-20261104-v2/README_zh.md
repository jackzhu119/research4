# NN v2独立审计准备：NOT_RUN

新脚本为 `scripts/audit_supplemental_multirepresentation_20261104_v2.py`，测试为
`tests/test_supplemental_multirepresentation_independent_audit_v2.py`。原v1审计脚本
SHA `d6bf78990ab226c9742d5ec3b6cd97c9874210c6ea92b35c53c49c62828c6971`及其历史收据保持不变。

本目录只证明源码、纯stdlib测试和静态审核已完成。**没有执行实际审计、
读取真实数组／标签／权重、导入数值库、重新拟合或运行量子计算。** NN v2由root
另行运行；其进度不能从本目录的NOT_RUN状态推断。审计须等待root确认全部六阶段健康闭合并释放执行。

## v2延续与时间坐标修复

NN v1在prepare严格均匀轴门禁失败，实际0次模型拟合；其失败、资源和原期限均保留。
root报告需携带的闭合成本为：v1实际2.258571017 CPU秒、受监督轴诊断实际
0.090860801秒，以及首次未完整测量启动的10秒保守收费，共charged
12.349431818秒。**10秒是保守收费，不能称为实际测量CPU。** 实际审计执行时会从五个
小型闭合收据的SHA和逐阶段数据独立重算这些金额，不只相信汇总字段。

v2只增加固定的6750原坐标→6750端点均匀坐标的无外推线性插值，再执行原
512／3375、Kaiser5、line边界polyphase至1024及L2归一化。审计独立检查原轴、
均匀轴、输出轴SHA、严格端点步长、坐标／间距偏差≤0.001步、未知时间单位、
source/query同一合同；不将原舍入坐标伪称为均匀采样，不由标签或预测表现选择修复。

固定原轴SHA：`fbf09eefe1c1970452b8b33c4dd91400768e3dc1d0065a9ac6e394824fb95887`。
固定均匀轴SHA：`fb8570076ac289d8362ffd8e518960fdfc7f5b549bf4ac71e5cd2c222e4b348e`。
架构、角色、种子42／43／44、轮次24／48／96及27源模型＋36最终模型的比较保持不变。

## 累计资源不能重置

原总CPU上限3600秒、原绝对期限2026-11-04T23:38:58.123369Z不变。v2训练上限
3400秒，审计预留最多180秒；charged prior＋v2逐阶段child/parent实际CPU之和＋
审计进程实际CPU必须≤3600。审计还独立核验逐阶段求和等于两个grand total，
低写汇总、未知成本或期限延长均拒绝。

审计默认180 CPU秒、600秒wall，wall进一步受原绝对期限限制，CPU/AS有OS上限，
RSS1.5GiB best effort、单CPU/单线程。健康NN链核验后，在该NN输出下新增一次性
`independent_audit_claim_v2.json`；其后失败或未知状态不能通过换审计输出目录自动重置预算。
原NN/v1账本不改写。成功时另写审计闭合及scope收据；硬杀缺少健康闭合仍为PARTIAL。

## root之后的实际命令

以下命令当前**未执行**，须使用NN相同的已安装Python且等待root六阶段健康放行：

```bash
PYTHONDONTWRITEBYTECODE=1 .runtime/bin/python scripts/audit_supplemental_multirepresentation_20261104_v2.py audit \
  --run reports/supplemental-multirepresentation-20261104-v2 \
  --output reports/supplemental-multirepresentation-independent-audit-20261104-v2 \
  --replay-models --cpu-seconds 180 --wall-seconds 600
```

仍会核27个源OOF模型的完整选轮、36预测／模型及C/M同参数配对、54逐种子指标、
全部逐类和混淆计数、18条三种子均值／样本SD、54条表示纠错／损害，以及同旧171
样品配对和新增3样品的27条重复预测。类别列／ID先检查，真值最后读取。

只有显式`--replay-models`才在终态Q授权、原Q全局锁、全部健康stage/hash门禁后
延迟导入Torch并真实重放36份权重；检查source scaler、初始化／拟合state dict、
原生标签、logits和由logits派生的softmax（atol=rtol=1e−5）、eval／BN状态不变。
没有重新拟合。不带该选项必须标记`NOT_RUN_HASH_VERIFICATION_ONLY`，不能把权重SHA
核验写成实际模型重放。

当前源码静态审核通过，61项合并纯stdlib测试通过；这些软件fixture不是科研样品或真实数值验证。
本次不改旧经典实验、v1失败产物、NN正在运行的数据或任何量子文件，不执行Git。
