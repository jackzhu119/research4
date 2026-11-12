# 多表示NN独立审计：仅准备完成

**当前状态：NOT_RUN。** 本目录是小源码、纯stdlib测试与冷启动计划收据；没有读取真实数组、预测、真值、权重或实际量子账本，没有导入NumPy、Torch、PySCF，没有执行科研训练或数值重放。

独立审计脚本：`scripts/audit_supplemental_multirepresentation_20261104.py`。
测试：`tests/test_supplemental_multirepresentation_independent_audit.py`。
原driver/leaf/tests均保持其作者管理，本任务不编辑它们；闭合的经典模型结果未改动。
`NOT_RUN.json`与首次`standardlib_review.json`保留早期准备状态；当前实现以
`NOT_RUN_FINAL.json`和`standardlib_review_resolution.json`为准，仍未实际运行数值阶段。

## 冷启动与未来实际命令

已经只执行冷计划及纯stdlib测试。冷计划不会打开NN运行目录或量子文件：

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/audit_supplemental_multirepresentation_20261104.py plan
```

以下命令**当前没有执行**。须等待Q终态、root登记的原Q授权/全局锁、NN全部六阶段健康闭合后，使用与NN相同的已安装Python执行；不安装新软件：

```bash
PYTHONDONTWRITEBYTECODE=1 .runtime/bin/python scripts/audit_supplemental_multirepresentation_20261104.py audit \
  --run reports/supplemental-multirepresentation-20261104-v1 \
  --output reports/supplemental-multirepresentation-independent-audit-20261104-v1 \
  --replay-models --cpu-seconds 180 --wall-seconds 600
```

实际审计输出必须为新目录；不覆盖NN产物或旧审计。只有`audit_stage_closed.json`为`EXITED_HEALTHY`才可认定审计完整。缺少闭合、超限或失败均保留PARTIAL/FAILED，不得仅凭部分summary声称全部通过。

## 预定核验内容

- 终态Q授权SHA、真实原Q全局锁和本次NN运行锁；六阶段闭合→worker proof→输入/源端/模型/预测锁的完整SHA链。
- 27个源模型：3个非扩展任务×3折×3种子，独立从源OOF logits复算24／48／96轮的指标与共同选轮；十类任务只继承七类epoch。
- 36个最终模型及36个预测：4任务×3臂×42／43／44种子。C和M的参数量、初始化、source IDs、class weights、epoch和batch顺序成对一致。
- 所有预测ID、唯一性、类别列及logits形状先通过检查，再读取真值。独立复算54条逐种子指标、18条均值/样本SD、全部逐类P/R/F1与完整混淆矩阵、54条表示对照纠正/损害记录。
- 对B的旧171个ID与A逐项配对；新增3个ID必须恰为B减A，不能只靠分层数量。输出9条旧171入组配对及27条新3重复预测，后者仍只有3个独立样品。

## 可选实际权重重放

只有显式`--replay-models`会在全部Q/闭合/hash/预测门禁之后延迟导入Torch。该分支独立重建同Encoder结构，核初始化与拟合state dict、参数数目、source scaler统计量、独立轴与query ID/类别顺序，加载36份`weights_only`权重并在完整query上执行eval/no_grad推断；没有优化器、梯度更新或重新拟合。

逐模型核验记录logits（atol=rtol=1e−5）、完全相同的类别预测、反序/单例推断及BN状态不变。原预测文件只存logits；概率核验是对冻结logits与重放logits分别计算稳定softmax后比较，不冒称存在额外已保存概率数组或概率已经校准。

不带该选项时，必须报告`NOT_RUN_HASH_VERIFICATION_ONLY`、实际重放0个模型；权重SHA通过不能等同实际推断通过。当前该分支同样为NOT_RUN。

实际审计作为**独立有界阶段**记录资源，不重置、不篡改NN或Q账本。默认180进程CPU秒、600秒wall、单线程/单CPU、RSS1.5GiB best effort、AS10GiB及OS CPU硬上限，另记录每个实际重放模型的CPU/时间和整个审计阶段的峰值RSS。硬终止没有健康闭合收据，不能算完成。

错误混种分类成绩已按2026-11-12用户决定撤回；论文使用main的paper/current白名单。
