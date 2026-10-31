# 酸枣仁既有源训练损失导出

仅复用冻结神经实验的480条inner source训练损失，不训练、不预测、不读取outer成绩。每条342个source ID、12／24轮；5tar和480成员SHA均实际核对。数据是训练态各batch优化CE的人数加权均值，含dropout与变化中的权重，不是验证loss或固定checkpoint的全源eval loss。

- `curves/`：480份逐曲线CSV；`epoch_loss_long.csv`合并8640条epoch记录。
- `curve_summary.csv`：480条首尾／末段下降描述；`group_summary.csv`：16组描述均值及样本SD。
- `group_epoch_summary.csv`：288条组×epoch均值／SD；两个profile各一张PNG及PDF。
- `member_provenance.json`：原tar／成员／SHA与源ID数；`validation.json`记录完整性负例与边界。

30条／组来自重叠五外折、两内折及三seed，不是30组生物独立重复；阴影为描述性SD，不是置信区间。不同profile改变类别权重及多个参数，曲线不用于跨profile排最优。没有每epoch验证曲线与中途权重，不能从末段loss下降断言延长训练一定提高泛化；后续实验仍NOT_RUN。

详见`docs/DEPLOYMENT_NEURAL_DIAGNOSTIC_20261031.md`。复现需要新空输出目录：

```bash
.runtime/bin/python scripts/export_deployment_source_curves.py --output outputs/source-curves-reproduction-new
```

有限SHA和schema检查不证明绝对零泄漏，不改变历史曝光／批次身份。没有新增真实样品或校准／分子计算。
