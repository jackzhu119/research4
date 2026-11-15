# 已启动的后台实验

方案见 `PRE_REGISTERED_DESIGN_zh.md`，机器配置见 `protocol.json`，真实启动验收见 `launch_verification.json`。

主进程与守护进程已脱离当前聊天的执行会话。四个单线程worker按冻结队列训练；9万CPU秒包含安全预留，累计账本为 `queue.sqlite3`。不以准确率未达标作为提前停止条件，也不根据外测结果追加有利配置。当前真实进度以 `status.json` 为准，不以本文件中的静态描述为准。

- `jobs/<job_id>/selection.json`：外层预测之前冻结的参数。
- `jobs/<job_id>/inner_scores/`：完成的内层折检查点。
- `jobs/<job_id>/predictions.csv`、`result.json`：逐样本预测及完成结果。
- `jobs/<job_id>/fitted_artifacts.npz`：所选变量、模型及缩放参数。
- `attempts/<attempt_id>/stdout.log`：各次运行日志；失败与重试均记CPU。
- `reports/`：定期汇总，只统计完整、有效的外层重复；未完成结果保持待定。

查看进度：

```bash
python /workspace/applied_spectroscopy_revision/background_study_20261110/start_background.py --root /workspace/applied_spectroscopy_revision/background_study_20261110 --status
```

需要主动停止时：

```bash
python /workspace/applied_spectroscopy_revision/background_study_20261110/start_background.py --root /workspace/applied_spectroscopy_revision/background_study_20261110 --stop
```

本地守护每15秒检查领导进程，应用中另建了每小时巡检（最多36次）。二者沿用同一账本恢复，不重置CPU预算。应用巡检继承当前环境选择，但返回结果未提供连接明细；云平台回收/休眠及文件丢失仍可能影响续跑，不能视为绝不中断的保证。

`selftest/`、`smoke_runtime/`、`smoke_recovery/`仅为隔离的启动验收，不进入科学结果。化学先验训练目前缺少真实配对输入，未启动，也不能从结构约束实验直接推导化学机制。
