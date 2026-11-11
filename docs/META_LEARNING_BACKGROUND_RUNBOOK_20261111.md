# 元学习后台启动与恢复

工作目录：`/workspace/research4`。分支：`research/meta-learning-continuous-20261111`。研究范围与限制见[实验协议](META_LEARNING_CONTINUOUS_PROTOCOL_20261111.md)。

## 启动

```bash
cd /workspace/research4
python scripts/manage_meta_learning_background.py start
```

启动器校验并复用18表特征缓存，缺失时按当前有效清单重建。随后分别脱离终端启动α、n、n+α三个supervisor，每个托管一个单线程训练worker；共同使用当前约4核等效CPU。每个表示均运行种子20261031、20261108、42的五折，共15个原E03上下文。训练入口不传`--once`；250步是保存检查点的粒度，之后继续训练。

启动器遇到已有活跃supervisor时保持原任务；存活worker遗留单实例锁时拒绝重复启动。人工停止标志不会被自动清除。后台任务不自动提交或推送Git。

## 查看状态

```bash
python scripts/manage_meta_learning_background.py status
```

状态包括supervisor/worker PID、心跳、当前上下文、训练步数和已累计训练CPU秒数。实时文件位于`reports/meta_learning_continuous_20261111/`，已加入gitignore：

- `launch_manifest.json`：实际启动命令及授权与运行边界。
- `supervisors/{alpha,n,fusion}/status.json`：原子心跳、重启次数、当前日志路径。
- `runs/{alpha,n,fusion}/frozen_protocol.json`：固定科学代码、输入、运行版本、配置和15套样品名单。
- `runs/{表示}/cells/{上下文}/checkpoint.json`和`checkpoint.previous.json`：完整权重、Adam、RNG及步数。
- `runs/{表示}/cells/{上下文}/baseline/`：匹配固定对照。
- `runs/{表示}/cells/{上下文}/snapshots/step_*/`：预先固定几何步点评估，含逐样品预测、指标和完整性封存。

源数据、代码或依赖版本变化会触发明确校验错误；这时应保留旧目录并为新协议另建输出，不能把新旧实验混在同一检查点中。不能依据Query分数选择模型或训练终点。

## 只读核验与恢复

以α为例，保留与启动一致的配置：

```bash
PYTHONPATH=src:scripts python scripts/run_meta_learning_continuous.py \
  --root /workspace/research4 \
  --output reports/meta_learning_continuous_20261111/runs/alpha \
  --feature-cache reports/meta_learning_continuous_20261111/.cache/features.npz \
  --representations alpha --outer-seeds 20261031 20261108 42 --verify
```

核验封存指标和逐样品预测的一致性，不启动新训练。完整检查点原子替换；latest损坏或缺失时，保留损坏原件及恢复记录，并从验证通过的previous继续。恢复后达到既有封存步点时保留原封存。

主机仍存活而worker意外退出时，supervisor自动重新启动并从检查点续跑；退避只延迟重启。若整机/云环境重启导致进程均已消失，恢复相同文件、代码和依赖后重新执行启动命令。若supervisor被强制杀死而worker仍在，不能重复启动训练；先检查现有PID和单实例锁，确认托管状态。

如需明确人工停止，创建相应`supervisors/{表示}/STOP`文件即可让supervisor转发终止信号，worker在完整chunk后保存退出。此机制不用于自动分数、时长、预算或平台规避。

## 资源与生命周期

启动时另保存了一份[45个上下文的初始训练检查点](../data/research/meta_learning_20261111/launch_backup/manifest.json)，共约1.8MB压缩数据，含优化器和RNG，可从Git恢复该时刻的进度。它不是后续所有更新的自动备份。原主机目录丢失时，在新目录恢复：

```bash
python scripts/restore_meta_learning_launch_backup.py \
  --output /workspace/research4/reports/meta_learning_restored_20261111
python scripts/manage_meta_learning_background.py start \
  --output /workspace/research4/reports/meta_learning_restored_20261111
```

恢复程序先核对全部压缩及解压内容SHA-256，拒绝覆盖已有目录；新启动器重建特征缓存后继续已恢复的训练状态。实时进程和步数以`manage_meta_learning_background.py status`为准，[启动交接记录](../data/research/meta_learning_20261111/launch_receipt.json)是当时的只读快照。

900000 CPU小时是用户授权记录，当前没有平台分配或预留这一额度。没有本任务自设的分数、耐心、CPU预算、运行时长或重试次数停训阈值。程序不抵抗平台终止，不能保证云环境暂停、销毁或整机故障后的不间断执行。进程脱离终端不等于获得跨聊天、跨主机的持久调度服务。
