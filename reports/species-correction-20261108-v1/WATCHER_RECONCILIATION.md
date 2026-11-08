# 纠错证据分析等待器与恢复边界

等待器源码为`scripts/species_afterq_watcher_20261108.py`。首次交付仅编写、
代码审查及27项stdlib/fake测试；这不表示实际分析已经启动或完成。实际状态
由root冻结配置所指定的`workroot/state.json`和`detached-launch.json`说明。

等待器不训练、不推断、不运行量子计算、不运行Git，也不发停止信号。它只在
既有72小时截止内等待当前恢复队列的真实父进程结束。即使父PID消失，也必须
额外验证完整队列终态、全部作业的真实闭合证据、原累计账本、源代码和输入
哈希，以及没有活动科学/Git进程，才能通过原守护器启动一次纠错证据分析。
旧队列中已认证闭合的科学失败不阻止独立证据分析，也不因此被改称成功。

## 状态含义

- `WAITING_FOR_QC_PARENT`：仅等待，尚未运行纠错分析。
- `RUNNING_GUARDED_EVIDENCE_JOB`：原guard已经实际启动分析，检查其PID、出生
  标识、UID、boot、实际CPU、账本和日志，不重复启动。
- `DONE_LOCAL_EVIDENCE_NOT_AUTO_PUBLISHED`：真实作业闭合且结果状态符合冻结
  成功列表；结果保留本地，不自动推送公开仓库。
- `PARTIAL_OR_FAILED_CLOSED_PRESERVED`：真实闭合，但计算或结果门禁未通过。
- `BLOCKED_REQUIRES_ROOT_RECONCILIATION`：不能确认安全推进；不是成功或退款。

## 中断后的处理

宿主机关闭不会因detachment而保持运行。若boot改变、PID复用、账本仍有
未知开放阶段或guard成本不明，不得直接重复原`--detach`命令，更不能重建
state、清除claim、改预算或覆盖旧输出。先由root核实原进程和所有后代的
真实闭合、实际失败CPU成本及现存结果，然后按原截止决定新独立attempt。
已有成功证据不重算；已有相同分析phase会阻止自动再次运行。

默认CLI只做只读预检。只有root审核过分析worker、完整精确配置及输入哈希
后，才以该配置的`--execute --detach`启动等待器。O_EXCL启动claim、不可覆盖
的启动收据与独立flock共同防止同机重复等待和执行。

本任务使用原72小时账本的一项最多2000 CPU秒、1200墙钟秒的串行分析阶段，
不初始化、复制或重新发放另一个云端90,000 CPU秒搜索的额度；不设置周期
聊天任务。原截止为北京时间2026年11月10日10:50:37。
