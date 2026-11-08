# 持久量子计算队列与恢复（仅准备，NOT_RUN）

工具：`scripts/research4_quantum_queue_20261105.py`。
配置和授权模板位于 `reports/research4-quantum-queue-preparation-20261105-v1/`。
模板明确禁用，包含待替换字段；没有启动真实队列、量子任务、后台服务或平台推送任务。
root可以在已有用户授权内生成具体配置并授权，不需要另向用户索取安装或运行权限。

队列只执行已经物化、给定参数和SHA的有限命令。例如，在真实XC计划完成后，root将每个待算grid块列为独立job；在真实H1完成后，再为response块生成另一份配置。工具不推测未来SHA、不生成科学component index、不改方法、不搜索参数，也不运行独立数值分析器。consumer及analyzer必须先最终冻结；`*_SHA = "NOT_FROZEN"` 源码会被拒绝。

## 原计算账本保持不变

每个实际job均启动未修改的 `b985d6…` guard，使用同一个 `reports/pyscf-spinosin-staged-20261105-v6b/cpu_budget.json`。原累计216000 CPU秒上限、原期限、已有失败成本、原资源监督和单一continuation claim由原guard再次核验；队列没有增加预算或重新开始72小时窗口的入口。

仅当实际guard阶段健康闭合、退出码为0，且实际source、proof、完整argv、日志及结果/payload哈希均验证通过，job才记录为 `HEALTHY_CLOSED`。文件存在、PID消失、guardian返回0均不足以证明科研阶段完成。完整Hessian已计算与正频局部极小值仍由后续独立分析区分。

每次成功后封存result、input、proof、日志、manifest及payload字节SHA；以后跳过前必须匹配原seal，不能悄悄重签。首次纳入队列的历史健康job需由root配置提供 `existing_seal`。`FAILED`、`RUNNING`、孤儿或任何blocked invocation都会停止：先由当前agent检查并完成真实forensic闭合，再创建明确的新root配置/授权及fresh phase/output，仅重算中断块。其余已健康块仍可按原seal复核后跳过。不得改旧状态为DONE来绕过门禁。

## 队列自身的资源与串行安排

单个授权最多128个job、3次invocation；每次launcher自身CPU上限最高600秒、墙钟最高86400秒，并受原量子期限限制。`ITIMER_PROF`/`ITIMER_REAL`仅约束launcher本进程，子进程不继承；不降低可能误伤guardian的CPU/AS/RSS/stack硬限制。首次CPU/墙钟超时引发停止，另留2秒清理窗口，清理卡住则按默认终止并保留未闭合状态。launcher自身RSS为只读轮询的best-effort门禁，不冒充内核硬RSS限制。

队列使用独立锁，原guard继续使用原量子全局/lineage锁。root必须保证Git、ML和大IO串行；进程扫描仅为补充的best-effort检查，不能宣称覆盖不可见或任意命名的外部进程。每次新job前重新检查source、config、grant和已绑定inputs。状态与monitor限定在独立queue目录，不能追加到原账本、源码或科研文件。

## 进度与恢复命令

运行时固定每1800秒写一条进度到 `monitor.jsonl` 并打印console，另记录关键阶段节点。记录含下一阶段、真实已闭合量子CPU、开放阶段列表和launcher自身CPU/RSS。开放阶段未结算的CPU不会被编造成最终成本。当前agent从console读取后在聊天中每30分钟汇报；没有创建平台后台自动推送。

root物化并封存具体config/grant后，可使用以下命令模板（所有占位符均须替换为真实文件及其SHA）：

```bash
.runtime/bin/python scripts/research4_quantum_queue_20261105.py \
  --config reports/APPROVED_QUEUE/config.json --config-sha256 ACTUAL_CONFIG_SHA256 \
  --grant reports/APPROVED_QUEUE/root_grant.json --grant-sha256 ACTUAL_GRANT_SHA256 \
  --attempt root-invocation-001 --execute
```

每次真实运行还会在 `state.json` 保存完整、可直接交给exec的 `resume_argv`，其中包含实际config/grant SHA及新invocation编号。没有 `--execute` 时仅检查静态配置，不启动guard。未闭合/blocked队列不能用原命令自动重试，须由当前agent先解决原量子账本的forensic问题，再生成新的明确配置；旧状态、日志、成本和seal均保留。

平台整体重启时，没有外部start hook就不能保证自启动。环境恢复后必须由当前agent重新检查账本和检查点，再调用已保存的命令或经复核的新attempt；本工具没有声称可在环境停止期间继续运算或推送消息。
