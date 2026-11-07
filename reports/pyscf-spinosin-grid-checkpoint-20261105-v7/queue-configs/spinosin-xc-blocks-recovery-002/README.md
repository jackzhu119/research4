# XC 恢复配置002：仅禁用准备

**PREPARATION_ONLY，未获执行授权。** `root_grant_TEMPLATE_DISABLED.json` 的两项批准标志均为 false；没有创建 enabled grant、启动 queue／Q、修改账本、读取 NPY 或执行 Git。

该配置复制原 `e169a079…` 的31个科学任务与每项20个输入引用，仅更新 orchestration 为独立审查通过的 queue v2 `3af76bd231b4a6caa1bf70fe788c2ad2be1dd9cec8b84d181a17e53efeb79dad`，使用 v2 queue/grant schema。A/core/consumer、snapshot/G、plan、实际控制 release、两线程、物理参数、216000累计CPU cap与原截止时刻均保持不变。

实际保留／更新：

- 第一项 `spinosin-C001-xc-diag-000-001` 已由旧 queue 真正健康闭合。其 `existing_seal` 从旧 state **完整原样复制5条记录**，保留路径、SHA、bytes与顺序；canonical SHA为 `8fc9036206dd610bfc2a5c36b2361e0410fc82e9afdc868a398413160ddfcf22`。准备阶段没有重新读取这些科学 payload；真正启用时仍须由 queue 原生 qualifier 复核。
- 旧失败 `v7-spinosin-C001-xc-deriv2-000-001` 及其成本、状态、日志均保留。第二项改用新 `v7-spinosin-C001-xc-deriv2-000-002`，只更新 id/phase/log/result/output 地址，其科学 argv 与20个输入引用未变。
- 其他29项未使用的 `001` 任务逐字段与旧配置相同。计划覆盖仍是20个diag块与11个deriv2块；这描述任务计划，不是31块已经计算完成。

root 已在该配置之外 DIRECT 启动同一个 deriv0 `000-002`。本准备不接管、不终止、不重复启动它，也不预先为它签发 seal。**该阶段仍 RUNNING、失败或缺少真实健康 seal 时，绝不可启用此配置。** 当前第二项的 `existing_seal` 故意为 null，因此不能自动跳过并认可该阶段。

DIRECT002真正健康闭合后，root须用实际 verifier取得其 seal，再新建例如 `config_healthy_direct_closed.json`：只在新文件中加入第二项的真实 `existing_seal`，刷新完整闭合累计成本与恢复状态，重新计算 config SHA、创建相应新 disabled／approved grant 并重新预检。不得覆盖本次已发布准备记录，也不得凭结果文件存在、PID消失或先前的 RUNNING 状态推断成功。如果DIRECT002失败，应保留成本并另作新的显式恢复决定，不能复用失败002。

禁用模板的 `opaque_process_identity_exceptions` 为 `[]`。root启用时只能绑定已批准、实际同boot核验的 dockerd/containerd精确身份；当前批准记录的位置作为提示保留，但准备未自动应用例外或扩大权限。现有可见性限制不能包装为完整进程可见性。

本次仅运行 queue v2 的默认 `execute=False` 配置预检，实际返回 `NOT_RUN_CONFIGURATION_CHECKED`，31 jobs；没有创建后台启动钩子或 queue state/monitor。基线数字只是准备时已闭合成本，未包括当时仍运行的DIRECT002最终成本，不是未来启动可用余额。实际执行始终重新读取原累计账本。恢复只按当前同boot证据准备，**没有声称跨boot恢复已经验证**。

完整配置、禁用模板、原seal／差分核查与预检证据见本目录JSON及索引。root执行前仍需新的串行调度与资源门禁，当前DIRECT任务不受本准备影响。
