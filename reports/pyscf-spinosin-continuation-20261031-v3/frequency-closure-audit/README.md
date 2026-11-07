# 斯皮诺素频率阶段：闭合结果与失败证据

审计状态：**DONE（只读证据核查）**。量子阶段状态：**PARTIAL；尚未得到斯皮诺素振动频率**。生成时间：2026-11-01T16:20:31.280743+00:00。本附录只新增文件，不更改冻结源、原账本、结果和已有报告。

本次依据root绑定的最终账本SHA `19fbbc80b5e839dc645f10a467bb54f32fdebae14309a95d824520cd885d3d2c` 与pipeline state SHA `a2346f01bb7997c0f5fdf84e23ee603cea8f1be8e484007791d76217bd23eca5`，确认所有账本阶段已经结束，控制器状态为PARTIAL并有结束时间。

## 实际退出及CPU用量

频率阶段 `spinosin-native4-frequency-001` 在2026-11-01T16:12:14.745534+00:00结束，父进程exit code **2**。worker和缓存manifest记录实际捕获 **signal 24（SIGXCPU）**。原始文本为：

> Worker interrupted by resource supervisor signal 24

**这段泛用handler文本不能证明信号由supervisor发送。** 冻结源码1058—1066行对SIGTERM／SIGXCPU统一使用这句话，没有保存siginfo／发送者轨迹。能够确认的是捕获了24号信号并经BudgetStop路径保存PARTIAL；确切发送者、信号递送过程及CPU超额根因均**未直接记录／未证实**。不能以措辞推断墙钟监控、内存保护或某个人主动终止。

| 资源核查项 | CPU秒 |
|---|---:|
| 启动配置soft | 86280 |
| 启动配置hard／阶段额度 | 86400 |
| 父进程wait4实际user＋system | 89154.115469 |
| worker process CPU计时 | 89153.496542459 |
| wait4超出hard | 2754.115469 |
| wait4超出soft | 2874.115469 |
| wait4与worker计时数值差 | 0.618926541 |
| 全部已闭合V3阶段实测合计 | 111023.488913 |
| V3新有限预算配置 | 172800 |
| 按完整实测扣除后的余额 | 61776.511087 |

该阶段wait4超出配置hard约 **3.187633645%**；严格86400秒实测上界未实现。退出失败没有把真实消耗截断为86400，原账本完整计入实际费用。四阶段V3合计低于其172800秒配置预算；余额不是已经完成的科研结果，也不构成自动启动新计算的声明。父进程标志 `wall_terminated`、`hard_killed`、`memory_terminated`、`monitor_failed` 均为false。它们描述父监控记录，不能反推所有潜在内存压力不存在。

实际墙钟22827.819999秒，配置43200秒。24 GiB AS仍是虚拟地址空间，不是物理RAM；RSS轮询属于best-effort而非hard RSS。优化阶段的独立CPU披露见[先前附录](../resource-limit-disclosure/README.md)，该附录只讨论当时已结束的优化前缀，不能据此判断后来频率阶段合规。

## 已完成与未完成的计算边界

本频率尝试先完成同方法fresh SCF和梯度，元数据记录能量-2210.445521964444 Eh、原子梯度RMS 6.264040860295175e-05、最大值0.000203903454559022 Eh/Bohr，阈值检查通过。本审计没有重新读取梯度数组或计算导数；这些是已保存元数据，不是新增复算结果，也不能验证正频极小点。

native driver记录 **155次进入、155次返回**。这是底层调用计数，不等于完整partial Hessian、Hessian核或振动计算返回。最终manifest为`PARTIAL`，`partial_hessian=null`、`H1AO_atoms=[]`、`response_blocks=[]`；本次闭合缓存目录为空。标准捕获器只有原函数正常返回并校验、保存后才建立完成记录，因此**没有可恢复的完整partial Hessian、H1／响应块或全Hessian**。预计中间量尺寸、geometry文件或正常模式准备文件都不能作为已计算振型。

最终raw Hessian、analysis Hessian、frequencies.csv、normal_modes.npy均不存在。SCF checkpoint实际保存，SHA为`d647b182bf16269f1293ac629394e85a1216ee684da517a01188cef462cd07fc`，大小5704080字节；本审计只做字节SHA核验，没有打开HDF5或提取轨道。它最多可供后续合规SCF初猜使用，不能宣称从已有Hessian／振动继续；任何后续工作仍需严格几何、方法、SCF及梯度门槛。跨进程Hessian加载仍`NOT_IMPLEMENTED_SAVE_ONLY`，没有伪造恢复命令。

## 频率表与理论解释的限制

[原分析汇总](../final-analysis/summary.json)的斯皮诺素已计算／已验证频率记录均为0。两份斯皮诺素CSV为空，仅表示**未计算成功**，绝不表示该分子没有低频振动、没有吸收或某成分不存在。75原子非线性分子的219个预期模式不是实际得到219个模式。主频率表中的3条真实记录均来自水；水及乙醇21模式仅作数值验证，不能填入斯皮诺素机制结果。

20个实验候选对应表均保留空理论字段，状态为`BLOCKED_NO_VALIDATED_SPINOSIN_MODES_IN_RANGE`；没有可用斯皮诺素理论峰进行化学归属。原[分析README](../final-analysis/README.md)已明确“未计算保持空表，不推断零吸收”，不存在把空CSV解释为无低频峰的结论。IR始终`NOT_COMPUTED`；没有修改96.0234%的分类结果、光谱或标签。

完整机器可读审计见[audit.json](audit.json)，来源及冻结源码SHA见[source_sha256.json](source_sha256.json)，本目录文件索引见[index.json](index.json)。索引不包含自身SHA，由上级交付绑定。没有启动量子计算、积分、信号或Git操作；所有科学来源仅只读。
