# 补充光谱基准：独立源端审计

结论：**PASS_SOURCE_ONLY_INDEPENDENT_AUDIT**。本次审计只读取源数据、源端OOF预测、拟合追踪和锁定协议；未打开目标support/query数据包、目标真值或目标预测，未拟合任何新模型，未载入或重放模型权重，未调用量子化学软件，未执行Git操作。

审计脚本独立采用NumPy复算指标、混淆矩阵、蒙特卡洛抽样及门禁决策，不调用原训练或指标函数。

- 锁定协议SHA-256：`981afb628b4bc7f6ac910d0ff2b9d8d9b74fa0117fb284e61ab6e813429ab02a`。
- 51个源码文件与协议内SHA一致；聚合源码标识为`36439ad90dce759903e28cb8bd3e28a613ca7b80dc657925dee149ba1c5f4b8e`。
- 633个实际拟合上下文完整闭合：567个源内CV、48个全source最终模型、18个源内校准伪任务。每次拟合的actual/allowed IDs均与该训练分区逐项一致。
- 45,816条追踪事件无未闭合、失败或越界记录。138次未缓存MC-UVE排名、3,570次真实MC子抽样及18,150次底层PLS拟合的ID与独立复建的分层PCG64采样一致。
- 432次缓存复用的实际数据、标签、ID、频率轴、fold、seed、预处理、MC参数及源码标识均绑定当前训练上下文，未见跨折复用。
- 三个可选参任务各21候选×3种子，189组源内OOF指标、混淆矩阵及63个候选均值复算一致；12个family选择复算一致。最大指标差为0。
- 扩展十类任务只继承原七类任务选定的配置，各模型均使用扩展任务自身516个source样品重新拟合；48个最终模型覆盖全部4任务×4家族×3种子。
- 504条源内校准评估、204次实际adapter拟合均只涉及当前伪support样品。三组任务均采用selection 3个episode、gate 3个episode，重算门禁后全部回退至不修正。十类任务因新类别仅各1个source样品，保守校准按协议为NOT_RUN并全局不修正。
- 合计通过219,105项断言；审计本身耗时约4.69秒。原source进程记录149.7868 CPU秒、峰值RSS约172.17MiB，未触及5880 CPU秒源端协作式停止阈值。

错误混种分类成绩已按2026-11-12用户决定撤回；论文使用main的paper/current白名单。

| 任务 | 选定MC-UVE＋SVM | Accuracy | Macro-F1 | Balanced Accuracy |
| --- | --- | ---: | ---: | ---: |
| 酸枣仁七类 | MC20、LV2、10%、SG一阶导数＋SNV、C10、gamma scale | 91.0981% | 0.823583 | 0.800794 |
| 黄芪四类 | MC20、LV2、20%、SNV、C100、gamma scale | 83.2642% | 0.832875 | 0.832741 |
| 天麻四类 | MC20、LV3、10%、SNV、C100、gamma scale | 97.1074% | 0.971029 | 0.971243 |

局限：源内selection/gate episode来自同一已存在的source人群，并非独立采集批次。追踪与哈希一致性只证明已记录接口和数据的一致性，不能建立生物学独立性，亦不是操作系统级绝对隔离。本轮没有重新计算所有PLS的浮点矩阵，也没有重放权重；源校准门禁从真实已保存的源端评估行复算，未伪造不存在的伪query逐样品预测。

主要证据：

- `source_audit.json`：机器可读审计结果，SHA-256为`54e423d19ed6bad4e8e8ac7db042cd2cc8502dee26d11d9acdef2db68688dfe6`。
- `source_fit_inventory.json`：633个实际拟合上下文、追踪及元数据哈希。
- `source_oof_metrics_recomputed.json`：189组独立复算源内OOF指标。
- `read_file_hashes.json`：只读输入文件清单及SHA-256。
- `artifact_index.json`：上述结果及审计脚本哈希。

复现命令（脚本拒绝覆盖自身既有审计结果，需将整份脚本复制至新的空审计目录后运行）：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python /path/to/new-empty-audit-directory/audit_source.py
```
