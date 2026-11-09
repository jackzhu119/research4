# 四轮序贯嵌套搜索：完整证据

先读[中文报告](../../../docs/STAGED_NESTED_OPTIMIZATION_REPORT.md)，或打开[21张表的完整Excel](tables/四轮嵌套CV_完整结果与855样品OOF对照.xlsx)。本轮为855份、旧5外折／2内折，和过去171份固定预测实验分开。

正式R0—R4的OOF准确率为90.4094%／92.5146%／96.0234%／95.4386%／94.9708%；对应宏F1为0.8092／0.8277／0.9197／0.8877／0.8820。结果不单调提高；R2是预设路线比较中的最好观察结果，未据此回填或重选后续外折策略。

| 路径 | 证据 |
| --- | --- |
| `core/` | R0—R3及固定锚／7种预处理诊断，1470内层case／502740条预测、60外层角色／10260条OOF；历史基线和恢复检查 |
| `combinations/` | R4的普通MC、预算MC、稳定MC／CARS／联合和CNN双头，70内层case／23940条预测、40外层角色／6840条OOF |
| `tables/` | 21张表及对应CSV，包含20方法角色、1540内层记录、25个独立逐折选参记录、全部855编号OOF、1345条错误、逐类P／R／F1、gamma别名、类别权重、频点／稳定性、匹配预算及CNN训练记录 |
| `figures/` | 三幅PNG、矢量PDF、直接读取冻结结果的绘图脚本与输入hash |
| `protocols/` | 总计划和两个实际运行前冻结协议 |
| `outer_evaluation_release.json` | 五个外折全部R0—R4内层选择锁定后，统一外层放行的真实UTC与SHA |
| `evaluation_invocations.jsonl` | 外部driver记录实际evaluation argv、开始／完成UTC和返回码 |
| `independent_selection_audit.json`、同名`.py` | 放行前按原字节封存的完整selection审计及执行脚本 |
| `independent_audit.json`、同名`.py` | 完整内外预测、模型、源统计、系数前缀、继承、稳定性、CNN配对及实际时序独立复核；新增拟合0 |
| `all-tests.log` | 264项项目测试通过，13.536秒 |
| `cnn_full_source_audit.json`、同名`.py` | 15个新source-only CNN及两头共享的只读补充核验 |
| `archive_validation.json`、`delivery_validation.json`、`files_sha256.json` | 字节副本、实际模型及协议／锁／输入完整性索引 |

核心使用5个`outer_XX_selection_evidence.tar.gz`保存原内层指标、预测和source统计；24MB原指标JSON和省略矩阵清单采用gzip，并保留原SHA及恢复索引。`coefficient_summaries.tar.gz`保存160份去重的精确mean／std／floor摘要。190份逐次MC系数矩阵留云端缓存，GitHub保存其原SHA／形状／有序源身份和重建说明；实际独立审计时核过原矩阵，不声称报告包含所有原系数。具体见`core/README.md`及`omitted_mc_coefficient_matrices.json.gz`。

组合开发证据同样按原字节打包，源重采样、选择频率、Jaccard、计算预算和CNN共用编码器均可追溯。原二进制内层缓存不重复提交；外层模型见[模型目录](../../../models/staged_nested_20261031/README.md)。

所有raw预测按保存的`y_predicted`评分，SVC保持标准投票；不能用OVR分数argmax替换。R0逐855标签复现旧773份正确，OVRargmax会变为772份；R4普通MC控制逐分数／标签与R3相同。原始错误、诊断和不利折没有筛除。

独立样品数仍为用户声明的855份。20条方法角色、折间重叠训练、源重抽样、3000个CNN训练epoch均不增加独立样品或批次。每产地只有一批，ZY02只有5份，当前是历史数据上的回顾性内部验证；真实跨批次小样本校准未运行。

归档由`scripts/archive_staged_nested_results.py`复制经过验证的紧凑交付；可读表由`scripts/export_staged_nested_tables.py`生成；外部UTC记录由`scripts/run_recorded_research_command.py`生成。它们不修改冻结研究算法。原记录里的`outputs/...`指实际运行目录。
