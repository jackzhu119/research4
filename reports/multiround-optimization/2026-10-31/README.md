# 四轮后续优化：完整证据

先读[中文报告](../../../docs/MULTIROUND_OPTIMIZATION_REPORT.md)或打开[11张表的完整Excel](tables/四轮优化_完整结果与样品预测对照.xlsx)。四轮按先定85个主要格点及控制运行，正式追加源训练方案90.2534%／宏F1 0.878657；本轮未得到通过开发选择的小样本标签增益。FO-MAML与同预算监督继续训练逐种子分类结果一致，不能把提升归因MAML。

| 路径 | 内容 |
| --- | --- |
| `neural-base/` | 24配置、432开发源模型、全候选指标；开发元数据和压缩预测的精确字节封存、45,009条可读总预测及原文件hash；有独立归档说明 |
| `support-svm/` | 21配置、12,600开发评价、400最终评价；保留不改善和身份回退 |
| `anchored-ft/` | 24配置、2160开发评价、1500最终评价；源回放、参数约束及匹配零支持对照 |
| `meta-refinement/` | 16主配置及同任务监督控制、2880开发评价、2700最终评价；源任务清单、FO／监督原始控制和额外只读源诊断 |
| `tables/` | 11张表的Excel及11个对应CSV，包含全部参数汇总、19种最终方法及控制、49行种子结果、133行逐类结果、980随机展示（86错保留）、8379逐样品统计和2565条只读源诊断 |
| `figures/` | 源识别能力／校准作用两幅PNG、矢量PDF及可重跑绘图脚本 |
| `all_development_locks.json` | 四轮开发锁、协议和运行清单的统一放行记录，最终输出之前生成 |
| `previous_neural_source_diagnostic.json`、同名`.py` | 上轮保存源权重的只读诊断，没有新增拟合或目标评分 |
| `independent_audit.json`、同名`.py` | 全部开发与最终预测的独立数值／隔离／选择／hash／时序复核 |
| `all-tests.log` | 208项项目测试通过，11.130秒 |
| `archive_validation.json`、`files_sha256.json` | 归档字节与模型副本核验、全目录逐文件hash |

后三轮的`development_evidence.tar.gz`保存原开发目录中全部已保存指标、元数据及预测文件的**原始字节**，成员路径从`development/`开始；逐成员SHA在`original_run_files.json`中。例如：

```bash
mkdir -p outputs/restored-support-svm
tar -xzf reports/multiround-optimization/2026-10-31/support-svm/development_evidence.tar.gz -C outputs/restored-support-svm
```

模型二进制不重复放进这些tar，入选／最终权重及预处理器见[模型目录](../../../models/multiround_20261031/README.md)。非入选开发权重缓存不提交，原文件SHA及重跑代码保留。运行方自己的审查、独立审查、实际调用日志、续跑检查均单独保存；原始记录中的`outputs/...`指实际执行位置。

共1,907,766条开发预测、787,113条最终主预测，另有2565条只读源对照；并未增加独立样品数。三种子、100次支持抽样和重复开发任务不是独立新批次。所有结果均为当前单批次、已曝光数据上的回顾性匿名组研究，真实地名及独立新批次验证未补齐。

本目录仅为结果与证据，不包含重训时所需的中间NPZ；输入从当前有效工作簿用生产准备脚本生成。先行总方案见[协议](../../../docs/MULTIROUND_OPTIMIZATION_PROTOCOL.md)，四轮具体JSON均在仓库`data/research/`。归档由`scripts/archive_multiround_results.py`生成，可读表由`scripts/export_multiround_tables.py`生成；二者不训练模型。
