# 方案C：酸枣仁主线的审计、对照与论文证据

2026-10-31，北京时间。主线为酸枣仁THz识别及优化，外部植物和校准为辅助。当前R2仍是已评价路线中的最强开发候选：821／855＝96.0234%，Macro-F1 0.9197，Balanced Accuracy 89.9184%。全部855及历史测试此前已曝光，且每类只有一批，不是独立批次确认。

先读[本次完整审计报告](../../../docs/RESEARCH4_EXPERIMENT_AUDIT.md)，再看[主计划](../../../docs/RESEARCH4_PAPER_C_MASTER_PLAN.md)与[论文证据提纲](../../../docs/RESEARCH4_PAPER_OUTLINE.md)。

## 实际完成的研究

| 证据 | 内容与入口 |
| --- | --- |
| 仓库状态 | [全部分支与起始提交](repository-state/repository_snapshot.json)、[12工作流台账](repository-state/historical_experiment_ledger.csv)、[原资料清单](repository-state/committed_provenance_inventory.json)、[运行环境](repository-state/environment_snapshot.json) |
| 旧结果独立审计 | [历史审计](history-audit/README.md)：五阶段773／791／821／816／812，1540内层、17100 OOF、25主角色模型重放 |
| 保存历史最佳候选 | [R2索引包与预测命令](best-candidate/README.md)：五折原模型引用、特征／源统计／SHA及855实际重放；未来集成无独立准确率 |
| 统一分类器及消融 | [新公平对照](fair-benchmark/README.md)：1140内层、45完整外层管线、7695预测，全谱SVM／LDA／MLP及六项条件控制 |
| 精确原强MLP | [强MLP](strong-mlp/README.md)：64–32网络200轮＋100次源更新，三seed全保留，30内层／15外层源上下文及成对权重 |
| 外部资格 | [黄芪／人参／公开THz审查](external-data/README.md)：真实读取及合法访问失败记录、数值接口、独立性／单位／标签阻塞；没有虚构外部成绩 |
| 保守校准 | [单独171协议](calibration/README.md)：60源开发、14400候选评价、1200最终评价／205200预测；保守退回identity，传统λ1降准 |
| 外测前审查 | [前两协议审查](development-review/README.md)、[强MLP审查](development-review/STRONG_MLP_REVIEW.md)，全部在新外层之前 |
| 外测与独立复核 | [统一真实放行](evaluation_release.json)、[调用起止](command_invocations.jsonl)、[最终独立审计](final-audit/README.md)：218025最终预测、45管线＋30强MLP权重＋2原171源重放 |
| 论文可用表图 | [14张表Excel与6幅图](paper-assets/README.md)：统一模型、消融、混淆、特征数、稳定性及流程；171结果独立表 |
| 检查与完整性 | [292测试日志](unit-tests.log)、`delivery_validation.json`、`files_sha256.json`；每子目录另保留原SHA／无损压缩与复制校验 |

## 同855份五折的主要模型

| 模型 | 正确数 | Accuracy | Macro-F1 | Balanced Accuracy |
| --- | ---: | ---: | ---: | ---: |
| 优化MC-UVE＋SVM（R2） | 821 | 96.0234% | 0.9197 | 89.9184% |
| 收缩LDA | 786 | 91.9298% | 0.8300 | 88.8571% |
| 原强MLP＋100源更新，固定seed42 | 777 | 90.8772% | 0.8319 | 80.9184% |
| sklearn单隐层MLP | 759 | 88.7719% | 0.7900 | 78.4490% |
| 全谱SNV-SVM | 753 | 88.0702% | 0.7963 | 80.4082% |

R2重新拟合的855标签和全部七维分数精确匹配旧结果。新全谱分类器内调；原强MLP是旧架构预先固定后的同折重训，架构曾在有历史重叠的旧513源选择，不声称重新无偏验证全部架构选择。各表示、网格和成本不同；相同数据访问不等于严格同预算。

所有新选择冻结并审查后，于北京时间16:32:33统一放行三个实际进程。原历史文件与权重未覆盖，不合并main。新的完整模型保存在各自报告目录；原R2包引用仓库内已有权重，不重复占空间。新增元数据与归档中的`outputs/...`指实际运行路径，具体重建／模型定位命令见各README。

## 主要边界和下一步

ZY02仍仅5份，R2仅3／5召回；文件名不是新增批次，真实生长地与来源日志待补。黄芪403是谱记录、人参42是压片，不自动等于403／42独立植物。跨植物B／C迁移和真实跨批次校准均未执行，因为当前数据资格不足。

保守限幅v1实际无分类作用，不能称新算法成功；全部100个支持抽样及伤害结果保留。标准MC-UVE、SVM、LDA和元学习方法均保留来源。后续优先补真实溯源、ZY02与新批次，外部资格合格后再检验迁移，不通过追逐现有测试分数包装创新。
