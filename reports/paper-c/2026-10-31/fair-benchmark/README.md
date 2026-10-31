# 方案C：酸枣仁同折分类器对照与R2条件消融

2026-10-31，使用同一855份用户声明独立样品、0.1—1.5主区612个候选频点，以及原5外折／2内折。此处是实际重新拟合和外层评价，不是把此前171份预测成绩挪入本表。每份样品每方法恰有一个OOF预测，预测标签与真实匿名组逐项比较。

全部新source选择锁齐，并经根agent统一放行后，才执行本次外层评价；新增另一强MLP对照也在这个统一屏障之前锁定。放行见[统一记录](../evaluation_release.json)与[本实验绑定记录](../evaluation-releases/fair-benchmark.json)。本实验新增45个外层完整管线、7695条外层预测，未根据结果再改格点或种子。

## 实际结果

| 方法 | 正确数／855 | Accuracy | Macro-F1 | Balanced Accuracy | 平均频点数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MC-UVE＋SVM：R2原样复现 | 821 | 96.0234% | 0.9197 | 89.9184% | 98.4 |
| 全谱SNV＋SVM（96候选） | 753 | 88.0702% | 0.7963 | 80.4082% | 612.0 |
| 全谱收缩LDA（10候选） | 786 | 91.9298% | 0.8300 | 88.8571% | 612.0 |
| 全谱小MLP（8候选） | 759 | 88.7719% | 0.7900 | 78.4490% | 612.0 |
| 条件消融：去MC、保留R2 SVM | 757 | 88.5380% | 0.7951 | 78.1633% | 612.0 |
| 条件消融：随机同K、保留R2 SVM | 770 | 90.0585% | 0.8142 | 80.1224% | 98.4 |
| 条件消融：C10／scale、保留权重 | 796 | 93.0994% | 0.8343 | 83.9388% | 98.4 |
| 条件消融：无类别权重 | 817 | 95.5556% | 0.8896 | 86.6327% | 98.4 |
| 条件消融：固定MC20／LV2／62点 | 786 | 91.9298% | 0.8365 | 82.3061% | 62.0 |

MC-R2实际重新拟合后的全部855个标签及每份7维决策分数与历史R2精确一致，821／855，未创造一组新的独立确认数据。新全谱SVM、LDA、小MLP均在同样品和同折独立选择，但输入维数和搜索预算与MC-R2不同；它们是任务层面的公平分区对照，不称只更换分类器的完全等预算实验。另一份64–32监督优化强MLP由独立协议保存，不能把此处8候选单隐藏层小MLP称为旧强MLP的精确复现。

R2去类别权重后ZY02仅2／5正确，完整R2为3／5；新LDA的ZY02召回为4／5，但精确率只有25%，F1为0.3810，故不能只按某个召回值排名。全部逐类Precision／Recall／F1、混淆矩阵、真实来源和编号见`pooled_metrics.json`及`outer_predictions.csv.gz`。

## 选择与消融定义

- full-SVM：6个C×4个gamma规则×4个训练内类别权重，共96。保留原标准SVC.predict投票，scale／auto可能数值等价，实际gamma逐项保存。
- 收缩LDA：shrinkage＝auto／0.01／0.1／0.5／1，prior＝源经验频率／uniform，共10。
- 小MLP：单隐藏层64／128，alpha＝0.001／0.1，源sample_weight＝none／balanced，共8；Adam200轮、batch64、学习率0.001、seed42，不按验证或外层早停。全部80个内层训练的实际200轮及80条ConvergenceWarning保存，0失败；达到预算上限不等于已充分收敛。
- 每外折分别按两内折平均Macro-F1→Balanced Accuracy→Accuracy→固定candidate ID选参数。SNV、源StandardScaler、类别频率权重每个训练分区重拟合；无验证或外层统计入模。内层共1140次候选评价、389880条预测。
- 六项条件诊断继承该外折历史R1/R2的训练内选择。移除MC、随机同K（固定seed42、与标签无关）、C10／scale、去权重、固定MC20／LV2／62点，均按预设执行；它们存在参数交互，不能相加解释为独立单因素因果贡献。`scale`规则在改变表示后依当前源表示重算。

## 计算成本与真实执行

内层选择实际152.0561秒，外层阶段实际25.7382秒，另有外部driver真实UTC开始／完成日志。每个外层方法独立新缓存，成本含源变换拟合、分类器拟合、预测和实际模型字节数；完整数值在`model_comparison.csv`和各方法`cost`。内层源变换可在同一有序训练分区的候选之间复用。共享CPU并发、不同搜索预算和完整MC统计缓存均影响成本，详见[COST_INTERPRETATION.md](COST_INTERPRETATION.md)，不称独占硬件benchmark。

## 证据与复现

[放行前独立代码审查](pre_outer_review.md)通过；纯只读审计复算全部内外预测／源均值方差及15个内层选择、重放45个模型、对齐旧R2的855标签和分数：1,923,269项检查通过，最大数值差0，新增审计拟合0。详见`selection_arithmetic_audit.json`和`completed_arithmetic_audit.json`。

- `models/`：45个完整新管线及元数据；均为684份源训练的折模型，不是一个全855部署模型。
- `all_inner_metrics.json.gz`：84MB原JSON的精确无损压缩，解压后SHA与选择锁一致，未删失败或warning。5份逐折原JSON可由其筛选按原JSON编码字节重建，规则及各原SHA见`archive_validation.json`。
- `inner_predictions.csv.gz`／`outer_predictions.csv.gz`：389880／7695条实际预测。
- `selection_lock.json`／`run_manifest.json`／`protocol_snapshot.json`／`splits.json`：全部选择、源码／输入／环境SHA及具体分区。
- `stage_events.jsonl`及[实际外部调用](../command_invocations.jsonl)：真实UTC过程；历史源文件、模型、数据与结果未覆盖。

新增源码为[paper_c_benchmark.py](../../../../src/herb_spectra/paper_c_benchmark.py)，入口为[run_paper_c_benchmark.py](../../../../scripts/run_paper_c_benchmark.py)，定义见[新协议](../../../../data/research/paper_c_benchmark_protocol_20261031.json)。真实环境Python3.12.14、NumPy2.3.5、SciPy1.17.0、scikit-learn1.8.0、joblib1.5.3；MLP的sample_weight需要支持该接口的sklearn版本，复现优先匹配manifest。

原输入NPZ不在时按[原数据准备和频率裁剪](../../../../docs/ZAOREN_APPROVED_EXECUTION_REPORT.md)重建并检查SHA。独立重跑选择应使用新空目录：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .runtime/bin/python scripts/run_paper_c_benchmark.py --protocol data/research/paper_c_benchmark_protocol_20261031.json --output outputs/reproduce-paper-c-fair-new --stage selection
```

新锁形成后检查全部来源与选择，按新锁生成新的绑定放行JSON，再运行同一命令的`--stage evaluation --release <新放行文件>`；本次放行只绑定本次锁，不能借用到不同选择。运行不覆盖已完成目录。只读复核实际云端运行可执行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .runtime/bin/python reports/paper-c/2026-10-31/fair-benchmark/audit_completed.py --run outputs/paper-c-20261031/fair-benchmark --protocol data/research/paper_c_benchmark_protocol_20261031.json
```

这里所有855份和历史成绩此前已曝光，每个产地只有一批、ZY02仅5份，故仍是回顾性批次内内部验证；不能称真实新批次泛化。原MC-R2维持酸枣仁主线，外部植物与迁移不可取代主结果。统计改进不等于提出新算法，特征频率也不自动构成化学峰归属。
