# 神经网络源内训练轮数诊断：已有证据与后续最小方案

2026-10-31，北京时间。诊断版本`deployment-source-curve-review-20261031-v1`。
用户当前允许按ZY01—ZY07匿名七类继续，不依赖真实地名。此次只读取已冻结神经实验的**内层训练记录**；没有训练新模型、更新旧权重或根据外层成绩选择方法。独立文件不改写`upgrade-20261031-v1`报告或SHA索引。

## 当前能得出的结论

**已有480条逐epoch源训练损失曲线，可以在不重训的情况下描述优化进度；它们不足以确认泛化欠拟合，更不能证明延长训练会提高准确率。**

已实际读取五个永久`inner_outer_XX_evidence.tar.gz`，核对每个tar与原`files_sha256.json`一致、每个`training.json`成员与原`archive_validation.json`记录逐字节SHA一致。共8家族×2profile×5外折×2内折×3seed＝480条曲线；每条仅342条当前内层源记录。没有读取`evaluation/`、外层真值或外层分数。

每条曲线完整含12或24个有限值。480条曲线均出现“最后3轮均值低于再前3轮均值”。这说明这些已运行配置在该训练损失定义下尚有末段优化进展；没有独立验证曲线，不能由此识别过拟合／欠拟合或预测收益。

## 实际训练损失汇总

每行30条曲线，来自重叠开发划分及固定seed42／43／44；不是30份独立物理重复。`末段相对下降`定义为`(mean(loss[-6:-3])−mean(loss[-3:])) / mean(loss[-6:-3])`，先逐曲线计算，再取中位数。表格不用于跨家族选优。

| 家族 | profile | 曲线数 | 第1轮损失均值 | 最后一轮损失均值 | 末段相对下降中位数 | 末段下降条数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cnn_alpha | regularized24 | 30 | 1.9700 | 1.3157 | 5.77% | 30 |
| cnn_alpha | small12 | 30 | 1.9742 | 1.1245 | 17.23% | 30 |
| cnn_n | regularized24 | 30 | 1.9522 | 1.3093 | 4.90% | 30 |
| cnn_n | small12 | 30 | 1.9360 | 1.1146 | 14.86% | 30 |
| dual_alpha_mcuve | regularized24 | 30 | 1.9486 | 1.0062 | 6.53% | 30 |
| dual_alpha_mcuve | small12 | 30 | 1.9169 | 0.9396 | 9.17% | 30 |
| fusion_all_four | regularized24 | 30 | 1.9226 | 0.7942 | 10.04% | 30 |
| fusion_all_four | small12 | 30 | 1.8945 | 0.7723 | 15.58% | 30 |
| fusion_alpha_n | regularized24 | 30 | 1.9467 | 1.0375 | 7.52% | 30 |
| fusion_alpha_n | small12 | 30 | 1.8990 | 0.8680 | 12.98% | 30 |
| fusion_time_frequency | regularized24 | 30 | 1.9617 | 1.0900 | 9.76% | 30 |
| fusion_time_frequency | small12 | 30 | 1.9122 | 0.9798 | 14.23% | 30 |
| resnet_alpha | regularized24 | 30 | 1.9539 | 1.1738 | 5.00% | 30 |
| resnet_alpha | small12 | 30 | 1.9658 | 0.9500 | 16.85% | 30 |
| se_alpha | regularized24 | 30 | 1.9757 | 1.3036 | 6.81% | 30 |
| se_alpha | small12 | 30 | 1.9211 | 1.0095 | 15.70% | 30 |

### 损失的确切含义和缺失记录

[`fit_source`](../src/herb_spectra/upgrade_neural.py)每轮在`model.train()`下更新权重，以每batch的`CrossEntropyLoss`乘batch人数后除当轮源记录数，保存为`metadata.loss_curve`。因此：

- 这是源训练阶段各batch优化损失的人数加权平均，包含训练态dropout和当轮逐batch不同的模型权重；不是某一个固定checkpoint在全源样品上重新计算的eval loss。
- `regularized24`含类别权重，而`small12`不含；PyTorch加权CE在每batch内先按权重归一化。该epoch平均值不等于一次性全数据加权CE；不同profile的损失不宜直接相减作拟合能力优劣。
- 两profile还同时改变width、blocks、dropout、学习率及MC参数，旧12与24轮差异不是只改变epoch的单因素试验。
- 原记录没有每epoch内层验证CE、Accuracy、Macro-F1、Balanced Accuracy或逐类Recall，也没有每epoch源eval指标和中途checkpoint／优化器／随机生成器状态。原完整管线保留的是训练终点。
- 故不能从已有终点权重恢复过去每轮验证预测；也不能用最终外层分数反推“第几轮最好”。已有末段损失下降不等于增加epoch必然改善未知样品。

本版已从以上冻结记录实际导出480份逐曲线CSV、8640条合并epoch记录、16组描述汇总与288条组×epoch均值／SD；每个profile分别生成PNG及PDF，两张图均已实际目检。8640个loss单元回读与原值完全一致，另以NumPy独立复算576个组均值／样本SD，最大差小于10⁻¹²。6项完整性正反探针通过，含NaN、截断曲线、重复源ID、坏成员SHA与不完整／重复总体拒绝；合成探针不是训练数据。

产物见[独立报告目录](../reports/zaoren-neural-source-diagnostic-20261031-v1/README.md)、[12轮图](../reports/zaoren-neural-source-diagnostic-20261031-v1/small12_source_loss.png)、[24轮图](../reports/zaoren-neural-source-diagnostic-20261031-v1/regularized24_source_loss.png)、[480条曲线汇总](../reports/zaoren-neural-source-diagnostic-20261031-v1/curve_summary.csv)与[16组汇总](../reports/zaoren-neural-source-diagnostic-20261031-v1/group_summary.csv)。导出脚本为`scripts/export_deployment_source_curves.py`，输出目录不覆盖原实验。实际新增模型fit／推理均为0；后续扩展训练仍NOT_RUN。

## 建议后续版本：仅源内最小轮数诊断（NOT_RUN）

以下是待root锁定并放行的后续方案，**未执行、不构成已注册的正式比较结果**。不得覆盖原协议、600个神经模型或96.02%的MC-UVE／SVM历史结果。

1. 固定原`outer00.inner00`：342条source和342条内层validation。外层171条query的所有表示、标签、损失均不加载。此单上下文仅是运行／优化诊断，不作五折总体性能声明；ZY02按原分区保留，不复制／删类。
2. 固定两个结构代表：`cnn_alpha`与`fusion_all_four`，对应单支和四表示任务需求。两者均固定原`regularized24`的width12、blocks2、dropout0.35、lr0.0005、Adam weight_decay0.001、balanced class weight、batch64；仅把最大训练上限固定为96轮。不得依据外层成绩改为其他家族或另挑seed。
3. 使用全部seed42／43／44；每家族每seed独立初始化模型、优化器、生成器和当前342条source的scaler。共6次最大96轮源拟合，保存第12／24／48／96轮checkpoint；四个快照来自同一轨迹，不算24次独立训练或新增实物样品。不恢复旧终点，也不使用全855部署模型／scaler预训练，避免把禁止样品带回内层。
4. 每轮记录训练态batch优化损失；另外在冻结变换下以`eval()`／`no_grad()`计算source及内层validation的CE、Accuracy、Macro-F1、Balanced Accuracy和逐类Recall。validation结束后核参数、BN buffers和优化器状态不变，再恢复训练态。源eval与validationeval可比；训练态优化损失另标。validation不得参与任何scaler／BN／特征fit。
5. 快照记录权重、优化器、生成器状态、代码／数据／分区SHA和fit IDs；日志、曲线、所有seed及未改进结果完整保存。训练上限事前固定，图形不按最后一个点是否上涨延长。
6. 首轮只回答“训练优化是否持续、源eval与内层validation是否分离、增加轮数是否存在源内收益线索”。若后续要选epoch或开展五外折研究，另锁候选、指标、并列规则、预算和所有source选择记录后，才允许对应外层评价。不能把已有855条重新称未见确认数据，也不能以超过96.02%作继续／停止规则。

在本单上下文内，各seed与checkpoint仅用于诊断；不计算样品独立显著性，不基于这6条轨迹给出普遍最优epoch。单因素epoch结果也不能替代分支宽度、类别权重、学习率及融合输入的分别消融。

## 来源与复核范围

- [原防泄漏协议](DATA_LEAKAGE_PREVENTION_PROTOCOL.md)、[历史曝光登记](DATA_EXPOSURE_REGISTER.md)。
- 原逐epoch写入逻辑：`src/herb_spectra/upgrade_neural.py:192–203`；profile定义在同文件顶部。
- 原五个tar与成员SHA均在此次读取中实际核对，下面为实际输入版本；没有修改任何tar或其原索引。

| 永久证据包 | SHA-256 |
| --- | --- |
| `reports/upgrade-20261031-v1/neural/inner_outer_00_evidence.tar.gz` | `41455163ce0b63427f649bba666acf5a05ed982e917f7faf1fc42c684419c525` |
| `reports/upgrade-20261031-v1/neural/inner_outer_01_evidence.tar.gz` | `f2f52fc81efbe3664bd83f12420c80de42c9852e540b04af00173b5fab5a476d` |
| `reports/upgrade-20261031-v1/neural/inner_outer_02_evidence.tar.gz` | `8a5837e46ab769c0a6a584d43e88403b296b57d8ae4c62aea72a503093a1e53a` |
| `reports/upgrade-20261031-v1/neural/inner_outer_03_evidence.tar.gz` | `e754df24155bfda6627509e93453b47ccd30d1383c590dae9bc693097c5937b4` |
| `reports/upgrade-20261031-v1/neural/inner_outer_04_evidence.tar.gz` | `2c6654ca82de26ced4f2582fb4c09337c4d7cca2e3690b7b21b7c02569b09924` |

复核实际包含：5个tar SHA、480个training.json成员SHA、480条长度／有限性、480组342条唯一source ID数，以及16组30条曲线的汇总计数。没有调用模型fit、优化器step或任何预测接口；没有新增计算理论谱、校准或外层评价。实际运行命令、脚本SHA、环境版本和完整性检查见新报告`command.json`、`validation.json`、`csv_readback.json`、`independent_group_arithmetic.json`及`files_sha256.json`。该描述不证明绝对零泄漏、独立批次或生物学泛化。
