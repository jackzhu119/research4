# 保守迁移预算消融：48轮上限与96轮上限

两轮均实际完成，结果表明：**扩大训练／轮数选择预算提高了本次三个神经
配方的表现，但未证明加入黄芪、天麻预训练能提高酸枣仁准确率。** 96轮上限
试验中，从头训练Accuracy为87.0565%，黄芪＋天麻预训练为85.3801%。
历史MC-UVE＋SVM的96.0234%仍单独保留，本轮没有替换它或声称新的独立确认。

## 完整结果

三种子42／43／44的均值；每个种子的目标分母为相同855条。各预算内所有
方法采用相同五外折、原两内折、输入表示及source选择的共同目标轮数。

| 最大候选轮数 | 配方 | Accuracy | Macro-F1 | Balanced Accuracy | 同预算较A准确率变化 |
| --- | --- | ---: | ---: | ---: | ---: |
| 48 | A从头训练 | 80.3509% | 0.678626 | 73.9592% | 0 |
| 48 | Aextra目标source预训练＋L2-SP | 78.3236% | 0.651480 | 70.1701% | −2.0273个百分点 |
| 48 | B黄芪＋天麻预训练＋L2-SP | 76.1404% | 0.630414 | 68.3333% | −4.2105个百分点 |
| 96 | A从头训练 | 87.0565% | 0.760988 | 79.1837% | 0 |
| 96 | Aextra目标source预训练＋L2-SP | 86.3548% | 0.747579 | 77.3878% | −0.7018个百分点 |
| 96 | B黄芪＋天麻预训练＋L2-SP | 85.3801% | 0.737772 | 77.4082% | −1.6764个百分点 |

| 配方 | 扩上限后的准确率变化 | 相对准确率变化率 | Macro-F1变化 |
| --- | ---: | ---: | ---: |
| A | ＋6.7057个百分点 | ＋8.3455% | ＋0.082362 |
| Aextra＋L2-SP | ＋8.0312个百分点 | ＋10.2539% | ＋0.096099 |
| B黄芪／天麻＋L2-SP | ＋9.2398个百分点 | ＋12.1352% | ＋0.107358 |

第二张表是**最大轮数选择预算的消融变化**，不能归因于新增外部观测。
两轮的外部池、预训练步数和数据都相同。完整CSV见
[两预算六行模型表](../reports/cross-plant-conservative-pilot-20261104-v2/two_budget_model_comparison.csv)及
[预算变化表](../reports/cross-plant-conservative-pilot-20261104-v2/epoch_budget_changes.csv)。

96上限时，B相对额外计算控制Aextra的Accuracy仍低0.9747个百分点，Macro-F1
低0.009807；Balanced Accuracy仅高**0.0204个百分点**，不能写成三项指标
都降低或整体迁移成功。B相对A的三项均值都较低。

| 种子 | A正确数 | Aextra正确数 | B正确数 | B相对A纠正／损害 | B相对Aextra纠正／损害 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 759/855 | 740/855 | 754/855 | 32／37 | 50／36 |
| 43 | 729/855 | 722/855 | 714/855 | 47／62 | 52／60 |
| 44 | 745/855 | 753/855 | 722/855 | 35／58 | 31／62 |

种子42的B相对A的Macro-F1及Balanced Accuracy局部改善，但正确数仍少5份；
种子43、44的表现更差。全部种子和所有指标都保留，不挑局部正例作为结论。

## 追加预算的事前理由与严格控制

第一轮source锁在outer预测前已固定：五折选48、47、48、48、44轮，其中三折
取原上限。它提示值得做**收敛边界检查**，不直接证明欠拟合。第二轮是在
首轮已曝光的开发研究后追加注册的消融，不能伪装成最初未见的确认队列。

第二轮只将最大候选轮数48扩至96，独立资源限额600实际CPU秒扩至900秒、
墙钟1800扩至2400秒。数据、五／二折、三种子、128点α、SNV、source scaler、
ResNet结构、Adam参数、L2-SP系数0.001、120步预训练及三arm定义完全保持。
共同轮数仍仅由当前outer的A两inner×三seed source OOF选择，不看outer真值。
新选轮数为**91、86、92、89、92**，均未取96上界；这不是全局收敛证明。

执行中还设置source-only硬核验：先按第一轮封存索引确认旧source文件SHA，
再逐项比较source／query IDs、原输入指纹、初始头／encoder及scaler mean／scale。
30条source轨迹的前48轮logits和loss必须与第一轮逐元素完全一致；失败就停止，
不进入outer。实际3,447,360个logit及1440个loss值全部相同，最大差0。
该检查只读inner source记录，未读取两轮outer真值或预测作为选参依据。

全部30条source训练结束并锁五个选择后才做45个outer管线，全部预测锁定后
由独立进程连接真值评分。两轮都执行全部预定种子，不以更高分作为停止条件。
没有添加λ、种子、预处理或数据删选搜索。

## 实际执行、复核和路径

第二轮协议SHA：`b9207b7cf048b610b976d4913767c80cfa03c8eba633fc63cea37de0c9d50890`。
第二轮脚本SHA：`d7eb20ac813185f777de8cd4ab63b360bbbecead6cf7219790a6df4e4209f78f`。
脚本为[独立v2入口](../scripts/run_conservative_cross_plant_pilot_20261104_v2.py)，原v1不改。

第二轮实际wait4 CPU为**456.892275秒**，墙钟524.51秒；两轮实际训练／评分
CPU共**720.283753秒**，分别按独立600／900秒预算记录。全程单线程、CPU设备，
RSS低于1.5GiB，未安装或升级任何量子环境依赖。与主DFT共享CPU期间运行，
不据墙钟声称算法加速。后处理和独立审计不新增训练拟合或量子计算。

8项新版软件测试及独立差异审查通过。独立复核480条source选轮曲线、30条
前48轮轨迹、45个模型和7695条预测，9组总体／逐类指标、混淆矩阵、6组修正
损害计数最大差为0。跨预算的初始化、预训练encoder、BN重置后状态、锚点及
预训练批次一致；目标loss／样品顺序的旧长度前缀也一致。
此外从全部45个保存权重和source scaler实际原生重放，logit最大差为0。
第一轮封存589份文件的字节／SHA全部保持一致。

- [第二轮证据与状态](../reports/cross-plant-conservative-pilot-20261104-v2/README.md)
- [独立审计](../reports/cross-plant-conservative-pilot-20261104-preflight-v2/actual-result-audit/audit_index.json)
- [source预算曲线](../reports/cross-plant-conservative-pilot-20261104-v2/source_epoch_curves.png)
- [两预算对照图](../reports/cross-plant-conservative-pilot-20261104-v2/epoch_budget_ablation.png)
- [全部三seed逐类Precision／Recall／F1](../reports/cross-plant-conservative-pilot-20261104-v2/per_class_metrics.csv)
- [全部逐条预测和真值](../reports/cross-plant-conservative-pilot-20261104-v2/all_outer_predictions.csv.gz)
- [全部混淆计数](../reports/cross-plant-conservative-pilot-20261104-v2/confusion_matrix_counts.csv)
- [45模型原生重放](../reports/cross-plant-conservative-pilot-20261104-v2/native_model_replay.json)
- [首轮完整报告](CONSERVATIVE_CROSS_PLANT_PILOT_20261104.md)

两轮均属于同855条曝光数据上的表示开发。黄芪／天麻用户声明独立，绝对α
单位、仪器及原料／批次元数据仍未完全核实，不能称正式跨仪器、跨批次
泛化。本轮没有加入新的公开谱、SDF、DFT频率或文献化学均值作为样品特征。
没有新独立确认集，也未在本版重新做完整标签置乱分布。现有结果支持保留
MC-UVE＋SVM强基线，并把这两项真实负迁移及预算消融写入辅助分析；不预先
宣称所有迁移方法无效，不把较好的预算版本包装成新的确认性最优模型。
