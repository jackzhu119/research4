# 保守跨植物预训练：48轮预算开发试验

本轮已真实完成，但没有取得外部预训练增益。三种子均值中，黄芪＋天麻
预训练及L2-SP的Accuracy为76.1404%，低于同协议从头训练的80.3509%；
Macro-F1和Balanced Accuracy也降低。全部结果保留，未更改标签、删除错例
或挑选有利种子。

运行ID：`cross-plant-conservative-pilot-20261104-v1`。
[完整证据](../reports/cross-plant-conservative-pilot-20261104-v1/README.md)；
[预注册协议](../reports/cross-plant-conservative-pilot-20261104-v1/protocol.json)。
协议SHA-256为`dee8a86b6f0200b3b9e3dd80e39c9d0a847cc6a52914531b931fe3b76255d50e`，
训练脚本SHA为`a0aed48133444a8b77abf79de3aca324efed5239914769ad5d033108c8af23c4`。

## 同协议的实际比较

下列为固定42／43／44三种模型种子的均值。每个种子的分母始终是同855份
酸枣仁；重复训练不增加独立样品数。SD仅描述模型种子波动。

| 方法 | Accuracy均值±样本SD | Macro-F1 | Balanced Accuracy | 较A准确率变化 |
| --- | ---: | ---: | ---: | ---: |
| A：从头训练 | 80.3509% ± 3.1644个百分点 | 0.678626 | 73.9592% | 0 |
| Aextra：目标source额外预训练＋L2-SP | 78.3236% ± 2.6004个百分点 | 0.651480 | 70.1701% | −2.0273个百分点 |
| B：黄芪＋天麻预训练＋L2-SP | 76.1404% ± 4.2024个百分点 | 0.630414 | 68.3333% | −4.2105个百分点 |

B相对A准确率变化率为−5.2402%，这与−4.2105**个百分点**是不同单位。
B相对额外计算控制Aextra还低2.1832个百分点，因此本轮没有证据支持外部
预训练带来收益。B−A比较整个预训练及约束配方；B−Aextra才是同预训练步数、
同L2-SP规则下外部source相对目标source的比较。临时分类头维度不同，
不称严格FLOP匹配。

| 种子 | A正确数 | Aextra正确数 | B正确数 | B相对A纠正／损害 | B相对Aextra纠正／损害 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 715/855 | 682/855 | 692/855 | 22／45 | 51／41 |
| 43 | 661/855 | 644/855 | 636/855 | 44／69 | 59／67 |
| 44 | 685/855 | 683/855 | 625/855 | 30／90 | 41／99 |

单看种子42，B相对Aextra多对10份；其余两个种子分别少对8及58份。
本轮完整结论依据全部预定种子，不能只选择这一项局部改善。

## 数据、模型及选参范围

酸枣仁855×612、黄芪403×2578和天麻405×2578的既有α矩阵及原件摘要均与
冻结记录一致。导出名义频率轴中，酸枣仁为0.100708—1.499176 THz，黄芪／
天麻为0.100708—5.998993 THz。复用既有0.21—1.49 THz固定128点线性插值，
无外推。上述是导出结构／数值轴核验，不能代替绝对α单位、仪器校准和批次
元数据核验。用户声明黄芪／天麻各谱独立；共同原料、批次和仪器等仍不完整。
本轮定位为**光谱表示开发探索**，不是跨仪器或跨批次的确认性证据。

三个arm使用相同128点α、逐谱SNV、当前目标source独立拟合的StandardScaler，
以及width12、两个残差块、dropout0.35的小型ResNet1D。Adam学习率0.0005、
weight decay0.001、目标batch64。黄芪、天麻保持独立四分类头，不转写为ZY标签。
人参未进入本轮，SDF、分子描述符、DFT频率和文献地区均值也未进入训练。

Aextra在当前ZY source上用临时七分类头训练120步；B在黄芪／天麻两个
source池各训练60步，batch32。之后均重新建立配对的ZY头及Adam优化器、
重置BatchNorm运行统计，目标样品顺序和dropout种子配对。L2-SP只约束encoder
可训练参数到实际预训练权重，系数事前固定0.001；不约束BN运行buffers。
该起点约束属于既有方法，非本项目原创。[原始方法论文，Li等，ICML 2018](https://proceedings.mlr.press/v80/li18a.html)。

沿用原五外折／二内折及42／43／44三个种子。只用A的两内折source OOF，
分别比较1—48轮，以三种子均值Macro-F1、Balanced Accuracy、Accuracy及较早
epoch依次排序。5个外折最终共同轮数为**48、47、48、48、44**，同一外折
三arm和各种子采用相同轮数。30个source完整轨迹全部结束、5个选择锁统一
固定后才开展45个outer管线；全部outer预测锁定后，另一进程才连接真实标签评分。

## 验证、成本及证据

8项软件检查和独立代码审查通过，覆盖固定角色、epoch排序、标签摘要、
L2-SP真实梯度／anchor隔离及有界资源异常处理。实际source／optimizer输入
都有ID、X／y指纹；所有outer管线完成单条／批量／反序推理和模型状态不变检查。
额外从45个已保存权重与source scaler重新原生推理7695条预测，最大logit差为0，
新增训练拟合和量子计算均为0。不能将这些有限检查扩大为“绝对零泄漏”。

独立审计另外从原保存角色重算240条source epoch曲线，选轮数完全一致；
检查全部8640个source优化batch的实际IDs及数值指纹。再从45个outer模型的
7695条预测独立复算9组总体／逐类指标和6组纠正／损害计数，最大指标差为0；
23265个目标更新与3600个预训练更新的来源和实际覆盖均核查。
[审计证据](../reports/cross-plant-conservative-pilot-20261104-preflight-v1/actual-result-audit/)。

实际独立worker／score的wait4 CPU分别为259.795884和3.595594秒，合计
**263.391478秒**；监督墙钟346.10秒，峰RSS约397.14 MiB。Torch、interop、
OMP及BLAS均为单线程，未安装或修改量子环境依赖。同期主DFT占用共享CPU，
因此本轮墙钟不用于算法加速比较。600 CPU秒／1800墙钟秒／1.5 GiB RSS
上限均未触发。

- [指标对照与提升／下降](../reports/cross-plant-conservative-pilot-20261104-v1/model_comparison.csv)
- [全部逐条预测及真实标签](../reports/cross-plant-conservative-pilot-20261104-v1/all_outer_predictions.csv.gz)
- [逐类Precision、Recall、F1](../reports/cross-plant-conservative-pilot-20261104-v1/per_class_metrics.csv)
- [全部混淆矩阵计数](../reports/cross-plant-conservative-pilot-20261104-v1/confusion_matrix_counts.csv)
- [真实模型重放](../reports/cross-plant-conservative-pilot-20261104-v1/native_model_replay.json)
- [准确率、Macro-F1、BA图](../reports/cross-plant-conservative-pilot-20261104-v1/model_metrics.png)
- [固定首种子42的混淆图](../reports/cross-plant-conservative-pilot-20261104-v1/confusion_seed42_layout2.png)
- [实际资源账本](../reports/cross-plant-conservative-pilot-20261104-v1/resource_ledger.json)

## 对论文主线的含义

保留MC-UVE＋SVM历史96.0234%强开发基线。本轮采用128点和最多48轮的新配方，
旧跨植物试验通常选96轮，两者不能直接拼成一次优化的提升阶梯。短预算是否
造成差异尚未通过本轮的独立预算消融证明。

855条均已多轮曝光，每来源只有一批；新编号、重划分和重复种子不构成新确认集。
本轮未重新执行完整outer真值置乱重训及训练标签置乱零分布，旧试验的检查
不能代替本轮未执行项目。结果仅支持“这个事前固定的保守配方没有获益”，
不能推出所有跨植物迁移都无效，更不能改标签或不断试到分数上升。
