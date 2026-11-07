# 已保存855份OOF预测的条件不确定性与特征稳定性分析

实际执行：固定种子20261106、2000次按真实类别分层的配对bootstrap；没有重新训练或选模。
所有五阶段逐样本ID、标签和外层fold一致；Accuracy、Macro-F1、Balanced Accuracy全部与历史汇总重算一致。

这些是**给定已保存预测的条件区间**。855份数据已有历史曝光；外层模型共享训练样品，区间不覆盖训练、选参和特征选择不确定性。

| 阶段 | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|
| R0 | 90.4094% | 0.8092 | 83.6735% |
| R1 | 92.5146% | 0.8277 | 86.6531% |
| R2 | 96.0234% | 0.9197 | 89.9184% |
| R3 | 95.4386% | 0.8877 | 86.3878% |
| R4 | 94.9708% | 0.8820 | 85.8163% |

区间及配对差值见CSV；McNemar为描述性的名义精确检验，七个预先写定比较进行Holm调整。不能将其当作独立批次确认或完整嵌套CV重训检验。

特征稳定性直接读取25份真实外层训练元数据，不加载模型权重。不同保留数量采用匹配集合大小的均匀随机子集超几何期望作参照；该参照忽略邻近频率相关性。外层训练集合重叠，特征稳定性仅作描述。

ZY02仅5份，所有bootstrap保持5次该类抽样，无法补足生物样本或跨批次证据。R4为每折内层选定的组合家族，家族可以不同。

## 源文件与复现

全部输入及SHA-256、实际CPU/wall、环境见summary.json。输出目录要求不存在，防止覆盖历史产物。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=.runtime/quantum-pilot-site:.runtime/quantum-upgrade-site .runtime/bin/python -B scripts/research4_existing_model_evidence_20261106.py --output reports/cloud-followup-20261106-v1/model-evidence
```

不能生成真正独立新批次、标准品或HPLC/LC-MS数据；这些实验仍未执行。
