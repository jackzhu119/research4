# 频率与候选峰导出准备

状态：**NOT_RUN**。没有读取当前模式／Hessian数组，没有生成Spinosin频率或实验归属结果。16项标准库测试使用明确的临时合成夹具验证单位、符号、219模式完整性、重复候选记录、并列最近距离和篡改拒绝，不是科研计算。

冻结76e2独立分析器已经提供以下输出，直接复用，不另写绘图／动画实现：

| 输出 | 含义 |
|---|---|
| `all_frequencies_cm1_THz.csv` | 全部219个有符号模式，负／零保留 |
| `normal_modes_0p1_to_1p5_THz.csv` | 固定THz区间内的正模式位置 |
| `all_experimental_theory_pairs.csv` | 仅合格全正极小点时的全部候选记录×模式差值 |
| `signed_mode_frequencies.png/pdf` | 有符号频率图 |
| `low_frequency_positions_not_IR.png/pdf` | 等高度模式位置及实验候选点，不是IR强度谱 |
| `animations/*.xyz` | 前6个低频正模式及最多3个虚频诊断模式的显示动画 |

完整219个振型仍保留在真实global的`normal_modes.npy`；不声称已经输出219个XYZ。XYZ使用任意0.15Å显示幅度与31帧，不是分子动力学。零模式保留在全频率表／原振型数组中，不被转成正频或隐藏。

实验候选来源为`reports/upgrade-20261031-v1/source-peaks/experimental_candidates_by_fold.csv`，SHA`b3a5f591…`。这是20条重叠fold记录，不是20个独立化学峰。

可选的新小脚本是 `scripts/summarize_v7_audited_peak_proximity_20261105.py`。它只在真实76e2审计index/checks/219频率CSV通过后，用标准库生成最近**位置距离**CSV，保留所有并列值；不读NPY，不重算频率，不产生IR强度，不确认成分，不修改分类器。非全正或未合格极小点只记录阻断，不强行配峰。即使最近距离很小，也不能确定化学归属。

未来由root在所有Q闭合后物化真实hash并执行：

```bash
.runtime/bin/python -B -S scripts/summarize_v7_audited_peak_proximity_20261105.py \
  --execute --audit-index ACTUAL_INDEPENDENT_ANALYSIS/artifact_index.json \
  --audit-index-sha256 ACTUAL_INDEX_SHA256 --global-result-sha256 ACTUAL_GLOBAL_RESULT_SHA256 \
  --output FRESH_NEAREST_POSITION_REPORT
```

当前只运行了默认NOT_RUN预检。理论比较属于孤立分子谐振动机制探索；没有纯品THz、配对化学含量或固态结构验证时，不得据此确认药材吸收峰的成分来源。
