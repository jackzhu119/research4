# R2最佳历史候选：可复现索引包

本包引用现有五个R2完整管线，不重复复制权重。每个模型只训练684份源样品；855份OOF由各自171份留出拼接。本次**未训练新模型**，855个标签逐项重放与冻结记录相同：821份正确，Accuracy 96.0234%，Macro-F1 0.9197，Balanced Accuracy 89.9184%。这不是新增独立确认成绩。

`candidate_index.json`保存每折源／留出编号、完整612点频轴、MC次数／潜变量／保留比例、实际频点索引与频率、C／gamma／类别权重、源StandardScaler统计、原环境和SHA。原权重路径从仓库根解析，原模型与历史结果不改动。`replay_metrics.json`含逐类指标和混淆矩阵；`heldout_replay.csv`是实际重放结果。

## 重现与预测命令

在仓库根、已安装本项目的Python环境运行：

```bash
.runtime/bin/python scripts/package_paper_c_best_candidate.py --output outputs/paper-c-r2-package-new
.runtime/bin/python scripts/predict_paper_c_candidate.py --input outputs/approved-zaoren-20261031/datasets/candidate_0p1_1p5.npz --mode heldout-replay --output outputs/paper-c-r2-heldout-new.csv
.runtime/bin/python scripts/predict_paper_c_candidate.py --input new_spectra.npz --mode future-ensemble --confirm-compatible-measurement --output outputs/paper-c-new-spectra.csv
```

当前打包命令读取云端原始冻结NPZ并检查其文件SHA；原缓存不在时，按[数据准备和频率裁剪](../../../../docs/ZAOREN_APPROVED_EXECUTION_REPORT.md)重建，再用heldout预测入口验证相同数值。重新运行内层选择可用原入口（完整重跑需要原分区记录及匹配输入，评价阶段还须新的选择锁和放行记录，不应复用旧放行）：

```bash
.runtime/bin/python scripts/run_staged_nested_search.py --input outputs/approved-zaoren-20261031/datasets/candidate_0p1_1p5.npz --protocol data/research/staged_nested_protocol_20261031.json --output outputs/paper-c-nested-reproduction-new --stage selection
```

实际冻结证据见[原核心证据](../../../staged-nested-optimization/2026-10-31/core/README.md)，实验定义见[四轮协议](../../../../docs/STAGED_NESTED_OPTIMIZATION_PROTOCOL.md)。本次未重新拟合，上述完整选择重跑命令未在本打包步骤再次执行。

输入NPZ必须含`X_alpha`（N×612）、`axis`（原顺序的612点）和唯一`record_ids`。原始数值应为同一定义、单位和测量条件下的吸收系数；训练导出文件的绝对吸收单位尚未独立核实，不能自动接收其他仪器的不同物理量。未来模式要求显式确认兼容性，拒绝旧855编号；不得将旧样品改名后当作新样品。

默认heldout模式只调用该编号原留出折，避免对见过样品的模型打分。future模式对真正未知、同轴兼容样品输出五折类别、投票和分歧；平票按固定类别顺序处理并标记。五折集成目前**没有独立外部准确率**，投票比例不是校准概率。所有预测仅在ZY01—ZY07匿名已知类别间作强制分类；真实产地名称、未知产地检测和跨批次适用性仍未验证。96.02%不可标到该未来集成上。

频轴不符、非有限输入、历史ID在future集成中出现、模型／冻结源码SHA不符均拒绝处理。原855份谱的逐行数值hash也保存在索引中，heldout模式核对编号与原谱数值对应，future模式拒绝将已知原谱改名当作新样品。预测和打包输出不覆盖已有文件／非空目录。
