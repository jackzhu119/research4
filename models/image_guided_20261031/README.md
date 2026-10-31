# 图片建议优化：模型与概率参数

本目录为实际执行产物的原样副本，原运行路径、保存位置和SHA列在[manifest.json](manifest.json)。结果与限制见[中文报告](../../docs/IMAGE_GUIDED_OPTIMIZATION_REPORT.md)。

`features/`保存7个去重候选×3种子＝21个最终完整joblib管线及元数据。它们只用513份源样品训练，内部包含源SNV／MSC参考、缩放、变量选择或PCA及分类器。输入必须是原频率坐标0.1—1.5范围的612点吸收系数，列顺序和单位不变；编号、文件名和标签不作为特征。选中方法是`f012_snv_lda`，158／171＝92.3977%，但宏F1仅0.8352；ZY02精确率20%，不能称七类均衡改善。

特征缓存V2使用保存后重载的稳定指纹，完整管线可以正常重载；PCA-LDA修复前后数值与预测一致。CARS／UVE／SPA及PCA-LDA家族是开发预选诊断，不因最终得分更高就替换正式方案。源码位于`src/herb_spectra/image_feature_models.py`，需要仓库代码和记录的sklearn／NumPy版本，不是原MATLAB权重。

`probability/final/source/seed_*/oof/`保存最终源OOF折的模型和预处理器，用于复核概率拟合；**它们不是最终全513份源训练的部署分类器**。本轮使用的全源SVM及两个MLP均复用此前冻结模型：

- SVM完整源管线：`models/representation_20261031/s010_snv_svc.joblib`。
- 基础MLP：`models/multiround_20261031/neural-base/final/seed_*/b006_mlp/model.pt`，配同目录`preprocessor.joblib`。
- 较强MLP：`models/multiround_20261031/meta-refinement/final/seed_*/m000/supervised.pt`，配同种子的上述基础源预处理器和64–32结构。

源OOF温度与每个候选判别参数原样保存在[概率证据](../../reports/image-guided-optimization/2026-10-31/probability/)中的`final_temperatures.csv`、`final_decision_parameters.csv`及context元数据。类别顺序为ZY01—ZY07，使用同种子、同provider的参数；不能将SVM margin当已校准概率。源温度使用`softmax(scores/T)`，T为正，分类不变。类别偏置属于判别规则，未将其当校准后验概率。

融合无需新增源二进制模型；完整recipe需要结合[融合证据](../../reports/image-guided-optimization/2026-10-31/fusion/)的`candidate_grid.json`与`selection_lock.json`：按候选ID查询组成模型和权重，由选择锁确定正式方案；`final_records.json`保存检查点路径及SHA，各`evaluation.json`保存实际温度、来源和结果。实际公式为`w*pSVM+(1-w)*pMLP`，先对齐类别列。正式选择的SVM权重为0，即温度校准较强MLP，不应把其90.2534%称作融合提升。

所有概率参数只使用源OOF标签，目标171份校准池标签使用数为0。当前数据仅单批次、预测组已有历史曝光，没有证明新批次或外部样品效果。全部原始概率、判别标签、source fingerprint和恢复检查已归档，原有模型未覆盖。
