# 论文启发的少样本分类源权重

本目录保存63份本项目训练的PyTorch `state_dict`，各文件hash、大小及用途列在[manifest.json](manifest.json)。不是原论文发布的权重。

- `final/seed_42/`、`seed_43/`、`seed_44/`：每种子3份，均只用固定513份源训练样品；后续读取应使用这些最终源权重。
- `development/episode_00/`至`episode_05/`：每任务3种子×3份，用于开发选择的307份源训练，不能冒充最终513份模型。
- 每个上下文分别包含`supervised.pt`、`prototype.pt`、`fomaml.pt`。共有18个开发上下文和3个最终上下文。21份共同监督初始化只记录指纹，未单独保存权重。

结构与运行方法见[协议](../../docs/PAPER_NEURAL_CLASSIFICATION_PROTOCOL.md)、[实现](../../src/herb_spectra/paper_fewshot_classification.py)及[实测报告](../../docs/PAPER_FEWSHOT_CLASSIFICATION_REPORT.md)。运行使用Python3.12.14、PyTorch2.9.0+cpu、NumPy2.3.5；源码和输入hash见[运行清单](../../reports/paper-fewshot-classification/2026-10-31/run_manifest.json)。重建网络需使用相同紧凑ResNet1D与七类词表，输入为相同612频点的逐谱SNV；权重本身不含工作簿读取和频段筛选。

本目录保存的是**校准前源权重**。MAML正式选择配置为学习率0.001、全网10步；需要从指定校准池按保存的支持编号取得每类1份标签，在权重副本上适配，然后预测固定171份。不能把源权重直接推理结果当作67.5458%的校准结果。100次支持抽样、训练清单、源指纹及每次结果保存在[证据目录](../../reports/paper-fewshot-classification/2026-10-31/README.md)，没有另存300份适配副本。

该MAML方案自身64.72%→67.55%，但SVM仍有88.89%；ZY02唯一预测样品在MAML校准前后都未识别。当前为单批次回顾性可行性试验，不能作为已验证的跨批次产地识别器。原有SVM模型保留在`models/zaoren_threeway_20261031/`。
