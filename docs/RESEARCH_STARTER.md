# 基础代码使用说明

> **2026-10-31起，以[有效工作簿清单](../data/active_workbooks.json)及[7份酸枣仁编号版](../data/processed/zaoren_20261031/README.md)为准。** 酸枣仁855条谱记录；旧未编号输入位于`data/archive/zaoren_superseded_20261031/`，不参与默认数据准备。当前研究主线暂不加入Gaussian。

算法问题及优先路线分别见 [算法审读](ALGORITHM_REVIEW.md)、[实验协议](RESEARCH_PROTOCOL.md)。原始资料仍位于 `references/`，本目录的新实现没有复制原压缩包源码。

后续已开展 [Octave/Python 云端可用性实测](RUNTIME_USABILITY_REPORT.md)，其中包含原算法入口运行、明确标注的兼容/逻辑补丁和 CNN 结构迁移。实验场景脚本位于 `scripts/runtime/`，与下面的正式基础接口分开记录。

## 安装与工程验证

在仓库根目录运行：

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

已有依赖时也可用 `PYTHONPATH=src python -m unittest discover -s tests -v`。测试包括错误 XLSX 尺寸声明、参考列排除、空表拒绝、重复扫描分组、训练统计量隔离、support/query 隔离、标签冲突，以及合成数据上的 SVC/SVR/PLS 流程。合成测试结果不是中药模型性能。

## 读取当前工作簿

```python
from herb_spectra.io import read_spectra

table = read_spectra(
    "data/processed/zaoren_20261031/样本1_编号版.xlsx",
    sheet="吸收系数",
    bounds=(0.1, 1.5),
)
print(table.values.shape)  # 谱记录条数 × 所选频率点数
```

该例只是读取已声明区间。坐标单位和区间适用性须按仪器记录及信噪比确认，不能用最终测试结果来反复选区间。读取器重新扫描工作表尺寸、排除参考列，并拒绝不完整/非有限的所选数据；甘肃空折射率表会明确报错。

## 元数据与评价

仓库中的[metadata_template.csv](../data/metadata_template.csv)按当前有效清单包含1258条谱列记录：黄芪403条，酸枣仁855条。酸枣仁对应ZY01为350条、ZY02为5条、ZY03—ZY07各100条，使用编号版工作簿的表头定位。用户声明每个文件对应一个产地，但ZY文件组编号不替代真实地名和采样记录；真实物理样品ID、重复扫描关系、批次、物种确认及化学值仍待补。同一物理样品所有扫描必须填相同`physical_sample_id`，不能直接把每个ZY谱记录编号当作独立物理样品。旧808行模板仅供历史追溯。

若需重新生成空模板，使用不同输出路径以免覆盖已填记录：

```bash
python scripts/build_metadata_template.py --output outputs/new_metadata_template.csv
python scripts/audit_workbooks.py --output outputs/workbook_profiles.json
```

这两个入口默认按`data/active_workbooks.json`选取工作簿，避免递归混入旧版或重复计数。检查历史文件时须显式指定范围，并将其结果与当前清单结果分开保存；不能把旧版与编号版拼接建模。

经人工核验并完成单位、同一频率网格与标签对齐后，可保存 NPZ：`X` 为扫描×特征的有限浮点矩阵，`y` 为每扫描一个真实产地标签或化学实测值，`groups` 为每扫描对应的物理样品 ID，`labels_verified` 为显式布尔真值。字符串数组使用常规 Unicode dtype，读取时禁止 pickle。

当前不自动从空模板生成训练数据，也不自动从文件名构造标签。使用已核验数据时：

```bash
python scripts/evaluate_baseline.py data/processed/reviewed.npz --model svc --output outputs/svc.json
python scripts/evaluate_baseline.py data/processed/reviewed.npz --model pls --output outputs/pls.json
python scripts/evaluate_baseline.py data/processed/reviewed.npz --model svr --output outputs/svr.json
```

训练集内层调原谱/SNV、C/gamma 或 PLS 潜变量数；外层物理样品分组评价，并保存每折训练/测试 group ID 和参数。该基线内层搜索以扫描级分数选参；重复次数不均衡时，应增加样品等权训练/选参对照或先按预设规则聚合重复扫描。最终报告同时给出样品级与扫描级指标，分类样品级采用多数投票（平票按标签排序决定），回归采用同一样品预测均值。

这些命令只评估数据集内的分组泛化，**不等于留批次/年份/仪器外测**。跨域实验需按实验协议先冻结源域与目标 query，并检查样品重叠。`support_query_split` 只负责分类目标域按独立样品划分校准池与测试池，尚未实现校准模型。

## 还需要实现/确认

CARS-style／MC-UVE、PLS-DA、重复筛选及分类分数校准已有独立工程实现，后续外部实测及其边界见[阶段二报告](RESEARCH_STAGE2_REPORT.md)。这些工程候选及旧数据运行不等于当前855条酸枣仁数据已经完成科学建模。当前仍需补足物理样品与批次映射，完成样品级基线、候选改进消融、真实新批次小样本分类校准及不确定性评价；Gaussian暂不纳入本轮主线。
