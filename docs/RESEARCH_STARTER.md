# 基础代码使用说明

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
    "data/raw/huangqi/黄芪甘肃202605310347.xlsx",
    sheet="吸收系数",
    bounds=(0.1, 1.5),
)
print(table.values.shape)  # (100, 612)，扫描条数 × 频率点
```

该例只是读取已声明区间。坐标单位和区间适用性须按仪器记录及信噪比确认，不能用最终测试结果来反复选区间。读取器重新扫描工作表尺寸、排除参考列，并拒绝不完整/非有限的所选数据；甘肃空折射率表会明确报错。

## 元数据与评价

仓库中的 [metadata_template.csv](../data/metadata_template.csv) 有 808 行谱列记录，科学标签及物理样品 ID 留空。同一物理样品所有扫描必须填相同 `physical_sample_id`，而不是把每个扫描编号当成独立样品。`origin_verified`、物种确认和化学值必须来自采样/检测记录。

若需重新生成空模板，使用不同输出路径以免覆盖已填记录：

```bash
python scripts/build_metadata_template.py --output outputs/new_metadata_template.csv
python scripts/audit_workbooks.py --output outputs/workbook_profiles.json
```

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

修正版 CARS/MC-UVE、分组稳定频段筛选、PLS-DA、SG/MSC、SSA、1D-CNN、批次外测、真实小样本校准、置信区间及文献新颖性检索均属于后续工作。本次只验证基础代码能正确读表和隔离分组，没有训练或宣称一个已验证的中药模型。
