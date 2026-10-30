# 人参太赫兹数据：原包检查与中文标注

2026-10-30，用户上传 `Dataset.rar` 后实际解压检查。包中确有标签，但原谱文件只在第一列写了 `Frequency (THz)`，其余样品列没有文字表头；6个标签文件也只有数字，没有解释文字或README。

## 标签在哪里

| 标签文件 | 原始7个数字 | 对应谱文件 |
| --- | --- | --- |
| `label_1.csv` | 1,1,1,0,0,0,0 | `data_n1.csv`、`data_a1.csv` |
| `label_2.csv` | 1,1,1,1,0,0,0 | `data_n2.csv`、`data_a2.csv` |
| `label_3.csv` | 1,1,1,0,0,0,0 | `data_n3.csv`、`data_a3.csv` |
| `label_4.csv` | 1,1,1,1,0,0,0 | `data_n4.csv`、`data_a4.csv` |
| `label_5.csv` | 1,1,1,0,0,0,0 | `data_n5.csv`、`data_a5.csv` |
| `label_6.csv` | 1,1,1,0,0,0,0 | `data_n6.csv`、`data_a6.csv` |

[发布者说明](https://data.mendeley.com/datasets/hvbp5mdtxx/2)及[版本比较页](https://data.mendeley.com/datasets/compare/hvbp5mdtxx)解释：**1＝亚洲人参（Panax ginseng），0＝西洋参（Panax quinquefolius）**；n为折射率、α为吸收系数cm⁻¹。包内本身没有解释这组编码；中文名称及单位取自外部来源说明。

整理时按相同文件后缀和组内顺序配对：标签第1–7项对应谱文件第2–8列，即 Excel 的 B–H 列。例如第1组 B、C、D 列为标签1，E、F、G、H列为标签0。来源没有原始样品ID对照表供独立核验身份，此顺序规则不能证明每片来自独立植株。

## 实际检查结果与交付文件

- 解压得到18个CSV：6个n、6个α、6个标签；无README或原始样品身份表。
- 每个光谱CSV有1行表头、65行数值、8列；12个文件频率列一致、严格递增，范围0.2–1.6 THz；读到的光谱数值均有限。
- 标签共42项：20个1、22个0。它们是42片压片的标签，不是已确认的42个独立植株或批次。
- [中文标注Excel](annotated/人参太赫兹数据_中文标注.xlsx)包含说明、样本标签、原始标签排列、合并n、合并α、原CSV校验；[sample_labels.csv](annotated/sample_labels.csv)保留每片整理编号与原文件／列位置。
- `G1_S01` 等是本次生成的整理编号；产地、原植物ID、批次字段留空。没有生成或猜测这些缺失标签。
- 两张合并光谱表的65×43数值（含频率列）已经与原CSV逐项复核，绝对误差不超过1e-12。未平滑、筛选、截断负值或填补数据；未运行模型。
- 完整文件哈希与检查摘要见 [inspection.json](annotated/inspection.json)。

**本包适合研究人参／西洋参物种二分类，不提供生长产地、年限、绝对成分含量及原始样品／参考时域波形，不能用作酸枣仁产地训练标签。**

## 原件、来源与复现

`original/` 保留用户上传RAR及解压CSV的原始字节。RAR为24,212字节，SHA-256：`3b1bbcff42c0825361c372cab329abee2c21c29d3b4b30ac7c1366865e7c3fcc`。上传包结构与所引V2仓储说明一致；未取得仓储端校验和进行独立字节比对。

引用：Mai, Jamil (2025), “Complex Refractive Indices of Ginseng/American Ginseng Tablets”, Mendeley Data, V2, DOI: [10.17632/hvbp5mdtxx.2](https://doi.org/10.17632/hvbp5mdtxx.2)。仓储许可为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。本仓库新增中文解释、整理编号和合并工作簿；保留原始CSV，不增加原研究未提供的身份信息。

从解压CSV重建标注文件（依赖NumPy和openpyxl）：

```bash
python scripts/annotate_ginseng_dataset.py \
  data/external/ginseng_hvbp5mdtxx_v2/original \
  data/external/ginseng_hvbp5mdtxx_v2/annotated
```
