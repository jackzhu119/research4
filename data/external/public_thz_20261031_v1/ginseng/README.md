# 人参／西洋参THz：42条压片观测的可追溯接口

2026-10-31，北京时间。本目录重新整理仓库已有公开数据包，不是新下载的第二个队列。原18个CSV及用户先前上传的RAR仍在`data/external/ginseng_hvbp5mdtxx_v2/original/`，字节不变。

[发布者V2说明](https://data.mendeley.com/datasets/hvbp5mdtxx/2)、DOI10.17632/hvbp5mdtxx.2，Jamil Mai，2025-05-08，National Tsing Hua University（新竹），[CC BY4.0](https://creativecommons.org/licenses/by/4.0/)。本轮重新核对官方元数据；没有取得仓储端RAR校验和，不能称原上传RAR已与远端逐字节验证。

## 可用于什么任务

这是**亚洲人参与西洋参物种二分类**，不是产地分类或酸枣仁新增标签。`1`＝Panax ginseng（亚洲人参），`0`＝Panax quinquefolius（西洋参）。共42个发布者描述的tablet观测，20个1、22个0；6个导出文件组，每组7条，每组均含两类。

`observations.npz`仅有以下六个数组：

- `X_alpha`：42×65，吸收系数，cm^-1。
- `X_n`：42×65，无量纲折射率。
- `axis`：65个原始频率点，0.2–1.6THz，无插值。
- `y`：42个字符串0／1，来自对应label_j.csv，不由光谱或文件名推断。
- `record_ids`：沿用原整理编号G1_S01—G6_S07，仅用于配对，不是原植株ID。
- `export_groups`：G1—G6，仅指文件组，不是已核实生物批次。

只有前两个数组是光谱特征。频率坐标用于解释和格式校验，编号、标签及组号不能作为特征列。

## 实际质量检查

18个CSV分别绑定原SHA，频轴与n／alpha按相同组后缀及列位置配对。全部5460个谱单元有限。alpha有141个负值（最小−45.042、最大66.71773），原样保留；n最小0.1823、最大1.69174，没有按预测正误删列、取绝对值、平滑或填补。

alpha、n及配对数组各861对观测均无精确重复；预先指定中心化cos≥0.99999的近重复诊断也无命中。n最大中心化cos为0.99976193。检查未发现命中不证明样品／植株／原料独立，高相关也不能直接认定生物学重复。

`observation_lineage.csv`共84行＝42观测×2表示，不增加观测数。`physical_sample_id`、`plant_id`、`parent_sample_id`、`preparation_id`、`batch_id`、`measurement_id`保留空白，独立性证据为unknown upstream。不能用整理编号给这些字段补造身份。

## 允许的评价范围

当前科学状态：`EXPLORATORY_OBSERVATION_LEVEL_UNKNOWN_UPSTREAM`。可在明确记录局限的前提下，用6个**导出文件组**进行外层leave-one-export-group-out、内层其余5组LOGO，只能称观察层的导出组留出探索；不能称独立植株／真实跨批次确认。原料、根、压片与生物批次对应未知，真实跨批次及跨植物预训练保持BLOCKED。

2026年[另一人参库](https://data.mendeley.com/datasets/k4pjjjmgnr/1)是4套特征方案和预导出的6折train/test文件。96文件不等于96新样品，也不能在未数值核对时与这42条合并。另有论文独立原根声明是否对应本V2，应按新版本审查证据另核，不能从另一版本直接迁移身份。

## 读取与重建

```python
from herb_spectra.public_thz_data import load_ginseng, prepare_ginseng
arrays = load_ginseng()
# 新目录；拒绝覆盖已有准备产物。
prepare_ginseng('outputs/ginseng-interface-rebuild-new')
```

源码：`src/herb_spectra/public_thz_data.py`。8项实际正反测试通过，包含原CSV逐值、标签／轴／原SHA篡改拒绝、跨组精确重复探针被报告且不删样、unknown身份不自动转PASS、NPZ无损回读及防覆盖。没有训练任何模型。

许可与原件来源：`source_provenance.json`；原件和接口哈希：`manifest.json`；数值质控：`duplicate_qc.json`。原文件和历史中文标注没有改写。
