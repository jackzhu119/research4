# 当前有效数据：以后以本版本为准

生效日期：2026-10-31。版本：`2026-10-31-zaoren-numbered-v1`。

**按用户明确要求，以下7份编号版是酸枣仁后续整理、建模、评价和论文引用的唯一有效输入。此前未编号、命名混乱或编号重复的版本停止使用。** 机器可读入口为[有效工作簿清单](active_workbooks.json)，[校验清单](manifest.csv)只列当前有效文件。

| 文件 | 产地组编号 | 谱记录数 | 基础编号范围 |
| --- | --- | ---: | --- |
| [样本1_编号版.xlsx](processed/zaoren_20261031/样本1_编号版.xlsx) | ZY01 | 350 | ZY01-001 至 ZY01-350 |
| [样本2_编号版.xlsx](processed/zaoren_20261031/样本2_编号版.xlsx) | ZY02 | 5 | ZY02-001 至 ZY02-005 |
| [样本3_编号版.xlsx](processed/zaoren_20261031/样本3_编号版.xlsx) | ZY03 | 100 | ZY03-001 至 ZY03-100 |
| [样本4_编号版.xlsx](processed/zaoren_20261031/样本4_编号版.xlsx) | ZY04 | 100 | ZY04-001 至 ZY04-100 |
| [样本5_编号版.xlsx](processed/zaoren_20261031/样本5_编号版.xlsx) | ZY05 | 100 | ZY05-001 至 ZY05-100 |
| [样本6_编号版.xlsx](processed/zaoren_20261031/样本6_编号版.xlsx) | ZY06 | 100 | ZY06-001 至 ZY06-100 |
| [样本7_编号版.xlsx](processed/zaoren_20261031/样本7_编号版.xlsx) | ZY07 | 100 | ZY07-001 至 ZY07-100 |

酸枣仁合计855个跨文件唯一谱记录编号，四表对应一致，原厚度后缀、坐标、参考值和全部光谱数值保留。编号说明及旧新名称对照见[编号目录](processed/zaoren_20261031/README.md)与[汇总Excel](processed/zaoren_20261031/酸枣仁_编号对照表.xlsx)。编号与入口整理本身不产生模型成绩；实际实验按[已批准方案](../docs/APPROVED_ZAOREN_EXPERIMENT_PLAN.md)执行，结果另行记录。

## 旧版如何处理

7份未编号原始输入从有效输入目录移至[停用旧版归档](archive/zaoren_superseded_20261031/README.md)，逐文件替代关系与SHA-256见[归档清单](archive/zaoren_superseded_20261031/manifest.csv)。归档只用于溯源，不得与新版合并、重复计数或作为默认训练输入。旧文件数值没有被认定错误；本次撤销的是旧命名和旧编号入口。

旧808行元数据模板另存[历史模板](archive/pre_numbering_20261030/README.md)。此前“酸枣仁405条”“合计808条”“仅5/6份”“样本1待补”等叙述不再代表当前数据范围；历史运行报告中的原始数量、路径和hash保留为当时证据。

## 当前清单和待补信息

有效清单还包含未变动的4份黄芪原始文件，共403条。因此[当前元数据模板](metadata_template.csv)合计1258条：黄芪403＋酸枣仁855。默认数据准备和审计流程按有效清单读取，核对文件存在及hash，不递归混入归档或汇总表。

用户已说明每份酸枣仁表对应一个产地，并进一步明确“每个编号均是独立样品”，证据见[用户声明](annotations/zaoren_user_declaration_20261031.json)。当前855条酸枣仁记录据此显式设置`physical_sample_id=record_id`，依据为`user_declared_independent_20261031`，`origin_group_verified=true`；这不是仅凭编号推断的独立性，也不等于已检查原始采样日志。

ZY01—ZY07仍作为匿名产地组标识；真实地名、批次和物种确认等未提供字段留空，黄芪注释不变。当前可开展“用户声明独立样品上的匿名产地组内部验证”，不能据此宣称真实地名核验或跨批次鉴别已经完成。实际批次信息未提供，真实小样本批次校准仍待启动条件满足。

当前科研主线为“酸枣仁产地识别 → 经对照验证的模型改进 → 已知产地新批次小样本校准”，暂不加入Gaussian。

用户随后进一步确认“每个产地目前只有一批”，已存入同一声明记录。当前批次编号仍未提供，不从文件序号伪造；当前数据无法分离产地与这一批来源样品的效应。第一轮样品级内部验证及对照结果见[执行报告](https://github.com/jackzhu119/research4/blob/research/algorithm-review-and-baselines-20261030/docs/ZAOREN_APPROVED_EXECUTION_REPORT.md)，真实跨批次校准需补独立新批次。
