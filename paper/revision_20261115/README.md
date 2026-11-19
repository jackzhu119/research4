# 最终论文与投稿文件

本版本面向Applied Spectroscopy，使用`paper/current/manifest.json`允许的纠正后分类结果及单独验收的两套原儿茶酸分子计算。错误混种模型、旧预测重贴标签分析和旧混合汇总不进入论文。

作者顺序：Ziyuan Zhu、Jiayi Ren、Shuyu Zhou、Mengting Zhu、Yin Shen。第一作者Ziyuan Zhu的ORCID为https://orcid.org/0009-0005-1153-4926。Yin Shen为末位通讯作者，邮箱shenyin285819016@126.com，ORCID为https://orcid.org/0000-0002-3221-7875。两项ORCID按姓名分别列明，已核验公开记录与校验位。全部作者单位为重庆医科大学人工智能医学学院；英文单位、共同工作地址、邮箱与贡献见[标题页](05_Title_Page.docx)和[作者信息](AUTHORSHIP.json)。

全体作者批准投稿并对准确性和完整性负责；无利益冲突，无资金支持，本稿未在其他期刊同时审理。实测数据由导师及通讯作者提供，也用于其指导的其他学生研究；稿件不宣称独占数据。

## 文件

|文件|用途|
|---|---|
|[01 英文正文](01_Manuscript_Optimized.docx)|期刊主稿，含4张可编辑表格、3个可编辑公式和4张主图|
|[02 英文补充材料](02_Supplementary_Information.docx)|完整方法、正负结果、来源划分和合格分子计算|
|[03 论文优化与清理说明](03_论文优化与清理说明.docx)|中文说明，供作者查看|
|[04 投稿信](04_Cover_Letter.docx)|Cover letter|
|[05 标题页](05_Title_Page.docx)|五位作者、通讯标记、邮箱、单位和ORCID|
|[06 投稿文件说明](06_投稿文件说明.docx)|上传类型、系统元信息和账户操作说明|
|[07 全文中英对照](07_Manuscript_EN_ZH.docx)|逐段正文、声明、图注及4张表的中文对照，供作者审阅|
|[08 独立声明](08_Declarations.docx)|系统要求单独声明文件时使用|
|[Supplementary Data S1](Supplementary_Data_S1.zip)|完整数据来源归档；期刊优先使用6份标准XLSX，归档保留纠正结果、预测/特征、协议、输入划分和两套合格分子模式/IR|
|[投稿元信息](SUBMISSION_METADATA.json)|可复制的题目、摘要、关键词、作者、声明和字数|

四张主图与一张补充图在[figures](figures/)中，由Matplotlib直接按记录的数值表绘制，统一完整外框。生产版提供1200 dpi RGB LZW TIFF及EPS，另备嵌入字体的PDF、PNG和SVG版本。上传时每图选择系统接受的一个版本，避免重复登记。四张可编辑表及四张主图已放回正文对应位置，表格采用简洁横线样式。

期刊补充数据优先使用6份标准XLSX：Supplementary_Tables.xlsx和Supplementary_B1.xlsx至Supplementary_B5.xlsx，均小于10 MiB，保留全部纳入表格的正负结果。来源SHA及逐格读回核查见[数据表核验](SUBMISSION_TABLES_VERIFICATION.json)。完整S1 ZIP保留为来源归档。

最终结构、引用、图件、作者及声明核查见[期刊格式核查](JOURNAL_COMPLIANCE.json)。正文已清除内部修订说明、占位符、评论与修订标记；按Applied Spectroscopy所链接的Sage政策，在致谢保留一条简短工具使用披露，无逐图AI标注。

中英对照保留原始英文参考文献和图内英文坐标、图例，另附中文图注与表格。其内容与最终英文稿对应，参见[双语核验](BILINGUAL_VERIFICATION.json)。

## 实验标注与主线

按照作者最终确认，已撤下旧实验参数补齐表。论文依据真实工作簿标注、导出数组及已确认条件完成，不再要求补填旧七组清单；未记录的制样或软件设置不补造，厚度按表头标称值表述，吸收保持原始导出尺度。

论文动机是发展太赫兹作为常规药材鉴定的补充光谱研究手段。英文题目更新为Terahertz fingerprints and machine learning for complementary discrimination of medicinal materials，参考文献由15篇补充至28篇，均在对应论点处引用。本文提供从已有光谱测量、成对光学参数到机器学习判别的可追踪流程，并通过消融实验明确有效信息和值得继续研究的策略，为后续更多药材的研究提供方法参考。

具体证据连接“候选光谱信息—最终保留变量—有效支持标签”：93.44%为内部嵌套主结果；固定62变量时联合SNV池87.11%相对SNV吸收81.00%，最终930次变量选择全部属于吸收。原始吸收同预算88.78%的对照及全部负结果保留。分子计算提供频率尺度参考，不把分类变量直接指认为实测化学峰。更多药材和独立新批次的效果是后续扩展问题。

## 溯源与打包

`Supplementary_Data_S1/`保留完整来源及冻结证据；历史证据中的原始缺失字段不改写，不构成本次待补要求。精简ZIP是供期刊使用的数据附件，完整证据用于复核。原18份有效光谱工作簿仍由仓库的`data/active_workbooks.json`定位，无需在包内重复复制。

`authoring_sources/`保存英文源稿、翻译映射和生成代码。最终交付压缩包按`01_Submit`（英文上传文件）、`02_Review`（作者中英审阅）和`03_Evidence`（完整溯源）组织，由以下脚本在最终排版核查后构建：

```bash
python paper/revision_20261115/authoring_sources/finalize_submission.py
```

[交付核验](DELIVERY_VERIFICATION.json)记录最终文档结构、页面检查和图片质量；[文件清单](FILE_MANIFEST.json)记录SHA-256。当前分类白名单仍为`paper/current`，本次写作未新增模型训练或量子作业。

文件已备齐，尚未正式提交期刊。ORCID号码已写入文件；投稿账号持有人须在系统完成本人ORCID授权，并查看系统生成的PDF。正式上传方法见06文件。
