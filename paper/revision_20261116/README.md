# 2026年11月16日修订稿

本轮以用户上传的research4.rar为作者信息与AI披露基线，对Review.zip中的意见逐项核对仓库原始结果。稿件主线为：面向中药材太赫兹辅助识别的多类别光谱选择思路、五来源酸枣仁任务的实际SVM收益、可供其他估计器接入的排名前端，以及消融分析揭示的表示和监督现象。

主分析采用内层选择K：93.44%对同ID全谱SVM的82.67%，提高10.78个百分点，平均保留220.6/612个坐标。全谱LDA的91.78%保留为强对照。这里的改善限于该内部来源判别任务；其他模型的接入能力与已经测量的跨模型收益分别表述。

## 文件

- 01_Manuscript.docx：英文主稿，4张可编辑表、6张主图、8个原生公式、28篇参考文献。
- 02_Title_Page.docx与03_Cover_Letter.docx：标题页与投稿信。
- 04_Supplementary_Information.docx：英文补充材料，包含稳定性、匹配对照、LOWO、MLP诊断及结果适用范围。
- 05_Declarations.docx：系统需要单独声明文件时使用；相同声明保留在正文中。
- 06_Manuscript_EN_ZH.docx：全文中英文对照，供作者阅读。
- 07_Revision_Notes_EN_ZH.docx：中英文修改记录，供作者核查审稿意见的处理。
- Supplementary_Tables.xlsx：更新主数据表；原B1–B5工作簿从上一修订目录按字节复制到投稿包。
- figures：按实际数据及流程绘制的EPS、SVG和PNG；所有图均有完整外框。

## 证据和复现

先运行仓库要求的current及withdrawn检查。新纳入的focused-followup源固定为5594d9c3326fcd607ee72fb7fc93895047de8f8a。主分析和新全谱对照的15组内外层ID相同；E1的900条预测来自300份实物。LOWO使用不同的固定全谱流程，二类98.84%，四类65.61%，不与93.44%拼接。完整正负结果保留。

authoring_sources/collect_sources.py显式导出固定Git对象；默认导出至outputs/revision_20261116/evidence。然后运行audit_evidence.py重算保存预测和掩码的统计量，再运行build_figures.py、build_documents.py及build_supplementary_tables.py。上述脚本不训练科学模型。生成文档后必须完成渲染核查，再运行finalize_submission.py打包。

EVIDENCE_MANIFEST.json、analysis/EVIDENCE_AUDIT.json及各verification文件记录来源、哈希和核查范围。扩展worker的首档gamma缓存缺陷已确认，受影响的性能和成本主张没有进入主比较。原分子计算不再用于本稿叙述，受保护的历史计算文件及旧版Supplementary_Data_S1保持不变。

期刊格式依据归档的Applied Spectroscopy官方指南及可访问的Sage通用说明检查。当天期刊专页返回403，不把归档指南冒充为当天重新获取的页面。系统账号授权ORCID及最终生成PDF由投稿账号持有人完成；本任务没有提交期刊。

下载使用聊天附件的英文短文件名；旧GitHub下载别名不代表本修订版。
