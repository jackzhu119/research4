# 2025人参THz数据独立性及输入接口审查

实际读取18个CSV、42条压片观测、65点频率，两表示配对；官方V2元数据已重新核对。没有模型训练，没有新增远程数值下载，没有新增独立植物计数。

数据接口：[准备目录](../../data/external/public_thz_20261031_v1/ginseng/README.md)。

- 发布者记录：Mai,Jamil(2025),Complex Refractive Indices of Ginseng/American Ginseng Tablets,MendeleyDataV2,DOI10.17632/hvbp5mdtxx.2，CC BY4.0。
- α负值141个全部保留，n无负值；所有谱数据有限。
- 三组QC（α／n／配对）各861对均无exact或预定cos≥0.99999命中。n最大cos0.99976193。无命中不是祖先独立性证明。
- 标签0为西洋参、1为亚洲人参，仅物种二分类；没有生长产地或含量目标。
- 物理、根、原料、制样和真实批次身份未知；42是观测／发布者描述的压片数，不是已核实独立根数。
- 文件组G1–G6每组7、两类别齐全，可作技术导出组留出探索；不能把导出组叫真实批次。
- 2026新库的四scheme／六折导出不可直接追加到旧42；需要单独重叠和身份对应证据。

`unit_tests.log`为8项真实正反接口测试；`command_events.jsonl`为实际命令UTC。所有旧原件SHA在准备前后相同。`validation.json`和两目录SHA索引记录新交付，不宣称绝对零泄漏或生物独立性PASS。
