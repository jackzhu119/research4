# 其他中药THz公开数据线索核查（2026-10-31）

本目录审查两条与中药主线相关的来源。此次新增可训练数值队列 **0**；没有运行外来代码、没有训练模型、没有联系作者，也没有把论文准确率当作复现结果。已经冻结的人参42条记录接口未改动。

| 线索 | 核验结论 | 实验准入 |
| --- | --- | --- |
| 柴胡／陈皮 ISEMCC（Heliyon 2024） | 原文明确数据需向作者合理申请，未发现公开数值下载 | BLOCKED：缺光谱、标签、批次映射与数据许可 |
| 贝母产地 THz（Chemical Physics Letters 2025） | 只能核验出版社预览；未核验完整数据声明或数值仓库 | BLOCKED：不能把“未找到”写成已确认拒绝公开 |
| jiaoyiping630/spectrum-preprocessing | 实际是通用近红外预处理源码，无随附数值队列 | CODE_ONLY；不能当作贝母THz数据 |

## 对实验设计的影响

[ISEMCC原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10950663/)描述柴胡130片、13个原料批次；陈皮288片、48个原料批次。同批原料分别制10片／6片，其评价采用按批次留一。若未来获得数组，必须保留原料批次分组，不能把各压片默认成独立植株。柴胡类别名称与陈皮年份也不能重命名为酸枣仁产地。

[贝母论文预览](https://www.sciencedirect.com/science/article/abs/pii/S0009261425004920)涉及不同贝母物种，因此不能预设其任务是无物种混杂的纯产地区分。本次直接打开出版社页面返回403，搜索索引可读部分预览；完整独立样品数、批次和数据许可尚未核验。DOI为10.1016/j.cplett.2025.142350。

## 实际检查的源码归属

[官方GitHub仓库](https://github.com/jiaoyiping630/spectrum-preprocessing)固定提交为45c3d67b5479ba82fa972f43eac12c3b1d945c53。已下载198,964字节的官方代码压缩包，只放在忽略的outputs运行目录。ZIP校验通过：10个MATLAB文件、1个README、3张JPEG，CSV／MAT／XLS(X)／NPZ／NPY／HDF5／DAT／TXT等数值扩展名文件均为0；没有明确LICENSE文件，因此未把源码归档重新分发进Git。

example.m引用MATLAB的spectra、NIR和octane示例，并非贝母THz原始数据。README指向Jiao等2020年的近红外预处理论文（10.1002/cem.3306）。聚合搜索页面上的相邻GitHub文字不能证明它是贝母论文的数据仓库。

## 证据文件

- source_audit.json：来源、DOI、许可状态、样品层次和阻塞原因。
- spectrum_preprocessing_tree_inventory.json：实际下载代码包的逐文件路径、字节数与SHA256。
- validation.json：实际ZIP与内容检查；训练次数0，新增数值队列0。
- files_sha256.json：本审查产物校验值。

上述结论限于已检查的一手页面、作者出版列表与固定源码树，不代表证明互联网上不存在其他数据。此审查不改变酸枣仁、人参或已有模型结果。
