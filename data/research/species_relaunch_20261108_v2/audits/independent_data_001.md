# 独立数据与固定划分核验

状态：PASS_METADATA_IDENTITY_RAW_SHA_CRC_AND_FIXED_ROLE_AUDIT。本次26964项检查，失败0项。

18份当前工作簿的字节数、SHA-256、ZIP全部成员CRC、四信号表结构通过；1672个record_id/physical_sample_id唯一，元数据、固定划分与有效清单逐ID一致。原855个角色逐ID保持不变；当前999训练／334校准／339预测。

| 种类 | 记录 | 训练 | 校准 | 预测 |
|---|---:|---:|---:|---:|
| 酸枣仁 | 500 | 300 | 100 | 100 |
| 枣仁 | 364 | 216 | 74 | 74 |
| 黄芪 | 403 | 241 | 80 | 82 |
| 天麻 | 405 | 242 | 80 | 83 |

ZY01/02为355份枣仁；ZY03—07为500份酸枣仁。新枣仁ZR01—03每组3份，分别1/1/1；ZY02为3/1/1。原855独立性声明与新817声明分别保留；所有batch_id为空，没有补造新批次。

- Kind/origin and physical independence are user declarations, not independent Latin botanical identification or sampling-log audit.
- Original ZY origins each have one user-declared lot; no new batch IDs supplied. Internal specimen-disjoint roles do not prove true cross-batch calibration.
- 山西 and 山西灵石 labels are retained separate source groups; geographic mutual exclusion is not established.
- Three added jujube groups each contain only3 specimens; 1/1/1 role split is diagnostic, not a stable origin-validation sample size.
- ZIP integrity/file hashes and metadata agree; this audit did not inspect full numeric spectra or independently establish absence of numeric duplicate columns. Root feature audit covers the numeric arrays.
- Four workbook signal views belong to the same records and are not fourfold samples.

实际CPU约2.804506秒；未拟合、推断、量子计算或运行Git，未改原文件。
