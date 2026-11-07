# 实验室与盲测准备包

状态：**PREPARATION_DONE；全部采样／采购／THz标准品／HPLC／LC-MS均LAB_REQUIRED；新增模型及理论四臂NOT_RUN。** 本目录不含真实新增样品、化学测量、理论频率或模型结果。

操作说明见[实验室与盲测交接](../../docs/RESEARCH4_LAB_AND_BLIND_VALIDATION_20261105.md)，V7控制元数据勘误另见[独立勘误](../../docs/PYSCF_V7_METADATA_ERRATA_20261105.md)。

## 空CSV

所有`templates/*.csv`均UTF-8 BOM编码，只有一行列名，**零数据行**。未填项保持空白；采集开始后明确UNKNOWN／TO_VALIDATE，不补猜测值。状态列填实际LAB_REQUIRED、NOT_RUN、MEASURED、QC_FAILED等事实，不把准备当测量。

| 表 | 保管方式与用途 |
| --- | --- |
| samples_private、batches_private | 私有：真实物理单位、匿名组、来源批次、species_name／voucher_id／authentication_basis；无凭证时UNKNOWN |
| blind_samples_public | 公开给模型操作者：不含ZY、地名或原料来源线索的opaque ID及最少QC信息；taxonomy保持UNKNOWN或经盲态审查的一致信息，不作特征 |
| preparations、aliquot_links | 扫描／压片技术重复与同原料分样血缘；THz和化学测量的实际配对依据 |
| standards | 标准品批号、CoA／纯度、固相、水合及身份依据；没有采购或测量的虚构行 |
| instrument_sessions、spectral_records | 原始时域／参考／轴及单位、SOP、仪器日、厚度水分、原件hash、QC和技术重复 |
| chemical_methods | 柱／流速／温度／波长／浓度等未可靠验证时TO_VALIDATE；需引用确切来源并验证本基质 |
| chemical_measurements | 实际同物理样品／aliquot的身份、定量与方法QC；低于LOQ不能自行填0 |
| role_allocation_private、blind_truth_private | 由真值保管者封存，不进入公开query包；支持与查询及共同原料严格隔离 |
| predictions_unscored | 无真值预测，全部模型／支持抽样完成后锁定，再释放真值 |
| blind_outcomes_after_release | 只在解盲后记录改错／改对与独立评分，不反过来选参 |
| sampling_plan | 仅规划目标，必须声明功效分析状态；不当作实际样品N |
| chain_of_custody | 实物／文件交接、封条及时间追踪 |

`template_columns.json`给出机器可读列名。字段名带`private`或表名带`private`的内容不能随本模板说明直接公开。即使表名不含private，parent、制样日志或仪器批次也可能泄露来源，公开盲包须由保管者检查并opaque化。

## 两个结构schema

- `blind_handoff.schema.json`：冻结模型／参数／source IDs／轴／query包后，锁预测，再解盲，再评分；结构要求私有真值只保留SHA承诺、不公开私有路径。
- `theory_four_arm.schema.json`：仅允许在真实完整219模式、全正频、合格独立分析及健康链后注册原MC／理论／联合／匹配随机联合四臂。

这两个文件是schema，**不是已冻结运行manifest**。JSON Schema的结构验证不证明时间先后、实际文件hash、盲态权限或物理独立性；未来执行需要独立门禁审核。当前没有伪造manifest实例。保护私有真值的权限／封存由实验室保管者实际落实。

## 静态准备验收范围

`preparation_receipt.json`只验证本目录空表列数／无重复列／零数据行、schema JSON可解析和固定四臂名称，以及原V7源码／旧MD SHA保持不变。它不是实验测量或任何Q数值验证。`artifact_index.json`记录本准备包及对应两份新MD的SHA与大小，方便后续交接追溯。
