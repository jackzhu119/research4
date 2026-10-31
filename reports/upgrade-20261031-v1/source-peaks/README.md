# 每外折源内的实验候选峰：描述性结果

2026-10-31。仅分析旧5个outer各自684份训练α；每折612个坐标点。既有855标签字段未读取，分析函数不接收测试矩阵或标签。**五折各检出4个候选，合计20条重叠折记录；不是20个独立化学峰，也不是成分鉴定或模型涨分。**

## 运行前固定的规则

- 当前外折源684条原始均值谱；不按类别求均值，不用全855均值。
- SG窗口15、三阶多项式、interp边界。
- prominence为当前源**原始均值谱**峰峰值的5%。
- 最小间隔0.03THz，按原轴median步长向上取整为14格。
- 坐标换算`1THz=33.3564095198cm^-1`；这是横轴波数换算，不是纵轴α单位确认。
- 参数、源码、输入、旧分区及三候选登记表的hash先写[协议锁](protocol_lock.json)，然后执行。未按所得峰调整阈值。

| 外折（展示从1起） | 源样品数 | 候选数 | prominence阈值（原导出纵轴单位） | 候选图 |
| --- | ---: | ---: | ---: | --- |
| 1 | 684 | 4 | 3.736892 | [PNG](outer_00/source_candidates.png) / [PDF](outer_00/source_candidates.pdf) |
| 2 | 684 | 4 | 3.752619 | [PNG](outer_01/source_candidates.png) / [PDF](outer_01/source_candidates.pdf) |
| 3 | 684 | 4 | 3.733766 | [PNG](outer_02/source_candidates.png) / [PDF](outer_02/source_candidates.pdf) |
| 4 | 684 | 4 | 3.728418 | [PNG](outer_03/source_candidates.png) / [PDF](outer_03/source_candidates.pdf) |
| 5 | 684 | 4 | 3.737937 | [PNG](outer_04/source_candidates.png) / [PDF](outer_04/source_candidates.pdf) |

[按折候选CSV](experimental_candidates_by_fold.csv)保存位置、峰高、prominence、基底坐标及波数；[源折汇总](source_fold_summary.csv)保存阈值。各`outer_XX/`另含684源ID／指纹／元数据和完整612点均值／平滑曲线。源集合高度重叠，不对各折计数作独立统计。

## 不能从这些候选得出什么

这些点只是预定数学规则找到的局部峰。仪器、水汽、制样、基线及平滑效应尚未排除，不能命名为某个分子吸收、证明产地机理或反推化学含量。纵轴保持原导出单位，绝对吸收单位和仪器仍待核。

没有跨折合并候选、构造分类mask、将其送入当前CNN／UVE模型或回头挑本版模型。既有96.02%及新神经流程的输入／权重未因本分析改变。

## 三种化合物的理论配对仍未执行

[理论配对状态表](theory_pairing_NOT_RUN.csv)包含每外折×三候选共15行：Spinosin、Jujuboside A、Jujuboside B。完整分子的DFT均`NOT_RUN`，3D结构校验仍`BLOCKED_PENDING_REVIEWED_3D_STRUCTURE`；实验配对位置、理论峰和assignment字段全部为空。

水分子环境smoke没有用作三候选理论峰；未匹配虚构频率或将公式正常模式计数当作已计算频率。[冻结候选登记快照](candidate_registry_snapshot.json)。

## 实际隔离与检查范围

`FitScope.operation`逐折记录684源身份及实际均值/检峰数据指纹；[fit范围记录](fit_scope_manifest.json)。加载器为解压旧NPZ会在内存物化原X数组，再仅将旧split对应684行传给分析函数，没有读取y；这是API级源隔离，不宣称操作系统文件访问沙箱。旧split与批准实验逐字节相同。

[独立数值审计](independent_source_peak_audit.json)：12454项检查、最大差0，重新由每源矩阵核均值、固定SG、候选位置/height/prominence、单位换算、源ID、fit范围及空理论配对。没有新增分类器拟合，也未确认生物学独立性。

4项[实际测试日志](source-peaks-tests.log)通过：THz／cm^-1往返、单条常数源NO_PEAKS、已知双峰合成fixture、scope越界及禁止标签API。合成fixture只用于软件单元测试，不是新增真实光谱。五张PNG均已view_image目检，无裁字或图例遮挡。

重现至新目录：

```bash
.runtime/bin/python scripts/analyze_upgrade_source_peaks.py --output outputs/source-peaks-reproduction-new
```

程序拒绝覆盖非空目录；依赖旧NPZ缓存、旧split及冻结候选登记。实际命令起止另记录在父目录`command_invocations.jsonl`，原执行stdout见[execution.log](execution.log)。
