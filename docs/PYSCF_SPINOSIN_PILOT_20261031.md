# 酸枣仁斯皮诺素PySCF分子振动试验

研究分支：`research/pyscf-spinosin-mechanism-20261031`。基于`3fc996e99cce8ff5799b5a2af9b565f982dfc899`，保留此前原始光谱、分类模型与历史报告。运行编号沿用用户指定日期20261031；实际运行记录使用UTC，日期为2026-11-01。

**这是开源PySCF云端实算，不是Gaussian计算。水分子的完整优化和频率验证已成功；斯皮诺素完成了4个真实SCF／核梯度几何评估，但未几何收敛，随后按资源保护安全停止。没有运行斯皮诺素Hessian，没有取得其理论频率，也不能作化学峰归属。**

全部量子阶段已结束，累计实际2666.234148 CPU秒（44.4372 CPU分钟），未启动任何后台续算。原3300秒量子预算余633.765852秒，未重置或扩大。

| 阶段 | 最终状态 | 实际证据／限制 |
|---|---|---|
| PySCF环境与方法核查 | DONE | 独立依赖、版本、CPU／内存及Hessian实现核查 |
| 水分子Opt/Freq | DONE | 两个DF-J路径均优化收敛，3个完整正频率 |
| Spinosin可信图与12构象 | DONE | 75原子／10指定手性中心；真实ETKDG／MMFF初筛 |
| Spinosin SCF及核梯度 | DONE（子阶段） | 4个完整配对能量／梯度点；不是完整研究完成 |
| Spinosin几何优化 | PARTIAL | 第5点梯度期间按CPU保护停止，4个完整点均未过梯度阈值 |
| Spinosin Hessian／频率 | NOT_RUN | BLOCKED：几何未收敛，资源有限；不存在可解释的219模式结果 |
| 理论—实验峰归属 | BLOCKED | 对照表保留20个实际实验候选，理论值和差值为空 |
| 皂苷A／B结构与规模审查 | DONE | 立体图、MMFF覆盖及存储规模；A身份歧义保留 |
| 皂苷A／B DFT | NOT_RUN | 当前缓存方案超16 GiB；没有启用更大计算 |
| 全部分子IR强度 | NOT_COMPUTED | 未实现／验证额外偶极导数，不生成吸收强度曲线 |

## 科学边界

本轮只研究孤立分子振动的可行性，不改变分类训练、样品标签、特征或参数。已审计MC-UVE＋SVM的96.0234%准确率和0.9197 Macro-F1保持其原有内部开发评价性质；本轮不声称改善它们。

`DONE`表示相应阶段实际完成；`PARTIAL`表示真实执行但未完成整个阶段；`NOT_RUN`表示没有执行；`BLOCKED`表示前置条件或资源不足。获得收敛电子能量不等于几何优化完成，得到检查点也不等于通过驻点或频率检查。

所有IR强度均为`NOT_COMPUTED`。PySCF标准`harmonic_analysis`不直接给出IR强度，本轮没有实施额外偶极导数方法，也没有生成伪造的理论吸收曲线。

## 环境与实际方法

[环境清单](../reports/pyscf-spinosin-pilot-20261031/environment/environment.json)、[依赖下载及构建哈希](../reports/pyscf-spinosin-pilot-20261031/environment/dependency_artifacts.json)。Python3.12.14、PySCF2.14.0、geomeTRIC1.1.1、RDKit2026.9.1、NumPy2.3.5、SciPy1.17.0、h5py3.16.0、LibXC7.0.0。

新增包安装到独立`.runtime/quantum-pilot-site/`，复用`.runtime/quantum-upgrade-site/`内PySCF，不改主机器学习环境。没有调用Gaussian、GPU或付费服务。可见5核、cgroup CPU配额4核、内存16 GiB；启动时磁盘可用约9.35 GiB。每个量子worker限制1线程、地址空间12 GiB；最初direct-J v1的PySCF内存参数4000 MB，缓存DF-J v2为8000 MB。

实算方法固定为：**PBE/def2-SVP、球形全电子基组、DF-J辅助基def2-universal-jfit、积分网格level1、电荷0、spin=0（单重态模型）**。没有D3修正，不能标为PBE-D3或B3LYP-D3(BJ)。`pyscf-dispersion`未安装；对源码的核查显示相关色散Hessian可能通过梯度有限差分实现，但本轮没有调用它。单重态与数据库中性闭壳层结构相符，不等于已经搜索所有自旋态。

SCF收敛阈值为1e−8 Eh、最多80循环。geomeTRIC最多35步；能量、原子矢量梯度RMS、最大梯度、位移RMS和最大位移阈值分别为1e−6 Eh、3e−4 Eh/Bohr、4.5e−4 Eh/Bohr、0.0012 Å、0.0018 Å。梯度使用PySCF默认`grid_response=False`。该粗网格、无色散、单构象方法是资源受限的初步试验，不是定量低频峰归属的最终精度方案。

本文原子矢量梯度RMS定义为`||G||F / sqrt(Natom)`，最大梯度为`max_i ||G_i||2`；它们不是将3N个分量直接平均的RMS及最大单分量。结果同时保存两套统计，判断阈值使用前者。

最初direct-J路径真实运行后因速度不足停止，保留全部日志和检查点。缓存DF-J v2在水分子上复测后才用于斯皮诺素；它使用同一泛函、基组及网格，每个几何点重新建立缓存，改变的是计算路径，不能计作物理方法创新。斯皮诺素762个轨道AO、2459个辅助AO；实际缓存2459×290703个float64，5,718,709,416字节（5.326 GiB），未写入报告或Git。

## 结构与构象证据

[结构记录与来源](../reports/pyscf-spinosin-pilot-20261031/structures/README.md)、[构象生成执行记录](../reports/pyscf-spinosin-pilot-20261031/conformer-execution/README.md)。斯皮诺素记录来自[ChEBI:81360](https://www.ebi.ac.uk/chebi/CHEBI:81360)，交叉引用PubChem CID155692，InChIKey为`VGGSULWDCMWZPO-ODEMIOGVSA-N`。公式C28H32O15、75原子、320电子、电荷0、自由基电子0、10个指定手性中心。

直接请求官方结构下载端点遭代理403，失败回执保留。没有冒称取得官方或实验3D坐标。已从官方页面取得完整立体SMILES/InChI并保存快照，再以RDKit ETKDGv3生成可追溯模型坐标，MMFF94s作初筛。12/12候选在750迭代上限内收敛，筛选CPU约2.56秒；坐标反推的立体InChIKey均与可信来源一致。

输入选为C001，MMFF94s初筛能量227.2838852 kcal/mol。这是12个已生成候选中的最低力场能量，不是DFT能量或全局最低构象证明。逐构象随机种子、参数、初始与筛选坐标、能量和两两重原子RMSD均已保存。C001的XYZ SHA-256：`4567bbacd7a1881e20d9ef823e241757a2c479d49ee20c98cb206c4aeeddb588`。

## 水分子完整Opt/Freq验证：DONE

[direct-J原始结果](../reports/pyscf-spinosin-pilot-20261031/water-optfreq/result.json)、[缓存DF-J结果](../reports/pyscf-spinosin-pilot-20261031/water-cached-optfreq/result.json)、[保存Hessian独立复算](../reports/pyscf-spinosin-pilot-20261031/method-audit/water_saved_result_review.json)。

两次均完成4个几何评估、优化收敛、新SCF与最终梯度检查、DF-RKS Hessian和去除平移转动后的正常模式分析。缓存后端结果如下：

| 模式 | cm⁻¹ | THz | IR强度 |
|---|---:|---:|---|
| 1 | 1605.356083 | 48.127365 | NOT_COMPUTED |
| 2 | 3688.836257 | 110.588529 | NOT_COMPUTED |
| 3 | 3787.669529 | 113.551476 | NOT_COMPUTED |

能量−76.27262382116325 Eh，最终原子梯度RMS0.000136759741、最大0.000231040863 Eh/Bohr；预期3N−6=3，实际3个正频率、0虚频，完整模式及正频率检查通过。Hessian最大置换非对称差约1.04e−7 Eh/Bohr²。两后端能量差约1.42e−14 Eh，最大频率差9.64e−5 cm⁻¹；使用保存的完整精度检查点和Hessian独立重对角化与原direct-J频率一致。

最初用12位小数XYZ重建几何的超严格复核未通过，重投影出现约1.36e−4 cm⁻¹的微小数值路径差异；失败尝试保留，改用原检查点完整精度几何后与原输出一致。额外[小矩阵诊断](../reports/pyscf-spinosin-pilot-20261031/delivery-preflight/water_projection_numerical_diagnostic.json)在仅用于诊断的对称化Hessian上将该差异降至约1e−12 cm⁻¹，说明差异涉及原Hessian微小非对称及投影数值路径，不能解释为分子物理频率对坐标舍入有同等敏感性。没有修改已保存Hessian或原始频率，也不以这项数值复核代替网格／基组精度研究。

这些是软件流程验证数据，不是斯皮诺素或药材振动，也不用于解释1 THz附近的实验峰。

## 斯皮诺素真实执行与停止条件

[累计CPU账本](../reports/pyscf-spinosin-pilot-20261031/cpu_budget.json)、[direct-J失败／部分结果](../reports/pyscf-spinosin-pilot-20261031/spinosin-scf-gradient/result.json)、[缓存优化日志](../reports/pyscf-spinosin-pilot-20261031/spinosin-cached-optimization/pyscf.log)。

为遵守用户总CPU上限3600秒，量子计算集中账本上限3300秒，另留300秒给准备及测试。账本以`wait4`实际user＋system CPU累计，拒绝并发量子阶段、重复阶段名、重用输出和自动扩大预算。运行中台账的`consumed`仅表示已结束阶段，不能作为实时剩余值；活动进程CPU需要另行叠加。

300秒是准备与测试预留，不是逐项测得的总CPU。最终冻结测试另行记录实际子进程CPU39.73秒，构象初筛约2.56秒；环境准备及早期测试的计时证据各自保留，不把量子账本误称为整个会话全部进程CPU实测。

最初direct-J尝试实际使用350.716438 CPU秒，未完成SCF/梯度阶段，状态PARTIAL。之后缓存路径从其检查点真实恢复，完成收敛SCF与逐几何核梯度，实际几何能量和梯度保存于`geometry_steps/`。尚未收敛时不得启动Hessian。

缓存优化有2400 CPU秒、3000秒墙钟上限，并设2310 CPU秒提前SIGTERM保存请求，避免只在硬上限前1秒才开始清理。提前请求不增加预算。若信号在底层计算段中等待处理，仍由内核硬上限兜底；最终实际状态以工作进程结果和父进程账本为准。

实际SIGTERM请求发生在进程CPU2311.16秒，worker成功保存PARTIAL；父进程`wait4`实测缓存阶段2312.144455 CPU秒，墙钟2313.070288秒，峰值RSS8,583,064 KiB（约8.19 GiB）。未触及2400秒硬上限，也没有因内存不足失败。具体UTC时间、命令、输出及哈希均在账本。几何优化阶段于2026-11-01 05:02:52 UTC结束。

| 完整评估点 | SCF能量／Eh | 原子梯度RMS／Eh·Bohr⁻¹ | 最大原子梯度／Eh·Bohr⁻¹ | 梯度阈值 |
|---|---:|---:|---:|---|
| 1 | −2210.4097641546923 | 0.0174866387484 | 0.0427715963020 | 未通过 |
| 2 | −2210.4234255782210 | 0.0129376886742 | 0.0294785511607 | 未通过 |
| 3 | −2210.4299551605445 | 0.00795537466865 | 0.0355764069933 | 未通过 |
| 4 | −2210.4359604904650 | 0.00703513281033 | 0.0329667830409 | 未通过 |

第4点RMS仍约为阈值23.45倍，最大原子梯度约73.26倍。轨迹中的RMS下降并不表示所有梯度统计单调下降。第5点日志记录SCF能量−2210.44035720308 Eh，但其梯度尚未完成即中止，不能将它与第4点梯度配对。结果JSON保留最后完整几何评估的能量／梯度；检查点可包含第5个试探几何，SCF状态字段也不能单独当作当前驻点证明。

恢复准备实际核对了检查点完整精度几何及其立体身份。它与第4点对齐后的75原子RMSD约0.14165 Å、重原子RMSD约0.12935 Å，确有内部几何变化，不只是坐标系平移／旋转。因此不能把第4点能量／梯度标在恢复XYZ上，必须重新计算。

完整Spinosin非线性分子的预期振动数219，只是3N−6算术，不是已计算模式数。本轮实际计算出的Spinosin频率和正常模式均为0条记录，含义是未计算，不能推断该分子在0.1—1.5 THz不存在振动。剩余预算不支持保证完成目前尚未收敛的优化及完整Hessian，因此没有自动追加阶段。

## 皂苷A／B：准备DONE，DFT NOT_RUN

| 候选 | 分子式／原子数 | 完整指定结构参考 | 指定手性中心 | def2-SVP AO | 当前缓存张量规模 |
|---|---|---|---:|---:|---:|
| Jujuboside A | C58H94O26／178 | PubChem CID51346169 | 34 | 1646 | 52.010 GiB |
| Jujuboside B | C52H84O21／157 | PubChem CID24721031 | 29 | 1442 | 34.890 GiB |

这两个完整图的MMFF参数覆盖均通过，strict重原子图可转动键分别13与10；没有运行它们的构象搜索、SCF、几何优化或频率计算。A另有CID171446记录包含两个未指定立体中心，与CID51346169不可静默等同，后续必须对照实际纯品与文献确定具体结构。

表中存储是按实际基组维数计算的满秩float64缓存，不是所有DFT方法的最低内存要求，也不是运行时间估计。当前方案仅缓存已超本机16 GiB内存及初始可用磁盘。direct-J等替代路径可能减少存储，但在Spinosin上的实测速率已不足以支持本轮大分子频率工作，因此完整A/B计算不启动。

## 理论与实测THz谱的关系

实验来源是历史[逐训练折候选峰](../reports/upgrade-20261031-v1/source-peaks/experimental_candidates_by_fold.csv)，实际20条记录，大致对应1.08、1.13—1.14、1.18—1.19、1.28—1.30 THz。后处理保留每条来源和频率，不依据测试准确率或峰位接近挑选理论结果。

0.1—1.5 THz约等于3.3356—50.0346 cm⁻¹。没有通过优化与完整正常模式检查的Spinosin理论结果时，对照表的理论频率及差值留空，比较为BLOCKED。水分子的高频模式不能填入这些空白。

即使未来得到频率接近的孤立分子模式，也只能形成待验证机制假设。真实药材固体包含氢键、晶格／集体振动、分子间作用和复杂基质；本轮不包含这些效应，也没有纯品实测THz或成分含量测定。不能用频率接近确认某成分或解释产地差异。

## 重现与下一步

新增脚本分工为：可信结构取得与验证、ETKDG/MMFF构象准备、带累计资源限制的量子worker、保存结果只读复算、检查点续算准备。恢复必须沿用当前账本，不自动初始化另一个60分钟批次，也不能将重新启动优化器说成保留了geomeTRIC历史；只恢复几何与可用SCF密度，重新进行SCF和梯度。

绑定最终账本、检查点和来源哈希的[续算准备](../reports/pyscf-spinosin-pilot-20261031/continuation/README.md)只提取并核验几何和生成命令，**没有执行续算**。当前云端工作目录内，先核对再按同一预算继续：

```bash
cd /workspace/research4
bash reports/pyscf-spinosin-pilot-20261031/continuation/resume_same_budget.sh
```

该脚本执行前逐项验证源文件、检查点和最终账本哈希；可请求的量子CPU最多631秒（来自剩余633.765852秒，另留边界余量），新输出目录为`calculations/spinosin-optimize-resume-001/`。它没有`--initialize`或扩大预算选项，重新启动geomeTRIC优化器，不能恢复先前信赖半径／优化Hessian历史。631秒不是预计收敛耗时，也不保证完成优化。若账本已变化应重新准备，不能绕过校验。

完整频率阶段仍要求优化器返回收敛、同方法新SCF与梯度阈值通过、对应Hessian实现检查及足够资源，不能从任意检查点直接计算并解释频率。新的较大预算批次应有明确的新资源授权；本轮未擅自启动。

## 图表、测试和论文使用范围

[只读分析目录](../reports/pyscf-spinosin-pilot-20261031/analysis/README.md)包含水频率CSV、实际优化步骤CSV、资源表及全部20条实验候选对照。PNG与PDF分别展示水模式位置、斯皮诺素真实优化轨迹、未收敛模型几何及理论比较阻塞状态；等高频率线仅为位置标记，不是吸收强度。没有合格斯皮诺素模式时，不生成其振型箭头或动画。

原封存几何图存在一个轴标签裁切，保留原件并另提供[完整边界几何图](../reports/pyscf-spinosin-pilot-20261031/analysis-execution/spinosin_model_geometry_uncropped.png)及同名PDF；它显示最后完整评估的第4点，不是检查点的第5点。图表逐项核验和解释见[执行记录](../reports/pyscf-spinosin-pilot-20261031/analysis-execution/README.md)。

冻结版本[559项测试全部通过](../reports/pyscf-spinosin-pilot-20261031/environment/project_tests_frozen_final.log)，覆盖预算限额、并发防护、身份／立体化学、错误检查点及运行中结果拒绝、模式数／单位／非正频率等。单元测试的合成夹具不计入科研结果；实际水Hessian另行复算，斯皮诺素保存梯度数组和日志另行独立核对。

[最终独立审计](../reports/pyscf-spinosin-pilot-20261031/final-audit/closed_phases_independent_review.json)通过：4步梯度数组的五项统计与日志配对能量一致，全部阶段实际CPU与账本一致，检查点SHA通过，旧数据／模型／报告保持不变。审计PASS仅指证据一致，斯皮诺素科研阶段仍是PARTIAL。

源码、文档和测试的Git空白检查通过。完整差异中的23份原始日志／固定格式结构等产物有空白格式提示，为保持已封存SHA按原样保留，详见[交付核验](../reports/pyscf-spinosin-pilot-20261031/delivery_validation.json)。[整体逐文件哈希索引](../reports/pyscf-spinosin-pilot-20261031/artifact_index.json)覆盖本次新增科研文件及授权说明，不覆盖或重写历史产物。

本轮可用于论文的内容是计算方法、可追溯构象、资源可行性和诚实的未收敛记录，**尚不能成为实验吸收峰化学归属的正面证据**。下一步首先完成几何优化，再开展网格／构象／方法敏感性和合格Hessian；若有真正低频模式，再讨论分子内振动，并通过纯品或含量实验增加验证。没有预设理论解释或分类增益。

所有失败和部分结果都保留；不上传5.326 GiB积分张量、安装包、临时运行环境。合理大小的计算检查点、正常模式原数组（仅实际完成的水测试）、日志、CSV、PNG/PDF和复现代码进入独立研究分支，不合并main。
