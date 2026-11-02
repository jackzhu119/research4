# 酸枣仁太赫兹产地识别与小样本校准研究

本仓库保存中药光谱研究资料及可复核实验。当前已对用户声明独立的855份酸枣仁样品完成基线、MC-UVE与SVM优化、同折模型对照及校准探索；已审计的主候选为96.02%准确率、0.9197宏F1。每产地目前仅一批，真实地名及独立新批次确认仍待补充，不能将内部识别分数当作跨批次产地鉴别结论。

## 最新：斯皮诺素PySCF真实DFT试验

[主报告](docs/PYSCF_SPINOSIN_PILOT_20261031.md)、[原始计算、结构及结果索引](reports/pyscf-spinosin-pilot-20261031/README.md)。云端已完成PBE/def2-SVP水分子Opt/Freq及两个DF-J后端核查，真实3个正频率；斯皮诺素可信立体结构与12个构象准备完成，并已执行真实SCF、核梯度和几何优化尝试。斯皮诺素尚未通过几何收敛检查，Hessian／振动频率未运行，不能作药材化学峰归属。无D3、IR强度NOT_COMPUTED，559项测试通过；保留中止日志和检查点，累计CPU限制及原96.02%分类成果不变。以下旧量子NOT_RUN段落保留为各自历史快照。

## 最新：外部植物预训练的真实深度学习对照

[完整报告](docs/CROSS_PLANT_DEEP_TRANSFER_REPORT_20261031.md)、[主表Excel、补充对照与图](reports/cross-plant-deep-transfer-20261031-v1/comparison/README.md)。同一酸枣仁855条、五／二折和三种子，完成240个内层＋60个外层ResNet管线：从头训练87.21%，人参预训练86.43%，黄芪／天麻／人参预训练85.65%，本轮未显示外部数据收益。保守策略87.95%只选择酸枣仁额外训练或原模型，不能归因跨植物。全部模型、负结果和标签隔离诊断保留，502项测试通过；原MC-UVE＋SVM96.02%保持不变。

## 最新：公开植物数据获取与多轮验证

[本轮方法和科学边界](docs/PUBLIC_PLANT_THZ_VALIDATION_20261031.md)、[实验与下载证据](reports/public-thz-20261031-v1/README.md)、[13表Excel与论文图](reports/public-thz-20261031-v1/comparison/README.md)。已实际下载约138MB水果亚太赫兹原件及植物叶片数值资料，复核既有人参／西洋参42条观测；18,720次source候选＋606次外层估计器拟合完成，463项项目测试通过。人参MC-UVE多轮未改善；描述性最高n＋收缩LDA为61.90%，水果部分任务改善、部分为负。许可未知的原件和水果逐条产物仅本地保存；不把曲线、时点、CSV文件数当独立样品，也不替换酸枣仁96.02%主线。

## 最新：ZY01—ZY07匿名部署与Excel预测

按用户新要求固定七个编号，不等待真实地名。[预测工具与使用说明](docs/ZAOREN_ANONYMOUS_DEPLOYMENT.md)：75次开发配置比较＋3次全855重拟合已完成，部署配置为SNV＋MC50/LV1/123点＋RBF-SVM（C1000、gamma0.1×scale、balanced），主seed42固定。支持CSV／Excel／NPZ、空白模板、历史留出重放与独立标签评分；模型新样品准确率尚未测，不能把旧96.02%赋给它。

[空白Excel模板](reports/zaoren-deployment-20261031-v1/deployment/templates/new_spectrum.xlsx)、[真实历史重放与全部34错误](reports/zaoren-deployment-historical-demo-20261031-v1/README.md)、[新的内验证学习曲线](reports/zaoren-neural-learning-pilot-20261031-v1/README.md)。399项测试通过；旧数据／模型不改。四表示96轮在单一内验证划分提高总体准确率，但Balanced Accuracy由48轮74.93%降至96轮69.44%，不预设加轮数一定更好。

## 最新执行：多表示、深度学习与严格防泄漏

[完整升级报告](docs/MULTIREPRESENTATION_UPGRADE_REPORT.md)和[要求逐项对应](docs/UPGRADE_REQUIREMENTS_TRACEABILITY.md)：保留96.0234%／宏F1 0.9197强MC-UVE＋SVM。新增八家族、三种子、600个实际神经管线；内层选择的四表示策略为85.0292%，未超过基线，负结果完整保存。359项项目测试通过，120个外层神经模型逐点推理／BN状态及独立预测重放通过；这些有限检查不证明绝对零泄漏。

[15表Excel及论文图](reports/upgrade-20261031-v1/paper-assets/README.md)、[统一多药材数据库](docs/MULTIPLANT_DATA_AUDIT.md)、[黄芪／天麻独立任务](reports/upgrade-20261031-v1/auxiliary/README.md)、[量子化学实际可行性](docs/QUANTUM_CHEMISTRY_FEASIBILITY.md)。HQ403／TM405已追加用户独立样品声明，独立批次及单位仍未知；旧快照保留，当前资格另存覆盖层。PySCF水环境测试真实完成，三候选DFT／理论峰均NOT_RUN，跨植物迁移BLOCKED。所有现有酸枣仁成绩均为历史曝光内部开发，尚无新批次确认。

## 当前数据版本：后续以此为准

老师本轮补充的[12份小麦／黄芪／天麻资料](data/incoming/teacher_20261031_v1/README.md)已单独归档，提供Excel清单、SHA及逐表完整性检查。3份黄芪完全重复，另有3份原件存在损坏，均明确标记；新增资料暂不替换以下有效研究输入。

**自2026-10-31起，酸枣仁仅使用7份编号版，共855个唯一谱记录编号。此前未编号的旧文件已停止使用并转入溯源归档。**

- [唯一有效数据及7份文件下载](data/CURRENT_DATASET.md)
- [机器可读有效清单](data/active_workbooks.json)与[当前校验清单](data/manifest.csv)
- [停用旧版及逐文件替代关系](data/archive/zaoren_superseded_20261031/README.md)

- [当前1258行元数据模板](data/metadata_template.csv)：黄芪403＋酸枣仁855。

当前主线以酸枣仁识别及可核验模型优化为核心；按用户本轮升级要求，新增多表示深度学习和分子振动机制准备。Gaussian候选计算尚未运行，不改写此前排除Gaussian时的历史实验。

## 科研起步与算法审读

- [方案C总计划](docs/RESEARCH4_PAPER_C_MASTER_PLAN.md)、[研究状态与准确率审计](docs/RESEARCH4_EXPERIMENT_AUDIT.md)及[论文证据提纲](docs/RESEARCH4_PAPER_OUTLINE.md)：酸枣仁优化为主线，外部植物与校准为辅助；保留855份OOF与旧171份预测两套协议，新增同折分类器对照、条件消融、最佳历史候选索引及外部数据资格审查。
- [最新：按用户顺序执行的四轮嵌套优化](docs/STAGED_NESTED_OPTIMIZATION_REPORT.md)：精确复现855份五折旧基线90.41%，MC-UVE＋SVM参数选择为96.02%／宏F1 0.9197；后续预处理、稳定组合及固定CNN未继续提高。1540次内层评价先全部锁定再统一外测，264项测试通过；提供21表Excel、全部错误和模型。与旧171份固定预测口径分开。
- [按图片开展的特征、概率与融合优化](docs/IMAGE_GUIDED_OPTIMIZATION_REPORT.md)：81主要设置，241项测试通过。SNV-LDA为92.40%准确率／0.8352宏F1；SVM温度概率NLL 0.541→0.410且分类不变；融合未通过开发检验。完整报告解释稀少类精确率、检查点修复及全部负结果，提供14张表的Excel。
- [最新四轮优化与完整台账](docs/MULTIROUND_OPTIMIZATION_REPORT.md)：85个主要设置及控制，208项测试通过。追加源训练后的正式零支持方案90.25%／宏F1 0.8787；同预算监督继续训练与FO-MAML分类结果相同，不能归因MAML；三个校准路径均回退不校准。提供每轮参数、负结果、逐样品Excel、模型和独立审计。
- [论文方法借鉴与真实分类试验](docs/PAPER_FEWSHOT_CLASSIFICATION_REPORT.md)：紧凑共享ResNet1D、分类ProtoNet、一阶MAML与微调实测。MAML校准64.72%→67.55%，三个种子平均均提高，仍低于SVM的88.89%；提供逐样品标签对照Excel、全部负结果、源权重与独立审计，172项测试通过。
- [表征与潜空间校准的后续优化](docs/REPRESENTATION_AND_LATENT_REPORT.md)：54条源管线、60种潜空间适配；开发阶段LDA胜出，但固定预测准确率88.30%仍低于SVM88.89%，三个源模型的正式校准选择均保留原方法。
- [小样本校准探索与标签复核](docs/CALIBRATION_OPTIMIZATION_REPORT.md)：116配置、69.6万次开发评价、9000组最终评价及153万余条预测。七类主选择保留原模型；强正则弱校准家族对照平均提高0.21个百分点，但获益仅涉及1份边界样品，尚不是确认提升。标签一致性与全部正负结果均归档。
- [训练／校准／预测拆分及实测](docs/ZAOREN_THREEWAY_REPORT.md)：21份Excel已按513／171／171份拆好，两个冻结模型已保存，114项测试通过。每类1份、20次分数校正均值未改善；SVM为88.89%→74.94%，MC20＋SVM为87.72%→72.89%，全部正负结果保留。
- [三组文件下载与归属](data/processed/zaoren_threeway_20261031_v1/README.md)及[先行协议](docs/ZAOREN_THREEWAY_PROTOCOL.md)：原件不变、四表同步、每份仅属一组；当前只能评价同批次回顾性校准，不能解释为新批次泛化。
- [已批准方案的首轮执行报告](docs/ZAOREN_APPROVED_EXECUTION_REPORT.md)：7表质量审查、13路径／65外折、11,115条真实预测、100项测试；候选区MC20＋SVM为90.41%准确率／0.8092宏F1，稳定筛选和拼接未胜出。全导出PLS-DA的100%仅列内部敏感性，真实跨批次校准未运行。
- [批准的实验方案与启动条件](docs/APPROVED_ZAOREN_EXPERIMENT_PLAN.md)：固定5外折／2内折／CARS内部2折；用户已确认每ZY编号独立、每产地仅一批，不含Gaussian。
- [Google Colab连接实测](docs/COLAB_CONNECTION.md)：官方CLI 0.7.5已隔离安装且依赖检查通过；Colab及OAuth服务目前被云端代理403拒绝，尚未授权、创建远程环境或执行远程代码。
- [酸枣仁工作簿统一编号（2026-10-31）](data/processed/zaoren_20261031/README.md)：样本1的3个RAR分卷已成功恢复，样本1—7全部完成，共855个跨文件唯一谱记录编号；四表同步、保留厚度后缀及全部数值，提供修改版和Excel对照表。
- [少样本光谱论文精读与酸枣仁适配](docs/FEWSHOT_PAPER_READING_AND_ADAPTATION.md)：首次全文审读与任务方案，保留当时未运行新模型的范围；后续实际分类试验见上方新报告。
- [Gaussian／GaussView安装记录](docs/GAUSSIAN_INSTALLATION.md)：上传GaussView6.0.16已通过Wine64启动并显示水分子；保留启动入口和截图，Gaussian计算引擎仍缺失。
- [老师源码性质与主线文献对照](docs/ALGORITHM_ORIGINALITY_AND_LITERATURE.md)：区分现成算法、任务适配和未确认的改进，列出中药THz、酸枣仁产地及小样本迁移参考，说明SCI／SCIE核验范围。
- [太赫兹数据来源核验](docs/THZ_DATA_SOURCES.md)：优先中国机构，锁定人参 n／α CSV、小麦粉含量 MAT 与代码、原始时域信号等；逐项区分文件说明、目录预览、受限访问和未下载状态。
- [枣类THz公开数据专项检索（2026-10-31）](docs/JUJUBE_THZ_DATA_SEARCH.md)：未核实到可下载且带产地标签的枣／酸枣仁THz数据；核实了枣果近红外高光谱入口及非枣种子THz记录，逐项说明用途与限制。
- [后续科研阶段二](docs/RESEARCH_STAGE2_REPORT.md)：稳定筛选与少样本校准实现，200个新药材分类场景、7200组迁移评价与正负结果。
- [科研数据目录](docs/RESEARCH_DATASET_CATALOG.md)：34条来源/数据表示记录，区分已下载、仅元数据、受限访问、同队列与样品层级。
- [外部中药光谱基准报告](docs/EXTERNAL_BENCHMARK_REPORT.md)：金银花/山银花产地与LC-MS相对丰度，330个真实运行场景，全部五折指标、原归档核心对照和复现命令。
- [安装与统一运行](docs/INSTALL_AND_RUN.md)：当前云端隔离环境、固定依赖、可运行家族入口、修正版与替代实现的对应关系。
- [云端运行与可用性实测报告](docs/RUNTIME_USABILITY_REPORT.md)：Octave/Python/PyTorch 的实际运行、阻塞点、兼容修复、数值对照及复现日志。未进行 MATLAB 原生验证。
- [算法逐文件审读](docs/ALGORITHM_REVIEW.md)：CARS/UVE、SSA-SVM、PLS、CNN、BPNN、PCA、LS-SVM 的实现差异、具体问题、源码位置及复用建议。
- [研究与实验协议](docs/RESEARCH_PROTOCOL.md)：黄芪产地鉴别、成分定量、跨批次泛化和小样本校准的数据条件与评价规则。
- [基础代码使用](docs/RESEARCH_STARTER.md)：正确读取尺寸声明异常的 Excel、SVC/SVR/PLS 嵌套分组验证及测试。
- [当前谱列元数据](data/metadata_template.csv)：1258条（黄芪403＋酸枣仁855）；酸枣仁物理ID与匿名组标签依据用户声明补齐，真实地名、批次标识和物种记录仍待补，黄芪注释未变。

历史首次数据检查覆盖9份文件、808条谱记录，其中旧枣仁405条；该范围已被当前7份酸枣仁855条替代。历史数值和hash保留在[检查记录](reports/audit/workbook_profiles.json)，不代表对当前数据的重新实测。当前有效清单共11份文件、1258条记录，编号不等于已核实的独立样品数。

## 已归档数据

以下四个黄芪 Excel 文件均来自用户上传，保持原始文件名和字节内容不变。产地文字仅依据文件名记录，尚未核验样品来源或实验记录；文件名中的数字串不作为已确认的采集时间。

| 文件 | 文件名所示产地 |
| --- | --- |
| 黄芪山西大同202605310205.xlsx | 山西大同 |
| 黄芪甘肃202605310347.xlsx | 甘肃 |
| 黄芪内蒙古固阳202605310518.xlsx | 内蒙古固阳 |
| 黄芪曲靖202605310559.xlsx | 曲靖 |

黄芪原始文件位于 `data/raw/huangqi/`。`data/manifest.csv` 只记录当前有效的4份黄芪及7份酸枣仁编号版的路径、字节数和SHA-256；停用原件另有归档清单。

## 文件完整性检查

四份黄芪工作簿均列有“时域、频域、折射率、吸收系数”工作表。当前版本的四份黄芪 Excel 文件均通过 XLSX/ZIP 全成员 CRC 完整性检查。这些检查仅验证文件容器完整性，不代表数值、单位、标签或实验有效性已经核验。

### 甘肃文件替换记录

按用户要求，`data/raw/huangqi/黄芪甘肃202605310347.xlsx` 已替换为本次重新上传的原始文件，保持新上传文件的字节内容不变；`data/manifest.csv` 已同步更新。旧版本仍保留在 Git 历史中。

**新文件的“折射率”工作表为空**：`xl/worksheets/sheet3.xml` 中 `sheetData` 没有行或单元格（0 行、0 单元格）。该表现在可以读取，但没有折射率数据；本次替换不代表该表数据已恢复。

| 版本 | 字节数 | SHA-256 | 容器检查结果 |
| --- | ---: | --- | --- |
| 先前上传 | 15,695,947 | `b43ad798b031177de19b23476901ff51b76e9f1cd55fbfe785129807ab63917a` | “折射率”工作表解压失败 |
| 本次重新上传 | 14,545,729 | `77f122be17bfc551e747d2101c2ab7910adca24f143aec74c96cfe57015bf04e` | 全成员 CRC 检查通过 |

## 酸枣仁当前归档：7/7全部完成

唯一有效文件位于`data/processed/zaoren_20261031/`，基础编号为ZY01—ZY07，共855条谱记录；样本1已由全部3个RAR分卷成功恢复，不再有待补工作簿。详细数量、替代关系及下载链接见[当前有效数据](data/CURRENT_DATASET.md)。

此前按品牌／日期命名的输入及未编号的样本1、4已移至`data/archive/zaoren_superseded_20261031/`，仅供溯源，不参与默认研究。旧版“5/6份”和“405条”叙述属于历史，后续不再采用。

## 后续研究需要确认的信息

- 光谱技术类型（是否为太赫兹光谱）、仪器型号、频率或时间坐标及单位、参考谱和实验条件。
- 样品物种与鉴定依据、样品及批次编号、重复测量关系、产地、含水率、制样方式和厚度。
- 成分检测所需的标准品或独立化学检测结果；真伪鉴别所需的可信标签及阴性、伪品或掺伪对照。
- 建立预处理、按独立样品或批次划分训练与测试集、模型评估和论文方法记录，避免同一样品的重复谱同时进入训练集和测试集。

这些信息有待补充。产地差异本身不能作为真伪标签，现有文件不代表成分检测或真伪鉴别已经得到实验验证。

## 论文参考资料（2026-10-30）

首批 47 项写作参考资料现已全部归档，其中此前超限的 `0308 CARS+UVE.rar` 已从本次补充的 4 个 Desktop RAR 分卷中恢复并单独保存；四个分卷也全部按原名保留。当前清单共 **51 个文件**，均为 `archived`。

参考文献、算法程序、论文稿件、流程图、教程、投稿材料和补充分卷按类别保存至 [论文资料库](references/paper-materials/README.md)，提供逐文件 SHA-256 清单。分卷内含文件、校验记录及解压方法见 [Desktop 分卷说明](references/paper-materials/archives/desktop/README.md)。

## 科研代码与实验说明

算法审读、当前有效清单的默认读取入口及实验协议同步维护在[研究分支](https://github.com/jackzhu119/research4/tree/research/algorithm-review-and-baselines-20261030)。7份酸枣仁工作簿及有效清单在main与研究分支保持一致。
