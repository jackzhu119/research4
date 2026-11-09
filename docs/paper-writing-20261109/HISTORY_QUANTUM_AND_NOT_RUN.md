# 历史算法、公开数据及量子化学：可引用证据与未完成事项

整理日期：2026-11-09。本文保存研究的完整来路和负结果，不把不同标签、样品、划分或方法下的成绩串成一个提升阶梯。当前物种/产地说明以有效18工作簿登记为准；旧文档的“855份酸枣仁”等称呼是当时注释，须结合下述勘误阅读。此次只归档、核对和复算已保存预测，没有新增模型拟合或量子计算。

[76项实验/计划登记CSV](../../reports/paper-evidence-20261109/tables/historical_quantum_experiment_register.csv)与[70份原始小文件及来源清单](../../reports/paper-evidence-20261109/sources/history_quantum/provenance.json)可直接下载。来源文件按原字节复制，含原SHA-256、Git blob SHA-1、固定提交及路径，共约3.4 MB；大型CHK/HDF/NPY缓存未复制。旧文件内的绝对工作路径和当时状态保留，不能据其`RUNNING`字段声称现在仍在运行。

状态含义：`DONE`仅指命名范围的实际工作完成；`PARTIAL`为实际执行但范围未完成；`NOT_RUN`为未执行；`BLOCKED`为现有条件/证据不能开展或不能支持结论。`DONE`不等于结果正面或物理资格通过。CSV的E1表示本轮读取实际数值/逐样品证据（旧R0–R4还独立重算），E2表示核对保存指标/报告及其引用链，E3表示计划或明确未做记录；这些不是外部机构认证等级。

## 1. 旧855份的五轮结果：计算可信，任务名称必须更正

旧数据版本`2026-10-31-zaoren-numbered-v1`使用855个不同编号、612个α特征、同一5外折/2内折，split seed 20261031、model seed 42。每方法每编号一次OOF。现已确认ZY01/ZY02为枣仁，其余ZY03–ZY07为酸枣仁，因此这是**混合种类的七来源组识别**，不能再作为纯酸枣仁产地识别的96.02%证据。

本次从两份原始`outer_predictions.csv.gz`重新逐ID统计，校验各方法恰855个不同ID并重算混淆矩阵、Accuracy、Macro-F1及Balanced Accuracy。结果与原报告一致，复算记录见[JSON](../../reports/paper-evidence-20261109/sources/history_quantum/historical_stages_metric_replay.json)。

| 方法 | 正确/855 | Accuracy | Macro-F1 | Balanced Accuracy | 相对R0变化，百分点 |
|---|---:|---:|---:|---:|---:|
| R0 原MC-UVE＋SVM | 773 | 90.4094% | 0.809185 | 83.6735% | 0 |
| R1 MC参数选择 | 791 | 92.5146% | 0.827691 | 86.6531% | +2.1053 |
| R2 继承MC，再选SVM | 821 | 96.0234% | 0.919685 | 89.9184% | +5.6140 |
| R3 再选预处理 | 816 | 95.4386% | 0.887720 | 86.3878% | +5.0292 |
| R4 再选组合家族 | 812 | 94.9708% | 0.882042 | 85.8163% | +4.5614 |

这里的提升是准确率差的**百分点**，不是相对提升百分率。R2相对R0纠正62份、伤害14份、净增48份；并非全部类别改善到96%。ZY02仍只有5份，R2召回3/5，两份误判保留。

R0不是固定C模型：内层选择Raw/SNV及C∈{0.1,1,10,100}，MC20/LV2/62点固定。R1固定SNV、C10，研究MC10/20/50、LV1/2/3、保留5/10/15/20%；故R1与R0差异不能全部归为MC的单因素效应。相对固定SNV/C10锚定88.1871%，R1为+4.3275个百分点。

R2各外折MC次数10/20/50/50/50，LV3/1/1/1/1，K92/62/123/123/92；所有外折仍用SNV。C为100/1000/1000/1000/100，gamma为0.1×scale/0.1×scale/scale/0.1×scale/scale；类别权重在训练内选择。它是五个外折模型的评价，不是一个已用全部855训练的部署模型。不能择一外折配方宣称唯一最优。

R3/R4下降不说明计算错误，也不证明复杂算法普遍无效：条件继承下，源内选中的更复杂路线未在本次外折取得额外收益。固定预处理诊断SNV96.0234%、SG一阶＋SNV92.3977%、MSC95.2047%；稳定MC95.2047%、稳定CARS89.7076%、CNN头87.0175%、CNN表示＋SVM86.0819%。全部正负结果和逐折选择保留，不能用外层最高家族替换内层选择策略。

固定来源：[报告](https://github.com/jackzhu119/research4/blob/3fc996e99cce8ff5799b5a2af9b565f982dfc899/docs/STAGED_NESTED_OPTIMIZATION_REPORT.md)、[协议及原始结果目录](https://github.com/jackzhu119/research4/tree/3fc996e99cce8ff5799b5a2af9b565f982dfc899/reports/staged-nested-optimization/2026-10-31)。新纯酸枣仁300TRAIN重复嵌套CV、旧855OOF、171固定预测与新100固定预测都是不同协议。

## 2. 早期算法审读、少样本与深度模型：已做事项不能遗漏

| 历史工作 | 实际范围与观察 | 论文位置 |
|---|---|---|
| Octave/Python可用性 | 92次执行/定向核验；原MATLAB入口多数有API/路径阻塞；MC-UVE/PLS/SVM核心及兼容适配可运行；未做MATLAB原生认证 | 方法实现与复现说明 |
| 老师源码性质核查 | CARS/UVE/IRIV/Random Frog/SVM/SSA/BPNN等有既有方法来源；发现CARS误差字段÷2.8、SSA同折寻优与报告等问题 | 说明真实修复，不包装标准算法原创 |
| 旧513/171/171固定角色 | SVM152/171=88.8889%，F1约0.8616；收缩LDA158/171=92.3977%，但F1约0.8352较低 | 历史补充，不能与855OOF排名 |
| 紧凑ResNet、ProtoNet、一阶MAML | 这些已实际运行。MAML自身64.7173%→67.5458%，F1约0.3922→0.4338；仍低于SVM。ProtoNet及全网微调有下降，开发不放行则回退 | 历史混种、同批次少样本探索 |
| 85主要候选四轮优化 | MLP充分源监督后90.2534%、F1约0.8787；FO-MAML与同更新监督控制的分类结果相同；7份支持校准未产生通过开发流程的净收益 | 区分源训练与适配收益 |
| 81主要设置图片指导优化 | CARS/UVE/SPA/PCA-LDA、概率偏置/温度、SVM/MLP软融合均已跑；温度改善NLL/Brier/ECE但不改变argmax；正式融合回退wSVM=0单MLP | 负结果和概率质量 |
| α、n、时域多表示与跨植物预算 | 旧任务报告均存在；更多表示、更多更新与外部知识的贡献须匹配控制；旧七ZY标签仍混种 | 历史对照，当前任务另看新矩阵 |

不能把旧SSA示例100%和含9条跨集相同特征行的CNN示例分数写进中药主结果。源OOF温度校准使用0份目标支持标签，不是新批次校准；旧目标支持三种子×100抽样也不是300个批次或新增样品。

主要固定报告：[源码及文献](https://github.com/jackzhu119/research4/blob/3fc996e99cce8ff5799b5a2af9b565f982dfc899/docs/ALGORITHM_ORIGINALITY_AND_LITERATURE.md)、[运行可用性](https://github.com/jackzhu119/research4/blob/3fc996e99cce8ff5799b5a2af9b565f982dfc899/docs/RUNTIME_USABILITY_REPORT.md)、[少样本论文适配](https://github.com/jackzhu119/research4/blob/3fc996e99cce8ff5799b5a2af9b565f982dfc899/docs/PAPER_FEWSHOT_CLASSIFICATION_REPORT.md)、[四轮优化](https://github.com/jackzhu119/research4/blob/3fc996e99cce8ff5799b5a2af9b565f982dfc899/docs/MULTIROUND_OPTIMIZATION_REPORT.md)、[图片指导优化](https://github.com/jackzhu119/research4/blob/3fc996e99cce8ff5799b5a2af9b565f982dfc899/docs/IMAGE_GUIDED_OPTIMIZATION_REPORT.md)。

## 3. 外部公开THz数据：取得、能训练、能证明什么分别登记

| 数据 | 实际取得/许可/粒度 | 已做结果或阻塞 |
|---|---|---|
| 人参/西洋参2025 V2，DOI10.17632/hvbp5mdtxx.2 | 18CSV、42压片、65点、α/n；CC BY4.0。共同原植物与独立批次未知 | 六导出组留出×3种子实测；R0 54.76%/F1 .5411，R1 45.24%，R2/R3 43.65%，source内R4 51.59%；n-LDA事后家族最高61.90%/F1 .6156，不能回填总体策略 |
| 人参2026 V1，DOI10.17632/k4pjjjmgnr.1 | 仅核到四方案×六折的96CSV目录；CC BY4.0；与2025队列关系不明 | 403未取得数值原件，不计N、不复现论文97.67% |
| Princeton亚THz水果作者仓库 | 已下载约138MB；30可追踪果实；120唯一时点记录；未见许可文件 | 按果实分组四个工程回归任务各10果实/40记录；MA-Brix/MA-DM共用40波形，不独立累加。source-policy RMSE .5171/.3370/.5144/.2652；目标/轴单位未确认，不作药材化学含量验证 |
| 叶片作者TikZ曲线 | 已下载68曲线，未见许可文件；曲线不是独立叶片 | 处理参考，缺原始样品标签，不训练分类器 |
| 稻谷、玉米DeepSpectra、Dryad小麦粉、Cambridge参考 | 有主源记录/下载失败证据；未取得可用数值原件或独立性/许可未核 | BLOCKED；不把文件数/像素/压片当独立植株 |
| 用户小麦表 | 50平均曲线列，标签和独立样品映射缺失，3非有限格 | BLOCKED；不能造类别进入产地分类 |

人参α/n是同一压片的不同派生表示，不能算84份样品。水果每种子40时点仍只有10果实，回归RMSE不能称准确率。未见明确许可的原件/衍生数值/模型仍保留原本地范围，本次只整理已有合法公开的自编报告、来源和汇总，不把未知许可材料重新上传。

[人参/水果完整报告](https://github.com/jackzhu119/research4/blob/3fc996e99cce8ff5799b5a2af9b565f982dfc899/docs/PUBLIC_PLANT_THZ_VALIDATION_20261031.md)、[主源检索及资格复核](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/RESEARCH4_EXTERNAL_EVIDENCE_20261104.md)。原报告写作时仍称855酸枣仁，需按当前勘误解读。黄芪/天麻新结果以当前物种更正后的专章为准，不用以上公开异任务替代主线。

## 4. Spinosin完整频率已真实完成，但最低点与峰归属未通过

原C001：Spinosin，C28H32O15，75原子/320电子，PubChem155692、ChEBI81360；可信连接/立体图→ETKDGv3/MMFF94s生成12构象→DFT模型构象。不是实验晶体坐标。PBE/def2-SVP、grid1、direct、完整762AO/MO、明确noD3。完整31XC→A→H1→10响应块→全Hessian于北京时间2026-11-06 14:20完成，独立模式数学重放随后通过。

| 证据 | 状态/值 | 可写结论 |
|---|---|---|
| 完整有符号模式 | DONE，219=3×75−6，194正/25虚 | 真实计算完整，不是合格最低点 |
| 虚频范围 | 约−312.34至−87.69cm⁻¹ | 保留符号；不删、不取绝对值 |
| 0.1–1.5THz正模式 | 1个，模式26：37.02737cm⁻¹=1.1100526THz | 当前未合格构象的诊断位置，不是确认斯皮诺素实验峰 |
| 独立正常模式重放 | 最大频差约6.14e−12cm⁻¹ | 核验保存Hessian到模式的数学转换；不是完整DFT独立重算 |
| 保存响应fresh方程残差 | 较新A004已补齐225/225 RHS | 不再停留早期6/225；但不认证H1/A/全部核二阶导 |
| 轨道稳定性 | 实际尝试后失败闭合 | PARTIAL，不能称稳定性通过或完全未尝试 |
| 固定/重建网格、方向能量/梯度控制 | 多个真实成功与失败均保存 | 原解析负曲率和部分数值正曲率不一致原因未确定，不擅自认定一个实现错误 |
| IR/化学峰归属/理论辅助训练 | NOT_COMPUTED / BLOCKED / NOT_RUN | 未产生可信理论吸收强度、成分确认或新分类增益 |

完整219表、结果和复核在本地整理包`quantum_historical/`；固定原件见[219模式CSV](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/reports/pyscf-spinosin-96h-continuation-20261105-v8/global-assemble-recovery-002/independent-modes/all_frequencies_cm1_THz.csv)、[完成报告](https://github.com/jackzhu119/research4/blob/cd4b50c37c1e26064f9f15f340d30e20ac386458/docs/PYSCF_SPINOSIN_COMPLETION_20261106.md)。较新的225RHS/失败与控制汇总见[backlog审计](https://github.com/jackzhu119/research4/blob/0f50b382c233887ceab1cb0e847b3eac99817488/docs/PYSCF_BACKLOG_AUDIT_20261108.md)。早期pilot/11月4日未完成报告只是历史阶段，不覆盖11月6日后的实际219结果。

## 5. 最新量子支线不能遗漏：原儿茶酸完整双步长正模式与Calycosin部分结果

此次重新核对较新分支`research/pyscf-autonomous-90000-20261108`，固定提交`0f50b382c233887ceab1cb0e847b3eac99817488`。它不是旧A006失败状态的重复描述。实际v3方法为PBE/def2-SVP、grid3、direct、fullAO/MO、noD3、grid_response=True，每周期重建native VEFF；普通电子容差1e−10Eh/轨道目标1e−7，旧严格门槛作为诊断警告保存，不把新版结果伪装成旧严格认证。

原儿茶酸C7H6O4/17原子：两套全核坐标数值Hessian各102/102真实位移梯度，h=.001/.0005Bohr；2026-11-09 00:09:50北京时间完成分析。每套45模式均正、0虚；原始相对反对称性约8.61e−5/2.35e−4，投影矩阵相对步长差1.38888e−4，`numerical_quality_diagnostics_passed=true`。最低模式约73.6463cm⁻¹≈2.20786THz，**不在0.1–1.5THz范围**。

但最终核梯度RMS1.14218e−5、最大分量3.62795e−5Eh/Bohr未满足原严格核门槛，`minimum_verified=false`。全部位移nativeSCF收敛，但旧严格诊断警告保留；因此不能仅凭全正频升级为经过严格驻点验证的稳定最低点，不能用它确认酸枣仁固体吸收峰。新grid3几何/方法和B旧grid1的45模式/4虚频分别保存，没有覆盖或“修掉”旧虚频。

Calycosin C16H12O5/33原子：旧B完整93模式含7虚；本轮较新grid3重新优化常规收敛，但核最大分量3.39903e−5仍未过原3e−5门槛。固定Git快照只有h=.001的**28/198**有限位移点，未取得本轮完整93模式或双步长分析，不补零拼矩阵。

关键实际原件：[原儿茶酸分析JSON](https://github.com/jackzhu119/research4/blob/0f50b382c233887ceab1cb0e847b3eac99817488/reports/pyscf-continuous-20261108-v3/jobs/acid-g3-analysis/result.json)、[Calycosin进度](https://github.com/jackzhu119/research4/blob/0f50b382c233887ceab1cb0e847b3eac99817488/reports/pyscf-continuous-20261108-v3/jobs/caly-g3-h001/progress.json)、[保存账本](https://github.com/jackzhu119/research4/blob/0f50b382c233887ceab1cb0e847b3eac99817488/reports/pyscf-continuous-20261108-v3/ledger.json)。账本更新时间为北京时间11月9日04:38，保存状态RUNNING、原90000授权累计保守计费61670.464636CPU秒，含先前费用/预留与当时活动观测；不是现在实时存活证明。最后快照存在`Argument list too long`上传失败记录，后来固定提交仍保存上述结果；不据此宣称之后未上传点已完成或丢失，也不在本归档任务接管续算。

另核B最新固定提交`c9f2fcf2b57934597c45caea2c113b713810d117`中的原4/7虚频子空间及组合方向数值验证。它们确有正曲率结果，但只是有限子空间，不是全部核空间正定认证；与新原儿茶酸完整数值Hessian应分别引用。B分支后来还有机器学习实验，由相应结果专章登记，不能计为新增分子计算。

## 6. 仍是设想、受阻或没有当前任务证据的事项

- **皂苷A/B完整DFT NOT_RUN**：已做结构、立体与成本审查。A178原子/1646AO、B157/1442AO；原满缓存估计52.010/34.890GiB是特定方案存储，不是所有DFT算法最低内存。A不同CID的立体歧义保留，不能随意选同名结构。
- **Gaussian药材原生计算 BLOCKED**：云端未确认合法可用引擎；PySCF实际结果不能标Gaussian计算。用户Windows水日志也不能当云端药材完整运行证据。B3LYP-D3(BJ)完整结果NOT_RUN；当前实际PBE明确noD3。
- **SNAIL/SimCLR NOT_RUN**：曾讨论原论文，但没有对应本任务实测。完整二阶MAML未完成；已有的是明确的一阶近似。Attention/理论先验网络若只有设计/模板，不能在论文中凭名称补写为已运行对照。
- **真实成分含量模型 NOT_RUN**：缺每个样品的HPLC/LC-MS参照，不能把产地标签、SDF、文献平均含量或理论全局频率填成逐样品化学真值。
- **真实跨批次/跨仪器校准 NOT_RUN**：样品独立性用户声明不等于多批次映射。内部support/query能讨论适配现象，不能改称独立新批次证据。
- **纯品THz、化学峰确认、分子辅助分类 BLOCKED/NOT_RUN**：缺标准品实测、合格最低点和可靠对应。理论频率接近不确认化学归属；孤立分子也不包含药材固体晶格/复杂氢键及集体效应。

用户已表示不再补实验和数据，以上缺口应作为论文范围与限制如实写出。现有云端主模型证据可以组织成一篇内部开发验证研究；不能因没有实验室补测就填造验证完成或把上述计划当结果。
