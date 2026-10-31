# 跨植物预训练：模型与执行预检

本次主任务是855条酸枣仁ZY01—ZY07分类。对外部数据能否改善深度学习作配对实验，不把外部类别改成酸枣仁类别，不改历史96.0234%强SVM基线。

当前预检完成，**真实科研数据模型拟合0次**。14项有意越界／状态／身份／标签隔离测试通过；另用完全合成数据、主模型宽度作四臂资源演练，共0.870秒。合成运行不构成科研结果。首次测试夹具把较长ID写进NumPy `U6`数组而截断，已仅修复夹具类型为`U64`，原失败日志保留。

15个准备上下文为原5外折／2内折（10内层342源／342查询、5外层684源／171查询）；另外三池黄芪403、天麻405、人参／西洋参42。所有输入统一线性插值到事先固定的0.21—1.49 THz 128点，拒绝外推。插值不增加物理分辨率或样品数。负值原样保留。

黄芪／天麻数据中的`labels`原本为空，明确复用已有匿名`filename_groups` HQ01—HQ04／TM01—TM04作为来源组监督，并非验证过的真实产地或物种标签。人参为公开0／1物种标签。四种任务头相互分开；网络输入只有数值光谱，编号、文件名、组码不作特征。

## 预先固定的配对设计

模型是既有Encoder的**紧凑ResNet1D**：宽12、两残差块、无注意力、Dropout0.35，16维编码，ZY七分类头，共3603个最终目标参数。所有臂目标模型完全一致。

| 臂 | 目标训练前的额外学习 | 目的 |
| --- | --- | --- |
| A | 无，从随机初始化训练 | 酸枣仁从头对照 |
| Aextra | 仅当前酸枣仁source，临时七类头预训练120步 | 同额外步数／批量对照，不多用样品 |
| B | 人参42条，独立二类头预训练120步 | 单外部植物任务预训练 |
| C | 黄芪／天麻／人参各40步，各自4／4／2类头 | 多植物任务平衡预训练 |

额外阶段均为每步32条、当步无放回抽样，不将重复访问计成新增样品。源分类头的类别数不同，因此是步数／批大小匹配，不宣称浮点算量完全相同。目标阶段各臂都用同一source、同一源内标准化、同一初始ZY头、同一新Adam优化器、同一目标随机序列与批次顺序，预训练后重置全部BN running buffers。每域分别拟合SNV后StandardScaler，绝不在查询上估计参考或统计量。

目标固定候选为48／96轮，lr0.0005、weight_decay0.001、batch64、源类频数平衡权重。每个外折仅用A的内层两查询合集684条、各seed一次OOF，计算指标后取三seed均值选择共享轮数；所有四臂使用同一个所选轮数。不是分别给迁移方法选择更多轮。

来源内策略先在共享轮数上比较Aextra／B／C，需Macro-F1比A至少高0.005且Balanced Accuracy、Accuracy均不低于A，才允许替换；否则回退A。策略只引用已经拟合的四臂权重，不增加模型。

预算：240个内层完整管线＋60个外层管线，三seed42／43／44全部保留，另15个策略别名。225条预训练轨迹共27000次额外优化步；目标内层103680步，外层31680—63360步。所有配置及共享轮数锁定后才允许生成外层预测；预测锁之后另进程评分。

## 防泄漏与可追溯边界

实际FitScope记录每个学习的scaler和source神经训练轨迹；每步另流式gzip记录实际IDs、数值／标签指纹、当前域头及损失，可从保存的完整批次计划和prepared packs逐条重建。保存预训练编码器／源头、BN重置前后状态、初始ZY头、初始优化器摘要、目标批序摘要和最终权重。

source／query严格schema，query不允许标签、批次、未知列；坐标逐值一致。actual trial期间用builtins.open、io.open、os.open拒绝全部15个命名真值文件的读写，并在真实拟合前逐一进行读拒绝探针。另一次0fit准备预检对真实15文件分别读／写探针，30次均被拒，文件hash前后相同。该机制是API／命名文件边界，不是操作系统沙箱，也不证明未知原植物身份。

`pipeline.joblib`是在后处理完整性字段加入metadata前序列化，因而其内部metadata不含自身文件SHA等4项完整性字段；外部`training.json`绑定pipeline、batch trace、batch schedule及FitScope四文件，完整试验记录再绑定training.json。比较内部metadata时应排除这4项，不能构造自引用hash。

外部仪器、吸收系数长度单位、共同原料等尚未核验，因此仅是明确授权的**EXPLORATORY形状工程迁移**，不是物理校准通过。水果时间单位未知、叶片只有曲线而无可信独立分类样品，未混入本试验。全部855条酸枣仁已曝光；本次不产生全新独立确认集。

## 运行入口

```bash
.runtime/bin/python scripts/run_cross_plant_transfer.py prepare --output <new_run_directory>
.runtime/bin/python scripts/run_cross_plant_transfer.py selection --output <run_directory> --release <source_release.json>
.runtime/bin/python scripts/run_cross_plant_transfer.py evaluation --output <run_directory> --release <outer_release.json>
.runtime/bin/python scripts/run_cross_plant_transfer.py score --output <run_directory>
```

准备、训练、外层预测及评分分开，已存在的试验目录或结果拒绝覆盖。当前实际准备目录为`../neural/`，尚未放行训练；数据hash、模型代码版本、14项测试及真实命令见[validation.json](validation.json)、[frozen_source_files.json](frozen_source_files.json)和`commands.jsonl`。已运行步骤及0fit状态不代表后续必然提高。
