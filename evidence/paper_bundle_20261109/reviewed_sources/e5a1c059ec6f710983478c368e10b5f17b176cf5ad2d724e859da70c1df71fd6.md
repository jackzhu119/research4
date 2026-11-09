# ZY01—ZY07匿名七分类：冻结模型与Excel预测工具

2026-10-31，北京时间。依据用户“继续开展，7组就按那个标号执行不用在意产地”，后续固定使用ZY01—ZY07，不要求补真实地名才执行预测。七组标签、855份原始数据和全部历史结果均未修改。

## 本次实际完成

建立了完整部署模型，不只保留研究交叉验证折模型。所有855份记录作为已曝光开发数据，固定使用五个历史源内R2配置作为候选、原五折成员作开发验证、seed42／43／44全部运行。75个真实候选管线按15个开发case的Macro-F1均值、Balanced Accuracy、Accuracy及固定ID顺序选配置，然后在完整855份上重新拟合三个模型。主模型固定seed42，43／44仅保留预测分歧参考，不按结果挑seed、不把三模型称已验证集成。

| 项目 | 最终配置 |
| --- | --- |
| 预处理 | 单谱SNV，ddof=1；训练源StandardScaler |
| MC-UVE | 50次，PLS 1潜变量，保留20%＝123点 |
| 分类器 | RBF-SVM，C=1000，gamma=0.1×当前源scale |
| 类别权重 | balanced，仅从当前训练源标签计算 |
| 最终训练样品 | 855，七组350／5／100／100／100／100／100 |
| 主种子 | 42；另存43、44 |
| 真正新样品的准确率 | 未测，null；不能填96.02% |

配置ID`r2_source_fold03`只是历史配置出处，不是类别ZY03。模型使用全部七组。最终123个频点、scaler和MC系数均重新从当前完整训练源拟合，没有取旧折频点并集。全部scaler、MC内部PLS和SVC进入实际FitScope，共23034次已完成学习操作；290MiB原日志无损压缩并逐字节SHA回读，原件另保存在outputs，未删除。75个开发模型和3个最终模型均保留。

独立审查重载75个候选模型、复算12825条开发预测和选择结果；最终三个模型各855条单点／逆序／batch13预测及状态检查一致。有限技术检查不证明绝对零泄漏，也不改变历史曝光和当前批次范围。

## 文件入口

- 完整模型包：`reports/zaoren-deployment-20261031-v1/deployment/`。
- 主模型：`model_seed42.joblib`；参数和真实选频点：同名JSON。
- 模型、源码、环境、训练历史匹配参考与SHA：`bundle_manifest.json`。
- 空白模板：`deployment/templates/new_spectrum.xlsx`和`.csv`，不含合成光谱。
- 独立技术审查：`reports/zaoren-deployment-review-20261031-v1/`。
- 选频随机性／训练折差异：[源内稳定性图表](../reports/zaoren-deployment-20261031-v1/feature-stability/README.md)。

同855份、不同MC种子的最终选点两两Jaccard平均0.9169；15个重叠开发训练模型中，同训练折不同种子平均0.8938，不同训练折平均0.6633。108对通过独立布尔掩码运算复算，差0。它描述源内选点一致性，不是独立批次稳定性、化学峰证据或稳定筛选提准；没有把汇总掩码反馈给模型。

## 预测新样品

在仓库根、当前固定Python环境运行；无需MATLAB或Gaussian。环境版本见bundle manifest，加载器核对源码、环境及全部权重SHA。

1. 使用已给Excel模板，或新建模板：

```bash
.runtime/bin/python scripts/run_zaoren_deployment.py template \
  --bundle reports/zaoren-deployment-20261031-v1/deployment \
  --output outputs/新样品模板.xlsx
```

2. 在`Spectra`表填真实吸收系数，第一列频率值保持原样；单条样品一列，批量可横向增加唯一编号列。样品编号只是配对元数据，不能加入真实标签、预测类别、批次或厚度数值列。空白模板不能直接用于预测。

3. 执行预测，输出目录必须是新的：

```bash
.runtime/bin/python scripts/run_zaoren_deployment.py predict \
  --bundle reports/zaoren-deployment-20261031-v1/deployment \
  --input outputs/填好真实光谱.xlsx \
  --output outputs/新样品预测001 \
  --mode future --confirm-compatible-measurement
```

输出`predictions.csv`含编号、ZY预测类别、七列决策分数、种子43／44结果、历史匹配和负值标志；`prediction_lock.json`固定输入、模型及输出SHA。分数不是校准概率。主预测始终用标准SVC投票，不能以七个决策分数的最大项替代。

支持CSV、XLSX和严格三字段NPZ（`X_alpha`、`axis`、`record_ids`）。老师格式Excel的`吸收系数`表也可读取；原2578点只在精确包含冻结612点时明确裁剪，不插值、不改变分辨率。参考列排除，厚度后缀仅作质控。错轴、重复ID、公式、非有限值、平谱、标签字段均拒绝；负值保留并标记。

`--confirm-compatible-measurement`记录数据提供者对吸收系数定义、单位与测量条件一致的声明；数值轴匹配不能由软件自动认证仪器兼容性。工具面向相同类型酸枣仁七个已知匿名组，不具备已验证的未知类别拒识能力，也不能直接把黄芪／天麻输入解释成酸枣仁组别。

## 原855份样品应怎样查看

未来模式会拒绝已训练ID、原谱改名及1e-12容差内的数值重导出。该匹配是质控，不能单靠相似程度认定生物学重复。

查看原样品的历史留出预测，用独立入口：

```bash
.runtime/bin/python scripts/predict_zaoren_historical_oof.py \
  --input data/processed/zaoren_20261031/样本2_编号版.xlsx \
  --output outputs/样本2历史留出预测
```

本次已实际读取全部七份原Excel，855个预测与原留出折记录逐字段一致，821正确＝96.0234%、Macro-F1 0.9197、Balanced Accuracy89.9184%。零新拟合，这只是历史重放。全部34个错误与固定seed抽取的10个例子保存于[重放示例](../reports/zaoren-deployment-historical-demo-20261031-v1/README.md)。

若只检查完整部署模型是否能读取原表，使用`--mode training-diagnostic`；输出明确为训练内诊断、accuracy=null，不能作为独立预测性能。本次真实样本2的五条诊断已经运行，三个历史标志均为true。原表、改名舍入谱、未填模板均已实测被future模式拒绝，失败日志保留。

## 有真实标签后独立评分

预测阶段不接收标签。先固定上述预测，再另提供UTF-8 CSV：

```csv
record_id,label
NEW-001,ZY01
NEW-002,ZY04
```

```bash
.runtime/bin/python scripts/score_zaoren_predictions.py \
  --predictions outputs/新样品预测001 \
  --labels outputs/真实匿名标签.csv \
  --output outputs/新样品评分001
```

评分器核对future模式和预测锁后才读取真值，拒绝训练诊断、预测被改、重复／少／多ID或未知标签；不加载模型、不拟合。输出逐样品正确标志、Accuracy、固定七类Macro-F1、真值中有支持类别的Balanced Accuracy、逐类Precision／Recall／F1和混淆矩阵。单样本或缺类别会标记partial_class_coverage，不能与完整七类结果直接比较。实际新样品尚未提供，当前真实未来评分为NOT_RUN；测试中的合成夹具只验证计数和边界。

## 神经网络的后续依据

已真实整理480条旧源训练曲线、8640条epoch记录和两组PNG／PDF，见[源训练损失诊断](DEPLOYMENT_NEURAL_DIAGNOSTIC_20261031.md)。它们显示训练损失还在下降，但缺逐epoch验证记录，不能单凭此断言欠拟合。独立新pilot已实际完成6条96轮轨迹、24个固定checkpoint；只使用原outer0.inner0的342训练／342内验证，不接触外层171、不改变本部署参数。三个固定种子内验证均值如下，不能直接与旧855份成绩比较：

| 模型／epoch | Accuracy | Macro-F1 | Balanced Accuracy |
| --- | ---: | ---: | ---: |
| α CNN，24轮 | 64.9123% | 0.3957 | 44.7619% |
| α CNN，96轮 | 75.9259% | 0.6130 | 62.5680% |
| 四表示，24轮 | 76.6082% | 0.6429 | 62.7211% |
| 四表示，48轮 | 79.7271% | 0.6922 | 74.9320% |
| 四表示，96轮 | 83.8207% | 0.6989 | 69.4388% |

四表示48→96轮时源eval CE由0.3481降至0.0505，验证CE约0.7320→0.7294近平台，而Balanced Accuracy下降。不能仅凭训练损失下降或总体准确率增加认定更好的少数类泛化。全部12／24／48／96、三seed和正负变化均保存，没有按外层成绩选epoch。已独立重放24权重、复核393984条源／内验证score和4656项指标／CE，分类计数差0，CE最大数值差2.93×10⁻⁷小于事前容差10⁻⁵。见[完整学习曲线与边界](../reports/zaoren-neural-learning-pilot-20261031-v1/README.md)。

## 复现开发与重拟合

```bash
.runtime/bin/python scripts/run_zaoren_deployment.py prepare --run outputs/部署重现新目录
# 新source授权需绑定该目录protocol/preparation/current-source SHA；不能复用旧gate。
.runtime/bin/python scripts/run_zaoren_deployment.py select --run outputs/部署重现新目录 --gate <新的source授权JSON>
# 独立核对75选择模型及锁后另发full-fit release。
.runtime/bin/python scripts/run_zaoren_deployment.py fit --run outputs/部署重现新目录 --release <新的full-fit授权JSON>
```

配置选择是全855开发内部的固化步骤，不能将其分数或全855回代成绩作为新的确认准确率。现阶段论文仍以已审计MC-UVE＋SVM优化为主证据；匿名标签不妨碍实现与内部比较，跨批次效力仍需相应独立数据。
