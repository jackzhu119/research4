# 酸枣仁源内学习曲线pilot：实际执行结果

2026-10-31，北京时间。用户允许继续按ZY01—ZY07匿名七类。协议`zaoren-neural-learning-pilot-20261031-v1`，仅原`outer00.inner00`的342训练／342内层验证；这684条都是已曝光开发记录，不是新确认。**没有读取外层171条的特征／真值字节，没有外层评价，也没有改变MC-UVE＋SVM部署选择。**

## 事前锁定和实际运行

原`regularized24`的宽12、blocks2、dropout0.35、lr0.0005、Adam weight_decay0.001、balanced源类别权重及batch64全部固定；只把训练上限预定为96。两个家族`cnn_alpha`／`fusion_all_four`×seed42／43／44共6个fresh模型，分别独立初始化模型、优化器、生成器和仅342source拟合的scaler。同一轨迹在12／24／48／96保存24个快照。训练没有调度器、早停或依据验证曲线追加epoch。

`protocol.json`、`input_manifest.json`、root `source_training_gate.json`在真实训练前锁定。9项边界测试实际通过，其中冻结与pilot两份14条／2epoch合成夹具的训练loss和最终权重精确相同；合成测试不是新增药材研究。原内层truth原件SHA对冻结报告索引核验，源ID另与原tar的同上下文training.json绑定。

训练期间Python审计hook故意尝试open外层truth，在读取字节前拒绝，见`outer_read_rejection.json`；该hook同时拒绝旧data／旧神经数据和结果路径。它不是OS沙箱，也不证明绝对零泄漏。

## 固定检查点结果：三seed均值

| 家族 | 预定epoch | 内层验证Accuracy均值±样本SD（百分点） | 内层验证Macro-F1均值±样本SD | 内层验证Balanced Accuracy均值 | Source eval CE | 内层验证eval CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cnn_alpha | 12 | 56.1404% ± 4.6417 | 0.2915 ± 0.0364 | 37.1939% | 1.6163 | 1.6562 |
| cnn_alpha | 24 | 64.9123% ± 2.4982 | 0.3957 ± 0.0414 | 44.7619% | 1.2515 | 1.3252 |
| cnn_alpha | 48 | 72.8070% ± 2.4982 | 0.5391 ± 0.0324 | 53.9796% | 0.8917 | 1.0248 |
| cnn_alpha | 96 | 75.9259% ± 0.8933 | 0.6130 ± 0.0260 | 62.5680% | 0.5486 | 0.8450 |
| fusion_all_four | 12 | 73.9766% ± 1.7786 | 0.5324 ± 0.0116 | 54.8980% | 1.1955 | 1.2994 |
| fusion_all_four | 24 | 76.6082% ± 2.6799 | 0.6429 ± 0.0629 | 62.7211% | 0.7195 | 0.9274 |
| fusion_all_four | 48 | 79.7271% ± 1.6882 | 0.6922 ± 0.0310 | 74.9320% | 0.3481 | 0.7320 |
| fusion_all_four | 96 | 83.8207% ± 0.6753 | 0.6989 ± 0.0548 | 69.4388% | 0.0505 | 0.7294 |

24→96轮，αCNN的本内层验证Accuracy由64.9123%到75.9259%、Macro-F1由0.3957到0.6130；四表示由76.6082%到83.8207%、Macro-F1由0.6429到0.6989。这些是同一预定source／内层validation上的描述性变化，不是新五折OOF或独立确认收益，也不能与旧855份96.02%直接相减。

负结果同样保留：四表示的Balanced Accuracy从48轮的74.9320%降至96轮69.4388%；48→96验证CE仅0.7320→0.7294，源eval CE却0.3481→0.0505，末段源／验证损失分离。不能用总Accuracy上涨宣称所有类别或泛化单调改善，也不据此将96选为普遍最优。ZY02在本source和validation各仅2份，逐类波动需谨慎解释。

固定训练预算内观察到增加轮数的内层收益线索，同时也看到拟合间隙及指标取舍。要决定正式epoch／正则化配置，须另立训练内部选择协议，再按相应固定外层流程评价；既有历史曝光限制仍保留。本轮不选择“最好seed”或最高曲线点作对外成绩。

## 实际验证和交付

- 6条完整96epoch轨迹、576条逐epoch指标。每轨迹保存source-eval及inner-validation完整分数，合计393984行；每角色每epoch各342唯一记录，重复epoch／seed不增加N。
- 21项实际完成学习操作＝15个源scaler＋6个固定96轮NN操作；所有fit IDs均在当前342source。
- 每轮验证使用eval／no_grad，核参数、BN、optimizer、grad buffer及Torch RNG不变；24个快照对全部342validation做单条／换batch／换序审计。
- 独立finish进程重放全部24快照logits最大差0；4656项指标／CE检查计数指标差0，最大CE浮点差2.93×10⁻⁷，小于事前1×10⁻⁵。
- 另以纯混淆计数核所有固定快照的逐类P／R／F1／support与矩阵，见`independent_checkpoint_classes.json`。三seed的样本SD仅描述训练敏感性，不是生物学置信区间。

`epoch_metrics.csv`为每seed／epoch原始汇总；`epoch_seed_summary.csv`为192条家族×epoch描述汇总；`checkpoint_all_seeds.csv`保留24条预定快照、全部种子；`checkpoint_seed_summary.csv`为8行预定均值表。`checkpoint_class_metrics.csv`与`checkpoint_confusion_matrices.csv`给出逐类／混淆记录；每轨迹的raw score和24个`.pt`权重均在`trajectories/`。

每家族PNG及PDF实际导出并目检：源eval与验证eval CE使用同一组固定源类别权重，可进行源／验证差异描述；`source_optimization_loss`另保存在CSV，它含训练态dropout与逐batch变化权重，不能冒充固定模型CE。图中阴影为三seed描述性SD。

原7类、5／2折及96.02%强传统基线不改。旧loss-only诊断的NOT_RUN说明保留其完成时状态；本版是另行root放行后的真实source-only执行，不回写旧报告。没有跨批次／迁移／校准／分子计算，也没有把这次内层结果加入部署模型选择。

复现需要新空目录及新的root source gate，禁止覆盖本目录或复用旧授权：

```bash
.runtime/bin/python scripts/run_source_learning_pilot.py prepare --output outputs/new-source-pilot
# 重新核输入、脚本、测试及guard SHA，生成只允许6条source轨迹的新授权。
.runtime/bin/python scripts/run_source_learning_pilot.py train --output outputs/new-source-pilot --release <new-root-gate.json>
.runtime/bin/python scripts/run_source_learning_pilot.py finish --output outputs/new-source-pilot
```

实际训练和finish命令／UTC见`driver_events.jsonl`、`finish-driver-events.jsonl`；源码、协议和环境版本见`protocol.json`；`files_sha256.json`是包含最终日志／README／图形复核的交付索引。初始finish阶段索引另存`finish_phase_index_snapshot.json`，仅记录补充报告前的阶段，不作为当前目录完整性索引。
