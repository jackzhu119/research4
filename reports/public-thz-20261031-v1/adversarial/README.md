# G1独立对抗诊断：先冻结后执行

状态更新：root放行后已完成固定小预算真实运行，见[实际结果与预算勘误](EXECUTED_RESULTS.md)。下文保留放行前说明，原计划/源码/fixture快照不改写。

这是原公开人参42条观察记录上的专门防泄漏诊断，不增加主基准候选，不参与主模型排名。它不会产生新的生物学独立确认集。G1的7条记录仅作固定诊断query，另外35条记录是source；export组不解释为真实原料批次。

## 已实现，当前未训练

`scripts/run_public_thz_adversarial.py` 实现独立计划、输入快照、哈希绑定root gate、source-only拟合及后验诊断；`tests/test_public_thz_adversarial.py` 使用合成数组和真实临时文件，不拟合研究模型。

预先固定seed42、alpha、SNV、MC-UVE(10次、LV1、保留13/65)、RBF-SVM的C=1或10、gamma=scale、balanced。每次运行以五个source export组LOGO评价两候选，按Macro-F1、Balanced Accuracy、Accuracy、候选ID排序，再拟合全部35条source；所有转换与MC内部PLS只在当下source拟合。

诊断包括：

1. 两次完整source选择/拟合。分别以原query真值文件及置乱真值文件作为禁止读取的陷阱；实际读取尝试必须被拒绝。比较参数、特征索引、拟合状态哈希、7条预测及分数。
2. 冻结同一个拟合模型，仅修改query的alpha[0,0]加0.01；拟合状态和其余6条分数/预测必须不变。
3. 在固定五组、固定成员下按10个预先固定种子分别组内置乱source标签；每轮重新做两候选完整内层选择、最终训练和预测。禁止只打乱评分标签或复用原标签训练的selector。
4. 全部诊断选择与预测文件落盘锁定之后读取真实G1标签。计算经验null分布与(1+超越次数)/11，最低分辨率1/11，不能据此报告高精度显著性。
5. 多数类由35条source标签选择，G1上实际评价，不能直接填50%。

预算：12次完整流程，每次11个SVM与60个内部PLS，合计132个SVM与720个PLS拟合。只用seed42；不因为外层表现挑种子或加候选。

## 真值边界的含义

Python命名文件入口`builtins.open`、`io.open/Path.open`、`os.open`实际拦截指定真值路径：拟合阶段拒绝读，所有阶段拒绝写；评分阶段允许只读。临时合成fixture已经验证这些拒绝实际发生，并验证hook在异常退出后恢复。

这是进程内特定入口的可执行检查，不是操作系统安全沙箱。它不声称拦截预先打开的整数文件描述符、任意本地扩展或所有潜在I/O。原始观察记录的上游独立性仍未知。

## 后续执行门

主runner代码冻结后才生成新的`prepared/`输入与`frozen_manifest.json`；之后由root审核source SHA与本目录计划，签发含`allow_adversarial_source_training=true`、本协议ID及manifest SHA的release文件。子任务不得自行签发release。

```bash
python -m unittest discover -s tests -p test_public_thz_adversarial.py -v
python scripts/run_public_thz_adversarial.py prepare --out reports/public-thz-20261031-v1/adversarial/prepared
python scripts/run_public_thz_adversarial.py run --out reports/public-thz-20261031-v1/adversarial/prepared --release ROOT_REVIEWED_RELEASE.json
```

本文件不代表已执行第二、三条命令；实际状态和SHA以新增preflight及prepared产物为准。没有训练结果时不得填入准确率、null值或PASS结论。
