# 当前论文结果入口：种类与产地修正后

本目录只收录2026-11-08确认种类和产地之后的分类、迁移及机制结果。
原855条旧混种任务的成绩、旧结果重贴标签得到的分析、旧论文汇编的历史章节均不在这里。

|入口|内容与适用范围|
|---|---|
|[材料、仪器、公式与划分写作稿](methods/MATERIALS_AND_METHODS_ZH.md)／[Excel明细](methods/data_splits.xlsx)|四药材统一测量条件；按18份有效Excel与逐ID冻结角色统计，另保留实际列头标称厚度|
|[任务范围](tables/task_scope.csv)／[六项固定留出](tables/fixed_classification.csv)|枣仁/酸枣仁二分类、四种类分类，以及酸枣仁、枣仁、黄芪、天麻各自的来源分类；训练/校准/预测数量分别登记|
|[种类判别按工作簿留一验证](results/kind_lowo_20261115/summary.json)／[固定协议](results/kind_lowo_20261115/protocol.json)|仅二种类与四种类主判别采用LOWO：10折864份、18折1672份；包含每工作簿指标、每ID折外预测、完整划分和独立核验回执；其余任务保留原协议与结果|
|[11月16日同划分全谱SVM对照](results/focused_followup_20261116/e1/summary.json)／[折外预测](results/focused_followup_20261116/e1/oof_predictions.csv.gz)|E1沿用V4的300酸枣仁TRAIN、15外折及原内层ID；375次已完成拟合，82.67%准确率；不增加独立样品数|
|[特征稳定性](results/focused_followup_20261116/e2/fixed_train_summary.json)／[交叉掩码](results/focused_followup_20261116/e3/summary.json)／[MLP诊断](results/focused_followup_20261116/e6/existing_checkpoint_diagnostic_audit.json)|E2只做排名；固定训练集部分的原始variation标签有误，新稿派生表更正为MC种子变化。E3参数近乎相同，不识别独立参数效应；E6回放旧模型，不是新神经训练|
|[固定留出迁移](tables/fixed_transfer_paired.csv)|三种子目标单训、额外目标预训、黄芪/天麻跨种类预训；负迁移保留|
|[主模型表](tables/all_final_model_settings.csv)|修正后原1305矩阵与新增V4 330项的已保存摘要；不同划分/预算分开解释|
|[协议目录](tables/corrected_protocol_register.csv)|仅修正后主线57项；display_id中的C1305对应原来源的OLD命名|
|[科学任务登记](tables/scientific_job_register.csv)|1305＋330已完成任务，备份与续算不重复计数|
|[B1](results/B1/results_summary.json)／[B2](results/B2/results_summary.json)／[B3](results/B3/results_summary.json)／[B4](results/B4/results_summary.json)／[B5](results/B5/results_summary.json)|五轮1380任务包的全部正负结果汇总；各轮指标/机制/不确定性明细以tables/B*/下CSV.gz保存|
|[四张主图](figures/)|仅从修正后的V4结果生成，提供PNG、SVG及PDF|
|[输入与协议](inputs/)|18表有效登记、1672行修正元数据、固定999/334/339角色；原始光谱仍使用仓库有效清单位置|
|[来源清单](manifest.json)／[当前包校验](verification.json)|每个文件的固定Git来源、原件SHA-256、变换方式与当前文件校验值|

## 任务边界

- 纯酸枣仁来源只包含ZY03—ZY07；纯枣仁来源包含ZY01、ZY02、ZR01—ZR03，种类标签分开。
- 种类判别LOWO按完整工作簿逐一留出，每份样品恰有一次折外预测；原TRAIN/CAL/PRED仍作为来源记录保留，不改写冻结角色。原固定留出种类结果保留供溯源，不能替代LOWO的主种类指标；来源判别、特征预算、迁移与支持集分析不改用LOWO。该评价仍基于既有集合，不能视为新采购、采集日期、仪器或真实独立新批次的前瞻验证。
- 枣仁新增三个来源各只有3份实物，每个角色1份；相关比率是小样本描述，不应与酸枣仁五来源300TRAIN三次五折成绩直接排名。
- `legacy_TRAIN_*`、`OLD-*`及`P-CV-OLD`出现在来源表中时，指修正种类后的原1305矩阵，不是修正前旧855混种结果；算法名`legacy_lambda1`也不是数据版本。
- 原1305最终轻量摘要仅汇总源模型预测，不覆盖全部校准明细；完整预测和校准输出保留于固定的纠正来源Git归档。本目录不复制大型预测包或模型权重，保留其已完成复核的来源链。
- 本目录完整保留所收录协议的负结果，不按准确率挑选设置。B1—5仍属已曝光样品上的内部探索，E05真实新批次评价为`NOT_RUN_NEEDS_NEW_BATCH_DATA`。
- 元学习尚未完成的最终结果和量子计算不进入这个分类结果入口；量子资格应由对应分子、数值模型和验收证据单独判断。
- `acceptance/`里的回执描述原固定提交的核验范围；新生成的`verification.json`只认证本目录文件、标签、计数与来源，不能当作全部模型重训或外部新批次认证。

维护命令：`python scripts/build_current_paper_results.py --build` 按固定白名单生成；`python scripts/build_current_paper_results.py --verify` 校验本地文件、固定Git原件与18份有效原谱。两者均不联网、不训练模型。

LOWO独立只读复核：`python scripts/verify_kind_lowo_20261115.py`。所复制回执中的README/manifest输入哈希对应原计算时的历史上下文，不能改写成扩展后当前包的哈希；特征矩阵和两任务拟合参数NPZ保留在固定来源提交，当前只复制22份必要协议、摘要、预测、划分和核验文件。

仪器和采购信息来自[本轮用户声明](../../data/annotations/acquisition_conditions_user_declaration_20261112.json)。新增方法表读取原Excel实际列头和冻结角色，Excel依赖版本见[requirements-paper-methods.txt](../../scripts/requirements-paper-methods.txt)；原始光谱与冻结分配保持不变。
