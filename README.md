# 中药太赫兹光谱：当前有效数据与论文结果

**论文从 [修正后的结果包](paper/current/README.md) 开始。** 枣仁、酸枣仁及各自产地已经分开登记。用户于2026-11-12撤回早期错误混种标签产生的模型结果、旧预测重贴标签分析和混合论文汇总；这些成果从当前分支版本删除，不再列作论文结果。

## 当前有效数据

当前版本为 `2026-11-08-confirmed-origin-kind-v1`，18份工作簿、1672条谱记录。原来的855条原始光谱继续有效，删除对象是错误标签产生的派生结果。

|种类|工作簿|谱记录|
|---|---:|---:|
|枣仁|5|364|
|酸枣仁|5|500|
|黄芪|4|403|
|天麻|4|405|
|合计|18|1672|

数据文件入口为 [有效工作簿清单](data/active_workbooks.json)、[文件与产地说明](data/CURRENT_DATASET.md)和[名称对应表](data/confirmed_names_20261108.csv)。论文结果使用 [冻结的修正输入与角色](paper/current/inputs/)，其18份原谱哈希已与当前文件逐项核对。

用户的原855条独立样品声明与[新增817条独立实物声明](data/annotations/specimen_independence_user_declaration_20261108.json)分别保留。中文种类、来源和独立实物属于用户声明，不等于拉丁植物学鉴定或真实跨批次验证；原上传文件拼写保持不变。

## 论文保留的结果

- [材料、仪器、THz公式与数据划分](paper/current/methods/MATERIALS_AND_METHODS_ZH.md)及[Excel明细](paper/current/methods/data_splits.xlsx)：四药材统一实验条件，18份原表逐表统计，另列1672条样品的冻结角色与原始列头。
- [六项修正分类任务](paper/current/tables/task_scope.csv)：枣仁/酸枣仁二分类、四种类分类，以及每一种类各自的来源分类，角色与分母分别列明。
- [修正后的1305＋V4 330项矩阵](paper/current/tables/scientific_job_register.csv)、[协议与模型设置](paper/current/tables/corrected_protocol_register.csv)、[主图](paper/current/figures/)及三种子迁移结果。
- B第1—5轮的全部正负结果，共1380唯一任务包；不把备份、续算、支持集抽样或验证当成新实物样本。
- [来源白名单](paper/current/manifest.json)与[清洁结果包核验](paper/current/verification.json)：没有复制旧混种成绩或旧整包合格证书。

原儿茶酸、毛蕊异黄酮和斯皮诺素的分子计算不受此次分类标签错误影响，保留在对应计算分支；其最低点、频率、IR与论文资格仍须按各分子验收，不能替代药材实测峰归属。元学习未完成的最终结果不列入已完成的论文分类结果。

## 写作与维护

论文输入只允许读取 `paper/current/manifest.json` 明确列入的文件。不得从Git旧提交、旧模型或旧汇总表恢复被撤回的分类成绩。名称中的 `legacy_final`、`legacy-prediction-completion` 和原1305来源的 `OLD-*` 指修正后的原矩阵，不属于此次错误混种结果。

```bash
python scripts/build_current_paper_results.py --verify
```

验证会检查固定来源、文件哈希、修正种类/产地任务、18份原谱和计数，并拒绝额外混入的文件；不会训练模型。正确实验的负结果继续保留，E05真实新批次验证仍未运行。

[当前分支与任务状态](docs/repository/README.md) · [删除规则与清理记录](maintenance/) · [论文使用规则](PAPER_RESULTS_POLICY.md) · [参考文献及原始资料](references/paper-materials/README.md)
