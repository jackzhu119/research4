# 水果分组回归：源内选择已完成，外层尚未执行

已依据root源release，在冻结代码及80个上下文上完成 **7560个源估计器拟合、840个候选／外层配置评价，0失败**。实际wall时间145.039秒，20组外层的PLSR、full-SVR、MLP、MC-SVR及source-policy配置均已锁定。这里没有外层预测或外层真值评分，不能将内层选择分数称为最终模型性能。

## 实际模型与计算

| 模型／组件 | 完成次数 |
| --- | ---: |
| PLSR估计器 | 1080 |
| full-SVR估计器 | 4320 |
| MLP估计器 | 720 |
| MC-SVR估计器 | 1440 |
| X StandardScaler／y StandardScaler | 各1080 |
| MC完整排序 | 720 |
| MC内部PLS | 10800 |

合法当前source缓存使两个保留比例共享同一排序；每次内部PLS均有FitScope记录。没有fit_failed或fit_rejected。720个MLP均依预注册设置跑满300迭代，并全部产生ConvergenceWarning（未证明已收敛）。这些警告已保留；不因为看到了本次结果增加epoch，也没有把警告改成成功收敛。

全部trial计时之和128.912秒，包含缓存、转换、估计器、推理检查和模型保存；145.039秒wall还包含启动、完整检查和汇总写出。不是纯CPU训练时间或可推广的速度基准。

## 配置锁与证据

私有运行目录：outputs/public-thz-fruit-regression-20261031-v1。selection_lock.json SHA256为50ddfb3d44003e38714cff5b6eb0fb462a82569a36aa9229d7430a1da0445783；冻结模块SHA256仍为bcfc427dd8d1c6a27d072304a9c40af38c65e3b6cdf12d0d7bd261c6e7467567。

source_selected_configurations.json包含全部20外层每个角色的选择、参数和各42／43／44种子的源内OOF汇总。每种子按32条当前source OOF预测计算RMSE／MAE，再按三种子平均进行选择。各任务的policy选择频数仅是源内诊断：

| 独立任务 | source-policy各家族被选外折数 |
| --- | --- |
| MA-Brix | MC-SVR 2，full-SVR 2，PLSR 1 |
| PM-Brix | full-SVR 3，MC-SVR 1，MLP 1 |
| PR-Brix | PLSR 3，MLP 1，full-SVR 1 |
| MA-DM | MC-SVR 4，full-SVR 1 |

这不是利用外层成绩选出的“最佳模型”；下一步仍须独立复算选择锁并由root签发新的外层release。

validation.json保存拟合／warning／成本汇总及私有原始证据哈希；source_integrity_recheck.json记录源选择完成后，对全部模型、元数据、缓存以及80个准备上下文的再次字节核验。执行计划、实际开始／完成时间和stdout／stderr为真实命令记录，不含逐样本数据。source_failures.json为空列表，保留此实际阴性事实。

原波形、目标值、逐样本源预测、完整trace和模型仍只留在忽略的outputs目录。没有公开未知许可的逐样本产物，没有改人参接口或酸枣仁历史结果，未提交Git。

## root后续消费的产物

- source阶段：candidate_summaries.json、selection_lock.json、inner_records.json均在私有运行目录；本公开报告提供配置／指标／成本聚合及哈希。
- 另gate后的evaluation：outer_model_records.json、outer_role_aliases.json、outer_predictions.csv.gz与prediction_lock.json。产生预测时不连接真值。
- 预测锁定后的score：seed_summary.json、pooled_metrics_by_seed.json和fold_metrics.json供最终表；scored_outer_predictions.csv.gz留本地，不能推送。

水果任务仍是10果实／40相关时间点记录的工程回归。时间／频率单位、Brix／DM测定方法及许可未完全核实，4个日期在source和query重叠；不是跨时间或新批次确认，更不是酸枣仁分类提高或THz-TDS跨药材迁移。
