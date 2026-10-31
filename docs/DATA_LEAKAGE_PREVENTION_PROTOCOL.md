# Research4升级：可执行的数据隔离与验证协议

协议ID：`upgrade-20261031-v1`。2026-10-31，北京时间。依据用户本轮完整的《严格的数据泄漏防护与验证规则》，数据隔离优先于涨分、模型数量和方法复杂度。

## 1. 结论边界先于模型选择

原96.02%＝821／855及全部历史实验保持原样，标记为已曝光、同批次、回顾性开发结果。重分855份或更换种子不能产生未见确认数据。哈希、锁和本项目执行日志是可追溯证据，不是外部独立执行认证。本协议及有限测试均不证明“绝对零泄漏”。

独立性为用户声明的酸枣仁可继续作明确标记的`EXPLORATORY`内部研究。黄芪、天麻缺实物映射时不能把谱列改成已核验独立样品；数据库保留`unknown`。新独立样品／批次确认集未建立时，明确`BLOCKED`，不静默降格为PASS。可明确授权探索，但不能将其计作独立确认。

最新任务允许研究多表示、深度学习及Gaussian／DFT，覆盖旧主线“不加入Gaussian”的方向性约束；不改变历史研究当时的范围。计算曲线不算新的真实样品，理论峰先验也遵守训练内建立或事先固定规则。

## 2. 数据血缘、身份与固定分区

元数据应具备dataset_id、record_id、physical_sample_id、parent_sample_id、preparation_id、batch_id、measurement_id、modality、label、source_file、source_hash、independence_evidence。物理、父样品、制样及测量ID须在项目中具备明确全局作用域。

`validate_lineage_partitions(records, partitions)`实际检查记录及共同祖先、重复测量／制样跨角色交集。父样品可以引用另一record，也可以引用已知物理实体；循环、同实体标签冲突和同记录的模态身份冲突会抛出`LeakageError`。不同批次标签不自动证明不同物理祖先；已知共同祖先不能因换厚度、重新压片、增强或改名跨分区。

缺身份的记录返回`BLOCKED`；允许的用户声明只返回`EXPLORATORY`。代码中的`TECHNICAL_CHECK_PASS`仅表示该次明确技术条件通过，返回值始终保留`independent_confirmation_established=false`。独立性断言的真伪仍需采样日志和来源证据，数字不重复或高相关都不能代替。

`validate_fixed_partitions`保存并复核角色ID及顺序摘要、唯一性和总体覆盖，不搜种子。`assert_modalities_aligned`要求各表示的ID与目标顺序完全一致；模态缺失、错序及重复不会静默对齐。一次比较应使用相同有效样品集合；集合改变时另列结果。

只有一个来源批次／产地时，已知产地跨批次问题不可识别；按文件留出未知类不能冒称已知类的批次验证。ZY02只有5份，继续固定旧5外折／2内折时核对内外类覆盖，不能复制或删类凑折数／shot。

## 3. 实际fit入口和源输入指纹

新入口使用`src/herb_spectra/leakage_guard.py`的`FitScope`。每个内层和外层源分别创建scope，不按模型名推断数据范围。

```python
scope = FitScope(source_ids, "outer0.inner0", dataset_version,
                 log_path, code_version=source_code_hash)
scaler = scope.fit(StandardScaler(), X_source,
                   sample_ids=source_ids, component="standard_scaler")
with scope.operation("neural_training", source_ids, X=X_source, y=y_source):
    # 独立初始化model／optimizer／scheduler，仅使用当前源训练。
    train_source_model()
```

守卫在fit执行前检查IDs、行数和当前允许集合，拒绝越界并写`fit_rejected`；成功和失败均保存`fit_sample_ids`、源数值／标签指纹、dataset_version、split_id、code_version及真实UTC。实际fit得到的对象附带源provenance，不只保存一个声明性“无泄漏”字符串。

普通数值特征接口拒绝字符串ID／路径矩阵；模型输入由数据库的光谱数组显式构造。**数字化的ID、顺序或批次若被调用者故意塞入数值矩阵，守卫不能自动识别其语义**，仍需输入schema及代码审查。ID声明不能凭日志自动变成生物学真值。

SNV及预定单谱滤波与跨样品学习分开。StandardScaler、MSC参考、PCA、PLS、特征筛选、类别权重、预训练、插补和概率映射等只用当前源；验证／测试仅transform。数据驱动频段、网格、阈值及网络配置均在内层选择，不能用全数据峰差后选窗口。

### MC-UVE内部PLS的实际钩子

```python
with instrument_mcuve_pls(scope, Z_source_float64, source_ids):
    selector = scope.fit(MCUVESelector(...), Z_source_float64, y_source,
                         sample_ids=source_ids, component="mcuve",
                         groups=source_ids)
```

该上下文暂时包裹既有MC-UVE系数函数与每个`PLSRegression.fit`。通过已绑定源行的精确数值指纹恢复实际MC子样本ID，核对其标准化Z，分别记录系数标准化和每个PLS拟合的真实子集。`finally`恢复原函数；不改旧源文件或MC数学。正对照测试逐元素确认原系数、稳定性和选点不变。

此钩子覆盖既有全宽、唯一源样品的MC-UVE系数路径；不把它冒称CARS／聚合重复组或任意第三方库的通用全局拦截器。数值完全重复而不能唯一绑定的源行会拒绝；需另提供可验证的组绑定，不凭分类分数删除。钩子期间禁止在同一进程并发PLS线程，独立拟合可用不同进程。

## 4. 训练、推理、评分及缓存

训练入口不接收外层真值，推理只接收features，先保存ID／类别顺序／预测／分数，评分再连接真值。数据进程若仍能访问磁盘上的全库，要如实标明是API级边界，并非已经实现操作系统文件权限沙箱。

`audit_pointwise_inference`实际调用`eval()`及`no_grad()`，比较单条、反序、固定置乱顺序和不同batch大小。逐张量比较参数和buffers，覆盖BatchNorm running mean／variance／计数器；拒绝`track_running_stats=False`的BatchNorm及检测到的测试批次统计依赖。误差界事先设为`atol=rtol=1e-5`。输出一致是当前覆盖样本的数值检查，不是所有未来输入的数学保证。

网络、优化器和调度器每内折／外折／种子独立初始化。早停／checkpoint／训练轮数只按内层规则。禁止测试损失选epoch，不能以超过96.02%作为搜索终止条件。所有固定种子均报告，不挑最佳。

`guarded_cache_key`绑定dataset_version、实际源数据／标签指纹、有序训练ID摘要、split_id、preprocessing、selector、code_version和seed。返回值兼容字符串，但保存/加载要求完整identity。`save_guarded_cache`验证实际fit provenance；`load_guarded_cache`再核上下文、源provenance及payload字节，跨折对象即使改文件名也会拒绝。缓存不可覆盖。此为科学来源检查，不是针对恶意pickle的安全沙箱。

纯逐样品固定转换可单独缓存，但仍保留来源。需要学习的缓存不能仅凭文件名、模型家族或全数据掩码复用。

## 5. 多表示、迁移、增强和校准

吸收系数、折射率、时域和频域是同一测量链的相关表示，共享物理分区；不是四批独立样品或四个独立传感器。MC选点只来自当前源折，不使用历史折频点并集。非连续UVE点用显式变量分支，不伪装成连续卷积轴。

迁移采用归纳式主协议：目标内层验证／外层测试的任何模态、增强或无标签版本均不得参与自监督、伪标签、域对齐或重构。预训练权重必须有源身份与hash；不同植物保持独立标签空间／分类头。来源及配比在允许开发数据内选择。无法证明来源、物理单位或频率兼容时，记录阻塞；不把没有运行误称负迁移。

增强和过采样在分区后、当前源内部执行，保存父ID，合成数量不增加真实独立样品数。固定测试扰动是原评价单位的鲁棒性诊断，不能按测试涨分选强度。

source、support、query共同祖先互斥；query先固定，再按预定seed抽support。每次支持抽样从同一冻结基础模型重新开始；回退规则和强度不看最终query标签。同结构零支持控制区分标签与额外计算。温度／概率校准只用有效source OOF或独立开发池；1-shot支持不能同时作为独立校准验证。

Gaussian分子／方法／理论峰先验需记录来源并事先固定或在当前源建立。事后SHAP、注意力及峰解释明确标诊断，不反馈同版本选择；不能按测试真实类别给不同先验。

## 6. 实际检查矩阵及覆盖边界

| 检查 | 可执行证据 | 当前最小核心状态 |
| --- | --- | --- |
| A 分区／共同祖先／增强后代 | 故意跨角色同实物、父样品、重压片、增强及循环样例均报警 | 已运行，通过其明确技术检查 |
| B fit范围 | 越界fit在执行前被拒；真实source/scaler和MC内部PLS逐call日志；原MC结果等同 | 已运行，通过其明确技术检查 |
| C 只置乱固定外层真值 | 需重跑完整内层选择并比配置／变换／频点／权重／预测 | 不由本15项测试代替；独立实测结果见run audit，未执行前NOT_RUN |
| D 只改变测试输入 | 需核训练产物不变，允许对应预测变化 | 同上，未执行前NOT_RUN |
| E 点式推理 | eval/no_grad、批量／反序／置乱、BN与故意状态更新或batch依赖 | 已运行，通过其明确技术检查 |
| F 缓存 | 8类identity字段变化、跨折对象和不完整identity拒绝；正例可还原 | 已运行，通过其明确技术检查 |
| G 多表示 | 误配、错序、重复及同实体另一模态跨角色均报警 | 已运行，通过其明确技术检查 |
| H 训练标签置乱 | 在固定物理分区重跑完整选择训练，经验分布与多数类基线 | 独立实验，未执行前NOT_RUN；不要求1/7 |
| I 真实预测指标复算 | 每外层／种子每record恰一条OOF，折均值与pooled分开 | 由实际runner独立计数；核心单元测试不冒充完成 |

核心实际命令和15项测试日志位于`reports/upgrade-20261031-v1/leakage-core/`。其他实测以对应run的`leakage_audit.json`、`fit_scope_manifest.json`、原始预测和选择锁为准。开发／确认数据身份见`docs/DATA_EXPOSURE_REGISTER.md`，血缘见`data/metadata/sample_lineage.csv`；缺项不补造。

## 7. 明确失败时如何处理

发现直接泄漏，停止该路径正式评分，保留失败产物；修复使用新代码／协议版本重跑，不仅改报告。发现生物学独立性、批次或许可不足，记录BLOCKED／EXPLORATORY，不能标独立性PASS。修订协议不覆盖旧96.02%或其他历史成果。

最终汇报分别给出实际执行通过的技术检查、未解决独立性／批次问题、结果的内部开发／跨任务／独立确认身份，以及证据不足不能写进论文的结论。任何检查未运行须写NOT_RUN；多任务重训不等于同一模型无需训练可识别其他药材。
