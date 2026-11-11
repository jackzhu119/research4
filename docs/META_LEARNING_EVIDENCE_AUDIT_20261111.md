# 最近回传结果审查与元学习补充方案（2026-11-11）

证据分支：`research/b-mechanism-round5-20261111`。本审查通过 GitHub connector 只读访问该分支的 AGENTS、物种修正结果及 B 方案一至五轮报告和执行模块。没有重训、修改原始数据或历史工件。

## 必须采用的最新边界

- 18 工作簿、1672 份用户声明独立实物；中文种类标签为枣仁364、酸枣仁500、黄芪403、天麻405。ZY01/02为枣仁，ZY03–07为酸枣仁。
- 2026-11-08 的独立性补充声明已覆盖新增817记录；不能继续沿用 main 中“新增记录独立性未知”的旧描述。独立性仍是用户声明，没有新增真实批次ID。
- 当前固定名单是 `data/research/species_revision_20261108/split_manifest.csv`，TRAIN999/CAL334/PRED339；酸枣仁300/100/100。新开发仅在 master TRAIN 内构造Source、Support、Query，不能重洗全500样品或重新使用CAL/PRED挑方案。
- 既有 E03 在酸枣仁300份TRAIN内采用三重复五折：每上下文Source190（每类38）、Support50（每类10）、Query60（每类12），15个上下文。重复Query行不增加独立实物数。
- 每个原ZY来源仅一批；五来源文件是五类，不是五域。此次元学习属于同批已知来源的内部表示学习探索，不是真实新批次或未见类别验证。

## 已经完成的证据，避免重复和过时结论

`PAPER_WRITING_START_HERE.md`及论文汇编：纯酸枣仁当前主开发候选是α＋SNV＋MC-UVE20＋调参RBF-SVM，V4内部开发Accuracy 93.4444%、Macro-F1 0.934401；MC100准确率相同，稳定5×MC20为92.7778%，收缩LDA91.7778%，n＋α双SNV融合91.6667%。该协议每外折训练240、测试60，K/C/γ由内2折选择。它与E03的Source190＋Support50＋Query60任务不同，不能把93.4444%和E03的80%直接相减解释算法增益。本轮因此在相同E03角色中补上Source内调参的MC20谱筛选强对照及完整支持池重训。

论文汇编的128点迁移神经协议和612点调优SVM也需分列；不能把训练步数、数据量或表示差异当作元学习目标本身的效果。

`docs/SPECIES_REVISION_RESULTS_20261108.md`：

- 纯酸枣仁五来源固定100份PRED上，R2训练300份为87%；加入100份CAL完整Pipeline重训为93%。R0分别94%/96%，与内层选参排序相反；不能看PRED后改称R0是预先最优。
- 强Anchor K1/3/5/10为75.90/81.50/84.05/86.30%，Identity87%；弱校准基本保持87%。完整重训不是冻结分数校准。
- 跨药材预训练三种子均值79.00%，目标单训81.00%，等额额外目标预训练80.33%；没有显示正迁移。
- 枣仁来源97.30%准确率伴宏F1约0.5972、平衡准确率60%；多数类基线94.59%，不能当可靠五来源识别。

`docs/B_MECHANISM_GAP_RESULTS_20261110.md`（105/105封存）：

- E03 Identity80.00%；强Anchor K1=69.89%、K10=78.78%；Bias K1=77.33%、K10=79.78%；弱修正几乎Identity。
- 完整Pipeline重训K1/3/5/10=80.00/80.67/82.11/83.89%。新增带标签样品重训是必须保留的强对照。

第二轮（345/345）：已经比较支持重排、乱标、对角Anchor、分数乘0.1/10、Source RMS归一化、Scaler×gamma、权重×来源覆盖、嵌套学习曲线。不要把这些再次当作新缺失对照。

第三轮（375/375）：已经交叉sum/mean损失和截距惩罚；用Source+Support无标签统计、真实标签及循环错标分解；10/20/40/100%支持污染；权重与SVC对偶边界审计。人工污染不证明原标签错误。

第四轮（375/375）：

- 冻结Scaler和数值gamma，增加Source C以匹配总损失质量，E03仍80%；新增Source+Support固定C或归一化损失质量K10均83.78%。不能把重训收益仅解释为正则总量变化。
- 已做C=0.1/1/10×balanced/None、弱修正margin激活证书、固定支持总量而改变类别构成及类别均衡残差加权。
- 归一化/加权可减少强Anchor误伤，但不保证胜过Identity。弱修正分数移动但不跨边界，不代表学到了有用新信息。

第五轮（180/180）：

- I01只在Source内部三次留出选择Identity/Bias/mean-Anchor/冻结坐标完整重训。E03选后K1/3/5/10为79.22/79.22/81.89/83.67%，同期完整重训80.11/80.22/82.11/83.78%；策略选择未稳定超过强重训。
- I02已做隐藏候选标签的随机、不确定性和多样性获取，覆盖与回退、实际观察标签量均计数。不要称主动获取是尚未探索的空白。
- I03已比较保持OVR输出锚的OVR/OVO/OVO-PCA残差输入；E03 K10分别80.00/80.44/80.33%，收益小，不能称确认性优胜。
- I04已比较λ1、λ=N和λ=N²/ESS。ESS收缩可缓和稀缺类损害，但多个条件仍低于80%Identity。

## 现阶段最值得补的真正算法缺口

1. **可学习谱表示的episodic元学习**。前五轮主要在冻结SVC分数几何中校正；不是表示层的few-shot元学习。建议ProtoNet作为首个实现：训练episode内support/query物理ID互斥，query损失必须反传至encoder，不能把重复普通训练或支持重抽样改名元学习。
2. **同网络、同输入、同优化预算的普通监督学习**。与episodic encoder共享初始化、网络容量、预处理和种子；最终都用同一Support原型接口评分，分离网络容量与episodic目标的效果。
3. **不学习/随机encoder和原始标准化谱的最近类原型**。相同Support与Query，只去掉表征学习。将其与support-only ridge/线性分类器、source-only模型和Source+Support完整SVC重训一并报告。
4. **零支持与接口一致性**。SVC原生投票、decision-score argmax及校准argmax分列；新原型网络的K0应明确是Source类质心或无适配读出，不能伪造没有支持的原型。
5. **支持标签贡献和采样方差**。保留K1/3/5/10、固定嵌套support、同一Query；报告纠错/误伤、逐类召回和配对结果。不另加大量重复，仅为变窄区间。

不建议继续大网格搜索Anchor强度/分数尺度，或再做已完成的C、污染、ESS、获取策略矩阵。以上结果表明首要瓶颈是可迁移表示和新批次数据，不是欠缺更多相同分数变换。

## 可复用实现和新的工程核查

- `scripts/species_revision_background_neural.py:97` 的 `load_train_only` 将完整特征包立即限到固定TRAIN白名单；`:131` 的 `development_partition` 为source/support/query保留互斥ID；`support_draws`保存20次嵌套K抽样。
- `src/herb_spectra/upgrade_neural.py` 的 Encoder/SNV/state hash可作为匹配既有网络的参考；现有脚本的A/Aextra/B是普通监督/预训练，不是episodic元学习。
- `scripts/b_mechanism_engine_20261110.py` 可复用E03设计、角色核验、原生预测与完整工件核验；`scripts/b_followup_common_20261110.py` 可参考TaskWriter与原子封存机制。
- 不要将旧128点0.21–1.49模型直接当612点0.1–1.5对照；比较必须同坐标。n+α应分模态SNV后拼接，并记录双倍输入维度。
- 新encoder最好采用LayerNorm/无运行统计的归一化，或保证评估BN只用Source冻结统计；不能把Support与Query拼一起更新BN，产生未声明的传导式适配。
- 检查点必须保存encoder、optimizer、完整RNG状态、global step、配置/输入/分区hash；恢复测试应证明连续训练与保存恢复后的下一步一致。训练不能因准确率、耗时或CPU利用率阈值自行早停；不可恢复的数据漂移与非有限数值仍需显式记录真实错误，不能静默继续产生无效结果。

## 来源

上述文件均读取自 `research/b-mechanism-round5-20261111`：AGENTS.md，SPECIES_REVISION_RESULTS_20261108.md，B_MECHANISM_EXISTING_EVIDENCE_AUDIT.md，B_MECHANISM_GAP_RESULTS_20261110.md，B_MECHANISM_FOLLOWUP_RESULTS_20261110.md，B_MECHANISM_ROUND3_RESULTS_20261110.md，B_MECHANISM_ROUND4_RESULTS_20261111.md，B_MECHANISM_ROUND5_RESULTS_20261111.md及其protocol、相关scripts。此次只读审查不代表复算全部逐预测记录；报告数字来自已封存报告，不将其当本轮新成绩。
