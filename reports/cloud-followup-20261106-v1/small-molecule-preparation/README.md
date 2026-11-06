# 原儿茶酸云端独立小分子准备

本目录实际完成：可信登记的中性原儿茶酸图复核，以及 ETKDGv3 / MMFF94s 模型构象生成和有上限的力场优化。没有进行 SCF、DFT 优化、Hessian 或频率计算，没有分配或增加量子预算。

结构来自先前已核验的 ChEBI:36062 / PubChem CID72 图；原始来源、图审计和登记CSV均以 SHA-256 固定。计算输入为 **模型三维坐标**，不是下载的实验晶体结构或已优化DFT结构。分子式 C7H6O4、17原子、80电子、中性单重态、完整Key `YQUVCSBJEUQKSH-UHFFFAOYSA-N` 已由实际 RDKit 图检查。所有候选、力场收敛状态与能量均保留；选择规则预先固定为力场已收敛候选中的最低MMFF94s能量，再按构象编号打破平局。力场能量不是DFT能量，也不保证全局最低构象。

`prepared_manifest.json` 是可供 root 明确授权的新独立 DFT pilot 输入。新 worker `scripts/research4_cloud_small_molecule_20261106.py compute` 默认不能执行；必须 `--execute`、实际原V8 guard、绑定脚本/输入/输出的root授权、原V8累计账本390160.830836 CPU秒上限和2026-11-09T12:30Z截止同时成立。

建议根任务在剩余资源评估后选择 **不超过10000 CPU秒** 的单次试算，最多35个优化step。该数是试算上限，不是经过实测的完成成本。已有75原子Spinosin的时间不能直接外推本分子。新优化会逐step保存坐标、梯度、能量及SCF checkpoint；优化、fresh SCF与梯度门禁通过后才运行原生完整stock RKS Hessian。全部预期45个有符号模式必须真实输出，保留虚频；IR始终NOT_COMPUTED。

本分子是酸枣仁相关文献中的小分子方法与氢键辅助候选。它不能代替斯皮诺素机制计算、纯品THz、配对HPLC/LC-MS测量或酸枣仁独立样品，不进入既有分类器。纯孤立分子频率不能确认复杂固体药材吸收峰归属。

8项保护性stdlib测试实际通过，均未导入PySCF或运行量子计算；它们只验证溯源、不可覆盖和执行门禁，不是数值Opt/Freq验证。

其他分子：Calycosin33已有上传SDF与中性图核验，旧V6b预备入口不可直接冒充V8授权执行。皂苷A178 / B157的已有图和公式级AO估计1646 / 1442仅供资源审查，完整DFT仍NOT_RUN；A的全指定与未完全指定立体记录差异需保留，不能靠同名替换。

最终执行入口追加了更严格的后续授权截止时间、已真实闭合Spinosin主队列state的path/SHA绑定，以及覆盖整个计算的原materialized队列锁。最新12项stdlib保护性检查实际通过，见 `prelaunch_worker_validation.json`。前面的8项回执和prepared_manifest的旧source哈希保留其生成当时真实版本，不重写历史。请求8候选后ETKDG RMS去重实际只保留1个构象；不能声称已有8个独立构象或DFT多构象对照。
