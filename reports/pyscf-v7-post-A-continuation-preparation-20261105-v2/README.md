# 有限后续阶段契约（仅准备）

`CONTINUATION_STAGE_CONTRACT.json` 给出冻结科学代码的实际 argv 模板与依赖字段；所有 CLI flag 已用真实源码 argparse 的 AST 核对。没有创建 enabled grant、计算未来 SHA 或执行科学阶段。generic_state_worker 的持久有限 driver 可直接按此契约物化，不需修改 A/core/6e9 consumer。

完整 XC 后共有 **13 个量子阶段**：A组装1项 → H1 1项 → 原B8响应10项 → global组装1项；随后是一次独立数值审计。原 materialized queue v2 不支持未生成结果的未来 SHA，必须在每个上游真正健康闭合后再生成后续输入。10个响应共享同一个已知A/B，可在B健康后一次物化；global的10项响应index要等全部响应真实闭合。

- A：`pyscf_xc_chunk_worker_20261105_v7.py --stage rks-partial --component-index ACTUAL_INDEX --component-index-sha256 ACTUAL_SHA --xc-max-memory-mb 2000`。index严格四键：`schema/RHF_partial/XC_plan/XC_chunks`，schema为`research4-v7-XC-component-index-v1`；31个XC必须真实健康且覆盖20diag＋11deriv2，不从目录glob混入失败尝试。
- H1：`pyscf_staged_consumer_20261105_v7.py --stage h1ao --partial-result ACTUAL_A --partial-result-sha256 ACTUAL_A_SHA`。只有完整正常返回且guard健康闭合的75项H1才可用于下一步。
- 响应：同consumer，`--stage response-block --block-index 0..9 --partial-result ACTUAL_A --partial-result-sha256 ACTUAL_A_SHA --h1-result ACTUAL_B --h1-result-sha256 ACTUAL_B_SHA`。保留全AO/MO响应与原子全局索引，不是独立子体系。
- global：同consumer，`--stage global-assemble` 加A/B及`--response-index ACTUAL_INDEX --response-index-sha256 ACTUAL_INDEX_SHA`。index为`research4-v7-closed-response-results-index-v1`，`results`是按0–9排列的真实两键引用。stock全交叉收缩、核项一次；219个有符号模式与全正极小点分开判定。

每项公共参数均在JSON完整列出：原snapshot3089、freshG6b14、实际release82d、`--execute --threads 2 --output FRESH`；目标不得带`--control-validation`。所有依赖ref仅`path＋sha256`，不得混入artifact的bytes。

约26k CPU是root当时估计的原池余量，不是新分配，也不是完整链条一定够用的证明。每阶段仍由原guard读取实际累计账本并收紧至原cap216000与23:20:50Z截止；不能把各job请求上限相加当资源保证。H1在6e9中是单体完整phase，不能从“已有部分H1文件”假称可恢复。H1／大分子CPHF实际成本未测，不能由水／乙醇时间推断；可先完成原固定响应block0观察真实资源，再决定是否物化余9块。不足时保留真实部分结果并停止，不自动提高cap或换科学参数。

已知新增payload约1.312GB：H1约1.045GB、全部MO1/MO_E1约265.5MB、global主要小数组约1.21MB，另留headers／日志／图表等。6e9每个响应都会读取全H1；global加载H1与全部响应约1.311GB匿名数组，同时page cache可另计源文件字节，native AO/FXC/Krylov scratch峰值未知。global不再保存一份重复H1 SAVE_ONLY。实际新磁盘观察在JSON内，只是时点值；A闭合、H1闭合、每响应闭合及global前均应由root检查资源，不改冻结科学代码。

真实global健康且所有Q闭合后运行冻结`analyze_pyscf_staged_modes_20261105_v7_v2.py`；其PNG/PDF/CSV/XYZ已覆盖基本导出，辅助最近位置表不代替独立审计。
