# V7 正常模式独立复算入口准备

当前状态：**V2 入口源码已冻结，21 项标准库测试通过；本准备记录不代表实际振动频率复算完成。**

当前入口是 `scripts/analyze_pyscf_staged_modes_20261105_v7_v2.py`。它不修改旧
`f137` V6b worker 或 `38dde` 分析器，也不将 V7 产物改写为旧版来源。
实际父监督器仍是冻结的 V6b `b985`，相应 proof/schema 保留 V6b 身份。
新 V7 worker/schema、物理状态绑定和正常模式产物分别核验。

分析前必须验证：

- 真实全闭合累计账本和现有全局 Q 排他锁。
- 原始 C001 完整 MF 快照、同状态梯度及 75 原子的顺序、电子数与全 AO/MO 空间。
- 独立健康 RHF、XC 计划、全部所需 diag/deriv2 网格块、合成 A 和完整 H1AO B。
- 每个完整体系 B8 右端项块的健康 producer、完整 MO1/MOE1、原子覆盖和实际 argv。
- `global-assemble` 使用完整 H1/MO1/MOE1；电子 partial 调用一次、核项一次，
  最终组装中 `make_h1` 和 `solve_mo1` 调用均为零。
- 各输入 JSON 的实际 SHA、对应闭合命令、所有 source/proof/manifest/payload 哈希。

完整物理态的 `research4-full-rks-physical-binding-v6b` 可以保留，因为数学状态相同；
这不允许将新的 V7 producer 认作旧 V6b producer。旧的失败阶段和中断阶段不能成为
任何健康组件来源。可变累计账本按真实闭合行哈希核验；最终持锁时再记录当前完整
账本哈希，不能要求早期 producer 之后从未正常追加账本。

上述门禁通过后才允许 NumPy 和已闭合数值载荷解码。独立质量、平移/转动投影、
特征值、简并子空间、完整有符号频率与振型复算使用冻结 `38dde` 的公式。
运行时 AST 门禁允许版本标签变化，禁止这些公式和来源门禁发生其他未审修改。
原始 V5 数值 helper 仅在门禁之后使用，不运行 PySCF 引擎。

`stable_MF_state_unchanged` 只表示恢复的 MF 数据未被原位改变，不代表开展了电子轨道稳定性分析。

完整 219 模式与通过极小点检查是不同状态。负频和零频保留；只有梯度合格、219
模式完整且全部为正的 Spinosin 才进行实验候选峰的描述性比较。水和乙醇只验证
实现，不是药材机制证据。IR 强度始终为 `NOT_COMPUTED`；图中的等高线标不代表
吸收强度。频率接近不能确认化学归属，理论值不进入已有分类器。

当前纯标准库测试命令：

```bash
python -B tests/test_analyze_pyscf_staged_modes_v7_v2_stdlib.py
```

准备期 `WORKER_SHA` 未冻结时，实际 `qualify` 必须拒绝执行；不会降级成仅凭布尔
结果放行。准备报告目录 `reports/pyscf-v7-modes-analyzer-preparation-20261105-v1/`
保留早期草稿的 NOT_RUN 快照，不代表后续源码或任何数值工作完成。源码最终冻结、
独立审查及实际水 3 / 乙醇 21 模式全矩阵比较通过后，由 root 签发真实控制发布记录，
再决定是否执行 Spinosin。所有实际计算和后处理命令、资源与输出仍需独立留证。

前一版执行源码为 `9eaf4a08b7073b39aca7ce5b91dd483ba3119ab83d8064b5f98aea2901067b23`，
绑定已冻结且实际使用的 consumer `6e9b835f4b49329cd52162b5862da14b3dcd87c80390c1e09b3cf62e14c07dbe`。
前一版源冻结和纯标准库预检见 `reports/pyscf-v7-modes-analyzer-preparation-20261105-v3/`。
V1/V2 未冻结草稿记录保留，不能作为当前源码或实际数值结论。真实水、乙醇和
Spinosin 结果仅以 root 执行后生成的闭合结果与独立分析报告为准。

2026-11-05 真实水控制后处理中，已执行的旧 `9eaf` 分析器因 `mo_occ` 键写错而停止，
未完成该次独立振型复算。冻结 V5 的真实接口使用 `MO_occ`、`MO_coeff` 和 `MO_energy`，
并原样返回 manifest 键。当前 V2 仅修正前两处下标，其他源码字节不变：
`76e2eddebf681f9adf2ab7c5f9fa0bdc126b0c93f081cc93a606009f31c94705`。
旧源、原审查和失败日志保留；昂贵的 RHF/XC/H1/CPHF/global 组件不需重算。
21 项测试和父代理独立审查均通过；新准备与失败说明位于
`reports/pyscf-v7-modes-analyzer-key-repair-20261105-v2/`。
首次失败的实际 CPU 未取得，保留 UNKNOWN；不能用工具 wall 时间冒充 CPU。
本修复子任务只执行标准库检查和无数值预检，真实水／乙醇复算由 root 在新目录执行。
