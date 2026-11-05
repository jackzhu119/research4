# V8 原生响应／全 Hessian 消费者

状态：源码已冻结，21项标准库测试通过，新 guardian SHA 已固定。**本准备没有启动量子计算，也没有产生新频率或 IR 强度。**

新增 `scripts/pyscf_staged_consumer_20261105_v8.py`，保留冻结6e9消费者的原生完整 AO/MO 数学：PBE/def2-SVP/grid1、非DF、显式无D3、B8完整响应、原CPHF容差与50次上限、level_shift=0、核项一次和所有原子间交叉项。`PHYSICAL_SCHEMA` 与 `NUMERICAL_POLICY` 逐字相同；新 result/cache/index 与运行监督使用明确的 V8 身份。

旧 A 与 H1 只能委托原00e/6e9消费者及原b985监督器验证。旧结果、旧缓存、旧授权和旧失败账本不重贴 V8 标签。新响应及全局组装只接受 V8 自身结果／监督证明。生产酸枣仁 C001 复用已完成 H1，不重新计算。

`NativeVindObserver` 仅记录原生诱导势算子调用的开始、结束、形状及CPU／墙钟。它只把原对象交原函数一次，原样返回同一结果对象，异常也原样重抛；不改输入、输出、Krylov求解策略或残差。调用次数不等于Krylov迭代次数，残差仍 `NOT_COMPUTED`。

## 真实控制流程

同目录 `ACTUAL_LEGACY_CONTROL_INPUTS_AND_NEW_NOT_RUN_INTERFACE.json` 给出已经真实完成的水／乙醇 A、H1、snapshot、梯度与原生完整参考的路径及哈希；所有未来 V8 结果均为 null/NOT_RUN。

root 使用新 V8 guard 启动 water response block0、ethanol block0与block1，再分别全局组装，所有阶段加 `--control-validation`。完成后使用新独立 V8 分析器做完整矩阵、全部有符号频率和模式子空间的实际复核。

目标 Spinosin 运行前必须有实际生成、通过的控制 release：

```json
{
  "schema": "research4-v8-actual-small-control-release-v1",
  "root_execution_authorized": true,
  "v8_actual_water_ethanol_controls_reviewed": true,
  "no_new_budget_beyond_registered_v8": true,
  "legacy_v7_control_release": {"path": "ACTUAL", "sha256": "ACTUAL"},
  "code_refs": ["ACTUAL V8 consumer, guardian, independent analyzer artifact refs"],
  "controls": [
    {"case": "water_software_control", "global_analysis_index": {"path": "ACTUAL", "sha256": "ACTUAL"}},
    {"case": "ethanol_software_control", "global_analysis_index": {"path": "ACTUAL", "sha256": "ACTUAL"}}
  ]
}
```

以上是字段说明，不是有效输入或执行证据。每个 analysis index 必须引用实际 V8 全局结果，独立 checks 状态为 `DONE_CONTROL_NUMERICAL_REPLAY`、完整3／21个正模式、IR未计算、原生全矩阵和所有模式比较 `PASS_ACTUAL_V8_VS_AUTHENTICATED_V5_STOCK_FULL_MATRIX_AND_ALL_MODES`。release内旧82d授权须另经原00e gate验证。不得拿准备报告或标准库测试替代实际控制。

新消费者 `--control-release` 指向上述真实 V8 release；它与旧 H1 的旧82d工程授权不同，物理输入仍须完全一致。

## 标准库验证

`tests/test_pyscf_staged_consumer_20261105_v8_stdlib.py`：21项通过，包括旧源完整性、物理合同相同、原生函数和关键数学调用AST相同、B8完整覆盖、旧A/H1严格分派、未知schema拒绝、observer对象身份及异常传播、准备授权不能放行新目标。它们没有执行量子计算。

后续所有新Q使用root新96小时授权与独立累计续算账本；旧210160.830836 CPU秒及失败成本完整保留，不自动增加新预算。此文件不授权代理自行启动Q。

最终消费者 SHA：`3a107671d9c8a5a47c25f631fd09ed70148dc82b529448d9bafc67179801a585`；guardian SHA：`7bc773076ecc5e0f90fb58b81bf0641948dd09c14b2816138a0469f16e511c65`。`preflight-001/preflight.json` 为真实元数据预检，状态 NOT_RUN，不能作为量子计算完成证据。
