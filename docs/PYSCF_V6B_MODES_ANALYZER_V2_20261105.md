# V6b 后处理器 v2：真实 V5 fresh-G 回执接口修复

原独立分析器 `scripts/analyze_pyscf_staged_modes_20261105_v6b.py`（SHA `8e96eebbb8b02f1e68a045f3b3615ff462de86c053358b5e11835789a64480a2`）在真实 water 分析的 source 元数据门禁失败：`retain_v5_closure()` 假定所有 V5 closure 都含 `log_sha256`，触发 KeyError。失败发生在数值库导入与目标数组读取之前；原源文件、失败日志及全部真实量子结果保持不变。

原失败证据：`reports/pyscf-spinosin-staged-20261105-v6b/execution/control-water-independent-analysis-001.root-console.log`。这属于分析软件接口错误，不是重新计算 Hessian 的理由，也不能把失败当成已完成独立数值复核。

新脚本为 `scripts/analyze_pyscf_staged_modes_20261105_v6b_v2.py`。只有 `retain_v5_closure()` 作如下逻辑修复：

- 原 state producer closure 有 `log_sha256`；原 fresh-G `verify_closed_execution()` closure 实际只有 `proof_sha256`、`producer_result_sha256`、`phase`、`actual_cpu_seconds`、`closed_row_sha256`。
- 从已与该 `closed_row_sha256` 完全一致的实际账本行读取 `row.log_sha256`，检查64位小写十六进制，再对实际 `row.log` 执行哈希核验。
- 如果 closure 自身另有 log SHA，仍要求与账本行相等。没有生成或补造旧 closure 字段，没有放松 source/proof/ledger/result、实际命令、fresh-G 或正常模式数学门禁。

`tests/test_analyze_pyscf_staged_modes_v6b_v2.py` 增加真实已闭合 water／ethanol fresh-G 元数据回归：通过原冻结 V5 函数重新资格检查，确认真实五字段结构，再验证新的 log 保留步骤。另保留缺失字段、冲突 log SHA、变更账本行及全部原拒绝测试。所有测试仅标准库与归档元数据，不读取 NPY／CHK／Hessian、不导入 NumPy／Torch／PySCF。

新 preflight 目录为 `reports/pyscf-v6b-modes-analysis-preflight-20261105-v2/`，明确 `NOT_RUN`。17项软件测试通过，只证明元数据接口已修复；V6b 对照的完整矩阵／模式独立复核仍待 NN 和审计结束后由 root 执行。当前不重新运行分析、量子计算或训练。

之后的执行命令仍依照前一版文档 `docs/PYSCF_V6B_MODES_ANALYZER_20261105.md`，仅替换脚本为 `_v6b_v2.py` 并使用全新输出目录。继续遵守全局 Q 锁、显式真实 result SHA、固定 resource limits、全部有符号模式保留及 IR `NOT_COMPUTED`。
