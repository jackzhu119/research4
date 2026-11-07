# PySCF只读分析预检

新增脚本不会运行SCF、梯度或Hessian。9项合成测试实际通过，包含完整合成ledger→图表输出、运行中阶段拒绝、预算总数不符拒绝、缺失/虚频/零频/非有限模式拒绝，以及水不能冒充斯皮诺素。

真实分析尚未执行。主代理结束全部计算后，提供绑定最终cpu_budget.json摘要的放行文件：allow_analysis_export=true、created_at_utc、cpu_budget_sha256。然后运行：

```bash
.runtime/bin/python scripts/analyze_pyscf_spinosin_pilot_20261031.py --run-root reports/pyscf-spinosin-pilot-20261031 --output reports/pyscf-spinosin-pilot-20261031/analysis --release <放行JSON>
```

输出包括真实水频率、各阶段资源/状态、实际优化梯度、当前计算几何图，以及从历史source-peaks读取的实际实验候选表。未得到通过全部检查的斯皮诺素模式时，理论列保持空值并标BLOCKED；频率位置标记等高不表示IR吸收强度。计算得到但未通过检查的频率单独保留，不用于峰位对照。全部后处理只读，不修改分类器或历史光谱。
