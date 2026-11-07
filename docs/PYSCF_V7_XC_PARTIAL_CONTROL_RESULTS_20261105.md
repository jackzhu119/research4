# V7网格分块的真实小分子部分Hessian对照

状态：**DONE：水／乙醇的XC与完整RKS电子partial差分通过；V7全局Hessian和频率尚未执行。** 本报告不是Spinosin振动结果，也不是Spinosin放行文件。

实际使用同一完整MF、原始网格、掩码和原生RKS XC函数。控制组显式固定224点原网格块；水10128点，每类46块；乙醇30376点，每类136块。两类XC各分3个独立、健康闭合的阶段，覆盖每个原网格块恰好一次。原生函数返回后立即按原表达式收缩为小Hessian贡献；RHF贡献仅加一次，没有核项或CPHF项。

| 对照量：最大绝对差 | 水 | 乙醇 |
| --- | ---: | ---: |
| 完整XC diagonal AO数组 | 7.1054×10⁻¹⁴ | 7.1054×10⁻¹⁴ |
| 完整XC deriv2 AO数组 | 2.8422×10⁻¹⁴ | 2.8422×10⁻¹⁴ |
| 完整RKS电子partial | 3.2951×10⁻¹³ | 1.1249×10⁻¹² |
| RHF电子partial | 1.7764×10⁻¹⁴ | 1.2434×10⁻¹⁴ |

比较固定绝对容差为10⁻⁸，保留实际差值。RKS/RHF partial单位为Eh/Bohr²；AO行按原生数组比较。该结果验证分块计算和收缩的数值一致性，不证明大分子提速，也不替代全局响应、振动模式或最小值检查。

实际产物位于 `reports/pyscf-spinosin-grid-checkpoint-20261105-v7/`：

- `control-water-partial-compare-001/partial_differential.json`
- `control-ethanol-partial-compare-001/partial_differential.json`
- `actual-small-partial-control-summary-001/`
- `resource-protocol/control-A-actual-completion-001.json`

源代码：core `da1e2b0267c48e748a2bcb7d3f2ee3110d8e2288aa340d23dab7a12c994ddf8e`；worker `00e710eb0d637dbd073939bea5d4496993cb68f29efad1439322df3972415f02`。27项标准库测试和独立源码／数学审查通过，随后才执行本报告数值实验。

共24个实际量子阶段，23个正常闭合、1个失败，合计75.493598 CPU秒，所有成本已计入原累计账本。保留两项入口记录：首个水计划省略显式内存参数，虽计算成功但不用于依赖；首个assembly索引加入可选`bytes`导致严格依赖字典比较拒绝，失败成本0.855151秒保留。新计划补齐实际参数，新索引仅使用`path`／`sha256`，在新阶段完成组装；未修改冻结科研代码、旧输入、旧结果或标签。

下一步必须完成真实V7全75索引H1流程的小分子版本、B8完整MO响应、全局Hessian、全部3／21个有符号模式及独立重算，之后才能启动Spinosin。IR强度仍为`NOT_COMPUTED`。
