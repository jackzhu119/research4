# V7完整Hessian与振动模式的真实控制结果

**DONE：水／乙醇完整计算、全部有符号模式、独立NumPy重放与原生stock结果对照均通过。** 这是分块恢复实现的数值验证，尚不是Spinosin结果。

| 真实对照 | 水 | 乙醇 |
| --- | ---: | ---: |
| 完整振动模式数 | 3 | 21 |
| 正频模式数 | 3 | 21 |
| 完整原始Hessian最大绝对差，Eh/Bohr² | 5.0692783×10⁻¹³ | 1.1253221×10⁻¹² |
| 全部有符号频率最大绝对差，cm⁻¹ | 4.0927262×10⁻¹¹ | 5.6502358×10⁻¹⁰ |

原生参考来自同一可信完整MF快照的真实V5 stock计算，不是虚构参考或仅比较选出的模式。独立检查包括全矩阵、去除平动／转动后的特征方程、质量加权正交性、完整模式数及模式子空间；全部正负模式均保留。投影特征方程残差不是CPHF残差，后者未独立测量，不能从正常返回中编造。

新流程使用原始全网格与原生XC数学表达式，把RHF贡献和完整XC网格块分别健康闭合；H1保留所有原子，B8只划分核扰动右端项，每块仍求完整AO／MO响应。最后向原生`hess_elec`提供完整H1、MO1和MOE1，保留全部原子交叉项，核Hessian只加一次。没有重复保存巨大H1，不改变基组、网格值、泛函或虚轨道空间。

控制结果及图表：`reports/pyscf-spinosin-grid-checkpoint-20261105-v7/`：

- `control-water-independent-analysis-002/`、`control-ethanol-independent-analysis-002/`：独立复算、检查、PNG/PDF及模式文件。
- `actual-full-control-summary-001/native_full_matrix_and_modes_comparison.csv`。
- `actual-full-control-summary-001/actual_small_molecule_frequencies.csv`：真实24个小分子模式，cm⁻¹及THz；不能充当Spinosin低频证据。
- `postprocessing-execution/`：两次成功独立复算的真实wait4收据及stdout。
- `resource-protocol/actual-water-ethanol-numerical-control-release-001.json`：真实控制通过后的source绑定放行文件。

独立分析器首版实际运行曾因`mo_occ`／`mo_coeff`与冻结快照真实键`MO_occ`／`MO_coeff`不一致而失败。保留旧源码`9eaf4a08…`、失败记录及CPU未知字段；未把0.413秒墙钟当成CPU。新分析器`76e2edde…`只修这两个键，21项接口回归测试及独立审查后，实际重放成功。两次成功重放wait4 CPU分别为1.503718、1.530404秒，没有重新运行DFT。

31个控制量子阶段合计105.007565 CPU秒，包含入口问题的失败成本；原累计预算和期限保持不变。控制通过时剩余61,488.854760 CPU秒。没有大分子提速结论。

IR强度始终为`NOT_COMPUTED`。图中的频率标记不代表吸收强度。后续Spinosin使用同一PBE／def2-SVP／grid1、75原子／762AO、完整MO、两线程，无DF及无D3的明确初步方案；只有真正完成219个模式并独立复核后才讨论其低频振动。

已准备持久队列和健康149单元的新运行副本；平台重启后由当前agent检查并调用恢复命令。启动前仍需完成Git及最新内存／磁盘门禁，不能从本报告推断任务已经启动或完成。
