# 已保存 CPHF 响应的可选方程残差复核

状态：**PREPARATION / NOT_RUN**。本文给出静态推导和可选执行入口。没有读取目标数值数组、调用量子算子或产生实际残差。本项目原响应结果的 `residual_norm=NOT_COMPUTED` 保持不变。10组响应、完整Hessian及219个有符号模式的必需独立复核继续优先。

## 精确约定与公式

针对当前实数、封闭壳层 RKS、`level_shift=0`；保持同 PBE/def2-SVP/grid1、完整AO/MO、原积分筛选及完整响应算子。设完整MO系数为 `C[AO,MO]`，AO重叠为 `S`，占据子集为 `Co=C[:,occ>0]`。保存的 `MO1_atom_NNN[xyz,AO,occ]` 是 **AO系数响应** `W=C U`，并不是包含AO基函数导数的整条轨道导数。

恢复MO系数必须使用重叠度量：

```text
U = C.T @ S @ W                         [1/Bohr]
C.T @ S @ C = I                        [dimensionless]
```

不能使用 `C.T @ W`。同时记录 `C U-W` 的重建差和 `C.T S C-I`，但不临时发明通过阈值。

对选定原子A，逐字沿用 `pyscf.hessian.rhf.solve_mo1` 的重叠导数构造：先计算 `dS=-mol.intor('int1e_ipovlp',comp=3)`，按该原子的AO切片 `[p0:p1]` 在矩阵行加入 `dS[:,p0:p1]`、在列加入其转置，得到 `S1_A`。定义：

```text
H = C.T @ H1AO_A @ Co                   [Eh/Bohr]
Q = C.T @ S1_A @ Co                     [1/Bohr]
Delta_ai = eps_a-eps_i                  [Eh]
D1(U) = 2 C U Co.T + (2 C U Co.T).T
V(U) = C.T @ native_RKS_response(D1(U)) @ Co   [Eh/Bohr]
```

`V(U)` 必须由冻结 `rhf.gen_vind(mf,C,occ)` **重新计算**。选择一个原子的3个核位移RHS时，分子、AO、全部占据/虚轨道、响应核及原子间耦合均保留；这只是少量方程的抽查，不能声明检查了全75原子的响应。

可复核量为：

```text
Rvirtual_ai = Delta_ai U_ai + H_ai - eps_i Q_ai + V_ai(U)    [Eh/Bohr]
Rpreconditioned_ai = Rvirtual_ai / Delta_ai                 [1/Bohr]
Roccupied_ji = U_ji + Q_ji/2                               [1/Bohr]
Rnormalization_ji = U_ji + U_ij + Q_ji                      [1/Bohr]
E1fresh_ji = (H-Q*eps_i+V(U))_ji + U_ji*(eps_j-eps_i)       [Eh/Bohr]
```

分别报告每个原子/轴的最大绝对值、Frobenius范数和RMS。另记录无量纲的**项尺度归一化残差**：

```text
||Rvirtual||F / (||Delta*Uvirtual||F + ||Hvirtual-Qvirtual*eps_i||F + ||Vvirtual(U)||F)
```

分母为零时返回 `null` 并标记未定义，不用任意epsilon伪造数值。它不是利用完整算子范数证明的误差界，不能叫做严格的全矩阵后向误差。

## 为什么正常返回和原生日志 r 不能替代真实残差

`lib.krylov` 求解 `(I+T)X=b`，其日志 `r` 是**新正交化试探方向的范数**。停止分支为 `max_innerprod<lindep OR max_innerprod<tol**2`，而不是重新计算最终 `(I+T)X-b`。

`rhf.solve_mo1` 将每个实际内部原子批次的请求tol设为 `mf.conv_tol_cpscf*该批原子数`。本项目B8请求为 `8e-8`，末3原子为 `3e-8`；已知762AO/160occ及2000MB下限使原生最小内部blksize为113，因此B8不会再次内部分裂。源码默认 `lindep=1e-13`，其平方根约 `3.16e-7`，可能比请求tol主导试探方向截断；运行时仍须读取实际配置。这个数既不是最终残差的上界，也不是最终残差的下界。

原生 `cphf.solve_withs1` 在Krylov返回 `X` 后，还重新计算 `V(X)` 并进行一次固定点修正得到保存的 `Y=U_saved`；保存的 `MO_E1` 仍用修正前的 `V(X)`。如果复用 `V(X)` 检查 `Y`，虚空间方程会按构造恒为零，形成假通过。真正的复核必须新算 `V(Y)`：

```text
Rvirtual(Y) = Vvirtual(Y)-Vvirtual(X)
E1fresh(Y)-E1saved = Voccupied(Y)-Voccupied(X)
```

因此 `MO_E1` 一致性差不应被误解为独立重跑了CPHF，也不能从保存结果倒推一个V并人为令残差为零。

## 最小真实验证顺序

1. 主Q全部真实闭合，主队列锁释放；root记录实际终态JSON及SHA。若主流程仍运行、处于未知状态或独立分析未结束，入口拒绝执行。
2. root使用相同V8 guard监督可选审计；先检查已完成的水响应全3原子，再检查乙醇B8及尾1原子响应。旧A/H1、新健康响应、快照/G、物理binding和原生模块哈希均重新资格核验。
3. 小控制先验证 `U=C.T S W` 重建和归一约束；使用fresh完整算子测量上述方程残差及 `MO_E1` 差。未设定新容差前仅报告真实数值，不把请求tol当作方程残差门槛。
4. 若控制无异常且剩余实际预算充足，可先检查一个已完成Spinosin响应块中的一个原子（3RHS），再决定是否检查该B8全部24RHS。报告必须列出检查范围，不能把3RHS结果写成全225RHS通过。
5. 结果写新的sidecar目录，状态为“残差已测量，尚无新收敛通过门槛”；旧响应、模式和分类器不修改。

该检查是“相同已审核响应算子下的方程残差复算”，不是独立量子化学引擎验证，不是新的全Spinosin DFT/Hessian计算。

## 成本依据与有界命令

首个真实健康Spinosin B8的事件日志中，12次完整native-vind调用均返回：24RHS单次约945–1450 CPU秒；一次3RHS调用为200.31 CPU秒；响应算子构建约35.61 CPU秒。该3RHS属于Krylov投影后的试探向量，并非某个原子的三个核位移；12次调用也不等于12个Krylov迭代。这些是旧求解器中的实测调用耗时，不是本残差审计已执行的耗时，也不是新调用的安全上界。候选起始请求可为一个原子900 CPU秒，整个B8约2400 CPU秒；root必须依据当时真实剩余预算决定，不能自动扩额。小分子控制成本须实际记录，不从小分子外推全Spinosin速度。

新默认安全入口：`scripts/audit_pyscf_saved_cphf_residual_20261105.py`。不带 `--execute` 只产生NOT_RUN准备元数据，不接受任何响应输入。实际示意命令如下；所有 `ACTUAL` 必须由真实闭合产物物化，不能填未来哈希：

```text
.runtime/bin/python scripts/run_pyscf_continuation_budget_20261105_v8.py run
  --ledger ACTUAL_V8_LEDGER --phase FRESH_AUDIT_PHASE
  --cpu-seconds 900 --wall-seconds 1800 --threads 2 --memory-gib 24
  --log FRESH_LOG --
  .runtime/bin/python scripts/audit_pyscf_saved_cphf_residual_20261105.py
  --execute --response-result ACTUAL_RESPONSE/result.json
  --response-result-sha256 ACTUAL_RESPONSE_SHA
  --main-pipeline-state ACTUAL_MAIN/state.json
  --main-pipeline-state-sha256 ACTUAL_CLOSED_MAIN_STATE_SHA
  --atom-index ACTUAL_ATOM_ID --threads 2 --output FRESH_REPORT
```

错误混种分类成绩已按2026-11-12用户决定撤回；论文使用main的paper/current白名单。
