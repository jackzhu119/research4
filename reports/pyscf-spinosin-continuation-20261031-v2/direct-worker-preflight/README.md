# V4：标准屏蔽直接RKS控制后端的准备

源码：`scripts/pyscf_spinosin_direct_worker_20261031.py`；18项纯防护测试通过。此目录没有实际分子计算结果，实际water Opt/Freq、Spinosin SCF／梯度／优化／Hessian均由root另行按账本执行。测试只构造Mole／梯度scanner／Hessian惰性对象并检查分派，不求积分或波函数。

V4保留PBE／def2-SVP／grid1／无D3、中性单重态及原SCF／几何收敛阈值，但**明确去除density fitting近似**。设置为`density_fitting=False`、`density_fit_backend=SCREENED_DIRECT_RKS_V4`，`auxbasis`及`hessian_auxbasis_response`为null。它不是将DF优化和直接Hessian静默混成同方法；频率入口要求真实V4优化收敛、重新SCF／梯度阈值通过和完整精度几何SHA匹配。

SCF子类的get_jk完整转发标准`pyscf.scf.hf.SCF.get_jk`，绕过RHF根据分子较小自动缓存全四中心ERI的分支。`mol.incore_anyway=False`、`mf.direct_scf=True`、`direct_scf_tol=1e-13`、`mf._eri=None`；若出现with_df或全ERI缓存立即拒绝。梯度调用标准`pyscf.grad.rks.Gradients.kernel`，只外加状态及计时，不调用DF梯度或其积分缓存。Hessian要求标准`pyscf.hessian.rks`类。

独立方法代理只读审查确认：标准RKS Hessian的电子部分使用RHf Hessian屏蔽direct_bindm导数收缩，没有DF实现rho／wj的强制数十GiB临时矩阵。**这不是资源足够或运行成功的证明**：H1／CPHF／网格及其他临时数组仍有GiB级成本，实际CPU、内存和磁盘必须由后续运行测量并受预算限制。不能从静态路径审查声称已经获得频率。

所有四阶段接口和每步能量／梯度／检查点、完整精度NPY及哈希均沿用V3。Hessian原矩阵与通过非对称阈值后使用的对称矩阵分别保留；完整219模式、非零实振型及全部正频率才可判驻点minimum。零／负／复数／NaN及缺失模式均被防护拒绝，IR强度始终NOT_COMPUTED。

actual water验证必须先于Spinosin direct计算，且与DF水之间允许真实density-fitting近似差异，不能要求跨方法数据完全一致或覆盖原DF结果。V3及所有历史报告保持原样。

`validation.json`保存本次测试耗时及V3模板、V4源码和测试SHA；脚本在测试和独立方法复核后冻结，没有自行启动worker或提交Git。
