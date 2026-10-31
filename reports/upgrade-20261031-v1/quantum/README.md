# 量子化学升级：真实技术测试与候选阻塞

先读[可行性报告](../../../docs/QUANTUM_CHEMISTRY_FEASIBILITY.md)和[三候选登记](../../../data/research/quantum_upgrade_20261031/candidate_registry.json)。

**本目录只有水分子软件环境测试，未计算Spinosin或Jujuboside A／B。Gaussian引擎计算0，候选优化／频率／IR强度0，新增真实药材样品0。**

- `environment_audit.json`及`gaussview-status.log`：已有GaussView私有载荷，未发现可核验g16／g09引擎或新授权安装介质；不把GUI当DFT。
- `downloaded_packages.json`、下载／安装日志：PySCF2.14.0、h5py3.16.0独立装到`.runtime/quantum-upgrade-site`。轮子留云端缓存，Git只保存版本、SHA、许可证与依赖记录。
- `water-smoke/`：首次真实PBE／STO-3G单点、梯度、解析Hessian。使用仓库手工水坐标，不是优化结果。
- `water-smoke-v2/`：频率文本解析器增加失败关闭检查后进行的同一技术复测，SCF能量仍为−75.22550440958054Hartree，差0。它不是第二份独立样品。
- `command_invocations.jsonl`：实际命令起止UTC及返回码；`source-snapshots/`绑定两次真实运行的源码SHA。
- `unit-tests-v2.log`：7个真实防护测试通过。解析频率是测试夹具，非计算产物。
- `execution_summary.json`：明确COMPLETED技术范围与NOT_RUN／BLOCKED的药材机制范围。

最新水技术计算耗时约0.659秒、梯度范数0.164746Hartree/Bohr，未做几何优化，所以没有从该Hessian声称得到有效振动频率或IR谱。数组保存在`water_derivatives.npz`，SCF记录及checkpoint同时保留；具体参数在各`result.json`中。

候选3D坐标／构象未核实，Jujuboside A同名数据库立体状态需解歧；没有纯品THz、当前样品HPLC与固态结构。模板只是输入准备，不能进入分类先验。孤立分子与固体THz机制分开论证，没有根据测试类别选理论峰。

有限格式／溯源单测不是绝对零泄漏证明，水环境测试也不是机制验证。全部历史模型、论文数据与旧Gaussian样例未改。
