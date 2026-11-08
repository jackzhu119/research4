# Calycosin33：后续可选Opt/Freq预备入口（NOT_RUN）

本目录仅完成源码、JSON来源核查、禁数值导入的静态测试和NOT_RUN预检。**没有读取或解析SDF/XYZ/NPY坐标，没有运行SCF、优化、梯度、Hessian或频率，也没有分配预算。** 当前Spinosin主任务继续原样运行。

## 身份与原始构象

候选为中性Calycosin（毛蕊异黄酮），PubChem CID5280448、ChEBI17793、完整Key `ZZAJQOPSWWVMBI-UHFFFAOYSA-N`，C16H12O5，33个显式原子、148电子、74占据轨道、单重态。来源是用户第二份SDF；不把该单分子等同于整株黄芪或任何酸枣仁样品。

- 原始SDF：`data/incoming/molecular_structures_20261103_v1/黄芪CID_5280448.sdf`；SHA `1bf68cb55ed29ac48017aa13b967b12922581290e89667efbd69bd42d9ac480f`。
- 原输入登记：`reports/pyscf-sdf-continuation-20261103-v1/state-worker-preflight/inputs/cid5280448_optimize_input.json`；SHA `842ebe92dad9e7d6d5a56b4c27c515fb10580966e0dd9495bc932fd3bfa76f49`。
- 原XYZ：`reports/pyscf-sdf-continuation-20261103-v1/structure-audit/derived/CID_5280448.xyz`；SHA `8fda598d70ff9eb70085022d42fb4922283e5d54e5d2fabc37e72d2f060f7236`。
- 原精确坐标：同目录 `CID_5280448_coordinates_angstrom.npy`；SHA `793776316ecf1a6f34b4f80fbd2467a2864d1be7eb5769c92bd96c05ef47b5fa`。
- 后续独立名称/图证据：`reports/external-evidence-20261104-v1/chemistry/uploaded_identity_overlay.json`；SHA `e00dd59f2aa058bd4a65e326ebc1bb041c3576c9650ff57c661385a12133acc1`。其原始ChEBI来源、全Key、规范SMILES与上传图一致。

原输入中的 `name_CID_mapping_independently_verified=false` 保留为历史记录；新入口明确引用后来追加的图/名称证据，没有覆盖旧审计。没有独立重新下载PubChem原3D字节，也没有晶体结构确认。仅对**一个用户提供构象**进行局部优化，不开展或宣称全局构象搜索。

## 新源码与既有实现的边界

- 计算入口：`scripts/pyscf_calycosin_optfreq_pilot_20261105.py`；冻结SHA `02491114cf2d7c8a7bd027a85db29c53964c5205297816e25708a621570e9963`。
- 独立数值复算：`scripts/analyze_calycosin_pilot_modes_20261105.py`；冻结SHA `4102925012c356bcd93a9b5bdf02666d525c75d8e5701a688662870b6cbd63a3`。
- 使用既有b985 V6b监督器；不注册新ledger，不增加或重置成本/截止时间。
- 仅复用冻结旧state的输入验证、SCF工厂、不可覆盖NPY写入，以及冻结V8的直接积分、梯度、内存/原生线程保护和正常模式检查函数；不调用旧V5的execute、监督资格、snapshot发布或恢复入口，不伪造旧source/proof。
- 33原子的完整Hessian使用未替换的stock `mf.Hessian().kernel(atmlst=list(range(33)))`；不复用75原子incremental缓存，不把Calycosin标成water/ethanol或Spinosin。
- 后处理新建Calycosin专用source/闭合/33→93检查；只复用旧a2已核验的泛型NumPy TR投影/模式审计数学函数。**旧Spinosin analyzer整入口不接受该新结果，也没有被宣称支持33原子。**

## 必须先通过的实际门禁

1. 新入口首先验证b985真实运行父进程、proof、线程、累计账本及资源限制。
2. 新root决策必须明确绑定此源码、输入、输出目录和有限请求，且来自**同一现有累计V6b ledger**。
3. Spinosin主任务必须已真实完成健康闭合C阶段，全部219个signed模式完成；f137 source/b985 parent/旧精确snapshot均吻合。
4. 当前冻结v2独立分析必须对该确切C结果完成219模式NumPy replay。计算完成与是否全正频分开；主结果有虚频不会被篡改或变成“未计算”。
5. root决定的实际时间必须晚于主C父进程闭合及独立分析。主任务仍RUNNING或PARTIAL时，Calycosin不会得到执行机会。
6. 以上均在NumPy/PySCF导入、SDF/XYZ/NPY数据读取之前完成；现有quantum-upgrade-site优先，不安装或升级软件。

父监督器实际CPU grant必须等于root明确记录的请求；如果现有剩余额度不足而被clamp，入口在数值导入前阻断。单次pilot仅允许125–7200实际CPU秒、61–7200墙钟秒、1或2线程、最多35个优化step。请求值是上限，不是完成时长承诺。没有自动重试、自动扩额或抢占主任务的安排。

## 实际计算流程（尚未运行）

PBE / spherical def2-SVP / grid1 / non-DF / explicitly noD3，SCF容差1e-8、最大80cycle、CPHF容差1e-8。几何优化上限35step；采用现有geomeTRIC阈值及直接RKS梯度。每step保留能量、SCF状态、坐标和梯度。

优化成功后，在最终几何新建独立 `final_scf.chk`（非复用优化梯度或直接沿用末步轨道），完成新的同方法SCF和梯度。原子向量梯度RMS≤3e-4、max≤4.5e-4 Eh/Bohr才允许进入Hessian。保留优化及最终SCF checkpoint、全精度几何、最终MOs、梯度、环境版本和日志。

完整stock Hessian在已核验的SCF8000/Hessian3000 MB作用域内执行，正常返回后保存原始矩阵；仅在raw不对称≤1e-5 Eh/Bohr²时做明确的对称化。排除平移/转动，输出全部93个signed频率、THz单位和振型。负/零模式保留，计算完成不等于已证实正定极小值。IR强度始终 `NOT_COMPUTED`。

超时、SCF/优化/梯度/Hessian失败均保存PARTIAL、真实日志和已有checkpoint；没有93模式就不声称完成频率。没有自动跨进程Hessian恢复协议，也不会释放失败父进程缓存。此粗网格无色散PBE阶段仅作可行性pilot；不能据此确认固态药材峰归属。

## root未来操作模板：现在不执行

本目录的 `AUTHORIZATION_TEMPLATE_DISABLED.json` 明确为disable模板，不是实际授权。待主任务完成后，root记录真正完成结果、检查文件SHA、剩余预算、选定线程/CPU/wall及输出新目录；无需另设预算池。不能把模板的NOT_RUN占位值当真实记录。

```bash
PYTHONPATH=/workspace/research4/.runtime/quantum-upgrade-site:/workspace/research4/.runtime/quantum-pilot-site \
.runtime/bin/python scripts/run_pyscf_continuation_budget_20261105_v6b.py run \
  --ledger reports/pyscf-spinosin-staged-20261105-v6b/cpu_budget.json \
  --phase v6b-calycosin-optfreq-pilot-001 \
  --cpu-seconds 7200 --wall-seconds 7200 --threads 2 --memory-gib 24 \
  --log reports/pyscf-calycosin-optfreq-pilot-20261105-v1/execution/pilot-001.log \
  -- .runtime/bin/python scripts/pyscf_calycosin_optfreq_pilot_20261105.py \
  --execute --threads 2 --maxsteps 35 \
  --authorization ACTUAL_POST_SPINOSIN_ROOT_DECISION_JSON \
  --authorization-sha256 ACTUAL_DECISION_SHA256 \
  --output reports/pyscf-calycosin-optfreq-pilot-20261105-v1/calculation-001
```

root可选择更少CPU/wall/step；CLI和实际决策记录必须一致。若不够做安全尝试，就维持NOT_RUN。只有pilot实际FREQUENCIES_COMPUTED且父监督器EXITED/0无资源/monitor故障后，才运行独立后处理：

```bash
PYTHONPATH=/workspace/research4/.runtime/quantum-upgrade-site:/workspace/research4/.runtime/quantum-pilot-site \
.runtime/bin/python scripts/analyze_calycosin_pilot_modes_20261105.py \
  --execute \
  --result reports/pyscf-calycosin-optfreq-pilot-20261105-v1/calculation-001/result.json \
  --result-sha256 ACTUAL_CLOSED_CALYCOSIN_RESULT_SHA256 \
  --output reports/pyscf-calycosin-optfreq-pilot-20261105-v1/independent-analysis-001
```

后处理先占同一全局Q锁并确认账本无RUNNING，验证真正Calycosin父行/source/log/结果、原输入和root主任务闭合许可后才读数组。限制1CPU、1GiB AS、120秒；以独立NumPy QR平移/转动投影、质量加权Hessian本征值重算93个signed频率，并检查振型正交、投影和本征方程残差。输出所有93频率CSV、0.1–1.5THz子表、检查JSON及哈希索引。它不重新计算电子结构，不生成IR强度曲线，不把频近当化学确认。

## 当前状态

- 小JSON来源/身份核查：DONE。
- worker与replay预检：NOT_RUN（仅源码及静态预备已运行）。
- 纯stdlib测试：27项通过；未导入NumPy/PySCF、未读数组。
- 新实际DFT、93频率、化学峰归属或分类准确率改善：NOT_RUN。
- 实际root授权、预算分配和新科学结果：NOT_RUN。

任何后续实际运行均由root在Spinosin优先事项完成、磁盘/内存/CPU/截止时间允许后决定。当前酸枣仁主线与96.0234%既有模型不变。
