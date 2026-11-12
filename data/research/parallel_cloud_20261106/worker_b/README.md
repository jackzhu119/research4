# 可移植B量子工作包输入

本目录只有小型可信结构图、来源证据与原始上传SDF。它不依赖A的私有工作目录、旧ledger、主机boot、.runtime安装目录、Spinosin大检查点或未跟踪缓存。

- `cases.json`固定water、原儿茶酸和Calycosin三项，以及PBE/def2-SVP/grid1/direct/明确noD3方法、门禁和预期模式数。预期数3/45/93不是已计算频率。
- `calycosin_CID5280448_user_supplied.sdf`逐字节复制自用户上传文件，SHA256 `1bf68cb55ed29ac48017aa13b967b12922581290e89667efbd69bd42d9ac480f`。它是用户提供的计算结构，不是实验晶体结构。
- `source_evidence.json`保留ChEBI图出处、CC BY4.0归属与原工程图复核哈希。原始A工程路径只是历史来源文字，B不会读取它们。
- `preparation_graph_validation.json`记录A上实际RDKit图/式/电荷/无自由基/手性检查，及上传Calycosin SDF与可信图的规范SMILES一致性。它不是B执行记录，也没有进行SCF、优化、Hessian或振动计算。

B worker为 `scripts/research4_parallel_b_quantum_20261106.py`。默认CLI始终NOT_RUN。所有实际操作都必须有B新独立父监督器签发的真实PID/birth/boot/预算/来源绑定proof，包括纯RDKit准备和独立NumPy分析。B预算与A分开建账，不能把A授权或已耗成本伪装成B资源。

各case阶段为：prepare（可信SMILES经ETKDGv3与MMFF94s生成并初筛模型3D；Calycosin先核原SDF图）→optimize（有上限的优化、逐step坐标/梯度/能量与CHK；freshSCF/G通过）→frequency（同几何freshSCF/G后stock全Hessian，保留全部signed模式）→analyze（独立NumPy平移/转动投影及全部频率/保存模式复核）。prepare实际去重后候选数按真实结果记录；力场最低不等于DFT全局最低。

每阶段独立新attempt目录；历史失败、实际资源成本、虚频和部分检查点保留。原生Hessian不声称支持中途任意贡献块续算；中断后的该阶段需新attempt并复用已成功的前序优化几何。优化中断保留last_geometry和SCF CHK供有审核的新恢复操作使用；自动重试不得覆盖旧结果。

输出工件path为checkout-relative，适合整体回传后在A解析。实际PID/proof/output_directory中的B绝对路径只描述B的运行观察，不是A需满足的进程身份。

错误混种分类成绩已按2026-11-12用户决定撤回；论文使用main的paper/current白名单。
