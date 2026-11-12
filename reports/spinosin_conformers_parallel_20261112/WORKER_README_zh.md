# 斯皮诺素不同构象补充计算入口

`worker.py --root ROOT --job ID`（或绝对 job.json 路径）由外部 supervisor 启动。加 `--preflight` 只检查输入，不计算量子化学；本次核查和十项自测均未启动 DFT。

使用固定提交 `cd4b50c37c1e26064f9f15f340d30e20ac386458` 的 C002、C003 ETKDGv3/MMFF94s 已收敛构象，均为 C28H32O15、75 个显式原子、320 电子、中性单重态；PubChem CID 155692、CHEBI:81360，立体结构标识为 `VGGSULWDCMWZPO-ODEMIOGVSA-N`。所有下载文件按 Git blob SHA 和 SHA256 核验，见 `source/source_manifest.json`。这两个构象与其他 host 使用的 C001 起点不同，最终是否归入同一极小值仍须实际计算确认；构象不增加独立化学或生物样本数。

依赖 `pyscf==2.13.0`、geometric、numpy、scipy、h5py。每个 worker 限一线程，模型内存 2200 MB；这些设置与实际 profile 均进入点的指纹。复用固定 publication 提交 `2eb6b6e1dfe6627f24c8f97ea268bdec8f636bcb` 的严格 SCF、独立梯度、轨道稳定性、几何优化、负曲率双向松弛和独立质量门。首先 df-grid5；无受支持下降方向或数值资格不足时，实际推进至 df-grid7、direct-grid7。支持负模方向的轮次没有任意六轮上限；全部预定方案未解决时明确阻塞本 job，不重复读取封存失败形成空转。

75 原子的两步数值 Hessian 和偶极导数需要 900 个位移点（0.002、0.001 Bohr）。只有完整原始数据、Hessian、模态与 IR 资格门重放通过，才写 `result.json` 中 `status=complete, complete=true` 及 `done.json`；原 C001 的 25 个虚频不构成任何新完成证据。退出码 75 表示断点重试，76 表示仅该任务科学阻塞，77 表示等待同盆地先行任务（至少 600 秒或先行任务状态事件后再查），78 表示完整性问题暂停。

优化步骤点放在完整科学指纹子目录，避免进程恰在点写完、优化进度未写完时被杀而错用旧 step 目录。正在进行的单个 SCF 可能重做，已经完整封存的点与优化几何可续用。`heartbeat.json`、`cpu_receipt.json` 按本进程实际 CPU 时间（含导入）记账，不生成子进程。

开始完整 Hessian 前，本机较后 ordinal 的任务会保守检查较前任务的严格驻点：相同实际设置，全部 75 原子在不反射的刚体对齐后 RMSD ≤1e-5 Bohr、最大偏离 ≤3e-5 Bohr，电子能差 ≤1e-7 Hartree。若同一盆地则等待或引用其通过验证的完整证据，标明 `basin_duplicate`，不重复算作样本。该检查不猜测对称原子映射，当前不合并未审查的远端 C001 状态。

所有旧仓库输入和代码保持原样。计算只支持明确模型下孤立分子的局部谐振和 IR 预测，不证实全局最低构象、药材固体 THz 峰归属或实测样本的成分含量。远端只发布较小的 JSON/NPY 证据，通常不含每点 density.npy，因此不能称为完整波函数备份或承诺跨 host 无损恢复；完整本地检查点保留。

恢复时重核上一个 case 的 SHA、种子源/负模选择文件 SHA、种子密度 SHA、完整有限坐标及其与源的逐值一致性。临时 DF 文件放在 supervisor 分配的磁盘 `root/tmpfiles/<attempt>`，在导入 PySCF 前设置 TMPDIR/PYSCF_TMPDIR。每个合格完整点的全部 array_refs 校验后，才移除未引用的 `attempt-*/.local/rolling.chk`；密度、坐标、梯度、偶极及结果均保留。进程结束后由 supervisor 清理该 attempt 的临时目录。
