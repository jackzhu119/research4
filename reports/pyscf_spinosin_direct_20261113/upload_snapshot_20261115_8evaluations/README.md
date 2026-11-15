# Spinosin 最新完整结果快照（2026-11-15，8 次完成的优化评估）

当前完成的是几何优化中的 8 次电子结构评估（step-00000 至 step-00007）。这些是优化器试探几何，不是 8 个独立构象，也不是 Hessian 位移点。8 次均为保存记录中 DONE、qualified、native_converged；逐点能量、核梯度、电子审计及几何哈希见 evaluations.csv。

结构优化尚未收敛。900 个计划频率位移点目前完成 0 个；本快照没有新 Hessian、振动频率或 IR 结果，不能从优化轨迹推导部分有限差分 Hessian 列。frozen 中名称含 historical 的输入仅为原有历史来源，不代表这轮已算出的频率结果。

step-00008 保存为待续的未完成点，仅含中断前输入、日志、电子尝试记录和 checkpoint；不计入 8 个合格结果。metadata_before_restoration 内 RUNNING 是云环境中断前保存的元数据，不能视作活进程证据。快照制作没有启动量子计算、修改科学代码、原始点、检查点、状态或进度。Calycosin 按用户指令继续停止，不能通过旧串行队列自动重启。

ZIP 内容：points/ 含 8 个完整点的 input/result、坐标、梯度、偶极、8 个密度矩阵及所有尝试日志/审计，并保存 pending step-00008；optimizer/ 含优化日志和冻结进度；recovery_checkpoints/ 含最后完整 step-00007 与 pending step-00008 的 HDF5 rolling.chk；metadata_before_restoration/ 含协议、授权账本、独立 8 点核验及原状态；frozen/ 含 Spinosin 冻结输入、科学脚本和版本 requirements；evaluations.csv 和 optimizer_trials.xyz 便于检查。

恢复请先在空目录解压，逐项按 manifest.json 核对 SHA-256；archive_path 对应包内文件，source_path/restore_path 对应原仓库位置。density.npy 在包内移出 .local 以便下载，实际恢复必须按 restore_path 放回逐点 .local/density.npy；两个 rolling.chk 同理恢复到原 attempt 的 .local。只恢复到独立仓库副本，先检查 manifest/协议/数组哈希，再由独立 Spinosin supervisor 恢复；不要覆盖正在运行的目录，也不要将 pending checkpoint 当成合格结果。元数据中的旧绝对路径及进程号只是历史记录。禁止启动 background_pyscf_repair_20261111.py 旧四分子队列。

所有大型数组/完整 payload 仅放在本地隐藏 .local/upload_pack_20261115_8evaluations 及单一 ZIP；普通快照文件夹仅放小型 README/CSV/manifest。已重开 ZIP 进行 CRC、逐文件 SHA-256 和 32 个数组逐元素比较；结果记录在 manifest.json 的 validation 中。
