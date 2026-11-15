完整分子结果快照。包含原始密度矩阵、持久波函数检查点、坐标、梯度、偶极、日志、代码、输入和一致性SQLite账本。
10条合格点记录对应8个不同几何，尚无收敛极小值/Hessian/IR结果。旧running文件为重启前历史证据。
按PARTS.json的顺序拼接*.part-*得到tar.gz，先核对总SHA256，再解压。MANIFEST.json列出每个文件的SHA256。
原JSON中的绝对路径以original_root为根；解压后按相对路径定位，不需要改写原证据文件。
