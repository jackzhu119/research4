# 斯皮诺素完整结果快照（2026-11-15）

本数据分支包含完整结果包分片、逐文件清单和校验值。GitHub Release 上传主机鉴权不可用，故使用仓库独立数据分支保存完整二进制数据。

10条合格能量/核梯度记录对应8个不同几何点；尚无收敛极小值、Hessian或IR结果。原始结果、密度矩阵、检查点、日志、代码、输入和一致性SQLite账本共293个文件均包含在压缩包中。

下载本分支全部文件后，在同一目录执行：

```sh
cat spinosin_complete_20261115T072056Z.tar.gz.part-* > spinosin_complete_20261115T072056Z.tar.gz
sha256sum -c SHA256SUMS
tar -xzf spinosin_complete_20261115T072056Z.tar.gz
```

PARTS.json提供分片顺序和SHA256，MANIFEST.json提供包内全部文件SHA256。每个上传分片已从GitHub重新下载并核对SHA256。

科研报告与运行代码：[研究分支](https://github.com/jackzhu119/research4/tree/research/spinosin-conformers-parallel-900000-20261112/reports/spinosin_conformers_parallel_20261112)。
