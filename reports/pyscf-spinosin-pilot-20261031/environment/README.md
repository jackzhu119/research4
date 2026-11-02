# 环境和复现入口

本轮真实使用Python3.12.14，复用`.runtime/quantum-upgrade-site/`内PySCF2.14.0／h5py3.16.0；新增geomeTRIC1.1.1／RDKit2026.9.1位于独立`.runtime/quantum-pilot-site/`。主ML环境未升级。工作脚本仅在自己的进程中加入独立路径。

- `environment.json`：资源、版本和新增下载摘要。
- `dft_backend.json`：LibXC7.0.0，PBE／grid1，无D3，梯度grid_response=False。
- `dependency_download.log`、`dependency_artifacts.json`：实际下载／构建日志、文件尺寸及SHA；二进制包保留于本地ignored目录，不上传Git。
- `software_metadata.json`：安装分发包的许可字段及官方项目链接。字段缺失不等于无许可证。
- 原PySCF依赖锁仍在`data/research/quantum_upgrade_20261031/environment.lock.json`，本轮没有覆盖。
- `project_tests_frozen_final.log/.json`：冻结新版代码后559项测试全部通过；实际测试子进程CPU约39.73秒，非量子计算。更早540／558项日志为当时快照，保留原样。

运行纯核验／后处理工具时使用：

```bash
PYTHONPATH=.runtime/quantum-pilot-site:.runtime/quantum-upgrade-site \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.runtime/bin/python -m unittest discover -s tests -v
```

不要为了复现水测试而覆盖已有结果目录。量子worker必须经过`scripts/run_quantum_budget_20261031.py`并使用现有累计账本；实际续算命令由计算结束后的`continuation/`提供。环境目录仅包含元数据和日志，运行时、安装包、巨型积分缓存均不进入Git。
