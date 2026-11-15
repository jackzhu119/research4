# 最新进度和完整结果下载入口（2026-11-15）

本轮按“保存一致性快照 → 上传并验证完整结果 → 原授权内恢复后台”的顺序执行。源文件、科学协议、已完成结果和原 CPU 账本均保留。旧云主机运行标记不作为当前进程存活证明。

## 斯皮诺素分子计算：完整上传已验证

- [完整数据分支](https://github.com/jackzhu119/research4/tree/research/spinosin-complete-results-20261115-data)
- [Release 下载入口](https://github.com/jackzhu119/research4/releases/tag/research-spinosin-results-20261115)
- 包大小：81,565,460 字节；5 个分片，293 个文件。
- 总包 SHA256：`ab0fdefdd2084c7a4079cccc8cb2a553b6a948e77ec97bbbf3eb113b03da7a64`。
- 所有分片已从 GitHub 下载并逐片核对 SHA256；包内每个文件也已反读并核对清单。
- 包含全部持久结果、密度矩阵、波函数检查点、坐标/梯度/偶极、SCF/优化日志、代码、输入、授权和恢复记录、一致性 SQLite 账本。

科学进度为 C002 的 10 条合格点记录、8 个不同几何；C003 的完整点为 0，保留检查点并排队。C002 尚未达到驻点标准，无已验证极小值、Hessian/振动频率或新 IR 结果。step 2/3、step 4/5 是重启后的同几何重算，不作为新结构或独立实验。

## 原光谱训练：完整归档正在上传

- [归档数据分支](https://github.com/jackzhu119/research4/tree/research/background-study-1360000-20261115-archive)
- [Release 入口](https://github.com/jackzhu119/research4/releases/tag/research-ml-results-20261115)

导出边界为 SQLite 已正式提交的 83,017 项完成任务。未完成和仅有文件、尚未正式提交的任务均单列，不增加完成数；重复划分不是独立样本。完整归档包括模型、预测、特征选择、诊断、分区、原始计算输入、报告、代码/协议及一致性账本。上传中的分支明确标记为 incomplete，全部分片和清单验证后才标记完整并恢复训练。

## 存储与恢复

GitHub Release 大文件上传主机当前鉴权不可用，完整二进制结果改存同一仓库的独立数据分支；Release 提供说明和下载入口。所有分片均低于 GitHub 单文件限制，下载后按 PARTS 清单顺序拼接并验证 SHA256。独立数据分支不修改论文主分支。

临时积分缓存、虚拟环境及假任务测试运行目录不作为科研结果。较大的实际科研数组和检查点已经包含，不再只提供小型报告。

累计授权保持分子 1,900,000 CPU 秒、原光谱训练 1,360,000 CPU 秒，互相独立。打包、发布和云重启期间的不确定 CPU 采用保守资源记账，保留总授权上限，不以准确率提前结束。
