# PySCF优化检查点续算准备

状态：`READY`。只提取并核验几何与生成命令；本阶段没有执行SCF、梯度或优化。

原累计量子CPU预算剩余633.765852秒；本次最多可请求631秒，不重置或扩大账本。

检查点可能包含下一试探几何，并不等于最后一个完成梯度或驻点；已保存完整精度.npy及带新SHA的XYZ。恢复命令始终要求新的SCF和梯度。

执行前必须核对全部冻结父证据及同一CPU账本。可运行`bash /workspace/research4/reports/pyscf-spinosin-pilot-20261031/continuation/resume_same_budget.sh`，本准备过程没有执行此命令。
