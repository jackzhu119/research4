# 酸枣仁斯皮诺素PySCF计算试验

运行编号沿用用户指定20261031；实际计算日志使用UTC，日期为2026-11-01。独立分支`research/pyscf-spinosin-mechanism-20261031`，基于`3fc996e`。不改原始光谱、历史分类器或main。

- [结构、立体化学与皂苷A/B资格](structures/README.md)
- [12个真实生成构象与来源](conformer-execution/README.md)
- [方法、Hessian、内存及水实测独立核查](method-audit/README.md)
- [总CPU预算与每阶段实际资源](cpu_budget.json)
- [依赖环境](environment/environment.json)
- [完整水Opt/Freq结果](water-optfreq/result.json)
- [缓存DF-J水Opt/Freq结果](water-cached-optfreq/result.json)
- [首次direct-J斯皮诺素SCF部分结果](spinosin-scf-gradient/result.json)
- [缓存DF-J斯皮诺素优化PARTIAL结果](spinosin-cached-optimization/result.json)
- [全部阶段最终状态](final_status.json)
- [完整科学报告与恢复命令](../../docs/PYSCF_SPINOSIN_PILOT_20261031.md)
- [实际CSV、PNG及PDF](analysis/README.md)
- [检查点核验及有界续算准备（没有执行）](continuation/README.md)
- [只读交付审查与数值诊断](delivery-preflight/)

最终：结构／12构象准备DONE；水完整Opt/Freq DONE；首次斯皮诺素direct-J SCF为PARTIAL；随后同方法缓存DF-J完成4个配对SCF／核梯度点，第5点梯度在CPU保护时中止，优化PARTIAL。全部量子进程已结束，累计2666.234148 CPU秒；原量子预算余633.765852秒，生成631秒恢复命令但未执行。没有通过几何收敛，斯皮诺素Hessian／正常模式NOT_RUN，理论比较BLOCKED，不能将水频率当作药材结果。

本轮使用PBE／def2-SVP／grid1，无D3，IR强度始终NOT_COMPUTED。所有坐标模型、计算状态和资源限制明确保存；不生成伪造理论吸收曲线或声称提高分类准确率。
