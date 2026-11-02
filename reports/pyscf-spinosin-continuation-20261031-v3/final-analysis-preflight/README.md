# V3最终结果分析：仅代码与纯测试完成

12项纯测试通过，未读取实际V3计算资料或活跃检查点，未导出真实final报告，没有SCF／核梯度／Hessian求值。测试中的频率数组、proof和ledger为临时合成夹具，不是科研结果；数组及小矩阵真复算仅在root最终放行后进行。

实际CLI：analyze_pyscf_monitored_results_20261031.py --run-root <V3目录> --output <新analysis目录> --release <root最终放行JSON>。放行JSON强制绑定最终ledger及已结束controller state路径／SHA，后者ledger_sha256_at_end必须同最终ledger；RUNNING及优化与频率间隙均拒绝。

采用官方V3验证器及独立CPU／closed V2／授权／执行源码／归档proof／日志检查；proof不冒充当前live PID或硬RSS限额。freq父优化还必须对应唯一EXITED0 supervisor owner和真实75x3 fresh gradient，不只信JSON标志。missing result按真实父退出和快照派生PARTIAL，不补造worker文件。

computed_Spinosin_frequency_rows_including_unvalidated是目标实算记录数（含负频），validated_spinosin_frequency_rows只有通过全部219正频／父opt／freshG／rawsymH／小矩阵replay的记录数。computed_frequency_rows_including_unvalidated总数还含水验证，不能当作斯皮诺素数量。乙醇21模式只在validation controls，不进药材理论表，多后端复测不当新增样品。IR NOT_COMPUTED，频率接近不确认化学归属。
