# 斯皮诺素受监控DFT批次：实际结束结果

本报告由已关闭控制器、最终CPU账本及保存计算文件的独立复核生成；没有启动新的SCF、梯度或Hessian计算。

控制器最终状态：`PARTIAL`。斯皮诺素频率状态：`BLOCKED_NO_VALIDATED_SPINOSIN_MODES`。

| 实际核查项目 | 结果 |
|---|---:|
| V3实际CPU秒 | 111023.48891300001 |
| 绑定V2实际CPU秒 | 14846.812969999999 |
| 斯皮诺素配对几何／梯度点 | 5 |
| 实际计算的斯皮诺素频率记录（含未验证及负频） | 0 |
| 通过完整检查的斯皮诺素模式记录 | 0 |
| 通过检查且位于0.1—1.5THz的模式 | 0 |
| 失败／部分完成阶段 | 1 |
| 已结束但没有worker结果文件的阶段 | 0 |

未计算、未收敛或尚未通过完整正频检查的结果均保留其实际状态；空频率表不代表零吸收。水和乙醇仅作软件／数值验证，不属于斯皮诺素机制证据。负频及缺少worker结果的失败证据保存在独立分析目录，未补造计算结果。

仅通过同方法父优化、fresh梯度、完整219个正频及原始Hessian复核的孤立分子模式，才进入实验候选峰比较。孤立分子与固态药材中的晶格、氢键和集体振动不同；频率接近不确认化学归属。IR强度为`NOT_COMPUTED`，不绘制虚构吸收强度。

本批次不修改酸枣仁光谱、标签、分类器或既有96.0234%分类结果，也不声称分子计算提高了分类准确率。RSS轮询属于尽力监测，24GiB地址空间不等于24GiB物理内存。

实际证据：[分析汇总](../reports/pyscf-spinosin-continuation-20261031-v3/final-analysis/summary.json)、[阶段成本与状态](../reports/pyscf-spinosin-continuation-20261031-v3/final-analysis/phase_status_and_resources.csv)、[失败证据](../reports/pyscf-spinosin-continuation-20261031-v3/final-analysis/actual_failure_evidence.json)、[理论与实验候选位置对照](../reports/pyscf-spinosin-continuation-20261031-v3/final-analysis/experiment_theory_pairs.csv)。

最终账本SHA-256：`19fbbc80b5e839dc645f10a467bb54f32fdebae14309a95d824520cd885d3d2c`。闭合控制器状态SHA-256：`a2346f01bb7997c0f5fdf84e23ee603cea8f1be8e484007791d76217bd23eca5`。

root条件授权SHA-256：`60d3d0e08f43f8c408b49fcbe924460368b2f4a8b77cb1a260d3c71232a916bd`。分析源码SHA-256：`92ef86dedcbef4a8879fd53a15fa2a209ea19c761c203b1a91177d60e2908a08`。
