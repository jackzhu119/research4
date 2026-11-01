# 原生进程无final result的后处理扩展

新V2分析器保留已封存旧源码，增加已结束但没有result.json的明确PARTIAL派生。只有父进程已结束且总账本没有RUNNING、root放行绑定最终SHA，才会读取input/progress及已有片段。原计算目录不写入伪造result.json，不把快照freqDONE提升为有效频率，缓存／Hessian片段只是部分执行证据。

7项纯合成测试通过，含native exit1、已计算片段、未完成频率声明、输入快照身份／方法冲突、运行中拒绝、FAILED_TO_START、完整纯测试导出（保留梯度点及空理论列）。没有读取当前活跃检查点或导出本轮真实final报告；无SCF／梯度／Hessian任务。
