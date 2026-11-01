# 9原子Hessian工程对照：运行前封存

当前仅代码／mock测试及静态来源解析完成，真实Opt/Hessian尚未运行。没有SCF、梯度、Hessian求值，也未读取活动检查点。

root提供的固定模型为已核验ChEBI乙醇C2H6O，9原子／26电子，显式XYZ SHA和原精度NPY双绑定；无默认或猜造结构。identity=validation_ethanol／validation_only=true，只作软件工程验证，不是酸枣仁或其他植物样本。源PubChem直接访问受限，不冒称已下载官方3D。

运行前固定容差：E1e-8Eh／梯度1e-8EhBohr^-1／完整raw Hessian1e-6EhBohr^-2／全部21频率1e-2cm^-1。比较原全局Hessian与V7B4、B8；同一freshMF／坐标／MO，三者native1和MF/Hessian3000MB上下文均一致，保留原内部4000MB默认形参和全部交叉原子项。不按测试结果挑种子／结构或改变阈值。

先完整Opt和freshSCF／梯度通过才进入Hessian；每种Hessian后实际重算同MF梯度并与初fresh梯度核对。数值等价与21频率全部正值的极小点判据分别报告；负频率如实保存。原生CPHF正常返回不当作已测残差，IR NOT_COMPUTED，缓存SAVE_ONLY不冒称恢复完成。

标准全局Hessian不安装capture wrappers；B4/B8使用同V7公开接口记录中间量。所有实际科研执行由root通过总资源账本串行启动，本预检目录不构成实际验证通过。
