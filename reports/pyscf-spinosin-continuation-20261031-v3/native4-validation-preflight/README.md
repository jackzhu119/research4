# V8 native4乙醇工程对照：运行前封存

当前仅新代码／9项纯测试与原静态来源解析完成，未执行Opt、SCF、核梯度或Hessian，不读取活跃检查点，也未创建或读取生产资源监督证明。旧V7脚本、旧预检和两个分析器保持字节原样。

源仍为既定9原子／26电子乙醇C2H6O，显式原XYZ SHA及fullprecision NPY双绑定，identity=validation_ethanol／validation_only=true，不是植物或酸枣仁数据。没有构造默认／猜造结构或按测试结果挑分子、种子。

实际执行前须通过新run_pyscf_monitored_budget_20261031.py监督器；程序在量子包导入和任何积分／SCF之前调用V8 verify_resource_supervision(expected_threads=4)，验证真实live父子PID、来源SHA、AS24GiB及RSS10GiB／cgroup12GiB监控合同。AS是虚拟地址上限；RSS／cgroup轮询为BEST_EFFORT，不能声称硬居民内存保证或证明未来监控健康。旧V2预算不可替代该合同。

初始Opt＋freshSG在native1单独scope完成并恢复；随后同一MF／几何／MO顺序执行标准full/native1、B4/native4、B8/native4，三者Hessian MF／Hobj memory3000上下文和原全局交叉耦合不变。每个scope独立不嵌套，异常也保存恢复控制记录，配置线程上限不冒称实时活跃worker数量。

事先固定阈值仍E1e-8Eh、梯度1e-8EhBohr^-1、全raw Hessian1e-6EhBohr^-2、全部21频率1e-2cm^-1，不根据实际测试结果修改。每种Hessian后重算实际同MF梯度与初freshG核对。DONE表示数值等价通过，21频率全部正值的stationary_minimum_verified另行报告；根进程不能仅看exit0或DONE。负频率保留，IR NOT_COMPUTED，原生CPHF返回不当作已测残差。

真实计算由root在新资源批次中串行启动，本预检不构成native4数值验证通过或药材分子振动结果。
