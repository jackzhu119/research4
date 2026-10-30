# 安装环境与统一运行入口

本次已在当前云端配置隔离 Python/Octave 环境，并建立统一入口。目标是让主要算法家族各有可以执行、标明来源的路径；不是把全部旧MATLAB文件自动改成未经修改即可运行，也不把替代实现写成老师原代码的复现。

本次最终统一检查通过：**34个现场运行场景、47项单元测试**，另有1个经来源、日志和环境校验的历史完整SSA运行。当前环境与从31个缓存包独立重建的Octave均通过检查。结果、逐场景日志、执行脚本和SHA-256清单见 [本次安装实测记录](../reports/runtime/2026-10-30-installation/RUN_REPORT.md)。

## 两条命令

在仓库根目录执行：

```bash
python scripts/setup_research_environment.py
python scripts/run_all_algorithms.py --profile full
```

第一条优先复用已验证组件，缺失时按固定版本安装到 `.runtime/`。当前云端也已验证离线复用和从31个缓存Debian包重新构建Octave；没有全局安装或修改原件。需要省略网络下载时可加 `--offline`，缺包会明确失败。

第二条依次完成环境检查、来源解压/校验、Python基础流程、缺失家族的开放替代、选择器和Octave兼容路线，结果保存在 `outputs/all-algorithms/时间戳/`。阅读 `RUN_REPORT.md` 与 `run_summary.json`；具体日志和每组 `results.json` 在同一目录。

## 已配置内容

| 内容 | 当前版本或角色 |
| --- | --- |
| Python | 3.12.14；隔离venv，可直接导入仓库 `src/` |
| 数值与模型 | NumPy 2.3.5、SciPy 1.17.0、scikit-learn 1.8.0、openpyxl 3.1.5 |
| CNN | PyTorch 2.9.0+cpu |
| Octave | 9.4.0，统计、信号、Excel IO、图像、优化等包 |
| 无界面绘图 | gnuplot与字体，工作副本可生成图形 |
| 原生MATLAB | 未安装或验证；需要真实学校/个人许可及对应工具箱 |

环境规格、锁文件、平台限制及医生程序见 [environment/README](../environment/README.md)。离线缓存重新构建并不是验证所有未来网络源永久可用；换到新环境应重新执行安装器。

稳定执行器为：

```bash
.runtime/bin/python -c "import herb_spectra, torch; print(torch.__version__)"
.runtime/bin/octave --eval "pkg load statistics signal io image optim; disp(version);"
python scripts/environment_doctor.py --output outputs/environment-doctor.json
```

环境中的Python会复用已检查版本的基础镜像包和已有CPU框架。移动或删除依赖前缀后，重新运行安装器；不要将几个启动脚本单独复制到另一台机器当成完整环境。

## 算法文件如何对应运行

| 原文件/算法 | 统一入口使用的路径 | 来源关系 |
| --- | --- | --- |
| `0308 CARS+UVE` PLS、CARS、噪声UVE | Octave核心/主流程，补 `sumsqr` 等已说明接口 | 兼容运行；参数与预算见结果记录 |
| `CARS（sy）` | 去掉错误误差缩放的工作副本 | 逻辑修正版，不改原包 |
| MC-UVE、分类变量筛选 | 绝对稳定性、one-hot多分类目标的Python实现 | 独立实现，保留来源关系 |
| IRIV | 原 `.m` 低维保护版，以及有界IRIV-inspired Python版本 | 保护性修正/独立替代，均不声称原始统计策略完全相同 |
| PLS-LDA、PLS CV、Random Frog | Octave的可用核心调用 | 随包示例与明确的小预算 |
| SVM、SVR | Linux LIBSVM兼容调用与sklearn实现 | 后端与接口不同，分别标注 |
| 手写BP | 梯度已核对的向量化工作副本 | 相同逐样本更新公式的性能改写 |
| `newff` 风格BP | sklearn sigmoid MLP | 独立替代，不是原newff或原手工MSE目标 |
| SSA-SVM | 现场小预算运行；完整100轮可核验历史证据或重新执行 | 兼容适配；寻优CV仍不等于外测 |
| CNN | PyTorch的2D层结构迁移，保持MATLAB列优先重排 | 结构迁移，3轮工程示例，不是原权重复现 |
| `LSSVM.txt` | 带独立偏置的RBF LS-SVM块方程求解器 | 独立实现，不是把SVR改名 |
| `PCA.txt/my_pca` | 保存训练均值和载荷的图像PCA | 独立替代 |
| 纹理DOCX、Sobel | 正确灰度邻接GLCM和Sobel运算 | 修正/标准操作实现；需要真实图像 |
| 平滑 | 显式信号轴的15点SG滤波 | 替代接口；未保证MATLAB所有端点行为一致 |
| `syqpcx` | 参数化行抽取与等长拼接 | 数据适配工具；拒绝90/91列不一致 |

统一入口覆盖主要建模、选择器和图像工具家族；UVE工具箱中ECR、OPLS、SPA、PHADIA等其他辅助函数并未逐一纳入就绪声明。它们的存在不自动等于已验证可用于你的中药任务。

新接口位于 `herb_spectra.algorithms` 和 `herb_spectra.selection`。选择器支持传入 `groups`，先按训练物理样品聚合重复扫描；多分类使用one-hot目标，避免把产地整数当连续含量。变量选择仍须在每个训练分区内拟合，内部选择误差不能替代独立外层评价。

## 时间预算与结果含义

```bash
# 更短的全部主要家族检查
python scripts/run_all_algorithms.py --profile quick

# 强制重新执行完整100轮SSA（而非读取既有完整证据）
python scripts/run_all_algorithms.py --profile full --rerun-slow

# 额外检查未修改旧入口的阻塞情况，独立标为diagnostic
python scripts/run_all_algorithms.py --profile full --diagnose-originals
```

完整100轮SSA之前已实际耗时约513秒。本次安装验证现场执行小预算SSA，并核验该历史完整记录；本次另外启动的完整重跑被中断，没有完整结果，未计入通过数。正常 `full` 模式在归档来源、日志、配置元数据、Octave及包版本校验通过时可以引用历史完整记录，状态写成 `cached_verified`，保留原运行日期，不伪装成本次新执行。缓存不满足时会重新计算；`--rerun-slow` 强制重新运行。

退出码0只表示选定的**可运行路径**全部完成；过程退出0但内部JSON中有失败，统一入口仍会报失败。故意复现缺陷的测试不能计入算法就绪。`--require-originals` 会在旧入口仍有阻塞时返回2，即使替代/兼容路径通过。

报告分别保留：`selected_paths_passed`、`original_files_all_unmodified_runnable`、`matlab_native_execution_verified`、`research_performance_verified`。后面三项当前为false。部分MLP示例有迭代收敛警告，警告单独保存；程序执行完成不能解释为该模型已充分收敛或优化。

## 原生MATLAB与真实中药模型的边界

安装器不会安装/激活MATLAB或补发许可证，也不会把Octave/Python成功当成MATLAB原生成功。`trainlssvm/my_pca/carsplslda`等原文件缺失、绝对路径、残缺条件及不正确的验证流程，都需要分别补资料、修源码或采用明示替代。

中药数据的真实样品编号、产地、批次和化学实测值仍待补。统一命令使用随包示例及合成工程夹具；真实中药部分只检查读取。CNN示例存在9条跨训练/测试相同特征行，示例高分不能作为论文结论。

原始材料、详细历史实测及科研协议分别见 [资料库](../references/paper-materials/README.md)、[可用性报告](RUNTIME_USABILITY_REPORT.md)、[实验协议](RESEARCH_PROTOCOL.md)。
