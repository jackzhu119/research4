# Spinosin 专用计算：实时结果

状态：PARTIAL。

仅运行 Spinosin；原 Calycosin 计算进程与报告保持独立。PBE/def2-SVP、无 D3、grid5、原生直接 SCF，完整网格响应梯度。

计划两个步长共 900 个真实位移点，保留全部有符号频率及真实偶极导数 IR 强度。完成与科学资格分开报告。

[机器可读汇总](../reports/pyscf_spinosin_direct_20261113/results_summary.json)。

计算范围为孤立分子，不据此确认药材固体 THz 峰或逐样品化学标签。按用户指令停止旧 Calycosin 队列后，独立分配 4 线程及 10000 MB PySCF 内存设置；本脚本不向其他进程发送信号。
