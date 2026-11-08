# 独立云端 Worker B 启动说明

Worker B 运行水控制、原儿茶酸及条件允许的毛蕊异黄酮计算。它有自己的运行目录、CPU账本、进程监督和结果分支；不依赖A机器的大缓存，不继承A的GitHub凭证，也不修改A账本。

运行包已经用全新隔离虚拟环境按相同依赖清单完成真实水分子四阶段测试：优化、fresh梯度、完整频率、独立正常模式重放全部通过，3个正频、0虚频，实际7.196529 CPU秒。100项代码／证明／临时Git集成检查通过；60个真实水结果工件已在临时bare仓库逐字节回传验证。记录见`data/research/parallel_cloud_20261106/worker_b/actual_water_qualification_and_return_test.json`。这些是本机运行包验证，另一账号的实际科研执行仍须由下列正常命令启动，不能称已在第二台主机运行。

在**另一个有权访问该仓库的账号／云端会话**中执行：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none --sparse --single-branch --branch research/parallel-worker-b-20261106 https://github.com/jackzhu119/research4.git research4-worker-b
cd research4-worker-b
git sparse-checkout set scripts requirements docs data/research/parallel_cloud_20261106
bash scripts/run_research4_worker_b_20261106.sh --run-id B20261106-01 --check-only
bash scripts/run_research4_worker_b_20261106.sh --run-id B20261106-01 --detach
```

执行前确认发行分支已由主任务实际推送。若提示远端分支不存在，此时B尚未安装或计算，不应改用main。上面只下载B相关源码和小输入；结果回传使用`git add --sparse`处理新增报告路径。

可以给另一个Codex会话复制这段指令：

> 使用已连接且有写权限的GitHub账号，在独立云端按`docs/RESEARCH4_WORKER_B_QUICKSTART_20261106.md`克隆Research4的B发行分支，先执行check-only，再执行`--run-id B20261106-01 --detach`。监督这一个有限队列，保留真实失败成本和检查点。不要运行A队列、改main、扩充预算或声称未完成实验。收尾核验实际结果分支与GitHub提交；凭证不足时准确报告阻塞。

启动器先核对仓库与分支，以本账号的`git push --dry-run`检查写入协商，再只提交`owner_claim.json`并以非强制push原子认领结果分支。先成功认领的主机才能继续；另一主机看到远端已认领且没有本机所有权标记时，会报`BLOCK_ALREADY_CLAIMED`。缺少授权会在计算前停止；最终科研结果回传仍以实际push和远端提交核验为准。

发行分支是`research/parallel-worker-b-20261106`；本次结果分支自动建立为`research/parallel-worker-b-20261106-B20261106-01`。所有运行产物保存在`reports/parallel-cloud-20261106/worker-b/B20261106-01/`。当前发行配置是`release_ROOT_APPROVED_v3.json`，只分配远端ID **B20261106-01** 一次，不能更换ID、重新克隆或切换账号再次领取相同预算。本机水包验证使用固定恢复ID **B-water-package-validation-003**；001、002的真实失败及累计0.196167 CPU秒成本保留，003仅使用剩余额度，不算第二云端已运行。

环境默认为新的`.runtime/worker-b-20261106/venv`。已有不属于此启动器的环境不会覆盖；本启动器创建但安装中断的环境可按同一版本清单重试安装。代理、CA和平台认证保持原配置，不使用网络绕过方式。

依赖固定为PySCF2.14.0、geomeTRIC1.1.1、NumPy2.3.5、SciPy1.17.0、RDKit2026.9.1。A实际使用Python3.12.14；B优先Python3.12，也接受3.11，但其安装与科学兼容性需由B的实际版本检查和水Opt/Freq控制确认。准备完成、安装完成、水控制通过和药材计算完成是不同状态。

`--check-only`不安装、不切分支、不启动计算、不推送。`--smoke-test --no-upload`仅供水控制测试；在A分支上使用这两个参数会明确标记`NO_REMOTE_B_RUN`，不能算B独立云端已启动。正常B运行默认开启收尾回传。

后台进程可独立于终端连接继续运行；实例关机／被销毁后不保证继续。使用保存的manifest、authorization、启动回执、账本与状态文件核验恢复；存在活跃／未知worker时不重复启动。修复须保留原失败目录和真实成本。

只回传许可范围内的小型JSON、CSV、日志、结构和图表。CHK、巨型缓存、虚拟环境及秘密文件保留本地或拒绝上传；不删除用于恢复的原文件。真实Git回执记录科学提交及后续回执提交，不填写虚构SHA。理论频率和孤立分子结果不会自动进入原酸枣仁分类器。
