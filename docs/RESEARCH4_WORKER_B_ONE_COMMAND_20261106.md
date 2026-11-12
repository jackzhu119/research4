# Worker B：一行安装、启动与回传

在另一Codex云端的Linux x86_64终端执行。标准入口需要bash/curl/coreutils；Git、Python及venv缺失时尝试有界apt安装，须有root或非交互sudo。Python限3.11/3.12；不兼容系统明确BLOCKED，不假称已完成安装。

```bash
bash -o pipefail -c 'curl -fsSL --max-time 60 https://raw.githubusercontent.com/jackzhu119/research4/4f854f20cd3a4e9f489c6f0abb9710a5070d653d/scripts/bootstrap_research4_worker_b_20261106.sh | bash -s -- --device-login'
```

入口已经实际推送，固定脚本提交为`4f854f20cd3a4e9f489c6f0abb9710a5070d653d`，SHA-256为`6c5181212ad5ec218f76c536028eadd0ad48fad4f6297ee264d8f4d4f557e6c2`。公开raw URL在A环境真实返回HTTP200且字节完全一致；尚不能据此声称另一环境可访问。

自动流程：核查基础环境→仅安装缺失工具→稀疏克隆固定B发行分支并核对完整c2284ad4755b223830aeb2283d7491a3a3f35a28→核验launcher哈希→配置缺失的本地机器人提交作者→实际Git写入dry-run→原子认领唯一运行→安装固定版本PySCF/geomeTRIC/NumPy/SciPy/RDKit→水完整控制→原儿茶酸、毛蕊异黄酮有限计算→检查点与监督→真实闭合后自动回传。

默认目录`/workspace/research4-worker-b`，不可写时使用当前目录的同名子目录。已有非目标目录、错误分支或错误代码版本都拒绝覆盖。固定run-id为`B20261106-01`，结果分支为`research/parallel-worker-b-20261106-B20261106-01`。不改main，不移动B冻结发行分支，不新增预算、运行ID、定时任务或A计算。

## GitHub授权和403

聊天中的操作授权不是GitHub登录凭证。优先在另一Codex账号连接有research4写权限的GitHub身份，可以连接同一个jackzhu119账号。没有从A复制、导出或公开任何token、私钥或注入凭据。

`--device-login`仅在实际写入检查失败且没有继承Git认证配置时提供用户自己的网页设备授权，需要真实终端；让另一Codex使用`tty:true`执行。缺少gh时有界安装；授权状态仅保存在该新B目录的忽略目录，脚本不读取PAT、不修改已有全局认证。已有认证失败时保留它，并提示通过Codex连接修复。网页确认不能由聊天文字代替。

公共仓库可下载不代表可以push。CONNECT/proxy403属于网络准入；GitHub permission denied属于写权限。脚本不改变代理、CA、路由或平台策略，不绕过网络拒绝。网络策略未允许GitHub或设备流程需要的端点时，通过该环境支持的配置/连接修复，然后重新检查；不声称一行代码能取消403。当前授权截止仍为2026-11-09T08:25:17.730193Z。

## 验证范围

`bash -n`、实际`--check-only`及7个离线集成检查通过。测试覆盖无副作用预检、参数错误、模拟完整委派、写失败阻断、完整SHA校验、未知目录保留、继承认证保护；模拟Git和子bash不执行真实clone、push、安装或Q。B原包另有真实水Opt/Freq与字节级临时Git回传证据，见原quickstart。新bootstrap没有冒充第二台机器已启动或已获得写权限。

错误混种分类成绩已按2026-11-12用户决定撤回；论文使用main的paper/current白名单。
