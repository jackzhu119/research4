# Google Colab连接实测（2026-10-31）

**当前状态：官方CLI已经安装并验证，尚未连接Google账号或Colab运行环境。** 当前云端的受限HTTP代理拒绝Colab及Google令牌服务访问，返回`Tunnel connection failed: 403 Forbidden`。没有创建远程CPU／GPU实例，也没有执行远程科研代码。

## 已实际完成

- 核对Google官方[Colab CLI仓库](https://github.com/googlecolab/google-colab-cli)，安装PyPI发布版`google-colab-cli==0.7.5`，许可证Apache-2.0。
- 使用独立Python环境`.runtime/colab-cli/`安装，保留现有光谱研究环境。
- `colab version`输出`Version: 0.7.5`，`pip check`通过；检查全局与`new`命令帮助。
- 创建`.runtime/bin/colab`入口，使用`umask 077`，将会话元数据放在被Git忽略的`.runtime/colab-state/`。该参数只控制会话元数据，不能误认为改变OAuth缓存路径。
- 执行`colab --auth oauth2 sessions`，到达远程浏览器授权提示；未提供授权码，没有完成授权。授权URL及原始认证输出未保存至报告。
- 正常通过现有代理请求下列服务，全部被代理拒绝；没有改用其他代理、IP或隧道。

| 目标 | 本次实测 |
| --- | --- |
| `colab.research.google.com` | 代理CONNECT返回403 |
| `colab.pa.googleapis.com` | 代理CONNECT返回403 |
| `oauth2.googleapis.com` | 代理CONNECT返回403 |

机器可读结果见[connection_status.json](../reports/software/2026-10-31-colab/connection_status.json)。当前可用工具中没有已连接的Colab运行环境或可控制的浏览器会话；官方CLI提供了无需浏览器控制扩展的后续途径。

## 恢复连接的前提

1. 在受支持的云端环境配置中允许上述3个目标。编辑执行机的网络策略快照文件不能放行；该策略由平台代理执行。用户完成Google授权的浏览器还需能访问`accounts.google.com`及`sdk.cloud.google.com`。
2. 网络放行后，运行下列命令，由用户在Google官方页面完成授权，再回到交互终端完成CLI提示。不要把密码、授权码或令牌写进GitHub、笔记本或报告。

```bash
.runtime/bin/colab --auth oauth2 sessions
```

实际安装版0.7.5的`--help`默认认证为`oauth2`，支持远程授权流程；认证逻辑使用用户目录中的OAuth缓存文件。GitHub最新开发版文档与发布版存在差异，本次没有把最新开发版的keyring功能描述为0.7.5的已装功能。

## 授权完成后的连接验收（尚未执行）

先创建普通CPU环境、执行一个最小算术测试，然后停止该测试环境：

```bash
.runtime/bin/colab new -s research4-connect-test
printf 'print(1 + 1)\n' | .runtime/bin/colab exec -s research4-connect-test
.runtime/bin/colab stop -s research4-connect-test
```

以Google返回的运行环境状态和远程执行输出作为连接成功证据。GPU／TPU和高内存设置不用于首次连接检查；之后按实际训练需求及账号资源决定。

后续科研输入仍只认[data/CURRENT_DATASET.md](../data/CURRENT_DATASET.md)中的7份编号版酸枣仁、855条谱记录。Colab接入不改变数据版本，也不代表样品身份、批次或模型性能已经验证。Colab的[local runtime](https://research.google.com/colaboratory/local-runtimes.html)使用自有机器算力，不等于取得Google托管GPU。

## 重建隔离安装

```bash
python3.12 -m venv .runtime/colab-cli
.runtime/colab-cli/bin/python -m pip install -r environment/requirements-colab-cli.txt
.runtime/colab-cli/bin/colab version
.runtime/colab-cli/bin/python -m pip check
```

上述命令重建CLI本体；当前工作区已有的`.runtime/bin/colab`入口还设置了私有文件权限和会话元数据路径。授权信息不随仓库分发，不能用已安装CLI代替Google账号授权。
