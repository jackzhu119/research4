# V8 有限后台响应流水线准备

状态：准备完成、目标计算尚未由本作者启动。Root需要独立审查、创建启用授权并做实际资源检查。

入口 `scripts/research4_response_pipeline_20261105_v8.py`。实际源码、测试、已填入真实V8控制release的禁用配置及授权模板的SHA见 `FREEZE_INDEX.json`。

## 执行范围

复用已通过旧producer严格资格校验的真实A/H1，只运行10组新V8完整B8响应、全局Hessian和新独立模式分析。每组仍保留全部AO/MO和原生full-system响应算子。第1组请求21600 CPU秒/14400墙钟秒，其余保持同一有界请求；不能增加已注册390160.8308359999累计CPU秒或2026-11-09T12:30Z期限。

首个响应真正健康闭合后，按已完成响应最大实际wait4成本×剩余块数×1.2+1800 CPU秒全局组装余量评估总预算；不够则保留健康结果并BLOCKED等root评估，不偷偷扩额。该预测不是完成保证。

每个新阶段启动前要求当前invocation剩余墙钟足够完整请求+180秒；不足则在健康边界正常退出，既有完成阶段不重算。下一次root或恢复任务可用同一不可变config/grant、全新安全attempt名、有限invocation次数续接；失败/孤儿/未闭合阶段不能这样直接重试，需要root取证和独立恢复配置。

`--detach`使用stdin DEVNULL及新独立session。每次attempt保存独立console与detached-launch-{attempt}.json，current-launch.json为可更新运行指针；不会覆盖历史launch。机器关闭/重启仍可能中断，不能承诺存活。

## 元数据预检（不执行）

```bash
.runtime/bin/python -B scripts/research4_response_pipeline_20261105_v8.py \
 --config reports/pyscf-spinosin-96h-continuation-20261105-v8/queue-configs/response-pipeline-001/config_ACTUAL_CONTROLS_DISABLED.json \
 --config-sha256 95c4dab131ff906ba9804026d809e3e39a09c8b4d46f18aad3da9ba0b339303a \
 --grant reports/pyscf-spinosin-96h-continuation-20261105-v8/queue-configs/response-pipeline-001/root_grant_ACTUAL_CONTROLS_TEMPLATE_DISABLED.json \
 --grant-sha256 2ba2fc87d511a603c9c6ba8987cf3f0695897e0ddbf04c6b0042c603653145d7
```

只有root创建与最终config/source绑定、明确enabled的授权后，才可另用新授权真实SHA并增加 `--attempt ROOT_FRESH_ID --detach`。上面禁用模板不能执行。

## Git/大数组

若采用默认工作目录，应只忽略以下两个大型响应载荷模式，保留manifest、小型Hessian、模式、CSV、图及日志：

```gitignore
/reports/pyscf-spinosin-96h-continuation-20261105-v8/response-pipeline-001/science/response-*/stage-cache/MO1_atom_*.npy
/reports/pyscf-spinosin-96h-continuation-20261105-v8/response-pipeline-001/science/response-*/stage-cache/MO_E1_atom_*.npy
```

已完成旧H1约1.045GB继续仅本地，原精确ignore不变。代码不执行Git，不创建服务或外部自动任务。源/配置/授权/真实结果校验不通过即停止；短暂app Git冲突只在新阶段前有限等待，不因它取消活跃Q。219个有符号模式与全正极小点分开记录，负频保留，IR及独立CPHF残差仍NOT_COMPUTED。
