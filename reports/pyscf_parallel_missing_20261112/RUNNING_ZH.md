# 本轮后台运行入口

- 共享新增授权：900000 CPU 秒。
- 斯皮诺素：严格几何修复与完整频率计算，已启动。
- 毛蕊异黄酮 IR：严格驻点前驱导数，已启动。
- 斯皮诺素 IR：进程已启动，等待本轮驻点后自动计算。
- 无评分、失败次数、几何轮数、无进展或墙钟时长退出条件；科学条件不放宽，授权 CPU 是唯一资源边界。
- 每600秒自动回传；状态见 background_state.json 与 workers/，检查命令见 docs/PYSCF_PARALLEL_MISSING_20261112.md。
- 草稿审阅入口：https://github.com/jackzhu119/research4/pull/15

当前是启动进度，不是完成或验收合格结论。
