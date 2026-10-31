# 人参冻结证据无损归档

本目录交付本轮CC BY 4.0人参任务的模型、变换器和实际fit事件日志，原文件保留本地不变，由上级`.gitignore`排除重复提交。

- 24,625个文件，原始437,320,263字节；11个归档共253,301,533字节。
- 最大单文件35,554,528字节，小于50 MiB及GitHub单文件限制。
- `archive_manifest.json`记录每个原路径、字节数、SHA256、ZIP成员或gzip分片，以及每个归档的字节数和SHA256。
- 原始事件日志162,790,740字节，分成四个gzip片段，恢复后逐字节一致。
- 归档范围：`selection_models/`、`evaluation_models/`、`transform_cache/`及`fit_scope_events.jsonl`。数据输入、协议、配置锁和指标仍按原路径直接交付。

从仓库根目录先验证、再恢复：

```bash
python scripts/archive_public_thz_evidence_20261031.py verify --archive-dir reports/public-thz-20261031-v1/ginseng-benchmark/artifact-archives-v1
python scripts/archive_public_thz_evidence_20261031.py restore --archive-dir reports/public-thz-20261031-v1/ginseng-benchmark/artifact-archives-v1
```

已有完全相同的文件会跳过；存在不同文件则拒绝覆盖。恢复过程可能需要数分钟。manifest自身的可信摘要由上级交付索引绑定，工具不提供数字签名或恶意并发环境下的OS沙箱。

真实新建临时目录恢复已完成：全部24,625文件字节验证、幂等恢复、冻结`verify_run`及`verify_selected`通过，11,160个源模型与306个外层模型及变换器摘要均复核。原配置锁、评分锁和日志前后不变；没有新增拟合或评分。实际命令及[结果记录](../../delivery-preflight/archive_execution.json)保留。

水果／叶片许可未核实的原件、衍生数值及模型不在本归档中。
