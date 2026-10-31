# 跨植物预训练：交付预检

本目录只做存储、历史保护及无损归档验证，不训练、选择或评分科研模型。实际归档必须等待全部实证／主实验评分及独立完整重放结束，并由root另行放行；当前归档状态为`NOT_RUN`。

初查见[initial_state_review.json](initial_state_review.json)：直接核对前次提交`3dc8b911`的5,953个Git blob，共2,812,408,650字节。既有数据、模型、历史报告、代码和测试均不变；仅获准的AGENTS和曝光登记文档更新。初查时剩余磁盘约10.7 GB，不构成当前阻塞。

计划只归档以下三处冻结证据目录，不移动或删除原件：

- `neural/source_models/`
- `neural/outer_models/`
- `empirical/prepared/fits/`

这些目录包含checkpoint、标准化器、实际fit范围、优化器批次轨迹／日程和预测锁等。酸枣仁、黄芪、天麻依据用户授权，人参依据CC BY 4.0；水果、叶片、小麦和未下载原件不进入本轮，也不进入本归档资格范围。

新增[归档入口](../../../scripts/archive_cross_plant_transfer_evidence_20261031.py)复用旧无损格式并校验旧helper固定SHA，旧helper本身保持不变。40 MiB未压缩分块、单件归档最多50 MiB，每个源文件／归档／gzip分片均记录字节数和SHA。恢复前拒绝所有已有不同文件、路径穿越、符号链接或范围变化，使用临时文件及独占链接发布，不覆盖原件。

本轮批量恢复为每个ZIP保持一个读取句柄，避免逐文件重复解析目录。六项真实合成夹具测试通过，见[archive_tests.log](archive_tests.log)与[helper_validation.json](helper_validation.json)：分块ZIP／gzip逐字节往返、幂等、原件不变、每归档一次读取、未完成目录／越界来源拒绝、目标冲突、解码摘要错误及旧依赖篡改拒绝。没有消耗科研拟合或评分预算。

归档后将新建临时目录完整恢复，校验全部训练摘要和冻结验证入口，并再次核对历史保护。仅在验证成功后新增Git忽略规则，Git交付压缩证据而本地原件保留。manifest自身须由上级交付索引绑定可信SHA；本工具不提供数字签名，也不是恶意并发文件系统修改的OS沙箱。
