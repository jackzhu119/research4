# 交付与无损归档预检

此预检不训练、不选择或评分模型，不修改冻结协议与历史产物。

`scripts/archive_public_thz_evidence_20261031.py`只对本轮CC BY 4.0人参实验提供CLI。默认归档`selection_models/`、`transform_cache/`与原始fit事件日志；40 MiB未压缩分块，单个归档不超过50 MiB。原件本地保持不动。每个源路径和归档均绑定字节数及SHA256；大日志分段gzip可逐字节还原。

恢复只写不存在的文件；已存在且完全相同则跳过，任何已有不同文件、路径穿越、符号链接、异常成员、归档或解码hash差异均拒绝。manifest的可信hash需要由上级交付索引绑定，本工具不提供数字签名或对恶意管理员的OS沙箱。

八项实际测试见[archive_tests_final.log](archive_tests_final.log)：多ZIP／分段gzip往返、幂等、原件不变、篡改、已有文件或父目录冲突、路径穿越、符号链接、额外ZIP成员和解码错误。测试使用合成夹具，不消耗科研模型拟合预算。早先七项测试日志及当时源码hash快照仍保留。

实际归档须在评分和独立模型重放均完成、root明确放行后执行。归档后将同时验证原路径不变、独立临时还原的每个路径字节一致，并实际调用冻结run的验证入口。届时再添加忽略规则，确保原件保留本地而Git交付压缩证据。未知许可的水果及叶片原件、派生数组、模型与逐条预测不进入此归档。

## 实际归档与历史保护检查

root在评分和独立完整重放后放行，真实归档及新建临时目录恢复已完成，见[archive_execution.json](archive_execution.json)。24,625文件、原437,320,263字节压缩为11个归档／253,301,533字节，最大35,554,528字节。全部文件SHA、幂等恢复、`verify_run`、`verify_selected`及306个外层模型摘要通过；原件、配置锁及日志保持不变，新增拟合／评分均为0。[恢复入口与说明](../ginseng-benchmark/artifact-archives-v1/README.md)。

[历史字节审计](protected_base_audit.json)直接按Git blob算法核对前次提交`ea7b6b7`的5,454个文件，共2,512,790,772字节：5,451个完全不变，只有3份获准更新的README／AGENTS／docs发生改变。既有数据、模型、历史报告、源代码、脚本和测试均没有改动。本预检不stage、不commit、不push、不合并main。
