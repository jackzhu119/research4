# 真实评分后的导出与全部单元格验证

根放行`root_export_release.json`绑定已完成评分清单及独立outer审查摘要。自编导出器源码保持预检时SHA不变；未训练、未调用benchmark score、未改模型／source／原指标。

实际命令及退出记录见`command_events.jsonl`，最初15表XLSX导出见`export.log`。报告追加覆盖解释和C−B整套方案对照：原自动生成README另存`initial_generated_README.md`；`validate_exports.py`记录新增说明、最初报告／索引摘要及原始评分清单未变。C−B同时改变人参剂量120→40并引入另外两任务，不宣称隔离药材单独作用。

`verify_final_exports.py`采用完整长度逐项比对，最后实际核验17张CSV表与15＋2表XLSX的549,505个单元格，均一致；5组PNG／PDF均解码／格式通过。图片已检查模型指标、负配对变化、15格混淆及真实学习曲线。最终[全部验证](../comparison/final_validation.json)和[报告产物索引](../comparison/files_sha256.json)保持明确范围。

结果：A87.21%、Aextra87.37%、B86.43%、C85.65%、保守source-policy87.95%。B／C的平均准确率、Macro-F1及BA均低于A和Aextra。source-policy五折Aextra、Aextra、A、A、A，没有使用外部预训练。所有种子与负结果保留，未把87.95%的提高归因外部数据。

原件可用源池与梯度实际覆盖按根代理提供的`pretraining-coverage/summary.json`解释；所有重复呈现不增加独立N。历史96.0234%仅作开发成绩背景，不是本次迁移收益基线。
