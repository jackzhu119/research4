# 新Excel入口的真实历史留出重放

7份原编号工作簿均通过新label-free I/O读取，2578点明确裁为冻结612点，仅调用该编号原留出折模型。855行与旧保存预测和折编号逐字段完全一致：821正确、Accuracy96.0234%、Macro-F1 0.9197、Balanced Accuracy89.9184%。零新拟合，这仍是旧历史开发成绩，不能称本次新增验证。

`random10_examples.csv`以固定seed20261031预定抽10行，无正确性过滤；`all34_historical_errors.csv`保存全部34错误。真标签在预测CSV写出并锁hash后由单独评分进程连接。

```bash
.runtime/bin/python scripts/predict_zaoren_historical_oof.py --input data/processed/zaoren_20261031/样本2_编号版.xlsx --output outputs/historical-sample2-new
```

全855部署模型的训练诊断与本历史留出重放严格区分。新样本使用`run_zaoren_deployment.py predict`，不能将已训练记录改名冒充新样本。
