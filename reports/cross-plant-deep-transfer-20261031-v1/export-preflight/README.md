# 酸枣仁外部预训练：报告导出预检

本阶段仅实现报告与合成夹具测试，尚未读取真实实验成绩／source选择／预测，未拟合或调用benchmark评分。

自编入口为`scripts/export_cross_plant_transfer_20261031.py`。9项合成测试实际通过，命令、日志及最终源码摘要见[validation.json](validation.json)。包含完整合成锁链到15表XLSX的往返；15个模型／种子病例分母均855，300份metadata／75组配对状态检验。合成数据绝不计入科研样品或正式性能。

独立只读代码复核发现的两个报告问题已修正：未绑定的fold指标从已评分预测重新计算再比对，stored NaN显式拒绝。模型、trace和schedule只验证字节摘要，不反序列化或训练。

真实导出必须等根代理提供绑定完成评分清单的放行文件：

```bash
.runtime/bin/python scripts/export_cross_plant_transfer_20261031.py \
  --run <冻结主实验目录> \
  --output reports/cross-plant-deep-transfer-20261031-v1/comparison \
  --release <root_aggregate_export_release.json>
```

放行至少包含`allow_aggregate_export:true`、`created_at_utc`、`scoring_manifest_sha256`；可记录`independent_evaluation_review_sha256`作为根代理提供的独立审查摘要引用。导出前逐项验证score/prediction/source/上游放行锁，然后读取真值已连接的预测，保存所有种子、负效应、校准前后错误转移、各组与A／Aextra的配对差。

历史96.0234%仅列上下文；本轮共同128点表示／ResNet／相同目标预算的A和Aextra才是迁移归因基线。SD不是生物重复或置信区间，未知单位／仪器及既有数据曝光边界不因导出改变。
