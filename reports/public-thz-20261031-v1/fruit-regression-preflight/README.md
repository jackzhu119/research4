# 水果分组回归：拟合前准备与冻结

状态：**PREPARED_NOT_TRAINED**。已生成80个固定上下文（20外层、60内层），实际水果模型拟合及预测均为0；尚无源训练或外层评价release。15项边界测试通过，测试中的小型随机数夹具不是植物实验。

## 范围与预算

输入为来源核验后的本地NPZ，SHA256为b3c5d8b6272038377cfdfe272a6d084895b9817325fdd9ec34bab6c9db1ec27b。MA-Brix、PM-Brix、PR-Brix和MA-DM分别建模，每任务10个持续果实ID、40条记录。MA两个任务共用果实和波形，不算额外独立样品。

每个果实的4个固定日期始终同组。外层GroupKFold5：8果实／32记录训练，2果实／8记录预测；内层GroupKFold3仅在当前外层训练的8果实中划分。4个日期同时覆盖训练与预测，因此这只是同一时间安排下对留出果实的探索，不能称跨时间、跨批次或新季节确认。

输入仅用7000点波形按事先固定的128个连续索引块取均值。只有工程块索引，时间单位与THz频轴未核；没有FFT、假频率轴、吸收系数／折射率转换或Background／Metal参考归一化。Brix和DM只称Excel报告的数值读数，单位及测定方法未确认。

R1固定34个候选：PLSR6、全特征SVR24、MLP4。R2固定8种MC-UVE参数，继承当前外层源内部选出的SVR预处理／C／gamma／epsilon。所有候选执行42、43、44种子；共上限7560个内层估计器拟合，合法source缓存后预计10800次MC内部PLS调用。含720个300轮MLP拟合，实际水果运行耗时尚未测量。R2仅保留32／64个工程块，不将这些块称为已经确认的物理吸收峰。

每候选先将3内折的32条OOF预测按种子汇总，计算该种子的RMSE与MAE，再按3种子平均RMSE、MAE、候选ID选取。外层固定报告mean dummy、PLSR、全SVR、MLP、MC-SVR及source-policy；policy只是内层所选模型的别名，无外层选优。外层最多300个实际估计器拟合、360个role/fold/seed情况，2880条预测；最终汇总72个task/role/seed情况，每个40条预测／10果实。

## 可执行边界

X和y的StandardScaler、MC选择及估计器都使用当前允许的源记录；MC的每次内部PLS有FitScope记录。MC采样的是源内相关时间点记录，不宣称这些时间点是独立果实。严格数值输入schema拒绝真值、日期、果实类型等附加字段，元数据不能从接口直接拼入X。

四阶段CLI为prepare、selection、evaluation、score。selection必须持root签发且绑定本次代码、协议、准备和manifest哈希的源release；全部20组选择锁定后，evaluation还需要单独root release。query推理进行单条／反序／置乱一致性与模型状态检查；所有预测落盘并锁定后，score才连接真值。所有真值文件的字节校验在任何相应语义解析前完成。此为API与证据边界，非操作系统权限沙箱，非“证明绝对无泄漏”。

## 重现与许可

完整运行目录固定在忽略的outputs/public-thz-fruit-regression-20261031-v1；代码强制数据、变换器、模型、trace、逐条预测留在outputs下。原始仓库未发现明确许可；不把原始波形、目标值、逐条派生数据或模型权重推送Git。此公开目录仅保存我们的方法、验证摘要、源文件哈希和预注册候选。

```bash
PYTHONPATH=src .runtime/bin/python scripts/run_public_fruit_regression.py prepare --output outputs/public-thz-fruit-regression-20261031-v1
PYTHONPATH=src .runtime/bin/python scripts/run_public_fruit_regression.py selection --output outputs/public-thz-fruit-regression-20261031-v1 --release ROOT_SOURCE_RELEASE.json
PYTHONPATH=src .runtime/bin/python scripts/run_public_fruit_regression.py evaluation --output outputs/public-thz-fruit-regression-20261031-v1 --release ROOT_OUTER_RELEASE.json
PYTHONPATH=src .runtime/bin/python scripts/run_public_fruit_regression.py score --output outputs/public-thz-fruit-regression-20261031-v1
```

prepare已执行，不可在同一目录再执行或覆盖。未获release的命令拒绝拟合。源/外层失败均应保留，不根据外层成绩改候选、删果实或挑种子。水果结果不能替代酸枣仁主线，也不是药材THz-TDS迁移证据。

日志保留两个成功测试版本（14项后补充全outer真值检查为15项）、实际prepare命令及时间。更早一次合成对抗夹具由于局部类不可pickle报错，移至模块级夹具后修复；该错误没有涉及水果模型训练。validation.json记录此历史，已有酸枣仁与冻结人参数据和模型均未修改。
