# 五折嵌套研究模型：按外折复现

本目录保存各外层684份源样品训练得到的模型，用于复现相应171份留出预测。**这些不是一个已经用全855份训练的部署模型。** 96.0234%来自五个R2模型各自留出预测拼接后的821／855，并非某一文件对全部训练样品的准确率。

- `core/`：48份去重完整管线及metadata，覆盖R0／R1／R2／R3／固定锚／7预处理诊断的60个方法折角色。按[core/manifest.json](core/manifest.json)中的`method_roles`及`outer_00`—`outer_04`找到实际文件；相同recipe共享文件，不当作额外模型。
- `combinations/`：35份外层完整管线（7家族×5折），另5份共享CNN `state_dict`及caller元数据。正式R4各折选择见对应选择锁，不能挑最终最好家族回填。CNN原头与SVM使用同一编码器，完整管线已包含预处理。
- [总清单](manifest.json)记录原运行路径、保存路径、SHA和大小。完整结果及来源见[四轮报告](../../docs/STAGED_NESTED_OPTIMIZATION_REPORT.md)。

输入为相同频率坐标和顺序下的612点吸收系数，样品编号和标签不进入数值特征。MC-UVE仅保留对应训练源选出的点；SG／MSC、StandardScaler和SVM都已在各管线内部源拟合，预测时不能重新用待测批次拟合统计量。

SVC使用标准`.predict`投票规则，不将OVR score argmax代替；分数不是校准概率。CNN头的原始输出为logits，其SVM分支仍为投票预测。网络32维表示不能称为32个原始频点，原输入仍为612点。

要复现R2，按每折自己的R1 MC选择和R2 SVM选择读取，R2五折都使用SNV；R3的SG一阶／MSC选择不能混进96.02%配方。对应源码为`src/herb_spectra/staged_nested_search.py`与`staged_nested_combinations.py`，依赖版本和源码hash已记录；属于Python适配实现，不是原MATLAB权重逐项复现。

R2的ZY02仍仅识别3／5，未解决稀少类问题。数据每产地只有一批，模型尚未经真正独立新批次确认。若后续生成全855份源训练的单一模型，应另在全源内层选择并拟合，明确它没有新增独立测试成绩，不直接将其称为“OOF96.02%的部署模型”。
