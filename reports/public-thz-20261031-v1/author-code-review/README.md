# 公开人参参考软件只读审查

来源：[原作者GitHub仓库](https://github.com/jamilmai/Analysis-and-ANN-for-TCM-DL)。本次实际下载归档10,951字节，SHA与全部五个文件摘要见[review.json](review.json)。归档没有数值数据或LICENSE文件，原件仅保存在本地忽略目录；未执行第三方代码，不把它计为新数据集。

实际代码存在以下问题，不能原样作为本研究的可靠对照：

- `main_classification.py:153`在验证循环把`val_images`再次赋成`torch.squeeze(train_images).cuda()`，覆盖此前取得的验证输入。
- `D2NN_classification.py:41–54`的隐藏ReLU及输出Sigmoid均被注释，实际forward是仿射层复合，不能仅根据文件名称其为有效非线性深度模型。
- `main_classification.py:39`对两个数据文件逐元素相加，但软件归档没有提供量纲与数据配对证据；不能直接照搬到α／n融合。

本次用独立审查的source-only实现对照SVM、MC-UVE、LDA、PCA-LDA、PLS-DA及MLP。以上结论只针对已下载代码字节，不证明已发表论文使用了相同版本，也不据此推断论文结果无效。
