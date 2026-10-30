# Desktop RAR 分卷归档

2026-10-30 补充上传的四个原始分卷已全部保存，保持原文件名和字节内容。整套分卷共 88,232,260 字节，完整读取及两个成员的 CRC-32 校验均通过。

## 使用方法

将下面四个分卷全部下载到同一文件夹，保留原文件名，使用支持 RAR 分卷的软件打开 `Desktop.part1.rar` 并解压。

| 原始分卷 | 字节数 | SHA-256 |
| --- | ---: | --- |
| [Desktop.part1.rar](Desktop.part1.rar) | 29,360,128 | `d70c182ea5e035daaed6b3860b5d2d74f78034f7a250f72451e1b2f1c121ae42` |
| [Desktop.part2.rar](Desktop.part2.rar) | 29,360,128 | `b2cb7c16c3fab3ff6e6fa35089c5fdda960c501ae7f98d487fd3612f51397fc0` |
| [Desktop.part3.rar](Desktop.part3.rar) | 29,360,128 | `1e62c8435621424b245ddcd76d567fe6561419bffb8fdad68371e6594a06fae0` |
| [Desktop.part4.rar](Desktop.part4.rar) | 151,876 | `9f8790e00559598051a346d625406f0e723aae1afa172495de81636a7fe89230` |

## 内含文件

以下是从整套四卷读取并按成员 CRC-32 核验的文件，不是四个分卷自身的校验值。

| 内含文件 | 字节数 | CRC-32 | SHA-256 |
| --- | ---: | --- | --- |
| 0308 CARS+UVE.rar | 34,847,824 | `bd234082` | `2762a1f8eaf845e083474a2d77b574564c0a6173aaffaa2a36d64d0a6f4e11a1` |
| GaussView 6.0.16 Win64安装包.rar | 53,383,453 | `c7c8b607` | `ebd1eeb421795b6051e17b17d638f77315a14b1a4311d2155a8a3f6119b0f382` |

`0308 CARS+UVE.rar` 已另存至 [代码目录中的原包](../../code/0308%20CARS%2BUVE.rar)，补齐此前缺失的同名资料。其来源是本次 Desktop 分卷，首批超限附件未直接下载。

`GaussView 6.0.16 Win64安装包.rar` 保留在四卷中，可按上述方法解压取得。本次只读取压缩档案并核验成员完整性，未运行代码或安装程序，也未进一步解压两个内层 RAR 包。

全库文件清单见 [manifest.csv](../../manifest.csv)。
