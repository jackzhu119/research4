# 1,635项已完成模型实验的无损结果归档

本目录是已完成产物的归档，不是新的训练，不更改数据、标签、模型或指标。
论文表格应优先阅读上一层的`tables/`、`sources/`及中文总索引；本目录供逐项复核和恢复。

## 覆盖范围

| 组成 | 覆盖 | 原始字节 | 归档字节 |
|---|---|---:|---:|
| `manifest.json`＋`scientific-results.tar.gz.part001…003` | 旧1,305项＋新330项，共91,529个文件；所有可用小型科研结果和监督回执，含新V4全部预测 | 981,501,071 | 74,635,537 |
| `legacy-prediction-completion/` | 旧预测中此前未在固定Git备份覆盖的591项 | 777,497,989 | 778,032,438 |
| 已有Git原路径或先前无损归档 | 旧预测余下714项 | 878,267,308 | 保留在下述固定提交，不重复上传 |

每个分卷不超过24MiB。分卷是同一连续tar.gz流的字节切片，**不能分别解压**。
第一项使用SHA-256内容寻址去重，91,529个原文件对应87,149个独立字节对象。
它保留所有任务的`summary.json`、`metrics_by_setting.json`、内层搜索、候选/选定特征、
支持抽样、真实标签、协议、完成封存及可用监督回执；含V4的CSV与gzip预测。
旧预测591项则由第二项补齐，字节完全不变。

旧预测已备份714项的确切来源逐文件记录在：

`legacy-prediction-completion/all-1305-prediction-coverage.json.gz`

核对的是实际文件字节与固定Git blob SHA-1，或先前无损归档清单的SHA-256；
不是仅凭文件名、目录存在或不完整的旧状态判断。固定来源为：

- [`8a388b8b4d8a5baf9f2e4321e76455c2c0b49eca`](https://github.com/jackzhu119/research4/tree/8a388b8b4d8a5baf9f2e4321e76455c2c0b49eca)：原路径直接保存524项旧预测。
- [`71a303e089c7c7b7b2557d62372aaceda44e2f91`](https://github.com/jackzhu119/research4/tree/71a303e089c7c7b7b2557d62372aaceda44e2f91)：直接保存400项旧预测，均与上述或归档交集；不重复计数。
- [`f38113e51072f0a99c864b2e6dae79a9169ff93c`](https://github.com/jackzhu119/research4/tree/f38113e51072f0a99c864b2e6dae79a9169ff93c)：此前`00575`截止的无损归档，额外覆盖190项旧预测；使用该提交的`manifest.json`、`restore_backfill.py`及`archives/`。

新增第二项591项与已有714项联合覆盖**全部1,305项旧预测**。最终远端是否完整应以本次根任务的
Git发布核验记录为准；只有本地存在分卷不证明已经上传。

本归档有意不复制模型权重、大型NPZ/NPY/CHK/HDF缓存、临时锁和临时日志；权重和恢复依据仍由
原分支或先前模型回传归档索引追溯。1635计数是这两轮已完成科学任务数，不包括历史所有训练
和失败尝试；历史实验、V3真实失败及量子化学的状态另见论文实验总台账，不能混称全部成功。

## 校验与恢复（仅Python标准库）

在仓库根目录执行：

```bash
python reports/paper-evidence-20261109/archives/restore_archive.py --verify-only
python reports/paper-evidence-20261109/archives/restore_archive.py \
  --manifest reports/paper-evidence-20261109/archives/legacy-prediction-completion/manifest.json \
  --verify-only
```

脚本逐个验证分卷SHA、完整流SHA、tar普通文件类型、路径安全、每个对象的SHA与长度，
并读至gzip尾部验证CRC。本次构建的全流验证记录已保存到各自的`full_stream_verification.json`；
日后运行该脚本时，核验结果输出到终端。
没有验证失败后自动重训、替换结果或覆盖结果的逻辑。

恢复到两个**新的空目录**，例如：

```bash
python reports/paper-evidence-20261109/archives/restore_archive.py --output /tmp/research4-small-results
python reports/paper-evidence-20261109/archives/restore_archive.py \
  --manifest reports/paper-evidence-20261109/archives/legacy-prediction-completion/manifest.json \
  --output /tmp/research4-newly-backed-predictions
```

恢复路径分别为`legacy1305/runs/<job>/…`和`v4_330/runs/<job>/…`。
这两份归档含不同文件，不要求将恢复目录指向原科研目录；脚本拒绝写入非空目录。
若需恢复全部旧逐样品预测，还需按coverage表取得已有Git524项与先前归档额外190项。
索引中的`source_path`只是原云端来源记录，不要求你的电脑具有同一绝对路径。

第一份`file-index.json.gz`的`excluded_legacy_predictions`列出全部旧大预测字节，因为其由
第二份和先前固定提交保存。这里的“excluded”指**未放在第一份包中**，不等于未做实验或
未保存。第一份归档的`metrics_by_setting.json`也包含校准指标，但旧1080NN最终compact
报告仍只统计无校准主预测；不能把归档完整误当作已新增一份统一校准效益统计分析。

`build_archive.py`和`build_legacy_prediction_completion.py`保留本次导出的操作逻辑。
它们是一次性来源导出脚本，依赖原工作区已完成产物，拒绝覆盖现有归档。
普通论文写作、校验和恢复只需要`restore_archive.py`，不需要安装机器学习软件或运行构建脚本。
