# 云端可用性测试复现说明

测试结论见 [实测报告](RUNTIME_USABILITY_REPORT.md)。建议从仓库根目录执行，给每轮测试使用新的输出目录，避免混入旧文件。

## 依赖

本次使用 Debian 13 的 Octave 9.4.0，以及 `octave-statistics octave-signal octave-io octave-image octave-optim gnuplot-nox fonts-freefont-otf`。还需要系统 `libarchive` 读取 RAR。常规已安装环境中，`/usr/bin/octave-cli` 可直接作为执行器。

本次云端没有管理员权限，将 Debian 官方包下载后解压到工作目录，配置 `OCTAVE_HOME`、动态库、字体和 gnuplot 路径，再执行 `pkg rebuild -global` 注册搬移后的包。所用执行器为 `/workspace/scratch/cloud-runtime/octave`。这只是当前会话中的路径，不保证云端实例重建后仍在。

Python 基础依赖见仓库 `pyproject.toml`。CNN 迁移另外使用 PyTorch `2.9.0+cpu`；本次放在隔离目录，通过 `PYTHONPATH` 选择，没有把大体积运行环境提交到 Git。

## 准备来源

```bash
python scripts/runtime/prepare_sources.py --output outputs/runtime-sources
```

该命令保留上传成员字节内容及 SHA-256，把 TXT 按字节复制成 `.m`，把两个 DOCX 的段落文本提取成 `.m`；不运行安装程序或 Windows MEX。原压缩包不会被修改。

## 运行分层测试

以下使用常规系统 Octave 路径；在本次隔离环境中将其替换为上面的执行器路径：

```bash
python scripts/runtime/run_octave_checks.py --phase first --octave /usr/bin/octave-cli --inventory outputs/runtime-sources/source_inventory.json --output outputs/runtime/first-pass
python scripts/runtime/run_octave_checks.py --phase compatibility --octave /usr/bin/octave-cli --inventory outputs/runtime-sources/source_inventory.json --output outputs/runtime/second-pass
python scripts/runtime/run_octave_checks.py --phase focused --octave /usr/bin/octave-cli --inventory outputs/runtime-sources/source_inventory.json --output outputs/runtime/focused
python scripts/runtime/run_octave_checks.py --phase vectorized --octave /usr/bin/octave-cli --inventory outputs/runtime-sources/source_inventory.json --output outputs/runtime/vectorized
python scripts/runtime/run_octave_checks.py --phase final --octave /usr/bin/octave-cli --inventory outputs/runtime-sources/source_inventory.json --output outputs/runtime/final-pass
python scripts/runtime/run_octave_checks.py --phase ssa-full --octave /usr/bin/octave-cli --inventory outputs/runtime-sources/source_inventory.json --output outputs/runtime/ssa-full
python scripts/runtime/run_octave_checks.py --phase targeted --octave /usr/bin/octave-cli --inventory outputs/runtime-sources/source_inventory.json --output outputs/runtime/targeted
python scripts/runtime/run_octave_checks.py --phase targeted-corrected --octave /usr/bin/octave-cli --inventory outputs/runtime-sources/source_inventory.json --output outputs/runtime/targeted-corrected
python scripts/runtime/run_octave_checks.py --phase texture-array --octave /usr/bin/octave-cli --inventory outputs/runtime-sources/source_inventory.json --output outputs/runtime/texture-array
```

`first` 不修改上传代码；`compatibility` 包含机械适配和明确标注的缺陷复现/逻辑补丁；`focused` 重现主要入口；`vectorized` 检查 BP 梯度与向量化；`final` 使用修正后的绘图适配和 MAT 输入夹具；`ssa-full` 给原默认100轮另设600秒预算；`targeted` 定向检查纹理条件、PCA中心化及提供的newff依赖。早期适配失败记录保留在本次证据中；重跑使用当前脚本时，这些已经修正的执行器问题不一定再次出现。

每项脚本单独限时并记录结果。超时表示该环境、线程和预算下没有完成，不自动表示算法错误；IRIV 的特定低维循环另有可证明的边界缺陷。脚本执行时会创建工作副本、图和 MAT 中间结果；这些是临时试验输出。

## Python 与证据整理

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/runtime/run_python_checks.py --inventory outputs/runtime-sources/source_inventory.json --output outputs/runtime/python
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/runtime/run_python_checks.py --only python_cnn_structure_migration --inventory outputs/runtime-sources/source_inventory.json --output outputs/runtime/python-cnn-corrected
```

所有来源谱列、示例 y 都保持其原含义；合成数据只用于工程夹具，不生成中药参考真值。`--only` 可在某个实现改变后只重跑相应场景。

本次证据使用 `collect_run_evidence.py` 整理，另保留了环境探测和示例重复谱的精确匹配记录。`source_hashes.json` 记录来源，`cases.json` 保留全部尝试，`numerical_checks.json` 汇总跨语言和补丁验证。脚本中有若干面向本次场景的固定目录名，扩展新场景时需同步更新收集列表。

运行全部场景后可整理到新的输出目录：

```bash
python scripts/runtime/collect_run_evidence.py --run-root outputs/runtime --inventory outputs/runtime-sources/source_inventory.json --output outputs/runtime-report
```

超过1MiB的日志同时保存首尾摘录和完整gzip，原始日志字节数、SHA-256写入场景记录；使用 `gzip -dc 文件.log.gz` 可还原完整日志。

本次所有耗时都是单次墙钟时间，含解释器启动/数据读取/绘图等开销；没有将其当作不同机器之间的标准性能基准。
