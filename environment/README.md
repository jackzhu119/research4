# 研究运行环境

本目录保存本次云端验证过的版本约束。运行时安装在仓库根目录 `.runtime/`，不提交解释器、模型框架或系统库二进制文件。

## 创建或复用

在仓库根目录执行：

```bash
python scripts/setup_research_environment.py
```

程序优先复用已经验证版本的 Octave；创建带已检查系统包访问权限的独立 Python venv，把仓库 `src/` 加入该 venv 的模块路径。缺少的固定版本 Python 包只安装到该 venv；CPU PyTorch 使用官方 CPU wheel 源。已有的本次云端 CPU PyTorch 目录可以复用，并记录路径。

没有可复用 Octave 时，在 **Debian 13/trixie x86_64** 上从 Debian 官方源下载固定版本及其依赖，使用 `dpkg-deb -x` 解压到私有前缀，注册 Octave 包并配置无界面绘图。不会执行全局系统安装、使用sudo、修改HOME或替换网络代理。其他系统应提供已有的兼容 Octave，不能混用 Debian 二进制包。

本次已有完整缓存，也实际验证了：

```bash
python scripts/setup_research_environment.py --offline
python scripts/setup_research_environment.py --runtime-dir outputs/bootstrap-rebuild --offline --rebuild-octave
```

第一条复用当前环境；第二条在另一个目录从缓存重新解包 Octave。离线模式不会访问网络，缺少固定版本时明确失败。`--check-only` 只执行环境检查；`--no-torch` 可建立不含CNN框架的基础环境，但医生报告会分别标注CNN状态。

固定版本在 `runtime-spec.json`、`requirements-core.lock.txt`、`requirements-torch.lock.txt` 中。安装后的实际路径、模式、规格校验值和解包文件校验值记录在 `.runtime/manifest.json`。这是针对已测试平台的组件版本约束，不是完整操作系统镜像或未来包仓库可用性的保证；固定版本不可取得时不会静默改装新版本。

阶段二的小麦RDA源文件解析另使用可选的 `requirements-rda.lock.txt`（rdata1.1.0、xarray2026.9.0），实际验证安装在outputs私有前缀，模型运行不依赖它。复现命令见 [阶段二报告](../docs/RESEARCH_STAGE2_REPORT.md)。

## 稳定入口与检查

```bash
.runtime/bin/python -c "import herb_spectra, torch; print(torch.__version__)"
.runtime/bin/octave --eval "pkg load statistics signal io image optim; disp(version);"
python scripts/environment_doctor.py --runtime-dir .runtime --output outputs/environment-doctor.json
```

`open_source_ready` 表示固定 Python 版本包可导入、CPU CNN 框架可用、libarchive可加载、Octave及所需包版本匹配。它与 `all_original_code_ready` 分开；原程序缺少数据、缺少函数或者存在逻辑错误，不会因环境安装而自动修复。

Python venv有意复用符合固定版本的基础镜像包及已有CPU框架；搬走其依赖目录后应重新执行 setup。需要部署到另一台机器时，使用同一规格重新创建，不要只复制几字节的启动包装器。

## MATLAB 原生运行

本安装器不安装、激活或仿造 MATLAB 许可证。MATLAB及专有工具箱需要真实的个人或学校授权，Octave成功不能记为MATLAB原生通过。

如果以后已经具备合法安装，可明确提供可执行文件并启动一次探测：

```bash
python scripts/environment_doctor.py --matlab /path/to/matlab --probe-matlab --output outputs/matlab-doctor.json
```

仅发现可执行文件时 `native_execution_verified` 仍为false；只有显式原生启动成功后才会标注已验证。医生不会输出许可证值、密码或完整进程环境。
