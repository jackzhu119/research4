# Gaussian与GaussView云端安装记录

检查与运行日期：2026-10-30。用户授权查找最新版本，无法取得时尝试上传包；随后确认目前暂无学校／课题组授权下载渠道，先处理上传包。

## 已完成的结果

**已在当前Debian 13云端完成用户上传GaussView 6.0.16的Wine64兼容部署，并实际打开水分子输入文件、显示三维结构。Gaussian计算引擎尚未安装，DFT优化及振动频率计算尚不可用。**

| 项目 | 实际状态 |
| --- | --- |
| GaussView界面 | 用户上传Windows x86-64程序，文件版本6.0.16.0，已启动 |
| 分子结构文件 | 实际打开`water.gjf`，界面显示3原子、10电子、中性、单重态 |
| 三维显示 | Xvfb虚拟显示与Mesa llvmpipe软件渲染通过，见下方截图 |
| Gaussian引擎 | 未在包内找到`g16/g09`或Windows对应引擎；程序明确弹出未安装提示 |
| Gaussian量子化学任务 | 执行数为0；没有生成优化结构、频率、能量或模拟光谱结果 |
| 图形桌面接入 | 当前运行于虚拟显示`:99`；未新增对外远程桌面服务 |

[实际三维显示截图](../reports/software/2026-10-30-gaussview/gaussview-water-success.png)与[缺少Gaussian引擎的提示](../reports/software/2026-10-30-gaussview/gaussian-engine-missing.png)分别保留，不能把界面运行当成计算引擎可用。

水分子是手工给定坐标的显示样例，不是Gaussian优化结果。相同输入保存在 [water.gjf](../examples/gaussian/water.gjf) 和 [water.xyz](../examples/gaussian/water.xyz)。GJF中的计算设置仅用于展示输入格式，未执行。

## 最新版本与下载渠道核验

| 软件 | 本轮能核验到的版本 | 来源及获取状态 |
| --- | --- | --- |
| Gaussian | Gaussian 16 Rev. C.02 | [官方平台表](https://gaussian.com/g16/g16_plat.pdf)；本轮未取得可公开下载的正式安装介质 |
| GaussView | 6.1.1 | [京都大学当前部署](https://web.kudpc.kyoto-u.ac.jp/manual/en/software/gaussian16)和[广岛大学站点说明](https://www.media.hiroshima-u.ac.jp/services/license/gauss/)列明此版本及Linux／Windows支持；其站点渠道有自身用户范围 |

以上是可核验版本，不是通过成功访问厂家全部发布记录建立的穷尽性最新版本证明。大学自己的下载／借用渠道不能当作面向所有用户的公共安装包来源。本轮没有从这些受限渠道取得软件，没有购置或配置新许可证。

[官方GaussView Windows安装说明](https://gaussian.com/g16/gv6win_install.pdf)要求Gaussian 16或Gaussian 16 Utilities，并说明安装信息及序列号。GaussView和Utilities都不能替代完整计算引擎。Gaussian C.02官方Linux平台表也未列Debian13，后续取得Linux介质时仍需实测兼容性。

## 上传包与部署方式

上传包来自已归档的[四个Desktop分卷](../references/paper-materials/archives/desktop/README.md)：

- 内层`GaussView 6.0.16 Win64安装包.rar`：53,383,453字节，SHA-256 `ebd1eeb421795b6051e17b17d638f77315a14b1a4311d2155a8a3f6119b0f382`，与原归档记录一致。
- 包含NSIS自解压EXE、安装文字和图片。EXE的SHA-256为`aca26014d666fbb659c5ab35d1b1ffc0639540b91d41883097238c184dead7e5`。
- 安装器stub是PE32，但实际`gview.exe`及附带Qt／VC运行库是PE32+ x86-64。解压得到1262个载荷文件，共96,343,800字节。
- 部署采用解压应用载荷、配置私有Wine64后启动的方式。原32位NSIS安装器未执行，未修改GaussView二进制，未添加虚构的Gaussian引擎或授权信息。
- Wine及依赖仅下载、解压到项目`.runtime/`，不修改系统安装、不重设`HOME`。该目录被Git忽略；本次提交仅保存启动代码、来源、检查记录和截图。

主要路径：

| 内容 | 当前路径 |
| --- | --- |
| 应用 | `/workspace/research4/.runtime/gaussian-app/GaussView6/gview.exe` |
| Wine64工具 | `/workspace/research4/.runtime/gaussview-tools/root/` |
| 独立Wine配置 | `/workspace/research4/.runtime/gaussview-wine/` |
| 终端启动入口 | `/workspace/research4/.runtime/bin/gaussview` |
| 桌面快捷方式 | `/home/agent/Desktop/GaussView.desktop` |

## 如何启动与检查

有交互图形桌面时，可使用桌面快捷方式，或运行：

```bash
.runtime/bin/gaussview examples/gaussian/water.gjf
```

当前无外部交互显示时，可启动至私有虚拟显示：

```bash
python scripts/launch_gaussview.py --headless examples/gaussian/water.gjf
```

虚拟显示运行不代表浏览器中自动获得远程桌面。没有交互桌面时，启动过程中缺少Gaussian的提示仍需要界面操作；本轮实测通过确认两个缺少引擎的提示后进入主界面。没有修改依赖检查或伪造引擎路径。

只检查部署路径、不启动界面：

```bash
python scripts/launch_gaussview.py --status
```

[启动脚本](../scripts/launch_gaussview.py)使用已经配置的私有运行环境，不会自动下载新的商业软件。新克隆仓库不包含`.runtime`二进制，不能仅凭该脚本视为已安装软件。

## 实际解决的兼容问题

私有Wine部署额外配置了以下路径与依赖，均限于项目目录：

1. `wineserver`指向私有`wineserver64`，处理Debian包装器的系统绝对路径。
2. 在私有`usr/share/wine/`下设置`wine -> .`，适配重定位后的NLS数据查找。
3. PE查找路径包含`wine/x86_64-windows`，并显式设置私有`WINELOADER`及Unix库路径。
4. 将Debian包里的64位`zlib1.dll`复制到私有Wine的`system32`；仅放在包目录不足以支持应用启动。
5. 初次配置失败的配置目录保留为`.incomplete-first-boot`，在依赖修复后重新初始化当前私有配置，`wineboot`退出码0。

Wineboot仍有未配置蓝牙、32位辅助程序和菜单生成器等日志，本轮确认它们未阻止当前64位GaussView读入并显示水分子。完整功能、长时间稳定性、其他文件格式和Gaussian任务提交未通过此单例证明。

可审计证据见 [installation_status.json](../reports/software/2026-10-30-gaussview/installation_status.json)、[官方下载与版本来源](../reports/software/2026-10-30-gaussview/official_sources.json)、[应用载荷哈希](../reports/software/2026-10-30-gaussview/uploaded_member_hashes.json)、[兼容依赖包](../reports/software/2026-10-30-gaussview/compatibility_packages.json)。

要开展老师建议的分子结构优化和振动频率计算，仍需可在该环境使用的完整Gaussian引擎及授权安装介质；也可以另行选择适合研究问题的开源量子化学工具。本轮只完成上传GaussView的兼容运行验证。
