# V6b staged Hessian 独立后处理器：准备记录

本文件只记录新后处理器的代码与纯标准库验证。**本轮 V6b 量子计算、V6b 水／乙醇数值对照、Spinosin 频率分析均为 NOT_RUN。** 当前运行的原量子任务、旧 V5 a2 分析器、原 group 分析器和历史结果均保持不变；不能把此代码准备当作频率计算完成。

新增脚本为 `scripts/analyze_pyscf_staged_modes_20261105_v6b.py`，测试为 `tests/test_analyze_pyscf_staged_modes_v6b.py`。冻结的旧 a2 明确拒绝 V6b；新脚本具有自己的 V6b 来源门禁与产物索引。只有通过这些门禁后才加载 a2 中经过既有水／乙醇验证的纯数值辅助函数，不绕过或改写旧 a2 的生产者验证。

## 执行顺序与证据门禁

1. 取得当前已有真实量子全局锁 `.runtime/pyscf-v4-global.lock` 的排他锁。命令行和公开 `analyze()` API 都在锁内进行分析；锁被真实 Q 占用时立即拒绝，不并行抢占资源。
2. 验证显式提供的完整 V6b global `result.json` SHA、冻结 V6b worker 与 parent 脚本 SHA、实际 parent proof、健康 EXITED/0 账本行、实际 CPU、日志及结果绑定。拒绝存在 RUNNING 项的 V6b 账本。
3. 对 global、A (`rks-partial`)、B (`h1ao`) 各自的实际闭合命令逐项校验 stage、output、threads、control、snapshot、fresh-G 路径和哈希；B 的实际命令必须使用 global 声明的同一个 A。
4. 用原 V5 源码中未修改的纯元数据函数校验真实 source snapshot 和 same-boot fresh-G 的原 V5 proof。使用 AST 仅提取固定命名的函数和字面量常量，不执行该源文件的顶层 NumPy import，不翻译 V5 schema。实际数值加载后仍再次使用完整原 V5 payload 校验器验证 snapshot 数组。
5. Spinosin 固定为原 C001 的 snapshot `3089d6014ba2d2283540eb13cc7dbcdbbf284721042e7250b624197fc15c3311` 和 fresh-G `6b143defc238471bc815857df08fc4034a9d8454a00567f3b084907a54a5c403`，75 原子、C₂₈H₃₂O₁₅、中性闭壳层。独立闭合完整 2N 条健康 RHF 单位来源，核对 A 的真实既有前缀与 `prior_RHF_unit_producer_checks`，拒绝 forensic／失败来源冒充健康结果。
6. 验证 A/B cache 全部原子、文件形状和哈希、相同物理 binding、原源状态与 fresh-G、A→B→global 来源链，以及 stock global capture 的 partial/H1 cache 来源、完整 B8 响应和核项恰好一次。A/B 已完成缓存的字节哈希发生在这些 source／命令元数据门禁之后；不重新加载全部巨大 H1 数组。
7. 此后才延迟导入 NumPy，校验完整原 snapshot 数组、真实梯度、原子顺序、Bohr/Å 换算、raw/used Hessian、全部正常模式、约化质量、频率 JSON/CSV。全过程不导入 PySCF，不执行 SCF、梯度或 Hessian。

冻结 worker SHA 为 `f13701964cd8a5fbda25a19ca27ec91b4a414f6b5cb49c14abacf6c82eaa448d`；冻结 parent SHA 为 `b985d6d6e20f5e4a48387094461f19b133dae761338664e88836c62051f993a5`；a2 数值辅助源码 SHA 为 `a2e3588c6dea121be58e4a69afb137bc6e5092199ca4180937fcdb34f6f57154`。任何冻结来源改变都应建立新审查版本，不能修改旧结果来通过门禁。

## 正常模式与最小值是不同结论

使用 H/C/O 标准平均原子质量 1.008/12.011/15.999 amu，从真实完整 Hessian 独立重建质量加权矩阵及平动／转动投影，核实非线性分子的 3N−6 个模式。保留全部有符号频率，以 1 cm⁻¹ = 0.0299792458 THz 转换。

频率复算容差为 `atol=1e-6 cm^-1, rtol=1e-9`；模式正交、TR 投影与投影本征方程残差不超过 `1e-8`。另独立重建完整本征向量；固定相邻本征值间隙 `1e-8 + 1e-6*max(abs(adjacent)) Eh/(amu Bohr²)` 划分近简并组，比较子空间投影与主重叠奇异值，容差 `1e-6`，允许向量符号及简并组内正交旋转。该本征残差不等于 CPHF 残差；后者仍为 `NOT_COMPUTED`。

完整 Hessian 和全部模式复算通过，可以写“完整有符号模式已获得并复核”。只有真实 fresh-G 合格且 219 个频率全部为正，才标记本构象为已验证局部谐振最小值并进入机制候选位置比较。负频和零频不删、不改、不取绝对值冒充真实频率；负频时保留诊断结果、明确未获全正局部最小值，不产生化学归属表。

## 软件对照及固定参考

`--control-validation` 只支持下列已存在的可信旧源，并非依据元素组成判断化学身份。C₂H₆O 同分子式存在异构体；源 snapshot、原 input recipe 的 SHA、明确 molecule_id 和原 connectivity_verified 声明必须共同匹配。

| 控制 | 原 snapshot SHA | 原 input SHA | 已存在 V5 stock result SHA |
| --- | --- | --- | --- |
| water | `123e3e760111f349315703fd213d94aac26eda87e02d03caf98dc9ebbd0cfaf4` | `865877bc001ebca8a5a9fde48dbb16cd9a21504a14b435c23a276bfb6f5e9ded` | `986be9aa60ff6810693169055e078d202dc7135f8a6ddf61c47f0a921018a930` |
| ethanol | `9966fbec6184b744e236471d3b38e0cac2d40cde7cb16b45cd444b1da1e35cba` | `7c3ddf300ab6cb806a475a8e08c3a11944c3cc1e744332dab65ffbd52b638e07` | `6c35b214b22e927185ca2298674fb8e20e1a6fede06bbecc33be22e7910d1e61` |

旧 stock 参考为 `reports/pyscf-spinosin-resumable-20261104-v5/v5-validation-{water,ethanol}-stock-hessian-001/result.json`。先按其真实 V5 schema/source/proof 健康闭合、验证相同 snapshot binding，再于 V6b 全部门禁通过后读旧参考数组。不把 V5 参考重标为 V6b。

未来真实执行时，比较完整 raw/used Hessian（最大绝对差均 ≤`1e-8 Eh/Bohr²`）、全部 3/21 个有符号频率（`atol=1e-5 cm^-1, rtol=1e-9`）和近简并模式子空间（投影最大差 ≤`1e-5`）。旧参考自身也独立进行 TR 模式复算。**目前这些 V6b-versus-stock 比较一项也未运行，不能声称 staged 与 stock 数值等价或大分子提速。**

## 未来真实输出

每次使用全新输出目录，不覆盖原结果。成功后生成以下产物：

| 文件 | 可支持的结论 |
| --- | --- |
| `checks.json`、`artifact_index.json` | 实际 source/proof/A/B/RHF/global 生产链、冻结分析源码、数值复算容差与文件哈希 |
| `all_frequencies_cm1_THz.csv` | 全部真实有符号频率及 IR `NOT_COMPUTED`，包括负频 |
| `normal_modes_0p1_to_1p5_THz.csv` | 固定 0.1–1.5 THz 窗口内正常模式位置 |
| `signed_mode_frequencies.png/.pdf` | 全部有符号模式频率，不是理论吸收谱 |
| `low_frequency_positions_not_IR.png/.pdf` | 等高位置刻线，不表示 IR 强度或 IR 活性 |
| `animations/*.xyz` | 原子顺序固定、任意 0.15 Å 显示振幅的模式动画，不是分子动力学轨迹 |
| `all_experimental_theory_pairs.csv` | 仅全正 Spinosin 构象：所有低频模式 × 全部 20 条重叠 source-fold 实验候选记录的笛卡尔位置比较，无最近峰匹配／化学归属 |

实验候选文件仍为 `reports/upgrade-20261031-v1/source-peaks/experimental_candidates_by_fold.csv`，固定 SHA `b3a5f59157f42859f68edf0a8760d9f811f83ea7947c978cc0a9cbc98a4ba8b4`。5 个训练折 × 4 条候选共 20 条记录，不是 20 个独立化学峰。纯品实测、含量检测和固体结构证据缺失时，频率接近不能确认成分；单一孤立构象也不覆盖晶格、氢键和分子间集体振动。现有 MC-UVE＋SVM 的 96.0234% 内部开发结果不接入这些理论位置、不被改写。

原 group participation 分析器暂不适用于 V6b；新 provenance adapter 当前 `NOT_RUN_NOT_IMPLEMENTED_FOR_V6b`，本次不修改旧 group 文件。

## 复现与资源

当前只允许执行纯准备，实际已执行的命令及小文件哈希保存在独立 preflight 目录；没有读取目标 NPY、CHK、Hessian、实验候选 CSV，也没有数值库冷加载：

```bash
cd /workspace/research4
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_analyze_pyscf_staged_modes_v6b.py -v
PYTHONDONTWRITEBYTECODE=1 python scripts/analyze_pyscf_staged_modes_20261105_v6b.py --preflight --output reports/pyscf-v6b-modes-analysis-preflight-20261105-v1
```

只有 Q 已健康闭合、实际 V6b global 结果及其真实 SHA 已由 root 选定后，才能执行下列命令。以下路径和 SHA 是需填入的真实输入，**不是已有结果**：

```bash
cd /workspace/research4
V6B_RESULT='/absolute/path/to/healthy-v6b-global/result.json'
V6B_RESULT_SHA='actual_64_character_result_sha256'
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/research4/.runtime/bin/python scripts/analyze_pyscf_staged_modes_20261105_v6b.py \
  --result "$V6B_RESULT" --result-sha256 "$V6B_RESULT_SHA" \
  --output reports/pyscf-v6b-modes-actual-analysis-UNUSED-ID --cpu-seconds 300 --wall-seconds 900
```

真实 V6b water/ethanol global 各用不同未存在目录并加 `--control-validation`；旧 V5 stock 文件不能作为新 V6b `--result`。后处理固定 1 CPU、1 GiB 地址空间，默认 300 actual process CPU 秒／900 wall 秒，命令行最多 600／1800 秒，超限明确失败，不扩大量子预算、不启动新量子计算。若内存限制导致解析／绘图失败，保留真实失败日志，另行审查新版本，不能降低来源与数学门禁来生成结果。

准备状态只证明小代码和拒绝规则可执行；完整 V6b 正常模式、图、机制比较与所有 control 数值结论仍待实际完成 Q 后验证。
