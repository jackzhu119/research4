# V6b Spinosin C001模式组份额：准备与门禁

**当前仅完成独立代码、静态C001图／原子映射复核及标准库契约测试。实际Spinosin组份额、CSV和热图均为NOT_RUN。** 大分子A阶段运行期间，没有读取当前Q账本、模式、Hessian、NPY／CHK或实际频率，没有导入NumPy／Torch／PySCF／RDKit，没有启动计算或Git操作。

新增 `scripts/analyze_spinosin_staged_groups_20261105_v6b.py`。旧 `analyze_spinosin_group_participation_20261104.py`（SHA `b997a51f1d57b29cb2f05b67209bf993d1882af504a4af164c6912ed497c2dcd`）的V5/a2门禁不改、仍不宣称接受V6b。新adapter只在所有新来源门禁通过后使用旧 `participation()` 数值辅助函数；当前不导入旧脚本的顶层NumPy。

## 真实计算前的强制条件

1. 命令行及公开 `analyze()` API取得已有 `.runtime/pyscf-v4-global.lock` 排他锁；实际量子账本必须由调用者提供路径与SHA，经冻结V6b parent验证全部阶段已终止、无RUNNING或未知CPU。
2. 提供真实独立分析目录及artifact index SHA。索引必须绑定新analyzer_v2源码 `38dde5e185c4fc48d58732982e0164b3a2f5430a26bdfc1076b21103d1d75c6a`、真实V6b worker和原a2数值辅助源码，拒绝旧V5/a2报告重标。
3. `checks.json`必须是原Spinosin、75原子、全部219模式、全正频率、虚频／零频均0、`qualified_for_mechanism=true`，原snapshot `3089d601...c3311` 与fresh-G `6b143def...c403`固定。完整计算完成与局部最小值是独立门槛，任何未通过项都阻塞组份额，不删除负／零频或伪造空结果。
4. 用新analyzer_v2的真实V6b门禁重新核A/B/global parent、实际argv、完整健康RHF链、原source/G、同一物理binding与cache。实际闭合记录必须与独立分析导出的记录完全一致。这一步仅在Q全闭合之后执行；当前preflight不调用它、不哈希巨大H1缓存。
5. 此后才延迟导入数值库，核219个Cartesian模式、真实几何与原C001坐标／顺序完全匹配，再独立重放完整TR模式及子空间，核`Σm*d²≈1`（绝对误差≤1e−8），最后计算描述份额。

## C001映射和组的确切含义

静态包 `reports/external-evidence-20261104-v1/mode-interpretation-preparation/files_sha256.json` 的固定SHA为 `5ecf8bbe7c5317d5ef0d18070f01307542c366f0700bcbaf43e5e7849a5aa246`。纯元数据阶段AST提取旧脚本未修改的 `static_groups/validate_groups` 等函数，复核包内所有小文件及原可信化学JSON、C001 SDF、坐标传递XYZ和生成源码的哈希。不运行RDKit；这是重新验证此前已审核的有序图与来源证据，不声称新完成化学图计算。

原schema有22个group：**4个 `DISJOINT_PARTITION` 组共同构成一个完整的75原子分区**，原子数为32／20／1／22（苷元／近端糖／氧桥／末端糖），合计75且每个原子恰好一次。另18个 `OVERLAPPING_GRAPH_SUBGROUP` 为重叠局部图描述，不能相加为1。新结果明确报告 `complete_atom_partitions=1`、`disjoint_groups_in_complete_partition=4`、`overlapping_graph_subgroups=18`。

仅原C001零基0—74及其一基对应可用。后上传SDF虽然化学身份相同，存在多个合法原子自同构映射，不能任取一个对接历史模式。元素顺序相同本身也不足以排除同元素交换，须共同满足已封存的有序图、索引传承、原snapshot和真实几何证据。

## 数学量与未来输出

使用已验证的旧辅助公式：

`share(group,k) = Σ_group m_i ||d_ki||² / Σ_all m_i ||d_ki||²`

PySCF保存的是Cartesian位移，`d=m^(-1/2)e`，因此此处乘质量而不是再把输入当作已质量加权向量。质量H/C/O为1.008/12.011/15.999 amu，旧helper先按模式最大幅度缩放避免溢出；新adapter另用标准库标量公式独立核对（未来真实数据最大差要求≤1e−12）。四个互斥组份额和须为1（误差≤1e−12），18个重叠组各自为描述量。

真实符合条件后才生成：219×22＝4818行 `mode_group_participation.csv`、219行 `partition_participation.csv`、固定0.1–1.5 THz低频视图，以及 `low_mode_group_participation_not_IR.png/.pdf`。热图将互斥组与重叠组分开；无窗口模式时明确写无模式，不补峰、不匹配实验候选。真实审计结果与来源哈希写入 `checks.json` 和新source绑定的artifact index。

错误混种分类成绩已按2026-11-12用户决定撤回；论文使用main的paper/current白名单。

## 复现与当前状态

10项纯标准库契约测试及独立静态复核通过；只包含小标量软件fixture及静态映射，不是实际Spinosin份额或旧NumPy helper重跑。旧数学软件验证证据仍保留于 `reports/pyscf-spinosin-resumable-20261104-v5/group-participation-preflight-20261104-v1/`，本轮不重跑它。

```bash
cd /workspace/research4
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_analyze_spinosin_staged_groups_v6b.py -v
PYTHONDONTWRITEBYTECODE=1 python scripts/analyze_spinosin_staged_groups_20261105_v6b.py \
  --preflight --output reports/pyscf-v6b-group-participation-preflight-20261105-v1
```

未来只有真实C阶段及独立模式分析合格后，由root选定实际路径与SHA，在全新目录运行：

```bash
/workspace/research4/.runtime/bin/python scripts/analyze_spinosin_staged_groups_20261105_v6b.py \
  --analysis ACTUAL_QUALIFIED_V6B_ANALYSIS_DIRECTORY --analysis-index-sha256 ACTUAL_INDEX_SHA256 \
  --quantum-ledger ACTUAL_TERMINAL_V6B_LEDGER --quantum-ledger-sha256 ACTUAL_TERMINAL_LEDGER_SHA256 \
  --output reports/pyscf-v6b-group-participation-ACTUAL-UNUSED-ID --cpu-seconds 180 --wall-seconds 600
```

上面大写输入是尚未提供的真实输入，不是现有结果。后处理限1 CPU、1 GiB地址空间，默认180 CPU／600 wall秒，最多300／900；不触发新量子计算或新增模型拟合。当前preflight明确NOT_RUN，没有真实数值表、热图或模式归属。
