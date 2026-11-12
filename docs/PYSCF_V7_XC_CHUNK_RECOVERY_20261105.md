# V7：RKS XC可恢复分块设计、契约与有限控制操作

原V6b大分子A在环境重启后未产生成功result或频率；终态进程CPU **24863.36秒必须完整计费**，不能以缺wait4回执计为0或重新归零。root已将原phase失败闭合并保留旧文件，当前剩余额度约61593.862325 CPU秒，没有增加预算。旧A没有磁盘XC中间检查点，不能从已终止进程恢复未写出的内存累加器。

本轮新增V7源码与标准库测试已经冻结、独立审查通过；**不表示实际XC控制或Spinosin已成功**。root已在原账本签发有限control-A授权并运行water `xc-plan`，健康EXITED/0、0.848962 CPU秒；这只建立真实分块计划，没有完成XC矩阵／完整Hessian／频率控制。Spinosin仍须等待水、乙醇完整真实控制及新的root control release，不能盲重试原A。

## 一、实际源码支持什么分解

本文只根据实际安装源码审查；未修改PySCF安装文件或冻结V6b worker。

| 来源 | 代码依据 | 含义 |
| --- | --- | --- |
| `pyscf/hessian/rks.py:66–69` | 原`rhf._partial_hess_ejk`，再`e1+ej-hyb*ek` | 当前纯PBE的完整RHF部分为e1+ej，仅计算／加入一次 |
| `rks.py:71–83` | native `_get_vxc_diag`及`_get_vxc_deriv2` | 两个XC贡献需分别完成，不得只算deriv2漏diag |
| `rks.py:353–354,381–404` | GGA各grid块累加raw vmat与共享ipip，遍历全部原子 | 固定完整AO／MO／占据时，每grid点的PBE XC贡献可加；缩小atmlst不会拆开此计算 |
| `rks.py:406–409` | 每原子AO列加入ipip及`transpose(1,0,3,2)` | 原生helper末尾处理只算本次所选网格的ipip，不在外部重复加入；不是普通全矩阵对称平均 |
| `rks.py:106,109,112` | diag自项、deriv2原子对的原einsum及下三角转置填充 | 每chunk原生helper返回后立即收缩为小的[natm,natm,3,3]矩阵 |
| `pyscf/dft/numint.py:2887–2928` | 完整ngrids决定blocksize、sparse、mask偏移，最后eval_ao/yield | 不切grids、不截mask；只有唯一循环头加入未选块continue，且在coords／AO计算前跳过 |

固定源码SHA：RKS `d61dbd1727b286845737853b92e1812a3e6a966dca29a9a8a79ff5e9a7725c03`；NumInt `ff7384551c7a6b80ff8a5ffcd915eb8bfbf6ca4b8f987f912aa82a1d6edda7e8`；grad/RKS `cffc3e9efa10cf1b547d757e9d165c1000377e9159c794f87464a79c8c93ab46`；gen_grid `44aa665fb25efc544a42cb80fdad737767c787c20479d4df0ae4b1649028ff14`。

新core对实际导入的4个module路径／SHA、RKS→NumInt／grad别名、原block_loop的globals和BLKSIZE56再次核验，防止哈希一套site而执行另一套。AST删除新增的唯一if/continue后必须与原函数完全相同。只替换当前NumInt实例方法，结束后恢复；不改全局类、AO计算、mask、cutoff、库函数或数学式。

适用范围仅当前完整AO、非DF、非混合、无range separation／NLC／dispersion的PBE/GGA。非局域VV10等含跨网格项，不能套用此分解。max_memory=2000 MB是新明确冻结的XC分块政策，不能声称已经证明等于历史运行按RSS动态选出的数值。实际完整网格／权重／mask不变，但块求和与收缩次序改变，仍必须实际验证浮点误差。

## 二、复杂度和恢复边界

Spinosin的75×9×762²双精度vmat约3.14 GB；每次原生deriv2仍会在RAM分配它，**本方案不是降低该峰值内存的承诺**。区别是每个成功chunk返回后马上收缩，仅保存405000字节、约0.386 MiB的XC小矩阵，而不是把每个chunk的3GB AO张量写磁盘。控制分子额外保存原生AO张量用于全元素差分，目标75原子禁止这样保存。

原PBE逐点XC总工作量量级保持相同，增加了多次进程恢复、vmat清零及文件校验开销；跳过未选原始grid块发生在eval_ao前，不会为了每个chunk重新评估所有未选AO。实际时间不能由11块简单平均保证，应从有限控制及首块真实CPU／RSS日志评估。

每chunk必须完整native返回、原MF／grid／方法恢复、实际parent健康EXITED/0并关闭计费后，才有资格复用。结果文件存在但parent失败／未知／重启中断的chunk一律不接受；重算该chunk用新phase和目录，不覆盖。RHF只用原健康149进入新namespace，原生RHF完整返回并独立健康关闭后保存e1+ej；失败phase新产生的last unit继续隔离。

diag与deriv2各有独立计划：导数阶3／2不同，不能假定默认blocksize相同。每kind实际原grid block均须完整且仅出现一次；两计划须同physical、同完整grid/mask、同nao/natm/ngrids、同memory/control政策。最终A为一份健康RHF矩阵＋全部健康XCdiag块＋全部健康XCderiv2块，按kind/start固定顺序加入，不加核项或CPHF、不用对称平均掩盖误差。

## 三、冻结文件与新producer契约

- 核心：`scripts/pyscf_xc_chunk_checkpoint_20261105_v7.py`，SHA `da1e2b0267c48e748a2bcb7d3f2ee3110d8e2288aa340d23dab7a12c994ddf8e`。
- A／control worker：`scripts/pyscf_xc_chunk_worker_20261105_v7.py`，SHA `00e710eb0d637dbd073939bea5d4496993cb68f29efad1439322df3972415f02`。
- 27项标准库测试：`test_pyscf_xc_chunk_core_v7.py`、`test_pyscf_xc_runtime_binding_v7.py`、`test_pyscf_xc_chunk_worker_v7.py`；独立源审及27项测试通过，不含NumPy／PySCF运行或真实AO计算。
- 静态preflight：`reports/pyscf-spinosin-grid-checkpoint-20261105-v7/metadata-preflight-001/`；`control_input_refs.json`只有既有可信source／fresh-G引用，不是执行授权。

使用已有真实b985 V6b预算parent及其原proof schema，费用继续进入原账本；**worker/results是V7，不冒充f137 producer**。结果schema为`research4-staged-full-rks-hessian-result-v7`，完整A cache schema为`research4-complete-full-rks-stage-cache-v7`。same physical schema只描述相同物理状态，不表示producer兼容。旧f137、38dde和旧group门禁仍拒V7。

| stage | 成功status | 作用 |
| --- | --- | --- |
| `xc-plan` | `XC_PLAN_COMPUTED` | 保存独立diag／deriv2完整固定计划 |
| `rhf-partial` | `RHF_PARTIAL_COMPUTED` | 原生完整e1+ej，无XC／核／CPHF |
| `xc-diag-chunk`／`xc-deriv2-chunk` | `XC_CHUNK_COMPUTED` | 原生所选grid块完成并收缩 |
| `rks-partial` | `STAGE_COMPLETE` | 独立健康A组装，保存`full_RKS_partial.npy` |
| `control-reference` | `CONTROL_REFERENCE_COMPUTED` | 小分子未改native完整XC AO及RKS partial基准 |
| `control-compare` | `CONTROL_PARTIAL_DIFFERENTIAL_PASS` | 全AO／RHF／完整partial差分通过，尚非完整频率控制 |

输出统一`reports/pyscf-spinosin-grid-checkpoint-20261105-v7/`，scratch统一`outputs/pyscf-spinosin-grid-checkpoint-20261105-v7/`，旧V6b目录／账本不搬动、不覆盖。所有阶段固定2线程。

### A组装输入的精确schema

`component_index.json`只有下面4个顶层键，记录必须引用真实健康关闭的`result.json`。每个ref至少有绝对path与SHA，bytes可带且带时必须精确。

```json
{
  "schema": "research4-v7-XC-component-index-v1",
  "RHF_partial": {"path": "ACTUAL_RHF_RESULT_JSON", "sha256": "ACTUAL_SHA256"},
  "XC_plan": {"path": "ACTUAL_PLAN_RESULT_JSON", "sha256": "ACTUAL_SHA256"},
  "XC_chunks": [
    {"path": "ACTUAL_HEALTHY_DIAG_CHUNK_RESULT_JSON", "sha256": "ACTUAL_SHA256"},
    {"path": "ACTUAL_HEALTHY_DERIV2_CHUNK_RESULT_JSON", "sha256": "ACTUAL_SHA256"}
  ]
}
```

示例中的两个chunk是结构示意，不是固定数量；实际列表必须覆盖两kind全部原blocks，每个一次。索引文件本身SHA作为`--component-index-sha256`。A cache只有一个float64 `[natm,natm,3,3]` 的`full_RKS_partial.npy`，不包含核Hessian／CPHF。

新B/C的stdlib接口为 `qualify_partial(record,physical_binding,atoms,nao,guard)`，返回`(result,manifest,cache_dir,closure,extra_chain)`。extra schema是`research4-v7-complete-xc-partial-lineage-v1`；真实验证后才导出all_chunk_producers_healthy、all_required_grid_blocks_covered_exactly_once、RHF_added_once、XCdiag_once、XCderiv2_once、actual_partial_command_verified、actual_component_commands_verified，以及nuclear_or_CPHF_included=false。immutable `verified_artifacts`去重；追加账本用`closed_ledger_rows`按实际phase行哈希核验，不把旧整账本SHA当永久不变。完整root控制release也进入target的实际argv和artifact链。

## 四、水／乙醇实际有限验证操作

以下只给root串行操作入口，不启动任何进程、不分配新预算。继续使用root已批准的原ledger及有限control-A额度，未批准的Spinosin不能运行。

每个原生stage都须由冻结budget parent `run` 启动，显式指定新phase名、CPU／wall／threads2／memory24GiB和新log路径，再传worker `--execute`。共同worker参数必须显式含：

```text
--threads 2 --snapshot ACTUAL_SOURCE_DIR --snapshot-manifest-sha256 ACTUAL_SOURCE_SHA
--gradient-check-result ACTUAL_G_RESULT --gradient-check-result-sha256 ACTUAL_G_SHA
--control-validation
```

旧可信水／乙醇源与G引用见`metadata-preflight-001/control_input_refs.json`。worker会再核真实same-boot G与原snapshot；准备引用不保证当前boot已合格。water snapshot SHA为123e3e...cfaf4、ethanol为9966fb...35cba，不能用任意同分子式结构代替。

1. `--stage xc-plan --xc-max-memory-mb 2000`：**该memory flag必须实际出现在argv**。小控制固定blksize224，保证真实多块；目标以后采用M2000的原自动公式，而不是控制224。
2. 独立`--stage rhf-partial`，小控制不提供rhf-cache；独立`--stage control-reference`保存未改native完整AO／partial基准。两者先健康关闭。
3. 对每kind运行若干`xc-*-chunk`：必须实际写`--plan-result`／SHA、`--start-block`／`--block-count`。可把原block索引按三个连续组分开，分组只依赖已冻结plan，不依赖数值／结果。
4. 将全部健康source refs写入上述component index；用`--stage rks-partial --component-index ... --component-index-sha256 ...`独立组装A并关闭。
5. `--stage control-compare --partial-result ACTUAL_A_RESULT --partial-result-sha256 ... --reference-result ACTUAL_REFERENCE --reference-result-sha256 ...`。固定要求完整diag AO、deriv2 AO、RHF部分和完整RKS partial最大绝对差均≤1e−8；失败不得放宽阈值或选对自己有利的块。
6. **还没有完成完整控制。** 使用新V7 B/C消费者完成H1AO、每个独立健康response-block、global-assemble，再由新V7独立分析器对全部raw/used Hessian、全部有符号频率和模式子空间与可信stock比较；水／乙醇分别应实际有3／21模式。新producer链不能送旧38dde来绕类型。

已完成water plan `control-water-xc-plan-001`有10128点，两kind各46个224-point blocks。可各用`(start,count)=(0,15),(15,15),(30,16)`，共6个XC阶段，恰好覆盖。实际plan result SHA `633c14885be44018c8d5c74ad6c14b2f6809b765c569960604b1a9849f950919`。乙醇必须等其真实plan后从其block列表确定，不能抄水的范围。

还应验证：分块前后／提交前后中断、失败chunk隔离、缺失或重复块拒绝、plan／mask／source变更拒绝、重复恢复不重复加入，以及原MF／grid／NumInt／native driver恢复。当前只有标准库拒绝规则与AST契约测试通过；真实控制尚未全部完成，不能宣称数学／速度实测成功。

## 五、Spinosin事前门禁与科学限度

`require_control_release(path,sha,consumer_source=...)`在NumPy导入前要求root显式release，绑定A/core/consumer/analyzer全部实际SHA，两个真实健康partial差分结果、两个实际完整控制index以及真实C健康producer。缺少任一水／乙醇AO、完整Hessian或全部signed频率控制即阻塞target；当前尚无Spinosin release。

通过控制后，root才可在原剩余预算内分配target：新namespace健康149 seed→RHF-only健康close；固定target计划→逐块健康close；A组装→新B/C及response恢复→独立模式分析。不得自动增加预算、重新开始72小时或使用失败A输出冒充健康native贡献。必要的目标有限梯度方向检查应另冻结方向／步长及一致网格语义并在剩余预算内批准，不能在此自动展开昂贵计算。

错误混种分类成绩已按2026-11-12用户决定撤回；论文使用main的paper/current白名单。
