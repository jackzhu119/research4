# 多表示固定小实验：完整分阶段CLI与执行边界

代码已补全为[独立driver](../scripts/run_supplemental_multirepresentation_20261104.py)，
复用原[冻结leaf](../scripts/draft_supplemental_multirepresentation_20261104.py)及
[固定配方](../data/research/supplemental_multirepresentation_draft_20261104.json)。
旧草案和静态预检保留其当时状态，原633-fit实验、MC-UVE结果、数据registry
与原拆分不改。**目前全部数值阶段仍NOT_RUN**：此轮只写小源码/文档并运行
纯stdlib fixture测试，未导入数值库、读取数组或训练。

固定比较仍是单支α128、容量控制C[α128,α128,α1024]、多表示M[α128,n128,
time1024]。α长支直接从原canonical612插值；time6750在均匀原轴检查后
固定polyphase抗混叠（512/3375、Kaiser5、line边界）到1024，单位仍unknown。
n不做行SNV；每个learned scaler仅fit当前source折。全部旧角色、原source
三折与seeds42/43/44固定。27个sourceα模型在24/48/96作source OOF共同
选轮，再36个最终source模型；扩10类继承旧7类epoch，无singleton CV。
主比较M−C，全部正负结果均保存，支持池本试验不参与。
第三α长支从原612直接派生，比单支α128保留不同的α采样表示，因此C−A
同时含分支容量与α表示分辨率变化，不能当作纯容量的因果消融。M−C保留
相同网络参数、各分支长度、初始化、epoch与batch顺序，回答的是该明确
控制下更换表示的效果；也不等于证明某个物理模态的独立化学贡献。

## Q之后的实际进入条件

数值CLI必须收到root独立写入的终态Q/实现审核release，且Q账本没有RUNNING
或未知实际CPU阶段。driver每阶段获取**原Q worktree的现有全局锁**，与Q
互斥；不是在新worktree伪造一个同名锁。用户既有授权无需重问，本release
是root对实际终态、输入/实现和资源的可追溯登记，当前尚未创建。

release格式示例（示例hash均须以实际闭合文件替换，不能直接运行）：

```json
{
  "schema": "research4-post-quantum-fusion-authority-v1",
  "root_execution_authorized": true,
  "quantum_terminal_reviewed": true,
  "implementation_review_passed": true,
  "quantum_worktree_root": "/workspace/research4",
  "quantum_global_lock": "/workspace/research4/.runtime/pyscf-v4-global.lock",
  "quantum_ledger": {"path": "ACTUAL_CLOSED_QUANTUM_LEDGER", "sha256": "ACTUAL_SHA256"},
  "limits": {"cpu_seconds": 3600, "wall_seconds": 14400, "threads": 1,
             "device": "cpu", "RSS_bytes": 1610612736, "AS_bytes": 10737418240}
}
```

3600实际CPU秒/4小时/单线程、RSS1.5GiB best-effort/AS10GiB是待root审定
上限，不是已授权或已发生的拟合。数值步骤在独立受监督进程执行，累计
wait4子进程CPU及已观测监督器CPU，不把进程墙钟当CPU；最终元数据写入/
返回的微小额外监督器CPU未包含在该采样总和，语义明确保留。CPU、墙钟、
RSS或未知wait4触发即失败关闭，保留断点/失败日志，不自动重置预算或重跑。
独立新输出目录下的RLIMIT_CPU、RLIMIT_AS、pidfd监控和父死亡SIGKILL
借用已审核的旧单进程Torch pilot思路，不使用GPU/付费服务。

## 可复现命令与阶段屏障

执行工作目录为`/workspace/research4-supplemental`，复用主worktree中的
`/workspace/research4/.runtime/bin/python`；supplemental目录没有独立`.runtime`。
数值依赖路径显式指向主worktree的既有site目录，`src`仍是当前supplemental
源码。以下仅是Q结束后的命令说明，本次未执行数值阶段。

```bash
PYTHONDONTWRITEBYTECODE=1 /workspace/research4/.runtime/bin/python \
  scripts/run_supplemental_multirepresentation_20261104.py plan

PYTHONPATH=/workspace/research4/.runtime/quantum-pilot-site:/workspace/research4/.runtime/quantum-upgrade-site:src \
/workspace/research4/.runtime/bin/python scripts/run_supplemental_multirepresentation_20261104.py register \
  --output reports/supplemental-multirepresentation-20261104-v1 \
  --authority PATH_TO_ACTUAL_ROOT_RELEASE --authority-sha256 ACTUAL_RELEASE_SHA

PYTHONPATH=/workspace/research4/.runtime/quantum-pilot-site:/workspace/research4/.runtime/quantum-upgrade-site:src \
/workspace/research4/.runtime/bin/python scripts/run_supplemental_multirepresentation_20261104.py run \
  --output reports/supplemental-multirepresentation-20261104-v1 --stage all
```

也可逐个使用`--stage prepare|bind|source|fit|predict|score`。每阶段仅接受
前一阶段的健康闭合receipt、实际result hash及相同注册代码。已有成功阶段
可由`all`检查后跳过；任何失败/未知阶段不能被当作成功跳过或重新覆盖。

| 阶段 | 实际执行时做什么 | 此源码交付时 |
| --- | --- | --- |
| prepare | 固定原角色；α继承旧canonical rolepack；旧ZA855的n/time按各自record_ids；新增ZA9及HQ/TM按incoming→canonical→entity显式映射；跟v2 registry实际路径而非猜v2目录；按独立轴固定转换后写新compact source/query包与独立manifest | NOT_RUN |
| bind | 校验全部实际input/code/axis/ID引用；用source小切片实测C/M参数与初始化相等、forward/BN点式不变、非法fit-ID在Scaler真正fit前拒绝；另用明确软件脉冲/高频控制检查抗混叠 | NOT_RUN，不能拿纯测试替代这些实际门禁 |
| source | 只读source_manifest与source数组；27个sourceα fit；所有候选/seed/OOF ID齐全后锁3任务共同epoch；B只继承A | NOT_RUN |
| fit | 不接真实query；全部36个source模型；核C/M初始化、参数、epoch及batch顺序；统一模型锁 | NOT_RUN |
| predict | 先复核36模型和source锁，再只读无标签query；eval/no_grad、点式/批次/BN复核；统一36预测锁 | NOT_RUN |
| score | 先复核全部36预测的bytes/IDs/类别顺序，之后独立进程才读truth.csv；不再训练或改变配置 | NOT_RUN |

原角色包只有α，不能直接冒充多表示。incoming ZA merged514不覆盖旧855，
准备器保留旧ZA库并补新9；HQ03格式等价overlay依已有可见映射，不重算
浮点身份后把它变新样品。缺行、多候选parent、跨角色alias、轴或单位契约
不合、非有限值和缺模态均阻止准备，不自动删query或造值。原参考列从未
进入X。support只保留角色/身份元数据，不参与本试验。

## 实际完成后才会产生的交付

每stage有console、监督proof、wait4行、worker_result及closed_stage；原始
输入、代码、准备包、source/模型/预测锁有SHA。每个神经batch的真实fit IDs
由旧FitScope记录；这仍是API/来源边界，不是OS真值沙箱。

完成后`score/`保存全seed逐样品预测、Accuracy/Macro-F1/BA、逐类P/R/F1、
混淆计数、三组配对纠正/伤害、参数/模型大小/CPU表、三seed均值及样本SD，
并生成PNG/PDF模型对照和source选轮曲线。扩10类单列总174、旧171和新3
（每新类仅1份query）；对子集按其真实支持类报告Macro-F1/BA，并保留完整
10类混淆矩阵，向旧类错分仍计FN。比较前所有arm按同一record_ID显式重排。

所有队列仍是已曝光开发或很小的新类入组探索，303条新可读表示不增加HQ/TM
独立N；结果不冒称新批次或物理单位已验证。任何零增益、负迁移和失败均
保留，历史96.0234%不会重新赋给此513-source/171-query神经协议。

源码交付时19项driver纯stdlib测试通过；独立复核同时重跑冻结leaf的24项，
共43项纯软件测试通过。静态审查中发现并修复了阶段锁SHA链未锚定、truth
前36条唯一网格/类别列校验不足以及关闭时累计CPU越限漏判，相关反例均
保留。当前实际forward、反向、输入数值准备、滤波和拟合仍NOT_RUN，不能
把软件fixture测试当作这些真实数值门禁的完成证明。
