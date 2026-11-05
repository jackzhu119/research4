# Historical NN verification: separate, bounded no-fit utility

Status: **PREPARED / NUMERICAL EXECUTION NOT_RUN**. Nineteen synthetic, standard-library tests passed. No current quantum ledger, experimental arrays, labels or weights were opened during this preparation. No training or numerical replay was performed. The original actual v2 independent audit remains unchanged and recorded PASS, including its actual 36-weight replay; this utility does not claim a second completed replay.

The original NN release authenticated a terminal quantum-ledger snapshot. A later legitimate quantum continuation changes the live ledger SHA. Separately, the original NN training deadline will expire. Neither event changes the recorded historical production times or permits editing the original authority, changing its deadline, rolling back the live ledger, or restarting training. The new utility verifies the old experiment and, only under a new explicit bounded grant, may recompute its metrics and replay its saved weights without fitting.

## Frozen implementation

- Wrapper: `scripts/audit_supplemental_multirepresentation_historical_20261104_v3.py`, SHA-256 `37d1fdc7ed3bbad2bc7b0da43f538a543d717271319fe6653d452e585966caca`.
- Deferred numerical leaf: `scripts/historical_supplemental_nn_numerical_20261104_v3.py`, SHA-256 `9c1bb1acba525642086d050e53025cdcfbc492977e0574c06a3ee3e7e472cda6`.
- Readonly numerical helpers: original v2 auditor, SHA-256 `7ef9829efd2bdfefc75e4a7e59813f6713ba8779342176aedc3adca899bc6306`. Its top-level audit, authority, budget, training and old claim functions are not called or monkeypatched.
- Frozen quantum metadata guard: `run_pyscf_continuation_budget_20261105_v6b.py`, SHA-256 `b985d6d6e20f5e4a48387094461f19b133dae761338664e88836c62051f993a5`.

The wrapper first checks a new grant that explicitly forbids training and pins the old registration, completion, ledger, successful v2 audit, code and archived quantum snapshot. It authenticates all six historical supervisor/worker closures and exact recorded worker commands. Their times must fit the **original** NN deadline, including the unchanged proof deadline. This is a comparison of past recorded times; it does not compare that old deadline with today's clock or revive an old parent process.

It then acquires the actual original nonblocking global quantum lock. Only while holding that lock can it read the current quantum ledger. The complete 55-phase archived prefix, immutable headers, budgets, sources, authorizations and lineage must remain identical. Only appended closed phases and their truthful CPU totals/remaining budget can differ; failed costs remain charged. The frozen quantum guard independently validates current metadata, with RUNNING, unknown CPU or residual-process flags rejected. The ledger is checked unchanged throughout the actual audit. The utility neither reads quantum scientific arrays nor launches quantum work.

Only after those gates does the numerical leaf check all 27 source models and all 36 final model/prediction identities, the three seeds, source-only epoch locks, model-role IDs, the time-axis repair metadata, and matched-capacity C/M controls. It then recomputes all metrics, same-query modality comparisons, the paired old 171 acid-jujube samples and the three new query samples. Optional replay loads all 36 saved states with the original scaler and class order; it does no fitting. All new outputs live in a fresh audit directory. Original NN artifacts, prior audits, original quantum/NN budgets and authorities stay unchanged.

## Cold checks and future invocation

Cold regression command, already run successfully:

```bash
python -B -m unittest discover -s tests -p 'test_*historical*v3.py' -v
```

The preparation directory is `reports/supplemental-multirepresentation-historical-audit-preparation-20261104-v3/`. Its `cold_plan_FINAL_NOT_RUN.json` is a preparation record. `root_grant_FINAL_TEMPLATE_NOT_AUTHORIZED.json` has `root_execution_authorized=false` and placeholder issue/expiry times, so it cannot start an audit. The older `root_grant_TEMPLATE_NOT_AUTHORIZED.json` is retained as an intermediate preparation artifact and binds an earlier source SHA; do not use it for execution.

After the actual quantum work has closed, the root controller may issue a **new** grant from the FINAL template, with current UTC issue/expiry times within a current user-authorization record and an unused output directory. The independent audit limit is at most **180 process CPU seconds, 600 wall seconds, one CPU thread, 1.5 GiB best-effort peak RSS and 10 GiB virtual address space**. This is a new verification allocation, not a reset of NN training or quantum budgets. A fresh output and one-use claim are required. A hard-killed audit has no healthy closure and must not be reported as completed.

Example future CLI, **not executed in this preparation**; use the existing runtime/environment that passed the original v2 replay:

```bash
/workspace/research4/.runtime/python/bin/python -B \
  scripts/audit_supplemental_multirepresentation_historical_20261104_v3.py audit \
  --run reports/supplemental-multirepresentation-20261104-v2 \
  --output reports/supplemental-multirepresentation-historical-independent-audit-20261104-v3 \
  --grant PATH_TO_NEW_ROOT_ISSUED_GRANT.json \
  --grant-sha256 SHA256_OF_THAT_NEW_GRANT \
  --replay-models
```

An expired current grant, active quantum phase, changed old artifact, changed frozen source, altered archived prefix or changed budget/lineage fails closed before numerical work. This preparation is not evidence of new scientific improvement and does not select another model from the exposed evaluation data.


2026-11-05 实际执行记录

状态：DONE。完成了新的历史核验，没有重新训练。量子孤儿阶段完成显式失败对账后，在独立180 CPU秒／600墙钟秒授权内，真实复算了36份预测产物和全部指标，并加载36个保存的模型执行无拟合重放。27个source模型与36个最终模型元数据通过检查，最大指标复算差为1.4211×10⁻¹⁴；实际审计进程CPU为4.744852527秒，峰值RSS为406458368字节。

本次核验保留原55阶段量子账本归档，核对当前65阶段账本的追加链和全部失败成本；未更改旧NN训练期限、授权、指标或权重。Spinosin计算成功与否不是NN验证通过的前提。

实际产物：`reports/supplemental-multirepresentation-historical-independent-audit-20261104-v3/`。实际授权：`reports/supplemental-multirepresentation-authority-20261104-v2/root_historical_no_fit_audit_grant_20261105_001.json`。
