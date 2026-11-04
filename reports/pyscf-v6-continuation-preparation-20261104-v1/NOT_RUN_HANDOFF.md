# V6 continuation guard — NOT_RUN contingency

This directory documents independent engineering preparation. It does not authorize a new allocation, initialize a real ledger, release failed scientific caches, or launch quantum work. The active V5 producer is untouched. If V5 succeeds, this contingency may remain unused.

## Immutable lineage and resource accounting

`scripts/run_pyscf_continuation_budget_20261105_v6.py` accepts only a SHA-bound, fully closed V5 ledger from the frozen guard `scripts/run_pyscf_actual_cpu_budget_20261104_v5.py` (SHA-256 `33f1dc3db5280f16c9898c561e345143614730dc052b171b4f11fbbb079cdeee`). It does not import V4 directly or reconstruct an earlier, cheaper prefix.

Registration copies the complete V5 ledger and phase array, including every failed phase and its actual wait4 CPU cost. The original `created_at_utc`, original V5 deadline, original authorization, source checkpoint, source revision, prior V3 provenance, and V4-to-V5 migration metadata remain traceable. Old phases and proofs retain their actual versions and statuses.

The root must supply a new explicit decision **after** actual V5 closure. It contains the exact closed V5 SHA and consumed CPU, explicit additional CPU (0–43,200 s), explicit deadline extension (0–43,200 s), and a conservative `research_authorization_deadline` supported by the user's existing authorization evidence. The effective deadline is exactly original V5 deadline plus the explicit extension, and cannot exceed that conservative endpoint. Registration time does not start or restart a 72-hour authorization clock.

The cumulative cap is original V5 cap plus the approved increment; consumed CPU is the complete old actual cost plus all subsequent actual costs, including failures. Unknown, nonfinite, unresolved, or residual-process costs block continuation. Re-registering or changing the SHA-bound V5 source is rejected. There are no automatic retries or allocation increases.

Only one V6 allocation is permitted for the original V5 budget lineage. Authorization binds the sole canonical V6 ledger path. Under the shared and lineage locks, registration first atomically publishes a no-replace claim in `.runtime/pyscf-v6-continuation-registry/`, keyed by original creation, V5 authorization SHA, source checkpoint SHA and V4 ancestor SHA. An annotated/reformatted V5 copy or later closed V5 phase cannot obtain another allocation. The claim additionally binds exact source payload/version, authorization, guard and sole ledger path; it is checked before every phase. A copied V6 ledger alias cannot run. A crash after the claim but before ledger publication leaves a blocked registration for explicit review; it never frees the allocation automatically.

The V5 owned-PID/start-time/UID/session checks, finite wait4 reaping grace, process-group checks, exact wait4 accounting, best-effort cumulative CPU/RSS/cgroup polling, RLIMIT_AS, supplementary RLIMIT_CPU, and finite phase wall limits are retained. The same `.runtime/pyscf-v4-global.lock` is used; old V5, V4 and V3 ledger locks are also acquired. RSS/cgroup sampling is a best-effort stop policy, not a precise hard RSS guarantee.

## Separate commands — templates only, not executed

The root must first inspect actual closure and costs, freeze the new guard and staged-worker sources, independently review the source, and write an actual root decision. No such real decision is created in this preparation directory.

```bash
.runtime/bin/python scripts/run_pyscf_continuation_budget_20261105_v6.py register \
  --ledger NEW_V6_REPORT/cpu_budget.json \
  --source-v5-ledger CLOSED_V5_REPORT/cpu_budget.json \
  --source-v5-sha256 ACTUAL_FINAL_CLOSED_V5_SHA256 \
  --authorization NEW_V6_REPORT/resource-protocol/authorization.json \
  --authorization-sha256 ACTUAL_EXPLICIT_AUTHORIZATION_SHA256 \
  --source-base-commit ACTUAL_FULL_40_HEX_REVIEWED_COMMIT
```

`register` performs no payload launch. It atomically publishes a new ledger without replacing an existing file. A subsequent independently requested bounded stage can use:

```bash
.runtime/bin/python scripts/run_pyscf_continuation_budget_20261105_v6.py run \
  --ledger NEW_V6_REPORT/cpu_budget.json \
  --phase UNIQUE_CLOSED_STAGE_NAME \
  --cpu-seconds ROOT_APPROVED_PHASE_CPU_SECONDS \
  --wall-seconds ROOT_APPROVED_PHASE_WALL_SECONDS \
  --threads 1 --memory-gib 24 \
  --log NEW_V6_REPORT/execution/UNIQUE_STAGE.log \
  -- .runtime/bin/python FROZEN_STAGED_WORKER.py EXPLICIT_REVIEWED_STAGE_ARGUMENTS
```

The actual allowed grant is reduced by remaining cumulative CPU, cleanup reserve, and effective remaining wall time. `run` does not register or expand a budget. A failed run still consumes its true cost.

## Authorization JSON field contract

This is a field contract, not an enabled authorization file. Root must provide all actual values and the exact new guard SHA. Do not substitute synthetic test timestamps, test costs or example deadlines for actual evidence.

- `schema`: `research4-v6-continuation-authorization-v1`
- Explicit booleans: `root_approved_for_execution: true`, `no_budget_reset: true`, `no_automatic_budget_growth: true`.
- `v6_supervisor_source_sha256`: frozen actual V6 source SHA.
- `continuation_ledger_path`: the sole canonical absolute new V6 ledger path; copies and subsequent fresh registrations for the same original budget are forbidden.
- `authorization_text`, `research_authorization_evidence`: the explicit bounded decision and the user's actual authorization text/evidence. No generated user-grant timestamp.
- `research_authorization_deadline`: timezone-aware, conservative endpoint selected from actual evidence; `created_at_utc`: actual root decision time.
- Optional `research_authorization_record`: `{ "path": "...", "sha256": "..." }` to bind an existing evidence file.
- `source_v5_ledger_path`, `source_v5_ledger_sha256`, `source_v5_runner_sha256`, `source_v5_consumed_cpu_seconds`: exact final closed V5 values.
- Exact original fields: `original_v5_created_at_utc`, `original_v5_run_deadline_at_utc`, `original_v5_quantum_cpu_cap_seconds`, `original_v5_run_wall_cap_seconds`, `original_v5_authorization_path`, `original_v5_authorization_sha256`.
- `source_base_commit`: complete reviewed 40-hex source commit.
- Explicit integers: `additional_cpu_seconds` and `deadline_extension_seconds`, each 0–43,200; `max_quantum_cpu_seconds = old_cap + additional`; `max_run_wall_seconds = old_wall_cap + extension`.
- Phase limits: `max_phase_cpu_seconds` ≤122,400; `max_phase_wall_seconds` ≤43,200; `max_threads` ≤4; `max_memory_gib` =24 (virtual-address cap).
- Exact inherited guard values from old V5 authorization: `guard_semantics`, `rss_stop_bytes`, `cgroup_kill_bytes`, `monitor_interval_seconds`, `memory_kill_grace_seconds`, `expected_cgroup_memory_max_bytes`, `actual_cpu_cleanup_reserve_seconds`, `cpu_guard_kill_grace_seconds`, `actual_cpu_guard_source`.
- Exact inherited V3 bank fields: `prior_closed_ledger_path`, `prior_closed_ledger_sha256`, `prior_consumed_cpu_seconds`, `prior_budget_is_historical_only`.

## Worker and producer interfaces

New ledger schema: `research4-pyscf-continuation-budget-v6`.

New live-parent proof schema: `research4-pyscf-continuation-supervisor-proof-v6`.

`verify_resource_supervision(proof_path=None, *, expected_threads=None)` checks the real current parent identity, source SHA, command, limits, proof, and corresponding unique RUNNING V6 ledger row before a new worker imports scientific libraries. It also rechecks the complete closed-V5 SHA, original prefix and explicit allocation.

`healthy_closed_row(row)` requires actual `EXITED`, exit code0, finite known CPU, end timestamp, no residual processes, and all CPU/memory/wall/monitor/hardkill flags false.

`verify_closed_phase(evidence, *, expected_worker_path, expected_worker_sha256, expected_result_schema, allowed_result_statuses, expected_output=None, expected_snapshot_sha256=None)` is a stdlib-only gate for a new completed stage. It binds actual frozen worker source, real V6 proof, healthy closed producer row, actual command/output path, closed log SHA, stage/result/snapshot and native-driver restoration. A different consumer may currently be RUNNING; the producer itself must already be closed and healthy. Old V5 snapshot and RHF-unit producers continue to be checked by the frozen V5 verifier, never relabelled as V6.

## Scientific return boundaries — preparation only

The separate staged-worker work is intended to return normally after (1) a complete native full RKS partial Hessian, (2) complete full-system H1AO, or (3) full original CPHF/global Hessian/modes. The first two stages have frequency status NOT_RUN and require healthy producer closure before reuse. A failed or resource-killed source cannot qualify its SAVE_ONLY partial/H1 output automatically. Missing or incomplete native output cannot be reconstructed by declaring a stage successful.

All stage caches must bind the same original geometry, full immutable MF/MO/grid snapshot, method and native implementation, with producer proof kept separately from stable physical binding. Full global assembly must use fresh copies of the partial Hessian (stock PySCF mutates it), all atoms, and add nuclear repulsion once. Actual water/ethanol whole-kernel cross-process validation is required before scientific Spinosin use. No such new scientific gate is claimed here.

## Validation status

Only stdlib synthetic tests, source compilation and static source identity checks are appropriate while V5 quantum work remains active. Synthetic fixtures use temporary files, fake costs and fake evidence; they do not read the active V5 ledger, invoke actual subprocess payloads, import numerical packages, or authorize a real allocation.

Quantum calculation / continuation registration / real authorization / new scientific results: **NOT_RUN**.

