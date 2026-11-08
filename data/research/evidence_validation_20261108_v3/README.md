# Evidence-driven validation: new independent90,000 CPU-second grant

Protocol and scientific limits are in
`docs/EVIDENCE_DRIVEN_VALIDATION_20261108.md`; cross-branch evidence is in
`docs/CROSS_BRANCH_EVIDENCE_AUDIT_20261108.md`.

The finite matrix is15 matched-feature jobs and315 representation/transfer
jobs. It uses only999 master TRAIN records, primarily300 pure-acid records.
CAL334/PRED339 are not exported. The source dataset, labels and roles are not
changed. This is exploratory internal validation, not a new external test.

The new budget is88,000 inner+1,000 preparation+1,000 controller CPU seconds.
Previous grants and failure costs are not reset. The new independent Git branch
is `research/evidence-driven-validation-90000-20261108`; main is untouched.

`protocol.json`, `jobs.json`, `input_lock.json` and approved resource config
are preparation evidence. Real execution is proven separately by
`resource-control/launch-*.json`, root launch receipts, actual PID/birth/boot
checks and both cumulative `state.json` ledgers. Do not call prepared jobs
completed. Admission WAIT is also not a completed fit.

The new admission stage waits for old heavy summaries/backups and adequate
resources. Its bounded memory recovery can stop only its own verified new
children. The old queue is never adopted or signalled by the new supervisor.
Its separate memory-pause recovery forensics are preserved in
`old-queue-recovery-002/`; old science remains charged to the old grant.

Closed new results are under `runs/`; compact readouts under `snapshots/`.
Only complete repeated5-fold groups may be called development candidates.
Latest compact summaries have priority in serial Git backups. Some trained
preprocessor NPZ caches remain local with hashes; Git contains source/data
needed to rebuild them, but not an entirely self-contained inference bundle.

Read-only status from this independent worktree:

```bash
/workspace/.venvs/research4-species-relaunch-20261108/bin/python -B scripts/species_background_budget.py status --state-dir data/research/evidence_validation_20261108_v3/control
```

No timed chat reports or main merges are scheduled. A detached process can
continue while the cloud host survives; host replacement is not prevented.
Unclean state recovery requires identity/cost verification and never refunds
unknown CPU use.
