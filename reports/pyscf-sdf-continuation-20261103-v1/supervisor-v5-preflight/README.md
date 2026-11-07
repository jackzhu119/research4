# V5 supervisor repair and same-pool migration

Frozen source: `scripts/run_pyscf_actual_cpu_budget_20261104_v5.py`, SHA-256
`33f1dc3db5280f16c9898c561e345143614730dc052b171b4f11fbbb079cdeee`.
The old V4 supervisor, state/consumer workers, core and historical ledgers remain
unchanged. This preflight ran 24 software tests; no quantum payload was launched.

## Actual defect and bounded repair

V4 could observe the original process-group leader as gone/Z while a nonblocking
`wait4` was briefly not ready. It treated this normal exit transition as an RSS
monitor failure, even when the eventual exit was 0 and the science result was
DONE. The three old failure rows remain failure rows, with their actual costs.

V5 independently rechecks the owned child's PID/start/UID/session/group identity.
Only confirmed Z/X/x or gone states can enter a terminal reaping grace of at most
0.5 seconds. A live R/S/D leader, permission failure, unstable identity or malformed
observation still fails closed; missing live RSS is never replaced by zero.
During the grace, cgroup and wall/deadline checks continue, and the last cumulative
CPU high-water mark is retained. Reaping an exit does not erase an already
observed cgroup violation or unexpected process-group member.

Real monitor failures leave the normal loop for bounded owned-child cleanup.
Signals use rechecked group identity or the original unreaped child's pidfd;
discovered/unrelated PIDs are not signalled. If final wait4 cannot be obtained
within bounded cleanup, the row becomes `UNRESOLVED_WAIT4_REAP_FAILURE`, with
actual CPU unknown and a sampled lower bound. The budget is then blocked for
explicit forensic review; no final CPU value or successful closure is invented.

CPU/RSS sampling remains best effort. This repair does not claim a kernel hard
RSS limit, exact scheduler timing, or that every possible exit race is validated
by pure fixtures. Root must perform ordinary short process validation before QM.

## Migration preserves the existing pool

The migration command is:

```sh
.runtime/bin/python scripts/run_pyscf_actual_cpu_budget_20261104_v5.py \
  --ledger NEW_V5_LEDGER.json \
  --migrate-from-closed-v4 ORIGINAL_V4_LEDGER.json \
  --migrate-from-sha256 ORIGINAL_V4_LEDGER_SHA \
  --authorization MIGRATION_AUTH.json --authorization-sha256 AUTH_SHA \
  --source-checkpoint ORIGINAL_CHECKPOINT \
  --source-checkpoint-sha256 ORIGINAL_CHECKPOINT_SHA \
  --source-base-commit FULL_REVIEWED_SOURCE_COMMIT
```

`--initialize` is forbidden. The destination must not exist. Migration validates
the immutable original V4 runner and its closed ledger/auth, all known per-phase
CPU costs, and the independently pinned original V3 ledger. It copies all old
phase JSON values unchanged. The current original pool contains 13 closed phases,
3605.273693 actual CPU seconds, a 129600-second CPU cap and an original deadline
`2026-11-05T01:20:50.391742+00:00`; these do not reset or expand. Original creation
time, wall cap, remaining budget and old failures also remain preserved.

Migration and all V5 phases share the existing `.runtime/pyscf-v4-global.lock`,
the original V3 prior-budget lock, and additionally the migrated V4 ledger lock.
The old V4 ledger bytes must remain equal to the migration SHA. Later V5 phases
must keep the complete old phase prefix exactly intact.

The new authorization uses schema
`research4-v5-budget-migration-authorization-v1`, sets
`migration_preserves_existing_budget=true`,
`new_budget_is_separately_authorized=false`, `prior_budget_preserved=true`, and
binds `migrate_from_closed_v4_path`, `migrate_from_closed_v4_sha256`,
`v5_supervisor_source_sha256`, `original_created_at_utc`,
`original_run_deadline_at_utc`, `original_quantum_cpu_cap_seconds`,
`original_run_wall_cap_seconds` and `original_consumed_cpu_seconds`.
All prior guard, source-checkpoint, source-base and V3 bindings are still required.
The user's later 72-hour research authorization does not silently change this pool.

The runtime schemas are `research4-pyscf-actual-cpu-budget-v5` and
`research4-pyscf-actual-cpu-supervisor-proof-v5`. Frozen V4-only workers reject
them by design; separate explicitly versioned V5 worker adapters are needed.

## Test scope

Pure tests cover delayed zombie/gone reaping, each terminal state, live/malformed/
unreadable identity rejection, identity changes, extra group members, 0.5-second
expiry, finite cleanup, known-failure cost preservation, migration cap/deadline/
sum invariants, and refusal of fresh initialization. Two tests execute the actual
monitor loop using a fake Popen and wait4 sequence: the legitimate exit race
closes healthily with actual wait4 CPU added to the prior cost, while a permission
failure still fails closed and retains the measured final cost. No process is
launched by these fixtures.

One development-only test initially demanded exactly one pidfd containment call;
the implementation also repeats containment during final cleanup. The assertion
was corrected to require at least one call and verify all calls target the same
owned pidfd. No scientific results or production guard semantics were changed
to satisfy that test.
