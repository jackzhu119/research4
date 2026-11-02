# Independent V4 resource authorization template

Status: TEMPLATE_NOT_EXECUTION_AUTHORIZATION. This directory does not initialize a scientific ledger or start a calculation. Root must finalize a separate dated `authorization.json`, set its explicit authorization flag, hash its actual bytes, and initialize once.

The user's continuation authorization is recorded verbatim in the template. The root-selected finite new budget is 129,600 actual CPU seconds (36 CPU hours), with each phase at most 122,400 CPU seconds (34 CPU hours), total wall 64,800 seconds and each phase wall at most 43,200 seconds. The closed V3 cost of 111,023.488913 CPU seconds is historical and is added when disclosing all research costs; it does not reduce or reset the new separately authorized pool. Earlier ledgers and their recorded overruns remain unchanged.

The budget supervisor uses process-birth cumulative `/proc/<TGID>/stat` user+system and waited-child CPU with the actual `SC_CLK_TCK`; native threads are already included and are not summed again. Only a single payload TGID is permitted. It refuses an unexpected live process-group member. Final `wait4` usage is the sole completed-phase charge. Sampling never subtracts the first observation. At a phase or new-pool boundary minus 120 CPU seconds it requests TERM; the configured escalation is at most one second. Polling, scheduling and signal delivery remain best effort. The supplementary profiling `RLIMIT_CPU` is not represented as an exact bound, in light of the recorded earlier overruns.

The address-space limit stays 24 GiB, the entire cgroup envelope stays 16 GiB, process-group RSS above 10 GiB requests TERM/KILL, and cgroup memory above 12 GiB requests immediate KILL. The guard samples each 0.1 seconds. AS is a virtual address-space limit; the sampled RSS guard is not a kernel hard RSS limit. No core, paid cloud service or GPU expansion is authorized here.

Each group signal is preceded by worker PID/start-tick/UID/PGID/SID checks. The fixed worker is its own process-group and session leader. The original Popen child's pidfd is used for containment on monitoring failure; a limited fallback can kill only the original unreaped child PID. A group signal still has a best-effort identity TOCTOU boundary. Live observations which cannot be trusted fail closed, with actual CPU retained via wait4. Lingering group members block another phase.

New and prior V3 ledger locks are held against concurrent use by their corresponding runners; this does not claim to lock every older V2 runner. Root owns serial launch control. The prior V3 lock is opened read-only and the old ledger is never appended. The source checkpoint is a warm start, not a verified Hessian restore. The old frequency calculation produced no completed partial Hessian; restoration remains NOT_IMPLEMENTED_NOT_VALIDATED.

Only the new pure-J candidate and source-frozen verification gates may be launched by root after this resource template is finalized. The physical molecule, basis, functional, grids and fresh-gradient requirements are outside this template and remain controlled by root's separate `ROOT_EXECUTION_PROTOCOL.md`. This document does not change those scientific requirements.

## Root CLI contract

Create a finalized new-root `authorization.json` from the template. Its `created_at_utc` must be a real UTC timestamp and `new_budget_is_separately_authorized` must be true. Bind the literal file hash; do not use the template hash as authority.

```sh
python scripts/run_pyscf_actual_cpu_budget_20261101.py \
  --ledger reports/pyscf-spinosin-pure-j-20261101-v4/cpu_budget.json \
  --initialize \
  --authorization reports/pyscf-spinosin-pure-j-20261101-v4/authorization.json \
  --authorization-sha256 ROOT_FINAL_AUTHORIZATION_SHA256 \
  --source-checkpoint reports/pyscf-spinosin-continuation-20261031-v3/spinosin-native4-optimization-001/calculation.chk \
  --source-checkpoint-sha256 aae612aca345048028f85bce27f647c01f0834da1f9d06ec6ca986c0a2fffe8c \
  --source-base-commit 0629d4c6a1496dc96b6852c07755b61c42efadea
```

Launch a fresh, non-reused phase/log name with `--phase NAME --cpu-seconds REQUEST --wall-seconds REQUEST --threads 4 --memory-gib 24 --log NEW_LOG -- COMMAND`. The supervisor independently clamps grants to remaining CPU and wall allowance and serializes phases. It publishes the actual V4 proof via `R4_RESOURCE_SUPERVISOR_PROOF`. The worker must call `verify_resource_supervision(..., expected_threads=4)` before importing/starting quantum work. No execution command for the unvalidated candidate is asserted here.
