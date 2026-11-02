# V3 monitored resource supervisor preflight

New implementation: `scripts/run_pyscf_monitored_budget_20261031.py`.
Tests: `tests/test_pyscf_monitored_budget.py`.
Authorization template:
`docs/templates/PYSCF_MONITORED_RESOURCE_AUTHORIZATION_20261031.template.json`.

This is a separately authorized, bounded resource protocol. Old runner sources,
V2/pilot ledgers and calculation files remain unchanged. The template is clearly
marked **TEMPLATE_NOT_AUTHORIZATION** and cannot initialize a run until root fills
actual closed-source/checkpoint/ledger hashes and CPU totals in a separate file.
No real quantum worker or scientific ledger was started by this implementation
agent. Test children are ordinary tiny Python programs.

## Actual resource semantics

- CPU: at most172800 actual child CPU seconds per new batch (48 CPU hours), at
  most86400 per phase (24 CPU hours), four existing cores; no GPU/paid service.
- Wall: at most86400 seconds for the entire new batch (24h, including gaps), at
  most43200 seconds per phase (12h). Existing CPU soft/hard limits and60-second
  wall cleanup remain bounded inside the original deadline.
- `RLIMIT_AS`: at most24 GiB of **virtual address space**, not24 GiB RAM.
- Every0.1 seconds, sum RSS across the payload's process group. Do not count
  `/proc/PID/task` threads as separate processes. Shared pages between processes
  may be conservatively counted twice.
- Group RSS above10 GiB requests SIGTERM and a nominal SIGKILL within1 second;
  the target leaves one polling interval of timing margin. Cgroup memory above
  12 GiB requests immediate SIGKILL. Missing leader observation or a monitor
  read failure closes the phase and kills the payload; an exited-leader race is
  distinguished with a second nonblocking wait4.
- The actual existing cgroup ceiling must remain16 GiB. It is not enlarged.
  This is **BEST_EFFORT_RSS_GUARD_NOT_HARD_RSS_LIMIT**. Polling and normal Linux
  scheduling cannot prevent every transient overshoot or guarantee6 GiB free
  physical headroom. This UID cannot create a child cgroup hard RSS/memory limit.

The child arms Linux `PR_SET_PDEATHSIG(SIGKILL)` before executing any payload and
checks that its supervisor has not already exited. An unexpected supervisor
exception kills/reaps its child, records actual wait4 CPU and closes the phase as
PARTIAL; cleanup checks actual child exit state, not a potentially non-None
WNOHANG rusage object.

## Provenance and mutual exclusion

Initialization binds the **closed** V2 ledger's exact SHA, independently summed
actual CPU and source checkpoint SHA. A RUNNING prior phase, inconsistent CPU
total, changed prior bytes, changed runner source, reused phase/log or reset of
an existing ledger is rejected. Both initialization and execution hold the old
V2 runner's exact `.lock` file exclusively, preventing simultaneous old/new
phases. A separate global V3 lock prevents V3 ledgers from running concurrently.

Each child receives `R4_RESOURCE_SUPERVISOR_PROOF`. An exclusive immutable launch
file is written in preexec **after actual limits and parent-death protection are
armed**, before the worker executes. It binds actual supervisor/child PID and
start ticks, supervisor source SHA, ledger/phase, thread allocation, CPU/wall/AS
limits, RSS/cgroup policy and Linux cgroup membership. The parent adds its file
SHA to the RUNNING ledger after spawn. This proof is a startup contract, not a
claim that future monitor health or future quantum work has succeeded. V8 checks
its real parent/process identities, live limits and source before quantum imports.

CPU accounting uses actual wait4 user+system time. The scientific payload is one
Python quantum worker with native threads; this is not an arbitrary multiprocess
job scheduler. Any remaining process-group members after the main payload exits
are killed and the phase is marked PARTIAL. No result is invented after a native
C exit; ended-ledger/progress evidence remains separate from worker result JSON.

## Root execution sequence

1. Let the V2 phase terminate and have its existing parent reap and save wait4
   usage. Freeze the resulting V2 ledger bytes; do not edit them afterward.
2. Select and hash an actually closed source checkpoint. Do not hash an active
   checkpoint or substitute an unfinished geometry for a converged result.
3. Copy and complete the authorization template into the new V3 report root.
   Bind closed ledger usage/hash, source checkpoint and explicit new limits.
4. Initialize the new ledger with the exact source/auth hashes. Then run small
   water and verified9-atom engineering gates through this supervisor. A higher
   AS limit and startup proof do not themselves qualify native4 Spinosin work.
5. Only root launches quantum phases. Preserve every unsuccessful test and all
   phase CPU/memory/exit evidence; never reset the new ledger or modify old caps.

CLI form (replace capitalized placeholders with the actual frozen values):

```bash
python scripts/run_pyscf_monitored_budget_20261031.py \
  --ledger reports/pyscf-spinosin-continuation-20261031-v3/cpu_budget.json \
  --initialize \
  --authorization reports/pyscf-spinosin-continuation-20261031-v3/authorization.json \
  --authorization-sha256 ACTUAL_AUTHORIZATION_SHA256 \
  --source-checkpoint ACTUAL_CLOSED_CHECKPOINT_PATH \
  --source-checkpoint-sha256 ACTUAL_CLOSED_CHECKPOINT_SHA256 \
  --source-base-commit aef64c65295bee03764da1dd72c21d32ed3e6c24
```

The subsequent phase command uses `--phase`, `--cpu-seconds`, `--wall-seconds`,
`--threads 4`, `--memory-gib 24`, a fresh `--log`, then `--` and the actual worker
command. Run roots and log names are supplied by root, never inferred by scanning
historical checkpoints.
