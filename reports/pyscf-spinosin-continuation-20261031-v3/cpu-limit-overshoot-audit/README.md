# Closed optimization CPU-limit overshoot audit

**DONE — operational evidence audit. Original optimization remains actually
DONE with a converged fresh SCF and passing fresh nuclear gradient.** Recovery
is **NOT_RUN: no recovery is needed**. The ongoing frequency phase was not
stopped, signaled, retuned or inspected through its checkpoint. No quantum
evaluation was launched by this audit.

## Actual closed evidence

| Quantity | CPU seconds |
|---|---:|
| Phase grant / configured kernel hard limit | 21600 |
| Configured kernel soft limit | 21480 |
| Actual parent wait4 user+system cost | **21681.889694** |
| Worker `time.process_time()` interval | 21681.306838823 |
| Actual wait4 above configured hard value | **81.889694** |
| Percentage above configured hard value | **0.3791189537%** |
| Parent-minus-worker startup/shutdown interval | 0.582855177 |

The exact excess is already charged in the original cumulative ledger; no CPU
was subtracted and no row or limit was rewritten. The closed row is EXITED0 with
no wall, memory, monitor or explicit hard-kill stop. Its launch proof and worker's
actual startup observation both contain RLIMIT_CPU `(21480,21600)`. The frozen
supervisor source actually calls `resource.setrlimit` before exec, and the worker
checks real limits before importing PySCF. The reviewed PySCF/geomeTRIC Python/C
source search found no code changing RLIMIT_CPU. These observations establish
configured and startup limits, not a continuous proof that no third-party native
component changed any state later.

Completed phases summed to21869.373444 CPU seconds at this audit, including the
full optimization cost. The new frequency's configured grant is86400 CPU seconds.
That configuration sum is108269.373444, leaving64530.626556 seconds below the
original172800-second (48 CPU hour) pool. It is **a configuration calculation,
not a measurement of the active phase or a guarantee of future enforcement**.
The current source, limits, requested frequency budget and deadline are unchanged.

## Mechanism assessment

The exact reason for the81.889694-second difference is **UNPROVEN**. The ended
worker did not preserve a final raw CPUCLOCK_PROF value or a kernel signal-delivery
trace. Python's delayed handling of a soft SIGXCPU inside a native call cannot
by itself explain failure of an uncatchable kernel SIGKILL.

Independent methods review retrieved official Linux tag v6.18.57 source, matching
the reported version tag but not proving identical vendor patches or compile
configuration. That source checks RLIMIT_CPU against PROF (raw user+system), while
process CPU time and wait4 accounting can use scheduler-runtime-adjusted values.
This supplies a concrete candidate clock-accounting mechanism. It does not prove
that it caused this particular long-run difference. Source and kernel evidence
is separately preserved in `../cpu-overshoot-kernel-audit/`.

The current parent monitor samples RSS and cgroup memory, plus wall time. It
relies on the kernel for CPU-limit signaling and **does not independently stop
the job using sampled process CPU**. Future separately reviewed supervisor work
could add an aggregate CPU guard and an empirically justified boundary allowance.
No such change was made to the frozen/live supervisor in this audit. The present
limits should be described as configured kernel limits, not a mathematically
exact guarantee on wait4-charged CPU.

## Actual low-cost ordinary probes

Two bounded C/OpenMP programs used the same environment, one and four threads,
with actual RLIMIT_CPU soft1/hard2 seconds. They perform elementary arithmetic,
contain no PySCF imports or quantum integrals, self-stop at2.25 process CPU seconds
if necessary, and have an independent10-second parent wall backstop.

| Native threads | Observed SIGXCPU | Exit code | Actual wait4 CPU | Parent wall kill |
|---:|---|---:|---:|---|
| 1 | Yes | -9 | 2.020569 s | No |
| 4 | Yes | -9 | 2.020243 s | No |

The repeated termination near2 CPU seconds is consistent with a working kernel
CPU-limit path in this short test. It does **not** reproduce or explain the
six-hour optimization overshoot. These tests must not be presented as a proof of
strict long-run CPU accounting or scientific model performance. Compilation and
both ordinary probes consumed4.259503 actual child CPU seconds in total, recorded
separately from the quantum ledger.

`native_cpu_limit_probe.c` and compile command/log preserve the actual inputs;
the binary remains under ignored `outputs/` with its SHA recorded. Logs retain
observed signal output and actual initial process/PROF/SCHED clock reads. The
process was killed before final clocks could print, so no final raw PROF value
is invented. Linux clock encodings were checked against the retrieved source.

## Reproducible evidence

- `closed_opt_row_snapshot.json`: a derived copy of the one ended ledger row,
  not an asserted final snapshot of the still-active whole V3 ledger.
- `closed_opt_cpu_audit.json`: actual costs, overshoot, exact closed-file hashes,
  unchanged budgets and explicitly unproven causal interpretation.
- `ordinary_probe_results.json` and ordinary logs: actual non-quantum microtests.
- `audit_closed_opt_cpu.py`: read-only reconciliation; refuses overwrite and
  never reads the active frequency checkpoint.

All prior records remain intact. The operational overshoot is recorded alongside
the genuine successful optimization; neither finding is used to fabricate or
erase the other.
