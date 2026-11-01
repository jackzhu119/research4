# Closed optimization CPU overshoot: source and clock audit

**Confirmed overshoot; exact cause UNPROVEN. Scientific optimization DONE is independently verified.**

| Quantity | Actual value |
|---|---:|
| Requested phase CPU / recorded hard limit | 21600 s |
| Recorded soft limit | 21480 s |
| Parent `wait4` actual CPU | 21681.889694 s |
| Worker-local process CPU | 21681.306838823 s |
| `wait4` CPU above requested hard value | **81.889694 s (0.37911895%)** |
| Parent minus worker timer | 0.582855177 s |

The closed phase is `spinosin-native4-optimization-001`, exit 0 with actual optimization and fresh-gradient checks passing. Its archived launch proof and worker observation both show CPU limits `[21480,21600]`. The frozen worker/supervisor SHA values match their actual files. Parent flags show no wall, memory, monitor or explicit hard-kill intervention. No historical row, actual CPU value, limit or active frequency parameter was changed by this audit. The measured phase did **not** meet a strict 21600-second bound expressed in the same `wait4` accounting used by the ledger. The separately authorized global 48 CPU-hour cap was not exceeded by completed totals at this snapshot.

The related [ordinary-process audit](../cpu-limit-overshoot-audit/README.md) includes short C/OpenMP CPU-limit probes. Those probes are not molecular calculations and do not establish a precise six-hour enforcement guarantee. The [scientific completion audit](../opt-native4-completion-audit/science-point-check/README.md) verifies all five optimizer evaluations, the additional fresh gradient, optimized coordinates, SCF evidence and molecular stereochemistry independently of resource compliance.

Seven public source files were fetched through the GitHub connector and their original Git blob identities were verified. Full source snapshots, URLs and hashes are retained. The Linux source tag `v6.18.57` matches this environment's `uname` version label; this does not prove the vendor build has no patches.

Relevant source evidence:

- `linux_posix_cpu_timers.c:1049–1064`: RLIMIT_CPU is checked against `CPUCLOCK_PROF`, with hard-limit SIGKILL and soft-limit SIGXCPU logic.
- `linux_posix_cpu_timers.c:216–218`: profiling CPU samples use accounting `stime+utime`; scheduling samples use `rtime`.
- `linux_posix_cpu_timers.c:1671`: public `CLOCK_PROCESS_CPUTIME_ID` maps to the scheduling clock. `cpython_timemodule.c:1348–1352` uses that public clock, corroborated by this interpreter's actual `time.get_clock_info('process_time')`.
- `linux_sched_cputime.c:527–542,546–617`: the generic adjustment path scales accounting user/system values against scheduler runtime and preserves monotonicity. `linux_sys.c:1917` uses adjusted group accounting; `linux_exit.c` routes wait rusage through `getrusage`.
- `linux_posix_timers.h:19–23` and `linux_posix_timers_types.h:27–29`: current-process raw PROF clock encoding is `-8`; SCHED is `-6`.

The actual embedded `/proc/config.gz` confirms `CONFIG_VIRT_CPU_ACCOUNTING_GEN=y`, `CONFIG_POSIX_CPU_TIMERS_TASK_WORK=y`, `CONFIG_NO_HZ_FULL=y` and `CONFIG_HZ=250`. Selected lines and hashes are saved in `audit.json`. These establish compiled options, not every CPU's actual full-dynticks runtime state. Missing `NATIVE`/`TICK_CPU_ACCOUNTING` lines are recorded as absent, rather than falsely quoted as explicit disabled lines.

The differing clocks/accounting paths and timer-delivery mechanisms are source-supported candidates. The old worker's final raw PROF counter and a complete midrun syscall trace were **not recorded**, so the precise reason for the 81.889694-second difference remains unproven. Python's delayed delivery of a catchable signal alone is not a sufficient explanation for hard SIGKILL behavior. A bounded textual scan found no CPU-limit or SIGXCPU manipulation in the installed PySCF/geomeTRIC sources; it does not exclude every unobserved native/runtime action.

This auditor generated no CPU workload, imported no quantum package and evaluated no SCF, gradient or Hessian. It read only the closed optimization result, its archived proof/log, source files, current clock metadata and kernel configuration. The active frequency directory/checkpoint was not read or signalled. The ledger remains mutable for that later phase; only the selected completed row is snapshotted here.

Future supervision can add a separately declared measured-CPU polling margin. No such change was applied to the current run. Recovery preparation remains **NOT_RUN** because the original optimization completed successfully; no V4 permission or V9 fallback artifacts were created.
