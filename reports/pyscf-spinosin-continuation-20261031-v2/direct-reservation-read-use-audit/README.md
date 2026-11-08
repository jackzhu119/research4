# Direct-J reservation and source-visible payload audit

Status: **DONE — source/arithmetic audit only**. No PySCF import, compilation, integral, SCF, gradient or Hessian was run. Only this new directory was written; installed sources, frozen workers and earlier reports were preserved. This audit does not authorize a quantum launch or establish whole-Hessian memory safety.

## Finding

The approximate3.2GB native direct-J budget is a **per-thread virtual allocation policy**, shared among that thread's `n_dm` JKArray buffers. `nr_direct.c:407–425` divides the budget by `n_dm` and creates the buffers inside the OpenMP region. `nr_direct.c:255` uses `malloc`; it does not explicitly clear the full allocation. Padding and the separate integral/cache buffer mean3.2GB is nominal rather than an exact total.

`nr_direct_dot.c:180–193` appends and clears an output tile only when its block-pair key is `NOVALUE`. `NPdset0` writes exactly the requested elements (`np_helper.c:20–25`). The key is unique **between write-backs**; write-back resets the keys and stack (`nr_direct_dot.c:80–113`). For fullAO output with matching disjoint block partitions, a JKArray stack contains at most `ncomp*nao²` doubles. Later epochs reuse the same prefix.

## Exact payload arithmetic

These are conditional bounds on **private JKArray.data array payload accessed by the reviewed source**, excluding the other memory listed below. The762-AO dimension is task-supplied; the CPHF24-density case is a scenario, not a measured call.

| Case | n_dm | ncomp | Per-thread private payload | At most4 native threads |
|---|---:|---:|---:|---:|
| Pure-J gradient | 1 | 3 | 13,935,456 B | 55,741,824 B |
| CPHF24-density scalar J | 24 | 1 | 111,483,648 B | 445,934,592 B |

Formula: `threads * n_dm * ncomp *762² *8`. The pure gradient call shape follows `grad/rks.py:79–80`, `grad/rhf.py:214–218` and `scf/_vhf.py:438–463`. Actual driver `n_dm` includes operator/script replication; do not infer it solely from the number of Python density objects for arbitrary calls.

The C code's `#pragma omp parallel` (`nr_direct.c:414`) does **not** fix the team at4. The four-thread totals apply only if the runtime actually limits this single direct-driver team to at most4 threads. No live team size was measured here.

## Limits of the memory interpretation

The nominal four-thread reserve is12,800,000,000B (11.92GiB), before padding, metadata and other allocations. It is not a measured RSS, and the source-visible tile payload bounds are not physical-page or totalRSS bounds. Allocator behavior, page rounding and huge pages may change physical residency. Virtual-address or commit limits can still reject `malloc` while RSS is low.

Excluded memory includes shared full outputs (`nr_direct.c:307–318,371–372`), reordered density matrices (`276–302,373–374`), thread integral/cache buffers (`427–428`), keys/outptr and allocator metadata (`248–264`), Python/BLAS state, and independent GGA numerical-grid and CPHF `xs/ax` histories. No conclusion about safe complete Hessian execution, native4 performance or successful convergence follows from this audit.

## One resource snapshot

At2026-11-01T07:59:15Z, `df -hT /workspace/research4` reported an overlay filesystem of32G total,21G used and9.3G available (human-readable rounded units). `/sys/fs/cgroup/memory.max` was17,179,869,184B (16GiB); `memory.current` was2,493,706,240B. The latter is cgroup-wide usage, not quantum-workerRSS or a peak. The raw one-time snapshot is in `resource_snapshot.json`; it does not certify later resource availability.

## Evidence

`audit.json` records source hashes, exact arithmetic, evidence line ranges and all preconditions. `source_excerpts.md` preserves numbered excerpts rather than full source copies. The installed C files are `nr_direct.c`, `nr_direct_dot.c` and shared `nr_direct.h`; no `nr_direct_dot.h` is installed. Supplemental call/zeroing sources are hashed as well. `index.json` binds every other file in this new audit directory. A second agent independently cross-checked the control flow and arithmetic without running quantum code.
