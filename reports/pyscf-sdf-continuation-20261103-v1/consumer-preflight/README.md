# Resumable full-Hessian consumer: software qualification

Frozen source: `scripts/pyscf_resumable_hessian_worker_20261103.py`, SHA-256
`d23c4a3185aa4f666e22d671034728c1ca2443676a05fc8bec0587d6c8d6bf03`.

20 pure metadata/NumPy tests passed. Independent review also checked the actual
PySCF factory/core method identities with numerical kernels mocked to fail if
called, and rejected custom hcore/X2C modifications. These tests are **not** the
real cross-process water/ethanol quantum gate. Only root launches actual quantum
calculations through `run_pyscf_actual_cpu_budget_20261101.py`.

## Three separate phases

```sh
.runtime/bin/python scripts/pyscf_resumable_hessian_worker_20261103.py \
  --phase restore-gradient-check \
  --snapshot SOURCE_PHASE/exact-mf-snapshot \
  --snapshot-manifest-sha256 SNAPSHOT_SHA \
  --output NEW_GRADIENT_CHECK_DIRECTORY --threads 4
```

This restores the exact source MF without SCF/grid rebuild, computes a true
same-state gradient, compares every component with the stored source gradient
using a fixed `1e-8 Eh/Bohr` tolerance, checks stationarity, and records Linux
boot identity. This gate is reused across subsequent segments on the same boot;
it is not rerun every ten atoms.

```sh
.runtime/bin/python scripts/pyscf_resumable_hessian_worker_20261103.py \
  --phase stock-hessian-reference \
  --snapshot SOURCE_PHASE/exact-mf-snapshot \
  --snapshot-manifest-sha256 SNAPSHOT_SHA \
  --gradient-check-result GRADIENT_PHASE/result.json \
  --gradient-check-result-sha256 GRADIENT_RESULT_SHA \
  --output NEW_STOCK_REFERENCE_DIRECTORY --threads 4
```

The stock reference **does not enter the incremental partial-helper context**.
It uses the frozen reviewed V8 complete stock RKS/global Hessian with B8 response
blocks, native four-thread policy and 3000 MB scoped Hessian memory. The native
full-Hessian response workflow is not replaced by fragment normal modes.

```sh
.runtime/bin/python scripts/pyscf_resumable_hessian_worker_20261103.py \
  --phase hessian-segment \
  --snapshot SOURCE_PHASE/exact-mf-snapshot \
  --snapshot-manifest-sha256 SNAPSHOT_SHA \
  --gradient-check-result GRADIENT_PHASE/result.json \
  --gradient-check-result-sha256 GRADIENT_RESULT_SHA \
  --rhf-cache outputs/DEDICATED_EXACT_STATE_CACHE \
  --max-new-units 10 --output NEW_SEGMENT_DIRECTORY --threads 4
```

Use `--max-new-units 2` for the intended water control and `5` for ethanol, so
the real control forces multiple closed processes. All source snapshot,
molecule, MO and actual grid bytes remain bound. Stock RKS XC/H1/CPHF/global
assembly continues only after every RHF diagonal/row unit is complete. The
existing V8 RKS/H1/response intermediate caches remain **SAVE_ONLY**.

## Status and provenance gates

- `DONE`: a successful restore-gradient check, with no frequencies computed.
- `CHECKPOINT_SEGMENT_COMPLETE`: a normal finite stop with complete saved RHF
  units; exit 0, `frequencies_status=NOT_RUN`, full Hessian false.
- `FREQUENCIES_COMPUTED`: real complete full Hessian and `3N-6` normal modes;
  signed negative frequencies are retained. Positive-frequency minimum
  qualification is an independent boolean, not inferred from process exit.
- `PARTIAL`: failure/interruption, exit 2; a hard kill may prevent result writing.

Every restored old unit is checked against its actual immutable producer proof,
closed healthy EXITED/0 V4 row, measured CPU, parent log, matching result,
reviewed entry source and unchanged source MF. A self-reported execution JSON is
insufficient. Producer execution evidence is separate from the immutable source
MF binding. Core Hamiltonian and gradient hooks must be standard unmodified
PySCF methods; X2C, solvent/MM decorators, pseudopotentials and ECPs are rejected.

An interrupted or unclean producer is blocked by default. Root may explicitly
approve exact record hashes through `--forensic-unit-release PATH` plus its
expected SHA. Such an override records the original failed/unknown parent check
and is labelled `EXPLICIT_ROOT_FORENSIC_RELEASE`; it never rewrites the original
ledger, invents CPU/time or claims healthy closure. Normal operation does not
require or create such a release.

The full mode table records both cm^-1 and THz and keeps IR intensities
`NOT_COMPUTED`. No theoretical result enters an existing classifier.
