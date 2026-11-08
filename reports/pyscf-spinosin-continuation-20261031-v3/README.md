# Spinosin monitored direct continuation V3

This run is additive to the closed V2 batch. The user explicitly authorized
continued long calculations. The recorded authorization and closed V2 ledger
hash preserve earlier CPU accounting; budgets are not reset.

Actual completed validation:

- `water-native4-validation-001/`: complete water optimization and three positive
  vibrational modes under the monitored four-thread V8 worker.
- `validation-ethanol-native4-001/`: nine-atom optimization, fresh gradient,
  standard full Hessian/native1 versus B4 and B8/native4; all full-matrix
  comparisons and all 21 positive-mode checks passed.
- `water-backend-comparison/` and `gate-independent-audit/`: independently
  recalculated numerical comparisons, modes, source/log/proof and array checks.
- `gate-completion-snapshot/`: immutable ledger snapshot after these two gates.

The live `cpu_budget.json` is authoritative for the actual quantum worker.
`pipeline-native4-001/` records the bounded serial controller and immutable
source plan. It requests Spinosin optimization, then only after a passing
same-method optimization and fresh gradient requests Hessian/frequency work.
The controller never initializes more budget, retries, or changes parameters.
A requested stage is not a completed calculation.

The actual run started at 2026-11-01 08:20:14 UTC. No Spinosin vibrational modes
had completed at that startup snapshot. Water and ethanol are software/numerical
validation molecules, not acid-jujube constituents or new biological samples.
IR intensities are `NOT_COMPUTED` throughout.

The resource policy uses AS24GiB plus a best-effort RSS/cgroup monitor; the RSS
monitor is not a hard per-job resident-memory limit. The finite batch and phase
budgets, source provenance and all failures remain inspectable. See
[`docs/PYSCF_MONITORED_DIRECT_PROTOCOL_20261031.md`](../../docs/PYSCF_MONITORED_DIRECT_PROTOCOL_20261031.md)
and the current
[`continuation report`](../../docs/PYSCF_SPINOSIN_CONTINUATION_20261031.md).

Large Hessian intermediates remain in ignored `outputs/`; their manifests bind
dimensions, hashes and actual completed blocks. Checkpointing is SAVE_ONLY until
a separately validated loader exists. Do not advertise cross-process cache
recovery as implemented. Do not read an active HDF5 checkpoint.

The original spectra, historical classifiers, the 96.0234% internal accuracy,
the old pilot/V2 records, and `main` are unchanged.
