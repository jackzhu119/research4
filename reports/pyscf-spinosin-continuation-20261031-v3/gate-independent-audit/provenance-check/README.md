# Independent closed-run provenance gate

Status: **PASS_WITH_STALE_PROGRESS_NOTE**. 437 checks; 14 saved gradient arrays recomputed. No PySCF import, quantum calculation, integral or numerical Hessian/mode matrix work was performed.

Only the two named closed V3 rows were audited. Later ledger phases may be appended; the selected closed rows were re-read and verified unchanged. Existing results and source files were not modified.

| Closed run | wait4 CPU seconds | Payload process CPU seconds | Fresh atom RMS | Fresh atom max |
|---|---:|---:|---:|---:|
| water-native4-validation-001 | 5.378027000 | 4.729362885 | 0.00013637558592029502 | 0.00023043787845722363 |
| validation-ethanol-native4-001 | 182.105723000 | 181.424017566 | 7.6396643735551008e-05 | 0.00015358569894336033 |

The closed supervisor CPU total is187.483750000 seconds. Proof-file SHA matches both result and ledger; embedded proofs, parent supervisor source SHA, worker/helper source SHA, resource contracts and declared file checksums were independently checked against actual files. ActualCPU is taken from closed wait4 accounting; it was not remeasured by this audit.

Both final results declare optimizer/SCF convergence and fresh-gradient thresholds passed. Execution logs contain optimizer `Converged!`; final converged SCF log energy matches result within1e-12 Hartree. All14 printed gradient blocks match saved arrays within5.01e-11 Hartree/Bohr. Fresh gradients pass atom RMS≤3e-4 and atom max≤4.5e-4.

**Nonblocking stale record:** both `progress.json` files still say PARTIAL. Final `result.json`, closed EXITED/exit0 ledger rows and completion events consistently show completion. Initial input flags and iteration-time SCF callback flags are retained separately; they were not relabeled as final failures.

This is a provenance/gradient gate only. It does not establish measured CPHF residuals, active OpenMP worker counts, whole-Hessian memory safety or a new Spinosin result. Numerical Hessian/mode validation is outside this component. See `audit.json`, `log_excerpts.md`, `closed_ledger_rows.snapshot.json` and `input_hashes.json`; `index.json` binds this new directory.
