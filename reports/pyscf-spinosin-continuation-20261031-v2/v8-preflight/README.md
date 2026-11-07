# V8 supervised parallel worker — pure preflight

Status: **DONE for engineering preparation and 14 pure tests; quantum calculations NOT_RUN here.**

The independent budget agent reviewed the startup-proof contract and V8 changes. The source is frozen at:

- Worker `scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py`: `acfe7708748ac237c9912aa9b1d98e047d842794fd0c6f08b1a67b993f5b277d`
- Tests `tests/test_pyscf_spinosin_parallel_hessian_worker.py`: `78b0ded68b4f23f0833cc90d82b985a61edebd1bd96f970c42efd2be67bf979c`

V7 and earlier frozen files were preserved. `pure_tests.log` and `pure_test_result.json` record the actual test command, timestamps, return code, source hashes and zero quantum computations.

The only computational policy changes are allowing native thread counts 1–4 and response atom blocks 1–8 within the newly authorized monitored environment. The SCF/gradient/direct-integral implementation, PBE/def2-SVP/grid settings, convergence thresholds, normal-mode checks, complete global Hessian assembly, and scoped 3000 MB MF/Hessian resource attributes remain unchanged from V7. The native driver's forwarding/restoration methods are AST-identical to V7; only its constructor allowlist and explanatory docstring changed.

Before any V8 execution, `verify_resource_supervision(proof_path=None, *, expected_threads=None)` requires a real launch proof from `scripts/run_pyscf_monitored_budget_20261031.py`, selected by `R4_RESOURCE_SUPERVISOR_PROOF`. It checks the live parent and child PID/start identities, parent command/source SHA, actual CPU and 24 GiB virtual-address limits, shared 16 GiB cgroup, the 10 GiB RSS and 12 GiB cgroup stop contract, monitor timing, and one RUNNING V3 ledger phase. The worker checks this before importing quantum dependencies. The standalone validation helper must explicitly invoke the same function before quantum work.

The 24 GiB limit caps virtual addresses. RSS polling is a **best-effort guard**, not a hard resident-memory limit; immutable startup proof does not establish future monitor health. The parent polls at up to 0.1 s intervals, terminates on 10 GiB process-group RSS, targets KILL within 1 s, immediately kills at 12 GiB cgroup usage, and fails closed on monitoring errors. OS scheduling can delay polling or signals.

The native allocation/read-use audit is in `../direct-reservation-read-use-audit/`. Its bounded explicit JKArray payload calculations do not bound total RSS. Real water and nine-atom multi-block numerical validation, followed by separately budgeted Spinosin work, remain necessary. This preflight reports no new molecular energy, gradient, Hessian or frequency.
