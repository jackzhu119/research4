# Direct resumption: live monitoring handoff

Snapshot: 2026-11-04 04:08 America/Los_Angeles (12:08 UTC). This file is a dated observation, not a terminal scientific result.

The user explicitly requested continuous monitoring and repair until completion. Keep the current scientific process running; do not finish with a claim that frequencies exist before actual completion.

- Current entry: frozen `scripts/run_pyscf_serial_segments_20261104_v5.py`, SHA `a01b6a79f457bca4b0ab90e63b45a3bb524b62fc586fab26905d6b43a2aba1d7`.
- Current tool session: `88969`; root controller PID initially 9504. The active child changes at segment boundaries; obtain it from the live ledger.
- Current output: this `direct-resume-001/` directory. Read `progress.json`, `events.jsonl`, the live `../cpu_budget.json`, and actual `/proc` state. Do not confuse the initialized controller `status: READY` with liveness; its event and supervised child establish current execution.
- Latest qualified prefix: **81/150** units. Active phase at snapshot: `v5-spinosin-resumable-segment-010`. Five additional markers are observations inside an unclosed segment, not qualified frequencies or a completed Hessian.
- Ledger cost after 81 closed units: 34929.664711 actual CPU seconds, cap 129600, remaining 94670.335289 before the active phase; original deadline remains 2026-11-05T01:20:50.391742Z. No reset or budget enlargement occurred.
- Actual start authorization: `../resource-protocol/direct-resume-001.json`, SHA `ffd1d632e94e4290b5c573d3f5bd3fbc7fe2417c309c1636b561f19af06af3cc`.
- Ongoing root observations: `../execution/root-observation.jsonl`; console: `../execution/direct-resume-001.root-console.log`.

## Why this is a separate continuation

The original backup job ended after a healthy 51-unit segment because GitHub returned an Internal Server Error pushing commit `009090db84641b86c136882344e01826e6c980f9`. Root successfully pushed that exact existing commit afterward. The original `auto-segments/job-status.json` and completion observer correctly remain PARTIAL / BLOCKED_PARTIAL_JOB; they are historical records, not the current execution status. The old observer has exited and must not be restarted against the new controller. The current controller has no Git dependency during quantum execution.

Independent review verified the 51-unit archive and healthy producers before this resumption. Subsequent segments 007, 008, and 009 closed cleanly and were sealed by the original controller. Keep the same snapshot, gradient receipt, cache, source hashes and budget ledger; do not launch a concurrent quantum worker or read active NPY/HDF5/CHK files.

## Actions after the current controller exits

1. Inspect the actual controller `result.json` and exit code. Require a healthy closed terminal producer, 150 units, complete global Hessian and 219 computed modes before describing the frequency stage as complete. Preserve any failure and actual CPU accounting; investigate before scientific retries.
2. For a valid DONE result, read its `final_result.path` and actual SHA. Execute the frozen analyzer `scripts/analyze_spinosin_v5_modes_20261104.py` (SHA `a2e3588c6dea121be58e4a69afb137bc6e5092199ca4180937fcdb34f6f57154`) with `--result PATH --result-sha256 SHA --output reports/pyscf-spinosin-resumable-20261104-v5/final-mode-analysis`. The output directory must be new. Its source/gradient/closure gates precede target array reads.
3. Preserve every negative/zero mode. Only if the actual analyzer qualifies the full minimum may `scripts/analyze_spinosin_group_participation_20261104.py` (SHA `b997a51f1d57b29cb2f05b67209bf993d1882af504a4af164c6912ed497c2dcd`) run with `--analysis CLOSED_A2_DIR --analysis-index-sha256 ACTUAL_INDEX_SHA --output NEW_DIR`. It reports mass-weighted displacement shares, not IR intensities, energies or confirmed chemical assignments. IR remains NOT_COMPUTED.
4. Generate a new final continuation report. Preserve the older `docs/PYSCF_SPINOSIN_FINAL_STATUS_20261104.md` BLOCKED snapshot and explicitly link the subsequent result. Use source peak comparisons descriptively; no classifier changes or claimed accuracy increase from chemistry.
5. After quantum phases are closed, commit scoped evidence and push only the research branch. For a transient server push failure retry the same existing commit, never force-push or rerun publication to fabricate another scientific result. Main remains untouched.

Prepared companion evidence: `docs/RESEARCH4_PAPER_EVIDENCE_UPDATE_20261104.md`, the static `reports/external-evidence-20261104-v1/mode-interpretation-preparation/` atom map, frozen analyzer/observer preflights, and the new group-participation preflight/review. These are not proof that Spinosin frequencies have already completed.
