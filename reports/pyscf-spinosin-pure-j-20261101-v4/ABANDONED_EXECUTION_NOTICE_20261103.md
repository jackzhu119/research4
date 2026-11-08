# Archived interrupted execution

On the replacement cloud host (observed 2026-11-04 UTC), the old supervisor and
quantum payload no longer exist. The persisted `RUNNING` row is an unchanged
historical snapshot, not a current live-process assertion. No terminal wait4
record, final CPU total, exit code, exact end time or cause was recovered.

The old SCF checkpoint, molecular coordinates and saved gradient have been
independently inspected read-only and match their original recorded hashes and
array fingerprints. This qualifies the SCF component only. No completed RHF
helper, partial/full Hessian or Spinosin frequencies were found.

Do not run the old finisher: its live-phase/deadline binding belongs to the
abandoned process. Do not synthesize a healthy completion row or reset this
ledger. This archive preserves the exact existing evidence, including its
incomplete status. The separate resumption evidence is in
`../pyscf-sdf-continuation-20261103-v1/{interruption-audit,saved-state-audit}/`.

The last main-phase CPU observation, 8417.97 seconds, is not a final wait4 cost.
The two closed control phases cost 112.63092 seconds in total. Neither a final
main cost nor a reusable remaining V4 balance is asserted.
