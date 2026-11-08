# Finite serial controller: engineering delivery

This directory records pure engineering tests only. No controller quantum phase
was launched by this task. The `phase_runner` is an in-process synthetic fake;
it never invokes the supervisor or imports PySCF. Actual global/cache locks are
mocked in these tests so they do not interfere with root's real molecule gates.

The implementation is `scripts/run_pyscf_serial_segments_20261103.py`.
Its `run` subcommand requires an explicit root-approved water/ethanol real-gate
report and SHA, the source snapshot directory and manifest SHA, a current-boot
same-state gradient receipt and SHA, the frozen consumer source SHA, and the new
actual-CPU ledger. `--dry-run` validates these inputs and records the proposed
first command without starting QM.

The controller allows at most 17 serial phases, each requesting at most ten new
RHF checkpoint units. Every payload goes through the frozen V4 supervisor.
Only a healthy closed EXITED/0 plus a verified consumer result permits continued
execution. Failure, a missing/unknown CPU cost, an unresolved running ledger,
no prefix progress, insufficient finite CPU/wall budget, or a changed binding
stops the controller. No budget or scientific validation is silently enlarged.

Use `--max-segments 1` for root's backup loop. A successful checkpoint-only
single phase ends with controller `PARTIAL`, reason
`FINITE_SEGMENT_COUNT_EXHAUSTED`, and process exit code 2. Root must inspect the
sealed successful segment before committing it and starting another new run.
The next invocation uses a fresh `--output` and the same `--rhf-cache`; phase
names automatically advance based on the real ledger. Final frequency
completion requires all 150 RHF units and 219 finite computed modes; it returns
controller `DONE` and exit 0. Positive-frequency validation remains explicit in
the final result and must not be inferred merely from `DONE`.

After each successful closed phase, the controller takes the actual V4 global
lock and RHF cache lock, repeats producer qualification, and copies only the
small immutable cache header/completed unit payloads/completion records to
`checkpoint-archive/`. Each immutable seal includes hashes, closed producer
qualifications, dependencies and an exact `restore_command`. The command uses
`restore-cache --seal PATH --seal-sha256 SHA --cache-dir ORIGINAL_PATH`. Restore
requires the original dedicated ignored cache path and acquires both locks
before inspecting any target payload. Conflicting old files fail rather than
being overwritten. Archives contain incomplete RHF intermediates, not an XC,
response or frequency result.

Root must commit/push the report directory, archive, source snapshot, same-state
gradient receipt, gate evidence, closed producer reports/proofs/logs and the
actual updated ledger. The controller performs no Git operation. Published H1
and other large save-only intermediates stay in ignored `outputs/`.

An optional `--forensic-unit-release PATH --forensic-unit-release-sha256 SHA`
supports a separately signed root release of specific old units. The consumer
checks it, the command forwards it unchanged, and seals retain its hash and
explicit qualification. It never rewrites a failed old row or permits automatic
continuation after a new failed phase.

The actual 15-case passing run is in `unittest.log` and `receipt.json` with the
source hashes and observed timing. Tests include a complete 15-segment synthetic
run, one-segment restart/unique names, archive byte-for-byte restoration, prefix
and hash checks, busy/unclosed archive guards, failed/no-progress stops, finite
budget checks, explicit forensic forwarding, and dry-run behavior.
