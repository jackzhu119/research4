# Independent finite serial-controller review

**Final engineering review: PASS**, 2026-11-04 UTC.

Final source: `scripts/run_pyscf_serial_segments_20261103.py`, SHA-256
`d0a67096215102a6e7b46ce9510e1eee6e3e81517de5f3e09e5f86f44c13bd6e`.
Use `watchdog-final/` for the final qualification. The earlier `8c300c50...`
receipt remains unchanged as historical evidence and is not the final version.
The frozen state worker, incremental core, Hessian consumer and V4 supervisor
were not modified by this review or by the controller changes.

The final source passed 16 independent pure tests. Five additional real
filesystem-lock checks ran under an isolated private temporary ROOT: same-cache
contention, global exclusion across different caches, refusal of an active
helper cache lock, release of the global lock after failed cache acquisition,
and normal reacquisition all passed. These did not touch the real research
global lock or any active molecular files. The watchdog test used only a mock
Popen object and clock; no supervisor was launched by the reviewer.

The review confirms:

- The controller accepts a hash-bound root gate qualifying actual water and
  ethanol full-Hessian/all-mode comparisons, separate-process restoration,
  same-state gradient validation, exact source versions and independent review.
  The controller does not create or replace those scientific results.
- It reads the existing finite ledger and reduces each request to remaining
  CPU/wall allowance. It never initializes, increases, repairs or extends the
  ledger. At most 17 sequential phases run per invocation, with ten new RHF
  units requested per phase. The unchanged V4 supervisor remains responsible
  for actual child accounting and resource containment.
- Healthy closed producer verification is required before continuation and
  archival. A new failure, unknown cost, unresolved running phase, missing
  progress, or inconsistent scientific result stops the loop without retry.
- An old abandoned/failed unit requires an explicit root hash-bound forensic
  release. That release is propagated to prefix qualification and the child;
  it neither clears old flags nor automatically releases a new failure.
- Archival and restoration acquire the actual V4 global-lock path and the
  helper's cache lock. Producer qualification and prefix count are checked
  again while locked before reading small closed payloads. Seals bind immutable
  records, dependencies and the exact original cache path; an old seal does not
  depend on a subsequently mutated whole-ledger SHA. No active HDF/NPY is read.
- A terminal scientific result requires all 150 Spinosin RHF units, a complete
  global Hessian and all 219 finite signed frequencies. Negative frequencies
  remain reportable and do not become a minimum certificate. A checkpoint-only
  phase cannot be promoted to frequency completion.
- A wall watchdog bounds waiting for the controller's own Popen supervisor:
  granted wall plus two seconds, TERM with a three-second wait, then KILL with
  another three-second wait. It does not automatically continue afterward.
  The reviewed unchanged supervisor source statically arms Linux parent-death
  SIGKILL for its payload and checks the parent PID during initialization.
- Git commits/pushes and any new authorization remain outside this controller.
  Report and checkpoint writes are immutable or explicitly transient progress;
  preserved scientific results and source labels are not rewritten.

This review performed **zero new quantum evaluations** and launched **zero
supervisor processes**. Test phases, spectra and payload bytes are synthetic.
The root researcher's actual molecule calculations and gate results are
separate evidence; this receipt cannot be cited as completed Spinosin DFT.

Reproduce the final pure/mock/private-lock review into a new output directory
by adapting only the review destination in `review_watchdog_final.py`; the
saved replay scripts refuse to overwrite their earlier output directories:

```bash
.runtime/bin/python reports/pyscf-sdf-continuation-20261103-v1/serial-controller-review/review_watchdog_final.py
```

All receipt source hashes, test logs, the original review script and the final
watchdog wrapper are retained. No real research global lock is taken by these
review scripts.
