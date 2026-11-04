# Independent resumable-Hessian consumer review

**PASS for the reviewed standard workflow**, 2026-11-04 UTC.

Reviewed consumer source:
`scripts/pyscf_resumable_hessian_worker_20261103.py`, SHA-256
`d23c4a3185aa4f666e22d671034728c1ca2443676a05fc8bec0587d6c8d6bf03`.
The state worker remains `dcfffc4d...` and incremental core `ce96055f...`.
No production source was changed by this reviewer.

Twenty pure tests passed independently. A separate actual-factory metadata
check constructed only small Mole/basis metadata and the frozen plain RKS /
stock Hessian objects. Integral, SCF and gradient entrypoints were mocked to
raise. The stock method-identity gate passed; a custom `get_hcore` and an X2C
decoration were rejected. This is not an electronic-structure calculation.
The initial inline receipt and a replay from the saved script are both retained.

The review found no blocking issue for starting the root-supervised real water
and ethanol restored-gradient tests. Specifically:

- Exact snapshots require a cleanly closed real source producer, source and
  array hashes, complete MOs/Mole/quadrature and matching environment. No fresh
  SCF is performed by this consumer.
- The entry gate restricts standard MF, Hessian and gradient core method
  identities and rejects X2C, solvent, QM/MM, pseudopotential/ECP and other
  decorated Hamiltonians. It inspects methods without requesting integrals.
- A restored-gradient receipt must match the exact snapshot, current boot,
  successful comparison and stationary thresholds, plus a healthy closed
  supervised producer.
- Every reused completed RHF unit requires independently verified proof,
  source, closed parent ledger, unchanged log and matching producer result.
  A numeric marker alone is insufficient. An explicit hash-bound root forensic
  release may qualify named abandoned-phase records without rewriting their
  unknown/failed exits or costs.
- A finite checkpoint stop is reported separately from full RHF, global
  Hessian and frequencies. Native guard and scoped Hessian methods restore
  on normal return and interruption. Source geometry/MO/grid stability is
  checked after a completed phase or checkpoint stop.
- Stock reference and incremental paths retain the same reviewed memory,
  global assembly and complete response workflow. The incremental context
  affects only its qualified RHF stage. Analysis checks raw matrix shape and
  asymmetry before the original symmetrization and harmonic analysis.
- Frequencies remain signed, the complete nonlinear `3N-6` count is required,
  and a computed spectrum is distinct from a verified positive-frequency
  minimum. IR intensities remain `NOT_COMPUTED`.

This review produced **zero new quantum evaluations**. It does not claim a
real restored-gradient match, a water/ethanol full-Hessian or mode match, or
Spinosin frequencies. Root must execute and independently assess those actual
scientific gates before large-molecule use.

Reproduce the metadata and pure checks in a fresh destination:

```bash
.runtime/bin/python reports/pyscf-sdf-continuation-20261103-v1/consumer-review/replay_consumer_entry_review.py --output /tmp/research4-consumer-review-fresh
```

The replay refuses to reuse an existing output directory and requires the
reviewed source SHA. `review.json` records the first independent run;
`replay/receipt.json` records execution of the saved replay script.
