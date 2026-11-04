# Independent checkpoint mathematics and state review

Recorded on 2026-11-04 UTC. **Static and synthetic engineering review: PASS.**
This is not a scientific release of molecular frequencies or an actual
cross-process PySCF validation. This review performed no SCF, gradient,
optimization, libcint integral, or molecular Hessian calculation.

## Final reviewed sources

| Source | SHA-256 | Review |
|---|---|---|
| `scripts/pyscf_molecular_state_worker_20261103.py` | `dcfffc4d0c68cd30c6d2c60597b77a4c3931516412b7f3308bb9d0b72ef45360` | 29 independent pure tests passed; restore/construction AST has no `kernel`, `run`, `build`, or `density_fit` call |
| `scripts/pyscf_incremental_rhf_checkpoint_20261103.py` | `ce96055f24b39833337844c178f94597a9ac4ba7721ba019816d03628f971f55` | 16 independent pure tests and six separate synthetic Python processes passed |

The earlier `88a9f150...` receipt is preserved. The final incremental revision
only prepends the original Apache copyright/license/author notice: hashing its
unchanged body reproduces the complete earlier SHA. The final licensed source
was independently retested; use `licensed-final/` as the final qualification.

## Mathematical findings

- Stock `int2e_ipip1` uses AO symmetry `s2kl` and the distinct contraction script
  `lk->s1ij`. Restricting only the first atom's shell range partitions output
  AO index `i`; `j`, `k`, `l`, and the density matrix remain global. The reviewed
  installed direct driver derives the output dimensions from `ij` and the
  density dimensions from `lk`, matching the proposed sliced shapes.
- The precontracted diagonal is exactly the original expression
  `2 * einsum('xypq,pq->xy', V_ipip1[:,:,P_i,:], D[P_i,:])`. No electron-space,
  atom-pair, occupied/virtual-MO, or response-space truncation is introduced.
- The row implementation retains the original `ipvip1` coefficient `0.5`,
  Coulomb row factor `4`, overlap AA/AB factors `-2`, and core-Hamiltonian
  factor `+1`. All units are inherited from the original analytical derivative
  expressions; no frequency conversion is performed in this helper.
- Diagonal contributions are added once when producing a complete lower row.
  Restored lower rows are assigned once. Upper blocks are restored with
  `E[j,i] = E[i,j].T`; diagonal blocks are not averaged. `ek` is a full-shape
  zero array only in the restricted nonhybrid PBE path.
- A full tuple is returned only after all `2 * natm` units exist. The RKS XC,
  H1, complete CPHF response, nuclear terms, and global assembly remain outside
  this helper. Returned arrays are fresh because stock RKS mutates `e1`.
- `with_k=True` delegates to the original helper. Explicit MO overrides must
  fingerprint-match the actual MF. Full original atom ordering is required.
  Nested contexts, state changes, duplicate completed units, discontinuous
  prefixes, changed source/hash, and malformed payloads fail closed.
- The reviewed derivative optimizer remains `None`; no unsafe suffix-based
  VHF optimizer is enabled. Splitting one global call into atom calls may
  increase overhead. This review establishes neither a speedup nor a new DFT
  method.

## Independent checks actually executed

1. `review_foundation.py` contracts one tiny synthetic four-index tensor in
   global and first-index-sliced forms. The maximum J difference was
   `1.42e-14`; the maximum precontracted diagonal difference was `1.85e-13`
   from floating-point summation order. Both were below `1e-12`. It separately
   replays 72 bit-equal full-array comparisons, 24 nonalias checks, and 18
   rejected invalid contracts from the existing synthetic algebra check.
2. The state worker's 29 pure tests passed independently. Exact source SCF/MO,
   serialized physical Mole state, full actual quadrature and screening masks,
   grid provenance, numerical settings, source/native implementation identity,
   and a cleanly closed producer proof are bound before scientific restoration.
   The restoration path constructs the MF without a fresh SCF or grid rebuild.
3. The incremental worker's 16 pure tests passed independently. Six **real
   separate Python processes**, each with a new synthetic fixture object,
   appended one unit per process to the same disk store. All processes exited
   successfully and released the file lock. The final synthetic full arrays
   differed from the frozen V9 global-J reference by at most `7.11e-15`.
   These are fixture arrays, not restored physical wavefunctions or a real
   molecular full Hessian. Both pre-license and final-license runs are retained.
4. All six pinned installed source/native file hashes matched. Completed
   payloads use closed, fsynced, no-replace publications; directory creation
   boundaries are synchronized. Unfinished attempts are retained as orphan
   evidence and are not consumed as completed units.

## Remaining mandatory scientific gates

Only root may launch quantum calculations. Before using this implementation for
Spinosin, root must run real stock-versus-incremental water and ethanol full
Hessians on the same verified source MF, then restore that exact MF and finish
the segmented calculation in separate processes. Compare the complete matrices
and all 3/21 normal modes, not merely the cached pre-XC rows. Also validate the
actual source/consumer supervision proof and recorded costs.

The incremental store records unit `execution_evidence` but does not authenticate
the producer's phase closure. The root consumer/controller must check each
unit's producer or explicitly release abandoned-phase units after forensic
review. A completed immutable row does not prove that its producing process
closed successfully. Never fabricate a missing final CPU cost or `DONE` status.

The abandoned old V4 process had no completed RHF checkpoint. These tests do not
recover its unfinished Hessian. New SDF atom ordering must not be mixed with old
MO or cache indices. No new Spinosin frequency, IR intensity, plant spectrum,
chemical assignment, classifier feature, or accuracy improvement was produced.

## Reproduction

The receipt-producing scripts refuse to overwrite their first outputs. Rerun in
a fresh review directory with the same depth, or preserve the original receipts
and change only the reviewer output destination. Commands originally executed:

```bash
.runtime/bin/python reports/pyscf-sdf-continuation-20261103-v1/checkpoint-math-review/review_foundation.py
.runtime/bin/python -m unittest discover -s tests -p test_pyscf_molecular_state_worker.py -v
.runtime/bin/python reports/pyscf-sdf-continuation-20261103-v1/checkpoint-math-review/review_implementation.py
.runtime/bin/python reports/pyscf-sdf-continuation-20261103-v1/checkpoint-math-review/review_licensed_final.py
```

The `synthetic-cross-process-cache/` directories are closed pure-fixture stores,
retained only for engineering inspection. Their contents must never be used as
molecular data or numerical quantum results.
