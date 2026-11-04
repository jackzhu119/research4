# Molecular state worker software preflight

The new `scripts/pyscf_molecular_state_worker_20261103.py` is an actual supervised
SCF/geometry-optimization entry point. This preflight ran **29 pure JSON/tiny
NumPy tests**, not quantum calculations. The independent review also checked
that restoration does not invoke SCF, integral, optimizer or grid-build kernels.
The root controller must still run real cross-process control calculations.

The reviewed frozen worker source SHA-256 is
`dcfffc4d0c68cd30c6d2c60597b77a4c3931516412b7f3308bb9d0b72ef45360`.
No historical worker, report, spectrum or model was modified by this subtask.

## Explicit input manifest

All file records are `{ "path": "...", "sha256": "...", "bytes": 123 }`.
Paths may be absolute or repository-relative. The CLI requires the manifest's
expected SHA-256 separately. The schema is `research4-molecular-input-v1`:

```json
{
  "schema": "research4-molecular-input-v1",
  "molecule_id": "explicit_provenance_qualified_identifier",
  "formula": "C16H12O5",
  "atom_symbols": ["the complete atom sequence in exact XYZ/NPY order"],
  "composition": {"C": 16, "H": 12, "O": 5},
  "charge": 0,
  "spin_2S": 0,
  "multiplicity": 1,
  "radical_electrons": 0,
  "electron_count": 148,
  "connectivity_verified": true,
  "stereochemistry_checked": true,
  "input_xyz": {"path": "...", "sha256": "...", "bytes": 123},
  "coordinates_angstrom_npy": {"path": "...", "sha256": "...", "bytes": 123},
  "source_artifacts": []
}
```

This is a schema illustration, **not runnable invented molecular coordinates**.
Use the intake audit's actual archived/derived files. A prepare-state input also
requires `optimized_source_result` as a file record. Its original DONE optimizer,
SCF, gradient, method and exact geometry must agree. An optional verified initial
checkpoint additionally requires `initial_checkpoint`,
`initial_checkpoint_atom_order_sha256` (the worker's `document_hash(atom_symbols)`)
and `initial_checkpoint_geometry_npy_sha256`. It must be the checkpoint bound by
the original optimization result. Never pair the uploaded reordered Spinosin SDF
with old C001 orbitals by numerical atom index.

Two real hash-bound input manifests have also been prepared and validated
without quantum execution:

- `inputs/spinosin_old_C001_prepare_input.json`: SHA-256
  `3511b8dc3208e0a93be1f000655fdcb9be4424d016244176d46b5f033b4a32fa`.
  Preserves the old C001 full-precision coordinates and atom order, original
  DONE optimization result and matching original checkpoint.
- `inputs/cid5280448_optimize_input.json`: SHA-256
  `842ebe92dad9e7d6d5a56b4c27c515fb10580966e0dd9495bc932fd3bfa76f49`.
  Uses the user's faithfully exported 33-atom structure and explicitly labels
  common-name/CID external provenance as pending. It is not a whole-plant sample
  or a claim that this compound was measured in the Huangqi workbooks.

## Execution interface

Only the root controller launches quantum work through the actual-CPU V4
supervisor. The child command is:

```sh
.runtime/bin/python scripts/pyscf_molecular_state_worker_20261103.py \
  --phase prepare-state --input-manifest INPUT.json \
  --input-manifest-sha256 SHA256 --output NEW_PHASE_DIRECTORY \
  --threads 4 --maxsteps 35
```

`prepare-state` rechecks an already optimized geometry using a fresh PBE/
def2-SVP/grid1 screened-direct SCF and fresh same-method gradient. It does not
rerun optimization. `optimize-state` performs real bounded geomeTRIC optimization
and then a fresh final SCF/gradient. Both explicitly use **no D3**, no density
fitting and no global four-center ERI cache. Neither computes a Hessian or IR
intensity. A failed or interrupted stage remains PARTIAL with available logs,
geometry steps, checkpoint and result; hard-killed processes can lack result.json.

## Exact state restoration

```python
mf, state_binding = state_worker.restore_snapshot(
    snapshot_directory,
    expected_manifest_sha256=frozen_snapshot_sha,
    expected_source_sha256=frozen_worker_sha,
    stdout=new_consumer_log,
)
```

An exact snapshot contains pure JSON Mole native state; all complete MO
coefficients/energies/occupations; actual quadrature coordinates/weights;
non0tab/screen_index where present; atom attribution/unpartitioned quadrature
weights; final gradient; method/grid/callable settings; and source/dependency
hashes. NPY loads forbid pickle. Mole restoration does not use PySCF's eval-based
`gto.loads`. Files are fsynced and atomically published without replacement, with
the manifest last.

Restoration requires the immutable source proof and a matching **closed healthy
EXITED/0 V4 producer row**, measured CPU, unchanged parent log, DONE scientific
result, exact source version and every array hash/dimension. A running/abandoned
producer is rejected. Exact Python/PySCF/NumPy/SciPy/geomeTRIC versions, selected
PySCF sources, native libcint/cgto/dft/xc/cvhf libraries and selected NumPy/SciPy
native wrappers are bound. Physical
Mole identity normalizes only reviewed transient `_env[17]` bookkeeping, with no
change to geometry or basis. Restored state is explicitly
`RESTORED_FROM_VERIFIED_SOURCE_SCF`, never a claimed new SCF run.

The returned binding is immutable source-state evidence. Consumer proof/PID
belongs separately in the new execution report. Enter the frozen V8
3000-MB Hessian memory context before building an incremental Hessian context.
Cached derivative rows must additionally match the actual restored MF/grid/
method fingerprint. Real water/ethanol source-to-restore Hessian comparisons
remain necessary before using this path for the main Spinosin calculation.
