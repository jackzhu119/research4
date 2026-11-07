# Synthetic lower-row algebra check

Status: `SYNTHETIC_ALGEBRA_ONLY_NOT_PYSCF_VALIDATION`.

The script was actually executed once and produced `result.json`. It extracts
only the frozen V9 `pure_j_partial_body` function via AST. Every integral,
overlap and core provider is replaced by an explicitly synthetic NumPy fixture;
the PBE/ABI predicates are mocked. No PySCF module is imported and no molecular
SCF, gradient, Hessian or normal mode is evaluated.

Two five-atom/nine-AO **synthetic** fixtures cover arbitrary finite values and
signed zero. For each fixture, the global-diagonal contraction and an equivalent
first-AO partition of its precomputed tensor are compared. Both diagonal
representations are then used with every contiguous completed lower-row prefix
from zero through five. Rows are round-tripped through NumPy's non-pickle format
in memory and restored into fresh full matrices, with upper Cartesian blocks
filled by transpose. The return buffers are subsequently mutated to ensure that
they do not alias the saved synthetic rows or diagonal.

Actual checks: 24 prefix restorations, 72 bitwise return-array comparisons,
24 cache-nonalias checks and 18 negative-contract rejections (nine distinct
malformed cases across the two fixtures). All returned e1/ej/zero-ek values
match the frozen V9 algebra bit for bit; the maximum absolute difference is zero.

These results establish only the tested algebra and serialization behavior.
The first-AO test slices an already generated arbitrary tensor; it does not test
the real PySCF/libcint `shls_slice` API, numerical integrals, screening, native
thread behavior or performance. The synthetic binding test compares opaque
markers and is not an implementation of real MO/grid validation. The check runs
within one process; actual cross-process quantum restoration is `NOT_RUN`.

Before a production continuation is accepted, real water and ethanol must pass
the predeclared full-global versus atom-partitioned diagonal, full-Hessian,
all-mode and separate-process restoration checks. These synthetic arrays must
not be reported as medicinal-plant or Spinosin calculation results.

Reproduce in the repository with a fresh output filename:

```sh
PYTHONPATH=.runtime/quantum-pilot-site:.runtime/quantum-upgrade-site \
  .runtime/bin/python \
  reports/pyscf-sdf-continuation-20261103-v1/incremental-rhf-design/synthetic-check/replay_synthetic_lower_rows.py \
  --output reports/pyscf-sdf-continuation-20261103-v1/incremental-rhf-design/synthetic-check/result-recheck-001.json
```

Existing receipts are not overwritten. The source and actual initial receipt
are bound in `files_sha256.json`.
