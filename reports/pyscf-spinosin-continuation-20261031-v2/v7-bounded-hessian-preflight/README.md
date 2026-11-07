# V7 bounded Hessian engineering preflight

**DONE: implementation, source review and 15 pure guard/dispatch tests.** These
tests run synthetic tiny fixtures; they are not water, ethanol or Spinosin
calculations. This agent started no quantum jobs, evaluated no integrals and read
no active checkpoint. Root must independently execute the real small-model
numerical gates before a Spinosin Hessian.

New worker: `scripts/pyscf_spinosin_bounded_hessian_worker_20261031.py`.
New tests: `tests/test_pyscf_spinosin_bounded_hessian_worker.py`.
Frozen V4/V5/V6 files and all earlier calculations remain unchanged.

V7 preserves `SCREENED_DIRECT_RKS_V4` as the physical integral backend and uses
the engineering version `BOUNDED_HESSIAN_DIRECT_V7_SAVE_ONLY`. Functional, basis,
grid points and weights, screening threshold, SCF convergence, optimizer
thresholds, original full Hessian assembly, native-driver forwarding and strict
Spinosin 219 positive-mode guard are unchanged. IR intensities remain
`NOT_COMPUTED` and cross-process Hessian restore remains
`NOT_IMPLEMENTED_SAVE_ONLY`.

The reviewed changes are limited to resources and response grouping:

- `--hessian-block-atoms` accepts 1–8 atoms and defaults to 4.
- Blocks above 4 atoms require native direct threads = 1; smaller blocks retain
  the reviewed native1/native2 options. The whole electronic response and all
  inter-atom Hessian couplings are preserved.
- During the **entire original Hessian kernel**, a context temporarily sets
  `mf.max_memory` and `hobj.max_memory` to 3000 MB. SCF and gradient calculations
  retain the original 8000 MB setting. The context restores both original values
  on normal return, solver exception, signal-style Python interruption or
  callback failure. It preserves inherited versus instance Hessian attributes.
- Manifest bindings contain the actual MF/Hessian memory attributes of 3000 MB
  alongside unchanged MO, geometry and initialized quadrature hashes. Worker
  progress and results record memory-context entry/exit and restoration.

These are PySCF **resource attributes**, not a strict 3 GB library-wide cap.
The original `hess_elec(..., max_memory=4000)` argument and derivative formulas
are intentionally unchanged. The key RKS partial Hessian and global response
construct their numerical-block budgets from the actual `mf.max_memory`, so the
3000 MB context affects both while retaining their native 2000 MB floor. At the
partial GGA stage, `max(2000, 2700-current_RSS_MB)` conditionally permits up to
40,264 grid points instead of the prior ceiling of 67,200, or fewer when memory
or the actual grid size constrains the block. This is a source-based estimate,
not a measured Spinosin peak. The supervisor's 12 GiB address-space limit remains
the real allocation backstop.

The independent methods agent reviewed the new source and found no blocking
method or restoration issue. Tests additionally compare the AST of the unchanged
physical-method guards and complete native-thread wrapper against frozen V6,
exercise synthetic nine-atom 8+1 grouping and every cross-block pair, verify
actual3000 MB cache bindings, and preserve a first completed cache block when a
later mocked solver call fails. Synthetic files are temporary and removed.

Required **real controls**, to be run only by root under the serial CPU ledger:

1. V4 versus V7 full water Opt/Freq.
2. Verified nine-atom ethanol at the same optimized geometry and same fresh MF:
   original full Hessian versus B4 versus B8, native1, identical3000 MB contexts.
3. Separate numerical equivalence, positive-minimum verification and actual
   resource/elapsed-time results. A three-atom water cannot qualify an
   eight-atom response block.

No completion-time guarantee follows from the earlier three-minute SCF-J
timing; response batch cost and Krylov iterations must be measured. The source
and integer-size analysis is separately frozen in `../b8-performance-audit/`.

Re-run the pure tests:

```bash
python -m unittest discover -s tests -p test_pyscf_spinosin_bounded_hessian_worker.py -v
```

Root may inspect CLI arguments without running a calculation:

```bash
python scripts/pyscf_spinosin_bounded_hessian_worker_20261031.py --help
```
