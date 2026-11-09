# Fixed-quadrature curvature diagnosis

The new run investigates the unresolved discrepancy between the archived
analytic Hessian and finite gradients which rebuilt atom-centred grids. It
does not relabel old frequencies, remove imaginary modes, or fit classifiers.

The prior219-mode Spinosin result remains194positive/25imaginary, with minimum
and chemical-mechanism qualification false. All225 original response-equation
right-hand sides were measured; that checks the given response equation, not
the correctness of H1, partial A, or the complete nuclear derivative.

## Frozen scope before execution

- Existing cumulative V8 cap:390160.8308359999 CPU seconds.
- Actual carried cost:381227.8519289999 CPU seconds; remaining8932.978907.
- Effective deadline:2026-11-09T08:25:17.730193Z, earlier than the legacy ledger
  deadline. The old ledger is not edited to change either allocation or dates.
- Water suite grant300 CPU seconds; conditional single Spinosin point7000.
  Frozen guard120-second reserves apply. No retries or additional large points.
- PBE/def2-SVP/grid1, full AO/MO, direct integrals, two threads, explicit noD3.
- No new scheduled tasks or hourly chat reports.

## Water gate

Restore the authentic water source state and evaluate its zero gradient anew.
Compare it with the archived actual gradient, requiring max difference no more
than1e-8 Eh/Bohr. Evaluate fixed-quadrature displacements along Cartesian
coordinates0,4,8 at both signs of0.005 and0.0025 Bohr. Predeclared acceptance:
full central finite-gradient Hessian-vector max error no more than1e-4 Eh/Bohr²;
step-halving full-vector difference no more than5e-5 Eh/Bohr². These are new
software-control criteria, fixed before live results, not retrospectively fitted
claims about Spinosin or the earlier water diagnostics.

Keep exactly the original coordinates, weights, ordering and padded points.
Configure grid attributes before attaching arrays; use the displaced molecule
to rebuild both AO mask fields. Require the zero-geometry mask to reproduce the
saved mask, grids.mol to be the actual molecule, no grid pruning or repartition,
and coordinate/weight/source-state fingerprints to remain unchanged. Keep
grid_response=False. A control failure blocks the Spinosin calculation.

## One affordable Spinosin discriminator

At mode1,+0.005Bohr, compute one real SCF and full nuclear gradient on that
same original quadrature. Compare (g_plus-g0)/h with archived Hq. g0 is the
authenticated historical gradient, not zero: q dot g0 is approximately
-5.91301919e-5 Eh/Bohr. Neglecting it would introduce a substantial error.

The result is a first-order, one-sided local diagnostic. It cannot establish
step convergence, validate all219 modes, prove a minimum, or compute IR
intensities. Both the old-boot reference and new live execution identity must
be explicit. Original same-boot Hessian gates are not waived or rewritten.

## Interpretation and delivery

Archive actual control values even when gates fail. Preserve every failure cost
and output; any subsequent repair needs a new independently reviewed attempt.
Compare with old rebuilt-grid and analytic values without forcing agreement.
Use isolated outputs, checkpoints, source/config hashes and actual Git return
receipts on the research branch; do not merge main. Background execution is
bounded and requires the cloud instance to remain alive.
