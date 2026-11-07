# Smaller-step and energy-only diagnostics

A005 preserved all13 fixed-quadrature water outputs. All calculations finished,
but the predeclared control failed: the oxygen c0 direction differed from the
analytic curvature by0.0191273 at h=.005 and0.00558747 at h=.0025. The two other
directions were much closer. The larger Spinosin point was never started.
The4.357029CPU-second actual cost remains in the original cumulative ledger.

This new experiment measures step convergence, without editing those results
or relaxing their gate. It uses the same original quadrature and fresh AO
screening masks, PBE/def2-SVP/fullAO/fullMO/direct/noD3.

## Predeclared water experiment

Cartesian coordinates0,4,8, both signs, at h=.00125,.000625,.0003125,
.00015625,.000078125Bohr, plus a fresh zero SCF/gradient:31 points. SCFs use
energy tolerance1e-13Eh and orbital-gradient tolerance1e-9, with an explicit
AND convergence hook including the final extra cycle. The default relaxed
extra-cycle OR rule is not treated as strict convergence.

All coarse and fine values remain in the report. A separate finest-pair gate
retains the original1e-4 Eh/Bohr² full-Hq error and5e-5 step-halving criteria.
The energy-curvature check at the intended Spinosin step .0003125Bohr must
also pass1e-4 Eh/Bohr² for all three water directions. Success of measurement
and success of these gates are distinct fields.

## Conditional Spinosin energy test

If those controls pass, a separate reviewed producer may perform three actual
SCFs at zero and ±.0003125Bohr along archived mode1. Both displaced guesses
start from the fresh zero density. It must use fresh E0, never archived E0,
and compute fsum(Eplus,Eminus,-2Ezero)/h² from full binary64 values.

Spinosin stopping criteria are predeclared1e-12Eh energy drift AND1e-9 orbital
residual, including the extra cycle. At |E|~2210Eh one binary64 ULP is~4.55e-13,
so a1e-13 drift bound would demand equal rounded floats and is unsuitable as a
robust mandatory gate. Save actual drift, orbital residual and ULP; these are
precision diagnostics, not rigorous error bounds.

One finite-energy step is a local diagnostic, not a step-converged full
Hessian, minimum, new normal-mode spectrum, or chemical peak assignment.
No nuclear gradients are computed in that large-molecule phase. Every partial
energy/checkpoint and failed cost must be retained if its resource ceiling
prevents all three SCFs from finishing.

## Resources and delivery

Water grant500CPU seconds and conditional energy grant7500 use the unchanged
390160.8308359999 cumulative cap; actual remaining after A005 is8928.621878.
The deadline remains2026-11-09T08:25:17.730193Z. No automatic retry, extra grant,
or periodic report task. All new source/input/output folders are independent.
Native logs retain their original bytes, including trailing spaces; transport
checks exact committed SHA rather than normalizing scientific evidence.

Original219 modes/25imaginary frequencies, IR NOT_COMPUTED, classifier96.0234%
historical internalCV and the outstanding laboratory gaps remain unchanged.
