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

错误混种分类成绩已按2026-11-12用户决定撤回；论文使用main的paper/current白名单。
historical internalCV and the outstanding laboratory gaps remain unchanged.

## Actual software-control results

All rows below are actual calculations on the same archived water geometry,
PBE/def2-SVP/grid1 and exact fixed original quadrature. Failed rows are retained.

| Attempt | Attempted points | Qualified endpoints | Actual CPU seconds | Numerical release |
|---|---:|---:|---:|---|
| A005 original steps | 13 | 13 | 4.357029 | Failed large-step curvature gate |
| A006 tight SCF | 31 | 16 | 13.712143 | Incomplete; final extra-cycle failures |
| A006 bounded polishing v2 | 31 | 29 | 25.342708 | Incomplete; two native incremental-energy failures |
| A006 full endpoint certification v3 | 31 | 31 | 24.472247 | All original numerical gates passed |

V3 preserves each native convergence flag and incremental-energy value. Its
explicit solver certificate instead requires an accepted normal iteration
followed by its actual final extra-cycle states in the same call, full native
VEFF evaluations on both consecutive densities, energy drift below1e-13Eh,
and the true unshifted full-Fock orbital residual below1e-9. Consecutive density
values may coincide at a fixed point; they are not manufactured duplicate
evaluations. Native failure records from v1/v2 remain unchanged. At most3
calls of100cycles are permitted per point; there is no automatic job retry.

The maximum water full-Hq error at the finest step is3.74893e-6Eh/Bohr²;
the maximum energy-curvature error at the actual Spinosin h=.0003125Bohr is
4.23483e-5Eh/Bohr². All finest-pair energy and5e-5Hq step-halving criteria
also passed. The successful producer is
`queue-host-a-006/science/water-full-audit-convergence-attempt001/result.json`,
SHA81f57122b72b86c45c17e534e9f249b3ca02a0fc4fde9fbe8b2de571076788e2.
This validates the small-molecule numerical control, not a new Spinosin minimum.

## Finite background continuation

The independently reviewed energy-v2 producer uses the same full-endpoint
certificate, with the previously selected Spinosin1e-12Eh/1e-9 thresholds.
It may perform fresh zero, plus and minus SCFs only. Large density/MO arrays
and checkpoints stay under the work directory's `local-checkpoints/` with
hashes, outside the small-result collector. It retains full AO/MO space,
direct native integrals and the explicitly non-D3 PBE pilot.

The A006 queue and detached wrapper record actual status and PID/birth/boot
in `queue-host-a-006/`; configuration alone is not evidence of execution.
After a complete triplet a separate non-quantum analyzer authenticates the
closed producer and replays the compensated energy curvature into CSV and
a figure. Small results return to the research branch only after Q closes.
The7500CPU phase, original cumulative cap, earlier deadline and failed costs
remain fixed. No periodic reporting task is created, and a host shutdown is
not guaranteed to preserve a live process. Source-qualified checkpoints and
failure evidence support a later explicit, independently checked recovery.

A single directional energy test does not create219 new frequencies, verify
all25 imaginary modes, establish a minimum, compute IR intensities or assign
an experimental peak. Those conclusions require further real evidence.
