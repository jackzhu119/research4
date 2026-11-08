# Independent scientific gap and checkpoint review

The next useful experiment is a bounded full-potential SCF test, before another displaced Spinosin calculation. The stopped fresh-zero did not qualify; it produced no certified E0, Eplus, Eminus or new curvature. This review performs no quantum calculation and is not a resource authorization receipt.

## What the stopped run establishes

A006 closed under `ACTUAL_CPU_GUARD` after **7381.797739 CPU seconds**. There were106 recorded ordinary SCF cycles:100 in call1 and6 in call2. The last energy was−2210.4455219696674Eh, with |deltaE|=2.09184e−10Eh and native orbital gradient norm4.29745e−9. These fail the final1e−12/1e−9 energy-and-gradient bounds; the stricter ordinary-loop trigger was1e−10.

The issue exceeds a last-digit stopping disagreement. Call1's last20 cycles plateaued at energy drifts around2e−10Eh. At its actual returned density, a full nonincremental rebuild lowered E by1.96933e−8Eh and found true unshifted Fock residual2.17968e−7, versus native1.24477e−8. At |E|≈2210Eh, one stored double ULP is4.54747e−13Eh; this discrepancy is approximately43,000ULPs. Call2 reset its initial energy to the rebuilt value, then drift resumed. Scalar storage rounding alone cannot explain this behavior.

Native `pyscf/scf/hf.py:178` passes density/potential history into every ordinary step. In `pyscf/dft/rks.py:96–107`, pure PBE accumulates J(deltaDM)+Jprevious while integrating XC on the current density. The custom `screened_direct_rks_type` at `scripts/pyscf_spinosin_checkpointed_direct_worker_20261031.py:170–181` delegates to the native direct contraction. Accumulated screened-J error is therefore a concrete suspect. It remains a hypothesis until a same-density J/VEFF/Fock comparison isolates the difference.

Root's proposed override—native `get_veff(..., dm_last=None, vhf_last=None)` at every SCF step—is an appropriate isolated test. Retain `direct_scf=True`, the reviewed cutoffs, full AO/MO basis and actual grid, and prohibit an in-core four-center ERI cache. A full potential build each cycle changes numerical iteration policy, not the declared PBE/def2-SVP/no-D3 physical method. It needs a separately frozen source and prospective protocol; the old failed run remains failed.

## Small controls and zero point first

1. On water and ethanol, compare cumulative versus forced-full J, VEFF, Fock, energy and the true orbital residual at the **same actual density**. A short full-build SCF pilot should record consecutive actual densities and both native flags and full-operator certification. Retain all numerical norms and raw flags.
2. After those endpoints qualify, compute a fresh native gradient and small native Hessian for each control. Test selected full-vector gradient differences and energy curvatures at multiple steps. A successful water v3 result already completed all31 points and passed the original fine-pair/target-step gates; that validates its explicit endpoint construction, not Spinosin's accumulated iteration.
3. Use an authenticated checkpoint only as the initial guess for a short Spinosin E0 pilot. Bind geometry, density/MOs, occupation, grid and method. Evaluate the actual full operator before continuation. Require the predeclared physical energy **and** full unshifted Fock residual gates. Do not spend another long block repeating a flat residual trend.

Newton refinement alone does not demonstrate that an accumulated potential has become consistent. A native Newton path can also pass incremental history to its underlying `get_veff`; audit its actual operator if used. Full builds or explicit verified resets are the relevant intervention. Consecutive-state energy drift is useful but is not itself an absolute energy-error bound.

## Directional energy design

For a Euclidean Cartesian unit direction q, use `kappa(h)=fsum(Eplus,Eminus,-2*Ezero)/h²`. If every energy has absolute error at most epsilon, then curvature error is bounded by4epsilon/h². For an illustrative absolute curvature tolerance1e−4Eh/Bohr²:

| h (Bohr) | Allowed numerator error (Eh) | Sufficient per-energy error bound (Eh) |
|---:|---:|---:|
| 0.0003125 | 9.765625e-12 | 2.441406e-12 |
| 0.00125 | 1.5625e-10 | 3.90625e-11 |
| 0.0025 | 6.25e-10 | 1.5625e-10 |
| 0.005 | 2.5e-09 | 6.25e-10 |
| 0.01 | 1e-08 | 2.5e-09 |

A new mode1 diagnostic at h=.005 and.0025Bohr uses **five** actual energies with a shared new E0. A predeclared third step.01 uses seven. These larger steps have a larger signal; comparison across h exposes finite-step bias. They are a new diagnostic, not a relaxation or retrospective PASS for A006. If the estimates follow a smooth O(h²) window, report the Richardson estimate and its assumptions; do not extrapolate through a noisy or inconsistent window. Only completed real pairs enter analysis. The earlier rebuilt-grid gradient curvature was positive at.01/.005 with a6.9e−7Eh/Bohr² change, versus analytic−.0375223; those observations do not prove fixed-grid agreement.

After energy points qualify, compute gradients for one completed fixed-grid pair and, if needed, its halved pair. Check both the directional q contraction and the **full Hq vector**, plus the first derivative `[Eplus−Eminus]/(2h)` against `q·g0`. A local second derivative can be measured at a nonminimum, but a minimum claim needs a fresh nuclear gradient and consistent final surface.

## Keep the two grid questions separate

The common-fixed-grid diagnostic defines F(R;G0): original spatial coordinates, weights, ordering and padding stay fixed, and AO screening masks are rebuilt for the actual geometry. This is the clean comparison with the current fixed-grid analytic derivatives.

Rebuilding the atom-centered grid defines F(R;G(R)). Its finite energy differences include grid-coordinate and partition-weight motion. In the pinned native gradient source, `pyscf/grad/rks.py:50–71` explicitly branches between full grid response and `grid_response=False`. A rebuilt-grid energy difference cannot be identified with the latter gradient without quantifying the missing terms. Test this first on small controls, record each actual grid fingerprint/count, and compare a denser grid on selected directions before expanding Spinosin work. Preserve PBE/def2-SVP/no-D3 in each numerical-policy comparison.

Rigid translations/rotations and grid-density checks are useful quadrature diagnostics. Projecting six rigid motions out of a Hessian does not remove an integration error affecting internal modes.

## What remains unresolved about the25 imaginary modes

The original result remains **219 signed modes:194 positive,25 imaginary; minimum=false; IR NOT_COMPUTED**. The same-operator audit measured all225 nuclear RHS, without a predeclared numerical PASS/FAIL gate. It does not independently prove H1/partial-A correctness or that the geometry is stationary on a newly qualified energy surface. The historical restored gradient passed its original optimization thresholds (component RMS3.61655e−5Eh/Bohr), but it used the archived source state; it is not a new full-operator gradient at the new SCF endpoint. No completed Spinosin internal orbital-stability result is present in the reviewed postrun summary.

Mode1, weakest negative mode25 and first positive mode26 are economical representative diagnoses. Agreement or disagreement for a sample cannot establish the exact number of genuine negative directions. Do not flip signs, delete modes or relabel all25 as numerical artifacts. First qualify the intended electronic solution, fresh nuclear stationarity and derivative/grid consistency. Only then decide whether a new complete Spinosin Hessian is warranted. Geometry optimization should use a declared, consistently differentiated numerical surface; a fixed G0 diagnosis does not automatically qualify an atom-centered moving-grid minimum.

## Storage and execution boundaries

Keep large existing checkpoint, density and MO arrays local and hash-bound. New return artifacts should contain scalar iteration summaries, small matrices/plots, source/input/proof hashes and local checkpoint references; do not restage every cached raw array. Root manages the prospective72h grant and guard separately. Historical costs, closed ledgers and failed outputs remain preserved. This document creates no frequencies, IR, mechanism assignment or classifier result.

`review.json` contains authenticated source hashes, exact scalar endpoint records, the finite-energy noise table and the ordered test plan. No array/checkpoint contents were loaded for this review.
