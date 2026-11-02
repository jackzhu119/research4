# B8 CPHF performance and live-memory audit

Status: **DONE — source/arithmetic audit only**. No quantum calculation, integral,
SCF, gradient or Hessian was run; no active checkpoint was read. No V7 was written
and frozen V4/V5/V6 sources and earlier results were not changed. The accompanying
JSON binds the installed PySCF 2.14 source hashes and contains exact integer array
sizes. GB below means 10^9 bytes; the worker address-space hard limit remains
12 GiB = 12,884,901,888 bytes. These estimates are **not measured peak memory or
strict bounds on the Python/native allocator**.

Independent source review by the methods agent confirmed both the response-stage
arithmetic and the old/new `aow` lifetime in the separate partial-GGA phase. The
parent subsequently authorized a new V7 engineering worker with an explicitly
recorded 3000 MB Hessian-only memory context; that later implementation and its
future real controls are separate from this immutable audit.

## Recommendation

1. For the **CPHF response stage**, an eight-atom block with **native direct
   threads = 1** has a useful estimated margin and merits a bounded engineering
   pilot after a nine-atom same-method control. It is not yet supported by the
   frozen V6 CLI (which permits only 1–4 atoms), and this audit does not authorize
   or implement another worker.
2. Do **not** make B8/native2 the default. With a deliberately conservative
   working-set model and 1 GiB additional allocator/BLAS/unmodeled reserve, its
   estimated margin is only 369,579,872 bytes (0.344 GiB). Actual prior failures
   were virtual allocation failures, so an RSS estimate alone is inadequate.
3. **The complete Spinosin Hessian remains gated on a separate numerical-grid
   block-memory review.** The partial GGA XC Hessian has a source-visible live
   array set near 11.74 GB at the permitted 67,200-point block, independent of the
   later CPHF block size. A smaller CPHF block or native thread count cannot fix
   this separate stage. No whole-Hessian memory-safety claim follows from item 1.

## Response-stage working set

Fixed dimensions are 75 atoms, 762 AO/MO functions, 160 occupied orbitals and
float64, as provided for the current Spinosin calculation. The full H1AO, global
MO1 and MO_E1 arrays total 1,310,695,200 bytes. This deliberately budgets all
completed response atoms even while a current block is being solved.

| Atoms per response block | Blocks | Maximum `xs+ax` at 50 iterations, no deflation | Native1 estimate including 1 GiB reserve | Native2 estimate including reserve |
|---|---:|---:|---:|---:|
| 2 | 38 | 0.585 GB | 7.045 GB | 10.250 GB |
| 4 | 19 | 1.170 GB | 7.800 GB | 11.005 GB |
| 8 | 10 | 2.341 GB | 9.310 GB | 12.515 GB |

For B8, one MO-space RHS block has 24×762×160 doubles = 23,408,640 bytes. PySCF
`lib/linalg_helper.py:1299–1375` retains the two Krylov bases as lists of vector
views and copies the **current** RHS block for orthogonalization; it does not
stack a second full Krylov history. Fifty no-deflation iterations therefore
contribute 2×50 RHS blocks. The estimate adds ten current-block equivalents for
CPHF input/control/QR/transformation arrays. The largest reduced Krylov matrix
is 1200² doubles (11.52 MB), not another AO/Krylov-sized tensor.

One B8 AO density batch is 24×762² doubles = 111,483,648 bytes. Four such matrices
budget `dm1`, XC result, J result and native reordered densities. The code builds
MO-space `v1vo` after the native J has returned. The GGA XC response's
`hermi_sum` briefly allocates a second output matrix (`numint.py:1510`,
`numpy_helper.py:764–778`). The source only symmetrizes density inputs by copying
when `hermi != 1`; the reviewed global response uses `hermi=1`.

The native C policy is approximately 3.2 GB **per native direct thread**, rather
than 3.2 GB per density. `nr_direct.c:408` divides its stack limit by `n_dm`, and
lines 422–427 allocate all density stacks in each OpenMP thread. The estimate
rounds the total, padding and small metadata up to 3,205,000,000 bytes per thread
for the B8 pure-DFT J case. This does not generalize to an arbitrary hybrid
functional or different backend. The estimate also includes 100 MB ground XC/grid
bookkeeping and an assumed 600 MB runtime/library baseline. Those two allowances
are assumptions, not measurements taken at a completed Spinosin CPHF stage.

The XC response and native J allocations have different lifetimes:
`scf/_response_functions.py:74–100` computes `nr_rks_fxc` first, then accumulates
J in place. AO/grid temporaries created inside that XC routine are no longer
live Python arrays when the native J is called. For a permitted 67,200-point GGA
block, four AO components use 1.639 GB and one extra grid×AO component is 0.410 GB.
Retained `aow` plus a density contraction can add two such components. This
response-XC phase can be large, but should not be added to the full native J
stack as if every array were simultaneously live. Freed malloc arenas or BLAS
workspaces can retain address mappings; the 1 GiB reserve is a cautionary budget,
not a proof that all such retention fits.

## Independent partial GGA Hessian risk

`hessian/rks.py:353` allocates a full 75×3×3×762×762 `vmat` before the numerical
integration loop. Its GGA branch (`381–404`) uses second AO derivatives and keeps
four `ao_dm0` components. During `aow = _make_dR_dao_w(...)`, Python retains the
old three-component list until the function returns a new three-component list;
the helper additionally creates a one-component scaling temporary while updating
its new arrays (`grad/rks.py:213–232`). At the allowed grid-block ceiling:

| Simultaneous array group | Bytes |
|---|---:|
| Full GGA Hessian `vmat` | 3,135,477,600 |
| Ten-component AO derivative buffer | 4,096,512,000 |
| Four `ao_dm0` components | 1,638,604,800 |
| Old/new three-component `aow` plus one temporary | 2,867,558,400 |
| **Visible array subtotal** | **11,738,152,800** |

This conditional scenario assumes the actual grid has at least 67,200 points and
the memory argument permits that block; it is not an observed allocation.
Adding the assumed 600 MB runtime plus 1 GiB reserve gives 13,411,894,624 bytes,
above the unchanged 12 GiB limit, before several smaller arrays. Source evidence
therefore does not support treating the present 8000 MB memory setting as safe
for the entire Hessian.

`NumInt.block_loop:2896–2916` determines the block from its passed memory argument
and caps it at 1200×56 points. Importantly, `rks.partial_hess_elec:71–83` resets
that memory argument from `max(2000, mf.max_memory*.9-current_RSS)`; changing only
`hobj.max_memory` does not control this stage. A future engineering change should
bound numerical **block size**, preserve the complete original grid and all
atoms/basis functions, and qualify the raw Hessian against an unchanged control.
This audit does not select or implement such a change.

## Performance evidence and limits

The timing input is **parent-reported actual direct1 Spinosin timing**: about
20 CPU minutes for the first five SCF cycles plus extra convergence cycle, and
about three minutes per native J. This audit did not independently time it and
SCF J is not the same workload as a 24-density CPHF J.

Larger response blocks can share integral traversal. In `nr_direct.c:66–93`, each
screened shell quartet evaluates its integral once and loops over density
contractions afterward. However, contractions grow with batch size and the
Krylov rank/iteration count may change; no linear speedup is guaranteed. Native
`cphf.solve_withs1:136` makes **one further `fvind` call after Krylov returns**.
If a full block uses K response iterations, the rough count is K+1 calls/block:

| Block | K=5 | K=8 | K=10 |
|---|---:|---:|---:|
| B2 | 228 | 342 | 418 |
| B4 | 114 | 171 | 209 |
| B8 | 60 | 90 | 110 |

At the illustrative three-minute SCF-J proxy, B8's K=8 row alone would be
270 CPU minutes, excluding XC work, partial Hessian, H1 construction, final
assembly and other overhead. B2 would be 1026 CPU minutes and already exceed
12 hours. These are conditional workload calculations, **not completion-time
forecasts**. Do not spend the rest of the run budget based on them alone.

## Qualification before any future large-block pilot

Use a documented fixed nine-atom three-water geometry, full-electron PBE/
def2-SVP/grid1/noDF, identical SCF orbitals/settings and the same raw Hessian
comparison conventions. Compare the complete original global Hessian against
B8 (8+1 atom RHS blocks), including every inter-atom block; do not concatenate
partial sub-Hessians. One three-atom water cannot exercise an eight-atom block.
This is a numerical engineering control and is not medicinal-plant evidence;
an arbitrary fixed cluster need not be a minimum or have all-positive modes.

Keep `conv_tol_cpscf=1e-8`, max_cycle=50, level_shift=0 and native lindep unchanged.
Full B8's requested native tolerance is 8e-8, stricter than a 75-atom request of
7.5e-7 but less strict than B2/B4's 2e-8/4e-8. Native normal return is not an
independent residual measurement, because the unchanged lindep criterion can
also terminate Krylov. Any solver exception remains failure/PARTIAL.

After a successful engineering gate, a potential B8/native1 Spinosin pilot should
record actual CPHF block time, iteration count, `/proc` VmSize/RSS before native
calls, completed cache hashes and remaining budget. Only measured results can
justify a full-Hessian continuation. Cross-process restore is still
`NOT_IMPLEMENTED_SAVE_ONLY`; this report supplies no fictitious resume command.

Recompute the arithmetic and source hash audit to a **new** path:

```bash
python reports/pyscf-spinosin-continuation-20261031-v2/b8-performance-audit/audit_memory_and_work.py --output /tmp/research4-b8-audit-recomputed.json
```
