# Incremental pure-PBE RHF-helper checkpoint design

Status: **IMPLEMENTED — PURE SYNTHETIC TESTS PASS; REAL PySCF VALIDATION NOT RUN**.
This document does not represent a completed Hessian, a restored calculation, a
new SCF evaluation, or a vibrational result. The earlier V4 phase has no completed
RHF helper state, so none of its unfinished integral work can be recovered by this
design. Only root may authorize and launch subsequent quantum verification.

## Frozen mathematical reference

The reference is the installed PySCF 2.14 `hessian/rhf.py:_partial_hess_ejk`
and the already tested V9 `pure_j_partial_body`. Only the non-DF, nonhybrid,
closed-shell PBE branch with all atoms in original order is in scope. Preserve
the reviewed derivative-optimizer `None` contract. No suffix fix, AO/MO-space
restriction, atom removal, changed response tolerance, density fitting,
dispersion correction or new functional is introduced.

Let `D=2*Cocc*Cocc.T` and `W=2*einsum(Cocc,Cocc,eps_occ)` be the original density
and energy-weighted density. For atom `i`, let `P_i` denote its complete AO slice.
The diagonal Coulomb contribution that currently keeps a `(3,3,nao,nao)` array
alive is only consumed through

```
Jdiag[i] = 2 * einsum('xypq,pq->xy', V_ipip1[:,:,P_i,:], D[P_i,:])
```

For Spinosin, the original AO tensor has 9*762*762*8 = 41,806,368 payload bytes;
all 75 precontracted `(3,3)` blocks need only 75*9*8 = 5,400 bytes.

For each `i`, keep the original two per-atom derivative calls and the original
J combination:

```
V_i = J_ip1ip2(i)
V_i[:,:,P_i] += 0.5 * transpose(J_ipvip1(i), (0,2,1))
V_i = reshape(V_i, (3,3,nao,nao))

EJrow[i] += Jdiag[i]
E1row[i] -= 2 * <S_aa[:,:,P_i,:], W[P_i,:]>
for j in 0..i:
    EJrow[j] += 4 * <V_i[:,:,P_j,:], D[P_j,:]>
    E1row[j] -= 2 * <S_ab[:,:,P_i,P_j], W[P_i,P_j]>
    E1row[j] += <Hcore_second(i,j), D>
```

The saved `E1row` and `EJrow` have shape `(i+1,3,3)` and already contain all
diagonal, overlap and core terms for that row, in the original arithmetic order.
On load, assign each completed lower row once and fill upper entries only with
`E[j,i]=E[i,j].T`. Do not add the diagonal again; do not independently symmetrize
the diagonal `(3,3)` block. `EK` remains an explicitly zero full-shape result in
this pure-PBE branch; every `with_k=True` request must still delegate unchanged.

## Two global-J checkpoint choices

1. **Minimal algebra reordering:** retain the one original global `ipip1` call,
   precontract every `Jdiag[i]` immediately, save the small map, and release the
   AO tensor. This retains the 151 derivative-call structure for 75 atoms. It
   cannot protect work until that first global call returns.
2. **More interruption-resistant partition:** evaluate `ipip1` with its first
   shell index restricted to `P_i`, while all other three AO indices and the
   contraction density remain complete. The `s2kl` symmetry, nine components,
   operator and J script stay unchanged. This is a partition of the original
   output index, not an electronic-space or molecular fragment approximation.
   The existing per-atom `ipvip1` J call demonstrates the same sliced-output
   shape convention. It replaces one global call with 75 atom-sliced calls,
   bringing the RHF-helper call count to 225. Timing may improve or worsen;
   neither speed nor numerical equivalence is established by this design.

The implemented second choice first publishes all atom-sliced `Jdiag` units,
then each complete lower row (two derivative calls per row). Each published
unit is independently durable. This avoids making a single large global
operation the first recoverable point. It still requires actual water/ethanol
verification against the frozen stock/full reference.

## Minimal durable state and trust boundary

Save a one-time immutable converged-MF snapshot before the Hessian: serialized
Mole/geometry/basis state, complete MO coefficients/energies/occupations and SCF
energy, actual initialized grid coordinates and weights, and the actual grid
screening mask when present. Spinosin's recorded grid coordinates and weights
are about 9.7 MB together; saving them removes the earlier hash-only restoration
gap without copying the approximately 1 GB H1 intermediates.

Separate three records:

- **Physical binding:** exact finite array dtype/shape/value hashes, canonical
  Mole identity (excluding only the already reviewed transient RINV marker),
  charge/spin/cartesian convention, full atom list, basis, PBE/non-DF/no-D3,
  grid values and masks, numerical settings and all loaded implementation hashes.
- **Producer evidence:** actual completed source SCF and same-state gradient,
  source checkpoint/input hashes, immutable source code and actual old execution
  proof. Never relabel an abandoned phase as a successful process exit.
- **Consumer evidence:** a new real resource proof and explicit restored-state
  status. It must match the old physical binding exactly. A new SCF creates a
  new wavefunction and cannot be mixed with cached old rows.

A resume should verify old converged-state evidence, restore the exact old MO
and grid values, and preferably recompute a gradient using that restored state.
Report this as restored SCF evidence plus a new same-state gradient, not a newly
performed SCF. Whether a producer abandoned during environment replacement can
be consumed requires an explicit root forensic release, never a fabricated
`DONE` row or guessed final CPU cost.

Each row must be an immutable publication: lower `e1`, lower `ej`, and a complete
JSON record binding their byte hashes, row/atom indices and the physical-header
hash. Maximum two-array payload at row 74 is only 10,800 bytes. Use closed-file
atomic no-replace publication and directory synchronization. Incomplete writes
are not checkpoints. Reattempts use new names and retain orphan evidence.
Require a unique contiguous completed prefix; duplicates, conflicts, missing
rows, changed source/binding, nonfinite values or unexpected shapes fail closed.

Published immutable row records can be backed up independently. No reader may
inspect an actively written HDF5/checkpoint file. A published row is an incomplete
pre-XC intermediate, never a PBE Hessian or a source of normal-mode frequencies.

## Proposed new code boundaries

- `scripts/pyscf_incremental_rhf_checkpoint_20261103.py`: binding, immutable
  store, atom-row evaluation/assembly and a scoped pure-PBE adapter.
- A separate V10 worker entry: preserve actual V8/V9 provenance, record the real
  new entry and incremental-module hashes, and use the unchanged real V4 resource
  supervisor when current hardware satisfies its policy. Do not spoof `__file__`
  or rewrite historical source SHA fields.
- Leave RKS XC, H1, complete CPHF response and native global/nuclear assembly
  unchanged initially. A complete restored RHF tuple is supplied to the existing
  RKS caller, which continues the original full-system calculation.

The newly authorized production implementation is
`scripts/pyscf_incremental_rhf_checkpoint_20261103.py`; pure tests are in
`tests/test_pyscf_incremental_rhf_checkpoint.py`. The old V8/V9/runner files
remain unchanged. No actual quantum calculation has been launched by this
implementation task. The source must be frozen before actual cross-process
validation, because its own SHA participates in the cache identity.

The public interface is `IncrementalPureJContext(hobj, cache_dir,
state_binding, max_new_units=10, *, rhf_module=None, execution_evidence=None)`.
Its finite segment stops with `CheckpointSegmentComplete.progress`, which is
an explicitly incomplete RHF-helper status. `state_binding` is stable verified
source-snapshot identity; consumer execution proof remains separate. This
low-level store does not certify supervisor ledgers: its caller must verify the
source and every consumer's resource execution evidence. Published small units
are valid intermediates even though they do not complete RKS XC/H1/response.

Atomic publication uses unique `*.writing-*` files plus no-replace hard links,
fsync of closed payloads and their directory ancestry. Failed write/link
temporaries and incomplete attempts are retained as orphan evidence. Only
unique completed units in a continuous prefix can be consumed.

The actual 16-case pure test run passed. It compares complete synthetic e1/ej
and zero-ek matrices with the frozen V9 full-global-J reference; tests every
single-unit boundary; checks no cached integral calls, mutable-return isolation,
MO/grid/snapshot changes, atom ordering, duplicate/hole/corruption rejection,
orphan preservation, derivative/model/resource gates, and context restoration.
Every synthetic sliced callback derives from one fixed full virtual AO tensor.
These are not real integral, real molecule, real separate-process, or frequency
results. A first test invocation found a tuple-item augmented-assignment error
in the test itself; slice assignment corrected it before the passing run.

## Mandatory verification before medicinal-molecule use

1. Pure synthetic algebra tests for every interruption boundary: no double-added
   diagonal, exact row transpose restoration, complete atom coverage, no cached
   row integral calls, and rejected tampered/partial/duplicate bindings. Such
   tests are explicitly not quantum calculations or chemical results.
2. Real water and ethanol on one physical MF state: stock complete Hessian versus
   uninterrupted incremental complete Hessian, full raw cross-coupling difference
   <=1e-6 Hartree/Bohr² and all-mode difference <=0.01 cm^-1.
3. A real separate-process interruption/resume for both molecules. Restore exact
   MO/grid state, complete previously unpublished rows, then the unchanged RKS
   XC/H1/CPHF/global assembly. Compare full matrices and all 3/21 modes with the
   uninterrupted reference; verify positive-frequency status independently.
4. Verify a new real resource supervisor and cumulative accounting for every
   source and consumer process. Preserve successful, negative and interrupted
   evidence with truthful costs and unique output paths.

Only after these gates may root run Spinosin. All 219 modes, signed imaginary
frequencies, IR `NOT_COMPUTED`, and experimental-peak interpretation limits remain
错误混种分类成绩已按2026-11-12用户决定撤回；论文使用main的paper/current白名单。
