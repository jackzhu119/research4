# V6 stage-boundary contingency — final static preparation, NOT_RUN

This is a new contingency implementation. It does not modify or replace the
currently running V5 calculation. No numerical package, target array, quantum
engine, real authorization, budget registration or Git operation was used to
prepare it. Water and ethanol numerical controls remain **NOT_RUN**. The pure
stdlib tests and static review do not establish numerical equivalence.

Entry: `scripts/pyscf_staged_hessian_worker_20261105_v6.py`. The script is
stdlib-only on import and in preparation mode. `--execute` first verifies an
actual frozen V6 supervisor, before numerical imports. Its frozen guard is
`scripts/run_pyscf_continuation_budget_20261105_v6.py`, SHA
`ae2cc7bc8ed4f3d6f39e13a842e635e5a26152c02aab52d6af25d91cc65f5b1b`.
Registration and root allocation are separate guard operations. This worker
does not initialize a ledger or create an authorization. Existing V5 costs
must remain included. The real research deadline and any finite new allocation
must be established by root after the current V5 phase actually stops.

The stages are:

1. `rks-partial`: restore the exact existing SCF, reuse only independently
   healthy compatible RHF units, recompute any final missing unit, and finish
   the **complete full-system RKS electronic partial Hessian**. For the target,
   only a healthy149/150 seed is accepted. The failed producer's final RHF unit
   or incomplete XC arrays are rejected, regardless of a completion filename.
   The resulting75×75×3×3 matrix includes the electronic RHF and XC terms,
   without nuclear repulsion or CPHF response.
2. `h1ao`: after the partial producer has exited healthily, compute and save
   every one of the75 full3×762×762 AO derivative blocks. No fragment or
   subset of atoms is accepted.
3. `global`: after both complete producers have exited healthily, call the
   original stock full kernel once. Its partial and H1 providers return fresh
   owned copies from the closed caches. Its full-AO/MO CPHF response uses the
   frozen V8 eight-atom scheduling and shared full-system response operator.
   The original nuclear contribution is added exactly once. The original
   whole-system assembly retains all cross-atom terms. The existing signed
   harmonic analysis then emits3N−6 modes,219 for Spinosin, with IR intensity
   explicitly `NOT_COMPUTED`.

The distinction between **RHF units** and a **full RKS partial Hessian** is
essential.150 completed RHF markers do not constitute an XC Hessian, H1,
response or a completed frequency calculation. Old failed/RUNNING producer
arrays are quarantined by default. Only genuine closed V5 producer proofs or
this new V6 producer schema are supported; there is no automatic forensic
override and no relabelling of old proofs as V6.

Physical bindings contain the exact snapshot, full MO/occupancy/energy arrays,
geometry, actual grid including screening metadata, derivative source hashes,
method and numerical settings. Current stage PID/proof details are separate
producer evidence. Default target inputs remain snapshot3089d601… and restored
gradient6b143def…. The original V5 state loader verifies genuine V5 source
closure; the fresh-gradient receipt is checked before restoring molecular
arrays. Stage proof, healthy closure, result and full manifest checks precede
stage payload reads. Cache manifests are published only after source MF
stability and operator restoration checks. Each stage requires a fresh output
directory and a unique output basename for its save-only capture.

The stock `hess_elec` passes `max_memory=4000` to partial/response methods even
while MF/Hessian resources are scoped to3000MB. The standalone partial stage
preserves this4000 argument. It does not silently substitute3000. No functional,
basis, dispersion, density fitting, quadrature, occupation or response-space
approximation is introduced. Different process RSS can change PySCF's grid
batch size, so mathematical equivalence does not imply bitwise identity;
water/ethanol comparison against genuine full stock controls is still required.

Stock electronic assembly modifies its partial matrix in place. Cache providers
therefore never return a source object or writable mmap. H1 is freshly loaded
per atom, avoiding a permanently retained second1GiB baseline in RAM. Original
instance methods are restored in `finally`, including on exceptions. Explicit
MO/energy/occupation arguments, atom order, the4000 argument and H1 `chkfile=None`
are checked; checking only the MF object's fields would be insufficient.

For continuity, the global stage still uses frozen V8 save-only capture. This
duplicates1,045,159,200 bytes of closed H1 payload into the new capture and must
be included in IO/page-cache planning. The new capture explicitly marks partial
and H1 as restored, not freshly recomputed native stages. CPHF and nuclear terms
are freshly evaluated by the stock global path. Additional IO has not been
hidden by changing the frozen V8 implementation. Cache eviction or other
resource scheduling is root's responsibility between genuinely closed phases.

These results have a new V6 schema and new proofs. The old frozen a2 analyzer
and observer pin the V5 consumer and **do not automatically accept V6 outputs**.
A reviewed V6 postprocessing gate is still needed before any new mechanism
错误混种分类成绩已按2026-11-12用户决定撤回；论文使用main的paper/current白名单。

Preparation command (no quantum work):

```bash
.runtime/bin/python -S scripts/pyscf_staged_hessian_worker_20261105_v6.py \
  --output NEW_PREPARATION_DIRECTORY
```

Execution requires `--execute`, `--stage`, explicit snapshot/G inputs, a real
V6 parent and a fresh output. `rks-partial` additionally needs `--rhf-cache`.
`h1ao` needs the exact `--partial-result` and its SHA. `global` needs that same
partial result and the exact `--h1-result` and SHA. The H1 producer must name
the same partial dependency. Small controls require `--control-validation`
and genuine closed water/ethanol source inputs; they cannot be presented as
Spinosin. No execution command or allocation was run during this preparation.

This v2 package records the final17-test source. The v1 package preserves the earlier14-test development state; its receipts have not been rewritten. Complete cached payload hashes are now checked before new scientific work, and partial provider-install failures restore all prior instance methods.
