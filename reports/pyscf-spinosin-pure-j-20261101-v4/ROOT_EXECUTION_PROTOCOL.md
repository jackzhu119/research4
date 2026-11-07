# V4 Spinosin pure-PBE calculation protocol

Status: design fixed before new quantum evaluation. Implementation review,
code checks, and actual water/ethanol equivalence gates must pass before the
Spinosin Hessian is launched. This document is not a calculation result.

## Authorization and scope

The current user instruction is: “你继续完成后续任务，可以一直运行我授予你权限”.
Root uses this continuing authorization for a new **finite** calculation round,
with 36 quantum CPU hours and an 18-hour wall limit. A single Spinosin frequency
phase may request at most 34 CPU hours and 12 wall hours. This is a separate
budget, not a claim that it fits inside the earlier V3 48-hour allocation.
Actual historical and new costs must be reported separately and cumulatively.

The reason for the new round is measured: the closed V3 frequency attempt used
89,154.115469 CPU seconds, received SIGXCPU, and did not complete the Hessian.
V3 used 111,023.488913 CPU seconds in total. Its configured phase limit of
86,400 seconds was exceeded; neither the sender nor the clock discrepancy was
directly recorded. All original budgets, logs and failed results remain intact.
The original one-thread/3,300-second AGENTS pilot limit describes an earlier
round; this current authorized round has its own explicit protocol and ledger.

No main merge, raw spectrum change, classifier update, or large jujuboside
calculation is authorized by this protocol. The branch remains
`research/pyscf-spinosin-mechanism-20261031`.

## Fixed physical calculation

- Spinosin C28H32O15: 75 atoms, 320 electrons, charge 0, singlet.
- Reuse the documented C001 conformer lineage and closed native direct
  optimization from V3. Reusing its geometry is not a new optimization or a
  new conformer. Checkpoint data may initialize a fresh SCF, not restore a
  previously unfinished Hessian.
- PBE/def2-SVP, spherical basis, grid level 1, no density fitting and no
  dispersion correction. This is explicitly **not** PBE-D3 or B3LYP-D3.
- Preserve direct-SCF tolerance 1e-13, SCF energy tolerance 1e-8 Hartree,
  full AO/MO space, the existing full-system Hessian assembly, full response
  coupling, response blocks of 8 atoms and native direct thread limit 4.
- Preserve the V8 scoped Hessian memory strategy. PySCF max_memory is a
  heuristic; AS 24 GiB is virtual address space, not physical RAM. Retain
  RSS 10 GiB/cgroup 12 GiB best-effort stop thresholds on a 16 GiB machine.

The sole intended numerical engineering change is omission of K contractions
whose result is not consumed by the pure-PBE partial-Hessian expression.
Original J/core terms, density slices, coefficients and return shapes must be
retained. The with_k=True path delegates to the unchanged stock implementation.
Do not enable the derivative-optimizer suffix branch: its full-density versus
slice contract has not been validated. Original PySCF and V8 files stay unchanged.

The SCF/CPHF screened-direct label does not establish that every derivative
integral uses a VHF optimizer. The ABI audit and its scoped limitations apply.
Any removed K bindings are recorded explicitly, independently of the thread
wrapper's claim that the wrapper itself preserves operator arguments.

## Required actual equivalence gates

Use the closed optimized water and ethanol coordinates. For each, calculate a
new converged SCF and gradient, followed by two complete Hessians on the same
MF, geometry, MOs and actual grid:

1. Stock PySCF partial Hessian with the already validated V8 B8 assembly.
2. Pure-J partial Hessian with the same B8 assembly and resource settings.

The reference is described as stock partial plus the V8 engineering wrapper,
not as an unmodified default full-batch solver. Earlier water/ethanol evidence
separately covers the B4/B8/native-thread engineering equivalence.

Predeclared gates are maximum full raw-Hessian difference <= 1e-6 Hartree/Bohr²,
maximum paired frequency difference <= 0.01 cm^-1, complete expected mode counts
and positive-frequency minima for both validation molecules. Verify all matrix
entries, including cross-block terms, and verify unchanged MF/grid bindings.
Record both timings without extrapolating small-molecule speedups to Spinosin.
All failures are retained. Failure blocks the Spinosin launch.

These are fresh SCF/gradient/Hessian comparisons, not newly run geometry
optimizations. Existing full Opt/Freq controls remain distinct from this gate.
Only root launches quantum workers through the new actual-CPU supervisor.

## Resource and evidence gates

The new supervisor must measure the child's entire phase CPU from /proc, not
CPU since the monitor starts. Check target PID/start time/UID/process-group and
session identity before any group signal. Preserve profiling RLIMIT as an
additional control, but do not describe it as a verified strict actual-CPU cap.
Use conservative cleanup reserves and bounded TERM/KILL escalation. Record
sampling and scheduling limitations, final wait4 cost, and every stop reason.
Stop/monitor failures must not become successful phase records.

Freeze reviewed source hashes, dependency versions, commands, input hashes and
the code commit before actual gates. Use unique output directories. Close and
audit every stage before starting the next. Do not append to the closed V3 ledger.

Save complete RHF core/J helper arrays before the subsequent RKS XC work where
possible. They are **incomplete intermediates**, not a complete PBE Hessian and
not vibrational modes. Cache restoration remains NOT_IMPLEMENTED unless a
separate cross-process implementation is actually validated.

## Scientific acceptance and interpretation

Spinosin requires a fresh same-method stationary gradient, a complete analytic
Hessian, saved raw/symmetrized matrices, actual normal modes and exactly 219
vibrational frequencies after translation/rotation removal. Preserve signed
negative frequencies. Computation complete and positive-frequency minimum
verified are different statuses. Do not derive modes from incomplete caches.

IR intensities remain NOT_COMPUTED. Mode-position diagrams are not absorption
intensity spectra. Only qualified complete positive-frequency results enter
the fixed 0.1–1.5 THz experimental-candidate comparison. Frequency conversion
uses 1 cm^-1 = 0.0299792458 THz without fitting a scaling factor to those peaks.
Keep all fixed experimental candidates, not only attractive nearest matches.
Proximity does not establish chemical attribution in solid medicinal material.

No theoretical frequency enters the existing 96.0234% classifier. Water and
ethanol are software controls, not additional medicinal-plant samples or
Spinosin modes. Report DONE/PARTIAL/NOT_RUN/BLOCKED according to actual evidence.
