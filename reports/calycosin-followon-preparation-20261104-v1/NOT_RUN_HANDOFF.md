# Calycosin neutral Opt/Freq handoff — NOT_RUN

Preparation only, 2026-11-04. This handoff read small source/JSON/MD files while
Spinosin remained active. It did not import NumPy/RDKit/Torch/PySCF, parse SDF or
arrays, access the network, hash large files, run Git, or start a calculation.
No Calycosin energy, optimized geometry, frequency or timing result is supplied.
Root must wait until the current quantum task closes and explicitly allocate the
follow-on resource budget before using any command below.

## Current identity evidence and preserved chronology

The old intake audit at
`reports/pyscf-sdf-continuation-20261103-v1/structure-audit/audit.json`
correctly retains `name_CID_mapping_independently_verified=false` from its
2026-11-04 06:52 UTC intake state. This flag is superseded for current chemical
identity by the later append-only evidence, without rewriting the old artifact:

- `docs/RESEARCH4_CHEMICAL_EVIDENCE_20261104.md` describes the current conclusion.
- `reports/external-evidence-20261104-v1/chemistry/uploaded_identity_overlay.json`,
  created08:20:13 UTC, matches the complete uploaded graph to neutral ChEBI17793
  and CID5280448. The source and uploaded canonical isomeric SMILES match;
  both full InChIKeys are `ZZAJQOPSWWVMBI-UHFFFAOYSA-N`.
- `chemistry/source_registry.json` record `CHEBI_CALYCOSIN` records a primary
  database web-open success and DFT_INPUT_IDENTITY role. `PUBCHEM_CALYCOSIN`
  corroborates the official indexed CID/name/formula/key; direct access had
  CAPTCHA. ChEBI reuse is CC BY4.0.
- `chemistry/graph_and_triage_replay.json` records PASS against the exact uploaded
  SDF SHA and overlay SHA
  `e00dd59f2aa058bd4a65e326ebc1bb041c3576c9650ff57c661385a12133acc1`.
- `chemistry/independent_review.json`, recorded08:32:14 UTC, preserves the
  independent graph/source review. That review performed no quantum calculation.

Use the **neutral** Calycosin graph: C16H12O5,33 explicit atoms,148electrons,
charge0,spin2S0,multiplicity1,no radicals or stereocentres. ChEBI77992 is the
different C16H11O5 anion and must not replace this input.

Naming/neutral-graph identity is now supported. The download channel and
byte-exact official3D export remain independently unverified. The uploaded3D
file is one computed conformer, not an experimental crystal or DFT minimum.
Neither the source register nor this handoff establishes Calycosin content in
the project's Huangqi specimens.

## Existing structure and executable input

- Original user file:
  `data/incoming/molecular_structures_20261103_v1/黄芪CID_5280448.sdf`
  (recorded SHA `1bf68cb55ed29ac48017aa13b967b12922581290e89667efbd69bd42d9ac480f`).
- Faithful source-order XYZ:
  `reports/pyscf-sdf-continuation-20261103-v1/structure-audit/derived/CID_5280448.xyz`
  (recorded SHA `8fda598d70ff9eb70085022d42fb4922283e5d54e5d2fabc37e72d2f060f7236`).
- Faithful coordinates:
  `reports/pyscf-sdf-continuation-20261103-v1/structure-audit/derived/CID_5280448_coordinates_angstrom.npy`
  (recorded SHA `793776316ecf1a6f34b4f80fbd2467a2864d1be7eb5769c92bd96c05ef47b5fa`).
- Existing immutable executable input:
  `reports/pyscf-sdf-continuation-20261103-v1/state-worker-preflight/inputs/cid5280448_optimize_input.json`
  (recorded SHA `842ebe92dad9e7d6d5a56b4c27c515fb10580966e0dd9495bc932fd3bfa76f49`).

The existing input preserves its historical `molecule_id` ending
`NAME_PROVENANCE_PENDING` and false naming flag. Those labels do not block the
generic worker's graph/coordinate validation. It is usable for an explicitly
identified historical-input pilot, with the later overlay cited separately.
For a clean named run, create a new small manifest with identical coordinate
records and atom order, a Calycosin identifier, the updated naming conclusion and
overlay/source-register/replay dependencies. Never overwrite the old manifest;
use the new manifest's actual SHA in the same CLI.

Required input schema is `research4-molecular-input-v1`. `validate_input` requires
all33 `atom_symbols` in exact source order, `composition={C:16,H:12,O:5}`,
`formula=C16H12O5`, `charge=0`, `spin_2S=0`, `multiplicity=1`,
`radical_electrons=0`, `electron_count=148`,
`connectivity_verified=true`, `stereochemistry_checked=true`,
`input_xyz` and `coordinates_angstrom_npy` artifact records, plus optional
`source_artifacts`. Each artifact record allows only `path`, `sha256` and
optional `bytes`. `molecule_id` must be a nonempty string; it is not restricted
to Spinosin. No initial checkpoint is needed for the uploaded starting conformer.
Do not combine its atom order with Spinosin MOs or derivative caches.

## Ledger registration and resource limits

All numerical child commands below must run through
`scripts/run_pyscf_actual_cpu_budget_20261104_v5.py`, source SHA
`33f1dc3db5280f16c9898c561e345143614730dc052b171b4f11fbbb079cdeee`.
`--phase` automatically registers a fresh RUNNING phase and real launch proof;
there is no separate register-only CLI. The guard refuses unresolved phases and
clamps each request to the **existing** remaining CPU and original wall deadline.

`CALY_LEDGER` below must be an already valid, closed V5 pool that root has
explicitly approved for these bounded phases. Using the existing pool does not
create extra budget. V5 forbids fresh `--initialize` and supports only migration
of a specifically closed V4 pool. **Do not remigrate the original old V4 snapshot
to reset the consumed Spinosin cost.** If a genuinely separate later pool is
needed, prepare a reviewed resource record/guard that binds the current closed
V5 totals; the existing V5 initializer is not a generic new-pool allocator.

The guard is not chemically hardcoded to Spinosin. Its authorization still binds
source checkpoint/revision and historical ledgers; these bindings must remain
valid and must not be faked or silently replaced for Calycosin.

Retain `--memory-gib 24`: it is the reviewed24-GiB virtual-address cap, not a
24-GiB physical-RAM claim. The worker verifies this exact cap. Existing physical
guards remain10-GiB process RSS/12-GiB total cgroup, with the16-GiB cgroup envelope.
Prefer one thread for the first33-atom pilot; native direct-driver scratch is
not simply proportional to molecular atom count. Allocate no concurrent ML,
Git, package changes, downloads or bulk I/O during the quantum phase.

## Exact supervised CLI sequence — run only after root authorization

These are bounded starting grants, not guaranteed completion times or a new
authorization. `set -e` stops the sequence if a supervisor phase fails. All
directories/phase names must be new. Root may choose smaller actual grants.

```bash
set -euo pipefail
export PYTHONPATH=.runtime/quantum-pilot-site:.runtime/quantum-upgrade-site
CALY_LEDGER=reports/pyscf-spinosin-resumable-20261104-v5/cpu_budget.json
CALY_RUN=reports/calycosin-pilot-20261104-v1
CALY_INPUT=reports/pyscf-sdf-continuation-20261103-v1/state-worker-preflight/inputs/cid5280448_optimize_input.json
CALY_INPUT_SHA=842ebe92dad9e7d6d5a56b4c27c515fb10580966e0dd9495bc932fd3bfa76f49

# Opt + fresh final same-method SCF/G; no Hessian in this state worker.
.runtime/bin/python scripts/run_pyscf_actual_cpu_budget_20261104_v5.py \
  --ledger "$CALY_LEDGER" --phase calycosin-opt-001 \
  --cpu-seconds 3600 --wall-seconds 7200 --threads 1 --memory-gib 24 \
  --log "$CALY_RUN/execution/calycosin-opt-001.log" -- \
  .runtime/bin/python scripts/pyscf_molecular_state_worker_20261104_v5.py \
  --phase optimize-state --input-manifest "$CALY_INPUT" \
  --input-manifest-sha256 "$CALY_INPUT_SHA" \
  --output "$CALY_RUN/calycosin-opt-001" --threads 1 --maxsteps 35

# Only read the small result JSON after the preceding supervisor returns0.
CALY_SNAPSHOT_SHA=$(.runtime/bin/python -c \
  'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="DONE" and r["geometry_optimized"] and r["scf_converged"] and r["final_gradient_thresholds_passed"]; print(r["snapshot_manifest"]["sha256"])' \
  "$CALY_RUN/calycosin-opt-001/result.json")
CALY_SNAPSHOT="$CALY_RUN/calycosin-opt-001/exact-mf-snapshot"

# A true cross-process restore/G gate; no new SCF and no grid rebuild.
.runtime/bin/python scripts/run_pyscf_actual_cpu_budget_20261104_v5.py \
  --ledger "$CALY_LEDGER" --phase calycosin-restoreg-001 \
  --cpu-seconds 900 --wall-seconds 1800 --threads 1 --memory-gib 24 \
  --log "$CALY_RUN/execution/calycosin-restoreg-001.log" -- \
  .runtime/bin/python scripts/pyscf_resumable_hessian_worker_20261104_v5.py \
  --phase restore-gradient-check --snapshot "$CALY_SNAPSHOT" \
  --snapshot-manifest-sha256 "$CALY_SNAPSHOT_SHA" \
  --output "$CALY_RUN/calycosin-restoreg-001" --threads 1

CALY_GRESULT="$CALY_RUN/calycosin-restoreg-001/result.json"
CALY_GSHA=$(.runtime/bin/python -c \
  'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' \
  "$CALY_GRESULT")

# The original full-system stock RKS Hessian + B8 full CPHF + 3N-6 modes.
.runtime/bin/python scripts/run_pyscf_actual_cpu_budget_20261104_v5.py \
  --ledger "$CALY_LEDGER" --phase calycosin-stockfreq-001 \
  --cpu-seconds 7200 --wall-seconds 10800 --threads 1 --memory-gib 24 \
  --log "$CALY_RUN/execution/calycosin-stockfreq-001.log" -- \
  .runtime/bin/python scripts/pyscf_resumable_hessian_worker_20261104_v5.py \
  --phase stock-hessian-reference --snapshot "$CALY_SNAPSHOT" \
  --snapshot-manifest-sha256 "$CALY_SNAPSHOT_SHA" \
  --gradient-check-result "$CALY_GRESULT" --gradient-check-result-sha256 "$CALY_GSHA" \
  --rhf-cache outputs/calycosin-pilot-20261104-v1/rhf-cache \
  --output "$CALY_RUN/calycosin-stockfreq-001" --threads 1
```

## Qualification and method details

The state worker is
`scripts/pyscf_molecular_state_worker_20261104_v5.py` (frozen SHA
`be4727b76d59a0b0dd3defba249ef833ee8ef0ea89473e390dcef389728415c8`).
It supports neutral singlet C/H/O compositions, so33 atoms are eligible. It uses
PBE/def2-SVP/grid1, noDF, **noD3**, screened direct SCF with1e-13 tolerance,
SCF convergence1e-8 Eh/max80cycles, fixed geomeTRIC thresholds and a fresh final
gradient. Atom gradient RMS must be<=3e-4 and maximum<=4.5e-4 Eh/Bohr.

`restore_snapshot` checks immutable source and arrays, DONE Opt result, optimized
geometry, fresh SCF/G stationarity and its **actual healthy closed V5 parent**:
EXITED/0, known wait4 CPU, unchanged proof/log/source, all resource-failure flags
false and no residual child group. A child-written DONE result alone is
insufficient. No orbital/geometry combination from another molecule is accepted.

The generic frequency consumer is
`scripts/pyscf_resumable_hessian_worker_20261104_v5.py` (SHA
`b40188f747c01e9efa97672af6eedfd281a9e9697d20bc7d57f125bb131fa190`).
It accepts the exact restored MF, a healthy closed same-boot gradient receipt,
full stock hcore/gradient/Hessian method identities and source-binding checks.
`stock-hessian-reference` does not enter the incremental partial-helper patch.
It runs the complete standard global RKS kernel, not a fragment Hessian.

Expected Calycosin dimensions are33atoms,99Cartesian coordinates and93nonlinear
vibrational modes. These are formula-level expectations, **not computed results**.
Successful full calculation writes signed cm^-1/THz `frequencies.csv`, raw and
analysis Hessians, normal modes, masses, settings and logs. All negative/zero
frequencies remain visible. `FREQUENCIES_COMPUTED` does not itself prove a
positive-frequency minimum; IR is alwaysNOT_COMPUTED. Requested CPHF tolerance
and normal solver return are not a measured independent residual norm.

For interruptible RHF work, use the same generic consumer's `hessian-segment`
with a dedicated cache, at most10newunits per cleanly closed phase, total66units
(33diagonals +33rows), then full XC/H1/CPHF and93modes. The current serial
controller `run_pyscf_serial_segments_20261104_v5.py` hardcodes Spinosin75/150/219
and cannot orchestrate this33-atom task without a separately reviewed version.
The stock single-phase workflow above avoids that new-controller dependency.

Other identity-specific tools must not be misused:

- `prepare_spinosin_conformers_20261031.py` hardcodes75atoms/Spinosin.
- `analyze_spinosin_v5_modes_20261104.py` permits75-atom Spinosin, or explicit
  water/ethanol controls; it does not accept Calycosin. Do not relabel the
  molecule to bypass the guard. A later generic33-atom independent NumPy mode
  replay/plotter needs a separate reviewed version.
- The consumer's ignored SAVE_ONLY intermediate directory name still contains
  `pyscf-spinosin-resumable-20261104-v5/hessian-saveonly`. Unique
  `calycosin-*` output basenames keep these new artifacts separate; report their
  real CID identity. No old Spinosin cache/geometry/model is overwritten.

## Relative resource feasibility and remaining scientific limits

The sealed graph triage estimates354spherical def2-SVP AOs and74occupied orbitals,
versus762/160 for Spinosin. Full AO H1 storage is formula-estimated near94.7MiB
versus996.7MiB (about9.5%). Such counts are resource proxies, not measured timing
or a guaranteed speedup. Native-driver scratch and the2000-MB XC memory floor do
not scale down proportionally. Keep the reviewed memory guards and exclusive
execution even for33atoms, profile actual SCF/G cost before granting more CPU,
and stop/save PARTIAL at the approved budget rather than extend automatically.

The smallest credible first experiment is one documented uploaded conformer,
real Opt/freshG, then all93modes; label it a single-conformer isolated-molecule
pilot. No local Calycosin conformer search has been run. The PubChem-style
MMFF energy field is a source-file claim, not our force-field or DFT result.
The literature Molclus/M06-2X/6-311+G(d,p) optimized-XYZ table and CCDC242382
are listed external leads, not acquired geometries. Additional conformer and
grid/method sensitivity studies require separate explicit runs. Isolated modes,
especially low-frequency torsions, are not solid Huangqi THz chemical
assignments or proof of the compound's concentration in the samples.
