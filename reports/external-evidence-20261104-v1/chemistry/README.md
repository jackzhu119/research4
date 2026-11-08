# Chemical source evidence, 2026-11-04

This independent subtask ran web searches, primary-source reading and RDKit graph
checks only. It did not run SCF, gradients, optimization, Hessians, force fields,
classifiers, or Git. The active quantum calculation is owned by root elsewhere.

- `source_registry.json` / `.csv`: 17 source records, authority, access, licence,
  intended use and limits. Source records are not independent specimens.
- `uploaded_identity_overlay.json`: append-only ChEBI-to-upload identity match.
  The two complete stereo keys and canonical isomeric graphs match exactly.
  Original SDFs and the prior audit are unchanged. 3D download-byte identity and
  experimental crystal identity are not asserted.
- `graph_and_triage_replay.json` / `chemical_candidates.csv`: actual graph replay
  and five candidate records. Atom/mode/AO counts are formula-level metadata;
  there are no new quantum frequencies or timing benchmarks.
- `external_assets.json`: five specifically identified external data assets,
  all still unacquired; stable download links and blockers are retained.
- `download_attempts.json`: two real bounded cloud HTTP attempts returned
  CONNECT403. No proxy bypass or fabricated downloads were used.
- `literature_peak_reference.csv`: three published pure-PA positions, with
  specimen/modality/grain restrictions. It is not an independently sampled raw
  spectrum or a chemical assignment for project855.
- `method_cautions.json`: the reference paper models isolated construction
  units, not a periodic solid. Its text/Table1 use1.22/1.20THz for one cocrystal
  comparison; preserve this difference rather than choose the better match.

Source material is summarized and linked, not mirrored wholesale. ChEBI's own
licensing page explicitly states CC BY4.0. PubChem references third-party spectra
with separate rights; none were copied. Two chemistry articles have CC BY-NC-ND
terms; full adapted tables and full text were not redistributed. There is no
new dataset cleared for direct origin-classifier training in this chemical
register. Future paired specimen assays or frozen physical priors need their
own source-isolated protocol and fair controls.

Recompute the graph and sizing checks into a fresh directory:

```bash
PYTHONPATH=.runtime/quantum-pilot-site:.runtime/quantum-upgrade-site \
  .runtime/bin/python \
  reports/external-evidence-20261104-v1/chemistry/replay_identity_and_triage.py \
  --output outputs/chemical-evidence-replay-new
```

Rebuild the manually checked source register into a fresh directory:

```bash
.runtime/bin/python \
  reports/external-evidence-20261104-v1/chemistry/build_chemical_registry.py \
  --output outputs/chemical-source-register-new
```

The linked report is `docs/RESEARCH4_CHEMICAL_EVIDENCE_20261104.md`.
