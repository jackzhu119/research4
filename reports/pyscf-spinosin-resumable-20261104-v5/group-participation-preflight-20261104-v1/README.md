# C001 normal-mode group participation: static preparation only

Status: **NOT_RUN for Spinosin mode participation**. Seventeen pure tests and
the real static preflight passed. No target normal-mode arrays, Hessian,
checkpoint or frequencies were read. No target participation values were
generated. No quantum calculation, training or Git operation was launched.
The original analyzer, quantum sources and historical outputs were not changed.

The new independent script is
`scripts/analyze_spinosin_group_participation_20261104.py` (SHA-256
`b997a51f1d57b29cb2f05b67209bf993d1882af504a4af164c6912ed497c2dcd`).
Its test source is `tests/test_spinosin_group_participation.py` (SHA-256
`638637e9153a81298695d37345fe23b5794b7c5fd9a7f2703bf37c7f81b061ba`).
`execution-receipt.json` records the commands and actual outcomes;
`pure-tests.log` retains the real test output. The nested `target-NOT_RUN/`
contains the static preflight and its artifact index, with no result CSV.

The quantity is the dimensionless mass-weighted squared Cartesian normal-mode
share:

`P(group,mode) = sum_group(m_i * |d_i|^2) / sum_all(m_i * |d_i|^2)`.

PySCF's saved `norm_mode` is the mass-weighted orthonormal eigenvector divided
by the square root of atomic mass. The exact inspected `thermo.py` source SHA is
`8143ebed7653bd79b4b8d7b2f5933671a33285b11aba47cf0d8134c229a031f8`.
Its lines 95–98 define this coordinate convention. Atomic masses match the
frozen analyzer: H 1.008, C 12.011, O 15.999 amu. The raw saved modes must pass
their mass-normalization check before calculating any shares.

Four mutually exclusive C001 structural regions cover all 75 atoms and must
sum to one for every mode. The 18 additional graph subgroups, including OH,
overlap and are never summed as a partition. These labels use the old trusted
C001 SDF indices. The newly uploaded SDF has a different atom order and is
excluded. Static package SHA:
`5ecf8bbe7c5317d5ef0d18070f01307542c366f0700bcbaf43e5e7849a5aa246`.

The actual entry requires the complete output of frozen analyzer
`a2e3588c6dea121be58e4a69afb137bc6e5092199ca4180937fcdb34f6f57154`,
an explicit expected artifact-index SHA, 219 positive modes, unchanged source
MF and the exact C001 snapshot. It independently rechecks the original healthy
V5 producer and source SCF closure and fresh-gradient receipt **before** any
target NPY hashing or loading. Negative/zero frequencies, incomplete results,
software-control results or a false mechanism qualification block the entry.
Only then does it verify and read the closed mode/geometry arrays and compare
the geometry to the pinned prepared C001 XYZ. It never reads the Hessian or
checkpoint and never invokes PySCF.

After those conditions actually hold, root may run:

```bash
.runtime/bin/python scripts/analyze_spinosin_group_participation_20261104.py \
  --analysis CLOSED_A2_ANALYSIS_DIRECTORY \
  --analysis-index-sha256 ACTUAL_CLOSED_ARTIFACT_INDEX_SHA256 \
  --output NEW_PARTICIPATION_DIRECTORY
```

The uppercase arguments are placeholders, not completed calculation paths.
The output refuses an existing directory. The CLI uses one CPU affinity,
one BLAS thread and an address-space limit of 1 GiB. A successful actual run
would produce `mode_group_participation.csv` (219×22 rows),
`partition_participation.csv` (219 rows and four regions),
`low_frequency_mode_group_participation.csv` (the 0.1–1.5 THz view),
`checks.json`, `README.md` and `artifact_index.json`.

These shares describe where an eigenvector is concentrated. They are not
vibrational energy, IR intensity, a measured displacement amplitude or an
experimental absorption assignment. They omit phase and motion type. OH
participation does not prove hydrogen bonding. Individual shares depend on the
chosen basis within degenerate mode subspaces, although sign and global
per-mode scaling leave them invariant. Isolated-conformer modes do not establish
solid-herb chemical peak assignments. IR remains `NOT_COMPUTED`; no frequency
or participation quantity enters the classifiers or changes the existing
96.0234% classification result.
