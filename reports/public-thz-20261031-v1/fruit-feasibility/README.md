# Princeton Meta-Sticker fruit: independent feasibility audit

**Status: conditional exploratory regression data, not an acid-jujube validation cohort.** The original archive contains real numerical waveforms and Brix/DM spreadsheet values. Its 30 persistent fruit IDs are repeatedly observed. The archive does not establish calibrated physical axes, reference processing, target measurement method or a redistribution license. No model was trained by this audit.

## Provenance and licensing

Original public author repository: <https://github.com/PrincetonUniversity/-Sub-Terahertz-Fruit-Sensing>, commit `16943a2c499ce7918afb1e20a223267d94ad7a93`.

- Downloaded ZIP: **138,001,691 bytes**, SHA-256 `902cca56a744187dd3eb3bbed41aa1ce28eb5e75e5b003946d29007a7fed603f`.
- Local original: `data/external/public_thz_20261031_v1/acquired/princeton_metasticker_fruit/author_repository.zip`. Acquisition is recorded separately under `../acquisition/`.
- `CITATION.cff` explicitly identifies **10.1145/3625687.3625815**, the SenSys conference paper, published in 2024. It does not identify the later Nature Food paper. Princeton's official publication record corroborates this citation.
- No LICENSE/LICENCE/COPYING file was detected. The public GitHub ZIP and its numerical derivatives remain locally ignored. Public availability is not treated as permission to mirror the numerical data into this GitHub project.
- The related **Nature Food 6, 97–104 (2025)** paper, DOI **10.1038/s43016-024-01083-x**, points to processed data at **10.6084/m9.figshare.27299859**. Figshare lists CC BY 4.0 and 265.05 kB. This is a separate deposit, not an inherited license for the 138 MB GitHub archive. Numeric cohort equivalence has not been established.
- The later paper describes NIR reference readings and explicitly discusses their limitations. That description does not independently establish the measurement method/calibration of this earlier workbook. No claim is made that these are chemical assays.

See `source_evidence.json` for source URLs and evidence boundaries. No data were downloaded again for this audit.

## What the original files actually contain

| Item | Actual count / meaning |
|---|---|
| MAT files | 2,525, all finite, nonconstant float64 waveforms of shape 1×7,000 |
| Fruit MAT files | 2,445, including 60 numerically identical alias copies at the first timepoint |
| Unique fruit/time/mode waveform keys | 2,385, after identifying exact duplicate aliases; originals are unchanged |
| Persistent fruit IDs | 30: MA1–10 mango, PR1–10 pear, PM1–10 persimmon |
| Scheduled times | 40, 12-hour AM/PM labels from 2023-05-13 21:00 through 2023-06-02 09:00 |
| Acquisition modes | `Tag` and `NT` are paired observations of the same fruit, not independent specimens |
| Background / Metal | 40 files each; **each same-session Background and Metal pair has exactly the same numerical array** |
| Workbook | 3 sheets, each 40 time rows ×10 fruit columns; mango cell order is DM, then Brix |
| Supporting ZIPs | 41 distinct byte hashes; all contain an identically dated 2021 `Reference.picotd` header |

There are **100 exact-array duplicate groups**: 60 first-timepoint fruit aliases plus 40 Background/Metal pairs. The alias files differ in MAT wrapper bytes/variable names (`data_F` versus `Fruit_data`) but match numerically. The auditor validates the numerical match before choosing the canonical explicitly T/NT-named member for an interface. This is not score-based sample deletion.

The same fruit ID across days, modes and file aliases must remain in one group. The spreadsheet and filenames consistently join by **(fruit ID, date, AM/PM)**. A path/filename fruit, date, mode or AM/PM disagreement causes a hard error.

| Fruit task | Fruit IDs | Unique Tag / NT traces | Labelled Tag / NT traces | Targets available |
|---|---:|---:|---:|---|
| Mango MA | 10 | 394 /394 | 384 /384 | DM and Brix |
| Pear PR | 10 | 400 /400 | 390 /390 | Brix |
| Persimmon PM | 10 | 397 /400 | 387 /390 | Brix |

Missingness is recorded, not repaired:

- The 2023-05-20 09:00 workbook row is blank for all 30 fruits, although traces exist.
- MA5 has no targets and no Tag/NT traces for the final six times, beginning 2023-05-30 21:00.
- PM7, PM8 and PM9 have no Tag trace at 2023-05-24 09:00; their workbook values and NT traces exist.
- No label imputation, interpolation or outlier removal was performed. Numerical values such as the mango Brix minimum 6.2 remain unchanged.

`audit.json` records aggregate target ranges/missing keys, duplicate mappings and calibration evidence. `mat_inventory.csv` records every MAT filename, variable, role, timestamp, group, shape, finite flag, file hash and numerical-array hash. Neither report exports the waveform values or row-level target values.

## Quantity, axes and compatibility

The MATLAB files contain an amplitude vector, not a frequency column, complex spectrum, refractive-index vector or absorption coefficient. The source experiment concerns **sub-THz reflection with/without a resonant sticker**. It cannot be concatenated with acid-jujube absorption spectra as if both measured the same quantity.

Each nested `Reference.picotd` has 7,000 numeric first-column points from 0.00 to 699.90, step 0.10, but its header is dated 2021-09-22 and does not explicitly identify the time unit. `Blank.xml` includes `Units=mm`; that product setting is not proof that waveform coordinates are millimetres, seconds or picoseconds. Matching vector lengths does not establish that this old reference is the calibrated axis for each 2023 MAT. There is no upstream MATLAB/Python processing code in the archive.

**Do not construct a Hz/THz axis by assuming 0.1 ps, and do not normalize by the identical Background/Metal arrays.** Correct frequency processing and reflection reference handling require an author/instrument clarification or a verifiable acquisition specification.

Target values are the workbook's reported `Brix` and `DM` readings. The conventional degree-Brix or percentage interpretation and the measurement/calibration procedure are not explicit in this archive. Model outputs should therefore be described in **reported reading units**, with the unit/method limitation stated. They are not origin labels, direct water-content measurements or verified wet-chemistry concentrations.

## Ready local numerical interface

An explicitly exploratory interface is available at:

`data/external/public_thz_20261031_v1/acquired/princeton_metasticker_fruit/prepared/fruit_fixed_Tag_time_index_v1.npz`

The dataset stays under the ignored original-data directory. `prepared_interface.json` binds its SHA-256 to the original ZIP and records its schema and selection rule.

- Fixed single mode: **Tag**.
- Fixed scheduled time indices: **0, 10, 20, 30**; selected before any model was evaluated, not on label values or prediction scores.
- Dates: May13PM, May18PM, May23PM, May28PM, 2023.
- **120 observations**, four observations per persistent fruit; each plant task has **40 observations /10 fruit IDs**.
- `X`: 120×7,000 exact raw float64 waveform amplitudes. `sample_index`: 0–6,999, with no physical time/frequency unit assigned.
- `y_Brix`: spreadsheet Brix for all rows. `y_DM`: spreadsheet DM for mango and NaN for the other fruits.
- Metadata: `groups`, `fruit_type`, `record_id`, `timestamp`, `condition`, `source_member`.
- Metadata and targets must not enter `X`. No filtering, FFT, normalization, fitted transformation, feature selection, label filling or training occurs in preparation.

This interface supports a **separate exploratory engineering regression**, with the limited question: do raw recorded waveform features predict the associated reported readings for unseen persistent fruit IDs within this collection? It cannot validate the original calibrated resonance method or demonstrate chemical accuracy, new-batch generalization or improved acid-jujube classification.

## Evaluation constraints for a subsequent runner

1. Run mango Brix/DM, pear Brix and persimmon Brix as separate regression tasks. Do not use plant identity to make a pooled target easier.
2. Group both outer and inner splits by persistent fruit ID. With 10 IDs/task, a fixed 5-outer/3-inner grouped design is feasible. All four times of each held-out fruit remain outside fitting/selection.
3. Use a train-mean regression baseline and train-only preprocessing/PCA/regularized regression or SVR controls. MC-UVE is a supervised feature selector and must also be refitted in every current training partition; latent components must respect the small training counts.
4. Declare candidates and all seeds before outer scores are available; keep negative R², every fold and every seed. Do not choose later models from outer errors. This already inspected collection is exploratory.
5. Report MAE, RMSE, pooled out-of-fold R² and per-fruit errors in reading units; classification accuracy and Macro-F1 are inappropriate for continuous Brix/DM targets.
6. For future NT or larger timepoint sensitivity analyses, retain paired groups and predeclare the comparison. Do not count more timepoints as more independent fruits. Cross-day deployment or joint fruit/date generalization would need an additional time split and calendar-effect controls.
7. The shared acquisition-session environment across fruits remains a potential confound. A fruit holdout is not a new acquisition batch. Instrument, batch and identity evidence are still limited.

## Reproduction and checks

```bash
python scripts/audit_princeton_fruit_20261031.py \
  --archive data/external/public_thz_20261031_v1/acquired/princeton_metasticker_fruit/author_repository.zip \
  --output /tmp/research4-fruit-audit-new
python scripts/prepare_princeton_fruit_exploration_20261031.py \
  --archive data/external/public_thz_20261031_v1/acquired/princeton_metasticker_fruit/author_repository.zip \
  --output data/external/public_thz_20261031_v1/acquired/princeton_metasticker_fruit/prepared/fruit_fixed_Tag_time_index_rerun.npz \
  --manifest /tmp/research4-fruit-interface-rerun.json
python -m unittest discover -s tests -p 'test_princeton_fruit_interface.py' -v
```

Both tools refuse to overwrite outputs. Source hashing verifies the raw ZIP remains unchanged. Tests check AM/PM matching, group persistence, mismatch rejection, missing targets, paired mango target ordering, raw-byte tampering and exact agreement of every local prepared waveform/target with its source. No model fits are claimed by this report; any subsequent regression must have its own locked protocol and separate report directory.
