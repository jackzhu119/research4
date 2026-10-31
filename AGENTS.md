# Research data conventions

The user's current data decision, effective 2026-10-31, is recorded in
`data/CURRENT_DATASET.md` and `data/active_workbooks.json`.

- Use only `data/processed/zaoren_20261031/样本1_编号版.xlsx` through
  `样本7_编号版.xlsx` for the current acid jujube kernel dataset: 855 spectrum
  records, with groups ZY01 through ZY07. These replace all earlier unnumbered
  jujube workbook inputs for future work.
- Obtain default inputs from the active registry. Do not discover current
  research inputs by recursively scanning `data/raw`, `data/archive`, or the
  processed directory; the latter also contains a non-spectral summary workbook.
- Earlier unnumbered originals and the old 808-row metadata template are
  superseded and retained under `data/archive` for provenance only. Do not mix
  them with the numbered workbooks or count both versions as observations.
- The current metadata template has 1258 spectrum records: 855 acid jujube
  kernel records and the unchanged 403 Huangqi records. The user subsequently
  confirmed that each numbered jujube record is an independent sample; the
  evidence is `data/annotations/zaoren_user_declaration_20261031.json`.
  Jujube physical_sample_id is explicitly mapped to record_id with basis
  user_declared_independent_20261031, and origin_group_verified is true.
  This is a user declaration, not an inference from numbering or a laboratory
  audit. Named origins, batches and species confirmation remain unfilled.
  Huangqi annotations are unchanged.
- The user also confirmed that each origin currently has only one batch.
  This is recorded in the same declaration JSON; no actual batch identifiers
  were supplied or invented. Origin and the particular source lot cannot be
  separated using the current collection. Sample-level CV is not a real
  cross-batch experiment, regardless of its accuracy.
- The approved current protocol is `docs/APPROVED_ZAOREN_EXPERIMENT_PLAN.md`
  and its JSON under `data/research/`. Use outer 5 / inner 2 / CARS-internal 2
  folds, split seed 20261031 and model seed 42; the ZY02 class has only five
  samples. Current results are internal CV on anonymous origin groups, not
  verified named-origin or cross-batch performance. Real batch calibration
  remains gated on new batch annotations and adequate support/query pools.
- Historical reports retain the data counts and hashes from their actual runs.
  Do not relabel earlier results as experiments on the new 855-record dataset.
- The current paper direction is origin classification, evaluated model
  improvements, and few-shot calibration on new batches of known origins.
  Gaussian is excluded from the current mainline by the user's decision.
- The user subsequently requested a separate retrospective within-batch
  train/calibration/prediction split. Follow
  `docs/ZAOREN_THREEWAY_PROTOCOL.md` and its JSON for this workflow: fixed
  seed 20261101, 513 training / 171 calibration-pool / 171 prediction samples,
  source-only 3-fold tuning, and frozen-source 1-shot-per-class score
  adjustment with fixed regularization 1. Seven-class 3/5/10-shot is
  unavailable because ZY02 has one calibration specimen. These derived
  workbooks must not enter the active registry as additional observations.
  All 855 records were used in prior CV, so this is retrospective internal
  validation, not a pristine confirmatory set or a real cross-batch result.
