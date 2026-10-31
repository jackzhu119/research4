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
  kernel records and the unchanged 403 Huangqi records. Record IDs do not
  establish independent physical specimens, repeats, collection batches, or
  named geographical origins. Leave unconfirmed scientific fields unfilled.
- Historical reports retain the data counts and hashes from their actual runs.
  Do not relabel earlier results as experiments on the new 855-record dataset.
- The current paper direction is origin classification, evaluated model
  improvements, and few-shot calibration on new batches of known origins.
  Gaussian is excluded from the current mainline by the user's decision.
