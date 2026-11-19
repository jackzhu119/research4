# Kind classification: leave-one-workbook-out

New retrospective validation protocol on all corrected records: four kinds, 18 folds/1672 specimens; Zaoren versus Suanzaoren, 10 folds/864 specimens.

Each complete workbook is tested once while all other eligible workbooks train the fixed full-spectrum SNV (ddof=1), training-only StandardScaler and C=10 RBF SVC baseline. There is no hyperparameter search. The existing TRAIN/CAL/PRED roles are preserved as provenance, not reused as this protocol's partition.

All folds, mistakes, per-record OOF predictions, explicit train/test IDs, training class/workbook counts, scaler parameters and majority-class controls are retained. A fold has one true kind, so fold accuracy is true-kind recall; pooled macro-F1 and balanced accuracy use the whole task. Equal-weight workbook mean accuracy gives each workbook one vote. Workbook grouping is not proof of prospective or independently acquired batch performance.

Recompute: `python scripts/validate_kind_lowo_20261115.py`

Verify existing outputs: `python scripts/validate_kind_lowo_20261115.py --verify-only`
