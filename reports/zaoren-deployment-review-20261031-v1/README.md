# Anonymous deployment: independent technical review

2026-10-31. This review used zero new estimator/calibration fits outside explicitly
synthetic unit-test fixtures. All855 acid-jujube records remain historically
exposed development data. No independent external or new-batch accuracy exists.
The native seven labels are ZY01–ZY07; real geographic names are not required.

| Phase | Technical checks | Actual scope |
| --- | ---: | --- |
| Before source training | 909 | Original Git inputs, all855 IDs/labels/hashes, five fixed folds, five source-defined candidates, 22 boundary tests, real2578→612 intake |
| Source selection replay | 221672 | 75 saved models, 12825 development predictions, five15-case means, 21825 actual learned operations |
| Full refit and serving | 12493 | Three final full855 models, 1209 appended learned operations, each855-row pointwise/order/batch13 state checks, real diagnostic/re-export/tamper checks |

Source replay comparisons had maximum floating arithmetic difference
3.469446951953614e-18; the final checks had maximum difference0. The original
285012450-byte source fit-scope prefix remained unchanged after the final fits.
`selection_fit_scope_prefix.jsonl.gz` preserves those exact source-only bytes.

## Selection and deployment meaning

Five complete configurations came from each historical R2 source-only choice,
without reading its outer predictions or frequency mask. The old five-fold
members were reused as development CV, all seeds42/43/44 were included, and the
selection rule was mean Macro-F1, Balanced Accuracy, Accuracy, then candidate ID.

The independently recovered winner was `r2_source_fold03`: SNV, MC50/LV1,
retention20% (123features), C1000, gamma0.1×scale and source-balanced weights.
Each final seed model fitted fresh scaler, MC selector and SVC on all855.
Historical fold masks were not pooled or reused.

The75source models used21825 actual learned operations, including18900 internal
PLS fits and2700 MC coefficient standardizations. The three final models added
1050 PLS fits,150 MC standardizations and9 primary scaler/selector/SVC fits:
1209 additional operations,23034total. Every logged fit ID and numerical X/y
fingerprint was reconstructed from its allowed source and deterministic MC
subsamples, without refitting PLS in this audit.

The old96.0234% remains the separate historical855OOF result. Neither source
selection means nor full-development predictions are independent performance.
This audit did not calculate a final model's resubstitution accuracy.

## Actual serving boundaries

The final three models were loaded from their exact saved hashes. Each source
spectrum was inferred alone, in reverse order, and in nondividing batch13;
outputs and serialized model state remained unchanged. Native SVC voting was
preserved; decision scores are not calibrated probabilities.

Five real historical spectra were processed only as explicit
`TRAINING_DIAGNOSTIC_NOT_PERFORMANCE`, with accuracyNone and no labels passed to
the prediction API. A historical spectrum renamed and altered by1e-14 roundtrip
noise was rejected in future mode before any forecast output. An altered model
in a temporary bundle copy was rejected before deserialization; actual weights
were unchanged. These checks add no physical specimens or successful external
predictions.

Input schema tests cover strict label-free NPZ/CSV/XLSX, source-ID independence,
reference exclusion, thickness metadata, axis/order/nonfinite checks, explicit
2578→612 cropping, flat spectra, hidden target fields, incorrect pairing and
exclusive blank templates. New input labels are never accepted for prediction.

## Files and reproduction

```bash
.runtime/bin/python reports/zaoren-deployment-review-20261031-v1/source_review.py --run <fresh-prepared-run>
# Source review must precede root source release and research fitting.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .runtime/bin/python reports/zaoren-deployment-review-20261031-v1/selection_review.py --run <completed-source-run>
# A separate root full-refit gate precedes the three full855 fits.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .runtime/bin/python reports/zaoren-deployment-review-20261031-v1/final_review.py --run <completed-full-refit-run>
```

Scripts refuse to overwrite completed phase reviews; reproduce in an explicitly
new review-output version. Original phase JSONs bind source versions, inputs,
model locks and actual test logs by SHA256. `review_common.py` implements separate
confusion-count arithmetic, coefficient ranking and full fit-input checks.

## Limits

This is a bounded technical review, not absolute proof of zero leakage or an
OS-level truth sandbox. Biological ancestors, batch identities, instrument and
absorption-unit compatibility remain incompletely documented. User IDs alone
cannot establish a new physical sample. Numeric near-match QC only detects
re-exports at the stated tolerance; it does not prove material identity.
Unknown-class detection and independent future accuracy remain unvalidated.
