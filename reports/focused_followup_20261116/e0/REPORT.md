# E0 evidence and paired archived-prediction analysis

Status: COMPLETED

This module fitted no classifier or PLS model.

Verified common complete MC20 repeats: 330; independent declared physical specimens: 300.

|Comparison|Accuracy difference, pp|Fixed archived-prediction paired resampling interval, pp|
|---|---:|---:|
|1: selection/alpha_snv vs baseline/svm_alpha_snv|11.1616|[8.4171, 14.0577]|
|2: selection/alpha_snv vs baseline/lda_alpha_raw_snv|2.3960|[0.3838, 4.4728]|
|3: spacing/alpha_inner_gap_k62_c2 vs selection/alpha_snv_c2_k62|11.8081|[9.0343, 14.7183]|
|4: spacing/alpha_inner_gap_k62_c2 vs coverage/uniform_k62|9.2657|[6.6373, 12.1172]|
|5: rank_context/true_n vs rank_context/shuffle_n|0.0646|[-1.1394, 1.2728]|
|6: spacing/alpha_inner_gap_k62_c2 vs selection/alpha_snv|-0.4061|[-1.1828, 0.3657]|

- Six explanatory comparisons were locked after archived outcomes were visible; they are not prospective preregistered confirmation experiments.
- Intervals are fixed archived-prediction paired resampling intervals (固定存档预测的配对重采样区间), conditional on these fitted models and specimens. They exclude model-development, historical-selection, and new-batch uncertainty.
- Outer repeats and repeated predictions are dependent descriptions, not additional independent specimens. No independent-sample t test or noninferiority/equivalence claim is made.
- Physical independence follows the frozen user declarations, not a sampling-log audit. Batch identifiers remain unknown where unrecorded; a workbook is not assigned a new batch ID.
- The 300 existing TRAIN specimens have been exposed in historical model development. Neither repeat count nor total completed-job count establishes external validation.

LOWO binary/four-kind results, all heldout-workbook outcomes, and all misclassified records are separate files. The five-source analysis does not combine them.

Source artifacts, identities, exact saved split IDs, final training-selected parameters, and missing evidence are recorded in evidence_registry.csv.gz and missing_artifacts.json.
