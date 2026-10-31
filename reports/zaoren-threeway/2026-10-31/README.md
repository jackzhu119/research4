# Retrospective within-batch frozen-source comparison

Actual run on the current855 user-declared independent acid-jujube records, partitioned513training/171calibration/171fixedprediction. This is not a cross-batch experiment or a first-use confirmatory test: all855records were previously explored in internalCV.

Two source models (full-band candidate alpha SVM and MCUVE20+SVM) were tuned on training-only3foldCV, fitted once, saved, then frozen. Both use the predeclared0.1–1.5candidate range and classify by aligned OVR-score argmax. Source fingerprints and local model-artifactSHA256 remain unchanged after calibration. Local `.joblib` models and `.cache` are excluded from this archive.

The dedicated calibration pool supplies one independent sample per known class,7labels per draw,20predeclared draws(seed42), with the same171prediction samples throughout. A fixed ridge score correction(lambda1) is fitted, without changing source feature selection/scaling/SVM. ZY02 has one calibration specimen and one prediction specimen. Uniform3/5/10shot are not evaluable; allclasses remain included. Draw0is the prespecified example, not a selected optimum.

Actual negative results are retained: fullSVM accuracy88.8889%, calibrated20draw mean74.9415%; MC20SVM87.7193%, calibrated mean72.8947%. These particular settings do not improve the current same-batch task. Support-drawSD is descriptive, not an independent-batch confidence interval. Calibration scores are not probabilities.

`prediction_bundle.json` contains predictions and fitting provenance generated before query-truth scoring. `sample_predictions.csv` has7182rows from42evaluations. `all_draw_metrics.csv` and `all_class_recalls.csv` expose every draw. `independent_arithmetic_audit.json` confirms manual confusion-matrix metrics and paired error-count arithmetic exactly agree(maxerror0), and originalNPZtruth matches allscored labels. Source-specific selected feature indices are relative to the612point candidate range, with a mapping to original exported indices.

`inputs.json` records actualargv,input/protocol/planSHA,Python/library versions,and usedsourcefile hashes. The protocol and split-plan snapshots preserve actualrun settings. `files_sha256.json` inventories every other archive file. No numericalNPZ/joblib/cache is tracked here.

The independently recomputed audit is in `independent_audit.json`; all114test results are in `all-tests.log`. The complete Chinese report is [ZAOREN_THREEWAY_REPORT.md](../../../docs/ZAOREN_THREEWAY_REPORT.md). Known-hash source model binaries are preserved separately in [models/zaoren_threeway_20261031](../../../models/zaoren_threeway_20261031/README.md); they are byte-identical copies of the actual frozen run artifacts, not additional fits. The figure includes every support draw and labels the612-point candidate input explicitly.
