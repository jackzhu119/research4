# Independent audit of the 2026-10-31 neural experiment

All three technical audit phases passed. This audit performed **zero new estimator or calibration fits**. It reconstructed saved training scope and arithmetic, reloaded the 600 saved neural models, and recomputed predictions and metrics. The original 855-specimen population and five-outer/two-inner folds were retained. This remains an **EXPLORATORY** analysis of previously exposed specimens, not independent biological or new-batch confirmation.

| Phase | Completed UTC | Independent checks | Principal evidence |
| --- | --- | ---: | --- |
| Source review | 10:23:00 | 14,045 | 22,141,080 numeric cells, 15 split contexts, 3,420 original value/axis bindings, strict input schema and 10 boundary tests |
| Inner selection review | 10:32:06 | 192,490 | 480 actual models, 164,160 predictions, 60 MC selectors, 40 profile choices and 5 overall policy choices |
| Outer replay and metrics | 10:40:36 | 83,983 | 120 actual models, 15 policy aliases, 23,085 prediction rows, 27 method/seed OOF metric cases |

The inner and outer reviews observed a maximum numerical difference of **0** for their comparisons. Each method/seed OOF case contains exactly **855 unique specimens**. Repeated seeds, inner validation appearances and policy aliases do not increase the specimen count.

## Frozen selection and result

Before outer prediction, the mean of the six inner cases for each outer context selected `fusion_all_four / regularized24` as the primary policy on every outer fold. The six cases comprise three fixed seeds and two inner folds. Selection used Macro-F1, Balanced Accuracy, Accuracy and the predefined lexical tie order. Neither architecture nor profile was selected on outer results.

| Primary policy seed | Correct / total | Accuracy | Macro-F1 | Balanced Accuracy |
| --- | ---: | ---: | ---: | ---: |
| 42 | 727 / 855 | 85.0292% | 0.744911 | 0.739796 |
| 43 | 718 / 855 | 83.9766% | 0.718695 | 0.771429 |
| 44 | 702 / 855 | 82.1053% | 0.701243 | 0.707143 |

These minimal neural configurations did not exceed the preserved historical MC-UVE/SVM R2 result. No parameters were changed after observing this outcome. The comparison does not establish that all deep architectures or training budgets are inferior. The existing R2 is a historical exploratory comparator, not a newly retrained model or fresh confirmation result.

## What was checked

- Source and inference array values were traced by canonical specimen ID to the current database, original workbook hashes and value/axis bindings; alpha and refractive-index values also matched the historical candidate dataset exactly.
- Every source/inference context matched the old fixed split and had disjoint specimen IDs. Inference packs carried only fixed numeric spectral arrays and record IDs. Their strict schema rejected unknown metadata and target fields.
- Each learned scaler, MC-UVE operation, coefficient standardization, individual one-versus-rest PLS fit and neural fit used current source IDs and independently reconstructed current source values/labels. The inner log contained 8,520 completed learned operations; the appended outer log contained 2,730. The original inner log prefix remained unchanged.
- Saved coefficients independently reproduced the 75 MC-UVE feature selections. No PLS estimator was fitted during this audit. Sparse frequency selections entered the MLP branch rather than being treated as contiguous CNN inputs.
- Saved model initialization, class weights, epochs, source provenance and cache identity matched the frozen configuration. Batch-normalization counters matched source minibatches. Evaluation used stored running statistics, and saved parameter/buffer state remained unchanged during replay.
- Independent confusion counts reproduced Accuracy, Macro-F1, Balanced Accuracy, every class Precision/Recall/F1, per-fold metrics, pooled metrics and descriptive three-seed means/standard deviations.
- The prediction lock preceded the separate scoring process. The 15 primary-policy cases were exact aliases of already trained, source-selected family predictions, with no additional fits.

## Files and immutable phase records

- `source_review.py` / `source_review.json`: pre-source input and boundary review.
- `audit_common.py`: independent numerical, saved-model and fit-scope checks shared by subsequent phases.
- `selection_review.py` / `selection_review.json`: post-inner selection review before outer release.
- `selection_fit_scope_prefix.jsonl.gz`: immutable original inner fit-scope bytes.
- `final_review.py` / `final_review.json`: independent outer model replay and metric recomputation.
- `outer_fit_scope_suffix.jsonl.gz`: appended outer scope independently validated against actual source values.
- `model_metrics_by_seed.csv` and `class_metrics_by_seed.csv`: independent tabular metric exports.
- `selection_review.log` and `final_review.log`: actual audit executions.

Completed phase JSON files bind the scripts and relevant immutable inputs by SHA-256. The source and selection snapshots were written before their respective release gates. Audit scripts refuse to overwrite completed reviews. Bytecode caches are not part of the deliverable.

## Limits of the conclusion

This validates the declared code paths, serialized values, saved state and execution order. File/API separation is not an operating-system security boundary, and these checks do not prove an unrestricted claim of absolute zero leakage. Higher biological ancestors, new source batches and provenance beyond the user's independent-specimen declaration remain unverified. Different neural branches and profiles have different parameter counts and compute costs; these comparisons are not equal-budget causal ablations. Three seeds measure training sensitivity rather than three independent material datasets. No Gaussian result, theoretical peak assignment, transfer-learning success or few-shot calibration benefit is implied by this audit.
