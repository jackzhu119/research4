# Independent auxiliary-plant experiment audit

The completed v3 source-selection and outer-replay audits passed. This audit fitted **zero** new estimators or calibration models. It independently reloaded **1,200 inner models and 180 outer models**, recomputed predictions and metrics, and reconstructed the actual numerical scope of every recorded learned operation.

These are two separate **EXPLORATORY source-file-group classification tasks**: Huangqi has 403 column samples and Tianma has 405. The user declared no repeated scanning or repressing. Shared upstream material and batch identities remain unknown; named locations and species were not independently authenticated. Results do not establish new-batch origin recognition or cross-plant transfer.

## Completed evidence

| Stage | Actual check | Result |
| --- | --- | --- |
| Full v2 source review | 26,645 checks, 19,989,491 numeric-cell comparisons, 8 original workbook hashes, 2,019 original-sheet row bindings and 30 prepared split contexts | Technical source PASS; no research fits at review time |
| v3 repair review | 722 checks; 13,674,180 prepared cells exactly equal the already audited v2 values; fixed IDs, labels, axes and partitions unchanged; 13 recorded tests and 72 independent tail-index cases | Technical source PASS; no new research fits at review time |
| Inner selection review | 902,452 checks; 720 SVC and 480 neural models; 193,920 predictions; 480 MC-UVE selectors; 60 family candidate choices and 10 neural-policy choices | PASS before outer release |
| Outer replay and scoring | 98,191 checks; 60 SVC and 120 neural models; 30 policy aliases; 16,968 prediction rows; 42 plant/method/seed OOF metric cases | PASS after separately locked prediction and scoring stages |

Each OOF method/seed case contains exactly 403 Huangqi or 405 Tianma specimens. Three seeds and policy aliases do not create additional physical specimens or additional policy-model fits. The largest numerical discrepancies were `6.94e-18` in the inner review and `3.47e-18` in the outer review, within arithmetic-rounding tolerances. Predicted classes and integer confusion counts matched.

All 39,000 inner learned operations and 3,420 outer learned operations were traced to current-source canonical IDs and independently reconstructed source X/y values. These include 31,800 inner and 2,820 outer explicit estimator/selector/scaler/neural fits, plus 7,200 inner and 600 outer MC coefficient-standardization operations. The 28,800 inner and 2,400 outer individual PLS fits are included in those counts; they are not additional classifier models. The original 170,578,440-byte inner fit-scope prefix remained unchanged after the outer append.

## Fixed-protocol outcomes

| Task | Method | Seed 42 correct / total | Seed 42 Accuracy | Seed 42 Macro-F1 | Three-seed Accuracy mean ± sample SD |
| --- | --- | ---: | ---: | ---: | ---: |
| Huangqi | Full-spectrum SVM | 347 / 403 | 86.1042% | 0.860863 | 86.1042% ± 0.0000 pp |
| Huangqi | MC-UVE + SVM | 349 / 403 | 86.6005% | 0.865740 | 87.0968% ± 0.4963 pp |
| Huangqi | Inner-selected neural policy | 251 / 403 | 62.2829% | 0.619829 | 61.2076% ± 1.2734 pp |
| Tianma | Full-spectrum SVM | 397 / 405 | 98.0247% | 0.980267 | 98.0247% ± 0.0000 pp |
| Tianma | MC-UVE + SVM | 398 / 405 | 98.2716% | 0.982769 | 97.6955% ± 0.5140 pp |
| Tianma | Inner-selected neural policy | 350 / 405 | 86.4198% | 0.864778 | 86.8313% ± 0.5140 pp |

Seed 42 was fixed before training. The three-seed summaries describe training sensitivity, not independent material datasets. Huangqi MC-UVE/SVM improved the mean Accuracy by about 0.993 percentage points over full SVM. Tianma MC-UVE/SVM improved seed 42 slightly but its three-seed mean was about 0.329 percentage points lower; seeds 43 and 44 were worse than full SVM. Both results are retained. The tested minimal neural candidates did not outperform the conventional SVMs. This does not establish that every architecture or training budget will behave similarly.

The 30 neural-policy cases exactly reuse predictions of source-selected models. Huangqi selected fusion on outer folds 0 and 1 and ResNet on folds 2–4. Tianma selected SE on fold 2 and ResNet on the others. Every selected profile was `regularized24`. These choices came from the six inner cases per candidate (two inner folds and three seeds), with Macro-F1, Balanced Accuracy, Accuracy and predefined lexical tie rules. Outer results did not select architectures or parameters.

## Preserved failures and review adjustments

Draft v1 had no real-data fits. Its pre-fit review identified missing immutable prediction/truth-integrity boundaries; the author archived it and implemented v2 before source authorization.

The first authorized v2 source trial completed **one SVC and one scaler fit**, then failed in the diagnostic batch comparison: checking seven rows in batches of three accidentally included nine rows from a larger inference pack. It returned before any inner-truth parsing, saved no pipeline or trial predictions, produced no accuracy, and never ran outer evaluation. Its exact code, scope, context, traceback and release were preserved in `../auxiliary/source_inference_failure_v2/`. This was a software-check failure, not a scientific negative-accuracy result. The v3 repair clamps diagnostic slice ends and checks batch sizes 1, 2, 3 and 7. AST comparison confirmed unchanged model fitting, prediction/vote, selection and scoring definitions; the prepared spectra and labels were exactly unchanged.

The independent review also retained its own unsuccessful script attempts: adapting the recorded provenance field names, recognizing the authorized anonymous-label overlay, and handling the empty other-plant score columns produced by a union CSV schema. These changes affected only review code. No model, input, prediction or scientific result was edited.

## Inspectable records

- `source_review.json` and `source_review_v3.json`: immutable pre-source evidence and the isolated repair review.
- `selection_review.json`: completed 1,200-model source replay and candidate-selection review before outer release.
- `final_review.json`: completed 180-model outer replay, independent metrics and scope verification.
- Corresponding `.py` and `.log` files: executable review logic and actual executions. The reviewer imports model-loading/inference code but never calls a fit entry point.
- `selection_fit_scope_prefix.jsonl.gz` and `outer_fit_scope_suffix.jsonl.gz`: independently checked original scope bytes.
- `model_metrics_by_seed.csv` and `class_metrics_by_seed.csv`: all 42 pooled cases and all per-class metrics. Confusion matrices are retained in `final_review.json`.
- `audit_deliverable_index.json`: recursive SHA-256/byte index, excluding bytecode caches and the index itself.

Successful model inputs and frozen evidence are under `outputs/upgrade-20261031-v1/auxiliary/v3/`; permanent model/trace archives are maintained separately by the experiment author. Completed audit JSON records bind the exact scripts and input evidence by SHA-256. The audit scripts refuse to overwrite completed review records.

## Scope of the technical conclusion

The audit checks fixed source/inference IDs and numeric bindings; source-only SNV/scalers, MC sampling and each PLS fit; original SVC voting and gamma; fresh neural initialization and optimizer; source-only class weights; BN source-minibatch counters; unchanged inference state; every pooled and per-class confusion-count metric; all scored correctness flags; frozen-source selection; release/prediction/scoring order; and unchanged historical core evidence.

File/API boundaries are not operating-system access control and do not prove unrestricted absolute zero leakage. No class was silently dropped to make a multimodal route succeed: Huangqi uses time/frequency fusion because its Gansu refractive-index sheet is empty; Tianma uses absorption/refractive-index fusion because two time sheets are corrupt. Different modality pairs and parameter counts limit causal attribution. Units, instrument compatibility and batch structure remain unresolved, so cross-plant transfer remains blocked. No theoretical chemistry feature, Gaussian calculation or small-sample calibration benefit is claimed by this experiment. The sour jujube seed task and its preserved historical strong baseline remain the main research task.
