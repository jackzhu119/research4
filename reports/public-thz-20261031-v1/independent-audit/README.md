# Independent verification of public THz experiments

This directory contains a second agent's read-only reviews. The auditor performed **zero model fits**, changed no labels, raw arrays, frozen training source, candidate grids, model weights or historical experiment products. Reviews bind the actual source, preparation, predictions, learned objects and metrics by SHA-256. A technical PASS is limited to the stated frozen implementation and is not a certification of specimen independence or an absolute proof of zero leakage.

## Ginseng and American ginseng

The existing uploaded archive contains 42 paired tablet observations, 65 native frequency points and two species labels. The original plant/root/material ancestry and biological batches remain unknown. The six export files are provenance groups, not verified independent batches; paired absorption coefficient and refractive index spectra do not create additional samples. The 141 negative absorption cells remain unchanged and flagged. This task is exploratory species classification, not geographical-origin prediction or acid-jujube validation.

- `independent_dataset_review.json`: independently reconstructs all 18 original numeric CSV files and checks the prepared values, native units, source hashes, label-column pairing and lineage.
- `source_review.json`: checks all 36 outer/inner source/query packs against that reconstruction, frozen source and actual 12 synthetic boundary-test logs. It independently checks the fixed 36+48+7+33 source grid, three seeds, 20 outer roles and 11,160 source classifier budget. The standalone cache API omits a separate axis hash; the approved run still binds every actual pack/axis byte. That limitation is explicit and the frozen source was not silently changed.
- `adversarial_review.json`: replays the separately predeclared G1 diagnostic. Query-truth permutation produces the same source-selected configuration, learned state and predictions; a postfit single-query perturbation leaves the other six unchanged. Ten within-source-export-group label permutations are preserved, including failures to outperform nulls. G1 observed 3/7 accuracy is below source-majority 4/7. This diagnostic exposed G1 labels and cannot make subsequent G1 evaluation a fresh independent confirmation.
- `selection_review.json`: replays every 11,160 source model,1,620 cached transforms and 78,120 source predictions/scores; independently recomputes metrics, stage/family winners, inherited configurations and actual current-source fingerprints, including each internal MC PLS subset. The actual source cost includes 64,800 internal binary one-hot PLS1 fits and 1,080 MLP warnings.
- `evaluation_review.json` and `independently_recomputed_ginseng_metrics.json` replay all 306 locked outer models and 54 exact-configuration aliases, all 2,520 outer predictions and 60 pooled role/seed metrics. The latter includes independent per-class precision, recall and F 1.

The diagnostic frozen estimate said 720 internal PLS fits but the actual trace contains 1,440: binary one-hot MC selection fits one PLS1 per target column. The incorrect estimate and actual count are retained explicitly;132 diagnostic SVC fits are correctly accounted for. Isolation/replay passed; the internal PLS budget estimate did not. This was not used to alter the main grid or select main candidates.

`postscore_checks.json` independently verifies all outer learned scaler statistics, raw-value/target cache lineage, MC ranking arithmetic and actual internal PLS subset fingerprints in both tasks. It also adds an explicitly **after-score descriptive** source-majority comparison: choose the most frequent label among each fixed 35-observation source, then predict its seven query observations. The pooled result is 22/42 correct (52.38%), Macro-F1 0.34375 and Balanced Accuracy 0.5. This was not a predeclared main role, did not alter the grid and is not presented as a new selected model.

## Separate fruit engineering regression

`fruit_source_review.json` independently reads all 120 selected original MAT waveforms and their real workbook target cells, then checks all 80 source/query contexts. There are 30 persistent fruits across three fruit types, each observed at four dates; each task has ten fruits and 40 dependent observations. All dates from one fruit stay together in both inner and outer splits. Target and input scalers are learned only from the current source. The 128 features are fixed waveform index blocks; an unverified time calibration is not converted into a THz frequency axis.

`fruit_selection_review.json` replays all 7,560 source models,80,640 source predictions, cached transforms, pooled 32-observation inner metrics per seed and all 20 family/policy locks. `fruit_evaluation_review.json` replays all 300 outer models,60 exact aliases,2,880 predictions and 72 pooled cases. All three seeds, negative R² values and models worse than the mean baseline remain included. Its exact source trace prefix remains unchanged after outer fits were appended.

The source repository does not establish a redistribution license, physical time unit or target-assay method. Raw/derived arrays, individual predictions, traces and models stay in ignored local directories. Only source code and aggregate reports are intended for Git. The same four dates overlap source/query fruit groups, so this is not future-time validation. These reflection-waveform regression results do not demonstrate medicinal THz-TDS transfer.

## Reproduction and preserved corrections

Run the Python audit scripts from this directory, in order, only after their corresponding immutable stage has completed. Source-stage scripts require outer prediction/release artifacts to be absent; they deliberately refuse a later-stage run as a new source review. Completed JSON reviews use exclusive file creation and must not be overwritten. A new audit requires a new report directory. The original actual execution logs remain here.

1. `audit_common.py`: independent raw ginseng dataset review.
2. `source_review.py`: frozen main preflight; `adversarial_review.py`: fixed actual G1 replay.
3. `fruit_source_review.py`: frozen fruit preflight.
4. `selection_review.py` and `fruit_selection_review.py`: independent source-stage replay before root outer release.
5. `evaluation_review.py ginseng` and `evaluation_review.py fruit`: locked outer-model replay and metric calculation.
6. `postscore_checks.py`: after-score descriptive majority and additional outer-transform/actual internal PLS checks.

Two auditor corrections are preserved: `review_attempt 01_header_whitespace/` contains an overly strict raw-header assertion, corrected to trim trailing whitespace; `source_review_attempt 01_unittest_format/` contains an assertion expecting pytest wording, corrected to the actual unittest log format. These were audit-reader corrections and caused no data, model, grid or production-source changes. They are not model-training failures.
