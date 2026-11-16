# Evidence audit of the two reviews (read-only repository audit)

The reviews identify useful clarity issues, but their proposed additions need different decisions. No training or PLS fitting was performed. Repository evidence was read from the explicitly corrected current view and immutable source objects. Source policy: `research4/AGENTS.md`, `paper/current/README.md`, `paper/current/manifest.json`; manuscript scope: `paper/revision_20261116/EVIDENCE_MANIFEST.json` and `analysis/EVIDENCE_AUDIT.json`.

## Decisions on doubao.txt

|Suggestion|Decision|Reason|
|---|---|---|
|Clarify 10.78 percentage points applies to matched full-spectrum SVM|Accept with precise wording|The LDA difference interval includes zero; that does not establish equivalence or prove equal performance.|
|Add background 330-repeat 93.20% estimate|Reject as current corrected evidence|Do not confuse background 330 partition repeats with the allowlisted V4 330 task bundles. The former uses the expanded worker whose gamma-grid defect is established.|
|Add random/uniform K scores as proof selector beats random|Reject as corrected evidence|coverage SVM configurations use the defective inner gamma cache; also comparing their K62 with adaptive mean K220.6 would not isolate selector value at equal budget.|
|Add 990 permutation results to rule out leakage|Reject|SVM/MC null branches have the gamma issue. Even a valid chance-level permutation null would not by itself certify absence of all leakage or external validity. LDA null is a separate estimator and cannot validate the other pipelines.|
|Add n shuffle/noise context accuracies to prove n has no information|Reject|rank_context SVM search has the defect; the proposed conclusion also exceeds the assay, since changed ranking can coexist with informative n under other predictors.|
|Add background PLS-DA 78.52%|Do not import under present allowlist; distinguish defect scope|PLS-DA does not use SVM gamma: source inspection does NOT justify calling its score gamma-corrupted. Its provenance/protocol/eligibility still needs separate explicit admission and verification before comparison. Do not claim all background numerical results are invalid.|
|Describe ZY07→ZY04 errors|Accept only from current 900 primary predictions|Derived below at three actual repeats, not330.|

The immutable background source is `93a607c8ce206a5bbbff22538233bdd0929bd04d:source/study_worker.py`. Its `nested()` cache key omits `gamma_multiplier`, while `transformed()` stores `numeric_gamma=gamma0*p['gamma_multiplier']`; the cached transform metadata therefore reuses the first gamma within a transform setting. Config loops enumerate C and gamma together. Affected SVM families include selection, coverage, rank_context and SVM/MC permutation variants. Non-SVM configurations contain no SVM gamma grid. The manuscript manifest admits this background commit for defect scope only; its excluded claims include expanded-worker performance and cost as corrected gamma-grid comparisons. V4 and the focused E1 bridge are separately verified; this defect is not evidence that they share it.

Safe EN: “The 10.78-percentage-point gain is established against the matched full-spectrum SVM; the paired MC20-minus-shrinkage-LDA interval includes zero.”

Safe ZH: “10.78个百分点的收益针对同划分全谱SVM成立；MC20减收缩LDA的配对区间包含零。”

## Fixed PRED / T27 suggestion in codex_revision_instructions.md

Accept the numerical supplement with qualifications; reject its final causal/task-general interpretation. Source: current `tables/fixed_classification.csv`, pinned to `0156e2189d13456d1faf05ebfd53d86eeeb4f5e7:data/research/species_revision_20261108/classification/metrics_overview.csv`. Use `method=source_only`, not score-argmax, calibration or anchor rows.

|Task/model|Correct/denominator|Accuracy|Macro-F1|Coordinates|
|---|---|---|---|---|
|acid_origins full_svm|85/100|85.00%|0.8446885|612|
|acid_origins fixed_MC20|87/100|87.00%|0.8694929|62|
|acid_origins R0_reselected|94/100|94.00%|0.9401195|62|
|four_kinds full_svm|331/339|97.64%|0.9750880|612|
|four_kinds fixed_MC20|313/339|92.33%|0.9187430|62|

Pinned per-task `selection_lock.json` and `*_fit_receipt.json` establish: fixed_MC20 is SNV, 20 draws, 2 components, ratio0.1, balanced SVM C10/gamma-scale; acid R0 is raw, 20 draws, 2 components, ratio0.1, balanced SVM C100/gamma-scale (`hist006`). All three selected models above have62 coordinates. R0 was selected using two-fold TRAIN-only historical-grid CV, not by CAL/PRED; selection lock says both calibration_used_to_select and prediction_used_to_select=false. Source-only rows train on TRAIN only. CAL is used in other methods, which are not these scores. Full model uses SNV/all612/C10/gamma-scale. Original specimens were historically exposed: frozen PRED is retrospective internal evidence, not a new blind external batch.

Thus R0 vs fixed_MC20 changes preprocessing and C, while keeping K62; its 94% cannot be attributed to a larger budget or the same fixed selector alone. One fixed-K comparison across different tasks does not establish that within-material tasks benefit from aggressive pruning or that broad separation requires spectral coverage. The primary nested MC20 additionally tunes K over31/62/123/184/245 and averages220.6 selected coordinates: it is not the same K62 pipeline.

Safe EN: “In the separate fixed-role source-only comparison (T27), the Suanzaoren five-source task yielded85/100 correct for full_svm,87/100 for fixed_MC20 (62 coordinates; macro-F1=0.8695), and94/100 for the TRAIN-CV-selected R0 configuration, which also used62 coordinates but changed preprocessing and SVM C. For four-material classification, full_svm yielded331/339 versus313/339 for fixed_MC20. These descriptive results are specific to the recorded pipelines and fixed split and do not isolate a causal feature-budget effect.”

Safe ZH: “在独立的固定角色source-only比较（T27）中，酸枣仁五来源任务的full_svm、fixed_MC20（62坐标；macro-F1=0.8695）及TRAIN交叉验证选择的R0配置分别正确85/100、87/100及94/100；R0同样使用62坐标，但改变了预处理和SVM的C。四药材任务中，full_svm为331/339，fixed_MC20为313/339。上述结果是所记录流程与固定划分下的描述，不能单独识别特征预算的因果效应。”

For the proposed LDA discussion, accept traceable coordinate sparsity and actual mean220.6 as a representational distinction. Modify “removing kernel degradation” to an observed accuracy comparison; the experiments do not identify redundant coordinates disturbing kernel distances as the causal mechanism. Do not call MC20 computationally “efficient” without a matched total cost comparison;80% fewer coefficient draws is the supported arithmetic only.

## Derived primary confusion (no fitting)

Input: `paper/revision_20261116/analysis/primary_predictions.csv.gz`; filter `family=alpha_snv20,is_primary_k=1`. Exactly900 rows,300 unique physical IDs, seeds42/20261031/20261108,15 outer folds. Correct841/900; errors59. Each source contributes180 prediction instances across60 specimens and three partitions. ZY07→ZY04 accounts for30/59 errors (50.85%); reverse direction ZY04→ZY07 accounts for6. Twelve distinct ZY07 IDs are predicted ZY04 at least once; eight do so in all three repeats. These repeated predictions are correlated and are not900 independent specimens.

Safe EN: “Across the three recorded partition repeats, ZY07-to-ZY04 confusions accounted for30 of59 errors in the primary MC20 predictions; eight specimens showed this confusion in all three repeats. This is a descriptive error pattern among the recorded source labels.”

Safe ZH: “在已记录的三次划分重复中，主MC20预测的59次错误里有30次为ZY07误判为ZY04，8份样品在三次重复中均出现该误判。这是记录来源标签之间的描述性错误模式。”

Derived CSVs in this directory preserve pooled confusion, seedwise counts, and the twelve recurrent IDs. Do not interpret this as chemical identity, duplicate procurement, geography-general mechanism, or330-repeat evidence.
