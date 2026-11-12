# Corrected-kind, training-only finite replication

This review governs the new `species-relaunch-20261108-v2` queue. It
does not certify results that have not yet been computed. The old host's live
runtime is unavailable; this is a separately authorized replication, not a
claim to resume an unknown process or recover its unspent budget.

## Evidence and protected data

The current user-confirmed registry contains 18 workbooks and 1,672 records:
500 酸枣仁, 364 枣仁, 403 黄芪 and 405 天麻. These are Chinese kind labels;
their confirmation is not independent Latin-taxonomic authentication.
错误混种分类成绩已按2026-11-12用户决定撤回；论文使用main的paper/current白名单。
result therefore describes anonymous-source recognition in a collection now
known to mix two kinds. It cannot serve as pure-酸枣仁 origin performance.

The existing split remains unchanged: 999 TRAIN, 334 CAL and 339 PRED. The new
workers receive only a 999-row TRAIN pack (酸枣仁300, 枣仁216, 黄芪241,
天麻242). Reading all raw arrays to verify hashes, pair columns and export
this pack is data packaging, not fitting a transformation on holdouts. The
derived audit records that distinction and every array fingerprint. Neither
CAL nor PRED values or truths are exported as worker inputs or used for model
selection. The six already completed fixed-holdout experiments are not
repeated by this queue.

Raw bytes, master record identities, master roles, labels, historical reports,
scientific worker implementations and main remain unchanged. The new source
audit separately binds registry, 18 workbook hashes, original extraction
audit, current cache, extractor and training-only arrays. A cache rebuilt to
the historical SHA proves byte equality; a different SHA must not be silently
accepted as equivalent to an unavailable historical array.

## Fixed experiments and priority

All 1,305 declared scientific jobs are retained: 225 classical jobs and 1,080
neural jobs. Ordering is deterministic and independent of scores. First run
the five pure-酸枣仁 alpha baseline folds, then all 15 pure-酸枣仁 neural
`pre120/p12_e96_full` fold/seed jobs. The remaining classical jobs put
酸枣仁 first, then 黄芪, 天麻 and kind-classification controls. Remaining
neural jobs likewise put the 酸枣仁 target first. This early complete neural
group checks runtime usability without replacing the MC-UVE/SVM main study.

The finite allocation is 88,000 CPU core-seconds for the supervisor and all
its experiment/summary/backup children, 1,900 for audited preparation and 100
for the outer watchdog. These sum to the newly authorized 90,000. Unused
reserves are not automatically transferred; previous grants and failures are
neither refunded nor carried. The queue may end before all jobs finish, and
it must retain unstarted, partial and failed groups without inventing values.

### Classical main study

Pure-酸枣仁 origin identification uses only its 300 master-TRAIN records.
The fixed five-fold outer development split is repeated using seeds
20261031, 20261108 and 42. Model seed remains 42. Inside each outer source
partition, two-fold selection determines preprocessing, MC-UVE and SVM
choices before that fold's validation prediction is produced. Historical
R0 and R2, full-spectrum SVM and PLS-DA remain explicit controls. R1 is an
intermediate MC-UVE-selection stage, not an independently evaluated model in
the declared four-method comparison.

Compare alpha, refractive index n, and n+alpha. In fusion, the two physical
quantities are processed separately before concatenation; one joint SNV of
mixed units is not justified. Record selected feature count and computational
cost because the fused representation has twice as many input dimensions.
Do not equate a higher-dimensional gain with proof of a molecular mechanism.

### Neural and cross-kind controls

Each target has its own source-label head. No 黄芪 or 天麻 source label is
treated as an 酸枣仁 source label. The preset arms are A (target-only), Aextra
(extra target training) and B (other-herb pretraining), alongside a matched
SVM source-data control. The fold, seed, source-count, support-count and
validation identities must match for a paired arm comparison. The complete
five folds and all three model seeds remain declared; no favourable seed is
selected retrospectively. Aextra is necessary to distinguish auxiliary-data
transfer from an increase in optimisation steps.

Do not initialise an outer-fold target model from one fitted on the whole
300-record acid TRAIN pool: that model has seen the outer validation records.
Only exact matching source-ID/label/code/axis cache identities may be reused.
Auxiliary-only pretrained encoders must not have been target-finetuned on the
current target validation records.

### Conservative support adaptation

Support comes from the outer source pool, never its validation query. For
酸枣仁, each fold has 190 source, 50 reserved support and 60 validation
records. K=1/3/5/10 per class uses nested support draws and 20 repeats. Report
zero-support/base, traditional update and the prespecified conservative
mechanism together. Count incorrect→correct and correct→incorrect changes;
report macro-F1 as well as accuracy, paired uncertainty and abstention/no-op
frequency. Support draws and seeds are repeated evaluations, not additional
independent physical samples or batches.

## Interpretation and stopping rules

Only complete five-fold pooled OOF groups, with exact sample coverage and all
declared repeats/seeds, may support a development recommendation. Partial
snapshots are useful operational evidence but not a final best-model claim.
Aggregate confusion counts before calculating each OOF accuracy, macro-F1
and balanced accuracy; do not substitute an unweighted mean of fold metrics
when fold sizes differ. Compare paired predictions under the same protocol.
Preserve negative transfer, calibration harm, failed candidates and unfinished
groups. Neither the old 96.02% nor the earlier four-kind 97.64% may replace a
new pure-酸枣仁 origin metric.

These are internal development experiments on previously used material.
Source group may remain confounded with one preparation/source batch; broad
“山西” and specific “山西灵石” annotations do not establish non-overlapping
geographical populations. The work does not establish real cross-batch,
cross-instrument or prospective generalisation. Laboratory standards,
HPLC/LC-MS and independent new batches remain NOT_RUN. This queue does not
perform DFT or infer IR intensities, molecular peak assignments or chemistry
from a classifier score.

Immutable summaries and filtered Git backups follow experiments 5, 10 and
each subsequent 25, with a final summary and backup. They run inside the same
inner CPU ceiling. A failed backup cannot turn a partial scientific job into
DONE, reset budget or modify a source implementation. No hourly notification
schedule is introduced.
