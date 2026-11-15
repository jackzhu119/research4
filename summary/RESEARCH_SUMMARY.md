# Old ML background study — completed-results publication

Captured 2026-11-15T07:24:21.121213+00:00. This study is independent of the chemistry computation.

The queue contains **83,017 completed jobs of 89,267** (93.00%). Another 6,246 jobs are queued and 4 were marked running on the previous boot; all 6,250 unfinished jobs are excluded from the completed-results archive. The study is still incomplete. At capture, canonical training was stopped after the cloud reboot; the saved report supervisor state is from the previous boot. Publication precedes the authorized resume.

The canonical report was refreshed at 2026-11-15T07:23:36.684795+00:00 without new model fitting. It validated 83,019 non-testing complete result files with 0 invalid and 0 testing exclusions. This is two more than SQLite-completed jobs because two previous-boot running attempts had written complete result files before queue commitment. Those two are excluded from the completed archive; their incomplete five-fold repeat is excluded from method-level source estimates. Reports are preserved as generated, with this distinction explicit.

## Current descriptive findings

There are 85 reportable source method/MC groups, 15,349 complete source repeats, 990 complete label-permutation repeats, and 1,322 fixed-role material fits. The source population remains the same 300 physical specimens.

Selected predeclared MC20 comparisons (or methods without an MC parameter):

| Family / variant | Complete repeats | Mean source accuracy | Conditional 95% interval |
|---|---:|---:|---:|
| baseline / svm_alpha_snv | 331 | 82.04% | 78.42%–85.41% |
| baseline / lda_alpha_raw_snv | 331 | 90.80% | 88.19%–93.21% |
| baseline / lda_joint_both_snv | 330 | 90.89% | 88.53%–93.07% |
| selection / alpha_snv | 330 | 93.20% | 90.83%–95.34% |
| selection / joint_both_snv | 330 | 91.93% | 89.49%–94.14% |
| selection / joint_nraw_asnv | 330 | 92.77% | 90.39%–94.95% |
| retain_drop / joint_both_snv | 330 | 93.04% | 90.67%–95.19% |
| spacing / alpha_gap4_k62_c2 | 330 | 92.81% | 90.24%–95.09% |
| coverage / uniform_k62 | 330 | 83.52% | 79.98%–86.75% |

The alpha-SNV selection and retain/drop workflows currently give about 93% mean internal source accuracy. Several full-spectrum LDA workflows give about 91%. These are descriptive comparisons over differing completed repeat sets; the reports do not establish a statistically significant winner or independent information from refractive index. MC100 and MC500 results remain separate sensitivity levels with fewer repeats.

Label-permutation controls remain near the five-class chance level: lda_alpha_raw_snv 20.09% (330 repeats), mc_alpha_snv 20.00% (330 repeats), svm_alpha_snv 20.05% (330 repeats). They are a separate null target.

Fixed-PRED material results are reported separately: binary_svm_alpha_snv 100.00% on 174 unique specimens, binary_svm_alpha_snv_fixed_C10 100.00% on 174 unique specimens, four_lda_alpha_raw_snv 95.28% on 339 unique specimens, four_plsda_alpha_raw_snv 94.10% on 339 unique specimens, four_svm_alpha_snv 96.46% on 339 unique specimens, four_svm_alpha_snv_fixed_C10 97.64% on 339 unique specimens. Repeated fits reuse these same fixed PRED specimens.

## Evidence and interpretation

All seven frozen source/input files match the pre-recovery SHA-256 checks; the current protocol SHA-256 is `f8d133f235a5f8ed66fce743404127a38baf175bd73fb1a56ed8daa3e074d593`. The cumulative authorization remains **1,360,000 CPU seconds**; publication adds no authorization. Older text mentioning an initial 90,000-second budget is historical and is superseded by the recorded authorizations and current protocol.

The archive preserves completed job results, predictions, selections, fitted artifacts or verified lossless reconstruction recipes, diagnostic archives, shared exact input NPZ and worker, reports, source/protocol, consistent SQLite snapshot, and file/shard manifests. `MODEL_ARTIFACT_CODEC_README.txt` explains restoration without training. Incomplete jobs remain labelled in queue provenance and are not counted as completed evidence.

Source intervals are conditional on trained outer-test predictions: per-record mean correctness is bootstrapped within class by physical ID (5,000 draws, seed 20261110), retaining repeated predictions. They do not include model refitting, multiple-comparison correction, independent specimens, or external/cross-batch generalization. Full five-fold source repeats alone enter the method summaries. Material and permutation targets must not be pooled with source OOF accuracy. Chemical mechanism inference remains blocked by absent same-ID quantitative chemistry and validated reference bands.

Primary report: `reports/summary_zh.md`; numerical evidence: `reports/source_accuracy_summary.csv`, `reports/permutation_null_summary.csv`, `reports/material_PRED_summary.csv`, `reports/repeat_completion.csv`; publication counts/hashes: `research_summary.json` and `frozen_hashes_verified.json`.
