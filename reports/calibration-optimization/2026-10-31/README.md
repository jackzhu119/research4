# Calibration exploration evidence, 2026-10-31

This archive preserves the finite, source-only selection run on real uploaded spectra with anonymous origin-group labels. The fixed outer roles remain513training/171calibration/171prediction. The six-class sample-size experiment explicitly excludesZY02 and is a different auxiliary task.

Actual work:116candidate configurations,120development source fits,696000development candidate evaluations,2new six-class final source fits,9000final evaluation cases and1531800per-ID prediction rows. Candidate execution failures:0. All final choices were locked before query scoring; SHA `efc9f8dcb010b3ba686b4942971aa9bbea0a703f1eb23abe8e60d48931fbff75`.

The seven-class primary selection retains the frozen sources, so accuracies stay88.8889% and87.7193%. A predeclared full-SVM family diagnostic (lambda1000,blend0.05) has mean89.0994% (+0.2105pp), but did not meet the first-stage development selection threshold. Its improvement involves onlyZY07-057 in36of100draws. This is an exploratory signal, not a selected/confirmed improvement. All negative and zero results remain included.

Important evidence:

- `candidate_grid.json`, `protocol_snapshot.json`, `split_plan_snapshot.json`, `run_manifest.json`: inputs, fixed configuration, source/runtime hashes and candidate definition.
- `development/`:120CSV.gz files plus episode metadata, containing every development evaluation, source role IDs, support draws, failure status and recorded source-freeze assertions.
- `development_aggregate.json`, `selection_lock.json`, `selection_lock_sha256.json`: all means, the24/6development selection/gate, matched zero-support controls and immutable choices.
- `final/`:10compressed per-ID prediction partitions plus all scene metrics, class recalls, confusion matrices and model metadata. `all_final_metrics.json` and `summary.json` provide combined versions.
- `random_query_ids.json`, `random_query_predictions.csv`: the precommitted random examples, including incorrect predictions. `exploratory_changed_predictions.csv` is a separate post-hoc explanation, not the random sample.
- `label-audit/`:855-row label lineage and7117consistency checks. Anonymous labels are checked against supplied files, not independently verified geographic origins.
- `independent_final_audit.json` and `independent_selection_audit.json`: independent final-score and development-choice recomputations. Their `.py` files retain the actual executed audit sources and original workspace paths; use the production CLI below for portable model reproduction.
- `all-tests.log`:132tests passed. `run.log` and `run_invocations.jsonl`: actual execution progress, argv, time and resume check. `files_sha256.json` inventories every other archived file.
- `校准探索_预测与标签对照.xlsx`: human-readable summary, all precommitted random examples, per-specimen seven-class statistics, and explicitly post-hoc changed-case diagnostics. The corresponding CSVs and `table_export_validation.json` preserve inspectable values and export checks. `boundary_sample_diagnostic.json` describes the single specimen behind the small positive family comparisons.

The source classifiers remained frozen. Seven-class binaries are in `models/zaoren_threeway_20261031`; the two six-class binaries are separately preserved in `models/calibration_optimization_20261031`. No NPZ or cache is stored here. Repeated draws are not independent specimens or new batches. All855records and the current query set were explored previously, so findings are retrospective within-batch evidence.

The Chinese interpretation and reproducible production commands are in [CALIBRATION_OPTIMIZATION_REPORT.md](../../../docs/CALIBRATION_OPTIMIZATION_REPORT.md). The production runner is `scripts/run_calibration_optimization.py`; append`--resume`only with identical inputs, protocol, dependencies and source hashes. Future data or method changes require a distinct protocol/output directory.
