# Research data conventions

The user's current data decision, effective 2026-10-31, is recorded in
`data/CURRENT_DATASET.md` and `data/active_workbooks.json`.

- Use only `data/processed/zaoren_20261031/样本1_编号版.xlsx` through
  `样本7_编号版.xlsx` for the current acid jujube kernel dataset: 855 spectrum
  records, with groups ZY01 through ZY07. These replace all earlier unnumbered
  jujube workbook inputs for future work.
- Obtain default inputs from the active registry. Do not discover current
  research inputs by recursively scanning `data/raw`, `data/archive`, or the
  processed directory; the latter also contains a non-spectral summary workbook.
- Earlier unnumbered originals and the old 808-row metadata template are
  superseded and retained under `data/archive` for provenance only. Do not mix
  them with the numbered workbooks or count both versions as observations.
- The current metadata template has 1258 spectrum records: 855 acid jujube
  kernel records and the unchanged 403 Huangqi records. The user subsequently
  confirmed that each numbered jujube record is an independent sample; the
  evidence is `data/annotations/zaoren_user_declaration_20261031.json`.
  Jujube physical_sample_id is explicitly mapped to record_id with basis
  user_declared_independent_20261031, and origin_group_verified is true.
  This is a user declaration, not an inference from numbering or a laboratory
  audit. Named origins, batches and species confirmation remain unfilled.
  Huangqi annotations are unchanged.
- The user also confirmed that each origin currently has only one batch.
  This is recorded in the same declaration JSON; no actual batch identifiers
  were supplied or invented. Origin and the particular source lot cannot be
  separated using the current collection. Sample-level CV is not a real
  cross-batch experiment, regardless of its accuracy.
- The approved current protocol is `docs/APPROVED_ZAOREN_EXPERIMENT_PLAN.md`
  and its JSON under `data/research/`. Use outer 5 / inner 2 / CARS-internal 2
  folds, split seed 20261031 and model seed 42; the ZY02 class has only five
  samples. Current results are internal CV on anonymous origin groups, not
  verified named-origin or cross-batch performance. Real batch calibration
  remains gated on new batch annotations and adequate support/query pools.
- Historical reports retain the data counts and hashes from their actual runs.
  Do not relabel earlier results as experiments on the new 855-record dataset.
- The current paper direction is origin classification, evaluated model
  improvements, and few-shot calibration on new batches of known origins.
  Gaussian is excluded from the current mainline by the user's decision.
- The user subsequently requested a separate retrospective within-batch
  train/calibration/prediction split. Follow
  `docs/ZAOREN_THREEWAY_PROTOCOL.md` and its JSON for this workflow: fixed
  seed 20261101, 513 training / 171 calibration-pool / 171 prediction samples,
  source-only 3-fold tuning, and frozen-source 1-shot-per-class score
  adjustment with fixed regularization 1. Seven-class 3/5/10-shot is
  unavailable because ZY02 has one calibration specimen. These derived
  workbooks must not enter the active registry as additional observations.
  All 855 records were used in prior CV, so this is retrospective internal
  validation, not a pristine confirmatory set or a real cross-batch result.
- The user further authorized repeated calibration algorithm exploration.
  Use `docs/CALIBRATION_OPTIMIZATION_PROTOCOL.md` and
  `data/research/calibration_optimization_protocol_20261031.json`: 116 locked
  candidates, 30 source-only development tasks (24 selection / 6 gate), then
  locked-method evaluation on the original prediction specimens. Require
  gains over both the source model and matched zero-support control; never
  choose a method or sample from final-query scores. Seven-class analysis
  remains 1-shot. A separately labelled six-class auxiliary analysis excludes
  ZY02 and tests 1/3/5/10-shot; it must not replace seven-class results. The
  user explicitly permits anonymous labels and will supply place names later.
- Later optimization separates source representation gains from calibration.
  `data/research/representation_calibration_protocol_20261031.json` fixes
  54 source pipelines and 60 latent adapters, with source-only 24/6 task
  selection/gating before the unchanged 171-record query is scored. Full
  exported-range results are sensitivity diagnostics, not calibration gains
  or proof of a physically valid range.
- The explicitly requested paper-inspired neural trial follows
  `data/research/paper_neural_classification_protocol_20261031.json` and
  `docs/PAPER_NEURAL_CLASSIFICATION_PROTOCOL.md`: compact shared ResNet1D,
  classification fine-tuning, episodic ProtoNet and first-order MAML, all
  seeds42/43/44. The cited paper is regression and this is not exact
  reproduction. Source-meta support/query records must be disjoint;
  eligibility excludes a class with only one source record from meta tasks,
  while retaining all seven classes in supervision and final evaluation.
- The user authorized a four-round follow-up search. Follow
  `docs/MULTIROUND_OPTIMIZATION_PROTOCOL.md` and
  `data/research/multiround_optimization_plan_20261031.json`. All four
  source-only development selections must be locked and hashed before any
  round scores the fixed final query. Round1 MLP source improvements,
  round2 weighted SVM parameter adaptation, round3 anchored replay
  fine-tuning and round4 first-order MAML refinement are separate paths.
  Preserve matched no-target-label controls, every seed and negative result;
  never use final-query scores to design or choose a later round.
- The user's three-image follow-up follows
  `docs/IMAGE_GUIDED_OPTIMIZATION_PROTOCOL.md` and
  `data/research/image_guided_plan_20261031.json`: 40 feature/classifier,
  21 probability/decision and 20 probability-fusion configurations.
  All three track locks precede a new shared final-evaluation release.
  Macro-F1 is primary with accuracy/balanced-accuracy guards. Source OOF
  calibration excludes singleton validation rows while keeping them in
  fold training; every fold refits its transformations. This round uses
  zero target-support labels. Positive temperature preserves argmax;
  decision bias, probability reliability and fusion gains are separate.
- The latest user-directed four-stage search follows
  `docs/STAGED_NESTED_OPTIMIZATION_PROTOCOL.md` and
  `data/research/staged_nested_plan_20261031.json`: restore the original
  855-sample5outer/2inner folds and standard SVC.predict voting. Reproduce
  the historical90.41% reference separately. R1 MC36, R2 SVM96,
  R3 preprocessing7 and R4 combination7 are selected separately within
  every outer fold; never pool outer-fold inner results into one winner.
  All core/combinations selection locks precede outer evaluation release.
  The old513/171/171 workbooks remain a separate unchanged experiment.
- The user's paper-C direction keeps acid jujube THz identification and model
  optimization as the primary study (about70% work); other plants (about20%)
  and transfer/calibration (about10%) are auxiliary. Follow
  `docs/RESEARCH4_PAPER_C_MASTER_PLAN.md`. Preserve the audited R2 candidate
  and all historical results. New comparisons reuse the original855-sample
  five-outer/two-inner folds; all transformations and candidate selection
  stay within source partitions. Fixed171 calibration remains separate.
  Exact former strong64-32 MLP and new sklearn MLP are distinct controls.
  External record counts must not become invented independent sample counts;
  THz units/axes, labels, identity, provenance and license must be qualified
  before external validation or transfer. Do not replace the jujube task,
  rename standard methods as new algorithms, or merge this work into main.
