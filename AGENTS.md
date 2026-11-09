# Research data conventions

The user's current data decision, effective 2026-11-08, is recorded in
`data/CURRENT_DATASET.md`, `data/active_workbooks.json` and
`data/annotations/origin_kind_confirmation_20261108.json`.

- Use the active registry as the default input entry point. Current version is
  `2026-11-08-confirmed-origin-kind-v1`: 18 workbooks and 1672 spectrum records.
  These comprise 枣仁 364, 酸枣仁 500, 黄芪 403 and 天麻 405 records. Do not
  recursively scan raw, archive or processed directories to discover inputs;
  those locations also contain historical versions and non-spectral summaries.
- The user has confirmed the origins and kinds represented by the uploaded
  filenames. `species` stores the user's Chinese kind labels, not a Latin
  botanical identification. Origin/species confirmation flags and the
  `origin_basis` / `species_basis` fields refer to this user declaration, not
  an independent laboratory identification or provenance audit. Preserve the
  uploaded spelling, including `天麻安徽毫州.xlsx`.
- The seven previously numbered workbooks remain in
  `data/processed/zaoren_20261031/`, with their ZY01--ZY07 record IDs unchanged.
  Their current basenames are 陕西延安枣仁.xlsx, 云南枣仁.xlsx,
  内蒙古酸枣仁.xlsx, 山西灵石酸枣仁.xlsx, 安徽亳州酸枣仁.xlsx,
  河北石家庄酸枣仁.xlsx and 山西酸枣仁.xlsx respectively. Renaming these
  files does not change their spectral values, column headers or hashes.
  ZY01 and ZY02 are now confirmed as 枣仁; the other five are 酸枣仁.
- Four Huangqi inputs use the user's short filenames in `data/raw/huangqi/`.
  The current 黄芪甘肃.xlsx includes the previously absent refractive-index
  table; the incomplete prior version is retained for provenance. Existing
  time-domain values are unchanged; eight frequency/absorption values differ
  only in export precision (maximum absolute difference about 1.42e-14).
- The three added 枣仁 workbooks are in `data/raw/zaoren_20261108/` (ZR01--ZR03)
  and the four added 天麻 workbooks are in `data/raw/tianma/` (TM01--TM04).
  Registry/metadata record IDs do not imply the raw workbook column headers
  were rewritten, or that individual physical samples or batches were verified.
- The current old/new filename map is `data/confirmed_names_20261108.csv`;
  the current 18-workbook summary is `data/confirmed_workbooks_20261108.xlsx`.
  The older `酸枣仁_编号对照表.xlsx`, numbered-name mapping CSVs and verification
  JSONs remain historical evidence. Preserve their original names, paths and
  hashes as historical fields; use their added current-path links or the
  active registry for current file locations. Never count archived originals,
  numbered outputs and renamed current files as separate observations.
- The 2026-10-31 user declaration of independent samples and one source batch
  per group applies only to the original 855 ZY records. Their
  `physical_sample_id=record_id` basis remains
  `user_declared_independent_20261031`. Do not extend this declaration to the
  new raw workbooks or Huangqi records, and do not invent actual batch IDs.
- Historical experiments on ZY01--ZY07 used anonymous group labels and were
  conducted on a collection now known to mix 枣仁 and 酸枣仁. Retain their
  original counts, hashes, metrics and protocols, but do not describe them as
  validated within-species acid-jujube origin identification. Do not transfer
  their scores to the current 18-workbook dataset or imply new experiments
  were run during the filename/annotation update.
- The historical approved protocol remains documented in
  `docs/APPROVED_ZAOREN_EXPERIMENT_PLAN.md` and `data/research/`: outer 5 /
  inner 2 / CARS-internal 2 folds, split seed 20261031, model seed 42. ZY02 has
  five records. Gaussian remains excluded from that approved experiment.
  Any new origin study must account for kind, group and batch confounding;
  sample-level validation is not a real cross-batch experiment. New-batch
  calibration still needs independently annotated support/query batches.

## New PySCF background queue (2026-11-08)

The current user explicitly authorized a new finite 90,000 CPU-second PySCF
background run, including continued execution after this chat ends. This is a
separate prospective authorization from the historical classification protocol.
Read `docs/PYSCF_BACKLOG_AUDIT_20261108.md` and
`docs/PYSCF_AUTONOMOUS_90000_20261108.md`, then use the frozen configuration and
independent ledger under `reports/pyscf-autonomous-90000-20261108/`.
Do not reset budgets, relabel failed results, duplicate a live queue, merge main,
or modify the frozen scientific/supervisor source while the queue is running.
The finite priority is water controls, new grid3 protocatechuic-acid optimization,
two complete numerical Hessians and signed-mode/step validation. Preserve the
Spinosin and Calycosin backlogs and all historical imaginary modes. IR remains
NOT_COMPUTED. An independent session can outlive the chat; cloud instance uptime
is not guaranteed. Unknown unclosed process/CPU identity blocks automatic resume.

## Latest user directive: continuous execution v3 (2026-11-08)

The latest actual user explicitly rejected stopping the pipeline on strict
scientific thresholds and requested continuous background computation. For NEW
v3 jobs this supersedes earlier scientific gating/no-retry prescriptions. Read
`docs/PYSCF_CONTINUOUS_EXECUTION_20261108.md`. Convergence, stationarity and
numerical consistency are truthful result-quality labels, not execution gates;
finite unconverged results must retain native false and their warnings. Recover
individual failures and continue independent points, Hessians and molecules.
Never manufacture missing data or relabel old results as passed. The original
90,000 CPU-second authorization still covers all runs: prior charged costs are
9678.939876, v3 cap 80,000, combined ceiling 89678.939876. Keep actual CPU costs,
source/input integrity, process cleanup and explicit STOP; never reset costs or
launch a duplicate live queue. The current driver and ledger are under
`reports/pyscf-continuous-20261108-v3/`; use its status instead of historical
v1/v2 ledgers. Do not edit frozen v3 producers while the queue is running.

## Explicit continuation and minimum repair (2026-11-09)

After being informed that the host restarted and the old final CPU tail cannot
be reconciled exactly, the user explicitly requested immediate continuation and
then requested repairing minimum certification alongside Calycosin. This is a
new prospective continuation instruction; do not reset or pretend to close the
old ledger. Read the authorization and allocation records under
`reports/pyscf-resume-20261109-v4/`. The assistant selected a finite 28,000 CPU
second execution window: 18,000 for acid minimum repair, 10,000 for Calycosin
continuation. These are operational limits, not a claim that the user explicitly
specified a new numeric CPU allowance or that the old 90,000-second grant was
exactly reconciled. Retain all old costs and unknown-tail evidence.

Reuse the 28 authenticated Calycosin displacement points and the unchanged v3
scientific source; write new attempt logs and current-boot costs separately.
Minimum repair must inspect actual force, electronic residual, derivative and
eigenvalue evidence. Distinguish native solver acceptance from fresh-residual
targets and stricter certification; never relabel old failed certificates or
claim a changed geometry has an old geometry's Hessian. Scientific quality
warnings remain nonblocking. Two allocations may run concurrently; never launch
a duplicate allocation or exceed the recorded combined continuation window.
