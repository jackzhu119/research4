<!-- corrected-paper-policy-20261112 -->
## Current user decision: withdraw mislabeled results

The 2026-11-12 user instruction withdraws pre-correction mixed 枣仁/酸枣仁 classifier outputs and old-prediction relabeling from current repository results and paper writing. This supersedes older requests to retain or cite those scores. Use only main:paper/current and its explicit manifest for paper claims. Preserve raw spectra, shared source code, corrected 1305/V4/B outputs and quantum computations. Do not rebuild removed paper bundles or import withdrawn results from Git history into current paper inputs.

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
  were rewritten. Physical independence is now separately user-declared below;
  actual sample logs and batch information remain unverified.
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
- The subsequent explicit user reply on 2026-11-08, “每列都是独立实物样品”,
  separately confirms physical independence for the remaining 817 records
  (HQ01--HQ04, TM01--TM04, ZR01--ZR03). See
  `data/annotations/specimen_independence_user_declaration_20261108.json`.
  These now have `physical_sample_id=record_id` with basis
  `user_declared_independent_20261108`; preserve the old 855-record basis.
  This reply supplies no new batch counts or IDs.
- The fixed current train/calibration/prediction assignment is
  `data/research/species_revision_20261108/split_manifest.csv`; its protocol
  and summary are in the same directory. Reuse this assignment for all tasks
  and methods; fit selectors and choose parameters using training data only.
  Calibration is support for prespecified adaptation; prediction labels are
  used only for the final comparison. Historical ZY samples have already been
  used in model development, so this is not a previously unseen external test.
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


## Current quantum queue assignment (2026-11-13)

The user explicitly stopped Calycosin and requested return of its existing partial results, assigning the available compute to Spinosin. Do not automatically relaunch `background_pyscf_repair_20261111.py` or the old four-molecule serial queue: that would restart the stopped Calycosin task. Calycosin has 90/396 qualified displacement points, not a complete frequency spectrum; preserve the partial-data handoff and stop receipt. Resume Calycosin only on a new explicit user instruction.

Spinosin is owned by branch `research/spinosin-direct-20261113`, worktree `/workspace/research4-spinosin`, and `scripts/background_pyscf_spinosin_direct_20261113.py --launch`. Its output is `reports/pyscf_spinosin_direct_20261113/`; use this dedicated supervisor after environment restoration. It uses the four-CPU allocation, with PySCF max_memory=10000 MB to leave operating-system and temporary-buffer headroom. Original frozen chemistry modules stay unchanged. Check actual process/heartbeat state rather than treating saved RUNNING metadata as proof of a live process. CPU authorizations are permissions, not purchased cloud credits or guarantees against platform suspension.
