# Current dataset and paper-result policy

The user explicitly withdrew pre-correction mixed 枣仁/酸枣仁 classifier results on 2026-11-12. This supersedes earlier instructions to retain those results as current paper evidence.

## Paper inputs

- Read `paper/current/README.md` and `paper/current/manifest.json` before writing or comparing classifier results. Only explicitly allowlisted corrected-kind results may enter the paper. Run `python scripts/build_current_paper_results.py --verify` to validate this view.
- Do not resurrect withdrawn model outputs, old predictions relabeled after training, old R-round scores, contaminated 93-family paper packages or mixed historical workbooks from Git history, backups or generators.
- Preserve every included corrected positive and negative result. Corrected six-task/three-seed experiments, the original corrected 1305 plus V4 330, and B1--B5 remain valid internal evidence. Backups, continuation and verification are not new training rounds or independent specimens.
- `legacy_final`, `legacy-prediction-completion`, `OLD-*`, `P-CV-OLD` and `legacy_TRAIN_*` in pinned corrected sources refer to the corrected 1305 matrix. Do not delete by those names or by a date/percentage alone.
- `maintenance/withdrawn_results_rules.json` records approved result-deletion scope. Source code sealed by corrected protocols, raw data, current model outputs and independent quantum calculations are protected. Use the paper manifest rather than recursively scanning report directories.

## Data and tasks

- Acquisition facts for all four herbs were supplied and confirmed by the user on 2026-11-12 in `data/annotations/acquisition_conditions_user_declaration_20261112.json`. The same Menlo TERA K15 setup, 292 K, nitrogen enclosure humidity below 4%, 1550 nm optical wavelength and 33 mW optical power apply to all four kinds. PE blank reference response correction is confirmed; software/settings remain unspecified. Use `paper/current/methods/` for paper-ready equipment, equations and actual workbook/record splits. Do not overwrite sealed metadata or raw spectra to fill acquisition fields.
- Current input version is `2026-11-08-confirmed-origin-kind-v1`: 18 workbooks and 1672 records: 枣仁364, 酸枣仁500, 黄芪403, 天麻405. Discover inputs through `data/active_workbooks.json`, never raw/archive recursion. Four workbook views do not quadruple specimen count.
- Preserve current workbook bytes, record IDs, uploaded spelling including 天麻安徽毫州.xlsx, and hashes. ZY01--ZY02 are 枣仁; ZY03--ZY07 are 酸枣仁. ZR01--ZR03 are additional 枣仁 sources. Original855 spectra remain inputs after label correction.
- Paper inputs are frozen in `paper/current/inputs/`: corrected registry, metadata and TRAIN999/CAL334/PRED339 split. Earlier root metadata flags predate the later independence declaration; do not override frozen roles or labels with historical flags.
- The 2026-10-31 declaration covers original 855; `data/annotations/specimen_independence_user_declaration_20261108.json` separately covers 817 additional records (黄芪403, 天麻405, 枣仁9). These are user declarations, not independent laboratory identification or proof of new batches. Do not invent batch IDs.
- Pure 酸枣仁 origin tasks use ZY03--ZY07; pure 枣仁 origin tasks use ZY01/ZY02/ZR01--ZR03. Newly added枣仁 origins have only 3 specimens each. Keep task roles and denominators separate rather than ranking unlike metrics.
- B experiments remain exploratory internal TRAIN-only comparisons on previously exposed specimens. E05 is `NOT_RUN_NEEDS_NEW_BATCH_DATA`; CPU work cannot create independent support/query batches.

## Branches and quantum work

- `docs/repository/` is a dated operational inventory, not another paper-result allowlist. Withdrawn classifier rounds are excluded. A chat, open PR or commit date alone does not prove liveness.
- Quantum calculations, controls, arrays, sealed numerical protocols and job ledgers are outside this deletion. Preserve them. Judge each molecule/model by real stationary-point, mode and IR acceptance; water controls or ML validation do not certify all targets or peak assignments.
- Ordinary deletion commits remove files from current versions. Historical Git objects remain for reproducibility, without permission to cite withdrawn classification scores as current results.
