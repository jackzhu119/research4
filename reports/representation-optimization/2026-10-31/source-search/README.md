# Stage A: source-only spectral pipeline search

This is compact evidence for the completed54-candidate source-model search. Each pipeline was fitted separately in30development tasks using only the original513-sample training role. Within each task,307specimens fit the source transforms/classifier,103form the reserved support pool, and103are developer queries. The outer171-sample prediction role was not scored in Stage A.

Actual work:1620source classifier fits,1620successful checkpoints,0failures,166860developer per-ID predictions. The original run took90.66seconds. A completed-run `--resume` check returned without refitting. These tasks overlap the same physical samples; they are not1620independent studies or new batches.

The source selection used the first24tasks for candidates and the last6for a fixed development gate. The selected source is `s024_sg0_snv_shrinkage_lda`:15-point/cubic SG smoothing, per-row SNV, train-fitted feature scaling and uniform-prior shrinkage LDA. Its mean selection/gate accuracies were85.8010%/87.8641%, compared with83.5356%/83.9806% for the fixed SNV-SVM reference. These are development results, not final171-query accuracies and not few-shot calibration gains.

The deduplicated Stage-B source set is the reference `s010_snv_svc`, the selected LDA, and `s014_snv_plsda` (10latent variables). Source selection lock SHA:

`86ebd3e827ec98285669a31f9c82e16f6112873b07808808ca6517a15ad63956`

## Compact files and exact recovery

- `source_checkpoints.jsonl.gz`:1620JSONL records. Each contains its original relative path, SHA256, candidate, episode, and **exact original UTF8JSON text** in `raw_json_utf8`. Decode this field to recover the original checkpoint file byte-for-byte. All1620reconstructions were checked.
- `source_development_predictions.csv.gz`:166860rows with added `candidate_id` and `episode` columns. Group by those columns, drop them, and write the four original columns (`record_id,y_true,y_predicted,is_correct`) with UTF8BOM and LF endings to recover each original predictionCSV exactly. All1620reconstructions were verified against the original bytes and SHA.
- `source_metrics.csv`:1620checkpoint-level rows including accuracy, balanced accuracy, macroF1, fit status, warnings, model/prediction SHA and fit time.
- `original_files_sha256.json`:4872original-file entries, including all1620joblib paths and hashes. The model bytes were checked without loading or fitting them; their1.2GB runtime cache is deliberately not copied into this compact archive.
- Source grid,30partitions, aggregate, selection lock, provenance, preparation, completed status, actual log and invocation records are copied unchanged. `archive_validation.json` records the recovery checks. `files_sha256.json` indexes this archive.

All180shrinkage-LDA fits recorded sklearn's singleton-class covariance warning, because one declared origin has only one source specimen in each development task. These warnings were preserved, not treated as extra data or erased; fits and finite-score checks succeeded. There is no independent geographic verification or cross-batch claim. Anonymous labels follow the user's supplied workbook groups.

`build_archive.py` is a read-only consolidation helper. It never fits or loads a model. It can rebuild compact evidence from a completed runtime source directory. To resume or reproduce the actual model search, use `scripts/run_spectral_source_search.py` with the recorded protocol and input files; all source statistics must remain fitted on each task's source partition only.
