# Staged nested-search evidence

All inner metrics and raw predictions are preserved byte-for-byte; large JSON files are gzip-compressed with original SHA/size listed in `compressed_json_files.json`. Five outer-fold tar archives retain original training statistics, frequency indices, signed stability, importance and rankings.

The original per-MC coefficient matrices remain in the cloud cache and are intentionally omitted from GitHub. `omitted_mc_coefficient_matrices.json.gz` records every original SHA/shape/dtype/seed and ordered source identity; `coefficient_summaries.tar.gz` extracts to `coefficient_summaries/` and stores exact mean, sample standard deviation and prefix-specific floor. Their original numerical arrays were independently audited before archiving. Rebuild full matrices using the frozen input, ordered source IDs and protocol; do not claim this archive contains the raw matrices.

Outer model roles may alias one underlying fitted pipeline. `deliverable_models/manifest.json` records every unique source model and each role mapping. No outer test label was used for selection, and no fitting occurred during archiving.
