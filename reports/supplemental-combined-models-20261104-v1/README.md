# Combined fixed-query model figures

Actual independently audited metrics only; no fitting, spectra, labels or model weights were loaded by this plotting script. Four task-specific comparisons share the registered source/support/query memberships within each task. Pipelines have different source-search budgets and preprocessing; this is not an isolated classifier or feature contribution experiment.

- `model_comparison.csv`: 28 pipeline/task summaries. `per_seed_metrics.csv`: all84 actual seed rows.
- `accuracy_comparison.png/.pdf` and `macro_f1_comparison.png/.pdf`: all seven no-calibration pipelines on each fixed query.
- `paired_old171_accuracy.csv`: 7 model summaries; `paired_old171_per_seed.csv`:21 paired seed rows; `paired_old171_accuracy.png/.pdf`: same old171 queries, A seven-class versus B ten-class models. B adds three source specimens and three classes, so the difference is not a causal estimate of additional-data benefit.

All error bars are **sample SD over the same three seeds**, not confidence intervals and not independent new subject cohorts. The classical original table used population SD; this build derives common sample SD directly from the three independently recomputed seed metrics and checks the original means/population SD. Neural sample SD and all means are checked against the original score table. All per-seed accuracy values are checked against actual correct/N; paired net gains are checked against corrected minus harmed counts.

These are exposed development cohorts. The old96.0234% five-outer/two-inner result is a separate protocol and is not plotted as a directly comparable line. New ten-class query groups have only one specimen each, and repeated seeds do not increase that independent N. Anonymous labels do not verify botanical identity, named geography or cross-batch performance. C-versus-single neural changes include extra alpha representation/resolution as well as network capacity; the matched multi-representation effect is M-versus-C.

Reproduce into a **new** directory with the existing CPU Python and installed plotting libraries:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/workspace/research4/.runtime/quantum-pilot-site:/workspace/research4/.runtime/quantum-upgrade-site /workspace/research4/.runtime/bin/python reports/supplemental-combined-models-20261104-v1/build_figures.py --output reports/supplemental-combined-models-REPRODUCTION
```
