# Actual read-only analysis execution

Actual exports: `../analysis/`. Numerical validation: `validation.json` and `validation_replay.json`. Interpretation: `INTERPRETATION_NOTES.md`. Recommended uncropped computational geometry figure: `spinosin_model_geometry_uncropped.png` / `.pdf`.

The original export is immutable and its 19 products match `../analysis/files_sha256.json`. Export and replay commands are preserved in `commands.jsonl`. The analyzer was frozen before the final ledger release and did not start quantum calculations.

To reproduce these checks and the supplemental geometry in a **fresh copy of this report directory**, from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .runtime/bin/python reports/pyscf-spinosin-pilot-20261031/analysis-execution/validate_actual_export.py
```

Do not rerun into this sealed evidence directory. A replay writes only its own validation and supplemental rendering products; it never writes `../analysis/` or any quantum inputs/results. Every synthetic fixture in the accompanying analyzer tests is explicitly a test fixture and is not a scientific result.
