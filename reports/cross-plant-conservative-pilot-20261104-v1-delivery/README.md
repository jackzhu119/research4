# Lossless full experiment delivery

The complete sealed v1 experiment is preserved in `full_artifacts.tar.gz`, including all native model weights, scalers and per-batch fit-scope logs. Every archived file was read back and matched to its original SHA-256. Original local files and the sealed index were not modified. Top-level report/metrics/predictions/figures remain directly browsable in Git.

From the repository root, restore only missing files (GNU tar; never overwrite existing weights):

```sh
tar --skip-old-files -xzf reports/cross-plant-conservative-pilot-20261104-v1-delivery/full_artifacts.tar.gz -C .
```

Verify the archive SHA in receipt.json and the restored files against the original report index.json before replay. The archive contains repository-relative report paths only. It is real development-experiment evidence, not new independent biological samples.
