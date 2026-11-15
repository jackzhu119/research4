# Completed-job storage maintenance

`storage_maintenance.py` is an independent, single-process storage worker. It does not edit the frozen scientific worker/spec/protocol, change the training queue, fit a model, or use outer-test scores. It reads SQLite with `mode=ro` and only accepts `completed` jobs whose result is genuinely `complete`, non-testing, and has no active attempt.

## Measured motivation

Read-only sampling selected the first/middle/latest completed job in each of nine families, 27 directories total. Temporary gzip-level-1 archives were measured in memory; official job files were untouched. Approximate mean logical KiB per job:

| Family | Total | Diagnostics | gz1 diagnostics | Fitted artifacts | Total after diagnostic packing |
|---|---:|---:|---:|---:|---:|
| amplitude | 667.3 | 8.1 | 2.6 | 583.1 | 661.8 |
| baseline | 1346.9 | 8.1 | 2.5 | 1259.6 | 1341.3 |
| coverage | 401.7 | 8.1 | 2.6 | 319.5 | 396.1 |
| material | 1305.6 | 8.0 | 2.4 | 1030.6 | 1300.0 |
| permutation | 639.2 | 158.5 | 92.3 | 404.2 | 573.0 |
| rank_context | 275.3 | 110.1 | 104.9 | 92.5 | 270.1 |
| retain_drop | 1109.9 | 911.2 | 526.4 | 125.4 | 725.1 |
| selection | 416.2 | 226.8 | 159.9 | 116.2 | 349.3 |
| spacing | 213.9 | 71.9 | 57.0 | 69.1 | 198.9 |

Current-family-weighted logical bytes decreased from about 569.7 to 531.9 KiB/job: diagnostic packing alone saves about 6.6%. This projects roughly 40.6/50.7 GiB for 80k/100k jobs, before other study files and filesystem-block/manifest overhead. This is a measured planning scenario, not a guarantee of future job mix. Official `jobs/` measured approximately 15 GiB. Fitted support-vector arrays dominate several families; their existing NPZ compression already retains around 96% of the uncompressed floating-point bytes. Therefore maintenance also calls the separately verified **lossless** `model_artifact_codec.encode_job(..., compact=True)`, after completed-job checks. It does not quantize arrays or implement its own model deletion.

## Diagnostics transaction and retained evidence

Only `rank_cache/`, `inner_scores/`, `candidate_grid.json`, and `inner_candidate_means.json` enter `diagnostics.tar.gz`. It explicitly drains gzip to EOF to check CRC, checks the exact member list, and verifies SHA-256/size of every member. The archive and its atomic, fsynced `diagnostics_archive_manifest.json` must be valid before any originals are removed. All remaining originals are checked before the first deletion. Job completion is checked again before cleanup and before every unlink. Interrupted cleanup is resumable from the verified archive/manifest.

The diagnostic archiver leaves these original paths unchanged: `result.json`, `predictions.csv`, `selection.json`, `job_frozen.json`, `split_frozen.json`, `checkpoint.json`, `cpu_receipt.json`, `heartbeat.json`, and `job.json`. It also leaves `fitted_artifacts.npz` untouched; only the separate root-authorized model codec may replace that container with its lossless encoding/recipe after bitwise reconstruction checks. Model-codec files are excluded from diagnostic archives. Shared `inputs/current1672_paired612.npz`, frozen `study_worker.py`, and per-job selection/lossless/recipe files are restoration dependencies and are never archived or removed by this diagnostic worker. Codec skips for `no_support_vectors` or `no_net_storage_saving` are successful no-op model states; they preserve the original container.

Never manually requeue an archived completed job while maintenance is running. Pause maintenance, restore diagnostics (and models through the model codec), verify them, and only then requeue. Queued/running/partial jobs are never candidates. Existing scientific reports continue to read the same result/prediction records.

## Invocation — official start remains the root's decision

```bash
python storage_maintenance.py --root /absolute/study/root --once
python storage_maintenance.py --root /absolute/study/root --start
```

`--start` is a foreground daemon in the same process. It creates no children. The caller may detach it. A process-level `flock` prevents duplicate maintenance. Natural `budget_exhausted` phase boundaries with a waiting/activating higher-budget handoff do not stop maintenance. A real root `STOP`, user stop, or final budget exhaustion does request graceful exit. Each batch handles at most 100 completed jobs; backlog batches pause 1 second, while an exhausted backlog waits 60 seconds. Priority is reduced to nice 10. SIGTERM/SIGINT requests graceful exit with an atomic final CPU ledger update. `--diagnostics-only` omits the separate codec. `--restore-job JOB_ID` verifies and restores diagnostics into a still-completed job; it refuses to overwrite differing files.

`storage_maintenance_ledger.json` has `accounts={boot:pid:start_ticks: {pid,start_ticks,boot_id,cpu_seconds,closed,...}}`. Each value is cumulative `process_time()` from process start (including imports), kept as a monotone high-water. The same account is updated; observations are not repeatedly added. No child CPU exists. Diagnostics and model-codec completion are tracked separately: if the codec is absent or diagnostics-only mode is used, model work remains pending and is backfilled after the codec becomes available; the completion cursor never marks missing model work done. Active ledger heartbeats occur at most about 0.75 seconds apart during archive streaming and waits, and after every job; model-codec calls execute inline and are charged once to this process. The budget guard must include these accounts and retain prior closed accounts. The final ledger write itself has a small unavoidable measurement tail; the global resource guard retains its shutdown safety reserve.

Below 1 GiB free, `STORAGE_WARNING.json` is written. If temporary space is insufficient, originals are preserved and the job is deferred; training scheduling is not changed. Verification/codec failures retain source data, write a warning, and retry with a cooldown up to three times. Source directories and unverified data are never deleted to make space.

## Tests

```bash
python -m unittest discover -s storage_maintenance_tests -v
```

Tests create and remove a separate temporary fake SQLite/job root. They check full round-trip hashes, protected-file byte identity, running/queued/partial/testing isolation, no deletion on injected verification failure, corrupt-gzip refusal, partial-cleanup restart, symlink refusal, CPU ledger monotonicity and failure retry behavior. Additional tests cover codec-unavailable backfill, separate component status, natural-budget handoff survival, and a job leaving completed state before cleanup. They do not start official maintenance or training.

## Component policy v2: completed split snapshots

Completed-only `split_frozen.json` is separately stored as `split_frozen.json.gz` with `split_frozen.archive_manifest.json`. The gzip EOF CRC, decompressed byte count, source SHA256, result identity and completed status are verified before the original is unlinked. Existing diagnostic tarballs and `DIAGNOSTIC_TOP` are unchanged. `--restore-job` now restores both diagnostics and the exact split bytes. Running, queued, partial and testing splits remain untouched.

On the first v2 batch, only this maintenance worker's own completion cursor is replayed from the beginning. `component_policy_version=2` and the previous maintenance cursor are recorded. This backfills splits of jobs already processed by v1. The scientific SQLite queue, attempts and cumulative CPU accounts are never reset or edited. The official maintenance daemon must be gracefully restarted by root to load v2; this patch does not signal or restart it.
