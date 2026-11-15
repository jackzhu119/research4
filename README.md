# Old ML evidence publication — 15 November 2026

This publication preserves the old ML background study independently of the chemistry computation. It contains **83,017 SQLite-committed completed jobs out of 89,267 scheduled jobs**. The 6,246 queued jobs and four previous-boot running jobs are unfinished and are not counted as completed evidence. Existing unfinished job directories are preserved under `incomplete_not_counted/` for provenance.

- [Summary/source branch](https://github.com/jackzhu119/research4/tree/research/background-study-1360000-20261115): research summary, source/protocol, publication manifests, and recovery evidence.
- [Multipart archive branch](https://github.com/jackzhu119/research4/tree/research/background-study-1360000-20261115-archive): `ARCHIVE_MANIFEST.json`, `SHA256SUMS`, and the complete `shards/` directory.
- [Prerelease pointer](https://github.com/jackzhu119/research4/releases/tag/research-ml-results-20261115): **pointer only, with no uploaded release assets**. The GitHub release upload host rejected the authenticated upload with HTTP 401, so the archive was published through Git Data API blobs on the archive branch. The release links to that archive; it does not contain the binary shards as release assets.

`archive_branch_publication.json` on the summary branch records the archive branch's immutable commit. Use that commit with the restore script to pin downloads. The complete archive uses one gzip-compressed PAX tar stream divided into numerically ordered 16 MiB shards (the final shard may be shorter). Every published shard was downloaded from GitHub and checked against its local SHA-256 before the temporary local shard was removed.

## Counts and current results

The refreshed canonical reports, generated at **2026-11-15T07:23:36Z**, validated **83,019** complete non-testing result files with zero invalid results and zero testing exclusions. Two files belonged to previous-boot running attempts that had written results before SQLite completion was committed. Those two jobs are preserved only as `incomplete_not_counted` and are excluded from the 83,017 completed-job archive count. Their source repeat has only two of five outer folds and does not enter method-level complete-repeat estimates.

There are 15,349 full source repeats, 990 full label-permutation repeats, and 1,322 fixed-role material fits. Source inference still concerns the same 300 physical specimens. Alpha-SNV selection currently averages about 93.20% internal source accuracy at MC20; this is a descriptive retrospective result, with conditional intervals and no claim of independent external validation or a statistically established winning method. The full research summary and per-method tables preserve the separate source, null, and fixed-PRED targets.

The cumulative CPU authorization remains **1,360,000 seconds**, independent of chemistry. This publication does not add budget or change the frozen protocol. Historical documents retain earlier budget phases. All seven sealed source/input hashes match the recorded pre-recovery hashes. The study remains incomplete.

## Download and verify

The restore helper uses Python 3.9+ and the standard library. It needs no GitHub token for this public repository. Download `restore_archive.py` from the summary branch, then fetch the manifest from the pinned archive commit:

```bash
ARCHIVE_COMMIT='REPLACE_WITH_COMMIT_FROM_archive_branch_publication.json'
curl -fL "https://raw.githubusercontent.com/jackzhu119/research4/${ARCHIVE_COMMIT}/ARCHIVE_MANIFEST.json" -o ARCHIVE_MANIFEST.json
python restore_archive.py --manifest ARCHIVE_MANIFEST.json --commit "$ARCHIVE_COMMIT" --verify-gzip
```

The script downloads `shards/<name>` from public raw GitHub URLs, verifies each shard's exact byte count and SHA-256, and reuses matching local files on subsequent runs. Failed downloads never replace a verified final file. `--verify-gzip` reads the ordered shards as one stream to EOF and checks gzip CRC and length without creating an additional full archive copy. By default, verified files remain in `ml_archive_download/`.

`--commit` is optional; without it, the script uses `archive_commit` if supplied by the manifest, otherwise the archive branch. An exact commit makes the download source immutable. Keep the manifest from the same commit.

To check already downloaded files without any network calls:

```bash
python restore_archive.py --manifest ARCHIVE_MANIFEST.json --offline --verify-gzip
```

Only request a concatenated tarball when needed:

```bash
python restore_archive.py --manifest ARCHIVE_MANIFEST.json --offline --reassemble
```

Reassembly writes an additional `ml_archive_download/ml-evidence-20261115.tar.gz` and refuses to overwrite an existing archive. Use `--archive-path PATH` to choose another filename. Separate shards occupy the manifest's `compressed_archive_bytes`; reassembly needs that amount again. Extraction also needs space for the archived files and filesystem metadata.

## Extract and restore the original study layout

After verifying all shards, extract directly from them to avoid a second compressed archive copy:

```bash
mkdir -p restored
cat ml_archive_download/ml-evidence-20261115.tar.gz.part* | tar -xz -C restored
python restore_archive.py --manifest ARCHIVE_MANIFEST.json --offline --verify-extracted restored
```

`--verify-extracted` verifies the SHA-256 of the archived member manifest against `ARCHIVE_MANIFEST.json`, then checks the size and SHA-256 of every file listed in it. It reads `restored/background_study_20261110/archive_members.jsonl.gz` by default; `--member-manifest PATH` can select another copy. The member manifest does not list itself; its own checksum comes from the outer archive manifest.

The extracted root is `restored/background_study_20261110/`:

- `jobs/`: all 83,017 completed jobs, including result, prediction, selection, split, fitted-model, and diagnostic evidence.
- `inputs/`: exact shared input NPZ and original workbook/input provenance.
- `snapshot_source/`: scripts, protocol, resource ledgers, authorizations, and a consistent SQLite backup.
- `reports/` and `publication_summary/`: refreshed reports and count/interpretation caveats.
- `attempts/`, `recovery_events/`, and `incomplete_not_counted/`: lineage and interruption evidence, kept distinct from completed results.

The source snapshot is deliberately separated from `jobs/` and `inputs/`. To use the original worker/model-codec APIs, copy `snapshot_source` contents into the extracted study root. Exclude disposable SQLite WAL/SHM sidecars; `queue.sqlite3` itself is the consistent backup. This example copies source files without launching training:

```bash
python - <<'PY'
from pathlib import Path
import shutil
study = Path('restored/background_study_20261110')
for source in (study / 'snapshot_source').iterdir():
    if source.name in {'queue.sqlite3-wal', 'queue.sqlite3-shm'}:
        continue
    target = study / source.name
    if target.exists():
        raise SystemExit(f'Refusing to overwrite existing path: {target}')
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)
PY
```

In particular, `study_worker.py` and `model_artifact_codec.py` must be at the study root beside the unchanged `inputs/` directory. Fitted support vectors may be stored as verified lossless NPZ plus reconstruction recipes. `MODEL_ARTIFACT_CODEC_README.txt` explains exact array reconstruction without model fitting. Completed diagnostics and split snapshots can also be compressed with their own verified manifests. Keep their compressed evidence and restoration dependencies together.

Restoring the evidence does not start workers, alter the queue, reset CPU accounts, or grant permission to rerun completed tasks. The archived runtime status describes the capture time, including the previous-boot interruption; subsequent resume verification is a separate publication record.

## Cumulative CPU accounting after publication

The summary branch also contains `resource_management/`: the supplementary publication/report CPU ledger, its enforcer, current status, and validation evidence. These resource files were created while the immutable evidence archive was streaming, so they are preserved separately on the summary branch. The original `snapshot_source/budget_guard_ledger.json` alone does not include the manually launched report or publication process.

Any later authorized continuation must retain both sets of accounts and run both resource guards under the unchanged 1,360,000-second cap. Restore the `resource_management` contents into the study's `ml_publication_20261115/` directory and read `RESOURCE_BUDGET_ENFORCEMENT.md` before a recovery. The extra guard deducts measured publication CPU plus conservative tail/control allowances and uses the canonical CPU budget stop marker. It does not grant new budget or change the frozen science.
