This is a complete, immutable evidence snapshot of the **old ML background study**, captured before authorized background recovery on 2026-11-15. It is independent of the chemistry computation.

- **83,017 queue-committed completed jobs** of 89,267 scheduled jobs. The 6,246 queued and 4 previous-boot running jobs are excluded from the completed result count; existing incomplete files are preserved under an explicit incomplete path.
- Complete result, prediction, selection, fitted-model/lossless-recipe, diagnostic and split evidence; exact frozen NPZ and original workbooks; source/protocol; consistent SQLite backup; all attempt/recovery provenance and refreshed reports.
- Archive: **11,966,219,460 bytes in 714 numbered 16MiB parts**, containing 1,323,080 regular files. Every shard was uploaded, downloaded, and SHA-256 verified. All seven frozen source/input hashes match.

**[Download the complete multipart archive](https://github.com/jackzhu119/research4/tree/7a145aaab9be8a8d969452352b9c8e4a9697c6f1)** · [Manifest and SHA-256 values](https://github.com/jackzhu119/research4/blob/7a145aaab9be8a8d969452352b9c8e4a9697c6f1/ARCHIVE_MANIFEST.json) · [Restore/download helper](https://github.com/jackzhu119/research4/blob/7a145aaab9be8a8d969452352b9c8e4a9697c6f1/restore_archive.py)

[Research summary, source, protocol and runtime records](https://github.com/jackzhu119/research4/tree/research/background-study-1360000-20261115)

Archive commit: `7a145aaab9be8a8d969452352b9c8e4a9697c6f1`. This release tag resolves to that exact archive commit. The archive branch is independent of the pre-existing multi-GB repository tree.

The release has **no binary release assets**: the authenticated `uploads.github.com` endpoint returned HTTP 401. Full data was successfully stored and verified through Git Data API blobs on the archive branch instead. Use the numbered shards and restore helper above. The helper verifies SHA-256 and supports streaming gzip CRC verification without an extra full archive copy.

Canonical reports were refreshed without fitting new models: 83,019 complete non-testing result files, zero invalid/testing. Two complete files were not yet queue-committed and are separately labelled incomplete, excluded from the 83,017 archived-completed count and from full-repeat source estimates. The study remains incomplete; reported uncertainty is conditional on the same 300 physical source specimens and is not external validation.

The original cumulative authorization remains **1,360,000 CPU seconds**. Publication/report CPU is conservatively deducted by the supplementary resource guard; no authorization or scientific protocol was changed. Canonical training resumes only after this verified upload. Later computation is recorded separately from this immutable cutoff.
