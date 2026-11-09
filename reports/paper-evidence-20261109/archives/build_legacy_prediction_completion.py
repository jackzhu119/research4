#!/usr/bin/env python3
"""Back up ONLY the completed legacy predictions absent from existing Git backups.

Coverage input is produced by matching actual bytes to the Git blob SHA-1 at
fixed remote-verified commits, or SHA-256 in the previous lossless archive.
No scientific values are changed. Uses gzip level 0 because inputs are already
compressed and measured recompression saves less than 0.05%.
"""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import time
import build_archive as common


def main():
    started = time.time()
    source = Path("/tmp/research4-legacy-prediction-coverage-20261109.json")
    coverage = json.loads(source.read_text())
    root = Path(__file__).resolve().parent / "legacy-prediction-completion"
    root.mkdir(exist_ok=True)
    if (root / "manifest.json").exists() or list(root.glob("scientific-results.tar.gz.part*")):
        raise SystemExit("Refusing to overwrite a previous prediction archive.")
    common.DEST = root
    missing = [r for r in coverage["files"] if not r["already_remote"]]
    records = []
    blobs = {}
    for r in missing:
        path = Path(r["source_path"])
        assert common.digest_file(path) == r["sha256"], "input changed after coverage audit"
        assert path.stat().st_size == r["bytes"]
        relative = r["path"].split("/runs/", 1)[1]
        records.append({"path": f"legacy1305/runs/{relative}", "bytes": r["bytes"], "sha256": r["sha256"]})
        blobs[r["sha256"]] = path
    index = {
        "schema": "research4-lossless-science-index-v1",
        "jobs": {"legacy_prediction_completion": [r["path"].split("/runs/", 1)[1].split("/", 1)[0] for r in missing]},
        "files": records, "excluded_legacy_predictions": [],
    }
    def write_gzip(path, value):
        with path.open("xb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=6) as zipped:
                zipped.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())
    write_gzip(root / "file-index.json.gz", index)
    coverage.update({
        "scope": "Exact byte coverage of all 1305 completed legacy validation_predictions.csv.gz files. New archive contains only 591 missing files; 714 are in fixed prior Git commits/backup archive.",
        "prior_remote_heads_verified_by_root": {
            "relaunch": "8a388b8b4d8a5baf9f2e4321e76455c2c0b49eca",
            "nextstage": "71a303e089c7c7b7b2557d62372aaceda44e2f91",
            "backfill": "f38113e51072f0a99c864b2e6dae79a9169ff93c",
        },
        "prior_backfill_manifest_sha256": common.digest_file(Path("/workspace/research4-backfill-20261108-001/manifest.json")),
    })
    write_gzip(root / "all-1305-prediction-coverage.json.gz", coverage)
    parts = common.Parts()
    with gzip.GzipFile(filename="", fileobj=parts, mode="wb", mtime=0, compresslevel=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for sha, path in sorted(blobs.items()):
                info = tarfile.TarInfo(f"blobs/{sha}")
                info.size = path.stat().st_size
                info.mode = 0o644
                info.mtime = 0
                with path.open("rb") as inp:
                    archive.addfile(info, inp)
    parts.close()
    manifest = {
        "schema": "research4-lossless-science-archive-v1",
        "status": "BUILT_NOT_YET_INDEPENDENTLY_VERIFIED",
        "job_counts": {"legacy_prediction_completion": len(missing)},
        "included_files": len(records), "unique_blobs": len(blobs),
        "included_original_bytes": sum(r["bytes"] for r in records),
        "archive_bytes": parts.total, "stream_sha256": parts.hasher.hexdigest(),
        "part_size_limit": common.PART_BYTES, "parts": parts.parts,
        "index": {"name": "file-index.json.gz", "bytes": (root / "file-index.json.gz").stat().st_size, "sha256": common.digest_file(root / "file-index.json.gz")},
        "coverage": {"name": "all-1305-prediction-coverage.json.gz", "bytes": (root / "all-1305-prediction-coverage.json.gz").stat().st_size, "sha256": common.digest_file(root / "all-1305-prediction-coverage.json.gz")},
        "excluded_legacy_prediction_files": 0,
        "scope": "Byte-for-byte completion archive of 591 legacy prediction gzip files; union with 714 fixed prior Git-backed predictions covers all 1305 legacy jobs. No models or spectral caches included.",
        "compression": "gzip level0 around existing .csv.gz bytes; one continuous tar stream split at 24MiB",
        "build_wall_seconds": time.time() - started,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:v for k,v in manifest.items() if k != "parts"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
