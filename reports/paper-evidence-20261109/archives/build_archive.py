#!/usr/bin/env python3
"""Build a byte-preserving, content-addressed paper result archive; never fit models.

This is a one-time export from immutable completed scientific outputs, not a
replacement for the experiment runners. Read README.md before interpreting it.
"""
from __future__ import annotations
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

DEST = Path(__file__).resolve().parent
ROOTS = {
    "legacy1305": Path("/workspace/research4-relaunch/data/research/species_relaunch_20261108_v2/runs"),
    "v4_330": Path("/workspace/research4-nextstage/data/research/evidence_validation_20261108_v4/runs"),
}
PART_BYTES = 24 * 1024 * 1024


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as inp:
        for block in iter(lambda: inp.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Parts:
    def __init__(self):
        self.handle = None
        self.parts = []
        self.total = 0
        self.hasher = hashlib.sha256()
        self.position = 0

    def write(self, data):
        view = memoryview(data)
        total = len(view)
        while view:
            if self.handle is None:
                name = f"scientific-results.tar.gz.part{len(self.parts) + 1:03d}"
                path = DEST / name
                self.handle = path.open("xb")
                self.parts.append({"name": name})
                self.position = 0
            take = min(len(view), PART_BYTES - self.position)
            piece = view[:take]
            self.handle.write(piece)
            self.hasher.update(piece)
            self.total += take
            self.position += take
            view = view[take:]
            if self.position == PART_BYTES:
                self.handle.close()
                self.handle = None
        return total

    def flush(self):
        if self.handle:
            self.handle.flush()

    def close(self):
        if self.handle:
            self.handle.close()
            self.handle = None
        for p in self.parts:
            path = DEST / p["name"]
            p.update(bytes=path.stat().st_size, sha256=digest_file(path))


def main():
    started = time.time()
    if (DEST / "manifest.json").exists() or list(DEST.glob("scientific-results.tar.gz.part*")):
        raise SystemExit("Refusing to overwrite existing archive; use a new output directory.")
    records = []
    omitted = []
    blobs = {}
    jobs = {}
    for label, root in ROOTS.items():
        jobdirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "complete.json").is_file())
        jobs[label] = [p.name for p in jobdirs]
        for job in jobdirs:
            for path in sorted(job.rglob("*")):
                if not path.is_file() or path.is_symlink():
                    continue
                included = path.suffix in {".json", ".csv", ".md"} or path.name.endswith(".csv.gz")
                if not included:
                    continue
                relative = f"{label}/runs/{path.relative_to(root).as_posix()}"
                sha = digest_file(path)
                record = {"path": relative, "bytes": path.stat().st_size, "sha256": sha}
                if label == "legacy1305" and path.name == "validation_predictions.csv.gz":
                    omitted.append({**record, "storage": "LOCAL_ONLY_OR_PREVIOUS_BACKFILL_SUBSET", "source_path": str(path)})
                    continue
                records.append(record)
                blobs.setdefault(sha, path)
        print(label, "jobs", len(jobdirs), "indexed files", len(records), flush=True)
    assert len(jobs["legacy1305"]) == 1305 and len(jobs["v4_330"]) == 330
    # Supervisory receipts preserve exact commands' input/output hashes and CPU closure evidence.
    for label, source in [
        ("legacy_prior_receipts", Path("/workspace/research4-relaunch/data/research/species_relaunch_20261108_v2/control/receipts")),
        ("legacy_tail_receipts", Path("/workspace/research4-runtime-20261108/work/control/receipts")),
        ("v4_receipts", Path("/workspace/research4-nextstage/data/research/evidence_validation_20261108_v4/control/receipts")),
    ]:
        for path in sorted(source.glob("*.json")):
            sha = digest_file(path)
            records.append({"path": f"{label}/{path.name}", "bytes": path.stat().st_size, "sha256": sha})
            blobs.setdefault(sha, path)
    index = {"schema": "research4-lossless-science-index-v1", "jobs": jobs, "files": records, "excluded_legacy_predictions": omitted}
    packed = json.dumps(index, ensure_ascii=False, separators=(",", ":")).encode()
    with (DEST / "file-index.json.gz").open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=6) as out:
            out.write(packed)
    print("files", len(records), "unique blobs", len(blobs), "uncompressed", sum(p.stat().st_size for p in blobs.values()), flush=True)
    parts = Parts()
    with gzip.GzipFile(filename="", fileobj=parts, mode="wb", mtime=0, compresslevel=6) as zipped:
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
        "job_counts": {k: len(v) for k, v in jobs.items()},
        "included_files": len(records), "unique_blobs": len(blobs),
        "included_original_bytes": sum(r["bytes"] for r in records),
        "archive_bytes": parts.total, "stream_sha256": parts.hasher.hexdigest(),
        "part_size_limit": PART_BYTES, "parts": parts.parts,
        "index": {"name": "file-index.json.gz", "bytes": (DEST / "file-index.json.gz").stat().st_size, "sha256": digest_file(DEST / "file-index.json.gz")},
        "excluded_legacy_prediction_files": len(omitted),
        "excluded_legacy_prediction_bytes": sum(r["bytes"] for r in omitted),
        "scope": "All 1635 completed jobs' small scientific outputs and available supervisor receipts; all V4 predictions; excludes legacy bulk prediction bytes, weights, NPZ, NPY, CHK, HDF and temporary logs/locks.",
        "metrics_are": "Existing true experiment outputs; export did not retrain, relabel or recompute new scientific performance.",
        "build_wall_seconds": time.time() - started,
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:v for k,v in manifest.items() if k not in {"parts"}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
