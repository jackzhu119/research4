#!/usr/bin/env python3
"""Verify every archived byte and optionally restore to a NEW empty directory.

Uses only Python's standard library. Does not execute any archived code or
deserialize pickle, joblib, NPY, or model files. It validates safe relative paths.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as inp:
        for block in iter(lambda: inp.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Concat(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.handle = None
        self.sha = hashlib.sha256()
        self.total = 0
    def readable(self): return True
    def readinto(self, buffer):
        while True:
            if self.handle is None:
                try: self.handle = next(self.paths).open("rb")
                except StopIteration: return 0
            n = self.handle.readinto(buffer)
            if n:
                self.sha.update(memoryview(buffer)[:n])
                self.total += n
                return n
            self.handle.close()
            self.handle = None


def safe_relative(value):
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or "\\" in value or not p.parts:
        raise ValueError(f"Unsafe path: {value}")
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent / "manifest.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only == bool(args.output):
        parser.error("Choose exactly one of --verify-only or --output EMPTY_DIRECTORY")
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.resolve().parent
    safe_relative(manifest["index"]["name"])
    index_file = root / manifest["index"]["name"]
    assert sha_file(index_file) == manifest["index"]["sha256"], "index SHA mismatch"
    index = json.loads(gzip.decompress(index_file.read_bytes()))
    by_sha = {}
    seen_paths = set()
    for record in index["files"]:
        safe_relative(record["path"])
        assert record["path"] not in seen_paths, "duplicate output path"
        seen_paths.add(record["path"])
        by_sha.setdefault(record["sha256"], []).append(record)
    if args.output:
        outdir = args.output.resolve()
        if outdir.exists() and any(outdir.iterdir()):
            raise SystemExit("Output must be empty; never overwrite scientific results.")
        outdir.mkdir(parents=True, exist_ok=True)
    for part in manifest["parts"]:
        safe_relative(part["name"])
        path = root / part["name"]
        assert path.stat().st_size == part["bytes"], "part size mismatch"
        assert sha_file(path) == part["sha256"], "part SHA mismatch"
    concat = Concat([root / p["name"] for p in manifest["parts"]])
    seen_blobs = set()
    with io.BufferedReader(concat) as inp:
        with gzip.GzipFile(fileobj=inp, mode="rb") as zipped:
            with tarfile.open(fileobj=zipped, mode="r|") as archive:
                for item in archive:
                    safe_relative(item.name)
                    assert item.isfile() and item.name.startswith("blobs/"), "non-regular archive member"
                    sha = item.name.split("/", 1)[1]
                    assert sha in by_sha and sha not in seen_blobs, "unexpected/repeated blob"
                    records = by_sha[sha]
                    assert all(r["bytes"] == item.size for r in records), "member size mismatch"
                    h = hashlib.sha256()
                    handle = None
                    if args.output:
                        first = outdir.joinpath(*PurePosixPath(records[0]["path"]).parts)
                        first.parent.mkdir(parents=True, exist_ok=True)
                        handle = first.open("xb")
                    with archive.extractfile(item) as blob:
                        for block in iter(lambda: blob.read(1024 * 1024), b""):
                            h.update(block)
                            if handle: handle.write(block)
                    if handle: handle.close()
                    assert h.hexdigest() == sha, "content SHA mismatch"
                    if args.output:
                        for record in records[1:]:
                            target = outdir.joinpath(*PurePosixPath(record["path"]).parts)
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with target.open("xb") as out:
                                with first.open("rb") as src: shutil.copyfileobj(src, out)
                    seen_blobs.add(sha)
            while zipped.read(1024 * 1024): pass  # Complete gzip CRC/trailer verification.
        while inp.read(1024 * 1024): pass
    assert seen_blobs == set(by_sha), "missing blobs"
    assert concat.total == manifest["archive_bytes"], "archive total byte mismatch"
    assert concat.sha.hexdigest() == manifest["stream_sha256"], "stream SHA mismatch"
    print(json.dumps({"status": "PASS", "mode": "verify-only" if args.verify_only else "restored", "jobs": manifest["job_counts"], "included_files": len(index["files"]), "verified_unique_blobs": len(seen_blobs), "verified_archive_bytes": concat.total, "excluded_legacy_prediction_files": len(index["excluded_legacy_predictions"]), "scope": manifest["scope"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
