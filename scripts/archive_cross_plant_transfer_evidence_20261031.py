"""Bounded lossless archives for the authorized cross-plant transfer evidence.

Preserve original files in place. Never train, score or deserialize model
checkpoints. The historical archive helper is a hash-checked, immutable dependency.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "reports/cross-plant-deep-transfer-20261031-v1"
OLD_HELPER = ROOT / "scripts/archive_public_thz_evidence_20261031.py"
OLD_HELPER_SHA256 = "432953764c8d57e16a792ba16e86a7dcecff259ee2a6953abbd9738d46184692"
ENTRIES = ("neural/source_models", "neural/outer_models", "empirical/prepared/fits")


def _load_old():
    if hashlib.sha256(OLD_HELPER.read_bytes()).hexdigest() != OLD_HELPER_SHA256:
        raise ValueError("Immutable archive dependency bytes changed")
    spec = importlib.util.spec_from_file_location("immutable_public_thz_archive", OLD_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = _load_old()
ArchiveError = old.ArchiveError


def verify_dependency():
    if old.digest(OLD_HELPER) != OLD_HELPER_SHA256:
        raise ArchiveError("Immutable archive dependency bytes changed")


def create_evidence_archive(source_root, archive_dir, chunk_bytes=old.CHUNK_BYTES,
                            max_archive_bytes=old.MAX_ARCHIVE_BYTES):
    verify_dependency()
    source_root = Path(source_root)
    for entry in ENTRIES:
        if not old.safe_target(source_root, entry).is_dir():
            raise ArchiveError("All three completed model/diagnostic directories are required")
    return old.create_archive(source_root, archive_dir, entries=ENTRIES,
                              chunk_bytes=chunk_bytes, max_archive_bytes=max_archive_bytes)


def verified_manifest(archive_dir):
    verify_dependency()
    manifest = old.load_verified_manifest(archive_dir)
    if manifest["source_entries"] != list(ENTRIES):
        raise ArchiveError("Archive source eligibility differs from the frozen transfer scope")
    for record in manifest["files"]:
        if not any(record["path"].startswith(entry + "/") for entry in ENTRIES):
            raise ArchiveError("Artifact outside the three qualified evidence directories")
    return manifest


def restore_evidence_archive(archive_dir, target_root):
    """Reuse verified format and safety checks; open each ZIP once for bulk restore."""
    archive_dir, target_root = Path(archive_dir), Path(target_root)
    manifest = verified_manifest(archive_dir)
    matching = 0
    for record in manifest["files"]:
        path = old.safe_target(target_root, record["path"])
        if path.exists():
            if not path.is_file() or path.stat().st_size != record["bytes"] or old.digest(path) != record["sha256"]:
                raise ArchiveError("Existing target differs; nothing overwritten")
            matching += 1
    restored, zip_opens = 0, 0
    with ExitStack() as stack:
        zips = {}
        for record in manifest["files"]:
            path = old.safe_target(target_root, record["path"])
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".transfer-restore-", delete=False) as dst:
                temporary = Path(dst.name)
                try:
                    if record["storage"] == "zip":
                        name = record["archive"]
                        if name not in zips:
                            zips[name] = stack.enter_context(zipfile.ZipFile(archive_dir / name))
                            zip_opens += 1
                        with zips[name].open(record["member"]) as src:
                            count, sha = old.copy_hashed(src, dst, record["bytes"])
                    else:
                        total = hashlib.sha256()
                        count = 0
                        for part in record["parts"]:
                            n = 0
                            h = hashlib.sha256()
                            with gzip.open(archive_dir / part["archive"], "rb") as src:
                                for block in iter(lambda: src.read(old.BUFFER), b""):
                                    n += len(block)
                                    if n > part["bytes"]:
                                        raise ArchiveError("Gzip decoded size exceeds manifest")
                                    dst.write(block)
                                    total.update(block)
                                    h.update(block)
                            if n != part["bytes"] or h.hexdigest() != part["sha256"]:
                                raise ArchiveError("Gzip part decoded bytes differ")
                            count += n
                        sha = total.hexdigest()
                    if count != record["bytes"] or sha != record["sha256"]:
                        raise ArchiveError("Restored bytes differ")
                    dst.flush()
                    os.fsync(dst.fileno())
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            try:
                old.safe_target(target_root, record["path"])
                os.link(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            restored += 1
    return {"source_files": manifest["source_files"], "restored": restored,
            "already_matching": matching, "zip_handles_opened_for_restore": zip_opens,
            "archive_manifest_sha256": old.digest(archive_dir / "archive_manifest.json"),
            "immutable_dependency_sha256": OLD_HELPER_SHA256}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("archive", "verify", "restore"))
    parser.add_argument("--archive-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.archive_dir.resolve().is_relative_to(RUN_ROOT.resolve()):
        parser.error("Archive directory must stay inside this transfer run")
    if args.stage == "archive":
        manifest = create_evidence_archive(RUN_ROOT, args.archive_dir)
        result = {"source_files": manifest["source_files"], "source_bytes": manifest["source_bytes"],
                  "archive_files": len(manifest["archives"]),
                  "archive_manifest_sha256": old.digest(args.archive_dir / "archive_manifest.json")}
    elif args.stage == "verify":
        manifest = verified_manifest(args.archive_dir)
        result = {"source_files": manifest["source_files"], "archive_files": len(manifest["archives"]),
                  "archive_manifest_sha256": old.digest(args.archive_dir / "archive_manifest.json")}
    else:
        result = restore_evidence_archive(args.archive_dir, RUN_ROOT)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
