"""Lossless, bounded evidence archives; never overwrite the original or a restore target.

The CLI is deliberately limited to this run's licensed ginseng evidence. Fruit
and leaf originals and derivatives with unknown licenses are not eligible.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

SCHEMA = "public-thz-evidence-archive-v1"
DEFAULT_ENTRIES = ("selection_models", "transform_cache", "fit_scope_events.jsonl")
MAX_ARCHIVE_BYTES = 50 * 1024 * 1024
CHUNK_BYTES = 40 * 1024 * 1024
BUFFER = 1024 * 1024


class ArchiveError(ValueError):
    pass


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(BUFFER), b""):
            h.update(block)
    return h.hexdigest()


def relative_name(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ArchiveError("Unsafe relative path")
    p = PurePosixPath(value)
    if p.is_absolute() or any(x in ("", ".", "..") for x in value.split("/")):
        raise ArchiveError("Unsafe relative path")
    if str(p) != value:
        raise ArchiveError("Noncanonical relative path")
    return value


def safe_target(root, rel):
    rel = relative_name(rel)
    if root.is_symlink():
        raise ArchiveError("Root symlink refused")
    if root.exists() and not root.is_dir():
        raise ArchiveError("Root is not a directory")
    target = root / rel
    current = root
    for part in PurePosixPath(rel).parts:
        current = current / part
        if current.is_symlink():
            raise ArchiveError("Symlink refused")
        if current != target and current.exists() and not current.is_dir():
            raise ArchiveError("Existing parent is not a directory")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ArchiveError("Path escapes root")
    return target


def copy_hashed(reader, writer, expected_bytes=None):
    h = hashlib.sha256()
    size = 0
    while True:
        block = reader.read(BUFFER)
        if not block:
            break
        size += len(block)
        if expected_bytes is not None and size > expected_bytes:
            raise ArchiveError("Decoded size exceeds manifest")
        writer.write(block)
        h.update(block)
    if expected_bytes is not None and size != expected_bytes:
        raise ArchiveError("Decoded size differs from manifest")
    return size, h.hexdigest()


def create_archive(source_root, archive_dir, entries=DEFAULT_ENTRIES,
                   chunk_bytes=CHUNK_BYTES, max_archive_bytes=MAX_ARCHIVE_BYTES):
    source_root = Path(source_root)
    archive_dir = Path(archive_dir)
    if source_root.is_symlink() or not source_root.is_dir():
        raise ArchiveError("Real source directory required")
    if chunk_bytes <= 0 or chunk_bytes >= max_archive_bytes:
        raise ArchiveError("Chunk must be positive and smaller than archive cap")
    paths = []
    for entry in entries:
        p = safe_target(source_root, entry)
        if not p.exists():
            raise FileNotFoundError(p)
        if p.is_file():
            paths.append(p)
        else:
            for q in sorted(p.rglob("*")):
                if q.is_symlink():
                    raise ArchiveError("Source symlink refused")
                if q.is_file():
                    paths.append(q)
    if len(paths) != len(set(paths)):
        raise ArchiveError("Overlapping archive entries")
    if any(archive_dir.resolve().is_relative_to((source_root / entry).resolve()) for entry in entries):
        raise ArchiveError("Archive output cannot be inside a source entry")
    records = []
    for path in sorted(paths):
        rel = relative_name(path.relative_to(source_root).as_posix())
        records.append({"path": rel, "bytes": path.stat().st_size,
                        "sha256": digest(path)})
    archive_dir.mkdir(parents=True, exist_ok=False)
    zip_parts = []
    pending = []
    pending_bytes = 0
    for record in records:
        if record["bytes"] >= chunk_bytes:
            continue
        # Bound uncompressed payload and ZIP filename/central-directory overhead.
        estimate = record["bytes"] + 512 + 2 * len(record["path"].encode("utf-8"))
        if pending and pending_bytes + estimate > chunk_bytes:
            zip_parts.append(pending)
            pending = []
            pending_bytes = 0
        pending.append(record)
        pending_bytes += estimate
    if pending:
        zip_parts.append(pending)
    archives = {}
    for index, group in enumerate(zip_parts, 1):
        name = f"models-and-metadata-{index:03d}.zip"
        p = archive_dir / name
        with zipfile.ZipFile(p, "x", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as z:
            for record in group:
                info = zipfile.ZipInfo(record["path"], (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                with (source_root / record["path"]).open("rb") as src, z.open(info, "w") as dst:
                    count, sha = copy_hashed(src, dst, record["bytes"])
                if sha != record["sha256"]:
                    raise ArchiveError("Source changed while archiving")
                record.update(storage="zip", archive=name, member=record["path"])
        archives[name] = {"bytes": p.stat().st_size, "sha256": digest(p), "format": "zip"}
    for index, record in enumerate(records, 1):
        if record["bytes"] < chunk_bytes:
            continue
        record.update(storage="gzip_parts", parts=[])
        total_sha = hashlib.sha256()
        offset = 0
        with (source_root / record["path"]).open("rb") as src:
            part = 0
            while offset < record["bytes"]:
                part += 1
                name = f"large-file-{index:05d}-part-{part:03d}.gz"
                count = 0
                part_sha = hashlib.sha256()
                with (archive_dir / name).open("xb") as raw, gzip.GzipFile(
                        filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as dst:
                    limit = min(chunk_bytes, record["bytes"] - offset)
                    while count < limit:
                        block = src.read(min(BUFFER, limit - count))
                        if not block:
                            raise ArchiveError("Source truncated while archiving")
                        dst.write(block)
                        total_sha.update(block)
                        part_sha.update(block)
                        count += len(block)
                p = archive_dir / name
                archives[name] = {"bytes": p.stat().st_size, "sha256": digest(p), "format": "gzip"}
                record["parts"].append({"archive": name, "offset": offset,
                                        "bytes": count, "sha256": part_sha.hexdigest()})
                offset += count
            if src.read(1) or total_sha.hexdigest() != record["sha256"]:
                raise ArchiveError("Source changed while archiving")
    if any(v["bytes"] > max_archive_bytes for v in archives.values()):
        raise ArchiveError("Archive exceeds requested file cap; original files unchanged")
    for record in records:
        if digest(source_root / record["path"]) != record["sha256"]:
            raise ArchiveError("Source changed before final archive lock")
    manifest = {"schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_root_at_creation": str(source_root), "source_entries": list(entries),
                "source_files": len(records), "source_bytes": sum(v["bytes"] for v in records),
                "originals_modified_or_deleted": False, "max_archive_bytes": max_archive_bytes,
                "archives": archives, "files": records}
    with (archive_dir / "archive_manifest.json").open("x", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return manifest


def load_verified_manifest(archive_dir):
    archive_dir = Path(archive_dir)
    if archive_dir.is_symlink():
        raise ArchiveError("Archive directory symlink refused")
    manifest = json.loads((archive_dir / "archive_manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("source_files") != len(manifest["files"]):
        raise ArchiveError("Archive schema/count mismatch")
    by_archive = {name: [] for name in manifest["archives"]}
    names = set()
    for rec in manifest["files"]:
        name = relative_name(rec["path"])
        if name in names or not isinstance(rec["bytes"], int) or rec["bytes"] < 0:
            raise ArchiveError("Duplicate path or invalid size")
        names.add(name)
        if rec["storage"] == "zip":
            relative_name(rec["member"])
            if rec["member"] != name or rec["archive"] not in by_archive:
                raise ArchiveError("ZIP member binding mismatch")
            by_archive[rec["archive"]].append(rec)
        elif rec["storage"] == "gzip_parts":
            expected_offset = 0
            for part in rec["parts"]:
                if part["archive"] not in by_archive or part["offset"] != expected_offset or part["bytes"] <= 0:
                    raise ArchiveError("Gzip part binding mismatch")
                expected_offset += part["bytes"]
                by_archive[part["archive"]].append(rec)
            if expected_offset != rec["bytes"]:
                raise ArchiveError("Gzip total mismatch")
        else:
            raise ArchiveError("Unknown archive format")
    if any(str(parent) in names for name in names for parent in PurePosixPath(name).parents):
        raise ArchiveError("Manifest file paths have conflicting ancestors")
    for name, proof in manifest["archives"].items():
        if relative_name(name) != Path(name).name or not by_archive[name]:
            raise ArchiveError("Archive basename or references invalid")
        p = safe_target(archive_dir, name)
        if p.stat().st_size != proof["bytes"] or digest(p) != proof["sha256"]:
            raise ArchiveError("Archive bytes changed")
        if proof["bytes"] > manifest["max_archive_bytes"]:
            raise ArchiveError("Archive size cap exceeded")
        if proof["format"] == "zip":
            with zipfile.ZipFile(p) as z:
                info = z.infolist()
                expected = {r["member"]: r for r in by_archive[name]}
                if len(info) != len(expected) or set(z.namelist()) != set(expected):
                    raise ArchiveError("Unexpected ZIP member")
                for item in info:
                    relative_name(item.filename)
                    if (item.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ArchiveError("ZIP symlink refused")
                    if item.file_size != expected[item.filename]["bytes"]:
                        raise ArchiveError("ZIP decoded size differs")
        elif proof["format"] != "gzip" or len(by_archive[name]) != 1:
            raise ArchiveError("Invalid gzip binding")
    return manifest


def restore_archive(archive_dir, target_root):
    archive_dir, target_root = Path(archive_dir), Path(target_root)
    manifest = load_verified_manifest(archive_dir)
    # Refuse every conflict before creating any result file.
    existing = 0
    for rec in manifest["files"]:
        p = safe_target(target_root, rec["path"])
        if p.exists():
            if not p.is_file() or p.stat().st_size != rec["bytes"] or digest(p) != rec["sha256"]:
                raise ArchiveError("Existing target differs; nothing overwritten")
            existing += 1
    restored = 0
    for rec in manifest["files"]:
        p = safe_target(target_root, rec["path"])
        if p.exists():
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=p.parent, prefix=".evidence-restore-", delete=False) as dst:
            tmp = Path(dst.name)
            try:
                if rec["storage"] == "zip":
                    with zipfile.ZipFile(archive_dir / rec["archive"]) as z, z.open(rec["member"]) as src:
                        count, sha = copy_hashed(src, dst, rec["bytes"])
                else:
                    total = hashlib.sha256()
                    count = 0
                    for part in rec["parts"]:
                        with gzip.open(archive_dir / part["archive"], "rb") as src:
                            n = 0
                            h = hashlib.sha256()
                            for block in iter(lambda: src.read(BUFFER), b""):
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
                if count != rec["bytes"] or sha != rec["sha256"]:
                    raise ArchiveError("Restored bytes differ")
                dst.flush()
                os.fsync(dst.fileno())
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        try:
            safe_target(target_root, rec["path"])
            os.link(tmp, p)  # Exclusive publish: a concurrently-created target is never replaced.
        finally:
            tmp.unlink(missing_ok=True)
        restored += 1
    return {"schema": SCHEMA, "source_files": manifest["source_files"],
            "restored": restored, "already_matching": existing,
            "manifest_sha256": digest(archive_dir / "archive_manifest.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["archive", "restore", "verify"])
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("reports/public-thz-20261031-v1/ginseng-benchmark"))
    args = parser.parse_args()
    expected = Path("reports/public-thz-20261031-v1/ginseng-benchmark")
    if args.run_root.resolve() != expected.resolve():
        parser.error("CLI is restricted to this run's ginseng CC-BY evidence; use import API for synthetic tests")
    if args.stage == "archive":
        result = create_archive(args.run_root, args.archive_dir)
        result = {"source_files": result["source_files"], "source_bytes": result["source_bytes"],
                  "archive_files": len(result["archives"]),
                  "manifest_sha256": digest(args.archive_dir / "archive_manifest.json")}
    elif args.stage == "restore":
        result = restore_archive(args.archive_dir, args.run_root)
    else:
        m = load_verified_manifest(args.archive_dir)
        result = {"source_files": m["source_files"], "archive_files": len(m["archives"]),
                  "manifest_sha256": digest(args.archive_dir / "archive_manifest.json")}
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
