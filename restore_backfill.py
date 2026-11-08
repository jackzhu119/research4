#!/usr/bin/env python3
"""Verify or restore sealed research files from a split continuous tar.gz.

Only the Python standard library is needed. Archive payloads are streamed in
1 MiB blocks; no scientific software, model fitting, or Git operation occurs.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile

CHUNK = 1024 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_PART_BYTES = 32 * 1024 * 1024
SCHEMA = "research4-closed-results-backfill-v1"


class VerificationError(ValueError):
    pass


def need(value, message):
    if not value:
        raise VerificationError(message)


def safe_name(value):
    need(isinstance(value, str) and value and "\x00" not in value
         and "\\" not in value and ":" not in value, "Unsafe archive/part path")
    p = PurePosixPath(value)
    need(not p.is_absolute() and p.as_posix() == value
         and all(x not in ("", ".", "..") for x in p.parts), "Unsafe archive/part path")
    return value


def checked_record(row):
    need(isinstance(row, dict), "Manifest entry must be an object")
    name = safe_name(row.get("path"))
    checksum, size = row.get("sha256"), row.get("bytes")
    need(isinstance(checksum, str) and re.fullmatch(r"[0-9a-f]{64}", checksum), "Invalid SHA-256")
    need(type(size) is int and size >= 0, "Invalid byte count")
    return name, checksum, size


def sha_stream(handle):
    digest, count = hashlib.sha256(), 0
    while True:
        block = handle.read(CHUNK)
        if not block:
            break
        digest.update(block)
        count += len(block)
    return digest.hexdigest(), count


def load_manifest(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "Manifest must be a regular non-symlink file")
    need(path.stat().st_size <= MAX_MANIFEST_BYTES, "Manifest exceeds bounded verification size")
    with path.open("rb") as handle:
        manifest_sha, _ = sha_stream(handle)
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    need(isinstance(data, dict) and data.get("schema") == SCHEMA, "Unsupported manifest schema")
    need(isinstance(data.get("files"), list) and isinstance(data.get("parts"), list), "Files/parts arrays required")
    files = {}
    for row in data.pop("files"):
        name, checksum, size = checked_record(row)
        need(name not in files, "Duplicate manifest file: " + name)
        files[name] = (checksum, size)
    need(files, "Empty file inventory")
    for name in files:
        need(all(str(parent) not in files for parent in PurePosixPath(name).parents
                 if str(parent) != "."), "File/directory inventory conflict")
    parts, names = [], set()
    for row in data.pop("parts"):
        name, checksum, size = checked_record(row)
        need(name not in names and 0 < size <= MAX_PART_BYTES, "Duplicate, empty or oversized part")
        names.add(name)
        parts.append((name, checksum, size))
    need(parts, "No archive parts")
    need(all(size == MAX_PART_BYTES for _, _, size in parts[:-1]), "Nonfinal part is not exactly32MiB")
    return files, parts, manifest_sha


def part_path(root, name):
    cursor = root
    for piece in PurePosixPath(name).parts:
        cursor /= piece
        need(not cursor.is_symlink(), "Symlink in part path")
    need(cursor.resolve().is_relative_to(root) and stat.S_ISREG(cursor.stat().st_mode), "Part must be a contained regular file")
    return cursor


class PartsReader(io.RawIOBase):
    """A bounded, sequential reader over files that form one gzip stream."""
    def __init__(self, paths):
        super().__init__()
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        need(not self.closed, "Archive reader is closed")
        target = memoryview(buffer)
        if not target:
            return 0
        while True:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    return 0
                self.current = path.open("rb")
            count = self.current.readinto(target)
            if count:
                return count
            self.current.close()
            self.current = None

    def close(self):
        if self.current is not None:
            self.current.close()
            self.current = None
        super().close()


def prepare_output(path):
    if path is None:
        return None
    raw = Path(path).absolute()
    need(not raw.is_symlink(), "Output cannot be a symlink")
    if raw.exists():
        need(raw.is_dir() and not any(raw.iterdir()), "Output directory must be empty")
    else:
        raw.mkdir(mode=0o700, parents=True)
    return raw.resolve()


def exclusive_target(root, name):
    target = root / name
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    cursor = target.parent
    while cursor != root:
        need(not cursor.is_symlink(), "Symlink in extraction target")
        cursor = cursor.parent
    need(target.parent.resolve().is_relative_to(root), "Extraction escaped output root")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    return os.fdopen(os.open(target, flags, 0o600), "wb")


def verify(manifest, *, output=None):
    manifest = Path(manifest).absolute()
    files, parts, manifest_sha = load_manifest(manifest)
    root = manifest.parent.resolve()
    paths = []
    # Validate every transport part before creating any extraction output.
    for name, checksum, size in parts:
        path = part_path(root, name)
        with path.open("rb") as handle:
            observed_sha, observed_size = sha_stream(handle)
        need(observed_sha == checksum and observed_size == size, "Part checksum/size mismatch: " + name)
        paths.append(path)
    destination = prepare_output(output)
    seen, total = set(), 0
    with PartsReader(paths) as raw, io.BufferedReader(raw, buffer_size=CHUNK) as stream, gzip.GzipFile(fileobj=stream, mode="rb") as uncompressed:
        # Continue across zero padding so concatenated tar members cannot be
        # hidden after the first conventional end-of-archive marker.
        with tarfile.open(fileobj=uncompressed, mode="r|", ignore_zeros=True) as archive:
            for member in archive:
                name = safe_name(member.name)
                need(member.isfile() and member.type in (tarfile.REGTYPE, tarfile.AREGTYPE), "Nonregular archive member: " + name)
                need(member.mode & 0o7777 == 0o600 and member.mtime == 0, "Unexpected member permissions/time: " + name)
                need(name in files and name not in seen, "Unknown/duplicate archive member: " + name)
                expected_sha, expected_size = files[name]
                need(member.size == expected_size, "Declared member size mismatch: " + name)
                handle = archive.extractfile(member)
                need(handle is not None, "Missing archive member data")
                sink = exclusive_target(destination, name) if destination is not None else None
                digest, count = hashlib.sha256(), 0
                try:
                    with handle:
                        while True:
                            block = handle.read(CHUNK)
                            if not block:
                                break
                            count += len(block)
                            need(count <= expected_size, "Oversized member: " + name)
                            digest.update(block)
                            if sink is not None:
                                sink.write(block)
                finally:
                    if sink is not None:
                        sink.close()
                need(count == expected_size and digest.hexdigest() == expected_sha, "Member checksum/size mismatch: " + name)
                seen.add(name)
                total += count
        # tar EOF can precede the gzip trailer. Drain to check CRC and ISIZE.
        while uncompressed.read(CHUNK):
            pass
    need(seen == files.keys(), "Archive is missing manifest members")
    return {"status": "PASS", "mode": "VERIFY_ONLY" if destination is None else "RESTORED",
            "manifest_sha256": manifest_sha, "files": len(seen), "uncompressed_bytes": total,
            "parts": len(parts), "part_bytes": sum(size for _, _, size in parts),
            "gzip_crc_checked": True, "output": str(destination) if destination is not None else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.report is not None:
            report = args.report.absolute()
            need(not report.exists() and not report.is_symlink(), "Report must not already exist")
            need(report.resolve() != args.manifest.resolve(), "Report cannot overwrite the manifest")
            if args.output is not None:
                need(not report.resolve().is_relative_to(args.output.resolve()), "Report must be separate from extraction output")
        result = verify(args.manifest, output=args.output)
        if args.report is not None:
            # A successful verifier must never replace existing evidence.
            with args.report.open("x", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
    except Exception as error:
        print(json.dumps({"status": "FAIL", "error_type": type(error).__name__, "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
