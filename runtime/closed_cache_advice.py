"""Per-file cache advice for sealed outputs and immutable Git objects.

No data bytes are written, no files are removed, and no process is signalled.
Large payloads are never read to hash them: only the small supervisor receipts
are SHA-checked before their completed outputs become eligible for advice.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

OLD = Path('/workspace/research4-relaunch/data/research/species_relaunch_20261108_v2')
OLD_OBJECTS = Path('/workspace/research4/.git/objects')
BACKFILL = Path('/workspace/research4-backfill-20261108-001')
MEMORY_CURRENT = Path('/sys/fs/cgroup/memory.current')
MIN_BYTES = 512 * 1024
SMALL_JSON_CAP = 2 * 1024 * 1024
STATE_CAP = 48 * 1024 * 1024
SHA256 = re.compile(r'[0-9a-f]{64}')
HEX2 = re.compile(r'[0-9a-f]{2}')
HEX38 = re.compile(r'[0-9a-f]{38}')
PACK = re.compile(r'pack-[0-9a-f]{40}\.(?:pack|idx)')
PART = re.compile(r'closed-results\.tar\.gz\.part[0-9]{3}')
JOB = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.-]*')


def _small_bytes(path, cap):
    """Bounded metadata read, rejecting a replaced leaf symlink."""
    if path.is_symlink():
        raise ValueError('Metadata symlink')
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > cap:
            raise ValueError('Metadata is not a bounded regular file')
        chunks, count = [], 0
        while True:
            block = os.read(fd, min(1024 * 1024, cap + 1 - count))
            if not block:
                break
            count += len(block)
            if count > cap:
                raise ValueError('Metadata grew beyond cap')
            chunks.append(block)
        return b''.join(chunks)
    finally:
        os.close(fd)


def _json(path, cap=SMALL_JSON_CAP):
    return json.loads(_small_bytes(path, cap).decode('utf-8'))


def _memory(errors):
    try:
        return int(MEMORY_CURRENT.read_text().strip())
    except (OSError, ValueError):
        errors['memory_read'] += 1
        return None


def _contained_no_links(path, root):
    """Reject links in every component, including the approved root itself."""
    if not path.is_absolute() or '..' in path.parts:
        return False
    try:
        pieces = path.relative_to(root).parts
    except ValueError:
        return False
    cursor = root
    if any(parent.is_symlink() for parent in (root, *root.parents)):
        return False
    for piece in pieces:
        cursor = cursor / piece
        if cursor.is_symlink():
            return False
    return path.resolve().is_relative_to(root.resolve())


def _git_candidates(root, files, errors):
    """Only immutable, final Git object names; never refs, locks or temporaries."""
    if any(parent.is_symlink() for parent in (root, *root.parents)) or not root.is_dir():
        return
    try:
        entries = list(root.iterdir())
    except OSError:
        errors['git_listing'] += 1
        return
    for directory in entries:
        if directory.is_symlink() or not directory.is_dir():
            continue
        loose = bool(HEX2.fullmatch(directory.name))
        packed = directory.name == 'pack'
        if not loose and not packed:
            continue
        try:
            for path in directory.iterdir():
                valid = HEX38.fullmatch(path.name) if loose else PACK.fullmatch(path.name)
                if valid and not path.is_symlink():
                    files[path] = root
        except OSError:
            # A concurrent legitimate Git GC may remove or replace a pack.
            errors['git_listing'] += 1


def _sealed_outputs(files, errors):
    verified = 0
    try:
        if not _contained_no_links(OLD / 'control/state.json', OLD):
            raise ValueError('Unapproved state metadata path')
        state = _json(OLD / 'control/state.json', STATE_CAP)
        jobs = state.get('jobs', {})
        if not isinstance(jobs, dict):
            raise ValueError('Invalid job state')
    except (OSError, ValueError, TypeError, UnicodeError):
        errors['state_read'] += 1
        return verified
    for name, record in jobs.items():
        if not isinstance(record, dict) or record.get('status') != 'completed':
            continue
        if not isinstance(name, str) or not JOB.fullmatch(name):
            errors['receipt_identity'] += 1
            continue
        expected = record.get('receipt_sha256')
        if not isinstance(expected, str) or not SHA256.fullmatch(expected):
            errors['receipt_identity'] += 1
            continue
        path = OLD / 'control/receipts' / (name + '.json')
        try:
            if not _contained_no_links(path, OLD):
                raise ValueError('Unapproved receipt path')
            payload = _small_bytes(path, SMALL_JSON_CAP)
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError('Receipt hash mismatch')
            receipt = json.loads(payload.decode('utf-8'))
            outputs = receipt.get('output_hashes', {})
            if not isinstance(outputs, dict):
                raise ValueError('Invalid output inventory')
            verified += 1
            for filename, checksum in outputs.items():
                if not isinstance(filename, str) or not isinstance(checksum, str) or not SHA256.fullmatch(checksum):
                    errors['output_inventory'] += 1
                    continue
                child = Path(filename)
                if child.suffix in ('.json', '.gz') and _contained_no_links(child, OLD):
                    files[child] = OLD
        except (OSError, ValueError, TypeError, UnicodeError):
            errors['receipt_read_or_seal'] += 1
    return verified


def _verified_backfill(files, errors):
    result_path = BACKFILL / 'backfill_result.json'
    if not result_path.exists():
        return
    try:
        if not _contained_no_links(result_path, BACKFILL):
            raise ValueError('Unapproved backfill metadata path')
        if _json(result_path).get('status') != 'DONE_REMOTE_BACKFILL_VERIFIED':
            return
        if not _contained_no_links(BACKFILL / 'manifest.json', BACKFILL):
            raise ValueError('Unapproved backfill manifest path')
        manifest = _json(BACKFILL / 'manifest.json', 8 * 1024 * 1024)
        for entry in manifest.get('parts', []):
            relative = entry.get('path', '')
            if not isinstance(relative, str):
                continue
            part = Path(relative)
            if len(part.parts) == 2 and part.parts[0] == 'archives' and PART.fullmatch(part.name):
                path = BACKFILL / part
                if _contained_no_links(path, BACKFILL):
                    files[path] = BACKFILL
        _git_candidates(BACKFILL / '.git/objects', files, errors)
    except (OSError, ValueError, TypeError, AttributeError, UnicodeError):
        errors['backfill_metadata'] += 1


def advise_closed():
    """Return measured, bounded metadata about best-effort read-only page advice."""
    started = time.process_time()
    errors = Counter()
    before = _memory(errors)
    files = {}
    verified = _sealed_outputs(files, errors)
    _git_candidates(OLD_OBJECTS, files, errors)
    _verified_backfill(files, errors)
    count, size, changed = 0, 0, 0
    for path, root in sorted(files.items()):
        fd = None
        try:
            if not _contained_no_links(path, root):
                continue
            prior = path.lstat()
            if not stat.S_ISREG(prior.st_mode) or prior.st_size < MIN_BYTES:
                continue
            fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
            opened = os.fstat(fd)
            identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
            if not stat.S_ISREG(opened.st_mode) or identity(prior) != identity(opened):
                errors['path_changed_before_advice'] += 1
                continue
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            after_fd = os.fstat(fd)
            after_path = path.lstat()
            if identity(opened) != identity(after_fd) or identity(opened) != identity(after_path):
                changed += 1
                errors['path_changed_during_advice'] += 1
                continue
            count += 1
            size += opened.st_size
        except (OSError, ValueError):
            # GC/deletion races and unsupported filesystems never stop science.
            errors['file_advice'] += 1
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    errors['close'] += 1
    after = _memory(errors)
    return {'status': 'READ_ONLY_CLOSED_CACHE_ADVICE', 'files': count,
            'bytes': size, 'file_bytes_advised': size,
            'memory_before_bytes': before, 'memory_after_bytes': after,
            'actual_cpu_seconds': time.process_time() - started,
            'errors': sum(errors.values()), 'error_counts': dict(errors),
            'verified_completed_receipts': verified, 'observed_metadata_changes': changed,
            'payload_bytes_read': False, 'file_bytes_written': False,
            'global_drop_caches_used': False, 'signals_sent': False}
