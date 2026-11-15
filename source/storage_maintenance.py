#!/usr/bin/env python3
"""Reversible, completed-job-only storage maintenance; never schedules or fits models.

Diagnostics archives are fully CRC/SHA verified and their manifest fsynced before
any original is removed. Model recoding is delegated to model_artifact_codec.
--start stays in this process; the caller controls backgrounding. No children.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import gzip
import hashlib
import importlib
import json
import os
for _thread_var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_thread_var] = '1'
from pathlib import Path, PurePosixPath
import signal
import sqlite3
import stat
import sys
import tarfile
import threading
import time
import uuid

DIAGNOSTIC_TOP = ('rank_cache', 'inner_scores', 'candidate_grid.json', 'inner_candidate_means.json')
PROTECTED = ('result.json', 'predictions.csv', 'fitted_artifacts.npz', 'selection.json',
             'job_frozen.json', 'split_frozen.json', 'checkpoint.json', 'cpu_receipt.json',
             'heartbeat.json', 'job.json')
ARCHIVE = 'diagnostics.tar.gz'
MANIFEST = 'diagnostics_archive_manifest.json'
SPLIT_NAME = 'split_frozen.json'
SPLIT_GZIP = 'split_frozen.json.gz'
SPLIT_MANIFEST = 'split_frozen.archive_manifest.json'
COMPONENT_POLICY_VERSION = 2
STOP = threading.Event()


class UnsafeJob(RuntimeError): pass
class StopRequested(RuntimeError): pass


def utc():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with tmp.open('w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if tmp.exists(): tmp.unlink()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path, pulse=None):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk: break
            h.update(chunk)
            if pulse: pulse()
    return h.hexdigest()


def process_identity():
    raw = Path('/proc/self/stat').read_text()
    fields = raw[raw.rfind(')') + 2:].split()
    return {'pid': os.getpid(), 'start_ticks': int(fields[19]),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def lifecycle_stop_reason(root):
    root = Path(root)
    if (root / 'STOP').exists(): return 'root_STOP_requested'
    try: status = read_json(root / 'status.json')
    except (OSError, ValueError): return None
    state = status.get('state')
    if state in ('stopped_by_user', 'fatal_configuration_error'): return state
    if state == 'budget_exhausted':
        try: pending = read_json(root / 'pending_budget_handoff.json')
        except (OSError, ValueError): pending = {}
        if pending.get('state') in ('waiting_natural_boundary', 'activating') and float(pending.get('cumulative_authorized_cpu_seconds', 0)) > float(status.get('cpu_budget_seconds', 0)):
            return None  # Authorized natural boundary; keep maintenance alive.
        return 'final_resource_budget_exhausted'
    return None


class Ledger:
    """Own process cumulative CPU high-water, including import/startup CPU."""
    def __init__(self, root):
        self.root = Path(root); self.ident = process_identity()
        self.key = f"{self.ident['boot_id']}:{self.ident['pid']}:{self.ident['start_ticks']}"
        self.last = 0.0; self.closed = False; self.last_stop_check = 0.0; self.stop_reason = None
        self.details = {'jobs_checked': 0, 'jobs_compacted': 0, 'errors': 0}
        self.update(force=True)

    def update(self, force=False, closed=False, reason=None):
        if not force and time.monotonic() - self.last < .75: return
        self.last = time.monotonic()
        cpu = time.process_time()  # deliberately no baseline subtraction
        path = self.root / 'storage_maintenance_ledger.json'
        with (self.root / 'storage_maintenance_ledger.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            old = read_json(path) if path.exists() else {'schema': 'storage-self-cpu-v1', 'accounts': {}}
            if not isinstance(old.get('accounts'), dict): raise RuntimeError('Invalid storage CPU ledger; refusing reset')
            previous = old['accounts'].get(self.key, {})
            old['accounts'][self.key] = {**previous, **self.ident, **self.details,
                'cpu_seconds': max(float(previous.get('cpu_seconds', 0)), cpu),
                'closed': bool(closed or previous.get('closed', False)),
                'updated_utc': utc(), 'accounting': 'process_time cumulative, imports included; no children'}
            if reason: old['accounts'][self.key]['reason'] = reason
            old['updated_utc'] = utc()
            atomic_json(path, old)
        self.closed = closed

    def pulse(self):
        self.update()
        if time.monotonic() - self.last_stop_check >= .75:
            self.last_stop_check = time.monotonic()
            self.stop_reason = lifecycle_stop_reason(self.root)
            if self.stop_reason: STOP.set()
        if STOP.is_set(): raise StopRequested(self.stop_reason or 'Graceful signal stop requested')


def connect_ro(root):
    c = sqlite3.connect((Path(root) / 'queue.sqlite3').resolve().as_uri() + '?mode=ro', uri=True, timeout=10)
    c.row_factory = sqlite3.Row
    c.execute('PRAGMA query_only=ON')
    return c


def eligible_job(root, job_id):
    if not job_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.' for c in job_id):
        raise UnsafeJob('Unsafe job ID')
    root = Path(root).resolve(); job = root / 'jobs' / job_id
    if job.is_symlink() or job.resolve().parent != (root / 'jobs').resolve(): raise UnsafeJob('Unsafe job directory')
    with contextlib.closing(connect_ro(root)) as c:
        r = c.execute('SELECT status,result_path FROM jobs WHERE id=?', (job_id,)).fetchone()
        if not r or r['status'] != 'completed': raise UnsafeJob('Job is not SQLite completed')
        active = c.execute("SELECT COUNT(*) FROM attempts WHERE job_id=? AND status IN('running','launching')", (job_id,)).fetchone()[0]
        if active: raise UnsafeJob('Job has an active attempt')
    result_path = job / 'result.json'
    if result_path.is_symlink() or not result_path.is_file(): raise UnsafeJob('Missing regular result')
    if r['result_path'] and Path(r['result_path']).resolve() != result_path.resolve(): raise UnsafeJob('Result path differs from SQLite')
    result = read_json(result_path)
    if result.get('status') != 'complete' or result.get('complete') is not True:
        raise UnsafeJob('Result is not complete')
    if result.get('testing_only') or result.get('job', {}).get('testing'):
        raise UnsafeJob('Testing result is excluded')
    if result.get('job', {}).get('id', job_id) != job_id: raise UnsafeJob('Result identity differs')
    return job, digest(result_path)


def source_inventory(job, pulse=None):
    items = []
    for name in DIAGNOSTIC_TOP:
        path = job / name
        if path.is_symlink(): raise UnsafeJob('Symlink diagnostic entry')
        if not path.exists(): continue
        if path.is_file(): paths = [path]
        elif path.is_dir():
            paths = []
            for parent, dirs, files in os.walk(path, followlinks=False):
                for x in dirs:
                    if (Path(parent) / x).is_symlink(): raise UnsafeJob('Symlink diagnostic directory')
                paths.extend(Path(parent) / x for x in files)
        else: raise UnsafeJob('Nonregular diagnostic entry')
        for p in sorted(paths):
            st = p.lstat()
            if not stat.S_ISREG(st.st_mode): raise UnsafeJob('Nonregular diagnostic file')
            rel = p.relative_to(job).as_posix()
            if PurePosixPath(rel).parts[0] not in DIAGNOSTIC_TOP: raise UnsafeJob('Unexpected archive member')
            before = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
            h = digest(p, pulse)
            now = p.stat()
            if before != (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns): raise UnsafeJob('Diagnostic changed while hashing')
            items.append({'name': rel, 'size': st.st_size, 'sha256': h,
                          'mode': stat.S_IMODE(st.st_mode), 'mtime_ns': st.st_mtime_ns,
                          'allocated_bytes': st.st_blocks * 512})
    return sorted(items, key=lambda x: x['name'])


def verify_archive(path, members, pulse=None):
    """Explicitly drain gzip to EOF (including CRC), then verify each tar member."""
    with gzip.open(path, 'rb') as f:
        while f.read(1024 * 1024):
            if pulse: pulse()
    wanted = {x['name']: x for x in members}
    if len(wanted) != len(members): raise UnsafeJob('Duplicate manifest member')
    seen = set()
    with tarfile.open(path, 'r:gz') as tar:
        for member in tar:
            name = member.name
            if not member.isfile() or name not in wanted or name in seen:
                raise UnsafeJob('Unexpected/nonregular/duplicate tar member')
            if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
                raise UnsafeJob('Unsafe tar member path')
            expected = wanted[name]
            if member.size != expected['size']: raise UnsafeJob('Tar member size mismatch')
            h = hashlib.sha256(); total = 0
            with tar.extractfile(member) as stream:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk: break
                    h.update(chunk); total += len(chunk)
                    if pulse: pulse()
            if total != expected['size'] or h.hexdigest() != expected['sha256']:
                raise UnsafeJob('Tar member SHA256 mismatch')
            seen.add(name)
    if seen != set(wanted): raise UnsafeJob('Missing archive members')
    return digest(path, pulse)


def warn(root, reason, **details):
    atomic_json(Path(root) / 'STORAGE_WARNING.json', {'utc': utc(), 'reason': reason,
                 'scope': 'storage only; no training scheduling changes', **details})


def compact_diagnostics(root, job_id, pulse=None):
    root = Path(root).resolve(); job, result_sha = eligible_job(root, job_id)
    archive = job / ARCHIVE; manifest_path = job / MANIFEST
    if archive.is_symlink() or manifest_path.is_symlink(): raise UnsafeJob('Symlink archive/manifest')
    if archive.exists() and not archive.is_file(): raise UnsafeJob('Nonregular archive')
    if manifest_path.exists():
        m = read_json(manifest_path)
        if m.get('job_id') != job_id or m.get('result_sha256') != result_sha:
            raise UnsafeJob('Existing archive manifest identity differs')
        if verify_archive(archive, m['members'], pulse) != m['archive_sha256']:
            raise UnsafeJob('Existing archive SHA256 differs')
        if m.get('cleanup_complete') and not any((job / x).exists() for x in DIAGNOSTIC_TOP):
            return {'status': 'already_compact', 'job_id': job_id, 'raw_bytes': m['source_bytes'], 'archive_bytes': m['archive_bytes']}
    else:
        members = source_inventory(job, pulse)
        if not members: return {'status': 'no_diagnostics', 'job_id': job_id, 'raw_bytes': 0, 'archive_bytes': 0}
        raw = sum(x['size'] for x in members)
        free = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
        if free < 2**30: warn(root, 'free_space_below_1GiB', free_bytes=free, job_id=job_id)
        if free < raw * 1.02 + 8 * 1024 * 1024:
            raise UnsafeJob('Insufficient temporary space; originals preserved')
        tmp = archive.with_name(archive.name + '.tmp-' + uuid.uuid4().hex)
        try:
            if not archive.exists():
                with tarfile.open(tmp, 'w:gz', compresslevel=1) as tar:
                    for x in members:
                        if pulse: pulse()
                        tar.add(job / x['name'], arcname=x['name'], recursive=False)
                with tmp.open('rb') as f: os.fsync(f.fileno())
                verify_archive(tmp, members, pulse)
                os.replace(tmp, archive)
            archive_sha = verify_archive(archive, members, pulse)
        finally:
            if tmp.exists(): tmp.unlink()
        # Recheck eligibility and every original before manifest commit/deletion.
        _, current = eligible_job(root, job_id)
        if current != result_sha or source_inventory(job, pulse) != members:
            raise UnsafeJob('Job changed during compression; originals preserved')
        m = {'schema': 'completed-job-diagnostics-archive-v1', 'job_id': job_id,
             'result_sha256': result_sha, 'archive': ARCHIVE,
             'archive_sha256': archive_sha, 'archive_bytes': archive.stat().st_size,
             'source_bytes': raw, 'source_allocated_bytes': sum(x['allocated_bytes'] for x in members),
             'members': members, 'protected_paths': list(PROTECTED),
             'verification': 'gzip EOF CRC + exact tar member list + every member SHA256',
             'verified_at_utc': utc(), 'cleanup_complete': False}
        atomic_json(manifest_path, m)
    # Verify ALL remaining originals before deleting ANY of them.
    for x in m['members']:
        p = job / x['name']
        if p.is_symlink(): raise UnsafeJob('Remaining diagnostic became a symlink')
        if p.exists() and (not p.is_file() or p.stat().st_size != x['size'] or digest(p, pulse) != x['sha256']):
            raise UnsafeJob('Remaining diagnostic differs; originals preserved')
    _, current = eligible_job(root, job_id)
    if current != result_sha: raise UnsafeJob('Result changed before cleanup')
    with contextlib.closing(connect_ro(root)) as current_db:
        for x in m['members']:
            if pulse: pulse()
            fresh = current_db.execute('SELECT status FROM jobs WHERE id=?', (job_id,)).fetchone()
            if not fresh or fresh['status'] != 'completed': raise UnsafeJob('Job left completed state; cleanup stopped')
            p = job / x['name']
            if p.exists(): p.unlink()
    for name in ('rank_cache', 'inner_scores'):
        p = job / name
        if p.is_dir():
            for parent, dirs, files in os.walk(p, topdown=False):
                if not list(Path(parent).iterdir()): Path(parent).rmdir()
    m.update(cleanup_complete=True, cleaned_at_utc=utc())
    atomic_json(manifest_path, m)
    return {'status': 'compacted', 'job_id': job_id, 'raw_bytes': m['source_bytes'],
            'archive_bytes': m['archive_bytes'], 'archive_manifest': str(manifest_path)}


def restore_diagnostics(root, job_id, pulse=None):
    """Explicit recovery helper. Pause maintenance before restoring for requeue."""
    job, result_sha = eligible_job(root, job_id); m = read_json(job / MANIFEST)
    if m['job_id'] != job_id or m['result_sha256'] != result_sha: raise UnsafeJob('Restore identity mismatch')
    if verify_archive(job / ARCHIVE, m['members'], pulse) != m['archive_sha256']: raise UnsafeJob('Restore archive hash mismatch')
    expected = {x['name']: x for x in m['members']}
    with tarfile.open(job / ARCHIVE, 'r:gz') as tar:
        for item in tar:
            x = expected[item.name]; p = job / item.name
            if p.is_symlink() or p.resolve().is_relative_to(job.resolve()) is False: raise UnsafeJob('Unsafe restore target')
            if p.exists():
                if digest(p, pulse) != x['sha256']: raise UnsafeJob('Refusing to overwrite a different original')
                continue
            p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_name(p.name + '.restore-' + uuid.uuid4().hex)
            try:
                with tar.extractfile(item) as src, tmp.open('wb') as dest:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk: break
                        dest.write(chunk)
                        if pulse: pulse()
                    dest.flush(); os.fsync(dest.fileno())
                if digest(tmp, pulse) != x['sha256']: raise UnsafeJob('Restored member hash mismatch')
                os.chmod(tmp, x['mode'] & 0o777); os.replace(tmp, p)
                os.utime(p, ns=(x['mtime_ns'], x['mtime_ns']))
            finally:
                if tmp.exists(): tmp.unlink()
    return {'status': 'restored', 'job_id': job_id, 'members': len(expected)}


def verify_split_gzip(path, expected_sha, expected_size, pulse=None):
    h = hashlib.sha256(); size = 0
    with gzip.open(path, 'rb') as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk: break
            h.update(chunk); size += len(chunk)
            if pulse: pulse()
    if size != expected_size or h.hexdigest() != expected_sha:
        raise UnsafeJob('Split gzip decompressed SHA/size mismatch')
    return digest(path, pulse)


def compact_completed_split(root, job_id, pulse=None):
    """Separate v2 component; never changes any existing diagnostics tarball."""
    root = Path(root).resolve(); job, result_sha = eligible_job(root, job_id)
    source = job / SPLIT_NAME; encoded = job / SPLIT_GZIP; manifest = job / SPLIT_MANIFEST
    if any(p.is_symlink() for p in (source, encoded, manifest)):
        raise UnsafeJob('Symlink split storage path')
    if manifest.exists():
        m = read_json(manifest)
        if m.get('job_id') != job_id or m.get('result_sha256') != result_sha:
            raise UnsafeJob('Split manifest identity mismatch')
        if verify_split_gzip(encoded, m['source_sha256'], m['source_bytes'], pulse) != m['gzip_sha256']:
            raise UnsafeJob('Split gzip container SHA mismatch')
        if not source.exists():
            if not m.get('cleanup_complete'):
                m.update(cleanup_complete=True, cleaned_at_utc=utc())
                atomic_json(manifest, m)
            return {'status': 'already_compact', 'raw_bytes': m['source_bytes'],
                    'encoded_bytes': m['gzip_bytes'], 'manifest': str(manifest)}
    else:
        if not source.is_file(): raise UnsafeJob('Completed split source/manifest unavailable')
        before = source.stat(); source_sha = digest(source, pulse)
        if source.stat().st_size != before.st_size or source.stat().st_mtime_ns != before.st_mtime_ns:
            raise UnsafeJob('Split changed during source hashing')
        if encoded.exists() and not encoded.is_file(): raise UnsafeJob('Nonregular split gzip')
        temporary = encoded.with_name(encoded.name + '.tmp-' + uuid.uuid4().hex)
        try:
            if not encoded.exists():
                with temporary.open('wb') as target:
                    with gzip.GzipFile(filename='', mode='wb', fileobj=target, compresslevel=1, mtime=0) as compressed:
                        with source.open('rb') as raw:
                            while True:
                                chunk = raw.read(1024 * 1024)
                                if not chunk: break
                                compressed.write(chunk)
                                if pulse: pulse()
                    target.flush(); os.fsync(target.fileno())
                verify_split_gzip(temporary, source_sha, before.st_size, pulse)
                os.replace(temporary, encoded)
            compressed_sha = verify_split_gzip(encoded, source_sha, before.st_size, pulse)
        finally:
            if temporary.exists(): temporary.unlink()
        _, current_result_sha = eligible_job(root, job_id)
        if current_result_sha != result_sha or digest(source, pulse) != source_sha or source.stat().st_size != before.st_size:
            raise UnsafeJob('Completed split/result changed; original retained')
        m = {'schema': 'completed-job-split-lossless-gzip-v1', 'job_id': job_id,
             'result_sha256': result_sha, 'source_name': SPLIT_NAME,
             'source_sha256': source_sha, 'source_bytes': before.st_size,
             'source_mode': stat.S_IMODE(before.st_mode), 'source_mtime_ns': before.st_mtime_ns,
             'gzip_name': SPLIT_GZIP, 'gzip_sha256': compressed_sha,
             'gzip_bytes': encoded.stat().st_size, 'verification': 'gzip EOF CRC and decompressed size/SHA256',
             'verified_at_utc': utc(), 'cleanup_complete': False}
        atomic_json(manifest, m)
    # The durable verified gzip+manifest now exist; recheck the unchanged source and role.
    if not source.is_file() or digest(source, pulse) != m['source_sha256'] or source.stat().st_size != m['source_bytes']:
        raise UnsafeJob('Split differs immediately before cleanup; retained')
    _, current_result_sha = eligible_job(root, job_id)
    if current_result_sha != result_sha: raise UnsafeJob('Result changed before split cleanup')
    if pulse: pulse()
    source.unlink()
    fd = os.open(job, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
    m.update(cleanup_complete=True, cleaned_at_utc=utc())
    atomic_json(manifest, m)
    return {'status': 'compacted', 'raw_bytes': m['source_bytes'],
            'encoded_bytes': m['gzip_bytes'], 'manifest': str(manifest)}


def restore_completed_split(root, job_id, pulse=None):
    job, result_sha = eligible_job(root, job_id)
    manifest = job / SPLIT_MANIFEST; source = job / SPLIT_NAME; encoded = job / SPLIT_GZIP
    if not manifest.exists():
        if source.is_file() and not source.is_symlink(): return {'status': 'not_compacted'}
        raise UnsafeJob('No split source or archive manifest')
    if any(p.is_symlink() for p in (manifest, source, encoded)): raise UnsafeJob('Symlink split restore path')
    m = read_json(manifest)
    if m['job_id'] != job_id or m['result_sha256'] != result_sha: raise UnsafeJob('Split restore identity mismatch')
    if verify_split_gzip(encoded, m['source_sha256'], m['source_bytes'], pulse) != m['gzip_sha256']:
        raise UnsafeJob('Split restore gzip SHA mismatch')
    if source.exists():
        if digest(source, pulse) != m['source_sha256']: raise UnsafeJob('Refusing to overwrite a different split')
        return {'status': 'already_restored'}
    temporary = source.with_name(source.name + '.restore-' + uuid.uuid4().hex)
    try:
        with gzip.open(encoded, 'rb') as compressed, temporary.open('wb') as raw:
            while True:
                chunk = compressed.read(1024 * 1024)
                if not chunk: break
                raw.write(chunk)
                if pulse: pulse()
            raw.flush(); os.fsync(raw.fileno())
        if digest(temporary, pulse) != m['source_sha256'] or temporary.stat().st_size != m['source_bytes']:
            raise UnsafeJob('Restored split SHA/size mismatch')
        eligible_job(root, job_id)
        os.chmod(temporary, m['source_mode'] & 0o777); os.replace(temporary, source)
        os.utime(source, ns=(m['source_mtime_ns'], m['source_mtime_ns']))
    finally:
        if temporary.exists(): temporary.unlink()
    return {'status': 'restored', 'sha256': m['source_sha256'], 'bytes': m['source_bytes']}


def candidates(root, state, limit):
    cursor = state.get('cursor', ['', ''])
    # Completion time (not ordinal) prevents missing slow, out-of-order jobs.
    with contextlib.closing(connect_ro(root)) as c:
        c.execute('BEGIN')
        rows = c.execute('''SELECT j.id,MAX(a.ended_utc) completed_at FROM jobs j
             JOIN attempts a ON a.job_id=j.id
             WHERE j.status='completed' AND a.status='succeeded'
             GROUP BY j.id HAVING MAX(a.ended_utc)>? OR
             (MAX(a.ended_utc)=? AND j.id>?) ORDER BY completed_at,j.id LIMIT ?''',
             (cursor[0], cursor[0], cursor[1], limit)).fetchall()
        result = [dict(x) for x in rows]; c.rollback()
    return result


def load_codec(root):
    sys.path.insert(0, str(root))
    try: return importlib.import_module('model_artifact_codec')
    finally: sys.path.pop(0)


def run_batch(root, ledger, max_jobs=100, use_codec=True):
    root = Path(root); state_path = root / 'storage_maintenance_state.json'
    state = read_json(state_path) if state_path.exists() else {'cursor': ['', ''], 'retry': {}, 'pending_model': []}
    state.setdefault('retry', {}); state.setdefault('pending_model', [])
    if int(state.get('component_policy_version', 1)) < COMPONENT_POLICY_VERSION:
        state['previous_policy_cursor'] = state.get('cursor', ['', ''])
        state['cursor'] = ['', '']  # Own maintenance replay only, never the training queue/CPU ledger.
        state['component_policy_version'] = COMPONENT_POLICY_VERSION
        state['policy_replay_started_utc'] = utc()
    free_now = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
    if free_now < 2**30: warn(root, 'free_space_below_1GiB', free_bytes=free_now)
    codec = None; codec_reason = 'diagnostics_only_requested'
    if use_codec:
        try: codec = load_codec(root)
        except ModuleNotFoundError as exc: codec_reason = 'codec_not_available: ' + str(exc)
    todo = []; seen = set()
    def due(jid):
        old = state['retry'].get(jid)
        return not old or (old['attempts'] < 3 and time.time() >= old['retry_after'])
    if codec:
        for jid in state['pending_model']:
            if due(jid): todo.append({'id': jid, 'completed_at': None}); seen.add(jid)
            if len(todo) >= max_jobs: break
    for jid in state['retry']:
        if jid not in seen and due(jid) and len(todo) < max_jobs:
            todo.append({'id': jid, 'completed_at': None}); seen.add(jid)
    todo += [r for r in candidates(root, state, max_jobs - len(todo)) if r['id'] not in seen]
    successes = 0; errors = []
    for item in todo:
        ledger.pulse(); jid = item['id']; ledger.details['jobs_checked'] += 1
        try:
            job, _ = eligible_job(root, jid)
            diagnostic = compact_diagnostics(root, jid, ledger.pulse)
            split_storage = compact_completed_split(root, jid, ledger.pulse)
            if codec:
                model = codec.encode_job(job, root=root, compact=True,
                    progress_callback=lambda _event: ledger.pulse())
                if model.get('status') == 'error': raise UnsafeJob('Model codec: ' + str(model.get('reason')))
                if model.get('status') not in ('encoded', 'compacted', 'already_compact', 'skipped'):
                    raise UnsafeJob('Unknown codec status: ' + str(model.get('status')))
                if model.get('status') == 'skipped' and model.get('reason') not in ('no_support_vectors', 'no_net_storage_saving'):
                    raise UnsafeJob('Unexpected codec skip: ' + str(model.get('reason')))
                state['pending_model'] = [x for x in state['pending_model'] if x != jid]
            else:
                model = {'status': 'pending', 'reason': codec_reason}
                if jid not in state['pending_model']: state['pending_model'].append(jid)
            # Separate component status: diagnostics completion never marks a missing codec done.
            component_path = job / MANIFEST
            if component_path.exists():
                component = read_json(component_path)
                component['component_status'] = {'diagnostics': diagnostic['status'], 'model_codec': model,
                                                  'split_storage': split_storage,
                                                  'updated_utc': utc()}
                atomic_json(component_path, component)
            if diagnostic['status'] == 'compacted' or model.get('status') == 'compacted' or split_storage.get('status') == 'compacted':
                successes += 1; ledger.details['jobs_compacted'] += 1
            state['retry'].pop(jid, None)
        except StopRequested: raise
        except Exception as exc:
            ledger.details['errors'] += 1
            previous = state['retry'].get(jid, {}).get('attempts', 0)
            state['retry'][jid] = {'attempts': previous + 1, 'retry_after': time.time() + 60,
                                   'error': type(exc).__name__ + ': ' + str(exc)}
            errors.append({'job_id': jid, 'error': state['retry'][jid]['error']})
            warn(root, 'storage_job_preserved_after_error', job_id=jid, error=state['retry'][jid]['error'])
        if item['completed_at'] is not None: state['cursor'] = [item['completed_at'], jid]
        atomic_json(state_path, state); ledger.update(force=True)
    status = {'utc': utc(), 'batch_jobs': len(todo), 'compacted_jobs': successes,
              'errors': errors, 'retry_jobs': len(state['retry']), 'pending_model_jobs': len(state['pending_model']),
              'model_codec_available': codec is not None, 'component_policy_version': COMPONENT_POLICY_VERSION, 'cursor': state['cursor'],
              'pid': os.getpid(), 'training_queue_modified': False}
    atomic_json(root / 'storage_maintenance_status.json', status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parent))
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--once', action='store_true')
    modes.add_argument('--start', action='store_true', help='Foreground daemon in this process; no fork or subprocess')
    modes.add_argument('--restore-job')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--diagnostics-only', action='store_true', help='Skip the separately verified model codec')
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 100: parser.error('--batch-size must be1..100')
    root = Path(args.root).resolve(); root.mkdir(parents=True, exist_ok=True)
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, lambda *_: STOP.set())
    try: os.setpriority(os.PRIO_PROCESS, 0, max(10, os.getpriority(os.PRIO_PROCESS, 0)))
    except (AttributeError, OSError): pass
    ledger = Ledger(root); reason = 'completed'
    lock = (root / 'storage_maintenance.lock').open('a')
    try:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            reason = 'another_maintenance_process_holds_lock'; return 0
        if args.restore_job:
            job, _ = eligible_job(root, args.restore_job)
            diagnostics = restore_diagnostics(root, args.restore_job, ledger.pulse) if (job / MANIFEST).exists() else {'status': 'not_compacted'}
            split_storage = restore_completed_split(root, args.restore_job, ledger.pulse)
            print(json.dumps({'diagnostics': diagnostics, 'split_storage': split_storage})); return 0
        while not STOP.is_set():
            result = run_batch(root, ledger, args.batch_size, not args.diagnostics_only)
            if args.once:
                print(json.dumps(result, ensure_ascii=False)); break
            wait = 1 if result['batch_jobs'] >= args.batch_size else 60
            end = time.monotonic() + wait
            while not STOP.is_set() and time.monotonic() < end:
                ledger.pulse(); STOP.wait(min(1, max(0, end - time.monotonic())))
    except StopRequested as exc: reason = str(exc)
    except Exception as exc:
        reason = type(exc).__name__ + ': ' + str(exc)
        warn(root, 'storage_maintenance_stopped_after_error', error=reason)
        return 1
    finally:
        if STOP.is_set() and reason == 'completed': reason = ledger.stop_reason or 'graceful_signal_stop'
        ledger.update(force=True, closed=True, reason=reason)
        lock.close()
    return 0


if __name__ == '__main__': sys.exit(main())
