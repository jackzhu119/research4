#!/usr/bin/env python3
"""Recover only manifest-selected corrected artifacts from the pinned archive.

The public archive is one gzip stream split into transport shards, so every
shard must be read. Four verified shards are prefetched in memory; the 12GB
archive is never written or extracted wholesale. Source bytes are authenticated
against shard SHA256 plus the final frozen per-member manifest. No model fits.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import concurrent.futures
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import tarfile
import time
import urllib.error
import urllib.request

from common import (BACKGROUND_DIR, REPORT_DIR, INPUT_SHA256, PINNED_ARCHIVE,
                    PINNED_BACKGROUND, atomic_json, load_json, now, sha256_file)

REPO = 'jackzhu119/research4'
PREFIX = 'background_study_20261110/'
SOURCE_FILES = ('protocol.json', 'study_worker.py', 'worker_spec.json',
                'extend_manifest.py', 'PRE_REGISTERED_DESIGN_zh.md',
                'model_artifact_codec.py', 'extract_current1672.py',
                'workbook_restoration_manifest.json', 'data_eligibility.json',
                'workbook_sheet_inventory.json', 'budget_guard_ledger.json',
                'budget_guard_status.json', 'status.json')
REPORT_FILES = ('completed_repeat_metrics.csv', 'source_accuracy_summary.csv',
                'repeat_completion.csv', 'progress.json', 'summary_manifest.json')
REQUIRED_METHODS = {
    ('selection', 'alpha_snv'), ('baseline', 'svm_alpha_snv'),
    ('baseline', 'lda_alpha_raw_snv'), ('spacing', 'alpha_inner_gap_k62_c2'),
    ('selection', 'alpha_snv_c2_k62'), ('coverage', 'uniform_k62'),
    ('rank_context', 'true_n'), ('rank_context', 'shuffle_n')}

def fetch(url, expected=None):
    """Finite retries apply only to identified transport failures, never scores."""
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'research4-focused-recovery/1'})
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read() if expected is None else response.read(expected['bytes'] + 1)
            if expected is not None:
                if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
                    raise ValueError('Pinned archive shard hash/size mismatch: ' + expected['name'])
            return data
        except urllib.error.HTTPError as error:
            if error.code not in (408, 429, 500, 502, 503, 504) or attempt == 4:
                raise
            retry = error.headers.get('Retry-After')
            delay = float(retry) if retry and retry.isdigit() else min(16, 2 ** attempt)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            if attempt == 4:
                raise
            delay = min(16, 2 ** attempt)
        time.sleep(delay)

def source_url(ref, path):
    return 'https://raw.githubusercontent.com/' + REPO + '/' + ref + '/' + path

def write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        stream.write(data)
    tmp.replace(path)

class ShardStream(io.RawIOBase):
    def __init__(self, shards, progress):
        self.shards = shards
        self.progress = progress
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.futures = {}
        self.index = 0
        self.offset = 0
        self.data = b''
        self.completed = []
        for i in range(min(4, len(shards))):
            self.submit(i)
    def submit(self, i):
        row = self.shards[i]
        url = source_url(PINNED_ARCHIVE, 'shards/' + row['name'])
        self.futures[i] = self.pool.submit(fetch, url, row)
    def readable(self):
        return True
    def readinto(self, buffer):
        while self.offset == len(self.data):
            if self.index == len(self.shards):
                return 0
            i = self.index
            self.data = self.futures.pop(i).result()
            self.offset = 0
            self.index += 1
            self.completed.append(self.shards[i])
            if i + 4 < len(self.shards):
                self.submit(i + 4)
            self.progress(self.index, len(self.shards))
        count = min(len(buffer), len(self.data) - self.offset)
        buffer[:count] = self.data[self.offset:self.offset + count]
        self.offset += count
        return count
    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        super().close()

def load_queue(root):
    database = root / 'snapshot_source/queue.sqlite3'
    if not database.exists():
        raise ValueError('Archive did not provide consistent queue.sqlite3 before jobs')
    connection = sqlite3.connect('file:' + str(database) + '?mode=ro', uri=True)
    rows = connection.execute('SELECT id,status,job_json FROM jobs ORDER BY ordinal').fetchall()
    connection.close()
    jobs = {job_id: json.loads(payload) for job_id, status, payload in rows if status == 'completed'}
    # Actual order comes from queue ordinal, never seed numeric order or scores.
    repeat_order = []
    folds = defaultdict(lambda: defaultdict(set))
    for job in jobs.values():
        if (job['family'], job['variant']) in REQUIRED_METHODS and int(job['mc_draws']) == 20:
            repeat = int(job['repeat'])
            if repeat not in repeat_order:
                repeat_order.append(repeat)
            folds[repeat][(job['family'], job['variant'])].add(int(job['fold']))
    common = [repeat for repeat in repeat_order if all(
        folds[repeat][method] == set(range(5)) for method in REQUIRED_METHODS)]
    return jobs, set(common[:3])

def verify_cached(root):
    """A completed flag alone is insufficient after a runner cache restore."""
    receipt = load_json(root / 'recovery_verification.json')
    for relative, expected in receipt['cache_stream_hashes'].items():
        if sha256_file(root / relative) != expected:
            raise ValueError('Recovered cache stream changed: ' + relative)
    for relative, expected in receipt.get('source_metadata_sha256', {}).items():
        if sha256_file(root / relative) != expected:
            raise ValueError('Recovered source metadata changed: ' + relative)
    with gzip.open(root / 'retained_member_hashes.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if 'local_path' not in row:
                continue
            path = root / row['local_path']
            if row.get('local_storage_transform'):
                with gzip.open(path, 'rb') as source:
                    data = source.read()
                actual = (len(data), hashlib.sha256(data).hexdigest())
            else:
                actual = (path.stat().st_size, sha256_file(path))
            if actual != (row['bytes'], row['sha256']):
                raise ValueError('Recovered cached member changed: ' + row['local_path'])
    return receipt

def publish_receipt(verification):
    task = os.environ.get('FOCUSED_TASK_ID')
    if not task:
        return
    artifact = REPORT_DIR / 'sources/recovery_verification.json'
    atomic_json(artifact, verification)
    atomic_json(REPORT_DIR / 'task_results' / (task + '.json'), {
        'status': 'REUSED_VERIFIED', 'detail': 'Pinned corrected source archive recovered and authenticated; no model fits',
        'artifacts': [str(artifact.relative_to(REPORT_DIR.parents[1]))], 'new_classifier_fits': 0,
        'new_pls_fits': 0, 'source_commit': PINNED_BACKGROUND, 'archive_commit': PINNED_ARCHIVE,
        'members_verified': verification['members_verified'],
        'cpu_seconds_recovery_this_invocation': verification['cpu_seconds_this_invocation']})

def selected_job(job):
    return ((job['family'], job['variant']) in REQUIRED_METHODS or
            job['family'] in ('selection', 'spacing', 'rank_context', 'retain_drop'))

def recover(root=BACKGROUND_DIR):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / 'recovery_status.json'
    if status_path.exists():
        old = load_json(status_path)
        if old.get('complete') and old.get('archive_commit') == PINNED_ARCHIVE:
            verification = verify_cached(root)
            if sha256_file(root / 'inputs/current1672_paired612.npz') != INPUT_SHA256:
                raise ValueError('Previously recovered dataset changed')
            publish_receipt(verification)
            return old
    start_cpu, start_wall = time.process_time(), time.monotonic()
    for filename in ('ARCHIVE_MANIFEST.json', 'MEMBER_MANIFEST_INDEX.json'):
        write_bytes(root / filename, fetch(source_url(PINNED_BACKGROUND, filename)))
    for filename in SOURCE_FILES:
        write_bytes(root / 'published_source' / filename,
                    fetch(source_url(PINNED_BACKGROUND, 'source/' + filename)))
    for filename in REPORT_FILES:
        write_bytes(root / 'published_reports' / filename,
                    fetch(source_url(PINNED_BACKGROUND, 'reports/' + filename)))
    manifest = load_json(root / 'ARCHIVE_MANIFEST.json')
    member_index = load_json(root / 'MEMBER_MANIFEST_INDEX.json')
    shards = manifest['shards']
    if [row['part'] for row in shards] != list(range(1, len(shards) + 1)):
        raise ValueError('Nonconsecutive pinned archive shard list')
    state = {'schema': 'focused-source-recovery-v1', 'complete': False,
             'archive_commit': PINNED_ARCHIVE, 'source_commit': PINNED_BACKGROUND,
             'started_utc': now(), 'verified_shards': 0, 'retained_members': 0,
             'scope': 'Pinned manifest input and completed jobs; no withdrawn outputs or model fits'}
    atomic_json(status_path, state)
    hashes = {}
    jobs = None
    first_repeats = set()
    gzip_logs = {}
    for name in ('all_attempt_receipts', 'all_attempt_heartbeats', 'all_attempt_errors', 'all_result_summaries'):
        gzip_logs[name] = gzip.open(root / (name + '.jsonl.gz'), 'wt', encoding='utf-8', compresslevel=1)
    retained_log = gzip.open(root / 'retained_member_hashes.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1)
    def retain(name, data, destination=None, log=None, value_key=None):
        sha = hashlib.sha256(data).hexdigest()
        row = {'archive_path': name, 'bytes': len(data), 'sha256': sha}
        hashes[name] = (len(data), sha)
        if destination:
            if name.endswith('/split_frozen.json'):
                # Lossless storage transform only; decompressed bytes retain the
                # original member hash and every actual sample ID.
                destination = Path(str(destination) + '.gz')
                write_bytes(destination, gzip.compress(data, compresslevel=1, mtime=0))
                row['local_storage_transform'] = 'gzip; original bytes authenticated before compression'
            else:
                write_bytes(destination, data)
            row['local_path'] = str(Path(destination).relative_to(root))
        if log:
            original = json.loads(data)
            gzip_logs[log].write(json.dumps({**row, value_key: original}, ensure_ascii=False, separators=(',', ':')) + '\n')
        retained_log.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
        state['retained_members'] += 1
    def progress(done, total):
        state.update(verified_shards=done, shard_count=total,
                     cpu_seconds_this_invocation=time.process_time()-start_cpu,
                     wall_seconds_this_invocation=time.monotonic()-start_wall,
                     heartbeat_utc=now())
        atomic_json(status_path, state)
        if done == 1 or done % 16 == 0 or done == total:
            print(json.dumps({'recovery': 'streaming_verified_archive', 'shards': done,
                              'of': total, 'retained_members': state['retained_members']}), flush=True)
    transport = ShardStream(shards, progress)
    member_manifest = None
    try:
        with io.BufferedReader(transport, buffer_size=1024 * 1024) as buffered:
            with gzip.GzipFile(fileobj=buffered, mode='rb') as uncompressed:
                with tarfile.open(fileobj=uncompressed, mode='r|') as archive:
                    for member in archive:
                        name = member.name
                        path = PurePosixPath(name)
                        if path.is_absolute() or '..' in path.parts or '\\' in name or not name.startswith(PREFIX):
                            raise ValueError('Unsafe or out-of-scope archive member ' + name)
                        if not member.isfile():
                            if member.isdir():
                                continue
                            raise ValueError('Nonregular evidence archive member ' + name)
                        relative = name[len(PREFIX):]
                        destination = None
                        log = None
                        value_key = None
                        if relative.startswith(('snapshot_source/', 'inputs/')):
                            destination = root / relative
                        elif relative == 'snapshot.json':
                            destination = root / relative
                        elif relative == 'archive_members.jsonl.gz':
                            member_manifest = root / relative
                            write_bytes(member_manifest, archive.extractfile(member).read())
                            continue
                        elif relative.startswith('jobs/'):
                            if jobs is None:
                                jobs, first_repeats = load_queue(root)
                                atomic_json(root / 'recovered_queue_identity.json', {
                                    'completed_jobs': len(jobs), 'first_three_repeat_ordinals': sorted(first_repeats),
                                    'selection_rule': 'First three qualifying repeats in original queue ordinal order'})
                            parts = relative.split('/')
                            job = jobs.get(parts[1])
                            if job is None:
                                raise ValueError('Job archive member is not queue-committed complete: ' + name)
                            leaf = '/'.join(parts[2:])
                            if leaf == 'result.json':
                                log, value_key = 'all_result_summaries', 'result'
                            elif leaf == 'cpu_receipt.json':
                                log, value_key = 'all_attempt_receipts', 'receipt'
                            elif leaf == 'heartbeat.json':
                                log, value_key = 'all_attempt_heartbeats', 'heartbeat'
                            if selected_job(job) and leaf in ('result.json', 'selection.json', 'job_frozen.json',
                                    'split_frozen.json', 'split_frozen.json.gz', 'split_frozen.archive_manifest.json',
                                    'cpu_receipt.json', 'heartbeat.json'):
                                destination = root / relative
                            if selected_job(job) and int(job['repeat']) in first_repeats:
                                if leaf in ('fitted_artifacts.npz', 'fitted_artifacts.lossless.npz',
                                            'fitted_artifacts.recipe.json') or (leaf.startswith('rank_cache/outer') and leaf.endswith('.npz')):
                                    destination = root / relative
                        elif relative.startswith('attempts/'):
                            leaf = relative.rsplit('/', 1)[-1]
                            if leaf.endswith('cpu_receipt.json'):
                                log, value_key = 'all_attempt_receipts', 'receipt'
                            elif leaf.endswith('heartbeat.json'):
                                log, value_key = 'all_attempt_heartbeats', 'heartbeat'
                            elif leaf.endswith('.json') and any(word in leaf for word in ('error', 'exit', 'failure', 'attempt')):
                                log, value_key = 'all_attempt_errors', 'error'
                        elif relative.startswith('incomplete_not_counted/jobs/'):
                            leaf = relative.rsplit('/', 1)[-1]
                            if leaf == 'cpu_receipt.json':
                                log, value_key = 'all_attempt_receipts', 'receipt'
                            elif leaf == 'heartbeat.json':
                                log, value_key = 'all_attempt_heartbeats', 'heartbeat'
                            elif leaf == 'result.json':
                                log, value_key = 'all_result_summaries', 'result'
                        if destination or log:
                            data = archive.extractfile(member).read()
                            retain(name, data, destination, log, value_key)
                # Validate gzip trailer/CRC and the complete transport stream.
                while uncompressed.read(1024 * 1024):
                    pass
        if member_manifest is None or sha256_file(member_manifest) != member_index['sha256']:
            raise ValueError('Final archive member-manifest SHA256 mismatch or missing')
        verified = 0
        with gzip.open(member_manifest, 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                original = hashes.pop(row['path'], None)
                if original is not None:
                    if original != (int(row['bytes']), row['sha256']):
                        raise ValueError('Retained archive member differs from frozen manifest: ' + row['path'])
                    verified += 1
        if hashes:
            raise ValueError('Retained members absent from frozen manifest: ' + str(list(hashes)[:3]))
        if sha256_file(root / 'inputs/current1672_paired612.npz') != INPUT_SHA256:
            raise ValueError('Frozen corrected input SHA256 mismatch')
        selected_count = 0
        for job_id, job in (jobs or {}).items():
            if not selected_job(job):
                continue
            selected_count += 1
            folder = root / 'jobs' / job_id
            required = ('result.json', 'selection.json', 'job_frozen.json')
            if any(not (folder / name).is_file() for name in required) or not any(
                    (folder / name).is_file() for name in ('split_frozen.json', 'split_frozen.json.gz')):
                raise ValueError('Incomplete retained evidence for completed source job: ' + job_id)
        if len(transport.completed) != len(shards):
            raise ValueError('Gzip stream did not consume every pinned transport shard')
        # Close compressed aggregate streams before authenticating their cache
        # representation for subsequent independent runner invocations.
        retained_log.close()
        for stream in gzip_logs.values():
            stream.close()
        cache_streams = [root / (name + '.jsonl.gz') for name in gzip_logs]
        cache_streams.extend([root / 'retained_member_hashes.jsonl.gz', member_manifest])
        source_metadata = [root / 'ARCHIVE_MANIFEST.json', root / 'MEMBER_MANIFEST_INDEX.json']
        source_metadata.extend((root / 'published_source').glob('*'))
        source_metadata.extend((root / 'published_reports').glob('*'))
        verification = {'schema': 'focused-recovery-verification-v1', 'complete': True,
            'archive_commit': PINNED_ARCHIVE, 'source_commit': PINNED_BACKGROUND,
            'shards_verified': len(transport.completed), 'compressed_bytes': sum(r['bytes'] for r in transport.completed),
            'members_verified': verified, 'member_manifest_sha256': member_index['sha256'],
            'completed_jobs_in_source_queue': len(jobs or {}), 'selected_jobs_retained': selected_count,
            'input_sha256': INPUT_SHA256, 'gzip_crc_verified_to_eof': True,
            'cache_stream_hashes': {str(p.relative_to(root)): sha256_file(p) for p in cache_streams},
            'source_metadata_sha256': {str(p.relative_to(root)): sha256_file(p) for p in source_metadata},
            'new_classifier_fits': 0, 'new_pls_fits': 0,
            'cpu_seconds_this_invocation': time.process_time()-start_cpu,
            'wall_seconds_this_invocation': time.monotonic()-start_wall, 'completed_utc': now()}
        atomic_json(root / 'recovery_verification.json', verification)
        state.update(verification)
        atomic_json(status_path, state)
        publish_receipt(verification)
        return state
    except BaseException as error:
        state.update(complete=False, error_type=type(error).__name__, error=str(error), failed_utc=now(),
                     cpu_seconds_this_invocation=time.process_time()-start_cpu,
                     wall_seconds_this_invocation=time.monotonic()-start_wall)
        atomic_json(status_path, state)
        raise
    finally:
        retained_log.close()
        for stream in gzip_logs.values():
            stream.close()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=BACKGROUND_DIR)
    args = parser.parse_args()
    print(json.dumps(recover(args.root), ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
