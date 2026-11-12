#!/usr/bin/env python3
"""Publish this run's small, inspectable evidence without cloning repository history."""
import argparse
import base64
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile


def api(path, payload=None, method=None):
    cmd = ['gh', 'api', path]
    if method:
        cmd += ['--method', method]
    if payload is not None:
        cmd += ['--input', '-']
    p = subprocess.run(cmd, input=json.dumps(payload) if payload is not None else None,
                       text=True, capture_output=True, timeout=60)
    if p.returncode:
        raise RuntimeError('GitHub publication failed: ' + p.stderr[-1000:])
    return json.loads(p.stdout)


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def publish(root):
    cfg = json.loads((root / 'publication.json').read_text())
    repo, branch, prefix = cfg['repository'], cfg['branch'], cfg['prefix'].rstrip('/')
    endpoint = 'repos/' + repo
    state_path = root / 'publication_state.json'
    old = json.loads(state_path.read_text()) if state_path.exists() else {}
    hashes = old.get('published_hashes', {})
    candidates = {}
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part.startswith('.') or part in ('__pycache__', 'tmp', 'tmpfiles') for part in rel.parts):
            continue
        if len(rel.parts) > 1 and rel.parts[0].startswith('runtime_selftest'):
            continue
        if rel.parts[0] == 'audit_sources' and path.name.endswith('_tree.json'):
            continue  # Raw repository listings are redundant with pinned audit references.
        if path.suffix not in ('.json', '.py', '.md', '.xyz', '.sdf', '.npy', '.npz', '.csv'):
            continue
        if path.name in ('publication_state.json', 'lease.json') or path.stat().st_size > 2500000:
            continue
        # Density matrices and live wavefunctions remain in local checkpoints;
        # manifests authenticate them without republishing large caches.
        if 'density' in path.name.lower() or path.name.endswith('.chk'):
            continue
        candidates[str(rel)] = path.read_bytes()
    # SQLite's online backup API supplies a consistent read-only snapshot.
    ledger = root / 'queue.sqlite3'
    if not ledger.exists():
        ledger = root / 'ledger.sqlite3'
    if ledger.exists():
        with tempfile.NamedTemporaryFile(suffix='.sqlite3') as tmp:
            source = sqlite3.connect(f'file:{ledger}?mode=ro', uri=True, timeout=10)
            dest = sqlite3.connect(tmp.name)
            source.backup(dest)
            dest.close()
            source.close()
            candidates['ledger_snapshot.sqlite3'] = Path(tmp.name).read_bytes()
    new_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in candidates.items()}
    changed = {n: d for n, d in candidates.items() if hashes.get(n) != new_hashes[n]}
    removed = [name for name in hashes if '/' in name and name.split('/')[0].startswith('runtime_selftest')]
    if not changed and not removed:
        return {'status': 'unchanged', 'commit': old.get('commit')}
    head = api(endpoint + '/git/ref/heads/' + branch)['object']['sha']
    commit = api(endpoint + '/git/commits/' + head)
    entries = []
    for name in removed:
        entries.append({'path': prefix + '/' + name, 'mode': '100644', 'type': 'blob', 'sha': None})
    for name, data in sorted(changed.items()):
        item = {'path': prefix + '/' + name, 'mode': '100644', 'type': 'blob'}
        try:
            item['content'] = data.decode('utf-8')
        except UnicodeDecodeError:
            item['sha'] = api(endpoint + '/git/blobs', {
                'content': base64.b64encode(data).decode(), 'encoding': 'base64'})['sha']
        entries.append(item)
    tree = api(endpoint + '/git/trees', {'base_tree': commit['tree']['sha'], 'tree': entries})
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    created = api(endpoint + '/git/commits', {'message': 'Checkpoint independent spinosin conformer calculations ' + now,
                  'tree': tree['sha'], 'parents': [head]})
    # Non-force fast-forward rejects competing ref movement. A failed upload
    # never edits the local scientific queue or its checkpoints.
    api(endpoint + '/git/refs/heads/' + branch, {'sha': created['sha'], 'force': False}, 'PATCH')
    result = {'status': 'published', 'at_utc': now, 'commit': created['sha'],
              'files_changed': len(changed), 'files_removed': len(removed),
              'published_hashes': {**{n: h for n, h in hashes.items() if n not in removed}, **new_hashes}}
    atomic(state_path, result)
    return {k: v for k, v in result.items() if k != 'published_hashes'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    with open(root / '.publisher.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(0)
        try:
            print(json.dumps(publish(root), ensure_ascii=False))
        except Exception as exc:
            atomic(root / 'publication_error.json', {'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'error': str(exc), 'training_affected': False})
            print(json.dumps({'status': 'publication_error', 'error': str(exc)}, ensure_ascii=False))
            raise SystemExit(1)
