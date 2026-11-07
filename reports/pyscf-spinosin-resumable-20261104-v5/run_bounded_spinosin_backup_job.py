"""Root's finite 17-round research job; publish only closed segment archives.

Each controller invocation runs at most ten new RHF units. Git publication is
explicitly authorized for this research branch and occurs between calculations.
Failures never trigger an automatic scientific retry or an enlarged budget.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
BRANCH = 'research/pyscf-spinosin-mechanism-20261031'
CONTROLLER = ROOT/'scripts/run_pyscf_serial_segments_20261104_v5.py'
CONTROLLER_SHA = 'a01b6a79f457bca4b0ab90e63b45a3bb524b62fc586fab26905d6b43a2aba1d7'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True, timeout=180,
            env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'}).strip()


def assert_branch_head(expected):
    if git('branch', '--show-current') != BRANCH or git('rev-parse', 'HEAD') != expected:
        raise RuntimeError('Concurrent Git branch/revision change; stop without committing')


def publish(paths, expected_head, message):
    """Commit only scoped evidence; explicit ref update can never commit to main."""
    with (ROOT/'.runtime/pyscf-v4-global.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _publish_closed(paths, expected_head, message)


def _publish_closed(paths, expected_head, message):
    assert_branch_head(expected_head)
    if git('diff', '--cached', '--name-only'):
        raise RuntimeError('Another task staged files; stop before Git mutation')
    relative = [str(p.resolve().relative_to(ROOT)) for p in paths]
    git('add', '--', *relative)
    assert_branch_head(expected_head)
    staged = git('diff', '--cached', '--name-only').splitlines()
    if not staged or any(not any(p == scope or p.startswith(scope+'/') for scope in relative) for p in staged):
        raise RuntimeError('Staged changes are empty or outside this closed evidence scope')
    tree = git('write-tree')
    actual = git('diff-tree', '-r', '--name-only', '--no-commit-id', expected_head, tree).splitlines()
    if not actual or any(not any(p == scope or p.startswith(scope+'/') for scope in relative) for p in actual):
        raise RuntimeError('Actual commit tree includes changes outside the closed evidence scope')
    assert_branch_head(expected_head)
    revision = git('commit-tree', tree, '-p', expected_head, '-m', message)
    assert_branch_head(expected_head)
    git('update-ref', 'refs/heads/'+BRANCH, revision, expected_head)
    if git('branch', '--show-current') != BRANCH:
        raise RuntimeError('Branch changed; the explicit research ref is preserved, main untouched')
    git('push', 'origin', revision+':refs/heads/'+BRANCH)
    return revision


def checked(record):
    p = Path(record['path']).resolve()
    if not p.is_relative_to(ROOT) or sha(p) != record['sha256']:
        raise RuntimeError('Job input hash/path mismatch: '+str(p))
    return p


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    manifest = checked({'path': str(a.manifest), 'sha256': a.manifest_sha256})
    spec = json.loads(manifest.read_text())
    if sha(CONTROLLER) != CONTROLLER_SHA or git('branch', '--show-current') != BRANCH:
        raise RuntimeError('Frozen controller or research branch differs')
    snapshot_manifest = checked(spec['snapshot_manifest'])
    gradient = checked(spec['gradient_check'])
    gate = checked(spec['scientific_gate'])
    if checked(spec['job_source']) != Path(__file__).resolve():
        raise RuntimeError('Job source must be explicitly frozen')
    ledger = REPORT/'cpu_budget.json'
    if any(r['status'] == 'RUNNING' for r in json.loads(ledger.read_text())['phases']):
        raise RuntimeError('A current quantum phase is unresolved')
    if git('diff', '--cached', '--name-only'):
        raise RuntimeError('Do not include another task\'s staged work')
    # All prerequisite arrays/logs/proofs and the frozen manifest/source must be
    # published before the deliberately narrower per-round backup scopes begin.
    prerequisite_paths = [snapshot_manifest.parent.parent, gradient.parent, gate,
                          manifest, Path(__file__).resolve()]
    for path in prerequisite_paths:
        if git('status', '--porcelain', '--', str(path.relative_to(ROOT))):
            raise RuntimeError('Uncommitted prerequisite evidence: '+str(path))
    local_head = git('rev-parse', 'HEAD')
    remote = git('ls-remote', 'origin', 'refs/heads/'+BRANCH).split()
    if not remote or remote[0] != local_head:
        raise RuntimeError('Publish prerequisite commit before starting the backup job')
    base = REPORT/'auto-segments'
    if base.exists():
        raise FileExistsError('Preserve an earlier finite backup job')
    if not a.execute:
        print(json.dumps({'status': 'NOT_RUN', 'max_rounds': 17, 'max_new_units_each': 10,
                          'branch': BRANCH, 'manifest_sha256': a.manifest_sha256}))
        return
    base.mkdir()
    status_path = base/'job-status.json'
    def emit(status, **values):
        v = {'utc': datetime.now(timezone.utc).isoformat(), 'status': status,
             'manifest_sha256': a.manifest_sha256, **values}
        status_path.write_text(json.dumps(v, indent=2)+'\n')
        print(json.dumps(v), flush=True)
    lock = ROOT/'.runtime/spinosin-backup-job.lock'
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for number in range(1, 18):
                expected_head = git('rev-parse', 'HEAD')
                assert_branch_head(expected_head)
                output = base/f'round-{number:03d}'
                cmd = [sys.executable, str(CONTROLLER), 'run', '--ledger', str(ledger),
                       '--snapshot', str(snapshot_manifest.parent),
                       '--snapshot-manifest-sha256', spec['snapshot_manifest']['sha256'],
                       '--gradient-check-result', str(gradient),
                       '--gradient-check-result-sha256', spec['gradient_check']['sha256'],
                       '--gate-report', str(gate), '--gate-report-sha256', spec['scientific_gate']['sha256'],
                       '--worker-sha256', spec['consumer_worker_sha256'], '--output', str(output),
                       '--rhf-cache', str(ROOT/'outputs/pyscf-spinosin-resumable-20261104-v5/spinosin-C001-restart'),
                       '--phase-prefix', 'v5-spinosin-resumable', '--max-segments', '1',
                       '--phase-cpu-seconds', '122400', '--phase-wall-seconds', '43200']
                emit('RUNNING', round=number, command=cmd)
                proc = subprocess.run(cmd, cwd=ROOT, check=False)
                result = json.loads((output/'result.json').read_text())
                ordinary_partial = (proc.returncode == 2 and result['status'] == 'PARTIAL'
                                    and result.get('stop_reason') == 'FINITE_SEGMENT_COUNT_EXHAUSTED'
                                    and len(result.get('segments', [])) == 1)
                done = proc.returncode == 0 and result['status'] == 'DONE'
                if not (ordinary_partial or done):
                    raise RuntimeError('Finite controller stopped; manual review required: '+str(result.get('stop_reason')))
                # Consumer/controller verified actual clean closure and sealed
                # all numeric artifacts before any backup step reads them.
                if any(r['status'] == 'RUNNING' for r in json.loads(ledger.read_text())['phases']):
                    raise RuntimeError('Never publish an active quantum segment')
                emit('CLOSED_READY_FOR_BACKUP', round=number, completed_units=result['completed_units'])
                revision = publish([output, ledger, status_path], expected_head,
                    f'Checkpoint Spinosin RHF: {result["completed_units"]}/150 units; '+
                    ('full frequencies computed' if done else 'frequency stage partial'))
                emit('BACKUP_PUSHED', round=number, commit=revision, completed_units=result['completed_units'])
                if done:
                    emit('DONE', round=number, commit=revision, final_result=result['final_result'])
                    terminal = publish([status_path], revision,
                        'Record completed finite Spinosin frequency job and published checkpoint')
                    print(json.dumps({'status': 'DONE_PUBLISHED', 'commit': terminal}), flush=True)
                    return
            emit('PARTIAL', reason='FINITE_17_ROUND_LIMIT_REACHED')
        except BaseException as exc:
            emit('PARTIAL', reason=str(exc), exception_type=type(exc).__name__)
            raise


if __name__ == '__main__':
    main()
