#!/usr/bin/env python3
"""Detached, resumable orchestration for three complementary quantum jobs.

Only the explicitly authorized aggregate CPU boundary ends unfinished work.
Scientific qualification gates remain in the driver and never stop this queue.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


BRANCH = 'research/parallel-spinosin-ir-900000-20261112'
AUTHORIZED_CPU_SECONDS = 900000.0
TASKS = {'publication_spinosin': 2, 'ir_calycosin': 1, 'ir_spinosin': 1}
TICKS = os.sysconf('SC_CLK_TCK')


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class Paths:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.output = self.root / 'reports/pyscf_parallel_missing_20261112'
        self.background = self.output / '.background'
        self.script = self.root / 'scripts/background_pyscf_parallel_missing_20261112.py'
        self.driver = self.root / 'scripts/run_pyscf_parallel_missing_20261112.py'
        self.state = self.output / 'background_state.json'
        self.ledger = self.background / 'cpu_ledger.json'
        self.stop = self.background / 'budget_exhausted.json'


@contextmanager
def lock(path, nonblocking=False):
    stream = Path(path).open('a+')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield stream
    finally:
        stream.close()


def proc(pid):
    try:
        raw = (Path('/proc') / str(pid) / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        return {'pid': int(pid), 'state': fields[0], 'ppid': int(fields[1]),
                'ticks': sum(int(fields[i]) for i in (11, 12, 13, 14)),
                'start_ticks': int(fields[19])}
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return None


def identity(pid):
    value = proc(pid)
    return {key: value[key] for key in ('pid', 'start_ticks')} if value else None


def alive(value):
    if not value:
        return False
    current = proc(value.get('pid', -1))
    return bool(current and current['start_ticks'] == value.get('start_ticks')
                and current['state'] not in ('Z', 'X'))


def process_snapshot():
    records = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            value = proc(int(entry.name))
            if value:
                records[value['pid']] = value
    return records


def tree_ticks(root, records):
    current = records.get(root['pid'])
    if not current or current['start_ticks'] != root['start_ticks']:
        return None
    descendants = {root['pid']}
    while True:
        extra = {pid for pid, value in records.items() if value['ppid'] in descendants}
        expanded = descendants | extra
        if expanded == descendants:
            break
        descendants = expanded
    # Every live process contributes its own and already-waited children's CPU.
    # A still-live child is not yet present in its parent's waited-child counters.
    return sum(records[pid]['ticks'] for pid in descendants)


def account(paths):
    with lock(paths.background / 'accounting.lock'):
        ledger = read(paths.ledger, {'schema': 'parallel-cpu-ledger-v1',
                      'authorized_cpu_seconds': AUTHORIZED_CPU_SECONDS,
                      'created_at_utc': utc(), 'generations': []})
        if ledger.get('authorized_cpu_seconds') != AUTHORIZED_CPU_SECONDS:
            raise ValueError('Existing CPU authorization differs; refusing to reset the ledger')
        records = process_snapshot()
        for generation in ledger['generations']:
            measured = tree_ticks(generation, records)
            if measured is not None:
                generation['charged_ticks'] = max(generation.get('charged_ticks', 0), measured)
            generation['alive'] = alive(generation)
        ledger['charged_cpu_seconds'] = sum(g.get('charged_ticks', 0) for g in ledger['generations']) / TICKS
        ledger['remaining_cpu_seconds'] = max(0.0, AUTHORIZED_CPU_SECONDS - ledger['charged_cpu_seconds'])
        ledger['heartbeat_at_utc'] = utc()
        ledger['accounting_method'] = 'monotonic per-generation own+waited-child+live-descendant /proc CPU ticks'
        write(paths.ledger, ledger)
        return ledger


def register(paths, task, child):
    value = identity(child.pid)
    if not value:
        raise RuntimeError('Worker process disappeared before registration')
    with lock(paths.background / 'accounting.lock'):
        ledger = read(paths.ledger, {'schema': 'parallel-cpu-ledger-v1',
                      'authorized_cpu_seconds': AUTHORIZED_CPU_SECONDS,
                      'created_at_utc': utc(), 'generations': []})
        ledger['generations'].append({**value, 'task': task, 'charged_ticks': 0,
                                      'started_at_utc': utc()})
        write(paths.ledger, ledger)
    return value


def receipt(paths, task):
    value = read(paths.output / 'task_completion' / (task + '.json'), {})
    return value.get('status') == 'COMPLETE' and value.get('paper_ready') is True


def stop_owned_workers(paths, reason):
    ledger = account(paths)
    write(paths.stop, {'status': 'AUTHORIZED_CPU_BUDGET_REACHED', 'at_utc': utc(),
                      'reason': reason, 'authorized_cpu_seconds': AUTHORIZED_CPU_SECONDS,
                      'charged_cpu_seconds': ledger['charged_cpu_seconds'],
                      'partial_checkpoints_preserved': True})
    for worker in ledger['generations']:
        if alive(worker):
            try:
                # Workers own a session, so no unrelated process can share this group.
                os.killpg(worker['pid'], signal.SIGTERM)
            except ProcessLookupError:
                pass


def spawn_role(paths, role):
    log = (paths.background / (role + '.log')).open('ab', buffering=0)
    try:
        child = subprocess.Popen([sys.executable, str(paths.script), '--root', str(paths.root),
                                  '--' + role], cwd=paths.root, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                 close_fds=True)
    finally:
        log.close()
    return child


def ensure_guard(paths):
    if not alive(read(paths.background / 'guard.json')):
        spawn_role(paths, 'guard')


def task_process(paths, task):
    env = os.environ.copy()
    threads = TASKS[task]
    env.update(OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads), NUMEXPR_NUM_THREADS=str(threads),
               PYSCF_MAX_MEMORY='2500', PYTHONUNBUFFERED='1')
    log = (paths.background / (task + '.log')).open('ab', buffering=0)
    try:
        child = subprocess.Popen([sys.executable, str(paths.driver), '--root', str(paths.root),
                                  '--task', task, '--threads', str(threads)], cwd=paths.root,
                                 env=env, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    finally:
        log.close()
    register(paths, task, child)
    return child


def publish(paths):
    try:
        with lock(paths.background / 'publication.lock', nonblocking=True):
            def git(*args):
                result = subprocess.run(['git', *args], cwd=paths.root, text=True,
                                        capture_output=True, timeout=55)
                if result.returncode:
                    raise RuntimeError(result.stderr[-2000:] or result.stdout[-2000:])
                return result.stdout.strip()
            if git('branch', '--show-current') != BRANCH:
                raise RuntimeError('Publication restricted to the dedicated computation branch')
            candidates = [paths.script, paths.driver]
            candidates += list((paths.root / 'docs').glob('*PARALLEL*20261112*'))
            for path in paths.output.rglob('*'):
                if not path.is_file() or any(part in ('.background', '.local') for part in path.parts):
                    continue
                if path.suffix.lower() == '.npy' and 'density' in path.name.lower():
                    continue
                if path.suffix.lower() in ('.json', '.csv', '.md', '.xyz', '.npy', '.npz') and path.stat().st_size < 8 * 1024 * 1024:
                    candidates.append(path)
            relative = sorted({str(p.relative_to(paths.root)) for p in candidates if p.exists()})
            for offset in range(0, len(relative), 100):
                git('add', '--', *relative[offset:offset + 100])
            changed = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=paths.root).returncode
            if changed == 1:
                git('commit', '-m', 'checkpoint: parallel Spinosin and complementary IR calculations')
            elif changed != 0:
                raise RuntimeError('Could not inspect staged checkpoint')
            git('push', 'origin', 'HEAD:refs/heads/' + BRANCH)
            result = {'status': 'PUBLISHED', 'commit_sha': git('rev-parse', 'HEAD'), 'at_utc': utc()}
    except BlockingIOError:
        return {'status': 'ANOTHER_PUBLICATION_IN_PROGRESS'}
    except Exception as error:
        result = {'status': 'RETRY_PENDING', 'error': str(error), 'at_utc': utc()}
    write(paths.output / 'publication.json', result)
    return result


def verify(paths):
    ledger = read(paths.ledger, {})
    latest = {}
    for value in ledger.get('generations', []):
        latest[value['task']] = value
    return {'supervisor_alive': alive(read(paths.background / 'supervisor.json')),
            'guard_alive': alive(read(paths.background / 'guard.json')),
            'tasks_alive': {task: alive(latest.get(task)) for task in TASKS},
            'tasks_complete': {task: receipt(paths, task) for task in TASKS},
            'cpu_accounting': ledger, 'state': read(paths.state, {}),
            'budget_boundary': read(paths.stop), 'publication': read(paths.output / 'publication.json')}


def supervise(paths):
    try:
        with lock(paths.background / 'supervisor.lock', nonblocking=True):
            write(paths.background / 'supervisor.json', {**identity(os.getpid()), 'started_at_utc': utc()})
            ensure_guard(paths)
            children = {}
            last_publish = time.monotonic()
            while True:
                ledger = account(paths)
                if paths.stop.exists() or ledger['charged_cpu_seconds'] >= AUTHORIZED_CPU_SECONDS:
                    stop_owned_workers(paths, 'Aggregate measured CPU authorization reached')
                    write(paths.state, {'status': 'AUTHORIZED_CPU_BUDGET_REACHED', 'at_utc': utc(),
                                        'cpu_accounting': ledger, 'partial_checkpoints_preserved': True})
                    publish(paths)
                    return
                ensure_guard(paths)
                live_by_task = {}
                for value in ledger['generations']:
                    if alive(value):
                        live_by_task[value['task']] = value
                for task in TASKS:
                    child = children.get(task)
                    if child and not alive(identity(child.pid)):
                        # account() above captures zombie's final CPU before wait() reaps it.
                        child.poll()
                        children.pop(task, None)
                    if not receipt(paths, task) and task not in live_by_task:
                        children[task] = task_process(paths, task)
                complete = {task: receipt(paths, task) for task in TASKS}
                write(paths.state, {'schema': 'parallel-background-v1',
                                    'status': 'COMPLETE' if all(complete.values()) else 'RUNNING',
                                    'heartbeat_at_utc': utc(), 'supervisor': identity(os.getpid()),
                                    'guard': read(paths.background / 'guard.json'),
                                    'tasks_complete': complete,
                                    'authorized_cpu_seconds': AUTHORIZED_CPU_SECONDS,
                                    'charged_cpu_seconds': ledger['charged_cpu_seconds'],
                                    'remaining_cpu_seconds': ledger['remaining_cpu_seconds'],
                                    'quality_thresholds_end_execution': False,
                                    'maximum_retries': None, 'maximum_rounds': None,
                                    'arbitrary_runtime_timeout': None})
                if all(complete.values()):
                    publish(paths)
                    return
                if time.monotonic() - last_publish >= 600:
                    publish(paths)
                    last_publish = time.monotonic()
                time.sleep(2)
    except BlockingIOError:
        return


def guard(paths):
    try:
        with lock(paths.background / 'guard.lock', nonblocking=True):
            write(paths.background / 'guard.json', {**identity(os.getpid()), 'started_at_utc': utc()})
            while True:
                ledger = account(paths)
                if paths.stop.exists() or ledger['charged_cpu_seconds'] >= AUTHORIZED_CPU_SECONDS:
                    stop_owned_workers(paths, 'Independent guardian observed aggregate CPU authorization boundary')
                    return
                if all(receipt(paths, task) for task in TASKS):
                    return
                if not alive(read(paths.background / 'supervisor.json')):
                    spawn_role(paths, 'supervise')
                time.sleep(2)
    except BlockingIOError:
        return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    group = parser.add_mutually_exclusive_group(required=True)
    for option in ('start', 'supervise', 'guard', 'verify', 'publish'):
        group.add_argument('--' + option, action='store_true')
    args = parser.parse_args()
    paths = Paths(args.root)
    if not args.verify:
        paths.background.mkdir(parents=True, exist_ok=True)
    if args.verify:
        print(json.dumps(verify(paths), ensure_ascii=False, indent=2))
    elif args.publish:
        print(json.dumps(publish(paths), ensure_ascii=False, indent=2))
    elif args.start:
        ledger = account(paths)
        if paths.stop.exists() or ledger['charged_cpu_seconds'] >= AUTHORIZED_CPU_SECONDS:
            raise SystemExit('This authorization is exhausted; existing accounting was preserved')
        with lock(paths.background / 'launch.lock'):
            ensure_guard(paths)
            if not alive(read(paths.background / 'supervisor.json')):
                child = spawn_role(paths, 'supervise')
                print(json.dumps({'supervisor_pid': child.pid, 'detached': True,
                                  'authorized_cpu_seconds': AUTHORIZED_CPU_SECONDS}))
            else:
                print(json.dumps({'already_running': True}))
    elif args.guard:
        guard(paths)
    else:
        supervise(paths)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
