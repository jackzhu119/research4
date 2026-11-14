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
import math
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
BOOT_ID = Path('/proc/sys/kernel/random/boot_id').read_text().strip()


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
        self.authorization = self.output / 'authorization.json'


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
    return {**{key: value[key] for key in ('pid', 'start_ticks')}, 'boot_id': BOOT_ID} if value else None


def alive(value):
    if not value or value.get('retired_at_utc') or value.get('boot_id') != BOOT_ID:
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
    if root.get('retired_at_utc') or root.get('boot_id') != BOOT_ID:
        return None
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


def empty_ledger():
    return {'schema': 'parallel-cpu-ledger-v1',
            'authorized_cpu_seconds': AUTHORIZED_CPU_SECONDS,
            'created_at_utc': utc(), 'generations': []}


def authorization_receipt(paths):
    value = read(paths.authorization)
    if value is None:
        value = {'schema': 'parallel-cpu-authorization-v1',
                 'initial_authorized_cpu_seconds': AUTHORIZED_CPU_SECONDS,
                 'worker_frozen_protocol_initial_authorization_cpu_seconds': AUTHORIZED_CPU_SECONDS,
                 'scientific_protocol_and_dispatcher_unchanged': True,
                 'events': [], 'cumulative_authorized_cpu_seconds': AUTHORIZED_CPU_SECONDS}
    if value.get('initial_authorized_cpu_seconds') != AUTHORIZED_CPU_SECONDS:
        raise ValueError('Initial authorization cannot be replaced')
    total = AUTHORIZED_CPU_SECONDS
    ids = set()
    for event in value.get('events', []):
        additional = event.get('additional_cpu_seconds')
        event_id = event.get('authorization_id')
        if (not isinstance(additional, (int, float)) or isinstance(additional, bool)
                or not math.isfinite(additional) or additional <= 0
                or not isinstance(event_id, str) or not event_id.strip() or event_id in ids):
            raise ValueError('Invalid or duplicate CPU authorization event')
        if event.get('previous_total_cpu_seconds') != total:
            raise ValueError('Authorization events do not form an append-only chain')
        total += additional
        if event.get('cumulative_total_cpu_seconds') != total:
            raise ValueError('Authorization event total differs from its increment')
        ids.add(event_id)
    if value.get('cumulative_authorized_cpu_seconds') != total:
        raise ValueError('Authorization receipt cumulative total is inconsistent')
    return value


def apply_authorization(paths, ledger):
    authorization = authorization_receipt(paths)
    old_events = ledger.get('authorization_events', [])
    events = authorization['events']
    if events[:len(old_events)] != old_events:
        raise ValueError('Existing authorization events cannot be removed or rewritten')
    previous_total = AUTHORIZED_CPU_SECONDS + sum(e['additional_cpu_seconds'] for e in old_events)
    if ledger.get('authorized_cpu_seconds') != previous_total:
        raise ValueError('CPU ledger total differs from its existing authorization events')
    ledger['authorization_events'] = [dict(event) for event in events]
    ledger['authorized_cpu_seconds'] = authorization['cumulative_authorized_cpu_seconds']
    return authorization


def update_accounting(paths, ledger):
    apply_authorization(paths, ledger)
    records = process_snapshot()
    for generation in ledger['generations']:
        measured = tree_ticks(generation, records)
        if measured is not None:
            generation['charged_ticks'] = max(generation.get('charged_ticks', 0), measured)
        generation['alive'] = alive(generation)
    ledger['charged_cpu_seconds'] = sum(g.get('charged_ticks', 0) for g in ledger['generations']) / TICKS
    ledger['remaining_cpu_seconds'] = max(0.0, ledger['authorized_cpu_seconds'] - ledger['charged_cpu_seconds'])
    ledger['heartbeat_at_utc'] = utc()
    ledger['accounting_method'] = 'monotonic per-generation own+waited-child+live-descendant /proc CPU ticks'
    write(paths.ledger, ledger)
    return ledger


def account(paths):
    with lock(paths.background / 'accounting.lock'):
        return update_accounting(paths, read(paths.ledger, empty_ledger()))


def budget_reached(paths, ledger):
    stopped = read(paths.stop, {})
    return (ledger['charged_cpu_seconds'] >= ledger['authorized_cpu_seconds']
            or stopped.get('authorized_cpu_seconds') == ledger['authorized_cpu_seconds'])


def retire_previous_runtime(ledger):
    retired = []
    previous_heartbeat = ledger.get('heartbeat_at_utc')
    for generation in ledger['generations']:
        if generation.get('retired_at_utc') or generation.get('boot_id') == BOOT_ID:
            continue
        current = proc(generation['pid'])
        if (not generation.get('boot_id') and current
                and current['start_ticks'] == generation['start_ticks']):
            raise ValueError('A legacy process identity still matches; establish it has stopped before migration')
        generation.update(alive=False, retired_at_utc=utc(),
                          retirement_reason='previous_runtime_no_longer_observable')
        retired.append({key: generation.get(key) for key in ('pid', 'start_ticks', 'task', 'boot_id')})
    if retired:
        recoveries = ledger.setdefault('runtime_recoveries', [])
        recoveries.append({'recovery_id': 'kernel-boot:' + BOOT_ID, 'at_utc': utc(),
                           'boot_id': BOOT_ID, 'previous_heartbeat_at_utc': previous_heartbeat,
                           'retired_generations': retired,
                           'preserved_measured_cpu_seconds': sum(g.get('charged_ticks', 0) for g in ledger['generations']) / TICKS,
                           'unobserved_tail_cpu_seconds': None, 'unobserved_tail_billed': False,
                           'paused_wall_time_is_not_cpu_usage': True})
    return retired


def extend_authorization(paths, additional, authorization_id):
    if (not math.isfinite(additional) or additional <= 0 or not authorization_id.strip()
            or len(authorization_id) > 200):
        raise ValueError('An extension requires positive finite CPU seconds and a stable authorization ID')
    with lock(paths.background / 'accounting.lock'):
        ledger = read(paths.ledger, empty_ledger())
        authorization = apply_authorization(paths, ledger)
        old_event = next((event for event in authorization['events']
                          if event['authorization_id'] == authorization_id), None)
        if old_event and old_event['additional_cpu_seconds'] != additional:
            raise ValueError('Authorization ID already exists with a different increment')
        retired = retire_previous_runtime(ledger)
        if not old_event:
            previous = authorization['cumulative_authorized_cpu_seconds']
            authorization['events'].append({'authorization_id': authorization_id,
                'additional_cpu_seconds': additional, 'previous_total_cpu_seconds': previous,
                'cumulative_total_cpu_seconds': previous + additional,
                'explicit_user_authorization': True, 'recorded_at_utc': utc()})
            authorization['cumulative_authorized_cpu_seconds'] = previous + additional
        authorization['runtime_recoveries'] = ledger.get('runtime_recoveries', [])
        write(paths.authorization, authorization)
        # The receipt is committed first. account() can reconcile a crash before
        # the following ledger write by accepting only the receipt's new suffix.
        ledger = update_accounting(paths, ledger)
        stopped = read(paths.stop)
        if stopped and stopped.get('authorized_cpu_seconds', float('inf')) < ledger['authorized_cpu_seconds']:
            if ledger['charged_cpu_seconds'] < ledger['authorized_cpu_seconds']:
                archive = paths.background / ('budget_exhausted_before_' + str(len(authorization['events'])) + '.json')
                if not archive.exists():
                    write(archive, stopped)
                paths.stop.unlink()
        return {'extension_applied': old_event is None, 'idempotent_replay': old_event is not None,
                'authorization_id': authorization_id,
                'authorized_cpu_seconds': ledger['authorized_cpu_seconds'],
                'charged_cpu_seconds': ledger['charged_cpu_seconds'],
                'remaining_cpu_seconds': ledger['remaining_cpu_seconds'],
                'retired_previous_runtime_generations': len(retired)}


def register(paths, task, child):
    value = identity(child.pid)
    if not value:
        raise RuntimeError('Worker process disappeared before registration')
    with lock(paths.background / 'accounting.lock'):
        ledger = read(paths.ledger, empty_ledger())
        apply_authorization(paths, ledger)
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
                      'reason': reason, 'authorized_cpu_seconds': ledger['authorized_cpu_seconds'],
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
            'authorization': authorization_receipt(paths),
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
                if budget_reached(paths, ledger):
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
                                    'authorized_cpu_seconds': ledger['authorized_cpu_seconds'],
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
                if budget_reached(paths, ledger):
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
    group.add_argument('--extend-cpu-seconds', type=float)
    parser.add_argument('--authorization-id', default='')
    args = parser.parse_args()
    paths = Paths(args.root)
    if not args.verify:
        paths.background.mkdir(parents=True, exist_ok=True)
    if args.extend_cpu_seconds is not None:
        if not args.authorization_id:
            parser.error('--extend-cpu-seconds requires --authorization-id')
        print(json.dumps(extend_authorization(paths, args.extend_cpu_seconds, args.authorization_id),
                         ensure_ascii=False, indent=2))
    elif args.verify:
        print(json.dumps(verify(paths), ensure_ascii=False, indent=2))
    elif args.publish:
        print(json.dumps(publish(paths), ensure_ascii=False, indent=2))
    elif args.start:
        ledger = account(paths)
        if budget_reached(paths, ledger):
            raise SystemExit('This authorization is exhausted; existing accounting was preserved')
        with lock(paths.background / 'launch.lock'):
            ensure_guard(paths)
            if not alive(read(paths.background / 'supervisor.json')):
                child = spawn_role(paths, 'supervise')
                print(json.dumps({'supervisor_pid': child.pid, 'detached': True,
                                  'authorized_cpu_seconds': ledger['authorized_cpu_seconds']}))
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
