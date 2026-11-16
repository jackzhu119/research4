#!/usr/bin/env python3
"""Detached IR consumer; repeat resumable scans without stall or retry limits.

The sole resource boundary is the user's 900000 cumulative CPU-second
authorization. CPU is sampled from Linux counters, including reaped children.
There is no wall-clock, idle, failure-count, or scientific-quality stop rule.
This does not keep a destroyed or suspended cloud environment alive.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

AUTHORIZED_CPU_SECONDS = 900000.0
POLL_SECONDS = 0.25
SCAN_INTERVAL_SECONDS = 60.0
PUBLISH_INTERVAL_SECONDS = 600.0
INTEGRITY_INTERVAL_SECONDS = 30.0
RUNNER_NAME = 'run_pyscf_ir_20261111.py'
SELF_NAME = 'background_pyscf_ir_20261111.py'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('w') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def proc_stat(pid):
    try:
        raw = Path(f'/proc/{int(pid)}/stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        ticks = os.sysconf('SC_CLK_TCK')
        return {'pid': int(pid), 'state': fields[0], 'ppid': int(fields[1]),
                'pgrp': int(fields[2]), 'sid': int(fields[3]),
                'own_cpu': (int(fields[11]) + int(fields[12])) / ticks,
                'waited_child_cpu': (int(fields[13]) + int(fields[14])) / ticks,
                'start_ticks': int(fields[19])}
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
        return None


def identity(pid):
    point = proc_stat(pid)
    return ({'pid': point['pid'], 'start_ticks': point['start_ticks'],
             'boot_id': boot_id()} if point else None)


def matching(expected):
    if not isinstance(expected, dict) or expected.get('boot_id') != boot_id():
        return False
    point = proc_stat(expected.get('pid', -1))
    return bool(point and point['state'] != 'Z'
                and point['start_ticks'] == expected.get('start_ticks'))


def signal_group(expected, signum=signal.SIGTERM):
    if not matching(expected):
        return False
    point = proc_stat(expected['pid'])
    if point['pgrp'] != point['pid'] or point['sid'] != point['pid']:
        raise ValueError('Refusing to signal a process that does not own its session')
    try:
        os.killpg(point['pid'], signum)
        return True
    except ProcessLookupError:
        return False


def group_snapshot(expected):
    if not matching(expected):
        return {}
    root = proc_stat(expected['pid'])
    if root['pgrp'] != root['pid'] or root['sid'] != root['pid']:
        raise ValueError('Managed process does not own its session')
    points = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdecimal():
            point = proc_stat(int(entry.name))
            if point and point['sid'] == root['pid'] and point['pgrp'] == root['pid']:
                points[point['pid']] = point
    return points


def cpu_snapshot(supervisor, worker=None, guard=None, publisher=None):
    """One coherent sum; reaped-child CPU may move between owners, not duplicate.

    Re-reading all counters avoids persisting a high-water value during the
    instant a child disappears and its CPU transfers to a parent's cutime.
    All managed processes own distinct sessions; their ordinary descendants
    must remain in that session. BLAS thread CPU is included by /proc counters.
    """
    if not matching(supervisor):
        return None
    owners = [p for p in (supervisor, worker, guard, publisher) if p]
    for _ in range(5):
        first = {pid: p for owner in owners for pid, p in group_snapshot(owner).items()}
        last = {pid: p for owner in owners for pid, p in group_snapshot(owner).items()}
        if first.keys() == last.keys() and all(
                first[pid]['start_ticks'] == last[pid]['start_ticks']
                and first[pid]['waited_child_cpu'] == last[pid]['waited_child_cpu']
                for pid in first):
            return sum(p['own_cpu'] + p['waited_child_cpu'] for p in last.values())
    return None


def session_cpu_delta(raw_counter, baseline):
    return None if raw_counter is None else max(0.0, float(raw_counter) - float(baseline))


def restart_baseline(paths, supervisor_identity):
    """Continue one Linux process across ledger sessions without recounting it."""
    counters = [0.0]
    for path in (paths.runtime / 'sessions').glob('*.json'):
        entry = read(path)
        if entry.get('supervisor_identity') == supervisor_identity:
            counters.append(float(entry.get('supervisor_raw_counter_at_final', 0.0)))
    return max(counters)


def cpu_capacity():
    capacity = float(len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity')
                     else (os.cpu_count() or 1))
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max':
            capacity = min(capacity, float(quota) / float(period))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    return capacity


@dataclass
class Paths:
    root: Path

    @property
    def output(self): return self.root / 'reports/pyscf_ir_20261111'
    @property
    def runtime(self): return self.output / '.background'
    @property
    def state(self): return self.runtime / 'state.json'
    @property
    def frozen(self): return self.runtime / 'frozen'
    @property
    def manifest(self): return self.output / 'source_manifest.json'
    @property
    def frozen_script(self): return self.frozen / 'scripts' / SELF_NAME
    @property
    def frozen_runner(self): return self.frozen / 'scripts' / RUNNER_NAME


def source_files(paths):
    """Freeze upstream Python modules as well as the new IR entry points."""
    files = list((paths.root / 'scripts').glob('*pyscf_ir*20261111.py'))
    files += list((paths.root / 'scripts').glob('pyscf_*20261111.py'))
    files += list(paths.root.glob('requirements*pyscf*.txt'))
    return {str(p.relative_to(paths.root)): sha(p) for p in sorted(files)
            if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts}


def freeze(paths):
    previous = read(paths.manifest)
    if previous is not None:
        check_integrity(paths)
        return previous
    files = source_files(paths)
    if 'scripts/' + RUNNER_NAME not in files or 'scripts/' + SELF_NAME not in files:
        raise ValueError('IR runner and supervisor must both be prepared in scripts/')
    paths.frozen.mkdir(parents=True, exist_ok=True)
    for relative in files:
        destination = paths.frozen / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paths.root / relative, destination)
        published = paths.output / 'frozen_source' / relative
        published.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(destination, published)
    value = {'schema': 'pyscf-ir-frozen-source-v1', 'files': files,
             'python_executable': sys.executable, 'python_version': sys.version,
             'frozen_at_utc': utc(), 'root_environment_variable': 'PYSCF_IR_REPOSITORY_ROOT',
             'published_frozen_source': 'reports/pyscf_ir_20261111/frozen_source'}
    write(paths.manifest, value)
    check_integrity(paths)
    return value


def check_integrity(paths):
    frozen = read(paths.manifest)
    if not frozen or not frozen.get('files'):
        raise ValueError('Frozen source manifest is missing')
    for relative, digest in frozen['files'].items():
        target = paths.frozen / relative
        if not target.is_file() or target.is_symlink() or sha(target) != digest:
            raise ValueError('Frozen scientific source changed: ' + str(target))
    return True


def workspace_changes(paths):
    manifest = read(paths.manifest, {})
    return [relative for relative, digest in manifest.get('files', {}).items()
            if not (paths.root / relative).is_file()
            or (paths.root / relative).is_symlink()
            or sha(paths.root / relative) != digest]


def record_preflight(paths, measured=None, source=None):
    previous = read(paths.output / 'preflight_cpu.json')
    if previous is not None:
        return previous
    if measured is None:
        stats = dict(line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
        measured = float(stats['usage_usec']) / 1e6
        source = 'Linux /sys/fs/cgroup/cpu.stat usage_usec sampled before detached launch'
    measured = float(measured)
    if not math.isfinite(measured) or measured < 0:
        raise ValueError('Preflight CPU must be a finite measured nonnegative number')
    value = {'observed_cpu_seconds': measured, 'conservative_cpu_seconds': measured,
             'measurement_source': source or 'Explicit measured CPU supplied to --start',
             'scope': 'This task environment cgroup, including preparation, installs, investigation, tests and launch preparation',
             'scope_authority': 'Primary agent confirmed the selected environment is dedicated to this IR task',
             'sampled_at_utc': utc(), 'invented_preflight_reserve_seconds': 0}
    write(paths.output / 'preflight_cpu.json', value)
    return value


def session_totals(paths, excluding=None):
    preflight = read(paths.output / 'preflight_cpu.json', {})
    measured = float(preflight.get('observed_cpu_seconds', 0.0))
    conservative = float(preflight.get('conservative_cpu_seconds', measured))
    if not math.isfinite(measured) or not math.isfinite(conservative) or measured < 0 or conservative < measured:
        raise ValueError('Preflight CPU ledger is invalid')
    for path in (paths.runtime / 'sessions').glob('*.json'):
        entry = read(path)
        if entry['session_id'] == excluding:
            continue
        actual = float(entry.get('observed_cpu_seconds', 0))
        charged = float(entry.get('conservative_cpu_seconds', actual))
        if not math.isfinite(actual) or not math.isfinite(charged) or actual < 0 or charged < actual:
            raise ValueError('CPU ledger is invalid: ' + str(path))
        measured += actual
        conservative += charged
    return measured, conservative


def eligible_files(paths):
    files = list((paths.root / 'scripts').glob('*pyscf_ir*20261111.py'))
    files += list((paths.root / 'tests').glob('test*pyscf_ir*20261111.py'))
    files += list((paths.root / 'docs').glob('PYSCF_IR*20261111.md'))
    files += list(paths.root.glob('requirements*pyscf*ir*.txt'))
    files += list((paths.root / 'data/research/pyscf_ir_20261111').rglob('*'))
    files += list(paths.output.rglob('*'))
    result = []
    for path in files:
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(paths.root)
        if any(part.startswith('.') or part == '__pycache__' for part in relative.parts):
            continue
        if path.name.endswith(('.log', '.pid', '.lock', '.chk', '.h5', '.pyc')) or '.tmp-' in path.name:
            continue
        result.append(relative.as_posix())
    return sorted(set(result))


def git(paths, *args, input_text=None):
    return subprocess.run(['git', *args], cwd=paths.root, text=True,
                          input=input_text, capture_output=True, timeout=45)


def current_branch(paths):
    result = git(paths, 'branch', '--show-current')
    branch = result.stdout.strip()
    if result.returncode or not branch:
        raise ValueError('A named IR branch is required')
    return branch


def publish(paths):
    """Publication failure is an independent retryable event, never compute stop."""
    state = read(paths.state, {})
    branch = state.get('branch')
    if not branch or current_branch(paths) != branch:
        raise ValueError('Publication branch changed; compute remains frozen and isolated')
    files = eligible_files(paths)
    staged = git(paths, 'diff', '--cached', '--name-only', '-z')
    if staged.returncode or any(p not in files for p in staged.stdout.split('\0') if p):
        raise ValueError('Unrelated staged files; publication deferred')
    if files:
        staged = git(paths, 'add', '--pathspec-from-file=-', '--pathspec-file-nul',
                     input_text='\0'.join(files) + '\0')
        if staged.returncode:
            raise RuntimeError('IR checkpoint staging failed')
    changed = git(paths, 'diff', '--cached', '--quiet')
    if changed.returncode == 1:
        result = git(paths, 'commit', '-m', 'Checkpoint independent PySCF IR calculations and audit')
        if result.returncode:
            raise RuntimeError('IR checkpoint commit failed')
    elif changed.returncode:
        raise RuntimeError('Cannot inspect checkpoint index')
    result = git(paths, 'push', 'origin', 'HEAD:refs/heads/' + branch)
    if result.returncode:
        raise RuntimeError('IR checkpoint push failed; local checkpoint retained')
    return git(paths, 'rev-parse', 'HEAD').stdout.strip()


def cleanup_own_descendants():
    """A timed-out Git command must not leave unmetered remote helpers behind."""
    own = identity(os.getpid())
    for point in group_snapshot(own).values():
        if point['pid'] != os.getpid():
            try:
                os.kill(point['pid'], signal.SIGKILL)
            except ProcessLookupError:
                pass


def publish_process(paths):
    try:
        commit = publish(paths)
        write(paths.runtime / 'publication.json', {'status': 'PUSHED', 'commit': commit, 'at_utc': utc()})
        return 0
    except Exception as error:
        write(paths.runtime / 'publication.json', {'status': 'RETRY_PENDING', 'error': repr(error), 'at_utc': utc()})
        return 1
    finally:
        cleanup_own_descendants()


def status(paths):
    state = read(paths.state, {})
    actual, conservative = session_totals(paths)
    return {'state': state, 'supervisor_alive': matching(state.get('supervisor_identity')),
            'worker_alive': matching(state.get('worker_identity')),
            'guard_alive': matching(state.get('guard_identity')),
            'publisher_alive': matching(state.get('publisher_identity')),
            'observed_cpu_seconds': actual, 'conservative_cpu_seconds': conservative,
            'authorized_cpu_seconds': AUTHORIZED_CPU_SECONDS,
            'remaining_cpu_seconds': max(0, AUTHORIZED_CPU_SECONDS - conservative),
            'cloud_lifetime_guaranteed': False}


def close_interrupted_session(paths, state, session_file):
    """Keep measurement and interruption uncertainty separate in the ledger."""
    for key in ('worker_identity', 'publisher_identity'):
        signal_group(state.get(key), signal.SIGKILL)
    old = read(session_file, {})
    if old.get('status') != 'RUNNING':
        return
    now = datetime.now(timezone.utc)
    if old.get('boot_id') == boot_id():
        gap = max(0.0, time.monotonic() - float(old.get('sample_monotonic', time.monotonic())))
    else:
        last = datetime.fromisoformat(old.get('sample_at_utc', utc()))
        gap = max(0.0, (now - last).total_seconds())
    observed = float(old.get('observed_cpu_seconds', 0))
    uncertainty = gap * float(old.get('cpu_capacity', cpu_capacity()))
    old.update(status='INTERRUPTED', observed_cpu_seconds=observed,
               conservative_cpu_seconds=observed + uncertainty,
               uncertain_upper_cpu_seconds=uncertainty, interrupted_at_utc=utc(),
               recovery_note='Lost CPU interval is an upper bound, not measured usage; no invented preflight charge.')
    write(session_file, old)


def guard(paths, session_id):
    session_file = paths.runtime / 'sessions' / (session_id + '.json')
    while True:
        state = read(paths.state, {})
        if state.get('session_id') != session_id or state.get('status') != 'RUNNING':
            return 0
        supervisor = state.get('supervisor_identity')
        if not matching(supervisor):
            close_interrupted_session(paths, state, session_file)
            if not (paths.runtime / 'STOP.json').exists() and not (paths.runtime / 'budget_stop.json').exists():
                try:
                    check_integrity(paths)
                    replacement = start_child(paths, paths.frozen_script,
                        ['--root', str(paths.root), '--supervise'], 'supervisor.log')
                    write(paths.runtime / 'recovery.json', {
                        'reason': 'Lost supervisor automatically replaced from frozen code',
                        'previous_session_id': session_id,
                        'replacement_identity': identity(replacement.pid), 'at_utc': utc()})
                except Exception as error:
                    write(paths.runtime / 'recovery.json', {'error': repr(error), 'at_utc': utc()})
            return 1
        raw_counter = cpu_snapshot(supervisor, state.get('worker_identity'),
                                   identity(os.getpid()), state.get('publisher_identity'))
        baseline = float(state.get('supervisor_counter_baseline', 0.0))
        measured = session_cpu_delta(raw_counter, baseline)
        old = read(session_file, {})
        if measured is not None:
            measured = max(measured, float(old.get('observed_cpu_seconds', 0)))
            write(session_file, {'session_id': session_id, 'status': 'RUNNING', 'boot_id': boot_id(),
                  'observed_cpu_seconds': measured, 'conservative_cpu_seconds': measured,
                  'uncertain_upper_cpu_seconds': 0.0, 'sample_monotonic': time.monotonic(),
                  'sample_at_utc': utc(), 'cpu_capacity': cpu_capacity(),
                  'supervisor_identity': supervisor,
                  'supervisor_counter_baseline': baseline,
                  'measurement': 'Coherent Linux own+waited-child counters of authenticated sessions'})
        _, used = session_totals(paths)
        if used >= AUTHORIZED_CPU_SECONDS:
            # No early safety reserve: stop at the measured authorization
            # boundary. Sampling/termination resolution is recorded honestly.
            write(paths.runtime / 'budget_stop.json', {'reason': 'AUTHORIZED_CPU_BOUNDARY',
                  'session_id': session_id, 'conservative_cpu_seconds': used,
                  'authorized_cpu_seconds': AUTHORIZED_CPU_SECONDS,
                  'poll_seconds': POLL_SECONDS, 'at_utc': utc()})
            for key in ('worker_identity', 'publisher_identity'):
                signal_group(state.get(key), signal.SIGKILL)
        time.sleep(POLL_SECONDS)


def start_child(paths, script, arguments, log_name):
    environment = os.environ.copy()
    environment['PYSCF_IR_REPOSITORY_ROOT'] = str(paths.root)
    environment['PYTHONUNBUFFERED'] = '1'
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[name] = '2'
    with (paths.runtime / log_name).open('ab', buffering=0) as stream:
        return subprocess.Popen([sys.executable, str(script), *arguments],
            cwd=paths.root, env=environment, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)


def _supervise_session(paths, counter_baseline=0.0):
    paths.runtime.mkdir(parents=True, exist_ok=True)
    lock = (paths.runtime / 'supervisor.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return 0
    old_state = read(paths.state, {})
    if matching(old_state.get('guard_identity')):
        signal_group(old_state.get('guard_identity'), signal.SIGTERM)
        deadline = time.monotonic() + 2
        while matching(old_state.get('guard_identity')) and time.monotonic() < deadline:
            time.sleep(.05)
        signal_group(old_state.get('guard_identity'), signal.SIGKILL)
        # Its last complete sample remains durable; account the intervening
        # interval conservatively before replacing this session's state.
    for previous in (paths.runtime / 'sessions').glob('*.json'):
        close_interrupted_session(paths, old_state, previous)
    check_integrity(paths)
    session_id = uuid.uuid4().hex
    state = {'schema': 'pyscf-ir-background-v1', 'status': 'RUNNING',
             'phase': 'STARTING', 'session_id': session_id,
             'branch': old_state.get('branch') or current_branch(paths), 'root': str(paths.root),
             'supervisor_identity': identity(os.getpid()), 'worker_identity': None,
             'supervisor_counter_baseline': float(counter_baseline),
             'guard_identity': None, 'publisher_identity': None,
             'started_at_utc': utc(), 'completed_scans': 0, 'failed_scans': 0,
             'authorization': 'User 2026-11-11 authorized 900000 cumulative CPU seconds',
             'authorized_cpu_seconds': AUTHORIZED_CPU_SECONDS,
             'no_idle_stall_retry_or_quality_exit_limit': True,
             'cloud_lifetime_guaranteed': False}
    write(paths.state, state)
    initial_raw = cpu_snapshot(state['supervisor_identity'])
    initial_cpu = session_cpu_delta(initial_raw, counter_baseline) or 0.0
    write(paths.runtime / 'sessions' / (session_id + '.json'), {
        'session_id': session_id, 'status': 'RUNNING', 'boot_id': boot_id(),
        'supervisor_identity': state['supervisor_identity'],
        'supervisor_counter_baseline': float(counter_baseline),
        'observed_cpu_seconds': initial_cpu, 'conservative_cpu_seconds': initial_cpu,
        'uncertain_upper_cpu_seconds': 0.0, 'sample_monotonic': time.monotonic(),
        'sample_at_utc': utc(), 'cpu_capacity': cpu_capacity(),
        'measurement': 'Initial supervisor CPU snapshot before auxiliary meter launch'})
    guard_process = start_child(paths, paths.frozen_script,
                                ['--root', str(paths.root), '--guard', session_id], 'guard.log')
    state['guard_identity'] = identity(guard_process.pid)
    worker = publisher = None
    finishing = False
    publisher_is_final = False
    next_scan = next_integrity = 0.0
    next_publish = time.monotonic() + PUBLISH_INTERVAL_SECONDS
    session_file = paths.runtime / 'sessions' / (session_id + '.json')
    terminal = 'STOPPED'
    should_stop = False

    def on_stop(signum, frame):
        nonlocal should_stop
        should_stop = True

    signal.signal(signal.SIGTERM, on_stop)
    signal.signal(signal.SIGINT, on_stop)

    def update(**values):
        state.update(values, heartbeat_at_utc=utc())
        write(paths.state, state)

    try:
        while True:
            now = time.monotonic()
            _, used = session_totals(paths)
            if used >= AUTHORIZED_CPU_SECONDS or (paths.runtime / 'budget_stop.json').exists():
                terminal = 'AUTHORIZED_CPU_BOUNDARY'
                break
            if should_stop or (paths.runtime / 'STOP.json').exists():
                terminal = 'STOPPED_BY_USER'
                break
            if now >= next_integrity:
                check_integrity(paths)
                next_integrity = now + INTEGRITY_INTERVAL_SECONDS
            if guard_process.poll() is not None:
                # Restart a lost auxiliary meter rather than abandoning the
                # scientific queue. Supervisor counters retain reaped CPU.
                guard_process.wait()
                guard_process = start_child(paths, paths.frozen_script,
                    ['--root', str(paths.root), '--guard', session_id], 'guard.log')
                update(guard_identity=identity(guard_process.pid), meter_restart_at_utc=utc())
            if worker is not None and worker.poll() is not None:
                code = worker.wait()
                state['completed_scans'] += 1
                if code:
                    state['failed_scans'] += 1
                update(worker_identity=None, last_scan_returncode=code,
                       last_scan_finished_at_utc=utc(), phase='WAITING_NEXT_SCAN')
                worker = None
                next_scan = now + SCAN_INTERVAL_SECONDS
                if code == 0 and read(paths.output / 'summary.json', {}).get('status') == 'COMPLETE':
                    finishing = True
                    next_publish = now
                    update(phase='FINAL_PUBLICATION_PENDING')
            if publisher is not None and publisher.poll() is not None:
                publication_code = publisher.wait()
                publisher = None
                update(publisher_identity=None)
                if finishing and publisher_is_final and publication_code == 0:
                    terminal = 'COMPLETE'
                    break
                if finishing:
                    next_publish = now + (0 if not publisher_is_final else SCAN_INTERVAL_SECONDS)
            if not finishing and worker is None and now >= next_scan:
                check_integrity(paths)
                try:
                    worker = start_child(paths, paths.frozen_runner, ['--once'], 'worker.log')
                    update(worker_identity=identity(worker.pid), phase='SCANNING')
                except OSError as error:
                    update(last_spawn_error=repr(error), phase='RETRY_PENDING')
                    next_scan = now + SCAN_INTERVAL_SECONDS
            if publisher is None and now >= next_publish:
                write(paths.output / 'cpu_accounting.json', {
                    'authorized_cpu_seconds': AUTHORIZED_CPU_SECONDS,
                    'observed_cpu_seconds': session_totals(paths)[0],
                    'conservative_cpu_seconds': used, 'at_utc': utc(),
                    'not_wall_time': True, 'early_cpu_reserve_seconds': 0,
                    'sampling_interval_seconds': POLL_SECONDS})
                try:
                    publisher = start_child(paths, paths.frozen_script,
                        ['--root', str(paths.root), '--publish'], 'publication.log')
                    publisher_is_final = finishing
                    update(publisher_identity=identity(publisher.pid))
                except OSError as error:
                    update(last_publication_spawn_error=repr(error))
                next_publish = now + PUBLISH_INTERVAL_SECONDS
            update()
            time.sleep(POLL_SECONDS)
    except Exception as error:
        try:
            check_integrity(paths)
        except (ValueError, OSError):
            terminal = 'SCIENTIFIC_INTEGRITY_CHANGED'
            update(last_error=repr(error))
        else:
            # The outer supervisor loop starts another accounted session;
            # there is no maximum number of such restarts.
            update(last_error=repr(error), phase='SUPERVISOR_RESTART_PENDING')
            terminal = 'SUPERVISOR_INTERRUPTED'
    finally:
        for child, key in ((worker, 'worker_identity'), (publisher, 'publisher_identity')):
            if child is not None:
                signal_group(state.get(key), signal.SIGKILL)
                child.wait()
        update(status=terminal, worker_identity=None, publisher_identity=None,
               phase='STOPPED', finished_at_utc=utc())
        guard_process.wait(timeout=5)
        old = read(session_file, {})
        raw_counter = cpu_snapshot(state['supervisor_identity'])
        measured = session_cpu_delta(raw_counter, counter_baseline)
        measured = max(float(old.get('observed_cpu_seconds', 0)), float(measured or 0))
        write(session_file, {'session_id': session_id, 'status': terminal,
              'boot_id': boot_id(), 'observed_cpu_seconds': measured,
              'supervisor_identity': state['supervisor_identity'],
              'supervisor_counter_baseline': float(counter_baseline),
              'supervisor_raw_counter_at_final': float(raw_counter or 0.0),
              'conservative_cpu_seconds': measured, 'uncertain_upper_cpu_seconds': 0.0,
              'sample_monotonic': time.monotonic(), 'sample_at_utc': utc(),
              'measurement': 'Final supervisor self+waited-child CPU, including exited guard'})
        actual, charged = session_totals(paths)
        write(paths.output / 'cpu_accounting.json', {
            'authorized_cpu_seconds': AUTHORIZED_CPU_SECONDS, 'observed_cpu_seconds': actual,
            'conservative_cpu_seconds': charged, 'status': terminal, 'at_utc': utc(),
            'not_wall_time': True, 'early_cpu_reserve_seconds': 0,
            'sampling_interval_seconds': POLL_SECONDS,
            'boundary_overshoot_cpu_seconds': max(0.0, charged - AUTHORIZED_CPU_SECONDS)})
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    return 0 if terminal in {'AUTHORIZED_CPU_BOUNDARY', 'STOPPED_BY_USER', 'COMPLETE'} else 1


def supervise(paths):
    while True:
        try:
            check_integrity(paths)
        except (ValueError, OSError) as error:
            state = read(paths.state, {})
            state.update(status='SCIENTIFIC_INTEGRITY_CHANGED', last_error=repr(error),
                         finished_at_utc=utc())
            write(paths.state, state)
            return 1
        try:
            # _supervise_session may repeat in this same process. Its Linux
            # self+waited-child counters are cumulative across those repeats.
            baseline = restart_baseline(paths, identity(os.getpid()))
            code = _supervise_session(paths, counter_baseline=baseline)
            state = read(paths.state, {})
            if state.get('status') in {'AUTHORIZED_CPU_BOUNDARY', 'STOPPED_BY_USER',
                                       'SCIENTIFIC_INTEGRITY_CHANGED', 'RUNNING', 'COMPLETE'}:
                return code
        except Exception as error:
            write(paths.runtime / 'supervisor_retry.json', {
                'error': repr(error), 'at_utc': utc(), 'retry_limit': None})
        time.sleep(1)


def start(paths, preflight_cpu_seconds=None, preflight_cpu_source=None):
    paths.runtime.mkdir(parents=True, exist_ok=True)
    launch_lock = (paths.runtime / 'launch.lock').open('a+')
    fcntl.flock(launch_lock, fcntl.LOCK_EX)
    try:
        previous = status(paths)
        if previous['supervisor_alive'] or previous['state'].get('status') == 'COMPLETE':
            return previous
        if (paths.runtime / 'STOP.json').exists():
            return {**previous, 'start_blocked': 'Explicit STOP.json must be removed to resume'}
        freeze(paths)
        record_preflight(paths, preflight_cpu_seconds, preflight_cpu_source)
        child = start_child(paths, paths.frozen_script,
                            ['--root', str(paths.root), '--supervise'], 'supervisor.log')
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            value = status(paths)
            if value['supervisor_alive'] and value['guard_alive']:
                return value
            if child.poll() is not None:
                return {**value, 'start_returncode': child.returncode}
            time.sleep(.1)
        return {**status(paths), 'launch_handshake': 'PENDING',
                'launched_pid': child.pid, 'worker_terminated': False}
    finally:
        fcntl.flock(launch_lock, fcntl.LOCK_UN)
        launch_lock.close()


def stop(paths):
    write(paths.runtime / 'STOP.json', {'reason': 'Explicit user stop', 'at_utc': utc()})
    state = read(paths.state, {})
    return {'supervisor_signalled': signal_group(state.get('supervisor_identity'))}


def verify(paths):
    value = status(paths)
    try:
        check_integrity(paths)
        integrity = True
    except (ValueError, OSError):
        integrity = False
    active = value['supervisor_alive'] and value['guard_alive']
    complete = value['state'].get('status') == 'COMPLETE'
    value['verified'] = bool(integrity and (active or complete))
    value['source_integrity_passed'] = integrity
    value['workspace_source_changes_ignored_by_frozen_worker'] = workspace_changes(paths)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
                        default=Path(os.environ.get('PYSCF_IR_REPOSITORY_ROOT', Path(__file__).resolve().parents[1])))
    parser.add_argument('--preflight-cpu-seconds', type=float)
    parser.add_argument('--preflight-cpu-source')
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ('start', 'verify', 'stop', 'status', 'supervise', 'publish'):
        group.add_argument('--' + flag, action='store_true')
    group.add_argument('--guard')
    args = parser.parse_args()
    paths = Paths(args.root.resolve())
    if args.supervise:
        return supervise(paths)
    if args.guard:
        return guard(paths, args.guard)
    if args.publish:
        return publish_process(paths)
    if args.start:
        result = start(paths, args.preflight_cpu_seconds, args.preflight_cpu_source)
    elif args.stop:
        result = stop(paths)
    elif args.verify:
        result = verify(paths)
    else:
        result = status(paths)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if args.verify and not result['verified'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
