#!/usr/bin/env python3
"""Detached, cumulatively budgeted PySCF recovery with inspectable checkpoints.

CPU time is never confused with elapsed time. Linux process counters include
reaped descendants; active worker-group counters are added only while live.
An independent guard checks the budget even while Git is blocked. Detached
processes survive the shell, not suspension/destruction of the cloud machine.
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
import signal
import subprocess
import sys
import time
import uuid

BRANCH = 'research/pyscf-publication-recovery-20261111'
AUTHORIZED_CPU = 900000.0
COMPUTE_RESERVE = 300.0
FINAL_RESERVE = 30.0
POLL_SECONDS = 1.0
CHECKPOINT_SECONDS = 600.0
INTEGRITY_ERROR = 42
RETRYABLE = 2


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
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
    """Linux own+waited-child CPU, never guessed from percentage CPU."""
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
    value = proc_stat(pid)
    if value is None:
        return None
    return {'pid': value['pid'], 'start_ticks': value['start_ticks'], 'boot_id': boot_id()}


def matching_process(expected):
    if not isinstance(expected, dict) or expected.get('boot_id') != boot_id():
        return False
    actual = proc_stat(expected.get('pid', -1))
    return bool(actual and actual['state'] != 'Z'
                and actual['start_ticks'] == expected.get('start_ticks'))


def _group_snapshot(group_id):
    members = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        point = proc_stat(int(entry.name))
        if point and point['pgrp'] == group_id and point['sid'] == group_id:
            members[point['pid']] = point
    return members


def group_cpu(expected):
    """Sum self+reaped descendants across one authenticated worker session.

    A parent's waited-child counters exclude children still present in /proc.
    No arbitrary command-name or recycled-PID match authorizes signalling.
    """
    if not matching_process(expected):
        return 0.0
    root = proc_stat(expected['pid'])
    if root['pgrp'] != root['pid'] or root['sid'] != root['pid']:
        raise ValueError('Managed worker must own its process group and session')
    for _ in range(5):
        first = _group_snapshot(root['pid'])
        last = _group_snapshot(root['pid'])
        # Reaping inside the worker group can otherwise double-count a child
        # read just before its CPU transfers to the parent's wait counters.
        coherent = first.keys() == last.keys() and all(
            first[pid]['start_ticks'] == last[pid]['start_ticks']
            and first[pid]['waited_child_cpu'] == last[pid]['waited_child_cpu']
            for pid in first)
        if coherent:
            return sum(p['own_cpu'] + p['waited_child_cpu'] for p in last.values())
    return None


def session_cpu(supervisor, worker=None, guard=None):
    """Coherent parent/live-group reading avoids reaping double counts."""
    if not matching_process(supervisor):
        return None
    for _ in range(5):
        first = proc_stat(supervisor['pid'])
        active = group_cpu(worker) if worker else 0.0
        if active is None:
            continue
        guard_stat = proc_stat(guard['pid']) if matching_process(guard) else None
        last = proc_stat(supervisor['pid'])
        if first and last and first['waited_child_cpu'] == last['waited_child_cpu']:
            return last['own_cpu'] + last['waited_child_cpu'] + active + (
                guard_stat['own_cpu'] + guard_stat['waited_child_cpu'] if guard_stat else 0.0)
    # A changing wait counter is a short race. Retry next poll; do not record
    # an inflated high-water value by counting the same child twice.
    return None


def terminate_group(expected, signum=signal.SIGTERM):
    if not matching_process(expected):
        return False
    observed = proc_stat(expected['pid'])
    if observed['pgrp'] != expected['pid'] or observed['sid'] != expected['pid']:
        raise ValueError('Refusing to signal an unauthenticated process group')
    os.killpg(expected['pid'], signum)
    return True


def close_orphan_worker(expected):
    """Terminate an authenticated orphan and account through its exit interval."""
    terminate_group(expected)
    deadline = time.monotonic() + 5
    while matching_process(expected) and time.monotonic() < deadline:
        time.sleep(.1)
    if matching_process(expected):
        terminate_group(expected, signal.SIGKILL)
        time.sleep(.1)


def conservative_recovery_charge(last_observation, current_boot, now_monotonic, cpu_capacity):
    """A lost accounting interval is uncertainty, never reported as measured CPU."""
    actual = float(last_observation.get('observed_cpu_seconds', 0.0))
    if last_observation.get('boot_id') != current_boot:
        return {'observed_cpu_seconds': actual, 'conservative_cpu_seconds': AUTHORIZED_CPU,
                'uncertain_upper_cpu_seconds': max(0.0, AUTHORIZED_CPU - actual),
                'reason': 'Environment reboot lost the unsampled interval; require a CPU audit before further compute'}
    gap = max(0.0, now_monotonic - float(last_observation.get('sample_monotonic', now_monotonic)))
    upper = gap * float(cpu_capacity)
    return {'observed_cpu_seconds': actual,
            'conservative_cpu_seconds': actual + upper,
            'uncertain_upper_cpu_seconds': upper,
            'reason': 'Interrupted session: all elapsed unsampled CPU capacity is reserved conservatively'}


@dataclass
class Paths:
    root: Path

    @property
    def output(self): return self.root / 'reports/pyscf_publication_20261111'
    @property
    def runtime(self): return self.output / '.background'
    @property
    def state(self): return self.runtime / 'state.json'
    @property
    def python(self): return self.root / '.venv-pyscf/bin/python'
    @property
    def runner(self): return self.root / 'scripts/run_pyscf_publication_20261111.py'
    @property
    def script(self): return self.root / 'scripts/background_pyscf_publication_20261111.py'


def scientific_snapshot(paths):
    files = list((paths.root / 'scripts').glob('*pyscf_publication*20261111.py'))
    files += [paths.root / 'scripts' / name for name in
              ('pyscf_repair_core_20261111.py', 'pyscf_repair_modes_20261111.py')]
    for folder in ('data/research/pyscf_publication_20261111',
                   'data/research/pyscf_repair_20261111/inputs',
                   'data/research/pyscf_repair_20261111/history_controls'):
        files += [p for p in (paths.root / folder).rglob('*') if p.is_file()]
    files += list(paths.root.glob('requirements-pyscf-publication*.txt'))
    if (paths.output / 'preflight_cpu.json').is_file():
        files.append(paths.output / 'preflight_cpu.json')
    result = {str(p.relative_to(paths.root)): sha(p) for p in sorted(set(files))
              if p.is_file() and not p.is_symlink() and '.local' not in p.parts}
    if not paths.runner.is_file() or not paths.python.is_file():
        raise ValueError('Prepared runner or PySCF virtual environment is missing')
    return result


def eligible_files(paths):
    selected = []
    files = list((paths.root / 'scripts').glob('*pyscf_publication*20261111.py'))
    files += list((paths.root / 'tests').glob('test_pyscf_publication*20261111.py'))
    files += list((paths.root / 'docs').glob('PYSCF_PUBLICATION*20261111.md'))
    files += list(paths.root.glob('requirements-pyscf-publication*.txt'))
    files += list((paths.root / 'data/research/pyscf_publication_20261111').rglob('*'))
    files += list(paths.output.rglob('*'))
    for path in files:
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(paths.root)
        if path == paths.output / '.gitignore':
            selected.append(relative.as_posix())
        elif (not any(p.startswith('.') or p in {'__pycache__', 'cache', 'checkpoints'} for p in relative.parts)
                and not path.name.endswith(('.log', '.lock', '.pid', '.tmp', '.chk', '.h5', '.pyc'))):
            selected.append(relative.as_posix())
    return sorted(set(selected))


def git(paths, *args, input_text=None, timeout=45):
    return subprocess.run(['git', *args], cwd=paths.root, text=True,
                          input=input_text, capture_output=True, timeout=timeout)


def require_branch(paths):
    current = git(paths, 'branch', '--show-current')
    if current.returncode or current.stdout.strip() != BRANCH:
        raise ValueError('Checkout must remain on ' + BRANCH)


def publish(paths, title):
    """Bounded, non-force publication; computation does not depend on Git health."""
    require_branch(paths)
    files = eligible_files(paths)
    staged = git(paths, 'diff', '--cached', '--name-only', '-z')
    if staged.returncode or any(p not in files for p in staged.stdout.split('\0') if p):
        raise ValueError('Unrelated staged files; refusing to commit them')
    if files:
        result = git(paths, 'add', '--pathspec-from-file=-', '--pathspec-file-nul',
                     input_text='\0'.join(files) + '\0')
        if result.returncode:
            raise RuntimeError('Git checkpoint staging failed')
    changed = git(paths, 'diff', '--cached', '--quiet')
    if changed.returncode == 1:
        result = git(paths, 'commit', '-m', title)
        if result.returncode:
            raise RuntimeError('Git checkpoint commit failed')
    elif changed.returncode:
        raise RuntimeError('Cannot inspect the Git index')
    result = git(paths, 'push', 'origin', 'HEAD:refs/heads/' + BRANCH)
    if result.returncode:
        raise RuntimeError('Git checkpoint push failed; local commit retained')
    return git(paths, 'rev-parse', 'HEAD').stdout.strip()


def preflight_charge(paths):
    data = read(paths.output / 'preflight_cpu.json', {})
    measured = data.get('actual_cpu_seconds')
    observed = 0.0 if measured is None else float(measured)
    configured = float(data.get('conservative_cpu_seconds', 1000.0))
    if not math.isfinite(observed) or not math.isfinite(configured) or observed < 0 or configured < 0:
        raise ValueError('Invalid preflight CPU ledger')
    conservative = max(observed, configured)
    return {'observed_cpu_seconds': observed, 'conservative_cpu_seconds': conservative,
            'uncertain_upper_cpu_seconds': conservative - observed,
            'measurement': data.get('measurement', '1000 CPU-second conservative preflight reserve; unmeasured remainder is not actual CPU')}


def cpu_capacity():
    """An upper CPU-seconds-per-second capacity, respecting a cgroup quota."""
    available = float(len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1))
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max':
            available = min(available, float(quota) / float(period))
    except (FileNotFoundError, ValueError, ZeroDivisionError):
        pass
    return max(available, 1.0)


def ledger_totals(paths, excluding=None):
    preflight = preflight_charge(paths)
    actual, conservative = preflight['observed_cpu_seconds'], preflight['conservative_cpu_seconds']
    for path in (paths.runtime / 'sessions').glob('*.json'):
        entry = read(path)
        if entry['session_id'] == excluding:
            continue
        observed = float(entry.get('observed_cpu_seconds', 0))
        charged = float(entry.get('conservative_cpu_seconds', observed))
        if not math.isfinite(observed) or not math.isfinite(charged) or observed < 0 or charged < observed:
            raise ValueError('Invalid persisted CPU session ledger: ' + str(path))
        actual += observed
        conservative += charged
    return actual, conservative


def progress_marker(paths):
    progress = read(paths.output / 'progress.json', {})
    if 'checkpoint_token' in progress:
        return json.dumps(progress['checkpoint_token'], sort_keys=True)
    sealed = {str(p.relative_to(paths.output)): sha(p)
              for p in paths.output.rglob('COMPLETE.json') if p.is_file()}
    return json.dumps(sealed, sort_keys=True)


def status(paths):
    state = read(paths.state, {})
    actual, conservative = ledger_totals(paths)
    return {'state': state, 'supervisor_alive': matching_process(state.get('supervisor_identity')),
            'worker_alive': matching_process(state.get('worker_identity')),
            'guard_alive': matching_process(state.get('guard_identity')),
            'authorized_cpu_seconds': AUTHORIZED_CPU, 'observed_cpu_seconds': actual,
            'conservative_cpu_seconds': conservative,
            'remaining_conservative_cpu_seconds': max(0.0, AUTHORIZED_CPU - conservative),
            'cloud_lifetime_guaranteed': False,
            'progress': read(paths.output / 'progress.json', {})}


def stop(paths):
    state = read(paths.state, {})
    write(paths.runtime / 'STOP.json', {'at_utc': utc(), 'source': 'explicit user stop'})
    return terminate_group(state.get('worker_identity'))


def guard(paths, session_id):
    """Independent budget protection remains active while Git or worker blocks."""
    session_path = paths.runtime / 'sessions' / (session_id + '.json')
    own = identity(os.getpid())
    while True:
        state = read(paths.state, {})
        if state.get('session_id') != session_id:
            return 0
        if state.get('status') in {'DONE', 'BUDGET_EXHAUSTED', 'STOPPED', 'BLOCKED_INTEGRITY', 'BLOCKED_ERROR'}:
            return 0
        supervisor, worker = state.get('supervisor_identity'), state.get('worker_identity')
        if not matching_process(supervisor):
            close_orphan_worker(worker)
            last = read(session_path, {'observed_cpu_seconds': 0, 'boot_id': boot_id()})
            repair = conservative_recovery_charge(last, boot_id(), time.monotonic(), cpu_capacity())
            last.update(repair, status='INTERRUPTED_GUARD_CLOSED', at_utc=utc())
            write(session_path, last)
            return 1
        measured = session_cpu(supervisor, worker, own)
        old = read(session_path, {})
        if measured is not None:
            measured = max(float(old.get('observed_cpu_seconds', 0)), measured)
            write(session_path, {'session_id': session_id, 'boot_id': boot_id(),
                  'observed_cpu_seconds': measured, 'conservative_cpu_seconds': measured,
                  'uncertain_upper_cpu_seconds': 0, 'sample_monotonic': time.monotonic(),
                  'sample_at_utc': utc(), 'status': 'RUNNING',
                  'measurement': 'Linux supervisor own+waited-child CPU + active authenticated worker session + guard own CPU; stable wait-counter snapshot'})
        _, conservative = ledger_totals(paths)
        is_compute = state.get('phase') == 'computing'
        threshold = AUTHORIZED_CPU - (COMPUTE_RESERVE if is_compute else FINAL_RESERVE)
        if conservative >= threshold or (paths.runtime / 'STOP.json').exists():
            reason = 'CPU_BUDGET' if conservative >= threshold else 'USER_STOP'
            write(paths.runtime / 'guard_stop.json', {'session_id': session_id, 'reason': reason,
                  'phase': state.get('phase'), 'worker_identity': worker,
                  'conservative_cpu_seconds': conservative, 'at_utc': utc()})
            terminate_group(worker)
        if conservative >= AUTHORIZED_CPU - (10 * cpu_capacity() + 10):
            # Final hard protection also covers a blocked Git child in the
            # supervisor's own authenticated session. The guard is detached
            # separately, so it can close the interrupted-session receipt.
            terminate_group(supervisor, signal.SIGKILL)
        time.sleep(POLL_SECONDS)


def recover_old_sessions(paths):
    for path in (paths.runtime / 'sessions').glob('*.json'):
        entry = read(path)
        if entry.get('status') != 'RUNNING':
            continue
        previous = read(paths.state, {})
        if matching_process(previous.get('supervisor_identity')):
            raise ValueError('Previous supervisor is still alive')
        if matching_process(previous.get('guard_identity')):
            deadline = time.monotonic() + 10
            while matching_process(previous.get('guard_identity')) and time.monotonic() < deadline:
                time.sleep(.2)
            entry = read(path)
            if entry.get('status') != 'RUNNING':
                continue
        close_orphan_worker(previous.get('worker_identity'))
        repair = conservative_recovery_charge(entry, boot_id(), time.monotonic(), cpu_capacity())
        entry.update(repair, status='INTERRUPTED_RECOVERY_CLOSED', at_utc=utc())
        write(path, entry)


def supervise(paths):
    require_branch(paths)
    paths.runtime.mkdir(parents=True, exist_ok=True)
    lock = (paths.runtime / 'supervisor.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return 0
    recover_old_sessions(paths)
    previous = read(paths.state, {})
    frozen = scientific_snapshot(paths)
    if previous and previous.get('scientific_files') != frozen:
        raise ValueError('Frozen scientific code/inputs changed; use a new run directory')
    session_id = uuid.uuid4().hex
    state = {'schema': 'pyscf-publication-background-v1', 'session_id': session_id,
             'status': 'RUNNING', 'phase': 'starting', 'supervisor_identity': identity(os.getpid()),
             'worker_identity': None, 'guard_identity': None, 'scientific_files': frozen,
             'authorized_cpu_seconds': AUTHORIZED_CPU, 'authorization': 'User 2026-11-11 authorized 900000 CPU seconds',
             'started_at_utc': utc(), 'cloud_lifetime_guaranteed': False, 'phase_attempts': [],
             'branch': BRANCH, 'qualification_is_independent_of_completion': True}
    write(paths.state, state)
    logfile = (paths.runtime / 'guard.log').open('ab', buffering=0)
    guard_process = subprocess.Popen([str(paths.python), str(paths.script), '--root', str(paths.root),
                   '--guard', session_id], cwd=paths.root, stdin=subprocess.DEVNULL,
                   stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    logfile.close()
    state['guard_identity'] = identity(guard_process.pid)

    def update(**values):
        state.update(values, heartbeat_at_utc=utc())
        write(paths.state, state)

    def check_frozen():
        require_branch(paths)
        if scientific_snapshot(paths) != frozen:
            raise ValueError('Frozen scientific code or inputs changed')

    def run_phase(phase, flag):
        check_frozen()
        update(phase=phase, worker_identity=None)
        environment = os.environ.copy()
        environment.update({key: '2' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
        environment['PYTHONUNBUFFERED'] = '1'
        command = [str(paths.python), str(paths.runner), flag]
        with (paths.runtime / (phase + '.log')).open('ab', buffering=0) as log:
            child = subprocess.Popen(command, cwd=paths.root, env=environment,
                  stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                  start_new_session=True, close_fds=True)
        child_identity = identity(child.pid)
        update(worker_identity=child_identity)
        checkpoint = time.monotonic()
        stop_started = None
        while child.poll() is None:
            check_frozen()
            if guard_process.poll() is not None:
                close_orphan_worker(child_identity)
                child.wait(timeout=5)
                raise RuntimeError('CPU guard exited unexpectedly; worker stopped for budget protection')
            update()
            stop_record = read(paths.runtime / 'guard_stop.json', {})
            should_stop = (stop_record.get('session_id') == session_id
                           and stop_record.get('phase') == phase
                           and stop_record.get('worker_identity') == child_identity)
            if should_stop:
                if stop_started is None:
                    terminate_group(child_identity)
                    stop_started = time.monotonic()
                elif time.monotonic() - stop_started > 15:
                    terminate_group(child_identity, signal.SIGKILL)
            if phase == 'computing' and time.monotonic() - checkpoint >= CHECKPOINT_SECONDS:
                try:
                    update(last_checkpoint_sha=publish(paths, 'Checkpoint PySCF publication recovery (partial)'),
                           last_publication_error=None)
                except (RuntimeError, subprocess.TimeoutExpired) as error:
                    update(last_publication_error=str(error))
                checkpoint = time.monotonic()
            time.sleep(POLL_SECONDS)
        code = child.wait()
        state['phase_attempts'].append({'phase': phase, 'returncode': code, 'finished_at_utc': utc()})
        update(worker_identity=None)
        return code

    terminal = 'BLOCKED_ERROR'
    exit_code = 1
    try:
        update()
        _, used = ledger_totals(paths)
        if (paths.runtime / 'STOP.json').exists():
            terminal = 'STOPPED'
        elif used >= AUTHORIZED_CPU - COMPUTE_RESERVE:
            terminal = 'BUDGET_EXHAUSTED'
        else:
            consecutive_stalls = 0
            while True:
                marker_before = progress_marker(paths)
                code = run_phase('computing', '--resume')
                stop_receipt = read(paths.runtime / 'guard_stop.json', {})
                if stop_receipt.get('session_id') == session_id:
                    terminal = 'BUDGET_EXHAUSTED' if stop_receipt['reason'] == 'CPU_BUDGET' else 'STOPPED'
                    break
                if code == 0:
                    terminal = 'DONE'
                    break
                if code == INTEGRITY_ERROR:
                    terminal = 'BLOCKED_INTEGRITY'
                    break
                consecutive_stalls = consecutive_stalls + 1 if progress_marker(paths) == marker_before else 0
                if code != RETRYABLE or consecutive_stalls >= 3:
                    update(last_error=f'Runner returned {code}; deterministic/repeated failure requires inspection')
                    terminal = 'BLOCKED_ERROR'
                    break
                time.sleep(2)
        # Summary does not turn partial scientific data into qualified results.
        if terminal != 'BLOCKED_INTEGRITY':
            update(phase='summarizing')
            code = run_phase('summarizing', '--summarize')
            if code and terminal == 'DONE':
                terminal = 'BLOCKED_ERROR'
            if terminal == 'DONE' and run_phase('verifying', '--verify'):
                terminal = 'BLOCKED_ERROR'
        actual, charged = ledger_totals(paths)
        write(paths.output / 'cpu_accounting.json', {'authorized_cpu_seconds': AUTHORIZED_CPU,
              'observed_cpu_seconds': actual, 'conservative_cpu_seconds': charged,
              'uncertain_upper_cpu_seconds': max(0, charged - actual), 'snapshot_at_utc': utc(),
              'preflight': preflight_charge(paths), 'not_wall_time': True})
        update(phase='publishing')
        try:
            update(final_commit_sha=publish(paths, 'Save PySCF publication recovery results and CPU accounting'),
                   publication='PUSHED')
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            update(publication='RETRY_REQUIRED', last_publication_error=str(error))
        exit_code = 0 if terminal in {'DONE', 'BUDGET_EXHAUSTED', 'STOPPED'} else (42 if terminal == 'BLOCKED_INTEGRITY' else 1)
        update(status=terminal, phase='completed', finished_at_utc=utc())
    except ValueError as error:
        terminate_group(state.get('worker_identity'))
        update(status='BLOCKED_INTEGRITY', last_error=str(error))
        terminal, exit_code = 'BLOCKED_INTEGRITY', 42
    except Exception as error:
        terminate_group(state.get('worker_identity'))
        update(status='BLOCKED_ERROR', last_error=repr(error))
        terminal, exit_code = 'BLOCKED_ERROR', 1
    finally:
        # Guard sees terminal state and exits; its CPU then moves exactly once
        # into supervisor waited-child usage.
        update(status=terminal, worker_identity=None)
        try:
            guard_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if matching_process(state.get('guard_identity')):
                os.kill(guard_process.pid, signal.SIGTERM)
            guard_process.wait(timeout=5)
        observed = session_cpu(state['supervisor_identity'])
        session_path = paths.runtime / 'sessions' / (session_id + '.json')
        old = read(session_path, {})
        observed = max(float(old.get('observed_cpu_seconds', 0)), float(observed or 0))
        write(session_path, {'session_id': session_id, 'boot_id': boot_id(),
              'observed_cpu_seconds': observed, 'conservative_cpu_seconds': observed,
              'uncertain_upper_cpu_seconds': 0, 'status': terminal,
              'finished_at_utc': utc(), 'sample_monotonic': time.monotonic(),
              'measurement': 'Final supervisor Linux self+waited-child counters, including exited guard'})
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    return exit_code


def launch(paths):
    require_branch(paths)
    scientific_snapshot(paths)
    current = status(paths)
    if current['supervisor_alive'] or current['state'].get('status') == 'DONE':
        return current
    paths.runtime.mkdir(parents=True, exist_ok=True)
    with (paths.runtime / 'supervisor.log').open('ab', buffering=0) as log:
        child = subprocess.Popen([str(paths.python), str(paths.script), '--root', str(paths.root), '--supervise'],
              cwd=paths.root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
              start_new_session=True, close_fds=True)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        result = status(paths)
        if result['supervisor_alive'] and result['state'].get('guard_identity'):
            return result
        if child.poll() is not None:
            return result
        time.sleep(.2)
    return {'launch_handshake': 'PENDING', 'pid': child.pid, 'computation_terminated': False}


def verify(paths):
    state = read(paths.state, {})
    if not state:
        return {'verified': False, 'reason': 'No launch recorded'}
    frozen = scientific_snapshot(paths)
    result = status(paths)
    valid = state.get('scientific_files') == frozen
    valid = valid and result['conservative_cpu_seconds'] <= AUTHORIZED_CPU
    if result['supervisor_alive']:
        valid = valid and result['guard_alive']
    return {'verified': bool(valid), **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ('launch', 'status', 'verify', 'dry-run', 'stop', 'supervise'):
        group.add_argument('--' + flag, action='store_true')
    group.add_argument('--guard', metavar='SESSION_ID')
    args = parser.parse_args()
    paths = Paths(args.root.resolve())
    if args.guard:
        return guard(paths, args.guard)
    if args.supervise:
        return supervise(paths)
    if args.dry_run:
        result = {'branch': BRANCH, 'authorized_cpu_seconds': AUTHORIZED_CPU,
                  'compute_cpu_reserve': COMPUTE_RESERVE,
                  'preflight': preflight_charge(paths), 'scientific_files': scientific_snapshot(paths),
                  'publication_files': len(eligible_files(paths)), 'cloud_lifetime_guaranteed': False}
    elif args.launch:
        result = launch(paths)
    elif args.stop:
        result = {'signalled_matching_worker': stop(paths)}
    elif args.verify:
        result = verify(paths)
    else:
        result = status(paths)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if args.verify and not result['verified'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
