#!/usr/bin/env python3
"""Supervise only the user's current Calycosin frequency matrix.

Uses the existing cumulative CPU ledger and frozen numerical implementation.
The superseded queue is never entered, and completed scientific files stay.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

PARENT = Path('reports/all_round_validation_20261112')
OUTPUT = PARENT / 'frequency_only'
RUNTIME = OUTPUT / '.background'
SELF = Path('scripts/background_frequency_only_20261116.py')
RUNNER = Path('scripts/run_frequency_matrix_only_20261116.py')
POLICY = Path('data/research/all_round_validation_20261112/frequency_only_policy.json')
SOURCE_SCOPE = Path('data/research/all_round_validation_20261112/execution_scope_20261116.json')
MANIFEST = OUTPUT / 'controller_source_manifest.json'
TERMINAL = {'COMPLETE', 'AUTHORIZED_CPU_BOUNDARY', 'STOPPED_BY_USER', 'SCIENTIFIC_INTEGRITY_CHANGED'}


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


def load_base(root):
    relative = PARENT / '.background/authorization_extension_frozen/scripts/extend_all_round_validation_20261112.py'
    source = root / relative
    manifest = read(root / PARENT / 'authorization_extension_manifest.json', {})
    if source.is_symlink() or sha(source) != manifest.get('wrapper_sha256'):
        raise ValueError('Existing authorization implementation is not intact')
    spec = importlib.util.spec_from_file_location('inherited_frequency_authorization', source)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source.read_bytes(), str(source), 'exec'), module.__dict__)
    base = module.load_runtime(root)
    return base


def frozen(root):
    return root / RUNTIME / 'frozen'


def policy(root):
    value = read(frozen(root) / POLICY)
    if not value or value.get('retained_task') != 'calycosin/df-grid5/round000 frequency matrix only':
        raise ValueError('A frozen frequency-only scope is required')
    if [item.get('id') for item in value.get('workers', [])] != ['frequency_matrix_0', 'frequency_matrix_1']:
        raise ValueError('Only the two cooperative matrix workers are authorized')
    return value


def check_integrity(root, base):
    base.check_integrity(root)
    manifest = read(root / MANIFEST, {})
    if not manifest.get('files'):
        raise ValueError('Frequency-only controller has not been prepared')
    if sha(Path(__file__).resolve()) != manifest['files'][str(SELF)]:
        raise ValueError('Executing frequency-only controller is not its frozen implementation')
    for relative, digest in manifest['files'].items():
        for path in (frozen(root) / relative, root / OUTPUT / 'frozen_source' / relative):
            if path.is_symlink() or not path.is_file() or sha(path) != digest:
                raise ValueError('Frozen frequency-only source changed: ' + relative)
    policy(root)


def prepare(root):
    base = load_base(root)
    base.check_integrity(root)
    if (root / MANIFEST).exists():
        check_integrity(root, base)
        return read(root / MANIFEST)
    scope = read(root / SOURCE_SCOPE, {})
    if not scope.get('cancelled_tasks') or not scope.get('user_instruction'):
        raise ValueError('The explicit user frequency-only scope receipt is missing')
    value = {
        'schema': 'calycosin-frequency-only-execution-policy-v1',
        'user_instruction': scope['user_instruction'],
        'source_scope_receipt': str(SOURCE_SCOPE), 'source_scope_receipt_sha256': sha(root / SOURCE_SCOPE),
        'retained_task': 'calycosin/df-grid5/round000 frequency matrix only',
        'cancelled_tasks': scope['cancelled_tasks'],
        'workers': [{'id': 'frequency_matrix_' + str(index),
                     'command': ['{python}', '{frozen}/' + str(RUNNER), '--root', '{root}',
                                 '--worker-id', str(index), '--once'],
                     'threads': 2} for index in (0, 1)],
        'completion': str(OUTPUT / 'summary.json'), 'completion_seal': str(OUTPUT / 'COMPLETE.json'),
        'cumulative_authorization': 'Existing additive ledger; no new grant and no reset',
        'phase_budget_at_preparation_cpu_seconds': base.AUTHORIZED_CPU_SECONDS,
        'new_geometry_profiles_molecules_or_ir_jobs_allowed': False,
        'created_at_utc': base.utc(),
    }
    existing = read(root / POLICY)
    if existing is not None:
        raise ValueError('Unexpected pre-existing unfrozen frequency-only policy; preserve it for review')
    # Validate inputs before creating a durable policy or manifest.
    for relative in (SELF, RUNNER):
        if not (root / relative).is_file() or (root / relative).is_symlink():
            raise FileNotFoundError(str(root / relative))
    write(root / POLICY, value)
    files = {}
    for relative in (SELF, RUNNER, POLICY):
        source = root / relative
        files[str(relative)] = sha(source)
        for target in (frozen(root) / relative, root / OUTPUT / 'frozen_source' / relative):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    manifest = {'schema': 'frequency-only-frozen-controller-v1', 'files': files,
                'prepared_at_utc': base.utc(), 'original_frozen_source_modified': False,
                'old_cpu_accounting_preserved': True}
    write(root / MANIFEST, manifest)
    check_integrity(root, base)
    return manifest


def completion(root):
    cfg = policy(root)
    result = read(root / cfg['completion'], {})
    seal = read(root / cfg['completion_seal'], {})
    path = root / cfg['completion']
    if result.get('status') != 'COMPLETE' or result.get('qualified_points') != 396 or result.get('planned_points') != 396:
        return False
    if not path.is_file() or seal.get('summary_sha256') != sha(path):
        return False
    hashes = seal.get('artifacts_sha256')
    if not hashes:
        return False
    for relative, digest in hashes.items():
        target = root / OUTPUT / relative
        if not target.resolve().is_relative_to((root / OUTPUT).resolve()) or not target.is_file() or sha(target) != digest:
            return False
    return True


def own_command(root, option):
    return [sys.executable, str(frozen(root) / SELF), '--root', str(root), option]


def spawn(root, command, name):
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', PYTHONPATH=os.pathsep.join([
        str(frozen(root) / 'scripts'), str(root / PARENT / '.background/frozen/scripts'),
        str(root / PARENT / '.background/frozen/src')]))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '2'
    replacements = {'{python}': sys.executable, '{root}': str(root), '{frozen}': str(frozen(root))}
    argv = []
    for item in command:
        for key, value in replacements.items():
            item = item.replace(key, value)
        argv.append(item)
    (root / RUNTIME).mkdir(parents=True, exist_ok=True)
    with (root / RUNTIME / (name + '.log')).open('ab', buffering=0) as stream:
        return subprocess.Popen(argv, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)


def persist(root, base, state):
    state.update(heartbeat_at_utc=base.utc(), authorized_cpu_seconds=base.AUTHORIZED_CPU_SECONDS,
                 authorization_cpu_seconds=base.AUTHORIZED_CPU_SECONDS,
                 scope='FREQUENCY_MATRIX_ONLY', queued_tasks=[],
                 cancelled_tasks=policy(root)['cancelled_tasks'],
                 current_control_entry=str(SELF), control_runtime=str(RUNTIME))
    write(root / RUNTIME / 'state.json', state)
    write(root / OUTPUT / 'status.json', state)
    # Preserve the established status and publication entry points. The old
    # queue is separately barred by its explicit STOP/scope-superseded marker.
    base.write(root / PARENT / '.background/state.json', state)
    base.write(root / PARENT / 'status.json', state)


def halt_workers(base, state):
    for task in state.get('tasks', {}).values():
        base.signal_session(task.get('identity'))
    base.signal_session(state.get('publisher_identity'))


def publish(root, base):
    previous = base.config
    def config(workspace):
        value = previous(workspace)
        value['publish_paths'] += [str(SELF), str(RUNNER), str(POLICY),
                                  'tests/test_frequency_only_controller_20261116.py',
                                  'tests/test_frequency_matrix_only_20261116.py']
        return value
    base.config = config
    return base.publish_process(root)


def guard(root, base):
    while True:
        state = read(root / RUNTIME / 'state.json', {})
        if state.get('status') in TERMINAL:
            return 0
        ledger = base.sample_cpu(root)
        if ledger['conservative_cpu_seconds'] >= base.AUTHORIZED_CPU_SECONDS:
            write(root / RUNTIME / 'budget_stop.json', {'at_utc': base.utc(), 'status': 'AUTHORIZED_CPU_BOUNDARY'})
            halt_workers(base, state)
        if not base.alive(state.get('supervisor_identity')):
            if (root / RUNTIME / 'budget_stop.json').exists() or (root / RUNTIME / 'STOP.json').exists():
                halt_workers(base, state)
                state['status'] = 'AUTHORIZED_CPU_BOUNDARY' if (root / RUNTIME / 'budget_stop.json').exists() else 'STOPPED_BY_USER'
                persist(root, base, state)
                return 0
            try:
                check_integrity(root, base)
                replacement = spawn(root, own_command(root, '--supervise'), 'supervisor')
                write(root / OUTPUT / 'controller_recovery.json', {'at_utc': base.utc(),
                    'replacement_identity': base.identity(replacement.pid), 'retry_limit': None})
                return 0
            except Exception as error:
                write(root / OUTPUT / 'guard_retry.json', {'error': repr(error), 'at_utc': base.utc()})
        time.sleep(.25)


def supervise(root, base):
    (root / RUNTIME).mkdir(parents=True, exist_ok=True)
    with (root / RUNTIME / 'supervisor.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        check_integrity(root, base)
        old = read(root / RUNTIME / 'state.json', {})
        if old.get('status') in TERMINAL:
            return 0
        base.signal_session(old.get('guard_identity'))
        base.signal_session(old.get('publisher_identity'))
        state = {**old, 'status': 'RUNNING', 'supervisor_identity': base.identity(os.getpid()),
                 'guard_identity': None, 'publisher_identity': None, 'tasks': old.get('tasks', {})}
        cfg = policy(root)
        for task in cfg['workers']:
            state['tasks'].setdefault(task['id'], {'identity': None, 'attempts': 0, 'status': 'PENDING'})
        # A crashed controller does not kill live scientific work; its exact
        # identities can be adopted without resetting CPU accounting.
        persist(root, base, state)
        guardian = spawn(root, own_command(root, '--guard'), 'guard')
        state['guard_identity'] = base.identity(guardian.pid)
        children = {}
        publisher = None
        next_publish, next_check = time.monotonic() + 600, 0.
        signal_received = False
        def requested(signum, frame):
            nonlocal signal_received
            signal_received = True
        signal.signal(signal.SIGTERM, requested)
        signal.signal(signal.SIGINT, requested)
        try:
            while True:
                now = time.monotonic()
                ledger = base.sample_cpu(root)
                if ledger['conservative_cpu_seconds'] >= base.AUTHORIZED_CPU_SECONDS or (root / RUNTIME / 'budget_stop.json').exists():
                    state['status'] = 'AUTHORIZED_CPU_BOUNDARY'; break
                if (root / RUNTIME / 'STOP.json').exists():
                    state['status'] = 'STOPPED_BY_USER'; break
                if signal_received:
                    state['status'] = 'CONTROLLER_RESTART_PENDING'; break
                if now >= next_check:
                    check_integrity(root, base); next_check = now + 15
                if guardian.poll() is not None:
                    guardian.wait(); guardian = spawn(root, own_command(root, '--guard'), 'guard')
                    state['guard_identity'] = base.identity(guardian.pid)
                done = completion(root)
                for task in cfg['workers']:
                    item = state['tasks'][task['id']]
                    child = children.get(task['id'])
                    if child is not None and child.poll() is not None:
                        item.update(identity=None, last_returncode=child.wait(), next_retry=now + 2,
                                    last_finished_at_utc=base.utc(), status='RETRY_PENDING')
                        del children[task['id']]
                    if done:
                        if base.alive(item.get('identity')):
                            base.signal_session(item['identity'])
                        item.update(identity=None, status='COMPLETE')
                        continue
                    if not base.alive(item.get('identity')) and now >= item.get('next_retry', 0):
                        try:
                            child = spawn(root, task['command'], task['id'])
                            children[task['id']] = child
                            item.update(identity=base.identity(child.pid), attempts=item['attempts'] + 1,
                                        status='RUNNING', last_started_at_utc=base.utc())
                        except OSError as error:
                            item.update(status='RETRY_PENDING', last_error=repr(error), next_retry=now + 2)
                if publisher is not None and publisher.poll() is not None:
                    code = publisher.wait(); publisher = None; state['publisher_identity'] = None
                    state['last_publication_returncode'] = code
                    if done and state.get('publisher_was_final') and code == 0:
                        state['status'] = 'COMPLETE'; break
                if publisher is None and (now >= next_publish or (done and not state.get('publisher_was_final'))):
                    publisher = spawn(root, own_command(root, '--publish'), 'publication')
                    state.update(publisher_identity=base.identity(publisher.pid), publisher_was_final=done)
                    next_publish = now + (2 if done else 600)
                persist(root, base, state)
                time.sleep(.25)
        except Exception as error:
            try:
                check_integrity(root, base)
            except Exception:
                state.update(status='SCIENTIFIC_INTEGRITY_CHANGED', last_error=repr(error))
            else:
                state.update(status='CONTROLLER_RESTART_PENDING', last_error=repr(error))
        finally:
            if state['status'] in TERMINAL:
                halt_workers(base, state)
                for child in children.values():
                    child.wait()
                if publisher is not None:
                    publisher.wait()
            state['finished_at_utc'] = base.utc()
            persist(root, base, state)
            base.sample_cpu(root)
            if state['status'] in TERMINAL:
                base.signal_session(state.get('guard_identity'))
            # For a nonterminal exception the live guard replaces this process
            # and adopts scientific workers instead of interrupting them.
        return 0 if state['status'] in TERMINAL else 1


def status(root, base, verify=False):
    state = read(root / RUNTIME / 'state.json', {})
    value = {'state': state, 'supervisor_alive': base.alive(state.get('supervisor_identity')),
             'guard_alive': base.alive(state.get('guard_identity')),
             'tasks_alive': {name: base.alive(item.get('identity')) for name, item in state.get('tasks', {}).items()},
             'cpu_accounting': read(root / PARENT / 'cpu_accounting.json', {}),
             'frequency_only_policy': str(POLICY)}
    if verify:
        try:
            check_integrity(root, base)
            intact = True
        except Exception as error:
            intact = False; value['integrity_error'] = repr(error)
        identities = [state.get('supervisor_identity'), state.get('guard_identity')]
        identities += [x.get('identity') for x in state.get('tasks', {}).values()]
        detached = all(not base.alive(x) or ((p := base.proc_stat(x['pid'])) and p['pgrp'] == p['pid'] == p['sid']) for x in identities)
        value['verified'] = bool(intact and detached and ((value['supervisor_alive'] and value['guard_alive']) or state.get('status') == 'COMPLETE'))
        value['source_integrity_passed'] = intact
    return value


def start(root, base):
    (root / RUNTIME).mkdir(parents=True, exist_ok=True)
    with (root / RUNTIME / 'launch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        check_integrity(root, base)
        previous = status(root, base)
        if previous['supervisor_alive'] or previous['state'].get('status') in TERMINAL:
            return previous
        if (root / RUNTIME / 'STOP.json').exists():
            raise ValueError('Explicit stop prevents frequency-only resume')
        # First handover must be performed explicitly by the primary agent;
        # never silently kill a still-running old controller or scientific job.
        if not previous['state']:
            legacy = read(root / PARENT / '.background/state.json', {})
            old = [legacy.get(k) for k in ('supervisor_identity', 'guard_identity', 'publisher_identity')]
            old += [item.get('identity') for item in legacy.get('tasks', {}).values()]
            if any(base.alive(item) for item in old):
                raise ValueError('The superseded queue still has live processes; finish explicit checkpoint handover first')
        ledger = base.sample_cpu(root)
        if ledger['conservative_cpu_seconds'] >= base.AUTHORIZED_CPU_SECONDS:
            raise ValueError('Cumulative authorization is exhausted')
        oldstate = read(root / PARENT / '.background/state.json', {})
        state = {**previous['state'], 'status': 'STARTING', 'branch': oldstate.get('branch'),
                 'initial_head': oldstate.get('initial_head'), 'tasks': previous['state'].get('tasks', {}),
                 'inherited_completed_tasks': oldstate.get('inherited_completed_tasks', []),
                 'authorization_ledger': 'data/research/all_round_validation_20261112/authorization_extensions.json'}
        persist(root, base, state)
        child = spawn(root, own_command(root, '--supervise'), 'supervisor')
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            value = status(root, base)
            if value['supervisor_alive'] and value['guard_alive']:
                return value
            if child.poll() is not None:
                return {**value, 'start_returncode': child.returncode}
            time.sleep(.05)
        return {**status(root, base), 'launch_handshake': 'PENDING'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/workspace/research4'))
    options = parser.add_mutually_exclusive_group(required=True)
    for flag in ('prepare', 'start', 'verify', 'status', 'supervise', 'guard', 'publish', 'stop'):
        options.add_argument('--' + flag, action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    base = load_base(root)
    if args.prepare:
        result = prepare(root)
    else:
        if args.supervise: return supervise(root, base)
        if args.guard: return guard(root, base)
        if args.publish: return publish(root, base)
        if args.start: result = start(root, base)
        elif args.stop:
            write(root / RUNTIME / 'STOP.json', {'reason': 'Explicit user stop', 'at_utc': base.utc()})
            result = status(root, base)
        else: result = status(root, base, verify=args.verify)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if args.verify and not result['verified'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
