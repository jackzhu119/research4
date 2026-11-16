#!/usr/bin/env python3
"""Extend and resume the original frozen validation without changing its code.

The extension ledger is additive and idempotent. All control roles re-enter
this separately frozen wrapper, while scientific workers use the original
frozen source, inputs and checkpoints. Completed ML is never reconstructed.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import uuid

OUTPUT = Path('reports/all_round_validation_20261112')
RUNTIME = OUTPUT / '.background'
SELF = Path('scripts/extend_all_round_validation_20261112.py')
OLD_SELF = Path('scripts/background_all_round_validation_20261112.py')
OLD_AUTHORIZATION = Path('data/research/all_round_validation_20261112/authorization.json')
EXTENSIONS = Path('data/research/all_round_validation_20261112/authorization_extensions.json')
MANIFEST = OUTPUT / 'authorization_extension_manifest.json'
WRAPPER_COPY = RUNTIME / 'authorization_extension_frozen' / SELF
PUBLISHED_COPY = OUTPUT / 'authorization_extension_source' / SELF
INHERITED = OUTPUT / 'authorization_inherited_completed_tasks.json'
DEFAULT_GRANT_ID = 'user-confirmed-additional-100000-cpu-seconds-20261112'
DEFAULT_GRANT_SECONDS = 100000.0


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


def original(root):
    source = root / RUNTIME / 'frozen' / OLD_SELF
    manifest = read(root / OUTPUT / 'source_manifest.json', {})
    expected = manifest.get('files', {}).get(str(OLD_SELF))
    if not expected or not source.is_file() or source.is_symlink() or sha(source) != expected:
        raise ValueError('Original frozen supervisor does not match its original manifest')
    spec = importlib.util.spec_from_file_location('all_round_original_' + uuid.uuid4().hex, source)
    module = importlib.util.module_from_spec(spec)
    # Loading the immutable control source must not create/update a .pyc in
    # the original frozen directory either.
    exec(compile(source.read_bytes(), str(source), 'exec'), module.__dict__)
    return module


def cumulative(ledger):
    if not isinstance(ledger, dict):
        raise ValueError('Additional authorization ledger is missing')
    total = float(ledger['initial_authorized_cpu_seconds'])
    if not math.isfinite(total) or total <= 0:
        raise ValueError('Invalid original CPU authorization')
    identities = set()
    for entry in ledger.get('grants', []):
        identifier = entry['grant_id']
        amount = float(entry['added_cpu_seconds'])
        if not identifier or identifier in identities or not math.isfinite(amount) or amount <= 0:
            raise ValueError('Duplicate or invalid additional CPU authorization')
        identities.add(identifier)
        total += amount
    if total != float(ledger['cumulative_authorized_cpu_seconds']):
        raise ValueError('Additional authorization total does not equal its grants')
    return total


def extension_lock(root, exclusive):
    path = root / RUNTIME / 'authorization_extension.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
    return lock


def _check_extension_unlocked(root, full=True):
    manifest = read(root / MANIFEST, {})
    if not manifest.get('wrapper_sha256'):
        raise ValueError('Run --prepare to freeze the authorization wrapper first')
    ledger_path = root / EXTENSIONS
    if sha(ledger_path) != manifest['authorization_ledger_sha256']:
        raise ValueError('Additional authorization ledger changed outside its preparation transaction')
    ledger = read(ledger_path)
    total = cumulative(ledger)
    if sha(root / OLD_AUTHORIZATION) != ledger['original_authorization_sha256']:
        raise ValueError('Original CPU authorization was changed')
    if total != manifest['cumulative_authorized_cpu_seconds']:
        raise ValueError('Extension manifest and additional authorization disagree')
    if full:
        if sha(Path(__file__).resolve()) != manifest['wrapper_sha256']:
            raise ValueError('Executing authorization wrapper differs from its frozen source')
        for relative in (WRAPPER_COPY, PUBLISHED_COPY):
            target = root / relative
            if target.is_symlink() or not target.is_file() or sha(target) != manifest['wrapper_sha256']:
                raise ValueError('Frozen authorization wrapper changed: ' + str(relative))
        if sha(root / OUTPUT / 'source_manifest.json') != manifest['original_source_manifest_sha256']:
            raise ValueError('Original frozen source manifest changed')
        if sha(root / INHERITED) != manifest['inherited_completed_tasks_sha256']:
            raise ValueError('Inherited completed-task receipt changed')
    return ledger, manifest, total


def check_extension(root, full=True):
    with extension_lock(root, False):
        return _check_extension_unlocked(root, full)


def inherited_receipt(root, base):
    """Preserve the prior completion as history, without executing its code."""
    cfg = base.config(root)
    state = read(root / RUNTIME / 'state.json', {})
    entries = []
    for task in cfg['tasks']:
        if task['id'] != 'ml_verification':
            continue
        marker_path = root / base.safe_relative(task['completion']['path'])
        if not base.completed(root, task):
            raise ValueError('Historical ML completion receipt is missing; rebuilding withdrawn paper inputs is forbidden')
        entries.append({'id': task['id'], 'status': 'COMPLETE',
                        'completion_receipt': task['completion']['path'],
                        'completion_receipt_sha256': sha(marker_path),
                        'historical_state': state.get('tasks', {}).get(task['id'], {}),
                        'execution_on_resume': 'EXCLUDED_ALREADY_COMPLETED',
                        'does_not_grant_current_paper_eligibility': True})
    return {'schema': 'all-round-inherited-completed-tasks-v1',
            'tasks': entries, 'at_utc': utc(),
            'policy': 'Current AGENTS.md withdraws mislabeled results; never rebuild removed paper bundles or import withdrawn paper inputs.',
            'scientific_workers_preserved': True}


def prepare(root, seconds=DEFAULT_GRANT_SECONDS, grant_id=DEFAULT_GRANT_ID):
    if not isinstance(grant_id, str) or not grant_id.strip() or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('A stable grant identity and positive finite CPU seconds are required')
    base = original(root)
    base.check_integrity(root)
    with extension_lock(root, True):
        old_manifest = read(root / MANIFEST)
        if old_manifest:
            _check_extension_unlocked(root)
        authorization = read(root / OLD_AUTHORIZATION)
        initial = float(authorization['authorized_cpu_seconds'])
        if initial != float(base.AUTHORIZED_CPU_SECONDS):
            raise ValueError('Original authorization does not match the frozen original supervisor')
        ledger = read(root / EXTENSIONS)
        if ledger is None:
            ledger = {'schema': 'all-round-additional-authorization-v1',
                      'original_authorization_file': str(OLD_AUTHORIZATION),
                      'original_authorization_sha256': sha(root / OLD_AUTHORIZATION),
                      'initial_authorized_cpu_seconds': initial,
                      'cumulative_authorized_cpu_seconds': initial, 'grants': [],
                      'cpu_accounting': 'Retain every original observed counter epoch and separately charged uncertainty; never reset used CPU.',
                      'early_reserve_cpu_seconds': 0}
        cumulative(ledger)
        existing = next((entry for entry in ledger['grants'] if entry['grant_id'] == grant_id), None)
        if existing and float(existing['added_cpu_seconds']) != float(seconds):
            raise ValueError('The same grant identity already has a different amount')
        if not existing:
            ledger['grants'].append({'grant_id': grant_id, 'added_cpu_seconds': float(seconds),
                'authorized_at_utc': utc(),
                'authority': 'Explicit user-confirmed additional CPU seconds in the current chat; additional amount, not a replacement total'})
            ledger['cumulative_authorized_cpu_seconds'] = initial + sum(float(entry['added_cpu_seconds']) for entry in ledger['grants'])
            ledger['updated_at_utc'] = utc()
        source = Path(__file__).resolve()
        digest = sha(source)
        if old_manifest and digest != old_manifest['wrapper_sha256']:
            raise ValueError('The separately frozen authorization wrapper cannot be silently replaced')
        for relative in (WRAPPER_COPY, PUBLISHED_COPY):
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.resolve() != source:
                shutil.copyfile(source, destination)
        if not (root / INHERITED).exists():
            write(root / INHERITED, inherited_receipt(root, base))
        if old_manifest and not existing:
            history = root / OUTPUT / 'authorization_extension_history' / (uuid.uuid4().hex + '.json')
            write(history, old_manifest)
        if not existing or not (root / EXTENSIONS).exists():
            write(root / EXTENSIONS, ledger)
        manifest = {'schema': 'all-round-authorization-wrapper-v1',
                    'wrapper_sha256': digest, 'frozen_wrapper': str(WRAPPER_COPY),
                    'published_wrapper': str(PUBLISHED_COPY),
                    'original_source_manifest_sha256': sha(root / OUTPUT / 'source_manifest.json'),
                    'original_supervisor_sha256': sha(root / RUNTIME / 'frozen' / OLD_SELF),
                    'authorization_ledger': str(EXTENSIONS), 'authorization_ledger_sha256': sha(root / EXTENSIONS),
                    'cumulative_authorized_cpu_seconds': cumulative(ledger),
                    'inherited_completed_tasks_sha256': sha(root / INHERITED),
                    'prepared_at_utc': old_manifest['prepared_at_utc'] if old_manifest else utc(),
                    'latest_grant_id': grant_id,
                    'original_frozen_files_modified': False,
                    'all_control_roles_use_frozen_wrapper': True,
                    'scientific_workers_use_original_frozen_code': True}
        if not old_manifest or not existing:
            write(root / MANIFEST, manifest)
        _check_extension_unlocked(root)
        return {'authorization': ledger, 'manifest': read(root / MANIFEST),
                'already_recorded': bool(existing), 'formal_processes_started': False}


def load_runtime(root):
    base = original(root)
    ledger, manifest, total = check_extension(root)
    base.AUTHORIZED_CPU_SECONDS = total
    old_check, old_config, old_sample, old_write = base.check_integrity, base.config, base.sample_cpu, base.write

    def augmented_check(workspace):
        old_check(workspace)
        check_extension(workspace)

    def augmented_config(workspace):
        value = copy.deepcopy(old_config(workspace))
        value['authorized_cpu_seconds'] = base.AUTHORIZED_CPU_SECONDS
        # The prior verifier included now-withdrawn paper material. Its sealed
        # completion is inherited; it must never be re-executed on restoration.
        value['tasks'] = [task for task in value['tasks'] if task['id'] != 'ml_verification']
        if not value['tasks']:
            raise ValueError('No remaining scientific workers in the frozen task configuration')
        for relative in (str(SELF), 'tests/test_all_round_extension_20261112.py'):
            if relative not in value.setdefault('publish_paths', []):
                value['publish_paths'].append(relative)
        return value

    def augmented_sample(workspace):
        # Every live supervisor and guard follows future explicit additive
        # grants through this same ledger, without a numerical-code restart.
        _, _, latest_total = check_extension(workspace, full=False)
        base.AUTHORIZED_CPU_SECONDS = latest_total
        return old_sample(workspace)

    def augmented_own_command(workspace, option, *args):
        return [sys.executable, str(workspace / WRAPPER_COPY), '--root', str(workspace), option, *args]

    def augmented_write(path, value):
        if Path(path) in {root / RUNTIME / 'state.json', root / OUTPUT / 'status.json'} and isinstance(value, dict):
            value.update(authorized_cpu_seconds=base.AUTHORIZED_CPU_SECONDS,
                         authorization_cpu_seconds=base.AUTHORIZED_CPU_SECONDS,
                         authorization_ledger=str(EXTENSIONS),
                         authorization_wrapper=str(WRAPPER_COPY))
        return old_write(path, value)

    base.check_integrity = augmented_check
    base.config = augmented_config
    base.sample_cpu = augmented_sample
    base.own_command = augmented_own_command
    base.write = augmented_write
    return base


def validate_resume_state(root, base):
    """Only an explicitly superseded CPU boundary may be cleared."""
    state_path = root / RUNTIME / 'state.json'
    state = read(state_path, {})
    if (root / RUNTIME / 'STOP.json').exists() or state.get('status') in {'STOPPED_BY_USER', 'SCIENTIFIC_INTEGRITY_CHANGED'}:
        raise ValueError('Explicit stop or scientific-integrity failure cannot be bypassed by an added grant')
    processes = [state.get(key) for key in ('supervisor_identity', 'guard_identity', 'publisher_identity')]
    processes.extend(item.get('identity') for item in state.get('tasks', {}).values())
    if any(base.alive(item) for item in processes):
        # A live extended supervisor is already following the grant ledger.
        if base.alive(state.get('supervisor_identity')) and float(state.get('authorized_cpu_seconds', 0)) == base.AUTHORIZED_CPU_SECONDS:
            return state
        raise ValueError('An existing control or scientific process is still alive; this recovery will not replace it')
    if state.get('status') == 'COMPLETE':
        return state
    ledger = base.sample_cpu(root)
    if ledger['conservative_cpu_seconds'] >= base.AUTHORIZED_CPU_SECONDS:
        raise ValueError('Cumulative charged CPU already reaches the extended authorization')
    boundary_path = root / RUNTIME / 'budget_stop.json'
    if state.get('status') == 'AUTHORIZED_CPU_BOUNDARY' or boundary_path.exists():
        previous_limit = float(state.get('authorized_cpu_seconds', state.get('authorization_cpu_seconds', 0)))
        if previous_limit <= 0 or previous_limit >= base.AUTHORIZED_CPU_SECONDS:
            raise ValueError('Only a boundary superseded by a larger explicit grant can be resumed')
        archive = root / OUTPUT / 'authorization_boundary_history' / uuid.uuid4().hex
        write(archive / 'state.json', state)
        if boundary_path.exists():
            os.replace(boundary_path, archive / 'budget_stop.json')
        state['status'] = 'AUTHORIZED_EXTENSION_RESUME_PENDING'
    state.update(authorization_cpu_seconds=base.AUTHORIZED_CPU_SECONDS,
                 authorized_cpu_seconds=base.AUTHORIZED_CPU_SECONDS,
                 authorization_ledger=str(EXTENSIONS),
                 inherited_completed_tasks=read(root / INHERITED)['tasks'],
                 restoration_reason='Platform environment restarted; resume original point checkpoints with additive grant',
                 authorization_wrapper=str(WRAPPER_COPY))
    write(state_path, state)
    write(root / OUTPUT / 'authorization_continuation.json', {
        'status': 'PREPARED_FOR_RESUME', 'at_utc': utc(),
        'authorized_cpu_seconds': base.AUTHORIZED_CPU_SECONDS,
        'observed_cpu_seconds': ledger['observed_cpu_seconds'],
        'conservative_cpu_seconds': ledger['conservative_cpu_seconds'],
        'uncertain_upper_cpu_seconds': ledger['uncertain_upper_cpu_seconds'],
        'remaining_cpu_seconds': ledger['remaining_cpu_seconds'],
        'old_frozen_source_unchanged': True,
        'inherited_completed_tasks': read(root / INHERITED)['tasks']})
    return state


def start(root, base):
    lock_path = root / RUNTIME / 'authorization_resume.lock'
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        base.check_integrity(root)
        validate_resume_state(root, base)
        return base.start(root)


def status(root, base, verify=False):
    value = base.verify(root) if verify else base.status(root)
    ledger, manifest, total = check_extension(root)
    value.update(cumulative_authorized_cpu_seconds=total, authorization_ledger=str(EXTENSIONS),
                 frozen_wrapper=str(WRAPPER_COPY), wrapper_integrity_passed=True,
                 all_control_role_command_paths=[base.own_command(root, flag) for flag in ('--supervise', '--guard', '--publish')],
                 inherited_completed_tasks=read(root / INHERITED)['tasks'])
    if verify:
        value['verified'] = bool(value['verified'] and value['state'].get('authorized_cpu_seconds') == total
                                 and value.get('cpu_accounting', {}).get('authorized_cpu_seconds') == total)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('ALL_ROUND_VALIDATION_ROOT', '/workspace/research4')))
    parser.add_argument('--grant-cpu-seconds', type=float)
    parser.add_argument('--grant-id')
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ('prepare', 'start', 'verify', 'status', 'supervise', 'guard', 'publish', 'stop'):
        group.add_argument('--' + flag, action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.prepare:
        result = prepare(root, args.grant_cpu_seconds if args.grant_cpu_seconds is not None else DEFAULT_GRANT_SECONDS,
                         args.grant_id or DEFAULT_GRANT_ID)
    else:
        if args.grant_cpu_seconds is not None or args.grant_id:
            raise ValueError('Additional grants are recorded only with --prepare')
        base = load_runtime(root)
        if args.supervise:
            return base.supervise(root)
        if args.guard:
            return base.guard(root)
        if args.publish:
            return base.publish_process(root)
        if args.start:
            result = start(root, base)
        elif args.stop:
            write(root / RUNTIME / 'STOP.json', {'reason': 'Explicit user stop', 'at_utc': utc()})
            result = status(root, base)
        else:
            result = status(root, base, verify=args.verify)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if args.verify and not result['verified'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
