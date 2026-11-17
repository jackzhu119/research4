#!/usr/bin/env python3
"""Run only Spinosin in its own worktree, checkpoints, supervision and report.

No CPU, elapsed-time or retry limit. Frozen chemistry modules are imported without
source modification. The dedicated process applies the user-directed four-thread
and 10000 MB runtime allocation. Other processes are never signalled here.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np
import background_pyscf_repair_20261111 as bg
import run_pyscf_repair_20261111 as study
import pyscf_repair_core_20261111 as core
from pyscf_repair_modes_20261111 import verify_analysis

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
OUTPUT = ROOT / 'reports/pyscf_spinosin_direct_20261113'
RUNTIME = OUTPUT / '.background'
STATE = RUNTIME / 'state.json'
PROTOCOL = OUTPUT / 'protocol.json'
DOC = ROOT / 'docs/PYSCF_SPINOSIN_DIRECT_RESULTS_20261113.md'
BRANCH = 'research/spinosin-direct-20261113'
PYTHON = ROOT / '.venv-pyscf/bin/python'
LEGACY = Path('/workspace/research4/reports/pyscf_repair_20261111')
INTEGRITY_ERROR = 42
# User-directed dedicated allocation; frozen source files remain byte-identical.
RESOURCE_OVERRIDES = {'threads': 4, 'max_memory_mb': 10000}
core.SCF_SETTINGS.update(RESOURCE_OVERRIDES)
study.OUTPUT = OUTPUT
utc = study.utc
read = study.read
write = study.write
sha = study.sha


def require_branch():
    result = bg.git('branch', '--show-current')
    if result.returncode or result.stdout.strip() != BRANCH:
        raise study.IntegrityError('Spinosin worktree must remain on ' + BRANCH)


def snapshot():
    value = bg.scientific_snapshot()
    files = dict(value['scientific_files'])
    for path in (SCRIPT, Path(bg.__file__).resolve()):
        files[str(path.relative_to(ROOT))] = sha(path)
    return {'scientific_files': files, 'scientific_input_sha256': hashlib.sha256(
        json.dumps(files, sort_keys=True).encode()).hexdigest()}


def check_snapshot(expected):
    if snapshot() != {key: expected.get(key) for key in snapshot()}:
        raise study.IntegrityError('Spinosin frozen scientific inputs or wrapper changed')


def protocol():
    symbols, coordinates, dm = study.molecule_inputs('spinosin')
    if len(symbols) != 75 or coordinates.shape != (75, 3):
        raise study.IntegrityError('Expected frozen 75-atom Spinosin input')
    return {'schema': 'research4-spinosin-direct-v1', 'molecule': 'spinosin',
            'source_protocol': read(bg.OUTPUT / 'frozen_protocol.json'), **snapshot(),
            'execution_resource_overrides': RESOURCE_OVERRIDES,
            'settings': core.settings(5), 'optimization_settings': core.OPTIMIZATION_SETTINGS,
            'finite_difference_steps_bohr': list(study.STEPS),
            'planned_primary_displaced_points': 900, 'planned_vibrational_modes_per_step': 219,
            'source_density_available': dm is not None, 'shared_inputs_read_only': True,
            'legacy_queue_ownership': 'User directed stopping the legacy Calycosin queue and assigning the full allocation to standalone Spinosin; this script performs observation only.',
            'quality_runtime_cpu_cutoffs': None, 'retry_count_limit': None,
            'scope': 'Isolated molecule PBE/def2-SVP/noD3; signed frequencies and actual dipole derivatives; no herb-solid peak assignment.'}


def prepare():
    require_branch()
    value = protocol()
    old = read(PROTOCOL)
    if old is not None and old != value:
        raise study.IntegrityError('Spinosin protocol or source/input manifest changed')
    if old is None:
        write(PROTOCOL, value)
    return value


def progress(**values):
    old = read(OUTPUT / 'progress.json', {})
    old.update(values, heartbeat_at_utc=utc())
    write(OUTPUT / 'progress.json', old)


study.progress = progress


def legacy_activity():
    current = read(LEGACY / 'progress.json', {})
    active = current.get('molecule') == 'spinosin' and current.get('status') not in ('DONE', 'FAILED')
    return {'legacy_output': str(LEGACY), 'legacy_reported_molecule': current.get('molecule'),
            'legacy_heartbeat_at_utc': current.get('heartbeat_at_utc'),
            'legacy_spinosin_activity_reported': active,
            'warning': 'Unexpected legacy Spinosin activity reported.' if active else None,
            'coordination': 'Observation only; legacy processes and outputs are untouched.'}


def status():
    state = read(STATE, {})
    return {'state': state, 'supervisor_alive': bg.matching_process(state.get('supervisor_pid'), [SCRIPT.name, '--supervise']),
            'worker_alive': bg.matching_process(state.get('worker_pid'), [SCRIPT.name, '--worker']),
            'progress': read(OUTPUT / 'progress.json', {}),
            'optimization': read(current_round() / 'grid5-optimize/progress.json', {}),
            'summary': read(OUTPUT / 'results_summary.json', {}), **legacy_activity(),
            'cloud_lifetime_guaranteed': False}


def current_round():
    root = OUTPUT / 'jobs/spinosin'
    state = read(root / 'repair_state.json', {'round': 0})
    return root if state['round'] == 0 else root / f"round{state['round']:03d}"


def validate_arrays(result, include_density=False):
    count = 0
    if 'gradient' in result and not {'coordinates_bohr', 'gradient', 'dipole_au', 'density'}.issubset(result.get('array_refs', {})):
        raise study.IntegrityError('Completed electronic point lacks required array seals')
    for key, ref in result.get('array_refs', {}).items():
        if '/.local/' in ref['path'] and not include_density:
            continue
        target = Path(ref['path'])
        if not target.is_file() or sha(target) != ref['sha256']:
            raise study.IntegrityError('Saved array hash mismatch: ' + str(target))
        if key in result and isinstance(result[key], list):
            if not np.array_equal(np.asarray(result[key]), np.load(target, allow_pickle=False)):
                raise study.IntegrityError('Saved JSON/array mismatch: ' + str(target))
        count += 1
    return count


def recover_optimization_steps():
    """Only under the exclusive worker lock; preserve interrupted raw evidence."""
    recovery = []
    for opt in (OUTPUT / 'jobs/spinosin').glob('**/grid5-optimize'):
        saved = read(opt / 'progress.json', {})
        if not saved or saved.get('status') == 'DONE':
            continue
        while True:
            index = int(saved.get('evaluations', 0))
            step = opt / 'steps' / f'step-{index:05d}'
            if not step.exists():
                break
            point = read(step / 'result.json', {})
            if point.get('status') == 'DONE' and point.get('qualified'):
                validate_arrays(point, include_density=True)
                if saved.get('density_path'):
                    incoming = np.load(saved['density_path'], allow_pickle=False)
                elif opt.parent.name.startswith('round'):
                    round_id = int(opt.parent.name[5:])
                    selection = read(OUTPUT / 'jobs/spinosin' / f'round{round_id - 1:03d}' / 'negative-mode-follow/selection.json')
                    incoming = np.load(selection['density_path'], allow_pickle=False)
                else:
                    incoming = study.molecule_inputs('spinosin')[2]
                coordinates = np.asarray(point['coordinates_bohr'])
                frozen_symbols = study.molecule_inputs('spinosin')[0]
                if point['symbols'] != frozen_symbols or coordinates.shape != (75, 3) or not np.isfinite(coordinates).all():
                    raise study.IntegrityError('Uncommitted optimization point atom-order mismatch')
                expected, _ = core._fingerprint(point['symbols'], coordinates, 5, incoming)
                if point.get('input_fingerprint') != expected:
                    raise study.IntegrityError('Uncommitted optimization point input fingerprint mismatch')
                proof = {'action': 'ROLL_FORWARD_REAL_COMPLETED_POINT', 'point_path': str(step / 'result.json'),
                         'point_sha256': sha(step / 'result.json'), 'previous_evaluations': index,
                         'array_refs': point['array_refs'], 'at_utc': utc()}
                # Persist the real-point hash proof before advancing the checkpoint.
                write(OUTPUT / 'recovery' / (utc().replace(':', '-') + '-' + uuid.uuid4().hex[:8] + '.json'), proof)
                saved.update(evaluations=index + 1, coordinates_bohr=point['coordinates_bohr'],
                             density_path=point['density_path'], energy_hartree=point['energy_hartree'],
                             nuclear_gradient_rms=point['nuclear_gradient_rms'], nuclear_gradient_max=point['nuclear_gradient_max'],
                             last_point_path=str(step / 'result.json'), heartbeat_at_utc=utc())
                write(opt / 'progress.json', saved)
                recovery.append(proof)
                continue
            archive = OUTPUT / '.local/interrupted-optimization' / (utc().replace(':', '-') + '-' + uuid.uuid4().hex[:8])
            archive.parent.mkdir(parents=True, exist_ok=True)
            proof = {'action': 'ARCHIVE_INTERRUPTED_STEP', 'old_path': str(step), 'archive_path': str(archive),
                     'point_status': point.get('status'), 'at_utc': utc()}
            shutil.move(str(step), str(archive))
            recovery.append(proof)
            break
    if recovery:
        write(OUTPUT / 'recovery' / (utc().replace(':', '-') + '.json'), recovery)
    return recovery


def make_summary():
    current = current_round()
    complete = study.completed_result(current)
    value = {'status': 'DONE' if complete else 'PARTIAL', 'molecule': 'spinosin',
             'planned_primary_displaced_points': 900, 'at_utc': utc(),
             'result_directory': str(current.relative_to(ROOT)), 'scope': protocol()['scope'],
             'legacy_queue': legacy_activity(), 'quality_is_separate_from_completion': True}
    if complete:
        value.update(center_quality=complete['center_quality'], analysis=complete['analysis'],
                     result_path=str(current / 'result.json'), modes_csv=str(current / 'modes/all_modes.csv'))
    write(OUTPUT / 'results_summary.json', value)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    text = ['# Spinosin 专用计算：实时结果', '', '状态：' + value['status'] + '。', '',
            '仅运行 Spinosin；原 Calycosin 计算进程与报告保持独立。PBE/def2-SVP、无 D3、grid5、原生直接 SCF，完整网格响应梯度。', '',
            '计划两个步长共 900 个真实位移点，保留全部有符号频率及真实偶极导数 IR 强度。完成与科学资格分开报告。', '',
            '[机器可读汇总](../reports/pyscf_spinosin_direct_20261113/results_summary.json)。']
    if complete:
        path = current.relative_to(ROOT).as_posix()
        text += ['', f'[完整结果](../{path}/result.json)；[全部有符号频率及 IR 强度](../{path}/modes/all_modes.csv)。', '',
                 '```json', json.dumps(complete['analysis'].get('qualification', {}), ensure_ascii=False, indent=2), '```']
    text += ['', '计算范围为孤立分子，不据此确认药材固体 THz 峰或逐样品化学标签。按用户指令停止旧 Calycosin 队列后，独立分配 4 线程及 10000 MB PySCF 内存设置；本脚本不向其他进程发送信号。', '']
    DOC.write_text('\n'.join(text), encoding='utf-8')
    return value


def worker():
    prepare()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / '.worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 1
        recover_optimization_steps()
        progress(status='RUNNING', molecule='spinosin', worker_pid=os.getpid(), legacy_queue=legacy_activity())
        study.run_molecule('spinosin')
        result = make_summary()
        progress(status=result['status'], stage='spinosin_computed', molecule_displacements_completed=900)
        return 0


def verify():
    prepare()
    current = current_round()
    if study.completed_result(current) is None:
        return {'status': 'PARTIAL', 'verified': False, 'reason': 'Spinosin is not complete'}
    checks = 0
    for path in (OUTPUT / 'jobs/spinosin').rglob('result.json'):
        result = read(path, {})
        if result.get('status') in ('DONE', 'UNQUALIFIED'):
            checks += validate_arrays(result)
    for path in (OUTPUT / 'jobs/spinosin').rglob('COMPLETE.json'):
        study.completed_result(path.parent)
        checks += 1
    opt = read(current / 'grid5-optimize/result.json')
    xyz = np.asarray(opt['coordinates_bohr'])
    if xyz.shape != (75, 3):
        raise study.IntegrityError('Spinosin optimized atom count mismatch')
    count = 0
    for hi, step in enumerate(study.STEPS):
        points = {}
        for sign, label in ((1, 'plus'), (-1, 'minus')):
            items = []
            for j in range(xyz.size):
                path = current / 'finite-differences' / f'h{hi}' / f'c{j:03d}-{label}' / 'result.json'
                point = read(path)
                if point is None or point.get('status') not in ('DONE', 'UNQUALIFIED'):
                    raise study.IntegrityError('Missing completed Cartesian point: ' + str(path))
                expected = np.array(xyz, copy=True)
                expected.flat[j] += sign * step
                if not np.array_equal(np.asarray(point['coordinates_bohr']), expected):
                    raise study.IntegrityError('Cartesian displacement coordinate mismatch')
                validate_arrays(point)
                items.append(point)
                count += 1
            points[sign] = items
        hessian, dipole = study.assemble_fd([p['gradient'] for p in points[1]], [p['gradient'] for p in points[-1]],
                                           [p['dipole_au'] for p in points[1]], [p['dipole_au'] for p in points[-1]], step)
        for filename, array in ((f'hessian_h{hi}.npy', hessian), (f'dipole_derivative_h{hi}.npy', dipole)):
            if not np.array_equal(array, np.load(current / 'finite-differences' / filename, allow_pickle=False)):
                raise study.IntegrityError('Raw-point finite-difference replay mismatch: ' + filename)
        with np.load(current / 'modes/analysis_arrays.npz', allow_pickle=False) as arrays:
            if not np.array_equal(hessian, arrays[f'step_{hi}_raw_hessian']) or not np.array_equal(dipole, arrays[f'step_{hi}_dipole_derivative_au']):
                raise study.IntegrityError('Mode inputs differ from raw-point derivatives')
    if count != 900:
        raise study.IntegrityError('Expected exactly 900 Cartesian points')
    replay = verify_analysis(current / 'modes')
    value = {'status': 'VERIFIED', 'verified': True, 'raw_displacement_points_verified': count,
             'array_and_completion_checks': checks, 'mode_replay': replay, 'at_utc': utc(),
             'result_sha256': sha(current / 'result.json'), 'scientific_input_sha256': snapshot()['scientific_input_sha256']}
    write(OUTPUT / 'verification.json', value)
    make_summary()
    return value


def eligible_files():
    paths = [SCRIPT, DOC] + list(OUTPUT.rglob('*')) + list((ROOT / 'tests').glob('*spinosin*20261113*'))
    selected = []
    for path in paths:
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(ROOT)
        if path.name == '.gitignore':
            selected.append(relative.as_posix())
            continue
        if any(part.startswith('.') or part in ('__pycache__', 'cache', 'checkpoint', 'checkpoints') for part in relative.parts):
            continue
        if path.suffix in ('.log', '.lock', '.pid', '.tmp', '.partial', '.pyc', '.chk', '.h5'):
            continue
        selected.append(relative.as_posix())
    return sorted(set(selected))


def publish(state, update):
    committed = state.get('final_commit_sha')
    while True:
        require_branch()
        check_snapshot(state)
        if not committed:
            files = eligible_files()
            staged = bg.git('diff', '--cached', '--name-only', '-z')
            if staged.returncode:
                raise RuntimeError('Cannot inspect Git index')
            if any(name and name not in files for name in staged.stdout.split('\0')):
                raise study.IntegrityError('Unrelated staged files prevent task-only finalization')
            result = bg.git('add', '--pathspec-from-file=-', '--pathspec-file-nul', input_text='\0'.join(files) + '\0')
            if result.returncode:
                update(publication='RETRYING_GIT_ADD', last_error=bg.safe_message(result.stderr))
                time.sleep(30)
                continue
            changed = bg.git('diff', '--cached', '--quiet')
            if changed.returncode == 1:
                result = bg.git('commit', '-m', 'Finalize standalone Spinosin computation and numerical verification')
                if result.returncode:
                    update(publication='RETRYING_COMMIT', last_error=bg.safe_message(result.stderr))
                    time.sleep(30)
                    continue
            elif changed.returncode != 0:
                time.sleep(30)
                continue
            committed = bg.git('rev-parse', 'HEAD').stdout.strip()
            update(final_commit_sha=committed)
        result = bg.git('push', 'origin', 'HEAD:refs/heads/' + BRANCH)
        if result.returncode == 0:
            update(publication='PUSHED', published_at_utc=utc(), last_error=None)
            write(OUTPUT / 'background_finalization.json', {'status': 'DONE', 'final_commit_sha': committed,
                  'branch': BRANCH, 'at_utc': utc(), 'scientific_input_sha256': state['scientific_input_sha256']})
            return
        update(publication='RETRYING_PUSH', last_error=bg.safe_message(result.stderr))
        time.sleep(30)


def supervise(instance):
    frozen = prepare()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / 'supervisor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        previous = read(STATE, {})
        if previous:
            check_snapshot(previous)
        state = {'status': 'RUNNING', 'instance': instance, 'supervisor_pid': os.getpid(),
                 'supervisor_sid': os.getsid(0), 'supervisor_ppid_at_start': os.getppid(),
                 'started_at_utc': utc(), 'quality_runtime_cpu_cutoffs': None, 'retry_count_limit': None,
                 'cloud_lifetime_guaranteed': False, **snapshot()}
        if previous.get('final_commit_sha'):
            state['final_commit_sha'] = previous['final_commit_sha']
        def update(**values):
            state.update(values, heartbeat_at_utc=utc(), legacy_queue=legacy_activity())
            write(STATE, state)
        env = os.environ.copy()
        env.update({key: '4' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')})
        env['PYTHONUNBUFFERED'] = '1'
        try:
            orphan = previous.get('worker_pid')
            while bg.matching_process(orphan, [SCRIPT.name, '--worker']):
                update(phase='waiting_for_previous_worker', worker_pid=orphan)
                time.sleep(5)
            for phase, option in (('computing', '--worker'), ('verifying', '--verify')):
                attempt = 0
                while True:
                    require_branch()
                    check_snapshot(state)
                    attempt += 1
                    command = [str(PYTHON), str(SCRIPT), option]
                    update(phase=phase, phase_attempt=attempt, command=command, worker_pid=None)
                    try:
                        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL)
                    except OSError as error:
                        update(last_error=str(error), retry_after_seconds=30)
                        time.sleep(30)
                        continue
                    update(worker_pid=child.pid)
                    while child.poll() is None:
                        update(worker_pid=child.pid)
                        time.sleep(5)
                    code = child.wait()
                    update(worker_pid=None, last_exit_code=code)
                    if code == 0:
                        break
                    if code == INTEGRITY_ERROR:
                        update(status='BLOCKED_INTEGRITY', last_error='Scientific manifest or saved artifact mismatch')
                        return code
                    update(last_error=f'{phase} exited {code}; resuming durable checkpoints', retry_after_seconds=5)
                    time.sleep(5)
            update(phase='publishing')
            publish(state, update)
            update(status='DONE', phase='completed', finished_at_utc=utc())
            return 0
        except Exception as error:
            integrity = study.is_integrity_error(error)
            update(status='BLOCKED_INTEGRITY' if integrity else 'SUPERVISOR_ERROR', last_error=str(error), traceback=traceback.format_exc())
            return INTEGRITY_ERROR if integrity else 1


def launch():
    prepare()
    current = read(STATE, {})
    if current:
        check_snapshot(current)
    if bg.matching_process(current.get('supervisor_pid'), [SCRIPT.name, '--supervise']) or current.get('status') == 'DONE':
        return status()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    instance = uuid.uuid4().hex
    with (RUNTIME / 'supervisor.log').open('ab', buffering=0) as log:
        child = subprocess.Popen([str(PYTHON), str(SCRIPT), '--supervise', '--instance', instance], cwd=ROOT,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, close_fds=True)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        state = read(STATE, {})
        if state.get('instance') == instance:
            return status()
        if child.poll() is not None:
            raise RuntimeError('Detached supervisor exited during launch: ' + str(child.returncode))
        time.sleep(.2)
    return {'launch_handshake': 'PENDING', 'supervisor_pid': child.pid, 'computation_terminated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    for action in ('launch', 'supervise', 'worker', 'verify', 'status', 'dry-run'):
        actions.add_argument('--' + action, action='store_true')
    parser.add_argument('--instance', default='manual-' + uuid.uuid4().hex)
    args = parser.parse_args()
    try:
        if args.dry_run:
            value = protocol()
        elif args.status:
            value = status()
        elif args.launch:
            value = launch()
        elif args.supervise:
            return supervise(args.instance)
        elif args.worker:
            return worker()
        else:
            value = verify()
            print(json.dumps(value, ensure_ascii=False, indent=2))
            return 0 if value.get('verified') else 1
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        integrity = study.is_integrity_error(error)
        if not args.dry_run and not args.status:
            write(OUTPUT / ('integrity_error.json' if integrity else 'retryable_error.json'),
                  {'error': str(error), 'traceback': traceback.format_exc(), 'at_utc': utc()})
        print(str(error), file=sys.stderr)
        return INTEGRITY_ERROR if integrity else 1


if __name__ == '__main__':
    raise SystemExit(main())
