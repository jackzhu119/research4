#!/usr/bin/env python3
"""Finish only the current calycosin Hessian, using two locked 2-thread workers.

All electronic work uses the original frozen implementation, geometry, density
seed and settings. Existing qualified points are reused without relabeling.
There is no IR queue, new geometry, sensitivity SCF or subsequent molecule.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

RELATIVE_OUTPUT = Path('reports/all_round_validation_20261112/frequency_only')
ORIGINAL_OUTPUT = Path('reports/all_round_validation_20261112')
CASE = Path('reports/all_round_validation_20261112/quantum/publication/jobs/calycosin/df-grid5/round000')
STEPS = (0.002, 0.001)
EXPECTED_ATOMS = 33
EXPECTED_POINTS = 396
REQUIRED_ARTIFACTS = {
    'hessian_h0.npy', 'hessian_h1.npy', 'frequency_analysis/result.json',
    'frequency_analysis/analysis_arrays.npz', 'frequency_analysis/all_modes.csv',
    'frequency_analysis/independent_mode_verification.json',
}
MODULES = ('pyscf_publication_core_20261111', 'pyscf_repair_modes_20261111',
           'run_pyscf_publication_20261111')


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


def lock_file(path, nonblocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a+')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
    except BlockingIOError:
        stream.close()
        return None
    return stream


def load_modules(root):
    frozen = root / ORIGINAL_OUTPUT / '.background/frozen'
    manifest = read(root / ORIGINAL_OUTPUT / 'source_manifest.json', {})
    for name in MODULES:
        relative = 'scripts/' + name + '.py'
        path = frozen / relative
        if not path.is_file() or path.is_symlink() or sha(path) != manifest.get('files', {}).get(relative):
            raise ValueError('Original frozen scientific module changed: ' + relative)
    sys.path.insert(0, str(frozen / 'scripts'))
    modules = []
    for name in MODULES:
        existing = sys.modules.get(name)
        if existing is not None and Path(existing.__file__).resolve() != (frozen / 'scripts' / (name + '.py')).resolve():
            raise ValueError('Refusing a non-frozen scientific module already imported: ' + name)
        module = importlib.import_module(name)
        modules.append(module)
    return modules


def context(root):
    core, modes, publication = load_modules(root)
    core.configure('df-grid5', threads=2)
    case = root / CASE
    optimization = read(case / 'optimize/result.json')
    stability = read(case / 'electronic-stability-00/result.json')
    if not optimization or optimization.get('status') != 'DONE' or not optimization.get('optimizer_converged') or not optimization.get('minimum_stationarity_passed'):
        raise ValueError('Current fixed optimized center is not strictly stationary')
    if not stability or stability.get('status') != 'DONE' or not stability.get('tested_electronic_stability_passed') or not stability.get('supported_for_primary_hessian'):
        raise ValueError('Current fixed electronic state is not supported for this Hessian')
    symbols = optimization['symbols']
    coordinates = np.asarray(optimization['coordinates_bohr'], dtype=float)
    density_path = Path(optimization['density_path'])
    density = np.load(density_path, allow_pickle=False)
    if len(symbols) != EXPECTED_ATOMS or coordinates.shape != (EXPECTED_ATOMS, 3) or not np.isfinite(coordinates).all() or not np.isfinite(density).all():
        raise ValueError('Scope is restricted to the existing 33-atom calycosin center')
    if not density_path.resolve().is_relative_to(case.resolve()):
        raise ValueError('Density seed escaped the existing calycosin case')
    value = {'root': root, 'output': root / RELATIVE_OUTPUT, 'case': case,
             'core': core, 'modes': modes, 'publication': publication,
             'symbols': symbols, 'coordinates': coordinates, 'density': density,
             'optimization': optimization, 'stability': stability}
    value['protocol'] = {'schema': 'calycosin-current-frequency-matrix-only-v1',
        'molecule': 'calycosin', 'profile': 'df-grid5', 'case': str(CASE),
        'steps_bohr': list(STEPS), 'planned_points': EXPECTED_POINTS,
        'worker_count': 2, 'threads_per_electronic_point': 2,
        'total_requested_cpu_threads': 4, 'max_memory_mb_per_point': core.SCF_SETTINGS['max_memory_mb'],
        'coordinates_sha256': publication.digest_array(coordinates),
        'density_seed_sha256': core._array_digest(density),
        'density_file_sha256': sha(density_path),
        'optimization_result_sha256': sha(case / 'optimize/result.json'),
        'stability_result_sha256': sha(case / 'electronic-stability-00/result.json'),
        'scientific_modules': {Path(module.__file__).name: sha(module.__file__) for module in (core, modes, publication)},
        'method': core.settings(), 'independent_ir_scf_points_authorized': 0,
        'queued_molecules_authorized': [], 'new_geometries_or_profiles_authorized': [],
        'qualification_thresholds_do_not_end_point_queue': True}
    return value


def jobs(ctx):
    result = []
    for hi, step in enumerate(STEPS):
        for coordinate in range(ctx['coordinates'].size):
            for sign in (1, -1):
                result.append({'index': len(result), 'step_index': hi, 'step_bohr': step,
                    'coordinate': coordinate, 'sign': sign,
                    'name': f'h{hi}-c{coordinate:03d}-' + ('plus' if sign == 1 else 'minus'),
                    'path': ctx['case'] / 'finite-differences' / f'h{hi}' / (f'c{coordinate:03d}-' + ('plus' if sign == 1 else 'minus'))})
    if len(result) != EXPECTED_POINTS:
        raise ValueError('Frequency-only scope differs from the frozen 396-point matrix')
    return result


def displaced(ctx, job):
    value = np.array(ctx['coordinates'], copy=True)
    value.flat[job['coordinate']] += job['sign'] * job['step_bohr']
    return value


def qualified_point(ctx, job):
    core = ctx['core']
    xyz = displaced(ctx, job)
    fingerprint, _ = core._fingerprint(ctx['symbols'], xyz, None, ctx['density'])
    point = core._read_completed(job['path'], fingerprint)
    if point is None:
        return None
    audit = point.get('electronic_audit', {})
    if point.get('qualified') is not True or point.get('native_converged') is not True or audit.get('qualified') is not True or audit.get('native_converged') is not True:
        return None
    if not np.array_equal(np.asarray(point.get('coordinates_bohr')), xyz):
        raise ValueError('Point coordinates differ from the fixed matrix displacement')
    gradient = np.asarray(point.get('gradient'), dtype=float)
    if gradient.shape != xyz.shape or not np.isfinite(gradient).all():
        raise ValueError('Completed point has incomplete gradient')
    limits = [('fresh_orbital_gradient_norm', core.SCF_SETTINGS['fresh_orbital_gradient_acceptance']),
              ('fresh_energy_difference_hartree', core.SCF_SETTINGS['fresh_energy_difference_acceptance_hartree']),
              ('electron_count_error', core.SCF_SETTINGS['electron_count_acceptance'])]
    for key, limit in limits:
        value = audit.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > limit:
            raise ValueError('Completed point electronic audit differs from the original acceptance: ' + key)
    return point


def prepare(ctx):
    path = ctx['output'] / 'protocol.json'
    old = read(path)
    if old is not None and old != ctx['protocol']:
        raise ValueError('Frozen frequency-only model, geometry, seed or source changed')
    if old is None:
        write(path, ctx['protocol'])
    return ctx['protocol']


def progress(ctx, worker=None, **extra):
    with lock_file(ctx['output'] / '.locks/progress.lock'):
        count = 0
        for job in jobs(ctx):
            point = read(job['path'] / 'result.json', {})
            count += point.get('status') == 'DONE' and point.get('qualified') is True
        value = {'status': 'RUNNING', 'qualified_points': int(count),
                 'planned_points': EXPECTED_POINTS, 'heartbeat_at_utc': utc(),
                 'molecule': 'calycosin', 'profile': 'df-grid5', 'threads_per_point': 2,
                 'maximum_concurrent_points': 2, 'independent_ir_enabled': False,
                 'subsequent_queue_enabled': False, **extra}
        write(ctx['output'] / 'progress.json', value)
        if worker is not None:
            write(ctx['output'] / 'workers' / (worker + '.json'), value)
        return value


def verify_complete(output):
    output = Path(output)
    seal = read(output / 'COMPLETE.json')
    summary = read(output / 'summary.json')
    if seal is None or summary is None:
        return None
    if sha(output / 'summary.json') != seal.get('summary_sha256'):
        raise ValueError('Frequency-only completion summary seal changed')
    if not REQUIRED_ARTIFACTS.issubset(seal.get('artifacts_sha256', {})):
        raise ValueError('Frequency-only completion receipt lacks matrix or replay artifacts')
    for relative, digest in seal.get('artifacts_sha256', {}).items():
        path = output / relative
        if not path.resolve().is_relative_to(output.resolve()) or not path.is_file() or sha(path) != digest:
            raise ValueError('Frequency-only completed scientific artifact changed: ' + relative)
    if summary.get('status') != 'COMPLETE' or summary.get('qualified_points') != EXPECTED_POINTS or summary.get('planned_points') != EXPECTED_POINTS:
        raise ValueError('Completion receipt does not cover the requested whole matrix')
    return summary


def finalize(ctx):
    old = verify_complete(ctx['output'])
    if old is not None:
        return old
    lock = lock_file(ctx['output'] / '.locks/assembly.lock', nonblocking=True)
    if lock is None:
        return None
    try:
        old = verify_complete(ctx['output'])
        if old is not None:
            return old
        points = {0: {1: [], -1: []}, 1: {1: [], -1: []}}
        ledger, authenticated = [], {}
        for job in jobs(ctx):
            point = qualified_point(ctx, job)
            if point is None:
                return None
            points[job['step_index']][job['sign']].append(point)
            audit = point['electronic_audit']
            ledger.append({'qualified': point['qualified'], 'native_converged': point['native_converged'],
                'fresh_orbital_gradient_norm': audit['fresh_orbital_gradient_norm'],
                'energy_recheck_difference_hartree': audit['fresh_energy_difference_hartree'],
                'electron_count_error': audit['electron_count_error']})
            authenticated[str((job['path'] / 'result.json').relative_to(ctx['root']))] = sha(job['path'] / 'result.json')
            for ref in point.get('array_refs', {}).values():
                path = Path(ref['path'])
                if path.resolve().is_relative_to(ctx['root']):
                    authenticated[str(path.relative_to(ctx['root']))] = ref['sha256']
        matrices, matrix_artifacts = [], {}
        for hi, h in enumerate(STEPS):
            # Reuse the original Hessian assembly; its cheap contraction of
            # saved dipoles is discarded and is never a new IR calculation.
            matrix, _ = ctx['publication'].assemble(points[hi], h)
            relative = f'hessian_h{hi}.npy'
            ctx['core'].atomic_array(ctx['output'] / relative, matrix)
            matrix_artifacts[relative] = {'sha256': sha(ctx['output'] / relative),
                                         'bytes': (ctx['output'] / relative).stat().st_size}
            matrices.append(matrix)
        analysis = ctx['modes'].analyze_modes(ctx['symbols'], ctx['coordinates'], matrices,
            None, ctx['output'] / 'frequency_analysis',
            center_quality=ctx['publication'].center_quality(ctx['optimization']['final_point'], ctx['stability']),
            point_quality=ledger, steps_bohr=STEPS)
        replay = ctx['modes'].verify_analysis(ctx['output'] / 'frequency_analysis')
        analysis_artifacts = {}
        for path in sorted((ctx['output'] / 'frequency_analysis').iterdir()):
            if path.is_file() and not path.name.endswith('.tmp'):
                relative = str(path.relative_to(ctx['output']))
                analysis_artifacts[relative] = {'sha256': sha(path), 'bytes': path.stat().st_size}
        summary = {'schema': 'current-frequency-matrix-only-completion-v1', 'status': 'COMPLETE',
            'qualified_points': EXPECTED_POINTS, 'planned_points': EXPECTED_POINTS,
            'molecule': 'calycosin', 'profile': 'df-grid5', 'case': str(CASE),
            'steps_bohr': list(STEPS), 'matrix_artifacts': matrix_artifacts,
            'analysis_artifacts': analysis_artifacts,
            'frequency_minimum_verified': analysis['minimum_verified'],
            'frequency_qualification': analysis['qualification'],
            'signed_frequency_steps': analysis['steps'], 'step_comparison': analysis['step_comparison'],
            'analysis_replay': replay, 'IR_status': 'CANCELLED_BY_USER_NOT_COMPUTED',
            'independent_ir_scf_points_computed': 0, 'new_geometries_or_profiles_computed': 0,
            'subsequent_queue_status': 'CANCELLED_BY_USER', 'matrix_completion_is_separate_from_minimum_qualification': True,
            'authenticated_original_point_sha256': authenticated, 'completed_at_utc': utc(),
            'scope': 'Only the current isolated-molecule Hessian and signed frequencies. No new IR, sensitivity SCF, geometry or later molecule.'}
        write(ctx['output'] / 'summary.json', summary)
        all_artifacts = {k: v['sha256'] for k, v in {**matrix_artifacts, **analysis_artifacts}.items()}
        write(ctx['output'] / 'COMPLETE.json', {'summary_sha256': sha(ctx['output'] / 'summary.json'),
                                               'artifacts_sha256': all_artifacts})
        return summary
    finally:
        lock.close()


def worker(root, worker_id):
    ctx = context(root)
    prepare(ctx)
    worker_name = 'first' if worker_id in ('0', 'first') else 'second'
    order = jobs(ctx)
    if worker_name == 'second':
        order.reverse()
    claimed = 0
    while True:
        complete = verify_complete(ctx['output'])
        if complete is not None:
            return complete
        worked = False
        for job in order:
            if qualified_point(ctx, job) is not None:
                continue
            handle = lock_file(ctx['output'] / '.locks' / (job['name'] + '.lock'), nonblocking=True)
            if handle is None:
                continue
            try:
                if qualified_point(ctx, job) is not None:
                    continue
                claimed += 1
                progress(ctx, worker_name, current_point=job['name'], worker_claimed_points=claimed)
                point = ctx['core'].calculate_point(ctx['symbols'], displaced(ctx, job),
                            job['path'], dm0=ctx['density'])
                if not point.get('qualified'):
                    write(ctx['output'] / 'workers' / (worker_name + '-point-retry.json'),
                          {'point': job['name'], 'status': 'RETRY_PENDING', 'at_utc': utc(),
                           'input_fingerprint': point.get('input_fingerprint'), 'quality_warnings': point.get('quality_warnings')})
                else:
                    qualified_point(ctx, job)
                worked = True
                progress(ctx, worker_name, last_point=job['name'], worker_claimed_points=claimed)
            finally:
                handle.close()
        completed = finalize(ctx)
        if completed is not None:
            return completed
        if not worked:
            time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--worker-id', choices=['0', '1', 'first', 'second'], default='first')
    parser.add_argument('--once', action='store_true', help='Run the fixed current matrix to completion; no subsequent queue')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.prepare:
            result = prepare(context(root))
        elif args.verify:
            result = verify_complete(root / RELATIVE_OUTPUT)
            if result is None:
                result = {'status': 'INCOMPLETE', 'completion_is_not_claimed': True}
        else:
            result = worker(root, args.worker_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        write(root / RELATIVE_OUTPUT / 'workers' / (args.worker_id + '-error.json'),
              {'status': 'RETRY_PENDING', 'error_type': type(error).__name__, 'error': str(error),
               'traceback': traceback.format_exc(), 'at_utc': utc(),
               'original_inputs_and_results_preserved': True})
        print(str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
