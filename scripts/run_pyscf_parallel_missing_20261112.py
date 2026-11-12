#!/usr/bin/env python3
"""Run the previously queued molecular jobs in separate persistent workers.

This dispatcher preserves the existing scientific implementations.  Qualification
failures are recorded and refined; they never exhaust the execution queue.  The
external supervisor owns the one shared CPU authorization for these workers.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

import run_all_round_quantum_20261112 as continuation
import run_pyscf_publication_20261111 as publication
import run_pyscf_ir_20261111 as infrared
import pyscf_ir_sources_20261111 as sources
import pyscf_publication_core_20261111 as electronic


TASK = 'reports/pyscf_parallel_missing_20261112'
TASKS = ('publication_spinosin', 'ir_calycosin', 'ir_spinosin')
READ = continuation.read
WRITE = continuation.write


def configure(root, task, threads):
    root = Path(root).resolve()
    continuation.TASK = TASK
    continuation.CURRENT_ROOT = root
    publication.ROOT = infrared.ROOT = root
    publication.INPUTS = root / 'data/research/pyscf_repair_20261111/inputs'
    publication.OUTPUT = root / TASK / 'publication'
    name = task.split('_', 1)[1]
    infrared.OUTPUT = root / TASK / 'ir' / name
    for folder in (publication.OUTPUT, infrared.OUTPUT):
        folder.mkdir(parents=True, exist_ok=True)
    publication.dm = continuation.density
    # Thread count is a documented runtime setting and remains in point-cache
    # fingerprints.  The physical XC/basis/grid/fitting model is unchanged.
    original_configure = electronic.configure

    def worker_configure(profile='direct-grid5', threads=threads):
        original_configure(profile, threads=worker_configure.threads)
        # Three concurrent workers share the host's 16 GiB limit.  This runtime
        # allocation is included in each original electronic fingerprint.
        electronic.SCF_SETTINGS['max_memory_mb'] = 2500
        return electronic.settings()

    worker_configure.threads = threads
    electronic.configure = worker_configure
    electronic.SCF_SETTINGS['max_memory_mb'] = 2500
    return root, name


def frozen_protocol(root, task, threads):
    numeric = (electronic, publication.modes, publication.quality,
               infrared.core, infrared.quality, sources, publication, infrared, continuation)
    value = {
        'schema': 'parallel-missing-molecular-jobs-v1', 'task': task,
        'all_workers': list(TASKS), 'shared_authorized_cpu_seconds': 900000,
        'independent_new_authorization': True,
        'authorization_is_shared_not_per_worker': True,
        'maximum_runtime_seconds': None, 'maximum_geometry_rounds': None,
        'maximum_retry_count': None, 'qualification_threshold_ends_execution': False,
        'scientific_qualification_thresholds_unchanged': True,
        'worker_runtime': {'threads': threads, 'max_memory_mb': 2500},
        'source_branches': list(sources.BRANCHES),
        'numeric_code_sha256': {Path(m.__file__).name: electronic.sha256(m.__file__) for m in numeric},
        'dispatcher_sha256': electronic.sha256(__file__),
        'input_sha256': {p.name: electronic.sha256(p) for p in
                         sorted(publication.INPUTS.iterdir()) if p.is_file()},
        'versions': {name: importlib.metadata.version(name) for name in
                     ('pyscf', 'geometric', 'numpy', 'scipy', 'h5py')},
        'scope': 'Qualified isolated-molecule local harmonic/IR data only.',
        'unqualified_historical_minima_are_not_recomputed_for_ir': True,
        'existing_remote_jobs_modified': False,
    }
    path = root / TASK / 'workers' / (task + '_protocol.json')
    old = READ(path)
    if old is not None and old != value:
        raise ValueError('Parallel task frozen scientific protocol changed')
    if old is None:
        WRITE(path, value)
    return value


def state(root, task, **values):
    path = root / TASK / 'workers' / (task + '.json')
    value = READ(path, {})
    value.update(values, task=task, pid=os.getpid(), heartbeat_at_utc=electronic.utc())
    WRITE(path, value)


def error_record(root, task, error, **values):
    record = dict(type=type(error).__name__, error=str(error),
                  traceback=traceback.format_exc(), at_utc=electronic.utc(), **values)
    path = root / TASK / 'errors' / task / (str(time.time_ns()) + '.json')
    WRITE(path, record)
    state(root, task, status='RETRY_PENDING', last_error=record,
          retry_limit=None, qualification_failure_ends_execution=False)


def control_evidence(root):
    """Replay the already completed upstream water cases before selection."""
    branch = sources.BRANCHES[0]
    commit = sources._git(root, 'rev-parse', 'refs/remotes/origin/' + branch).decode().strip()
    reader = sources.GitBlobs(root, commit, sources.PUBLICATION_ROOT + '/controls/water-final')
    summary_path = sources.PUBLICATION_ROOT + '/controls/water-final/result.json'
    summary_raw = reader.read(summary_path)
    seal = json.loads(reader.read(summary_path.rsplit('/', 1)[0] + '/COMPLETE.json'))
    if seal.get('result_sha256') != hashlib.sha256(summary_raw).hexdigest():
        raise ValueError('Upstream aggregate water-control seal changed')
    if json.loads(summary_raw).get('control_passed') is not True:
        raise ValueError('Upstream aggregate water control did not pass')
    controls = []
    for profile in ('direct-grid5', 'df-grid5'):
        path = sources.PUBLICATION_ROOT + '/controls/water-final/' + profile
        value = sources.authenticate_case(reader, path)
        if not value['source_minimum_qualified'] or not value['source_ir_qualified']:
            raise ValueError('Upstream water control is not independently qualified: ' + profile)
        controls.append({'profile': profile, 'source_id': value['id']})
    receipt = {'status': 'AUTHENTICATED_UPSTREAM_CONTROLS', 'branch': branch,
               'commit': commit, 'controls': controls, 'at_utc': electronic.utc()}
    WRITE(publication.OUTPUT / 'controls_replayed.json', receipt)
    return receipt


def next_resolution(cycle, coordinates):
    if cycle == 0:
        return cycle, .002
    found = continuation.cycle_resolution(cycle - 1, coordinates)
    return None if found is None else (found[0] + 1, found[1])


def publication_once(root, name):
    """Advance one profile/geometry cycle without the legacy six-round cap."""
    task = 'publication_' + name
    output = publication.OUTPUT
    root_result = output / 'jobs' / name
    previous = publication.sealed(root_result)
    if previous is not None and previous.get('paper_ready'):
        continuation.replay_selected(root, previous)
        return previous
    controls = control_evidence(root)
    _, reference, initial_density = publication.inputs(name)
    queue_path = output / 'queue.json'
    queue = READ(queue_path, {'cycle': 0, 'profile_index': 0, 'seeds': {}, 'attempts': []})
    profiles = tuple(publication.PROFILES)
    cycle = int(queue['cycle'])
    index = int(queue['profile_index'])
    profile = profiles[index]
    seed = queue['seeds'].get(profile)
    coords = np.asarray(seed['coordinates_bohr']) if seed else np.array(reference)
    if seed and Path(seed.get('density_path', '')).is_file():
        density = np.load(seed['density_path'], allow_pickle=False)
    elif seed and (seed.get('gradient_path') or seed.get('final_point')):
        density = continuation.density(seed)
    else:
        # A density is a reconstructible guess, not a scientific observation.
        # The strict optimization still re-establishes the electronic state.
        density = initial_density if not seed else None
    resolution = next_resolution(cycle, coords)
    if resolution is None:
        state(root, task, status='WAITING_FOR_RESOLVABLE_NUMERICAL_MODEL',
              queue_remains_active=True, paper_ready=False)
        return {'paper_ready': False, 'status': 'WAITING_FOR_RESOLVABLE_NUMERICAL_MODEL'}
    actual_cycle, h = resolution
    if actual_cycle != cycle:
        queue['cycle'] = actual_cycle
        cycle = actual_cycle
        WRITE(queue_path, queue)
    old_steps = publication.STEPS
    publication.STEPS = (h, h / 2)
    out = root_result / profile / f'round{cycle:05d}'
    state(root, task, status='RUNNING', phase='STRICT_GEOMETRY_AND_FULL_HESSIAN',
          molecule=name, profile=profile, cycle=cycle, steps_bohr=list(publication.STEPS),
          maximum_geometry_rounds=None, maximum_retries=None)
    try:
        case = publication.stationary_case(name, profile, coords, density, out)
        attempt = {'profile': profile, 'round': cycle, 'path': str(out),
                   'status': case['status'], 'paper_ready': case.get('paper_ready', False)}
        queue['attempts'].append(attempt)
        if case.get('coordinates_bohr') and case.get('density_path'):
            center = READ(case['center_point_path'])
            queue['seeds'][profile] = dict(center, coordinates_bohr=case['coordinates_bohr'],
                                           density_path=case['density_path'])
        if case.get('paper_ready'):
            sensitivity = publication.direct_soft_checks(name, profile, case, out / 'sensitivity')
            if sensitivity.get('directional_sensitivity_passed'):
                value = {'status': 'QUALIFIED_LOCAL_HARMONIC_DATA', 'molecule': name,
                    'paper_ready': True, 'selected': {'profile': profile, 'round': cycle,
                        'path': str(out), 'case_result_sha256': electronic.sha256(out / 'result.json'),
                        'sensitivity': sensitivity}, 'attempts': queue['attempts'],
                    'water_controls': controls, 'finished_at_utc': electronic.utc(),
                    'scope': 'Isolated-molecule local harmonic data only.'}
                publication.seal(root_result, value)
                continuation.replay_selected(root, value)
                return value
            attempt['status'] = 'REFINE_GRID_OR_FITTING_REQUIRED'
        elif 'analysis' in case:
            follow = publication.negative_branches(name, profile, case, out / 'negative-mode-recovery')
            if follow.get('selected'):
                selected = follow['selected']
                relaxed = out / 'negative-mode-recovery' / f"sign{selected['sign']:+d}" / 'relax/result.json'
                queue['seeds'][profile] = READ(relaxed, selected)
        queue['profile_index'] = (index + 1) % len(profiles)
        if queue['profile_index'] == 0:
            queue['cycle'] += 1
        WRITE(queue_path, queue)
        state(root, task, status='REFINEMENT_PENDING', paper_ready=False,
              scientific_failure_ends_execution=False)
        return case
    finally:
        publication.STEPS = old_steps


def local_reader(root):
    return continuation.DirectoryReader(root, publication.OUTPUT)


def descriptors_for_target(root, name):
    output = infrared.OUTPUT
    # Both IR workers serialize remote-ref refresh, while keeping every result
    # and scientific source snapshot in their own output directory.
    lock_path = root / TASK / '.remote-refresh.lock'
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        descriptors = sources.sync_sources(root, output, refresh=True)
    result = {d['id']: d for d in descriptors if d['molecule'] == name}
    reader = local_reader(root)
    registry_path = output / 'sources_state.json'
    registry = READ(registry_path, {})
    manifests = registry.setdefault('snapshot_manifest_sha256', {})
    errors = {}
    for path in sorted(reader.objects):
        if not path.endswith('/COMPLETE.json'):
            continue
        folder = path.rsplit('/', 1)[0]
        if folder + '/modes/result.json' not in reader.objects:
            continue
        try:
            case = json.loads(reader.read(folder + '/result.json'))
            if case.get('molecule') != name:
                continue
            descriptor = sources.import_case(reader, folder, output,
                'local/parallel-missing-20261112', 'consumer-seal:' + electronic.sha256(root / path))
            result[descriptor['id']] = descriptor
            manifests[descriptor['id']] = descriptor['source_record_sha256']
        except Exception as error:
            errors[folder] = str(error)
    WRITE(registry_path, registry)
    WRITE(output / 'local_sources_state.json', {'errors': errors,
          'local_source_count': len(result), 'at_utc': electronic.utc()})
    return sorted(result.values(), key=source_priority)


def source_priority(case):
    profile = {'df-grid5': 0, 'df-grid7': 1, 'direct-grid5': 2, 'direct-grid7': 3}.get(case['profile'], 4)
    branch = case.get('source', {}).get('branch', '')
    origin = 0 if branch.startswith('local/') else 1 if 'publication-recovery' in branch else 2
    return origin, profile, case['id']


def previews_for_target(root, name):
    output = infrared.OUTPUT
    previews = {p['id']: p for p in sources.preview_centers(root, output) if p['molecule'] == name}
    reader = local_reader(root)
    for path in sorted(reader.objects):
        if not path.endswith('/result.json'):
            continue
        folder = path.rsplit('/', 2)[-2]
        if folder != 'optimize' and not folder.startswith('reoptimized-stable-state-'):
            continue
        try:
            if f'/jobs/{name}/' not in path:
                continue
            candidate = sources.authenticate_preview(reader, path, name)
            evidence = candidate.pop('_evidence')
            candidate['source'].update(branch='local/parallel-missing-20261112',
                commit='optimization-sha256:' + hashlib.sha256(reader.read(path)).hexdigest())
            destination = output / 'precursors' / candidate['id']
            manifest_path = destination / 'source.json'
            if not manifest_path.exists():
                hashes = {}
                for relative, raw in evidence.items():
                    target = destination / 'evidence' / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw)
                    hashes['evidence/' + relative] = hashlib.sha256(raw).hexdigest()
                candidate['snapshot_hashes'] = hashes
                WRITE(manifest_path, candidate)
            else:
                frozen = READ(manifest_path)
                identity = {'symbols': frozen['symbols'], 'coordinates_bohr': frozen['coordinates_bohr'],
                            'method': frozen['method']}
                digest = hashlib.sha256(sources._canonical(identity)).hexdigest()
                if frozen['id'] != candidate['id'] or digest != frozen['id']:
                    raise ValueError('Local precursor scientific identity changed')
                if frozen.get('source_minimum_qualified') is not False or frozen.get('source_ir_qualified') is not False:
                    raise ValueError('Local precursor qualification was changed')
                for relative, digest in frozen['snapshot_hashes'].items():
                    if electronic.sha256(destination / relative) != digest:
                        raise ValueError('Local precursor frozen artifact changed')
                candidate = frozen
            previews[candidate['id']] = candidate
        except Exception as error:
            error_record(root, 'ir_' + name, error, source_path=path)
    return sorted(previews.values(), key=source_priority)


def infrared_once(root, name):
    task = 'ir_' + name
    descriptors = descriptors_for_target(root, name)
    completed = []
    for descriptor in descriptors:
        if not descriptor.get('source_minimum_qualified'):
            # Old imaginary-mode diagnostics cannot qualify molecular IR.
            # Spend this independent budget on current stationary precursors
            # and verified minima, retaining the old source as evidence only.
            WRITE(infrared.OUTPUT / 'waiting_sources' / (descriptor['id'] + '.json'), {
                'source_id': descriptor['id'], 'molecule': name,
                'status': 'WAITING_FOR_QUALIFIED_MINIMUM', 'paper_ready': False,
                'source': descriptor.get('source'), 'at_utc': electronic.utc()})
            continue
        folder = infrared.OUTPUT / 'jobs' / descriptor['id']
        results = [infrared._verify_completed(p.parent)
                   for p in sorted(folder.glob('round*/result.json'))]
        if any(r.get('paper_ready') or r.get('status') == 'DIAGNOSTIC_COMPLETE' for r in results):
            completed.extend(results)
            continue
        state(root, task, status='RUNNING', phase='INDEPENDENT_FULL_IR',
              molecule=name, profile=descriptor['profile'], source_id=descriptor['id'])
        try:
            result = infrared.process_case(descriptor, len(results))
            completed.append(result)
        except Exception as error:
            error_record(root, task, error, source_id=descriptor['id'])
    previews = []
    for candidate in previews_for_target(root, name):
        if any(d['profile'] == candidate['profile'] and np.array_equal(
                sources.load_source(d)['coordinates_bohr'], candidate['coordinates_bohr'])
                for d in descriptors):
            continue
        state(root, task, status='RUNNING', phase='STATIONARY_PRECURSOR_IR_DERIVATIVES',
              molecule=name, profile=candidate['profile'], precursor_id=candidate['id'],
              source_minimum_qualified=False)
        try:
            previews.append(infrared.process_preview(candidate))
        except Exception as error:
            error_record(root, task, error, precursor_id=candidate['id'])
    WRITE(infrared.OUTPUT / 'precursor_progress.json', {'cases': previews,
          'updated_at_utc': electronic.utc(), 'precursors_never_grant_paper_qualification': True})
    result = infrared.summarize(descriptors)
    ready = any(p.get('paper_ready') for p in completed)
    state(root, task, status='QUALIFIED' if ready else 'WAITING_OR_REFINING',
          paper_ready=ready, source_count=len(descriptors), precursor_count=len(previews),
          maximum_retries=None, quality_failure_ends_execution=False)
    return {'paper_ready': ready, 'summary': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=TASKS)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--threads', type=int, choices=(1, 2), default=1)
    parser.add_argument('--once', action='store_true', help='Advance one resumable queue scan.')
    parser.add_argument('--poll-seconds', type=float, default=30.)
    args = parser.parse_args()
    root, name = configure(args.root, args.task, args.threads)
    worker_dir = root / TASK / 'workers'
    worker_dir.mkdir(parents=True, exist_ok=True)
    with (worker_dir / (args.task + '.lock')).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen_protocol(root, args.task, args.threads)
        state(root, args.task, status='STARTING', started_at_utc=electronic.utc(),
              threads=args.threads, max_memory_mb=2500, runtime_timeout=None, maximum_retries=None,
              maximum_refinement_rounds=None)
        for scan in itertools.count():
            try:
                result = (publication_once(root, name) if args.task.startswith('publication_')
                          else infrared_once(root, name))
                state(root, args.task, completed_scans=scan + 1)
                if result.get('paper_ready'):
                    state(root, args.task, status='QUALIFIED', paper_ready=True,
                          finished_at_utc=electronic.utc())
                    WRITE(root / TASK / 'task_completion' / (args.task + '.json'), {
                        'task': args.task, 'status': 'COMPLETE', 'paper_ready': True,
                        'completed_at_utc': electronic.utc(),
                        'output': str(publication.OUTPUT if args.task.startswith('publication_') else infrared.OUTPUT),
                        'scientific_qualification_checked': True})
                    return 0
            except Exception as error:
                error_record(root, args.task, error, scan=scan)
                if args.once:
                    return 2
            if args.once:
                return 0
            time.sleep(max(1., args.poll_seconds))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
