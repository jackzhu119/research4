#!/usr/bin/env python3
"""Continue authenticated quantum checkpoints in an independent output tree.

Numerical SCF, optimization, vibrational and IR implementations are imported
unchanged. This module handles location, provenance, checkpoint compatibility
and queue lifetime only. Historical source objects are never edited.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

import pyscf_publication_core_20261111 as electronic
import pyscf_ir_core_20261111 as dipoles
import pyscf_ir_sources_20261111 as sources
import run_pyscf_publication_20261111 as publication
import run_pyscf_ir_20261111 as infrared

TASK = 'reports/all_round_validation_20261112/quantum'
BRANCHES = {
    'publication': 'research/pyscf-publication-recovery-20261111',
    'ir': 'research/pyscf-ir-strength-repair-20261111',
    'frequency': 'research/pyscf-frequency-repair-20261111',
}
OLD_ROOTS = {'publication': 'reports/pyscf_publication_20261111',
             'ir': 'reports/pyscf_ir_20261111'}
TARGETS = ('acid', 'calycosin', 'spinosin')
CURRENT_ROOT = None


def sha(path):
    return electronic.sha256(path)


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    electronic.atomic_json(Path(path), value)


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], check=True,
                          capture_output=True).stdout


def configure(root):
    global CURRENT_ROOT
    root = Path(root).resolve()
    CURRENT_ROOT = root
    out = root / TASK
    out.mkdir(parents=True, exist_ok=True)
    publication.ROOT = root
    publication.OUTPUT = out / 'publication'
    frozen_inputs = Path(__file__).resolve().parents[1] / 'data/research/pyscf_repair_20261111/inputs'
    publication.INPUTS = frozen_inputs if frozen_inputs.is_dir() else root / 'data/research/pyscf_repair_20261111/inputs'
    infrared.ROOT = root
    infrared.OUTPUT = out / 'ir'
    return root, out


def relocate(value, root):
    """Relocate paths, preserving scientific values and original fingerprints."""
    if isinstance(value, list):
        return [relocate(x, root) for x in value]
    if isinstance(value, dict):
        result = {relocate(k, root): relocate(v, root) for k, v in value.items()}
        # Density matrices are explicitly unpublished, reconstructible caches.
        # Do not remove required coordinates, gradients, or dipole references.
        refs = result.get('array_refs')
        if isinstance(refs, dict) and 'density' in refs:
            density = refs['density'].get('path', '')
            if '/.local/' in density and not Path(density).exists():
                refs.pop('density')
                result['unpublished_density_requires_reconstruction'] = True
        return result
    if isinstance(value, str):
        for kind, old in OLD_ROOTS.items():
            marker = '/' + old
            if marker in value and value.startswith('/'):
                return str(root / TASK / kind) + value.split(marker, 1)[1]
        return value
    return value


def _selected_paths(reader, kind):
    old = OLD_ROOTS[kind] + '/'
    keep = ('jobs/', 'controls/', 'derivative_pools/') if kind == 'ir' else ('jobs/', 'controls/water-final/')
    return [p for p in reader.objects if p.startswith(old)
            and p[len(old):].startswith(keep)
            and not any(x.startswith('.') for x in Path(p).parts)]


def authenticate_originals(reader, paths, kind):
    """Check original seals BEFORE any path conversion or consumer receipt."""
    checks = {'result_seals': 0, 'array_seals': 0, 'complete_cases': 0,
              'partial_point_inputs': 0, 'unpublished_density_refs': 0}
    blobs = reader.read_many(paths)
    for path, raw in blobs.items():
        if not path.endswith('.json'):
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            continue
        prefix = path.rsplit('/', 1)[0] + '/'
        if path.endswith('/COMPLETE.json'):
            if 'result_sha256' in value:
                result = reader.read(prefix + 'result.json')
                if hashlib.sha256(result).hexdigest() != value['result_sha256']:
                    raise ValueError('Original source COMPLETE seal mismatch: ' + path)
                checks['result_seals'] += 1
            if 'array_sha256' in value:
                array = reader.read(prefix + 'dipole_derivative.npy')
                if hashlib.sha256(array).hexdigest() != value['array_sha256']:
                    raise ValueError('Original derivative seal mismatch: ' + path)
                checks['array_seals'] += 1
        if path.endswith('/modes/result.json'):
            for artifact in value.get('artifacts', {}).values():
                raw_array = reader.read(prefix + artifact['path'])
                if hashlib.sha256(raw_array).hexdigest() != artifact['sha256']:
                    raise ValueError('Original mode artifact mismatch: ' + path)
                checks['array_seals'] += 1
        if path.endswith('/result.json') and value.get('status') == 'DONE' and 'array_refs' in value:
            for key, reference in value.get('array_refs', {}).items():
                relative = sources._relative(reference['path'])
                if key == 'density' and '/.local/' in relative and relative not in reader.objects:
                    checks['unpublished_density_refs'] += 1
                    continue
                raw_array = reader.read(relative)
                if hashlib.sha256(raw_array).hexdigest() != reference['sha256']:
                    raise ValueError('Original point array mismatch: ' + relative)
                if key in value and key in ('coordinates_bohr', 'gradient', 'dipole_au'):
                    import io
                    if not np.array_equal(np.load(io.BytesIO(raw_array), allow_pickle=False), value[key]):
                        raise ValueError('Original point JSON/array disagreement: ' + relative)
                checks['array_seals'] += 1
            input_path = prefix + 'input.json'
            if input_path in reader.objects:
                inputs = json.loads(reader.read(input_path))
                expected = inputs.pop('input_fingerprint')
                compact = inputs.get('schema') != dipoles.SCHEMA
                canonical = json.dumps(inputs, sort_keys=True, allow_nan=False,
                    **({'separators': (',', ':')} if compact else {})).encode()
                if hashlib.sha256(canonical).hexdigest() != expected or value.get('input_fingerprint') != expected:
                    raise ValueError('Original point input fingerprint mismatch: ' + input_path)
                checks['partial_point_inputs'] += 1
        if kind == 'ir' and path.endswith('/result.json') and 'array_sha256' in value:
            for filename, key in (('ir_arrays.npz', 'array_sha256'), ('ir_modes.csv', 'csv_sha256')):
                if hashlib.sha256(reader.read(prefix + filename)).hexdigest() != value[key]:
                    raise ValueError('Original IR artifact seal mismatch: ' + path)
                checks['array_seals'] += 1
    if kind == 'publication':
        for path in paths:
            if path.endswith('/COMPLETE.json') and path.rsplit('/', 1)[0] + '/modes/result.json' in reader.objects:
                sources.authenticate_case(reader, path.rsplit('/', 1)[0])
                checks['complete_cases'] += 1
    return checks


def import_snapshot(root, kind, commit):
    """Fill absent checkpoints only; never replace newer local computation.

    Every transformed JSON has both source and consumer SHA receipts. Complete
    cases undergo maintained independent authentication before downstream use.
    """
    out = root / TASK
    ledger_path = out / 'imports' / kind / (commit + '.json')
    previous = read(ledger_path)
    if previous is not None:
        return previous
    # A newer remote snapshot is consumed through its own authenticated Git
    # descriptors; it must not be merged into an already-running local case.
    prior = list(ledger_path.parent.glob('*.json')) if ledger_path.parent.exists() else []
    if prior:
        return {'status': 'READ_FROM_FIXED_GIT_SOURCE_WITHOUT_MERGING_ACTIVE_CHECKPOINTS',
                'source_commit': commit, 'kind': kind}
    reader = sources.GitBlobs(root, commit, OLD_ROOTS[kind])
    paths = _selected_paths(reader, kind)
    original_checks = authenticate_originals(reader, paths, kind)
    blobs = reader.read_many(paths)
    imported, skipped = {}, []
    old_prefix = OLD_ROOTS[kind] + '/'
    destination = out / kind
    json_paths = []
    for relative, raw in blobs.items():
        target = destination / relative[len(old_prefix):]
        if target.exists():
            skipped.append(str(target.relative_to(root)))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        imported[str(target.relative_to(root))] = {
            'source_path': relative, 'source_sha256': hashlib.sha256(raw).hexdigest(),
            'source_git_blob': reader.objects[relative],
        }
        if target.suffix == '.json':
            json_paths.append(target)
    # All arrays now exist; path-dependent JSON is converted after the copy.
    for path in json_paths:
        value = read(path)
        converted = relocate(value, root)
        write(path, converted)
    # Independently replay the complete point ledger at the relocated location.
    for path in json_paths:
        if path.name == 'publication_quality.json':
            case_path = path.parent.parent / 'result.json'
            if case_path.exists():
                case = read(case_path)
                if 'publication_gate' in case:
                    value = publication.quality.audit_publication(path.parent,
                        path.parent.parent / 'finite-differences', read(case['center_point_path']),
                        electronic_stability_passed=case['electronic_stability']['tested_electronic_stability_passed'])
                    case['publication_gate'] = value
                    write(case_path, case)
    for path in json_paths:
        if path.name == 'COMPLETE.json':
            result = path.parent / 'result.json'
            value = read(path)
            if 'result_sha256' in value and result.is_file():
                value['result_sha256'] = sha(result)
                write(path, value)
    for relative, record in imported.items():
        record['consumer_sha256'] = sha(root / relative)
    receipt = {'schema': 'quantum-consumer-relocation-v1', 'kind': kind,
               'source_branch': BRANCHES[kind], 'source_commit': commit,
               'original_authentication': original_checks,
               'imported': imported, 'existing_local_files_not_overwritten': skipped,
               'numeric_arrays_changed': False, 'source_objects_immutable': True,
               'density_cache_policy': 'Unpublished caches rebuilt before use; no invented density arrays',
               'at_utc': electronic.utc()}
    write(ledger_path, receipt)
    return receipt


def refresh(root, kinds=('publication', 'ir', 'frequency')):
    """Read-only fetch of the three known source branches at most every 600s."""
    out = root / TASK
    state_path = out / 'source_refresh.json'
    with (out / '.source-refresh.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read(state_path, {'branches': {}})
        now = time.time()
        due = now - state.get('last_fetch_attempt_epoch', 0) >= 600
        if due:
            state['last_fetch_attempt_epoch'] = now
        for kind in kinds:
            branch = BRANCHES[kind]
            entry = state['branches'].setdefault(kind, {})
            if due:
                try:
                    git(root, 'fetch', '--no-tags', 'origin',
                        f'refs/heads/{branch}:refs/remotes/origin/{branch}')
                    entry.update(fetch_status='FETCHED', fetched_at_utc=electronic.utc())
                except Exception as error:
                    entry.update(fetch_status='RETRY_PENDING', fetch_error=str(error))
            try:
                commit = git(root, 'rev-parse', 'refs/remotes/origin/' + branch).decode().strip()
                entry['commit'] = commit
                if kind in OLD_ROOTS:
                    import_snapshot(root, kind, commit)
            except Exception as error:
                entry['import_error'] = str(error)
        write(state_path, state)
    return state


class DirectoryReader:
    """The maintained importer interface over sealed, local consumer results."""
    def __init__(self, root, directory):
        self.root = Path(root).resolve()
        self.objects = {str(p.relative_to(self.root)): 'local-consumer'
                        for p in Path(directory).rglob('*')
                        if p.is_file() and not p.is_symlink()
                        and not any(x.startswith('.') for x in p.relative_to(self.root).parts)}

    def read(self, path):
        if path not in self.objects:
            raise FileNotFoundError(path)
        target = (self.root / path).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError('Scientific source path escaped repository')
        return target.read_bytes()

    def read_many(self, paths):
        return {p: self.read(p) for p in paths}


def authenticated_sources(root):
    out = root / TASK / 'ir'
    descriptors = sources.sync_sources(root, out, refresh=False)
    by_id = {d['id']: d for d in descriptors}
    reader = DirectoryReader(root, root / TASK / 'publication')
    errors = {}
    state = read(out / 'sources_state.json', {})
    manifests = state.setdefault('snapshot_manifest_sha256', {})
    for path in sorted(reader.objects):
        if not path.endswith('/COMPLETE.json'):
            continue
        case = path.rsplit('/', 1)[0]
        if case + '/modes/result.json' not in reader.objects:
            continue
        try:
            descriptor = sources.import_case(reader, case, out,
                'local/all-round-validation-20261112', 'consumer-seal:' + sha(root / path))
            by_id[descriptor['id']] = descriptor
            manifests[descriptor['id']] = descriptor['source_record_sha256']
        except Exception as error:
            errors[case] = str(error)
    write(out / 'local_sources_state.json', {'errors': errors, 'local_case_count': len(by_id),
                                           'at_utc': electronic.utc()})
    write(out / 'sources_state.json', state)
    return sorted(by_id.values(), key=lambda d: ({'water': 0, 'acid': 1,
        'calycosin': 2, 'spinosin': 3}[d['molecule']], d['profile'], d['id']))


def _archive_incompatible(folder, reason):
    """Retain a source cache when the current exact fingerprint cannot reuse it."""
    folder = Path(folder)
    if not folder.exists():
        return
    if CURRENT_ROOT is None:
        raise ValueError('Repository must be configured before cache relocation')
    archive = CURRENT_ROOT / TASK / 'archived_checkpoints' / (str(time.time_ns()) + '-' + folder.name)
    archive.parent.mkdir(parents=True, exist_ok=True)
    original_hashes = {str(p.relative_to(folder)): sha(p) for p in folder.rglob('*')
                       if p.is_file() and not p.is_symlink()}
    folder.rename(archive)
    write(archive / 'consumer_cache_compatibility.json', {
        'reused': False, 'reason': reason, 'at_utc': electronic.utc(),
        'original_checkpoint_path': str(folder),
        'original_files_sha256': original_hashes,
        'original_checkpoint_preserved': True})


def compatible_dipole(*args, **kwargs):
    try:
        return _original_dipole(*args, **kwargs)
    except ValueError as error:
        if 'IR point input fingerprint changed' not in str(error):
            raise
        folder = args[2] if len(args) > 2 else kwargs['output']
        _archive_incompatible(folder, str(error))
        return _original_dipole(*args, **kwargs)


_original_dipole = dipoles.calculate_dipole
dipoles.calculate_dipole = compatible_dipole


_original_completed = electronic._read_completed


def compatible_completed(out, fingerprint, result_name='result.json'):
    try:
        return _original_completed(out, fingerprint, result_name)
    except ValueError as error:
        if 'Scientific input fingerprint changed' not in str(error):
            raise
        _archive_incompatible(out, str(error))
        Path(out).mkdir(parents=True, exist_ok=True)
        return None


electronic._read_completed = compatible_completed

_original_optimize = electronic.optimize_geometry


def compatible_optimize(*args, **kwargs):
    try:
        return _original_optimize(*args, **kwargs)
    except ValueError as error:
        if 'Optimization scientific inputs changed' not in str(error):
            raise
        folder = args[2] if len(args) > 2 else kwargs['out']
        _archive_incompatible(folder, str(error))
        return _original_optimize(*args, **kwargs)


electronic.optimize_geometry = compatible_optimize


def density(point):
    target = Path(point['density_path'])
    if target.is_file():
        return np.load(target, allow_pickle=False)
    center = point.get('final_point', point)
    folder = Path(center['gradient_path']).parent / 'consumer-density-rebuild'
    result = dipoles.qualified_dipole(center['symbols'], np.asarray(center['coordinates_bohr']),
                                     folder, require_density=True)
    energy_error = abs(result['energy_hartree'] - center['energy_hartree'])
    dipole_error = float(np.linalg.norm(np.asarray(result['dipole_au']) - center['dipole_au']))
    if energy_error > 1e-8 or dipole_error > 1e-5:
        raise ValueError('Rebuilt source density does not reproduce its electronic state')
    write(folder / 'source_state_comparison.json', {'qualified': True,
        'source_energy_difference_hartree': energy_error, 'source_dipole_difference_au': dipole_error,
        'source_density_was_unpublished': True, 'at_utc': electronic.utc()})
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(result['density_path'], target)
    return np.load(target, allow_pickle=False)


def prepare_protocol(root):
    out = root / TASK
    path = out / 'protocol.json'
    value = {'schema': 'all-round-quantum-continuation-v1', 'authorized_cpu_seconds': 990000,
        'independent_new_task_budget': True, 'scientific_thresholds_unchanged': True,
        'targets': list(TARGETS), 'source_branches': BRANCHES,
        'runtime_or_retry_limit': None, 'all_attempted_is_not_complete': True,
        'numeric_implementations': {Path(m.__file__).name: sha(m.__file__) for m in (
            electronic, dipoles, publication.modes, publication.quality, infrared.quality)},
        'inputs': {p.name: sha(p) for p in sorted(publication.INPUTS.iterdir()) if p.is_file()},
        'runtime': {key: publication.importlib.metadata.version(key) for key in
                    ('pyscf', 'geometric', 'numpy', 'scipy', 'h5py')},
        'scope': 'Qualified isolated molecule local harmonic/IR; no solid THz peak assignment'}
    old = read(path)
    if old is not None and old != value:
        raise ValueError('Quantum continuation scientific code changed')
    if old is None:
        write(path, value)
    return value


def selected_publication(root, name):
    folder = publication.OUTPUT / 'jobs' / name
    selection = read(folder / 'continuation_selected.json')
    if selection:
        if sha(selection['result_path']) != selection['result_sha256']:
            raise ValueError('Continuation selection seal changed')
        return publication.sealed(Path(selection['result_path']).parent)
    return publication.sealed(folder)


def replay_selected(root, value):
    selected = value.get('selected')
    if not value.get('paper_ready') or not selected:
        return False
    path = Path(selected['path'])
    reader = DirectoryReader(root, path)
    record = sources.authenticate_case(reader, str(path.relative_to(root)))
    if not record['source_minimum_qualified'] or not record['source_ir_qualified']:
        raise ValueError('Selected source case failed full independent qualification')
    return True


def cycle_resolution(cycle, coordinates):
    """Skip unsafe resolutions on either side without sticking at one index."""
    floor = 128 * np.finfo(float).eps * max(1., float(np.max(np.abs(coordinates))))
    unavailable = set()
    while True:
        exponent = (cycle // 2 + 1) * (1 if cycle % 2 == 0 else -1)
        try:
            h = .002 * 2. ** (-exponent)
        except OverflowError:
            h = float('inf')
        if floor < h <= .02:
            return cycle, h
        unavailable.add('small' if h <= floor else 'large')
        if len(unavailable) == 2:
            return None
        cycle += 1


def publication_molecule(root, name):
    out = root / TASK
    previous = selected_publication(root, name)
    if previous is not None and replay_selected(root, previous):
        return previous
    if previous is not None:
        # An exhausted historical finite profile matrix is a retained
        # negative result. A distinct continuation gets new checkpoint paths
        # and a new pair of displacement resolutions, not relaxed gates.
        attempts = sorted((publication.OUTPUT / 'continuations' / name).glob('cycle*'))
        cycle = int(attempts[-1].name[5:]) + 1 if attempts else 0
        if attempts and not (attempts[-1] / 'jobs' / name / 'COMPLETE.json').exists():
            cycle = int(attempts[-1].name[5:])
        old_output, old_steps = publication.OUTPUT, publication.STEPS
        publication.OUTPUT = old_output / 'continuations' / name / f'cycle{cycle:05d}'
        resolution = cycle_resolution(cycle, publication.inputs(name)[1])
        if resolution is None:
            publication.OUTPUT = old_output
            write(out / 'publication_continuation_waiting.json', {
                'molecule': name, 'reason': 'New displacement resolution is outside the resolvable local range',
                'paper_ready': False, 'queue_remains_active': True})
            return {'paper_ready': False, 'status': 'WAITING_FOR_RESOLVABLE_NUMERICAL_MODEL'}
        cycle, h = resolution
        publication.OUTPUT = old_output / 'continuations' / name / f'cycle{cycle:05d}'
        publication.STEPS = (h, h / 2)
        try:
            value = publication.molecule_recovery(name)
            if value.get('paper_ready'):
                write(old_output / 'jobs' / name / 'continuation_selected.json', {
                    'result_path': str(publication.OUTPUT / 'jobs' / name / 'result.json'),
                    'result_sha256': sha(publication.OUTPUT / 'jobs' / name / 'result.json')})
        finally:
            publication.OUTPUT, publication.STEPS = old_output, old_steps
    else:
        return publication.molecule_recovery(name)


def publication_once(root):
    root, out = configure(root)
    refresh(root)
    prepare_protocol(root)
    publication.dm = density
    reader = DirectoryReader(root, publication.OUTPUT / 'controls/water-final')
    for profile in ('direct-grid5', 'df-grid5'):
        case_path = str((publication.OUTPUT / 'controls/water-final' / profile).relative_to(root))
        if case_path + '/COMPLETE.json' in reader.objects:
            checked = sources.authenticate_case(reader, case_path)
            if not checked['source_minimum_qualified'] or not checked['source_ir_qualified']:
                raise ValueError('Replayed water control is scientifically unqualified: ' + profile)
    publication.water_control()
    errors = []
    for name in TARGETS:
        try:
            publication_molecule(root, name)
        except Exception as error:
            errors.append({'molecule': name, 'error': str(error), 'traceback': traceback.format_exc()})
        summarize(root)
    write(out / 'publication_retry_state.json', {'errors': errors, 'retry_limit': None,
                                               'at_utc': electronic.utc()})
    return summarize(root)['publication']


def ir_once(root):
    root, out = configure(root)
    refresh(root)
    prepare_protocol(root)
    descriptors = authenticated_sources(root)
    errors = []
    for descriptor in descriptors:
        folder = infrared.OUTPUT / 'jobs' / descriptor['id']
        completed = [infrared._verify_completed(p.parent) for p in sorted(folder.glob('round*/result.json'))]
        if any(r.get('paper_ready') or r.get('status') == 'DIAGNOSTIC_COMPLETE' for r in completed):
            continue
        try:
            infrared.process_case(descriptor, len(completed))
        except Exception as error:
            errors.append({'id': descriptor['id'], 'molecule': descriptor['molecule'],
                           'error': str(error), 'traceback': traceback.format_exc()})
    # The scientific source ID includes exact geometry/model. Precursors are
    # calculations only; full modes remain necessary to grant qualification.
    for candidate in sources.preview_centers(root, infrared.OUTPUT):
        if any(d['molecule'] == candidate['molecule'] and d['profile'] == candidate['profile']
               and np.array_equal(sources.load_source(d)['coordinates_bohr'], candidate['coordinates_bohr'])
               for d in descriptors):
            continue
        try:
            infrared.process_preview(candidate)
        except Exception as error:
            errors.append({'precursor': candidate['id'], 'error': str(error)})
    write(out / 'ir_retry_state.json', {'errors': errors, 'retry_limit': None,
                                      'at_utc': electronic.utc()})
    infrared.summarize(descriptors)
    return summarize(root)['ir']


def summarize(root):
    root, out = configure(root)
    rows = []
    for name in TARGETS:
        result = read(publication.OUTPUT / 'jobs' / name / 'result.json', {})
        selection = read(publication.OUTPUT / 'jobs' / name / 'continuation_selected.json')
        if selection:
            if sha(selection['result_path']) != selection['result_sha256']:
                raise ValueError('Continuation selection seal changed')
            result = read(selection['result_path'])
        rows.append({'molecule': name, 'paper_ready': result.get('paper_ready') is True,
                     'status': result.get('status', 'PENDING_OR_CHECKPOINTED')})
    pub = {'status': 'COMPLETE' if all(x['paper_ready'] for x in rows) else 'RUNNING_OR_WAITING',
           'molecules': rows, 'all_attempted_is_not_complete': True, 'at_utc': electronic.utc()}
    write(out / 'publication_summary.json', pub)
    ir = read(infrared.OUTPUT / 'summary.json', {'status': 'RUNNING_OR_WAITING'})
    # All authenticated, eligible source geometries are retained independently.
    # One positive acid source cannot hide an unresolved second acid method.
    descriptors = []
    for path in (infrared.OUTPUT / 'sources').glob('*/source.json'):
        case = read(path)
        if case.get('source_minimum_qualified'):
            results = [infrared._verify_completed(p.parent) for p in
                       (infrared.OUTPUT / 'jobs' / case['id']).glob('round*/result.json')]
            descriptors.append({'source_id': case['id'], 'molecule': case['molecule'],
                'profile': case['profile'], 'paper_ready': any(r.get('paper_ready') for r in results)})
    eligible_complete = bool(descriptors) and all(x['paper_ready'] for x in descriptors)
    ir = {**ir, 'eligible_source_cases': descriptors,
          'all_eligible_source_cases_qualified': eligible_complete,
          'status': 'COMPLETE' if ir.get('status') == 'COMPLETE' and eligible_complete else 'RUNNING_OR_WAITING'}
    write(out / 'ir_summary.json', ir)
    return {'publication': pub, 'ir': ir}


def verify(root):
    """Read-only replay of available scientific arrays and result seals."""
    root, out = configure(root)
    checks = []
    for path in publication.OUTPUT.glob('**/modes/result.json'):
        checks.append(publication.modes.verify_analysis(path.parent, write_receipt=False))
    reader = DirectoryReader(root, publication.OUTPUT)
    case_checks = []
    for path in sorted(reader.objects):
        if path.endswith('/COMPLETE.json') and path.rsplit('/', 1)[0] + '/modes/result.json' in reader.objects:
            record = sources.authenticate_case(reader, path.rsplit('/', 1)[0])
            case_checks.append({'id': record['id'], 'minimum_verified': record['source_minimum_qualified']})
    ir_checks = [infrared._verify_completed(p.parent) for p in infrared.OUTPUT.glob('jobs/*/round*/COMPLETE.json')]
    result = {'status': 'VERIFIED_AVAILABLE_ARTIFACTS', 'mode_replays': len(checks),
              'full_source_case_authentications': case_checks,
              'ir_result_seals': len(ir_checks), 'does_not_claim_unfinished_targets_qualified': True,
              'at_utc': electronic.utc()}
    write(out / 'verification.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get(
        'ALL_ROUND_REPOSITORY_ROOT', Path(__file__).resolve().parents[1])))
    group = parser.add_mutually_exclusive_group(required=True)
    for action in ('publication-once', 'ir-once', 'verify', 'summary', 'prepare'):
        group.add_argument('--' + action, action='store_true')
    args = parser.parse_args()
    root, out = configure(args.root)
    try:
        if args.publication_once:
            result = publication_once(root)
        elif args.ir_once:
            result = ir_once(root)
        elif args.verify:
            result = verify(root)
        elif args.summary:
            result = summarize(root)
        else:
            refresh(root)
            result = prepare_protocol(root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        mode = 'publication' if args.publication_once else 'ir' if args.ir_once else 'preparation'
        write(out / (mode + '_last_error.json'), {'error': str(error),
              'traceback': traceback.format_exc(), 'at_utc': electronic.utc(),
              'queue_policy': 'Supervisor retries; scientific gates never relaxed'})
        print(str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
