"""Restartable, geometry-isolated Spinosin optimization and full FD Hessians.

There is no global geometry-round, CPU, wall-time or retry limit. Finite native
SCF/optimizer segments are individual recovery/checkpoint units. Scientific
qualification remains strict; exhausted numerical alternatives produce an
explicit waiting-for-new-protocol state, never a false minimum certificate.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VENDOR = Path(__file__).resolve().parent / 'quantum_gap_vendor_20261112'
DESIGN = {
    'schema': 'spinosin-gap-science-20261112-v1',
    'production_cases': ['C003', 'C008'], 'formula': 'C28H32O15',
    'atoms': 75, 'electron_count': 320, 'charge': 0, 'spin': 0,
    'profiles': ['df-grid5', 'df-grid7', 'direct-grid7'],
    'steps_bohr': [0.002, 0.001], 'gradients_and_total_dipoles_per_center': 900,
    'additional_ir_cartesian_dipoles': 450, 'normal_mode_ir_dipoles': 1314,
    'soft_direction_sensitivity': {'directions': 3, 'step_bohr': 0.002,
        'df_grid5_controls': ['direct-grid5', 'df-grid7'],
        'maximum_relative_curvature_change': 0.1,
        'scope': 'Directional sensitivity only, not a full independent direct-integral Hessian'},
    'negative_mode_amplitudes_bohr': [0.01, 0.03, 0.08, 0.2],
    'optimizer_segment_maxsteps': 12,
    'global_geometry_round_limit': None, 'global_retry_limit': None,
    'cpu_stop_limit': None, 'wall_stop_limit': None,
    'threads_per_lane': 2, 'max_memory_mb_hint': 3000,
    'dense_df_tensor_forbidden': True,
    'point_cache_policy': 'Seal geometry/gradient/dipole/audit; remove reconstructible point DM/chk after commit',
    'scope': 'Gas-phase local harmonic/IR predictions at the explicitly qualified numerical model; no global minimum, direct-model equivalence or herb-solid assignment claim',
}
PROTOCOL = DESIGN


class ScientificIntegrityError(ValueError):
    pass


class UnsupportedStreamingPath(RuntimeError):
    pass


def install_streaming_guard(core=None):
    """Keep native pure-PBE J-only DF; prohibit full three-index tensors.

    This wraps the factory, not the scientific vendor source. Never inspect
    ``mf._cderi``: its property can itself build the forbidden tensor.
    """
    core = _vendor()['core'] if core is None else core
    factory = getattr(core, 'make_scf', None)
    if factory is None or getattr(factory, '_spinosin_j_only_streaming_guard', False):
        return
    def guarded_factory(*args, **kwargs):
        mf = factory(*args, **kwargs)
        fitting = getattr(mf, 'with_df', None)
        if fitting is None:
            return mf
        if fitting._cderi is not None:
            raise UnsupportedStreamingPath('DENSE_DF_DISABLED_J_ONLY_STREAMING_REQUIRED: prebuilt _cderi')
        if getattr(mf.mol, 'incore_anyway', False):
            raise UnsupportedStreamingPath('DENSE_DF_DISABLED_J_ONLY_STREAMING_REQUIRED: incore_anyway')
        def forbidden(*unused_args, **unused_kwargs):
            raise UnsupportedStreamingPath('DENSE_DF_DISABLED_J_ONLY_STREAMING_REQUIRED: build/loop')
        original_get_jk = fitting.get_jk
        signature = inspect.signature(original_get_jk)
        def guarded_get_jk(*call_args, **call_kwargs):
            bound = signature.bind_partial(*call_args, **call_kwargs)
            with_k = bound.arguments.get('with_k', signature.parameters['with_k'].default)
            if with_k:
                raise UnsupportedStreamingPath('DENSE_DF_DISABLED_J_ONLY_STREAMING_REQUIRED: exchange/with_k')
            if fitting._cderi is not None:
                raise UnsupportedStreamingPath('DENSE_DF_DISABLED_J_ONLY_STREAMING_REQUIRED: unexpected _cderi')
            result = original_get_jk(*call_args, **call_kwargs)
            if fitting._cderi is not None:
                raise UnsupportedStreamingPath('DENSE_DF_DISABLED_J_ONLY_STREAMING_REQUIRED: constructed _cderi')
            return result
        fitting.build = forbidden
        fitting.loop = forbidden
        fitting.get_jk = guarded_get_jk
        return mf
    guarded_factory._spinosin_j_only_streaming_guard = True
    guarded_factory._quantum_gap_streaming_jonly_guard = True
    core.make_scf = guarded_factory


install_jonly_streaming_guard = install_streaming_guard


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _array_hash(value):
    value = np.ascontiguousarray(value, dtype='<f8')
    return hashlib.sha256(str(value.shape).encode() + value.tobytes()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.tmp-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _record(path, payload):
    _atomic_json(path, {'payload': payload, 'payload_sha256': _hash(payload)})
    return payload


def _read_record(path):
    if not Path(path).exists():
        return None
    value = json.loads(Path(path).read_text())
    if value.get('payload_sha256') != _hash(value.get('payload')):
        raise ScientificIntegrityError('Scientific record seal mismatch: ' + str(path))
    return value['payload']


def _array(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.tmp-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.save(stream, np.asarray(value), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {'path': str(path.resolve()), 'sha256': _file_hash(path)}


def _check_refs(refs):
    for ref in refs.values():
        if not Path(ref['path']).is_file() or _file_hash(ref['path']) != ref['sha256']:
            raise ScientificIntegrityError('Scientific array changed or missing: ' + ref['path'])


def _load_density(ref):
    if ref is None:
        return None
    _check_refs({'density': ref})
    density = np.load(ref['path'], allow_pickle=False)
    if not np.isfinite(density).all():
        raise ScientificIntegrityError('Restart density is not finite')
    return density


def _vendor():
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    return {
        'core': importlib.import_module('pyscf_publication_core_20261111'),
        'modes': importlib.import_module('pyscf_repair_modes_20261111'),
        'quality': importlib.import_module('pyscf_publication_quality_20261111'),
        'helpers': importlib.import_module('run_pyscf_publication_20261111'),
    }


def _runtime_versions():
    versions = {'numpy': np.__version__}
    for name in ('pyscf', 'geometric', 'scipy', 'h5py'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


@contextmanager
def _bounded_optimizer_segment(core):
    """Use short geomeTRIC restart intervals without editing vendored bytes."""
    if not getattr(core, '__name__', '').startswith('pyscf_publication_core_'):
        yield
        return
    solver = importlib.import_module('pyscf.geomopt.geometric_solver')
    original = solver.kernel
    def bounded(*args, **kwargs):
        kwargs['maxsteps'] = min(int(kwargs.get('maxsteps', DESIGN['optimizer_segment_maxsteps'])),
                                 DESIGN['optimizer_segment_maxsteps'])
        return original(*args, **kwargs)
    solver.kernel = bounded
    try:
        yield
    finally:
        solver.kernel = original


def _resource_guard_evidence(folder, error=None):
    marker = 'DENSE_DF_DISABLED_J_ONLY_STREAMING_REQUIRED'
    if error is not None and marker in str(error):
        return True
    for path in Path(folder).rglob('electronic-attempts.json'):
        if marker in path.read_text():
            return True
    return False


def _validate_case(case):
    case = copy.deepcopy(case)
    symbols = case.get('symbols')
    coordinates = np.asarray(case.get('initial_geometry_bohr'), dtype=float)
    control = case.get('engineering_control') is True and case.get('case_id') == 'WATER_CONTROL'
    expected = Counter({'O': 1, 'H': 2}) if control else Counter({'C': 28, 'H': 32, 'O': 15})
    if not isinstance(symbols, list) or Counter(symbols) != expected:
        raise ValueError('Complete correct molecular identity is required')
    if coordinates.shape != (len(symbols), 3) or not np.isfinite(coordinates).all():
        raise ValueError('Complete finite initial_geometry_bohr is required')
    if case.get('charge') != 0 or case.get('spin') != 0:
        raise ValueError('Only the explicitly authorized neutral closed-shell state is supported')
    if not control and case.get('case_id') not in DESIGN['production_cases']:
        raise ValueError('Only independently assigned C003/C008 production cases are authorized here')
    electrons = sum({'C': 6, 'H': 1, 'O': 8}[symbol] for symbol in symbols)
    if electrons != (10 if control else 320) or case.get('formula') != ('H2O' if control else 'C28H32O15'):
        raise ValueError('Molecular formula/electron count mismatch')
    if case.get('electron_count', electrons) != electrons:
        raise ValueError('Declared electron count mismatch')
    if not control:
        for kind in ('xyz', 'sdf'):
            source = case.get(f'source_{kind}_path')
            digest = case.get(f'source_{kind}_sha256')
            if source is not None and digest is not None:
                path = Path(source)
                path = path if path.is_absolute() else ROOT / path
                if not path.is_file() or _file_hash(path) != digest:
                    raise ScientificIntegrityError('Original conformer provenance changed: ' + str(path))
    return case, symbols, coordinates


def prepare_fingerprint(case, threads=2, max_memory_mb=3000):
    case, _, _ = _validate_case(case)
    if isinstance(threads, bool) or threads not in (1, 2):
        raise ValueError('Validated thread counts are one or two')
    if isinstance(max_memory_mb, bool) or not isinstance(max_memory_mb, int) or max_memory_mb <= 0:
        raise ValueError('A positive memory hint is required')
    code = {Path(__file__).name: _file_hash(__file__)}
    code.update({path.name: _file_hash(path) for path in sorted(VENDOR.glob('*.py'))})
    ir = Path(__file__).with_name('spinosin_gap_ir_20261112.py')
    if ir.exists():
        code[ir.name] = _file_hash(ir)
    payload = {'design': DESIGN, 'case': case, 'threads': threads,
               'max_memory_mb_hint': max_memory_mb, 'code_sha256': code,
               'runtime_versions': _runtime_versions()}
    return {'source_fingerprint': _hash(payload), 'payload': payload}


def _stop(stop_requested):
    if stop_requested is not None and stop_requested():
        raise InterruptedError('Manual stop requested at a committed scientific boundary')


def _progress(progress, stop_requested, **values):
    values.setdefault('checkpoint_committed', True)
    if progress is not None:
        progress(values)
    _stop(stop_requested)


def _save_state(output, state, progress, stop_requested, **event):
    _record(output / 'workflow_state.json', state)
    _progress(progress, stop_requested, case_id=state['case_id'], generation=state['generation'], **event)


def _prune_local(root, keep=()):
    root = Path(root)
    keep = {str(Path(path).resolve()) for path in keep}
    removed = []
    locals_to_check = ([root] if root.name == '.local' and root.is_dir() else []) + list(root.rglob('.local'))
    for local in locals_to_check:
        if not local.is_dir() or local.is_symlink():
            continue
        for path in list(local.rglob('*')):
            if path.is_file() and str(path.resolve()) not in keep:
                removed.append({'path': str(path), 'sha256': _file_hash(path),
                                'bytes': path.stat().st_size, 'reconstructible_cache': True})
                path.unlink()
        for path in sorted(local.rglob('*'), key=lambda item: len(item.parts), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        if not any(local.iterdir()):
            local.rmdir()
    if removed:
        ledger_root = root.parent if root.name == '.local' else root
        _record(ledger_root / f'cache_cleanup_{time.time_ns()}.json', {'removed_local_caches': removed})


def _clean_point_result(raw, folder, request, density_ref=None):
    coordinates = np.asarray(raw.get('coordinates_bohr'))
    gradient, dipole = np.asarray(raw.get('gradient')), np.asarray(raw.get('dipole_au'))
    if (coordinates.shape != np.asarray(request['coordinates_bohr']).shape
            or not np.array_equal(coordinates, request['coordinates_bohr'])
            or gradient.shape != coordinates.shape or dipole.shape != (3,)
            or not all(np.isfinite(value).all() for value in (coordinates, gradient, dipole))):
        raise ScientificIntegrityError('Incomplete or mismatched scientific point arrays')
    if raw.get('symbols') != request['symbols'] or raw.get('settings') != request['settings']:
        raise ScientificIntegrityError('Point molecular identity/model differs from its frozen request')
    refs = {'coordinates_bohr': _array(folder / 'coordinates_bohr.npy', coordinates),
            'gradient': _array(folder / 'gradient.npy', gradient),
            'dipole_au': _array(folder / 'dipole_au.npy', dipole)}
    value = copy.deepcopy(raw)
    value.update(array_refs=refs, gradient_path=refs['gradient']['path'],
                 density_path=None if density_ref is None else density_ref['path'],
                 input_fingerprint=_hash({'request': request, 'vendor_input_fingerprint': raw.get('input_fingerprint')}),
                 request_fingerprint=_hash(request), source_fingerprint=request['source_fingerprint'],
                 density_cache_policy='Only explicit center/restart densities retained; displaced densities are reconstructible')
    if density_ref is not None:
        value['array_refs']['density'] = density_ref
    return value


def _point(symbols, coordinates, folder, core, source_fingerprint, density,
           keep_density, progress=None, stop_requested=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    request = {'symbols': list(symbols), 'coordinates_bohr': np.asarray(coordinates).tolist(),
               'settings': core.settings(), 'source_fingerprint': source_fingerprint,
               'initial_density_hash': None if density is None else _array_hash(density),
               'keep_density': bool(keep_density)}
    saved = _read_record(folder / 'POINT_SEAL.json')
    if saved is not None:
        if saved['request'] != request:
            raise ScientificIntegrityError('Point request changed; geometry/model/seed caches cannot be mixed')
        _check_refs(saved['result']['array_refs'])
        path = folder / 'result.json'
        if path.exists() and json.loads(path.read_text()) != saved['result']:
            raise ScientificIntegrityError('Saved point result differs from its sealed scientific record')
        if not path.exists():
            _atomic_json(path, saved['result'])
        _prune_local(folder / 'vendor_attempts')
        return saved['result']
    retry = _read_record(folder / 'retry_state.json')
    if retry is None:
        retry = {'request_fingerprint': _hash(request), 'attempt': 0, 'density_ref': None}
    if retry['request_fingerprint'] != _hash(request):
        raise ScientificIntegrityError('Incomplete point retry request changed')
    while True:
        _stop(stop_requested)
        attempt = retry['attempt']
        seed = density if retry['density_ref'] is None else _load_density(retry['density_ref'])
        out = folder / 'vendor_attempts' / f'attempt_{attempt:06d}'
        try:
            raw = core.calculate_point(symbols, coordinates, out, dm0=seed)
            if raw.get('qualified') is not True or raw.get('status') != 'DONE':
                if 'DENSE_DF_DISABLED_J_ONLY_STREAMING_REQUIRED' in json.dumps(raw):
                    raise UnsupportedStreamingPath('DENSE_DF_DISABLED_J_ONLY_STREAMING_REQUIRED')
                raise RuntimeError('Native electronic audit did not qualify this point')
            density_ref = None
            if keep_density:
                density_ref = _array(folder / '.local' / 'frozen_density.npy',
                                     np.load(raw['density_path'], allow_pickle=False))
            value = _clean_point_result(raw, folder, request, density_ref)
            _atomic_json(folder / 'result.json', value)
            _record(folder / 'POINT_SEAL.json', {'request': request, 'result': value})
            _prune_local(folder / 'vendor_attempts')
            _prune_local(folder, keep=() if density_ref is None else (density_ref['path'],))
            _progress(progress, stop_requested, stage='point_committed', point_path=str(folder),
                      source_fingerprint=source_fingerprint)
            return value
        except (InterruptedError, ScientificIntegrityError, UnsupportedStreamingPath):
            raise
        except Exception as error:
            if (folder / 'POINT_SEAL.json').exists():
                # A publisher/cache-cleanup failure after scientific commit is
                # operational; it must not trigger another electronic solve.
                raise
            raw_path = out / 'result.json'
            raw = json.loads(raw_path.read_text()) if raw_path.exists() else {}
            if ('DENSE_DF_DISABLED_J_ONLY_STREAMING_REQUIRED' in json.dumps(raw)
                    or _resource_guard_evidence(out, error)):
                raise UnsupportedStreamingPath('DENSE_DF_DISABLED_J_ONLY_STREAMING_REQUIRED') from error
            next_density = None
            if raw.get('density_path') and Path(raw['density_path']).exists():
                next_density = _array(folder / '.local' / f'retry_density_{attempt + 1:06d}.npy',
                                      np.load(raw['density_path'], allow_pickle=False))
            _record(folder / f'failed_attempt_{attempt:06d}.json',
                    {'request': request, 'error_type': type(error).__name__, 'error': str(error),
                     'vendor_result_path': str(raw_path), 'qualified': False})
            retry = {'request_fingerprint': _hash(request), 'attempt': attempt + 1, 'density_ref': next_density}
            _record(folder / 'retry_state.json', retry)
            _prune_local(folder / 'vendor_attempts')
            _prune_local(folder / '.local', keep=() if next_density is None else (next_density['path'],))
            _progress(progress, stop_requested, stage='electronic_retry_committed',
                      point_path=str(folder), attempt=attempt, qualified=False)


def _generation(output, state):
    return output / 'generations' / f'g{state["generation"]:06d}_{DESIGN["profiles"][state["profile_index"]]}'


def _copy_restart_density(output, source, index):
    if source is None or not Path(source).exists():
        return None
    density = np.load(source, allow_pickle=False)
    return _array(output / '.local' / f'optimization_seed_{index:08d}_{_array_hash(density)}.npy', density)


def _recover_optimizer_chunk(output, state, chunk, error=None):
    progress_path = chunk / 'progress.json'
    vendor_progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    coordinates = vendor_progress.get('coordinates_bohr', state['coordinates_bohr'])
    if np.asarray(coordinates).shape != np.asarray(state['coordinates_bohr']).shape or not np.isfinite(coordinates).all():
        raise ScientificIntegrityError('Invalid optimizer continuation geometry')
    density_ref = (_copy_restart_density(output, vendor_progress.get('density_path'), state['chunk_index'] + 1)
                   or state['density_ref'])
    n = int(vendor_progress.get('evaluations', 0))
    orphaned = [str(path) for path in (chunk / 'steps').glob('step-*')
                if path.name[5:].isdigit() and int(path.name[5:]) >= n]
    checkpoint = {'status': 'CONTINUATION_FROM_LAST_COMMITTED_VENDOR_PROGRESS',
                  'coordinates_bohr': coordinates, 'density_ref': density_ref,
                  'optimizer_converged': False, 'minimum_stationarity_passed': False,
                  'orphan_point_directories_preserved': orphaned,
                  'error_type': None if error is None else type(error).__name__,
                  'error': None if error is None else str(error)}
    _record(chunk / 'WORKFLOW_CHUNK.json', checkpoint)
    return checkpoint


def _optimize(output, state, symbols, core, progress, stop_requested):
    while state['phase'] == 'OPTIMIZE':
        _stop(stop_requested)
        generation = _generation(output, state)
        chunk = generation / 'optimization' / f'chunk_{state["chunk_index"]:08d}'
        checkpoint = _read_record(chunk / 'WORKFLOW_CHUNK.json')
        active = state.get('active_chunk') == str(chunk)
        if checkpoint is None and active and chunk.exists() and any(chunk.iterdir()):
            # Never re-enter a partially persisted vendor scanner: a point may
            # precede its vendor-progress commit. Preserve it and use a new chunk.
            result_path = chunk / 'result.json'
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            if result.get('optimizer_converged') and result.get('minimum_stationarity_passed'):
                checkpoint = {'status': 'OPTIMIZER_SEGMENT_FINISHED',
                              'coordinates_bohr': result['coordinates_bohr'],
                              'density_ref': _copy_restart_density(output, result.get('density_path'), state['chunk_index'] + 1),
                              'optimizer_converged': True, 'minimum_stationarity_passed': True,
                              'vendor_result': result}
                _record(chunk / 'WORKFLOW_CHUNK.json', checkpoint)
            else:
                checkpoint = _recover_optimizer_chunk(output, state, chunk)
        if checkpoint is None:
            state['active_chunk'] = str(chunk)
            _save_state(output, state, progress, stop_requested, stage='geometry_chunk_started',
                        optimization_chunk=state['chunk_index'])
            try:
                with _bounded_optimizer_segment(core):
                    result = core.optimize_geometry(symbols, np.asarray(state['coordinates_bohr']), chunk,
                                                    dm0=_load_density(state['density_ref']))
                checkpoint = {'status': 'OPTIMIZER_SEGMENT_FINISHED',
                              'coordinates_bohr': result['coordinates_bohr'],
                              'density_ref': _copy_restart_density(output, result.get('density_path'), state['chunk_index'] + 1),
                              'optimizer_converged': result.get('optimizer_converged') is True,
                              'minimum_stationarity_passed': result.get('minimum_stationarity_passed') is True,
                              'vendor_result': result}
                _record(chunk / 'WORKFLOW_CHUNK.json', checkpoint)
            except (InterruptedError, UnsupportedStreamingPath):
                raise
            except Exception as error:
                if _resource_guard_evidence(chunk, error):
                    raise UnsupportedStreamingPath('DENSE_DF_DISABLED_J_ONLY_STREAMING_REQUIRED') from error
                if any(text in str(error).lower() for text in ('hash changed', 'fingerprint changed', 'integrity error')):
                    raise ScientificIntegrityError(str(error)) from error
                checkpoint = _recover_optimizer_chunk(output, state, chunk, error)
        state.update(coordinates_bohr=checkpoint['coordinates_bohr'], density_ref=checkpoint['density_ref'],
                     chunk_index=state['chunk_index'] + 1, active_chunk=None)
        if checkpoint['optimizer_converged'] and checkpoint['minimum_stationarity_passed']:
            state['phase'] = 'CERTIFY_CENTER'
        _save_state(output, state, progress, stop_requested, stage='geometry_chunk_committed',
                    optimizer_converged=checkpoint['optimizer_converged'])
        _prune_local(chunk)


def _center_quality(point, stability):
    audit = point['electronic_audit']
    return {'qualified': point['qualified'], 'native_converged': point['native_converged'],
            'fresh_orbital_gradient_norm': audit['fresh_orbital_gradient_norm'],
            'energy_recheck_difference_hartree': audit['fresh_energy_difference_hartree'],
            'electron_count_error': audit['electron_count_error'],
            'nuclear_gradient_max': point['nuclear_gradient_max'],
            'nuclear_gradient_rms': point['nuclear_gradient_rms'],
            'orbital_stability_passed': stability.get('tested_electronic_stability_passed') is True}


def _waiting(output, state, reason, details, computed=False):
    value = {'schema': DESIGN['schema'], 'case_id': state['case_id'], 'status': 'REFINEMENT_REQUIRED',
             'computed': computed, 'minimum_verified': False, 'ir_verified': False,
             'paper_ready_molecular_harmonic_data': False, 'continuation_eligible': False,
             'waiting_for_new_protocol': True, 'reason': reason, 'details': details,
             'generation': state['generation'], 'source_fingerprint': state['source_fingerprint'],
             'scope': DESIGN['scope']}
    state.update(phase='WAITING_FOR_NEW_PROTOCOL', waiting_result=value)
    _record(output / 'workflow_state.json', state)
    _atomic_json(output / 'result.json', value)
    return value


def _escalate(output, state, reason, details, progress, stop_requested, computed=False):
    if state['profile_index'] + 1 >= len(DESIGN['profiles']):
        return _waiting(output, state, reason, details, computed)
    state.update(generation=state['generation'] + 1, profile_index=state['profile_index'] + 1,
                 phase='OPTIMIZE', active_chunk=None, certification_round=0)
    state['history'].append({'reason': reason, 'details': details,
                             'new_generation': state['generation'], 'new_profile_index': state['profile_index']})
    _save_state(output, state, progress, stop_requested, stage='new_precision_generation_committed')
    return None


def _certify(output, state, symbols, reference, vendor, progress, stop_requested):
    core = vendor['core']
    generation = _generation(output, state)
    coordinates = np.asarray(state['coordinates_bohr'])
    round_id = state.get('certification_round', 0)
    point = _point(symbols, coordinates, generation / f'center_attempt_{round_id:06d}', core,
                   state['source_fingerprint'], _load_density(state['density_ref']), True, progress, stop_requested)
    identity = vendor['helpers'].identity_check(symbols, reference, coordinates)
    if not identity.get('passed'):
        return _waiting(output, state, 'SOURCE_RELATIVE_CHEMICAL_IDENTITY_NOT_PRESERVED', identity)
    stable_dir = generation / f'stability_{round_id:06d}'
    stability = core.check_electronic_stability(symbols, coordinates, stable_dir,
                                               dm0=np.load(point['density_path'], allow_pickle=False))
    _record(stable_dir / 'WORKFLOW_STABILITY.json', stability)
    _progress(progress, stop_requested, stage='electronic_stability_committed',
              case_id=state['case_id'], generation=state['generation'])
    if stability.get('geometry_reoptimization_required'):
        state.update(phase='OPTIMIZE', certification_round=round_id + 1,
                     density_ref=_copy_restart_density(output, stability.get('density_path'), state['chunk_index'] + 1))
        _save_state(output, state, progress, stop_requested, stage='repaired_electronic_state_requires_reoptimization')
        return None
    if not stability.get('supported_for_primary_hessian'):
        return _escalate(output, state, 'UNRESOLVED_ELECTRONIC_STABILITY', stability,
                         progress, stop_requested)
    if not (point['nuclear_gradient_rms'] <= 3e-6 and point['nuclear_gradient_max'] <= 1e-5):
        state.update(phase='OPTIMIZE', certification_round=round_id + 1,
                     density_ref=point['array_refs']['density'])
        _save_state(output, state, progress, stop_requested, stage='fresh_center_gradient_requires_reoptimization')
        return None
    center_fingerprint = _hash({'source_fingerprint': state['source_fingerprint'],
                                'symbols': symbols, 'coordinates_bohr': coordinates.tolist(),
                                'settings': core.settings(),
                                'density_sha256': point['array_refs']['density']['sha256']})
    artifact = {'case_id': state['case_id'], 'symbols': symbols,
                'coordinates_bohr': coordinates.tolist(), 'center_point': point,
                'electronic_stability': stability, 'identity': identity,
                'settings': core.settings(), 'source_fingerprint': center_fingerprint,
                'case_source_fingerprint': state['source_fingerprint'],
                'steps_bohr': DESIGN['steps_bohr'], 'generation': state['generation']}
    _record(generation / 'CENTER.json', artifact)
    state.update(phase='FINITE_DIFFERENCES', center_fingerprint=center_fingerprint,
                 center_artifact_path=str(generation / 'CENTER.json'))
    _save_state(output, state, progress, stop_requested, stage='frozen_center_committed',
                center_fingerprint=center_fingerprint)
    return None


def _finite_differences(output, state, vendor, progress, stop_requested):
    center = _read_record(state['center_artifact_path'])
    if center['source_fingerprint'] != state['center_fingerprint']:
        raise ScientificIntegrityError('Frozen center fingerprint changed')
    core = vendor['core']
    if center['settings'] != core.settings():
        raise ScientificIntegrityError('Frozen center method/settings changed')
    symbols, coordinates = center['symbols'], np.asarray(center['coordinates_bohr'])
    density = np.load(center['center_point']['density_path'], allow_pickle=False)
    _check_refs(center['center_point']['array_refs'])
    generation = _generation(output, state)
    fd = generation / 'finite-differences'
    hessians, derivatives, ledger, refs = [], [], [], {}
    completed = 0
    for hi, step in enumerate(DESIGN['steps_bohr']):
        points = {1: [], -1: []}
        for coordinate in range(coordinates.size):
            for sign, label in ((1, 'plus'), (-1, 'minus')):
                displaced = coordinates.copy()
                displaced.flat[coordinate] += sign * step
                folder = fd / f'h{hi}' / f'c{coordinate:03d}-{label}'
                point = _point(symbols, displaced, folder, core, state['center_fingerprint'],
                               density, False, progress, stop_requested)
                points[sign].append(point)
                for name, ref in point['array_refs'].items():
                    refs[str(Path(ref['path']).resolve())] = ref
                refs[str((folder / 'POINT_SEAL.json').resolve())] = {
                    'path': str((folder / 'POINT_SEAL.json').resolve()), 'sha256': _file_hash(folder / 'POINT_SEAL.json')}
                audit = point['electronic_audit']
                ledger.append({'qualified': point['qualified'], 'native_converged': point['native_converged'],
                               'fresh_orbital_gradient_norm': audit['fresh_orbital_gradient_norm'],
                               'energy_recheck_difference_hartree': audit['fresh_energy_difference_hartree'],
                               'electron_count_error': audit['electron_count_error']})
                completed += 1
                _progress(progress, stop_requested, stage='finite_difference_point_committed',
                          case_id=state['case_id'], generation=state['generation'],
                          center_fingerprint=state['center_fingerprint'], completed_points=completed,
                          expected_points=coordinates.size * 4, coordinate=coordinate, sign=sign, step_bohr=step)
        hessian = np.stack([(np.asarray(p['gradient']) - np.asarray(m['gradient'])).ravel() / (2 * step)
                            for p, m in zip(points[1], points[-1])], axis=1)
        derivative = np.stack([(np.asarray(p['dipole_au']) - np.asarray(m['dipole_au'])) / (2 * step)
                               for p, m in zip(points[1], points[-1])], axis=1)
        hessians.append(hessian)
        derivatives.append(derivative)
        refs[f'hessian_{hi}'] = _array(fd / f'hessian_h{hi}.npy', hessian)
        refs[f'dipole_derivative_{hi}'] = _array(fd / f'dipole_derivative_h{hi}.npy', derivative)
    evidence = {'center_fingerprint': state['center_fingerprint'], 'center': center,
                'expected_points': coordinates.size * 4, 'completed_points': completed,
                'all_displacements_qualified': all(point['qualified'] for point in ledger),
                'steps_bohr': DESIGN['steps_bohr'], 'artifact_refs': refs,
                'point_quality': ledger}
    _record(generation / 'HESSIAN_EVIDENCE.json', evidence)
    analysis = vendor['modes'].analyze_modes(symbols, coordinates, hessians, derivatives,
                generation / 'modes', center_quality=_center_quality(center['center_point'], center['electronic_stability']),
                point_quality=ledger, steps_bohr=DESIGN['steps_bohr'])
    gate = vendor['quality'].audit_publication(generation / 'modes', fd, center['center_point'],
                   electronic_stability_passed=center['electronic_stability'].get('tested_electronic_stability_passed') is True)
    minimum = {**gate, 'center_quality': _center_quality(center['center_point'], center['electronic_stability']),
               'all_displacements_qualified': evidence['all_displacements_qualified'],
               'expected_points': evidence['expected_points'], 'completed_points': completed,
               'electronic_stability': center['electronic_stability']}
    center.update(hessian_paths=[refs[f'hessian_{hi}']['path'] for hi in range(2)],
                  dipole_derivative_paths=[refs[f'dipole_derivative_{hi}']['path'] for hi in range(2)],
                  minimum_evidence=minimum, hessian_evidence_path=str(generation / 'HESSIAN_EVIDENCE.json'),
                  analysis_path=str(generation / 'modes'))
    _record(generation / 'DERIVATIVES_COMPLETE.json', {'center_artifact': center, 'analysis': analysis,
                                                     'publication_gate': gate})
    state['phase'] = 'HESSIAN_QUALIFICATION'
    _save_state(output, state, progress, stop_requested, stage='complete_hessian_audit_committed',
                completed_points=completed, expected_points=coordinates.size * 4,
                minimum_verified=gate.get('minimum_verified') is True)
    return center, analysis, gate


def _supported_negative(analysis):
    try:
        coarse, fine = analysis['steps'][0], analysis['steps'][-1]
        error = analysis['step_comparison']['projected_spectral_step_difference_hartree_per_amu_bohr2']
        return (coarse['imaginary_count'] > 0 and fine['imaginary_count'] > 0
                and fine['lowest_signed_eigenvalue_hartree_per_amu_bohr2'] < -error)
    except (KeyError, TypeError):
        return False


def _negative_seed(output, state, center, vendor, progress, stop_requested):
    generation = _generation(output, state)
    saved = _read_record(generation / 'NEGATIVE_MODE_FOLLOW.json')
    if saved is not None:
        return saved
    arrays = np.load(Path(center['analysis_path']) / 'analysis_arrays.npz', allow_pickle=False)
    direction = arrays['step_1_cartesian_unit_display_modes'][:, 0].reshape(np.asarray(center['coordinates_bohr']).shape)
    direction = direction / np.max(np.abs(direction))
    coordinates, symbols = np.asarray(center['coordinates_bohr']), center['symbols']
    density = np.load(center['center_point']['density_path'], allow_pickle=False)
    candidates = []
    for sign in (1, -1):
        for amplitude in DESIGN['negative_mode_amplitudes_bohr']:
            displaced = coordinates + sign * amplitude * direction
            identity = vendor['helpers'].identity_check(symbols, np.asarray(state['initial_geometry_bohr']), displaced)
            if not identity.get('passed'):
                continue
            point = _point(symbols, displaced, generation / 'negative-mode-follow' / f's{sign:+d}_a{amplitude:g}',
                           vendor['core'], state['center_fingerprint'], density, False, progress, stop_requested)
            candidates.append({'coordinates_bohr': displaced.tolist(), 'energy_hartree': point['energy_hartree'],
                               'sign': sign, 'amplitude_bohr': amplitude, 'identity': identity,
                               'point_fingerprint': point['input_fingerprint']})
    value = {'selected': min(candidates, key=lambda value: value['energy_hartree']) if candidates else None,
             'candidates': candidates, 'scope': 'Finite local negative-mode seed search, not a global conformer search'}
    _record(generation / 'NEGATIVE_MODE_FOLLOW.json', value)
    return value


def _ir(center, output, progress):
    module = importlib.import_module('spinosin_gap_ir_20261112')
    return module.run_ir_check(center, output, progress=progress)


def _soft_sensitivity(center, generation, vendor, progress, stop_requested):
    """Apply the upstream three-direction sensitivity formula to this case.

    The old driver cannot be called because its input lookup names the old
    molecule rather than these conformers. No acceptance formula is relaxed.
    """
    output = generation / 'soft-direction-controls'
    existing = _read_record(output / 'SENSITIVITY.json')
    if existing is not None:
        if existing['center_fingerprint'] != center['source_fingerprint']:
            raise ScientificIntegrityError('Sensitivity controls belong to a different center')
        _check_refs(existing['artifact_refs'])
        return existing
    arrays = np.load(Path(center['analysis_path']) / 'analysis_arrays.npz', allow_pickle=False)
    coordinates = np.asarray(center['coordinates_bohr'])
    directions = arrays['step_1_cartesian_unit_display_modes'][:, :min(3, coordinates.size - 6)]
    hessian = arrays['step_1_raw_hessian']
    derivative = arrays['step_1_dipole_derivative_au']
    profile = center['settings']['profile']
    checks = (['direct-grid5', 'df-grid7'] if profile == 'df-grid5' else
              ['direct-grid7'] if profile == 'df-grid7' else ['df-grid7'])
    core, original = vendor['core'], copy.deepcopy(vendor['core'].SCF_SETTINGS)
    rows, refs = [], {}
    try:
        for check_profile in checks:
            core.configure(check_profile, threads=original['threads'])
            core.SCF_SETTINGS['max_memory_mb'] = original['max_memory_mb']
            folder = output / check_profile
            checked_center = _point(center['symbols'], coordinates, folder / 'center', core,
                                    center['source_fingerprint'], None, True, progress, stop_requested)
            density = np.load(checked_center['density_path'], allow_pickle=False)
            for index in range(directions.shape[1]):
                direction = directions[:, index].reshape(coordinates.shape)
                plus = _point(center['symbols'], coordinates + 0.002 * direction,
                              folder / f'mode_{index}' / 'plus', core,
                              center['source_fingerprint'], density, False, progress, stop_requested)
                minus = _point(center['symbols'], coordinates - 0.002 * direction,
                               folder / f'mode_{index}' / 'minus', core,
                               center['source_fingerprint'], density, False, progress, stop_requested)
                force_difference = (np.asarray(plus['gradient']) - np.asarray(minus['gradient'])) / 0.004
                curvature = float(np.sum(direction * force_difference))
                base = float(direction.ravel() @ hessian @ direction.ravel())
                dipole_difference = (np.asarray(plus['dipole_au']) - np.asarray(minus['dipole_au'])) / 0.004
                rows.append({'profile': check_profile, 'mode_zero_based': index,
                    'all_points_qualified': all(point['qualified'] for point in (checked_center, plus, minus)),
                    'gradient_curvature': curvature, 'base_gradient_curvature': base,
                    'relative_curvature_change': abs(curvature - base) / max(abs(base), 1e-12),
                    'energy_curvature': (plus['energy_hartree'] + minus['energy_hartree'] - 2 * checked_center['energy_hartree']) / 0.002 ** 2,
                    'dipole_derivative_absolute_change': float(np.linalg.norm(dipole_difference - derivative @ direction.ravel())),
                    'reference_center_gradient_max': checked_center['nuclear_gradient_max']})
        for seal in output.rglob('POINT_SEAL.json'):
            record = _read_record(seal)
            refs[str(seal.resolve())] = {'path': str(seal.resolve()), 'sha256': _file_hash(seal)}
            refs.update({str(Path(ref['path']).resolve()): ref for ref in record['result']['array_refs'].values()})
        passed = bool(rows and all(row['all_points_qualified'] and row['gradient_curvature'] > 0
                                  and row['relative_curvature_change'] <= 0.1 for row in rows))
        value = {'center_fingerprint': center['source_fingerprint'], 'rows': rows,
                 'directional_sensitivity_passed': passed,
                 'maximum_allowed_relative_curvature_change': 0.1, 'artifact_refs': refs,
                 'scope': 'Three soft directions only; not a full independent direct-integral minimum certificate',
                 'exact_J_method_equivalence_claimed': False}
        _record(output / 'SENSITIVITY.json', value)
        return value
    finally:
        core.SCF_SETTINGS.clear()
        core.SCF_SETTINGS.update(original)


def _verify_final_artifacts(result):
    _check_refs(result['center_artifact']['center_point']['array_refs'])
    evidence = _read_record(result['center_artifact']['hessian_evidence_path'])
    _check_refs(evidence['artifact_refs'])
    _check_refs(result.get('directional_sensitivity', {}).get('artifact_refs', {}))
    for path, digest in result['publication_gate'].get('authenticated_artifact_sha256', {}).items():
        if not Path(path).is_file() or _file_hash(path) != digest:
            raise ScientificIntegrityError('Authenticated minimum evidence changed: ' + path)
    for key in ('array_refs', 'input_refs'):
        _check_refs(result['ir_result'].get(key, {}))
    ledger = result['ir_result'].get('point_ledger')
    if ledger is not None:
        _check_refs({'ir_point_ledger': ledger})


def _verify_derivatives_complete(complete, vendor):
    center, gate = complete['center_artifact'], complete['publication_gate']
    evidence = _read_record(center['hessian_evidence_path'])
    if evidence['center_fingerprint'] != center['source_fingerprint']:
        raise ScientificIntegrityError('Derivative evidence belongs to another frozen center')
    _check_refs(evidence['artifact_refs'])
    _check_refs(center['center_point']['array_refs'])
    for path, digest in gate.get('authenticated_artifact_sha256', {}).items():
        if not Path(path).is_file() or _file_hash(path) != digest:
            raise ScientificIntegrityError('Authenticated derivative evidence changed: ' + path)
    vendor['modes'].verify_analysis(center['analysis_path'], write_receipt=False)


def _cleanup_completed_optimizer_caches(output, state):
    keep = [] if state.get('density_ref') is None else [state['density_ref']['path']]
    for checkpoint_path in output.glob('generations/*/optimization/*/WORKFLOW_CHUNK.json'):
        checkpoint = _read_record(checkpoint_path)
        if (state.get('active_chunk') == str(checkpoint_path.parent)
                and checkpoint.get('density_ref') is not None):
            keep.append(checkpoint['density_ref']['path'])
        _prune_local(checkpoint_path.parent)
    _prune_local(output / '.local', keep=keep)


def _run_conformer_impl(case, output, threads=2, max_memory_mb=3000, progress=None, stop_requested=None):
    case, symbols, reference = _validate_case(case)
    frozen = prepare_fingerprint(case, threads, max_memory_mb)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = _read_record(output / 'SCIENCE_PROTOCOL.json')
    if protocol is not None and protocol != frozen:
        raise ScientificIntegrityError('Case input/code/config/runtime fingerprint changed')
    if protocol is None:
        if (output / 'workflow_state.json').exists() or (output / 'generations').exists():
            raise ScientificIntegrityError('Scientific data without its original protocol seal cannot be reused')
        _record(output / 'SCIENCE_PROTOCOL.json', frozen)
    receipt = _read_record(output / 'COMPLETE.json')
    if receipt is not None:
        result = json.loads((output / 'result.json').read_text())
        if (receipt.get('source_fingerprint') != frozen['source_fingerprint']
                or result.get('source_fingerprint') != frozen['source_fingerprint']
                or result.get('case_id') != case['case_id']):
            raise ScientificIntegrityError('Completed science belongs to another case/input fingerprint')
        if receipt['result_sha256'] != _file_hash(output / 'result.json'):
            raise ScientificIntegrityError('Final result seal mismatch')
        _verify_final_artifacts(result)
        return result
    state = _read_record(output / 'workflow_state.json')
    if state is None:
        state = {'case_id': case['case_id'], 'source_fingerprint': frozen['source_fingerprint'],
                 'initial_geometry_bohr': reference.tolist(), 'coordinates_bohr': reference.tolist(),
                 'density_ref': None, 'generation': 0, 'profile_index': 0, 'chunk_index': 0,
                 'phase': 'OPTIMIZE', 'active_chunk': None, 'certification_round': 0, 'history': []}
        _record(output / 'workflow_state.json', state)
    if state['source_fingerprint'] != frozen['source_fingerprint']:
        raise ScientificIntegrityError('Workflow state belongs to a different scientific case')
    if state['phase'] == 'WAITING_FOR_NEW_PROTOCOL':
        return state['waiting_result']
    vendor = _vendor()
    while True:
        _stop(stop_requested)
        _cleanup_completed_optimizer_caches(output, state)
        core = vendor['core']
        core.configure(DESIGN['profiles'][state['profile_index']], threads=threads)
        core.SCF_SETTINGS['max_memory_mb'] = max_memory_mb
        if state['phase'] == 'OPTIMIZE':
            _optimize(output, state, symbols, core, progress, stop_requested)
        if state['phase'] == 'CERTIFY_CENTER':
            waiting = _certify(output, state, symbols, reference, vendor, progress, stop_requested)
            if waiting is not None:
                return waiting
            continue
        if state['phase'] == 'FINITE_DIFFERENCES':
            center, analysis, gate = _finite_differences(output, state, vendor, progress, stop_requested)
        else:
            complete = _read_record(_generation(output, state) / 'DERIVATIVES_COMPLETE.json')
            if complete is None:
                raise ScientificIntegrityError('Missing completed derivative audit')
            center, analysis, gate = complete['center_artifact'], complete['analysis'], complete['publication_gate']
        _verify_derivatives_complete({'center_artifact': center, 'publication_gate': gate}, vendor)
        if not gate.get('minimum_verified'):
            if _supported_negative(analysis):
                follow = _negative_seed(output, state, center, vendor, progress, stop_requested)
                if follow['selected'] is None:
                    return _waiting(output, state, 'NO_IDENTITY_PRESERVING_NEGATIVE_MODE_SEED', follow, True)
                state.update(generation=state['generation'] + 1, phase='OPTIMIZE',
                             coordinates_bohr=follow['selected']['coordinates_bohr'],
                             density_ref=center['center_point']['array_refs']['density'],
                             active_chunk=None, certification_round=0)
                state['history'].append({'reason': 'SUPPORTED_NEGATIVE_MODE_FOLLOW', 'follow': follow})
                _save_state(output, state, progress, stop_requested, stage='negative_mode_new_geometry_committed')
                continue
            waiting = _escalate(output, state, 'INDEPENDENT_HESSIAN_AUDIT_REQUIRES_REFINEMENT', gate,
                                 progress, stop_requested, True)
            if waiting is not None:
                return waiting
            continue
        sensitivity = _soft_sensitivity(center, _generation(output, state), vendor, progress, stop_requested)
        if not sensitivity['directional_sensitivity_passed']:
            waiting = _escalate(output, state, 'SOFT_DIRECTION_MODEL_SENSITIVITY_REQUIRES_REFINEMENT',
                                 sensitivity, progress, stop_requested, True)
            if waiting is not None:
                return waiting
            continue
        def ir_progress(event):
            if progress is not None:
                progress({'case_id': state['case_id'], 'generation': state['generation'], **event})
            if event.get('checkpoint_committed'):
                _stop(stop_requested)
        ir_result = _ir(center, _generation(output, state) / 'ir-refinement', ir_progress)
        if ir_result.get('source_fingerprint') not in (None, center['source_fingerprint']):
            raise ScientificIntegrityError('IR result belongs to a different frozen center')
        ir_verified = (ir_result.get('status') == 'QUALIFIED' and ir_result.get('computed_complete') is True
                       and ir_result.get('minimum_verified') is True
                       and ir_result.get('ir_numerically_qualified') is True
                       and ir_result.get('paper_ready') is True
                       and ir_result.get('source_fingerprint') == center['source_fingerprint'])
        if not ir_verified:
            if ir_result.get('status') == 'IR_REFINEMENT_REQUIRED' and ir_result.get('computed_complete') is True:
                state['phase'] = 'IR_REFINEMENT'
                _save_state(output, state, progress, stop_requested, stage='ir_refinement_continuation_committed')
                continue
            return _waiting(output, state, 'INDEPENDENT_IR_REQUIRES_NEW_EVIDENCE_OR_PROTOCOL', ir_result, True)
        value = {'schema': DESIGN['schema'], 'case_id': case['case_id'], 'status': 'COMPLETE',
                 'computed': True, 'minimum_verified': True, 'ir_verified': True,
                 'paper_ready_molecular_harmonic_data': True, 'continuation_eligible': False,
                 'waiting_for_new_protocol': False, 'source_fingerprint': frozen['source_fingerprint'],
                 'center_fingerprint': center['source_fingerprint'], 'generation': state['generation'],
                 'center_artifact': center, 'publication_gate': gate, 'ir_result': ir_result,
                 'directional_sensitivity': sensitivity, 'directional_sensitivity_evaluated': True,
                 'history': state['history'], 'scope': DESIGN['scope']}
        _atomic_json(output / 'result.json', value)
        _record(output / 'COMPLETE.json', {'result_sha256': _file_hash(output / 'result.json'),
                                         'source_fingerprint': frozen['source_fingerprint'],
                                         'status': 'COMPLETE', 'minimum_verified': True, 'ir_verified': True})
        state['phase'] = 'COMPLETE'
        _save_state(output, state, progress, stop_requested, stage='case_complete_committed')
        return value


def run_conformer(case, output, threads=2, max_memory_mb=3000, progress=None, stop_requested=None):
    install_streaming_guard(_vendor()['core'])
    while True:
        try:
            return _run_conformer_impl(case, output, threads, max_memory_mb, progress, stop_requested)
        except UnsupportedStreamingPath as error:
            state = _read_record(Path(output).resolve() / 'workflow_state.json')
            if state is None:
                raise
            waiting = _escalate(Path(output).resolve(), state, 'UNSUPPORTED_DENSE_DF_PATH',
                                {'error': str(error)}, progress, stop_requested,
                                computed=state['phase'] in ('HESSIAN_QUALIFICATION', 'IR_REFINEMENT'))
            if waiting is not None:
                return waiting


run_case = run_conformer


def water_control_case():
    return {'case_id': 'WATER_CONTROL', 'engineering_control': True, 'formula': 'H2O',
            'symbols': ['O', 'H', 'H'], 'charge': 0, 'spin': 0, 'electron_count': 10,
            'initial_geometry_bohr': [[0.0, 0.0, 0.0], [1.43, 0.0, 1.11], [-1.43, 0.0, 1.11]],
            'provenance': 'Explicit independent engineering control; not either production Spinosin case'}


def run_water_control(output, threads=2, max_memory_mb=3000, progress=None, stop_requested=None):
    return run_conformer(water_control_case(), output, threads, max_memory_mb, progress, stop_requested)
