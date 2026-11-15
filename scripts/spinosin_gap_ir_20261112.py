"""Case-local, resumable Cartesian and independently displaced modal IR checks.

Only C003/C008 Spinosin centers and explicit water implementation controls are
accepted. No upstream scanner, network operation, publisher or budget scheduler
is called. Qualification failures request refinement; they never stop a queue.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import resource
import sys
import tempfile

import numpy as np

VENDOR = Path(__file__).resolve().parent / 'quantum_gap_vendor_20261112'
sys.path.insert(0, str(VENDOR))
import pyscf_publication_core_20261111 as electronic
import pyscf_ir_core_20261111 as ir_core
import pyscf_ir_quality_20261111 as quality
import pyscf_repair_modes_20261111 as modes

assert ir_core.electronic is electronic

PROTOCOL = {
    'schema': 'spinosin-gap-independent-ir-20261112-v1',
    'allowed_spinosin_cases': ['C003', 'C008'],
    'source_scope': 'this case exact center, atom order, model and original electronic density seed only',
    'third_cartesian_step': 'half the smaller upstream Cartesian step',
    'direct_modal_steps': 'same three distinct step sizes, in Bohr*sqrt(amu)',
    'spinosin_normal_modes': 219,
    'initial_additional_cartesian_points': 450,
    'initial_direct_modal_points': 1314,
    'mass_normalization': 'L.T M_amu L = I; no Euclidean renormalization',
    'dipole_convention': 'total nuclear plus electronic; fixed laboratory origin; atomic units',
    'intensity_convention': 'standard integrated km/mol, no frequency factor, no ln(10)',
    'completed_points': 'immutable SHA seals; only explicitly rebuildable displaced DM/CHK caches removed',
    'DF_resource_path': 'same science-wrapper guarded native pure-PBE J-only streaming; full DF.build/loop, prebuilt cderi and exchange routes rejected',
    'threads_per_lane': 2,
    'max_memory_mb_hint_per_lane': 3000,
    'refinement': 'tighten electronic precision, then continue smaller distinct local steps; no global round, runtime, CPU or retry stop threshold',
    'host_uptime_guarantee': False,
}
DESIGN = PROTOCOL


def _now():
    return datetime.now(timezone.utc).isoformat()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _array_hash(value):
    value = np.ascontiguousarray(value, dtype='<f8')
    return _hash({'shape': list(value.shape), 'sha256': hashlib.sha256(value.tobytes()).hexdigest()})


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False,
        sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read(path):
    return json.loads(Path(path).read_text())


def _ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': _sha(path), 'bytes': path.stat().st_size}


def _verify_ref(ref):
    path = Path(ref['path'])
    if not path.is_file() or _sha(path) != ref['sha256'] or ('bytes' in ref and path.stat().st_size != ref['bytes']):
        raise ValueError('IR scientific artifact checksum changed: ' + str(path))


def _array(path, value):
    stream = io.BytesIO()
    np.save(stream, np.asarray(value), allow_pickle=False)
    _write(path, stream.getvalue())
    return _ref(path)


def _progress(callback, **event):
    if callback:
        callback({'component': 'spinosin_gap_ir_20261112', **event})


def _versions():
    result = {}
    for package in ('numpy', 'scipy', 'pyscf'):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _resource_observation():
    current = None
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                current = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError):
        pass
    return {'pid': os.getpid(), 'process_rss_bytes': current,
            'process_high_water_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            'max_memory_mb_is_a_hint_not_a_hard_RSS_cap': True}


def ensure_streaming_jonly_guard(core=None):
    """Use the shared science factory guard, including IR precision refinements."""
    import spinosin_gap_science_20261112 as science
    core = electronic if core is None else core
    science.install_jonly_streaming_guard(core)
    if not getattr(core.make_scf, '_quantum_gap_streaming_jonly_guard', False):
        raise ValueError('The native J-only streaming resource guard was not installed')
    if core is electronic and ir_core.electronic is not core:
        raise ValueError('IR and source science are using different electronic core objects')
    return {'native_j_only_streaming_guard_installed': True,
            'guard_implementation_sha256': _sha(science.__file__),
            'full_DF_tensor_build_or_loop_allowed': False,
            'mf_cderi_property_inspected': False}


def _verify_point(directory, request_hash=None):
    directory = Path(directory)
    seal = _read(directory / 'POINT_COMPLETE.json')
    if request_hash is not None and seal['request_hash'] != request_hash:
        raise ValueError('IR point belongs to another geometry/model/mode/seed')
    _verify_ref(seal['result'])
    point = _read(directory / 'result.json')
    if point.get('qualified') is not True or point.get('status') != 'DONE':
        raise ValueError('Committed IR point is not electronically qualified')
    for key, ref in point.get('array_refs', {}).items():
        if key == 'density' and not Path(ref['path']).exists():
            continue
        _verify_ref(ref)
    return point, seal


def _prune_rebuildable_point_cache(directory):
    """Keep every scientific array/result; remove only this displaced point's DM/CHK."""
    directory = Path(directory).resolve()
    candidates = [directory / '.local/density.npy', *directory.rglob('*.chk')]
    for path in candidates:
        if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(directory):
            path.unlink()


def _point(symbols, coordinates, directory, seed, context, evaluator, progress, *, keep_density=False, **event):
    directory = Path(directory)
    request = {'context': context, 'symbols': symbols, 'coordinates_bohr': coordinates.tolist(),
               'settings': electronic.settings(), 'seed_sha256': _array_hash(seed)}
    identity = _hash(request)
    if (directory / 'POINT_COMPLETE.json').exists():
        point, seal = _verify_point(directory, identity)
        if not np.array_equal(point['coordinates_bohr'], coordinates) or point['symbols'] != symbols:
            raise ValueError('Committed IR displacement/atom order differs')
        _progress(progress, stage='IR_POINT_REUSED', checkpoint_committed=True, **event)
        return point, {'directory': str(directory.resolve()), 'seal': _ref(directory / 'POINT_COMPLETE.json'),
                       'input_fingerprint': point['input_fingerprint']}
    _progress(progress, stage='IR_POINT_COMPUTING', checkpoint_committed=False,
              resource_observation=_resource_observation(), **event)
    directory.mkdir(parents=True, exist_ok=True)
    _write(directory / 'request.json', dict(request, request_hash=identity))
    point = evaluator(symbols, coordinates, directory, seed, require_density=keep_density)
    if (point.get('qualified') is not True or point.get('status') != 'DONE'
            or point.get('symbols') != symbols or not np.array_equal(point.get('coordinates_bohr'), coordinates)
            or point.get('settings') != electronic.settings()):
        raise RuntimeError('IR point has not supplied matching qualified electronic evidence')
    dipole = np.asarray(point.get('dipole_au'), dtype=float)
    if dipole.shape != (3,) or not np.isfinite(dipole).all() or not point.get('input_fingerprint'):
        raise ValueError('IR point is missing a finite total dipole/fingerprint')
    for ref in point.get('array_refs', {}).values():
        _verify_ref(ref)
    # The original solver result is preserved byte-for-byte behind its own seal.
    if _read(directory / 'result.json') != point:
        raise ValueError('Returned IR point differs from its saved result')
    _write(directory / 'POINT_COMPLETE.json', {'request_hash': identity,
        'result': _ref(directory / 'result.json'), 'qualified': True, 'sealed_at_utc': _now(),
        'resource_observation': _resource_observation()})
    if not keep_density:
        _prune_rebuildable_point_cache(directory)
    _progress(progress, stage='IR_POINT_COMMITTED', checkpoint_committed=True,
              resource_observation=_resource_observation(), **event)
    return point, {'directory': str(directory.resolve()), 'seal': _ref(directory / 'POINT_COMPLETE.json'),
                   'input_fingerprint': point['input_fingerprint']}


def _cartesian(symbols, coords, directory, h, seed, context, evaluator, progress):
    columns, ledger = [], []
    for coordinate in range(coords.size):
        points = []
        for sign, label in ((1, 'plus'), (-1, 'minus')):
            displaced = coords.copy()
            displaced.flat[coordinate] += sign * h
            if np.array_equal(displaced, coords):
                raise ArithmeticError('Cartesian displacement is below floating-point coordinate resolution')
            point, receipt = _point(symbols, displaced, directory / f'c{coordinate:04d}-{label}', seed,
                context, evaluator, progress, coordinate=coordinate, sign=sign, step_bohr=float(h),
                scientific_phase='IR_CARTESIAN_DIPOLE')
            points.append(np.asarray(point['dipole_au']))
            ledger.append(dict(receipt, coordinate=coordinate, sign=sign, step_bohr=float(h)))
        columns.append((points[0] - points[1]) / (2 * h))
    return np.stack(columns, axis=1), ledger


def _directional(symbols, coords, vectors, directory, steps, seed, context, evaluator, progress):
    series, ledger = [], []
    for h in steps:
        columns = []
        for index in range(vectors.shape[1]):
            points = []
            for sign, label in ((1, 'plus'), (-1, 'minus')):
                displacement = sign * h * vectors[:, index].reshape(coords.shape)
                displaced = coords + displacement
                if np.array_equal(displaced, coords):
                    raise ArithmeticError('Normal displacement is below floating-point coordinate resolution')
                point, receipt = _point(symbols, displaced,
                    directory / float(h).hex() / f'm{index:04d}-{label}', seed, context, evaluator,
                    progress, mode_index_one_based=index + 1, sign=sign,
                    step_bohr=float(h), planned_points=2 * len(steps) * vectors.shape[1],
                    scientific_phase='IR_DIRECT_NORMAL_MODE_DIPOLE')
                points.append(np.asarray(point['dipole_au']))
                ledger.append(dict(receipt, mode_index=index, sign=sign,
                    normal_coordinate_step_bohr_sqrt_amu=float(h),
                    maximum_cartesian_displacement_bohr=float(np.max(np.abs(displacement)))))
            columns.append((points[0] - points[1]) / (2 * h))
        series.append(np.stack(columns, axis=1))
    return np.stack(series), ledger


def sampling_plan(index, original_steps):
    """A new configuration per completed failed IR attempt, with no round cap."""
    precision = min(index, 2)
    factor = np.ldexp(1.0, -max(0, index - 2))
    steps = [float(max(original_steps) * factor), float(min(original_steps) * factor),
             float(min(original_steps) * factor * 0.5)]
    if any(not np.isfinite(h) or h <= 0 for h in steps) or len(set(steps)) != 3:
        return None
    return {'sampling_index': index, 'precision_level': precision, 'cartesian_steps_bohr': steps,
            'normal_coordinate_steps_bohr_sqrt_amu': list(steps)}


def _configure(settings, precision):
    electronic.SCF_SETTINGS.update(settings)
    electronic.SCF_SETTINGS['threads'] = 2
    electronic.SCF_SETTINGS['max_memory_mb'] = min(int(settings.get('max_memory_mb', 3000)), 3000)
    if precision:
        electronic.SCF_SETTINGS.update(conv_tol_hartree=10.0 ** (-12 - precision),
            conv_tol_grad=10.0 ** (-8 - precision), polish_conv_tol_hartree=10.0 ** (-12 - precision),
            polish_conv_tol_grad=10.0 ** (-9 - precision),
            fresh_orbital_gradient_acceptance=10.0 ** (-8 - precision))


def _input(center, output, testing):
    case = str(center['case_id'])
    symbols = list(center['symbols'])
    coords = np.asarray(center['coordinates_bohr'], dtype=float)
    water = case.lower().startswith('water') and symbols == ['O', 'H', 'H']
    if not (case in ('C003', 'C008') or water):
        raise ValueError('Only the two new Spinosin cases or explicit water controls are permitted')
    if coords.shape != (len(symbols), 3) or not np.isfinite(coords).all():
        raise ValueError('A complete finite atom-order-preserving center is required')
    if not testing and not water and len(symbols) != 75:
        raise ValueError('Production Spinosin must retain all 75 atoms')
    point = center['center_point']
    if point.get('symbols') != symbols or not np.array_equal(point.get('coordinates_bohr'), coords):
        raise ValueError('Electronic center and new case geometry/atom order differ')
    if not isinstance(center['source_fingerprint'], str) or not center['source_fingerprint']:
        raise ValueError('A frozen same-case source fingerprint is required')
    settings = center['settings']
    if point.get('settings') != settings:
        raise ValueError('Upstream center model differs from the frozen IR input')
    if settings.get('charge') != 0 or settings.get('spin') != 0:
        raise ValueError('This scoped molecular calculation requires the neutral closed-shell model')
    source_refs = {}
    density_path = Path(point['density_path']).resolve()
    density_ref = point.get('array_refs', {}).get('density')
    if not isinstance(density_ref, dict) or Path(density_ref['path']).resolve() != density_path:
        raise ValueError('The center density is not bound to its upstream scientific reference')
    for ref in point.get('array_refs', {}).values():
        _verify_ref(ref)
    _verify_ref(density_ref)
    source_refs['original_center_density'] = _ref(density_path)
    seed = np.load(density_path, allow_pickle=False)
    if not np.isfinite(seed).all():
        raise ValueError('The frozen center density seed is not finite')
    codes = {Path(module.__file__).name: _sha(module.__file__)
             for module in (electronic, ir_core, quality, modes)}
    codes[Path(__file__).name] = _sha(__file__)
    guard_file = Path(__file__).with_name('spinosin_gap_science_20261112.py')
    codes[guard_file.name] = _sha(guard_file)
    identity = {'protocol': PROTOCOL, 'case_id': case, 'symbols': symbols,
        'coordinates_bohr': coords.tolist(), 'source_fingerprint': center['source_fingerprint'],
        'settings': settings, 'density_seed_sha256': _array_hash(seed),
        'implementation_sha256': codes, 'runtime': _versions()}
    pool_id = _hash(identity)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    return case, symbols, coords, seed, settings, source_refs, identity, pool_id, output


def _csv_bytes(frequencies, thz, cartesian, direct, result):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['mode_one_based', 'signed_frequency_cm1', 'signed_frequency_THz',
        'imaginary', 'cartesian_IR_km_mol', 'direct_IR_km_mol', 'minimum_verified',
        'IR_numerically_qualified', 'paper_ready'])
    for i, (cm, hz) in enumerate(zip(frequencies, thz)):
        writer.writerow([i + 1, float(cm), float(hz), bool(cm < 0), float(cartesian[i]), float(direct[i]),
            result['minimum_verified'], result['ir_numerically_qualified'], result['paper_ready']])
    return stream.getvalue().encode()


def verify_ir_result(directory):
    """Pure replay of a sealed IR attempt, without electronic evaluations."""
    directory = Path(directory)
    seal = _read(directory / 'COMPLETE.json')
    _verify_ref(seal['result'])
    result = _read(directory / 'result.json')
    if result['input_fingerprint'] != seal['input_fingerprint']:
        raise ValueError('IR result seal input identity differs')
    for ref in list(result['array_refs'].values()) + list(result['input_refs'].values()) + [result['csv'], result['point_ledger']]:
        _verify_ref(ref)
    for path, expected in result['minimum_evidence'].get('authenticated_artifact_sha256', {}).items():
        _verify_ref({'path': path, 'sha256': expected})
    frozen = _read(result['input_refs']['refinement_input']['path'])
    coordinates = np.asarray(frozen['coordinates_bohr'])
    vectors = np.load(result['array_refs']['mass_normalized_modes']['path'], allow_pickle=False)
    cartesian, directional = {}, {}
    ledger = _read(result['point_ledger']['path'])
    for entry in ledger:
        _verify_ref(entry['seal'])
        point, _ = _verify_point(entry['directory'])
        expected = coordinates.copy()
        if 'coordinate' in entry:
            expected.flat[entry['coordinate']] += entry['sign'] * entry['step_bohr']
            key = (entry['step_bohr'], entry['coordinate'], entry['sign'])
            target = cartesian
        elif 'mode_index' in entry:
            expected += (entry['sign'] * entry['normal_coordinate_step_bohr_sqrt_amu']
                         * vectors[:, entry['mode_index']].reshape(coordinates.shape))
            key = (entry['normal_coordinate_step_bohr_sqrt_amu'], entry['mode_index'], entry['sign'])
            target = directional
        else:
            key, target = 'center', {}
        if point['symbols'] != frozen['symbols'] or not np.array_equal(point['coordinates_bohr'], expected):
            raise ValueError('IR replay point geometry/atom identity differs')
        if point['settings'] != result['effective_settings']:
            raise ValueError('IR replay point physical/precision model differs')
        if key in target:
            raise ValueError('Duplicate IR displacement identity')
        target[key] = np.asarray(point['dipole_au'])
    plan = result['sampling_plan']
    recovered_direct = np.asarray([np.stack([
        (directional[(h, mode, 1)] - directional[(h, mode, -1)]) / (2 * h)
        for mode in range(vectors.shape[1])], axis=1)
        for h in plan['normal_coordinate_steps_bohr_sqrt_amu']])
    saved_direct = np.load(result['array_refs']['direct_modal_derivative_series']['path'], allow_pickle=False)
    if not np.array_equal(recovered_direct, saved_direct):
        raise ValueError('IR modal derivative replay differs from raw dipole points')
    recovered_cartesian = []
    for h in plan['cartesian_steps_bohr']:
        if (h, 0, 1) in cartesian:
            value = np.stack([(cartesian[(h, coordinate, 1)] - cartesian[(h, coordinate, -1)]) / (2 * h)
                              for coordinate in range(coordinates.size)], axis=1)
        else:
            index = frozen['steps_bohr'].index(h)
            value = np.load(result['input_refs'][f'upstream_dipole_derivative_{index}']['path'], allow_pickle=False)
        recovered_cartesian.append(value)
    if not np.array_equal(np.stack(recovered_cartesian), np.load(
            result['array_refs']['cartesian_derivative_series']['path'], allow_pickle=False)):
        raise ValueError('IR Cartesian derivative replay differs from raw dipole points')
    return result


def run_ir_check(center_artifact, output, *, progress=None, point_evaluator=None):
    """Compute or resume one qualified new-center IR refinement configuration.

    Hessian/dipole inputs are two complete same-center matrices with explicit
    steps. The caller supplies its independently authenticated publication gate.
    A subsequent call reuses a qualified attempt, resumes an incomplete one,
    or advances an unqualified completed attempt to a genuinely new sampling
    configuration. This function never executes a source scanner or network call.
    """
    c = center_artifact
    guard_evidence = ensure_streaming_jonly_guard()
    case, symbols, coords, seed, settings, refs, identity, pool_id, out = _input(c, output, point_evaluator is not None)
    _progress(progress, stage='IR_INPUT_VALIDATED', case_id=case, source_fingerprint=c['source_fingerprint'])
    steps = [float(value) for value in c.get('steps_bohr', (0.002, 0.001))]
    if len(steps) != 2 or len(set(steps)) != 2 or any(not np.isfinite(h) or h <= 0 for h in steps):
        raise ValueError('Two distinct positive upstream Hessian steps are required')
    hp, dp = c.get('hessian_paths'), c.get('dipole_derivative_paths')
    if not hp or not dp or len(hp) != 2 or len(dp) != 2:
        waiting = {'status': 'DERIVATIVES_ONLY_AWAITING_FULL_HESSIAN', 'case_id': case,
            'computed_complete': False, 'minimum_verified': False, 'ir_numerically_qualified': False,
            'paper_ready': False, 'source_fingerprint': c['source_fingerprint'],
            'reason': 'No complete pair of same-center Hessian/Cartesian dipole matrices supplied',
            'quality_thresholds_stop_execution': False}
        _write(out / 'waiting.json', waiting)
        return waiting
    hessians, derivatives = [], []
    for i, (hpath, dpath) in enumerate(zip(hp, dp)):
        refs[f'hessian_{i}'], refs[f'upstream_dipole_derivative_{i}'] = _ref(hpath), _ref(dpath)
        h, d = np.load(hpath, allow_pickle=False), np.load(dpath, allow_pickle=False)
        if h.shape != (coords.size, coords.size) or d.shape != (3, coords.size) or not np.isfinite(h).all() or not np.isfinite(d).all():
            raise ValueError('A complete finite same-center Hessian/dipole pair is required')
        hessians.append(h)
        derivatives.append(d)
    for key in ('hessian_evidence_path',):
        if c.get(key):
            refs[key] = _ref(c[key])
    minimum = c.get('minimum_evidence', {})
    if not isinstance(minimum, dict):
        raise ValueError('Independent minimum evidence must be a dictionary')
    minimum_verified = minimum.get('minimum_verified') is True
    if minimum_verified:
        gates = minimum.get('gates', {})
        required = ('source_artifacts_authenticated', 'distinct_steps',
                    'center_electronic_quality', 'center_stationary', 'orbital_stability_passed',
                    'complete_unique_displacement_ledger', 'all_points_electronically_qualified',
                    'consistent_electronic_model_settings', 'raw_derivatives_reassembled',
                    'all_vibrational_eigenvalues_positive', 'raw_antisymmetry_passed',
                    'all_step_comparisons_passed', 'force_sum_rules_passed')
        if not all(gates.get(key) is True for key in required):
            raise ValueError('The minimum gate lacks required independently authenticated evidence')
        if minimum.get('original_displacement_point_count') != coords.size * 4:
            raise ValueError('The minimum gate did not authenticate every original displacement')
        expected_center = hashlib.sha256(json.dumps(c['center_point'], sort_keys=True, allow_nan=False).encode()).hexdigest()
        if minimum.get('center_point_sha256') != expected_center:
            raise ValueError('The minimum gate belongs to another electronic center')
    if not c.get('analysis_path'):
        raise ValueError('Complete mode-analysis provenance is required before IR mode validation')
    result_path = Path(c['analysis_path']) / 'result.json'
    refs['source_mode_analysis'] = _ref(result_path)
    provenance = _read(result_path)['input_provenance']
    if (provenance['symbols'] != symbols or provenance['coordinates_hash'] != modes._array_hash(coords)
        or provenance['hessian_hashes'] != [modes._array_hash(h) for h in hessians]
        or provenance['dipole_derivative_hashes'] != [modes._array_hash(d) for d in derivatives]
        or provenance['steps_bohr'] != steps):
        raise ValueError('Full-mode provenance differs from this new case Hessian/center')
    modes.verify_analysis(c['analysis_path'], write_receipt=False)
    authenticated = minimum.get('authenticated_artifact_sha256', {})
    if minimum_verified:
        if authenticated.get(str(result_path.resolve())) != refs['source_mode_analysis']['sha256']:
            raise ValueError('Minimum gate is not bound to the supplied mode-analysis provenance')
        for path, expected in authenticated.items():
            _verify_ref({'path': path, 'sha256': expected})
    fine = int(np.argmin(steps))
    analyses = [modes.normal_modes(symbols, coords, h, d) for h, d in zip(hessians, derivatives)]
    analysis = analyses[fine]
    delta = analyses[0]['projected_hessian'] - analyses[1]['projected_hessian']
    spectral_difference = float(np.linalg.norm(delta, ord=2))
    relative_difference = float(np.linalg.norm(delta) / max(np.linalg.norm(analysis['projected_hessian']), np.finfo(float).tiny))
    numerical_minimum = {
        'both_actual_hessians_all_modes_strictly_positive': bool(all(np.all(a['eigenvalues'] > 0) for a in analyses)),
        'both_actual_raw_antisymmetry_passed': bool(all(a['diagnostics']['raw_relative_antisymmetry'] <= 1e-3 for a in analyses)),
        'projected_relative_step_difference': relative_difference,
        'projected_step_spectral_difference': spectral_difference,
        'actual_fine_minimum_eigenvalue': float(np.min(analysis['eigenvalues'])),
    }
    actual_minimum_supported = (numerical_minimum['both_actual_hessians_all_modes_strictly_positive']
        and numerical_minimum['both_actual_raw_antisymmetry_passed'] and relative_difference <= 5e-3
        and numerical_minimum['actual_fine_minimum_eigenvalue'] > spectral_difference)
    if minimum_verified and not actual_minimum_supported:
        raise ValueError('Actual Hessians contradict the supplied verified minimum evidence')
    vectors, masses = analysis['cartesian_mass_normalized_modes'], analysis['masses_amu']
    expected_modes = coords.size - 6
    if vectors.shape != (coords.size, expected_modes):
        raise ValueError('Expected complete nonlinear vibrational basis; no mode may be omitted')
    norm = quality.validate_mode_normalization(vectors.T, masses)
    if not norm['qualified']:
        raise ValueError('Mass-normalized modes failed independent validation')
    source_identity = dict(identity, hessians=[ref['sha256'] for key, ref in refs.items() if key.startswith('hessian_')],
        dipoles=[ref['sha256'] for key, ref in refs.items() if key.startswith('upstream_dipole')],
        minimum_evidence=minimum, modes_sha256=_array_hash(vectors), steps_bohr=steps)
    context_id = _hash(source_identity)
    context = out / 'analyses' / context_id
    context.mkdir(parents=True, exist_ok=True)
    completed = sorted(context.glob('sample*/COMPLETE.json'))
    if completed:
        latest = verify_ir_result(completed[-1].parent)
        if latest['ir_numerically_qualified']:
            return latest
    if not minimum_verified:
        waiting = {'status': 'AWAITING_QUALIFIED_MINIMUM', 'case_id': case, 'computed_complete': False,
            'minimum_verified': False, 'ir_numerically_qualified': False, 'paper_ready': False,
            'signed_frequencies_cm1': analysis['signed_frequencies_cm1'].tolist(),
            'minimum_evidence': minimum, 'independent_actual_hessian_check': numerical_minimum,
            'quality_thresholds_stop_execution': False,
            'source_fingerprint': c['source_fingerprint']}
        _write(context / 'waiting.json', waiting)
        return waiting
    pending = sorted(path for path in context.glob('sample*') if path.is_dir() and not (path / 'COMPLETE.json').exists())
    index = int(pending[-1].name[6:]) if pending else (int(completed[-1].parent.name[6:]) + 1 if completed else 0)
    plan = sampling_plan(index, steps)
    if plan is None:
        waiting = {'status': 'WAITING_FOR_NEW_SOURCE_OR_MODEL', 'case_id': case, 'computed_complete': False,
            'minimum_verified': minimum_verified, 'ir_numerically_qualified': False, 'paper_ready': False,
            'reason': 'No new representable positive local displacement configuration',
            'quality_thresholds_stop_execution': False}
        _write(context / 'waiting.json', waiting)
        return waiting
    attempt = context / f'sample{index:06d}'
    attempt.mkdir(parents=True, exist_ok=True)
    input_data = dict(source_identity, sampling_plan=plan)
    input_hash = _hash(input_data)
    input_path = attempt / 'input.json'
    if input_path.exists() and _read(input_path) != input_data:
        raise ValueError('Frozen IR refinement input changed')
    if not input_path.exists():
        _write(input_path, input_data)
    refs['refinement_input'] = _ref(input_path)
    before = dict(electronic.SCF_SETTINGS)
    before['electronic_recovery_order'] = list(electronic.SCF_SETTINGS['electronic_recovery_order'])
    evaluator = ir_core.qualified_dipole if point_evaluator is None else point_evaluator
    ledger = []
    try:
        _configure(settings, plan['precision_level'])
        pool = out / 'pools' / pool_id / f'precision{plan["precision_level"]}'
        point_context = {'geometry_model_pool_id': pool_id, 'precision_level': plan['precision_level'],
                         'effective_settings': electronic.settings()}
        center, receipt = _point(symbols, coords, pool / 'center', seed, point_context,
            evaluator, progress, keep_density=True, scientific_phase='IR_CENTER_RECHECK')
        ledger.append(receipt)
        original = c['center_point']
        energy_difference = abs(float(center['energy_hartree']) - float(original['energy_hartree']))
        dipole_difference = float(np.max(np.abs(np.asarray(center['dipole_au']) - np.asarray(original['dipole_au']))))
        if energy_difference > 1e-8 or dipole_difference > 1e-5:
            raise RuntimeError('Recomputed IR center does not reproduce the frozen electronic state')
        cartesian_series = []
        for h in plan['cartesian_steps_bohr']:
            matching = next((i for i, old_h in enumerate(steps) if h == old_h), None)
            if plan['precision_level'] == 0 and matching is not None:
                value = derivatives[matching]
            else:
                value, points = _cartesian(symbols, coords, pool / 'cartesian' / float(h).hex(),
                    h, seed, point_context, evaluator, progress)
                ledger.extend(points)
            cartesian_series.append(value)
        cartesian_series = np.stack(cartesian_series)
        mode_context = dict(point_context, mode_basis_sha256=_array_hash(vectors))
        direct, points = _directional(symbols, coords, vectors,
            pool / 'normal_modes' / mode_context['mode_basis_sha256'],
            plan['normal_coordinate_steps_bohr_sqrt_amu'], seed, mode_context, evaluator, progress)
        ledger.extend(points)
        common = quality.common_frame_derivative_comparison(
            cartesian_series.transpose(0, 2, 1), plan['cartesian_steps_bohr'], vectors.T, masses)
        projected = (cartesian_series[-1] @ vectors).T
        directional = quality.directional_comparison(projected, direct.transpose(0, 2, 1),
            plan['normal_coordinate_steps_bohr_sqrt_amu'])
        rigid = quality.rigid_body_residuals(cartesian_series[-1].T, coords,
            center['dipole_au'], float(settings['charge']))
        qualified = all(check.get('qualified') is True for check in (norm, common, directional, rigid))
        intensity = quality.prefactor_si() * np.sum(projected ** 2, axis=1)
        direct_intensity = quality.prefactor_si() * np.sum(direct[-1].T ** 2, axis=1)
        arrays = {name: _array(attempt / f'{name}.npy', value) for name, value in (
            ('mass_normalized_modes', vectors), ('masses_amu', masses),
            ('signed_frequencies_cm1', analysis['signed_frequencies_cm1']),
            ('signed_frequencies_THz', analysis['signed_frequencies_THz']),
            ('cartesian_derivative_series', cartesian_series), ('direct_modal_derivative_series', direct),
            ('projected_dipole_derivatives', projected), ('cartesian_IR_km_mol', intensity),
            ('direct_IR_km_mol', direct_intensity))}
        _write(attempt / 'point_ledger.json', ledger)
        result = {'schema': PROTOCOL['schema'], 'case_id': case, 'status': 'QUALIFIED' if qualified else 'IR_REFINEMENT_REQUIRED',
            'input_fingerprint': input_hash, 'source_fingerprint': c['source_fingerprint'],
            'geometry_model_pool_id': pool_id, 'sampling_plan': plan, 'computed_complete': True,
            'minimum_verified': minimum_verified, 'ir_numerically_qualified': qualified,
            'paper_ready': bool(minimum_verified and qualified), 'mode_count': vectors.shape[1],
            'imaginary_mode_count': int(np.sum(analysis['eigenvalues'] < 0)),
            'signed_frequencies_cm1': analysis['signed_frequencies_cm1'].tolist(),
            'signed_frequencies_THz': analysis['signed_frequencies_THz'].tolist(),
            'original_hessian_displacement_count': coords.size * 4,
            'additional_cartesian_points': sum('coordinate' in point for point in ledger),
            'direct_normal_mode_points': len(points), 'total_ir_point_evidence_count': len(ledger),
            'minimum_evidence': minimum, 'mass_normalization': norm, 'common_frame_check': common,
            'independent_actual_hessian_check': numerical_minimum,
            'directional_check': directional, 'rigid_body_check': rigid,
            'center_reproduction': {'energy_difference_hartree': energy_difference, 'dipole_max_difference_au': dipole_difference},
            'effective_settings': electronic.settings(), 'array_refs': arrays, 'input_refs': refs,
            'resource_observation': _resource_observation(),
            'resource_path_guard': guard_evidence,
            'point_ledger': _ref(attempt / 'point_ledger.json'),
            'quality_thresholds_stop_execution': False, 'next_sampling_index': None if qualified else index + 1,
            'chemical_peak_assignment_verified': False, 'global_conformer_minimum_verified': False,
            'scope': 'This exact isolated new conformer and numerical model only; no solid-state chemical assignment or classifier improvement is established.',
            'finished_at_utc': _now()}
        _write(attempt / 'all_modes.csv', _csv_bytes(analysis['signed_frequencies_cm1'],
            analysis['signed_frequencies_THz'], intensity, direct_intensity, result))
        result['csv'] = _ref(attempt / 'all_modes.csv')
        _write(attempt / 'result.json', result)
        _write(attempt / 'COMPLETE.json', {'input_fingerprint': input_hash,
            'result': _ref(attempt / 'result.json'), 'completed_at_utc': _now()})
        _write(out / 'latest.json', {'directory': str(attempt), 'result': _ref(attempt / 'result.json'),
            'status': result['status'], 'case_id': case, 'source_fingerprint': c['source_fingerprint']})
        _progress(progress, stage='IR_ATTEMPT_COMMITTED', checkpoint_committed=True,
                  case_id=case, sampling_index=index, paper_ready=result['paper_ready'])
        return verify_ir_result(attempt)
    finally:
        electronic.SCF_SETTINGS.clear()
        electronic.SCF_SETTINGS.update(before)


run_ir_validation = run_ir_check
