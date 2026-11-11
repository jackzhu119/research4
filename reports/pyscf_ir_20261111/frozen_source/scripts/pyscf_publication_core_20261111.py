#!/usr/bin/env python3
"""Profile-bound PySCF points and unconstrained geometry optimization.

This module never converts an unconverged finite electronic state to an accepted
scientific point.  Unqualified states are retained for diagnosis, independently
of the caller's decision to continue the experiment queue.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import time

import numpy as np


SCHEMA = 'research4-pyscf-publication-core-v1'
PROFILES = {
    'direct-grid5': {'density_fitting': False, 'grid_level': 5},
    'direct-grid7': {'density_fitting': False, 'grid_level': 7},
    'df-grid5': {'density_fitting': True, 'grid_level': 5},
    'df-grid7': {'density_fitting': True, 'grid_level': 7},
}
SCF_SETTINGS = {
    'xc': 'PBE', 'basis': 'def2-SVP', 'charge': 0, 'spin': 0,
    'dispersion': 'NONE_EXPLICIT', 'density_fitting': False,
    'direct_scf_tol': 1e-14, 'conv_tol_hartree': 1e-12,
    'conv_tol_grad': 1e-8, 'fresh_orbital_gradient_acceptance': 1e-8,
    'fresh_energy_difference_acceptance_hartree': 1e-10,
    'electron_count_acceptance': 1e-7,
    'grid_response': True, 'full_native_veff_every_cycle': True,
    'effective_potential_update': 'Full effective potential from the current density; no delta-DM update.',
    'small_rho_cutoff': 0,
    'electronic_recovery_order': ['NATIVE_DIIS', 'TIGHT_NATIVE_POLISH', 'NEWTON_SEED_NATIVE_POLISH'],
    'polish_conv_tol_hartree': 1e-12, 'polish_conv_tol_grad': 2e-9,
    'threads': 2, 'max_memory_mb': 5000,
    'profile': 'direct-grid5', 'grid_level': 5,
    'density_fitting_auxbasis': None,
    'auxbasis_response': False,
    'model_scope': 'Native direct electron repulsion integrals; no density fitting.',
    'internal_stability_max_repairs': 3,
}
OPTIMIZATION_SETTINGS = {
    'convergence_energy': 1e-8,
    'convergence_grms': 3e-6,
    'convergence_gmax': 1e-5,
    'convergence_drms': 1e-5,
    'convergence_dmax': 3e-5,
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_array(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, np.asarray(value), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {'path': str(path.resolve()), 'sha256': sha256(path)}


def configure(profile='direct-grid5', threads=2):
    """Select a frozen scientific model before running a worker process.

    Profiles must not be changed concurrently inside one Python process. The
    complete effective settings, including thread count and any explicit grid
    override, enter every cache fingerprint. DF profiles approximate Coulomb/JK
    integrals and require their own independently validated results.
    """
    if profile not in PROFILES:
        raise ValueError('Unknown publication profile: ' + str(profile))
    if isinstance(threads, bool) or threads not in (1, 2):
        raise ValueError('The validated thread choices are 1 or 2')
    chosen = PROFILES[profile]
    fitting = chosen['density_fitting']
    SCF_SETTINGS.update(
        chosen, profile=profile, threads=int(threads),
        density_fitting_auxbasis='def2-universal-jkfit' if fitting else None,
        auxbasis_response=bool(fitting),
        model_scope=(
            'Density-fitted Coulomb/JK integrals with def2-universal-jkfit; '
            'PBE XC quadrature remains numerical. This is a model approximation '
            'relative to the direct-integral reference, not a direct-SCF result.'
            if fitting else 'Native direct electron repulsion integrals; no density fitting.'
        ),
    )
    return settings()


def _grid_level(value=None):
    value = SCF_SETTINGS['grid_level'] if value is None else value
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value <= 9:
        raise ValueError('Grid level must be an integer from 0 through 9')
    return int(value)


def settings(grid_level=None):
    value = dict(SCF_SETTINGS)
    value['electronic_recovery_order'] = list(SCF_SETTINGS['electronic_recovery_order'])
    value.update(grid_level=_grid_level(grid_level), grid_pruning='PySCF_default_NWChem',
                 grid_construction='native_atom_centered_rebuilt_for_each_geometry')
    return value


def _validate(symbols, coordinates):
    symbols = list(symbols)
    coordinates = np.asarray(coordinates, dtype=float)
    if not symbols or coordinates.shape != (len(symbols), 3) or not np.isfinite(coordinates).all():
        raise ValueError('Complete finite Cartesian coordinates required')
    if not all(isinstance(symbol, str) and symbol for symbol in symbols):
        raise ValueError('Atomic symbols required')
    return symbols, coordinates


def _array_digest(array):
    if array is None:
        return None
    if np.iscomplexobj(array):
        raise ValueError('This real closed-shell route does not accept complex initial densities')
    array = np.asarray(array, dtype='<f8')
    if not np.isfinite(array).all():
        raise ValueError('Initial density must be finite')
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def _fingerprint(symbols, coordinates, grid_level, dm0, operation='point'):
    import pyscf
    payload = {
        'schema': SCHEMA, 'operation': operation,
        'source_sha256': sha256(__file__), 'pyscf_version': pyscf.__version__,
        'settings': settings(grid_level), 'symbols': symbols,
        'coordinates_bohr': coordinates.tolist(), 'initial_density_sha256': _array_digest(dm0),
    }
    if operation == 'optimize':
        import geometric
        payload['geometric_version'] = geometric.__version__
        payload['optimization_settings'] = OPTIMIZATION_SETTINGS
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return digest, payload


def new_molecule(symbols, coordinates, output=None):
    from pyscf import gto, lib
    symbols, coordinates = _validate(symbols, coordinates)
    lib.num_threads(SCF_SETTINGS['threads'])
    mol = gto.M(atom=[(s, tuple(map(float, row))) for s, row in zip(symbols, coordinates)],
                unit='Bohr', basis=SCF_SETTINGS['basis'], charge=0, spin=0,
                symmetry=False, verbose=0, max_memory=SCF_SETTINGS['max_memory_mb'])
    if output is not None:
        mol.stdout = output
        mol.verbose = 4
    return mol


def make_scf(mol, out, grid_level=None):
    from pyscf import dft
    from pyscf.dft import rks
    class FullNativeRKS(rks.RKS):
        def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
            return rks.get_veff(self, mol, dm, dm_last=0, vhf_last=0, hermi=hermi)
    mf = FullNativeRKS(mol)
    if SCF_SETTINGS['density_fitting']:
        mf = mf.density_fit(auxbasis=SCF_SETTINGS['density_fitting_auxbasis'])
    mf.xc = SCF_SETTINGS['xc']
    mf.grids.level = _grid_level(grid_level)
    # Preserve geometry/species-defined quadrature independently of the initial DM.
    mf.small_rho_cutoff = 0
    mf.max_memory = SCF_SETTINGS['max_memory_mb']
    mf.direct_scf = True
    mf.direct_scf_tol = SCF_SETTINGS['direct_scf_tol']
    mf.conv_tol = SCF_SETTINGS['conv_tol_hartree']
    mf.conv_tol_grad = SCF_SETTINGS['conv_tol_grad']
    mf.max_cycle = 120
    local = Path(out) / '.local'
    local.mkdir(parents=True, exist_ok=True)
    mf.chkfile = str(local / 'rolling.chk')
    return mf


def _audit(mf):
    dm = np.asarray(mf.make_rdm1())
    overlap = mf.get_ovlp()
    hcore = mf.get_hcore()
    veff = mf.get_veff(mf.mol, dm)
    energy = float(mf.energy_tot(dm, hcore, veff))
    fock = mf.get_fock(hcore, overlap, veff, dm, cycle=-1, diis=None, level_shift_factor=0)
    residual = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ, fock)))
    electron_count = float(np.einsum('ij,ji->', dm, overlap))
    difference = float(energy - mf.e_tot)
    if not np.isfinite(dm).all() or not all(np.isfinite(x) for x in (energy, residual, electron_count, difference)):
        raise ValueError('Nonfinite density or electronic audit')
    warnings = []
    if not mf.converged:
        warnings.append('NATIVE_SCF_NOT_CONVERGED')
    if residual > SCF_SETTINGS['fresh_orbital_gradient_acceptance']:
        warnings.append('FRESH_ORBITAL_GRADIENT_TARGET_NOT_MET')
    if abs(difference) > SCF_SETTINGS['fresh_energy_difference_acceptance_hartree']:
        warnings.append('FRESH_ENERGY_RECHECK_TARGET_NOT_MET')
    if abs(electron_count - mf.mol.nelectron) > SCF_SETTINGS['electron_count_acceptance']:
        warnings.append('ELECTRON_COUNT_TARGET_NOT_MET')
    return dm, {
        'native_converged': bool(mf.converged), 'fresh_energy_hartree': energy,
        'native_energy_hartree': float(mf.e_tot),
        'fresh_energy_difference_hartree': difference,
        'fresh_orbital_gradient_norm': residual,
        'electron_count': electron_count,
        'expected_electron_count': int(mf.mol.nelectron),
        'electron_count_error': electron_count - mf.mol.nelectron,
        'qualified': not warnings, 'quality_warnings': warnings,
    }


def _run_electronic(symbols, coordinates, out, grid_level, dm0, log):
    mol = new_molecule(symbols, coordinates, log)
    calls = []
    mf = make_scf(mol, out, grid_level)
    guess = None if dm0 is None else np.asarray(dm0).copy()
    audit = None
    for stage in SCF_SETTINGS['electronic_recovery_order']:
        started = time.monotonic()
        try:
            if stage == 'NEWTON_SEED_NATIVE_POLISH':
                seed = mf.newton()
                seed.max_cycle = 60
                seed.max_cycle_inner = 12
                # Newton supplies a basin seed; fresh native polish owns strict acceptance.
                seed.conv_tol_grad = 1e-7
                seed.kernel(dm0=guess)
                calls.append({'solver': 'NEWTON_SEED', 'native_converged': bool(seed.converged),
                              'actual_conv_tol_hartree': seed.conv_tol,
                              'actual_conv_tol_grad': seed.conv_tol_grad})
                guess = np.asarray(seed.make_rdm1()).copy()
                mf = make_scf(mol, out, grid_level)
            if stage in ('TIGHT_NATIVE_POLISH', 'NEWTON_SEED_NATIVE_POLISH'):
                mf = make_scf(mol, out, grid_level)
                mf.conv_tol = SCF_SETTINGS['polish_conv_tol_hartree']
                mf.conv_tol_grad = SCF_SETTINGS['polish_conv_tol_grad']
                mf.diis_space = 12
            mf.kernel(dm0=guess)
            guess, audit = _audit(mf)
            calls.append({'solver': stage, 'native_converged': bool(mf.converged),
                          'actual_conv_tol_hartree': mf.conv_tol,
                          'actual_conv_tol_grad': mf.conv_tol_grad,
                          'audit': audit, 'wall_seconds': time.monotonic() - started})
            if audit['qualified']:
                return mf, guess, audit, calls
        except Exception as error:
            calls.append({'solver': stage, 'error_type': type(error).__name__,
                          'error': str(error), 'wall_seconds': time.monotonic() - started})
            if getattr(mf, 'mo_coeff', None) is not None:
                guess = np.asarray(mf.make_rdm1()).copy()
    if getattr(mf, 'mo_coeff', None) is None:
        atomic_json(Path(out) / 'electronic-attempts.json', calls)
        raise RuntimeError('No finite electronic state; see electronic-attempts.json')
    # A failed later strategy must never leave an audit from an earlier object.
    guess, audit = _audit(mf)
    return mf, guess, audit, calls


def _read_completed(out, fingerprint, result_name='result.json'):
    path = Path(out) / result_name
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get('input_fingerprint') != fingerprint:
        raise ValueError('Scientific input fingerprint changed: ' + str(out))
    if value.get('status') != 'DONE':
        return None
    for ref in value.get('array_refs', {}).values():
        if sha256(ref['path']) != ref['sha256']:
            raise ValueError('Saved point array hash changed: ' + ref['path'])
    return value


def calculate_point(symbols, coords, out, grid_level=None, dm0=None):
    """Return a persisted native energy, grid-response gradient and dipole.

    ``qualified`` requires native SCF convergence and independent fresh Fock,
    energy and electron-count audits. Unqualified finite results are retained.
    All returned quantities are JSON-serializable; coordinates and gradient are
    in Bohr and Hartree/Bohr, dipole in atomic units.
    """
    symbols, coords = _validate(symbols, coords)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    fingerprint, payload = _fingerprint(symbols, coords, grid_level, dm0)
    old = _read_completed(out, fingerprint)
    if old is not None:
        return old
    atomic_json(out / 'input.json', {**payload, 'input_fingerprint': fingerprint})
    attempts = sorted(out.glob('attempt-*'))
    attempt = out / f'attempt-{len(attempts) + 1:03d}'
    attempt.mkdir()
    start, cpu_start = time.monotonic(), time.process_time()
    result = {'schema': SCHEMA, 'status': 'RUNNING', 'input_fingerprint': fingerprint,
              'settings': settings(grid_level), 'symbols': symbols,
              'coordinates_bohr': coords.tolist(), 'started_at_utc': utc(),
              'attempt_path': str(attempt), 'array_refs': {}}
    atomic_json(out / 'result.json', result)
    try:
        with (attempt / 'pyscf.log').open('w', buffering=1) as log:
            mf, dm, audit, calls = _run_electronic(symbols, coords, attempt, grid_level, dm0, log)
            atomic_json(attempt / 'electronic-attempts.json', calls)
            grad_driver = mf.nuc_grad_method()
            grad_driver.grid_response = True
            if SCF_SETTINGS['density_fitting']:
                if not hasattr(grad_driver, 'auxbasis_response'):
                    raise RuntimeError('DF gradient auxiliary-basis response is unsupported by this PySCF build')
                grad_driver.auxbasis_response = True
            gradient = np.asarray(grad_driver.kernel())
            dipole = np.asarray(mf.dip_moment(unit='AU', verbose=0))
            if gradient.shape != coords.shape or not np.isfinite(gradient).all() or not np.isfinite(dipole).all():
                raise ValueError('Nonfinite or incomplete gradient/dipole')
            refs = {
                'coordinates_bohr': atomic_array(out / 'coordinates_bohr.npy', coords),
                'gradient': atomic_array(out / 'gradient.npy', gradient),
                'dipole_au': atomic_array(out / 'dipole_au.npy', dipole),
                'density': atomic_array(out / '.local/density.npy', dm),
            }
            result.update(status='DONE' if audit['qualified'] else 'UNQUALIFIED',
                          qualified=audit['qualified'], native_converged=audit['native_converged'],
                          energy_hartree=audit['fresh_energy_hartree'],
                          gradient=gradient.tolist(), dipole_au=dipole.tolist(),
                          gradient_path=refs['gradient']['path'], density_path=refs['density']['path'],
                          nuclear_gradient_rms=float(np.sqrt(np.mean(gradient ** 2))),
                          nuclear_gradient_max=float(np.max(np.abs(gradient))),
                          gradient_response={'grid_response': True,
                                             'auxbasis_response': bool(SCF_SETTINGS['density_fitting'])},
                          electronic_audit=audit, quality_warnings=audit['quality_warnings'],
                          calls=calls, array_refs=refs)
    except Exception as error:
        result.update(status='ERROR', qualified=False, error_type=type(error).__name__, error=str(error))
        atomic_json(attempt / 'result.json', result)
        atomic_json(out / 'result.json', result)
        raise
    finally:
        result.update(wall_seconds=time.monotonic() - start, cpu_seconds=time.process_time() - cpu_start,
                      finished_at_utc=utc())
        atomic_json(attempt / 'result.json', result)
        atomic_json(out / 'result.json', result)
    return result


def optimize_geometry(symbols, coords, out, grid_level=None, dm0=None):
    """Run one resumable geomeTRIC segment, preserving every electronic point.

    A new call resumes the last finite geometry if the previous segment stopped.
    Individual calls have 100 optimizer steps; this is a checkpoint interval,
    not a queue CPU/time/retry cap. Optimizer convergence and the independently
    checked final nuclear stationary criteria are reported separately.
    """
    from pyscf.geomopt import addons, geometric_solver
    symbols, coords = _validate(symbols, coords)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    fingerprint, payload = _fingerprint(symbols, coords, grid_level, dm0, 'optimize')
    old = _read_completed(out, fingerprint)
    if old is not None and old.get('optimizer_converged') and old.get('minimum_stationarity_passed'):
        return old
    atomic_json(out / 'input.json', {**payload, 'input_fingerprint': fingerprint})
    progress_path = out / 'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    if progress and progress.get('input_fingerprint') != fingerprint:
        raise ValueError('Optimization scientific inputs changed')
    current = np.asarray(progress.get('coordinates_bohr', coords), dtype=float)
    if progress.get('density_path') and Path(progress['density_path']).exists():
        density = np.load(progress['density_path'], allow_pickle=False)
    else:
        density = dm0
    step_index = int(progress.get('evaluations', 0))
    segment = int(progress.get('segments_started', 0)) + 1
    progress.update(status='RUNNING', input_fingerprint=fingerprint, segments_started=segment,
                    optimizer_settings=OPTIMIZATION_SETTINGS, started_at_utc=progress.get('started_at_utc', utc()))
    atomic_json(progress_path, progress)
    last_result = None
    def scan_function(mol):
        nonlocal density, step_index, last_result
        coords_now = np.asarray(mol.atom_coords())
        point = calculate_point(symbols, coords_now, out / 'steps' / f'step-{step_index:05d}',
                                grid_level=grid_level, dm0=density)
        step_index += 1
        last_result = point
        density = np.load(point['density_path'], allow_pickle=False)
        progress.update(evaluations=step_index, coordinates_bohr=coords_now.tolist(),
                        density_path=point['density_path'], energy_hartree=point['energy_hartree'],
                        nuclear_gradient_rms=point['nuclear_gradient_rms'],
                        nuclear_gradient_max=point['nuclear_gradient_max'],
                        last_point_path=str(out / 'steps' / f'step-{step_index - 1:05d}' / 'result.json'),
                        heartbeat_at_utc=utc())
        atomic_json(progress_path, progress)
        if not point['qualified']:
            raise RuntimeError('Electronic point unqualified; raw result retained at ' + point['attempt_path'])
        return point['energy_hartree'], np.asarray(point['gradient'])
    try:
        with (out / f'geometric-segment-{segment:03d}.log').open('w', buffering=1) as log:
            mol = new_molecule(symbols, current, log)
            method = addons.as_pyscf_method(mol, scan_function)
            converged, optimized_mol = geometric_solver.kernel(method, assert_convergence=True,
                                                              maxsteps=100, **OPTIMIZATION_SETTINGS)
        final_coords = np.asarray(optimized_mol.atom_coords())
        final = calculate_point(symbols, final_coords, out / f'final-center-segment-{segment:03d}',
                                grid_level=grid_level, dm0=density)
        stationary = bool(final['qualified'] and final['nuclear_gradient_rms'] <= 3e-6
                          and final['nuclear_gradient_max'] <= 1e-5)
        refs = {'coordinates_bohr': atomic_array(out / 'coordinates_bohr.npy', final_coords)}
        progress.update(status='DONE' if converged and stationary else 'SEGMENT_COMPLETE_CONTINUATION_NEEDED',
                        coordinates_bohr=final_coords.tolist(), density_path=final['density_path'],
                        optimizer_converged=bool(converged), minimum_stationarity_passed=stationary,
                        finished_at_utc=utc())
        result = {
            'schema': SCHEMA, 'status': progress['status'], 'input_fingerprint': fingerprint,
            'symbols': symbols, 'coordinates_bohr': final_coords.tolist(),
            'density_path': final['density_path'], 'grid_level': _grid_level(grid_level),
            'settings': settings(grid_level),
            'optimizer_settings': OPTIMIZATION_SETTINGS, 'optimizer_converged': bool(converged),
            'minimum_stationarity_passed': stationary,
            'electronic_qualified': bool(final['qualified']),
            'final_point': final, 'array_refs': refs,
            'evaluations': step_index, 'segments_started': segment, 'finished_at_utc': utc(),
            'scope': 'Unconstrained molecular geometry; nuclear stationarity alone does not certify a minimum or electronic stability.',
        }
        atomic_json(progress_path, progress)
        atomic_json(out / 'result.json', result)
        return result
    except Exception as error:
        progress.update(status='ERROR_CONTINUATION_NEEDED', error_type=type(error).__name__,
                        error=str(error), heartbeat_at_utc=utc())
        atomic_json(progress_path, progress)
        raise


def _diagnostic_mf(symbols, coords, out, grid_level, dm0):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = (out / 'pyscf.log').open('w', buffering=1)
    try:
        mf, dm, audit, calls = _run_electronic(symbols, coords, out, grid_level, dm0, log)
        atomic_json(out / 'electronic-attempts.json', {'audit': audit, 'calls': calls})
        if not audit['qualified']:
            raise RuntimeError('Diagnostic electronic state is unqualified')
        return mf, log
    except Exception:
        log.close()
        raise


def native_analytic_hessian(symbols, coords, out, grid_level=None, dm0=None):
    """Native analytic Hessian for derivative comparison, not the primary result."""
    symbols, coords = _validate(symbols, coords)
    out = Path(out).resolve()
    fingerprint, payload = _fingerprint(symbols, coords, grid_level, dm0, 'analytic_hessian')
    out.mkdir(parents=True, exist_ok=True)
    old = _read_completed(out, fingerprint)
    if old is not None:
        return old
    mf, log = _diagnostic_mf(symbols, coords, out, grid_level, dm0)
    try:
        driver = mf.Hessian()
        supported = hasattr(driver, 'grid_response')
        if supported:
            driver.grid_response = True
        auxiliary_supported = hasattr(driver, 'auxbasis_response')
        if SCF_SETTINGS['density_fitting'] and auxiliary_supported:
            driver.auxbasis_response = 2
        hessian = np.asarray(driver.kernel())
        if hessian.shape != (len(symbols), len(symbols), 3, 3) or not np.isfinite(hessian).all():
            raise ValueError('Incomplete or nonfinite analytic Hessian')
        ref = atomic_array(out / 'hessian_au.npy', hessian)
        value = {'schema': SCHEMA, 'status': 'DONE', 'input_fingerprint': fingerprint,
                 'settings': settings(grid_level), 'grid_response_option_supported': supported,
                 'grid_response_requested': supported, 'array_refs': {'hessian_au': ref},
                 'auxbasis_response_option_supported': auxiliary_supported,
                 'auxbasis_response_requested': 2 if SCF_SETTINGS['density_fitting'] and auxiliary_supported else None,
                 'hessian_path': ref['path'], 'finished_at_utc': utc(),
                 'scope': 'Native analytic diagnostic; agreement with full gradient finite differences must be checked.'}
        atomic_json(out / 'input.json', {**payload, 'input_fingerprint': fingerprint})
        atomic_json(out / 'result.json', value)
        return value
    finally:
        log.close()


def _native_stability_status(status):
    """Unknown solver statuses must never become scientific passes."""
    return bool(status) if isinstance(status, (bool, np.bool_)) else None


def check_electronic_stability(symbols, coords, out, grid_level=None, dm0=None):
    """Repair real closed-shell internal instabilities and test the final state.

    A changed orbital state invalidates the old geometry's stationary-point
    certificate. The caller must recalculate its gradient and reoptimize when
    ``geometry_reoptimization_required`` is true before forming a Hessian.
    External instabilities are retained as unsupported electronic-state cases;
    this routine never silently changes RKS into UKS or complex-orbital RKS.
    """
    symbols, coords = _validate(symbols, coords)
    out = Path(out).resolve()
    fingerprint, payload = _fingerprint(symbols, coords, grid_level, dm0, 'electronic_stability')
    out.mkdir(parents=True, exist_ok=True)
    old = _read_completed(out, fingerprint)
    if old is not None:
        return old
    value = {'schema': SCHEMA, 'status': 'RUNNING', 'input_fingerprint': fingerprint,
             'settings': settings(grid_level), 'internal_stable': None, 'external_stable': None,
             'real_to_complex_stable': None, 'rks_to_uks_stable': None,
             'array_refs': {}, 'started_at_utc': utc(),
             'internal_repair_attempts': [], 'internal_repair_performed': False,
             'geometry_reoptimization_required': False,
             'density_path': None, 'supported_for_primary_hessian': False}
    atomic_json(out / 'input.json', {**payload, 'input_fingerprint': fingerprint})
    mf, log = _diagnostic_mf(symbols, coords, out, grid_level, dm0)
    try:
        initial_dm, initial_audit = _audit(mf)
        final_audit = initial_audit
        value['initial_energy_hartree'] = initial_audit['fresh_energy_hartree']
        repairs = SCF_SETTINGS['internal_stability_max_repairs']
        for round_id in range(repairs + 1):
            attempt = {'round': round_id, 'checked_at_utc': utc()}
            try:
                internal = mf.stability(internal=True, external=False, return_status=True)
                state = _native_stability_status(internal[2])
                value['internal_stable'] = state
                attempt['internal_stable'] = state
                value['internal_repair_attempts'].append(attempt)
                if state is True or state is None or round_id == repairs:
                    break
                rotated = np.asarray(internal[0])
                if (rotated.shape != np.asarray(mf.mo_coeff).shape
                        or not np.isfinite(rotated).all() or np.iscomplexobj(rotated)):
                    raise ValueError('Internal stability did not return finite real closed-shell orbitals')
                rotated_dm = np.asarray(mf.make_rdm1(mo_coeff=rotated, mo_occ=mf.mo_occ))
                if not np.isfinite(rotated_dm).all() or np.iscomplexobj(rotated_dm):
                    raise ValueError('Invalid closed-shell density from internal stability orbitals')
                repair_dir = out / f'internal-repair-{round_id + 1:02d}'
                repair_dir.mkdir(parents=True, exist_ok=True)
                seed_ref = atomic_array(repair_dir / '.local/rotated_density.npy', rotated_dm)
                candidate, candidate_dm, candidate_audit, calls = _run_electronic(
                    symbols, coords, repair_dir, grid_level, rotated_dm, log)
                atomic_json(repair_dir / 'electronic-attempts.json',
                            {'audit': candidate_audit, 'calls': calls})
                attempt.update(repair_seed_sha256=seed_ref['sha256'],
                               repaired_energy_hartree=candidate_audit['fresh_energy_hartree'],
                               repaired_electronic_qualified=candidate_audit['qualified'],
                               repair_directory=str(repair_dir))
                if not candidate_audit['qualified']:
                    value['internal_repair_error'] = {
                        'type': 'UnqualifiedElectronicRecovery',
                        'message': 'Rotated-orbital strict SCF recovery failed; candidate retained diagnostically.'}
                    atomic_array(repair_dir / '.local/unqualified_density.npy', candidate_dm)
                    break
                mf = candidate
                final_audit = candidate_audit
                value['internal_stable'] = None  # A new density requires its own next test.
                value['internal_repair_performed'] = True
                value['geometry_reoptimization_required'] = True
                atomic_json(out / 'repair-progress.json', value)
            except Exception as error:
                value['internal_error'] = {'type': type(error).__name__, 'message': str(error)}
                if not value['internal_repair_attempts'] or value['internal_repair_attempts'][-1] is not attempt:
                    value['internal_repair_attempts'].append(attempt)
                break
        final_dm, final_audit = _audit(mf)
        density_ref = atomic_array(out / '.local/stability_density.npy', final_dm)
        value['array_refs']['density'] = density_ref
        value.update(density_path=density_ref['path'], candidate_density_path=density_ref['path'],
                     final_electronic_audit=final_audit,
                     final_energy_hartree=final_audit['fresh_energy_hartree'],
                     energy_change_hartree=final_audit['fresh_energy_hartree'] - value['initial_energy_hartree'],
                     density_change_frobenius=float(np.linalg.norm(final_dm - initial_dm)))
        try:
            external = mf.stability(internal=False, external=True, return_status=True)
            value['rks_to_uks_stable'] = _native_stability_status(external[3])
            log.flush()
            logged = (out / 'pyscf.log').read_text()
            matches = re.findall(r'rhf_real2complex: lowest eigs of H =\s*(\[[^\]]*\]|[^\n]*)', logged)
            if matches:
                eigenvalues = [float(x) for x in re.findall(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?', matches[-1])]
                if eigenvalues and np.isfinite(eigenvalues).all():
                    value['real_to_complex_lowest_eigenvalue_hartree'] = min(eigenvalues)
                    value['real_to_complex_stable'] = bool(min(eigenvalues) >= -1e-5)
            if value['real_to_complex_stable'] is not None and value['rks_to_uks_stable'] is not None:
                value['external_stable'] = bool(value['rks_to_uks_stable'] and value['real_to_complex_stable'])
            value['external_native_return_status_scope'] = 'PySCF2.13 rhf_external return_status reports RKS-to-UKS only; real-to-complex is explicitly parsed from native solver eigenvalue log.'
        except Exception as error:
            value['external_error'] = {'type': type(error).__name__, 'message': str(error)}
        passed = bool(final_audit['qualified'] and value['internal_stable'] is True
                      and value['external_stable'] is True)
        value.update(status='DONE', finished_at_utc=utc(),
                     tested_electronic_stability_passed=passed,
                     supported_for_primary_hessian=passed and not value['geometry_reoptimization_required'],
                     electronic_state_disposition=(
                         'INTERNAL_REPAIR_REQUIRES_GEOMETRY_REOPTIMIZATION'
                         if passed and value['geometry_reoptimization_required'] else
                         'STABLE_REAL_CLOSED_SHELL_RKS' if passed else
                         'UNSUPPORTED_OR_UNRESOLVED_ELECTRONIC_STABILITY'),
                     scope='Actual RKS internal orbital repair plus RKS-to-UKS and real-to-complex tests; external or unknown/failed tests are not passes.')
        atomic_json(out / 'result.json', value)
        return value
    finally:
        log.close()
