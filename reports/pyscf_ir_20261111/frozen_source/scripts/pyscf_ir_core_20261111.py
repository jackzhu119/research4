#!/usr/bin/env python3
"""Audited total-dipole points without unnecessary nuclear gradients.

The electronic solver is shared with the publication recovery, but this IR
path deliberately does not request a nuclear gradient at each displaced point.
Every cache includes the geometry, SCF model, density seed and implementation.
"""
from pathlib import Path
import hashlib
import json
import platform
import time

import numpy as np
import pyscf_publication_core_20261111 as electronic

SCHEMA = 'research4-pyscf-ir-dipole-v1'


def fingerprint(symbols, coordinates, density):
    import pyscf
    import scipy
    coords = np.asarray(coordinates, dtype=float)
    payload = {
        'schema': SCHEMA, 'symbols': list(symbols),
        'coordinates_bohr': coords.tolist(), 'settings': electronic.settings(),
        'density_seed_sha256': None if density is None else electronic._array_digest(density),
        'implementation_sha256': electronic.sha256(__file__),
        'electronic_implementation_sha256': electronic.sha256(electronic.__file__),
        'runtime': {'python':platform.python_version(),'pyscf':pyscf.__version__,
                    'numpy':np.__version__,'scipy':scipy.__version__},
        'dipole_convention': 'total nuclear plus electronic, atomic units, fixed laboratory origin (0,0,0)',
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return digest, payload


def calculate_dipole(symbols, coordinates, output, density=None, *, require_density=False):
    """Persist a qualified SCF dipole and density; retry unqualified points."""
    symbols, coords = electronic._validate(symbols, coordinates)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    digest, payload = fingerprint(symbols, coords, density)
    result_path = out / 'result.json'
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous.get('input_fingerprint') != digest:
            raise ValueError('IR point input fingerprint changed: ' + str(out))
        if previous.get('status') == 'DONE' and previous.get('qualified') is True:
            density_available=False
            for key,ref in previous['array_refs'].items():
                if key=='density' and not Path(ref['path']).exists():
                    continue  # rebuildable, deliberately unpublished cache
                if electronic.sha256(ref['path']) != ref['sha256']:
                    raise ValueError('IR point array fingerprint changed: ' + ref['path'])
                if key=='density':
                    density_available=True
            if not require_density or density_available:
                return previous
    attempt = out / f'attempt-{len(list(out.glob("attempt-*"))) + 1:04d}'
    attempt.mkdir()
    electronic.atomic_json(out / 'input.json', {**payload, 'input_fingerprint': digest})
    started, cpu = time.monotonic(), time.process_time()
    result = {**payload, 'input_fingerprint': digest, 'status': 'RUNNING',
              'qualified': False, 'started_at_utc': electronic.utc(), 'array_refs': {}}
    electronic.atomic_json(result_path, result)
    try:
        with (attempt / 'pyscf.log').open('w', buffering=1) as log:
            mf, dm, audit, calls = electronic._run_electronic(
                symbols, coords, attempt, None, density, log)
            dipole = np.asarray(mf.dip_moment(unit='AU', verbose=0))
        if dipole.shape != (3,) or not np.isfinite(dipole).all():
            raise ValueError('Nonfinite total dipole')
        refs = {
            'coordinates_bohr': electronic.atomic_array(out / 'coordinates_bohr.npy', coords),
            'dipole_au': electronic.atomic_array(out / 'dipole_au.npy', dipole),
            'density': electronic.atomic_array(out / '.local/density.npy', dm),
        }
        result.update(status='DONE' if audit['qualified'] else 'UNQUALIFIED',
                      qualified=bool(audit['qualified']), dipole_au=dipole.tolist(),
                      density_path=refs['density']['path'], energy_hartree=audit['fresh_energy_hartree'],
                      electronic_audit=audit, electronic_attempts=calls, array_refs=refs,
                      nuclear_gradient_evaluated=False)
    except Exception as error:
        result.update(status='ERROR', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result.update(finished_at_utc=electronic.utc(), wall_seconds=time.monotonic()-started,
                      cpu_seconds=time.process_time()-cpu)
        electronic.atomic_json(attempt / 'result.json', result)
        electronic.atomic_json(result_path, result)
    return result


def qualified_dipole(symbols, coordinates, output, density=None, *, require_density=False):
    result = calculate_dipole(symbols, coordinates, output, density,require_density=require_density)
    if result['qualified'] is not True:
        raise RuntimeError('IR electronic point requires retry: ' + str(output))
    return result


def directional_derivatives(symbols, coordinates, modes, output, steps_bohr, density, progress=None):
    """Direct simultaneous displacements with steps in Bohr*sqrt(amu).

    Modes are Cartesian L with L.T M_amu L=I. The returned derivative is
    per Bohr*sqrt(amu); actual maximum Cartesian moves are also recorded.
    No symmetry, atom reordering, reorientation or nuclear relaxation is used.
    """
    coords, vectors = np.asarray(coordinates), np.asarray(modes)
    if vectors.ndim != 2 or vectors.shape[0] != coords.size or not np.isfinite(vectors).all():
        raise ValueError('Invalid normal-mode matrix')
    if not steps_bohr or any(not np.isfinite(h) or h <= 0 for h in steps_bohr):
        raise ValueError('Positive finite displacement steps required')
    out = Path(output)
    derivatives, ledger = [], []
    for hi, h in enumerate(steps_bohr):
        columns = []
        for mi in range(vectors.shape[1]):
            mode = vectors[:, mi]
            largest = float(np.max(np.abs(mode)))
            if largest <= 0:
                raise ValueError('Zero mode direction')
            qstep = float(h)
            points = []
            for sign in (1, -1):
                if progress:
                    progress(stage='DIRECT_NORMAL_MODE_DIPOLE', step_bohr=float(h),
                             mode_index_one_based=mi+1, sign=sign,
                             planned_points=len(steps_bohr)*vectors.shape[1]*2)
                displaced = coords + sign*qstep*mode.reshape(coords.shape)
                point = qualified_dipole(symbols, displaced,
                    out/f'h{hi}'/f'm{mi:04d}-{"plus" if sign == 1 else "minus"}', density)
                points.append(np.asarray(point['dipole_au']))
                ledger.append({'mode_index': mi, 'step_bohr': float(h),
                    'normal_coordinate_step_bohr_sqrt_amu': qstep, 'sign': sign,
                    'maximum_cartesian_displacement_bohr': qstep*largest,
                    'input_fingerprint': point['input_fingerprint'], 'qualified': point['qualified']})
            columns.append((points[0]-points[1])/(2*qstep))
        array = np.stack(columns, axis=1)
        electronic.atomic_array(out/f'direct_derivative_h{hi}.npy', array)
        derivatives.append(array)
    return np.stack(derivatives), ledger


def cartesian_derivative(symbols, coordinates, output, step_bohr, density, progress=None):
    coords = np.asarray(coordinates)
    columns = []
    out = Path(output)
    for coordinate in range(coords.size):
        points = []
        for sign in (1, -1):
            if progress:
                progress(stage='REFINED_CARTESIAN_DIPOLE', coordinate=coordinate,
                         step_bohr=float(step_bohr), sign=sign)
            displaced = np.array(coords, copy=True)
            displaced.flat[coordinate] += sign*step_bohr
            point = qualified_dipole(symbols, displaced,
                out/f'c{coordinate:04d}-{"plus" if sign == 1 else "minus"}', density)
            points.append(np.asarray(point['dipole_au']))
        columns.append((points[0]-points[1])/(2*step_bohr))
    array = np.stack(columns, axis=1)
    electronic.atomic_array(out/'dipole_derivative.npy', array)
    return array
