"""Independent memory-repair bridge to the immutable Spinosin workflow.

The molecular model and scientific acceptance tests are unchanged. Only the
full grid-response accumulation is blocked before allocating large Becke/AO
arrays. Imported SCF checkpoints supply unqualified initial densities, never a
gradient, stationary point, Hessian, or minimum certificate.
"""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import numpy as np

import spinosin_gap_science_20261112 as original

ROOT = Path(__file__).resolve().parents[1]
REPAIR_ROOT = ROOT / 'reports/spinosin_ir_memory_repair_20261112'
ScientificIntegrityError = original.ScientificIntegrityError
DESIGN = {
    'schema': 'spinosin-memory-repair-bridge-20261112-v1',
    'cases': ['C003', 'C008'],
    'default_grid_block_size': 2048,
    'model_unchanged': 'PBE/def2-SVP, neutral closed-shell, native grid5/7, full grid and auxiliary-basis responses',
    'changed': 'Block grid-response Becke partition and AO work before allocation; preserve original atom owner and all response terms',
    'accumulation_order_changed': True,
    'scientific_acceptance_thresholds_changed': False,
    'warm_start_scope': 'Unqualified density reconstructed from a same-case, same-geometry electronic checkpoint; fresh strict SCF/gradient required',
    'completed_gradient_reused_from_old_run': False,
    'old_run_modified': False,
    'global_round_cpu_time_retry_stop_limits': None,
}
PROTOCOL = DESIGN


def _adapter():
    return importlib.import_module('spinosin_grid_response_streaming_20261112')


def _path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _recovery_receipt(case, recovery_root=None):
    root = REPAIR_ROOT / 'recovery_inputs' if recovery_root is None else _path(recovery_root)
    path = root / case['case_id'] / 'recovery_input.json'
    if not path.is_file():
        raise ScientificIntegrityError('Missing explicit unqualified warm-start receipt: ' + str(path))
    receipt = json.loads(path.read_text())
    if receipt.get('case_id') != case['case_id'] or receipt.get('case_declared_input_hash') != case.get('input_hash'):
        raise ScientificIntegrityError('Warm-start receipt belongs to a different conformer/input')
    if (receipt.get('scientifically_qualified') is not False or receipt.get('gradient_available') is not False
            or receipt.get('stationary_point_verified') is not False):
        raise ScientificIntegrityError('Recovery input must explicitly be an unqualified electronic guess')
    refs = {
        'snapshot': {'path': str(_path(receipt['snapshot_path'])), 'sha256': receipt['snapshot_sha256']},
        'density': {'path': str(_path(receipt['warm_density_path'])), 'sha256': receipt['warm_density_sha256']},
    }
    original._check_refs(refs)
    return path.resolve(), receipt, refs


def _validate_warm_density(case, receipt, refs):
    """Validate the actual frozen checkpoint, not just historical receipt flags."""
    from pyscf import lib
    from pyscf.scf import chkfile
    # This read-only validation does not run SCF, nuclear gradients or optimizer.
    # Use one thread for the small reconstruction/overlap check.
    previous_threads = lib.num_threads()
    lib.num_threads(1)
    try:
        mol, scf = chkfile.load_scf(refs['snapshot']['path'])
        symbols = [mol.atom_symbol(index) for index in range(mol.natm)]
        coordinates = np.asarray(mol.atom_coords())
        expected = np.asarray(case['initial_geometry_bohr'])
        if symbols != case['symbols'] or coordinates.shape != expected.shape:
            raise ScientificIntegrityError('Recovery checkpoint atom order/count differs from its conformer')
        difference = float(np.max(np.abs(coordinates - expected)))
        if difference > 1e-12:
            raise ScientificIntegrityError('Warm-start geometry differs from the new workflow initial geometry')
        if mol.charge != 0 or mol.spin != 0 or mol.nelectron != 320 or mol.nao_nr() != 762:
            raise ScientificIntegrityError('Warm-start molecular charge/spin/electron/basis dimensions changed')
        if str(mol.basis).lower() != 'def2-svp':
            raise ScientificIntegrityError('Warm-start AO basis differs from def2-SVP')
        coefficient = np.asarray(scf['mo_coeff'])
        occupation = np.asarray(scf['mo_occ'])
        energy = np.asarray(scf['mo_energy'])
        density = np.load(refs['density']['path'], allow_pickle=False)
        if (coefficient.shape != (762, 762) or occupation.shape != (762,) or energy.shape != (762,)
                or density.shape != (762, 762) or np.iscomplexobj(density)
                or not all(np.isfinite(value).all() for value in (coefficient, occupation, energy, density))
                or not np.all((occupation == 0) | (occupation == 2)) or occupation.sum() != 320):
            raise ScientificIntegrityError('Incomplete finite real closed-shell recovery orbital/density data')
        reconstructed = (coefficient * occupation) @ coefficient.T
        density_difference = float(np.max(np.abs(density - reconstructed)))
        if not np.allclose(density, reconstructed, rtol=1e-12, atol=1e-12):
            raise ScientificIntegrityError('Warm density is not the density of its authenticated checkpoint orbitals')
        overlap = mol.intor_symmetric('int1e_ovlp')
        electrons = float(np.einsum('ij,ji->', density, overlap))
        if abs(electrons - 320) > 1e-7:
            raise ScientificIntegrityError('Warm density has the wrong overlap-weighted electron count')
        return {'atom_count': mol.natm, 'ao_count': mol.nao_nr(), 'charge': mol.charge, 'spin': mol.spin,
                'overlap_weighted_electrons': round(electrons, 9), 'orbital_occupation_sum': float(occupation.sum()),
                'geometry_max_abs_difference_bohr': round(difference, 12),
                'density_reconstruction_max_abs_difference': round(density_difference, 12),
                'reporting_note': 'Acceptance uses unrounded residuals; diagnostic rounding prevents irrelevant BLAS last-bit changes in frozen provenance',
                'seed_is_scientifically_qualified': False, 'nuclear_gradient_imported': False}
    finally:
        lib.num_threads(previous_threads)


def execution_dependencies(case, recovery_root=None):
    case, _, _ = original._validate_case(case)
    receipt_path, _, refs = _recovery_receipt(case, recovery_root)
    adapter = _adapter()
    files = {Path(__file__).resolve(), Path(original.__file__).resolve(), Path(adapter.__file__).resolve(), receipt_path}
    files.update(Path(ref['path']) for ref in refs.values())
    files.update(path.resolve() for path in original.VENDOR.glob('*.py'))
    files.add(Path(original.__file__).with_name('spinosin_gap_ir_20261112.py').resolve())
    return {str(path): original._file_hash(path) for path in sorted(files)}


def _prepared(case, threads, max_memory_mb, block_size, recovery_root):
    case, _, _ = original._validate_case(case)
    if case['case_id'] not in DESIGN['cases']:
        raise ValueError('The memory-repair bridge is limited to C003/C008')
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError('Grid block size must be a positive integer')
    receipt_path, receipt, refs = _recovery_receipt(case, recovery_root)
    validation = _validate_warm_density(case, receipt, refs)
    adapter = _adapter()
    execution = {
        'design': DESIGN,
        'bridge_source_sha256': original._file_hash(__file__),
        'adapter_source_sha256': original._file_hash(adapter.__file__),
        'immutable_original_science_sha256': original._file_hash(original.__file__),
        'adapter_provenance': adapter.provenance(),
        'requested_grid_block_size': block_size,
        'warm_start_receipt': {'path': str(receipt_path), 'sha256': original._file_hash(receipt_path)},
        'warm_density': refs['density'], 'checkpoint_snapshot': refs['snapshot'],
        'old_source_fingerprint': receipt['old_source_fingerprint'],
        'source_checkpoint_sha256_at_snapshot': receipt['source_sha256_at_snapshot'],
        'snapshot_at_utc': receipt['snapshot_at_utc'],
        'warm_density_validation': validation,
        'completed_gradient_reused': False, 'stationary_point_or_minimum_imported': False,
    }
    supplied = case.get('memory_repair_execution')
    if supplied is not None and supplied != execution:
        raise ScientificIntegrityError('Caller supplied different repair execution provenance')
    case['memory_repair_execution'] = execution
    fingerprint = original.prepare_fingerprint(case, threads=threads, max_memory_mb=max_memory_mb)
    return case, fingerprint, refs['density']


def prepare_fingerprint(case, threads=2, max_memory_mb=3000, *, block_size=2048, recovery_root=None):
    return _prepared(case, threads, max_memory_mb, block_size, recovery_root)[1]


def _initialize_new_run(case, fingerprint, density_ref, output, progress=None, stop_requested=None):
    output = Path(output).resolve()
    if not output.is_relative_to(REPAIR_ROOT.resolve()):
        raise ScientificIntegrityError('Repaired scientific artifacts must use the new repair namespace')
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / 'SCIENCE_PROTOCOL.json'
    protocol = original._read_record(protocol_path)
    if protocol is not None and protocol != fingerprint:
        raise ScientificIntegrityError('New repair run already has a different frozen scientific protocol')
    if protocol is None:
        if any((output / name).exists() for name in ('workflow_state.json', 'generations', 'COMPLETE.json', 'result.json')):
            raise ScientificIntegrityError('Cannot import old scientific artifacts into a new repair run')
        original._record(protocol_path, fingerprint)
    state_path = output / 'workflow_state.json'
    state = original._read_record(state_path)
    if state is not None:
        if state.get('source_fingerprint') != fingerprint['source_fingerprint'] or state.get('case_id') != case['case_id']:
            raise ScientificIntegrityError('Repair workflow state belongs to a different frozen case')
        return
    if (output / 'generations').exists() or (output / 'COMPLETE.json').exists():
        raise ScientificIntegrityError('Cannot infer an initial repaired state from existing scientific artifacts')
    coordinates = np.asarray(case['initial_geometry_bohr']).tolist()
    state = {'case_id': case['case_id'], 'source_fingerprint': fingerprint['source_fingerprint'],
             'initial_geometry_bohr': coordinates, 'coordinates_bohr': coordinates,
             'density_ref': copy.deepcopy(density_ref), 'generation': 0, 'profile_index': 0,
             'chunk_index': 0, 'phase': 'OPTIMIZE', 'active_chunk': None, 'certification_round': 0,
             'history': [{'reason': 'UNQUALIFIED_SCF_WARM_START_IMPORTED',
                          'warm_density_sha256': density_ref['sha256'],
                          'gradient_qualification_imported': False,
                          'minimum_or_stationarity_imported': False}]}
    original._record(state_path, state)
    original._progress(progress, stop_requested, stage='repaired_warm_start_state_committed',
                       case_id=case['case_id'], source_fingerprint=fingerprint['source_fingerprint'],
                       seed_scientifically_qualified=False)


def run_conformer(case, output, threads=2, max_memory_mb=3000, progress=None, stop_requested=None,
                  *, block_size=2048, recovery_root=None):
    case, fingerprint, density_ref = _prepared(case, threads, max_memory_mb, block_size, recovery_root)
    _initialize_new_run(case, fingerprint, density_ref, output, progress, stop_requested)
    adapter = _adapter()
    def block_progress(event):
        if progress is not None:
            # A completed integration block is an operational heartbeat only.
            progress({**event, 'case_id': case['case_id'], 'checkpoint_committed': False})
    adapter.install(block_size=block_size, progress=block_progress)
    # Direct call; no sys.modules aliasing or replacement of the original module.
    return original.run_conformer(case, output, threads=threads, max_memory_mb=max_memory_mb,
                                  progress=progress, stop_requested=stop_requested)


run_case = run_conformer
