#!/usr/bin/env python3
"""Read-only audit of existing v3 acid results; no quantum calculation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import re
import numpy as np
from pyscf.scf import hf
from pyscf.data import elements

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / 'reports/pyscf-continuous-20261108-v3/jobs'
OUTPUT = Path(__file__).with_name('independent_acid_existing_audit.json')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def ref(path):
    p = Path(path)
    return {'path': str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
            'sha256': sha(p), 'bytes': p.stat().st_size}

def read(path, expected=None):
    if expected is not None:
        assert sha(path) == expected, ('JSON hash mismatch', str(path))
    return json.loads(Path(path).read_text())

def array(record):
    assert sha(record['path']) == record['sha256'], ('Array hash mismatch', record)
    value = np.load(record['path'], allow_pickle=False)
    assert np.isfinite(value).all()
    return value

def raw_sha(value):
    return hashlib.sha256(np.asarray(value, dtype='<f8').tobytes()).hexdigest()

def finite_point(path, expected=None):
    d = read(path, expected)
    assert d['status'] == 'POINT_COMPUTED'
    assert d['native_pyscf_converged'] and d['diagnostics']['native_converged']
    assert d['settings'] == SETTINGS and d['source_sha256'] == SOURCE_SHA
    g = array(d['gradient']); x = array(d['coordinates_bohr'])
    assert g.shape == x.shape == (17, 3)
    assert d['symbols'] == SYMBOLS
    assert d['inputs']['coordinates_sha256'] == raw_sha(x)
    assert hashlib.sha256(json.dumps(d['inputs'], sort_keys=True).encode()).hexdigest() == d['input_fingerprint']
    a = d['diagnostics']; residual = a['fresh_orbital_gradient_norm']
    assert np.isfinite(residual) and np.isfinite(a['energy_recheck_difference_hartree'])
    assert abs(a['fresh_energy_hartree'] - d['energy_hartree']) < 1e-14
    assert abs(float(np.sqrt(np.mean(g*g))) - d['nuclear_gradient_rms']) < 1e-14
    assert abs(float(np.max(np.abs(g))) - d['nuclear_gradient_max']) < 1e-14
    log = Path(d['gradient']['path']).parent / 'pyscf.log'
    log_text = log.read_text()
    cycles = re.findall(r'^cycle=\s*\d+ E=.*?delta_E=\s*([^ ]+)\s+\|g\|=\s*([^ ]+)', log_text, re.M)
    extra = re.findall(r'^Extra cycle\s+E=.*?delta_E=\s*([^ ]+)\s+\|g\|=\s*([^ ]+)', log_text, re.M)
    assert cycles and extra, ('Missing native SCF convergence evidence', str(log))
    last_main = tuple(float(v) for v in cycles[-1]); last_extra = tuple(float(v) for v in extra[-1])
    # Logs are printed with three significant figures; raw post-SCF diagnostics
    # remain separately reported and are never recast as passing 1e-7.
    assert abs(last_main[0]) < 1e-10 and last_main[1] < 1e-7
    assert abs(last_extra[0]) < 1e-9 or last_extra[1] < 3e-7
    info = {'result': ref(path), 'coordinates_bohr': d['coordinates_bohr'], 'gradient': d['gradient'],
            'pyscf_log': ref(log), 'input_fingerprint': d['input_fingerprint'],
            'native_pyscf_converged': True, 'fresh_orbital_gradient_norm': residual,
            'fresh_energy_recheck_difference_hartree': a['energy_recheck_difference_hartree'],
            'fresh_residual_le_1e_7': bool(residual <= 1e-7),
            'fresh_residual_le_3e_7': bool(residual <= 3e-7),
            'fresh_residual_le_former_strict_1e_9': bool(residual <= 1e-9),
            'main_cycle_last_printed_delta_energy_and_orbital_gradient': last_main,
            'extra_cycle_last_printed_delta_energy_and_orbital_gradient': last_extra,
            'quality_warnings': d['quality_warnings']}
    return d, x, g, info

source = ROOT / 'scripts/pyscf_continuous_science_20261108_v3.py'
SOURCE_SHA = sha(source)
geometry_path = JOBS / 'acid-g3-optimize/result.json'
geometry = read(geometry_path)
SETTINGS = geometry['settings']; SYMBOLS = geometry['symbols']
assert SETTINGS['energy_tolerance_hartree'] == 1e-10 and SETTINGS['orbital_gradient_tolerance'] == 1e-7
assert SETTINGS['grid_level'] == 3 and SETTINGS['grid_response'] is True
assert geometry['source_sha256'] == SOURCE_SHA
spec = importlib.util.spec_from_file_location('existing_v3_science_for_algebra_only', source)
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)

center_path = JOBS / 'acid-g3-optimize/final-center/result.json'
center, coordinates, gradient, center_info = finite_point(center_path)
assert geometry['coordinates_bohr'] == center['coordinates_bohr']
assert geometry['density'] == center['density']
assert geometry['derivative_settings'] == SETTINGS
density = array(center['density']); density_raw_sha = raw_sha(density)
atom_gradient_norms = np.linalg.norm(gradient, axis=1)
center_info.update(density=center['density'], nuclear_component_rms=float(np.sqrt(np.mean(gradient**2))),
                   nuclear_component_max=float(np.max(np.abs(gradient))),
                   geometric_atom_gradient_rms=float(np.sqrt(np.mean(atom_gradient_norms**2))),
                   geometric_atom_gradient_max=float(np.max(atom_gradient_norms)))
assert geometry['optimizer_converged'] is True
ordinary_force_pass = bool(center_info['geometric_atom_gradient_rms'] < 1e-4 and
                           center_info['geometric_atom_gradient_max'] < 3e-4)
strict_force_pass = bool(center_info['nuclear_component_rms'] <= 1e-5 and
                         center_info['nuclear_component_max'] <= 3e-5)

hessians = []; all_points = []
for name in ('acid-g3-h001', 'acid-g3-h0005'):
    path = JOBS / name / 'result.json'; h = read(path)
    assert h['status'] == 'HESSIAN_COMPUTED' and h['actual_finite_points'] == h['expected_points'] == 102
    assert not h['failed_points'] and h['native_unconverged_points'] == 0
    assert h['coordinates_bohr'] == center['coordinates_bohr'] and h['center_density_sha256'] == center['density']['sha256']
    assert h['settings'] == SETTINGS and h['geometry_source_derivative_settings'] == SETTINGS
    assert h['center_stationarity_settings_match'] is True
    signed = {1: {}, -1: {}}; receipts = []
    for point in h['point_results']:
        p = Path(point['path']); d, x, g, info = finite_point(p, point['sha256'])
        assert d['inputs']['initial_density_sha256'] == density_raw_sha
        match = re.fullmatch(r'c(\d+)-(plus|minus)', p.parent.name); assert match
        coordinate = int(match[1]); sign = 1 if match[2] == 'plus' else -1
        expected = coordinates.copy(); expected.flat[coordinate] += sign * h['step_bohr']
        assert np.array_equal(x, expected), ('Displacement mismatch', str(p))
        signed[sign][coordinate] = g; info.update(point=p.parent.name, step_bohr=h['step_bohr'])
        receipts.append(info)
    assert len(signed[1]) == len(signed[-1]) == 51
    rebuilt = ((np.array([signed[1][i] for i in range(51)]) - np.array([signed[-1][i] for i in range(51)]))/(2*h['step_bohr'])).reshape(51,51).T
    raw = array(h['raw_hessian']); assert np.array_equal(rebuilt, raw)
    asym = float(np.linalg.norm(raw-raw.T)/np.linalg.norm(raw))
    assert abs(asym-h['raw_relative_asymmetry']) < 1e-15
    masses = np.array([elements.MASSES[elements.charge(s)] for s in SYMBOLS])
    frequencies, values, modes, rank = worker.vibrational_modes(raw, coordinates, masses)
    assert len(frequencies) == 45 and rank == 6
    hessians.append({'result':ref(path), 'raw_hessian':h['raw_hessian'], 'step_bohr':h['step_bohr'],
                     'raw_relative_asymmetry':asym, 'signed_frequencies_cm1':frequencies.tolist(),
                     'signed_eigenvalues':values.tolist(), 'minimum_signed_frequency_cm1':float(frequencies.min()),
                     'minimum_eigenvalue_hartree_per_bohr2_amu':float(values.min()),
                     'imaginary_count':int((frequencies < 0).sum()), 'external_rank':rank,
                     'all_finite_points':102, 'point_receipts':receipts,
                     '_matrix':raw,'_modes':modes})
    all_points.extend(receipts)

analysis_path = JOBS / 'acid-g3-analysis/result.json'; analysis = read(analysis_path)
assert analysis['minimum_verified'] is False
for h, old in zip(hessians, analysis['input_hessians']):
    assert h['result']['sha256'] == old['sha256']
scale = np.repeat(np.sqrt(masses),3); common = hessians[0]['_modes']
projected = [common.T @ (.5*(h['_matrix']+h['_matrix'].T)/scale[:,None]/scale[None,:]) @ common for h in hessians]
difference = projected[1]-projected[0]
empirical = float(np.linalg.norm(difference,2))
relative = float(np.linalg.norm(difference)/np.linalg.norm(projected[1]))
assert abs(empirical-analysis['empirical_step_uncertainty_margin']) < 1e-12
assert abs(relative-analysis['projected_relative_step_difference']) < 1e-12
for i,h in enumerate(hessians):
    np.testing.assert_allclose(h['signed_frequencies_cm1'],analysis['signed_frequencies_cm1'][i],rtol=1e-9,atol=1e-7)
numerical_pass = bool(max(h['raw_relative_asymmetry'] for h in hessians) <= 1e-3 and relative <= 1e-3 and
                      all(min(h['signed_eigenvalues']) > empirical for h in hessians))
native_ordinary_protocol_pass = bool(ordinary_force_pass and numerical_pass and all(p['native_pyscf_converged'] for p in all_points))
fresh_1e7_pass = bool(center_info['fresh_residual_le_1e_7'] and all(p['fresh_residual_le_1e_7'] for p in all_points))
strict_pass = bool(strict_force_pass and center_info['fresh_residual_le_former_strict_1e_9'] and
                   all(p['fresh_residual_le_former_strict_1e_9'] and abs(p['fresh_energy_recheck_difference_hartree']) <= 1e-12 for p in all_points) and numerical_pass)
for h in hessians:
    del h['_matrix']; del h['_modes']

receipt = {
 'schema':'research4-independent-existing-acid-protocol-audit-v1',
 'created_at_utc':datetime.now(timezone.utc).isoformat(),
 'audit_source':ref(__file__), 'scientific_producer':ref(source), 'pyscf_scf_implementation':ref(hf.__file__),
 'scope':'Post-run audit of pre-existing ordinary native-PySCF protocol. A numerical approximate local minimum of this isolated neutral singlet PBE/def2-SVP/grid3 model; not an exact stationary point, global minimum, experimental structure, solid-state spectrum or rigorous mathematical error bound.',
 'original_results_modified':False, 'quantum_calculations_performed_by_auditor':False,
 'protocol_definition_timing':'Named post-run certificate for already executed settings; not a retrospectively preregistered strict residual protocol.',
 'protocol_minimum_verified':native_ordinary_protocol_pass,
 'protocol_minimum_verified_meaning':'Approximate local minimum under the original optimizer thresholds and PySCF native convergence semantics, with two-step positive projected numerical curvature. This does not assert final fresh orbital residual <=1e-7.',
 'strict_minimum_verified':strict_pass, 'final_fresh_residual_1e_7_certificate':fresh_1e7_pass,
 'original_analysis_minimum_verified':analysis['minimum_verified'],
 'native_scf_semantics':{'version':'2.13.0','configured_conv_tol':1e-10,'configured_conv_tol_grad':1e-7,
   'main_cycle':'abs(delta_E)<1e-10 AND orbital-gradient norm<1e-7',
   'extra_cycle':'After main-loop convergence, native driver accepts abs(delta_E)<1e-9 OR orbital-gradient norm<3e-7.',
   'all_center_and_204_point_main_and_extra_logs_consistent':True,
   'fresh_residual_independent_check':'Center and many displacement final fresh norms exceed1e-7, all below3e-7; they must not be called an independent1e-7 or1e-9 residual certificate.'},
 'optimizer_protocol':{'source':ref(source),'native_optimizer_converged':True,'energy_tolerance_hartree':1e-6,
  'gradient_rms_per_atom_tolerance_hartree_per_bohr':1e-4,'gradient_max_per_atom_tolerance_hartree_per_bohr':3e-4,
  'displacement_rms_angstrom':1.2e-3,'displacement_max_angstrom':1.8e-3,
  'all_final_fresh_force_targets_met':ordinary_force_pass,
  'former_strict_component_rms_1e_5_max_3e_5_met':strict_force_pass,
  'optimization_log':ref(JOBS/'acid-g3-optimize/attempts/attempt-001.log')},
 'settings':SETTINGS,'geometry':ref(geometry_path),'center':center_info,'hessians':hessians,'original_analysis':ref(analysis_path),
 'independent_numerical_reconstruction':{'all_json_and_gradient_coordinate_hashes_verified':True,
  'all204_displacements_exactly_equal_center_plus_or_minus_named_step':True,
  'all204_initial_density_fingerprints_equal_verified_center_density':True,
  'both_raw_hessians_exactly_reconstructed_from_saved_gradients':True,
  'all_point_settings_and_source_match_geometry_and_hessians':True,
  'all_center_gradients_and_hessians_same_grid3_grid_response_true':True,
  'both_external_ranks':6,'vibrational_modes_each':45,'imaginary_counts':[h['imaginary_count'] for h in hessians],
  'raw_relative_asymmetry':[h['raw_relative_asymmetry'] for h in hessians],
  'projected_relative_step_difference':relative,'empirical_step_difference_spectral_norm_hartree_per_bohr2_amu':empirical,
  'minimum_eigenvalue_minus_empirical_step_difference':[min(h['signed_eigenvalues'])-empirical for h in hessians],
  'empirical_step_difference_is_rigorous_error_bound':False,'two_step_numerical_diagnostics_passed':numerical_pass},
 'fresh_residual_summary':{'center':center_info['fresh_orbital_gradient_norm'],
  'displacement_points':len(all_points),'native_converged_points':sum(p['native_pyscf_converged'] for p in all_points),
  'max_displacement_fresh_residual':max(p['fresh_orbital_gradient_norm'] for p in all_points),
  'displacement_points_above_1e_7':sum(not p['fresh_residual_le_1e_7'] for p in all_points),
  'displacement_points_above_3e_7':sum(not p['fresh_residual_le_3e_7'] for p in all_points),
  'displacement_points_above_1e_9':sum(not p['fresh_residual_le_former_strict_1e_9'] for p in all_points),
  'max_fresh_energy_recheck_difference_hartree':max(abs(p['fresh_energy_recheck_difference_hartree']) for p in all_points)},
 'professional_recommendation':'The ordinary native-protocol approximate-local-minimum label is defensible if displayed with these semantics and false strict/fresh1e-7 flags. To repair the stricter certificate, continue an actual tighter SCF/geometry refinement and recompute its Hessian; changing only the validator must not be described as new strict scientific convergence.',
 'warnings':['Existing false minimum_verified retained. New protocol label is explicitly weaker and method-specific.',
  'PySCF native convergence true does not imply final independently rebuilt orbital gradient below the configured main-loop1e-7.',
  'Positive curvature and an empirical two-step margin support a local numerical conclusion; they do not certify a global minimum or rigorous continuum error bound.',
  'No IR intensities, solid-state THz assignments or classifier results are certified.']}
OUTPUT.write_text(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
print(json.dumps({'receipt':ref(OUTPUT),'protocol_minimum_verified':receipt['protocol_minimum_verified'],
 'strict_minimum_verified':strict_pass,'final_fresh_residual_1e_7_certificate':fresh_1e7_pass,
 'fresh_residual_summary':receipt['fresh_residual_summary'],
 'independent_numerical_reconstruction':receipt['independent_numerical_reconstruction']},indent=2))
