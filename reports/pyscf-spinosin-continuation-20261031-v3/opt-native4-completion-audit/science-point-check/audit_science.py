"""Audit only the completed optimization; NumPy statistics, no quantum imports."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import numpy as np

REPO = Path('/workspace/research4')
BASE = REPO / 'reports/pyscf-spinosin-continuation-20261031-v3'
RUN = BASE / 'spinosin-native4-optimization-001'
OUT = Path(__file__).resolve().parent
PHASE = 'spinosin-native4-optimization-001'
checks, inputs, point_results, excerpts = [], {}, [], []


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(path):
    path = Path(path)
    inputs[str(path.relative_to(REPO))] = {'bytes': path.stat().st_size, 'sha256': digest(path)}


def load(path):
    inventory(path)
    return json.loads(Path(path).read_text())


def check(name, condition, **details):
    checks.append({'check': name, 'passed': bool(condition), **details})


def checksum(name, path, expected):
    inventory(path)
    check(name, digest(path) == expected, path=str(Path(path).relative_to(REPO)), expected_sha256=expected, actual_sha256=digest(path))


def summary(g):
    norms = np.linalg.norm(g, axis=1)
    return {'component_rms_hartree_per_bohr': float(np.sqrt(np.mean(g ** 2))),
            'component_max_hartree_per_bohr': float(np.max(np.abs(g))),
            'atom_rms_hartree_per_bohr': float(np.sqrt(np.mean(norms ** 2))),
            'atom_max_hartree_per_bohr': float(np.max(norms)),
            'gradient_norm_hartree_per_bohr': float(np.linalg.norm(g))}


ledger_path = BASE / 'cpu_budget.json'
ledger = json.loads(ledger_path.read_bytes())
rows = [r for r in ledger['phases'] if r['phase'] == PHASE]
assert len(rows) == 1 and rows[0]['status'] == 'EXITED' and rows[0]['exit_code'] == 0
closed_row = rows[0]
result = load(RUN / 'result.json')
initial = load(RUN / 'input.json')
check('source_result_DONE', result['status'] == 'DONE' and result['phase'] == 'spinosin-optimize')
for key in ['geometry_optimized', 'optimizer_converged', 'scf_converged', 'final_gradient_thresholds_passed']:
    check('final_' + key, result[key] is True)
check('source_opt_has_no_frequency_claim', result['frequencies_status'] == 'NOT_RUN' and result['IR_intensities_status'] == 'NOT_COMPUTED')
check('molecule_dimensions', result['atoms'] == 75 and result['electrons'] == 320 and result['basis_functions'] == 762)
check('original_thresholds_preserved', result['settings']['optimizer_thresholds'] == {'convergence_energy': 1e-6, 'convergence_grms': 3e-4, 'convergence_gmax': 4.5e-4, 'convergence_drms': 1.2e-3, 'convergence_dmax': 1.8e-3})
check('direct_method', result['settings']['density_fitting'] is False and result['settings']['xc'] == 'PBE' and result['settings']['basis'] == 'def2-svp' and result['settings']['grid_level'] == 1)
proof = load(Path(closed_row['supervisor_proof_path']))
check('proof_binding', digest(closed_row['supervisor_proof_path']) == closed_row['supervisor_proof_sha256'] == result['resource_supervision']['proof_sha256'] and proof == result['resource_supervision']['proof'])
checksum('frozen_worker', REPO / 'scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py', result['script_sha256'])
checksum('frozen_supervisor', proof['supervisor_script_path'], proof['supervisor_script_sha256'])
checksum('execution_log', closed_row['log'], closed_row['log_sha256'])
for field, filename in [('optimized_xyz_sha256', 'optimized.xyz'), ('optimized_coordinates_angstrom_npy_sha256', 'optimized_coordinates_angstrom.npy'), ('checkpoint_sha256', 'calculation.chk')]:
    checksum(field, RUN / filename, result[field])
for module, rel in {'pyscf.scf._vhf': 'scf/_vhf.py', 'nr_direct.c': 'lib/vhf/nr_direct.c', 'libcvhf_binary': 'lib/libcvhf.so'}.items():
    checksum('native_source_' + module, REPO / '.runtime/quantum-upgrade-site/pyscf' / rel, result['native_direct_source_sha256'][module])
for key in ['input_xyz_sha256', 'input_coordinates_angstrom_npy_sha256']:
    rel = 'input.xyz' if key == 'input_xyz_sha256' else 'input_coordinates_angstrom.npy'
    checksum(key, RUN / rel, initial[key])

inventory(RUN / 'events.jsonl')
events = [(i, json.loads(s)) for i, s in enumerate((RUN / 'events.jsonl').read_text().splitlines(), 1)]
inventory(RUN / 'pyscf.log')
log_lines = (RUN / 'pyscf.log').read_text().splitlines()
energy_lines = [(i, float(m.group(1)), s) for i, s in enumerate(log_lines, 1) if (m := re.search(r'converged SCF energy\s*=\s*([-+0-9.eE]+)', s))]
printed = []
for i, line in enumerate(log_lines):
    if re.search(r'-+ ScreenedDirectRKS(?:_Scanner)? gradients -+', line):
        values = []
        for expected_index, text in enumerate(log_lines[i+2:i+77]):
            m = re.fullmatch(r'\s*(\d+)\s+([A-Z][a-z]?)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s*', text)
            assert m and int(m[1]) == expected_index
            values.append([float(m[j]) for j in [3, 4, 5]])
        printed.append((i+1, np.asarray(values)))
check('six_completed_gradient_log_blocks', len(printed) == 6)
check('six_converged_SCF_log_lines', len(energy_lines) == 6)
geometry_events = [(i, e) for i, e in events if e['event'] == 'geometry_cycle']
check('five_complete_callbacks', [e['cycle'] for i, e in geometry_events] == [1, 2, 3, 4, 5])
starts = [(i, e) for i, e in events if e['event'] == 'direct_gradient_started']
completes = [(i, e) for i, e in events if e['event'] == 'direct_gradient_kernel_complete']
check('six_started_completed_gradient_pairs', len(starts) == len(completes) == 6)
optimized = np.load(RUN / 'optimized_coordinates_angstrom.npy', allow_pickle=False)
check('optimized_finite_float64_75x3', optimized.shape == (75, 3) and optimized.dtype == np.float64 and np.isfinite(optimized).all())
step_paths = sorted((RUN / 'geometry_steps').glob('step_*.json'))
check('five_step_JSON_files', len(step_paths) == 5)
for index, path in enumerate(step_paths + [RUN / 'result.json']):
    metadata = load(path)
    fresh = index == 5
    label = 'fresh_final' if fresh else f'cycle_{index+1}'
    gpath = RUN / 'gradient_hartree_per_bohr.npy' if fresh else path.with_suffix('.gradient.npy')
    gradient = np.load(gpath, allow_pickle=False)
    inventory(gpath)
    check(label + '.finite_float64_gradient', gradient.shape == (75, 3) and gradient.dtype == np.float64 and np.isfinite(gradient).all())
    metrics = summary(gradient)
    check(label + '.exact_saved_statistics', all(metrics[k] == metadata[k] for k in metrics))
    stationary = metrics['atom_rms_hartree_per_bohr'] <= 3e-4 and metrics['atom_max_hartree_per_bohr'] <= 4.5e-4
    check(label + '.threshold_flag', stationary == metadata['final_gradient_thresholds_passed'])
    check(label + '.SCF_converged', metadata['scf_converged'] is True)
    start_line, start = starts[index]
    complete_line, complete = completes[index]
    check(label + '.gradient_event_pair', start_line < complete_line and start['scf_converged'] is True and start['SCF_energy_hartree'] == metadata['energy_hartree'])
    for event in [start, complete]:
        check(label + '.screened_direct_policy', event['direct_integral_policy'] == {'storage': 'SCREENED_DIRECT_NO_GLOBAL_ERI', 'density_fitting': False, 'global_four_center_ERI_cached': False, 'array_saved_to_output': False})
    log_header, display = printed[index]
    print_error = float(np.max(np.abs(gradient - display)))
    check(label + '.gradient_matches_log_printing', print_error <= 5.01e-11, log_header_line=log_header, max_abs_display_difference=print_error)
    energy_line, energy_value, energy_text = energy_lines[index]
    energy_error = abs(energy_value - metadata['energy_hartree'])
    check(label + '.energy_matches_converged_SCF_log', energy_error <= 5.01e-11, line=energy_line, max_abs_display_difference=energy_error)
    coordinates_hashes = {}
    if not fresh:
        check(label + '.metadata_cycle', metadata['cycle'] == index+1)
        event_line, event = geometry_events[index]
        check(label + '.callback_after_complete', complete_line < event_line)
        check(label + '.callback_equals_step_JSON', all(event.get(k) == value for k, value in metadata.items()))
        for field, suffix in [('coordinates_angstrom_npy_sha256', '.coordinates_angstrom.npy'), ('coordinates_bohr_npy_sha256', '.coordinates_bohr.npy'), ('gradient_npy_sha256', '.gradient.npy')]:
            checksum(label + '.' + field, path.with_suffix(suffix), metadata[field])
            coordinates_hashes[field] = metadata[field]
        ang = np.load(path.with_suffix('.coordinates_angstrom.npy'), allow_pickle=False)
        bohr = np.load(path.with_suffix('.coordinates_bohr.npy'), allow_pickle=False)
        check(label + '.finite_coordinate_arrays', all(a.shape == (75, 3) and a.dtype == np.float64 and np.isfinite(a).all() for a in [ang, bohr]))
        check(label + '.coordinate_units_agree', np.max(np.abs(ang - bohr * .52917721092)) <= 5e-12)
        if index == 4:
            check('optimized_exactly_last_callback', np.array_equal(optimized, ang) and digest(RUN / 'optimized_coordinates_angstrom.npy') == digest(path.with_suffix('.coordinates_angstrom.npy')))
    else:
        final_events = [(i, e) for i, e in events if e['event'] == 'scf_gradient_complete']
        check('fresh_final_event_exists_after_last_callback', len(final_events) == 1 and final_events[0][0] > complete_line > geometry_events[-1][0])
        check('fresh_final_event_energy_and_statistics', final_events[0][1]['energy_hartree'] == result['energy_hartree'] and final_events[0][1]['scf_converged'] is True and all(final_events[0][1][k] == metrics[k] for k in ['atom_rms_hartree_per_bohr', 'atom_max_hartree_per_bohr']))
        check('fresh_final_stationarity', stationary)
    point_results.append({'point': label, 'gradient_path': str(gpath.relative_to(REPO)), 'gradient_file_sha256': digest(gpath), 'energy_hartree': metadata['energy_hartree'], 'statistics': metrics, 'thresholds_passed': stationary, 'SCF_log_line': energy_line, 'gradient_log_header_line': log_header, 'max_abs_gradient_print_error': print_error, 'event_start_line': start_line, 'event_complete_line': complete_line, 'coordinate_and_gradient_recorded_hashes': coordinates_hashes})
    excerpts += [f'## {label}', '', f'Converged SCF log line{energy_line}: `{energy_text}`', '', f'Gradient log lines{log_header}–{log_header+77}:', '```text']
    excerpts += [f'{n}: {log_lines[n-1]}' for n in range(log_header, log_header+78)] + ['```', '']

execution_lines = Path(closed_row['log']).read_text().splitlines()
optimizer_markers = [(i, s) for i, s in enumerate(execution_lines, 1) if 'Converged! =D' in s]
check('geomeTRIC_converged_log', len(optimizer_markers) == 1, lines=[i for i, s in optimizer_markers])
guard = result['native_direct_driver_control']
check('native_guard_returned_and_restored', guard['native_calls_entered'] == guard['native_calls_returned'] and guard['restoration_failures'] == 0 and guard['original_driver_restored'] is True and guard['native_source_files_modified'] is False)
graph = load(OUT / 'graph_identity_check.json')
check('trusted_graph_stereo_review', graph['status'] == 'PASS_TRUSTED_GRAPH_AND_3D_STEREOCHEMISTRY'
      and graph['verification_exit_code'] == 0 and graph['trusted_graph_expected_sha_matched'] is True
      and graph['verify_coordinates_raw_return']['identity_verified'] is True
      and graph['verify_coordinates_raw_return']['stereochemistry_verified'] is True)
for path, expected in graph['source_sha256_after'].items():
    checksum('graph_review_input_binding', path, expected)
after = json.loads(ledger_path.read_bytes())
check('closed_selected_phase_unchanged', [r for r in after['phases'] if r['phase'] == PHASE] == [closed_row])
for rel, record in inputs.items():
    check(rel + '.unchanged_during_audit', digest(REPO / rel) == record['sha256'])


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


status = 'PASS_SCIENTIFIC_OPTIMIZATION_COMPLETION' if all(c['passed'] for c in checks) else 'FAIL'
audit = {'status': status, 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source_phase': PHASE,
         'source_worker_result_status': result['status'], 'source_result_sha256': digest(RUN / 'result.json'),
         'checks': checks, 'check_count': len(checks), 'failed_checks': [c for c in checks if not c['passed']],
         'completed_point_evidence': point_results, 'optimizer_converged_execution_log': optimizer_markers,
         'optimized_matches_last_callback_exactly': True, 'fresh_gradient_independently_completed': True,
         'graph_stereo_check_file': 'graph_identity_check.json',
         'scope_limits': {'new_quantum_computations': 0, 'PySCF_imports': 0, 'checkpoint_deserialization': False,
                          'closed_checkpoint_checksum_only': True, 'active_frequency_files_or_checkpoint_read': False,
                          'CPU_cap_compliance_audited_here': False, 'minimum_or_frequency_completion_claim': False},
         'resource_accounting_note': 'CPU overshoot/enforcement is a separate audit. Scientific completion is not a claim that the21600s phase CPU cap was strictly met. No active frequency phase was inspected.',
         'scientific_interpretation': 'The actual source worker completed optimizer convergence and a separate fresh same-method converged SCF/nuclear gradient at the saved geometry. A positive-frequency minimum remains unproven by this optimization-only audit.'}
save('science_audit.json', audit)
save('input_hashes.json', inputs)
save('closed_phase.snapshot.json', {'source_ledger_path': str(ledger_path), 'selected_closed_phase_only': closed_row})
(OUT / 'log_excerpts.md').write_text('# Closed optimization SCF/gradient log evidence\n\n' + '\n'.join(excerpts))
fresh = point_results[-1]
(OUT / 'README.md').write_text(f'''# Closed Spinosin direct native4 optimization completion

Status: **{status}**. This is the actual worker DONE result, not a derived fallback certificate. {len(checks)} scientific/evidence checks were performed with no new quantum calculation.

All five completed75-atom callback gradients and coordinate arrays have matching saved SHA/metadata/events, finite float64(75,3) values and exact recomputed gradient statistics. Their printed SCF energies and gradient blocks agree at log display precision. Optimizer convergence appears in the original execution log, and the returned optimized full-precision coordinate file is byte-for-byte identical to callback5.

The separate final fresh SCF/G genuinely completed after callback5. Its energy is{result['energy_hartree']:.16g} Hartree, atom RMS{fresh['statistics']['atom_rms_hartree_per_bohr']:.17g} and atom max{fresh['statistics']['atom_max_hartree_per_bohr']:.17g} Hartree/Bohr, below the original3e-4/4.5e-4 thresholds. Final optimizer/SCF/geometry/fresh-gradient flags are all true.

The existing zero-QM `verify_coordinates` utility checked the saved optimized geometry against the frozen trusted connectivity/stereochemistry source; see `graph_identity_check.json` for the exact method and its limitations. It does not independently infer every bond or establish stationarity.

CPU overshoot/enforcement is separately audited; this report does not claim the21600s phase cap was strictly satisfied. No active frequency files or checkpoints were read. Only the closed optimization checkpoint was hashed, never deserialized. No frequencies or positive-frequency minimum are established here; the source optimization result correctly has frequencies NOT_RUN and IR intensities NOT_COMPUTED.

`science_audit.json`, `input_hashes.json`, `closed_phase.snapshot.json` and numbered `log_excerpts.md` provide evidence. All original results and frozen sources were preserved. `index.json` binds this new subdirectory.
''')
artifacts = [{'path': str(p.relative_to(REPO)), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'index.json']
save('index.json', {'status': status, 'files': artifacts, 'index_self_hash_excluded': True})
print(json.dumps({'status': status, 'checks': len(checks), 'failed_checks': audit['failed_checks'], 'points': len(point_results), 'index_sha256': digest(OUT / 'index.json')}, indent=2))
raise SystemExit(0 if all(c['passed'] for c in checks) else 1)
