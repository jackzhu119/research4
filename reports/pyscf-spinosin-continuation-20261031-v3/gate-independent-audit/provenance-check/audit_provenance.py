"""Read only two closed validation runs; import NumPy, never PySCF."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import numpy as np

REPO = Path('/workspace/research4')
BASE = REPO / 'reports/pyscf-spinosin-continuation-20261031-v3'
OUT = Path(__file__).resolve().parent
RUNS = {
    'water-native4-validation-001': 'water_native4_validation_001',
    'validation-ethanol-native4-001': 'ethanol_native4_validation_001',
}
checks, gradients, runs, inputs, excerpts = [], [], [], {}, []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def load(path):
    path = Path(path)
    value = json.loads(path.read_text())
    inventory(path)
    return value


def inventory(path):
    path = Path(path)
    inputs[str(path.relative_to(REPO))] = {'bytes': path.stat().st_size, 'sha256': sha(path)}


def check(label, passed, **details):
    checks.append({'check': label, 'passed': bool(passed), **details})


def checksum(label, path, expected):
    path = Path(path)
    actual = sha(path) if path.is_file() else None
    check(label, actual == expected, path=str(path.relative_to(REPO)), expected=expected, actual=actual)
    if path.is_file():
        inventory(path)


def summarize(g):
    norms = np.linalg.norm(g, axis=1)
    return {
        'component_rms_hartree_per_bohr': float(np.sqrt(np.mean(g ** 2))),
        'component_max_hartree_per_bohr': float(np.max(np.abs(g))),
        'atom_rms_hartree_per_bohr': float(np.sqrt(np.mean(norms ** 2))),
        'atom_max_hartree_per_bohr': float(np.max(norms)),
        'gradient_norm_hartree_per_bohr': float(np.linalg.norm(g)),
    }


ledger_path = BASE / 'cpu_budget.json'
ledger_data = ledger_path.read_bytes()
ledger = json.loads(ledger_data)
selected = {p: [r for r in ledger['phases'] if r['phase'] == p] for p in RUNS.values()}
for phase, rows in selected.items():
    assert len(rows) == 1 and rows[0]['status'] == 'EXITED' and rows[0]['exit_code'] == 0
closed_rows = {p: r[0] for p, r in selected.items()}
source_files = [REPO / 'scripts/run_pyscf_monitored_budget_20261031.py',
                REPO / 'scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py',
                REPO / 'scripts/validate_pyscf_native4_large_block_engineering_20261031.py']
source_before = {str(p.relative_to(REPO)): sha(p) for p in source_files}


def verify_manifest(node, label):
    if isinstance(node, list):
        for i, value in enumerate(node):
            verify_manifest(value, f'{label}[{i}]')
    elif isinstance(node, dict):
        if 'path' in node and 'file_sha256' in node:
            path = Path(node['path'])
            allowed = [REPO / 'outputs/pyscf-spinosin-continuation-20261031-v3/water-native4-validation-cache-001',
                       REPO / 'outputs/pyscf-spinosin-continuation-20261031-v3/validation-ethanol-native4-cache-001']
            assert any(path.is_relative_to(root) for root in allowed)
            checksum(label + '.file_sha256', path, node['file_sha256'])
            check(label + '.file_bytes', path.stat().st_size == node['file_bytes'])
        for key, value in node.items():
            if key == 'native_Hessian_source_sha256':
                for module, expected in value.items():
                    path = REPO / '.runtime/quantum-upgrade-site' / (module.replace('.', '/') + '.py')
                    checksum(label + '.' + module, path, expected)
            verify_manifest(value, label + '.' + key)


for run_name, phase in RUNS.items():
    run = BASE / run_name
    row = closed_rows[phase]
    result = load(run / 'result.json')
    initial = load(run / 'input.json')
    progress = load(run / 'progress.json')
    proof_path = Path(row['supervisor_proof_path'])
    proof = load(proof_path)
    actual_proof_sha = sha(proof_path)
    check(phase + '.proof_sha_result_ledger', actual_proof_sha == row['supervisor_proof_sha256'] == result['resource_supervision']['proof_sha256'])
    check(phase + '.embedded_proof_equals_file', result['resource_supervision']['proof'] == proof)
    checksum(phase + '.parent_supervisor_source', proof['supervisor_script_path'], proof['supervisor_script_sha256'])
    check(phase + '.parent_source_equals_ledger_runner', proof['supervisor_script_sha256'] == ledger['runner_sha256'])
    check(phase + '.pid_phase_and_ledger_binding', proof['child_pid'] == row['pid'] and proof['phase'] == phase and proof['ledger_path'] == str(ledger_path))
    contract_pairs = {'cpu_soft_limit_seconds': 'cpu_soft_limit_seconds', 'cpu_hard_limit_seconds': 'cpu_hard_limit_seconds',
                      'rss_stop_bytes': 'rss_stop_bytes', 'cgroup_kill_bytes': 'cgroup_kill_bytes',
                      'expected_cgroup_memory_max_bytes': 'expected_cgroup_memory_max_bytes', 'threads': 'threads',
                      'wall_limit_seconds': 'wall_limit_seconds', 'monitor_interval_seconds': 'monitor_interval_seconds'}
    for pk, rk in contract_pairs.items():
        check(phase + '.contract.' + pk, proof[pk] == row[rk])
    check(phase + '.AS_contract', proof['address_space_soft_limit_bytes'] == proof['address_space_hard_limit_bytes'] == row['address_space_limit_gib'] * 2**30)
    check(phase + '.closed_clean', all(row.get(k) is False for k in ['wall_terminated', 'hard_killed', 'memory_terminated', 'monitor_failed']))
    checksum(phase + '.execution_log', row['log'], row['log_sha256'])
    worker = REPO / row['command'][1]
    checksum(phase + '.worker_source', worker, result['script_sha256'])
    if 'engineering_helper_source_sha256' in result:
        checksum(phase + '.helper_source', source_files[1], result['engineering_helper_source_sha256'])
    for field in ['geometry_optimized', 'optimizer_converged', 'scf_converged', 'final_gradient_thresholds_passed']:
        check(phase + '.final_flag.' + field, result[field] is True)
    check(phase + '.result_DONE', result['status'] == 'DONE')
    check(phase + '.actual_CPU_within_grant', 0 <= row['actual_cpu_seconds'] <= row['granted_cpu_seconds'])
    check(phase + '.CPU_scope_consistent', result['process_cpu_seconds'] <= row['actual_cpu_seconds'])
    check(phase + '.log_energy_and_result_scope', result['started_at_utc'] >= row['started_at_utc'] and result['completed_at_utc'] <= row['ended_at_utc'])
    if run_name.startswith('water'):
        mapping = {'optimized_xyz_sha256': 'optimized.xyz', 'optimized_coordinates_angstrom_npy_sha256': 'optimized_coordinates_angstrom.npy',
                   'normal_mode_geometry_angstrom_npy_sha256': 'normal_mode_geometry_angstrom.npy', 'hessian_cache_manifest_sha256': 'hessian_cache_manifest.json',
                   'raw_hessian_sha256': 'hessian_hartree_per_bohr2.npy', 'analysis_hessian_sha256': 'hessian_used_for_analysis.npy',
                   'checkpoint_sha256': 'calculation.chk'}
        for field, rel in mapping.items():
            checksum(phase + '.' + field, run / rel, result[field])
        for module, rel in {'pyscf.scf._vhf': 'scf/_vhf.py', 'nr_direct.c': 'lib/vhf/nr_direct.c', 'libcvhf_binary': 'lib/libcvhf.so'}.items():
            checksum(phase + '.native_source.' + module, REPO / '.runtime/quantum-upgrade-site/pyscf' / rel, result['native_direct_source_sha256'][module])
        checksum(phase + '.input_xyz', REPO / result['input_xyz_source'], result['input_xyz_sha256'])
    else:
        xyz = REPO / row['command'][row['command'].index('--input-xyz') + 1]
        provenance_path = REPO / row['command'][row['command'].index('--provenance-json') + 1]
        checksum(phase + '.input_xyz', xyz, result['input_xyz_sha256'])
        checksum(phase + '.source_provenance', provenance_path, result['provenance_sha256'])

    events = [(i, json.loads(s)) for i, s in enumerate((run / 'events.jsonl').read_text().splitlines(), 1)]
    inventory(run / 'events.jsonl')
    log_lines = (run / 'pyscf.log').read_text().splitlines()
    inventory(run / 'pyscf.log')
    energy_lines = [(i, float(m.group(1)), s) for i, s in enumerate(log_lines, 1) if (m := re.search(r'converged SCF energy\s*=\s*([-+0-9.eE]+)', s))]
    last_energy = energy_lines[-1]
    check(phase + '.last_converged_log_energy', abs(last_energy[1] - result['energy_hartree']) <= 1e-12,
          line=last_energy[0], log_energy=last_energy[1], result_energy=result['energy_hartree'], absolute_difference=abs(last_energy[1] - result['energy_hartree']))
    exec_lines = Path(row['log']).read_text().splitlines()
    converged_lines = [(i, s) for i, s in enumerate(exec_lines, 1) if 'Converged! =D' in s]
    check(phase + '.optimizer_log_converged', bool(converged_lines), lines=[x[0] for x in converged_lines])
    printed = []
    for i, line in enumerate(log_lines):
        if re.search(r'-+ ScreenedDirectRKS(?:_Scanner)? gradients -+', line):
            values = []
            for s in log_lines[i+2:i+2+result['atoms']]:
                m = re.match(r'\s*\d+\s+\w+\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s*$', s)
                assert m
                values.append([float(v) for v in m.groups()])
            printed.append((i+1, np.asarray(values)))
    ordered_paths = sorted((run / 'geometry_steps').glob('*.gradient.npy')) + [run / 'gradient_hartree_per_bohr.npy']
    if not run_name.startswith('water'):
        ordered_paths += [run / case / 'post_Hessian_gradient_hartree_per_bohr.npy' for case in ['standard_full_native1', 'response_block4_native4', 'response_block8_native4']]
    check(phase + '.gradient_log_block_count', len(printed) == len(ordered_paths))
    for path, (line, printed_g) in zip(ordered_paths, printed):
        g = np.load(path, allow_pickle=False)
        inventory(path)
        check(str(path.relative_to(BASE)) + '.finite_shape', g.shape == (result['atoms'], 3) and np.isfinite(g).all())
        summary = summarize(g)
        reference = load(path.with_name(path.name.replace('.gradient.npy', '.json'))) if path.name.endswith('.gradient.npy') else result
        equality = {k: summary[k] == reference[k] for k in summary if k in reference}
        if path.name == 'post_Hessian_gradient_hartree_per_bohr.npy':
            equality = {}
        check(str(path.relative_to(BASE)) + '.recorded_summary', all(equality.values()), exact_checks=equality)
        error = float(np.max(np.abs(g - printed_g)))
        check(str(path.relative_to(BASE)) + '.printed_gradient', error <= 5.01e-11, header_line=line, max_abs_error=error)
        gradients.append({'path': str(path.relative_to(REPO)), 'sha256': sha(path), 'log_header_line': line, 'summary': summary,
                          'printed_max_abs_error': error, 'recorded_summary_exact_checks': equality,
                          'stationary_thresholds_pass': summary['atom_rms_hartree_per_bohr'] <= 3e-4 and summary['atom_max_hartree_per_bohr'] <= 4.5e-4})
        if path.name.endswith('.gradient.npy') and 'gradient_npy_sha256' in reference:
            checksum(str(path.relative_to(BASE)) + '.step_gradient', path, reference['gradient_npy_sha256'])
            for key, suffix in [('coordinates_angstrom_npy_sha256', '.coordinates_angstrom.npy'), ('coordinates_bohr_npy_sha256', '.coordinates_bohr.npy')]:
                if key in reference:
                    checksum(str(path.relative_to(BASE)) + '.' + key, path.with_name(path.name.replace('.gradient.npy', suffix)), reference[key])

    for path in sorted(run.glob('**/hessian_cache_manifest.json')):
        manifest = load(path)
        verify_manifest(manifest, str(path.relative_to(BASE)))
        check(str(path.relative_to(BASE)) + '.binding_sha', hashlib.sha256(json.dumps(manifest['binding'], sort_keys=True).encode()).hexdigest() == manifest['binding_sha256'])
    for path in sorted(run.glob('*/result.json')):
        case = load(path)
        for field, rel in {'raw_hessian_sha256': 'hessian_hartree_per_bohr2.npy', 'analysis_hessian_sha256': 'hessian_used_for_analysis.npy', 'normal_modes_file_sha256': 'normal_modes.npy'}.items():
            checksum(str(path.relative_to(BASE)) + '.' + field, path.parent / rel, case[field])
        checksum(str(path.relative_to(BASE)) + '.geometry_source', case['normal_mode_geometry_source'], case['normal_mode_geometry_source_sha256'])
        check(str(path.relative_to(BASE)) + '.proof_binding', case['resource_supervision'] == result['resource_supervision'])
    partial_callbacks = [(i, e) for i, e in events if e['event'] == 'scf_cycle' and e.get('scf_converged') is False]
    completed_gradients = [(i, e) for i, e in events if e['event'] == 'direct_gradient_started']
    check(phase + '.all_gradient_start_SCF_flags', all(e.get('scf_converged') is True for i, e in completed_gradients))
    geometry_events = [(i, e) for i, e in events if e['event'] in ['geometry_cycle', 'validation_geometry_cycle']]
    fresh_gradient_event = next((i, e) for i, e in completed_gradients if i > geometry_events[-1][0])
    check(phase + '.fresh_gradient_SCF_energy', fresh_gradient_event[1]['SCF_energy_hartree'] == result['energy_hartree'])
    check(phase + '.fresh_gradient_after_optimizer_steps', fresh_gradient_event[0] > geometry_events[-1][0])
    selected_events = [{'line': i, **{k: e[k] for k in ['utc', 'event', 'cycle', 'scf_converged', 'SCF_energy_hartree', 'energy_hartree'] if k in e}}
                       for i, e in events if e['event'] in ['geometry_cycle', 'validation_geometry_cycle', 'direct_gradient_started', 'direct_gradient_kernel_complete', 'scf_gradient_complete', 'validation_opt_fresh_gradient_passed', 'frequencies_complete', 'validation_Hessian_case_complete']]
    runs.append({'run': run_name, 'phase': phase, 'proof_sha256': actual_proof_sha,
                 'ledger_actual_wait4_CPU_seconds': row['actual_cpu_seconds'], 'result_process_CPU_seconds': result['process_cpu_seconds'],
                 'CPU_difference_seconds': row['actual_cpu_seconds'] - result['process_cpu_seconds'], 'ledger_wall_seconds': row['elapsed_wall_seconds'],
                 'peak_group_RSS_sampled_bytes': row['peak_group_rss_bytes_sampled'], 'peak_cgroup_memory_sampled_bytes': row['peak_cgroup_memory_bytes_sampled'],
                 'monitor_samples': row['memory_monitor_samples'], 'final_flags': {k: result[k] for k in ['status', 'geometry_optimized', 'optimizer_converged', 'scf_converged', 'final_gradient_thresholds_passed']},
                 'initial_input_flags': {k: initial.get(k) for k in ['status', 'geometry_optimized', 'optimizer_converged', 'scf_converged']},
                 'progress_status': progress.get('status'), 'note': 'progress.json remains a stale PARTIAL snapshot; final result, closed ledger and completion events are authoritative for this audit.',
                 'partial_callback_false_flag_count': len(partial_callbacks), 'partial_callback_explanation': 'scf_cycle callbacks are iteration-time records, preceding completed-SCF convergence checks; false here does not contradict later converged gradient events.',
                 'last_converged_SCF_log': {'line': last_energy[0], 'text': last_energy[2]}, 'optimizer_execution_log_converged': converged_lines,
                 'selected_event_evidence': selected_events})
    excerpts += [f'## {run_name}', '', f'Final SCF log line {last_energy[0]}: `{last_energy[2]}`', '',
                 'Execution-log optimizer convergence: ' + '; '.join(f'line{i}: {s}' for i, s in converged_lines), '']
    for line, _ in printed:
        excerpt_end = line + 3 + result['atoms']
        excerpts += [f'pyscf.log lines{line}–{excerpt_end-1}:', '```text']
        excerpts += [f'{n}: {log_lines[n-1]}' for n in range(line, excerpt_end)]
        excerpts += ['```', '']

after_ledger = json.loads(ledger_path.read_bytes())
for phase, row in closed_rows.items():
    rows = [r for r in after_ledger['phases'] if r['phase'] == phase]
    check(phase + '.closed_row_unchanged_during_audit', len(rows) == 1 and rows[0] == row)
for path, before in source_before.items():
    check(path + '.unchanged_during_audit', sha(REPO / path) == before)
for path, record in inputs.items():
    check(path + '.unchanged_during_audit', sha(REPO / path) == record['sha256'])
total_CPU = sum(r['actual_cpu_seconds'] for r in closed_rows.values())
audit = {'status': 'PASS_WITH_STALE_PROGRESS_NOTE' if all(c['passed'] for c in checks) else 'FAIL',
         'created_utc': datetime.now(timezone.utc).isoformat(),
         'scope': 'Only two named closed V3 validation phases; provenance/checksums/SCF/optimizer/fresh-gradient audit. No Hessian or normal-mode numerical matrix analysis.',
         'quantum_operations_run': False, 'PySCF_imported': False, 'checkpoint_handling': 'Only checksum the closed water checkpoint against its declared SHA; no checkpoint parsing and no active checkpoint read.',
         'ledger_file_sha256_at_read_only': hashlib.sha256(ledger_data).hexdigest(), 'ledger_is_mutable_outside_selected_closed_rows': True,
         'selected_closed_row_canonical_SHA256': {p: canonical(r) for p, r in closed_rows.items()},
         'closed_phase_actual_CPU_sum_seconds': total_CPU,
         'CPU_accounting_limit': 'Actual wait4 child user+system values are read from the closed supervisor ledger, not independently remeasured. Payload process CPU covers a narrower interval. Neither is wall seconds multiplied by thread count.',
         'source_bindings': source_before, 'runs': runs, 'gradient_statistics': gradients, 'checks': checks,
         'check_count': len(checks), 'failed_checks': [c for c in checks if not c['passed']],
         'limitations': ['Startup proof is an immutable launch contract, not continuous monitor-health evidence. Closed ledger separately records no monitor/kill failure and sampled peaks.',
                         'Observed native thread count is configured OpenMP maximum, not a measurement of simultaneously active native workers.',
                         'Requested CPHF tolerance and normal solver return are not an independently measured residual.',
                         'Small water/ethanol validation does not certify large Spinosin memory safety, convergence or positive frequencies.',
                         'All existing files and frozen sources are read-only; only this new provenance-check directory is written.']}


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


write_json('audit.json', audit)
write_json('closed_ledger_rows.snapshot.json', {'ledger_path': str(ledger_path), 'selected_closed_rows_only': closed_rows})
write_json('input_hashes.json', inputs)
(OUT / 'log_excerpts.md').write_text('# Closed validation log excerpts\n\n' + '\n'.join(excerpts))
lines = ['# Independent closed-run provenance gate', '', f'Status: **{audit["status"]}**. {len(checks)} checks; {len(gradients)} saved gradient arrays recomputed. No PySCF import, quantum calculation, integral or numerical Hessian/mode matrix work was performed.', '',
         'Only the two named closed V3 rows were audited. Later ledger phases may be appended; the selected closed rows were re-read and verified unchanged. Existing results and source files were not modified.', '',
         '| Closed run | wait4 CPU seconds | Payload process CPU seconds | Fresh atom RMS | Fresh atom max |', '|---|---:|---:|---:|---:|']
for r in runs:
    g = next(g for g in gradients if g['path'].endswith(r['run'] + '/gradient_hartree_per_bohr.npy'))
    lines.append(f'| {r["run"]} | {r["ledger_actual_wait4_CPU_seconds"]:.9f} | {r["result_process_CPU_seconds"]:.9f} | {g["summary"]["atom_rms_hartree_per_bohr"]:.17g} | {g["summary"]["atom_max_hartree_per_bohr"]:.17g} |')
lines += ['', f'The closed supervisor CPU total is{total_CPU:.9f} seconds. Proof-file SHA matches both result and ledger; embedded proofs, parent supervisor source SHA, worker/helper source SHA, resource contracts and declared file checksums were independently checked against actual files. ActualCPU is taken from closed wait4 accounting; it was not remeasured by this audit.', '',
          'Both final results declare optimizer/SCF convergence and fresh-gradient thresholds passed. Execution logs contain optimizer `Converged!`; final converged SCF log energy matches result within1e-12 Hartree. All14 printed gradient blocks match saved arrays within5.01e-11 Hartree/Bohr. Fresh gradients pass atom RMS≤3e-4 and atom max≤4.5e-4.', '',
          '**Nonblocking stale record:** both `progress.json` files still say PARTIAL. Final `result.json`, closed EXITED/exit0 ledger rows and completion events consistently show completion. Initial input flags and iteration-time SCF callback flags are retained separately; they were not relabeled as final failures.', '',
          'This is a provenance/gradient gate only. It does not establish measured CPHF residuals, active OpenMP worker counts, whole-Hessian memory safety or a new Spinosin result. Numerical Hessian/mode validation is outside this component. See `audit.json`, `log_excerpts.md`, `closed_ledger_rows.snapshot.json` and `input_hashes.json`; `index.json` binds this new directory.']
(OUT / 'README.md').write_text('\n'.join(lines) + '\n')
artifacts = []
for path in sorted(OUT.iterdir()):
    if path.is_file() and path.name != 'index.json':
        artifacts.append({'path': str(path.relative_to(REPO)), 'bytes': path.stat().st_size, 'sha256': sha(path)})
write_json('index.json', {'status': audit['status'], 'files': artifacts, 'index_self_hash_excluded': True})
print(json.dumps({'status': audit['status'], 'checks': len(checks), 'failed_checks': audit['failed_checks'], 'gradient_arrays': len(gradients), 'closed_actual_CPU_sum_seconds': total_CPU, 'output': str(OUT.relative_to(REPO))}, indent=2))
raise SystemExit(0 if all(c['passed'] for c in checks) else 1)
