"""Prepare fixed-quadrature and directional response tests in the remaining B budget."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'reports/parallel-cloud-20261106/worker-b/B20261106-01'
PREVIOUS = BASE / 'followup-diagnostics-recovery-v2'
WORK = BASE / 'followup-rootcause-v3'


def ref(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}


def checked(binding):
    p = Path(binding['path'])
    p = p if p.is_absolute() else ROOT / p
    actual = ref(p)
    assert actual['sha256'] == binding['sha256'], str(p)
    return actual


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    prior_manifest = json.loads((PREVIOUS / 'manifest_ROOT_AUTHORIZED.json').read_text())
    prior_state = json.loads((PREVIOUS / 'state.json').read_text())
    prior_ledger = json.loads((PREVIOUS / 'cpu_budget.json').read_text())
    assert prior_state['invocations'][-1]['status'] == 'CLOSED'
    assert len(prior_state['results']) == 56
    assert all(p['status'] == 'EXITED' for p in prior_ledger['phases'])
    assert prior_ledger['conservative_debit_cpu_seconds'] == 1000
    assert abs(prior_ledger['budget_charged_cpu_seconds'] - 40559.109664) < 1e-6
    assert datetime.now(timezone.utc) < datetime.fromisoformat(prior_manifest['deadline_utc'])
    original_state = json.loads((BASE / 'state.json').read_text())
    frequencies = {case: checked(original_state['results'][case + '-frequency'])
                   for case in ('water', 'protocatechuic_acid', 'calycosin')}
    regrid = {case: checked(prior_state['results'][job]) for case, job in (
        ('water', 'water-control-comparison'), ('protocatechuic_acid', 'acid-baseline-comparison'),
        ('calycosin', 'calycosin-baseline-comparison'))}
    original_worker = ref(ROOT / 'scripts/research4_parallel_b_quantum_20261106.py')
    worker = ROOT / 'scripts/research4_b_rootcause_worker_20261107.py'
    supervisor = ROOT / 'scripts/research4_b_rootcause_supervisor_20261107.py'
    grant = {
        'schema': 'research4-b-rootcause-user-authorization-v3',
        'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'user_instruction': '继续寻找原因以及相关的测试，后台自主运行',
        'run_id': 'B20261106-01', 'new_allocation_or_refund': False,
        'cpu_cap_seconds': 58800, 'previous_actual_closed_cpu_seconds': prior_ledger['actual_closed_cpu_seconds'],
        'permanent_conservative_debit_cpu_seconds': 1000,
        'previous_budget_charged_cpu_seconds': prior_ledger['budget_charged_cpu_seconds'],
        'remaining_cpu_seconds': prior_ledger['remaining_cpu_seconds'],
        'deadline_utc': prior_manifest['deadline_utc'],
        'scientific_question': 'Does fixed versus rebuilt numerical quadrature explain the stored analytic Hessian discrepancy, or are explicit Fock derivatives or the CPHF response inconsistent?',
        'predeclared_tests': [
            'Freeze actual center Cartesian grid coordinates and weights; verify their hashes before and after every displaced SCF/gradient.',
            'Water translation and vibrational direction finite differences at0.01/0.005/0.001/0.0005Bohr.',
            'Protocatechuic acid and Calycosin original mode1 fixed-grid differences at0.005/0.001Bohr.',
            'Single-direction CPHF equation residuals, occupied constraints and coefficient-density derivative versus fixed-grid SCF density differences.',
            'Explicit Fock derivative finite differences and a stock-direct/wrapped-direct control where supported.'],
        'decision_rules': {
            'quadrature_semantics': 'Consistent fixed-grid analytic/FD agreement and disagreement with rebuilt-grid FD supports the missing grid-response/quadrature-semantics mechanism; do not assume agreement in advance.',
            'response_solver': 'Measure independent virtual equation and occupied-constraint residuals; normal solver return alone is insufficient.',
            'explicit_derivative': 'If explicit Fock derivative differs from fixed-density finite differences, investigate H1/XC/J construction separately from iterative solve convergence.',
            'density_comparison': 'Compare AO-label coefficient density derivatives; do not describe these as complete real-space density derivatives.'},
        'method_qualifiers': ['PBE/def2-SVP, noD3 and original molecular geometries retained.',
                              'New point SCF tolerance1e-12 versus prior independent1e-10 and original1e-8.',
                              'Fixed quadrature is a diagnostic discretization; overall translation need not be invariant relative to a stationary external numerical grid.',
                              'No full molecular Hessian recomputation or frequency relabeling.'],
        'preserve': 'All previous science, negative results, source snapshots, ledgers, failed artifacts and allocation claims remain unchanged.'}
    grant['local_only_grid_snapshots'] = ['grids_coords.npy', 'grids_weights.npy', 'grid_non0tab.npy']
    grant['grid_backup_policy'] = 'Exact large quadrature arrays and AO screening masks remain local with published hashes/counts; selected density/response/gradient arrays and metadata are returned within the20MiB return limit.'
    grant_path = WORK / 'preparation/user_authorization_rootcause.json'
    save(grant_path, grant)
    jobs, request_refs, all_points = [], [], []
    common = {'schema': 'research4-b-diagnostic-request-v1', 'threads': 2, 'source_worker': original_worker}
    def add(name, request, cpu, wall, dependencies=(), center=None, minus=None, plus=None, points=()):
        p = WORK / 'requests' / (name + '.json'); save(p, {**common, **request}); request_refs.append(ref(p))
        argv = [str(ROOT / prior_manifest['python_executable']), str(worker), '--request', str(p),
                '--request-sha256', ref(p)['sha256'], '--output', '{output}', '--execute']
        for flag, job_id in (('center', center), ('minus', minus), ('plus', plus)):
            if job_id:
                argv.extend(['--' + flag + '-result', '{result:' + job_id + '}',
                             '--' + flag + '-result-sha256', '{sha256:' + job_id + '}'])
        for point in points:
            argv.extend(['--point-result', '{result:' + point + '}', '--point-result-sha256', '{sha256:' + point + '}'])
        status = {'fixed_grid_center': 'FIXED_GRID_CENTER_COMPUTED', 'fixed_grid_point': 'POINT_COMPUTED',
                  'response_audit': 'RESPONSE_AUDIT_COMPLETE', 'aggregate': 'AGGREGATE_COMPLETE'}[request['operation']]
        jobs.append({'id': name, 'argv': argv, 'cpu_seconds': cpu, 'wall_seconds': wall,
                     'depends_on': list(dependencies), 'max_attempts': 2,
                     'result_schema': 'research4-b-diagnostic-result-v1', 'success_statuses': [status]})

    def tag(direction):
        return 'mode%02d' % direction['index'] if direction['type'] == 'mode' else 'translate%d' % direction['axis']

    def batch(label, case, directions, steps, cpu, wall, audit_cpu, audit_wall):
        base = {'case': case, 'frequency_result': frequencies[case], 'grid_level': 1,
                'gradient_grid_response': False, 'scf_conv_tol': 1e-12}
        center = label + '-center'
        add(center, {**base, 'operation': 'fixed_grid_center'}, cpu, wall)
        points = [center]; by_direction = {}
        for direction in directions:
            for step in steps:
                pair = {}
                for sign in (-1, 1):
                    name = label + '-' + tag(direction) + '-h%04d' % round(step * 10000) + ('-minus' if sign < 0 else '-plus')
                    add(name, {**base, 'operation': 'fixed_grid_point', 'direction': direction,
                               'step_bohr': step, 'sign': sign}, cpu, wall, dependencies=[center], center=center)
                    points.append(name); pair[sign] = name
                by_direction[(tag(direction), step)] = pair
        add(label + '-comparison', {'operation': 'aggregate', 'expected_points': points,
                                    'regrid_comparison': regrid[case]}, 30, 180,
            dependencies=points, points=points)
        for direction in directions:
            pair = by_direction[(tag(direction), .001)]
            add(label + '-response-' + tag(direction), {**base, 'operation': 'response_audit',
                'direction': direction, 'finite_difference_step_bohr': .001,
                'cphf_tol': 1e-10, 'cphf_max_cycle': 100, 'include_fock_fd': True,
                'include_stock_direct_control': case == 'water'}, audit_cpu, audit_wall,
                dependencies=[center, pair[-1], pair[1]], center=center, minus=pair[-1], plus=pair[1])
        all_points.extend(points)

    batch('water-fixed', 'water', [{'type': 'translation', 'axis': 0}, {'type': 'mode', 'index': 1}],
          [.01, .005, .001, .0005], 30, 120, 150, 600)
    batch('acid-fixed', 'protocatechuic_acid', [{'type': 'mode', 'index': 1}], [.005, .001],
          240, 900, 1200, 1800)
    batch('calycosin-fixed', 'calycosin', [{'type': 'mode', 'index': 1}], [.005, .001],
          1200, 2400, 4000, 3600)
    add('all-fixed-comparison', {'operation': 'aggregate', 'expected_points': all_points}, 60, 240,
        dependencies=all_points, points=all_points)
    prefix = str(WORK.relative_to(BASE))
    public_arrays = ('gradient.npy', 'density.npy', 'direction.npy', 'geometry_bohr.npy', 'predicted_Hq.npy',
                     'mo_coeff.npy', 'mo_energy.npy', 'mo_occ.npy', 'H1q_AO.npy', 'S1q_AO.npy', 'H1q_MO.npy',
                     'S1q_MO.npy', 'U_MO.npy', 'induced_response_MO.npy', 'occupied_energy_response.npy',
                     'density_derivative_analytic.npy', 'density_derivative_FD.npy', 'explicit_H1_Fock_FD.npy',
                     'stock_direct_Fock_difference.npy', 'stock_direct_induced_response_difference.npy')
    return_manifest = {'schema': 'research4-b-rootcause-return-manifest-v3', 'role': 'B',
        'run_id': 'B20261106-01', 'work_directory': str(BASE.relative_to(ROOT)),
        'ledger': str((WORK / 'cpu_budget.json').relative_to(ROOT)),
        'upload': {'branch': prior_manifest['branch'], 'allowlist': [prefix + '/*.json', prefix + '/*.jsonl',
                    prefix + '/detached-*.console.log', prefix + '/preparation/**/*.json', prefix + '/requests/*.json',
                    *[prefix + '/science/**/*' + suffix for suffix in ('.json', '.jsonl', '.csv', '.log', '.txt', '.png', '.pdf')],
                    *[prefix + '/science/**/' + name for name in public_arrays],
                    prefix + '/execution/*.json', prefix + '/execution/*.log']}}
    return_path = WORK / 'return_manifest.json'; save(return_path, return_manifest)
    helper = ref(ROOT / 'scripts/research4_parallel_b_git_return_20261106.py')
    manifest = {k: prior_manifest[k] for k in ('run_id', 'branch', 'python_executable', 'cpu_cap_seconds',
                'deadline_utc', 'threads', 'memory_gib', 'rss_stop_gib', 'reserve_cpu_seconds')}
    manifest.update(schema='research4-b-rootcause-manifest-v3', work_directory=str(WORK.relative_to(ROOT)),
        ledger=str((WORK / 'cpu_budget.json').relative_to(ROOT)),
        predecessor_manifest=ref(PREVIOUS / 'manifest_ROOT_AUTHORIZED.json'),
        predecessor_ledger=ref(PREVIOUS / 'cpu_budget.json'), predecessor_state=ref(PREVIOUS / 'state.json'),
        predecessor_supervisor=ref(ROOT / 'scripts/research4_b_diagnostic_supervisor_recovery_20261107.py'),
        predecessor_allocation=ref(ROOT / '.runtime/worker-b-diagnostic-recovery-allocation.json'),
        source_refs=[ref(worker), ref(supervisor), ref(__file__), helper, *prior_manifest['source_refs']],
        input_refs=[*prior_manifest['input_refs'], *request_refs, *frequencies.values(), *regrid.values(),
                    ref(grant_path), ref(return_path), ref(WORK / 'preparation/offline_localization.json')],
        jobs=jobs, authorization_ref=ref(grant_path), same_allocation_no_budget_reset=True,
        upload={'enabled': True, 'helper': helper, 'manifest': ref(return_path)})
    target = WORK / 'manifest_ROOT_AUTHORIZED.json'; save(target, manifest)
    print(json.dumps({'manifest': ref(target), 'jobs': len(jobs), 'fixed_points_and_centers': len(all_points),
                      'first_attempt_CPU_caps_sum': sum(j['cpu_seconds'] for j in jobs),
                      'remaining_CPU': prior_ledger['remaining_cpu_seconds'], 'new_quantum_evaluations': 0}, indent=2))


if __name__ == '__main__':
    main()
