"""Prepare the user-authorized, finite B diagnostic continuation; no quantum work."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'reports/parallel-cloud-20261106/worker-b/B20261106-01'
WORK = ORIGINAL / 'followup-diagnostics-v1'
SCHEMA = 'research4-b-diagnostic-result-v1'
DEADLINE = '2026-11-09T08:25:17.730193+00:00'


def ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}


def checked_ref(record):
    path = Path(record['path'])
    if not path.is_absolute():
        path = ROOT / path
    assert path.resolve().is_relative_to(ROOT) and not path.is_symlink()
    actual = ref(path)
    assert actual['sha256'] == record['sha256'], str(path)
    if 'bytes' in record:
        assert actual['bytes'] == record['bytes'], str(path)
    return actual


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    ledger = json.loads((ORIGINAL / 'cpu_budget_B.json').read_text())
    state = json.loads((ORIGINAL / 'state.json').read_text())
    assert ledger['cpu_cap_seconds'] == 58800
    assert all(p['status'] == 'EXITED' for p in ledger['phases'])
    assert all(v['status'] == 'DONE_SEALED' for v in state['jobs'].values())
    assert all(i['status'] == 'CLOSED' for i in state['invocations'])
    assert abs(ledger['consumed_cpu_seconds'] - 23291.367197) < 1e-8
    assert datetime.now(timezone.utc) < datetime.fromisoformat(DEADLINE)
    worker = ROOT / 'scripts/research4_b_diagnostic_worker_20261106.py'
    supervisor = ROOT / 'scripts/research4_b_diagnostic_supervisor_20261106.py'
    original_worker = ref(ROOT / 'scripts/research4_parallel_b_quantum_20261106.py')
    assert original_worker['sha256'] == 'a95d7cf972ba2b95da3f14a44bcfdf094d5ceb1db23bad2de559a123ee8bc9df'
    for job, record in state['results'].items():
        checked_ref(record)
        sealed = state['jobs'][job]['seal']
        assert sealed['result'] == record
        row = next(p for p in ledger['phases'] if p['phase'] == state['jobs'][job]['phase'])
        assert row['result_sha256'] == record['sha256'] and row['exit_code'] == 0
        for item in sealed['files']:
            checked_ref({'path': str(Path(row['output_directory']) / item['relative_path']),
                         'sha256': item['sha256'], 'bytes': item['size_bytes']})
    frequencies = {case: checked_ref(state['results'][case + '-frequency'])
                   for case in ('water', 'protocatechuic_acid', 'calycosin')}
    authorization = {
        'schema': 'research4-b-diagnostic-user-authorization-v1',
        'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'user_instruction': '对于结果进行处理，探究一些深度问题和反常问题。设置后台自动执行运行不要中断',
        'scope': 'Analyze and independently test the completed Worker B anomalous Hessians; bounded diagnostic continuation of the same allocation.',
        'run_id': 'B20261106-01', 'cpu_cap_seconds': 58800,
        'initial_consumed_cpu_seconds': ledger['consumed_cpu_seconds'],
        'available_cpu_seconds_before_followup': ledger['remaining_cpu_seconds'],
        'deadline_utc': DEADLINE, 'new_allocation_or_budget_reset': False,
        'same_result_branch': 'research/parallel-worker-b-20261106-B20261106-01',
        'independent_question': 'Do the saved imaginary modes represent actual local potential-energy curvature or numerical derivative/response inconsistency?',
        'hypotheses': ['True negative potential curvature', 'Incomplete or inaccurate electronic response in analytic Hessian',
                       'Quadrature/grid-response sensitivity', 'Postprocessing or rigid-body invariance error'],
        'predeclared_diagnostics': [
            'Independent array, atom-order, rigid-body and mode-localization audit.',
            'Double-step central differences of full fresh gradients and energies along saved Cartesian-normalized modes.',
            'Water vibration and whole-molecule translation software controls.',
            'Grid-level and gradient-grid-response sensitivities on the same original geometries.'],
        'decision_rules': {
            'derivative_consistency': 'Compare analytic Hq and qTHq with central full-gradient derivative and energy curvature at h=0.005 and 0.01 Bohr.',
            'followup_flag': 'Opposite signs or >10% relative curvature difference trigger further investigation, not an automatic bug declaration.',
            'true_negative_curvature': 'Requires reproducible negative finite-gradient and energy curvatures across both step sizes; this does not certify a full stationary minimum.',
            'quadrature_limitation': 'Every displaced geometry rebuilds atom-centered grids. gradient_grid_response=False is not the exact derivative of a common fixed quadrature functional; disagreement alone does not identify a Hessian bug.',
            'electronic_tolerance': 'Followup SCF tolerance=1e-10; original SCF tolerance=1e-8. This is a recorded numerical sensitivity, not a relabeling of the historical result.'},
        'preservation': 'Original outputs, science source, manifests, costs and checkpoints remain unchanged.',
        'not_claimed': ['Experimental chemical peak assignment', 'IR intensities', 'Global minimum',
                        'New classifier performance', 'New independent plant or batch data']}
    grant = WORK / 'preparation/user_authorization_followup.json'
    save(grant, authorization)
    requests = []
    jobs = []
    common = {'schema': 'research4-b-diagnostic-request-v1', 'threads': 2,
              'source_worker': original_worker}

    def job(name, request, cpu, wall, dependencies=(), point_ids=(), operation=None):
        request_path = WORK / 'requests' / (name + '.json')
        save(request_path, {**common, **request})
        requests.append(ref(request_path))
        argv = [str(ROOT / '.runtime/worker-b-20261106/venv/bin/python'), str(worker),
                '--request', str(request_path), '--request-sha256', ref(request_path)['sha256'],
                '--output', '{output}', '--execute']
        for point_id in point_ids:
            argv.extend(['--point-result', '{result:' + point_id + '}',
                         '--point-result-sha256', '{sha256:' + point_id + '}'])
        status = {'point': 'POINT_COMPUTED', 'postprocess': 'POSTPROCESS_COMPLETE',
                  'aggregate': 'AGGREGATE_COMPLETE'}[operation or request['operation']]
        jobs.append({'id': name, 'argv': argv, 'cpu_seconds': cpu, 'wall_seconds': wall,
                     'depends_on': list(dependencies), 'max_attempts': 2,
                     'result_schema': SCHEMA, 'success_statuses': [status]})

    job('original-array-audit', {'operation': 'postprocess', 'frequency_results': frequencies}, 60, 180)
    all_points = []

    def batch(label, case, directions, steps, cpu, wall, grid=1, response=False):
        ids = []
        center_id = label + '-center'
        base = {'operation': 'point', 'case': case, 'frequency_result': frequencies[case],
                'grid_level': grid, 'gradient_grid_response': response, 'scf_conv_tol': 1e-10}
        job(center_id, {**base, 'direction': {'type': 'mode', 'index': 1},
                        'step_bohr': 0, 'sign': 0}, cpu, wall)
        ids.append(center_id)
        for direction in directions:
            tag = ('mode%02d' % direction['index']) if direction['type'] == 'mode' else ('translate%d' % direction['axis'])
            for step in steps:
                for sign in (-1, 1):
                    name = label + '-' + tag + '-h' + ('%04d' % round(step * 10000)) + ('-minus' if sign < 0 else '-plus')
                    job(name, {**base, 'direction': direction, 'step_bohr': step, 'sign': sign}, cpu, wall)
                    ids.append(name)
        all_points.extend(ids)
        job(label + '-comparison', {'operation': 'aggregate', 'expected_points': ids}, 30, 180,
            dependencies=ids, point_ids=ids)

    batch('water-control', 'water', [{'type': 'mode', 'index': 1}, {'type': 'translation', 'axis': 0}],
          [.005, .01], 30, 120)
    batch('acid-baseline', 'protocatechuic_acid', [{'type': 'mode', 'index': x} for x in (1, 4, 5)],
          [.005, .01], 180, 600)
    batch('calycosin-baseline', 'calycosin', [{'type': 'mode', 'index': x} for x in (1, 7, 8)],
          [.005, .01], 1000, 1800)
    batch('acid-grid3', 'protocatechuic_acid', [{'type': 'mode', 'index': 1}],
          [.005, .01], 400, 900, grid=3)
    batch('acid-gridresponse', 'protocatechuic_acid', [{'type': 'mode', 'index': 1}],
          [.005, .01], 240, 600, response=True)
    batch('calycosin-grid3', 'calycosin', [{'type': 'mode', 'index': 1}],
          [.01], 1600, 2400, grid=3)
    job('all-point-comparison', {'operation': 'aggregate', 'expected_points': all_points},
        60, 180, dependencies=all_points, point_ids=all_points)

    prefix = str(WORK.relative_to(ORIGINAL))
    return_manifest = {
        'schema': 'research4-b-diagnostic-return-manifest-v1', 'role': 'B',
        'run_id': 'B20261106-01', 'work_directory': str(ORIGINAL.relative_to(ROOT)),
        'ledger': str((WORK / 'cpu_budget.json').relative_to(ROOT)),
        'upload': {'branch': 'research/parallel-worker-b-20261106-B20261106-01',
                   'allowlist': [prefix + '/*.json', prefix + '/detached-*.console.log',
                                 prefix + '/preparation/*.json', prefix + '/requests/*.json',
                                 *[prefix + '/science/**/*' + suffix for suffix in
                                   ('.json', '.jsonl', '.csv', '.npy', '.xyz', '.log', '.md', '.png', '.pdf')],
                                 prefix + '/execution/*.json', prefix + '/execution/*.log']}}
    return_path = WORK / 'return_manifest.json'
    save(return_path, return_manifest)
    return_helper = ref(ROOT / 'scripts/research4_parallel_b_git_return_20261106.py')
    inputs = [ref(grant), ref(ROOT / 'data/research/parallel_cloud_20261106/worker_b/cases.json'),
              ref(WORK / 'preparation/independent_hessian_audit.json'), *requests,
              *frequencies.values(), ref(return_path)]
    for case, frequency_ref in frequencies.items():
        result = json.loads(Path(frequency_ref['path']).read_text())
        for array_ref in result['arrays'].values():
            inputs.append(checked_ref(array_ref))
        inputs.append(checked_ref(result['optimized_geometry']))
        assert result['source_worker']['sha256'] == original_worker['sha256']
    unique = {r['path']: r for r in inputs}
    manifest = {'schema': 'research4-b-diagnostic-manifest-v1', 'run_id': 'B20261106-01',
                'branch': 'research/parallel-worker-b-20261106-B20261106-01',
                'work_directory': str(WORK.relative_to(ROOT)),
                'ledger': str((WORK / 'cpu_budget.json').relative_to(ROOT)),
                'python_executable': '.runtime/worker-b-20261106/venv/bin/python',
                'cpu_cap_seconds': 58800, 'deadline_utc': DEADLINE, 'threads': 2,
                'memory_gib': 12, 'rss_stop_gib': 8, 'reserve_cpu_seconds': 500,
                'carry_ledger': ref(ORIGINAL / 'cpu_budget_B.json'),
                'carry_state': ref(ORIGINAL / 'state.json'),
                'source_refs': [ref(worker), ref(supervisor), original_worker, ref(__file__), return_helper],
                'input_refs': list(unique.values()), 'jobs': jobs,
                'authorization_ref': ref(grant), 'same_allocation_no_budget_reset': True,
                'upload': {'enabled': True, 'helper': return_helper, 'manifest': ref(return_path)}}
    path = WORK / 'manifest_ROOT_AUTHORIZED.json'
    save(path, manifest)
    print(json.dumps({'manifest': ref(path), 'jobs': len(jobs), 'point_jobs': len(all_points),
                      'sum_first_attempt_CPU_caps': sum(j['cpu_seconds'] for j in jobs),
                      'remaining_original_CPU_seconds': ledger['remaining_cpu_seconds'],
                      'new_quantum_evaluations': 0}, indent=2))


if __name__ == '__main__':
    main()
