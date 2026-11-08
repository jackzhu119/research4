"""Root-started finite numerical controls; never runs the Spinosin target."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import subprocess

ROOT = Path('/workspace/research4')
BASE = ROOT / 'reports/pyscf-spinosin-96h-continuation-20261105-v8'
OUT = BASE / 'controls'
LEDGER = BASE / 'cpu_budget.json'
GUARD = ROOT / 'scripts/run_pyscf_continuation_budget_20261105_v8.py'
WORKER = ROOT / 'scripts/pyscf_staged_consumer_20261105_v8.py'
ANALYZER = ROOT / 'scripts/analyze_pyscf_staged_modes_20261105_v8.py'
PYTHON = ROOT / '.runtime/bin/python'
PINS = [(GUARD, '7bc773076ecc5e0f90fb58b81bf0641948dd09c14b2816138a0469f16e511c65'),
        (WORKER, '3a107671d9c8a5a47c25f631fd09ed70148dc82b529448d9bafc67179801a585'),
        (ANALYZER, 'da2a53d85015e0ced6445e7766e18ae4fbe58e098e0eb077719759be7b18cab6')]

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def ref(p):
    return {'path': str(Path(p).resolve()), 'sha256': sha(p)}

def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')

def checked_command(command, console):
    for p, expected in PINS:
        assert sha(p) == expected, f'Frozen source changed: {p}'
    with console.open('x') as log:
        code = subprocess.call(command, cwd=ROOT, env=ENV, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    assert code == 0, f'Control failed; preserve outputs, no automatic retry: {console}'

def parent_cpu_timeout(signum, frame):
    raise TimeoutError('Finite root-only control-orchestration CPU envelope')
signal.signal(signal.SIGPROF, parent_cpu_timeout)
signal.setitimer(signal.ITIMER_PROF, 120)
signal.alarm(7200)
ENV = dict(os.environ, PYTHONPATH=str(ROOT / '.runtime/quantum-pilot-site') + os.pathsep + str(ROOT / '.runtime/quantum-upgrade-site'),
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
spec = importlib.util.spec_from_file_location('root_controls_v8_guard', GUARD)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
assert not any(p['status'] == 'RUNNING' for p in read(LEDGER)['phases'])
inputs = read(ROOT / 'reports/pyscf-v8-native-consumer-preparation-20261105-v1/ACTUAL_LEGACY_CONTROL_INPUTS_AND_NEW_NOT_RUN_INTERFACE.json')
completed = []
assert [c['atoms'] for c in inputs['controls']] == [3, 9]
for case in inputs['controls']:
    short = 'water' if case['atoms'] == 3 else 'ethanol'
    common = ['--execute', '--control-validation', '--threads', '2', '--snapshot', case['snapshot'],
              '--snapshot-manifest-sha256', case['snapshot_manifest_sha256'],
              '--gradient-check-result', case['gradient_check_result'],
              '--gradient-check-result-sha256', case['gradient_check_result_sha256'],
              '--partial-result', case['legacy_A']['path'], '--partial-result-sha256', case['legacy_A']['sha256'],
              '--h1-result', case['legacy_H1']['path'], '--h1-result-sha256', case['legacy_H1']['sha256']]
    responses = []
    for stage, block in [('response-block', i) for i in case['response_block_indices']] + [('global-assemble', None)]:
        name = f'v8-control-{short}-response-{block:03d}-003' if block is not None else f'v8-control-{short}-global-003'
        reuse_closed_water = short == 'water' and block == 0
        if reuse_closed_water:
            name = 'v8-control-water-response-000-002'
        output = OUT / name
        payload = [str(PYTHON), str(WORKER), *common, '--stage', stage, '--output', str(output)]
        if block is not None:
            payload += ['--block-index', str(block)]
        else:
            index = OUT / f'v8-control-{short}-response-index-003.json'
            write(index, {'schema': 'research4-v8-closed-response-results-index-v1', 'results': responses})
            payload += ['--response-index', str(index), '--response-index-sha256', sha(index)]
        command = [str(PYTHON), str(GUARD), 'run', '--ledger', str(LEDGER), '--phase', name,
                   '--cpu-seconds', '600', '--wall-seconds', '900', '--threads', '2', '--memory-gib', '24',
                   '--log', str(OUT / (name + '.log')), '--', *payload]
        if not reuse_closed_water:
            write(OUT / (name + '-root-command.json'), {'argv': command, 'status_before_launch': 'NOT_RUN'})
            checked_command(command, OUT / (name + '-guard-console.log'))
        result_path = output / 'result.json'
        value = read(result_path)
        status = 'STAGE_COMPLETE' if block is not None else 'FREQUENCIES_COMPUTED'
        assert value['status'] == status
        evidence = {k: value[k] for k in ('source_worker', 'resource_supervision', 'snapshot_manifest_sha256', 'stage')}
        evidence['output_directory'] = str(output)
        guard.verify_closed_phase(evidence, expected_worker_path=WORKER, expected_worker_sha256=PINS[1][1],
                                  expected_result_schema='research4-staged-full-rks-hessian-result-v8',
                                  allowed_result_statuses=(status,), expected_output=output,
                                  expected_snapshot_sha256=case['snapshot_manifest_sha256'])
        if block is not None:
            responses.append(ref(result_path))
        print(json.dumps({'phase': name, 'status': status, 'actual_result': ref(result_path)}), flush=True)
    analysis = OUT / f'v8-control-{short}-independent-modes-003'
    command = [str(PYTHON), str(ANALYZER), '--control-validation', '--result', str(result_path),
               '--result-sha256', sha(result_path), '--output', str(analysis), '--cpu-seconds', '300', '--wall-seconds', '900']
    checked_command(command, OUT / f'v8-control-{short}-analysis-console.log')
    checks = read(analysis / 'checks.json')
    assert checks['normal_modes'] == 3 * case['atoms'] - 6 and checks['normal_mode_count_complete'] is True
    assert checks['all_frequencies_positive'] is True and checks['IR_intensities_status'] == 'NOT_COMPUTED'
    completed.append({'case': case['case'], 'global_analysis_index': ref(analysis / 'artifact_index.json')})
    print(json.dumps({'control': short, 'actual_audit': checks['status'], 'modes': checks['normal_modes']}), flush=True)
write(OUT / 'root-actual-control-results-003.json', {'status': 'ACTUALLY_RUN_CONTROLS_AND_ANALYSES_COMPLETE',
      'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'controls': completed,
      'IR_intensities': 'NOT_COMPUTED', 'new_Spinosin_target_evaluations': 0})
signal.alarm(0)
