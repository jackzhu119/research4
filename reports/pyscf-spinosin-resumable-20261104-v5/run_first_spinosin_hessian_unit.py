"""Root's finite early-checkpoint pilot, after a healthy same-state gradient.

Waits only on JSON, then computes one real RHF unit through frozen supervision.
It archives only closed immutable files, performs no Git operations, and does
not label a derivative contribution as a complete Hessian or frequency result.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
CONTROLLER_SHA = 'a01b6a79f457bca4b0ab90e63b45a3bb524b62fc586fab26905d6b43a2aba1d7'
WORKER_SHA = 'b40188f747c01e9efa97672af6eedfd281a9e9697d20bc7d57f125bb131fa190'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(path, expected, name):
    if sha(path) != expected:
        raise RuntimeError('Frozen source changed')
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def record(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def main():
    worker = load(ROOT/'scripts/pyscf_resumable_hessian_worker_20261104_v5.py', WORKER_SHA, 'first_unit_consumer')
    controller = load(ROOT/'scripts/run_pyscf_serial_segments_20261104_v5.py', CONTROLLER_SHA, 'first_unit_archiver')
    gradient = REPORT/'v5-spinosin-C001-restore-gradient-001/result.json'
    source = REPORT/'v5-spinosin-C001-prepare-001/result.json'
    source_value = json.loads(source.read_text())
    snapshot = Path(source_value['snapshot_manifest']['path']).parent
    snapshot_sha = source_value['snapshot_manifest']['sha256']
    ledger = REPORT/'cpu_budget.json'
    wait_deadline = time.monotonic()+3600
    while True:
        rows = [r for r in json.loads(ledger.read_text())['phases']
                if r['phase'] == 'v5-spinosin-C001-restore-gradient-001']
        if len(rows) != 1:
            raise RuntimeError('Same-state gradient phase is not unique')
        if rows[0]['status'] != 'RUNNING':
            if not worker.healthy_closed_row(rows[0]) or not gradient.exists():
                raise RuntimeError('Restored gradient did not close cleanly; no Hessian launch')
            break
        if time.monotonic() >= wait_deadline:
            raise TimeoutError('Finite wait exhausted; gradient is not declared complete')
        time.sleep(5)
    value = json.loads(gradient.read_text())
    if value['status'] != 'DONE' or not value['restore_gradient_comparison_passed']:
        raise RuntimeError('Actual restored gradient check failed')
    gate = REPORT/'resource-protocol/spinosin-root-numerical-gate.json'
    controller.verify_gate(gate, '28832b15aea8b2a0c7947d133c6be752b545babc5081bcdcd6b7f3deb547692b', WORKER_SHA, worker.STATE_SHA)
    name = 'v5-spinosin-resumable-segment-001'
    out = REPORT/name
    cache = ROOT/'outputs/pyscf-spinosin-resumable-20261104-v5/spinosin-C001-restart'
    command = [sys.executable, str(ROOT/'scripts/run_pyscf_actual_cpu_budget_20261104_v5.py'),
        '--ledger', str(ledger), '--phase', name, '--cpu-seconds', '18000', '--wall-seconds', '7200',
        '--threads', '4', '--memory-gib', '24', '--log', str(REPORT/'execution'/(name+'.log')),
        '--', sys.executable, str(ROOT/'scripts/pyscf_resumable_hessian_worker_20261104_v5.py'),
        '--phase', 'hessian-segment', '--snapshot', str(snapshot), '--snapshot-manifest-sha256', snapshot_sha,
        '--output', str(out), '--threads', '4', '--max-new-units', '1', '--rhf-cache', str(cache),
        '--gradient-check-result', str(gradient), '--gradient-check-result-sha256', sha(gradient)]
    with (REPORT/'execution'/(name+'.root-command.json')).open('x') as h:
        json.dump({'command': command, 'purpose': 'EARLY_ONE_UNIT_CHECKPOINT_PILOT'}, h, indent=2)
    with (REPORT/'execution'/(name+'.supervisor-console.log')).open('x') as h:
        p = subprocess.run(command, cwd=ROOT, stdout=h, stderr=subprocess.STDOUT, check=False)
    if p.returncode:
        raise RuntimeError('First supervised unit failed; preserve all evidence without retry')
    result = json.loads((out/'result.json').read_text())
    if result['status'] != 'CHECKPOINT_SEGMENT_COMPLETE' or result['incremental_after']['total_units_completed'] != 1:
        raise RuntimeError('Expected exactly one actual completed RHF unit, no frequencies')
    def verify():
        return worker.verify_cached_unit_producers(cache, snapshot_sha)
    closed = [r for r in json.loads(ledger.read_text())['phases'] if r['phase'] == name]
    if len(closed) != 1 or not worker.healthy_closed_row(closed[0]):
        raise RuntimeError('Unit producer did not close cleanly')
    row = REPORT/'execution'/(name+'.closed-ledger-row.json')
    controller.atomic_json(row, closed[0], immutable=True)
    seal = controller.archive_checkpoint(cache, REPORT/'first-unit-checkpoint-archive', name,
        [record(snapshot/'manifest.json'), record(gradient), record(gate), record(out/'result.json'), record(row)],
        verify_closed=verify, expected_units=1)
    controller.atomic_json(REPORT/'first-unit-pilot-result.json',
        {'status': 'DONE_EARLY_CHECKPOINT_PILOT', 'scientific_frequency_stage': 'PARTIAL',
         'completed_RHF_units': 1, 'total_RHF_units': 150, 'complete_global_Hessian': False,
         'frequencies_status': 'NOT_RUN', 'IR_intensities_status': 'NOT_COMPUTED',
         'source': record(__file__), 'producer_result': record(out/'result.json'),
         'closed_phase_actual_CPU_seconds': closed[0]['actual_cpu_seconds'], 'archive_seal': seal}, immutable=True)
    print(json.dumps({'status': 'EARLY_CHECKPOINT_ARCHIVED', 'completed_RHF_units': 1, 'archive': seal}), flush=True)


if __name__ == '__main__':
    main()
