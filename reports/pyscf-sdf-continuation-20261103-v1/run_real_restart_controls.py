"""Root's finite actual water/ethanol stock-versus-restart validation commands.

This script launches quantum work only through the unchanged actual-CPU guard.
No phase or output is overwritten; a failure stops the finite sequence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
WORKER = ROOT / 'scripts/pyscf_resumable_hessian_worker_20261103.py'
WORKER_SHA = 'd23c4a3185aa4f666e22d671034728c1ca2443676a05fc8bec0587d6c8d6bf03'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('molecule', choices=('water', 'ethanol'))
    a = p.parse_args()
    if sha(WORKER) != WORKER_SHA:
        raise RuntimeError('Numerical consumer is not the frozen reviewed version')
    m = a.molecule
    snapshot = REPORT / f'validation-{m}-prepare-001/exact-mf-snapshot'
    gradient = REPORT / f'validation-{m}-restore-gradient-001/result.json'
    if json.loads(gradient.read_text())['status'] != 'DONE':
        raise RuntimeError('Actual restored gradient gate has not passed')
    stages = [('stock-hessian-reference', f'validation-{m}-stock-hessian-001')]
    # Water: six RHF units in three separate processes; ethanol: eighteen in four.
    count, units = (3, 2) if m == 'water' else (4, 5)
    stages += [('hessian-segment', f'validation-{m}-segment-{i:03d}') for i in range(1, count+1)]
    final = None
    for phase, name in stages:
        out = REPORT / name
        if out.exists():
            raise FileExistsError('Do not overwrite an earlier validation: ' + str(out))
        command = [sys.executable, str(ROOT / 'scripts/run_pyscf_actual_cpu_budget_20261101.py'),
            '--ledger', str(REPORT / 'cpu_budget.json'), '--phase', name,
            '--cpu-seconds', '3600', '--wall-seconds', '1800', '--threads', '4', '--memory-gib', '24',
            '--log', str(REPORT / 'execution' / (name+'.log')), '--', sys.executable, str(WORKER),
            '--phase', phase, '--snapshot', str(snapshot), '--snapshot-manifest-sha256', sha(snapshot/'manifest.json'),
            '--output', str(out), '--threads', '4', '--max-new-units', str(units),
            '--rhf-cache', str(ROOT/'outputs/pyscf-sdf-continuation-20261103-v1'/f'{m}-restart-validation'),
            '--gradient-check-result', str(gradient), '--gradient-check-result-sha256', sha(gradient)]
        launch = REPORT/'execution'/(name+'.root-command.json')
        with launch.open('x') as handle:
            json.dump({'command': command, 'worker_sha256': WORKER_SHA}, handle, indent=2)
            handle.write('\n')
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            raise RuntimeError(f'Supervised control failed: {name}; returncode={result.returncode}')
        r = json.loads((out/'result.json').read_text())
        rows = json.loads((REPORT/'cpu_budget.json').read_text())['phases']
        row = [x for x in rows if x['phase'] == name]
        if len(row) != 1 or row[0]['status'] != 'EXITED' or row[0]['exit_code'] != 0:
            raise RuntimeError('Control has no clean actual parent closure')
        if phase == 'stock-hessian-reference' and r['status'] != 'FREQUENCIES_COMPUTED':
            raise RuntimeError('Stock reference has not produced full frequencies')
        if phase == 'hessian-segment':
            if r['status'] not in ('CHECKPOINT_SEGMENT_COMPLETE', 'FREQUENCIES_COMPUTED'):
                raise RuntimeError('Unexpected incomplete incremental validation')
            final = r
            if r['status'] == 'FREQUENCIES_COMPUTED':
                break
    if final is None or final['status'] != 'FREQUENCIES_COMPUTED':
        raise RuntimeError('Finite restart validation ended without complete Hessian')
    print(json.dumps({'molecule': m, 'status': 'FULL_CALCULATIONS_COMPLETED_PENDING_INDEPENDENT_COMPARISON'}))


if __name__ == '__main__':
    main()
