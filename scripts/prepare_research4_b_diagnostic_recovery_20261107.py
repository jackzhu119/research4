"""Prepare the reviewed continuation after a platform-query subprocess failure."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'reports/parallel-cloud-20261106/worker-b/B20261106-01'
OLD = BASE / 'followup-diagnostics-v1'
WORK = BASE / 'followup-diagnostics-recovery-v2'


def ref(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
            'bytes': p.stat().st_size}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    predecessor_path = OLD / 'manifest_ROOT_AUTHORIZED.json'
    assert ref(predecessor_path)['sha256'] == 'ca1464be64de265a1adadbcb9ffcac8cb53e32af6c8d2b05478db40bce9bc006'
    previous = json.loads(predecessor_path.read_text())
    state = json.loads((OLD / 'state.json').read_text())
    ledger = json.loads((OLD / 'cpu_budget.json').read_text())
    resolution_path = WORK / 'preparation/forensic_resolution_ROOT_REVIEWED.json'
    resolution = json.loads(resolution_path.read_text())
    assert resolution['approved_for_recovery'] and resolution['processes_verified_closed']
    assert resolution['actual_wait4_cpu_seconds'] is None
    assert resolution['conservative_debit_cpu_seconds'] == 1000
    assert resolution['predecessor_ledger_sha256'] == ref(OLD / 'cpu_budget.json')['sha256']
    assert resolution['predecessor_state_sha256'] == ref(OLD / 'state.json')['sha256']
    assert len(state['results']) == 34
    assert all(state['jobs'][name]['status'] == 'DONE_SEALED' for name in state['results'])
    worker = ROOT / 'scripts/research4_b_diagnostic_worker_recovery_20261107.py'
    supervisor = ROOT / 'scripts/research4_b_diagnostic_supervisor_recovery_20261107.py'
    jobs = []
    for old_job in previous['jobs']:
        if old_job['id'] in state['results']:
            continue
        job = json.loads(json.dumps(old_job))
        job['argv'][1] = str(worker)
        previous_attempts = sum(p['job_id'] == old_job['id'] for p in ledger['phases'])
        job['max_attempts'] = old_job.get('max_attempts', 1) - previous_attempts
        assert job['max_attempts'] >= 1
        jobs.append(job)
    assert len(jobs) == 22
    helper = ROOT / 'scripts/research4_parallel_b_git_return_20261106.py'
    patterns = []
    for work in (OLD, WORK):
        prefix = str(work.relative_to(BASE))
        patterns.extend([prefix + '/*.json', prefix + '/detached-*.console.log',
                         prefix + '/preparation/**/*.json', prefix + '/preparation/**/*.txt',
                         prefix + '/preparation/**/*.log', prefix + '/requests/*.json',
                         *[prefix + '/science/**/*' + suffix for suffix in
                           ('.json', '.jsonl', '.csv', '.npy', '.xyz', '.log', '.txt', '.md', '.png', '.pdf')],
                         prefix + '/execution/*.json', prefix + '/execution/*.log'])
    return_manifest = {
        'schema': 'research4-b-diagnostic-recovery-return-manifest-v2', 'role': 'B',
        'run_id': 'B20261106-01', 'work_directory': str(BASE.relative_to(ROOT)),
        'ledger': str((WORK / 'cpu_budget.json').relative_to(ROOT)),
        'upload': {'branch': previous['branch'], 'allowlist': patterns}}
    return_path = WORK / 'return_manifest.json'
    save(return_path, return_manifest)
    grant = {
        'schema': 'research4-b-diagnostic-recovery-authorization-v2',
        'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'user_authorization': '继续监督、处理并在必要时修复用户新授权的Research4 Worker B深度异常诊断后台队列；修复常规问题不用逐次确认。',
        'reason': 'The strict scientific-process guard interrupted h5py platform processor discovery before quantum entry; preserve the frozen failed attempt and use a nonforking processor-information query.',
        'scope': 'Resume only the22 unfinished jobs; import34 independently verified successful seals.',
        'actual_predecessor_closed_cpu_seconds': ledger['consumed_cpu_seconds'],
        'conservative_failed_phase_budget_debit_cpu_seconds': 1000,
        'conservative_debit_is_actual_measured_cpu': False,
        'original_aggregate_cpu_cap_seconds': 58800,
        'initial_recovery_budget_charged_cpu_seconds': ledger['consumed_cpu_seconds'] + 1000,
        'remaining_budget_cpu_seconds': 58800 - ledger['consumed_cpu_seconds'] - 1000,
        'deadline_utc': previous['deadline_utc'],
        'forensic_resolution': ref(resolution_path),
        'preservation': 'No old source, state, ledger, manifest, seal, ownership marker or failed artifact is rewritten.',
        'scientific_method_change': False,
        'allowed_runtime_fix': 'After live resource-proof validation, cache actual processor information from os.uname or proc CPU information without a subprocess.',
        'new_runtime_accounting_fix': 'Persist actual wait4 evidence before making process-tree qualification decisions.',
        'no_new_allocation_or_refund': True}
    grant_path = WORK / 'preparation/user_authorized_recovery.json'
    save(grant_path, grant)
    manifest = {k: previous[k] for k in ('run_id', 'branch', 'python_executable', 'cpu_cap_seconds',
                 'deadline_utc', 'threads', 'memory_gib', 'rss_stop_gib', 'reserve_cpu_seconds',
                 'carry_ledger', 'carry_state')}
    manifest.update(
        schema='research4-b-diagnostic-recovery-manifest-v2',
        work_directory=str(WORK.relative_to(ROOT)),
        ledger=str((WORK / 'cpu_budget.json').relative_to(ROOT)),
        predecessor_manifest=ref(predecessor_path),
        predecessor_ledger=ref(OLD / 'cpu_budget.json'),
        predecessor_state=ref(OLD / 'state.json'),
        predecessor_supervisor=ref(ROOT / 'scripts/research4_b_diagnostic_supervisor_20261106.py'),
        predecessor_allocation=ref(ROOT / '.runtime/worker-b-diagnostic-allocation.json'),
        forensic_resolution=ref(resolution_path),
        inherited_results=state['results'], jobs=jobs,
        source_refs=[ref(worker), ref(supervisor), ref(__file__), ref(helper),
                     *previous['source_refs']],
        input_refs=[*previous['input_refs'], ref(resolution_path), ref(grant_path), ref(return_path)],
        authorization_ref=ref(grant_path), same_allocation_no_budget_reset=True,
        upload={'enabled': True, 'helper': ref(helper), 'manifest': ref(return_path)})
    target = WORK / 'manifest_ROOT_AUTHORIZED.json'
    save(target, manifest)
    print(json.dumps({'manifest': ref(target), 'inherited_results': 34, 'pending_jobs': len(jobs),
                      'pending_first_attempt_CPU_admission': sum(j['cpu_seconds'] for j in jobs),
                      'remaining_CPU_after_conservative_debit': grant['remaining_budget_cpu_seconds'],
                      'new_quantum_evaluations': 0}, indent=2))


if __name__ == '__main__':
    main()
