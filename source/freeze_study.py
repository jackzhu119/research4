#!/usr/bin/env python3
"""Seal experiment inputs and methods before launching real training."""
import hashlib,json,os,platform,sys
from pathlib import Path
from datetime import datetime, timezone
import extend_manifest

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    path=ROOT/'protocol.json'
    if path.exists():
        p=json.loads(path.read_text())
        for rel,digest in p['sealed_files'].items():
            assert sha(ROOT/rel)==digest, f'Frozen file changed: {rel}'
        print('Existing protocol verified; CPU ledger must never be reset.')
        return
    spec=json.loads((ROOT/'worker_spec.json').read_text())
    sealed=['study_worker.py','worker_spec.json','supervisor.py','start_background.py','extend_manifest.py','PRE_REGISTERED_DESIGN_zh.md','inputs/current1672_paired612.npz']
    p={
      'schema':'thz-mechanism-background-protocol-v1',
      'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
      'authorized_cpu_budget_seconds':90000,'cpu_budget_seconds':90000,
      'max_workers':4,'blas_threads_per_worker':1,'worker_cpu_lease_seconds':1800,
      'management_cpu_reserve_seconds':500,'poll_seconds':2,'max_attempts':3,
      'accuracy_early_stop':False,'score_adaptive_scheduling':False,
      'input_path':str(ROOT/'inputs/current1672_paired612.npz'),
      'input_sha256':sha(ROOT/'inputs/current1672_paired612.npz'),
      'outer_cv':{'n_splits':5,'shuffle':True,'seed_rule':'extend_manifest.seed_for(repeat); SHA256 frozen prefix'},
      'inner_cv':{'n_splits':5,'n_repeats':2,'must_be_training_only':True},
      'scope':'Retrospective internal re-evaluation of existing specimens; no independent external validation',
      'selection_metric':'macro_f1, then balanced_accuracy, then accuracy, then predeclared complexity/order',
      'reporting':{'complete_outer_repeats_only':True,'bootstrap_unit':'physical specimen, stratified by class; retain repeated predictions within specimen','split_variation':'descriptive; repeats are not independent specimens','chemical_mechanism':'blocked: no same-ID quantified chemistry or validated reference bands'},
      'queue_rule':'Initial paired blocks: 3 new split seeds MC20; same3 MC100; firstseed MC500 convergence. Extend with32 fixed seeds, MC100 every8th andMC500 every32nd. Method order and variants frozen; never read outer scores.',
      'stop_conditions':['authorized CPU safety budget reached','user requested stop','irrecoverable infrastructure/input failure'],
      'resume_policy':'same durable CPU ledger; PID/starttime/boot identity; conservative charge for uncertain orphan tail; no reset',
      'platform_limit':'Local detached supervisor/watchdog cannot prevent cloud suspension or environment destruction.',
      'worker_spec':spec,'sealed_files':{rel:sha(ROOT/rel) for rel in sealed},
      'python':sys.executable,'platform':platform.platform()
    }
    extend_manifest.atomic(path,p)
    (ROOT/'protocol.sha256').write_text(sha(path)+'\n')
    extend_manifest.extend(ROOT,initial=True)
    # Add a predeclared large second block. Subsequent blocks follow exactly the same frozen rule.
    extend_manifest.extend(ROOT)
    print(json.dumps({'protocol_sha256':sha(path),'budget_cpu_seconds':90000,'parallel_workers':4},ensure_ascii=False))
if __name__=='__main__':main()
