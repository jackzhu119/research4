"""Actual short C CPU test, no quantum engine, scientific ledger or checkpoint.

Deliberately small test-only budgets exercise the frozen guard helpers. This does
not claim validation of a complete production V4 root authorization/proof.
"""
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    output=HERE/'ordinary_cpu_guard_evidence.json'
    if output.exists():raise FileExistsError('Preserve ordinary probe evidence')
    source=ROOT/'scripts/run_pyscf_actual_cpu_budget_20261101.py'
    spec=importlib.util.spec_from_file_location('frozen_actual_cpu_guard_probe',source)
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    scratch=ROOT/'outputs/pyscf-pure-j-20261101-v4-ordinary-cpu-probe'
    scratch.mkdir(parents=True,exist_ok=False)
    binary=scratch/'ordinary_cpu_spin'
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    compile_result=subprocess.run(['gcc','-O2','-std=c11','-fopenmp',str(HERE/'ordinary_cpu_spin.c'),'-o',str(binary)],
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=20)
    (HERE/'ordinary_compile.log').write_bytes(compile_result.stdout)
    if compile_result.returncode:raise RuntimeError('Ordinary compile failed, no child run')
    def limits():
        resource.setrlimit(resource.RLIMIT_CPU,(60,120))
        resource.setrlimit(resource.RLIMIT_AS,(1024**3,)*2)
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    events=[];started=time.monotonic();term_at=None;kill_at=None;identity=None
    child_log=HERE/'ordinary_native_child.log'
    with child_log.open('xb') as handle:
        child=subprocess.Popen([str(binary)],stdout=handle,stderr=subprocess.STDOUT,
            start_new_session=True,preexec_fn=limits,env={**os.environ,'OMP_NUM_THREADS':'4'})
        identity=guard.read_process_identity(child.pid)
        while True:
            pid,status,usage=os.wait4(child.pid,os.WNOHANG)
            if pid:
                child.returncode=os.waitstatus_to_exitcode(status);break
            snap=guard.process_group_snapshot(child.pid)
            if child.pid not in snap['pids']:continue
            observation=guard.actual_cpu_observation(snap,identity,int(os.sysconf('SC_CLK_TCK')),
                closed_cpu=0,batch_cap=5,phase_grant=5,reserve=4)
            now=time.monotonic()
            if observation['should_stop'] and term_at is None:
                event=guard.signal_bound_group(identity,signal.SIGTERM)
                events.append({'elapsed':now-started,'action':'TERM','result':event,'actual_observation':observation})
                term_at=now
            if term_at is not None and now-term_at>=.2 and kill_at is None:
                event=guard.signal_bound_group(identity,signal.SIGKILL)
                events.append({'elapsed':now-started,'action':'KILL','result':event,'actual_observation':observation})
                kill_at=now
            if now-started>=5 and kill_at is None:
                event=guard.signal_bound_group(identity,signal.SIGKILL)
                events.append({'elapsed':now-started,'action':'EXTERNAL_WALL_BACKSTOP','result':event})
                kill_at=now
            time.sleep(.02)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    value={'status':'DONE_ORDINARY_NATIVE_CPU_GUARD_TEST_NOT_QUANTUM',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'supervisor_source_sha256':sha(source),
        'test_method':'Actual frozen actual_cpu_observation and signal_bound_group helpers with real native4 child/proc ticks',
        'test_only_limits':{'sample_seconds':.02,'phase_CPU_seconds':5,'cleanup_reserve_CPU_seconds':4,
            'software_stop_CPU_seconds':1,'TERM_to_KILL_seconds':.2,'independent_child_self_stop_CPU_seconds':3,
            'external_wall_backstop_seconds':5,'supplementary_kernel_CPU_soft_hard_seconds':[60,120]},
        'production_root_authorization_or_proof_claimed':False,'production_reserve_stays_120_seconds':True,
        'new_quantum_evaluations':0,'real_scientific_ledgers_initialized':0,'checkpoints_used':0,
        'identity':{key:identity[key] for key in ['pid','start_ticks','uid','pgrp','session']},
        'events':events,'child_exit_code':child.returncode,'actual_child_wait4_CPU_seconds':usage.ru_utime+usage.ru_stime,
        'elapsed_wall_seconds':time.monotonic()-started,
        'total_compile_and_child_CPU_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        'child_log_sha256':sha(child_log),'ignored_binary_path':str(binary),'binary_sha256':sha(binary),
        'CPU_sampling_and_signal_delivery_are_best_effort_not_mathematically_exact_bounds':True}
    output.write_text(json.dumps(value,indent=2)+'\n')
    assert term_at is not None and kill_at is not None and child.returncode==-9
    assert all(event['result']['sent'] for event in events)
    print(json.dumps({key:value[key] for key in ['status','actual_child_wait4_CPU_seconds','child_exit_code','total_compile_and_child_CPU_seconds']}))

if __name__=='__main__':main()
