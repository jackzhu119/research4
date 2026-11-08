#!/usr/bin/env python3
"""Continue a closed research queue with cache-aware memory monitoring.

The original scientific source, job manifest, historical outer ledger and
inner CPU ceiling are unchanged. Only the original supervisor may continue its
own ledger. Page cache triggers read-only file advice, never a stop condition.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import uuid

HERE=Path(__file__).resolve().parent
SELF=Path(__file__).resolve()
TARGET=Path('/workspace/research4-relaunch')
OLD_RUN=TARGET/'data/research/species_relaunch_20261108_v2'
SUPERVISOR=TARGET/'scripts/species_background_budget.py'
SUPERVISOR_SHA='3ca443016719c90697ed9e7f82db2a6205cdb8f07b1f78ad31db117869667874'
DEADLINE=datetime.fromisoformat('2026-11-10T02:50:37.289380+00:00')
SCHEMA='research4-cache-aware-controller-v1'
TOKEN='RESEARCH4_CACHE_AWARE_CONTINUATION_TOKEN'
GIB=1024**3
TOTAL_CAP=90000.
CPU_CLEANUP_RESERVE=64.


def need(value,message):
    if not value:raise RuntimeError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def finite(value,name):
    need(type(value) in (int,float) and math.isfinite(value) and value>=0,'Unknown/nonfinite actual cost: '+name)
    return float(value)


def checked(ref,*,under=None):
    p=Path(ref['path']).resolve()
    need(p.is_file() and (under is None or p.is_relative_to(under)) and sha(p)==ref['sha256'],
         'Pinned input missing or changed: '+str(p))
    return p


def helpers():
    need(sha(SUPERVISOR)==SUPERVISOR_SHA,'Original reviewed supervisor changed')
    spec=importlib.util.spec_from_file_location('cache_aware_original_supervisor',SUPERVISOR)
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    b.TOKEN_KEY=TOKEN
    return b


def frozen_inputs(inner):
    result={}
    for mapping in inner['input_hashes'].values():
        for name,checksum in mapping.items():
            p=Path(name).resolve()
            need(p.is_relative_to(TARGET),'Original queue input escaped its research worktree')
            need(name not in result or result[name]==checksum,'Inconsistent original input seals')
            result[name]=checksum
    need(result,'Original frozen inputs are unavailable')
    for name,checksum in result.items():need(sha(name)==checksum,'Frozen science input changed: '+name)
    return result


def matching_prior_alive(b,identity):
    need(isinstance(identity,dict),'Closed prior process identity missing')
    if not b.same_process(identity):return False
    actual=b.proc_stat(identity['pid'])
    return actual is not None and actual['state']!='Z'


def scientific_processes(b):
    """Refuse duplicate source workers; never signal processes discovered here."""
    names={'species_revision_background_classical.py','species_revision_background_neural.py',
           'summarize_species_background.py','species_relaunch_backup_20261108.py'}
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            stat=b.proc_stat(int(p.name))
            if not stat or stat['state']=='Z' or p.stat().st_uid!=os.getuid():continue
            args=(p/'cmdline').read_bytes().split(b'\0')
            if not args or not Path(os.fsdecode(args[0])).name.startswith('python'):continue
            decoded=[os.fsdecode(x) for x in args if x]
            for arg in decoded[1:]:
                source=Path(arg)
                if source.parent!=TARGET/'scripts':continue
                if source.name in names or (source.name=='species_background_budget.py' and any(x in decoded for x in ('run','resume'))) or (source.name=='species_relaunch_resources_20261108.py' and '--execute' in decoded):
                    found.append({k:stat[k] for k in ('pid','start_ticks','pgrp','session','state')});break
        except (OSError,ValueError):continue
    return found


def execution_plan(config,old_inner,old_config,*,resume):
    mode=config.get('execution_mode','resume_original')
    need(mode in ('resume_original','remaining_manifest'),'Unknown finite execution mode')
    original_ref=config.get('original_manifest_ref',config['manifest_ref'])
    original=checked(original_ref)
    need(original==OLD_RUN/'jobs.json' and original_ref['sha256']==old_config['manifest_ref']['sha256'],
         'Original frozen manifest must not change')
    manifest=checked(config['manifest_ref'])
    if mode=='remaining_manifest':
        need(manifest==HERE/'remaining_jobs.json','Exact independent remaining-job manifest required')
        old_jobs={job['id']:job for job in read(original)['jobs']}
        pending={name for name,job in old_jobs.items() if job.get('kind') in ('classical','neural') and
                 old_inner['jobs'].get(name,{}).get('status')!='completed'}
        data=read(manifest)
        need(data.get('cpu_budget_seconds')==88000,'Local remaining supervisor cap cannot expand')
        observed=set();all_ids=set()
        for job in data['jobs']:
            need(job['id'] not in all_ids,'Duplicate remaining queue job ID');all_ids.add(job['id'])
            if job.get('kind') in ('classical','neural') or job['id'] in pending:
                need(job['id'] in pending and job==old_jobs[job['id']],
                     'Remaining scientific command/inputs/outputs differ from the frozen original')
                observed.add(job['id'])
            else:
                need(job['id'] not in old_jobs,'Do not rerun original giant auxiliary jobs')
                sources=[Path(x).resolve() for x in job['command'] if x.endswith('.py')]
                need(len(sources)==1 and sources[0].is_relative_to(HERE),
                     'Only the independent compact runtime may supply new auxiliary jobs')
        need(observed==pending,'Remaining manifest must contain exactly all unfinished original science jobs')
        state_dir=Path(config['state_dir']).resolve()
        need(state_dir.is_relative_to(HERE) and state_dir!=HERE and state_dir!=Path(config['workdir']).resolve(),
             'Remaining local ledger must be distinct from source and controller state')
        if resume:
            current=read(state_dir/'state.json')
            need(current.get('status')=='paused' and current.get('active') is None,
                 'Local remaining ledger is not a clean pause')
        else:
            need(not (state_dir/'state.json').exists(),'Existing local ledger cannot be reset')
        action='resume' if resume else 'run'
    else:
        need(manifest==original,'Resume mode must use the original frozen manifest')
        state_dir=OLD_RUN/'control';action='resume';pending=set()
    return {'mode':mode,'manifest':manifest,'state_dir':state_dir,'action':action,
            'remaining_scientific_jobs':len(pending)}


def validate(config,b,*,resume=False):
    need(config.get('schema')==SCHEMA and config['controller_source_sha256']==sha(SELF),'Exact reviewed controller required')
    need(Path(config['target_root']).resolve()==TARGET and Path(config['run_root']).resolve()==OLD_RUN,'Exact original queue required')
    need(config['total_cpu_cap_seconds']==90000,'The original total CPU cap cannot change')
    end=datetime.fromisoformat(config['deadline_utc'])
    need(end==DEADLINE and datetime.now(timezone.utc)<end,'Original deadline cannot extend or has expired')
    need(checked(config['supervisor_ref'])==SUPERVISOR and config['supervisor_ref']['sha256']==SUPERVISOR_SHA,'Original supervisor source required')
    prior_outer_path=checked(config['prior_outer_ref'],under=HERE/'evidence')
    prior_inner_path=checked(config['prior_inner_ref'],under=HERE/'evidence')
    prior_config_path=checked(config['prior_config_ref'],under=HERE/'evidence')
    old_outer,old_inner,old_config=read(prior_outer_path),read(prior_inner_path),read(prior_config_path)
    need(old_outer.get('status') in ('PAUSED_RESOURCE','PAUSED_MEMORY') and old_outer.get('active') is None
         and old_outer.get('closure_confirmed') is True,'Prior outer is not cleanly closed')
    need(old_inner.get('status')=='paused' and old_inner.get('active') is None,'Prior inner is not cleanly paused')
    need(old_outer.get('config_sha256')==sha(prior_config_path),'Original config seal differs')
    need(sha(TARGET/'scripts/species_relaunch_resources_20261108.py')==old_config['wrapper_source_sha256'],
         'Historical original outer source changed')
    need(datetime.fromisoformat(old_config['deadline_utc'])==end and old_config['total_budget_cpu_seconds']==90000
         and old_config['inner_budget_cpu_seconds']==88000 and old_inner['budget_cpu_seconds']==88000,
         'Original grant/deadline/inner cap must remain unchanged')
    prior=finite(old_outer['spent_cpu_seconds'],'prior outer total')
    need(finite(config['prior_total_cpu_seconds'],'declared carry')==prior,'Actual prior total cannot be refunded or reset')
    need(finite(old_outer['inner_actual_spent_cpu_seconds'],'prior actual child CPU')>=finite(old_inner['spent_cpu_seconds'],'prior inner ledger'),
         'Authoritative prior total must retain all observed child CPU')
    need(not matching_prior_alive(b,old_outer.get('identity')) and not matching_prior_alive(b,old_inner.get('owner')),
         'An original process remains alive; no parallel continuation')
    need(sha(OLD_RUN/'resource-control/state.json')==sha(prior_outer_path),'Historical original outer ledger changed')
    plan=execution_plan(config,old_inner,old_config,resume=resume)
    inputs=frozen_inputs(old_inner)
    need(old_inner['supervisor_code_sha256']==SUPERVISOR_SHA,'Original supervisor ledger source differs')
    helper=checked(config['helper_ref'],under=HERE)
    need(helper.name=='closed_cache_advice.py','Local hash-pinned read-only cache helper required')
    work=Path(config['workdir']).resolve()
    need(work.is_relative_to(HERE) and work!=HERE and not work.is_relative_to(TARGET),'Distinct runtime workdir required')
    state_path=work/'state.json'
    need(resume==state_path.exists(),'Fresh continuation versus cumulative resume must be explicit')
    current=read(OLD_RUN/'control/state.json')
    need(current.get('status')=='paused' and current.get('active') is None,'Original inner has running, unclean or completed work; do not restart')
    need(not matching_prior_alive(b,current.get('owner')),'Current source supervisor is still alive')
    need(not scientific_processes(b),'Source scientific processes remain active or orphaned; no duplicate launch')
    if resume:
        old=read(state_path)
        need(old.get('active') is None and old.get('closure_confirmed') is True and old.get('status') in
             ('PAUSED_EXTERNAL','PAUSED_EMERGENCY_MEMORY','PAUSED_DISK','PAUSED_CPU_BUDGET','PAUSED_DEADLINE'),
             'Only this controller’s clean closed pause may resume')
        need(old.get('config_sha256')==config['_config_sha256'],'Continuation config changed')
        need(finite(old['spent_cpu_seconds'],'controller cumulative total')>=prior,'Resume cannot refund prior total')
    if not resume or plan['mode']=='remaining_manifest':
        need(sha(OLD_RUN/'control/state.json')==sha(prior_inner_path),'Original actual inner state differs from the closed carry snapshot')
    need(Path(config['python']).is_absolute() and Path(config['python']).is_file(),
         'Explicit installed Python required')
    return {'prior_total':prior,'prior_wall':finite(old_outer['wall_seconds'],'prior wall'),
            **plan,'helper':helper,'work':work,'frozen_input_count':len(inputs)}


def usage(kind):
    u=resource.getrusage(kind);return u.ru_utime+u.ru_stime


def memory_info():
    stats={}
    for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines():
        key,value=line.split();stats[key]=int(value)
    kernel=stats.get('kernel',sum(stats.get(x,0) for x in ('kernel_stack','pagetables','sock','slab','percpu','vmalloc')))
    return {'current_bytes':int(Path('/sys/fs/cgroup/memory.current').read_text()),
            'anon_bytes':stats.get('anon',0),'kernel_bytes':kernel,'file_bytes':stats.get('file',0),
            'anon_plus_kernel_bytes':stats.get('anon',0)+kernel}


def stop_reason(total,now,disk,mem):
    if total>=TOTAL_CAP-CPU_CLEANUP_RESERVE:return 'CPU_BUDGET'
    if now>=DEADLINE:return 'DEADLINE'
    if disk<GIB:return 'DISK'
    if mem['anon_plus_kernel_bytes']>=14*GIB:return 'EMERGENCY_MEMORY'
    return None


def should_advise(mem,last,now):
    return (mem['current_bytes']>=9*GIB or mem['file_bytes']>=5*GIB) and (last is None or now-last>=15)


def load_adviser(config):
    path=checked(config['helper_ref'],under=HERE)
    spec=importlib.util.spec_from_file_location('cache_aware_pinned_advice',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    need(callable(getattr(module,'advise_closed',None)),'Read-only cache advice function required')
    return module


def run(config,config_path,b,*,resume=False):
    proof=validate(config,b,resume=resume);work=proof['work'];work.mkdir(parents=True,exist_ok=True)
    state_path=work/'state.json'
    with (work/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        proof=validate(config,b,resume=resume)
        old=read(state_path) if resume else {}
        base=finite(old.get('spent_cpu_seconds',proof['prior_total']),'invocation carry')
        wall_base=finite(old.get('wall_seconds',proof['prior_wall']),'wall carry')
        child_start=usage(resource.RUSAGE_CHILDREN);began=time.monotonic();high=base
        actual=b.proc_stat(os.getpid());need(actual is not None,'Actual controller birth identity unavailable')
        identity={**actual,'uid':os.getuid(),'boot_id':b.boot_id()}
        nonce=uuid.uuid4().hex;attempt=uuid.uuid4().hex
        state={'schema':SCHEMA,'status':'STARTING','active':None,'closure_confirmed':False,'attempt':attempt,
            'config_sha256':config['_config_sha256'],'controller_identity':identity,'token':nonce,
            'prior_total_cpu_seconds':proof['prior_total'],'invocation_base_total_cpu_seconds':base,
            'authoritative_total_cpu_cap_seconds':90000,'original_inner_cap_seconds':88000,
            'deadline_utc':config['deadline_utc'],'frozen_input_count':proof['frozen_input_count'],
            'old_outer_ledger_is_historical_not_live':True,'cache_only_memory_stop_disabled':True,
            'controller_suballocation_stop_disabled':True,'actual_grant_cpu_not_refunded':True,
            'execution_mode':proof['mode'],'inner_state_dir':str(proof['state_dir']),
            'inner_local_ledger_is_not_authoritative_grant_total':proof['mode']=='remaining_manifest'}
        b.atomic_json(work/('launch-'+attempt+'.json'),{'status':'ACTUAL_CONTROLLER_STARTED_NOT_SCIENCE_COMPLETE',
            'controller_identity':identity,'config_sha256':config['_config_sha256'],
            'source_sha256':sha(SELF),'at_utc':datetime.now(timezone.utc).isoformat()})
        process=None;active=None;reason=None;stop_at=None;term=False;kill=False
        handlers={};external=[]
        for sig in (signal.SIGTERM,signal.SIGINT):
            handlers[sig]=signal.getsignal(sig);signal.signal(sig,lambda number,frame:external.append(signal.Signals(number).name))

        def members():
            rows=b.owned_members(active) if active else []
            for row in rows:
                try:need(Path('/proc',str(row['pid'])).stat().st_uid==os.getuid(),'Owned group UID changed')
                except FileNotFoundError:pass
            return rows

        def sample():
            nonlocal high
            rows=members();live=sum(x['cpu_seconds'] for x in rows)
            self_cpu=usage(resource.RUSAGE_SELF);waited=max(0.,usage(resource.RUSAGE_CHILDREN)-child_start)
            high=max(high,base+self_cpu+waited+live)
            mem=memory_info();v=os.statvfs(TARGET);disk=v.f_bavail*v.f_frsize
            state.update(spent_cpu_seconds=high,remaining_cpu_seconds=max(0.,90000-high),
                invocation_self_cpu_seconds=self_cpu,invocation_waited_child_cpu_seconds=waited,
                invocation_live_group_cpu_seconds=live,wall_seconds=wall_base+time.monotonic()-began,
                heartbeat_utc=datetime.now(timezone.utc).isoformat(),active=active,memory=mem,free_disk_bytes=disk)
            b.atomic_json(state_path,state)
            return rows,stop_reason(high,datetime.now(timezone.utc),disk,mem)

        try:
            _,initial=sample();need(initial is None,'Actual resource/grant exhausted before launch: '+str(initial))
            console=work/('inner-'+attempt+'.log')
            command=[config['python'],'-B',str(SUPERVISOR),proof['action'],'--manifest',str(proof['manifest']),
                     '--state-dir',str(proof['state_dir']),'--budget-cpu-seconds','88000']
            env=dict(os.environ);env.update(b.THREAD_ENV);env[TOKEN]=nonce
            with console.open('x') as log:
                process=subprocess.Popen(command,cwd=TARGET,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
                state['provisional_child_pid']=process.pid;b.atomic_json(state_path,state)
                child=b.proc_stat(process.pid);need(child is not None,'Original supervisor exited before actual birth observation')
                active={k:child[k] for k in ('pid','start_ticks','session','pgrp')}
                active.update(uid=os.getuid(),boot_id=b.boot_id(),token=nonce)
                state.update(status='RUNNING',active=active,actual_command=command,inner_console=str(console))
                last_advice=old.get('last_advice_monotonic') if resume and old.get('controller_identity',{}).get('boot_id')==b.boot_id() else None
                while True:
                    code=process.poll();rows,why=sample();alive=[r for r in rows if r['state']!='Z']
                    if code is not None and not alive:break
                    if external:why='EXTERNAL'
                    if sha(config_path)!=config['_config_sha256'] or sha(SELF)!=config['controller_source_sha256']:
                        why='PROVENANCE'
                    if code is not None and alive:why='ORPHAN_DESCENDANTS'
                    now=time.monotonic()
                    if why is None and should_advise(state['memory'],last_advice,now):
                        last_advice=now
                        state['last_advice_monotonic']=now
                        try:
                            adviser=load_adviser(config)
                            state['last_cache_advice']={'status':'DONE','at_utc':datetime.now(timezone.utc).isoformat(),
                                                       'result':adviser.advise_closed()}
                        except Exception as error:
                            state['last_cache_advice']={'status':'ERROR_NOT_SCIENTIFIC_FAILURE',
                                'error_type':type(error).__name__,'error':str(error),
                                'at_utc':datetime.now(timezone.utc).isoformat()}
                            # A helper that drifted is never executed. Advice
                            # failure alone does not stop valid scientific work.
                        _,why_after=sample()
                        if why_after is not None:why=why_after
                    if why is not None and reason is None:
                        reason=why;stop_at=time.monotonic();state.update(status='STOPPING',stop_reason=reason)
                    if reason:
                        if not term:b.signal_owned_groups(active,signal.SIGTERM);term=True
                        if time.monotonic()-stop_at>=2 and not kill:
                            b.signal_owned_groups(active,signal.SIGKILL);kill=True
                        need(time.monotonic()-stop_at<7,'Owned closure not verified; retain forensic state')
                    time.sleep(.25 if reason else 5)
                process.wait();sample();active=None
                inner=read(proof['state_dir']/'state.json')
                if reason:
                    status='PAUSED_'+reason if reason in ('EXTERNAL','EMERGENCY_MEMORY','DISK','CPU_BUDGET','DEADLINE') else 'BLOCKED_'+reason
                elif process.returncode==0 and inner.get('status')=='completed':status='COMPLETED'
                elif inner.get('status')=='completed_with_errors':status='COMPLETED_WITH_ERRORS'
                elif inner.get('status')=='budget_exhausted':status='INNER_BUDGET_EXHAUSTED'
                else:status='INNER_FINISHED_NOT_COMPLETE'
                state.update(status=status,active=None,closure_confirmed=True,inner_exit_code=process.returncode,
                             inner_final_status=inner.get('status'))
        except BaseException as error:
            state.update(status='BLOCKED_FORENSIC',error_type=type(error).__name__,error=str(error))
            if process is not None and active is not None:
                try:
                    b.signal_owned_groups(active,signal.SIGTERM)
                    until=time.monotonic()+2
                    while time.monotonic()<until and any(x['state']!='Z' for x in members()):
                        process.poll();sample();time.sleep(.1)
                    if any(x['state']!='Z' for x in members()):b.signal_owned_groups(active,signal.SIGKILL)
                    process.wait(timeout=3)
                    if not any(x['state']!='Z' for x in members()):
                        sample();active=None;state.update(active=None,closure_confirmed=True)
                except BaseException as close_error:state['closure_error']=type(close_error).__name__+': '+str(close_error)
            elif process is not None:
                try:process.wait(timeout=.1)
                except subprocess.TimeoutExpired:state['unknown_child_state']='ROOT_RECONCILIATION_REQUIRED_NO_SIGNAL_NO_AUTORESTART'
            else:
                state.update(active=None,closure_confirmed=True)
        finally:
            try:sample()
            except BaseException as error:state['final_sample_error']=type(error).__name__+': '+str(error)
            state['finished_or_yielded_at_utc']=datetime.now(timezone.utc).isoformat();b.atomic_json(state_path,state)
            for sig,handler in handlers.items():signal.signal(sig,handler)
        return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();need(sha(args.config)==args.config_sha256,'Runtime config changed')
    config=read(args.config);config['_config_sha256']=args.config_sha256;b=helpers()
    proof=validate(config,b,resume=args.resume)
    if not args.execute:
        print(json.dumps({'status':'READ_ONLY_PREFLIGHT_PASS_NOT_STARTED','prior_total_cpu_seconds':proof['prior_total'],
                          'total_cpu_cap_seconds':90000,'frozen_input_count':proof['frozen_input_count'],
                          'cache_only_stop_disabled':True}));return 0
    state=run(config,args.config,b,resume=args.resume)
    print(json.dumps({'status':state['status'],'spent_cpu_seconds':state['spent_cpu_seconds']}))
    return 0 if state['status']=='COMPLETED' else 1


if __name__=='__main__':raise SystemExit(main())
