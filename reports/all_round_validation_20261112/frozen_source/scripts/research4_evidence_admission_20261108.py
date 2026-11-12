#!/usr/bin/env python3
"""Resource admission for one new science job, without signalling the old queue.

Waiting, real attempts and resource interruptions are all children of the new
cumulative CPU supervisor. A missing old PID is not interpreted as completion.
Only this wrapper's newly minted child nonce can be cleaned up.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
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
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'data/research/evidence_validation_20261108_v3'
SUPERVISOR = ROOT/'scripts/species_background_budget.py'
SUPERVISOR_SHA = '3ca443016719c90697ed9e7f82db2a6205cdb8f07b1f78ad31db117869667874'
OLD_STATE = Path('/workspace/research4-relaunch/data/research/species_relaunch_20261108_v2/control/state.json')
SCHEMA = 'research4-evidence-admission-v1'
TOKEN = 'RESEARCH4_EVIDENCE_CHILD_TOKEN'
GIB = 1024**3


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def helper():
    need(sha(SUPERVISOR) == SUPERVISOR_SHA, 'Reviewed supervisor helper changed')
    spec = importlib.util.spec_from_file_location('evidence_admission_helper', SUPERVISOR)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.TOKEN_KEY = TOKEN
    return m


def usage(kind):
    u = resource.getrusage(kind)
    return u.ru_utime + u.ru_stime


def limits(config):
    need(config.get('schema') == SCHEMA and config.get('run_id') == 'evidence-validation-20261108-v3',
         'Exact new admission run required')
    observed = datetime.fromisoformat(config['authorization_observed_utc'])
    end = datetime.fromisoformat(config['deadline_utc'])
    need(observed.tzinfo is not None and end.tzinfo is not None and observed < end <= observed+timedelta(hours=48),
         'At most48h deadline from new grant observation')
    config = dict(config)
    defaults = {'start_cgroup_bytes':9*GIB, 'stop_cgroup_bytes':11*GIB,
                'own_rss_stop_bytes':int(1.5*GIB), 'minimum_disk_bytes':2*GIB,
                'poll_seconds':5., 'wait_seconds':10., 'max_attempts':4,
                'old_state_path':str(OLD_STATE)}
    for k,v in defaults.items():config.setdefault(k,v)
    need(0 < config['start_cgroup_bytes'] <= 9*GIB and
         config['start_cgroup_bytes'] < config['stop_cgroup_bytes'] <= 11*GIB,
         'Admission memory safety limits cannot expand')
    need(0 < config['own_rss_stop_bytes'] <= int(1.5*GIB) and config['minimum_disk_bytes'] >= 2*GIB,
         'Finite owned RSS and disk safety floor required')
    need(1 <= config['poll_seconds'] <= 5 and 5 <= config['wait_seconds'] <= 10,
         'Bounded monitor and wait intervals required')
    need(type(config['max_attempts']) is int and 1 <= config['max_attempts'] <= 4,
         'At most4 independently logged resource attempts')
    need(Path(config['old_state_path']).resolve() == OLD_STATE, 'Only the designated old queue may be observed')
    return config,end


def old_queue_reason(path, b):
    """Observe only; absence/staleness cannot authorise taking over old work."""
    try:old = read(path)
    except (OSError,ValueError):return 'old_queue_state_unavailable'
    active = old.get('active')
    if old.get('status') == 'running':
        owner = old.get('owner')
        if not owner or not b.same_process(owner):return 'old_queue_identity_unverified'
        # A real owner can spend minutes checking sealed historical jobs with
        # no active child. The separate memory admission guard still applies.
        if not active:return None
        job = active.get('job_id','')
        if 'snapshot' in job or 'backup' in job or 'summary' in job:
            return 'old_queue_heavy_summary_or_backup'
        if not b.same_process(active):return 'old_worker_identity_unverified'
        return None
    if active:return 'old_queue_unclean_active_state'
    if old.get('status') in ('completed','completed_with_errors','budget_exhausted','paused','blocked_provenance'):
        return None
    return 'old_queue_unknown_status'


def admission_reason(config, b, *, memory, free_disk):
    if memory >= config['start_cgroup_bytes']:return 'global_memory_admission'
    if free_disk < config['minimum_disk_bytes']:return 'disk_admission'
    return old_queue_reason(config['old_state_path'],b)


def memory_current():
    return int(Path('/sys/fs/cgroup/memory.current').read_text())


def free_disk():
    s = os.statvfs(ROOT)
    return s.f_bavail*s.f_frsize


def load_closed_cache_advice(ref):
    path=Path(ref['path']).resolve()
    need(path==ROOT/'scripts/research4_closed_cache_advice_20261108.py' and
         path.is_file() and sha(path)==ref['sha256'], 'Closed-cache helper path/hash changed')
    spec=importlib.util.spec_from_file_location('research4_pinned_closed_cache_advice',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    need(callable(getattr(module,'advise_closed',None)), 'Pinned closed-cache advice function required')
    return module


def maybe_advise_closed(root, ref, b):
    """At most once60s across all new jobs; only the new runtime clock is written."""
    if ref is None:return None
    module=load_closed_cache_advice(ref)
    folder=Path(root)/'cache-advice';folder.mkdir(parents=True,exist_ok=True)
    clock=folder/'clock.json'
    with (folder/'clock.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return None
        now=time.monotonic();boot=b.boot_id()
        if clock.exists():
            previous=read(clock)
            if previous.get('boot_id')==boot:
                last=float(previous['started_monotonic'])
                need(math.isfinite(last) and 0<=last<=now, 'Unknown closed-cache advice clock')
                if now-last<60:return None
        record={'status':'STARTED','boot_id':boot,'started_monotonic':now,
                'started_at_utc':datetime.now(timezone.utc).isoformat(),
                'helper_ref':ref,'new_admission_pid':os.getpid(),
                'old_queue_bytes_or_ledger_modified':False,'signals_sent':False}
        b.atomic_json(clock,record)
        result=module.advise_closed()
        record.update(status='DONE',result=result,finished_at_utc=datetime.now(timezone.utc).isoformat())
        b.atomic_json(clock,record)
        return record


def owned_rows(b, identity):
    rows = b.owned_members(identity)
    for row in rows:
        try:need(Path('/proc',str(row['pid'])).stat().st_uid == os.getuid(), 'Owned process UID changed')
        except FileNotFoundError:pass
    return rows


def owned_rss(rows):
    total = 0
    for row in rows:
        try:
            for line in Path('/proc',str(row['pid']),'status').read_text().splitlines():
                if line.startswith('VmRSS:'):total+=int(line.split()[1])*1024
        except FileNotFoundError:pass
    return total


def close_owned(b, identity, process):
    """No global signals and no adoption of any historical process identity."""
    b.signal_owned_groups(identity,signal.SIGTERM)
    until=time.monotonic()+1.
    while time.monotonic()<until:
        process.poll()
        if not any(r['state']!='Z' for r in owned_rows(b,identity)):break
        time.sleep(.1)
    if any(r['state']!='Z' for r in owned_rows(b,identity)):
        b.signal_owned_groups(identity,signal.SIGKILL)
    process.wait(timeout=3)
    need(not any(r['state']!='Z' for r in owned_rows(b,identity)), 'Owned child closure unknown; no automatic restart')


def science_output(command):
    need(command and Path(command[0]).is_absolute() and Path(command[0]).is_file(), 'Installed absolute Python required')
    need(Path(command[0]).name.startswith('python'), 'Only a Python science argv is permitted')
    scripts=[Path(x).resolve() for x in command[1:] if x.endswith('.py')]
    need(len(scripts)==1 and scripts[0].is_relative_to(ROOT/'scripts') and scripts[0].is_file(),
         'One local science source script required')
    need(command.count('--output')==1 and command.index('--output')+1<len(command), 'One explicit scientific output required')
    output=Path(command[command.index('--output')+1]).resolve()
    need(output.is_relative_to(BASE/'runs') and output!=BASE/'runs', 'Science output must be under new run root')
    return scripts[0],output


def validate_complete(output):
    done=read(output/'complete.json')
    need(done.get('status') in ('complete','DONE'), 'Science result is not actually complete')
    artifacts=done.get('artifacts_sha256',done.get('artifacts'))
    need(isinstance(artifacts,dict) and artifacts, 'Scientific artifact hashes required')
    for name,expected in artifacts.items():
        p=(output/name).resolve()
        need(p.is_relative_to(output) and p.is_file() and sha(p)==expected, 'Scientific artifact seal mismatch')
    return {'complete_sha256':sha(output/'complete.json'),'artifact_count':len(artifacts)}


def run(root, output, command, b):
    root=Path(root).resolve();output=Path(output).resolve()
    need(root==BASE and output.is_relative_to(BASE/'admission'), 'Distinct new admission root required')
    config_path=root/'admission_config.json';source_sha=sha(__file__);config_sha=sha(config_path)
    config,end=limits(read(config_path));script,scientific_output=science_output(command)
    cache_ref=config.get('closed_cache_advice_ref')
    if cache_ref is not None:load_closed_cache_advice(cache_ref)
    refs=config.get('approved_worker_refs')
    if refs is not None:
        matches=[r for r in refs if Path(r['path']).resolve()==script]
        need(len(matches)==1 and sha(script)==matches[0]['sha256'],'Unapproved or changed scientific worker')
    output.mkdir(parents=True,exist_ok=True)
    handlers={};external=[];stop_event=threading.Event()
    def external_stop(number,frame):
        external.append(signal.Signals(number).name);stop_event.set()
    identity=None;process=None;child_start=usage(resource.RUSAGE_CHILDREN)
    child_base=0.;own_base=0.
    command_sha=hashlib.sha256(json.dumps(command,separators=(',',':')).encode()).hexdigest()
    state={'schema':SCHEMA,'run_id':config['run_id'],'status':'WAITING_ADMISSION','active':None,
           'config_sha256':config_sha,'source_sha256':source_sha,'science_script_sha256':sha(script),
           'command_sha256':command_sha,
           'science_output':str(scientific_output),'old_queue_is_observed_never_signalled':True,
           'attempts':[],'created_at_utc':datetime.now(timezone.utc).isoformat()}
    path=output/'state.json'
    # A second wrapper cannot overlap an earlier output. Root must reconcile an
    # unclean state rather than overwrite its console or refund failed CPU.
    with (output/'admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if path.exists():
            previous=read(path)
            need(previous.get('status')=='BLOCKED' and previous.get('active') is None and
                 previous.get('closure_confirmed') is True and previous.get('resume_eligible') is True,
                 'Unclean, failed or completed admission state cannot be automatically adopted')
            for key in ('config_sha256','source_sha256','science_script_sha256','command_sha256'):
                need(previous.get(key)==state[key], 'Frozen admission resume identity differs')
            state['attempts']=previous['attempts']
            need(len(state['attempts'])<config['max_attempts'], 'Finite attempts already exhausted')
            child_base=float(previous['child_cpu_seconds']);own_base=float(previous['own_cpu_seconds'])
            need(math.isfinite(child_base) and math.isfinite(own_base) and child_base>=0 and own_base>=0,
                 'Unknown prior admission CPU cannot be refunded')
            archive=output/('closed-invocation-'+uuid.uuid4().hex+'.json')
            b.atomic_json(archive,previous)
            state.update(previous_closed_invocation=str(archive),previous_cpu_seconds=child_base+own_base)
        actual_wrapper=b.proc_stat(os.getpid())
        need(actual_wrapper is not None,'Actual admission wrapper identity unavailable')
        state['wrapper_identity']={**actual_wrapper,'uid':os.getuid(),'boot_id':b.boot_id()}
        for sig in (signal.SIGTERM,signal.SIGINT):
            handlers[sig]=signal.getsignal(sig)
            signal.signal(sig,external_stop)
        try:
            for index in range(len(state['attempts'])+1,config['max_attempts']+1):
                while True:
                    need(sha(config_path)==config_sha and sha(__file__)==source_sha, 'Admission config/source drift')
                    need(sha(script)==state['science_script_sha256'],'Scientific source changed during admission wait')
                    need(not external,'External supervisor stop; do not start more science')
                    need(datetime.now(timezone.utc)<end,'New grant deadline reached')
                    reason=admission_reason(config,b,memory=memory_current(),free_disk=free_disk())
                    if reason is not None and reason!='old_queue_state_unavailable':
                        advice=maybe_advise_closed(root,cache_ref,b)
                        if advice is not None:
                            state['last_closed_cache_advice']=advice
                            reason=admission_reason(config,b,memory=memory_current(),free_disk=free_disk())
                    state.update(status='WAITING_ADMISSION',waiting_reason=reason,
                                 own_cpu_seconds=own_base+usage(resource.RUSAGE_SELF),
                                 child_cpu_seconds=child_base+max(0.,usage(resource.RUSAGE_CHILDREN)-child_start),
                                 heartbeat_utc=datetime.now(timezone.utc).isoformat())
                    b.atomic_json(path,state)
                    if reason is None:break
                    stop_event.wait(config['wait_seconds'])
                nonce=uuid.uuid4().hex;env=dict(os.environ);env.update(b.THREAD_ENV);env[TOKEN]=nonce
                attempt={'number':index,'status':'STARTING','started_at_utc':datetime.now(timezone.utc).isoformat(),
                         'cpu_before':usage(resource.RUSAGE_CHILDREN),'command':command}
                state['attempts'].append(attempt);b.atomic_json(path,state)
                console=output/('science-attempt%02d.log'%index)
                with console.open('x') as log:
                    need(sha(script)==state['science_script_sha256'],'Scientific source changed before spawn')
                    process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,
                          stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
                    actual=b.proc_stat(process.pid)
                    need(actual is not None,'Science child exited before identity observation')
                    identity={k:actual[k] for k in ('pid','start_ticks','session','pgrp')}
                    identity.update(uid=os.getuid(),boot_id=b.boot_id(),token=nonce)
                    attempt.update(status='RUNNING',identity=identity,console=str(console))
                    state.update(status='RUNNING',active=identity);b.atomic_json(path,state)
                    reason=None
                    while process.poll() is None:
                        rows=owned_rows(b,identity);rss=owned_rss(rows);memory=memory_current()
                        if external:reason='external_supervisor_stop'
                        elif datetime.now(timezone.utc)>=end:reason='grant_deadline'
                        elif sha(config_path)!=config_sha or sha(__file__)!=source_sha:reason='admission_source_drift'
                        elif sha(script)!=state['science_script_sha256']:reason='scientific_source_drift'
                        elif memory>=config['stop_cgroup_bytes']:reason='resource_global_memory'
                        elif rss>=config['own_rss_stop_bytes']:reason='resource_owned_memory'
                        elif free_disk()<config['minimum_disk_bytes']:reason='resource_disk_floor'
                        state.update(heartbeat_utc=datetime.now(timezone.utc).isoformat(),rss_bytes=rss,
                                     cgroup_memory_bytes=memory,own_cpu_seconds=own_base+usage(resource.RUSAGE_SELF))
                        b.atomic_json(path,state)
                        if reason:
                            close_owned(b,identity,process);break
                        stop_event.wait(min(.5,config['poll_seconds']))
                    code=process.wait()
                    if any(r['state']!='Z' for r in owned_rows(b,identity)):
                        close_owned(b,identity,process)
                        need(False,'Orphan descendants after scientific child exit')
                    attempt.update(exit_code=code,finished_at_utc=datetime.now(timezone.utc).isoformat(),
                         actual_child_cpu_seconds=max(0.,usage(resource.RUSAGE_CHILDREN)-attempt.pop('cpu_before')),
                         status='RESOURCE_INTERRUPTED' if reason else ('EXITED_ZERO' if code==0 else 'FAILED'),reason=reason)
                    identity=None;process=None;state['active']=None;b.atomic_json(path,state)
                if reason and reason.startswith('resource_'):
                    continue
                need(reason is None and code==0,'Scientific worker failed or externally interrupted')
                seals=validate_complete(scientific_output)
                state.update(status='DONE',closure_confirmed=True,scientific_seals=seals,
                             finished_at_utc=datetime.now(timezone.utc).isoformat(),
                             own_cpu_seconds=own_base+usage(resource.RUSAGE_SELF),
                             child_cpu_seconds=child_base+max(0.,usage(resource.RUSAGE_CHILDREN)-child_start))
                b.atomic_json(path,state);b.atomic_json(output/'receipt.json',state)
                return 0
            raise RuntimeError('Four finite resource attempts exhausted; no automatic fifth attempt')
        except BaseException as error:
            state.update(status='BLOCKED',error_type=type(error).__name__,error=str(error))
            if process is not None and identity is not None:
                try:close_owned(b,identity,process);identity=None;state.update(active=None,closure_confirmed=True)
                except BaseException as closure_error:state['closure_error']=str(closure_error)
            elif process is not None:
                # No signalling if the actual birth identity was never observed.
                try:process.wait(timeout=.1)
                except subprocess.TimeoutExpired:state['unknown_child_state']='ROOT_RECONCILIATION_REQUIRED'
            else:
                state.update(active=None,closure_confirmed=True)
            state.update(child_cpu_seconds=child_base+max(0.,usage(resource.RUSAGE_CHILDREN)-child_start),
                         own_cpu_seconds=own_base+usage(resource.RUSAGE_SELF),
                         resume_eligible=bool(external) and identity is None and datetime.now(timezone.utc)<end,
                         finished_at_utc=datetime.now(timezone.utc).isoformat())
            b.atomic_json(path,state);b.atomic_json(output/'receipt.json',state)
            return 1
        finally:
            for sig,handler in handlers.items():signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args();command=args.command
    if command[:1]==['--']:command=command[1:]
    return run(args.root,args.output,command,helper())


if __name__=='__main__':raise SystemExit(main())
