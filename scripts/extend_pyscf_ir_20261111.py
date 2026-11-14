#!/usr/bin/env python3
"""Append CPU authorization and continue a frozen IR run at its phase boundary.

The existing supervisor, guard, worker, scientific snapshot and ledgers remain
untouched while they run. A separate watcher arms a successor that loads the
same frozen supervisor with the new cumulative allowance. All successor guard,
publisher and recovery subprocesses use this entry point; scientific workers
continue to execute their original frozen entry point.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temp.open('w') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output(root):
    return Path(root)/'reports/pyscf_ir_20261111'


def allowance(root):
    value=read(output(root)/'authorization.json')
    if not value or value.get('schema')!='pyscf-ir-cumulative-authorization-v1':
        raise ValueError('Cumulative authorization ledger is missing')
    base=float(value['initial_authorized_cpu_seconds'])
    if not math.isfinite(base) or base<=0:
        raise ValueError('Invalid initial CPU authorization')
    ids=set();total=base
    for entry in value['grants']:
        amount=entry['added_cpu_seconds']
        if isinstance(amount,bool) or not isinstance(amount,(int,float)) or not math.isfinite(amount) or amount<=0:
            raise ValueError('Invalid additional CPU grant')
        if entry['grant_id'] in ids:
            raise ValueError('Duplicate grant identity')
        ids.add(entry['grant_id']);total+=amount
    if total!=value['cumulative_authorized_cpu_seconds']:
        raise ValueError('CPU authorization sum mismatch')
    return total


def original_module(root):
    path=output(root)/'.background/frozen/scripts/background_pyscf_ir_20261111.py'
    name='pyscf_ir_frozen_background_for_authorization'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    module.check_integrity(module.Paths(Path(root)))
    return module


def grant(root, amount, grant_id):
    if isinstance(amount,bool) or not isinstance(amount,(int,float)) or not math.isfinite(amount) or amount<=0:
        raise ValueError('CPU grant must be positive and finite')
    root=Path(root);out=output(root)
    lock=(out/'.background/authorization.lock').open('a+')
    with lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        previous=read(out/'authorization.json')
        if previous is None:
            original=original_module(root)
            previous={'schema':'pyscf-ir-cumulative-authorization-v1',
                'initial_authorized_cpu_seconds':original.AUTHORIZED_CPU_SECONDS,
                'cumulative_authorized_cpu_seconds':original.AUTHORIZED_CPU_SECONDS,
                'grants':[], 'accounting':'Existing preflight and session CPU ledgers are retained; no reset.'}
        else:
            allowance(root)
        for entry in previous['grants']:
            if entry['grant_id']==grant_id:
                if entry['added_cpu_seconds']!=amount:
                    raise ValueError('Existing CPU grant has a different amount')
                return previous
        previous['grants'].append({'grant_id':grant_id,'added_cpu_seconds':amount,
            'authorized_at_utc':utc(),'authority':'Explicit user CPU extension request in this chat'})
        previous['cumulative_authorized_cpu_seconds']+=amount
        previous['updated_at_utc']=utc()
        write(out/'authorization.json',previous)
        allowance(root)
        return previous


def extension_file(root):
    return output(root)/'authorization/frozen_extension.py'


def verify_extension(root):
    manifest=read(output(root)/'authorization_extension_manifest.json')
    if not manifest or sha(extension_file(root))!=manifest['sha256']:
        raise ValueError('Authorization continuation implementation seal mismatch')
    return True


def configured_module(root, entry=None):
    verify_extension(root)
    module=original_module(root)
    total=allowance(root)
    module.AUTHORIZED_CPU_SECONDS=total
    original_start=module.start_child
    original_write=module.write
    entry=Path(entry or extension_file(root))

    def start_child(paths,script,arguments,log_name):
        if Path(script).resolve()==paths.frozen_script.resolve():
            script=entry
        return original_start(paths,script,arguments,log_name)

    def corrected_write(path,value):
        if isinstance(value,dict) and value.get('schema')=='pyscf-ir-background-v1':
            effective_total=module.AUTHORIZED_CPU_SECONDS
            value={**value,'authorization':f'User explicitly authorized {effective_total:g} cumulative CPU seconds',
                'authorized_cpu_seconds':effective_total,'authorization_ledger':'reports/pyscf_ir_20261111/authorization.json'}
        return original_write(path,value)

    module.start_child=start_child
    module.write=corrected_write
    return module


def verify(root):
    module=configured_module(root);paths=module.Paths(Path(root))
    value=module.verify(paths)
    value['cumulative_authorized_cpu_seconds']=allowance(root)
    value['current_phase_cpu_boundary']=value['state'].get('authorized_cpu_seconds')
    value['pending_additional_cpu_seconds']=max(0,allowance(root)-float(value['current_phase_cpu_boundary'] or 0))
    state=read(output(root)/'authorization_continuation.json',{})
    value['continuation']=state
    value['continuation_watcher_alive']=module.matching(state.get('watcher_identity'))
    return value


def boundary_ready(module,paths,state):
    if state.get('status')!='AUTHORIZED_CPU_BOUNDARY':
        return False
    if any(module.matching(state.get(key)) for key in
           ('supervisor_identity','worker_identity','guard_identity','publisher_identity')):
        return False
    lock=(paths.runtime/'supervisor.lock').open('a+')
    try:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True
    finally:
        lock.close()


def watch(root, interval=2.0):
    root=Path(root);module=configured_module(root);paths=module.Paths(root)
    lock=(paths.runtime/'authorization_watcher.lock').open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close();return 0
    session_id='authorization-watcher-'+uuid.uuid4().hex
    identity=module.identity(os.getpid())
    state={'schema':'pyscf-ir-authorization-continuation-v1','status':'ARMED_WAITING',
        'watcher_identity':identity,'started_at_utc':utc(),'authorization_total_cpu_seconds':allowance(root),
        'current_worker_interrupted':False,'scientific_source_changed':False}
    ledger=paths.runtime/'sessions'/f'{session_id}.json'
    def record(status):
        point=module.proc_stat(os.getpid())
        own=float(point['own_cpu']) if point else 0
        # Only this watcher's own CPU: the successor's descendants have their
        # own existing meter and must never be recharged as waited children.
        write(ledger,{'session_id':session_id,'status':'AUTHORIZATION_WATCHER_'+status,
            'observed_cpu_seconds':own,'conservative_cpu_seconds':own,
            'measurement':'Watcher own process CPU only; excludes successor child CPU',
            'sample_at_utc':utc()})
        state.update(status=status,heartbeat_at_utc=utc())
        write(paths.output/'authorization_continuation.json',state)
    try:
        while True:
            verify_extension(root)
            state_now=read(paths.state,{})
            status=state_now.get('status')
            if (paths.runtime/'STOP.json').exists() or status in (
                    'COMPLETE','STOPPED_BY_USER','SCIENTIFIC_INTEGRITY_CHANGED'):
                record('NO_RESTART_'+str(status));return 0
            if boundary_ready(module,paths,state_now):
                module.AUTHORIZED_CPU_SECONDS=allowance(root)
                _,used=module.session_totals(paths)
                if used>=module.AUTHORIZED_CPU_SECONDS:
                    record('CUMULATIVE_AUTHORIZATION_EXHAUSTED');return 0
                old_stop=paths.runtime/'budget_stop.json'
                if old_stop.exists():
                    archive=paths.output/'authorization'/('previous_boundary_'+utc().replace(':','-')+'.json')
                    archive.parent.mkdir(parents=True,exist_ok=True)
                    os.replace(old_stop,archive)
                # start() retains the prior sessions and preflight CPU. Every
                # spawned supervisor/guard/publisher entry is routed above.
                activation=module.start(paths)
                if activation.get('supervisor_alive') and activation.get('guard_alive'):
                    state['activated_supervisor_identity']=activation['state']['supervisor_identity']
                    state['activated_at_utc']=utc()
                    record('ACTIVATED');return 0
            record('ARMED_WAITING')
            time.sleep(interval)
    finally:
        record(state['status'])
        lock.close()


def arm(root):
    root=Path(root);out=output(root);allowance(root)
    existing=read(out/'authorization_continuation.json',{})
    module=original_module(root)
    if module.matching(existing.get('watcher_identity')):
        return verify(root)
    target=extension_file(root)
    target.parent.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve()
    if source!=target.resolve():
        shutil.copyfile(source,target)
    write(out/'authorization_extension_manifest.json',{'schema':'pyscf-ir-authorization-extension-code-v1',
        'path':'reports/pyscf_ir_20261111/authorization/frozen_extension.py',
        'sha256':sha(target),'created_at_utc':utc(),
        'original_scientific_source_manifest_changed':False})
    with (out/'.background/authorization_watcher.log').open('ab',buffering=0) as stream:
        child=subprocess.Popen([sys.executable,str(target),'--root',str(root),'--watch'],
            cwd=root,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
            start_new_session=True,close_fds=True)
    for _ in range(50):
        current=read(out/'authorization_continuation.json',{})
        if module.matching(current.get('watcher_identity')):
            return verify(root)
        if child.poll() is not None:
            raise RuntimeError('Authorization continuation watcher failed to start')
        time.sleep(.1)
    raise RuntimeError('Authorization watcher launch handshake pending')


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--root',type=Path,default=Path(os.environ.get('PYSCF_IR_REPOSITORY_ROOT','/workspace/research4')))
    parser.add_argument('--grant',type=float)
    parser.add_argument('--grant-id')
    parser.add_argument('--arm',action='store_true')
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--verify',action='store_true')
    args,remainder=parser.parse_known_args()
    root=args.root.resolve()
    if args.grant is not None:
        if not args.grant_id:
            raise ValueError('Explicit grant ID required for idempotent authorization')
        result=grant(root,args.grant,args.grant_id)
        if args.arm:result=arm(root)
    elif args.arm:result=arm(root)
    elif args.watch:return watch(root)
    elif args.verify:result=verify(root)
    else:
        module=configured_module(root)
        sys.argv=[str(__file__),'--root',str(root),*remainder]
        return module.main()
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
