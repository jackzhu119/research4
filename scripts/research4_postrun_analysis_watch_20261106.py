"""Finite local post-run observer: analyze sealed Host-A004 results, then return.

Default --check-only is read-only. No hosted schedules or chat notifications
are created. The observer never launches quantum/scientific training workers,
changes a ledger, repairs numerical results, or signals existing processes.
It starts only its own bounded read-only summary and unchanged Git-return helper
after the original wrapper/queue are actually closed. Detachment does not
guarantee survival of a machine shutdown or external environment replacement.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode=True

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'reports/cloud-followup-20261106-v1/queue-host-a-004'
CONFIG_SCHEMA='research4-postrun-analysis-watch-config-v1'
STATE_SCHEMA='research4-postrun-analysis-watch-state-v1'
DEADLINE='2026-11-09T08:25:17.730193+00:00'
UPLOADER=ROOT/'scripts/research4_host_a_git_return_20261106_v4.py'
WRAPPER=ROOT/'scripts/run_research4_host_a_20261106_v4.py'
SUMMARY=ROOT/'scripts/research4_cloud_followup_summary_20261106.py'
SAFE=re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,59}\Z')
TERMINAL={'ALL_APPROVED_JOBS_CLOSED','BLOCKED_REQUIRES_ROOT_REVIEW_NO_AUTOMATIC_RETRY',
          'BLOCKED_REQUIRES_ROOT_REVIEW','BLOCKED','FAILED_CLOSED','PARTIAL_SCIENTIFIC_CLOSED',
          'HEALTHY_BOUNDARY_CLOSED','COMPLETED','COMPLETED_WITH_BLOCKED_JOBS'}


def require(ok,message):
    if not ok:raise RuntimeError(message)


def utc():return datetime.now(timezone.utc).isoformat()


def aware(value):
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    require(result.tzinfo is not None,'Timezone-aware deadline required')
    return result.astimezone(timezone.utc)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read(path):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=32*1024**2,'Bounded real metadata file required')
    return json.loads(path.read_text())


def checked(record):
    raw=Path(record['path']);path=raw.resolve()
    require(not raw.is_symlink() and path.is_relative_to(ROOT) and path.is_file(),'Exact repository source file required')
    require(sha(path)==record['sha256'],'Hash-bound source/input changed: '+str(path))
    return path


def ref(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':sha(path)}


def save(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temporary.open('x') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def identity(pid):
    proc=Path('/proc')/str(pid)
    raw=(proc/'stat').read_text();values=raw[raw.rfind(')')+2:].split()
    uid=next(x for x in (proc/'status').read_text().splitlines() if x.startswith('Uid:'))
    return {'pid':int(pid),'start_ticks':int(values[19]),'uid':int(uid.split()[1]),'state':values[0],
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def identity_state(expected,reader=identity):
    try:actual=reader(expected['pid'])
    except (FileNotFoundError,ProcessLookupError):return 'GONE'
    for key in ('pid','start_ticks','uid','boot_id'):
        require(actual[key]==expected[key],'Expected process identity is reused, changed, or from another boot')
    return 'CLOSED_ZOMBIE' if actual['state'] in ('Z','X','x') else 'LIVE'


def load_uploader(config):
    path=checked(config['uploader_source'])
    require(path==UPLOADER,'Only the separately reviewed and root-hash-bound A004 uploader is allowed')
    spec=importlib.util.spec_from_file_location('postrun_frozen_host_a_uploader_v4',path)
    module=importlib.util.module_from_spec(spec)
    old=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True;spec.loader.exec_module(module)
    finally:sys.dont_write_bytecode=old
    return module


def validate_config(config_file,expected_sha):
    config=read(checked({'path':str(Path(config_file).resolve()),'sha256':expected_sha}))
    require(config.get('schema')==CONFIG_SCHEMA,'Root watch configuration schema required')
    for key in ('source_worker','summary_source','uploader_source','queue_config','queue_grant',
                'return_manifest','original_return_manifest','wrapper_source','wrapper_launch'):
        checked(config[key])
    require(checked(config['source_worker'])==Path(__file__).resolve(),'Watch config binds another observer source')
    require(checked(config['summary_source'])==SUMMARY,'Only the separately reviewed postrun summary source is allowed')
    require(checked(config['uploader_source'])==UPLOADER and checked(config['wrapper_source'])==WRAPPER,'A004 uploader/wrapper binding changed')
    require(Path(config['work_directory']).resolve()==WORK,'Only actual Host-A004 work is eligible')
    require(Path(config['watch_directory']).resolve()==WORK/'postrun-watch','Watch writes have one narrow new prefix')
    require(Path(config['summary_output']).resolve()==WORK/'science/postrun-analysis-001','One pre-approved fresh summary output required')
    require(Path(config['wrapper_latest']).resolve()==ROOT/'.runtime/research4-host-a-wrapper-latest.json','Exact wrapper upload observation required')
    require(Path(config['python_executable']).resolve()==(ROOT/'.runtime/bin/python').resolve(),'Use the existing root-approved Python interpreter')
    require(aware(config['deadline_utc'])==aware(DEADLINE) and config.get('poll_seconds')==30,'Fixed deadline and 30-second local polling required')
    qconfig=read(checked(config['queue_config']));grant=read(checked(config['queue_grant']))
    require(aware(qconfig['deadline_utc'])==aware(DEADLINE) and aware(grant['expires_at_utc'])==aware(DEADLINE),'Original A004 finite window changed')
    require(grant.get('config_sha256')==config['queue_config']['sha256'] and grant.get('root_execution_approved') is True,'Original root grant/config binding differs')
    launch=read(checked(config['wrapper_launch']))
    launch_path=checked(config['wrapper_launch'])
    require(launch_path.parent==WORK and launch_path.name.endswith('.launch.json'), 'Actual A004 launch receipt must stay in its fixed work prefix')
    require(Path(config['wrapper_state']).resolve()==launch_path.with_name(launch_path.name.removesuffix('.launch.json')+'.json'),
            'Wrapper state must belong to the exact launched wrapper attempt')
    require(launch.get('status')=='ACTUAL_WRAPPER_STARTED_NOT_SCIENCE_COMPLETE'
            and launch.get('config_sha256')==config['queue_config']['sha256']
            and launch.get('source_sha256')==config['wrapper_source']['sha256'],'Actual original wrapper launch receipt required')
    expected=launch.get('identity',{})
    identity_keys=('pid','start_ticks','uid','boot_id')
    pinned=config.get('expected_wrapper_identity',{})
    require(set(pinned)==set(identity_keys) and all(expected.get(k)==pinned[k] for k in identity_keys)
            and type(pinned['pid']) is int and pinned['pid']>0 and type(pinned['start_ticks']) is int and pinned['start_ticks']>0
            and pinned['uid']==1000 and isinstance(pinned['boot_id'],str) and len(pinned['boot_id'])==36,
            'Only a root-pinned actual wrapper PID/birth/UID/boot identity is eligible')
    return config,qconfig,grant,launch


def require_no_live_git(opaque_exceptions=()):
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            raw=(proc/'stat').read_text();tail=raw[raw.rfind(')')+2:].split()
            if tail[0] in ('Z','X','x'):continue
            args=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
            require(not args or Path(args[0]).name not in ('git','git-remote-https','git-remote-http','git-receive-pack','git-upload-pack'),
                    'An existing Git process is active; do not run concurrent Git')
        except (FileNotFoundError,ProcessLookupError):continue
        except PermissionError:
            # Reuse the exact double-read daemon exception of the frozen scanner;
            # unknown opaque scientific processes remain a blocking condition.
            first=identity(int(proc.name));second=identity(int(proc.name))
            require(all(first[k]==second[k] for k in ('pid','start_ticks','uid','boot_id')) and first['uid']==0,
                    'Unknown process identity prevents Git exclusion')
            comm=(proc/'comm').read_text().strip()
            require(comm in ('dockerd','containerd') and any(all(first.get(k)==record.get(k) for k in
                    ('pid','start_ticks','uid','boot_id')) and record.get('comm')==comm for record in opaque_exceptions),
                    'Opaque process is not an exact root-approved non-scientific daemon identity')


def verify_git_receipt(value,manifest_sha,helper_sha):
    require(value.get('status')=='DONE_ACTUAL_PUSH_AND_REMOTE_REF_VERIFIED'
            and value.get('actual_result_upload')=='DONE'
            and value.get('manifest_sha256')==manifest_sha and value.get('source_helper_sha256')==helper_sha
            and isinstance(value.get('science_commit'),str) and re.fullmatch(r'[0-9a-f]{40}',value['science_commit'])
            and value.get('all_committed_scientific_blobs_match') is True and value.get('main_modified') is False
            and value.get('force_push') is False and value.get('branch')=='research/pyscf-spinosin-mechanism-20261031',
            'Actual Git receipt lacks a real SHA or exact helper/manifest/branch verification')


def observe(config,qconfig,launch):
    """No state write, subprocess, Git command, quantum work, or signal."""
    if datetime.now(timezone.utc)>=aware(config['deadline_utc']):return {'status':'BLOCKED_DEADLINE_EXPIRED'}
    wrapper_live=identity_state(launch['identity'])
    if wrapper_live=='LIVE' and not Path(config['wrapper_state']).is_file():
        return {'status':'WAIT_ORIGINAL_WRAPPER_METADATA_INITIALIZATION'}
    wrapper=read(config['wrapper_state'])
    require(all(wrapper.get('identity',{}).get(k)==launch['identity'][k] for k in ('pid','start_ticks','uid','boot_id')),'Wrapper state identity differs from launch')
    require(wrapper.get('source_sha256')==config['wrapper_source']['sha256'],'Wrapper state source binding differs')
    if wrapper_live=='LIVE' and not Path(qconfig['state']).is_file():
        return {'status':'WAIT_ORIGINAL_QUEUE_METADATA_INITIALIZATION'}
    queue=read(qconfig['state'])
    require(queue.get('config_sha256')==config['queue_config']['sha256'] and queue.get('grant_sha256')==config['queue_grant']['sha256'],
            'Actual queue state source/config/grant differs')
    invocations=queue.get('invocations',[])
    require(invocations and wrapper.get('queue_invocations'),'Original queue has no actual invocation receipt')
    for recorded in wrapper['queue_invocations']:
        current=identity_state(recorded['identity'])
        if current=='LIVE':return {'status':'WAIT_ORIGINAL_QUEUE_ALIVE','wrapper_state':wrapper_live,'done_jobs':sum(j.get('status')=='DONE_SEALED' for j in queue.get('jobs',{}).values())}
    if wrapper_live=='LIVE':return {'status':'WAIT_ORIGINAL_WRAPPER_ALIVE','done_jobs':sum(j.get('status')=='DONE_SEALED' for j in queue.get('jobs',{}).values())}
    last=invocations[-1]
    require(last.get('status') in TERMINAL and last.get('ended_at_utc'),'Original queue has not reached an actual recorded terminal boundary')
    for invocation in invocations:
        require(invocation.get('status')!='RUNNING','An original queue invocation is still recorded RUNNING')
        if invocation.get('identity'):require(identity_state(invocation['identity'])!='LIVE','An original queue parent is still alive')
    require(wrapper.get('status')==last['status'] and wrapper.get('quantum_phases_closed') is True and wrapper.get('ended_at_utc'),
            'Closed wrapper state is missing actual queue closure')
    uploader=load_uploader(config)
    manifest=read(checked(config['return_manifest']))
    closure=uploader.require_no_active_quantum(ROOT,manifest)
    require_no_live_git(read(checked(config['queue_grant'])).get('opaque_process_identity_exceptions',[]))
    latest_path=Path(config['wrapper_latest'])
    require(latest_path.is_file(),'Original wrapper upload outcome is unknown; require actual closure observation')
    latest=read(latest_path)
    require(all(latest.get('identity',{}).get(k)==launch['identity'][k] for k in ('pid','start_ticks','uid','boot_id'))
            and latest.get('status')==wrapper['status'],'Latest upload observation belongs to a different wrapper boundary')
    succeeded=latest.get('actual_upload_succeeded')
    require(type(succeeded) is bool,'Original upload outcome is unknown')
    latest_attempt=wrapper['queue_invocations'][-1]['attempt']
    receipts=[]
    for p in sorted((WORK/'git-return').glob(latest_attempt+'-upload*/return_receipt.json')):
        value=read(p)
        if value.get('status')=='DONE_ACTUAL_PUSH_AND_REMOTE_REF_VERIFIED':
            verify_git_receipt(value,config['original_return_manifest']['sha256'],config['uploader_source']['sha256'])
            receipts.append(ref(p))
    if succeeded:require(receipts,'Original reported upload success has no actual verified return receipt')
    return {'status':'READY_ANALYSIS_AND_RETURN' if succeeded else 'READY_ANALYSIS_ORIGINAL_UPLOAD_FAILED',
            'original_queue_terminal_status':last['status'],'original_upload_succeeded':succeeded,
            'original_upload_receipts':receipts,'closure':closure,'all_existing_wrapper_queue_and_Q_closed':True}


@contextmanager
def sole_watch_lock(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/'.postrun-analysis-watch.lock'
    require(not path.is_symlink(),'Symlink observer lock rejected')
    with path.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def child_limits(cpu_soft,cpu_hard):
    def limit():
        resource.setrlimit(resource.RLIMIT_CPU,(cpu_soft,cpu_hard))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    return limit


def bounded_child(command,console,wall_seconds,cpu_soft=120,cpu_hard=150):
    """May terminate only the newly owned non-Q analysis/upload child on timeout."""
    require(not Path(console).exists(),'Preserve earlier owned child logs')
    with Path(console).open('x') as handle:
        try:
            result=subprocess.run(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                timeout=wall_seconds,preexec_fn=child_limits(cpu_soft,cpu_hard))
            return {'returncode':result.returncode,'timed_out':False,'console':ref(console)}
        except subprocess.TimeoutExpired:
            return {'returncode':None,'timed_out':True,'console':ref(console),'only_newly_owned_child_contained':True}


def execute(config,config_ref,attempt):
    start_cpu=time.process_time();start_wall=time.perf_counter()
    with sole_watch_lock(config['watch_directory']):
        directory=Path(config['watch_directory'])/attempt
        require(not directory.exists(),'Fresh observer attempt required; preserve earlier state and failures')
        directory.mkdir()
        state={'schema':STATE_SCHEMA,'status':'WAITING_FOR_ORIGINAL_A004_CLOSURE','started_at_utc':utc(),
            'identity':identity(os.getpid()),'config':config_ref,'source_worker':ref(__file__),
            'deadline_utc':config['deadline_utc'],'new_quantum_evaluations':0,'models_fitted':0,
            'notifications':'NONE','hosted_scheduled_tasks_created':False,'signals_sent_to_existing_processes':False,
            'original_modes_retained':219,'original_imaginary_modes_retained':25,'minimum_qualification_changed':False,
            'summary_invocations':0,'upload_invocations':[]}
        def publish(event,details=None):
            state.update(updated_at_utc=utc(),own_CPU_seconds=time.process_time()-start_cpu,wall_seconds=time.perf_counter()-start_wall)
            if details:state.update(details)
            save(directory/'state.json',state)
            with (directory/'events.jsonl').open('a') as handle:
                handle.write(json.dumps({'utc':utc(),'event':event,'status':state['status'],'own_CPU_seconds':state['own_CPU_seconds']},allow_nan=False)+'\n')
        def stop(signum,frame):raise RuntimeError('Observer own bounded/external signal '+str(signum)+'; no existing process signalled')
        old={sig:signal.signal(sig,stop) for sig in (signal.SIGXCPU,signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
        resource.setrlimit(resource.RLIMIT_CPU,(120,150));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        signal.alarm(max(1,math.ceil((aware(config['deadline_utc'])-datetime.now(timezone.utc)).total_seconds())))
        publish('OBSERVER_STARTED')
        last_status=None;heartbeat=0.
        try:
            while True:
                require(time.process_time()-start_cpu<100,'Observer own CPU reserve reached; preserve state without new work')
                config,qconfig,grant,launch=validate_config(config_ref['path'],config_ref['sha256'])
                observation=observe(config,qconfig,launch)
                status=observation['status'];state['latest_observation']=observation
                if status.startswith('BLOCKED'):raise RuntimeError(status)
                if status.startswith('READY_'):break
                state['status']=status
                if status!=last_status or time.monotonic()-heartbeat>=900:
                    publish('WAITING_HEALTH_CHECK');last_status=status;heartbeat=time.monotonic()
                time.sleep(min(30,max(0.,(aware(config['deadline_utc'])-datetime.now(timezone.utc)).total_seconds())))
            require(datetime.now(timezone.utc)<aware(config['deadline_utc']),'Deadline expired before new analysis')
            output=Path(config['summary_output'])
            require(not output.exists(),'Pre-existing summary output requires root review; never rerun or overwrite it')
            state['status']='RUNNING_ONE_BOUNDED_READONLY_SUMMARY';state['summary_invocations']=1
            publish('SUMMARY_ACTUAL_START')
            argv=[config['python_executable'],'-B',str(checked(config['summary_source'])),'--config',str(checked(config['queue_config'])),
                  '--grant',str(checked(config['queue_grant'])),'--execute','--attempt','postrun-analysis-001','--output',str(output)]
            summary=bounded_child(argv,directory/'summary.console.log',min(300,max(1.,(aware(config['deadline_utc'])-datetime.now(timezone.utc)).total_seconds())))
            state['summary_execution']=summary
            require(summary['returncode']==0 and not summary['timed_out'],'Actual bounded summary failed/pending; preserve evidence without automatic rerun')
            result=read(output/'result.json')
            require(result.get('schema')=='research4-cloud-followup-postrun-analysis-v1' and result.get('status')=='POSTRUN_ANALYSIS_COMPLETED'
                    and result.get('new_quantum_evaluations')==0 and result.get('models_fitted')==0,'Actual summary result does not match approved analysis scope')
            state['summary_result']=ref(output/'result.json');state['scientific_anomalies_not_repaired']=True
            if not observation['original_upload_succeeded']:
                state['status']='PARTIAL_UPLOAD_NEEDS_REVIEW';publish('ANALYSIS_DONE_ORIGINAL_UPLOAD_FAILED_NO_NEW_GIT')
                return state
            uploader=load_uploader(config)
            manifest=read(checked(config['return_manifest']))
            for index in range(1,4):
                require(datetime.now(timezone.utc)<aware(config['deadline_utc']),'Deadline expired before new Git return')
                require(identity_state(launch['identity'])!='LIVE','Original wrapper unexpectedly active before Git return')
                uploader.require_no_active_quantum(ROOT,manifest);require_no_live_git(grant.get('opaque_process_identity_exceptions',[]))
                require(checked(config['uploader_source'])==UPLOADER,'Uploader source changed')
                name=attempt+'-return'+str(index)
                state['status']='RETURNING_ACTUAL_ANALYSIS';publish('UPLOAD_ACTUAL_START')
                argv=[config['python_executable'],'-B',str(UPLOADER),'--repo',str(ROOT),'--manifest',str(checked(config['return_manifest'])),
                      '--manifest-sha256',config['return_manifest']['sha256'],'--attempt',name,'--execute']
                returned=bounded_child(argv,directory/('upload-'+str(index)+'.console.log'),
                    min(900,max(1.,(aware(config['deadline_utc'])-datetime.now(timezone.utc)).total_seconds())))
                state['upload_invocations'].append({'attempt':name,**returned})
                if returned['timed_out']:
                    state['status']='PARTIAL_UPLOAD_NEEDS_REVIEW'
                    publish('OWNED_UPLOADER_TIMEOUT_NO_AUTOMATIC_RETRY_UNTIL_PROCESS_FORENSICS')
                    return state
                receipt=WORK/'git-return'/name/'return_receipt.json'
                if returned['returncode']==0 and receipt.is_file():
                    value=read(receipt)
                    verify_git_receipt(value,config['return_manifest']['sha256'],config['uploader_source']['sha256'])
                    state['status']='DONE_ANALYSIS_AND_ACTUAL_GIT_RETURN_SCIENTIFIC_LIMITATIONS_RETAINED'
                    state['actual_git_return_receipt']=ref(receipt);state['science_commit']=value['science_commit']
                    publish('FINITE_POSTRUN_ENDPOINT_DONE');return state
                publish('BOUNDED_UPLOAD_FAILURE_PRESERVED')
                if index<3:time.sleep(min(30,max(0.,(aware(config['deadline_utc'])-datetime.now(timezone.utc)).total_seconds())))
            state['status']='PARTIAL_UPLOAD_NEEDS_REVIEW';publish('MAX_THREE_UPLOAD_ATTEMPTS_REACHED');return state
        except BaseException as error:
            state.update(status='BLOCKED_REQUIRES_ROOT_REVIEW',error_type=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
            publish('OBSERVER_BLOCKED_NO_SCIENCE_REWRITE');return state
        finally:
            signal.alarm(0)
            state['ended_at_utc']=utc();save(directory/'state.json',state)
            for sig,handler in old.items():signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--check-only',action='store_true');parser.add_argument('--execute',action='store_true')
    parser.add_argument('--detach',action='store_true');parser.add_argument('--attempt',default='postrun-watch-001')
    args=parser.parse_args();require(not(args.execute and args.check_only),'Choose read-only check or actual execution')
    require(SAFE.fullmatch(args.attempt) and len(args.attempt)<=52,'Fresh bounded safe attempt identifier with room for uploader suffix required')
    config,qconfig,grant,launch=validate_config(args.config,args.config_sha256)
    config_ref={'path':str(args.config.resolve()),'sha256':args.config_sha256}
    if not args.execute:
        require(not args.detach,'Detachment requires explicit execution')
        result=observe(config,qconfig,launch)
        print(json.dumps({'mode':'CHECK_ONLY_NO_WRITES_NO_SUBPROCESSES',**result},ensure_ascii=False));return 0
    require(datetime.now(timezone.utc)<aware(config['deadline_utc']),'Finite deadline expired; no new observer')
    if args.detach:
        directory=Path(config['watch_directory']);directory.mkdir(parents=True,exist_ok=True)
        # Brief probe only. The child owns the sole lock for its actual lifetime.
        with sole_watch_lock(directory):pass
        receipt=directory/(args.attempt+'.launch.json');console=directory/(args.attempt+'.console.log')
        require(not receipt.exists() and not console.exists() and not(directory/args.attempt).exists(),'Preserve previous observer attempts')
        argv=[config['python_executable'],'-B',str(Path(__file__).resolve()),'--config',str(args.config.resolve()),
              '--config-sha256',args.config_sha256,'--execute','--attempt',args.attempt]
        with console.open('x') as handle:
            child=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        value={'schema':'research4-postrun-observer-detached-launch-v1','status':'ACTUAL_OBSERVER_STARTED_NOT_ANALYSIS_OR_GIT_DONE',
            'created_at_utc':utc(),'identity':identity(child.pid),'config':config_ref,'source_worker':ref(__file__),
            'console':str(console),'deadline_utc':config['deadline_utc'],'machine_shutdown_survival_not_guaranteed':True}
        save(receipt,value);print(json.dumps(value));return 0
    result=execute(config,config_ref,args.attempt)
    print(json.dumps({k:result.get(k) for k in ('status','summary_invocations','science_commit')},ensure_ascii=False))
    return 0 if result['status'].startswith('DONE_') else 2


if __name__=='__main__':raise SystemExit(main())
