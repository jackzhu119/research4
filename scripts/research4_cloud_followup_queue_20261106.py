"""Finite sequential follow-up queue; preparation is the default, not execution.

The unchanged V8 guard owns all quantum accounting and resource enforcement.
This launcher never edits a ledger, upgrades a method, reallocates CPU, repairs
unknown/orphan phases, or overwrites a failed attempt. Detachment survives a
terminal disconnect, not an instance shutdown. Only explicitly approved,
resource-closed failures may receive one separately named bounded retry.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "scripts/run_pyscf_continuation_budget_20261105_v8.py"
GUARD_SHA = "7bc773076ecc5e0f90fb58b81bf0641948dd09c14b2816138a0469f16e511c65"
HELPERS = ROOT / "scripts/research4_quantum_queue_20261105_v2.py"
HELPERS_SHA = "3af76bd231b4a6caa1bf70fe788c2ad2be1dd9cec8b84d181a17e53efeb79dad"
LEDGER = ROOT / "reports/pyscf-spinosin-96h-continuation-20261105-v8/cpu_budget.json"
CAP = 390160.8308359999
DEADLINE = "2026-11-09T12:30:00+00:00"
SCHEMA = "research4-cloud-followup-queue-config-v1"
GRANT_SCHEMA = "research4-cloud-followup-root-grant-v1"
STATE_SCHEMA = "research4-cloud-followup-state-v1"
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
TOKEN = re.compile(r"\{(result|sha256):([A-Za-z0-9_-]+)\}\Z")
MIN_DISK = 200 * 1024**2


def require(ok, message):
    if not ok:
        raise ValueError(message)


class ScientificQualificationError(ValueError):
    """A genuinely closed, source-bound execution lacks scientific release."""


def scientific_require(ok, message):
    if not ok:
        raise ScientificQualificationError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def path(value):
    candidate = Path(value)
    candidate = candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()
    require(candidate.is_relative_to(ROOT), "All queue paths must remain in research4")
    return candidate


def read(file, expected=None):
    file = path(file)
    require(file.stat().st_size <= 2 * 1024**2, "Unexpectedly large metadata JSON")
    if expected:
        require(sha(file) == expected, "Pinned metadata changed: " + str(file))
    return json.loads(file.read_text())


def atomic(file, value):
    file = path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(file.suffix + ".writing")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(file)


def module(file, expected, name):
    require(sha(file) == expected, "Frozen orchestration source changed")
    spec = importlib.util.spec_from_file_location(name, file)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def modules():
    return module(HELPERS, HELPERS_SHA, "followup_frozen_helpers"), module(GUARD, GUARD_SHA, "followup_unchanged_v8_guard")


def checked(ref):
    p = path(ref["path"])
    require(sha(p) == ref["sha256"], "Source/input hash changed: " + str(p))
    return p


def queue_deadline(config):
    value=datetime.fromisoformat(config['deadline_utc'])
    require(value.tzinfo is not None and value<=datetime.fromisoformat(DEADLINE),"Queue cannot extend original V8 deadline")
    return value


def output_artifacts(output):
    values=[];total=0
    for file in sorted(output.rglob('*')):
        require(not file.is_symlink(),"Symlink scientific artifact rejected")
        if not file.is_file():continue
        size=file.stat().st_size;total+=size
        require(size<=64*1024**2 and total<=128*1024**2,"Follow-up delivery artifacts must remain reasonably small")
        values.append({'relative_path':str(file.relative_to(output)),'size_bytes':size,'sha256':sha(file)})
    return values


def validate(config, grant, config_sha):
    require(config.get("schema") == SCHEMA and grant.get("schema") == GRANT_SCHEMA, "Explicit follow-up schemas required")
    require(grant.get("config_sha256") == config_sha and grant.get("driver_sha256") == sha(__file__), "Source/config-bound root grant required")
    require(grant.get("root_execution_approved") is True and grant.get("root_serial_scheduling_confirmed") is True, "Explicit root approval required")
    require(grant.get("no_new_quantum_budget") is True and grant.get("no_deadline_extension") is True, "Fixed existing pool only")
    require(path(config["ledger"]) == LEDGER and config.get("quantum_cpu_cap_seconds") == CAP and grant.get("expires_at_utc") == config.get("deadline_utc"), "Exact existing V8 ledger/cap and explicit no-later queue deadline required")
    queue_deadline(config)
    require(type(grant.get("max_invocations")) is int and 1 <= grant["max_invocations"] <= 4, "At most four finite invocations")
    require(type(grant.get("launcher_cpu_seconds_per_invocation")) is int and 30 <= grant["launcher_cpu_seconds_per_invocation"] <= 600, "Finite own CPU bound required")
    require(type(grant.get("wall_seconds_per_invocation")) is int and 600 <= grant["wall_seconds_per_invocation"] <= 86400, "Finite own wall bound required")
    require(type(grant.get("launcher_rss_stop_bytes")) is int and 32*1024**2 <= grant["launcher_rss_stop_bytes"] <= 256*1024**2, "Finite own RSS threshold required")
    require(type(config.get("minimum_remaining_cpu_seconds")) in (int, float) and 125 <= config["minimum_remaining_cpu_seconds"] <= CAP, "Explicit minimum available CPU required")
    require(type(config.get("CPU_reserve_seconds")) in (int, float) and 600 <= config["CPU_reserve_seconds"] <= 20000, "Explicit unallocated cleanup/diagnostic reserve required")
    require(isinstance(config.get('phase_prefix'),str) and SAFE_ID.fullmatch(config['phase_prefix']),"Safe unique phase prefix required")
    work = path(config["work_directory"])
    require(work.is_relative_to(ROOT/"reports") and work != ROOT/"reports", "Fresh report namespace required")
    require(path(config["state"]) == work/"state.json" and path(config["monitor"]) == work/"monitor.jsonl", "Narrow state and monitor paths required")
    require(not LEDGER.is_relative_to(work), "Ledger must remain outside queue output")
    refs = {str(checked(r)):r["sha256"] for r in config["source_refs"]}
    require(refs.get(str(GUARD)) == GUARD_SHA and refs.get(str(HELPERS)) == HELPERS_SHA, "Pinned original guard and helpers required")
    for ref in config.get("immutable_inputs", []):
        require(not checked(ref).is_relative_to(work), "Protect scientific inputs from output namespace")
    require(config.get("python_executable") == ".runtime/bin/python" and config.get("pythonpath") == [".runtime/quantum-pilot-site", ".runtime/quantum-upgrade-site"], "Reviewed Python environment required")
    jobs = config["jobs"]
    require(isinstance(jobs, list) and 1 <= len(jobs) <= 32, "One to32 finite jobs required")
    previous = set()
    for job in jobs:
        identity = job["id"]
        require(isinstance(identity, str) and SAFE_ID.fullmatch(identity) and identity not in previous, "Unique safe job ID required")
        require(job.get("kind") in ("quantum", "analysis"), "Only quantum/analysis jobs supported")
        worker = checked(job["worker"])
        require(worker.is_relative_to(ROOT/"scripts") and refs.get(str(worker)) == job["worker"]["sha256"], "Worker must be explicitly pinned source")
        argv = job["payload_argv"]
        require(isinstance(argv, list) and all(isinstance(t,str) and "\x00" not in t for t in argv), "Literal argv only; no shell")
        require(argv and path(argv[0]) == ROOT/".runtime/bin/python" and str(worker) in [str(path(t)) for t in argv if t.endswith(".py")], "Exact Python/worker argv required")
        require(argv.count("--output") == 1 and argv[argv.index("--output")+1] == "{output}", "Unique fresh output placeholder required")
        for token in argv:
            if "{" in token or "}" in token:
                match = TOKEN.fullmatch(token)
                require(token == "{output}" or (match and match.group(2) in previous), "Only previous sealed result/hash placeholders allowed")
        deps = job.get("depends_on", [])
        require(isinstance(deps,list) and set(deps).issubset(previous), "Dependencies must name earlier jobs")
        require(isinstance(job.get("success_statuses"), list) and job["success_statuses"] and all(isinstance(s,str) for s in job["success_statuses"]), "Explicit result success statuses required")
        require(isinstance(job.get("result_schema"),str), "Explicit result schema required")
        require(type(job.get("optional",False)) is bool and type(job.get("retry_closed_resource_failure",False)) is bool, "Boolean scheduling policies required")
        require(isinstance(job.get("required_result_fields",{}),dict), "Result qualification assertions must be a dictionary")
        if job["kind"] == "quantum":
            require(job.get("threads",2) == 2 and job.get("memory_gib",24) == 24, "Unchanged reviewed V8 resource policy required")
            for request in [job]+([job["retry_request"]] if job.get("retry_closed_resource_failure") else []):
                require(type(request.get("cpu_seconds")) is int and 125 <= request["cpu_seconds"] <= 122400 and type(request.get("wall_seconds")) is int and 181 <= request["wall_seconds"] <= 43200, "Finite quantum phase CPU/wall required")
        else:
            require(job.get("retry_closed_resource_failure",False) is False, "No automatic analysis retries")
            require('--analyze-only' in argv and '--execute' not in argv,"Analysis must use the worker's explicit non-quantum entry point")
            require(type(job.get("cpu_seconds")) is int and 1 <= job["cpu_seconds"] <= 120 and type(job.get("wall_seconds")) is int and 1 <= job["wall_seconds"] <= 600, "Lightweight analysis only")
        previous.add(identity)
    exceptions = grant.get("opaque_process_identity_exceptions", [])
    require(isinstance(exceptions,list) and len(exceptions) <= 2, "At most two exact known daemon exceptions")
    for value in exceptions:
        require(value.get("uid") == 0 and value.get("comm") in ("dockerd","containerd") and type(value.get("pid")) is int and type(value.get("start_ticks")) is int and isinstance(value.get("boot_id"),str), "Exact daemon identity exception required")


def active_quantum(q, configured_worker_names=()):
    records=[]
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            first=q.proc_identity(int(directory.name))
            if first['state'] in ('Z','X','x'):
                continue
            raw=(directory/'cmdline').read_bytes()
            tokens=[t.decode(errors='replace') for t in raw.split(b'\0') if t]
            if not tokens or not Path(tokens[0]).name.startswith(('python','pypy')):
                continue
            scripts=[Path(t).name for t in tokens[1:] if t.endswith('.py')]
            names=[n for n in scripts if (n in configured_worker_names or any(s in n for s in ('pyscf','quantum','cloud_followup_worker','cloud_orbital_stability','cloud_small_molecule'))) and not n.startswith(('analyze_','summarize_'))]
            if not names:
                continue
            second=q.proc_identity(int(directory.name))
            require(q.same_identity(first,second),"Unstable quantum process identity")
            records.append({**first,'script_basenames':names,'full_argv_logged':False})
        except (FileNotFoundError,ProcessLookupError):
            continue
    return records


def no_open_quantum(q, guard, config, grant):
    # The worker owns the old main-queue lock while doing science. Probe it
    # here, but never hold that lock across launching a worker that needs it.
    with q.queue_lock():
        pass
    ledger=read(LEDGER);guard.validate_ledger(ledger)
    require(not any(row.get('status')=='RUNNING' for row in ledger['phases']),"Existing live/orphan ledger phase requires root forensics")
    require(ledger['quantum_cpu_cap_seconds']==CAP and ledger['run_deadline_at_utc']==DEADLINE,"Registered allocation changed")
    scan=q.process_observation(opaque_exceptions=grant.get('opaque_process_identity_exceptions',[]))
    require(not scan['conflicts'],"External ML/Git/unknown process conflict")
    worker_names={path(j['worker']['path']).name for j in config['jobs'] if j['kind']=='quantum'}
    quantum=active_quantum(q,worker_names);require(not quantum,"Existing quantum guard/worker must not be duplicated")
    return ledger,{'observed_at_utc':utc(),'redacted_process_scan':scan,'active_quantum':quantum}


@contextmanager
def followup_lock():
    file=ROOT/'.runtime/research4-cloud-followup-20261106.lock'
    file.parent.mkdir(parents=True,exist_ok=True)
    require(not file.is_symlink(),"Symlink follow-up queue lock rejected")
    with file.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def materialize(argv, output, results):
    actual=[]
    for token in argv:
        if token=='{output}': actual.append(str(output));continue
        match=TOKEN.fullmatch(token)
        if match:
            value=results[match.group(2)]
            require(sha(path(value['path']))==value['sha256'],"Previously sealed result changed")
            actual.append(value['path'] if match.group(1)=='result' else value['sha256'])
        else: actual.append(token)
    return actual


def qualify_result(job, output):
    result=read(output/'result.json')
    scientific_require(result.get('schema')==job['result_schema'] and result.get('status') in job['success_statuses'],"Scientific result not in pre-approved successful scope")
    if job['kind']=='analysis':
        scientific_require(result.get('new_quantum_evaluations')==0,"Analysis result must explicitly report zero quantum evaluations")
    for dotted,expected in job.get('required_result_fields',{}).items():
        value=result
        for key in dotted.split('.'):
            scientific_require(isinstance(value,dict) and key in value,"Missing scientific qualification: "+dotted)
            value=value[key]
        scientific_require(value==expected,"Scientific qualification failed: "+dotted)
    return result


def quantum_seal(job, output, phase, log, guard, *, require_scientific_release=True):
    result=read(output/'result.json')
    ledger=read(LEDGER);guard.validate_ledger(ledger)
    rows=[r for r in ledger['phases'] if r['phase']==phase]
    require(len(rows)==1 and guard.healthy_closed_row(rows[0]),"Actual successful guard closure required")
    row=rows[0];proof_path=path(row['supervisor_proof_path']);proof=read(proof_path)
    require(sha(proof_path)==row['supervisor_proof_sha256'] and path(proof['ledger_path'])==LEDGER and proof['phase']==phase and proof['supervisor_script_sha256']==GUARD_SHA,"Guard proof/closed row binding mismatch")
    require(path(row['log'])==log and sha(log)==row['log_sha256'],"Actual closed log changed")
    source=result.get('source_worker',{});supervision=result.get('resource_supervision',{})
    require(path(source.get('path',''))==checked(job['worker']) and source.get('sha256')==job['worker']['sha256'],"Result worker provenance changed")
    require(supervision.get('proof')==proof and supervision.get('proof_sha256')==sha(proof_path) and supervision.get('live_parent_process_and_source_verified') is True,"Real live worker supervision verification required")
    require(path(result.get('output_directory',''))==output and '--output' in row['command'] and path(row['command'][row['command'].index('--output')+1])==output,"Actual output/command binding mismatch")
    if require_scientific_release:qualify_result(job,output)
    return {'result':{'path':str(output/'result.json'),'sha256':sha(output/'result.json')},'artifacts':output_artifacts(output),'phase':phase,'closed_row_sha256':hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest(),'proof':{'path':str(proof_path),'sha256':sha(proof_path)},'log':{'path':str(log),'sha256':sha(log)},'actual_cpu_seconds':row['actual_cpu_seconds']}


def close_optional_scientific_partial(state,job,current,seal,result,reason,publish):
    require(job.get('optional',False),"Mandatory scientific result requires root review")
    require(job['id'] not in state['results'],"Partial scientific output must never have been released")
    current.update(status='PARTIAL_SCIENTIFIC_CLOSED',transport_seal=seal,
                   actual_scientific_status=result.get('status'),actual_scientific_schema=result.get('schema'),
                   scientific_release=False,scientific_qualification_reason=str(reason),ended_at_utc=utc())
    current.pop('_last_sample',None)
    publish('OPTIONAL_SCIENTIFIC_PARTIAL_CLOSED_NO_RESULT_RELEASE',{'job':job['id'],'actual_status':result.get('status'),'scientific_release':False})
    return None


def retryable_closed_resource_failure(guard, ledger, phase):
    guard.validate_ledger(ledger)
    rows=[r for r in ledger['phases'] if r['phase']==phase]
    if len(rows)!=1 or rows[0].get('status')=='RUNNING':return False
    row=rows[0]
    # Require an explicit recorded guard reason, not a missing PID, exception,
    # unexplained SIGKILL, failed scientific gate, or merely nonzero exit code.
    return (row.get('status')=='PARTIAL_OR_FAILED' and isinstance(row.get('actual_cpu_seconds'),(int,float))
            and row.get('resource_stop_reason') in ('ACTUAL_CPU_GUARD','PROCESS_GROUP_RSS_GUARD','CGROUP_MEMORY_IMMEDIATE_KILL','WALL_CLEANUP_WINDOW')
            and any(row.get(key) is True for key in ('wall_terminated','cpu_terminated','memory_terminated'))
            and not row.get('residual_processes_require_review_after_main_payload_exit')
            and not row.get('monitor_failed'))


def launch_job(args,q,guard,config,grant,state,job,publish,end):
    work=path(config['work_directory']);old=state['jobs'].get(job['id'],{})
    if old.get('status')=='DONE_SEALED':
        require(sha(path(old['seal']['result']['path']))==old['seal']['result']['sha256'],"Previously successful result changed")
        if job['kind']=='quantum':
            actual=quantum_seal(job,path(old['output']),old['phase'],path(old['log']),guard)
            require(actual==old['seal'],"Do not silently re-seal changed successful dependencies")
        require(output_artifacts(path(old['output']))==old['seal']['artifacts'],"Successful scientific artifacts changed")
        require(sha(path(old['log']))==old['seal']['log']['sha256'],"Successful log changed")
        return old['seal']['result']
    if old.get('status') in ('FAILED_CLOSED','PARTIAL_SCIENTIFIC_CLOSED','BLOCKED_DEPENDENCY_NOT_SUCCESS','BLOCKED_RESOURCE_FEASIBILITY') and job.get('optional',False):return None
    require(not old,"Failed/unresolved job requires a fresh reviewed recovery configuration")
    dependencies=set(job.get('depends_on',[])) | {m.group(2) for t in job['payload_argv'] if (m:=TOKEN.fullmatch(t))}
    if not dependencies.issubset(state['results']):
        state['jobs'][job['id']]={'status':'BLOCKED_DEPENDENCY_NOT_SUCCESS'};publish('JOB_BLOCKED_DEPENDENCY',{'job':job['id']});return None
    count=2 if job.get('retry_closed_resource_failure') else 1
    for index in range(count):
        ledger,observation=no_open_quantum(q,guard,config,grant)
        remaining=ledger['remaining_quantum_cpu_seconds'];required=config['minimum_remaining_cpu_seconds']
        require(remaining>=required,"Fixed pool below explicit minimum remaining CPU")
        require(shutil.disk_usage(ROOT).free>=MIN_DISK,"Less than200MiB disk available")
        request=job if index==0 else job['retry_request']
        if job['kind']=='quantum' and remaining-config['CPU_reserve_seconds']<request['cpu_seconds']:
            state['jobs'][job['id']]={'status':'BLOCKED_RESOURCE_FEASIBILITY','remaining_cpu_seconds':remaining};publish('JOB_BLOCKED_RESOURCE_FEASIBILITY',{'job':job['id'],'remaining':remaining});return None
        require(time.time()+request['wall_seconds']+120<end,"Healthy boundary before a stage whose wall ceiling does not fit invocation")
        label=f"{job['id']}-attempt{index+1:03d}";output=work/'science'/label;log=work/'execution'/(label+'.log')
        require(not output.exists() and not log.exists(),"Preserve all earlier attempts/output/logs")
        command=materialize(job['payload_argv'],output,state['results'])
        phase=config['phase_prefix']+'-'+label
        current={'status':'RUNNING','attempt':index+1,'output':str(output),'log':str(log),'phase':phase,'prelaunch_process_observation':observation,'payload_argv':command,'started_at_utc':utc()}
        state['jobs'][job['id']]=current;publish('JOB_MATERIALIZED_BEFORE_LAUNCH',{'job':job['id'],'attempt':index+1})
        log.parent.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,PYTHONPATH=os.pathsep.join(str(ROOT/p) for p in config['pythonpath']),PYTHONUNBUFFERED='1')
        env.update({k:'2' if job['kind']=='quantum' else '1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
        if job['kind']=='quantum':
            guarded=[str(ROOT/config['python_executable']),str(GUARD),'run','--ledger',str(LEDGER),'--phase',phase,'--log',str(log),'--cpu-seconds',str(request['cpu_seconds']),'--wall-seconds',str(request['wall_seconds']),'--threads','2','--memory-gib','24','--',*command]
            console=work/'execution'/(label+'.guard-console.log')
            with console.open('x') as handle:
                child=subprocess.Popen(guarded,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                current['guardian_identity']=q.proc_identity(child.pid);publish('ACTUAL_GUARD_STARTED',{'job':job['id'],'identity':current['guardian_identity']})
                while child.poll() is None:
                    require(time.time()<end,"Invocation wall expired with guardian unresolved; do not retry")
                    if time.monotonic()-current.get('_last_sample',0)>=30:
                        current['_last_sample']=time.monotonic()
                        observed=q.proc_identity(child.pid)
                        require(q.same_identity(current['guardian_identity'],observed),"Guardian PID identity changed")
                        current['latest_guard_identity']=observed;publish('OWNED_GUARD_ACTIVE',{'job':job['id'],'identity':observed})
                    time.sleep(1)
                current['guard_returncode']=child.returncode
            # Never infer payload closure from parent exit; validate the ledger.
            closed_ledger=read(LEDGER);guard.validate_ledger(closed_ledger)
            require(not any(r['status']=='RUNNING' for r in closed_ledger['phases']),"Guard returned with unresolved quantum phase")
            if child.returncode==0:
                seal=quantum_seal(job,output,phase,log,guard,require_scientific_release=False)
                try:qualify_result(job,output)
                except ScientificQualificationError as error:
                    if job.get('optional',False):
                        return close_optional_scientific_partial(state,job,current,seal,read(output/'result.json'),error,publish)
                    raise
            else:
                rows=[r for r in closed_ledger['phases'] if r['phase']==phase]
                require(len(rows)==1 and isinstance(rows[0].get('actual_cpu_seconds'),(int,float)),"Actual failed phase cost must be closed before continuation")
                current.update(status='FAILED_CLOSED',ended_at_utc=utc(),actual_cpu_seconds=rows[0]['actual_cpu_seconds'],resource_stop_reason=rows[0].get('resource_stop_reason'),scientific_release=False);state.setdefault('failed_attempts',[]).append(dict(current));publish('ACTUAL_FAILED_ATTEMPT_PRESERVED',{'job':job['id'],'phase':phase,'actual_cpu_seconds':current['actual_cpu_seconds']})
                if index==0 and count==2 and retryable_closed_resource_failure(guard,closed_ledger,phase):
                    no_open_quantum(q,guard,config,grant);publish('BOUNDED_RESOURCE_CLOSED_RETRY_AUTHORIZED',{'job':job['id']});continue
                if job.get('optional',False):return None
                raise RuntimeError('Closed quantum failure requires root review: '+job['id'])
        else:
            def limits():
                resource.setrlimit(resource.RLIMIT_CPU,(request['cpu_seconds'],request['cpu_seconds']+2))
                resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2)
                resource.setrlimit(resource.RLIMIT_CORE,(0,0))
            with log.open('x') as handle:
                proc=subprocess.run(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,timeout=request['wall_seconds'],preexec_fn=limits)
            require(proc.returncode==0,"Bounded analysis failed; preserve output")
            seal={'result':{'path':str(output/'result.json'),'sha256':sha(output/'result.json')},'artifacts':output_artifacts(output),'log':{'path':str(log),'sha256':sha(log)},'new_quantum_evaluations':0}
            try:qualify_result(job,output)
            except ScientificQualificationError as error:
                if job.get('optional',False):
                    return close_optional_scientific_partial(state,job,current,seal,read(output/'result.json'),error,publish)
                raise
        current.update(status='DONE_SEALED',seal=seal,ended_at_utc=utc());current.pop('_last_sample',None)
        state['results'][job['id']]=seal['result'];publish('JOB_DONE_SEALED',{'job':job['id'],'result':seal['result']})
        return seal['result']
    return None


def run(args,q,guard,config,grant):
    work=path(config['work_directory']);work.mkdir(parents=True,exist_ok=True)
    with followup_lock():
        state=read(config['state']) if path(config['state']).exists() else {'schema':STATE_SCHEMA,'config_sha256':args.config_sha256,'grant_sha256':args.grant_sha256,'invocations':[],'jobs':{},'results':{}}
        require(state['config_sha256']==args.config_sha256 and state['grant_sha256']==args.grant_sha256,"Checkpoint belongs to another immutable plan")
        require(all(r['status'] in ('HEALTHY_BOUNDARY_CLOSED','COMPLETED','COMPLETED_WITH_BLOCKED_JOBS') for r in state['invocations']),"Unresolved/failed parent requires explicit root forensics")
        require(len(state['invocations'])<grant['max_invocations'] and args.attempt not in [r['id'] for r in state['invocations']],"Finite fresh invocation required")
        inv={'id':args.attempt,'status':'RUNNING','identity':q.proc_identity(os.getpid()),'started_at_utc':utc()};state['invocations'].append(inv)
        end=min(time.time()+grant['wall_seconds_per_invocation'],queue_deadline(config).timestamp())
        def publish(event,details):
            state['last_event']=event;state['updated_at_utc']=utc();inv['own_CPU_seconds']=time.process_time();atomic(config['state'],state)
            record={'utc':utc(),'event':event,'details':details,'own_CPU_seconds':time.process_time(),'own_RSS_bytes':q.own_rss()}
            file=path(config['monitor']);file.parent.mkdir(parents=True,exist_ok=True)
            with file.open('a') as handle:handle.write(json.dumps(record,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())
            print(json.dumps(record),flush=True)
        try:
            no_open_quantum(q,guard,config,grant);publish('FINITE_FOLLOWUP_STARTED',{})
            with q.self_limits(grant):
                for job in config['jobs']:
                    require(time.process_time()<grant['launcher_cpu_seconds_per_invocation']-5 and q.own_rss()<grant['launcher_rss_stop_bytes'],"Launcher own resource boundary")
                    validate(config,grant,args.config_sha256)
                    for ref in config.get('immutable_inputs',[]):checked(ref)
                    if time.time()+job['wall_seconds']+120>=end:
                        inv['status']='HEALTHY_BOUNDARY_CLOSED';publish('HEALTHY_BOUNDARY_NO_NEW_Q',{'next_job':job['id']});return
                    launch_job(args,q,guard,config,grant,state,job,publish,end)
            no_open_quantum(q,guard,config,grant)
            inv['status']='COMPLETED' if all(r.get('status')=='DONE_SEALED' for r in state['jobs'].values()) else 'COMPLETED_WITH_BLOCKED_JOBS'
            publish('FINITE_FOLLOWUP_CLOSED',{'status':inv['status'],'not_all_paper_experiments_complete':True})
        except BaseException as error:
            inv.update(status='BLOCKED_REQUIRES_ROOT_REVIEW',error_type=type(error).__name__,reason=str(error));publish('FOLLOWUP_BLOCKED_PRESERVED',{'error_type':type(error).__name__,'reason':str(error)})
            raise
        finally:
            inv['ended_at_utc']=utc();publish('FOLLOWUP_INVOCATION_ENDED',{})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','config-sha256','grant','grant-sha256'):p.add_argument('--'+key,required=True)
    p.add_argument('--attempt',default='PREPARATION_ONLY');p.add_argument('--execute',action='store_true');p.add_argument('--detach',action='store_true')
    args=p.parse_args();config=read(args.config,args.config_sha256);grant=read(args.grant,args.grant_sha256);validate(config,grant,args.config_sha256)
    require(SAFE_ID.fullmatch(args.attempt),"Safe fresh invocation ID required")
    q,guard=modules()
    if not args.execute:
        ledger,obs=no_open_quantum(q,guard,config,grant)
        require(ledger['remaining_quantum_cpu_seconds']>=config['minimum_remaining_cpu_seconds'],"Insufficient available CPU for approved queue preflight")
        require(shutil.disk_usage(ROOT).free>=MIN_DISK,"Less than200MiB disk available")
        print(json.dumps({'status':'PREFLIGHT_ONLY_NOT_RUN','jobs':len(config['jobs']),'remaining_CPU_seconds':ledger['remaining_quantum_cpu_seconds'],'deadline':config['deadline_utc'],'process_observation':obs}));return
    require(datetime.now(timezone.utc)<queue_deadline(config),"Queue deadline expired")
    if args.detach:
        work=path(config['work_directory']);work.mkdir(parents=True,exist_ok=True)
        receipt=work/('detached-launch-'+args.attempt+'.json');console=work/('detached-'+args.attempt+'.console.log')
        require(not receipt.exists() and not console.exists(),"Preserve all earlier detach receipts/consoles")
        no_open_quantum(q,guard,config,grant)
        command=[str(ROOT/config['python_executable']),str(Path(__file__).resolve()),'--config',str(path(args.config)),'--config-sha256',args.config_sha256,'--grant',str(path(args.grant)),'--grant-sha256',args.grant_sha256,'--attempt',args.attempt,'--execute']
        with console.open('x') as handle:child=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        value={'status':'ACTUAL_DETACHED_PARENT_LAUNCHED_NOT_SCIENCE_COMPLETE','launched_at_utc':utc(),'identity':q.proc_identity(child.pid),'driver_sha256':sha(__file__),'config_sha256':args.config_sha256,'grant_sha256':args.grant_sha256,'console':str(console),'machine_shutdown_survival_not_guaranteed':True,'no_unbounded_service_or_retries':True}
        atomic(receipt,value);atomic(work/'current-launch.json',value);print(json.dumps(value));return
    run(args,q,guard,config,grant)


if __name__=='__main__':main()
