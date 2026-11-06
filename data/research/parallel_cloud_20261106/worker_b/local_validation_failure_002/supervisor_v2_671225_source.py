"""Portable, finite Worker-B supervisor with an independent, non-transferable budget.

Default commands only check/prepare metadata. No A ledger or A host proof is used.
All executed payloads are single TGIDs (native threads allowed), in new sessions;
unexpected subprocess trees close as BLOCKED rather than inventing CPU totals.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SELF=Path(__file__).resolve()
DATA=ROOT/'data/research/parallel_cloud_20261106/worker_b'
RELEASE=DATA/'release_ROOT_APPROVED_v2.json'
MANIFEST_SCHEMA='research4-parallel-b-manifest-v1'
AUTH_SCHEMA='research4-parallel-b-authorization-v1'
PROOF_SCHEMA='research4-parallel-b-resource-proof-v1'
LEDGER_SCHEMA='research4-parallel-b-cpu-budget-v1'
DEADLINE='2026-11-09T08:25:17.730193+00:00'
BRANCH_PREFIX='research/parallel-worker-b-20261106-'
SAFE=re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,59}\Z')
TOKEN=re.compile(r'\{(result|sha256):([A-Za-z0-9_-]+)\}\Z')

def require(value,message):
    if not value:raise ValueError(message)

def utc():return datetime.now(timezone.utc).isoformat()

def sha(file):
    value=hashlib.sha256()
    with Path(file).open('rb') as f:
        for data in iter(lambda:f.read(1024**2),b''):value.update(data)
    return value.hexdigest()

def path(value):
    p=Path(value);p=p.resolve() if p.is_absolute() else (ROOT/p).resolve()
    require(p.is_relative_to(ROOT),'Portable files must remain in this checkout')
    return p

def python_path(value):
    # A real venv's bin/python normally is a symlink to the system interpreter.
    # Pin the dedicated selector lexically without pretending its target is a
    # repository data file or replacing it with the A interpreter.
    p=Path(value);p=Path(os.path.abspath(p if p.is_absolute() else ROOT/p))
    require(p.is_relative_to(ROOT/'.runtime'),'Dedicated B venv selector must be under .runtime')
    return p

def read(file,expected=None):
    file=path(file);require(file.stat().st_size<=2*1024**2,'Metadata exceeds2MiB')
    if expected:require(sha(file)==expected,'Pinned bytes changed: '+str(file))
    return json.loads(file.read_text())

def save(file,value,exclusive=False):
    file=path(file);file.parent.mkdir(parents=True,exist_ok=True)
    if exclusive:
        with file.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    else:
        temp=file.with_suffix(file.suffix+'.writing')
        with temp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        temp.replace(file)

def ref(file):return {'path':str(path(file).relative_to(ROOT)),'sha256':sha(path(file))}

def checked(value):
    p=path(value['path']);require(sha(p)==value['sha256'],'Source/input changed: '+str(p));return p

def identity(pid):
    base=Path('/proc')/str(pid);first=(base/'stat').read_text();fields=first.rsplit(')',1)[1].split()
    uid=next(x for x in (base/'status').read_text().splitlines() if x.startswith('Uid:')).split()[1:]
    again=(base/'stat').read_text().rsplit(')',1)[1].split()
    require(fields[19]==again[19] and len(uid)==4 and len(set(uid))==1,'Unstable process identity')
    return {'pid':int(pid),'start_ticks':int(fields[19]),'uid':int(uid[0]),'pgrp':int(fields[2]),'session':int(fields[3]),'state':fields[0],
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'CPU_seconds':sum(int(fields[k]) for k in (11,12,13,14))/os.sysconf('SC_CLK_TCK'),
            'rss_bytes':max(0,int(fields[21]))*os.sysconf('SC_PAGE_SIZE')}

def same(a,b):return all(a.get(k)==b.get(k) for k in ('pid','start_ticks','uid','pgrp','session','boot_id'))

def session_members(expected):
    result=[]
    for base in Path('/proc').iterdir():
        if not base.name.isdigit():continue
        try:
            fields=(base/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])!=expected['pgrp'] and int(fields[3])!=expected['session']:continue
            value=identity(int(base.name))
            if value['state'] not in ('Z','X','x'):result.append(value)
        except (FileNotFoundError,ProcessLookupError):pass
    return result

def no_external_scientific_processes(args):
    conflicts=[]
    for base in Path('/proc').iterdir():
        if not base.name.isdigit() or int(base.name)==os.getpid():continue
        try:
            observed=identity(int(base.name))
            if observed['uid']!=os.getuid() or observed['state'] in ('Z','X','x'):continue
            tokens=[x.decode(errors='replace') for x in (base/'cmdline').read_bytes().split(b'\0') if x]
            if not tokens or not Path(tokens[0]).name.startswith(('python','pypy')):continue
            names=[Path(t).name for t in tokens[1:] if t.endswith('.py')]
            scientific=[n for n in names if any(k in n for k in ('pyscf','quantum','parallel_b_supervisor','parallel_b_quantum','cloud_orbital_stability','cloud_small_molecule'))]
            if not scientific:continue
            # The short-lived, exact same-manifest detach caller is metadata,
            # not another computational parent; every other scientific process
            # blocks a new launch without receiving any signal.
            own_detacher=(observed['pid']==os.getppid() and SELF.name in names and '--detach' in tokens
                          and '--manifest-sha256' in tokens and tokens[tokens.index('--manifest-sha256')+1]==args.manifest_sha256)
            if own_detacher:continue
            require(same(observed,identity(observed['pid'])),'Unstable external scientific process identity')
            conflicts.append({'identity':observed,'script_basenames':scientific,'full_argv_logged':False})
        except (FileNotFoundError,ProcessLookupError):continue
    require(not conflicts,'Existing scientific process blocks duplicate/competing computation')
    return {'checked_at_utc':utc(),'conflicts':conflicts}

@contextmanager
def exclusive_lock(file):
    file=path(file);file.parent.mkdir(parents=True,exist_ok=True);require(not file.is_symlink(),'Symlink lock rejected')
    with file.open('a') as handle:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);yield

def dotted(value,key):
    for piece in key.split('.'):
        require(isinstance(value,dict) and piece in value,'Missing result assertion: '+key);value=value[piece]
    return value

def checked_owner_claim(run_id,release_sha,work):
    claim_file=work/'owner_claim.json';marker_file=ROOT/'.runtime/worker-b-ownership'/(run_id+'.json')
    claim=read(claim_file);marker=read(marker_file)
    for key in ('run_id','nonce','release_sha256','source_commit'):
        require(claim.get(key) is not None and marker.get(key)==claim[key],'B ownership marker/remote claim mismatch: '+key)
    require(claim['run_id']==run_id and claim['release_sha256']==release_sha and isinstance(claim['nonce'],str) and len(claim['nonce'])>=16,'Exact root release/run ownership required')
    require(isinstance(claim['source_commit'],str) and re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}',claim['source_commit']),'Actual Git source commit identity required')
    require(marker.get('remote_verified') is True and marker.get('branch')==BRANCH_PREFIX+run_id,'Actual remote B branch claim verification required')
    require(marker.get('repository_root')==str(ROOT),'Local B owner marker belongs to another checkout')
    return {'claim':ref(claim_file),'local_marker_sha256':sha(marker_file),'source_commit':claim['source_commit'],'nonce_sha256':hashlib.sha256(claim['nonce'].encode()).hexdigest()}

def local_validation_carry(manifest):
    prior=manifest.get('local_validation_prior_cost')
    if not prior:return None
    require(manifest.get('local_smoke_test_without_remote_B_claim') is True and manifest['cpu_cap_seconds']==1200,'Only the explicitly root-pinned local validation may carry prior local costs')
    release=read(checked(manifest['root_release']))
    require(release.get('local_validation_prior_cost')==prior,'Historical cost carry must be an exact ROOT release field')
    ledger_path=checked(prior['ledger_ref']);old=read(ledger_path);old_manifest=read(checked(prior['manifest_ref']))
    require(old.get('schema')==LEDGER_SCHEMA and old_manifest.get('schema')==MANIFEST_SCHEMA and old_manifest.get('local_smoke_test_without_remote_B_claim') is True,'Never import A or remote-B accounting into local recovery')
    require(old['cpu_cap_seconds']==old_manifest['cpu_cap_seconds']==1200 and old.get('prior_A_budget_or_proof_reused') is False and old_manifest['upload']['enabled'] is False and all(j['case_id']=='water' for j in old_manifest['jobs']),'Only a same-envelope local water validation may be carried')
    require(old['manifest_sha256']==prior['manifest_ref']['sha256'] and old_manifest['run_id']==old['run_id'] and old_manifest['allocation_ref']==manifest['allocation_ref'],'Historical run/manifest/allocation binding differs')
    require(not old.get('blocked_unknown_cpu_or_process_tree') and not old.get('initial_consumed_cpu_seconds') and not old.get('carried_local_validation'),'This explicit recovery accepts one original closed local validation, not recursive or unknown costs')
    require(old['phases'] and all(r.get('status') in ('EXITED','FAILED_CLOSED') and r.get('CPU_accounting')=='ACTUAL_WAIT4_SELF_AND_WAITED_CHILDREN_SINGLE_TGID_ENFORCED' and isinstance(r.get('actual_cpu_seconds'),(int,float)) and math.isfinite(r['actual_cpu_seconds']) and r['actual_cpu_seconds']>=0 for r in old['phases']),'Every historical phase must have real closed wait4 accounting; RUNNING or unknown is blocked')
    actual=sum(r['actual_cpu_seconds'] for r in old['phases']);declared=prior['actual_closed_CPU']
    require(type(declared) in (int,float) and math.isfinite(declared) and 0<=declared<1200 and math.isclose(actual,declared,rel_tol=1e-12,abs_tol=1e-9) and math.isclose(old['consumed_cpu_seconds'],actual,abs_tol=1e-9) and math.isclose(old['remaining_cpu_seconds'],1200-actual,abs_tol=1e-9),'No failed CPU cost may be dropped, refunded or fabricated')
    bundle=ledger_path.parent
    require(sha(bundle/'supervisor_c2cb60_source.py')==old['supervisor_sha256'],'Archived actual old supervisor source bytes differ')
    for row in old['phases']:
        # The v1 failure bundle contains one actual phase; any broader recovery
        # needs a separate reviewed mapping rather than guessed proof filenames.
        require(len(old['phases'])==1,'Explicit one-phase local failure bundle required')
        proof=read(bundle/'payload.proof.json');log=bundle/'payload.log'
        require(sha(bundle/'payload.proof.json')==row['proof_sha256'] and sha(log)==row['log_sha256'],'Historical actual proof/log bytes differ')
        require(proof.get('schema')==PROOF_SCHEMA and proof['phase']==row['phase'] and proof['case_binding_sha256']==row['case_binding_sha256'] and proof['manifest_sha256']==prior['manifest_ref']['sha256'] and proof['supervisor_source_sha256']==old['supervisor_sha256'],'Historical producer proof/source/config binding differs')
        jobs=[j for j in old_manifest['jobs'] if j['case_id']==row['case_id'] and j['operation']==row['operation']]
        require(len(jobs)==1 and hashlib.sha256(json.dumps(jobs[0],sort_keys=True).encode()).hexdigest()==row['case_binding_sha256'] and proof['input_manifest_sha256']==old_manifest['input_manifest']['sha256'],'Historical actual case/input binding differs')
    return {'schema':'research4-root-pinned-local-validation-carry-v1','ledger_ref':prior['ledger_ref'],'manifest_ref':prior['manifest_ref'],'actual_closed_CPU':actual,'historical_supervisor_sha256':old['supervisor_sha256'],'scope':'HISTORICAL_CLOSED_LOCAL_COST_NOT_A_LIVE_V2_PROOF','no_failed_cost_refund':True,'no_budget_transfer_from_A':True}

def register_allocation_claim(manifest,manifest_sha):
    role='local-validation' if manifest.get('local_smoke_test_without_remote_B_claim') else 'remote-B'
    registry=ROOT/'.runtime/parallel-b-allocation-registry'
    claim_path=registry/(manifest['allocation_ref']['sha256']+'-'+role+'.json')
    claim={'allocation_ref':manifest['allocation_ref'],'role':role,'manifest_sha256':manifest_sha,'ledger':manifest['ledger'],'run_id':manifest['run_id']}
    if not claim_path.exists():save(claim_path,claim,exclusive=True);return
    old_claim=read(claim_path)
    if old_claim==claim:return
    carry=local_validation_carry(manifest)
    require(role=='local-validation' and carry,'This allocation already belongs to another run; no new CPU allowance')
    old_manifest=read(checked(carry['manifest_ref']))
    expected_old={'allocation_ref':old_manifest['allocation_ref'],'role':'local-validation','manifest_sha256':carry['manifest_ref']['sha256'],'ledger':old_manifest['ledger'],'run_id':old_manifest['run_id']}
    require(old_claim==expected_old,'Root recovery does not bind the actual original allocation claim')
    recovery={**claim,'predecessor_claim':ref(claim_path),'carried_local_validation':carry,'not_a_new_allocation':True}
    recovery_path=registry/(manifest['allocation_ref']['sha256']+'-local-recovery-'+manifest['run_id']+'.json')
    if recovery_path.exists():require(read(recovery_path)==recovery,'A changed recovery claim cannot silently reallocate CPU')
    else:save(recovery_path,recovery,exclusive=True)

def validate(manifest,auth,manifest_sha):
    require(manifest.get('schema')==MANIFEST_SCHEMA and auth.get('schema')==AUTH_SCHEMA,'Explicit portable B schemas required')
    require(auth.get('enable_execution') is True and auth.get('manifest_sha256')==manifest_sha and auth.get('supervisor_sha256')==sha(SELF),'Explicit B authorization/source/config binding required')
    require(auth.get('no_budget_transfer_from_A') is True and manifest.get('no_budget_transfer_from_A') is True,'Never copy A allocation/proofs')
    run=manifest['run_id'];require(isinstance(run,str) and SAFE.fullmatch(run),'Safe unique run ID required')
    work=path(manifest['work_directory']);require(work==ROOT/'reports/parallel-cloud-20261106/worker-b'/run,'Narrow per-run B namespace required')
    require(path(manifest['ledger'])==work/'cpu_budget_B.json','Unique independent B ledger required')
    cap=manifest['cpu_cap_seconds'];require(type(cap) is int and 300<=cap<=60000 and auth.get('cpu_cap_seconds')==cap,'Explicit finite new B allocation <=60000CPU')
    allocation=read(checked(manifest['allocation_ref']));require(auth.get('allocation_ref')==manifest['allocation_ref'] and allocation.get('schema')=='research4-two-account-resource-allocation-v1' and allocation.get('root_approved') is True,'Actual B envelope allocation binding required')
    allocated=allocation['new_B_envelope']['local_package_validation_reserved_CPU_seconds' if manifest.get('local_smoke_test_without_remote_B_claim') else 'actual_second_account_CPU_cap_seconds']
    require(cap<=allocated and allocation['new_B_envelope']['maximum_additional_CPU_seconds']==60000,'Local and remote B allocations may not be copied or combined')
    deadline=datetime.fromisoformat(manifest['deadline_utc']);require(deadline.tzinfo and deadline<=datetime.fromisoformat(DEADLINE) and auth.get('deadline_utc')==manifest['deadline_utc'],'No later deadline or automatic extension')
    require(manifest.get('threads') in (1,2) and 2<=manifest.get('memory_gib',0)<=24 and 1<=manifest.get('rss_stop_gib',0)<manifest['memory_gib'],'Finite thread/address-space/RSS policy required')
    require(125<=manifest.get('reserve_cpu_seconds',0)<=2000,'Fixed reserve required')
    checked(manifest['input_manifest'])
    sources={str(checked(x)):x['sha256'] for x in manifest['source_refs']}
    require(sources.get(str(SELF))==sha(SELF),'Pin actual B supervisor')
    for value in manifest.get('input_refs',[]):checked(value)
    python=python_path(manifest['python_executable'])
    require(manifest['upload']['branch']==BRANCH_PREFIX+run,'Exact per-run B branch required')
    require(type(manifest['upload'].get('enabled')) is bool,'Explicit upload setting required')
    if not manifest.get('local_smoke_test_without_remote_B_claim'):
        release_ref=manifest['root_release'];checked(release_ref)
        owner=checked_owner_claim(run,release_ref['sha256'],work)
        require(owner==manifest.get('ownership'),'The same verified B ownership must precede every real payload')
    require(auth.get('local_validation_prior_cost')==manifest.get('local_validation_prior_cost'),'Historical local carry must also be explicitly authorization-bound')
    local_validation_carry(manifest)
    jobs=manifest['jobs'];require(isinstance(jobs,list) and 1<=len(jobs)<=16,'Finite jobs required')
    if manifest.get('local_smoke_test_without_remote_B_claim'):
        require(cap<=1200 and manifest['upload']['enabled'] is False and all(j.get('case_id')=='water' for j in jobs),'Local smoke is only water/no-upload/<=1200CPU and never remote B')
    earlier=set()
    for job in jobs:
        require(isinstance(job['id'],str) and SAFE.fullmatch(job['id']) and job['id'] not in earlier,'Safe unique job ID required')
        require(job['case_id'] in ('water','protocatechuic_acid','calycosin') and job['operation'] in ('prepare','optimize','frequency','analyze'),'Reviewed cases/operations only')
        worker=checked(job['worker']);require(sources.get(str(worker))==job['worker']['sha256'],'Explicit pinned worker required')
        require(job.get('input_manifest_sha256')==manifest['input_manifest']['sha256'],'Case manifest source binding required')
        require(set(job.get('depends_on',[])).issubset(earlier),'Only prior dependencies; no cycles')
        argv=job['payload_argv'];require(isinstance(argv,list) and all(isinstance(t,str) and '\x00' not in t for t in argv),'Literal argv required')
        require(python_path(argv[0])==python and str(worker) in [str(path(t)) for t in argv if t.endswith('.py')],'Pinned Python/worker only')
        require(argv.count('--output')==1 and argv[argv.index('--output')+1]=='{output}','Fresh output placeholder required')
        for value in argv:
            if '{' in value or '}' in value:
                m=TOKEN.fullmatch(value);require(value=='{output}' or (m and m.group(2) in earlier),'Only sealed earlier result/hash placeholders')
        require(argv[argv.index('--case')+1]==job['case_id'] and argv[argv.index('--operation')+1]==job['operation'] and '--execute' in argv,'Actual case/operation/execute argv binding required')
        for request in [job]+([job['retry_request']] if job.get('retry_closed_resource_failure') else []):
            require(type(request.get('cpu_seconds')) is int and 125<=request['cpu_seconds']<=60000 and type(request.get('wall_seconds')) is int and 61<=request['wall_seconds']<=43200,'Finite payload ceiling required')
        require(isinstance(job.get('success_statuses'),list) and job['success_statuses'] and isinstance(job.get('result_schema'),str),'Predeclared scientific success scope required')
        require(type(job.get('optional',False)) is bool and type(job.get('retry_closed_resource_failure',False)) is bool,'Finite boolean policies required')
        earlier.add(job['id'])

def validate_ledger(ledger,manifest,manifest_sha):
    require(ledger.get('schema')==LEDGER_SCHEMA and ledger.get('manifest_sha256')==manifest_sha and ledger.get('supervisor_sha256')==sha(SELF),'Wrong/source-changed B ledger')
    require(ledger.get('run_id')==manifest['run_id'] and ledger.get('cpu_cap_seconds')==manifest['cpu_cap_seconds'] and ledger.get('deadline_utc')==manifest['deadline_utc'],'B allocation or run binding changed')
    phases=ledger['phases'];require(len({p['phase'] for p in phases})==len(phases),'Duplicate phase identity')
    require(sum(p['status']=='RUNNING' for p in phases)<=1,'Multiple B live phases')
    require(all(p['status'] in ('RUNNING','EXITED','FAILED_CLOSED') for p in phases),'Unknown B phase status requires forensic review')
    closed=[p for p in phases if p['status']!='RUNNING']
    require(all(p.get('CPU_accounting')=='ACTUAL_WAIT4_SELF_AND_WAITED_CHILDREN_SINGLE_TGID_ENFORCED' and isinstance(p.get('actual_cpu_seconds'),(int,float)) and math.isfinite(p['actual_cpu_seconds']) and p['actual_cpu_seconds']>=0 for p in closed),'Unknown subtree costs block further computation')
    carry=local_validation_carry(manifest);initial=carry['actual_closed_CPU'] if carry else 0.
    require(ledger.get('initial_consumed_cpu_seconds',0.)==initial and ledger.get('carried_local_validation')==carry,'New local ledger must preserve its exact approved historical starting cost')
    used=initial+sum(p['actual_cpu_seconds'] for p in closed)
    require(math.isclose(used,ledger['consumed_cpu_seconds'],abs_tol=1e-6) and math.isclose(max(0.,manifest['cpu_cap_seconds']-used),ledger['remaining_cpu_seconds'],abs_tol=1e-6),'Never refund failed costs or enlarge remaining pool')
    require(not ledger.get('blocked_unknown_cpu_or_process_tree'),'Unresolved accounting/process tree requires forensic review')
    require(used<=manifest['cpu_cap_seconds'],'Actual B cap overrun is retained and blocks continuation')
    return ledger

def verify_b_resource_supervision(*,expected_threads,expected_worker_path=None,expected_case=None,expected_input_manifest_sha256=None,expected_operation=None):
    supplied=os.environ.get('R4_B_RESOURCE_PROOF');require(supplied,'Actual independent B supervision required')
    proof=read(supplied);require(proof.get('schema')==PROOF_SCHEMA and proof.get('supervisor_source_sha256')==sha(SELF),'Authentic B source/proof required')
    me=identity(os.getpid());parent=identity(os.getppid())
    require(same(proof['child_identity'],me) and same(proof['supervisor_identity'],parent),'Live B PID/birth/UID/boot identity mismatch')
    require(me['pid']==me['pgrp']==me['session'] and parent['pid']==os.getppid(),'Dedicated actual worker session required')
    tokens=[v.decode(errors='replace') for v in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes().split(b'\0') if v]
    require(any(t.endswith('.py') and path(t)==SELF for t in tokens),'Actual parent is not the B supervisor')
    manifest=read(proof['manifest_path'],proof['manifest_sha256']);auth=read(proof['authorization_path'],proof['authorization_sha256']);validate(manifest,auth,proof['manifest_sha256'])
    require(Path(os.path.abspath(sys.executable))==python_path(manifest['python_executable']),'Actual worker Python selector differs from independent B environment')
    ledger=validate_ledger(read(proof['ledger_path']),manifest,proof['manifest_sha256']);active=[p for p in ledger['phases'] if p['status']=='RUNNING']
    require(len(active)==1 and active[0]['phase']==proof['phase'] and active[0]['case_binding_sha256']==proof['case_binding_sha256'],'Unique B running phase/source binding required')
    require(expected_threads==proof['threads']==manifest['threads'],'B thread allocation mismatch')
    if expected_worker_path:require(path(expected_worker_path)==checked(proof['worker']),'Unexpected B worker source')
    for expected,key in ((expected_case,'case_id'),(expected_operation,'operation'),(expected_input_manifest_sha256,'input_manifest_sha256')):
        if expected is not None:require(proof.get(key)==expected,'Unexpected B case/input/operation: '+key)
    require(tuple(resource.getrlimit(resource.RLIMIT_CPU))==(proof['cpu_soft_limit_seconds'],proof['cpu_seconds']) and tuple(resource.getrlimit(resource.RLIMIT_AS))==(proof['address_space_bytes'],)*2,'Actual B kernel limits differ')
    require(time.time()<datetime.fromisoformat(proof['deadline_utc']).timestamp(),'B authorization expired')
    return {'proof_path':str(path(supplied)),'proof_sha256':sha(path(supplied)),'proof':proof,'verified_at_utc':utc(),'live_parent_process_and_source_verified':True,'independent_B_budget_not_A':True}

def materialize(argv,output,results):
    values=[]
    for index,token in enumerate(argv):
        if index==0:values.append(str(python_path(token)));continue
        if token=='{output}':values.append(str(output));continue
        match=TOKEN.fullmatch(token)
        if match:
            ref=results[match.group(2)];checked(ref);values.append(str(path(ref['path'])) if match.group(1)=='result' else ref['sha256'])
        else:values.append(str(path(token)) if token.endswith('.py') else token)
    return values

def result_seal(output,row,job):
    result=read(output/'result.json')
    require(result.get('schema')==job['result_schema'] and path(result.get('output_directory',''))==output,'Actual scientific schema/output mismatch')
    source=result.get('source_worker',{});require(checked(source)==checked(job['worker']),'Actual scientific source mismatch')
    supervision=result.get('resource_supervision',{});proof=read(row['proof_path'])
    require(sha(path(row['proof_path']))==row['proof_sha256'] and sha(path(row['log']))==row['log_sha256'],'Actual closed proof/log bytes changed')
    require(proof['phase']==row['phase'] and proof['case_binding_sha256']==row['case_binding_sha256']==hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest(),'Actual closed case configuration binding changed')
    require(supervision.get('proof')==proof and supervision.get('proof_sha256')==sha(path(row['proof_path'])) and supervision.get('live_parent_process_and_source_verified') is True,'Actual B worker proof verification missing')
    require(row['status']=='EXITED' and row['exit_code']==0 and row['result_sha256']==sha(output/'result.json'),'Real successful process closure required')
    files=[]
    for file in sorted(output.rglob('*')):
        require(not file.is_symlink(),'Symlink result rejected')
        if file.is_file():
            size=file.stat().st_size
            files.append({'relative_path':str(file.relative_to(output)),'sha256':sha(file),'size_bytes':size,'upload_eligible':file.suffix.lower() not in ('.chk','.h5','.hdf5') and size<=5*1024**2})
    return result,{'result':ref(output/'result.json'),'phase':row['phase'],'closed_row_sha256':hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest(),'files':files,'CPU_accounting':row['CPU_accounting'],'actual_cpu_seconds':row['actual_cpu_seconds']}

def water_qualified(results):
    for value in results.values():
        result=read(checked(value))
        if result.get('case')=='water' and result.get('operation')=='analyze' and result.get('status')=='MODES_ANALYZED':
            return all(result.get(k)==v for k,v in {'normal_mode_count':3,'imaginary_frequency_count':0,'all_frequencies_real_positive':True,'optimizer_converged':True,'fresh_gradient_thresholds_passed':True,'final_gradient_thresholds_passed':True,'independent_mode_replay_passed':True,'stationary_minimum_verified':True}.items())
    return False

def scientific_qualified(result,job):
    return result.get('status') in job['success_statuses'] and all(dotted(result,k)==v for k,v in job.get('required_result_fields',{}).items())

def bound_signal(expected,sig):
    try:
        observed=identity(expected['pid'])
        if not same(expected,observed) or observed['state'] in ('Z','X','x'):return False
        os.killpg(expected['pgrp'],sig);return True
    except (FileNotFoundError,ProcessLookupError):return False

def run_payload(manifest,auth,args,ledger,job,attempt,publish):
    no_external_scientific_processes(args)
    work=path(manifest['work_directory']);label=job['id']+f'-attempt{attempt:03d}'
    output=work/'science'/label;log=work/'execution'/(label+'.log');proofpath=work/'execution'/(label+'.proof.json')
    require(not output.exists() and not log.exists() and not proofpath.exists(),'Preserve every previous attempt')
    request=job if attempt==1 else job['retry_request'];remaining=ledger['remaining_cpu_seconds']
    require(remaining>=request['cpu_seconds']+manifest['reserve_cpu_seconds'],'Insufficient independent B budget; do not launch')
    deadline=datetime.fromisoformat(manifest['deadline_utc']).timestamp();wall=min(request['wall_seconds'],math.floor(deadline-time.time())-2)
    require(wall>=61,'B deadline leaves no safe bounded phase')
    argv=materialize(job['payload_argv'],output,args.results)
    log.parent.mkdir(parents=True,exist_ok=True)
    phase=manifest['run_id']+'-'+label
    case_sha=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()
    cpu=request['cpu_seconds'];margin=min(60,max(15,cpu//10));asbytes=manifest['memory_gib']*1024**3
    supervisor=identity(os.getpid());libc=ctypes.CDLL(None,use_errno=True)
    row={'phase':phase,'case_id':job['case_id'],'operation':job['operation'],'case_binding_sha256':case_sha,'attempt':attempt,'status':'RUNNING','started_at_utc':utc(),'command':argv,'output_directory':str(output),'log':str(log),'proof_path':str(proofpath),'cpu_seconds':cpu,'wall_seconds':wall,'actual_cpu_seconds':None}
    ledger['phases'].append(row);save(manifest['ledger'],ledger)
    def limits():
        require(libc.prctl(1,signal.SIGKILL,0,0,0)==0 and os.getppid()==supervisor['pid'],'Cannot bind worker lifetime to real supervisor')
        resource.setrlimit(resource.RLIMIT_CPU,(cpu-margin,cpu));resource.setrlimit(resource.RLIMIT_AS,(asbytes,)*2);resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        proof={'schema':PROOF_SCHEMA,'supervisor_source_sha256':sha(SELF),'supervisor_identity':supervisor,'child_identity':identity(os.getpid()),'manifest_path':str(path(args.manifest)),'manifest_sha256':args.manifest_sha256,'authorization_path':str(path(args.authorization)),'authorization_sha256':args.authorization_sha256,'ledger_path':str(path(manifest['ledger'])),'phase':phase,'case_id':job['case_id'],'operation':job['operation'],'case_binding_sha256':case_sha,'input_manifest_sha256':manifest['input_manifest']['sha256'],'worker':job['worker'],'threads':manifest['threads'],'cpu_seconds':cpu,'cpu_soft_limit_seconds':cpu-margin,'actual_cpu_stop_threshold_seconds':cpu-margin,'address_space_bytes':asbytes,'rss_stop_bytes':manifest['rss_stop_gib']*1024**3,'deadline_utc':manifest['deadline_utc'],'wall_seconds':wall,'prior_A_allocation_copied':False,'single_TGID_native_threads_only':True}
        save(proofpath,proof,exclusive=True)
    env=dict(os.environ,R4_B_RESOURCE_PROOF=str(proofpath),PYTHONUNBUFFERED='1')
    # Keep session proxy/CA authentication, but never import numerical packages
    # through an inherited A PYTHONPATH/PYTHONHOME as a portable B dependency.
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    if manifest.get('local_smoke_test_without_remote_B_claim'):
        env['PYTHONPATH']=os.pathsep.join(str(path(p)) for p in manifest.get('local_validation_pythonpath',[]))
    env.update({k:str(manifest['threads']) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    started=time.monotonic();waited=None;stop_reason=None;stop_at=None;unsupported=False;peakcpu=0.;peakrss=0
    with log.open('x') as handle:
        child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits)
        expected=identity(child.pid);row['child_identity']=expected;row['supervisor_identity']=supervisor;save(manifest['ledger'],ledger);publish('ACTUAL_B_PAYLOAD_STARTED',{'phase':phase,'identity':expected})
        while True:
            value=os.wait4(child.pid,os.WNOHANG)
            if value[0]:waited=value;child.returncode=os.waitstatus_to_exitcode(value[1]);break
            try:
                observed=identity(child.pid);require(same(expected,observed),'Worker PID identity changed')
                members=session_members(expected)
                if any(not same(expected,p) for p in members):unsupported=True;stop_reason='UNSUPPORTED_SUBPROCESS_TREE_CPU_BLOCKED'
                peakcpu=max(peakcpu,observed['CPU_seconds']);peakrss=max(peakrss,observed['rss_bytes'])
                if observed['CPU_seconds']>=cpu-margin:stop_reason=stop_reason or 'ACTUAL_CPU_GUARD'
                if observed['rss_bytes']>=manifest['rss_stop_gib']*1024**3:stop_reason=stop_reason or 'RSS_GUARD'
                if time.monotonic()-started>=wall-30 or time.time()>=deadline-30:stop_reason=stop_reason or 'WALL_GUARD'
                if time.process_time()>600:stop_reason=stop_reason or 'SUPERVISOR_CPU_GUARD'
            except (FileNotFoundError,ProcessLookupError):
                # Only wait4 can establish closure. A short terminal transition
                # gets at most one second before marking monitor uncertainty.
                stop_reason=stop_reason or 'PROCESS_OBSERVATION_UNCERTAIN'
            except (OSError,ValueError):unsupported=True;stop_reason=stop_reason or 'PROCESS_MONITOR_FAILED'
            if stop_reason and stop_at is None:
                stop_at=time.monotonic();row['stop_reason']=stop_reason;row['TERM_sent']=bound_signal(expected,signal.SIGTERM);save(manifest['ledger'],ledger)
            if stop_at is not None and time.monotonic()-stop_at>=1:
                row['KILL_sent']=bound_signal(expected,signal.SIGKILL)
                if time.monotonic()-stop_at>3:
                    unsupported=True;row['unresolved_wait4']=True;break
            row.update(latest_CPU_seconds=peakcpu,peak_RSS_bytes=peakrss)
            if int(time.monotonic()-started)%30==0:save(manifest['ledger'],ledger)
            time.sleep(.1)
    if waited is None or session_members(expected):unsupported=True
    if unsupported:
        row.update(status='BLOCKED_UNACCOUNTED_PROCESS_TREE_OR_WAIT4',CPU_accounting='UNRESOLVED_NOT_ZERO',actual_cpu_lower_bound_seconds=peakcpu,ended_at_utc=utc())
        ledger['blocked_unknown_cpu_or_process_tree']=True;save(manifest['ledger'],ledger)
        raise RuntimeError('No trustworthy complete subtree CPU/closure; automatic restart forbidden')
    actual=waited[2].ru_utime+waited[2].ru_stime
    require(math.isfinite(actual) and actual>=0,'Invalid actual wait4 CPU')
    row.update(status='EXITED' if child.returncode==0 and not stop_reason else 'FAILED_CLOSED',exit_code=child.returncode,actual_cpu_seconds=actual,CPU_accounting='ACTUAL_WAIT4_SELF_AND_WAITED_CHILDREN_SINGLE_TGID_ENFORCED',elapsed_wall_seconds=time.monotonic()-started,peak_RSS_bytes=peakrss,stop_reason=stop_reason,ended_at_utc=utc(),log_sha256=sha(log),proof_sha256=sha(proofpath))
    if (output/'result.json').exists():row['result_sha256']=sha(output/'result.json')
    ledger['consumed_cpu_seconds']+=actual;ledger['remaining_cpu_seconds']=max(0.,manifest['cpu_cap_seconds']-ledger['consumed_cpu_seconds']);save(manifest['ledger'],ledger)
    require(ledger['consumed_cpu_seconds']<=manifest['cpu_cap_seconds'],'B cap overrun retained; no new work')
    publish('ACTUAL_B_PAYLOAD_CLOSED',{'phase':phase,'status':row['status'],'actual_cpu_seconds':actual,'stop_reason':stop_reason})
    return row

def load_release():
    release=read(RELEASE);require(release.get('schema')=='research4-parallel-b-release-v1' and release.get('release_enabled') is True and release.get('supervisor_sha256')==sha(SELF),'Root-reviewed B release/source required')
    require(release.get('no_budget_transfer_from_A') is True,'No A budget transfer allowed')
    require(time.time()<datetime.fromisoformat(release['deadline_utc']).timestamp(),'Original authorization expired')
    require(datetime.fromisoformat(release['deadline_utc'])<=datetime.fromisoformat(DEADLINE),'Release cannot extend the72h deadline')
    checked(release['worker']);checked(release['input_manifest'])
    for value in release['source_refs']+release.get('input_refs',[]):checked(value)
    require(any(path(x['path'])==SELF and x['sha256']==sha(SELF) for x in release['source_refs']),'Release must pin actual supervisor')
    return release

def validate_release_run_id(release,run_id,local_smoke):
    key='local_smoke_run_id' if local_smoke else 'remote_run_id'
    require(isinstance(release.get(key),str) and run_id==release[key],'This fixed allocation permits only the exact root-released '+key)

def prepare_manifest(run_id,python_executable,smoke_test=False,no_upload=False):
    require(SAFE.fullmatch(run_id),'Safe run ID required')
    release=load_release()
    validate_release_run_id(release,run_id,bool(smoke_test and no_upload))
    work=ROOT/'reports/parallel-cloud-20261106/worker-b'/run_id
    local_smoke=bool(smoke_test and no_upload)
    previous_receipt=work/'prepared-launch.json'
    if previous_receipt.exists():
        # Re-enter exactly the original prepared plan. Preparation never closes
        # a phase, refunds CPU, edits state, or treats a missing PID as success.
        old=read(previous_receipt)
        mp=path(old.get('manifest_path',old['manifest']));ap=path(old.get('authorization_path',old['authorization']))
        require(mp==work/'manifest.json' and ap==work/'authorization.json','Existing preparation points outside its original namespace')
        manifest=read(mp,old['manifest_sha256']);auth=read(ap,old['authorization_sha256']);validate(manifest,auth,old['manifest_sha256'])
        require(manifest['run_id']==run_id and manifest.get('local_smoke_test_without_remote_B_claim')==local_smoke and python_path(manifest['python_executable'])==python_path(python_executable),'Resume must keep exact run/role/Python selector')
        require(manifest['root_release']==ref(RELEASE) and auth.get('root_release')==ref(RELEASE),'A changed release needs explicit recovery, not a new allowance')
        require(manifest['cpu_cap_seconds']==(1200 if local_smoke else release['cpu_cap_seconds']) and manifest['deadline_utc']==release['deadline_utc'],'Resume cannot alter the original cap/deadline')
        ledger_path=path(manifest['ledger']);ledger=None
        if ledger_path.exists():ledger=validate_ledger(read(ledger_path),manifest,old['manifest_sha256'])
        else:
            allowed={'manifest.json','authorization.json','prepared-launch.json'} | ({'owner_claim.json'} if not local_smoke else set())
            require({p.name for p in work.iterdir()}.issubset(allowed),'No ledger with unexplained execution artifacts; preserve for forensics')
        return {**old,'status':'EXISTING_PREPARED_PLAN_REUSED_NO_NEW_ALLOCATION','existing_plan_reused':True,'actual_saved_CPU_seconds':ledger['consumed_cpu_seconds'] if ledger else 0,'live_or_missing_PID_not_interpreted_by_prepare':True}
    if local_smoke:require(not work.exists(),'Fresh local validation namespace required')
    else:
        require(work.exists() and {p.name for p in work.iterdir()}=={'owner_claim.json'},'Only the actual atomically claimed B namespace may be prepared')
        ownership=checked_owner_claim(run_id,sha(RELEASE),work)
    worker=release['worker'];checked(worker);checked(release['input_manifest'])
    cases=['water'] if smoke_test else ['water','protocatechuic_acid','calycosin'];jobs=[]
    requests={'water':[(125,600),(500,1200),(400,1200),(125,300)],'protocatechuic_acid':[(300,900),(6000,10800),(4000,7200),(125,300)],'calycosin':[(600,1200),(18000,28800),(22000,43200),(125,300)]}
    for case in cases:
        previous=None
        for operation,(cpu,wall) in zip(('prepare','optimize','frequency','analyze'),requests[case]):
            name=case+'-'+operation;deps=[] if previous is None else [previous]
            if case!='water':deps.append('water-analyze')
            argv=[python_executable,worker['path'],'--execute','--case',case,'--operation',operation,'--threads',str(release.get('threads',2)),'--maxsteps','25','--output','{output}']
            if previous:argv+=['--source-result','{result:'+previous+'}','--source-result-sha256','{sha256:'+previous+'}']
            jobs.append({'id':name,'case_id':case,'operation':operation,'worker':worker,'input_manifest_sha256':release['input_manifest']['sha256'],'payload_argv':argv,'depends_on':deps,'result_schema':'research4-parallel-b-quantum-result-v1','success_statuses':[{'prepare':'PREPARED_FORCE_FIELD_MODELS','optimize':'OPTIMIZATION_CONVERGED','frequency':'FREQUENCIES_COMPUTED','analyze':'MODES_ANALYZED'}[operation]],'required_result_fields':{},'cpu_seconds':cpu,'wall_seconds':wall,'optional':case!='water','retry_closed_resource_failure':False});previous=name
    if local_smoke:require(release.get('allow_local_smoke') is True and release.get('local_smoke_cpu_cap_seconds')==1200,'Explicit fixed local validation allocation required')
    cap=1200 if local_smoke else min(60000,release['cpu_cap_seconds']);work.mkdir(parents=True,exist_ok=not local_smoke)
    allocation_ref=next(r for r in release.get('input_refs',[]) if path(r['path']).name=='resource_allocation_ROOT_APPROVED.json')
    manifest={'schema':MANIFEST_SCHEMA,'run_id':run_id,'work_directory':str(work.relative_to(ROOT)),'ledger':str((work/'cpu_budget_B.json').relative_to(ROOT)),'python_executable':python_executable,'cpu_cap_seconds':cap,'deadline_utc':release['deadline_utc'],'reserve_cpu_seconds':125 if local_smoke else 500,'threads':release.get('threads',2),'memory_gib':release.get('memory_gib',12),'rss_stop_gib':release.get('rss_stop_gib',8),'source_refs':release['source_refs'],'input_refs':release.get('input_refs',[])+[ref(RELEASE)],'input_manifest':release['input_manifest'],'no_budget_transfer_from_A':True,'local_smoke_test_without_remote_B_claim':local_smoke,'jobs':jobs,'upload':{'enabled':not no_upload,'branch':BRANCH_PREFIX+run_id,'allowlist':release.get('upload_allowlist',['*.json','science/**/*.json','science/**/*.csv','science/**/*.xyz','science/**/*.png','science/**/*.pdf','execution/*.log','execution/*.json'])},'root_release':ref(RELEASE)}
    manifest['allocation_ref']=allocation_ref
    manifest['role']='B'
    if local_smoke and release.get('local_validation_prior_cost'):manifest['local_validation_prior_cost']=release['local_validation_prior_cost']
    if local_smoke:manifest['local_validation_pythonpath']=release.get('local_validation_pythonpath',['.runtime/quantum-pilot-site','.runtime/quantum-upgrade-site'])
    else:manifest['ownership']=ownership
    mp=work/'manifest.json';save(mp,manifest,exclusive=True)
    auth={'schema':AUTH_SCHEMA,'enable_execution':True,'manifest_sha256':sha(mp),'supervisor_sha256':sha(SELF),'cpu_cap_seconds':cap,'deadline_utc':manifest['deadline_utc'],'no_budget_transfer_from_A':True,'root_release':ref(RELEASE),'allocation_ref':allocation_ref,'user_authorization_is_not_an_A_budget_copy':True}
    if manifest.get('local_validation_prior_cost'):auth['local_validation_prior_cost']=manifest['local_validation_prior_cost']
    ap=work/'authorization.json';save(ap,auth,exclusive=True);validate(manifest,auth,sha(mp))
    receipt={'status':'PREPARED_NOT_RUN','manifest':str(mp.relative_to(ROOT)),'manifest_path':str(mp.relative_to(ROOT)),'manifest_sha256':sha(mp),'authorization':str(ap.relative_to(ROOT)),'authorization_path':str(ap.relative_to(ROOT)),'authorization_sha256':sha(ap),'run_id':run_id,'remote_worker_B_execution':'NOT_RUN','new_CPU_consumed':0};save(work/'prepared-launch.json',receipt,exclusive=True);return receipt

def run(args,manifest,auth):
    work=path(manifest['work_directory']);work.mkdir(parents=True,exist_ok=True)
    with exclusive_lock(ROOT/'.runtime/parallel-b-exclusive.lock'):
        # A new run ID is a namespace, never a new authorization or CPU refund.
        # Persist the one local-validation and one remote allocation claim on
        # this host. Cross-host duplication still requires the user's central
        # allocation record; this is not a global distributed lock service.
        register_allocation_claim(manifest,args.manifest_sha256)
        lp=path(manifest['ledger']);statep=work/'state.json'
        if lp.exists():ledger=validate_ledger(read(lp),manifest,args.manifest_sha256)
        else:
            carry=local_validation_carry(manifest);initial=carry['actual_closed_CPU'] if carry else 0.
            ledger={'schema':LEDGER_SCHEMA,'run_id':manifest['run_id'],'manifest_sha256':args.manifest_sha256,'supervisor_sha256':sha(SELF),'cpu_cap_seconds':manifest['cpu_cap_seconds'],'deadline_utc':manifest['deadline_utc'],'initial_consumed_cpu_seconds':initial,'carried_local_validation':carry,'consumed_cpu_seconds':initial,'remaining_cpu_seconds':float(manifest['cpu_cap_seconds'])-initial,'phases':[],'created_at_utc':utc(),'prior_A_budget_or_proof_reused':False};save(lp,ledger,exclusive=True)
        require(not any(p['status']=='RUNNING' for p in ledger['phases']),'Missing PID is not closure; unresolved B phase requires forensics')
        state=read(statep) if statep.exists() else {'schema':'research4-parallel-b-state-v1','manifest_sha256':args.manifest_sha256,'invocations':[],'jobs':{},'results':{}}
        require(state['manifest_sha256']==args.manifest_sha256 and len(state['invocations'])<4,'Fixed source-bound state and at most4 invocations')
        require(all(i['status'] in ('CLOSED','CLOSED_WITH_BLOCKED_JOBS') for i in state['invocations']),'Unresolved previous parent requires review')
        inv={'id':args.attempt,'status':'RUNNING','identity':identity(os.getpid()),'started_at_utc':utc()};require(args.attempt not in [i['id'] for i in state['invocations']],'Fresh invocation ID required');state['invocations'].append(inv);args.results=state['results']
        def publish(event,details):
            state['last_event']=event;state['updated_at_utc']=utc();save(statep,state)
            with (work/'monitor.jsonl').open('a') as f:f.write(json.dumps({'utc':utc(),'event':event,'details':details},allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        publish('B_QUEUE_STARTED_NO_A_LEDGER',{})
        try:
            for job in manifest['jobs']:
                validate(manifest,auth,args.manifest_sha256);validate_ledger(ledger,manifest,args.manifest_sha256)
                if state['jobs'].get(job['id'],{}).get('status')=='DONE_SEALED':
                    previous=state['jobs'][job['id']];rows=[r for r in ledger['phases'] if r['phase']==previous['phase']]
                    require(len(rows)==1,'Cached successful stage lacks unique real closed row')
                    _,actual=result_seal(path(rows[0]['output_directory']),rows[0],job)
                    require(actual==previous['seal'],'Cached artifacts/source/proof differ; never re-seal changed checkpoints')
                    checked(state['results'][job['id']]);continue
                require(job['id'] not in state['jobs'],'Never overwrite failed/unresolved case; new reviewed manifest needed')
                if not set(job.get('depends_on',[])).issubset(state['results']):state['jobs'][job['id']]={'status':'BLOCKED_DEPENDENCY'};publish('B_DEPENDENCY_BLOCKED',{'job':job['id']});continue
                if job['case_id']!='water' and job['operation'] in ('optimize','frequency') and not water_qualified(state['results']):state['jobs'][job['id']]={'status':'BLOCKED_WATER_CONTROL'};publish('B_WATER_NOT_QUALIFIED',{'job':job['id']});continue
                if ledger['remaining_cpu_seconds']<job['cpu_seconds']+manifest['reserve_cpu_seconds'] or time.time()+61>=datetime.fromisoformat(manifest['deadline_utc']).timestamp() or shutil.disk_usage(ROOT).free<200*1024**2:
                    state['jobs'][job['id']]={'status':'BLOCKED_RESOURCE_OR_DEADLINE'};publish('B_RESOURCE_BLOCKED',{'job':job['id']});continue
                maximum=2 if job.get('retry_closed_resource_failure') else 1
                for attempt in range(1,maximum+1):
                    row=run_payload(manifest,auth,args,ledger,job,attempt,publish)
                    state['jobs'][job['id']]={'status':row['status'],'phase':row['phase'],'actual_cpu_seconds':row['actual_cpu_seconds']}
                    if row['status']=='EXITED':
                        result,seal=result_seal(path(row['output_directory']),row,job)
                        if scientific_qualified(result,job):state['jobs'][job['id']].update(status='DONE_SEALED',seal=seal);state['results'][job['id']]=seal['result']
                        else:state['jobs'][job['id']].update(status='PARTIAL_SCIENTIFIC_CLOSED',actual_scientific_status=result.get('status'),transport_seal=seal,scientific_release=False)
                        break
                    if not (attempt==1 and maximum==2 and row['stop_reason'] in ('ACTUAL_CPU_GUARD','RSS_GUARD','WALL_GUARD')):break
                    require(not session_members(row['child_identity']),'No retry while any prior group process remains')
                publish('B_JOB_CLOSED',{'job':job['id'],'status':state['jobs'][job['id']]['status']})
                if not job.get('optional',False):require(state['jobs'][job['id']]['status']=='DONE_SEALED','Mandatory B control failed; preserve result and stop')
            inv['status']='CLOSED' if all(x['status']=='DONE_SEALED' for x in state['jobs'].values()) else 'CLOSED_WITH_BLOCKED_JOBS';publish('B_FINITE_QUEUE_CLOSED',{'status':inv['status'],'not_all_thesis_experiments_complete':True})
        except BaseException as error:
            inv.update(status='BLOCKED_REQUIRES_FORENSICS',error_type=type(error).__name__,reason=str(error));publish('B_QUEUE_BLOCKED_PRESERVED',{'reason':str(error)});raise
        finally:inv['ended_at_utc']=utc();publish('B_INVOCATION_ENDED',{})
        if manifest['upload']['enabled']:
            helper=ROOT/'scripts/research4_parallel_b_git_return_20261106.py';require(any(path(x['path'])==helper for x in manifest['source_refs']),'Pinned Git return helper required')
            for return_index in range(1,4):
                require(not any(p['status']=='RUNNING' for p in validate_ledger(read(manifest['ledger']),manifest,args.manifest_sha256)['phases']),'No Git while B payload active')
                no_external_scientific_processes(args)
                checked(next(x for x in manifest['source_refs'] if path(x['path'])==helper))
                attempt=f'return-{return_index:03d}'
                command=[str(python_path(manifest['python_executable'])),str(helper),'--repo',str(ROOT),'--manifest',str(path(args.manifest)),'--manifest-sha256',args.manifest_sha256,'--attempt',attempt,'--execute']
                try:
                    completed=subprocess.run(command,cwd=ROOT,timeout=300,check=False)
                    value={'status':'HELPER_EXITED0_VERIFY_HELPER_RECEIPT' if completed.returncode==0 else 'BLOCKED_UPLOAD_FAILED_SCIENCE_RETAINED','returncode':completed.returncode,'attempt':attempt,'attempted_at_utc':utc()}
                except subprocess.TimeoutExpired:
                    value={'status':'BLOCKED_UPLOAD_TIMEOUT_SCIENCE_RETAINED','attempt':attempt,'attempted_at_utc':utc()}
                state.setdefault('git_return_attempts',[]).append(value);save(statep,state)
                if value.get('returncode')==0:break
                if return_index<3:time.sleep(2)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','manifest-sha256','authorization','authorization-sha256'):parser.add_argument('--'+key)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--detach',action='store_true');parser.add_argument('--check-only',action='store_true');parser.add_argument('--prepare',action='store_true');parser.add_argument('--run-id');parser.add_argument('--python-executable');parser.add_argument('--smoke-test',action='store_true');parser.add_argument('--no-upload',action='store_true');parser.add_argument('--attempt',default='B-invocation-001');args=parser.parse_args()
    if args.prepare:
        require(not args.execute and args.run_id and args.python_executable,'Preparation is not computation')
        if args.check_only:
            load_release();print(json.dumps({'status':'RELEASE_CHECKED_NOT_RUN','new_quantum_evaluations':0}));return
        print(json.dumps(prepare_manifest(args.run_id,args.python_executable,args.smoke_test,args.no_upload)));return
    require(all(getattr(args,k) for k in ('manifest','manifest_sha256','authorization','authorization_sha256')),'Explicit manifest/auth hashes required')
    manifest=read(args.manifest,args.manifest_sha256);auth=read(args.authorization,args.authorization_sha256);validate(manifest,auth,args.manifest_sha256)
    require(SAFE.fullmatch(args.attempt),'Safe invocation ID required')
    if not args.execute or args.check_only:print(json.dumps({'status':'CHECKED_NOT_RUN','jobs':len(manifest['jobs']),'independent_CPU_cap':manifest['cpu_cap_seconds'],'deadline':manifest['deadline_utc'],'new_quantum_evaluations':0}));return
    require(time.time()<datetime.fromisoformat(manifest['deadline_utc']).timestamp(),'B deadline expired')
    if not manifest.get('local_smoke_test_without_remote_B_claim'):
        actual_branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
        require(actual_branch==manifest['upload']['branch'],'Actual B per-run result branch required')
    if args.detach:
        work=path(manifest['work_directory']);work.mkdir(parents=True,exist_ok=True);receipt=work/('detached-'+args.attempt+'.json');console=work/('detached-'+args.attempt+'.console.log');require(not receipt.exists() and not console.exists(),'Preserve prior detached receipts/consoles')
        command=[str(python_path(manifest['python_executable'])),str(SELF),'--manifest',str(path(args.manifest)),'--manifest-sha256',args.manifest_sha256,'--authorization',str(path(args.authorization)),'--authorization-sha256',args.authorization_sha256,'--attempt',args.attempt,'--execute']
        with console.open('x') as f:child=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        value={'status':'ACTUAL_B_PARENT_LAUNCHED_NOT_SCIENCE_COMPLETE','identity':identity(child.pid),'manifest_sha256':args.manifest_sha256,'supervisor_sha256':sha(SELF),'machine_shutdown_survival_not_guaranteed':True,'launched_at_utc':utc(),'remote_B_claim':not manifest.get('local_smoke_test_without_remote_B_claim',False)};save(receipt,value,exclusive=True);save(work/'current-launch.json',value);print(json.dumps(value));return
    run(args,manifest,auth)

if __name__=='__main__':main()
