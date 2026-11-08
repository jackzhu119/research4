"""Explicit forensic closure of an orphaned V6b phase; default DRY_RUN.

No wait4 is fabricated. A terminal process-wide /proc tick measurement is tagged
PROC_STAT_FINAL_TICKS, not WAIT4; the failed phase can never qualify a cache. No
Q work, signals, allocation, cap/deadline extension or reserve-as-actual row is
implemented. Full frozen guard validation may hash CLOSED ancestor checkpoints
opaquely; no array decoder or numerical package is imported.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime,timezone
from decimal import Decimal
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
GUARD=ROOT/'scripts/run_pyscf_continuation_budget_20261105_v6b.py'
GUARD_SHA='b985d6d6e20f5e4a48387094461f19b133dae761338664e88836c62051f993a5'
SCHEMA='research4-orphan-phase-reconciliation-v1'
ADDENDUM_SCHEMA='research4-orphan-phase-reconciliation-addendum-v1'
SOURCE_SHAS={
    'scripts/pyscf_staged_hessian_worker_20261105_v6b.py':'f13701964cd8a5fbda25a19ca27ec91b4a414f6b5cb49c14abacf6c82eaa448d',
    'scripts/pyscf_molecular_state_worker_20261104_v5.py':'be4727b76d59a0b0dd3defba249ef833ee8ef0ea89473e390dcef389728415c8',
    'scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py':'acfe7708748ac237c9912aa9b1d98e047d842794fd0c6f08b1a67b993f5b277d',
    'scripts/pyscf_resumable_hessian_worker_20261104_v5.py':'b40188f747c01e9efa97672af6eedfd281a9e9697d20bc7d57f125bb131fa190',
    'scripts/pyscf_incremental_rhf_checkpoint_20261103.py':'ce96055f24b39833337844c178f94597a9ac4ba7721ba019816d03628f971f55'}
OPAQUE_DAEMONS={125:'dockerd',176:'containerd'}


def require(value,message):
    if not value:raise ValueError(message)


def utc():return datetime.now(timezone.utc).isoformat()


def aware(value):
    result=datetime.fromisoformat(value)
    require(result.tzinfo is not None and result.utcoffset() is not None,'Timezone-aware evidence time required')
    return result


def root_path(value):
    path=Path(value)
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()


def small_bytes(path,expected=None,maximum=2*1024**2):
    supplied=Path(path)
    require(not supplied.is_symlink() and supplied.is_file(),'Missing/symlink metadata source')
    require(supplied.stat().st_size<=maximum,'Oversized forensic metadata')
    raw=supplied.read_bytes()
    require(expected is None or hashlib.sha256(raw).hexdigest()==expected,'Metadata SHA differs: '+str(path))
    return raw


def read_json(path,expected=None):return json.loads(small_bytes(path,expected))


def artifact(path):
    return {'path':str(Path(path).resolve()),'sha256':sha(path),'bytes':Path(path).stat().st_size}


def encoded(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def immutable_bytes(path,raw):
    with Path(path).open('xb') as handle:
        handle.write(raw);handle.flush();os.fsync(handle.fileno())


def guard_module():
    require(sha(GUARD)==GUARD_SHA,'Frozen guard changed')
    spec=importlib.util.spec_from_file_location('frozen_v6b_orphan_metadata_guard',GUARD)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def no_spawn_source_audit():
    results=[]
    for relative,expected in SOURCE_SHAS.items():
        path=ROOT/relative;require(sha(path)==expected,'Frozen worker/helper changed')
        tree=ast.parse(path.read_text());bad=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                bad.extend(v.name for v in node.names if v.name.split('.')[0] in ('subprocess','multiprocessing'))
            elif isinstance(node,ast.ImportFrom) and (node.module or '').split('.')[0] in ('subprocess','multiprocessing'):
                bad.append(node.module)
            elif isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                if isinstance(node.func.value,ast.Name) and node.func.value.id=='os' and (
                    node.func.attr in ('fork','forkpty','system','popen','posix_spawn','posix_spawnp')
                    or node.func.attr.startswith(('spawn','exec'))):bad.append('os.'+node.func.attr)
        require(not bad,'Explicit process-spawn source outside single-TGID recovery scope: '+repr(bad))
        results.append({'path':str(path),'sha256':expected,'explicit_process_spawn_detected':False})
    return {'files':results,'scope':'Pinned Python worker/helpers have no explicit process spawn. Native/transitive library implementation is not exhaustively proved.'}


def parse_proc(raw):
    text=raw.decode();right=text.rfind(')');require(right>0,'Malformed proc stat')
    fields=text[right+1:].split();require(len(fields)>=50,'Full proc stat exit status required')
    return {'pid':int(text[:text.find('(')].strip()),'state':fields[0],'ppid':int(fields[1]),
        'pgrp':int(fields[2]),'session':int(fields[3]),'utime_ticks':int(fields[11]),
        'stime_ticks':int(fields[12]),'cutime_ticks':int(fields[13]),'cstime_ticks':int(fields[14]),
        'start_ticks':int(fields[19]),'rss_pages':int(fields[21]),'raw_exit_status':int(fields[49])}


def parse_status(raw):
    fields={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in raw.decode().splitlines() if ':' in line}
    return {'uids':[int(v) for v in fields['Uid'].split()], 'FDSize':int(fields['FDSize']),
        'Threads':int(fields['Threads']),'state':fields['State'].split()[0],
        'VmRSS_kib':int(fields.get('VmRSS','0 kB').split()[0]),
        'VmSize_kib':int(fields.get('VmSize','0 kB').split()[0])}


def terminal_identity(stat_raw,status_raw,expected):
    value=parse_proc(stat_raw);status=parse_status(status_raw)
    require(all(value[k]==v for k,v in expected.items()),'PID/start/group/session identity differs')
    require(value['state'] in ('Z','X','x') and status['state'] in ('Z','X','x')
        and value['state']==status['state']
        and value['rss_pages']==0 and status['FDSize']==0 and status['Threads']==1
        and status['VmRSS_kib']==status['VmSize_kib']==0,'Process is not a proven terminal no-files/no-mm task')
    require(value['cutime_ticks']==value['cstime_ticks']==0,'Nonzero waited-child CPU outside this narrow recovery scope')
    require(value['raw_exit_status']==9 and os.waitstatus_to_exitcode(value['raw_exit_status'])==-9,
        'This recovery is scoped to observed SIGKILL, not a guessed exit status')
    return {**value,**status,'raw_stat_sha256':hashlib.sha256(stat_raw).hexdigest(),
        'raw_status_sha256':hashlib.sha256(status_raw).hexdigest()}


def process_snapshot(pid,proc_root=Path('/proc')):
    base=proc_root/str(pid)
    return small_bytes(base/'stat',maximum=8192),small_bytes(base/'status',maximum=32768)


def no_live_groups(groups,proc_root=Path('/proc')):
    members=[];opaque=[];vanished=[]
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():continue
        pid=int(directory.name)
        try:
            raw=(directory/'stat').read_bytes()
            require(len(raw)<=8192,'Oversized process stat metadata')
            text=raw.decode();fields=text[text.rfind(')')+1:].split()
            require(len(fields)>=22,'Malformed process metadata')
            value={'pid':pid,'state':fields[0],'pgrp':int(fields[2]),'session':int(fields[3]),
                'start_ticks':int(fields[19]),'uid':directory.stat().st_uid}
        except (FileNotFoundError,ProcessLookupError):vanished.append(pid);continue
        except PermissionError:
            name=(directory/'comm').read_text().strip();uid=directory.stat().st_uid
            require(OPAQUE_DAEMONS.get(pid)==name and uid==0,'Unapproved opaque process; cannot exclude live group membership')
            opaque.append({'pid':pid,'name':name,'uid':uid,'stat_unreadable':True,
                'root_explicit_daemon_exception':True,'all_visibility_claimed':False});continue
        if value['pgrp'] in groups or value['session'] in groups:
            require(value['state'] in ('Z','X','x'),'Live process remains in original parent/child group or session')
            members.append(value)
    return {'terminal_group_members':members,'opaque_processes':opaque,'vanished_during_scan':vanished,
        'full_proc_visibility_claimed':not opaque,'cmdline_or_secrets_read':False}


def current_terminal_evidence(proof,raw_records,proc_root=Path('/proc')):
    expected_child={'pid':proof['child_pid'],'start_ticks':proof['child_start_ticks'],
        'pgrp':proof['target_pgid'],'session':proof['target_session_id']}
    archived_child=terminal_identity(small_bytes(raw_records['child_stat']['path'],raw_records['child_stat']['sha256']),
        small_bytes(raw_records['child_status']['path'],raw_records['child_status']['sha256']),expected_child)
    parent_archived=parse_proc(small_bytes(raw_records['supervisor_stat']['path'],raw_records['supervisor_stat']['sha256']))
    expected_parent={'pid':proof['supervisor_pid'],'start_ticks':proof['supervisor_start_ticks'],
        'pgrp':parent_archived['pgrp'],'session':parent_archived['session']}
    archived_parent=terminal_identity(small_bytes(raw_records['supervisor_stat']['path'],raw_records['supervisor_stat']['sha256']),
        small_bytes(raw_records['supervisor_status']['path'],raw_records['supervisor_status']['sha256']),expected_parent)
    observed=[];raw=[]
    for _ in range(2):
        one={'observed_at_utc':utc()}
        for role,expected,archived in [('child',expected_child,archived_child),('supervisor',expected_parent,archived_parent)]:
            stat_raw,status_raw=process_snapshot(expected['pid'],proc_root)
            value=terminal_identity(stat_raw,status_raw,expected)
            require(value['uids']==[proof['target_uid']]*4,'Current terminal UID differs from launch contract')
            require(archived['uids']==value['uids'],'Archived terminal UID differs')
            require(all(value[k]==archived[k] for k in ('utime_ticks','stime_ticks','cutime_ticks','cstime_ticks','raw_exit_status')),
                'Terminal CPU/wait status changed from preserved observation')
            one[role]=value;raw.append((role,stat_raw,status_raw))
        observed.append(one)
    critical=('pid','start_ticks','pgrp','session','uids','state','utime_ticks','stime_ticks','cutime_ticks',
        'cstime_ticks','raw_exit_status','rss_pages','FDSize','Threads','VmRSS_kib','VmSize_kib')
    require(all(observed[0][role][key]==observed[1][role][key] for role in ('child','supervisor') for key in critical),
        'Critical terminal identity/ticks did not remain identical across double read')
    group_audit=no_live_groups({expected_child['pgrp'],expected_child['session'],expected_parent['pgrp'],expected_parent['session']},proc_root)
    return {'double_read':observed,'group_audit':group_audit,'wait4_obtained':False},raw


def raw_records(observation_path,proof):
    base=Path(observation_path).parent
    records={}
    for role,pid in [('supervisor',proof['supervisor_pid']),('child',proof['child_pid'])]:
        for kind in ('stat','status'):
            path=base/f'proc_{pid}_{kind}'
            raw=small_bytes(path,maximum=8192 if kind=='stat' else 32768)
            records[role+'_'+kind]={'path':str(path.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    return records


def validate_addendum(path,expected_sha,args,proof,records,holdback,scientific_input):
    require(path and expected_sha,'--execute requires an explicit root addendum path/SHA')
    value=read_json(path,expected_sha)
    require(value.get('schema')==ADDENDUM_SCHEMA and value.get('root_approved_for_reconciliation') is True
        and value.get('reconciler_source_sha256')==sha(__file__),'Actual source-bound root reconciliation approval required')
    expected={'original_ledger_path':str(args.ledger.resolve()),'original_ledger_sha256':args.ledger_sha256,
        'phase':args.phase,'guard_source_sha256':GUARD_SHA,'launch_proof_sha256':proof['sha256'],
        'observation_sha256':args.observation_sha256,'raw_process_snapshots':records,
        'scientific_input_sha256':scientific_input['sha256'],
        'accounting_source':'PROC_STAT_FINAL_TICKS','wait4_obtained':False,
        'required_non_actual_holdback_seconds':holdback,'holdback_covered_by_unchanged_guard_boundary':True,
        'selected_future_threads':2,'no_resource_growth':True,'no_deadline_extension':True,'no_healthy_cache_release':True}
    require(all((value.get(k) is v) if isinstance(v,bool) else value.get(k)==v for k,v in expected.items())
        and type(value.get('required_non_actual_holdback_seconds')) in (int,float),'Root addendum binding/policy differs')
    created=aware(value['created_at_utc'])
    observed=aware(read_json(args.observation,args.observation_sha256)['observed_at_utc'])
    require(observed<=created<=datetime.now(timezone.utc),'Root decision time must follow actual terminal observation')
    return value


def build_proposal(original,phase,observation,evidence,records,original_sha,observation_record,holdback,guard):
    require(type(holdback) in (int,float) and math.isfinite(holdback) and holdback==1,'This narrow reconciliation records exactly1second NOT_ACTUAL holdback')
    matches=[i for i,r in enumerate(original['phases']) if r.get('phase')==phase]
    require(len(matches)==1,'Unique exact orphan phase required');index=matches[0];old=original['phases'][index]
    require(old.get('status')=='RUNNING' and sum(r.get('status')=='RUNNING' for r in original['phases'])==1,
        'Only one original RUNNING orphan may be reconciled; refuse already closed/idempotent repeat')
    child=evidence['double_read'][-1]['child'];parent=evidence['double_read'][-1]['supervisor']
    clock=observation['clock_ticks_per_second']
    require(type(clock) is int and clock>0,'Positive recorded clock required')
    child_cpu=float(Decimal(child['utime_ticks']+child['stime_ticks'])/Decimal(clock))
    parent_cpu=float(Decimal(parent['utime_ticks']+parent['stime_ticks'])/Decimal(clock))
    require(child_cpu==observation['terminal_process_cpu_seconds_tick_precision'],'Observed terminal CPU disagrees with raw tick conversion')
    require(all(observation.get('final_'+key)==child[key] for key in ('utime_ticks','stime_ticks','cutime_ticks','cstime_ticks')),
        'Declared observation CPU counters differ from preserved terminal task')
    proposal=deepcopy(original);row=proposal['phases'][index]
    row.update(status='PARTIAL_OR_FAILED',ended_at_utc=observation['observed_at_utc'],
        ended_at_utc_semantics='TERMINAL_OBSERVED_UPPER_BOUND_NOT_EXACT_DEATH_TIME',exact_death_time_known=False,
        actual_cpu_seconds=child_cpu,actual_cpu_accounting_source='PROC_STAT_FINAL_TICKS',
        actual_cpu_precision_seconds=1/clock,actual_wait4_cpu_seconds=None,wait4_obtained=False,
        actual_CPU_is_unknown_do_not_resume_budget=False,
        proc_terminal_CPU_measurement={'utime_ticks':child['utime_ticks'],'stime_ticks':child['stime_ticks'],
            'cutime_ticks':0,'cstime_ticks':0,'clock_ticks_per_second':clock,
            'process_wide_native_threads_already_included':True,'no_threads_multiplier':True,
            'component_quantization_uncertainty_upper_bound_seconds':2/clock,'same_dead_task_double_read':True},
        supervisor_process_CPU_seconds_tick_precision_NOT_CHILD_LEDGER=parent_cpu,
        exit_code=-9,exit_code_source='PROC_STAT_RAW_WAIT_STATUS9_INFERRED_SIGKILL_NOT_WAIT4',raw_terminal_exit_status=9,
        wall_terminated=None,memory_terminated=None,cpu_terminated=None,
        elapsed_wall_seconds=None,peak_rss_kib=None,
        termination_flags_unknown_because_supervisor_did_not_close=True,
        hard_killed=True,hard_killed_semantics='SIGKILL_OBSERVED_NOT_A_CLAIM_THIS_TOOL_OR_GUARD_SENT_IT',
        monitor_failed=True,supervisor_terminated_without_wait4=True,
        residual_processes_require_review_after_main_payload_exit=[],
        source_for_scientific_cache_qualified=False,healthy_producer=False,
        orphan_reconciliation={'schema':SCHEMA,'original_ledger_sha256':original_sha,
            'observation':observation_record,'raw_process_snapshots':records,
            'root_addendum':None,'approval_status':'DRY_RUN_NOT_GRANTED','reconciliation_receipt_path':None,
            'conservative_non_actual_holdback_seconds':holdback,'holdback_is_actual_cost':False,
            'scientific_result_created':False,'failed_cache_released':False})
    # Only a genuine child CPU measurement is charged. Reserve is never a fake
    # actual phase/cost. Future b985 boundary already holds6s at selected2threads.
    proposal['consumed_cpu_seconds']=original['consumed_cpu_seconds']+child_cpu
    proposal['remaining_quantum_cpu_seconds']=max(0.,proposal['quantum_cpu_cap_seconds']-proposal['consumed_cpu_seconds'])
    row['orphan_reconciliation']['accounting_addendum']={'phase':phase,'schema':SCHEMA,
        'source':'PROC_STAT_FINAL_TICKS_NOT_WAIT4','charged_child_cpu_seconds':child_cpu,
        'separately_reported_supervisor_CPU_not_charged':parent_cpu,
        'non_actual_holdback_seconds':holdback,'holdback_covered_by_unchanged_existing_boundary':True,
        'selected_future_threads':2,'existing_boundary_reserve_seconds':6,
        'no_additional_reserve_cost_or_phase':True,'original_ledger_sha256':original_sha,
        'observation':observation_record,'reconciler_source_sha256':sha(__file__)}
    unchanged=set(original)-{'phases','consumed_cpu_seconds','remaining_quantum_cpu_seconds'}
    require(all(proposal[k]==original[k] for k in unchanged),'Immutable original header/cap/deadline/auth/history changed')
    require(all(proposal['phases'][i]==r for i,r in enumerate(original['phases']) if i!=index),'Unrelated phase changed')
    require(not guard.healthy_closed_row(row),'Failed orphan must never become a healthy producer')
    require(math.isclose(guard.closed_costs(proposal['phases']),proposal['consumed_cpu_seconds'],abs_tol=1e-7,rel_tol=1e-12),'Nominal actual cost sum differs')
    require(proposal['consumed_cpu_seconds']<=proposal['quantum_cpu_cap_seconds'],'Recovered actual CPU exceeds unchanged cap; separate violation review required')
    return proposal


@contextmanager
def locks(ledger,guard,auth):
    local=ledger.with_suffix('.lock');global_path=ROOT/'.runtime/pyscf-v4-global.lock'
    handles=[]
    try:
        for path in (local,global_path):
            require(path.is_file() and not path.is_symlink(),'Existing original lock inode required')
            fd=os.open(path,os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC);handles.append(fd)
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with guard.lineage_locks(auth):yield
    finally:
        for fd in reversed(handles):os.close(fd)


def atomic_cas_ledger(path,expected_sha,raw):
    require(sha(path)==expected_sha,'Canonical ledger changed before atomic CAS')
    info=path.stat();require(info.st_uid==os.geteuid(),'Preserve ledger owner; unreviewed identity cannot replace it')
    fd,name=tempfile.mkstemp(prefix=path.name+'.reconcile-',dir=path.parent)
    temporary=Path(name)
    try:
        os.fchmod(fd,stat.S_IMODE(info.st_mode))
        with os.fdopen(fd,'wb') as handle:handle.write(raw);handle.flush();os.fsync(handle.fileno())
        require(sha(path)==expected_sha,'Ledger changed under held locks; stop without replacement')
        os.replace(temporary,path)
        directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:temporary.unlink(missing_ok=True)


def prepare(args,guard,*,proc_root=Path('/proc')):
    raw=small_bytes(args.ledger,args.ledger_sha256);original=json.loads(raw)
    auth=guard.validate_ledger(original,allow_running=True)
    observation=read_json(args.observation,args.observation_sha256)
    require(observation.get('schema')=='research4-unclosed-phase-terminal-observation-v1'
        and observation.get('phase')==args.phase and observation.get('wait4_obtained') is False
        and observation.get('exact_death_time_known') is False,'Original immutable terminal observation differs')
    saved=observation['original_ledger_bytes_preserved']['cpu_budget.json']
    require(saved['sha256']==args.ledger_sha256 and small_bytes(root_path(saved['preserved_copy']),saved['sha256'])==raw,
        'Exact original RUNNING ledger bytes must already be preserved')
    rows=[r for r in original['phases'] if r.get('phase')==args.phase]
    require(len(rows)==1 and rows[0].get('status')=='RUNNING','Refuse missing or already-closed phase')
    row=rows[0];proof_path=root_path(row['supervisor_proof_path'])
    proof=read_json(proof_path,row['supervisor_proof_sha256'])
    require(proof.get('schema')==guard.PROOF_SCHEMA and proof.get('supervisor_script_sha256')==GUARD_SHA
        and root_path(proof.get('supervisor_script_path',''))==GUARD and proof['phase']==args.phase
        and root_path(proof['ledger_path'])==args.ledger.resolve()
        and row['pid']==proof['child_pid'] and row['threads']==proof['threads']==2,
        'Original live-launch contract/source/phase/ledger differs')
    observed_time=aware(observation['observed_at_utc'])
    require(observed_time>=aware(row['started_at_utc']) and observed_time>=aware(proof['started_at_utc']),
        'Terminal observation cannot predate the original phase/proof start')
    saved_proof=observation['original_ledger_bytes_preserved']['execution/'+args.phase+'.log.supervisor-proof.json']
    require(saved_proof['sha256']==row['supervisor_proof_sha256']
        and small_bytes(root_path(saved_proof['preserved_copy']),saved_proof['sha256'])==small_bytes(proof_path),
        'Preserved launch proof differs')
    require(proof['actual_cpu_clock_ticks_per_second']==observation['clock_ticks_per_second']==os.sysconf('SC_CLK_TCK'),
        'Clock tick conversion must agree with original proof and current kernel')
    command=row['command'];script=next((root_path(t) for t in command if isinstance(t,str) and t.endswith('.py')),None)
    require(script==ROOT/'scripts/pyscf_staged_hessian_worker_20261105_v6b.py'
        and '--execute' in command and command[command.index('--stage')+1]=='rks-partial','Only the exact native A worker is supported')
    scientific_output=root_path(command[command.index('--output')+1])
    require(observation.get('result_json_exists') is False and not (scientific_output/'result.json').exists(),
        'Unexpected scientific result needs separate review; never create or replace it here')
    input_path=scientific_output/'input.json';run_input=read_json(input_path)
    require(run_input.get('schema')=='research4-staged-full-rks-hessian-result-v6b'
        and run_input.get('stage')==run_input.get('phase')=='rks-partial'
        and run_input.get('source_worker',{}).get('sha256')==SOURCE_SHAS['scripts/pyscf_staged_hessian_worker_20261105_v6b.py']
        and run_input.get('resource_supervision',{}).get('proof')==proof
        and run_input['resource_supervision'].get('proof_sha256')==row['supervisor_proof_sha256'],
        'Original partial input lacks the exact frozen worker and launch contract')
    boot=(proc_root/'sys/kernel/random/boot_id').read_text().strip()
    require(len(boot)==36 and run_input.get('boot_id')==boot,'Kernel boot identity differs; PID/start alone is insufficient')
    require(observed_time>=aware(run_input['started_at_utc']),'Observation predates original scientific input')
    records=raw_records(args.observation,proof)
    for role in ('child','supervisor'):
        for kind in ('stat','status'):
            key=role+'_'+kind;record=records[key]
            old=observation['raw_observations'].get('/proc/'+str(proof[role+'_pid'])+'/'+kind)
            if old:require(old['sha256']==record['sha256'] and old['bytes']==record['bytes'],'Preserved raw observation hash differs')
    evidence,current_raw=current_terminal_evidence(proof,records,proc_root)
    sources=no_spawn_source_audit()
    proposal=build_proposal(original,args.phase,observation,evidence,records,args.ledger_sha256,
        artifact(args.observation),args.holdback_seconds,guard)
    proposal['phases'][original['phases'].index(row)]['orphan_reconciliation'].update(
        original_partial_input=artifact(input_path),kernel_boot_id=boot)
    guard.validate_ledger(proposal)
    # Frozen metadata-only policy confirms1s is contained inside existing6s.
    limit_args=SimpleNamespace(threads=2,memory_gib=24,cpu_seconds=auth['max_phase_cpu_seconds'],wall_seconds=auth['max_phase_wall_seconds'])
    try:limits=guard.phase_limits(proposal,auth,limit_args)
    except ValueError as error:
        if 'Insufficient original CPU/wall budget' not in str(error):raise
        # Closing historical accounting is still valid after budget/deadline
        # exhaustion. This does not authorize or pretend a future phase can run.
        limits={'status':'BLOCKED_INSUFFICIENT_REMAINING_CPU_OR_WALL_NO_NEW_PHASE',
            'reason':str(error),'threads':2,'cpu_boundary_reserve_seconds':2*2+2}
    require(limits['cpu_boundary_reserve_seconds']>=args.holdback_seconds,'Unchanged future boundary does not cover measured holdback')
    return {'original_raw':raw,'original':original,'auth':auth,'observation':observation,'proof_record':artifact(proof_path),
        'records':records,'current_evidence':evidence,'current_raw':current_raw,'no_spawn_source_audit':sources,
        'scientific_input':artifact(input_path),'kernel_boot_id':boot,
        'proposal':proposal,'future_phase_limits_metadata_only_NOT_A_LAUNCH':limits}


def reconcile(args):
    guard=guard_module()
    require(not args.ledger.is_symlink() and not args.observation.is_symlink(),'Symlink ledger/observation rejected')
    args.ledger=root_path(args.ledger);args.observation=root_path(args.observation)
    require(not args.output.exists() and args.output.resolve().is_relative_to(ROOT/'reports'),'Fresh separate report directory required')
    # Validate first without mutation, then repeat every check under the exact
    # original lock order before any execute write. Dry-run never changes ledger.
    preliminary=read_json(args.ledger,args.ledger_sha256)
    auth=guard.validate_ledger(preliminary,allow_running=True)
    with locks(args.ledger,guard,auth):
        data=prepare(args,guard)
        output=args.output.resolve()
        addendum=None
        if args.execute:
            addendum=validate_addendum(args.addendum,args.addendum_sha256,args,data['proof_record'],data['records'],args.holdback_seconds,data['scientific_input'])
        target=next(r for r in data['proposal']['phases'] if r['phase']==args.phase)['orphan_reconciliation']
        target.update(root_addendum=artifact(args.addendum) if addendum else None,
            approval_status='EXPLICIT_ROOT_APPROVED' if addendum else 'DRY_RUN_NOT_GRANTED',
            reconciliation_receipt_path=str(output/'result.json'))
        guard.validate_ledger(data['proposal'])
        output.mkdir(parents=True)
        immutable_bytes(output/'original_RUNNING_ledger.json',data['original_raw'])
        immutable_bytes(output/'proposed_FAILED_ledger.json',encoded(data['proposal']))
        for index,(role,stat_raw,status_raw) in enumerate(data['current_raw']):
            immutable_bytes(output/f'current_{index}_{role}_stat',stat_raw)
            immutable_bytes(output/f'current_{index}_{role}_status',status_raw)
        report={'schema':SCHEMA,'mode':'EXECUTE_APPROVED' if args.execute else 'DRY_RUN_NO_LEDGER_CHANGE',
            'created_at_utc':utc(),'source_reconciler':artifact(__file__),'guard_source_sha256':GUARD_SHA,
            'original_ledger_sha256':args.ledger_sha256,'planned_new_ledger_sha256':hashlib.sha256(encoded(data['proposal'])).hexdigest(),
            'observation':artifact(args.observation),'launch_proof':data['proof_record'],
            'original_scientific_partial_input':data['scientific_input'],'kernel_boot_id':data['kernel_boot_id'],
            'terminal_evidence':data['current_evidence'],'no_spawn_source_audit':data['no_spawn_source_audit'],
            'actual_child_CPU_proc_ticks_seconds':next(r['actual_cpu_seconds'] for r in data['proposal']['phases'] if r['phase']==args.phase),
            'consumed_cpu_seconds':data['proposal']['consumed_cpu_seconds'],
            'remaining_cpu_seconds':data['proposal']['remaining_quantum_cpu_seconds'],
            'NOT_ACTUAL_holdback_seconds':args.holdback_seconds,
            'holdback_covered_by_existing_boundary_without_double_debit':True,
            'future_phase_limits_metadata_only_NOT_A_LAUNCH':data['future_phase_limits_metadata_only_NOT_A_LAUNCH'],
            'wait4_obtained':False,'peak_RSS_or_exact_death_or_wall_duration_known':False,
            'full_frozen_guard_chain_validation':'PASS_OPAQUE_HASHES_OF_CLOSED_ANCESTORS_NO_ARRAY_DECODING',
            'scientific_result_created':False,'failed_cache_released':False,'new_Q_or_Git_or_signal_or_allocation':False,
            'root_addendum':artifact(args.addendum) if addendum else None}
        immutable_bytes(output/'prepared_reconciliation.json',encoded(report))
        if args.execute:
            require(sha(args.addendum)==args.addendum_sha256,'Root addendum changed before CAS')
            atomic_cas_ledger(args.ledger,args.ledger_sha256,encoded(data['proposal']))
            guard.validate_ledger(read_json(args.ledger,report['planned_new_ledger_sha256']))
            report.update(status='DONE_FAILED_PHASE_RECONCILED_NOT_A_HEALTHY_PRODUCER',actual_new_ledger_sha256=sha(args.ledger))
        else:
            require(sha(args.ledger)==args.ledger_sha256,'Ledger changed during dry-run')
            report['status']='DRY_RUN_VALIDATED_NO_LEDGER_CHANGE'
        immutable_bytes(output/'result.json',encoded(report))
    print(json.dumps({k:report.get(k) for k in ('status','consumed_cpu_seconds','remaining_cpu_seconds','NOT_ACTUAL_holdback_seconds','actual_new_ledger_sha256')},indent=2),flush=True)
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--ledger',type=Path,required=True);p.add_argument('--ledger-sha256',required=True)
    p.add_argument('--phase',required=True);p.add_argument('--observation',type=Path,required=True)
    p.add_argument('--observation-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--holdback-seconds',type=float,default=1.)
    p.add_argument('--addendum',type=Path);p.add_argument('--addendum-sha256')
    return p


if __name__=='__main__':reconcile(parser().parse_args())
