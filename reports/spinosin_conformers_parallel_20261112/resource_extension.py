#!/usr/bin/env python3
"""Append-only resource authorizations over the original sealed chemistry plan.
Does not edit scientific files, the original meta budget, or input_seals.
Grant/recovery/start require separate explicit operator commands.
"""
import argparse,fcntl,hashlib,json,math,os,resource,signal,sqlite3,subprocess,sys,time,uuid,shutil
from pathlib import Path
from datetime import datetime,timezone
import supervisor as base
ROOT=Path(__file__).resolve().parent

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def ownsha():return base.sha(Path(__file__))
def dt(v):
 t=datetime.fromisoformat(str(v).replace('Z','+00:00'));assert t.tzinfo is not None,'UTC/offset required';return t.astimezone(timezone.utc)
def connect(root,readonly=False):
 d=sqlite3.connect('file:'+str(root/'queue.sqlite3')+('?mode=ro'if readonly else ''),uri=True,timeout=30);d.row_factory=sqlite3.Row
 if not readonly:d.execute('PRAGMA journal_mode=WAL');d.execute('PRAGMA synchronous=FULL');d.execute('PRAGMA busy_timeout=30000')
 return d

def schema(d):
 d.executescript('''CREATE TABLE IF NOT EXISTS budget_authorizations(seq INTEGER PRIMARY KEY AUTOINCREMENT,authorization_id TEXT UNIQUE NOT NULL,record_json TEXT NOT NULL,previous_hash TEXT NOT NULL,entry_hash TEXT NOT NULL);
 CREATE TRIGGER IF NOT EXISTS budget_auth_no_update BEFORE UPDATE ON budget_authorizations BEGIN SELECT RAISE(ABORT,'budget authorizations are append-only');END;
 CREATE TRIGGER IF NOT EXISTS budget_auth_no_delete BEFORE DELETE ON budget_authorizations BEGIN SELECT RAISE(ABORT,'budget authorizations are append-only');END;
 CREATE TABLE IF NOT EXISTS resource_recoveries(seq INTEGER PRIMARY KEY AUTOINCREMENT,recovery_id TEXT UNIQUE NOT NULL,record_json TEXT NOT NULL,entry_hash TEXT NOT NULL,extra_resource_charge_cpu REAL NOT NULL DEFAULT 0);
 CREATE TRIGGER IF NOT EXISTS recovery_no_update BEFORE UPDATE ON resource_recoveries BEGIN SELECT RAISE(ABORT,'resource recoveries are append-only');END;
 CREATE TRIGGER IF NOT EXISTS recovery_no_delete BEFORE DELETE ON resource_recoveries BEGIN SELECT RAISE(ABORT,'resource recoveries are append-only');END;
 CREATE TABLE IF NOT EXISTS resource_attempt_bindings(attempt_id TEXT PRIMARY KEY,authorization_id TEXT NOT NULL,record_json TEXT NOT NULL,entry_hash TEXT NOT NULL);
 CREATE TRIGGER IF NOT EXISTS binding_no_update BEFORE UPDATE ON resource_attempt_bindings BEGIN SELECT RAISE(ABORT,'attempt resource bindings are append-only');END;
 CREATE TRIGGER IF NOT EXISTS binding_no_delete BEFORE DELETE ON resource_attempt_bindings BEGIN SELECT RAISE(ABORT,'attempt resource bindings are append-only');END;''');d.commit()

def science_check(root,d):
 row=d.execute("SELECT value FROM meta WHERE key='input_seals'").fetchone();assert row,'Original frozen seals required';seals=json.loads(row['value']);bad=[f for f,h in seals.items()if not Path(f).is_file()or base.sha(f)!=h];assert not bad,'Original scientific/input seals changed: '+str(bad)
 b=float(d.execute("SELECT value FROM meta WHERE key='budget'").fetchone()['value']);_,p=base.config(root);assert float(p['cpu_budget_seconds'])==b,'Base scientific protocol/meta budget mismatch'
 return b,digest(seals)

def authority(root,d,files=True):
 original,seals_sha=science_check(root,d);exists=d.execute("SELECT 1 FROM sqlite_master WHERE name='budget_authorizations' AND type='table'").fetchone()
 if not exists:return {'total':original,'original':original,'latest':None,'tip':'GENESIS','count':0}
 tip='GENESIS';total=original;latest=None;rows=d.execute('SELECT *FROM budget_authorizations ORDER BY seq').fetchall()
 for row in rows:
  record=json.loads(row['record_json']);assert record['original_seals_digest']==seals_sha,'Original input_seals digest changed since resource authorization';assert row['previous_hash']==tip==record['previous_hash'];assert digest(record)==row['entry_hash'];assert float(record['previous_total_cpu_seconds'])==total;assert float(record['original_base_budget_cpu_seconds'])==original;assert float(record['additional_cpu_seconds'])>=0;assert float(record['cumulative_authorized_cpu_seconds'])==total+float(record['additional_cpu_seconds']);total=float(record['cumulative_authorized_cpu_seconds']);tip=row['entry_hash'];latest=record
 if latest and files:
  assert latest['controller_code_sha256']==ownsha(),'Resource controller changed since authorization'
  s=latest['resource_settings'];adapter=root/s['worker_adapter'];assert adapter.is_file()and base.sha(adapter)==s['worker_adapter_sha256'],'Authorized resource worker adapter changed';assert 1<=int(s['max_workers'])<=2
 return {'total':total,'original':original,'latest':latest,'tip':tip,'count':len(rows)}

def grant(args):
 root=args.root;assert args.backup and args.backup.is_file(),'Explicit pre-change SQLite backup required';d=connect(root);original,seals_sha=science_check(root,d);schema(d);a=authority(root,d,files=False);existing=d.execute('SELECT record_json FROM budget_authorizations WHERE authorization_id=?',(args.authorization_id,)).fetchone()
 if existing:
  saved=json.loads(existing['record_json']);assert float(saved['additional_cpu_seconds'])==args.additional and saved['user_authorization']==args.user_text and saved['resource_settings']['max_workers']==args.max_workers and saved['resource_settings']['worker_adapter']==args.worker_adapter,'Authorization ID reused with different amount/text/settings';assert saved['resource_settings']['worker_adapter_sha256']==base.sha(root/args.worker_adapter),'Same-ID adapter changed';print(existing['record_json']);d.close();return
 adapter=root/args.worker_adapter;assert adapter.resolve().is_relative_to(root)and adapter.is_file(),'Adapter must exist inside this study root';assert args.additional>=0 and args.user_text
 record={'authorization_id':args.authorization_id,'authorized_at_utc':base.now(),'user_authorization':args.user_text,'original_base_budget_cpu_seconds':original,'previous_total_cpu_seconds':a['total'],'additional_cpu_seconds':args.additional,'cumulative_authorized_cpu_seconds':a['total']+args.additional,'previous_hash':a['tip'],'controller_code_sha256':ownsha(),'original_seals_digest':seals_sha,'prior_ledger_backup':str(args.backup.resolve()),'prior_ledger_backup_sha256':base.sha(args.backup),'resource_settings':{'max_workers':args.max_workers,'worker_adapter':str(adapter.relative_to(root)),'worker_adapter_sha256':base.sha(adapter)},'scientific_protocol_changed':False,'original_meta_budget_changed':False,'original_input_seals_changed':False,'CPU_ledger_reset':False}
 with d:d.execute('INSERT INTO budget_authorizations(authorization_id,record_json,previous_hash,entry_hash)VALUES(?,?,?,?)',(args.authorization_id,canonical(record),a['tip'],digest(record)))
 base.event(d,'resource_budget_authorized',record);base.atomic(root/'resource_authorization_latest.json',{**record,'entry_hash':digest(record)});d.close();print(json.dumps(record,ensure_ascii=False,indent=2))

def recover_boot(args):
 """Explicitly adjudicate stale-boot attempts using a conservative wall/host CPU ceiling."""
 root=args.root;d=connect(root);schema(d);a=authority(root,d);assert a['latest'],'Append-only authorization required before recovery';assert args.backup and args.backup.is_file(),'Backup required';cutoff=dt(args.cutoff_utc);assert args.clock_margin>=0 and args.old_host_cpu_cap>=1;current_boot=base.boot();records=[];now_epoch=time.time();host_tail=0.;worker_uncertain=0.
 rows=d.execute("SELECT *FROM attempts WHERE state!='closed'").fetchall()
 for row in rows:
  old=dict(row)
  if old['boot_id']==current_boot:continue
  folder=Path(old['attempt_dir']);wr=base.read(folder/'wait4_receipt.json',{});sources={};observed=max(0.,float(old['observed_cpu']));times=[];verified_final=False
  for filename in [folder/'runner_heartbeat.json',root/'jobs'/old['job_id']/'cpu_receipt.json']:
   doc=base.read(filename,{})
   if not doc:continue
   if filename.name=='runner_heartbeat.json':
    ident=doc.get('worker_identity',{});valid=ident.get('pid')==old['worker_pid']and ident.get('start_ticks')==old['worker_start']and ident.get('boot_id')==old['boot_id'];value=doc.get('worker_observed_cpu');stamp=doc.get('updated_utc')
   else:
    valid=doc.get('attempt_id')==old['id']and doc.get('pid')==old['worker_pid']and doc.get('start_ticks')==old['worker_start']and doc.get('boot_id')==old['boot_id'];value=doc.get('process_cpu_seconds',doc.get('cpu_seconds'));stamp=doc.get('updated_at_utc',doc.get('closed_utc'))
   if valid and isinstance(value,(int,float)):
    observed=max(observed,float(value));sources[str(filename)]={'sha256':base.sha(filename),'value':float(value),'timestamp':stamp}
    if stamp:times.append(dt(stamp))
  if wr.get('attempt_id')==old['id']and wr.get('worker_identity',{}).get('boot_id')==old['boot_id']and wr.get('worker_identity',{}).get('pid')==old['worker_pid']and wr.get('worker_identity',{}).get('start_ticks')==old['worker_start']:
   verified_final=True;observed=max(observed,float(wr['verified_worker_cpu']));charge=observed;code=int(wr['exit_code']);bound=None;source='restored_persistent_wait4_receipt'
  else:
   earliest=min(times)if times else dt(old['started_utc']);elapsed=(cutoff-earliest).total_seconds()+args.clock_margin;assert elapsed>=0,'Cutoff precedes last observation beyond clock margin';tail=elapsed*args.old_host_cpu_cap;host_tail=max(host_tail,tail);charge=max(observed,min(float(old['allocation_cpu']),observed+tail));worker_uncertain+=max(0,charge-observed);code=75;bound={'observation_utc':earliest.isoformat(),'operator_old_runtime_ended_by_utc_bound':cutoff.isoformat(),'clock_uncertainty_seconds':args.clock_margin,'elapsed_wall_ceiling_seconds':elapsed,'old_host_CPU_cap':args.old_host_cpu_cap,'wall_times_host_CPU_ceiling':tail,'capped_at_prior_allocation':old['allocation_cpu']};source='boot_changed_wall_time_host_CPU_upper_bound_NOT_exact_usage'
  result=base.read(root/'jobs'/old['job_id']/'result.json',{});complete=verified_final and code==0 and result.get('complete')is True and result.get('status')=='complete';blocked=verified_final and code==76
  records.append({'before_attempt':old,'observed_worker_cpu_lower_bound':observed,'charged_worker_CPU_upper_bound':charge,'uncertain_worker_CPU_upper_bound':max(0,charge-observed),'source':source,'bound':bound,'source_receipts':sources,'verified_final_wait4':verified_final,'exit_code':code,'job_status':'completed'if complete else 'scientific_blocked'if blocked else 'queued'})
 extra=max(0.,host_tail-worker_uncertain) # Covers a global host-tail upper bound, including unobserved manager CPU.
 recovery_id=args.recovery_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_boot_resource_recovery';record={'recovery_id':recovery_id,'authorization_id':a['latest']['authorization_id'],'at_utc':base.now(),'current_boot_id':current_boot,'cutoff_utc_bound':cutoff.isoformat(),'clock_margin_seconds':args.clock_margin,'old_host_cpu_cap':args.old_host_cpu_cap,'backup_path':str(args.backup.resolve()),'backup_sha256':base.sha(args.backup),'attempts':records,'extra_global_unobserved_resource_CPU_charge':extra,'observed_values_are_not_final_exact_CPU':True,'bound_requires_old_runtime_ended_by_cutoff':True,'original_scientific_meta_or_seals_changed':False,'already_closed_attempts_changed':False}
 d.execute('BEGIN IMMEDIATE')
 try:
  if d.execute('SELECT 1 FROM resource_recoveries WHERE recovery_id=?',(recovery_id,)).fetchone():d.rollback();print('Recovery ID already present; no double charge');return
  for r in records:
   old=r['before_attempt'];current=d.execute('SELECT state FROM attempts WHERE id=?',(old['id'],)).fetchone();assert current['state']!='closed','Concurrent recovery detected; do not double-account'
   d.execute("UPDATE attempts SET state='closed',verified_cpu=?,charged_cpu=?,uncertain_cpu=?,account_source=?,ended_utc=?,exit_code=? WHERE id=?",(r['observed_worker_cpu_lower_bound'],r['charged_worker_CPU_upper_bound'],r['uncertain_worker_CPU_upper_bound'],r['source'],base.now(),r['exit_code'],old['id']))
   d.execute('UPDATE jobs SET status=?,next_ready=?,last_error=? WHERE id=?',(r['job_status'],now_epoch,'Old-boot interruption; checkpoint preserved; CPU bounded conservatively',old['job_id']))
  d.execute('INSERT INTO resource_recoveries(recovery_id,record_json,entry_hash,extra_resource_charge_cpu)VALUES(?,?,?,?)',(recovery_id,canonical(record),digest(record),extra));d.commit()
 except BaseException:d.rollback();raise
 base.event(d,'audited_boot_resource_recovery',{'recovery_id':recovery_id,'attempt_count':len(records),'record_hash':digest(record)});base.atomic(root/'resource_recovery_latest.json',{**record,'entry_hash':digest(record)});d.close();print(json.dumps(record,ensure_ascii=False,indent=2))

class ExtendedLeader(base.Leader):
 def __init__(self,root):
  d=connect(root,True);auth=authority(root,d);assert auth['latest'],'No additional resource authorization';old=d.execute("SELECT COUNT(*)FROM attempts WHERE state!='closed'AND boot_id!=?",(base.boot(),)).fetchone()[0];d.close();assert old==0,'Explicit audited --recover-boot required before starting on a new boot'
  super().__init__(root);self.resource_tip=None;self.apply_authority();info=base.read(root/'leader.json',{});info.update(resource_extension=True,effective_budget_cpu_seconds=self.budget,resource_authorization_id=auth['latest']['authorization_id']);base.atomic(root/'leader.json',info)
 def apply_authority(self):
  auth=authority(self.root,self.d)
  if auth['tip']!=self.resource_tip:
   assert auth['total']>=getattr(self,'budget',0),'Resource budget cannot decrease';self.budget=auth['total'];self.maxworkers=int(auth['latest']['resource_settings']['max_workers']);self.resource_tip=auth['tip'];self.resource_auth=auth['latest'];base.event(self.d,'resource_budget_applied',{'authorization_id':self.resource_auth['authorization_id'],'effective_cpu_budget_seconds':self.budget,'max_workers':self.maxworkers,'original_scientific_budget_unchanged':auth['original']})
 def accounting(self):
  self.apply_authority();a=super().accounting();extra=self.d.execute('SELECT COALESCE(SUM(extra_resource_charge_cpu),0)FROM resource_recoveries').fetchone()[0];a['management_cpu']+=float(extra);return a
 def check_integrity(self,force=False):
  return super().check_integrity(force=force)and bool(authority(self.root,self.d))
 def launch(self,row,allocation):
  self.apply_authority();aid=uuid.uuid4().hex;out=self.root/'jobs'/row['id'];out.mkdir(parents=True,exist_ok=True);base.atomic(out/'job.json',json.loads(row['job_json']));adir=self.root/'attempts'/aid;adir.mkdir(parents=True);binding={'attempt_id':aid,'authorization_id':self.resource_auth['authorization_id'],'controller_sha256':ownsha(),'resource_settings':self.resource_auth['resource_settings'],'created_utc':base.now()}
  with self.d:
   self.d.execute("INSERT INTO attempts(id,job_id,state,allocation_cpu,boot_id,started_utc,attempt_dir)VALUES(?,?,'launching',?,?,?,?)",(aid,row['id'],allocation,base.boot(),base.now(),str(adir)));self.d.execute("UPDATE jobs SET status='running' WHERE id=?",(row['id'],));self.d.execute('INSERT INTO resource_attempt_bindings(attempt_id,authorization_id,record_json,entry_hash)VALUES(?,?,?,?)',(aid,binding['authorization_id'],canonical(binding),digest(binding)))
  with(adir/'runner.log').open('ab',buffering=0)as log:p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(self.root),'--runner',aid],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
  self.children[p.pid]='runner';base.event(self.d,'authorized_resource_runner_launched',{'attempt_id':aid,'pid':p.pid,'allocation_cpu':allocation,'binding_hash':digest(binding)})
 def report(self,state='running',reason=None):
  value=super().report(state,reason);value.update(resource_extension=True,original_scientific_budget_cpu_seconds=float(self.p['cpu_budget_seconds']),resource_authorization_id=self.resource_auth['authorization_id'],resource_authorization_tip=self.resource_tip);base.atomic(self.root/'status.json',value);return value


def extended_runner(root,aid):
 d=base.dbopen(root);a=dict(d.execute('SELECT *FROM attempts WHERE id=?',(aid,)).fetchone());bindingrow=d.execute('SELECT *FROM resource_attempt_bindings WHERE attempt_id=?',(aid,)).fetchone();assert bindingrow,'Resource binding missing';binding=json.loads(bindingrow['record_json']);assert digest(binding)==bindingrow['entry_hash']and binding['controller_sha256']==ownsha();science_check(root,d);adapter=root/binding['resource_settings']['worker_adapter'];assert base.sha(adapter)==binding['resource_settings']['worker_adapter_sha256'];ident=base.self_identity();base.manager(d,'runner');out=root/'jobs'/a['job_id'];adir=Path(a['attempt_dir']);_,p=base.config(root);python=p.get('worker_python',str(root/'.venv/bin/python')if(root/'.venv/bin/python').exists()else sys.executable);allocation=int(a['allocation_cpu']);child=None;wi=None
 def forward(sig,frame):
  target=base.matches(child.pid,wi['start_ticks'],wi['boot_id'])if child is not None and child.returncode is None and wi else None
  if target and target['state']!='Z':
   try:os.killpg(child.pid,signal.SIGTERM)
   except ProcessLookupError:pass
 for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,forward)
 signal.signal(signal.SIGHUP,signal.SIG_IGN)
 with d:d.execute("UPDATE attempts SET state='running',runner_pid=?,runner_start=?,boot_id=? WHERE id=?",(ident['pid'],ident['start_ticks'],ident['boot_id'],aid))
 base.atomic(adir/'runner_identity.json',ident);tmp=root/'tmpfiles'/aid;tmp.mkdir(parents=True,exist_ok=True,mode=0o700);os.chmod(tmp,0o700);env=os.environ.copy();env.update({k:'1'for k in base.THREAD_VARS});env.update(CHEM_ATTEMPT_ID=aid,CHEM_JOB_ID=a['job_id'],TMPDIR=str(tmp),PYSCF_TMPDIR=str(tmp))
 def prepare():resource.setrlimit(resource.RLIMIT_CPU,(max(1,allocation-2),max(2,allocation-1)));base.atomic(adir/'worker_identity.json',base.self_identity())
 cmd=[python,'-u',str(adapter),'--root',str(root),'--job',str(out/'job.json')]
 try:
  with(adir/'worker.log').open('ab',buffering=0)as log:child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=prepare,close_fds=True,env=env)
  wi=base.read(adir/'worker_identity.json');assert wi['pid']==child.pid
  with d:d.execute('UPDATE attempts SET worker_pid=?,worker_start=? WHERE id=?',(child.pid,wi['start_ticks'],aid))
  base.event(d,'resource_adapter_worker_started',{'attempt_id':aid,'command':cmd,'adapter_sha256':binding['resource_settings']['worker_adapter_sha256'],'worker_identity':wi,'scientific_plan_unchanged':True})
  while True:
   got,status,usage=os.wait4(child.pid,os.WNOHANG)
   if got:
    code=os.waitstatus_to_exitcode(status);child.returncode=code;used=usage.ru_utime+usage.ru_stime;base.atomic(adir/'wait4_receipt.json',{'attempt_id':aid,'exit_code':code,'verified_worker_cpu':used,'worker_identity':wi,'runner_identity':ident,'resource_adapter_sha256':binding['resource_settings']['worker_adapter_sha256'],'ended_utc':base.now()});base.finish_attempt(d,root,a,code,used,used,'authorized_resource_runner_wait4_exactly_once')
    if tmp.resolve().parent==(root/'tmpfiles').resolve():shutil.rmtree(tmp,ignore_errors=True)
    break
   wp=base.matches(child.pid,wi['start_ticks'],wi['boot_id'])
   if wp:
    with d:d.execute('UPDATE attempts SET observed_cpu=MAX(observed_cpu,?)WHERE id=?',(wp['self_cpu'],aid))
   base.manager(d,'runner');base.atomic(adir/'runner_heartbeat.json',{'updated_utc':base.now(),'identity':ident,'worker_identity':wi,'worker_observed_cpu':wp['self_cpu']if wp else None,'resource_adapter_sha256':binding['resource_settings']['worker_adapter_sha256']});time.sleep(2)
 except BaseException as e:
  base.event(d,'resource_runner_error',{'attempt_id':aid,'error':repr(e)})
  if child and wi and base.matches(child.pid,wi['start_ticks'],wi['boot_id']):raise
  base.finish_attempt(d,root,a,1,0.,float(a['allocation_cpu']),'resource_spawn_failure_conservative_allocation')
 finally:base.manager(d,'runner',final=time.process_time());d.close()


def live(info):
 if not isinstance(info,dict):return False
 p=base.matches(info.get('pid'),info.get('start_ticks'),info.get('boot_id'));return bool(p and p['state']!='Z')

def leader(root):
 with(root/'leader.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  control=None
  try:control=ExtendedLeader(root);control.run()
  except BaseException as e:
   import traceback;traceback.print_exc();base.atomic(root/'PAUSE',{'reason':'resource_extension_or_integrity_error','error':repr(e),'at_utc':base.now()})
   if control:control.shutdown('paused_integrity')
   else:base.atomic(root/'status.json',{'state':'paused_integrity','reason':repr(e),'updated_utc':base.now()})
   raise

def launch_leader(root):
 with(root/'resource_supervisor.log').open('ab',buffering=0)as log:p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(root),'--leader'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 print(json.dumps({'resource_leader_pid':p.pid}))

def watchdog(root):
 with(root/'watchdog.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  d=base.dbopen(root);ident=base.self_identity();base.atomic(root/'watchdog.json',{**ident,'resource_extension':True,'started_utc':base.now()});signal.signal(signal.SIGHUP,signal.SIG_IGN)
  try:
   while True:
    base.manager(d,'watchdog',include_children=True);state=base.read(root/'status.json',{}).get('state');info=base.read(root/'leader.json',{})
    if state in base.TERMINAL:return
    if(root/'STOP').exists()or(root/'PAUSE').exists():
     if live(info):os.kill(info['pid'],signal.SIGTERM)
     else:return
    elif not live(info):subprocess.run([sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--launch-leader'],stdin=subprocess.DEVNULL,check=False)
    time.sleep(float(base.config(root)[1].get('watchdog_seconds',15)))
  finally:
   p=base.proc(os.getpid());base.manager(d,'watchdog',final=p['self_cpu']+p['child_cpu'],include_children=True);d.close()

def start(args):
 root=args.root;d=connect(root,True);a=authority(root,d);assert a['latest'];old=d.execute("SELECT COUNT(*)FROM attempts WHERE state!='closed'AND boot_id!=?",(base.boot(),)).fetchone()[0];d.close();assert old==0,'Audited recovery of old-boot attempts is required first';assert not(root/'STOP').exists()and not(root/'PAUSE').exists(),'STOP/PAUSE must be explicitly reviewed; never silently clear it';status=base.read(root/'status.json',{});assert status.get('state')not in ['all_tasks_completed','scientific_blocked'],'No repeat of completed or scientifically blocked plan'
 current=base.read(root/'leader.json',{});assert not live(current)or current.get('resource_extension'),'Original live controller requires an explicit controller-only takeover; do not interrupt DFT runners'
 if status.get('state')in ['budget_exhausted','stopped_by_user','paused_integrity']:raise RuntimeError('Review and archive the previous terminal state before an explicitly authorized continuation')
 with(root/'launcher.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if not live(base.read(root/'watchdog.json',{})):
   with(root/'resource_watchdog.log').open('ab',buffering=0)as log:p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(root),'--watchdog'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
  deadline=time.monotonic()+5
  while time.monotonic()<deadline:
   if live(base.read(root/'leader.json',{})):break
   time.sleep(.1)
 print(json.dumps({'authorization':a,'leader':base.read(root/'leader.json',{}),'watchdog':base.read(root/'watchdog.json',{}),'status':base.read(root/'status.json',{})},ensure_ascii=False,indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);action=p.add_mutually_exclusive_group(required=True);action.add_argument('--grant',dest='additional',type=float);action.add_argument('--recover-boot',action='store_true');action.add_argument('--start',action='store_true');action.add_argument('--watchdog',action='store_true');action.add_argument('--leader',action='store_true');action.add_argument('--launch-leader',action='store_true');action.add_argument('--runner');action.add_argument('--inspect',action='store_true');p.add_argument('--authorization-id');p.add_argument('--user-text');p.add_argument('--backup',type=Path);p.add_argument('--max-workers',type=int,choices=[1,2],default=1);p.add_argument('--worker-adapter',default='resource_worker.py');p.add_argument('--cutoff-utc');p.add_argument('--clock-margin',type=float,default=300);p.add_argument('--old-host-cpu-cap',type=float,default=4);p.add_argument('--recovery-id');a=p.parse_args();a.root=a.root.resolve()
 if a.additional is not None:assert a.authorization_id and a.user_text;grant(a)
 elif a.recover_boot:assert a.cutoff_utc;recover_boot(a)
 elif a.start:start(a)
 elif a.watchdog:watchdog(a.root)
 elif a.leader:leader(a.root)
 elif a.launch_leader:launch_leader(a.root)
 elif a.runner:extended_runner(a.root,a.runner)
 else:
  d=connect(a.root,True);value=authority(a.root,d);active=[dict(r)for r in d.execute("SELECT *FROM attempts WHERE state!='closed'")];d.close();print(json.dumps({'authorization':value,'current_boot':base.boot(),'active_attempts':active,'leader':base.read(a.root/'leader.json',{}),'status':base.read(a.root/'status.json',{})},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
