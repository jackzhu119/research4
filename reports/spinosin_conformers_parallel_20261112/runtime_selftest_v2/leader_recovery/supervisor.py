#!/usr/bin/env python3
"""Independent molecular-computation queue with persistent per-worker reapers.
A leader crash leaves each thin runner owning its DFT child, so wait4 accounting
remains exact and is committed once by attempt ID. No scientific imports here.
"""
import argparse,fcntl,hashlib,json,math,os,resource,signal,sqlite3,subprocess,sys,time,uuid,shutil
from pathlib import Path
from datetime import datetime,timezone
TERMINAL={'budget_exhausted','stopped_by_user','all_tasks_completed','scientific_blocked','paused_integrity'}
THREAD_VARS=['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS']

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p,default=None):
 try:return json.loads(Path(p).read_text())
 except (OSError,ValueError):return default

def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex)
 with t.open('w')as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
 try:fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 except OSError:pass

def boot():return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def proc(pid):
 try:
  s=Path(f'/proc/{int(pid)}/stat').read_text();v=s[s.rfind(')')+2:].split();hz=os.sysconf('SC_CLK_TCK')
  return {'pid':int(pid),'ppid':int(v[1]),'state':v[0],'start_ticks':int(v[19]),'boot_id':boot(),'self_cpu':(int(v[11])+int(v[12]))/hz,'child_cpu':(int(v[13])+int(v[14]))/hz}
 except (OSError,ValueError,TypeError,IndexError):return None

def matches(pid,start,b):
 p=proc(pid);return p if p and p['start_ticks']==start and p['boot_id']==b else None

def self_identity():
 p=proc(os.getpid());return {'pid':p['pid'],'start_ticks':p['start_ticks'],'boot_id':p['boot_id'],'sid':os.getsid(0)}
def dbopen(root):
 d=sqlite3.connect(root/'queue.sqlite3',timeout=15);d.row_factory=sqlite3.Row;d.execute('PRAGMA journal_mode=WAL');d.execute('PRAGMA synchronous=FULL');d.execute('PRAGMA busy_timeout=15000');d.executescript('''
 CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
 CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,ordinal INTEGER UNIQUE,job_json TEXT NOT NULL,status TEXT NOT NULL DEFAULT 'queued',failure_count INTEGER NOT NULL DEFAULT 0,next_ready REAL NOT NULL DEFAULT 0,last_error TEXT);
 CREATE TABLE IF NOT EXISTS attempts(id TEXT PRIMARY KEY,job_id TEXT NOT NULL,state TEXT NOT NULL,allocation_cpu REAL NOT NULL,runner_pid INTEGER,runner_start INTEGER,worker_pid INTEGER,worker_start INTEGER,boot_id TEXT,observed_cpu REAL NOT NULL DEFAULT 0,verified_cpu REAL NOT NULL DEFAULT 0,charged_cpu REAL NOT NULL DEFAULT 0,uncertain_cpu REAL NOT NULL DEFAULT 0,account_source TEXT,started_utc TEXT,ended_utc TEXT,exit_code INTEGER,attempt_dir TEXT NOT NULL);
 CREATE TABLE IF NOT EXISTS managers(identity TEXT PRIMARY KEY,role TEXT NOT NULL,pid INTEGER,start_ticks INTEGER,boot_id TEXT,observed_cpu REAL NOT NULL DEFAULT 0,final_cpu REAL,charge_cpu REAL NOT NULL DEFAULT 0,uncertain_cpu REAL NOT NULL DEFAULT 0,state TEXT NOT NULL DEFAULT 'running');
 CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,utc TEXT,event TEXT,detail TEXT);
 ''');d.commit();return d

def event(d,name,detail):
 with d:d.execute('INSERT INTO events(utc,event,detail)VALUES(?,?,?)',(now(),name,json.dumps(detail,ensure_ascii=False)))
def mkey(p):return f'{p["boot_id"]}:{p["pid"]}:{p["start_ticks"]}'
def manager(d,role,identity=None,final=None,include_children=False):
 p=proc(os.getpid())if identity is None else identity;k=mkey(p);value=p['self_cpu']+(p.get('child_cpu',0)if include_children else 0)
 with d:
  d.execute('INSERT OR IGNORE INTO managers(identity,role,pid,start_ticks,boot_id)VALUES(?,?,?,?,?)',(k,role,p['pid'],p['start_ticks'],p['boot_id']))
  d.execute('UPDATE managers SET observed_cpu=MAX(observed_cpu,?),charge_cpu=MAX(charge_cpu,?) WHERE identity=?',(value,value,k))
  if final is not None:d.execute("UPDATE managers SET final_cpu=?,charge_cpu=MAX(charge_cpu,?),state='closed' WHERE identity=?",(final,final+.1,k))
 return k

def freeze(d,key,value):
 old=d.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone()
 if old and old['value']!=value:raise RuntimeError('Frozen metadata mismatch: '+key)
 with d:d.execute('INSERT OR IGNORE INTO meta(key,value)VALUES(?,?)',(key,value))

def config(root):
 path=root/'protocol.json'if(root/'protocol.json').exists()else root/'chemistry_protocol.json';p=read(path);assert isinstance(p,dict),'Missing protocol';return path,p

def seal_inputs(root,p,jobs):
 paths={root/'worker.py',root/'jobs.json'}
 protocol,_=config(root);paths.add(protocol)
 for key in ['input_hashes','sealed_files','sealed_hashes']:
  for name,value in p.get(key,{}).items():
   f=Path(name);f=f if f.is_absolute()else root/f;assert f.is_file(),'Missing sealed input: '+str(f)
   if isinstance(value,str)and len(value)==64:assert sha(f)==value,'Declared hash mismatch: '+str(f)
   paths.add(f)
 for job in jobs:
  for key in ['input_path','input_file','xyz_path','sdf_path','geometry_path','xyz','sdf']:
   value=job.get(key)
   if isinstance(value,str)and '\n'not in value:
    f=Path(value);f=f if f.is_absolute()else root/f
    if f.is_file():paths.add(f)
 return {str(f.resolve()):sha(f)for f in sorted(paths)}

def finish_attempt(d,root,a,code,verified,charge,source):
 """A transaction owns final accounting and job transition, exactly once."""
 result=read(root/'jobs'/a['job_id']/'result.json',{});ok=code==0 and result.get('complete')is True and result.get('status')=='complete';blocked=code==76;waiting=code==77;integrity=code==78
 protocol,p=config(root);initial=float(p.get('retry_initial_seconds',10));maximum=float(p.get('retry_max_seconds',600))
 d.execute('BEGIN IMMEDIATE')
 try:
  old=d.execute('SELECT state FROM attempts WHERE id=?',(a['id'],)).fetchone()
  if not old or old['state']=='closed':d.rollback();return
  row=d.execute('SELECT failure_count FROM jobs WHERE id=?',(a['job_id'],)).fetchone();fail=int(row['failure_count'])+(0 if code==75 or waiting or ok or blocked else 1);backoff=600 if waiting else 1 if code==75 else min(maximum,initial*2**min(fail,12));status='completed'if ok else 'scientific_blocked'if blocked else 'queued'
  d.execute("UPDATE attempts SET state='closed',verified_cpu=?,charged_cpu=?,uncertain_cpu=?,account_source=?,ended_utc=?,exit_code=? WHERE id=?",(verified,charge,max(0,charge-verified),source,now(),code,a['id']))
  d.execute('UPDATE jobs SET status=?,failure_count=?,next_ready=?,last_error=? WHERE id=?',(status,fail,0 if ok or blocked else time.time()+backoff,None if ok else f'exit={code}; {source}',a['job_id']));d.commit()
 except BaseException:d.rollback();raise
 if integrity:atomic(root/'PAUSE',{'reason':'worker_integrity_exit78','attempt_id':a['id'],'at_utc':now()})
 event(d,'attempt_closed',{'attempt_id':a['id'],'job_id':a['job_id'],'exit_code':code,'complete':ok,'scientific_blocked':blocked,'dependency_wait':waiting,'verified_cpu':verified,'conservative_charge_cpu':charge,'source':source,'failure_count':fail,'retry_backoff_seconds':None if ok or blocked else backoff})


def runner(root,aid):
 """This process survives the leader and remains the worker's wait4 parent."""
 d=dbopen(root);a=dict(d.execute('SELECT * FROM attempts WHERE id=?',(aid,)).fetchone());ident=self_identity();manager(d,'runner');out=root/'jobs'/a['job_id'];adir=Path(a['attempt_dir']);protocol,p=config(root);python=p.get('worker_python',str(root/'.venv/bin/python')if(root/'.venv/bin/python').exists()else sys.executable);allocation=int(a['allocation_cpu']);child=None;wi=None;stopping=False
 def forward(sig,frame):
  nonlocal stopping;stopping=True
  if child is not None and child.returncode is None and wi is not None and matches(child.pid,wi['start_ticks'],wi['boot_id'])and proc(child.pid)['state']!='Z':
   try:os.killpg(child.pid,signal.SIGTERM)
   except ProcessLookupError:pass
 for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,forward)
 signal.signal(signal.SIGHUP,signal.SIG_IGN)
 with d:d.execute("UPDATE attempts SET state='running',runner_pid=?,runner_start=?,boot_id=? WHERE id=?",(ident['pid'],ident['start_ticks'],ident['boot_id'],aid))
 atomic(adir/'runner_identity.json',ident)
 tmpdir=root/'tmpfiles'/aid;tmpdir.mkdir(parents=True,exist_ok=True,mode=0o700);os.chmod(tmpdir,0o700)
 env=os.environ.copy();env.update({k:'1'for k in THREAD_VARS});env.update(CHEM_ATTEMPT_ID=aid,CHEM_JOB_ID=a['job_id'],TMPDIR=str(tmpdir),PYSCF_TMPDIR=str(tmpdir))
 def prepare():
  resource.setrlimit(resource.RLIMIT_CPU,(max(1,allocation-2),max(2,allocation-1)));w=self_identity();atomic(adir/'worker_identity.json',w)
  nice=int(p.get('worker_nice',0))
  if nice:os.nice(nice)
 cmd=[python,'-u',str(root/'worker.py'),'--root',str(root),'--job',str(out/'job.json')]
 try:
  with(adir/'worker.log').open('ab',buffering=0)as log:child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=prepare,close_fds=True,env=env)
  wi=read(adir/'worker_identity.json');assert wi and wi['pid']==child.pid
  with d:d.execute('UPDATE attempts SET worker_pid=?,worker_start=? WHERE id=?',(child.pid,wi['start_ticks'],aid))
  event(d,'worker_started',{'attempt_id':aid,'worker_identity':wi,'runner_identity':ident,'allocated_cpu_limit':allocation,'single_thread':True})
  while True:
   got,status,usage=os.wait4(child.pid,os.WNOHANG)
   if got:
    code=os.waitstatus_to_exitcode(status);child.returncode=code;used=usage.ru_utime+usage.ru_stime;atomic(adir/'wait4_receipt.json',{'attempt_id':aid,'exit_code':code,'verified_worker_cpu':used,'worker_identity':wi,'runner_identity':ident,'ended_utc':now()});finish_attempt(d,root,a,code,used,used,'persistent_runner_wait4_exactly_once')
    # DFT scratch is separate from all scientific checkpoints and removed only after wait4.
    if tmpdir.resolve().parent==(root/'tmpfiles').resolve():shutil.rmtree(tmpdir,ignore_errors=True)
    break
   wp=matches(child.pid,wi['start_ticks'],wi['boot_id'])
   if wp:
    with d:d.execute('UPDATE attempts SET observed_cpu=MAX(observed_cpu,?) WHERE id=?',(wp['self_cpu'],aid))
   manager(d,'runner');atomic(adir/'runner_heartbeat.json',{'updated_utc':now(),'identity':ident,'worker_identity':wi,'worker_observed_cpu':wp['self_cpu']if wp else None,'worker_stopping':stopping});time.sleep(2)
 except BaseException as e:
  event(d,'runner_error',{'attempt_id':aid,'error':repr(e)})
  if child and matches(child.pid,wi['start_ticks'],wi['boot_id']):
   # Never abandon an unaccounted live DFT process. Preserve its identity for recovery.
   atomic(adir/'runner_error.json',{'error':repr(e),'worker_identity':wi,'at_utc':now()});raise
  finish_attempt(d,root,a,1,0.,float(a['allocation_cpu']),'spawn_or_unreapable_failure_conservative_allocation')
 finally:manager(d,'runner',final=time.process_time());d.close()

class Leader:
 def __init__(self,root):
  self.root=root;self.d=dbopen(root);self.identity=self_identity();self.protocol_path,self.p=config(root);self.budget=float(self.p.get('cpu_budget_seconds',900000));self.reserve=float(self.p.get('management_cpu_reserve_seconds',self.p.get('manager_cpu_reserve_seconds',5000)));self.margin=float(self.p.get('shutdown_cpu_margin_seconds',200));self.maxworkers=int(self.p.get('max_workers',2));self.poll=float(self.p.get('poll_seconds',5));assert 1<=self.maxworkers<=2 and self.budget>self.reserve>=0
  manifest=read(root/'jobs.json');jobs=manifest['jobs']if isinstance(manifest,dict)else manifest;assert isinstance(jobs,list)and jobs,'No frozen jobs';seen=set()
  for i,j in enumerate(jobs):
   j.setdefault('ordinal',i);assert isinstance(j.get('id'),str)and all(c.isalnum()or c in '_-.'for c in j['id'])and j['id']not in seen;seen.add(j['id'])
  for j in jobs:assert set(j.get('depends_on',[]))<=seen
  self.seals=seal_inputs(root,self.p,jobs);freeze(self.d,'input_seals',json.dumps(self.seals,sort_keys=True));freeze(self.d,'budget',str(self.budget));self.last_hash=0.;self.stopping=False;self.stop_reason=None;self.publish=None;self.last_publish=time.monotonic();self.children={}
  with self.d:
   for j in jobs:
    text=json.dumps(j,sort_keys=True,separators=(',',':'));old=self.d.execute('SELECT job_json FROM jobs WHERE id=?',(j['id'],)).fetchone();assert not old or old['job_json']==text,'Frozen job changed'
    self.d.execute('INSERT OR IGNORE INTO jobs(id,ordinal,job_json)VALUES(?,?,?)',(j['id'],j['ordinal'],text))
  atomic(root/'leader.json',{**self.identity,'started_utc':now(),'budget_cpu_seconds':self.budget});manager(self.d,'leader');event(self.d,'leader_started',self.identity)
  for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,self.stop_signal)
  signal.signal(signal.SIGHUP,signal.SIG_IGN)
 def stop_signal(self,sig,frame):self.stopping=True;self.stop_reason='stopped_by_user'
 def check_integrity(self):
  if time.monotonic()-self.last_hash<30:return True
  self.last_hash=time.monotonic();changed=[f for f,h in self.seals.items()if not Path(f).is_file()or sha(f)!=h]
  if changed:atomic(self.root/'PAUSE',{'reason':'sealed_input_or_configuration_changed','files':changed,'at_utc':now()});event(self.d,'integrity_pause',{'changed_files':changed});return False
  return True
 def recover_and_monitor(self):
  for rr in self.d.execute("SELECT * FROM attempts WHERE state!='closed'").fetchall():
   a=dict(rr);adir=Path(a['attempt_dir']);ri=read(adir/'runner_identity.json',{});wi=read(adir/'worker_identity.json',{});receipt=read(adir/'wait4_receipt.json',{})
   if receipt.get('attempt_id')==a['id']:
    finish_attempt(self.d,self.root,a,receipt['exit_code'],receipt['verified_worker_cpu'],receipt['verified_worker_cpu'],'recovered_persistent_wait4_receipt');continue
   rp=matches(ri.get('pid',a.get('runner_pid')),ri.get('start_ticks',a.get('runner_start')),ri.get('boot_id',a.get('boot_id')))
   wp=matches(wi.get('pid',a.get('worker_pid')),wi.get('start_ticks',a.get('worker_start')),wi.get('boot_id',a.get('boot_id')))
   if wp:
    with self.d:self.d.execute('UPDATE attempts SET worker_pid=?,worker_start=?,boot_id=?,observed_cpu=MAX(observed_cpu,?) WHERE id=?',(wp['pid'],wp['start_ticks'],wp['boot_id'],wp['self_cpu'],a['id']))
   if rp and rp['state']!='Z':continue
   if wp and wp['state']!='Z':continue # Do not restart a still-running orphan worker.
   if wp and wp['state']=='Z':
    used=wp['self_cpu'];worker_result=read(self.root/'jobs'/a['job_id']/'result.json',{});code=0 if worker_result.get('complete')is True and worker_result.get('status')=='complete'else 75;finish_attempt(self.d,self.root,a,code,used,used+.03,'orphan_worker_final_proc_ticks_plus_guard')
   elif time.time()-Path(a['attempt_dir']).stat().st_mtime>10:
    # Neither a reaper receipt nor a final kernel counter survives: do not forgive unverified CPU.
    finish_attempt(self.d,self.root,a,75,float(a['observed_cpu']),max(float(a['allocation_cpu']),float(a['observed_cpu'])),'unverifiable_orphan_conservative_allocation')
  # Reap runner/publisher children; their nested DFT CPU was already accounted by the runner.
  for pid,role in list(self.children.items()):
   try:
    got,status,usage=os.wait4(pid,os.WNOHANG)
    if got:
     if role=='publisher':
      row=self.d.execute("SELECT * FROM managers WHERE pid=? AND role='publisher' ORDER BY rowid DESC LIMIT 1",(pid,)).fetchone()
      if row:
       with self.d:self.d.execute("UPDATE managers SET final_cpu=?,charge_cpu=MAX(charge_cpu,?),state='closed' WHERE identity=?",(usage.ru_utime+usage.ru_stime,usage.ru_utime+usage.ru_stime+.1,row['identity']))
      if os.waitstatus_to_exitcode(status)!=0:event(self.d,'publisher_sync_error',{'exit_code':os.waitstatus_to_exitcode(status)})
     self.children.pop(pid,None)
   except ChildProcessError:self.children.pop(pid,None)
  for row in self.d.execute("SELECT * FROM managers WHERE state='running'").fetchall():
   if row['role']=='leader'and row['pid']==os.getpid():continue
   p=matches(row['pid'],row['start_ticks'],row['boot_id'])
   if p:
    value=p['self_cpu']+(p['child_cpu']if row['role']in ['publisher','watchdog']else 0)
    with self.d:self.d.execute('UPDATE managers SET observed_cpu=MAX(observed_cpu,?),charge_cpu=MAX(charge_cpu,?) WHERE identity=?',(value,value,row['identity']))
   elif row['role']!='runner' or not self.d.execute("SELECT 1 FROM attempts WHERE runner_pid=? AND state!='closed'",(row['pid'],)).fetchone():
    with self.d:self.d.execute("UPDATE managers SET charge_cpu=MAX(charge_cpu,observed_cpu+100),uncertain_cpu=MAX(uncertain_cpu,100),state='lost_conservative' WHERE identity=? AND state='running'",(row['identity'],))
 def accounting(self):
  c=self.d.execute("SELECT COALESCE(SUM(charged_cpu),0)c,COALESCE(SUM(verified_cpu),0)v,COALESCE(SUM(uncertain_cpu),0)u FROM attempts WHERE state='closed'").fetchone();r=self.d.execute("SELECT COALESCE(SUM(allocation_cpu),0)a,COALESCE(SUM(observed_cpu),0)o,COUNT(*)n FROM attempts WHERE state!='closed'").fetchone();m=self.d.execute('SELECT COALESCE(SUM(charge_cpu),0)c FROM managers').fetchone();return {'closed_charge':float(c['c']),'closed_verified':float(c['v']),'worker_uncertain':float(c['u']),'active_allocated':float(r['a']),'active_observed':float(r['o']),'active_workers':r['n'],'management_cpu':float(m['c'])}
 def report(self,state='running',reason=None):
  manager(self.d,'leader');a=self.accounting();counts={r['status']:r['n']for r in self.d.execute('SELECT status,COUNT(*)n FROM jobs GROUP BY status')};active=[]
  for row in self.d.execute("SELECT * FROM attempts WHERE state!='closed'"):
   h=read(self.root/'jobs'/row['job_id']/'heartbeat.json',{});active.append({'attempt_id':row['id'],'job_id':row['job_id'],'worker_pid':row['worker_pid'],'worker_start_ticks':row['worker_start'],'runner_pid':row['runner_pid'],'boot_id':row['boot_id'],'worker_cpu_observed':row['observed_cpu'],'budget_allocation_cpu':row['allocation_cpu'],'heartbeat':h})
  v={'state':state,'reason':reason,'updated_utc':now(),'leader_identity':self.identity,'cpu_budget_seconds':self.budget,'management_reserve_cpu_seconds':self.reserve,'verified_completed_worker_cpu':a['closed_verified'],'worker_cpu_observed':a['closed_verified']+a['active_observed'],'conservative_completed_worker_charge':a['closed_charge'],'unknown_worker_cpu_upper_bound':a['worker_uncertain'],'active_allocated_worker_cpu':a['active_allocated'],'management_observed_or_conservative_cpu':a['management_cpu'],'total_cpu_observed_or_conservative':a['closed_charge']+a['active_observed']+a['management_cpu'],'active_workers':a['active_workers'],'max_workers':self.maxworkers,'jobs':counts,'workers':active,'stop_file':str(self.root/'STOP'),'pause_file':str(self.root/'PAUSE'),'no_score_or_improvement_stopping':True,'frozen_input_hashes':self.seals,'old_ML_budget_or_processes_modified':False};atomic(self.root/'status.json',v);return v
 def launch(self,row,allocation):
  aid=uuid.uuid4().hex;out=self.root/'jobs'/row['id'];out.mkdir(parents=True,exist_ok=True);atomic(out/'job.json',json.loads(row['job_json']));adir=self.root/'attempts'/aid;adir.mkdir(parents=True)
  with self.d:
   self.d.execute("INSERT INTO attempts(id,job_id,state,allocation_cpu,boot_id,started_utc,attempt_dir)VALUES(?,?,'launching',?,?,?,?)",(aid,row['id'],allocation,boot(),now(),str(adir)));self.d.execute("UPDATE jobs SET status='running' WHERE id=?",(row['id'],))
  with(adir/'runner.log').open('ab',buffering=0)as log:p=subprocess.Popen([sys.executable,'-u',str(self.root/'supervisor.py'),'--root',str(self.root),'--runner',aid],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
  self.children[p.pid]='runner';event(self.d,'runner_launched',{'attempt_id':aid,'pid':p.pid,'job_id':row['id'],'allocated_budget_cpu':allocation})
 def publish_if_due(self):
  if not(self.root/'publisher.py').exists()or time.monotonic()-self.last_publish<600:return
  if any(role=='publisher'for role in self.children.values()):return
  with(self.root/'publisher.log').open('ab',buffering=0)as log:p=subprocess.Popen([sys.executable,str(self.root/'publisher.py'),'--root',str(self.root)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
  ident=proc(p.pid)
  if ident:manager(self.d,'publisher',identity=ident)
  self.children[p.pid]='publisher';self.last_publish=time.monotonic()
 def final_command(self,script,role):
  if not(self.root/script).exists():return
  a=self.accounting();remaining=self.budget-a['closed_charge']-a['management_cpu']-a['active_observed'];limit=int(math.floor(remaining-3))
  if limit<3:event(self.d,'final_report_skipped_budget',{'role':role,'remaining_cpu':remaining});return
  def prepare():resource.setrlimit(resource.RLIMIT_CPU,(max(1,limit-2),max(2,limit-1)))
  with(self.root/'publication.log').open('ab',buffering=0)as log:p=subprocess.Popen([sys.executable,str(self.root/script),'--root',str(self.root)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,preexec_fn=prepare)
  ident=proc(p.pid)
  if ident:manager(self.d,role,identity=ident)
  pid,status,usage=os.wait4(p.pid,0);p.returncode=os.waitstatus_to_exitcode(status);used=usage.ru_utime+usage.ru_stime
  if ident:
   with self.d:self.d.execute("UPDATE managers SET final_cpu=?,charge_cpu=MAX(charge_cpu,?),state='closed' WHERE identity=?",(used,used+.1,mkey(ident)))
  if p.returncode:event(self.d,'final_publication_error',{'role':role,'exit_code':p.returncode,'cpu':used})
 def shutdown(self,reason):
  self.report('stopping',reason)
  def signal_workers(sig):
   for a in self.d.execute("SELECT * FROM attempts WHERE state!='closed'"):
    wi=read(Path(a['attempt_dir'])/'worker_identity.json',{});p=matches(wi.get('pid',a['worker_pid']),wi.get('start_ticks',a['worker_start']),wi.get('boot_id',a['boot_id']))
    if p and p['state']!='Z':
     try:os.killpg(p['pid'],sig)
     except ProcessLookupError:pass
  signal_workers(signal.SIGTERM);deadline=time.monotonic()+float(self.p.get('shutdown_grace_seconds',10))
  while time.monotonic()<deadline:
   self.recover_and_monitor()
   if not self.accounting()['active_workers']:break
   time.sleep(.2)
  if self.accounting()['active_workers']:signal_workers(signal.SIGKILL)
  deadline=time.monotonic()+15
  while self.accounting()['active_workers']and time.monotonic()<deadline:self.recover_and_monitor();time.sleep(.2)
  self.report('finalizing',reason);self.final_command('summarize_results.py','final_summary');self.final_command('publisher.py','final_publisher');manager(self.d,'leader',final=time.process_time());self.report(reason,reason);event(self.d,'leader_terminal',{'state':reason});self.d.close()
 def run(self):
  while True:
   self.recover_and_monitor()
   if self.stopping or(self.root/'STOP').exists():self.shutdown('stopped_by_user');return
   if(self.root/'PAUSE').exists()or not self.check_integrity():self.shutdown('paused_integrity');return
   a=self.accounting()
   if a['closed_charge']+a['active_observed']+a['management_cpu']>=self.budget-self.margin:self.shutdown('budget_exhausted');return
   # Reservations cover whole active worker lifetimes, not arbitrary hourly timeouts.
   free=self.budget-max(self.reserve,a['management_cpu']+self.margin)-a['closed_charge']-a['active_allocated'];slots=self.maxworkers-a['active_workers']
   while slots>0 and free>=3:
    ready=None;done={r['id']for r in self.d.execute("SELECT id FROM jobs WHERE status='completed'")}
    for row in self.d.execute("SELECT * FROM jobs WHERE status='queued' AND next_ready<=? ORDER BY ordinal",(time.time(),)):
     if set(json.loads(row['job_json']).get('depends_on',[]))<=done:ready=dict(row);break
    if ready is None:break
    allocation=int(math.floor(free/slots));self.launch(ready,allocation);a=self.accounting();free=self.budget-max(self.reserve,a['management_cpu']+self.margin)-a['closed_charge']-a['active_allocated'];slots=self.maxworkers-a['active_workers']
   total=self.d.execute('SELECT COUNT(*)FROM jobs').fetchone()[0];completed=self.d.execute("SELECT COUNT(*)FROM jobs WHERE status='completed'").fetchone()[0]
   if completed==total and not self.accounting()['active_workers']:self.shutdown('all_tasks_completed');return
   blocked=self.d.execute("SELECT COUNT(*)FROM jobs WHERE status='scientific_blocked'").fetchone()[0]
   if completed+blocked==total and not self.accounting()['active_workers']:self.shutdown('scientific_blocked');return
   if not self.accounting()['active_workers']and free<3:self.shutdown('budget_exhausted');return
   self.report('running'if self.accounting()['active_workers']else 'retry_backoff_or_dependencies');self.publish_if_due();time.sleep(self.poll)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--runner');a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
 if a.runner:runner(root,a.runner);return
 with(root/'leader.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  leader=None
  try:leader=Leader(root);leader.run()
  except BaseException as e:
   import traceback;traceback.print_exc();atomic(root/'PAUSE',{'reason':'supervisor_or_integrity_error','error':repr(e),'at_utc':now()})
   if leader:leader.shutdown('paused_integrity')
   else:atomic(root/'status.json',{'state':'paused_integrity','reason':repr(e),'updated_utc':now()})
   raise
if __name__=='__main__':main()
