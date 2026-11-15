#!/usr/bin/env python3
"""Detached, budget-reserved SQLite training supervisor. Accuracy is never a stop input."""
import argparse,contextlib,fcntl,hashlib,json,math,os,resource,signal,sqlite3,subprocess,sys,time,uuid
from pathlib import Path
from datetime import datetime,timezone
TERMINAL={'budget_exhausted','stopped_by_user','fatal_configuration_error'}
THREAD_VARS=['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS']

def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex)
 with t.open('w')as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
 try:
  fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 except OSError:pass

def read(p,default=None):
 try:return json.loads(Path(p).read_text())
 except (OSError,ValueError):return default

def proc(pid):
 try:
  s=Path(f'/proc/{int(pid)}/stat').read_text();a=s[s.rfind(')')+2:].split();return {'pid':int(pid),'state':a[0],'ppid':int(a[1]),'pgid':int(a[2]),'sid':int(a[3]),'cpu_seconds':(int(a[11])+int(a[12]))/os.sysconf('SC_CLK_TCK'),'start_ticks':int(a[19])}
 except (OSError,ValueError,IndexError,TypeError):return None

def alive_identity(pid,start,boot,expected_boot):
 p=proc(pid);return p if p and p['start_ticks']==start and boot==expected_boot else None

def connect(root):
 db=sqlite3.connect(root/'queue.sqlite3',timeout=30);db.row_factory=sqlite3.Row;db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL');db.execute('PRAGMA busy_timeout=30000')
 db.executescript('''CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
 CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,ordinal INTEGER UNIQUE NOT NULL,job_json TEXT NOT NULL,status TEXT NOT NULL DEFAULT 'queued',failure_count INTEGER NOT NULL DEFAULT 0,attempt_count INTEGER NOT NULL DEFAULT 0,result_path TEXT,last_error TEXT);
 CREATE TABLE IF NOT EXISTS attempts(id TEXT PRIMARY KEY,job_id TEXT NOT NULL,pid INTEGER,start_ticks INTEGER,boot_id TEXT,pgid INTEGER,status TEXT NOT NULL,lease_cpu REAL NOT NULL,observed_cpu REAL NOT NULL DEFAULT 0,verified_actual_cpu REAL NOT NULL DEFAULT 0,charged_cpu REAL NOT NULL DEFAULT 0,uncertain_upper_cpu REAL NOT NULL DEFAULT 0,final_source TEXT,started_utc TEXT,ended_utc TEXT,exit_code INTEGER,exit_signal INTEGER,out_dir TEXT,attempt_dir TEXT,heartbeat_path TEXT,last_heartbeat_utc TEXT);
 CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,utc TEXT NOT NULL,event TEXT NOT NULL,detail TEXT NOT NULL);''');db.commit();return db

def event(db,name,detail):db.execute('INSERT INTO events(utc,event,detail)VALUES(?,?,?)',(utc(),name,json.dumps(detail,ensure_ascii=False)));db.commit()

def meta_freeze(db,key,value):
 old=db.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone()
 if old and old['value']!=value:raise RuntimeError(f'Frozen ledger metadata changed: {key}')
 if not old:db.execute('INSERT INTO meta VALUES(?,?)',(key,value));db.commit()

def import_manifest(db,root):
 manifest=read(root/'job_manifest.json');assert isinstance(manifest,dict)and isinstance(manifest.get('jobs'),list),'Missing jobs manifest'
 added=0
 with db:
  for j in manifest['jobs']:
   assert all(k in j for k in ['id','ordinal','seed','fold','family','mc_draws','task','data_path']),f'Invalid job:{j}'
   assert isinstance(j['id'],str)and all(c.isalnum()or c in '_-.'for c in j['id']),'Unsafe job ID'
   s=json.dumps(j,sort_keys=True,separators=(',',':'),ensure_ascii=False);old=db.execute('SELECT job_json FROM jobs WHERE id=?',(j['id'],)).fetchone()
   if old:assert old['job_json']==s,f'Existing frozen job changed:{j["id"]}'
   else:db.execute('INSERT INTO jobs(id,ordinal,job_json)VALUES(?,?,?)',(j['id'],j['ordinal'],s));added+=1
 return added

class Supervisor:
 def __init__(self,root):
  self.root=root;self.protocol=read(root/'protocol.json');assert isinstance(self.protocol,dict),'protocol.json missing';p=self.protocol
  self.budget=float(p.get('cpu_budget_seconds',90000));self.workers=int(p.get('max_workers',4));self.lease=int(p.get('worker_cpu_lease_seconds',1800));self.reserve=float(p.get('management_cpu_reserve_seconds',min(100,self.budget*.02)));self.poll=float(p.get('poll_seconds',2));self.retries=int(p.get('max_attempts',3));assert self.budget>self.reserve>=0 and self.workers>=1 and self.lease>=3
  self.allowance=self.budget-self.reserve;self.boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();self.db=connect(root);meta_freeze(self.db,'protocol_sha256',sha(root/'protocol.json'));meta_freeze(self.db,'worker_sha256',sha(root/'study_worker.py'));self.worker_hash=sha(root/'study_worker.py');self.running={};self.stopping=False;self.stop_reason=None;self.run_id=uuid.uuid4().hex;self.started=utc();self.started_wall=time.monotonic();self.control_cpu_start=time.process_time();self.management_children_cpu=0.;self.last_extension=0.;import_manifest(self.db,root)
  for signum in [signal.SIGTERM,signal.SIGINT]:signal.signal(signum,self.signal_stop)
  signal.signal(signal.SIGHUP,signal.SIG_IGN)
  ident=proc(os.getpid());atomic(root/'leader.json',{'pid':os.getpid(),'pgid':os.getpgrp(),'sid':os.getsid(0),'start_ticks':ident['start_ticks'],'boot_id':self.boot,'run_id':self.run_id,'started_utc':self.started,'protocol_sha256':sha(root/'protocol.json')});event(self.db,'supervisor_started',{'run_id':self.run_id,'pid':os.getpid(),'protocol_sha256':sha(root/'protocol.json')});self.recover()
 def signal_stop(self,signum,frame):self.stopping=True;self.stop_reason=f'signal_{signum}'
 def accounting(self):
  r=self.db.execute("SELECT COALESCE(SUM(charged_cpu),0)charged,COALESCE(SUM(verified_actual_cpu),0)actual,COALESCE(SUM(uncertain_upper_cpu),0)uncertain FROM attempts WHERE status NOT IN ('launching','running')").fetchone();active=self.db.execute("SELECT COALESCE(SUM(lease_cpu),0)reserved,COALESCE(SUM(observed_cpu),0)observed FROM attempts WHERE status IN ('launching','running')").fetchone();return {'final_charged':float(r['charged']),'verified_actual':float(r['actual']),'uncertain_upper':float(r['uncertain']),'active_reserved':float(active['reserved']),'active_observed':float(active['observed'])}
 def recover(self):
  for rr in self.db.execute("SELECT * FROM attempts WHERE status IN ('launching','running')").fetchall():
   a=dict(rr);identity=read(Path(a['attempt_dir'])/'identity.json',{})
   if a['pid']is None and identity.get('attempt_id')==a['id']:
    a.update(pid=identity.get('pid'),start_ticks=identity.get('start_ticks'),boot_id=identity.get('boot_id'),pgid=identity.get('pid'))
    with self.db:self.db.execute('UPDATE attempts SET pid=?,start_ticks=?,boot_id=?,pgid=? WHERE id=?',(a['pid'],a['start_ticks'],a['boot_id'],a['pgid'],a['id']))
   current=alive_identity(a.get('pid'),a.get('start_ticks'),a.get('boot_id'),self.boot)
   if current and current['state']!='Z':self.running[a['id']]={'attempt':a,'child':False,'popen':None};event(self.db,'orphan_adopted_for_monitoring',{'attempt_id':a['id'],'pid':a['pid'],'reservation_retained':a['lease_cpu']})
   else:self.finish(a,None,None,'recovery_missing_or_zombie',orphan=True)
 def finish(self,a,exit_code,usage,source,orphan=False):
  old=self.db.execute('SELECT status FROM attempts WHERE id=?',(a['id'],)).fetchone()
  if not old or old['status']not in ['launching','running']:return
  observed=float(a.get('observed_cpu',0));verified=float(usage.ru_utime+usage.ru_stime)if usage is not None else 0.;charge=max(verified,observed)if usage is not None else max(float(a['lease_cpu']),observed);uncertain=0. if usage is not None else max(0.,charge-observed)
  sig=-exit_code if exit_code is not None and exit_code<0 else None;out=Path(a['out_dir']);receipt=read(out/'cpu_receipt.json',{});result=read(out/'result.json',{})
  receipt_match=receipt.get('attempt_id')==a['id']or receipt.get('pid')==a.get('pid')
  if orphan and receipt_match:
   r_cpu=float(receipt.get('process_cpu_seconds',receipt.get('cpu_seconds',receipt.get('attempt_cpu_seconds',0)))or 0);verified=max(observed,r_cpu);uncertain=max(0.,charge-verified)
  # Orphan reservations are never refunded based solely on an unreapable process.
  success=exit_code==0 or (orphan and bool(result)and receipt_match and receipt.get('exit_code',receipt.get('status'))in [0,'complete','completed','success'])
  interrupted=(exit_code==75 or sig==signal.SIGXCPU or (sig==signal.SIGKILL and observed>=a['lease_cpu']-3)or self.stopping or orphan)
  if success and not(result.get('complete')is True and result.get('status')=='complete'):
   success=False;interrupted=False;source+=':exit0_without_complete_result'
  if success:astate='succeeded';jstate='completed';failure_add=0
  elif interrupted:astate='interrupted';jstate='queued';failure_add=0
  else:astate='failed';failure_add=1;jstate='queued'
  with self.db:
   j=self.db.execute('SELECT failure_count FROM jobs WHERE id=?',(a['job_id'],)).fetchone();fails=int(j['failure_count'])+failure_add
   if not success and fails>=self.retries:jstate='failed'
   self.db.execute('UPDATE attempts SET status=?,verified_actual_cpu=?,charged_cpu=?,uncertain_upper_cpu=?,final_source=?,ended_utc=?,exit_code=?,exit_signal=? WHERE id=?',(astate,verified,charge,uncertain,source,utc(),exit_code,sig,a['id']))
   self.db.execute('UPDATE jobs SET status=?,failure_count=?,result_path=?,last_error=? WHERE id=?',(jstate,fails,str(out/'result.json')if success else None,None if success else source,a['job_id']))
  event(self.db,'attempt_finalized',{'attempt_id':a['id'],'job_id':a['job_id'],'state':astate,'job_state':jstate,'exit_code':exit_code,'verified_cpu_seconds':verified,'conservative_charge_seconds':charge,'uncertain_upper_seconds':uncertain,'final_source':source});self.running.pop(a['id'],None)
 def launch(self,job,lease):
  aid=uuid.uuid4().hex;adir=self.root/'attempts'/aid;adir.mkdir(parents=True);out=self.root/'jobs'/job['id'];out.mkdir(parents=True,exist_ok=True);jobfile=out/'job.json';payload=json.loads(job['job_json']);atomic(jobfile,payload);hb=out/'heartbeat.json'
  for old in ['cpu_receipt.json','heartbeat.json']:
   p=out/old
   if p.exists():
    try:(adir/('previous_'+old)).write_bytes(p.read_bytes())
    except OSError:pass
  a={'id':aid,'job_id':job['id'],'pid':None,'start_ticks':None,'boot_id':self.boot,'pgid':None,'status':'launching','lease_cpu':float(lease),'observed_cpu':0.,'out_dir':str(out),'attempt_dir':str(adir),'heartbeat_path':str(hb)}
  with self.db:self.db.execute('INSERT INTO attempts(id,job_id,status,lease_cpu,started_utc,out_dir,attempt_dir,heartbeat_path,boot_id)VALUES(?,?,?,?,?,?,?,?,?)',(aid,job['id'],'launching',lease,utc(),str(out),str(adir),str(hb),self.boot));self.db.execute("UPDATE jobs SET status='running',attempt_count=attempt_count+1 WHERE id=?",(job['id'],))
  env=os.environ.copy();env.update({k:'1'for k in THREAD_VARS});env.update(STUDY_ATTEMPT_ID=aid,STUDY_JOB_ID=job['id']);boot=self.boot
  def prepare():
   resource.setrlimit(resource.RLIMIT_CPU,(max(1,int(lease)-2),max(2,int(lease)-1)));p=proc(os.getpid());atomic(adir/'identity.json',{'attempt_id':aid,'job_id':job['id'],'pid':os.getpid(),'start_ticks':p['start_ticks'],'boot_id':boot})
  cmd=[sys.executable,'-u',str(self.root/'study_worker.py'),'--job',str(jobfile),'--out',str(out),'--cpu-limit',str(max(1,int(lease)-2))]
  try:
   with (adir/'stdout.log').open('ab',buffering=0)as log:
    p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,preexec_fn=prepare,close_fds=True)
   ident=read(adir/'identity.json',{});current=proc(p.pid);a.update(pid=p.pid,start_ticks=ident.get('start_ticks',current['start_ticks']if current else None),pgid=p.pid,status='running')
   with self.db:self.db.execute("UPDATE attempts SET pid=?,start_ticks=?,pgid=?,status='running' WHERE id=?",(p.pid,a['start_ticks'],p.pid,aid))
   self.running[aid]={'attempt':a,'child':True,'popen':p};event(self.db,'worker_started',{'attempt_id':aid,'job_id':job['id'],'pid':p.pid,'lease_cpu_seconds':lease,'command':cmd})
  except BaseException as e:self.finish(a,1,None,'spawn_failure:'+repr(e),orphan=False)
 def monitor(self):
  for aid,entry in list(self.running.items()):
   a=entry['attempt'];p=alive_identity(a['pid'],a['start_ticks'],a['boot_id'],self.boot)
   if p:a['observed_cpu']=max(a['observed_cpu'],p['cpu_seconds'])
   h=read(a['heartbeat_path'],{});match=h.get('attempt_id')==aid or h.get('pid')==a['pid'];hbtime=h.get('updated_utc',h.get('time_utc',h.get('utc',h.get('timestamp'))))
   if match:
    cpu=h.get('process_cpu_seconds',h.get('cpu_seconds',h.get('attempt_cpu_seconds')))
    if isinstance(cpu,(float,int)):a['observed_cpu']=max(a['observed_cpu'],float(cpu))
   with self.db:self.db.execute('UPDATE attempts SET observed_cpu=?,last_heartbeat_utc=? WHERE id=?',(a['observed_cpu'],str(hbtime)if match and hbtime else None,aid))
   if entry['child']:
    try:
     pid,status,usage=os.wait4(a['pid'],os.WNOHANG)
     if pid:
      code=os.waitstatus_to_exitcode(status);entry['popen'].returncode=code;self.finish(a,code,usage,'wait4_exactly_once');continue
    except ChildProcessError:entry['child']=False
   elif not p or p['state']=='Z':self.finish(a,None,None,'orphan_exit_conservative_lease',orphan=True)
 def status(self,state='running',reason=None):
  a=self.accounting();jobs={r['status']:r['n']for r in self.db.execute('SELECT status,COUNT(*)n FROM jobs GROUP BY status')};worker=[]
  for aid,e in self.running.items():
   q=e['attempt'];worker.append({'attempt_id':aid,'job_id':q['job_id'],'pid':q['pid'],'pgid':q['pgid'],'start_ticks':q['start_ticks'],'observed_cpu_seconds':q['observed_cpu'],'lease_cpu_seconds':q['lease_cpu'],'reapable_child':e['child'],'heartbeat':read(q['heartbeat_path'],{})})
  control=time.process_time()-self.control_cpu_start
  v={'state':state,'reason':reason,'updated_utc':utc(),'leader_pid':os.getpid(),'leader_start_ticks':proc(os.getpid())['start_ticks'],'boot_id':self.boot,'run_id':self.run_id,'started_utc_this_supervisor':self.started,'elapsed_wall_seconds_this_supervisor':time.monotonic()-self.started_wall,'cpu_budget_seconds':self.budget,'management_cpu_reserve_seconds':self.reserve,'training_cpu_allowance_seconds':self.allowance,'verified_completed_training_cpu_seconds':a['verified_actual'],'active_observed_training_cpu_seconds':a['active_observed'],'observed_training_cpu_seconds':a['verified_actual']+a['active_observed'],'conservative_completed_training_charge_seconds':a['final_charged'],'uncertain_training_cpu_upper_bound_seconds':a['uncertain_upper'],'active_cpu_lease_reserved_seconds':a['active_reserved'],'unallocated_training_cpu_seconds':max(0.,self.allowance-a['final_charged']-a['active_reserved']),'conservative_training_upper_committed_seconds':a['final_charged']+a['active_reserved'],'supervisor_cpu_seconds_this_run':control,'management_subprocess_cpu_seconds_this_run':self.management_children_cpu,'management_reserve_is_not_reported_as_actual_training':True,'max_workers':self.workers,'active_workers':len(worker),'workers':worker,'job_counts':jobs,'accuracy_used_as_stopping_criterion':False,'stop_file':str(self.root/'STOP'),'lifecycle_limit':'Detached local processes cannot guarantee survival of cloud sleep or destruction.'};atomic(self.root/'status.json',v);return v
 def extend(self):
  now=time.monotonic()
  if now-self.last_extension<15:return False
  self.last_extension=now;script=self.root/'extend_manifest.py'
  if not script.exists():event(self.db,'manifest_extension_missing',{});return False
  with (self.root/'management.log').open('ab',buffering=0)as log:
   p=subprocess.Popen([sys.executable,str(script),'--root',str(self.root)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True)
   pid,status,usage=os.wait4(p.pid,0);p.returncode=os.waitstatus_to_exitcode(status);self.management_children_cpu+=usage.ru_utime+usage.ru_stime
  if p.returncode!=0:event(self.db,'manifest_extension_failed',{'exit_code':p.returncode});return False
  added=import_manifest(self.db,self.root);event(self.db,'manifest_extended',{'jobs_added':added});return added>0
 def shutdown(self,reason):
  self.stopping=True;self.status('stopping',reason)
  for e in self.running.values():
   a=e['attempt'];p=alive_identity(a['pid'],a['start_ticks'],a['boot_id'],self.boot)
   if p and p['state']!='Z':
    try:os.killpg(a['pgid'],signal.SIGTERM)
    except ProcessLookupError:pass
  deadline=time.monotonic()+min(5,max(1,float(self.protocol.get('shutdown_grace_seconds',3))))
  while self.running and time.monotonic()<deadline:self.monitor();time.sleep(.1)
  for e in list(self.running.values()):
   a=e['attempt'];p=alive_identity(a['pid'],a['start_ticks'],a['boot_id'],self.boot)
   if p and p['state']!='Z':
    try:os.killpg(a['pgid'],signal.SIGKILL)
    except ProcessLookupError:pass
  deadline=time.monotonic()+5
  while self.running and time.monotonic()<deadline:self.monitor();time.sleep(.05)
  for e in list(self.running.values()):self.finish(e['attempt'],None,None,'shutdown_unreapable_conservative_lease',orphan=True)
  self.status(reason,reason);event(self.db,'supervisor_terminal',{'state':reason});self.db.close()
 def run(self):
  self.status()
  while True:
   self.monitor()
   if self.stopping or (self.root/'STOP').exists():self.shutdown('stopped_by_user');return
   if sha(self.root/'study_worker.py')!=self.worker_hash:raise RuntimeError('Worker script changed after frozen ledger start')
   a=self.accounting();room=self.allowance-a['final_charged']-a['active_reserved']
   if a['final_charged']>=self.allowance or (not self.running and room<3):self.shutdown('budget_exhausted');return
   while len(self.running)<self.workers and room>=3:
    job=self.db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY ordinal LIMIT 1").fetchone()
    if not job:break
    lease=min(self.lease,int(math.floor(room)));self.launch(dict(job),lease);a=self.accounting();room=self.allowance-a['final_charged']-a['active_reserved']
   queued=self.db.execute("SELECT COUNT(*) FROM jobs WHERE status='queued'").fetchone()[0]
   if not queued and not self.running:
    self.status('waiting_manifest','Frozen deterministic continuation required; not a completed study');self.extend()
   else:self.status('running'if self.running else 'waiting_for_cpu_lease')
   time.sleep(self.poll)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
 with (root/'supervisor.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:print('A live supervisor already holds the leader lock',flush=True);return
  supervisor=None
  try:supervisor=Supervisor(root);supervisor.run()
  except BaseException as e:
   import traceback;traceback.print_exc()
   if supervisor:
    try:supervisor.shutdown('fatal_configuration_error')
    except BaseException:pass
   else:atomic(root/'status.json',{'state':'fatal_configuration_error','reason':repr(e),'updated_utc':utc(),'leader_pid':os.getpid()})
   raise
if __name__=='__main__':main()
