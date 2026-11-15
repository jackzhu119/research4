#!/usr/bin/env python3
"""Local CPU limit enforcement only. No app automation, model selection, or messages."""
import argparse,fcntl,json,os,signal,sqlite3,subprocess,sys,time,uuid
from pathlib import Path
from datetime import datetime,timezone

TERMINAL={'budget_exhausted','stopped_by_user','fatal_configuration_error'}
def read(p,default=None):
 try:return json.loads(Path(p).read_text())
 except (OSError,ValueError):return default
def atomic(p,value):
 p=Path(p);tmp=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex)
 with tmp.open('w')as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def proc(pid):
 try:
  raw=Path('/proc')/str(int(pid))/'stat';t=raw.read_text();a=t[t.rfind(')')+2:].split();tick=os.sysconf('SC_CLK_TCK')
  return {'pid':int(pid),'state':a[0],'ppid':int(a[1]),'pgid':int(a[2]),'start_ticks':int(a[19]),'self_cpu':(int(a[11])+int(a[12]))/tick,'child_cpu':(int(a[13])+int(a[14]))/tick}
 except (OSError,ValueError,TypeError,IndexError):return None
def matches(p,q,boot):return bool(p and p['start_ticks']==q.get('start_ticks')and q.get('boot_id')==boot)
def sample(root,ledger):
 boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();status=read(root/'status.json',{});leaders={role:read(root/(role+'.json'),{})for role in ['leader','watchdog']};accounts=ledger.setdefault('management_accounts',{})
 leader_pid=leaders['leader'].get('pid');present=set()
 for role,q in leaders.items():
  p=proc(q.get('pid'))
  if not matches(p,q,boot):continue
  key=f"{role}:{boot}:{p['pid']}:{p['start_ticks']}";present.add(key);used=p['self_cpu']
  if role=='watchdog':
   # Its children are report processes and the short launch helper. The detached
   # supervisor is PPID1 and must never be counted here as well as in the ledger.
   used+=p['child_cpu']
   for child in Path('/proc').iterdir():
    if not child.name.isdigit():continue
    c=proc(child.name)
    if c and c['ppid']==p['pid']and c['pid']!=leader_pid:used+=c['self_cpu']+c['child_cpu']
  a=accounts.setdefault(key,{'role':role,'cpu_high_water':0.,'uncertain_tail_charge':0.})
  a['cpu_high_water']=max(a['cpu_high_water'],used)
 # Account for this guard across restarts without counting its live snapshots twice.
 own=proc(os.getpid());key=f"guard:{boot}:{own['pid']}:{own['start_ticks']}";present.add(key)
 a=accounts.setdefault(key,{'role':'guard','cpu_high_water':0.,'uncertain_tail_charge':0.})
 a['cpu_high_water']=max(a['cpu_high_water'],own['self_cpu'])
 for key,a in accounts.items():
  if key not in present:a['uncertain_tail_charge']=max(a.get('uncertain_tail_charge',0.),100.)
 db=sqlite3.connect(f'file:{root}/queue.sqlite3?mode=ro',uri=True,timeout=10);db.row_factory=sqlite3.Row;db.execute('BEGIN')
 final=float(db.execute("select coalesce(sum(charged_cpu),0)from attempts where status not in('launching','running')").fetchone()[0]);active=[];live=0.
 for rr in db.execute("select * from attempts where status in('launching','running')"):
  q=dict(rr);p=proc(q['pid']);seen=max(float(q['observed_cpu']),p['self_cpu'])if matches(p,q,boot)else float(q['lease_cpu']);live+=seen;active.append(q)
 db.rollback();db.close()
 management=sum(a['cpu_high_water']+a.get('uncertain_tail_charge',0.)for a in accounts.values())
 # Lossless completed-artifact maintenance is a standalone process, rather than
 # a watchdog child, and therefore needs its own cumulative CPU accounting.
 maintenance_charge=0.
 maintenance=read(root/'storage_maintenance_ledger.json',{})
 for record in maintenance.get('accounts',{}).values():
  used=float(record.get('cpu_seconds',0.));process=proc(record.get('pid'))
  if matches(process,record,boot):used=max(used,process['self_cpu'])
  elif not record.get('closed',False):used+=100.
  maintenance_charge+=used
 management+=maintenance_charge
 total=final+live+management+150.
 protocol=read(root/'protocol.json',{});authorized=float(protocol.get('authorized_cpu_budget_seconds',protocol.get('cpu_budget_seconds',90000)));assert authorized>=200
 reserve=2000. if authorized>=1000000 else 200.
 snap={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'guard_pid':os.getpid(),'guard_start_ticks':own['start_ticks'],'boot_id':boot,'state':status.get('state'),'training_final_conservative_cpu':final,'training_live_conservative_cpu':live,'management_cpu_charge':management,'storage_maintenance_cpu_charge':maintenance_charge,'preflight_conservative_cpu_charge':150.,'total_conservative_cpu_charge':total,'authorized_cpu_seconds':authorized,'trigger_cpu_seconds':authorized-reserve,'shutdown_cpu_reserve_seconds':reserve,'accuracy_used':False}
 return snap,active,leaders,boot
def run(root,once=False):
 with (root/'budget_guard.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return 0
  path=root/'budget_guard_ledger.json';ledger=read(path,{'version':1,'management_accounts':{}});above=0;maintenance_child=None;last_maintenance_launch=0.
  while True:
   try:
    snap,active,leaders,boot=sample(root,ledger);ledger['snapshot']=snap;atomic(path,ledger);atomic(root/'budget_guard_status.json',snap)
    if snap['state']in TERMINAL or(root/'STOP').exists():return 0
    config=read(root/'storage_maintenance_config.json',{})
    if maintenance_child is not None and maintenance_child.poll() is not None:maintenance_child=None
    if config.get('enabled') and time.monotonic()-last_maintenance_launch>=60:
     storage=read(root/'storage_maintenance_ledger.json',{});storage_alive=maintenance_child is not None
     for record in storage.get('accounts',{}).values():
      process=proc(record.get('pid'))
      if matches(process,record,boot) and process['state']!='Z':storage_alive=True;break
     if not storage_alive:
      with(root/'storage_maintenance.log').open('ab',buffering=0)as log:
       maintenance_child=subprocess.Popen([sys.executable,'-u',str(root/'storage_maintenance.py'),'--root',str(root),'--start'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
      last_maintenance_launch=time.monotonic()
    above=above+1 if snap['total_conservative_cpu_charge']>=snap['trigger_cpu_seconds'] else 0
    if above>=3:
     marker=dict(snap,reason='total_authorized_CPU_safety_limit',scientific_protocol_changed=False,resource_stop_not_accuracy_stop=True)
     atomic(root/'CPU_BUDGET_STOP.json',marker)
     with (root/'STOP').open('x')as f:f.write('Total CPU safety limit; see CPU_BUDGET_STOP.json. Not an accuracy stop.\n');f.flush();os.fsync(f.fileno())
     leader=leaders['leader'];p=proc(leader.get('pid'))
     if matches(p,leader,boot):
      try:os.kill(p['pid'],signal.SIGTERM)
      except ProcessLookupError:pass
     for q in active:
      p=proc(q['pid'])
      if matches(p,q,boot)and p['pgid']==q['pid']:
       try:os.killpg(p['pgid'],signal.SIGTERM)
       except ProcessLookupError:pass
     return 0
    if once:return 0
   except FileExistsError:return 0
   except Exception as e:
    atomic(root/'budget_guard_error.json',{'time_utc':datetime.now(timezone.utc).isoformat(),'error':repr(e),'no_ledger_reset':True})
    if once:raise
   time.sleep(2)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--start',action='store_true');p.add_argument('--once',action='store_true');a=p.parse_args();root=a.root.resolve()
 if a.start:
  with (root/'budget_guard.log').open('ab',buffering=0)as log:q=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
  print(json.dumps({'resource_guard_pid':q.pid,'app_automation_enabled':False}));return 0
 return run(root,a.once)
if __name__=='__main__':raise SystemExit(main())
