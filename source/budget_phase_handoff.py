#!/usr/bin/env python3
"""One local phase handoff; no app automation, user messages, or accuracy decisions."""
import argparse,fcntl,json,os,signal,sqlite3,subprocess,sys,time,uuid,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
def read(p,default=None):
 try:return json.loads(Path(p).read_text())
 except (OSError,ValueError):return default
def atomic(p,v):
 p=Path(p);t=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex)
 with t.open('w')as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def proc(pid):
 try:
  s=(Path('/proc')/str(int(pid))/'stat').read_text();a=s[s.rfind(')')+2:].split();return {'pid':int(pid),'state':a[0],'start_ticks':int(a[19])}
 except (OSError,ValueError,TypeError,IndexError):return None
def alive(q,boot):
 if not isinstance(q,dict):return False
 p=proc(q.get('pid',q.get('guard_pid')));return bool(p and p['state']!='Z'and p['start_ticks']==q.get('start_ticks',q.get('guard_start_ticks'))and q.get('boot_id')==boot)
def status(root,phase,pending,**more):
 p=proc(os.getpid());atomic(root/'budget_handoff_status.json',{'phase':phase,'checked_at_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'start_ticks':p['start_ticks'],'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'authorization_id':pending.get('authorization_id'),'target_cpu_seconds':pending.get('cumulative_authorized_cpu_seconds'),'accuracy_used':False,**more})
def run(root):
 with (root/'budget_handoff.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return 0
  signal.signal(signal.SIGHUP,signal.SIG_IGN);pending=read(root/'pending_budget_handoff.json',{});assert pending.get('state')in ['waiting_natural_boundary','activating'],'No pending authorized handoff'
  boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
  while True:
   s=read(root/'status.json',{});state=s.get('state')
   if(root/'STOP').exists()or state=='stopped_by_user':
    pending['state']='cancelled';atomic(root/'pending_budget_handoff.json',pending);status(root,'cancelled_stop_requested',pending);return 0
   if state=='fatal_configuration_error':status(root,'blocked_configuration_error',pending);return 1
   if state=='running'and s.get('run_id')!=pending['prior_run_id']and s.get('cpu_budget_seconds')==pending['cumulative_authorized_cpu_seconds']:
    pending.update(state='completed',new_run_id=s['run_id'],completed_at_utc=datetime.now(timezone.utc).isoformat(),activation_path='existing_recovery_loaded_authorized_budget');atomic(root/'pending_budget_handoff.json',pending);status(root,'completed',pending,new_leader_pid=s['leader_pid']);return 0
   if state=='budget_exhausted'and s.get('run_id')==pending['prior_run_id']:
    controllers=[read(root/'leader.json',{}),read(root/'watchdog.json',{}),read(root/'budget_guard_status.json',{})]
    if any(alive(q,boot)for q in controllers):status(root,'waiting_controller_cleanup',pending);time.sleep(2);continue
    db=sqlite3.connect(f'file:{root}/queue.sqlite3?mode=ro',uri=True);active=db.execute("SELECT COUNT(*)FROM attempts WHERE status IN('launching','running')").fetchone()[0];meta=db.execute("SELECT value FROM meta WHERE key='protocol_sha256'").fetchone()[0];db.close()
    if active:status(root,'blocked_unreaped_attempts',pending,active_attempts=active);time.sleep(2);continue
    assert meta==pending['protocol_sha256']and(root/'protocol.sha256').read_text().strip()==meta,'Protocol/ledger mismatch'
    stage=Path(pending['archive_path'])/'natural_boundary';stage.mkdir(exist_ok=True)
    for name in ['status.json','leader.json','watchdog.json','budget_guard_status.json']:
     if(root/name).exists():shutil.copy2(root/name,stage/name)
    pending['state']='activating';atomic(root/'pending_budget_handoff.json',pending);s.update(state='resuming_with_additional_budget',reason='authorized_natural_budget_phase_handoff',cpu_budget_seconds=pending['cumulative_authorized_cpu_seconds'],active_workers=0,updated_utc=datetime.now(timezone.utc).isoformat());atomic(root/'status.json',s);status(root,'starting_authorized_phase',pending)
    with(root/'budget_handoff.log').open('ab',buffering=0)as log:
     subprocess.run([sys.executable,str(root/'budget_guard.py'),'--root',str(root),'--start'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
     subprocess.run([sys.executable,str(root/'start_background.py'),'--root',str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
     current=read(root/'status.json',{})
     if current.get('state')=='running'and current.get('run_id')!=pending['prior_run_id']and current.get('cpu_budget_seconds')==pending['cumulative_authorized_cpu_seconds']:
      pending.update(state='completed',new_run_id=current['run_id'],completed_at_utc=datetime.now(timezone.utc).isoformat());atomic(root/'pending_budget_handoff.json',pending);status(root,'completed',pending,new_leader_pid=current['leader_pid']);return 0
     time.sleep(.5)
    status(root,'blocked_startup_not_confirmed',pending);return 1
   status(root,'waiting_natural_boundary',pending,current_runtime_cpu_budget=s.get('cpu_budget_seconds'),current_training_cpu=s.get('observed_training_cpu_seconds'));time.sleep(10)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--start',action='store_true');a=p.parse_args();root=a.root.resolve()
 if a.start:
  with(root/'budget_handoff.log').open('ab',buffering=0)as log:q=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
  print(json.dumps({'local_budget_handoff_pid':q.pid,'app_automation_enabled':False}));return 0
 return run(root)
if __name__=='__main__':raise SystemExit(main())
