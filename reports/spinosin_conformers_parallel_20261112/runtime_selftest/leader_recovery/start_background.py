#!/usr/bin/env python3
"""Independent chemistry watchdog. No app automation or old-study operations."""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
from supervisor import TERMINAL,read,atomic,proc,matches,dbopen,manager,self_identity,config

def live(info):
 if not isinstance(info,dict):return False
 p=matches(info.get('pid'),info.get('start_ticks'),info.get('boot_id'));return bool(p and p['state']!='Z')

def launch_leader(root):
 with(root/'supervisor.log').open('ab',buffering=0)as log:p=subprocess.Popen([sys.executable,'-u',str(root/'supervisor.py'),'--root',str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 print(json.dumps({'spawned_leader_pid':p.pid}),flush=True)

def watchdog(root):
 with(root/'watchdog.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  d=dbopen(root);ident=self_identity();atomic(root/'watchdog.json',{**ident,'watchdog_interval_seconds':config(root)[1].get('watchdog_seconds',15)});signal.signal(signal.SIGHUP,signal.SIG_IGN)
  try:
   while True:
    manager(d,'watchdog',include_children=True);state=read(root/'status.json',{}).get('state');leader=read(root/'leader.json',{})
    if state in TERMINAL:return
    if(root/'STOP').exists()or(root/'PAUSE').exists():
     if live(leader):
      try:os.kill(leader['pid'],signal.SIGTERM)
      except ProcessLookupError:pass
     else:return
    elif not live(leader):
     # A short child launches a detached leader and exits, preserving PPID1.
     subprocess.run([sys.executable,str(root/'start_background.py'),'--root',str(root),'--launch-leader'],stdin=subprocess.DEVNULL,check=False)
    time.sleep(float(config(root)[1].get('watchdog_seconds',15)))
  finally:
   p=proc(os.getpid());manager(d,'watchdog',final=p['self_cpu']+p['child_cpu'],include_children=True);d.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--watchdog',action='store_true');p.add_argument('--launch-leader',action='store_true');p.add_argument('--status',action='store_true');p.add_argument('--stop',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
 if a.status:print(json.dumps({'status':read(root/'status.json',{}),'leader':read(root/'leader.json',{}),'watchdog':read(root/'watchdog.json',{})},ensure_ascii=False,indent=2));return
 if a.stop:
  (root/'STOP').write_text('User requested independent chemistry stop\n');leader=read(root/'leader.json',{})
  if live(leader):os.kill(leader['pid'],signal.SIGTERM)
  print(json.dumps({'stop_requested':True,'root':str(root)}));return
 if a.launch_leader:launch_leader(root);return
 if a.watchdog:watchdog(root);return
 for file in ['supervisor.py','worker.py','jobs.json']:
  if not(root/file).is_file():raise SystemExit('Missing required file: '+file)
 config(root)
 if a.resume:
  # Resume authorizes restarting the same frozen plan, never changing seals/budget.
  db=dbopen(root);old=db.execute("SELECT value FROM meta WHERE key='input_seals'").fetchone();db.close()
  if old:
   import hashlib
   seals=json.loads(old['value']);bad=[f for f,h in seals.items()if not Path(f).is_file()or hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h]
   if bad:raise SystemExit('Frozen inputs changed; resume refused: '+str(bad))
  state=read(root/'status.json',{}).get('state')
  if state in ['budget_exhausted','all_tasks_completed','scientific_blocked']:raise SystemExit('Terminal study requires an explicitly new authorized plan, not a ledger reset.')
  for file in ['STOP','PAUSE']:
   if(root/file).exists():(root/file).unlink()
  s=read(root/'status.json',{});s['state']='resuming_same_frozen_plan';atomic(root/'status.json',s)
 if(root/'STOP').exists()or(root/'PAUSE').exists():raise SystemExit('STOP/PAUSE exists; inspect state and use explicit --resume only after restoring the original sealed files.')
 if read(root/'status.json',{}).get('state')in TERMINAL:raise SystemExit('Study is in terminal state; no computations repeated.')
 with(root/'launcher.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if not live(read(root/'watchdog.json',{})):
   with(root/'watchdog.log').open('ab',buffering=0)as log:q=subprocess.Popen([sys.executable,'-u',str(root/'start_background.py'),'--root',str(root),'--watchdog'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
  deadline=time.monotonic()+5
  while time.monotonic()<deadline:
   if live(read(root/'leader.json',{})):break
   if read(root/'status.json',{}).get('state')in TERMINAL:break
   time.sleep(.1)
  print(json.dumps({'root':str(root),'leader':read(root/'leader.json',{}),'watchdog':read(root/'watchdog.json',{}),'status':read(root/'status.json',{})},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
