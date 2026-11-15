#!/usr/bin/env python3
"""Launch detached study watchdog/supervisor; --root permits isolated smoke tests."""
import argparse,fcntl,json,os,signal,subprocess,sys,time,uuid
from pathlib import Path
TERMINAL={'budget_exhausted','stopped_by_user','fatal_configuration_error'}
def read(p,default=None):
 try:return json.loads(Path(p).read_text())
 except (OSError,ValueError):return default

def proc(pid):
 try:
  s=Path(f'/proc/{int(pid)}/stat').read_text();a=s[s.rfind(')')+2:].split();return {'pid':int(pid),'state':a[0],'start_ticks':int(a[19]),'ppid':int(a[1]),'sid':int(a[3])}
 except (OSError,ValueError,IndexError,TypeError):return None

def live(info):
 if not isinstance(info,dict):return False
 p=proc(info.get('pid',info.get('leader_pid')))
 return bool(p and p['state']!='Z'and p['start_ticks']==info.get('start_ticks',info.get('leader_start_ticks'))and info.get('boot_id')==Path('/proc/sys/kernel/random/boot_id').read_text().strip())

def atomic(p,v):
 t=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex)
 with t.open('w')as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)

def launch_supervisor(root):
 # This helper immediately exits; the session leader is then adopted by PID1.
 with (root/'supervisor.log').open('ab',buffering=0)as log:
  p=subprocess.Popen([sys.executable,'-u',str(root/'supervisor.py'),'--root',str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 print(json.dumps({'spawned_supervisor_pid':p.pid}),flush=True)

def watchdog(root):
 with (root/'watchdog.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  p=proc(os.getpid());atomic(root/'watchdog.json',{'pid':os.getpid(),'start_ticks':p['start_ticks'],'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'sid':os.getsid(0),'interval_seconds':15});signal.signal(signal.SIGHUP,signal.SIG_IGN);summary=None;last_summary=time.monotonic()
  while True:
   state=read(root/'status.json',{}).get('state')
   if (root/'STOP').exists()or state in TERMINAL:
    # Let an orderly STOP finish its accounting before the final report.
    deadline=time.monotonic()+15
    while (root/'STOP').exists()and live(read(root/'leader.json',{}))and read(root/'status.json',{}).get('state')not in TERMINAL and time.monotonic()<deadline:time.sleep(.5)
    if summary is not None and summary.poll()is None:
     try:summary.wait(timeout=120)
     except subprocess.TimeoutExpired:pass
    if (root/'summarize_results.py').exists():
     with (root/'summary.log').open('ab',buffering=0)as log:
      try:subprocess.run([sys.executable,str(root/'summarize_results.py'),'--root',str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=120,check=False)
      except subprocess.TimeoutExpired:pass
    return
   leader=read(root/'leader.json',{})
   if not live(leader):
    subprocess.run([sys.executable,str(root/'start_background.py'),'--root',str(root),'--launch-supervisor'],stdin=subprocess.DEVNULL,check=False)
   # Reporting is independent of training and never reads a score to stop the study.
   if time.monotonic()-last_summary>=600 and (root/'summarize_results.py').exists()and (summary is None or summary.poll()is not None):
    with (root/'summary.log').open('ab',buffering=0)as log:summary=subprocess.Popen([sys.executable,str(root/'summarize_results.py'),'--root',str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    last_summary=time.monotonic()
   time.sleep(15)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--watchdog',action='store_true');p.add_argument('--launch-supervisor',action='store_true');p.add_argument('--status',action='store_true');p.add_argument('--stop',action='store_true');a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
 if a.status:print(json.dumps({'status':read(root/'status.json',{}),'leader':read(root/'leader.json',{}),'watchdog':read(root/'watchdog.json',{})},ensure_ascii=False,indent=2));return
 if a.stop:
  (root/'STOP').write_text('User requested stop\n');leader=read(root/'leader.json',{})
  if live(leader):os.kill(leader['pid'],signal.SIGTERM)
  print(json.dumps({'stop_requested':True,'stop_file':str(root/'STOP')}));return
 if a.launch_supervisor:launch_supervisor(root);return
 if a.watchdog:watchdog(root);return
 if (root/'STOP').exists():raise SystemExit('STOP file exists; remove it deliberately before restarting this study.')
 for name in ['protocol.json','job_manifest.json','study_worker.py','supervisor.py']:
  if not(root/name).exists():raise SystemExit('Missing required study file: '+name)
 if read(root/'status.json',{}).get('state')in TERMINAL:raise SystemExit('Study has a terminal state; inspect status before any deliberate new study.')
 with (root/'launcher.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  info=read(root/'watchdog.json',{})
  if not live(info):
   with (root/'watchdog.log').open('ab',buffering=0)as log:p=subprocess.Popen([sys.executable,'-u',str(root/'start_background.py'),'--root',str(root),'--watchdog'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
  deadline=time.monotonic()+5
  while time.monotonic()<deadline:
   if live(read(root/'leader.json',{})):break
   time.sleep(.1)
  print(json.dumps({'root':str(root),'leader':read(root/'leader.json',{}),'watchdog':read(root/'watchdog.json',{}),'status':read(root/'status.json',{}),'stop_command':[sys.executable,str(root/'start_background.py'),'--root',str(root),'--stop']},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
