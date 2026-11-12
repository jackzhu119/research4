"""Isolated arithmetic-only queue tests. Never starts the real chemistry worker."""
from pathlib import Path
import json,os,signal,sqlite3,subprocess,sys,time,shutil
ROOT=Path(__file__).resolve().parent;BASE=ROOT/'runtime_selftest_final';BASE.mkdir(exist_ok=True)
FAKE='''import argparse,json,os,time,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--job');a=p.parse_args();r=Path(a.root);j=json.loads(Path(a.job).read_text());o=r/'jobs'/j['id'];c=o/'counter.json';n=json.loads(c.read_text())if c.exists()else 0;c.write_text(json.dumps(n+1));t=time.process_time();duration=float(j.get('cpu',.12));x=1
while time.process_time()-t<duration:
 for _ in range(5000):x=(x*13+7)%100003
status=76 if j.get('blocked')else 1 if j.get('fail_once')and n==0 else 0
(o/'result.json').write_text(json.dumps({'complete':status==0,'status':'complete'if status==0 else 'scientific_blocked'if status==76 else 'failed','job_id':j['id']}));(o/'cpu_receipt.json').write_text(json.dumps({'attempt_id':os.environ['CHEM_ATTEMPT_ID'],'pid':os.getpid(),'process_cpu_seconds':time.process_time(),'exit_code':status}));sys.exit(status)
'''
def create(name,jobs,maxworkers=2):
 r=BASE/name;r.mkdir(exist_ok=True)
 for file in ['supervisor.py','start_background.py']:shutil.copy2(ROOT/file,r/file)
 (r/'worker.py').write_text(FAKE);(r/'jobs.json').write_text(json.dumps({'jobs':jobs}));(r/'protocol.json').write_text(json.dumps({'cpu_budget_seconds':600,'management_cpu_reserve_seconds':150,'shutdown_cpu_margin_seconds':.5,'max_workers':maxworkers,'poll_seconds':.08,'watchdog_seconds':.2,'retry_initial_seconds':.05,'retry_max_seconds':.1,'shutdown_grace_seconds':1}));return r
def launch(r):
 log=(r/'leader-test.log').open('ab');return subprocess.Popen([sys.executable,str(r/'supervisor.py'),'--root',str(r)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
def waitstate(r,states,timeout=35):
 end=time.monotonic()+timeout
 while time.monotonic()<end:
  try:s=json.loads((r/'status.json').read_text())
  except(Exception):time.sleep(.05);continue
  if s['state']in states:return s
  time.sleep(.05)
 raise RuntimeError('Timed out '+str(r)+' '+(r/'leader-test.log').read_text()[-2000:])
def summary(r):
 d=sqlite3.connect(r/'queue.sqlite3');a=d.execute('SELECT COUNT(*),SUM(verified_cpu),SUM(charged_cpu),SUM(uncertain_cpu)FROM attempts').fetchone();j=d.execute('SELECT id,status,failure_count FROM jobs ORDER BY ordinal').fetchall();m=d.execute('SELECT role,state,charge_cpu FROM managers').fetchall();d.close();return {'attempts':a,'jobs':j,'managers':m}
results={}
r=create('normal',[{'id':'a','ordinal':0},{'id':'b','ordinal':1}]);p=launch(r);s=waitstate(r,{'all_tasks_completed'});p.wait(timeout=5);q=summary(r);assert q['attempts'][0]==2 and q['attempts'][3]==0 and abs(q['attempts'][1]-q['attempts'][2])<1e-12;results['normal']={'state':s['state'],**q}
r=create('retry_blocked',[{'id':'retry','ordinal':0,'fail_once':True},{'id':'blocked','ordinal':1,'blocked':True}]);p=launch(r);s=waitstate(r,{'scientific_blocked'});p.wait(timeout=5);q=summary(r);assert q['attempts'][0]==3 and q['jobs'][0][1]=='completed'and q['jobs'][1][1]=='scientific_blocked';results['retry_blocked']={'state':s['state'],**q}
r=create('leader_recovery',[{'id':'a','ordinal':0,'cpu':1.5},{'id':'b','ordinal':1,'cpu':1.5}]);p=launch(r);waitstate(r,{'running'});end=time.monotonic()+8
while time.monotonic()<end:
 d=sqlite3.connect(r/'queue.sqlite3');n=d.execute('SELECT COUNT(*)FROM attempts WHERE worker_pid IS NOT NULL').fetchone()[0];d.close()
 if n==2:break
 time.sleep(.03)
assert n==2;p.kill();p.wait();p2=launch(r);s=waitstate(r,{'all_tasks_completed'});p2.wait(timeout=5);q=summary(r);assert q['attempts'][0]==2 and q['attempts'][3]==0 and abs(q['attempts'][1]-q['attempts'][2])<1e-12;results['leader_recovery']={'state':s['state'],**q}
r=create('integrity',[{'id':'a','ordinal':0,'cpu':1.5}],1);p=launch(r);waitstate(r,{'running'});(r/'PAUSE').write_text('{}');s=waitstate(r,{'paused_integrity'});p.wait(timeout=5);results['integrity']={'state':s['state'],**summary(r)}
r=create('watchdog_recovery',[{'id':'a','ordinal':0,'cpu':2.},{'id':'b','ordinal':1,'cpu':2.}]);subprocess.run([sys.executable,str(r/'start_background.py'),'--root',str(r)],stdout=subprocess.DEVNULL,check=True);waitstate(r,{'running'});wd=json.loads((r/'watchdog.json').read_text());before=json.loads((r/'leader.json').read_text());os.kill(wd['pid'],signal.SIGKILL);subprocess.run([sys.executable,str(r/'start_background.py'),'--root',str(r)],stdout=subprocess.DEVNULL,check=True);s=waitstate(r,{'all_tasks_completed'});after=json.loads((r/'leader.json').read_text());assert before['pid']==after['pid']and before['start_ticks']==after['start_ticks'];q=summary(r);assert q['attempts'][0]==2 and q['attempts'][3]==0;results['watchdog_recovery']={'state':s['state'],'leader_identity_unchanged':True,**q}
(ROOT/'runtime_selftest_verification.json').write_text(json.dumps({'arithmetic_fake_worker_only':True,'real_DFT_started':False,'old_ML_modified':False,'tests':results},indent=2));print(json.dumps(results,indent=2))
