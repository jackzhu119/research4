"""Arithmetic fake-only budget/recovery tests; no real chemistry ledger or processes touched."""
from pathlib import Path
import json,os,sqlite3,subprocess,sys,time,hashlib,shutil,importlib.util
from datetime import datetime,timezone,timedelta
R=Path(__file__).resolve().parent;T=R/'runtime_selftest_resource_extension'/'run2';T.mkdir(parents=True,exist_ok=True)
FAKE='''import argparse,json,os,time,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--job');a=p.parse_args();r=Path(a.root);j=json.loads(Path(a.job).read_text());out=r/'jobs'/j['id'];(out/'called_resource_adapter').write_text(os.environ['CHEM_ATTEMPT_ID']);start=time.process_time();x=1
while time.process_time()-start<float(j.get('fake_cpu',.08)):
 for i in range(3000):x=(x*7+3)%100003
(out/'result.json').write_text(json.dumps({'status':'complete','complete':True,'job_id':j['id']}));(out/'done.json').write_text('{}')
'''
def hs(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def setup(name,long=False):
 root=T/name;root.mkdir()
 for file in ['supervisor.py','resource_extension.py']:shutil.copy2(R/file,root/file)
 (root/'worker.py').write_text("raise RuntimeError('Original worker sentinel: resource controller must never execute this')\n");(root/'resource_worker.py').write_text(FAKE);(root/'science_input.txt').write_text('original frozen scientific content\n');(root/'jobs.json').write_text(json.dumps({'jobs':[{'id':'a','ordinal':0,'fake_cpu':1. if long else .08},{'id':'b','ordinal':1,'fake_cpu':.08}]}));(root/'chemistry_protocol.json').write_text(json.dumps({'cpu_budget_seconds':900000,'manager_cpu_reserve_seconds':5000,'max_workers':2,'poll_seconds':.05,'watchdog_seconds':.2,'retry_initial_seconds':.1,'retry_max_seconds':.2,'shutdown_grace_seconds':1,'sealed_files':{'science_input.txt':hs(root/'science_input.txt')}}))
 spec=importlib.util.spec_from_file_location('fakebase_'+name,root/'supervisor.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base);d=base.dbopen(root);seals={str(root/f):hs(root/f)for f in ['worker.py','science_input.txt','chemistry_protocol.json','jobs.json']};base.freeze(d,'budget','900000.0');base.freeze(d,'input_seals',json.dumps(seals,sort_keys=True));d.close();backup=root/'backup.sqlite3';source=sqlite3.connect(root/'queue.sqlite3');target=sqlite3.connect(backup);source.backup(target);target.close();source.close();return root,seals

def call(root,*args,ok=True):
 p=subprocess.run([sys.executable,str(root/'resource_extension.py'),'--root',str(root),*map(str,args)],capture_output=True,text=True)
 if ok and p.returncode:raise RuntimeError(p.stdout+'\n'+p.stderr)
 return p

def grant(root,aid='grant1',extra=100000,text='FAKE arithmetic-only authorization'):
 return call(root,'--grant',extra,'--authorization-id',aid,'--user-text',text,'--backup',root/'backup.sqlite3','--max-workers',1)

def wait(root):
 deadline=time.monotonic()+30
 while time.monotonic()<deadline:
  try:s=json.loads((root/'status.json').read_text())
  except(Exception):time.sleep(.05);continue
  if s['state']=='all_tasks_completed':return s
  if s['state']=='paused_integrity':raise RuntimeError(json.dumps(s))
  time.sleep(.05)
 raise RuntimeError('Fake controller did not complete: '+str(root))

def launch(root):
 log=(root/'test-controller.log').open('ab');return subprocess.Popen([sys.executable,str(root/'resource_extension.py'),'--root',str(root),'--leader'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
results={};root,seals=setup('normal',True);grant(root);grant(root);bad=call(root,'--grant',999,'--authorization-id','grant1','--user-text','different','--backup',root/'backup.sqlite3',ok=False);assert bad.returncode
p=launch(root);deadline=time.monotonic()+8
while time.monotonic()<deadline:
 try:s=json.loads((root/'status.json').read_text())
 except(Exception):time.sleep(.03);continue
 if s.get('active_workers')==1:break
 time.sleep(.03)
assert s['max_workers']==1 and s['cpu_budget_seconds']==1000000;grant(root,'grant2',100,'FAKE live hot resource extension');s=wait(root);p.wait(timeout=5);assert s['cpu_budget_seconds']==1000100 and all((root/'jobs'/j/'called_resource_adapter').exists()for j in ['a','b'])
d=sqlite3.connect(root/'queue.sqlite3');assert d.execute("SELECT value FROM meta WHERE key='budget'").fetchone()[0]=='900000.0';assert json.loads(d.execute("SELECT value FROM meta WHERE key='input_seals'").fetchone()[0])==seals;assert d.execute('SELECT COUNT(*)FROM budget_authorizations').fetchone()[0]==2;assert d.execute('SELECT COUNT(*)FROM resource_attempt_bindings').fetchone()[0]==2;assert d.execute('SELECT SUM(uncertain_cpu)FROM attempts').fetchone()[0]==0
for sql in ['UPDATE budget_authorizations SET entry_hash="changed"','DELETE FROM budget_authorizations']:
 try:d.execute(sql);raise AssertionError('Append-only protection missing')
 except sqlite3.IntegrityError:d.rollback()
# A simultaneous physical-file/meta-seal edit must still fail the authorization's original-seal digest.
(root/'science_input.txt').write_text('changed');changed=dict(seals);changed[str(root/'science_input.txt')]=hs(root/'science_input.txt');d.execute('UPDATE meta SET value=?WHERE key="input_seals"',(json.dumps(changed,sort_keys=True),));d.commit();assert call(root,'--inspect',ok=False).returncode;d.execute('UPDATE meta SET value=?WHERE key="input_seals"',(json.dumps(seals,sort_keys=True),));d.commit();(root/'science_input.txt').write_text('original frozen scientific content\n');d.close();results['authorization_and_live_override']={'original_budget':900000,'first_effective_budget':1000000,'second_fake_hot_budget':1000100,'original_seals_unchanged':True,'resource_worker_adapter_used':True,'max_active_workers':1,'append_only_protected':True,'same_id_different_grant_rejected':True,'simultaneous_original_seal_tamper_rejected':True,'CPU_uncertainty':0}
root,seals=setup('boot_recovery');grant(root);d=sqlite3.connect(root/'queue.sqlite3');stamp=(datetime.now(timezone.utc)-timedelta(seconds=20)).isoformat();original_jobs=json.loads((root/'jobs.json').read_text())['jobs']
for i,j in enumerate(original_jobs):
 d.execute('INSERT INTO jobs(id,ordinal,job_json,status)VALUES(?,?,?,"running")',(j['id'],i,json.dumps(j,sort_keys=True,separators=(',',':'))));aid='old_'+j['id'];out=root/'attempts'/aid;out.mkdir(parents=True);obs=2.+i;pid=900001+i;ws=123+i;d.execute('INSERT INTO attempts(id,job_id,state,allocation_cpu,worker_pid,worker_start,boot_id,observed_cpu,started_utc,attempt_dir)VALUES(?,?,"running",100000,?,?,"dead-fake-boot",?,?,?)',(aid,j['id'],pid,ws,obs,stamp,str(out)));(out/'runner_heartbeat.json').write_text(json.dumps({'updated_utc':stamp,'worker_identity':{'pid':pid,'start_ticks':ws,'boot_id':'dead-fake-boot'},'worker_observed_cpu':obs}))
d.execute('INSERT INTO attempts(id,job_id,state,allocation_cpu,verified_cpu,charged_cpu,started_utc,attempt_dir)VALUES("closed_history","a","closed",10,10,10,?,?)',(stamp,str(root/'attempts/closed_history')));d.commit();closed_before=d.execute('SELECT *FROM attempts WHERE id="closed_history"').fetchone();d.close();cutoff=datetime.now(timezone.utc).isoformat();call(root,'--recover-boot','--cutoff-utc',cutoff,'--clock-margin',300,'--old-host-cpu-cap',4,'--backup',root/'backup.sqlite3','--recovery-id','fake-recovery');d=sqlite3.connect(root/'queue.sqlite3');rows=d.execute('SELECT observed_cpu,verified_cpu,charged_cpu,uncertain_cpu,allocation_cpu FROM attempts WHERE id LIKE"old_%"').fetchall();assert len(rows)==2 and all(v==o and o<c<a and abs(u-(c-v))<1e-9 for o,v,c,u,a in rows);assert d.execute('SELECT *FROM attempts WHERE id="closed_history"').fetchone()==closed_before;assert d.execute("SELECT value FROM meta WHERE key='budget'").fetchone()[0]=='900000.0';d.close();p=launch(root);s=wait(root);p.wait(timeout=5);results['boot_recovery']={'two_old_attempts_audited':True,'observed_and_unknown_separate':True,'whole_long_allocations_not_charged':True,'closed_history_unchanged':True,'original_budget_unchanged':True,'row_bounds':rows,'resumed_scientific_points_once_via_resource_worker':True,'terminal':s['state']}
(R/'resource_extension_selftest_verification.json').write_text(json.dumps({'arithmetic_fake_only':True,'real_DFT_started':False,'real_chemistry_ledger_modified':False,'tests':results},indent=2));print(json.dumps(results,indent=2))
