#!/usr/bin/env python3
"""Append authorized CPU credit while current workers continue to their phase boundary."""
import argparse,hashlib,json,os,shutil,sqlite3,uuid
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,v):
 p=Path(p);t=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex)
 with t.open('w')as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def tablehash(db,name):
 h=hashlib.sha256()
 for row in db.execute(f'SELECT * FROM {name} ORDER BY id'):h.update(json.dumps(tuple(row),ensure_ascii=False,separators=(',',':')).encode());h.update(b'\n')
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--additional',type=int,required=True);p.add_argument('--user-authorization',required=True);a=p.parse_args();assert a.additional>0
 state=json.loads((ROOT/'status.json').read_text());protocol=json.loads((ROOT/'protocol.json').read_text());oldsha=sha(ROOT/'protocol.json');leader=json.loads((ROOT/'leader.json').read_text())
 assert state['state']=='running'and not(ROOT/'STOP').exists(),'Use stopped-budget authorization when stopped'
 prior_pending=ROOT/'pending_budget_handoff.json'
 if prior_pending.exists():assert json.loads(prior_pending.read_text()).get('state')in {'completed','cancelled'},'A budget handoff is already pending'
 assert oldsha==(ROOT/'protocol.sha256').read_text().strip()
 for f,h in protocol['sealed_files'].items():assert sha(ROOT/f)==h,f'Frozen file changed: {f}'
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');archive=ROOT/'authorizations'/f'{stamp}_additional_{a.additional}_live';archive.mkdir(parents=True)
 for file in ['protocol.json','protocol.sha256','status.json','leader.json','watchdog.json','budget_guard_status.json','budget_guard_ledger.json','pending_budget_handoff.json','budget_handoff_status.json','reports/progress.json','reports/summary_manifest.json','reports/中文简报.md']:
  src=ROOT/file
  if src.exists():dst=archive/file;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 db=sqlite3.connect(ROOT/'queue.sqlite3',timeout=30);db.execute('PRAGMA synchronous=FULL');backup=sqlite3.connect(archive/'queue.sqlite3');db.backup(backup);backup_hashes={t:tablehash(backup,t)for t in ['jobs','attempts']};backup.close()
 db.execute('BEGIN IMMEDIATE')
 try:
  assert db.execute("SELECT value FROM meta WHERE key='protocol_sha256'").fetchone()[0]==oldsha
  before={t:tablehash(db,t)for t in ['jobs','attempts']};oldbudget=int(protocol['authorized_cpu_budget_seconds']);newbudget=oldbudget+a.additional
  auth={'authorization_id':archive.name,'authorized_at_utc':datetime.now(timezone.utc).isoformat(),'user_authorization':a.user_authorization,'previous_authorized_cpu_seconds':oldbudget,'additional_authorized_cpu_seconds':a.additional,'cumulative_authorized_cpu_seconds':newbudget,'prior_protocol_sha256':oldsha,'prior_table_sha256':before,'archive_table_sha256':backup_hashes,'archive_is_online_snapshot':True,'archive_path':str(archive),'scientific_configuration_changed':False,'CPU_ledger_reset':False,'current_workers_interrupted':False,'activation':'dynamic_guard_now; next_supervisor_at_natural_budget_boundary','previous_leader':leader,'current_phase_cpu_budget_seconds':state['cpu_budget_seconds']}
  atomic(archive/'authorization_prepared.json',auth);protocol['authorized_cpu_budget_seconds']=newbudget;protocol['cpu_budget_seconds']=newbudget;protocol.setdefault('budget_authorization_history',[]).append(auth);atomic(ROOT/'protocol.json',protocol);newsha=sha(ROOT/'protocol.json');(ROOT/'protocol.sha256').write_text(newsha+'\n')
  assert db.execute("UPDATE meta SET value=? WHERE key='protocol_sha256'AND value=?",(newsha,oldsha)).rowcount==1
  db.execute('INSERT INTO events(utc,event,detail)VALUES(?,?,?)',(datetime.now(timezone.utc).isoformat(),'additional_CPU_budget_authorized_live',json.dumps(dict(auth,new_protocol_sha256=newsha),ensure_ascii=False)))
  assert before=={t:tablehash(db,t)for t in ['jobs','attempts']};db.commit()
 except BaseException:db.rollback();raise
 finally:db.close()
 auth.update(new_protocol_sha256=newsha,table_rows_preserved_during_authorization=True);atomic(archive/'authorization_committed.json',auth)
 atomic(ROOT/'pending_budget_handoff.json',{'state':'waiting_natural_boundary','authorization_id':archive.name,'archive_path':str(archive),'current_phase_cpu_budget_seconds':state['cpu_budget_seconds'],'cumulative_authorized_cpu_seconds':newbudget,'prior_run_id':leader['run_id'],'prior_leader':leader,'protocol_sha256':newsha,'worker_interruptions_requested':False})
 print(json.dumps(auth,ensure_ascii=False))
if __name__=='__main__':main()
