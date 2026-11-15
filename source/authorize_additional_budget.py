#!/usr/bin/env python3
"""Record a user-authorized budget extension without clearing results or CPU history."""
import argparse,hashlib,json,os,shutil,sqlite3,uuid
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,v):
 p=Path(p);tmp=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex)
 with tmp.open('w')as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def table_digest(db,name):
 h=hashlib.sha256()
 for row in db.execute(f'SELECT * FROM {name} ORDER BY id'):h.update(json.dumps(tuple(row),ensure_ascii=False,separators=(',',':')).encode());h.update(b'\n')
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--additional',type=int,required=True);p.add_argument('--user-authorization',required=True);a=p.parse_args()
 assert a.additional>0
 current=json.loads((ROOT/'protocol.json').read_text());oldhash=sha(ROOT/'protocol.json');status=json.loads((ROOT/'status.json').read_text())
 assert status['state']in {'budget_exhausted','stopped_by_user'}and status['active_workers']==0,'Study must be fully stopped before budget authorization'
 assert oldhash==(ROOT/'protocol.sha256').read_text().strip()
 for file,digest in current['sealed_files'].items():assert sha(ROOT/file)==digest,f'Frozen file changed: {file}'
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');archive=ROOT/'authorizations'/f'{stamp}_additional_{a.additional}';archive.mkdir(parents=True,exist_ok=False)
 for filename in ['protocol.json','protocol.sha256','status.json','STOP','CPU_BUDGET_STOP.json','budget_guard.py','budget_guard_status.json','budget_guard_ledger.json','reports/progress.json','reports/summary_manifest.json','reports/中文简报.md']:
  src=ROOT/filename
  if src.exists():dst=archive/filename;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 db=sqlite3.connect(ROOT/'queue.sqlite3',timeout=30);db.execute('PRAGMA synchronous=FULL')
 assert db.execute("SELECT value FROM meta WHERE key='protocol_sha256'").fetchone()[0]==oldhash
 assert db.execute("SELECT COUNT(*) FROM attempts WHERE status IN ('launching','running')").fetchone()[0]==0
 backup=sqlite3.connect(archive/'queue.sqlite3');db.backup(backup);backup.close()
 before={name:table_digest(db,name)for name in ['jobs','attempts']};baseline=float(db.execute('SELECT COALESCE(SUM(charged_cpu),0) FROM attempts').fetchone()[0]);oldbudget=int(current['cpu_budget_seconds']);newbudget=oldbudget+a.additional
 auth={'authorization_id':archive.name,'authorized_at_utc':datetime.now(timezone.utc).isoformat(),'user_authorization':a.user_authorization,'previous_authorized_cpu_seconds':oldbudget,'additional_authorized_cpu_seconds':a.additional,'cumulative_authorized_cpu_seconds':newbudget,'resume_unfinished_tasks':True,'prior_protocol_sha256':oldhash,'prior_training_conservative_charge_seconds':baseline,'prior_table_sha256':before,'archive_path':str(archive),'scientific_configuration_changed':False,'prior_CPU_ledger_reset':False}
 atomic(archive/'authorization_prepared.json',auth)
 current['cpu_budget_seconds']=newbudget;current['authorized_cpu_budget_seconds']=newbudget
 current.setdefault('budget_authorization_history',[]).append(auth)
 atomic(ROOT/'protocol.json',current);newhash=sha(ROOT/'protocol.json');(ROOT/'protocol.sha256').write_text(newhash+'\n')
 with db:
  changed=db.execute("UPDATE meta SET value=? WHERE key='protocol_sha256' AND value=?",(newhash,oldhash)).rowcount;assert changed==1
  db.execute('INSERT INTO events(utc,event,detail)VALUES(?,?,?)',(datetime.now(timezone.utc).isoformat(),'additional_CPU_budget_authorized',json.dumps(dict(auth,new_protocol_sha256=newhash),ensure_ascii=False)))
 after={name:table_digest(db,name)for name in ['jobs','attempts']};assert before==after,'Jobs or CPU attempt rows changed during authorization'
 for filename in ['STOP','CPU_BUDGET_STOP.json']:
  src=ROOT/filename
  if src.exists():src.rename(archive/('retired_'+filename))
 status.update(state='resuming_with_additional_budget',reason='explicit_user_authorized_CPU_extension',cpu_budget_seconds=newbudget,authorized_additional_cpu_seconds=a.additional,updated_utc=datetime.now(timezone.utc).isoformat(),active_workers=0)
 atomic(ROOT/'status.json',status);auth.update(new_protocol_sha256=newhash,table_history_preserved=True);atomic(archive/'authorization_committed.json',auth);db.close();print(json.dumps(auth,ensure_ascii=False))
if __name__=='__main__':main()
