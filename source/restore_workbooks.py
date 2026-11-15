"""Restore only registry-addressed workbooks, preserving immutable SHA256 identities.
No classification or alteration of the user's repository is performed.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import urllib.request,urllib.parse,hashlib,json,shutil,time,os
B=Path(__file__).resolve().parent
SRC=Path('/workspace/applied_spectroscopy_revision/source/research4_Applied_Spectroscopy')
LOCAL=Path('/workspace/research4')
REG=json.loads((SRC/'evidence/active_workbooks.json').read_text())
OLD=json.loads((LOCAL/'data/active_workbooks.json').read_text())
OUT=B/'inputs/workbooks';OUT.mkdir(parents=True,exist_ok=True)
COMMIT='0156e2189d13456d1faf05ebfd53d86eeeb4f5e7'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
local_bysha={}
for entry in OLD['workbooks']:
 p=LOCAL/entry['path']
 if p.exists():
  h=sha(p)
  if h!=entry['sha256']:raise ValueError('Old registry hash mismatch '+str(p))
  local_bysha[h]=p

def restore(w):
 gid=w['origin_group_id'];target=OUT/(gid+'__'+Path(w['path']).name);expect=w['sha256']
 record={'origin_group_id':gid,'species':w['species'],'record_count':w['record_count'],'registry_path':w['path'],'target_path':str(target),'expected_sha256':expect,'immutable_commit':COMMIT}
 try:
  if target.exists()and sha(target)==expect:record.update(status='verified',method='existing_task_copy')
  elif expect in local_bysha:
   shutil.copyfile(local_bysha[expect],target);record.update(status='verified',method='exact_sha_copy_from_local_active_registry',local_source=str(local_bysha[expect]))
  else:
   url='https://raw.githubusercontent.com/jackzhu119/research4/'+COMMIT+'/'+urllib.parse.quote(w['path'],safe='/')
   record['source_url']=url
   req=urllib.request.Request(url,headers={'User-Agent':'research4-readonly-input-restoration'})
   tmp=target.with_suffix('.download')
   with urllib.request.urlopen(req,timeout=55) as response,tmp.open('wb')as f:
    record['http_status']=response.status
    while True:
     chunk=response.read(1024*1024)
     if not chunk:break
     f.write(chunk)
   got=sha(tmp)
   if got!=expect:raise ValueError(f'SHA256 mismatch expected{expect} received{got}; not used')
   tmp.replace(target);record.update(status='verified',method='immutable_public_GitHub_download')
  record['actual_sha256']=sha(target);record['bytes']=target.stat().st_size
  assert record['actual_sha256']==expect
 except Exception as e:record.update(status='blocked',error=type(e).__name__+': '+str(e))
 return record
records=[]
with ThreadPoolExecutor(max_workers=4)as pool:
 futures=[pool.submit(restore,w)for w in REG['workbooks']]
 for future in as_completed(futures):
  record=future.result();records.append(record)
  print(record['origin_group_id'],record['status'],record.get('method',record.get('error')),record.get('bytes'),flush=True)
  (B/'workbook_restoration_manifest.json').write_text(json.dumps({'registry_path':str(SRC/'evidence/active_workbooks.json'),'registry_sha256':sha(SRC/'evidence/active_workbooks.json'),'source_commit':COMMIT,'verified':sum(r['status']=='verified'for r in records),'required':18,'records':sorted(records,key=lambda r:r['origin_group_id'])},ensure_ascii=False,indent=2))
print('RESTORATION COMPLETE',sum(r['status']=='verified'for r in records),'/',len(records),flush=True)
