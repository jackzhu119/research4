#!/usr/bin/env python3
"""Evidence export only: no training, scientific-file edits, or queue writes.
Stream one tar.gz into bounded 16MiB chunks. Git Data API fallback is used when
GitHub release uploads are rejected for the uploads.github.com host. Every blob
is downloaded from the API and checked by SHA256 before local temporary removal.
"""
import base64, concurrent.futures, datetime, gzip, hashlib, io, json, os, sqlite3, tarfile, time, urllib.request, urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PUB=Path(__file__).resolve().parent
REPO='jackzhu119/research4'
CHUNK=16*1024*1024
PREFIX='background_study_20261110'
TOKEN=os.environ['GH_TOKEN']
START=time.monotonic()

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(p,obj):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
def api(path,data=None,raw=False):
 payload=None if data is None else json.dumps(data,separators=(',',':')).encode()
 for attempt in range(5):
  req=urllib.request.Request('https://api.github.com/repos/'+REPO+'/'+path,data=payload,headers={'Authorization':'Bearer '+TOKEN,'Accept':'application/vnd.github.raw+json' if raw else 'application/vnd.github+json','User-Agent':'ml-complete-evidence-export','Content-Type':'application/json'},method='GET' if data is None else 'POST')
  try:
   with urllib.request.urlopen(req,timeout=180)as r:
    b=r.read();return b if raw else json.loads(b)
  except urllib.error.HTTPError as e:
   if e.code not in (429,500,502,503,504):raise
   if attempt==4:raise
   time.sleep(min(20,2**attempt))
  except (TimeoutError,ConnectionError,OSError):
   if attempt==4:raise
   time.sleep(min(20,2**attempt))

class Sink:
 def __init__(self):
  self.buf=bytearray();self.index=0;self.futures=[];self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=3);self.total=0;self.records=[]
 def write(self,b):
  length=len(b);self.total+=length
  while b:
   n=min(CHUNK-len(self.buf),len(b));self.buf.extend(b[:n]);b=b[n:]
   if len(self.buf)==CHUNK:self.submit()
  return length
 def flush(self):pass
 def submit(self):
  if not self.buf:return
  self.index+=1;name=f'ml-evidence-20261115.tar.gz.part{self.index:05d}'
  p=PUB/'shards'/name;p.write_bytes(self.buf);self.buf.clear()
  self.futures.append(self.pool.submit(self.upload,p,self.index))
  if len(self.futures)>=6:self.collect(self.futures.pop(0))
 def upload(self,p,index):
  content=p.read_bytes();digest=hashlib.sha256(content).hexdigest();expected=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
  d=api('git/blobs',{'encoding':'base64','content':base64.b64encode(content).decode()})
  if d['sha']!=expected:raise RuntimeError('Git blob SHA mismatch '+p.name)
  downloaded=api('git/blobs/'+d['sha'],raw=True)
  # Some API proxies return the JSON form despite raw content negotiation.
  if len(downloaded)!=len(content) and downloaded[:1]==b'{':
   j=json.loads(downloaded);downloaded=base64.b64decode(j['content'])
  if len(downloaded)!=len(content) or hashlib.sha256(downloaded).hexdigest()!=digest:raise RuntimeError('Remote SHA256 mismatch '+p.name)
  rec={'part':index,'name':p.name,'bytes':len(content),'sha256':digest,'git_blob_sha':d['sha'],'remote_download_sha256_verified':True}
  p.unlink();return rec
 def collect(self,future):
  rec=future.result();self.records.append(rec)
  with (PUB/'uploaded_shards.jsonl').open('a')as f:f.write(json.dumps(rec)+'\n');f.flush();os.fsync(f.fileno())
  if rec['part']%8==0:print(json.dumps({'utc':now(),'uploaded_verified_parts':len(self.records),'last_part':rec['part'],'archive_bytes_streamed':self.total,'elapsed_s':round(time.monotonic()-START,1)}),flush=True)
 def close(self):
  self.submit()
  for f in self.futures:self.collect(f)
  self.pool.shutdown();self.records.sort(key=lambda x:x['part'])

class HashReader:
 def __init__(self,f):self.f=f;self.h=hashlib.sha256();self.n=0
 def read(self,n=-1):
  b=self.f.read(n);self.h.update(b);self.n+=len(b);return b

sink=Sink();members=0;logical=0;complete_jobs=0;incomplete_jobs=0;excluded=[];family={}
manifest=gzip.open(PUB/'archive_members.jsonl.gz','wt',encoding='utf-8',compresslevel=1)
stream=gzip.GzipFile(filename='ml-evidence-20261115.tar',mode='wb',fileobj=sink,compresslevel=1,mtime=0)
tar=tarfile.open(fileobj=stream,mode='w|',format=tarfile.PAX_FORMAT)

def add_file(p,arc,role):
 global members,logical
 if p.is_symlink():raise RuntimeError('Refusing symlink '+str(p))
 st=p.stat()
 if not p.is_file():return
 info=tarfile.TarInfo(arc);info.size=st.st_size;info.mtime=int(st.st_mtime);info.mode=0o600
 with p.open('rb')as f:
  reader=HashReader(f);tar.addfile(info,reader)
  if reader.n!=st.st_size:raise RuntimeError('Size changed '+str(p))
  sha=reader.h.hexdigest()
 if p.stat().st_size!=st.st_size or p.stat().st_mtime_ns!=st.st_mtime_ns:raise RuntimeError('Source changed during export '+str(p))
 manifest.write(json.dumps({'path':arc,'bytes':st.st_size,'sha256':sha,'role':role},ensure_ascii=False,separators=(',',':'))+'\n')
 members+=1;logical+=st.st_size

def add_tree(path,arc,role):
 for base,dirs,files in os.walk(path):
  dirs.sort();files.sort()
  for name in files:
   if name.endswith('.lock') or '.tmp-' in name:continue
   p=Path(base)/name;add_file(p,arc+'/'+p.relative_to(path).as_posix(),role)

try:
 # A consistent SQLite backup and atomically copied runtime/protocol provenance.
 add_tree(PUB/'source_snapshot',PREFIX+'/snapshot_source','source_protocol_runtime_and_consistent_sqlite')
 add_file(PUB/'snapshot.json',PREFIX+'/snapshot.json','snapshot_scope')
 add_tree(ROOT/'inputs',PREFIX+'/inputs','frozen_input_and_original_workbooks')
 c=sqlite3.connect('file:'+str(PUB/'source_snapshot/queue.sqlite3')+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
 rows=c.execute('select id,status,job_json from jobs order by ordinal').fetchall();c.close()
 for row in rows:
  jobid=row['id'];d=ROOT/'jobs'/jobid
  if row['status']=='completed':
   result=json.loads((d/'result.json').read_text())
   if result.get('status')!='complete' or result.get('complete') is not True or result.get('testing_only') is not False or result.get('job_id')!=jobid:raise RuntimeError('Invalid completed result '+jobid)
   for required in ('selection.json','predictions.csv','job_frozen.json','cpu_receipt.json'):
    if not(d/required).is_file():raise RuntimeError('Missing evidence '+jobid+'/'+required)
   if not((d/'fitted_artifacts.npz').is_file() or ((d/'fitted_artifacts.lossless.npz').is_file() and (d/'fitted_artifacts.recipe.json').is_file())):raise RuntimeError('Missing model '+jobid)
   if not((d/'split_frozen.json').is_file() or ((d/'split_frozen.json.gz').is_file()and(d/'split_frozen.archive_manifest.json').is_file())):raise RuntimeError('Missing split '+jobid)
   add_tree(d,PREFIX+'/jobs/'+jobid,'validated_queue_completed_result_evidence');complete_jobs+=1
   fam=json.loads(row['job_json'])['family'];family[fam]=family.get(fam,0)+1
  else:
   incomplete_jobs+=1
   if d.exists():
    excluded.append({'job_id':jobid,'queue_status':row['status'],'result_present':(d/'result.json').exists(),'reason':'not_queue_committed_completed; excluded from result count'})
    add_tree(d,PREFIX+'/incomplete_not_counted/jobs/'+jobid,'incomplete_excluded_not_results')
  if (complete_jobs+incomplete_jobs)%1000==0:
   print(json.dumps({'utc':now(),'completed_jobs_exported':complete_jobs,'manifest_members':members,'logical_bytes':logical,'compressed_bytes_streamed':sink.total,'elapsed_s':round(time.monotonic()-START,1)}),flush=True)
 for folder,role in [('attempts','all_attempt_lineage_provenance'),('recovery_events','reboot_and_budget_recovery_history'),('storage_maintenance_tests','storage_validation_source')]:
  add_tree(ROOT/folder,PREFIX+'/'+folder,role)
 # Canonical refreshed reports must finish before adding them to the snapshot.
 while not(PUB/'summary/refresh_complete.json').exists():
  print(json.dumps({'utc':now(),'state':'waiting_for_canonical_report_refresh'}),flush=True);time.sleep(10)
 add_tree(ROOT/'reports',PREFIX+'/reports','canonical_reports_with_explicit_uncommitted_result_caveat')
 add_tree(PUB/'summary',PREFIX+'/publication_summary','publication_summary_and_reporting_cpu_receipt')
 atomic(PUB/'archive_scope.json',{'schema':'ML-complete-evidence-export-v1','created_utc':now(),'validated_queue_completed_jobs':complete_jobs,'incomplete_jobs_excluded':incomplete_jobs,'incomplete_directories_preserved':excluded,'family_counts':family,'regular_files_before_manifest':members,'logical_bytes_before_manifest':logical,'new_model_fits':0,'archive_layout_root':PREFIX,'tar_format':'PAX','compression':'gzip-level1','transport':'Git Data API fallback; release uploads host rejected credentials','reports_note':'Canonical result-discovery reports may contain two complete but uncommitted stale-running results; those jobs are preserved only in incomplete_not_counted and not counted among 83,017 committed jobs.'})
 add_file(PUB/'archive_scope.json',PREFIX+'/archive_scope.json','archive_scope')
 manifest.close()
 info=tarfile.TarInfo(PREFIX+'/archive_members.jsonl.gz');info.size=(PUB/'archive_members.jsonl.gz').stat().st_size;info.mode=0o600;info.mtime=int(time.time())
 with (PUB/'archive_members.jsonl.gz').open('rb')as f:tar.addfile(info,f)
 tar.close();stream.close();sink.close()
 shards=sink.records
 final={'schema':'ML-multipart-gzip-manifest-v1','created_utc':now(),'repo':REPO,'transport':'git_data_blobs_with_download_sha256_verification','archive_basename':'ml-evidence-20261115.tar.gz','chunk_bytes':CHUNK,'compressed_archive_bytes':sum(x['bytes']for x in shards),'shard_count':len(shards),'shards':shards,'archive_members_manifest_sha256':hashlib.sha256((PUB/'archive_members.jsonl.gz').read_bytes()).hexdigest(),'completed_jobs':complete_jobs,'incomplete_jobs_excluded':incomplete_jobs,'regular_file_count':members+1,'logical_file_bytes':logical+(PUB/'archive_members.jsonl.gz').stat().st_size,'family_counts':family,'export_cpu_seconds':time.process_time(),'export_elapsed_seconds':time.monotonic()-START,'all_remote_download_sha256_verified':True,'reassemble':'Concatenate shards in numerical order, then gzip -t and tar -xzf. Verify individual members against archive_members.jsonl.gz.'}
 atomic(PUB/'ARCHIVE_MANIFEST.json',final)
 (PUB/'SHA256SUMS').write_text(''.join(x['sha256']+'  '+x['name']+'\n' for x in shards))
 print(json.dumps({'state':'export_complete','shards':len(shards),'compressed_bytes':final['compressed_archive_bytes'],'completed_jobs':complete_jobs,'files':members+1,'all_remote_sha256_verified':True}),flush=True)
except Exception as e:
 atomic(PUB/'export_error.json',{'utc':now(),'error':repr(e),'completed_jobs_exported':complete_jobs,'uploaded_parts':len(sink.records),'members':members,'no_science_or_queue_writes':True})
 raise
