#!/usr/bin/env python3
"""Publish existing verified evidence blobs and readable sources; no study edits."""
import argparse,base64,concurrent.futures,hashlib,json,os,time,urllib.request,urllib.error
from pathlib import Path
P=Path(__file__).resolve().parent
REPO='jackzhu119/research4'
ARCHIVE='research/background-study-1360000-20261115-archive'
SUMMARY='research/background-study-1360000-20261115'
TOKEN=os.environ['GH_TOKEN']
def api(path,data=None,method=None):
 for retry in range(12):
  req=urllib.request.Request('https://api.github.com/repos/'+REPO+'/'+path,data=None if data is None else json.dumps(data,separators=(',',':')).encode(),headers={'Authorization':'Bearer '+TOKEN,'Accept':'application/vnd.github+json','User-Agent':'ml-publication','Content-Type':'application/json'},method=method or ('POST' if data is not None else 'GET'))
  try:
   with urllib.request.urlopen(req,timeout=180)as r:return json.load(r)
  except urllib.error.HTTPError as e:
   if e.code==404:raise
   body=e.read().decode(errors='replace');limited=e.code in(403,429)and('rate limit'in body.lower()or e.headers.get('Retry-After'))
   transient_tree=e.code==422 and path=='git/trees'
   if(not limited and not transient_tree and e.code not in(500,502,503,504))or retry==11:raise RuntimeError('GitHub HTTP '+str(e.code)+' '+body[:500])
   delay=min(30,max(2,int(e.headers.get('Retry-After','5'))if limited else 2**min(retry,4)));print(json.dumps({'retrying_path':path,'http_status':e.code,'delay_seconds':delay}),flush=True);time.sleep(delay)
  except (OSError,TimeoutError):
   if retry==11:raise
   time.sleep(min(20,2**retry))
def blob_bytes(data):
 expected=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
 d=api('git/blobs',{'encoding':'base64','content':base64.b64encode(data).decode()})
 if d['sha']!=expected:raise RuntimeError('Git content SHA mismatch')
 return expected
def file_entry(item):
 name,path=item;return {'path':name,'mode':'100644','type':'blob','sha':blob_bytes(path.read_bytes())}
def commit_tree(branch,entries,message):
 tree=api('git/trees',{'tree':entries})
 try:ref=api('git/ref/heads/'+branch);parents=[ref['object']['sha']]
 except urllib.error.HTTPError as e:
  if e.code!=404:raise
  parents=[]
 commit=api('git/commits',{'message':message,'tree':tree['sha'],'parents':parents})
 if parents:api('git/refs/heads/'+branch,{'sha':commit['sha'],'force':False},'PATCH')
 else:api('git/refs',{'ref':'refs/heads/'+branch,'sha':commit['sha']})
 return {'branch':branch,'commit':commit['sha'],'tree':tree['sha'],'url':'https://github.com/'+REPO+'/tree/'+branch,'files':len(entries),'orphan_root':not parents}

a=argparse.ArgumentParser();a.add_argument('--checkpoint',action='store_true');args=a.parse_args()
records=[json.loads(x) for x in(P/'uploaded_shards.jsonl').read_text().splitlines()if x.strip()]
if not args.checkpoint:
 manifest=json.loads((P/'ARCHIVE_MANIFEST.json').read_text());assert manifest['all_remote_download_sha256_verified']and manifest['completed_jobs']==83017
 assert len(records)==manifest['shard_count']
records.sort(key=lambda x:x['part'])
assert [x['part']for x in records]==list(range(1,len(records)+1))
archive_entries=[{'path':'shards/'+r['name'],'mode':'100644','type':'blob','sha':r['git_blob_sha']}for r in records]
if args.checkpoint:
 readme=f'# ML evidence upload in progress\n\nVerified upload checkpoint: {len(records)} numbered16MiB shards. The complete archive manifest and final result will replace this checkpoint. Do not treat the current partial archive as complete.\n\nScope:83,017 queue-completed jobs;6,250 unfinished excluded. Original cumulative cap1,360,000CPU seconds, unchanged.\n'
 archive_entries.append({'path':'README.md','mode':'100644','type':'blob','sha':blob_bytes(readme.encode())})
 archive_entries.append(file_entry(('snapshot.json',P/'snapshot.json')))
 partial={'schema':'ML-incomplete-upload-checkpoint-v1','complete':False,'verified_uploaded_shards':len(records),'shards':records,'note':'Partial archive; do not infer study or archive completion.'}
 archive_entries.append({'path':'PARTIAL_ARCHIVE_MANIFEST.json','mode':'100644','type':'blob','sha':blob_bytes((json.dumps(partial,indent=2)+'\n').encode())})
 archive_entries.append({'path':'SHA256SUMS.partial','mode':'100644','type':'blob','sha':blob_bytes(''.join(r['sha256']+'  shards/'+r['name']+'\n'for r in records).encode())})
 result=commit_tree(ARCHIVE,archive_entries,'Checkpoint verified ML evidence archive upload')
 (P/'archive_branch_checkpoint.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));raise SystemExit
member_index={'schema':'ML-archive-member-manifest-index-v1','path_inside_archive':'background_study_20261110/archive_members.jsonl.gz','sha256':manifest['archive_members_manifest_sha256'],'compressed_manifest_bytes':(P/'archive_members.jsonl.gz').stat().st_size,'regular_file_count':manifest['regular_file_count'],'availability':'Included inside the complete multipart archive; no redundant standalone large Git blob.','verification':'Restore helper --verify-extracted checks this manifest SHA256 and all listed member bytes.'}
(P/'MEMBER_MANIFEST_INDEX.json').write_text(json.dumps(member_index,indent=2)+'\n')
archive_files=[('README.md',P/'PUBLICATION_README.md'),('restore_archive.py',P/'restore_archive.py'),('ARCHIVE_MANIFEST.json',P/'ARCHIVE_MANIFEST.json'),('SHA256SUMS',P/'SHA256SUMS'),('MEMBER_MANIFEST_INDEX.json',P/'MEMBER_MANIFEST_INDEX.json'),('archive_scope.json',P/'archive_scope.json'),('snapshot.json',P/'snapshot.json')]
with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:archive_entries+=list(pool.map(file_entry,archive_files))
archive=commit_tree(ARCHIVE,archive_entries,'Publish complete ML evidence:83,017 completed jobs with verified multipart archive')
(P/'archive_branch_publication.json').write_text(json.dumps(archive,indent=2)+'\n')
# Readable source/report branch has a separate orphan root, without the multi-GB
# default branch tree or individual per-job directories.
files=[]
for path in sorted((P/'source_snapshot').rglob('*')):
 if path.is_file()and not path.name.startswith('queue.sqlite3')and path.stat().st_size<=2*1024*1024:
  files.append(('source/'+path.relative_to(P/'source_snapshot').as_posix(),path))
for path in sorted((P/'summary').rglob('*')):
 if path.is_file():files.append(('summary/'+path.relative_to(P/'summary').as_posix(),path))
for path in sorted((P.parent/'reports').rglob('*')):
 if path.is_file()and path.stat().st_size<=2*1024*1024 and not path.name.startswith('.'):
  files.append(('reports/'+path.relative_to(P.parent/'reports').as_posix(),path))
files += [('README.md',P/'PUBLICATION_README.md'),('restore_archive.py',P/'restore_archive.py'),('ARCHIVE_MANIFEST.json',P/'ARCHIVE_MANIFEST.json'),('SHA256SUMS',P/'SHA256SUMS'),('MEMBER_MANIFEST_INDEX.json',P/'MEMBER_MANIFEST_INDEX.json'),('archive_scope.json',P/'archive_scope.json'),('snapshot.json',P/'snapshot.json'),('archive_branch_publication.json',P/'archive_branch_publication.json'),('export_complete_snapshot.py',P/'export_complete_snapshot.py'),('publish_git_snapshot.py',P/'publish_git_snapshot.py'),('verify_published_snapshot.py',P/'verify_published_snapshot.py'),('finalize_release_pointer.py',P/'finalize_release_pointer.py'),('resume_after_verified_publication.py',P/'resume_after_verified_publication.py')]
for name in ('supplemental_budget_guard.py','publication_cpu_ledger.json','supplemental_budget_status.json','SUPPLEMENTAL_BUDGET_REQUIRED.json','RESOURCE_BUDGET_ENFORCEMENT.md','SIDEGUARD_VALIDATION.json'):
 files.append(('resource_management/'+name,P/name))
with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:entries=list(pool.map(file_entry,files))
summary=commit_tree(SUMMARY,entries,'Publish ML progress, source and protocol with complete verified evidence archive')
(P/'summary_branch_publication.json').write_text(json.dumps(summary,indent=2)+'\n')
result={'archive':archive,'summary':summary,'remote_shards_sha256_verified':len(records),'completed_jobs':83017,'incomplete_jobs_excluded':6250}
(P/'GITHUB_PUBLICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
