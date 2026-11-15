#!/usr/bin/env python3
"""Read-only completeness and remote-ref verification after publication."""
import base64,gzip,hashlib,json,os,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;REPO='jackzhu119/research4';TOKEN=os.environ['GH_TOKEN']
def api(path):
 q=urllib.request.Request('https://api.github.com/repos/'+REPO+'/'+path,headers={'Authorization':'Bearer '+TOKEN,'User-Agent':'ml-publication-verification','Accept':'application/vnd.github+json'})
 with urllib.request.urlopen(q,timeout=120)as r:return json.load(r)
def blob(sha):return base64.b64decode(api('git/blobs/'+sha)['content'])
manifest=json.loads((P/'ARCHIVE_MANIFEST.json').read_text());pub=json.loads((P/'GITHUB_PUBLICATION.json').read_text());snapshot=json.loads((P/'snapshot.json').read_text())
assert manifest['completed_jobs']==83017 and manifest['incomplete_jobs_excluded']==6250
assert manifest['all_remote_download_sha256_verified'] and all(x['remote_download_sha256_verified']for x in manifest['shards'])
assert sum(x['bytes']for x in manifest['shards'])==manifest['compressed_archive_bytes']
assert [x['part']for x in manifest['shards']]==list(range(1,manifest['shard_count']+1))
assert hashlib.sha256((P/'archive_members.jsonl.gz').read_bytes()).hexdigest()==manifest['archive_members_manifest_sha256']
result_jobs=set();models=set();selections=set();predictions=set();incomplete_results=[];frozen_seen={};files=0;logical=0
with gzip.open(P/'archive_members.jsonl.gz','rt',encoding='utf-8')as f:
 for line in f:
  r=json.loads(line);path=r['path'];files+=1;logical+=r['bytes'];parts=path.split('/')
  if len(parts)>=4 and parts[1]=='jobs':
   job=parts[2];name=parts[-1]
   if name=='result.json':assert job not in result_jobs;result_jobs.add(job)
   if name in ('fitted_artifacts.npz','fitted_artifacts.lossless.npz'):models.add(job)
   if name=='selection.json':selections.add(job)
   if name=='predictions.csv':predictions.add(job)
  if '/incomplete_not_counted/jobs/'in path and path.endswith('/result.json'):incomplete_results.append(path)
  for name,hashes in snapshot['frozen_seven'].items():
   expected='background_study_20261110/'+(name if name.startswith('inputs/')else'snapshot_source/'+name)
   if path==expected:assert r['sha256']==hashes['expected'];frozen_seen[name]=r['sha256']
assert len(result_jobs)==83017 and models==result_jobs and selections==result_jobs and predictions==result_jobs
assert len(frozen_seen)==7
assert files+1==manifest['regular_file_count']
assert logical+(P/'archive_members.jsonl.gz').stat().st_size==manifest['logical_file_bytes']
remote={}
for kind in ('archive','summary'):
 info=pub[kind];ref=api('git/ref/heads/'+info['branch']);assert ref['object']['sha']==info['commit']
 tree=api('git/trees/'+info['tree']+'?recursive=1');assert not tree['truncated'];paths={r['path']:r['sha']for r in tree['tree']if r['type']=='blob'}
 if kind=='archive':
  for shard in manifest['shards']:assert paths['shards/'+shard['name']]==shard['git_blob_sha']
  assert blob(paths['ARCHIVE_MANIFEST.json'])==(P/'ARCHIVE_MANIFEST.json').read_bytes()
  assert blob(paths['SHA256SUMS'])==(P/'SHA256SUMS').read_bytes()
  assert blob(paths['snapshot.json'])==(P/'snapshot.json').read_bytes()
 if kind=='summary':
  assert hashlib.sha256(blob(paths['source/study_worker.py'])).hexdigest()==snapshot['frozen_seven']['study_worker.py']['expected']
  assert blob(paths['ARCHIVE_MANIFEST.json'])==(P/'ARCHIVE_MANIFEST.json').read_bytes()
  assert blob(paths['summary/research_summary.json'])==(P/'summary/research_summary.json').read_bytes()
 remote[kind]={'ref_commit_matches':True,'tree_files':len(paths),'tree_not_truncated':True,'key_file_readback_matches':True}
verification={'schema':'ML-GitHub-complete-publication-verification-v1','validated_completed_results':len(result_jobs),'model_prediction_selection_membership_exact':True,'incomplete_complete_files_preserved_not_counted':len(incomplete_results),'frozen7_manifest_hashes_match':True,'regular_files':files+1,'logical_bytes':manifest['logical_file_bytes'],'shards_remote_download_sha256_verified':manifest['shard_count'],'compressed_archive_bytes':manifest['compressed_archive_bytes'],'remote_refs_trees_and_key_readback':remote,'no_new_model_fits':True}
(P/'PUBLICATION_VERIFICATION.json').write_text(json.dumps(verification,indent=2)+'\n');print(json.dumps(verification))
