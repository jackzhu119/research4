#!/usr/bin/env python3
"""Add measured post-publication recovery records to the readable source branch."""
import base64,concurrent.futures,datetime,hashlib,json,os,time,urllib.request,urllib.error
from pathlib import Path
P=Path(__file__).resolve().parent;REPO='jackzhu119/research4';TOKEN=os.environ['GH_TOKEN']
def api(path,data=None,method=None):
 for retry in range(12):
  q=urllib.request.Request('https://api.github.com/repos/'+REPO+'/'+path,data=None if data is None else json.dumps(data).encode(),headers={'Authorization':'Bearer '+TOKEN,'Accept':'application/vnd.github+json','Content-Type':'application/json','User-Agent':'ml-runtime-publication'},method=method or('POST'if data is not None else'GET'))
  try:
   with urllib.request.urlopen(q,timeout=120)as r:return json.load(r)
  except urllib.error.HTTPError as e:
   b=e.read().decode(errors='replace');limited=e.code in(403,429)and('rate limit'in b.lower()or e.headers.get('Retry-After'));transient=e.code==422 and path=='git/trees'
   if(not limited and not transient and e.code not in(500,502,503,504))or retry==11:raise RuntimeError('GitHub HTTP '+str(e.code)+' '+b[:500])
   delay=min(30,max(2,int(e.headers.get('Retry-After','5'))if limited else 2**min(retry,4)));print(json.dumps({'waiting_http':e.code,'seconds':delay}),flush=True);time.sleep(delay)
def add(item):
 path,file=item;b=file.read_bytes();expected=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();r=api('git/blobs',{'encoding':'base64','content':base64.b64encode(b).decode()});assert r['sha']==expected
 return {'path':path,'mode':'100644','type':'blob','sha':expected},b
r=json.loads((P/'RESUME_VERIFICATION.json').read_text());assert r['live_worker_count']==4 and r['training_cpu_growth_seconds']>0 and r['published_completed_jobs_requeued_or_reattempted']==0
pub=json.loads((P/'GITHUB_PUBLICATION.json').read_text());branch=pub['summary']['branch'];ref=api('git/ref/heads/'+branch);assert ref['object']['sha']==pub['summary']['commit']
status=f'''# Canonical background recovery verified

Verified {r['verified_utc']} after the complete archive publication and release tag readback.

- Current boot: `{r['boot_id']}`.
- Canonical supervisor PID {r['sample_b']['leader']['pid']}; watchdog PID {r['sample_b']['watchdog']['pid']}.
- Four current-identity workers observed; training CPU grew {r['training_cpu_growth_seconds']:.3f} seconds between samples.
- Queue counts at verification: `{json.dumps(r['sample_b']['counts'],sort_keys=True)}`.
- Zero of the83,017 published completed jobs were requeued or reattempted. The frozen seven hashes and protocol remain unchanged.
- Original cumulative cap:1,360,000 CPU seconds. Combined conservative charge:{r['combined_budget_guard']['combined_conservative_cpu_seconds']:.3f}; remaining:{r['combined_budget_guard']['remaining_to_cap_cpu_seconds']:.3f}. Publication/report CPU is deducted by the supplementary guard.

The immutable archive is the capture before recovery; subsequent training results are later progress. Detached local processes cannot survive cloud destruction. Preserve the original and supplemental ledgers for any later authorized recovery.
'''.replace('the83,017','the 83,017').replace('cap:1','cap: 1').replace('charge:', 'charge: ').replace('remaining:', 'remaining: ')
(P/'RUNNING_STATUS.md').write_text(status)
files=[('runtime/'+name,P/name)for name in('RESUME_VERIFICATION.json','PUBLICATION_VERIFICATION.json','RELEASE_PUBLICATION_VERIFIED.json','canonical_resume_launch.json','RUNNING_STATUS.md','RELEASE_NOTES.md','tag_before_archive_move.json')]
files += [('resource_management/'+name,P/name)for name in('publication_cpu_ledger.json','supplemental_budget_status.json','SUPPLEMENTAL_BUDGET_REQUIRED.json')]
files += [('publish_runtime_verification.py',Path(__file__))]
with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:results=list(pool.map(add,files))
entries=[x[0]for x in results];captured={x[0]['path']:x[1]for x in results}
tree=api('git/trees',{'base_tree':pub['summary']['tree'],'tree':entries});commit=api('git/commits',{'tree':tree['sha'],'parents':[pub['summary']['commit']],'message':'Record verified four-worker recovery after complete ML evidence publication'})
api('git/refs/heads/'+branch,{'sha':commit['sha'],'force':False},'PATCH')
ref2=api('git/ref/heads/'+branch);assert ref2['object']['sha']==commit['sha']
remote=api('git/trees/'+tree['sha']+'?recursive=1');assert not remote['truncated'];paths={x['path']:x['sha']for x in remote['tree']if x['type']=='blob'}
for name in('runtime/RESUME_VERIFICATION.json','runtime/PUBLICATION_VERIFICATION.json','runtime/RELEASE_PUBLICATION_VERIFIED.json','resource_management/publication_cpu_ledger.json','resource_management/supplemental_budget_status.json'):
 content=base64.b64decode(api('git/blobs/'+paths[name])['content']);assert content==captured[name]
assert paths['ARCHIVE_MANIFEST.json']==hashlib.sha1(b'blob '+str((P/'ARCHIVE_MANIFEST.json').stat().st_size).encode()+b'\0'+(P/'ARCHIVE_MANIFEST.json').read_bytes()).hexdigest()
pub['summary'].update(commit=commit['sha'],tree=tree['sha'],files=len(paths));(P/'GITHUB_PUBLICATION.json').write_text(json.dumps(pub,indent=2)+'\n')
verification={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'summary_branch':branch,'summary_commit':commit['sha'],'summary_tree':tree['sha'],'summary_files':len(paths),'runtime_file_readbacks_match':True,'archive_manifest_unchanged':True,'archive_commit':pub['archive']['commit'],'release_url':r['release_url'],'original_cap_unchanged':1360000,'live_worker_count_at_resume_verification':4,'training_cpu_growth_seconds':r['training_cpu_growth_seconds'],'remaining_original_cap_at_resume':r['combined_budget_guard']['remaining_to_cap_cpu_seconds']}
(P/'RUNTIME_PUBLICATION_VERIFIED.json').write_text(json.dumps(verification,indent=2)+'\n');print(json.dumps(verification))
