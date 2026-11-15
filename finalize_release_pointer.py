#!/usr/bin/env python3
"""Point only this session's new ML release tag at the verified full archive."""
import datetime,json,os,time,urllib.request,urllib.error
from pathlib import Path
P=Path(__file__).resolve().parent;REPO='jackzhu119/research4';TAG='research-ml-results-20261115';TOKEN=os.environ['GH_TOKEN']
def api(path,data=None,method=None):
 for retry in range(12):
  req=urllib.request.Request('https://api.github.com/repos/'+REPO+'/'+path,data=None if data is None else json.dumps(data).encode(),headers={'Authorization':'Bearer '+TOKEN,'Accept':'application/vnd.github+json','User-Agent':'ml-release-pointer','Content-Type':'application/json'},method=method or('POST'if data is not None else'GET'))
  try:
   with urllib.request.urlopen(req,timeout=120)as r:return json.load(r)
  except urllib.error.HTTPError as e:
   msg=e.read().decode(errors='replace').lower();limited=e.code in(403,429)and(any(x in msg for x in('rate limit','secondary rate','temporarily blocked'))or e.headers.get('Retry-After'))
   if not limited and e.code not in(500,502,503,504):raise RuntimeError('GitHub request failed '+str(e.code)+' '+msg[:150])
   if retry==11:raise
   delay=min(30,max(2,int(e.headers.get('Retry-After','5'))if limited else 2**min(retry,4)));print(json.dumps({'waiting_after_http':e.code,'seconds':delay}),flush=True);time.sleep(delay)
pub=json.loads((P/'GITHUB_PUBLICATION.json').read_text());m=json.loads((P/'ARCHIVE_MANIFEST.json').read_text());v=json.loads((P/'PUBLICATION_VERIFICATION.json').read_text());archive=pub['archive']['commit'];assert v['validated_completed_results']==83017 and v['shards_remote_download_sha256_verified']==m['shard_count']
old=json.loads((P/'tag_before_archive_move.json').read_text());current=api('git/ref/tags/'+TAG)
assert old['ref']==current['ref']=='refs/tags/'+TAG
if current['object']['sha']!=archive:
 if current['object']['sha']!=old['object']['sha']:raise RuntimeError('New ML tag changed concurrently; refusing to move unexpected target')
 api('git/refs/tags/'+TAG,{'sha':archive,'force':True},'PATCH')
base='https://github.com/'+REPO
body=f'''This is a complete, immutable evidence snapshot of the **old ML background study**, captured before authorized background recovery on 2026-11-15. It is independent of the chemistry computation.

- **83,017 queue-committed completed jobs** of89,267 scheduled jobs. The6,246 queued and4 previous-boot running jobs are excluded from the completed result count; existing incomplete files are preserved under an explicit incomplete path.
- Complete result, prediction, selection, fitted-model/lossless-recipe, diagnostic and split evidence; exact frozen NPZ and original workbooks; source/protocol; consistent SQLite backup; all attempt/recovery provenance and refreshed reports.
- Archive: **{m['compressed_archive_bytes']:,} bytes in {m['shard_count']} numbered16MiB parts**, containing {m['regular_file_count']:,} regular files. Every shard was uploaded, downloaded, and SHA-256 verified. All seven frozen source/input hashes match.

**[Download the complete multipart archive]({base}/tree/{archive})** · [Manifest and SHA-256 values]({base}/blob/{archive}/ARCHIVE_MANIFEST.json) · [Restore/download helper]({base}/blob/{archive}/restore_archive.py)

[Research summary, source, protocol and runtime records]({base}/tree/{pub['summary']['branch']})

Archive commit: `{archive}`. This release tag resolves to that exact archive commit. The archive branch is independent of the pre-existing multi-GB repository tree.

The release has **no binary release assets**: the authenticated `uploads.github.com` endpoint returned HTTP401. Full data was successfully stored and verified through Git Data API blobs on the archive branch instead. Use the numbered shards and restore helper above. The helper verifies SHA-256 and supports streaming gzip CRC verification without an extra full archive copy.

Canonical reports were refreshed without fitting new models:83,019 complete non-testing result files, zero invalid/testing. Two complete files were not yet queue-committed and are separately labelled incomplete, excluded from the83,017 archived-completed count and from full-repeat source estimates. The study remains incomplete; reported uncertainty is conditional on the same300 physical source specimens and is not external validation.

The original cumulative authorization remains **1,360,000 CPU seconds**. Publication/report CPU is conservatively deducted by the supplementary resource guard; no authorization or scientific protocol was changed. Canonical training resumes only after this verified upload. Later computation is recorded separately from this immutable cutoff.
'''
body=body.replace('of89','of 89').replace('The6,246','The 6,246').replace('and4','and 4').replace('numbered16MiB','numbered 16MiB').replace('HTTP401','HTTP 401').replace(':83,019',': 83,019').replace('the83,017','the 83,017').replace('same300','same 300')
(P/'RELEASE_NOTES.md').write_text(body)
release=api('releases/tags/'+TAG)
release=api('releases/'+str(release['id']),{'name':'ML background study:83,017 completed jobs, complete verified evidence (2026-11-15)','body':body,'prerelease':True,'target_commitish':archive,'make_latest':'false'},'PATCH')
ref=api('git/ref/tags/'+TAG);assert ref['object']['sha']==archive
assert release['prerelease']and len(release['assets'])==0 and release['target_commitish']==archive
result={'tag':TAG,'tag_commit':archive,'tag_prior_commit':old['object']['sha'],'release_url':release['html_url'],'release_id':release['id'],'is_prerelease':release['prerelease'],'release_assets':len(release['assets']),'archive_bytes':m['compressed_archive_bytes'],'verified_shards':m['shard_count'],'actual_tag_target_verified':True,'no_other_tags_modified':True,'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(P/'RELEASE_PUBLICATION_VERIFIED.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
