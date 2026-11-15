#!/usr/bin/env python3
"""Recover the canonical four-worker study only after verified full publication."""
import datetime,hashlib,json,os,sqlite3,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parent;BOOT=Path('/proc/sys/kernel/random/boot_id').read_text().strip();TERMINAL={'budget_exhausted','stopped_by_user','fatal_configuration_error'}
def read(p):return json.loads(p.read_text())
def proc(pid):
 try:
  s=Path('/proc')/str(int(pid))/'stat';t=s.read_text();a=t[t.rfind(')')+2:].split();hz=os.sysconf('SC_CLK_TCK')
  return {'pid':int(pid),'state':a[0],'start_ticks':int(a[19]),'cpu_seconds':(int(a[11])+int(a[12]))/hz}
 except(OSError,ValueError,TypeError,IndexError):return None
def match(info,pidkey='pid',startkey='start_ticks'):
 p=proc(info.get(pidkey));return bool(p and p['state']!='Z'and p['start_ticks']==info.get(startkey)and info.get('boot_id')==BOOT)
def frozen_verify():
 protocol=read(ROOT/'protocol.json');assert protocol['authorized_cpu_budget_seconds']==1360000 and protocol['cpu_budget_seconds']==1360000 and protocol['max_workers']==4
 snapshot=read(P/'snapshot.json');assert hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()==snapshot['protocol_sha256']
 for name,expected in protocol['sealed_files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected
 return True
def sample():
 s=read(ROOT/'status.json');leader=read(ROOT/'leader.json');watchdog=read(ROOT/'watchdog.json');workers=[]
 for w in s.get('workers',[]):
  p=proc(w['pid'])
  if p and p['state']!='Z'and p['start_ticks']==w['start_ticks']and s['boot_id']==BOOT:workers.append(dict(p,job_id=w['job_id'],attempt_id=w['attempt_id'],observed_cpu_seconds=w['observed_cpu_seconds']))
 c=sqlite3.connect('file:'+str(ROOT/'queue.sqlite3')+'?mode=ro',uri=True)
 counts=dict(c.execute('select status,count(*) from jobs group by status'))
 cpu=c.execute("select coalesce(sum(verified_actual_cpu),0)+coalesce(sum(case when status in('launching','running')then observed_cpu else 0 end),0)from attempts").fetchone()[0];c.close()
 return {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':s['state'],'boot_id':s['boot_id'],'leader':leader,'leader_identity_live':match(leader),'watchdog':watchdog,'watchdog_identity_live':match(watchdog),'live_workers':workers,'live_worker_count':len(workers),'counts':counts,'verified_plus_active_observed_training_cpu_seconds':cpu}
def ready_sample(timeout=45):
 deadline=time.monotonic()+timeout
 while time.monotonic()<deadline:
  s=sample()
  if s['boot_id']==BOOT and s['leader_identity_live']and s['watchdog_identity_live']and s['live_worker_count']==4:return s
  time.sleep(.25)
 raise RuntimeError('Canonical4-workerpool did not verify before deadline')
assert not(ROOT/'STOP').exists()and read(ROOT/'status.json').get('state')not in TERMINAL
assert read(P/'PUBLICATION_VERIFICATION.json')['validated_completed_results']==83017
release=read(P/'RELEASE_PUBLICATION_VERIFIED.json');publication=read(P/'GITHUB_PUBLICATION.json');assert release['actual_tag_target_verified']and release['tag_commit']==publication['archive']['commit']
assert frozen_verify()
guard=read(ROOT/'budget_guard_status.json');side=read(P/'supplemental_budget_status.json');assert match(guard,'guard_pid','guard_start_ticks')and match(side)
assert side['combined_conservative_cpu_seconds']<side['trigger_cpu_seconds']and side['authorized_cpu_seconds']==1360000
before=sample()
with(P/'canonical_resume_launch.json').open('w')as f:subprocess.run([sys.executable,str(ROOT/'start_background.py'),'--root',str(ROOT)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=30)
a=ready_sample();time.sleep(12);b=ready_sample();assert b['verified_plus_active_observed_training_cpu_seconds']>a['verified_plus_active_observed_training_cpu_seconds']
c=sqlite3.connect('file:'+str(ROOT/'queue.sqlite3')+'?mode=ro',uri=True)
c.execute('attach database ? as frozen',('file:'+str(P/'source_snapshot/queue.sqlite3')+'?mode=ro',))
changed=c.execute("select count(*)from main.jobs j join frozen.jobs s on j.id=s.id where s.status='completed'and(j.status<>'completed'or j.attempt_count<>s.attempt_count)").fetchone()[0];assert changed==0
old_counts=dict(c.execute('select status,count(*)from frozen.jobs group by status'));c.close()
assert frozen_verify();assert not(ROOT/'STOP').exists()
side=read(P/'supplemental_budget_status.json');guard=read(ROOT/'budget_guard_status.json');assert match(guard,'guard_pid','guard_start_ticks')and match(side)
instances={'canonical_supervisor':[],'canonical_watchdog':[]}
for entry in Path('/proc').iterdir():
 if not entry.name.isdigit():continue
 try:args=(entry/'cmdline').read_bytes().split(b'\0')
 except OSError:continue
 if str(ROOT/'supervisor.py').encode()in args:instances['canonical_supervisor'].append(int(entry.name))
 if str(ROOT/'start_background.py').encode()in args and b'--watchdog'in args:instances['canonical_watchdog'].append(int(entry.name))
assert len(instances['canonical_supervisor'])==len(instances['canonical_watchdog'])==1
result={'schema':'ML-verified-publication-then-canonical-resume-v1','verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'boot_id':BOOT,'publication_archive_commit':publication['archive']['commit'],'release_url':release['release_url'],'snapshot_queue_counts':old_counts,'before_recovery':before,'sample_a':a,'sample_b':b,'training_cpu_growth_seconds':b['verified_plus_active_observed_training_cpu_seconds']-a['verified_plus_active_observed_training_cpu_seconds'],'live_worker_count':b['live_worker_count'],'canonical_instances':instances,'published_completed_jobs_requeued_or_reattempted':changed,'frozen_seven_and_protocol_unchanged':True,'original_guard':guard,'combined_budget_guard':side,'original_cap_unchanged':1360000,'background_process_lifecycle_limit':'Detached local processes do not survive cloud destruction; saved ledgers remain authoritative for recovery.'}
(P/'RESUME_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
event=ROOT/'recovery_events/20261115T020550Z_bootc1337_authorized_resume_publish/after_verified_upload_resume.json';event.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'canonical_restored':True,'leader_pid':b['leader']['pid'],'watchdog_pid':b['watchdog']['pid'],'workers':b['live_workers'],'counts':b['counts'],'training_cpu_growth_seconds':result['training_cpu_growth_seconds'],'combined_charge':side['combined_conservative_cpu_seconds'],'remaining_original_cap_cpu_seconds':side['remaining_to_cap_cpu_seconds'],'no_published_completed_jobs_reattempted':changed==0}))
