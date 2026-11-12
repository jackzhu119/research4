import argparse,json,os,time,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--job');a=p.parse_args();r=Path(a.root);j=json.loads(Path(a.job).read_text());o=r/'jobs'/j['id'];c=o/'counter.json';n=json.loads(c.read_text())if c.exists()else 0;c.write_text(json.dumps(n+1));t=time.process_time();duration=float(j.get('cpu',.12));x=1
while time.process_time()-t<duration:
 for _ in range(5000):x=(x*13+7)%100003
status=76 if j.get('blocked')else 1 if j.get('fail_once')and n==0 else 0
(o/'result.json').write_text(json.dumps({'complete':status==0,'status':'complete'if status==0 else 'scientific_blocked'if status==76 else 'failed','job_id':j['id']}));(o/'cpu_receipt.json').write_text(json.dumps({'attempt_id':os.environ['CHEM_ATTEMPT_ID'],'pid':os.getpid(),'process_cpu_seconds':time.process_time(),'exit_code':status}));sys.exit(status)
