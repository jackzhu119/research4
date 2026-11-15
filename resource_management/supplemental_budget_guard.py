#!/usr/bin/env python3
"""Resource-only enforcement of publication/report CPU under the original cap.
Never edits protocol, queue, or original budget_guard_ledger.json. Uses separate
persistent monotone accounts and the canonical STOP plus current-identity SIGTERM
path at original guard total + supplementary charges >= original trigger.
"""
import argparse,datetime,fcntl,json,os,signal,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PUB=Path(__file__).resolve().parent
BOOT=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
TERMINAL={'budget_exhausted','stopped_by_user','fatal_configuration_error'}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p,default=None):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return default

def atomic(p,v):
 t=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex)
 with t.open('w')as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def proc(pid):
 try:
  t=Path('/proc')/str(int(pid))/'stat';s=t.read_text();a=s[s.rfind(')')+2:].split();hz=os.sysconf('SC_CLK_TCK')
  return {'pid':int(pid),'state':a[0],'start_ticks':int(a[19]),'pgid':int(a[2]),'cpu_seconds':(int(a[11])+int(a[12]))/hz}
 except(OSError,ValueError,TypeError,IndexError):return None
def match(p,r):return bool(p and p['state']!='Z'and p['start_ticks']==r.get('start_ticks')and r.get('boot_id')==BOOT)
def account(ledger,pid,role):
 p=proc(pid)
 if not p:raise RuntimeError('Cannot identify '+role+' PID '+str(pid))
 key=f'{role}:{BOOT}:{pid}:{p["start_ticks"]}'
 ledger.setdefault('accounts',{}).setdefault(key,dict(p,boot_id=BOOT,role=role,cpu_high_water=0.,uncertain_tail_charge=100.))
 return key

a=argparse.ArgumentParser();a.add_argument('--export-pid',type=int);a.add_argument('--once',action='store_true');args=a.parse_args()
with(PUB/'supplemental_budget_guard.lock').open('a')as lock:
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('Supplemental guard already running')
 ledger=read(PUB/'publication_cpu_ledger.json',{'schema':'supplemental-publication-cpu-v1','accounts':{},'fixed_charges':{}})
 report=read(PUB/'summary/manual_report_cpu_receipt.json',{})
 report_charge=float(report.get('conservative_cpu_seconds',270.39))
 # Fixed allowance covers unregistered short CLI/shell/publisher control work.
 # It is an extra deduction, not additional authorization or a spent-CPU claim.
 ledger['fixed_charges'].setdefault('manual_canonical_report',{'conservative_cpu_seconds':report_charge,'receipt':'summary/manual_report_cpu_receipt.json'})
 ledger['fixed_charges'].setdefault('publication_control_allowance',{'conservative_cpu_seconds':1000.,'purpose':'Conservative deduction for short publication/API/publisher/control processes not in original guard'})
 ownkey=account(ledger,os.getpid(),'supplemental_guard')
 if args.export_pid:account(ledger,args.export_pid,'evidence_export_and_upload')
 atomic(PUB/'SUPPLEMENTAL_BUDGET_REQUIRED.json',{'schema':'supplemental-budget-requirement-v1','original_cap_unchanged':1360000.,'ledger':'publication_cpu_ledger.json','enforcer':'supplemental_budget_guard.py','rule':'original guard total plus persistent supplementary conservative charge must remain below original trigger','resume_requires_this_guard':True})
 above=0
 while True:
  for key,r in ledger['accounts'].items():
   p=proc(r['pid'])
   if match(p,r):r['cpu_high_water']=max(r.get('cpu_high_water',0.),p['cpu_seconds']);r['live']=True
   else:r['live']=False
  final=read(PUB/'ARCHIVE_MANIFEST.json',{})
  if final.get('export_cpu_seconds'):
   for r in ledger['accounts'].values():
    if r['role']=='evidence_export_and_upload':r['cpu_high_water']=max(r['cpu_high_water'],float(final['export_cpu_seconds']))
  supplement=sum(x['conservative_cpu_seconds']for x in ledger['fixed_charges'].values())+sum(r['cpu_high_water']+r['uncertain_tail_charge']for r in ledger['accounts'].values())
  original=read(ROOT/'budget_guard_status.json',{})
  protocol=read(ROOT/'protocol.json',{})
  if float(protocol.get('authorized_cpu_budget_seconds',0))!=1360000.:raise RuntimeError('Original cap changed; supplemental guard refuses to grant authorization')
  original_total=float(original.get('total_conservative_cpu_charge',0));trigger=float(original.get('trigger_cpu_seconds',1358000.));combined=original_total+supplement
  snap={'checked_at_utc':now(),'pid':os.getpid(),'start_ticks':proc(os.getpid())['start_ticks'],'boot_id':BOOT,'state':'running','original_guard_total_cpu_seconds':original_total,'supplementary_conservative_cpu_seconds':supplement,'combined_conservative_cpu_seconds':combined,'authorized_cpu_seconds':1360000.,'trigger_cpu_seconds':trigger,'remaining_to_cap_cpu_seconds':1360000.-combined,'remaining_to_trigger_cpu_seconds':trigger-combined,'protocol_or_historical_ledger_changed':False,'all_publication_cpu_included_conservatively':True}
  ledger['snapshot']=snap;atomic(PUB/'publication_cpu_ledger.json',ledger);atomic(PUB/'supplemental_budget_status.json',snap)
  if(ROOT/'STOP').exists()or read(ROOT/'status.json',{}).get('state')in TERMINAL:break
  above=above+1 if combined>=trigger else 0
  if above>=2:
   snap.update(state='resource_stop_requested',reason='supplemental_publication_cpu_budget_exhausted',scientific_protocol_changed=False,resource_stop_not_accuracy_stop=True)
   atomic(PUB/'SUPPLEMENTAL_CPU_BUDGET_STOP.json',snap)
   atomic(ROOT/'CPU_BUDGET_STOP.json',snap)
   try:
    with(ROOT/'STOP').open('x')as f:f.write('Resource budget exhausted: supplemental_publication_cpu_budget_exhausted; see CPU_BUDGET_STOP.json. Not a user/accuracy stop.\n');f.flush();os.fsync(f.fileno())
   except FileExistsError:pass
   leader=read(ROOT/'leader.json',{});p=proc(leader.get('pid'))
   if match(p,leader):
    try:os.kill(p['pid'],signal.SIGTERM)
    except ProcessLookupError:pass
   status=read(ROOT/'status.json',{})
   if status.get('boot_id')==BOOT:
    for w in status.get('workers',[]):
     p=proc(w.get('pid'))
     if p and p['start_ticks']==w.get('start_ticks')and p['pgid']==p['pid']:
      try:os.killpg(p['pgid'],signal.SIGTERM)
      except ProcessLookupError:pass
   atomic(PUB/'supplemental_budget_status.json',snap);break
  if args.once:break
  time.sleep(2)
