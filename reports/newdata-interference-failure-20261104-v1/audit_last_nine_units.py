"""Independent read-only failed-phase audit; only18 small closed RHF NPYs.

No PySCF import, integrals, SCF, gradient, Hessian evaluation, or Git operation.
This emits a root-review template with approval=false, never an approval.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import os,resource,sys,json,hashlib

for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS,(int(1.5*1024**3),int(1.5*1024**3)))
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
BASE=ROOT/'reports/pyscf-spinosin-resumable-20261104-v5'
PHASE='v5-spinosin-resumable-segment-016'
FOLDER=BASE/'direct-resume-001'/PHASE
CACHE=ROOT/'outputs/pyscf-spinosin-resumable-20261104-v5/spinosin-C001-restart'
SNAPSHOT_SHA='3089d6014ba2d2283540eb13cc7dbcdbbf284721042e7250b624197fc15c3311'
WORKER_SHA='b40188f747c01e9efa97672af6eedfd281a9e9697d20bc7d57f125bb131fa190'
PROOF_SHA='ec1ebd3e5117f1831fa355fa3a0a21103bea7dabbb9bd4a57b715bb3a7f97a68'
INPUTS={};CHECKS=[]
def require(x,label):
    CHECKS.append({'check':label,'pass':bool(x)})
    if not x:raise ValueError(label)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def record(p,expected=None):
    p=Path(p).resolve();require(p.is_file()and not p.is_symlink(),'Regular saved file '+p.name)
    digest=sha(p);require(expected is None or digest==expected,'Source SHA '+str(p.relative_to(ROOT)))
    INPUTS[str(p.relative_to(ROOT))]={'sha256':digest,'bytes':p.stat().st_size};return p
def read(p,expected=None):return json.loads(record(p,expected).read_text())
def healthy(row):return row['status']=='EXITED'and row['exit_code']==0 and all(row.get(k)is False for k in ('wall_terminated','memory_terminated','cpu_terminated','monitor_failed','hard_killed'))
def converted(fp):return {'shape':fp['shape'],'dtype':fp['dtype'],'bytes':fp['payload_bytes'],'sha256':fp['payload_sha256_C_order']}

def main():
    require(not(OUT/'audit.json').exists(),'Fresh independent audit')
    ledgerpath=record(BASE/'cpu_budget.json');ledgerbytes=ledgerpath.read_bytes();ledger=json.loads(ledgerbytes)
    matches=[x for x in ledger['phases']if x['phase']==PHASE];require(len(matches)==1,'Unique failed phase016');row=matches[0]
    require(row['status']=='PARTIAL_OR_FAILED'and row['exit_code']==-9 and row['memory_terminated']is True and row['hard_killed']is True,'Preserve actual memory/SIGKILL failure flags')
    require(row['resource_stop_reason']=='CGROUP_MEMORY_IMMEDIATE_KILL'and row['monitor_failed']is False
            and row['cpu_terminated']is False and row['wall_terminated']is False,'Known cgroup guard trigger; not timeout or monitor failure')
    require(row['actual_cpu_seconds']==8627.573778 and bool(row['ended_at_utc']),'Closed actual wait4 cost')
    require(not (Path('/proc')/str(row['pid'])).exists(),'Original worker no longer exists')
    require(row['memory_stop_cgroup']['current_bytes']>=row['cgroup_kill_bytes']and row['memory_stop_snapshot']['rss_bytes']<row['rss_stop_bytes'],'Cgroup threshold exceeded despite RSS below threshold')
    require(any(x['sent']is True and x['signal']==9 for x in row['identity_checked_signal_attempts']),'Actual identity-checked group SIGKILL sent')
    proof=read(row['supervisor_proof_path'],PROOF_SHA);require(proof['child_pid']==row['pid']and proof['phase']==PHASE,'Failed parent/proof identity')
    record(row['log'],row['log_sha256']);entry=read(FOLDER/'input.json');progress=read(FOLDER/'progress.json');capture=read(FOLDER/'hessian_cache_manifest.json')
    require(entry['resource_supervision']['proof']==proof and progress['resource_supervision']['proof']==proof,'Input/progress exact failed proof')
    require(entry['source_worker']['sha256']==WORKER_SHA and entry['snapshot_manifest_sha256']==SNAPSHOT_SHA,'Frozen consumer and exact source MF')
    record(entry['source_worker']['path'],WORKER_SHA)
    record(ROOT/'scripts/pyscf_molecular_state_worker_20261104_v5.py','be4727b76d59a0b0dd3defba249ef833ee8ef0ea89473e390dcef389728415c8')
    record(ROOT/'scripts/pyscf_incremental_rhf_checkpoint_20261103.py','ce96055f24b39833337844c178f94597a9ac4ba7721ba019816d03628f971f55')
    record(proof['supervisor_script_path'],proof['supervisor_script_sha256'])
    snapshot=read(Path(entry['snapshot_directory'])/'manifest.json',SNAPSHOT_SHA);header=read(CACHE/'header.json')
    require(header['natm']==75 and header['nao']==762 and header['total_units']==150,'Expected complete-helper dimensions')
    require(hashlib.sha256(canonical(header['binding'])).hexdigest()==header['binding_sha256'],'Canonical cache binding SHA')
    binding=header['binding'];stable=binding['stable_snapshot'];actual=binding['actual_runtime']
    require(stable==progress['source_state_binding']and stable['snapshot_manifest_sha256']==SNAPSHOT_SHA,'Exact stable MF binding in cache/progress')
    require(stable['arrays']==snapshot['arrays']and stable['molecule_fingerprint']==snapshot['molecule_fingerprint']
            and actual['energy_hartree']==snapshot['energy_hartree'],'Healthy source energy/geometry/array fingerprints')
    for k in ('MO_coeff','MO_energy','MO_occ'):require(actual[k]==converted(snapshot['arrays'][k]),'Source MO payload binding '+k)
    for a,b in [('coords','grid_coordinates'),('weights','grid_weights'),('atm_idx','grid_atm_idx'),('non0tab','grid_non0tab'),('screen_index','grid_screen_index'),('quadrature_weights','grid_quadrature_weights')]:
        require(actual['grid'][a]==converted(snapshot['arrays'][b]),'Exact saved quadrature binding '+a)
    physical=snapshot['molecule_fingerprint'];mol=actual['molecule']
    for a,b in [('atm','atm'),('bas','bas'),('charges','charges'),('coords_bohr','coordinates_bohr'),('env_except_17','env_except_transient17')]:
        require(mol[a]==converted(physical[b]),'Molecule fingerprint naming-normalized equality '+a)
    require(all(mol[k]==physical[k]for k in ('charge','spin','natm','nao','nbas','nelectron')),'Molecule scalar identity equality')
    allrecords=sorted((CACHE/'units').glob('*/attempt-*/complete.json'));indices=[];last=[];prior=Counter();allowed=[];arraychecks=[]
    byphase={r['phase']:r for r in ledger['phases']}
    for path in allrecords:
        rec=read(path);i=rec['index'];indices.append(i);ev=rec['execution_evidence'];producer=ev['resource_supervision']['proof']['phase']
        require(rec['header_sha256']==sha(CACHE/'header.json')and rec['binding_sha256']==header['binding_sha256'],'Unit exact cache header '+str(i))
        require(ev['snapshot_manifest_sha256']==SNAPSHOT_SHA and ev['source_worker']==entry['source_worker'],'Unit source consumer/MF '+str(i))
        if i<141:
            require(healthy(byphase[producer]),'Prior unit has healthy closed producer '+str(i));prior[producer]+=1;continue
        require(141<=i<=149 and producer==PHASE and ev['resource_supervision']==entry['resource_supervision']
                and Path(ev['output_directory'])==FOLDER,'Last-nine failed-parent binding '+str(i))
        require(rec['scope']=='ONE_COMPLETE_RHF_UNIT_NOT_A_COMPLETE_HESSIAN'and rec['unit_name']==f'{i:06d}-row-{i-75:04d}','Complete row-only scope '+str(i))
        require(set(rec['arrays'])=={'e1','ej'},'Exactly complete E1/EJ row arrays '+str(i))
        require(rec['completed_utc']<row['memory_stop_cgroup']['observed_at_utc'],'Unit completion strictly before memory stop '+str(i))
        for name,info in rec['arrays'].items():
            ap=record(path.parent/info['file'],info['file_sha256']);require(ap.stat().st_size<=2**20,'Small closed unit array '+str(i)+name)
            a=np.load(ap,allow_pickle=False);expected=(i-75+1,3,3);fp={'shape':list(a.shape),'dtype':str(a.dtype),'bytes':a.nbytes,'sha256':hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}
            require(a.dtype==np.float64 and a.shape==expected and np.isfinite(a).all()and fp==info['payload'],'Actual finite file/payload/shape '+str(i)+name)
            arraychecks.append({'unit':i,'array':name,'path':str(ap),'file_sha256':sha(ap),'payload_sha256':fp['sha256'],'shape':list(a.shape),'dtype':str(a.dtype),'payload_bytes':a.nbytes,'all_finite':True})
        allowed.append({'path':str(path.resolve()),'sha256':sha(path),'producer_proof_sha256':PROOF_SHA});last.append({'index':i,'unit_name':rec['unit_name'],'completed_utc':rec['completed_utc'],'record_sha256':sha(path)})
    require(sorted(indices)==list(range(150))and len(allowed)==9 and len(arraychecks)==18,'150complete markers;141healthy plus9specific complete units')
    require(progress['incremental_before']['total_units_completed']==141,'Actual phase016 started from qualified141 prefix')
    require(capture['status']=='RUNNING_CAPTURE_ONLY'and capture['partial_hessian']is None and not capture['H1AO_atoms']and not capture['response_blocks'],'No completed RKS partial/H1/response allowed for reuse')
    missing=['result.json','hessian_hartree_per_bohr2.npy','hessian_used_for_analysis.npy','normal_modes.npy','frequencies.csv']
    require(all(not(FOLDER/p).exists()for p in missing),'No complete Hessian/frequency result exists')
    require('pyscf'not in sys.modules,'No quantum engine imported')
    for relative,meta in INPUTS.items():require(sha(ROOT/relative)==meta['sha256'],'Input byte stability '+relative)
    checks_count=len(CHECKS)
    report={'schema':'research4-failed-memory-phase-independent-last-nine-audit-v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
      'status':'PASS_NINE_COMPLETE_INTERMEDIATE_UNITS_FOR_ROOT_FORENSIC_REVIEW','checks_count':checks_count,'checks':CHECKS,
      'parent_phase':PHASE,'parent_row_preserved':row,'parent_final_status_remains':'PARTIAL_OR_FAILED','known_stop_cause':'Supervisor cgroup12GiB threshold triggered identity-checked groupSIGKILL; actual wait4 exit-9.',
      'cgroup_observed_bytes':row['memory_stop_cgroup']['current_bytes'],'cgroup_guard_bytes':row['cgroup_kill_bytes'],'payload_RSS_observed_bytes':row['memory_stop_snapshot']['rss_bytes'],
      'interference_attribution_scope':'Root reports Gitadd began15:41:06Z; guard snapshot confirms aggregate cgroup exceeded12GiB with payloadRSS below10GiB. This audit does not independently partition file-cache vs other-process memory or claim a kernelOOM.',
      'actual_parent_CPU_seconds':row['actual_cpu_seconds'],'closed_total_CPU_seconds':ledger['consumed_cpu_seconds'],'remaining_CPU_seconds':ledger['remaining_quantum_cpu_seconds'],
      'unchanged_cap_CPU_seconds':ledger['quantum_cpu_cap_seconds'],'unchanged_deadline':ledger['run_deadline_at_utc'],
      'healthy_prefix_units':141,'completed_marker_units':150,'last_nine_units':last,'verified_small_arrays':arraychecks,'prior_healthy_unit_producers':dict(prior),
      'header_sha256':sha(CACHE/'header.json'),'cache_binding_sha256':header['binding_sha256'],'snapshot_manifest_sha256':SNAPSHOT_SHA,'worker_sha256':WORKER_SHA,
      'source_MO_grid_geometry_binding_verified':True,'MF_final_stability_status':'NOT_VERIFIED_PARENT_KILLED',
      'latched_progress_warning':'Progress native-call counters and stable_MF_state_unchanged=false are stale startup capture, not a final-state check or evidence of a specific last C call. No final operator restoration is claimed.',
      'allowed_records_for_explicit_root_review':allowed,'root_approval_issued_by_auditor':False,
      'recommendation':'Root may explicitly hash-release only these nine completed E1/EJ units. Preserve failed parent flags and charge all actualCPU. Re-run the global RKS/XC/H1/response and harmonic analysis under the same verified source; do not release a full Hessian from this failed phase.',
      'schedule_recommendation':'Complete large Git/archive/dataset I/O first. Use strict time separation from QM/training. Before resume inspect cgroup current/anonymous/file-cache headroom; release only completed-file cache via reviewed controls. Do not alter cap/deadline.',
      'complete_Hessian_released':False,'Spinosin_frequencies_obtained':False,'IR_intensities_status':'NOT_COMPUTED','full_Hessian_artifacts_absent':missing,
      'quantum_evaluations':0,'SCF_gradient_Hessian_or_integral_calls':0,'PySCF_imported':False,'HDF_reads':0,'only_closed_last9_NPY_arrays_read':18,
      'Git_operations':0,'old_files_or_ledger_modified':False,'audit_source_sha256':sha(__file__),'input_files':INPUTS}
    (OUT/'closed_ledger_observed.json').write_bytes(ledgerbytes)
    (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    template={'schema':'research4-explicit-root-rhf-unit-forensic-release-v1','explicit_root_forensic_approval':False,
      'snapshot_manifest_sha256':SNAPSHOT_SHA,'hessian_worker_sha256':WORKER_SHA,
      'reason':'NOT_APPROVED_TEMPLATE: explicit root review/signature required; nine complete small RHF E1/EJ units predate known memory protection kill; failed phase and full Hessian remain unqualified.',
      'allowed_records':allowed,'independent_audit':{'path':str(OUT/'audit.json'),'sha256':sha(OUT/'audit.json')}}
    (OUT/'release_template_NOT_APPROVED.json').write_text(json.dumps(template,indent=2)+'\n')
    (OUT/'README.md').write_text('Independent audit PASS for nine specific complete small RHF units141–149 (rows66–74),18 finite float64 E1/EJ arrays, file/payload SHA and exact source MF/header bindings. These markers were complete15:04:03–15:18:10Z, before the15:41:12Z cgroup guardSIGKILL. Parent remains failed exit-9, memory_terminated/hard_killed; actualCPU8627.573778s is fully charged.141 earlier units have healthy producers. Root may explicitly release only the nine immutable units; this auditor has not approved release. No full RKS partial/H1/response/Hessian/frequencies are released or claimed. Source-grid/MO/geometry metadata match; final MF/operator restoration was not verified after kill. Large I/O must finish before a strictly separate QM resume with original budget/deadline and measured cgroup headroom. No QM/PySCF/HDF/Git or historical modification occurred.\n')
    files=[p for p in OUT.iterdir()if p.is_file()and p.name!='index.json'];index={'status':'SEALED_INDEPENDENT_LAST_NINE_FORENSIC_AUDIT','files':[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size}for p in sorted(files)]}
    (OUT/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'units':len(allowed),'arrays':len(arraychecks),'earliest':min(x['completed_utc']for x in last),'latest':max(x['completed_utc']for x in last),'audit_sha256':sha(OUT/'audit.json'),'index_sha256':sha(OUT/'index.json')}))

if __name__=='__main__':main()
