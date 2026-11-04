"""Root-authorized read-only forensic inspection; no quantum-engine import.

This does NOT issue approval or relabel a failed supervised phase healthy.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, sys
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
BASE = ROOT / 'reports/pyscf-sdf-continuation-20261103-v1'
PHASE = 'validation-ethanol-segment-002'
INPUTS = {}

def require(ok, message):
    if not ok: raise ValueError(message)

def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def record(p, expected=None):
    p = Path(p).resolve(); require(p.is_file() and not p.is_symlink(), 'Source unavailable')
    digest = sha(p); require(expected is None or digest == expected, 'SHA mismatch: '+str(p))
    INPUTS[str(p.relative_to(ROOT))] = {'sha256':digest,'bytes':p.stat().st_size}
    return p

def read(p, expected=None): return json.loads(record(p,expected).read_text())

def main():
    require(not (OUT/'audit.json').exists(), 'Refuse overwrite sealed audit')
    ledger = read(BASE/'cpu_budget.json'); rows=[r for r in ledger['phases'] if r['phase']==PHASE]
    require(len(rows)==1, 'Ambiguous parent'); row=rows[0]
    require(row['status']=='PARTIAL_OR_FAILED' and row['exit_code']==0, 'Unexpected terminal status')
    require(row['resource_stop_reason']=='MONITOR_FAILED_CLOSED' and row['monitor_failed'] is True
            and row['hard_killed'] is True, 'Failure classification changed')
    require(all(row[k] is False for k in ['wall_terminated','cpu_terminated','memory_terminated']), 'Resource stop')
    require(row['identity_checked_signal_attempts'] and all(x['sent'] is False for x in row['identity_checked_signal_attempts']), 'Signal was sent')
    proof=read(row['supervisor_proof_path'],row['supervisor_proof_sha256'])
    require(proof['child_pid']==row['pid'] and proof['phase']==PHASE,'Parent proof mismatch')
    record(row['log'],row['log_sha256'])
    result=read(BASE/PHASE/'result.json')
    require(result['resource_supervision']['proof']==proof,'Result proof mismatch')
    require(result['status']=='CHECKPOINT_SEGMENT_COMPLETE' and result['complete_global_Hessian'] is False,'Not a completed checkpoint-only segment')
    require(result['stable_MF_state_unchanged'] is True and result['no_new_SCF_performed'] is True,'MF stability not demonstrated')
    native=result['native_direct_driver_control']
    require(native['native_calls_entered']==native['native_calls_returned'] and native['restoration_failures']==0
            and native['original_driver_restored'] is True,'Native restoration incomplete')
    source=result['source_worker']; record(source['path'],source['sha256'])
    snapshot=Path(result['snapshot_directory'])/'manifest.json'
    read(snapshot,result['snapshot_manifest_sha256'])
    cache=Path(result['incremental_after']['cache_dir']); header=read(cache/'header.json')
    require(hashlib.sha256(canonical(header['binding'])).hexdigest()==header['binding_sha256'],'Binding digest differs')
    require(header['binding_sha256']==result['incremental_after']['binding_sha256'],'Result cache binding differs')
    allrecords=sorted((cache/'units').glob('*/attempt-*/complete.json'))
    indices=[]; allowed=[]; payloads=[]
    for p in allrecords:
        rec=read(p); indices.append(rec['index'])
        if rec['index'] not in range(5,10): continue
        require(rec['header_sha256']==sha(cache/'header.json') and rec['binding_sha256']==header['binding_sha256'],'Unit header mismatch')
        ev=rec['execution_evidence']
        require(ev['source_worker']==source and ev['resource_supervision']==result['resource_supervision']
                and Path(ev['output_directory'])==BASE/PHASE and ev['snapshot_manifest_sha256']==result['snapshot_manifest_sha256'],'Unit producer mismatch')
        require(rec['completed_utc']<=result['completed_at_utc']<=row['identity_checked_signal_attempts'][0]['utc']<row['ended_at_utc'],'Completion/monitor timeline mismatch')
        for name,entry in rec['arrays'].items():
            ap=record(p.parent/entry['file'],entry['file_sha256']); a=np.load(ap,allow_pickle=False);fp=entry['payload']
            require(a.dtype==np.float64 and np.isfinite(a).all() and list(a.shape)==fp['shape']
                    and str(a.dtype)==fp['dtype'] and a.nbytes==fp['bytes']
                    and hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()==fp['sha256'],'Saved array fingerprint differs')
            payloads.append({'unit_index':rec['index'],'array':name,'path':str(ap),'shape':list(a.shape),'sha256':sha(ap),'all_finite':True})
        allowed.append({'path':str(p.resolve()),'sha256':sha(p),'producer_proof_sha256':row['supervisor_proof_sha256']})
    require(indices==list(range(10)) and len(allowed)==5,'Not contiguous complete prefix0..9')
    require('pyscf' not in sys.modules,'Unexpected quantum engine import')
    # Full ledger may later append phases. Freeze its bytes for this observation.
    ledgerbytes=(BASE/'cpu_budget.json').read_bytes()
    require(hashlib.sha256(ledgerbytes).hexdigest()==INPUTS[str((BASE/'cpu_budget.json').relative_to(ROOT))]['sha256'],'Ledger changed during forensic observation')
    (OUT/'ledger_observed.json').write_bytes(ledgerbytes)
    report={'schema':'research4-independent-failed-unit-forensic-audit-v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'status':'COMPLETE_SAVED_UNITS_VERIFIED_NOT_HEALTHY_PARENT','parent_phase':PHASE,'parent_row':row,
            'parent_status_remains':'PARTIAL_OR_FAILED','monitor_failure_remains':True,'root_approval_issued_by_this_audit':False,
            'recommendation':'Eligible for explicit root hash-bound forensic release of these five completed units only; retain failed parent flags. Require final full Hessian and all frequencies to match the healthy stock002 reference before any launch gate.',
            'snapshot_manifest_sha256':result['snapshot_manifest_sha256'],'hessian_worker_sha256':source['sha256'],
            'complete_prefix_indices':indices,'allowed_records_for_root_review':allowed,'array_payloads':payloads,
            'native_calls_all_returned':True,'state_unchanged':True,'no_signal_sent':True,
            'actual_parent_wait4_CPU_seconds':row['actual_cpu_seconds'],'scientific_scope':'Five complete RHF helper units; no full Hessian or frequencies in this segment.',
            'quantum_evaluations':0,'pyscf_imported':False,'audit_source_sha256':sha(__file__),'input_files':INPUTS}
    (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'README.md').write_text('Five complete RHF units (indices5–9; six finite saved arrays) verified by file and payload SHA, exact MF snapshot/source/proof bindings and completion timeline. The closed supervisor remains PARTIAL_OR_FAILED / MONITOR_FAILED_CLOSED / hard_killed=true, despite actual exit0 and no identity-checked signal sent. This audit does not issue root approval or relabel the parent healthy. Root may explicitly hash-release only these five records, provided the eventual full Hessian and spectrum match healthy stock002. No quantum calculation was run.\n')
    files=[p for p in OUT.iterdir() if p.is_file() and p.name!='index.json']
    index={'status':'SEALED_FAILED_PARENT_SAVED_UNIT_AUDIT','files':[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size}for p in sorted(files)]}
    (OUT/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'records':len(allowed),'payloads':len(payloads),'audit_sha256':sha(OUT/'audit.json'),'index_sha256':sha(OUT/'index.json')}))

if __name__=='__main__':main()
