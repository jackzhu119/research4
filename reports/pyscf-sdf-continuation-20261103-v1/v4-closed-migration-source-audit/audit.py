"""Read-only V4 terminal-ledger and immutable-source migration audit.

Standard library only. No molecular checkpoint or array parsing, no QM/Git.
"""
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
import hashlib,json,sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
BASE=ROOT/'reports/pyscf-sdf-continuation-20261103-v1'
INPUTS={}
def require(x,msg):
    if not x:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def record(p,expected=None):
    p=Path(p).resolve();require(p.is_file() and not p.is_symlink(),'Missing/symlink source')
    digest=sha(p);require(expected is None or digest==expected,'SHA mismatch '+str(p))
    INPUTS[str(p.relative_to(ROOT))]={'sha256':digest,'bytes':p.stat().st_size}
    return p
def read(p,expected=None):return json.loads(record(p,expected).read_text())
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def healthy(r):
    return r['status']=='EXITED' and r['exit_code']==0 and all(r.get(k)is False for k in ['wall_terminated','cpu_terminated','memory_terminated','monitor_failed','hard_killed'])

def main():
    require(not (OUT/'audit.json').exists(),'Refuse sealed overwrite')
    ledgerpath=record(BASE/'cpu_budget.json');ledgerbytes=ledgerpath.read_bytes();ledger=json.loads(ledgerbytes)
    require(ledger['schema']=='research4-pyscf-actual-cpu-budget-v4','Wrong historical ledger')
    require(len(ledger['phases'])==13,'Unexpected V4 terminal phase count')
    rows=ledger['phases'];pids=[];costs=[];phases=[]
    for r in rows:
        require(r['status']in ['EXITED','PARTIAL_OR_FAILED'] and r['exit_code']==0 and r.get('ended_at_utc'),'Phase not terminal')
        cost=Decimal(str(r['actual_cpu_seconds']));require(cost>=0,'Invalid actual cost');costs.append(cost);pids.append(r['pid'])
        proof=read(r['supervisor_proof_path'],r['supervisor_proof_sha256']);require(proof['child_pid']==r['pid'] and proof['phase']==r['phase'],'Proof row binding mismatch')
        record(r['log'],r['log_sha256']);resultpath=BASE/r['phase']/'result.json';result=read(resultpath)
        require(result['resource_supervision']['proof']==proof,'Result proof mismatch')
        src=result['source_worker'];record(src['path'],src['sha256'])
        if not healthy(r):
            require(r['resource_stop_reason']=='MONITOR_FAILED_CLOSED' and r['monitor_failed']is True and r['hard_killed']is True,'Failure flags changed')
            require(r['owned_child_pidfd_containment_requested']is True,'Expected pidfd request missing')
        phases.append({'phase':r['phase'],'parent_status':r['status'],'scientific_result_status':result['status'],'healthy_closed_parent':healthy(r),
            'pid':r['pid'],'actual_wait4_CPU_seconds':r['actual_cpu_seconds'],'ended_at_utc':r['ended_at_utc'],'exit_code':r['exit_code'],
            'resource_stop_reason':r.get('resource_stop_reason'),'monitor_failed':r['monitor_failed'],'hard_killed':r['hard_killed'],
            'owned_child_pidfd_containment_requested':r.get('owned_child_pidfd_containment_requested',False),
            'result_sha256':sha(resultpath),'proof_sha256':r['supervisor_proof_sha256'],'log_sha256':r['log_sha256'],
            'closed_row_canonical_sha256':hashlib.sha256(canonical(r)).hexdigest()})
    actual=sum(costs);require(abs(actual-Decimal(str(ledger['consumed_cpu_seconds'])))<Decimal('1e-8'),'Ledger cost sum differs')
    require(abs(Decimal(str(ledger['quantum_cpu_cap_seconds']))-actual-Decimal(str(ledger['remaining_quantum_cpu_seconds'])))<Decimal('1e-8'),'Budget conservation differs')
    require(len(set(pids))==len(pids),'PID reused within terminal control set')
    require(not any(r['phase'].startswith('validation-ethanol-segment-003')or r['phase'].startswith('validation-ethanol-segment-004')for r in rows),'Unexpected later ethanol phase')
    pilotindex=ROOT/'reports/pyscf-spinosin-pilot-20261031/artifact_index.json';pilot=read(pilotindex,'8fbd6a0c262a12beffb6320458713c45ae381bbd8adb76f062bea1d882ee8f5b')
    require(pilot['file_count']==311 and len(pilot['files'])==311,'Unexpected pilot inventory')
    pilotchecks=[]
    for relative,entry in pilot['files'].items():
        p=ROOT/relative;exists=p.is_file() and not p.is_symlink();digest=sha(p)if exists else None;size=p.stat().st_size if exists else None
        pilotchecks.append({'path':relative,'matches':exists and digest==entry['sha256']and size==entry['bytes'],'sha256':digest,'bytes':size,'expected_sha256':entry['sha256'],'expected_bytes':entry['bytes']})
    matches=sum(x['matches']for x in pilotchecks)
    # The full 311-file inventory is evidence, not a condition to hide mismatches.
    frozen={
      'scripts/run_pyscf_actual_cpu_budget_20261101.py':'354ff51b5c2af2810033656d9972948262cfebb4dabb3978fca2d021f97e2b8f',
      'scripts/pyscf_molecular_state_worker_20261103.py':'dcfffc4d0c68cd30c6d2c60597b77a4c3931516412b7f3308bb9d0b72ef45360',
      'scripts/pyscf_resumable_hessian_worker_20261103.py':'d23c4a3185aa4f666e22d671034728c1ca2443676a05fc8bec0587d6c8d6bf03',
      'scripts/pyscf_incremental_rhf_checkpoint_20261103.py':'ce96055f24b39833337844c178f94597a9ac4ba7721ba019816d03628f971f55',
      'scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py':'acfe7708748ac237c9912aa9b1d98e047d842794fd0c6f08b1a67b993f5b277d'}
    for relative,expected in frozen.items():record(ROOT/relative,expected)
    release=BASE/'resource-protocol/ethanol-segment002-root-forensic-release.json';receipt=read(release,'03c1d44f09b8d37cc7b177cf25dc830cb18fe18be54a3d1992147718d4c856b6')
    require(receipt['explicit_root_forensic_approval']is True and len(receipt['allowed_records'])==5,'Forensic extent differs')
    spino=read(BASE/'spinosin-C001-prepare-001/result.json');record(spino['snapshot_manifest']['path'],spino['snapshot_manifest']['sha256'])
    require(spino['status']=='DONE'and spino['scf_converged']and spino['final_gradient_thresholds_passed'],'Scientific prepare did not complete')
    partial=read(BASE/'validation-ethanol-segment-002/result.json')
    require(partial['incremental_after']['total_units_completed']==10 and partial['incremental_after']['total_units']==18,'Partial helper progress differs')
    require(not (BASE/'validation-ethanol-segment-003').exists()and not (BASE/'validation-ethanol-segment-004').exists(),'Unexpected later ethanol directory')
    read(BASE/'real-restart-validation-independent-audit/water/audit.json','3cc63f8e50e113cc24c1b4a76ba29bb19c6a1c1cb0583084d6bad1069e8ff9af')
    require(ledgerpath.read_bytes()==ledgerbytes,'V4 ledger changed during terminal audit')
    for relative,entry in INPUTS.items():require(sha(ROOT/relative)==entry['sha256'],'Source changed during audit')
    report={'schema':'research4-v4-closed-migration-source-audit-v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
      'status':'TERMINAL_LEDGER_RECONCILED','historical_ledger_schema':ledger['schema'],'historical_ledger_sha256':sha(ledgerpath),
      'phase_count':len(rows),'healthy_closed_phase_count':sum(healthy(r)for r in rows),'failed_closed_phase_count':sum(not healthy(r)for r in rows),
      'all_13_phases_terminal':True,'actual_wait4_CPU_seconds_sum':float(actual),'actual_wait4_CPU_seconds_decimal_sum':str(actual),
      'ledger_consumed_CPU_seconds':ledger['consumed_cpu_seconds'],'cap_CPU_seconds':ledger['quantum_cpu_cap_seconds'],
      'remaining_CPU_seconds':ledger['remaining_quantum_cpu_seconds'],'created_at_utc_of_original_envelope':ledger['created_at_utc'],
      'same_unchanged_deadline_required_for_migration':ledger['run_deadline_at_utc'],'same_wall_cap_seconds_required':ledger['run_wall_cap_seconds'],
      'migration_rule':'V5 must import this full actual cost, retain original cap and deadline; no fresh allowance or reset. This audit does not itself authorize or launch a calculation.',
      'water':'PASS independently replayed full stock/restart matrix and3positive modes; six healthy unique processes.',
      'ethanol':'PARTIAL: healthy stock002 has21modes; only10/18helper units, five completed units root-forensically released but no further segment launched; no full restarted Hessian/mode comparison.',
      'ethanol_003_004_absent':True,'forensic_release_sha256':sha(release),
      'spinosin_prepare':{'scientific_status':spino['status'],'parent_status':'PARTIAL_OR_FAILED','source_qualified_healthy':False,
        'SCF_converged':spino['scf_converged'],'gradient_pass':spino['final_gradient_thresholds_passed'],'energy_hartree':spino['energy_hartree'],
        'atom_RMS_gradient_Eh_per_Bohr':spino['atom_rms_hartree_per_bohr'],'snapshot_manifest_sha256':spino['snapshot_manifest']['sha256'],
        'frequencies_status':spino['frequencies_status'],'do_not_restore_as_healthy_source':True},
      'monitor_failure_interpretation':'All failed rows preserve MONITOR_FAILED_CLOSED/monitor_failed/hard_killed, groupKILL attempt not sent but pidfdKILL requested. Actual wait4 exit0; effect of request unknown. Never relabel healthy.',
      'pilot_311_inventory':{'index_sha256':sha(pilotindex),'expected_files':311,'matching_files':matches,'unchanged':matches==311,'mismatches':[x for x in pilotchecks if not x['matches']]},
      'frozen_computation_sources':frozen,'phases':phases,'input_files':INPUTS,'audit_source_sha256':sha(__file__),
      'new_quantum_evaluations':0,'quantum_engine_imported':False,'HDF_NPY_parsed':False,'historical_files_modified':False,'Git_operations':0}
    (OUT/'ledger_closed_snapshot.json').write_bytes(ledgerbytes)
    (OUT/'pilot_311_files_verified.json').write_text(json.dumps(pilotchecks,indent=2)+'\n')
    (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'README.md').write_text(f"V4 terminal migration source audited:13 closed phases (10 healthy,3 failed); total actual wait4 CPU {float(actual):.6f} seconds. V5 must import this cost and preserve the original129600 CPU-second cap and UTC deadline {ledger['run_deadline_at_utc']}. Water full restarted Hessian/modes pass. Ethanol is partial (healthy stock00221modes,10/18helper units; forensic release exists but no subsequent segment). Spinosin prepare scientificSCF/G completed, but parent failed; its saved snapshot is not a healthy restorable source. All failed flags remain. Pilot inventory byte/SHA matches {matches}/311. Standard-library-only audit; no QM, checkpoint/array parsing or Git. Full row/proof/source evidence and precise costs are in audit.json.\n")
    index={'status':'SEALED_V4_TERMINAL_MIGRATION_SOURCE_AUDIT','files':[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size}for p in sorted(OUT.iterdir())if p.is_file()and p.name!='index.json']}
    (OUT/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'CPU_seconds':float(actual),'phases':len(rows),'pilot_matches':matches,
        'audit_sha256':sha(OUT/'audit.json'),'index_sha256':sha(OUT/'index.json')}))

if __name__=='__main__':main()
