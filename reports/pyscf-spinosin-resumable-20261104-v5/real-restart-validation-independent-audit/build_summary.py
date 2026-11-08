"""Seal source-backed V5 numeric validation summary without a quantum engine."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,csv

OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    if (OUT/'validation_summary.json').exists():raise ValueError('Summary already sealed')
    rows=[];sources=[]
    expected={'water':'5ff9eb7ac56fa1634e2d2f1dc979900efa5b7696594a904121a4e52cef829caa',
              'ethanol':'e30b330ba053cee68e7b5d01aa42ae3a5913252d44d6e4f70f82bc0a8aeef5b9'}
    for identity,digest in expected.items():
        p=OUT/identity/'audit.json';assert sha(p)==digest;d=json.loads(p.read_text())
        assert d['status']=='PASS_INDEPENDENT_NUMPY_REPLAY' and d['forensic_units_reused']==0
        assert d['proof_schema']=='research4-pyscf-actual-cpu-supervisor-proof-v5'
        assert d['all_payload_PIDs_unique'] and d['same_exact_MF_snapshot_all_stages']
        assert all(r['status']=='EXITED' and r['exit_code']==0 and all(r[k]is False for k in ['monitor_failed','hard_killed','cpu_terminated','memory_terminated','wall_terminated'])for r in d['parent_rows'])
        sources.append({'path':str(p),'sha256':digest})
        rows.append({'molecule':identity,'raw_and_sym_Hessian_elements_each':d['all_raw_Hessian_elements_compared'],
           'positive_modes_each':d['stock']['modes'],'all_frequencies_positive':d['stock']['all_positive']and d['segmented']['all_positive'],
           'max_raw_difference_Eh_per_Bohr2':d['max_abs_differences']['raw'],'max_sym_difference_Eh_per_Bohr2':d['max_abs_differences']['sym'],
           'max_frequency_difference_cm1':d['max_abs_differences']['frequency'],'max_frequency_difference_THz':d['max_abs_differences']['thz'],
           'max_independent_frequency_replay_difference_cm1':max(d[t]['max_independent_frequency_difference_cm1']for t in ['stock','segmented']),
           'minimum_stock_segment_mode_absolute_overlap':d['minimum_stock_segment_mode_absolute_overlap'],
           'restore_gradient_difference_Eh_per_Bohr':d['restore_gradient_difference_Eh_per_Bohr'],
           'payload_process_count':len(d['independent_payload_PIDs']),'actual_wait4_CPU_seconds':d['actual_wait4_CPU_seconds_for_selected_lineage'],
           'snapshot_manifest_sha256':d['snapshot_manifest_sha256']})
    report={'schema':'research4-v5-independent-real-restart-validation-summary-v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
       'status':'PASS_BOTH_CLOSED_REAL_RESTART_NUMPY_VALIDATIONS','source_audits':sources,'comparisons':rows,
       'current_guard_sha256':'33f1dc3db5280f16c9898c561e345143614730dc052b171b4f11fbbb079cdeee',
       'current_state_worker_sha256':'be4727b76d59a0b0dd3defba249ef833ee8ef0ea89473e390dcef389728415c8',
       'current_consumer_sha256':'b40188f747c01e9efa97672af6eedfd281a9e9697d20bc7d57f125bb131fa190',
       'NumPy_audit_source_sha256':sha(OUT/'replay_restarts_v5.py'),'builder_sha256':sha(__file__),
       'forensic_units_reused':0,'no_active_Spinosin_array_or_checkpoint_reads':True,'new_quantum_evaluations':0,
       'IR_intensities_status':'NOT_COMPUTED','measured_CPHF_residual_norm':'NOT_COMPUTED',
       'interpretation':'Software numerical and restart controls pass on water/ethanol only. No Spinosin frequencies, physical assignment, large-molecule speedup, or classification accuracy improvement is established.',
       'reaping_grace_interpretation':'V5 proof contains a bounded reaping-grace configuration; successful controls do not by themselves prove that the race/grace branch was actually entered. Actual transition fields require separate metadata audit.'}
    (OUT/'validation_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    with (OUT/'validation_summary.csv').open('w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    index={'status':'SEALED_V5_INDEPENDENT_NUMERIC_SUMMARY','files':[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size}for p in [OUT/'validation_summary.json',OUT/'validation_summary.csv',Path(__file__)]]}
    (OUT/'summary_index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps({'summary_sha256':sha(OUT/'validation_summary.json'),'summary_index_sha256':sha(OUT/'summary_index.json')}))

if __name__=='__main__':main()
