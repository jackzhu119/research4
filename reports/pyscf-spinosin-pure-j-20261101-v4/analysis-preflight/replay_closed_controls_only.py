"""Finite saved-matrix postprocessing fixture; no quantum evaluations.

Only the two explicitly closed software controls from the immutable gate-prefix
snapshot are read. Never read the current Spinosin worker directory or ledger.
"""
from pathlib import Path
import importlib.util
import json
import time

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'scripts/analyze_pyscf_pure_j_results_20261101.py'
spec=importlib.util.spec_from_file_location('closed_pure_j_control_fixture',SOURCE)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
base=ROOT/'reports/pyscf-spinosin-pure-j-20261101-v4'
prefix=base/'gate-prefix-snapshots/both-validations-closed-ledger.json'
ledger=audit.read(prefix)
assert len(ledger['phases'])==2 and all(audit.healthy_parent(row) for row in ledger['phases'])
start=time.monotonic();records=[]
for row in ledger['phases']:
    directory=audit.resolved(audit.option(row['command'],'--output'))
    assert directory.name in ('water-pure-j-equivalence-001','ethanol-pure-j-equivalence-001')
    result=audit.read(directory/'result.json');hashes={}
    proof=audit.audit_closed_proof(base,row,hashes);audit.assert_entry(result,hashes)
    replay,timings=audit.audit_control(directory,result,row,hashes)
    records.append({'identity':result['identity'],'archived_healthy_proof':proof['healthy_parent'],
        'saved_matrix_frequency_and_mode_audit':replay,'actual_original_case_timings':timings,
        'input_sha256':hashes})
print(json.dumps({'status':'PASS_CLOSED_CONTROLS_ONLY_SAVED_MATRIX_REPLAY','source_sha256':audit.sha(SOURCE),
    'closed_gate_prefix_ledger_sha256':audit.sha(prefix),'new_SCF_gradient_Hessian_evaluations':0,
    'active_Spinosin_files_read':False,'Mole_build_and_saved_matrix_diagonalization_only':True,
    'elapsed_wall_seconds':time.monotonic()-start,'process_CPU_seconds':time.process_time(),'controls':records},indent=2))
