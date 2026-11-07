"""Stdlib text/AST comparison only. No actual data/proof/ledger/allocation."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
OLD=ROOT/'scripts/pyscf_staged_hessian_worker_20261105_v6.py'
NEW=ROOT/'scripts/pyscf_staged_hessian_worker_20261105_v6b.py'
OLD_TEST=ROOT/'tests/test_pyscf_staged_hessian_worker_v6_stdlib.py'
NEW_TEST=ROOT/'tests/test_pyscf_staged_hessian_worker_v6b_stdlib.py'
THREAD_TEST=ROOT/'tests/test_pyscf_staged_hessian_v6b_thread_contract_stdlib.py'
OLD_GUARD='ae2cc7bc8ed4f3d6f39e13a842e635e5a26152c02aab52d6af25d91cc65f5b1b'
NEW_GUARD='b985d6d6e20f5e4a48387094461f19b133dae761338664e88836c62051f993a5'
EXPECTED={OLD:'1e160efbeca8f986c329202ecafe1b10063373855fd9198e1ab6e36498d4d9ed',
 NEW:'f13701964cd8a5fbda25a19ca27ec91b4a414f6b5cb49c14abacf6c82eaa448d',
 OLD_TEST:'70b117c7d85b65ffdf86e6158b7d49193e9ac556db8ac8039be9e1ac9e6b01e1',
 NEW_TEST:'142faf13112bb8d3b479d0b1d04a79ebe7d11c7f279e66349a75ffe8a6a11dc5',
 THREAD_TEST:'553b5080530d1782b5092815afc155c9e2f7948bcdcfdcfa2d061f334bef1359',
 ROOT/'scripts/run_pyscf_continuation_budget_20261105_v6b.py':NEW_GUARD,
 ROOT/'scripts/pyscf_incremental_rhf_checkpoint_20261103.py':'ce96055f24b39833337844c178f94597a9ac4ba7721ba019816d03628f971f55',
 ROOT/'scripts/pyscf_molecular_state_worker_20261104_v5.py':'be4727b76d59a0b0dd3defba249ef833ee8ef0ea89473e390dcef389728415c8'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def function(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
def normalize(text):return text.replace(NEW_GUARD,OLD_GUARD).replace('v6b','v6').replace('V6B','V6')
def main():
    assert not(OUT/'review.json').exists()
    for p,digest in EXPECTED.items():assert sha(p)==digest
    old,new=OLD.read_text(),NEW.read_text();back=normalize(new)
    assert back==old and ast.dump(ast.parse(back),include_attributes=False)==ast.dump(ast.parse(old),include_attributes=False)
    backtest=normalize(NEW_TEST.read_text());assert backtest==OLD_TEST.read_text()
    assert ast.dump(ast.parse(backtest),include_attributes=False)==ast.dump(ast.parse(OLD_TEST.read_text()),include_attributes=False)
    trees={p:ast.parse(p.read_text())for p in [NEW,ROOT/'scripts/pyscf_incremental_rhf_checkpoint_20261103.py',ROOT/'scripts/pyscf_molecular_state_worker_20261104_v5.py']}
    core=function(trees[ROOT/'scripts/pyscf_incremental_rhf_checkpoint_20261103.py'],'runtime_state_fingerprint')
    environment=function(trees[ROOT/'scripts/pyscf_molecular_state_worker_20261104_v5.py'],'environment_identity')
    for node in (core,environment):
        assert not any(isinstance(n,ast.Constant)and isinstance(n.value,str)and'thread'in n.value.lower()for n in ast.walk(node))
        assert not any(isinstance(n,ast.Attribute)and'thread'in n.attr.lower()for n in ast.walk(node))
    for name in ('make_scf','restore_snapshot'):
        node=function(trees[ROOT/'scripts/pyscf_molecular_state_worker_20261104_v5.py'],name)
        assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='num_threads'for n in ast.walk(node))
    execute=function(trees[NEW],'execute')
    physical=next(n.value for n in ast.walk(execute)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='physical'for t in n.targets))
    values={k.value:v for k,v in zip(physical.keys,physical.values)if isinstance(k,ast.Constant)}
    selected=values['native_threads'];assert isinstance(selected,ast.Attribute)and isinstance(selected.value,ast.Name)and selected.value.id=='args'and selected.attr=='threads'
    calls=[n for n in ast.walk(execute)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)]
    configured=next(n for n in calls if n.func.attr=='num_threads');restored=next(n for n in calls if n.func.attr=='restore_snapshot');assert configured.lineno<restored.lineno
    assert ast.dump(configured.args[0])==ast.dump(selected)
    log=(OUT/'stdlib_tests.log').read_text();assert'Ran 19 tests'in log and'\nOK\n'in log and'forbidden_numeric_or_quantum_imports []'in log
    forbidden=[k for k in sys.modules if k.split('.')[0]in {'numpy','scipy','pyscf','h5py'}];assert not forbidden
    report={'schema':'research4-independent-v6b-mechanical-preparation-review-v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
      'status':'MECHANICAL_AND_STDLIB_REVIEW_PASSED_NUMERICAL_NOT_RUN','blocking_worker_static_findings':[],
      'worker_reverse_text_equal':True,'worker_reverse_AST_equal':True,'copied17tests_reverse_text_and_AST_equal':True,
      'literal_replacements_only':{'lowercase_v6_to_v6b':old.count('v6'),'uppercase_V6_to_V6B':old.count('V6'),'guard_SHA_pin':old.count(OLD_GUARD)},
      'old_V6_frozen_worker_and_tests_unchanged':True,'source_files':{str(p.relative_to(ROOT)):{'sha256':digest,'bytes':p.stat().st_size}for p,digest in EXPECTED.items()},
      'independent_test_count':19,'test_scope':'17mechanical stdlib tests+2source-AST thread contract tests, no actual electronic structure or real provenance inputs.',
      'independent_test_log_sha256':sha(OUT/'stdlib_tests.log'),'tests_exit_code':0,'test_resources':{'AS_bytes':536870912,'CPU_affinity':[0],'peak_RSS_kib':22428},
      'single_thread_old149_contract':{'status':'STATIC_BINDING_COMPATIBLE_ONLY','old_snapshot_environment_and_RHF_runtime_bind_thread_count':False,
        'restore_snapshot_and_make_scf_explicitly_reset_runtime_threads':False,'new_stage_physical_native_threads_uses_args_threads':True,
        'lib_num_threads_selected_before_snapshot_restore':True,'all_new_stages_must_use_same_selected_threads':True,
        'old4thread_source_proof_relabelled':False,'source_MO_gauge_geometry_grid_screening_and_dependency_SHA_relaxed':False,
        'actual149unit_reuse_performed':False,'actual_numerical_equivalence_or_speed_claim':False},
      'caller_requirement':'Worker default remains4 by exact mechanical equivalence. For the proposed1thread execution, explicitly pass --threads1 in worker and matching parent/proof; mismatched proof/args fails closed before numeric imports.',
      'guard_scope':'New V6B guard is pinned by exact source SHA. This worker review does not create authorization/continuation claims or register/allocate a real ledger; guard authority needs its separate reviewed root approval.',
      'remaining_requirements':['ActualV6B small-molecule partial/H1/global numerical controls must be run only after activeV5 is closed and a valid exclusive allocation exists.','Compare full raw/sym Hessians, frequencies and modes under the selected thread policy; static compatibility alone proves neither speed nor floating-point equivalence.','Preserve genuineV5 proofs and all historical failed costs; use explicit V6B-compatible postprocessor/observer for any new schema.'],
      'real_QM_registration_allocation_and_execution':'NOT_RUN','actual_arrays_or_HDF_read':0,'actual_proof_or_active_ledger_read':0,
      'numeric_or_quantum_imports':forbidden,'Git_operations':0,'signals_sent':0,'old_files_modified':False,'audit_source_sha256':sha(__file__)}
    for p,digest in EXPECTED.items():assert sha(p)==digest
    (OUT/'review.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'README.md').write_text('No static worker blocker. FinalV6Bf1370196... is exactly the frozenV6 worker after only version-label andguard-path/SHA replacements; reverse wholetext/AST and copied17test text/AST match.19 independentpython-S stdlib/AST tests pass. A selected1thread stage is statically compatible with genuineold4thread frozenMO/grid/RHF source bindings, and newstages bind the selected thread count consistently. This is not actual149unit reuse, numerical equivalence, speed evidence, realregistration/allocation or execution. Worker default remains4; proposed1thread execution must explicitly pass--threads1 with matching parentproof. ActualV6B controls/QM stayNOT_RUN. No numeric libraries, realarrays/proofs/activeledger, Git, signal or oldfile modification occurred.\n')
    (OUT/'index.json').write_text(json.dumps({'status':'SEALED_V6B_MECHANICAL_STDLIB_REVIEW_NOT_NUMERICAL_GATE','files':[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size}for p in sorted(OUT.iterdir())if p.is_file()and p.name!='index.json']},indent=2)+'\n')
    print(json.dumps({'status':report['status'],'review_sha256':sha(OUT/'review.json'),'index_sha256':sha(OUT/'index.json')}))

if __name__=='__main__':main()
