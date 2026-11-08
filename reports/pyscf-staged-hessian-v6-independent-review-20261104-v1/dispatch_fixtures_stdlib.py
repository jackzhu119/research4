"""Synthetic JSON-only dispatch fixtures; no actual phase/cache/proof is read."""
from pathlib import Path
import importlib.util,hashlib,json,sys,tempfile,types
from unittest.mock import patch

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
SOURCE=ROOT/'scripts/pyscf_staged_hessian_worker_20261105_v6.py'
EXPECTED='1e160efbeca8f986c329202ecafe1b10063373855fd9198e1ab6e36498d4d9ed'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert sha(SOURCE)==EXPECTED and not(OUT/'dispatch_results.json').exists()
    spec=importlib.util.spec_from_file_location('independent_V6_dispatch_fixture',SOURCE);w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
    results=[]
    with tempfile.TemporaryDirectory(prefix='synthetic-json-',dir=OUT)as tmp:
        fake_root=Path(tmp);cache=fake_root/'outputs/synthetic-cache';units=cache/'units';units.mkdir(parents=True)
        oldcalls=[];newcalls=[]
        def oldcheck(evidence,**kw):oldcalls.append((evidence,kw));return {'fixture':'OLD_HEALTHY_STDLIB_ONLY'}
        def newcheck(evidence,**kw):newcalls.append((evidence,kw));return {'fixture':'NEW_HEALTHY_STDLIB_ONLY'}
        v5=types.SimpleNamespace(EXECUTION_SCHEMA='research4-rhf-unit-producer-v1',verify_closed_execution=oldcheck)
        guard=types.SimpleNamespace(verify_closed_phase=newcheck)
        for i in range(150):
            p=units/f'{i:06d}-fixture'/'attempt-000001';p.mkdir(parents=True)
            ev={'schema':v5.EXECUTION_SCHEMA if i<149 else w.RHF_EVIDENCE_SCHEMA,
                'source_worker':{'sha256':w.V5_SHA if i<149 else EXPECTED},
                'phase':'hessian-segment'if i<149 else'rks-partial'}
            if i==149:ev['stage']='rks-partial'
            (p/'complete.json').write_text(json.dumps({'index':i,'execution_evidence':ev,'fixture':'JSON_ONLY_NOT_SCIENTIFIC_UNIT'}))
        last=units/'000149-fixture/attempt-000001/complete.json';saved=last.read_bytes()
        with patch.object(w,'ROOT',fake_root):
            qualified=w.verify_rhf_producers(cache,'s'*64,guard,v5)
            assert len(qualified)==150 and len(oldcalls)==149 and len(newcalls)==1
            assert all(x['qualification']=='CLOSED_HEALTHY_V5_PRODUCER'for x in qualified[:149])
            assert qualified[-1]['qualification']=='CLOSED_HEALTHY_V6_FULL_RKS_PARTIAL_PRODUCER'
            assert newcalls[0][1]['expected_worker_sha256']==EXPECTED and newcalls[0][1]['allowed_result_statuses']==('STAGE_COMPLETE',)
            results.append({'case':'149V5_plus1V6_routes_genuine_verifiers','passed':True,'old_verifier_calls':149,'V6_verifier_calls':1})
            def reject(*a,**kw):raise ValueError('SYNTHETIC_FAILED_PARENT')
            with patch.object(guard,'verify_closed_phase',side_effect=reject):
                try:w.verify_rhf_producers(cache,'s'*64,guard,v5)
                except ValueError as exc:assert str(exc)=='SYNTHETIC_FAILED_PARENT'
                else:raise AssertionError('Failed V6 parent was accepted')
            results.append({'case':'failed_new_unit_parent_rejected_no_forensic_fallback','passed':True})
            bad=json.loads(saved);bad['execution_evidence']['source_worker']['sha256']='0'*64;last.write_text(json.dumps(bad))
            try:w.verify_rhf_producers(cache,'s'*64,guard,v5)
            except ValueError as exc:assert 'schema/source'in str(exc)
            else:raise AssertionError('Unknown V6 source was accepted')
            results.append({'case':'unfrozen_new_unit_source_rejected','passed':True})
            bad=json.loads(saved);bad['index']=150;last.write_text(json.dumps(bad))
            try:w.verify_rhf_producers(cache,'s'*64,guard,v5)
            except ValueError as exc:assert 'continuous'in str(exc)
            else:raise AssertionError('Noncontinuous prefix accepted')
            results.append({'case':'gapped_149_to150_prefix_rejected','passed':True})
    forbidden=[n for n in sys.modules if n.split('.')[0]in {'numpy','scipy','pyscf','h5py'}];assert not forbidden
    report={'status':'PASS_FOUR_STDLIB_SYNTHETIC_DISPATCH_FIXTURES','cases':results,'source_sha256':EXPECTED,
      'actual_QM_or_numeric_execution':'NOT_RUN','real_array_reads':0,'actual_phase_or_ledger_or_proof_reads':0,
      'mock_producer_qualification_is_not_real_parent_validation':True,'forbidden_imports':forbidden,'fixture_source_sha256':sha(__file__)}
    (OUT/'dispatch_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'results_sha256':sha(OUT/'dispatch_results.json')}))

if __name__=='__main__':main()
