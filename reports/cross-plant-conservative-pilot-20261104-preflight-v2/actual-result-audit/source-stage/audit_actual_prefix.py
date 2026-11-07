"""Read-only independent replay of the registered source-only 48-epoch prefix."""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
V1=ROOT/'reports/cross-plant-conservative-pilot-20261104-v1'
V2=ROOT/'reports/cross-plant-conservative-pilot-20261104-v2'
OLD=ROOT/'reports/cross-plant-deep-transfer-20261031-v1/neural'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def payload_sha(a):
    a=np.asarray(a);h=hashlib.sha256(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(np.ascontiguousarray(a).tobytes());return h.hexdigest()


def main():
    result={'status':'PARTIAL','scope':'SOURCE_ONLY_INDEPENDENT_PREFIX_CHECK','new_model_fits':0,'quantum_calls':0,'outer_model_array_files_opened':0,'outer_truth_read':False,'started_at_utc':datetime.now(timezone.utc).isoformat()}
    checked=[];inputs={}
    try:
        p=read(V2/'protocol.json');spec=p['budget_ablation'];idx=read(V1/'index.json')
        assert sha(V1/'index.json')==spec['prior_sealed_index_sha256']=='15ee1e7764c1bb98472d119867a949356fdfc2068374b9583de09c1099e0eee4'
        assert sha(V1/'source_epoch_lock.json')==spec['prior_source_epoch_lock_sha256']
        assert sha(V1/'protocol.json')==spec['prior_protocol_sha256']
        assert sha(ROOT/'scripts/run_conservative_cross_plant_pilot_20261104.py')==spec['prior_training_source_sha256']=='a0aed48133444a8b77abf79de3aca324efed5239914769ad5d033108c8af23c4'
        records={r['path']:r for r in idx['files']};prep=read(OLD/'preparation.json');max_diff=0.
        for c in prep['contexts']:
            if c['stage']!='inner':continue
            for seed in (42,43,44):
                rel=Path('source_models')/c['split_id']/f'seed{seed}';a,b=V1/rel,V2/rel
                for name in ('training.json','predictions.npz','target-scaler.npz'):
                    old=a/name;new=b/name;rec=records[str(old.relative_to(V1))]
                    assert sha(old)==rec['sha256'] and old.stat().st_size==rec['bytes']
                    for file in (old,new):inputs[str(file.relative_to(ROOT))]={'sha256':sha(file),'bytes':file.stat().st_size}
                m,n=read(a/'training.json'),read(b/'training.json')
                assert m['epochs']==48 and n['epochs']==96
                for k in ('source_ids','query_ids','source_fingerprint','initial_encoder_sha256','initial_ZY_head_sha256'):
                    assert m[k]==n[k],k
                assert len(m['target_epoch_loss'])==48 and len(n['target_epoch_loss'])==96
                loss1=np.asarray(m['target_epoch_loss'],dtype=np.float64);loss2=np.asarray(n['target_epoch_loss'][:48],dtype=np.float64)
                assert payload_sha(loss1)==payload_sha(loss2)
                with np.load(a/'target-scaler.npz',allow_pickle=False) as x,np.load(b/'target-scaler.npz',allow_pickle=False) as y:
                    assert x.files==y.files
                    scaler={}
                    for k in x.files:
                        assert payload_sha(x[k])==payload_sha(y[k]);scaler[k]=payload_sha(x[k])
                with np.load(a/'predictions.npz',allow_pickle=False) as x,np.load(b/'predictions.npz',allow_pickle=False) as y:
                    assert payload_sha(x['record_ids'])==payload_sha(y['record_ids'])
                    left=x['epoch_scores'];right=y['epoch_scores'][:48]
                    assert left.shape==right.shape==(48,342,7)
                    assert np.isfinite(left).all() and np.isfinite(right).all()
                    assert payload_sha(left)==payload_sha(right)
                    max_diff=max(max_diff,float(np.max(np.abs(left-right))))
                    prefix_hash=payload_sha(left)
                checked.append({'split_id':c['split_id'],'seed':seed,'source_records':342,'source_query_records':342,
                    'source_scaler_payloads':scaler,'first48_logits_payload_sha256':prefix_hash,'first48_loss_payload_sha256':payload_sha(loss1),
                    'first48_logits_and_loss_bit_payload_identical':True,'source_IDs_and_initial_state_identical':True})
        assert len(checked)==30
        result.update(status='PASS_SOURCE_PREFIX_ABLATION',source_trajectories=30,source_prefix_epochs=48,
            logit_max_abs_difference=max_diff,prefix_logit_scalars_compared=30*48*342*7,
            source_loss_values_compared=30*48,trajectories=checked,
            V1_sealed_index_sha256=sha(V1/'index.json'),V2_protocol_sha256=sha(V2/'protocol.json'))
    except BaseException as exc:result.update(status='FAIL',error=repr(exc))
    result.update(ended_at_utc=datetime.now(timezone.utc).isoformat(),auditor_sha256=sha(__file__),input_files=inputs,numpy_version=np.__version__)
    (OUT/'prefix-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result.get(k) for k in ('status','error','source_trajectories','logit_max_abs_difference')}))
    return 0 if result['status'].startswith('PASS') else 1


if __name__=='__main__':raise SystemExit(main())
