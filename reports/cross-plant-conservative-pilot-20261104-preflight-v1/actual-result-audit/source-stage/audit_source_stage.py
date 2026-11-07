"""Independent, read-only NumPy replay of the already closed source stage.

No Torch/model fit or quantum imports. Never opens outer-model array files.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import os
import sys

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
RUN=ROOT/'reports/cross-plant-conservative-pilot-20261104-v1'
OLD=ROOT/'reports/cross-plant-deep-transfer-20261031-v1/neural'
OUT=Path(__file__).resolve().parent
CLASSES=[f'ZY{i:02d}' for i in range(1,8)]
SEEDS=(42,43,44)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(1024**2),b''):h.update(part)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def objhash(v):return hashlib.sha256(canonical(v).encode()).hexdigest()


def arrayfp(a):
    a=np.asarray(a)
    head=canonical({'shape':a.shape,'dtype':str(a.dtype)})
    h=hashlib.sha256(head.encode());h.update(np.ascontiguousarray(a).tobytes());return h.hexdigest()


def sourcefp(x,y):return objhash({'X':arrayfp(x),'y':arrayfp(y)})


def metrics(truth,predicted):
    lookup={c:i for i,c in enumerate(CLASSES)}
    a=np.asarray([lookup[str(x)] for x in truth]);b=np.asarray([lookup[str(x)] for x in predicted])
    matrix=np.bincount(7*a+b,minlength=49).reshape(7,7)
    tp=np.diag(matrix).astype(float);actual=matrix.sum(1);found=matrix.sum(0)
    if not np.all(actual):raise ValueError('Source OOF lost a true class')
    f=np.divide(2*tp,actual+found,out=np.zeros(7),where=actual+found!=0)
    return {'accuracy':float(tp.sum()/len(a)),'macro_f1':float(f.mean()),
            'balanced_accuracy':float((tp/actual).mean())}


def main():
    checks=[];inputs={};source_info=[];preds={};truth_by_context={};max_scaler_error=0.;trace_batches=0;trace_presentations=0
    def check(name,condition):
        checks.append({'name':name,'passed':bool(condition)})
        if not condition:raise ValueError('FAILED '+name)
    def bind(p):
        p=Path(p);inputs[str(p.relative_to(ROOT))]={'sha256':sha(p),'bytes':p.stat().st_size}
    result={'status':'PARTIAL','scope':'INDEPENDENT_SOURCE_STAGE_REPLAY_ONLY','training_fits':0,'quantum_kernel_calls':0,
        'outer_model_arrays_opened':0,'new_independent_confirmation':False,'started_at_utc':datetime.now(timezone.utc).isoformat()}
    try:
        protocol=read(RUN/'protocol.json');registration=read(RUN/'registration.json');lock=read(RUN/'source_epoch_lock.json');prep=read(OLD/'preparation.json')
        for p in (RUN/'protocol.json',RUN/'registration.json',RUN/'source_epoch_lock.json',OLD/'preparation.json'):bind(p)
        check('registered_protocol_bytes',registration['protocol_sha256']==sha(RUN/'protocol.json')==lock['protocol_sha256'])
        check('prepared_metadata_bytes',protocol['preparation_sha256']==sha(OLD/'preparation.json'))
        check('source_stage_lock',lock['status']=='ALL_SOURCE_CHOICES_LOCKED_BEFORE_OUTER' and lock['outer_truth_used'] is False)
        check('source_frozen_at_registration',protocol['source_sha256']['scripts/run_conservative_cross_plant_pilot_20261104.py']==registration['source_sha256'])
        for p,expected in protocol['source_sha256'].items():
            check('frozen_code:'+p,sha(ROOT/p)==expected);bind(ROOT/p)
        split_path=ROOT/protocol['split_path'];bind(split_path)
        check('original_split_sha',sha(split_path)==protocol['split_sha256']=='fcf14231b4dc1a1e15af8c46310ee6f39e71520b76a7b62fe0e176485f6f5cfe')
        splits={r['fold']:r for r in read(split_path)}
        inner=[c for c in prep['contexts'] if c['stage']=='inner']
        check('ten_original_inner_contexts',len(inner)==10)
        check('all_five_choices',sorted(c['outer_fold'] for c in lock['choices'])==list(range(5)))
        for c in inner:
            sid=c['split_id'];ref=splits[c['outer_fold']]['inner_folds'][c['inner_fold']]
            check(sid+'.original_role_order',c['source_ids']==ref['train_sample_ids'] and c['query_ids']==ref['validation_sample_ids'])
            for role in ('source','query'):
                p=OLD/c[role+'_pack'];bind(p);check(sid+'.'+role+'.input_bytes',sha(p)==c[role+'_sha256'])
            with np.load(OLD/c['source_pack'],allow_pickle=False) as f:source={k:f[k] for k in f.files}
            with np.load(OLD/c['query_pack'],allow_pickle=False) as f:query={k:f[k] for k in f.files}
            check(sid+'.schema_source',set(source)=={'X','record_ids','axis','source_labels'})
            check(sid+'.schema_query_without_labels',set(query)=={'X','record_ids','axis'})
            check(sid+'.source_ids',source['record_ids'].tolist()==c['source_ids'])
            check(sid+'.query_ids',query['record_ids'].tolist()==c['query_ids'])
            check(sid+'.role_disjoint',not(set(c['source_ids'])&set(c['query_ids'])) and len(c['source_ids'])==len(c['query_ids'])==342)
            expected_axis=np.linspace(.21,1.49,128)
            check(sid+'.axis',np.array_equal(source['axis'],expected_axis) and np.array_equal(query['axis'],expected_axis))
            truth_path=OLD/c['truth_path'];bind(truth_path);check(sid+'.source_truth_sha',sha(truth_path)==c['truth_sha256'])
            with truth_path.open() as f:truth={r['record_id']:r['label'] for r in csv.DictReader(f)}
            check(sid+'.truth_matches_query_ids',set(truth)==set(c['query_ids']));truth_by_context[sid]=truth
            x=source['X'];ys=source['source_labels'];ids=source['record_ids'];lookup={str(v):i for i,v in enumerate(ids)}
            normalized=(x-x.mean(1,keepdims=True))/x.std(1,ddof=1,keepdims=True)
            encoded=np.asarray([CLASSES.index(str(y)) for y in ys],dtype=np.int64)
            check(sid+'.all_source_classes',sorted(np.unique(ys).tolist())==CLASSES)
            for seed in SEEDS:
                d=RUN/'source_models'/sid/f'seed{seed}';meta=read(d/'training.json')
                for n in ('training.json','predictions.npz','target-scaler.npz','last-state.pt','fit_scope.jsonl','batch_trace.jsonl.gz'):bind(d/n)
                check(f'{sid}.{seed}.scientific_role',meta['status']=='FITTED_PREDICTION_LOCKED' and meta['arm']=='A' and meta['seed']==seed and meta['epochs']==48)
                check(f'{sid}.{seed}.fit_ids',meta['source_ids']==c['source_ids'] and meta['query_ids']==c['query_ids'])
                check(f'{sid}.{seed}.raw_source_values',meta['source_fingerprint']==sourcefp(x,ys))
                check(f'{sid}.{seed}.prediction_bytes',meta['prediction_sha256']==sha(d/'predictions.npz'))
                check(f'{sid}.{seed}.model_bytes',meta['model_sha256']==sha(d/'last-state.pt'))
                check(f'{sid}.{seed}.scratch_source_only',meta['encoder_L2SP_lambda']==0 and meta['pretraining_pool_steps']=={} and not meta['query_labels_received'])
                check(f'{sid}.{seed}.scratch_encoder_reused_before_fit',meta['initial_encoder_sha256']==meta['pretrained_encoder_sha256']==meta['reset_encoder_sha256'])
                with np.load(d/'target-scaler.npz',allow_pickle=False) as f:mean=f['mean'];scale=f['scale']
                diff=max(float(np.max(np.abs(mean-normalized.mean(0)))),float(np.max(np.abs(scale-normalized.std(0)))))
                max_scaler_error=max(max_scaler_error,diff);check(f'{sid}.{seed}.scaler_source_recomputed',diff<=1e-12)
                z=((normalized-mean)/scale).astype(np.float32)
                with np.load(d/'predictions.npz',allow_pickle=False) as f:pids=f['record_ids'];scores=f['scores'];epoch_scores=f['epoch_scores']
                check(f'{sid}.{seed}.epoch_arrays',pids.tolist()==c['query_ids'] and scores.shape==(342,7) and epoch_scores.shape==(48,342,7) and np.isfinite(epoch_scores).all())
                check(f'{sid}.{seed}.last_epoch_matches_final',np.array_equal(scores,epoch_scores[-1]))
                preds[sid,seed]=np.asarray(CLASSES)[epoch_scores.argmax(2)]
                events=[]
                with (d/'fit_scope.jsonl').open() as f:
                    for line in f:
                        e=json.loads(line);check(f'{sid}.{seed}.scope_allowed',set(e['fit_sample_ids'])<=set(c['source_ids']) and not(set(e['fit_sample_ids'])&set(c['query_ids'])))
                        check(f'{sid}.{seed}.scope_version',e['code_version']==registration['source_sha256'] and e['dataset_version']==protocol['run_id']);events.append(e)
                starts=[e for e in events if e['event']=='fit_start'];ends=[e for e in events if e['event']=='fit_completed']
                check(f'{sid}.{seed}.scope_complete',len(starts)==len(ends)==289)
                orders=[];epoch_order=[]
                with gzip.open(d/'batch_trace.jsonl.gz','rt') as f:
                    for step,line in enumerate(f):
                        e=json.loads(line);selected=[lookup[r] for r in e['fit_sample_ids']]
                        check(f'{sid}.{seed}.batch_role',e['stage']=='target' and e['head']=='ZY' and e['step']==step and len(selected)==len(set(selected)))
                        check(f'{sid}.{seed}.batch_numeric_inputs',e['X_fingerprint']==arrayfp(z[selected]) and e['y_fingerprint']==arrayfp(encoded[selected]))
                        epoch_order.extend(selected);trace_batches+=1;trace_presentations+=len(selected)
                        if (step+1)%6==0:
                            check(f'{sid}.{seed}.each_epoch_all_source',sorted(epoch_order)==list(range(342)))
                            orders.append(epoch_order);epoch_order=[]
                check(f'{sid}.{seed}.epoch_batch_schedule',len(orders)==48 and objhash(orders)==meta['target_batch_schedule_sha256'] and meta['target_steps']==288)
                source_info.append({'split_id':sid,'seed':seed,'records':342,'epochs':48,'target_steps':288,'model_sha256':meta['model_sha256'],'prediction_sha256':meta['prediction_sha256']})
        choices=[];curve_error=0.
        for outer in range(5):
            contexts=[c for c in inner if c['outer_fold']==outer];curves=[]
            saved=next(c for c in lock['choices'] if c['outer_fold']==outer)
            for epoch in range(1,49):
                per_seed=[]
                for seed in SEEDS:
                    truth=[];found=[];ids=[]
                    for c in contexts:
                        ids+=c['query_ids'];truth += [truth_by_context[c['split_id']][rid] for rid in c['query_ids']]
                        found.extend(preds[c['split_id'],seed][epoch-1].tolist())
                    check(f'outer{outer}.{seed}.source_OOF_identity',len(ids)==len(set(ids))==684 and set(ids)==set(splits[outer]['train_sample_ids']))
                    per_seed.append(metrics(truth,found))
                row={'epoch':epoch,**{k:float(np.mean([m[k] for m in per_seed])) for k in ('macro_f1','balanced_accuracy','accuracy')}}
                curves.append(row)
                expected=saved['source_curve'][epoch-1]
                curve_error=max(curve_error,max(abs(row[k]-expected[k]) for k in ('macro_f1','balanced_accuracy','accuracy')))
            chosen=min(curves,key=lambda r:(-r['macro_f1'],-r['balanced_accuracy'],-r['accuracy'],r['epoch']))
            check(f'outer{outer}.selected_epoch_recomputed',chosen['epoch']==saved['epochs']==saved['source_selected_metrics']['epoch'])
            choices.append({'outer_fold':outer,'selected_epoch':chosen['epoch'],'source_metrics':chosen})
        check('all_240_curve_rows_recomputed',curve_error<=1e-12)
        check('all_30_source_fits',len(source_info)==30)
        result.update(status='PASS_SOURCE_STAGE_ONLY',source_fit_count=30,selected_epochs=choices,
            source_curve_rows_recomputed=240,source_metric_max_abs_difference=curve_error,
            source_scaler_max_abs_difference=max_scaler_error,actual_source_optimizer_batches=trace_batches,
            actual_source_optimizer_presentations=trace_presentations,
            epoch_prediction_observations_recomputed=30*48*342,
            source_training_state_metadata=source_info,source_epoch_lock_sha256=sha(RUN/'source_epoch_lock.json'))
    except BaseException as exc:
        result.update(status='FAIL',error=repr(exc))
    result.update(ended_at_utc=datetime.now(timezone.utc).isoformat(),checks_performed=len(checks),checks_passed=sum(c['passed'] for c in checks),checks_failed=[c for c in checks if not c['passed']],input_files=inputs,auditor_sha256=sha(__file__),numpy_version=np.__version__)
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (OUT/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result.get(k) for k in ('status','error','checks_performed','source_metric_max_abs_difference','selected_epochs')}))
    return 0 if result['status'].startswith('PASS') else 1


if __name__=='__main__':raise SystemExit(main())
