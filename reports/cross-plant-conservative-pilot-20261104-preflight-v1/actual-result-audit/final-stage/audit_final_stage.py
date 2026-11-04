"""Independent read-only scoring/provenance replay; no Torch and no fitting."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import gzip
import importlib.util
import json
import os
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np

OUT=Path(__file__).resolve().parent
helper=OUT.parent/'source-stage/audit_source_stage.py'
spec=importlib.util.spec_from_file_location('independent_source_auditor',helper)
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
ROOT,RUN,OLD,CLASSES,SEEDS=h.ROOT,h.RUN,h.OLD,h.CLASSES,h.SEEDS
ARMS=('A','Aextra_L2SP','B_HQ_TM_L2SP')


def full_metrics(truth,pred):
    lut={c:i for i,c in enumerate(CLASSES)}
    a=np.asarray([lut[str(v)] for v in truth]);b=np.asarray([lut[str(v)] for v in pred])
    matrix=np.bincount(7*a+b,minlength=49).reshape(7,7)
    tp=np.diag(matrix).astype(float);actual=matrix.sum(1);found=matrix.sum(0)
    precision=np.divide(tp,found,out=np.zeros(7),where=found!=0)
    recall=np.divide(tp,actual,out=np.zeros(7),where=actual!=0)
    f1=np.divide(2*tp,actual+found,out=np.zeros(7),where=actual+found!=0)
    return {'accuracy':float(tp.sum()/len(a)),'macro_f1':float(f1.mean()),'balanced_accuracy':float(recall.mean()),
        'correct':int(tp.sum()),'total':int(matrix.sum()),'confusion_matrix':matrix.tolist(),
        'per_class':{c:{'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'support':int(actual[i])} for i,c in enumerate(CLASSES)},
        'class_counts':dict(zip(CLASSES,map(int,actual))),'class_recall':dict(zip(CLASSES,map(float,recall)))}


def main():
    started=time.process_time();checks=[];files={};actual_predictions={};fitted=[];covered={};metric_error=0.;pretrain_batches=target_batches=0
    def check(name,condition):
        checks.append({'name':name,'passed':bool(condition)})
        if not condition:raise ValueError(name)
    def bind(p):
        p=Path(p);files[str(p.relative_to(ROOT))]={'sha256':h.sha(p),'bytes':p.stat().st_size}
    result={'status':'PARTIAL','study_status':'DEVELOPMENT_NOT_NEW_CONFIRMATION','training_fits_by_auditor':0,
        'quantum_kernel_calls':0,'Torch_imported':False,'model_weight_replay_performed':False,'started_at_utc':datetime.now(timezone.utc).isoformat()}
    try:
        completion=h.read(RUN/'completion.json');ledger=h.read(RUN/'resource_ledger.json');protocol=h.read(RUN/'protocol.json')
        lock=h.read(RUN/'outer_prediction_lock.json');epochs=h.read(RUN/'source_epoch_lock.json');saved=h.read(RUN/'metrics.json');prep=h.read(OLD/'preparation.json')
        for name in ('completion.json','resource_ledger.json','protocol.json','outer_prediction_lock.json','source_epoch_lock.json','metrics.json','environment.json','worker_result.json'):bind(RUN/name)
        check('actual_complete',completion['status']=='DONE_DEVELOPMENT_ONLY' and completion['all_predeclared_seeds_and_folds_complete'] is True)
        check('no_new_confirmation',not completion['new_independent_confirmation'] and saved['study_status']=='DEVELOPMENT_NOT_NEW_CONFIRMATION')
        check('source_lock_unchanged',lock['source_epoch_lock_sha256']==h.sha(RUN/'source_epoch_lock.json'))
        check('all_45_prediction_locks',len(lock['files'])==45 and lock['source_fits']==30 and lock['outer_fits']==45)
        check('two_actual_clean_child_processes',len(ledger['phases'])==2 and [p['phase'] for p in ledger['phases']]==['worker','score'] and len({p['pid'] for p in ledger['phases']})==2)
        for p in ledger['phases']:
            check('actual_child_closed',p['status']=='EXITED' and p['exit_code']==0 and p['stop_reason'] is None and p['monitor_error'] is None)
            bind(RUN/(p['phase']+'.log'));check('closed_log_sha',p['log_sha256']==h.sha(RUN/(p['phase']+'.log')))
        total=sum(p['actual_CPU_seconds'] for p in ledger['phases'])
        check('wait4_cost_total',abs(total-ledger['actual_CPU_seconds'])<=1e-9 and abs(total-completion['actual_CPU_seconds'])<=1e-9)
        check('no_phase_CPU_reset',ledger['phases'][1]['prior_actual_CPU_seconds']==ledger['phases'][0]['actual_CPU_seconds'])
        check('inside_preregistered_budget',total<=600 and ledger['wall_seconds']<=1800 and max(p['wait4_peak_RSS_KiB']*1024 for p in ledger['phases'])<=int(1.5*1024**3))
        env=h.read(RUN/'environment.json');check('actual_CPU_single_threads',env['device']=='cpu' and env['torch_threads']==env['torch_interop_threads']==1 and set(env['thread_environment'].values())=={'1'})
        check('scoring_after_prediction_lock',datetime.fromisoformat(ledger['phases'][1]['started_at_utc']).timestamp() >= (RUN/'outer_prediction_lock.json').stat().st_mtime)
        source_audit=h.read(OUT.parent/'source-stage/audit.json');check('independent_source_replay_passed',source_audit['status']=='PASS_SOURCE_STAGE_ONLY' and source_audit['source_epoch_lock_sha256']==h.sha(RUN/'source_epoch_lock.json'));bind(OUT.parent/'source-stage/audit.json')
        auxiliary={}
        for name in ('huangqi','tianma'):
            p=OLD/prep['auxiliary'][name]['path'];bind(p);check('external_'+name+'_bytes',h.sha(p)==prep['auxiliary'][name]['sha256'])
            with np.load(p,allow_pickle=False) as a:auxiliary[name]={'record_ids':a['record_ids']}
        contexts=[c for c in prep['contexts'] if c['stage']=='outer'];check('five_outer_contexts',len(contexts)==5)
        all_truth={};source_last=0.;outer_first=float('inf')
        for c in contexts:
            sid=c['split_id'];qids=c['query_ids'];sids=c['source_ids'];sindex={rid:i for i,rid in enumerate(sids)};allowed=set(sids);forbidden=set(qids)
            truth_path=OLD/c['truth_path'];bind(truth_path);check(sid+'.truth_hash',h.sha(truth_path)==c['truth_sha256'])
            with truth_path.open() as f:truth={r['record_id']:r['label'] for r in csv.DictReader(f)}
            check(sid+'.query_truth_roles',set(truth)==forbidden and len(qids)==171 and len(sids)==684 and not allowed&forbidden)
            check(sid+'.truth_global_unique',not set(all_truth)&set(truth));all_truth.update(truth)
            for role in ('source','query'):
                p=OLD/c[role+'_pack'];bind(p);check(sid+'.'+role+'.hash',h.sha(p)==c[role+'_sha256'])
            with np.load(OLD/c['source_pack'],allow_pickle=False) as a:raw=a['X'];labels=a['source_labels'];stored_sids=a['record_ids']
            check(sid+'.source_order',stored_sids.tolist()==sids)
            expected_epoch=next(row['epochs'] for row in epochs['choices'] if row['outer_fold']==c['outer_fold'])
            for seed in SEEDS:
                paired=[]
                for arm in ARMS:
                    d=RUN/'outer_models'/sid/f'seed{seed}'/arm;m=h.read(d/'training.json');paired.append(m)
                    for name in ('training.json','predictions.npz','last-state.pt','fit_scope.jsonl','batch_trace.jsonl.gz','target-scaler.npz'):bind(d/name)
                    check(sid+'.'+arm+'.metadata_role',m['seed']==seed and m['arm']==arm and m['epochs']==expected_epoch and m['status']=='FITTED_PREDICTION_LOCKED')
                    check(sid+'.'+arm+'.source_exact',m['source_ids']==sids and m['query_ids']==qids and m['source_fingerprint']==h.sourcefp(raw,labels))
                    check(sid+'.'+arm+'.no_query_labels',not m['query_labels_received'] and not m['numerical_features_contain_ID_paths_or_chemical_descriptors'])
                    check(sid+'.'+arm+'.byte_locks',m['prediction_sha256']==h.sha(d/'predictions.npz')==lock['files'][str((d/'predictions.npz').relative_to(RUN))] and m['model_sha256']==h.sha(d/'last-state.pt'))
                    with np.load(d/'predictions.npz',allow_pickle=False) as a:ids=a['record_ids'];scores=a['scores'];ep=a['epoch_scores']
                    check(sid+'.'+arm+'.query_array',ids.tolist()==qids and scores.shape==(171,7) and ep.size==0 and np.isfinite(scores).all())
                    found=np.asarray(CLASSES)[scores.argmax(1)]
                    for rid,pred,ss in zip(ids,found,scores):
                        key=(seed,arm,str(rid));check('unique_original_outer_prediction',key not in actual_predictions)
                        actual_predictions[key]={'true_label':truth[str(rid)],'prediction':str(pred),'scores':ss,'outer_fold':c['outer_fold']}
                    expected_pools={} if arm=='A' else {'target':120} if arm=='Aextra_L2SP' else {'huangqi':60,'tianma':60}
                    check(sid+'.'+arm+'.pretraining_steps',m['pretraining_pool_steps']==expected_pools)
                    check(sid+'.'+arm+'.lambda',m['encoder_L2SP_lambda']==(0. if arm=='A' else .001))
                    for name,value in m['BN_after_reset'].items():
                        a=np.asarray(value);expected=1 if name.endswith('running_var') else 0
                        check(sid+'.'+arm+'.BN_reset',np.all(a==expected))
                    ids_seen={p:set() for p in expected_pools};pool_steps={p:0 for p in expected_pools};orders=[];order=[];tsteps=0
                    with gzip.open(d/'batch_trace.jsonl.gz','rt') as f:
                        for line in f:
                            b=json.loads(line);selected=b['fit_sample_ids'];check('actual_batch_unique',len(selected)==len(set(selected)))
                            if b['stage']=='pretrain':
                                name=b['head'];check('registered_pretraining_head',name in expected_pools)
                                pool_allowed=allowed if name=='target' else set(auxiliary[name]['record_ids'].tolist())
                                check('actual_pretrain_scope',set(selected)<=pool_allowed and not set(selected)&forbidden and len(selected)==32)
                                if arm=='B_HQ_TM_L2SP':check('fixed_external_task_rotation',name==('huangqi','tianma')[b['step']%2])
                                ids_seen[name].update(selected);pool_steps[name]+=1;pretrain_batches+=1
                            else:
                                check('actual_target_scope',b['stage']=='target' and b['head']=='ZY' and b['step']==tsteps and set(selected)<=allowed and not set(selected)&forbidden)
                                order.extend([sindex[rid] for rid in selected]);tsteps+=1;target_batches+=1
                                if tsteps%11==0:check('each_target_epoch_has_source_once',sorted(order)==list(range(684)));orders.append(order);order=[]
                    check(sid+'.'+arm+'.actual_pretrain_counts',pool_steps==expected_pools)
                    check(sid+'.'+arm+'.actual_gradient_coverage',m['pretraining_unique_gradient_records']=={p:len(v) for p,v in ids_seen.items()})
                    check(sid+'.'+arm+'.actual_target_schedule',tsteps==m['target_steps']==11*expected_epoch and len(orders)==expected_epoch and h.objhash(orders)==m['target_batch_schedule_sha256'])
                    covered[sid+'.'+str(seed)+'.'+arm]={p:len(v) for p,v in ids_seen.items()}
                    with (d/'fit_scope.jsonl').open() as f:
                        first=True
                        for line in f:
                            e=json.loads(line)
                            if first:outer_first=min(outer_first,datetime.fromisoformat(e['utc']).timestamp());first=False
                            sp=e['split_id'];pool=next((name for name in ('huangqi','tianma') if sp.endswith('.pretrain.'+name)),None)
                            permitted=allowed if pool is None else set(auxiliary[pool]['record_ids'].tolist())
                            check('fit_scope_actual_IDs',set(e['fit_sample_ids'])<=permitted and not set(e['fit_sample_ids'])&forbidden)
                    fitted.append({'outer_fold':c['outer_fold'],'seed':seed,'arm':arm,'epoch':expected_epoch,'prediction_sha256':m['prediction_sha256'],'weight_sha256':m['model_sha256']})
                for field in ('initial_encoder_sha256','initial_ZY_head_sha256','target_batch_schedule_sha256'):
                    check(sid+'.paired_'+field,len({m[field] for m in paired})==1)
        check('locked_source_before_first_outer_fit',outer_first >= (RUN/'source_epoch_lock.json').stat().st_mtime)
        check('855_unique_target_and45_models',len(all_truth)==855 and len(fitted)==45 and len(actual_predictions)==7695)
        replay={};means={};pair_changes=[]
        for seed in SEEDS:
            byarm={}
            for arm in ARMS:
                ids=sorted(all_truth);values=[actual_predictions[seed,arm,rid] for rid in ids]
                actual=full_metrics([v['true_label'] for v in values],[v['prediction'] for v in values]);expected=saved['per_seed'][f'{seed}.{arm}']
                for metric in ('accuracy','macro_f1','balanced_accuracy'):
                    err=abs(actual[metric]-expected[metric]);metric_error=max(metric_error,err);check('metric_'+str(seed)+arm+metric,err<=1e-12)
                for key in ('correct','total','confusion_matrix','class_counts'):check('integer_metric_'+key,actual[key]==expected[key])
                for c in CLASSES:
                    for field in ('precision','recall','f1','support'):
                        err=abs(actual['per_class'][c][field]-expected['per_class'][c][field]);metric_error=max(metric_error,err);check('perclass_'+c+field,err<=1e-12)
                replay[f'{seed}.{arm}']=actual;byarm[arm]={rid:actual_predictions[seed,arm,rid]['prediction'] for rid in ids}
            for ref in ('A','Aextra_L2SP'):
                b=byarm[ref];candidate=byarm['B_HQ_TM_L2SP'];corrected=sum(b[k]!=all_truth[k] and candidate[k]==all_truth[k] for k in all_truth);harmed=sum(b[k]==all_truth[k] and candidate[k]!=all_truth[k] for k in all_truth)
                row={'seed':seed,'candidate':'B_HQ_TM_L2SP','reference':ref,'corrected':corrected,'harmed':harmed,'net_correct':corrected-harmed,'accuracy_delta_pp':100*(corrected-harmed)/855};pair_changes.append(row)
                expected=next(x for x in saved['paired_changes'] if x['seed']==seed and x['reference']==ref)
                check('paired_changes_'+str(seed)+ref,row==expected)
        for arm in ARMS:
            means[arm]={k:float(np.mean([replay[f'{seed}.{arm}'][k] for seed in SEEDS])) for k in ('accuracy','macro_f1','balanced_accuracy')}
            check('arm_mean_'+arm,all(abs(means[arm][k]-saved['mean'][arm][k])<=1e-12 for k in means[arm]))
        combined=RUN/'all_outer_predictions.csv.gz';bind(combined);csv_seen=set()
        with gzip.open(combined,'rt') as f:
            for row in csv.DictReader(f):
                key=(int(row['seed']),row['arm'],row['record_id']);check('CSV_unique',key not in csv_seen);csv_seen.add(key);v=actual_predictions[key]
                check('CSV_truth_prediction',row['true_label']==v['true_label'] and row['prediction']==v['prediction'] and int(row['outer_fold'])==v['outer_fold'])
                check('CSV_scores',max(abs(float(row['score_'+c])-float(v['scores'][i])) for i,c in enumerate(CLASSES))<=1e-12)
        check('CSV_7695_rows_match',csv_seen==set(actual_predictions))
        result.update(status='PASS_FINAL_DEVELOPMENT_METRICS',mean=means,per_seed=replay,paired_changes=pair_changes,
            actual_outer_model_count=45,all_predeclared_seeds=list(SEEDS),target_denominator_per_seed=855,total_unique_target_records=855,
            total_outer_prediction_rows=7695,metric_max_abs_difference=metric_error,resource_cost_actual_child_CPU_seconds=total,
            actual_outer_target_optimizer_batches=target_batches,actual_outer_pretraining_batches=pretrain_batches,pretraining_unique_gradient_records=covered,
            model_and_prediction_bindings=fitted,source_epoch_lock_sha256=h.sha(RUN/'source_epoch_lock.json'),outer_prediction_lock_sha256=h.sha(RUN/'outer_prediction_lock.json'))
    except BaseException as exc:result.update(status='FAIL',error=repr(exc))
    result.update(ended_at_utc=datetime.now(timezone.utc).isoformat(),auditor_process_CPU_seconds=time.process_time()-started,
        auditor_sha256=h.sha(__file__),source_audit_helper_sha256=h.sha(helper),checks_performed=len(checks),checks_passed=sum(r['passed'] for r in checks),checks_failed=[r for r in checks if not r['passed']],input_files=files)
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (OUT/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result.get(k) for k in ('status','error','mean','metric_max_abs_difference','checks_performed','auditor_process_CPU_seconds')}))
    return 0 if result['status'].startswith('PASS') else 1


if __name__=='__main__':raise SystemExit(main())
