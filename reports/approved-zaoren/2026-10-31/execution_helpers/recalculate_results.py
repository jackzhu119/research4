"""Independently audit stored OOF predictions; no model fitting or selection."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np

ROOT=Path('/workspace/research4')
BASE=ROOT/'outputs/approved-zaoren-20261031'
EXP=BASE/'experiments'
OUT=BASE/'independent-audit'
if OUT.exists():raise FileExistsError(OUT)
OUT.mkdir()
SHA=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def write_csv(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def metrics(rows,classes):
    index={c:i for i,c in enumerate(classes)};matrix=np.zeros((len(classes),len(classes)),dtype=int)
    for r in rows:matrix[index[r['y_true']],index[r['y_predicted']]]+=1
    tp=matrix.diagonal().astype(float);truth=matrix.sum(axis=1);prediction=matrix.sum(axis=0)
    recall=np.divide(tp,truth,out=np.zeros(len(classes)),where=truth>0)
    f1=np.divide(2*tp,truth+prediction,out=np.zeros(len(classes)),where=(truth+prediction)>0)
    return {'accuracy':float(tp.sum()/len(rows)),'balanced_accuracy':float(recall.mean()),'macro_f1':float(f1.mean()),
            'class_recall':dict(zip(classes,recall.tolist())),'confusion_matrix':matrix.tolist(),
            'class_counts':dict(zip(classes,truth.tolist()))}

with np.load(BASE/'datasets/candidate_0p1_1p5.npz',allow_pickle=False) as data:
    y_by_sample=dict(zip(data['groups'].tolist(),data['y'].tolist()));classes=np.unique(data['y']).tolist();candidate_axis=data['axis'].copy()
with np.load(BASE/'data-preparation/dataset.npz',allow_pickle=False) as data:full_axis=data['axis'].copy()
protocol=json.loads((EXP/'frozen_protocol.json').read_text())
code_unchanged=all(SHA(ROOT/name)==value for name,value in protocol['source_code_sha256'].items())
assert code_unchanged
jobs=['alpha_primary','modality_controls','stability_controls','full_export_sensitivity']
metrics_rows=[];recall_rows=[];tie_rows=[];selection_rows=[];cases={};split_hashes={};checks=[];max_diff=0.
for job in jobs:
    folder=EXP/job;summary=json.loads((folder/'summary.json').read_text());assert summary['status']=='completed'
    assert summary['physical_sample_count']==855
    config=summary['configuration'];assert (config['outer_splits'],config['inner_splits'],config['cars_internal_cv'],config['split_seed'],config['model_seed'])==(5,2,2,20261031,42)
    split_file=folder/'splits.json';split_hashes[job]=SHA(split_file);splits=json.loads(split_file.read_text());seen=[]
    for fold in splits:
        train=set(fold['train_sample_ids']);test=set(fold['test_sample_ids']);assert not train&test
        assert train|test==set(y_by_sample) and len(test)==171
        assert Counter(y_by_sample[s] for s in test)['ZY02']==1
        seen.extend(test)
        for inner in fold['inner_folds']:
            a=set(inner['train_sample_ids']);b=set(inner['validation_sample_ids']);assert not a&b
            assert a|b==train
            assert Counter(y_by_sample[s] for s in a)['ZY02']==2
            assert Counter(y_by_sample[s] for s in b)['ZY02']==2
    assert len(seen)==855 and len(set(seen))==855
    for result in summary['results']:
        rep=result['representation'];method=result['method'];case=folder/f'{rep}_{method}'
        with (case/'predictions.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        ids=[r['sample_id'] for r in rows];assert len(rows)==len(set(ids))==855 and set(ids)==set(y_by_sample)
        assert all(r['y_true']==y_by_sample[r['sample_id']] and int(r['scan_count'])==1 and not r['batch_id'] for r in rows)
        for fold in splits:
            subset={r['sample_id'] for r in rows if int(r['outer_fold'])==fold['fold']};assert subset==set(fold['test_sample_ids'])
        recomputed=metrics(rows,classes);saved=result['metrics']
        for name in ['accuracy','balanced_accuracy','macro_f1']:
            difference=abs(recomputed[name]-saved[name]);max_diff=max(max_diff,difference);assert difference<1e-12
        assert recomputed['confusion_matrix']==saved['confusion_matrix']
        assert recomputed['class_counts']==saved['class_counts']
        assert all(abs(recomputed['class_recall'][c]-saved['class_recall'][c])<1e-12 for c in classes)
        for fold in result['folds']:
            sub=[r for r in rows if int(r['outer_fold'])==fold['fold']];m=metrics(sub,classes)
            assert m['confusion_matrix']==fold['metrics']['confusion_matrix']
            for name in ['accuracy','balanced_accuracy','macro_f1']:
                difference=abs(m[name]-fold['metrics'][name]);max_diff=max(max_diff,difference);assert difference<1e-12
        key=job+'/'+rep+'_'+method
        preprocessing=dict(Counter(f['best_params']['preprocess__mode'] for f in result['folds']))
        row={'case':key,'frequency_scope':'full exported range sensitivity' if job=='full_export_sensitivity' else 'predeclared 0.1-1.5 THz candidate',
             'representation':rep,'method':method,'samples':855,'accuracy':recomputed['accuracy'],'balanced_accuracy':recomputed['balanced_accuracy'],
             'macro_f1':recomputed['macro_f1'],'mean_selected_features':result['mean_selected_features'],'total_seconds':result['total_seconds'],
             'best_preprocessing_by_fold':json.dumps(preprocessing,sort_keys=True),'prediction_sha256':SHA(case/'predictions.csv')}
        metrics_rows.append(row);cases[key]={'rows':rows,'metrics':recomputed,'result':result}
        for label in classes:recall_rows.append({'case':key,'origin_group':label,'sample_count':recomputed['class_counts'][label],'recall':recomputed['class_recall'][label]})
        mismatches=[]
        for record in rows:
            scores=[float(record['score_'+c]) for c in classes];assert all(math.isfinite(v) for v in scores)
            label=classes[int(np.argmax(scores))]
            if label!=record['y_predicted']:
                ordered=sorted(scores,reverse=True)
                mismatches.append({'sample_id':record['sample_id'],'predict':record['y_predicted'],'score_argmax':label,'top_two_score_gap':ordered[0]-ordered[1]})
        tie_rows.append({'case':key,'method':method,'is_svc':method=='svc' or method.endswith('_svc'),'disagreement_count':len(mismatches),'examples':mismatches[:12],
                         'scope':'diagnostic only; frozen SVC predict is retained for all primary metrics; OVR scores are not probabilities'})
        if method in {'mcuve20_svc','mcuve100_svc','stable_svc','cars_svc'}:
            axis=full_axis if job=='full_export_sensitivity' else candidate_axis
            frequency=np.zeros(len(axis),int)
            for fold in result['folds']:
                selected=np.asarray(fold['selected_feature_indices'],dtype=int)
                assert len(selected)==len(set(selected.tolist())) and np.all((selected>=0)&(selected<len(axis)))
                frequency[selected]+=1
                if method in {'mcuve20_svc','mcuve100_svc','stable_svc'}:assert len(selected)==62
            for index,count in enumerate(frequency):selection_rows.append({'case':key,'feature_index_zero_based':index,'frequency_coordinate':float(axis[index]),'selected_outer_fold_count':int(count),'outer_folds':5,
                         'scope':'frequency across overlapping outer-training sets, not independent batches, p-values, or FDR-controlled stability'})
        checks.append({'case':key,'prediction_rows':855,'metrics_match':True,'split_and_ids_match':True,'ZY02_one_per_test_fold':True,'prediction_file_sha256':row['prediction_sha256']})
assert len(set(split_hashes.values()))==1 and len(metrics_rows)==13
write_csv(OUT/'pooled_metrics.csv',metrics_rows);write_csv(OUT/'class_recalls.csv',recall_rows);write_csv(OUT/'outer_fold_selection_frequency.csv',selection_rows)
comparisons=[('MC20_vs_full_SVM','alpha_primary/alpha_mcuve20_svc','alpha_primary/alpha_svc'),
 ('CARS_vs_full_SVM','alpha_primary/alpha_cars_svc','alpha_primary/alpha_svc'),
 ('MC100_vs_MC20','stability_controls/alpha_mcuve100_svc','alpha_primary/alpha_mcuve20_svc'),
 ('stable_vs_MC20','stability_controls/alpha_stable_svc','alpha_primary/alpha_mcuve20_svc'),
 ('stable_vs_MC100','stability_controls/alpha_stable_svc','stability_controls/alpha_mcuve100_svc'),
 ('n_vs_alpha_SVM','modality_controls/n_svc','alpha_primary/alpha_svc'),
 ('concat_vs_alpha_SVM','modality_controls/concat_svc','alpha_primary/alpha_svc'),
 ('concat_vs_alpha_PLSDA','modality_controls/concat_plsda','alpha_primary/alpha_plsda'),
 ('full_range_vs_candidate_SVM','full_export_sensitivity/alpha_svc','alpha_primary/alpha_svc'),
 ('full_range_vs_candidate_PLSDA','full_export_sensitivity/alpha_plsda','alpha_primary/alpha_plsda')]
deltas=[]
for title,test,reference in comparisons:
    a,b=cases[test],cases[reference];r={'comparison':title,'method':test,'reference':reference,'direction':'positive means higher metric; descriptive paired OOF difference, not independent-batch evidence'}
    for name in ['accuracy','balanced_accuracy','macro_f1']:r[name+'_difference']=a['metrics'][name]-b['metrics'][name]
    pa={x['sample_id']:x['y_predicted']==x['y_true'] for x in a['rows']};pb={x['sample_id']:x['y_predicted']==x['y_true'] for x in b['rows']}
    r['test_correct_reference_wrong']=sum(pa[k] and not pb[k] for k in pa);r['test_wrong_reference_correct']=sum(not pa[k] and pb[k] for k in pa);deltas.append(r)
write_csv(OUT/'comparison_deltas.csv',deltas);write(OUT/'tie_policy_diagnostics.json',tie_rows)
execution=json.loads((EXP/'runs_manifest.json').read_text())['runs'];wall=(max(datetime.fromisoformat(r['ended_at_shanghai']) for r in execution)-min(datetime.fromisoformat(r['started_at_shanghai']) for r in execution)).total_seconds()
report={'scope':'User-declared independent sample-level internal CV over seven anonymous origin groups; no known batches or named-origin mapping; not cross-batch validation',
        'model_paths':13,'outer_fold_models':65,'predictions_per_path':855,'paired_n_alpha_records':855,
        'metric_formula':'CM diagonal / N; balanced accuracy = mean class recall; macro F1 = mean(2*TP/(true_count+predicted_count)); no model helper reused',
        'maximum_saved_metric_absolute_difference':max_diff,'all_job_split_sha256':split_hashes,'all_jobs_share_identical_splits':True,
        'physical_id_train_test_overlap':0,'each_sample_tested_once_per_path':True,'ZY02_test_samples_per_outer_fold':1,'ZY02_train_samples_per_inner_training_fold':2,
        'all_code_unchanged_from_frozen_snapshot':code_unchanged,'total_svc_predict_argmax_disagreements':sum(r['disagreement_count'] for r in tie_rows if r['is_svc']),
        'all_negative_and_positive_results_preserved':True,'matrix_wall_seconds':wall,'per_job_seconds':{r['job']:r['seconds'] for r in execution},
        'checks':checks,'limitations':['Independent physical-sample interpretation follows user declaration; original collection logs not inspected.','Real geographical names and source/development/target batches remain unavailable.','ZY02 has only five samples; recall changes in steps of 0.2.','THz candidate interval is a predeclared engineering comparison, not instrument-validated optimal band.','Full exported range includes negative absorption values; high sensitivity scores require independent batch and physical validation.','SVC predict versus score argmax differences are retained as a tie-policy diagnostic, not used to change the frozen primary scoring.']}
write(OUT/'independent_recalculation.json',report)
print(json.dumps({'paths':13,'outer_fold_models':65,'prediction_rows':13*855,'max_metric_difference':max_diff,'split_sha256':next(iter(split_hashes.values())),'svc_predict_argmax_disagreements':report['total_svc_predict_argmax_disagreements'],'matrix_wall_seconds':wall,'audit_dir':str(OUT)},ensure_ascii=False))
for row in metrics_rows:print(row['case'],round(100*row['accuracy'],4),round(100*row['balanced_accuracy'],4),round(row['macro_f1'],6),row['mean_selected_features'])
