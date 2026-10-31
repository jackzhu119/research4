"""Final prediction arithmetic and verified-resume audit; no model fitting."""
from pathlib import Path
import csv,gzip,hashlib,json,time
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
import herb_spectra.neural_base_search as neural
from herb_spectra.paper_fewshot_classification import state_fingerprint

root=Path(__file__).resolve().parents[2]
run=root/'outputs/multiround-20261031/neural-base'
input_path=root/'outputs/approved-zaoren-20261031/data-preparation/dataset.npz'
plan_path=root/'data/processed/zaoren_threeway_20261031_v1/split_plan.json'
protocol_path=root/'data/research/neural_base_search_protocol_20261031.json'
checks=0;maximum=0.
def check(value,label):
    global checks
    checks+=1
    if not value:raise AssertionError(label)
def close(a,b,label):
    global maximum
    maximum=max(maximum,abs(float(a)-float(b)))
    check(abs(float(a)-float(b))<1e-12,label)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
contexts,truth=neural.load_locked_contexts(input_path,plan_path,run,scope='final')
check(len(contexts)==3,'contexts');classes=contexts[0]['classes'];lookup={str(c):i for i,c in enumerate(classes)}
summary=json.loads((run/'summary.json').read_text());rows=json.loads((run/'final_metrics.json').read_text())
recomputed=[];records_checked=0
for context,row in zip(contexts,rows):
    prediction=context['prediction'];check('y' not in prediction,'truthseparated')
    sets=[set(context[k]['record_ids']) for k in ('source','support_pool','prediction')]
    check([len(s) for s in sets]==[513,171,171],'rolecounts')
    check(not any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)),'roleisolation')
    directory=Path(context['directory']);path=directory/'predictions.csv.gz'
    check(digest(path)==row['prediction_sha256'],'CSVhash')
    with gzip.open(path,'rt',encoding='utf-8',newline='') as f:records=list(csv.DictReader(f))
    check([r['record_id'] for r in records]==prediction['record_ids'].tolist(),'queryIDorder')
    y=[];p=[]
    for r in records:
        check(r['y_true']==truth[r['record_id']],'truth')
        check(int(r['is_correct'])==int(r['y_true']==r['y_predicted']),'correctflag')
        y.append(lookup[r['y_true']]);p.append(lookup[r['y_predicted']]);records_checked+=1
    before=state_fingerprint(context['model'])
    replay=neural.model_scores(context['model'],prediction['X']).argmax(1)
    check(np.array_equal(replay,p),'inference_replay')
    for i in [0,20,90,150,170]:check(int(neural.model_scores(context['model'],prediction['X'][i:i+1]).argmax(1)[0])==int(replay[i]),'pointwise')
    check(state_fingerprint(context['model'])==before,'sourcefreeze')
    metrics={'accuracy':accuracy_score(y,p),'balanced_accuracy':balanced_accuracy_score(y,p),'macro_f1':f1_score(y,p,labels=list(range(7)),average='macro',zero_division=0)}
    for k,v in metrics.items():close(v,row[k],k)
    check(confusion_matrix(y,p,labels=list(range(7))).tolist()==row['confusion_matrix'],'CM')
    recomputed.append(metrics)
for k in recomputed[0]:close(float(np.mean([r[k] for r in recomputed])),summary[k+'_mean'],'summary'+k)
before={str(p):p.stat().st_mtime_ns for p in run.rglob('model.pt')}
original=neural.fit_model
def no_fit(*args,**kwargs):raise AssertionError('Resume must not fit')
neural.fit_model=no_fit
try:
    resumed=neural.run_search(input_path,plan_path,protocol_path,run,resume=True,stage='final')
finally:neural.fit_model=original
check(resumed==summary,'resumedsummary')
check(before=={str(p):p.stat().st_mtime_ns for p in run.rglob('model.pt')},'resumestateunchanged')
result={'status':'pass','scope':'run-side final arithmetic, source replay and no-fit resume',
        'checks_evaluated':checks,'maximum_numeric_difference':maximum,'prediction_rows':records_checked,'seeds':[c['seed'] for c in contexts],
        'source_inference_replays':3,'pointwise_inference_checks':15,'source_weights_unchanged':len(before),
        'resumed_final_without_fit':True,'additional_model_fits':0,'query_truth_separate_from_context':True,
        'audit_script_sha256':digest(Path(__file__)),'selection_lock_sha256':digest(run/'selection_lock.json')}
neural.write_json(run/'final_arithmetic_and_resume_audit.json',result)
print(json.dumps(result))
