"""Read-only arithmetic and lineage audit of the probability track."""
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
RUN=Path(__file__).resolve().parent
checks=0;maximum=0.


def checked(condition,message):
    global checks
    checks+=1
    if not condition:raise AssertionError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def near(a,b,message):
    global maximum
    difference=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    maximum=max(maximum,difference);checked(difference<1e-10,message)


def score(y,pred,p,classes):
    cm=np.zeros((classes,classes),dtype=int)
    np.add.at(cm,(y,pred),1)
    support=cm.sum(1);predicted=cm.sum(0);diag=np.diag(cm)
    recalls=np.divide(diag,support,out=np.zeros(classes),where=support>0)
    f1=np.divide(2*diag,support+predicted,out=np.zeros(classes),where=support+predicted>0)
    confidence=p.max(1);correct=p.argmax(1)==y;ece=0.
    bucket=np.minimum((10*confidence).astype(int),9)
    for i in range(10):
        mask=bucket==i
        if mask.any():ece+=mask.mean()*abs(correct[mask].mean()-confidence[mask].mean())
    return {'accuracy':float(diag.sum()/len(y)),'balanced_accuracy':float(recalls[support>0].mean()),
        'macro_f1':float(f1.mean()),'nll':float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-15)).mean()),
        'brier':float(np.square(p-np.eye(classes)[y]).sum(1).mean()),'ece10':float(ece),
        'confusion_matrix':cm.tolist(),'correct':int(diag.sum()),'total':len(y)}


with np.load(ROOT/'outputs/approved-zaoren-20261031/data-preparation/dataset.npz',allow_pickle=False) as data:
    truth=dict(zip(data['record_ids'].tolist(),data['y'].tolist()))
manifest=json.loads((RUN/'run_manifest.json').read_text())
for path,digest in manifest['source_sha256'].items():checked(sha(ROOT/path)==digest,('source',path))
all_case_rows={};total_prediction_rows=0;contexts=0;oofrows=0;models=0
for stage in ('development','final'):
    index=RUN/(stage+'_contexts.json')
    if not index.exists():continue
    case_rows=[]
    for relative in json.loads(index.read_text()):
        directory=RUN/relative;meta=json.loads((directory/'context.json').read_text());contexts+=1
        checked(sha(directory/'scores.npz')==meta['scores_npz_sha256'],('npz',relative))
        with np.load(directory/'scores.npz',allow_pickle=False) as handle:v={k:handle[k] for k in handle.files}
        classes=v['classes'].tolist();vocab={c:i for i,c in enumerate(classes)}
        checked(set(v['source_ids']).isdisjoint(set(v['prediction_ids'])),('roles',relative))
        checked(meta['target_support_labels_used']==0,('support use',relative))
        checked(meta['source_fingerprints']==meta['source_fingerprints_after'],('source freeze',relative))
        checked(('prediction_y' not in v) if stage=='final' else ('prediction_y' in v),('truth API',relative))
        for artifact in meta['source_artifacts'].values():checked(sha(ROOT/artifact['path'])==artifact['sha256'],('outer model',relative))
        oofrows+=len(v['oof_ids']);seen=[]
        for fold in meta['folds']:
            train=set(fold['source_train_ids']);valid=set(fold['source_validation_ids']);seen.extend(valid)
            checked(train.isdisjoint(valid),('OOF overlap',relative,fold['fold']))
            checked(train|valid==set(v['source_ids']),('OOF coverage',relative,fold['fold']))
            checked(set(meta['oof_policy']['excluded_oof_ids']).issubset(train),('rare anchor',relative))
            checked(fold['preprocessor_fit_source_rows']==len(train),('scaler fit',relative))
            for name,digest in fold['artifacts'].items():
                checked(sha(directory/'oof'/f"fold_{fold['fold']:02d}"/name)==digest,('OOF artifact',relative,name))
            models+=3
        checked(sorted(seen)==sorted(v['oof_ids'].tolist()),('OOF eligible exact',relative))
        for identifier,label in zip(v['oof_ids'],v['oof_y']):checked(truth[str(identifier)]==classes[int(label)],('OOF truth',relative))
        expected={r['candidate_id']:r for r in json.loads((directory/'metrics.json').read_text())}
        fits={r['candidate']['id']:r for r in json.loads((directory/'decision_fits.json').read_text())}
        grouped=defaultdict(list)
        with gzip.open(directory/'predictions.csv.gz','rt',encoding='utf-8',newline='') as handle:
            for row in csv.DictReader(handle):
                total_prediction_rows+=1;grouped[row['candidate_id']].append(row)
                checked(truth[row['record_id']]==row['y_true'],('prediction truth',relative))
                checked(int(row['is_correct'])==int(row['y_true']==row['y_predicted']),('correctness',relative))
        for key,records in grouped.items():
            fitted=fits[key];provider=fitted['candidate']['provider'];temperature=fitted['temperature']
            p=np.asarray([[float(r[f'p{i}']) for i in range(7)] for r in records]);y=np.array([vocab[r['y_true']] for r in records]);pred=np.array([vocab[r['y_predicted']] for r in records])
            checked([r['record_id'] for r in records]==v['prediction_ids'].tolist(),('prediction order',relative,key))
            scores=np.asarray(v['prediction_scores_'+provider],dtype=float)/temperature;scores-=scores.max(1,keepdims=True)
            probability=np.exp(scores);probability/=probability.sum(1,keepdims=True)
            near(p,probability,('softmax',relative,key));near(p.sum(1),np.ones(len(p)),('probability sum',relative,key))
            checked(np.array_equal(np.argmax(np.log(np.maximum(p,1e-15))+np.asarray(fitted['bias']),axis=1),pred),('decisionrule',relative,key))
            checked([r['probability_argmax'] for r in records]==[classes[i] for i in p.argmax(1)],('p argmax',relative,key))
            if fitted['candidate']['kind']=='temperature':checked(np.array_equal(pred,v['prediction_scores_'+provider].argmax(1)),('temperature invariant',relative,key))
            actual=score(y,pred,p,7)
            for name,value in actual.items():near(value,expected[key][name],('metric',relative,key,name))
            case_rows.append(expected[key])
    expected_path=RUN/('all_development_metrics.json' if stage=='development' else 'final_metrics.json')
    checked(case_rows==json.loads(expected_path.read_text()),('all case aggregate',stage))
    all_case_rows[stage]=case_rows
lock=json.loads((RUN/'selection_lock.json').read_text());grouped=defaultdict(list)
for r in all_case_rows['development']:grouped[(r['candidate_id'],r['stage'])].append(r)
for key,stages in lock['means'].items():
    for stage,means in stages.items():
        for metric,value in means.items():near(np.mean([r[metric] for r in grouped[(key,stage)]]),value,('development aggregate',key,stage,metric))
if 'final' in all_case_rows:
    for summary in json.loads((RUN/'summary.json').read_text()):
        cases=[r for r in all_case_rows['final'] if r['candidate_id']==summary['id']]
        for metric in ('accuracy','balanced_accuracy','macro_f1','nll','brier','ece10'):
            near(np.mean([r[metric] for r in cases]),summary[metric+'_mean'],('final summary',summary['id'],metric))
result={'status':'pass','checks_evaluated':checks,'maximum_numeric_difference':maximum,
    'prediction_rows':total_prediction_rows,'contexts':contexts,'source_oof_rows':oofrows,'source_oof_models_checked':models,
    'development_metric_cases':len(all_case_rows['development']),'final_metric_cases':len(all_case_rows.get('final',[])),
    'new_fits_in_audit':0,'target_support_labels_used':0,'script_sha256':sha(Path(__file__))}
(RUN/('final_self_audit.json' if 'final' in all_case_rows else 'development_self_audit.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
