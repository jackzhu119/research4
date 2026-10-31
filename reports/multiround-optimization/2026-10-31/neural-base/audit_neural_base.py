"""Recompute the source-only MLP search metrics without training or selecting anew.

This is a run-side arithmetic/lineage audit, not external origin verification.
It does not use project metric or source-selection functions.
"""
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix

root=Path(__file__).resolve().parents[2]
run=root/'outputs/multiround-20261031/neural-base'
checks=0;maximum=0.
def check(value,label):
    global checks
    checks+=1
    if not value:raise AssertionError(label)
def hash_file(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a,b,label):
    global maximum
    maximum=max(maximum,abs(float(a)-float(b)))
    check(abs(float(a)-float(b))<1e-12,label)

manifest=json.loads((run/'run_manifest.json').read_text())
for rel,digest in manifest['source_sha256'].items():check(hash_file(root/rel)==digest,'code:'+rel)
check(hash_file(run/'protocol_snapshot.json')==manifest['protocol_sha256'],'protocol')
plan=json.loads((root/'data/processed/zaoren_threeway_20261031_v1/split_plan.json').read_text())
truth={r['record_id']:r['origin_group_id'] for r in plan['records']}
outer={role:{r['record_id'] for r in plan['records'] if r['role']==role} for role in ('training','calibration','prediction')}
partitions=json.loads((run/'development_partitions.json').read_text())
classes=sorted(set(truth.values()))
lookup={c:i for i,c in enumerate(classes)}
for part in partitions:
    sets=[set(part[k]) for k in ('source','support_pool','query')]
    check([len(s) for s in sets]==[307,103,103],'rolecounts')
    check(set.union(*sets)==outer['training'],'allonlytraining')
    check(not any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)),'disjoint')
rows=json.loads((run/'all_development_metrics.json').read_text())
check(len(rows)==432,'developmentrows')
for row in rows:
    directory=run/'development'/f"episode_{row['episode']:02d}"/f"seed_{row['seed']}"/row['candidate_id']
    path=directory/'predictions.csv.gz'
    check(hash_file(path)==row['prediction_sha256'],'predictionhash')
    with gzip.open(path,'rt',encoding='utf-8',newline='') as f:records=list(csv.DictReader(f))
    check(len(records)==103,'predictioncount')
    expected=set(partitions[row['episode']]['query'])
    check({r['record_id'] for r in records}==expected,'queryIDs')
    y=[];p=[]
    for r in records:
        check(r['y_true']==truth[r['record_id']],'truth')
        check(int(r['is_correct'])==int(r['y_true']==r['y_predicted']),'flag')
        y.append(lookup[r['y_true']]);p.append(lookup[r['y_predicted']])
    close(row['accuracy'],accuracy_score(y,p),'accuracy')
    close(row['balanced_accuracy'],balanced_accuracy_score(y,p),'balanced')
    close(row['macro_f1'],f1_score(y,p,labels=list(range(7)),average='macro',zero_division=0),'macro')
    check(row['confusion_matrix']==confusion_matrix(y,p,labels=list(range(7))).tolist(),'confusion')
    meta=json.loads((directory/'training.json').read_text())
    check(set(meta['source_ids'])==set(partitions[row['episode']]['source']),'sourceIDs')
    check(hash_file(directory/'model.pt')==meta['model_sha256'],'weightsha')
    check(hash_file(directory/'preprocessor.joblib')==meta['preprocessor_sha256'],'preprocsha')
    check(meta['fingerprint']==row['model_fingerprint'],'sourcefingerprint')
    preprocessor=joblib.load(directory/'preprocessor.joblib')
    check(int(preprocessor['scaler'].n_samples_seen_)==307,'scalerfitcount')
    check(preprocessor['input_features']==612,'frequency')
grouped=defaultdict(list)
for row in rows:grouped[(row['candidate_id'],row['stage'])].append(row)
means={stage:{c:{m:float(np.mean([r[m] for r in group])) for m in ['accuracy','balanced_accuracy','macro_f1']}
              for (c,s),group in grouped.items() if s==stage} for stage in ['selection','gate']}
grid=json.loads((run/'candidate_grid.json').read_text());protocol=json.loads((run/'protocol_snapshot.json').read_text())
rules=protocol['selection'];ref=rules['reference_candidate_id']
def eligible(c,stage):
    a,b=means[stage][c['id']],means[stage][ref]
    return a['accuracy']-b['accuracy']>rules['minimum_accuracy_gain'] and a['macro_f1']-b['macro_f1']>=-rules['max_macro_f1_loss'] and a['balanced_accuracy']-b['balanced_accuracy']>=-rules['max_balanced_accuracy_loss']
def rank(candidates):
    best=max(means['selection'][c['id']]['accuracy'] for c in candidates)
    near=[c for c in candidates if best-means['selection'][c['id']]['accuracy']<=rules['accuracy_tolerance']]
    return sorted(near,key=lambda c:(-means['selection'][c['id']]['macro_f1'],-means['selection'][c['id']]['balanced_accuracy'],c['pca_components'] or 10**6,len(c['hidden_sizes']),c['epochs'],0 if c['class_weight']=='none' else 1,c['id']))[0]
viable=[c for c in grid if eligible(c,'selection')]
proposed=rank(viable or grid)
selected=proposed if viable and eligible(proposed,'gate') else next(c for c in grid if c['id']==ref)
lock=json.loads((run/'selection_lock.json').read_text())
check(lock['choice']['selected']==selected,'independentchoice')
check(lock['choice']['proposed']==proposed,'independentproposal')
for stage in means:
    for c in means[stage]:
        for metric,value in means[stage][c].items():close(value,lock['choice']['means'][stage][c][metric],'devmean')
check(not (run/'final').exists(),'no_final_before_all_round_locks')
result={'status':'pass','audit_type':'run-side independent arithmetic formulas, no training','checks_evaluated':checks,
        'development_evaluations':len(rows),'development_prediction_rows':432*103,'failed_checks':0,
        'maximum_numeric_difference':maximum,'independent_selected_candidate':selected,'independent_proposed_candidate':proposed,
        'selection_lock_sha256':hash_file(run/'selection_lock.json'),'final_prediction_scored':False,
        'audit_script_sha256':hash_file(Path(__file__)),'additional_model_fits':0}
(run/'development_arithmetic_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
