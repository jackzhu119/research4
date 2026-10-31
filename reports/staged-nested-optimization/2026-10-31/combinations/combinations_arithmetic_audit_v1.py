"""Read-only R4 arithmetic, source resampling and matched-R3 audit; no fits."""
from collections import defaultdict
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import time

import joblib
import numpy as np


ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'outputs/staged-nested-20261031/combinations'
CORE=ROOT/'outputs/staged-nested-20261031/core'
FAMILIES=('ordinary_mcuve','budget_mcuve','stable_mcuve','stable_cars','stable_joint','cnn_softmax','cnn_embedding_svm')
checks=0;maximum=0.


def check(condition,message):
    global checks
    checks+=1
    if not condition:raise AssertionError(message)


def close(a,b,message):
    global maximum
    delta=float(np.max(np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))))
    maximum=max(maximum,delta);check(delta<1e-9,message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric_rows(rows,vocabulary):
    truth=np.asarray([vocabulary[r['y_true']] for r in rows]);prediction=np.asarray([vocabulary[r['y_predicted']] for r in rows])
    cm=np.zeros((7,7),dtype=int);np.add.at(cm,(truth,prediction),1);d=cm.diagonal();a=cm.sum(1);b=cm.sum(0)
    recall=np.divide(d,a,out=np.zeros(7),where=a>0);f1=np.divide(2*d,a+b,out=np.zeros(7),where=a+b>0)
    return {'accuracy':float(d.sum()/cm.sum()),'balanced_accuracy':float(recall[a>0].mean()),
        'macro_f1':float(f1.mean()),'confusion_matrix':cm,'class_counts':a,'class_recall':recall}


def top(frequency,ranking,K):
    rank=np.empty(len(ranking),dtype=int);rank[ranking]=np.arange(len(ranking))
    return np.sort(np.lexsort((rank,-np.asarray(frequency)))[:K])


def verify_caller(directory,raw_lookup):
    common=json.loads((directory/'caller_common.json').read_text());checkpoint=json.loads((directory/'source_checkpoint.json').read_text())
    ids=common['source_ids'];sourceX=np.asarray([raw_lookup[i][0] for i in ids]);y=np.asarray([raw_lookup[i][1] for i in ids])
    check(sha(directory/'caller_common.json')==checkpoint['common_sha256'],'common source SHA')
    check(ids==checkpoint['expected']['source_ids'],'source ID declaration')
    check(joblib.hash(sourceX)==common['source_value_hash'],'source X numerical identity')
    check(joblib.hash(y)==common['source_label_hash'],'source Y numerical identity')
    rankings=np.asarray(common['ordinary_ranking']);importance=np.asarray(common['ordinary_importance'])
    check(np.array_equal(rankings,np.argsort(-importance,kind='stable')),'ordinary ranking from source stability')
    K=int(np.ceil(common['inherited']['mc']['ratio']*612));check(K==common['requested_selected_features'],'retainedceil ratio')
    stable=common['stability'];parts=stable['source_resamples'];rng=np.random.default_rng(42+9000)
    mc_sets=[];cars_sets=[];mc_counts=np.zeros(612);cars_counts=np.zeros(612)
    for repeat,part in enumerate(parts):
        positions=[]
        for c in np.unique(y):
            group=np.flatnonzero(y==c);take=min(len(group),max(2,int(np.ceil(.8*len(group)))))
            positions.extend(rng.permutation(group)[:take].tolist())
        positions=np.asarray(sorted(positions))
        check(part['repeat']==repeat and part['source_indices']==positions.tolist(),'recreate source resample indices')
        check(part['source_ids']==np.asarray(ids)[positions].tolist(),'recreate source resample IDs')
        check(len(set(part['source_ids']))==len(positions),'without replacement')
        counts={str(c):int(np.sum(y[positions]==c)) for c in np.unique(y)}
        check(part['class_counts']==counts and min(counts.values())>=2,'resample classes >=2 for internal CV2')
        a=np.asarray(part['mcuve_indices']);b=np.asarray(part['cars_indices']);mc_counts[a]+=1;cars_counts[b]+=1
        check(len(a)==K and len(set(a.tolist()))==K,'MC source support fixedK')
        best=min(range(12),key=lambda i:(part['cars_inner_mse_path'][i],len(part['cars_path_indices'][i])))
        check(part['cars_best_iteration']==best and part['cars_indices']==part['cars_path_indices'][best],'CARS source-only subset choice')
        mc_sets.append(a);cars_sets.append(b)
    close(stable['selection_frequencies']['stable_mcuve'],mc_counts/10,'MC frequency')
    close(stable['selection_frequencies']['stable_cars'],cars_counts/10,'CARS frequency')
    close(stable['selection_frequencies']['stable_joint'],.5*(mc_counts+cars_counts)/10,'joint frequency')
    for family in ['stable_mcuve','stable_cars','stable_joint']:
        frequency=np.asarray(stable['selection_frequencies'][family]);selected=top(frequency,rankings,K)
        check(stable['selected_indices'][family]==selected.tolist(),'fixedK tie ranking')
        check(stable['supplementation'][family]['zero_frequency_supplement_count']==int(np.sum(frequency[selected]==0)),'zero-frequency supplements')
    joint=[]
    for a,b in zip(mc_sets,cars_sets):
        count=np.zeros(612);count[a]+=.5;count[b]+=.5;joint.append(top(count,rankings,K))
    for family,sets in [('mcuve',mc_sets),('cars',cars_sets),('joint_resample_diagnostic',joint)]:
        expected=[]
        for i,j in itertools.combinations(range(10),2):
            left,right=set(sets[i]),set(sets[j]);expected.append(len(left&right)/len(left|right))
        detail=stable['pairwise_jaccard'][family]
        close([r['jaccard'] for r in detail['pairs']],expected,'Jaccard45')
        close(detail['mean'],np.mean(expected),'Jaccardmean')
    budget=stable['fit_budget'];runs=common['inherited']['mc']['runs']
    check(budget['budget_mcuve_coefficient_rounds']==10*runs==budget['stable_mcuve_coefficient_rounds'],'MC round budget control')
    check(budget['stable_cars_ovr_pls_fits']==840 and budget['stable_cars_multivariate_internal_cv_pls_fits']==240,'CARS extra fit budget explicit')
    check(budget['tie_breaker_extra_pls_fits']==0,'reuse ordinary source ranking')
    for family,artifact in checkpoint['artifacts'].items():
        check(sha(directory/artifact['file'])==artifact['sha256'],'model binary SHA')
        check(artifact['source_roundtrip_predictions_equal'],'checkpoint prediction roundtrip')
        check(artifact['source_roundtrip_max_score_difference']<1e-8,'checkpoint score roundtrip')
    check(sha(directory/'cnn_source_state.pt')==checkpoint['cnn_state_sha256'],'shared CNN weights SHA')
    neural=common['neural_training'];check(neural['epochs']==200 and len(neural['losses'])==200,'fixed CNN200')
    check(neural['seed']==42 and neural['parameter_count']==9223,'CNN fixed architecture/seed')
    check(checkpoint['cnn_source_training_invocations']==1,'single CNNfit sharedhead')
    return common


def main():
    started=time.monotonic();manifest=json.loads((RUN/'run_manifest.json').read_text())
    for p,digest in manifest['source_sha256'].items():check(sha(ROOT/p)==digest,'source code SHA')
    check(sha(RUN/'protocol_snapshot.json')==manifest['protocol_sha256'],'protocol SHA')
    with np.load(ROOT/'outputs/approved-zaoren-20261031/datasets/candidate_0p1_1p5.npz',allow_pickle=False) as a:
        raw={str(i):(x,y) for i,x,y in zip(a['groups'],a['X_alpha'],a['y'])};classes=np.unique(a['y'])
    vocabulary={str(c):n for n,c in enumerate(classes)}
    corelock=json.loads((CORE/'selection_lock.json').read_text());rows=[];raw_count=0;baseline_cases=[]
    for directory in sorted((RUN/'selection').glob('outer_*/inner_*')):
        outer=int(directory.parent.name.split('_')[-1]);inner=int(directory.name.split('_')[-1]);common=verify_caller(directory,raw)
        c=corelock['outer_folds'][outer]['choices'];expected={'preprocessing':c['R3']['selected']['preprocessing'],'mc':c['R1']['selected']['mc'],'svm':c['R2']['selected']['svm']}
        check(common['inherited']==expected,'per-outer independent inheritance')
        groups=defaultdict(list)
        with gzip.open(directory/'predictions.csv.gz','rt') as f:
            for row in csv.DictReader(f):
                raw_count+=1;check(row['record_id'] not in common['source_ids'],'validation not in source')
                check(row['y_true']==raw[row['record_id']][1],'truth matches canonical NPZ')
                check(int(row['is_correct'])==int(row['y_true']==row['y_predicted']),'correct flag')
                groups[row['family']].append(row)
        for row in json.loads((directory/'metrics.json').read_text()):
            predicted=groups[row['family']];m=metric_rows(predicted,vocabulary)
            check(len(predicted)==342,'inner validation342')
            for k in ['accuracy','balanced_accuracy','macro_f1','confusion_matrix']:close(row['metrics'][k],m[k],'inner metric '+k)
            rows.append(row)
        cid=c['R3']['selected']['id']
        with gzip.open(CORE/f'selection/outer_{outer:02d}/R3/predictions.csv.gz','rt') as f:
            matched={r['record_id']:r for r in csv.DictReader(f) if r['candidate_id']==cid and int(r['inner_fold'])==inner}
        original={r['record_id']:r for r in groups['ordinary_mcuve']};check(set(original)==set(matched),'matched R3 query IDs')
        check(all(original[i]['y_predicted']==matched[i]['y_predicted'] for i in original),'ordinary matched R3 predictions')
        score_diff=max(abs(float(original[i][k])-float(matched[i][k])) for i in original for k in original[i] if k.startswith('score_'))
        close(score_diff,0,'ordinary matched R3 decision scores');baseline_cases.append({'outer_fold':outer,'inner_fold':inner,'samples':342,'max_score_difference':score_diff})
    lock=json.loads((RUN/'selection_lock.json').read_text())
    for entry in lock['outer_folds']:
        local=[r for r in rows if r['outer_fold']==entry['outer_fold']]
        means={family:{k:float(np.mean([r['metrics'][k] for r in local if r['family']==family])) for k in ['macro_f1','balanced_accuracy','accuracy']} for family in FAMILIES}
        maximum=max(v['macro_f1'] for v in means.values());near=[f for f in FAMILIES if means[f]['macro_f1']>=maximum-1e-12]
        chosen=min(near,key=lambda f:(-means[f]['balanced_accuracy'],-means[f]['accuracy'],FAMILIES.index(f)))
        check(entry['choice']['selected']['family']==chosen,'independent R4inner winner')
        for family in FAMILIES:
            for k,v in means[family].items():close(entry['choice']['all_candidate_means'][family][k],v,'candidate inner mean')
    final_count=0;final_cases=0
    if (RUN/'outer_predictions.csv.gz').exists():
        for directory in sorted((RUN/'evaluation').glob('outer_*')):verify_caller(directory,raw)
        groups=defaultdict(list)
        with gzip.open(RUN/'outer_predictions.csv.gz','rt') as f:
            for row in csv.DictReader(f):
                final_count+=1;check(row['y_true']==raw[row['record_id']][1],'OOF truth')
                check(int(row['is_correct'])==int(row['y_true']==row['y_predicted']),'OOF correct')
                groups[(row['method'],int(row['outer_fold']))].append(row)
        for row in json.loads((RUN/'outer_metrics.json').read_text()):
            values=groups[(row['method'],row['outer_fold'])];expected=metric_rows(values,vocabulary);final_cases+=1
            check(len(values)==171,'outer test171')
            for k in ['accuracy','macro_f1','balanced_accuracy','confusion_matrix']:close(row[k],expected[k],'outer metric '+k)
        for row in json.loads((RUN/'summary.json').read_text()):
            values=sum([v for (method,_),v in groups.items() if method==row['method']],[])
            check(len(values)==855==len(set(r['record_id'] for r in values)),'855uniqueOOF')
            expected=metric_rows(values,vocabulary)
            for k in ['accuracy','macro_f1','balanced_accuracy','confusion_matrix']:close(row[k],expected[k],'pooled OOF '+k)
    result={'status':'pass','checks_evaluated':checks,'maximum_numeric_difference':maximum,'inner_metric_cases':len(rows),
        'inner_prediction_rows':raw_count,'outer_metric_cases':final_cases,'outer_prediction_rows':final_count,'matched_R3_inner':baseline_cases,
        'new_model_fits':0,'seconds':time.monotonic()-started,'script_sha256':sha(__file__)}
    name='final_arithmetic_audit.json' if final_count else 'selection_arithmetic_audit.json'
    target=RUN/name
    if target.exists() and not final_count:raise ValueError('Do not overwrite sealed selection audit')
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
