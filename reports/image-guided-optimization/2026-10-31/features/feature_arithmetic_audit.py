"""Read-only arithmetic/label/hash audit; no model fitting or project metrics."""
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
RUN=ROOT/'outputs/image-guided-20261031/features-v2'
V1=ROOT/'outputs/image-guided-20261031/features-v1-checkpoint-issue'
checks=0;maximum=0.


def check(condition,message):
    global checks
    checks+=1
    if not condition:raise AssertionError(message)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a,b,message):
    global maximum
    diff=float(np.max(np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))))
    maximum=max(maximum,diff);check(diff<1e-10,message)


def metrics(truth,prediction):
    cm=np.zeros((7,7),dtype=int)
    np.add.at(cm,(truth,prediction),1)
    d=cm.diagonal();actual=cm.sum(1);predicted=cm.sum(0)
    recall=np.divide(d,actual,out=np.zeros(7),where=actual>0)
    f1=np.divide(2*d,actual+predicted,out=np.zeros(7),where=(actual+predicted)>0)
    return {'accuracy':d.sum()/cm.sum(),'macro_f1':f1.mean(),'balanced_accuracy':recall[actual>0].mean(),
            'confusion_matrix':cm,'class_recall':recall,'class_counts':actual,'correct':int(d.sum()),'total':int(cm.sum())}


def main():
    started=time.monotonic()
    manifest=json.loads((RUN/'run_manifest.json').read_text())
    for p,sha in manifest['source_sha256'].items():check(digest(ROOT/p)==sha,'source SHA')
    check(digest(RUN/'protocol_snapshot.json')==manifest['protocol_sha256'],'protocol SHA')
    with np.load(ROOT/'outputs/approved-zaoren-20261031/data-preparation/dataset.npz',allow_pickle=False) as a:
        arrays={k:a[k] for k in a.files}
    ids=arrays['record_ids'];labels=arrays['y'];axis=arrays['axis'];band=(axis>=.1)&(axis<=1.5)
    X=arrays['X_alpha'][:,band];axis=axis[band]
    lookup={str(i):n for n,i in enumerate(ids)};classes=np.unique(labels);vocab={str(c):n for n,c in enumerate(classes)}
    truth={str(i):str(y) for i,y in zip(ids,labels)}
    plan=json.loads((ROOT/'data/processed/zaoren_threeway_20261031_v1/split_plan.json').read_text())
    training_ids={r['record_id'] for r in plan['records'] if r['role']=='training'}
    final_ids={r['record_id'] for r in plan['records'] if r['role']=='prediction'}
    training=np.asarray([n for n,i in enumerate(ids) if str(i) in training_ids]);partitions=[]
    orders=list(itertools.permutations(range(3)))
    for ep in range(6):
        rng=np.random.default_rng(20261103+ep);parts={k:[] for k in ['source','support_pool','query']}
        for c in classes:
            group=training[labels[training]==c];group=group[np.argsort(ids[group])]
            ordered=group[list(orders[ep%6])] if len(group)==3 else rng.permutation(group)
            source=max(1,int(.6*len(group)));support=max(1,int(.2*len(group)))
            parts['source'].extend(ordered[:source]);parts['support_pool'].extend(ordered[source:source+support]);parts['query'].extend(ordered[source+support:])
        partitions.append({k:ids[np.asarray(sorted(v))].tolist() for k,v in parts.items()})
    check(partitions==json.loads((RUN/'development_partitions.json').read_text()),'independent PCG partitions')
    raw_rows=0;case_rows=[];final_case_rows=[];artifact_count=0;warning_count=0;fallback_count=0
    cross_version_cases=0;cross_version_weight_bytes_equal=0;pca_roundtrips=0;dump_load_max=0.
    for directory in sorted((RUN/'development').glob('episode_*/seed_*'))+sorted((RUN/'final').glob('seed_*')):
        final=directory.parent.name=='final';context=json.loads((directory/'context.json').read_text())
        source_ids=context['source_ids'];query_ids=context['query_ids']
        check(not set(source_ids)&set(query_ids),'source/query disjoint')
        if final:
            check(set(source_ids)==training_ids,'final source train513');check(set(query_ids)==final_ids,'final query171')
        else:
            episode=int(directory.parent.name.split('_')[-1]);part=partitions[episode]
            check(source_ids==part['source'],'source307 IDs');check(query_ids==part['query'],'query103 IDs')
        source=np.asarray([lookup[str(i)] for i in source_ids]);sourceX=X[source]
        source_counts={str(c):int(np.sum(labels[source]==c)) for c in classes}
        check(digest(directory/'predictions.csv.gz')==context['predictions_sha256'],'raw CSV SHA')
        groups=defaultdict(list)
        with gzip.open(directory/'predictions.csv.gz','rt',encoding='utf-8') as f:
            for row in csv.DictReader(f):
                raw_rows+=1;check(row['record_id'] in query_ids,'known query ID')
                check(row['y_true']==truth[row['record_id']],'NPZ truth equality')
                check(int(row['is_correct'])==int(row['y_true']==row['y_predicted']),'correctness flag')
                groups[row['candidate_id']].append(row)
        for row in context['rows']:
            candidates=groups[row['candidate_id']];check(len(candidates)==len(query_ids),'one row per query')
            actual=np.asarray([vocab[r['y_true']] for r in candidates]);pred=np.asarray([vocab[r['y_predicted']] for r in candidates])
            expected=metrics(actual,pred)
            for k,v in expected.items():close(row[k],v,'CM-derived '+k)
            metadata=json.loads((directory/(row['candidate_id']+'.json')).read_text());fit=metadata['fit']
            check(metadata['expected']['source_ids']==source_ids,'model source IDs')
            check(metadata['expected']['candidate']['id']==row['candidate_id'],'candidate identity')
            check(fit['source_class_counts']==source_counts,'source class counts')
            check(digest(directory/(row['candidate_id']+'.joblib'))==metadata['model_sha256'],'source weight SHA');artifact_count+=1
            restored=joblib.load(directory/(row['candidate_id']+'.joblib'))
            check(joblib.hash(restored)==metadata['model_fingerprint'],'stable persisted object fingerprint')
            check(metadata['source_dump_load_labels_equal'],'source labels preserved on checkpoint roundtrip')
            dump_load_max=max(dump_load_max,metadata['source_dump_load_max_score_difference'])
            if fit['candidate']['family']=='pca_lda':pca_roundtrips+=1
            if not final:
                original_directory=V1/directory.relative_to(RUN)
                original=json.loads((original_directory/(row['candidate_id']+'.json')).read_text())
                check(original['fit']['selected_indices']==fit['selected_indices'],'v1/v2 feature indices same')
                original_row=next(r for r in json.loads((original_directory/'context.json').read_text())['rows'] if r['candidate_id']==row['candidate_id'])
                for k in ['accuracy','macro_f1','balanced_accuracy','selected_features','confusion_matrix']:
                    close(row[k],original_row[k],'v1/v2 unchanged '+k)
                cross_version_cases+=1
                if metadata['model_sha256']==original['model_sha256']:cross_version_weight_bytes_equal+=1
            idx=np.asarray(fit['selected_indices'],dtype=int)
            check(len(idx)==len(set(idx.tolist())) and np.all((idx>=0)&(idx<612)),'valid unique selected features')
            close(fit['selected_frequencies'],axis[idx],'original physical frequency mapping')
            if fit['candidate']['preprocessing']=='snv':
                processed=(sourceX-sourceX.mean(1,keepdims=True))/sourceX.std(1,ddof=1,keepdims=True)
            else:
                ref=sourceX.mean(0);close(fit['msc_reference'],ref,'source-only MSC reference')
                centered=ref-ref.mean();slopes=(sourceX-sourceX.mean(1,keepdims=True))@centered/(centered@centered)
                processed=(sourceX-(sourceX.mean(1)-slopes*ref.mean())[:,None])/slopes[:,None]
                close(fit['msc_source_slope_min'],slopes.min(),'source MSC slope min')
                close(fit['msc_source_slope_max'],slopes.max(),'source MSC slope max')
            close(fit['source_only_scaler_mean'],processed.mean(0),'source-only scaler mean')
            close(fit['source_only_scaler_variance'],processed.var(0),'source-only scaler variance')
            if fit['candidate']['family']=='noise_uve_svm':
                details=fit['selector'];real=np.asarray(details['real_stability']);noise=np.asarray(details['noise_stability'])
                threshold=np.quantile(noise,details['noise_quantile']);close(details['noise_threshold'],threshold,'noise UVE threshold')
                expected_idx=np.flatnonzero(real>threshold)
                if len(expected_idx)==0:expected_idx=np.asarray([int(np.argmax(real))]);fallback_count+=1
                check(np.array_equal(idx,expected_idx),'noise UVE kept variables')
            if fit['candidate']['family']=='cars_svm':
                policy=fit['selector']['internal_cv'];rare=[str(c) for c in classes if source_counts[str(c)]<3]
                local=[int(np.where(classes==c)[0][0]) for c in rare]
                check(policy['anchor_classes']==local,'CARS anchored rare classes')
                check(policy['validation_excludes_anchor_classes'],'CARS anchored val excludes rare')
            warning_count+=len(fit['warnings'])
            (final_case_rows if final else case_rows).append(row)
    saved=json.loads((RUN/'all_development_metrics.json').read_text())
    for directory in sorted((RUN/'development').glob('episode_*/seed_*')):
        original=V1/directory.relative_to(RUN)
        with gzip.open(directory/'predictions.csv.gz','rb') as f:new_bytes=f.read()
        with gzip.open(original/'predictions.csv.gz','rb') as f:old_bytes=f.read()
        check(new_bytes==old_bytes,'v1/v2 every raw prediction label and record byte equal')
    check(sorted(saved,key=lambda r:(r['episode'],r['seed'],r['candidate_id']))==sorted(case_rows,key=lambda r:(r['episode'],r['seed'],r['candidate_id'])),'all dev rows match contexts')
    grid=json.loads((RUN/'candidate_grid.json').read_text());reference=next(c for c in grid if c['preprocessing']=='snv' and c['family']=='full_svm' and c['C']==10.)
    aggregate={}
    for c in grid:
        for stage in ['selection','gate']:
            rows=[r for r in saved if r['candidate_id']==c['id'] and r['stage']==stage]
            aggregate[(c['id'],stage)]={k:float(np.mean([r[k] for r in rows])) for k in ['accuracy','macro_f1','balanced_accuracy','selected_features']}
    def passes(c,stage):
        a,b=aggregate[(c['id'],stage)],aggregate[(reference['id'],stage)]
        return a['macro_f1']-b['macro_f1']>.002 and a['accuracy']-b['accuracy']>=-.005 and a['balanced_accuracy']-b['balanced_accuracy']>=-.005
    def best(options):
        top=max(aggregate[(c['id'],'selection')]['macro_f1'] for c in options)
        tied=[c for c in options if aggregate[(c['id'],'selection')]['macro_f1']>=top-.002]
        return min(tied,key=lambda c:(-aggregate[(c['id'],'selection')]['accuracy'],aggregate[(c['id'],'selection')]['selected_features'],c.get('components',0),c.get('C',0),c['id']))
    eligible=[c for c in grid if passes(c,'selection')];proposed=best(eligible) if eligible else reference
    selected=proposed if eligible and passes(proposed,'gate') else reference
    choice=json.loads((RUN/'selection_lock.json').read_text())['choice']
    check(choice['selected']==selected,'independent source selection/gate')
    check(choice['proposed']==proposed,'independent source proposal')
    check(choice['eligible_selection_candidates']==len(eligible),'eligible count')
    for family,c in choice['family_diagnostics'].items():check(c==best([v for v in grid if v['family']==family]),'family diagnostic lock')
    if final_case_rows:
        by={(r['seed'],r['candidate_id']):r for r in final_case_rows}
        for row in json.loads((RUN/'final_metrics.json').read_text()):
            expected=by[(row['seed'],row['candidate_id'])]
            for k in ['accuracy','macro_f1','balanced_accuracy','selected_features']:close(row[k],expected[k],'final role metric')
        for row in json.loads((RUN/'summary.json').read_text()):
            cases=[r for r in json.loads((RUN/'final_metrics.json').read_text()) if r['method']==row['method']]
            for k in ['accuracy','macro_f1','balanced_accuracy']:close(row[k+'_mean'],np.mean([r[k] for r in cases]),'summary mean')
    result={'status':'pass','scope':'Read-only arithmetic, IDs, source-preprocessing and saved-weight hashes; no classifier fits',
        'checks_evaluated':checks,'maximum_numeric_difference':maximum,'development_cases':len(case_rows),'final_unique_cases':len(final_case_rows),
        'prediction_rows_checked':raw_rows,'source_model_file_hashes':artifact_count,'classifier_warnings_preserved':warning_count,
        'noise_uve_empty_fallback_count':fallback_count,'selected_candidate':selected,'new_model_fits':0,'seconds':time.monotonic()-started,
        'v1_v2_comparison_cases':cross_version_cases,'v1_v2_binary_model_files_byte_equal':cross_version_weight_bytes_equal,
        'pca_lda_roundtrips_verified':pca_roundtrips,'source_checkpoint_roundtrip_max_score_difference':dump_load_max,
        'audit_script_sha256':digest(__file__)}
    (RUN/'arithmetic_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
