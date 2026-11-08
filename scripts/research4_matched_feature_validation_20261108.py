"""TRAIN-only, nested matched-feature and modality-preprocessing validation.

Every fixed-K outer prediction is retained as an ablation. Only inner-fold
scores determine each family's primary K, kernel settings and modality weight.
This is exploratory development on previously used samples, not a fresh test.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import time
import uuid
import warnings

for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from herb_spectra.classification import sample_folds
from herb_spectra.selection import MCUVESelector
from herb_spectra.spectral_models import SpectralPreprocessor
from herb_spectra.staged_nested_search import gamma_spec

K_VALUES = (31,62,123,184,245)
C_VALUES = (.1,1.,10.,100.)
GAMMAS = ('scale','scale_x0.1','scale_x10')
WEIGHTS = (1.,.5,.25)
REPEATS = (20261031,20261108,42)
FAMILIES = {
    'alpha_raw20': (None,'raw','mc20'),
    'alpha_snv20': (None,'snv','mc20'),
    'n_raw20': ('raw',None,'mc20'),
    'n_snv20': ('snv',None,'mc20'),
    'concat_both_snv20': ('snv','snv','mc20'),
    'concat_nraw_asnv20': ('raw','snv','mc20'),
    'block_half_nraw_asnv20': ('raw','snv','block20'),
    'weighted_concat_nraw_asnv20': ('raw','snv','mc20'),
    'alpha_snv100': (None,'snv','mc100'),
    'alpha_snv_stability5x20': (None,'snv','rankvote5x20'),
}
AUXILIARY = ('fullalpha_shrinkLDA','fullalpha_MLP')
SCIENTIFIC_SOURCE_FILES = ('src/herb_spectra/selection.py','src/herb_spectra/spectral_models.py',
                          'src/herb_spectra/classification.py','src/herb_spectra/staged_nested_search.py')


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def array_sha(value):
    a=np.ascontiguousarray(value)
    return object_sha({'shape':list(a.shape),'dtype':a.dtype.str,'payload':hashlib.sha256(a.tobytes()).hexdigest()})


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)


def freeze(path, value):
    path=Path(path)
    if path.exists():need(read(path)==value,'Frozen job identity changed: '+str(path))
    else:atomic(path,value)


def checkpoint(path, identity, produce):
    """A successful immutable stage can be reused; failures are separate attempts."""
    path=Path(path);seal=path.with_name(path.name+'.receipt.json')
    need(not seal.exists() or path.exists(),'Sealed stage payload missing; do not silently regenerate')
    if path.exists() and seal.exists():
        proof=read(seal)
        need(proof=={'identity':identity,'artifact_sha256':sha(path)},'Checkpoint seal changed')
        value=read(path);need(value['identity']==identity,'Checkpoint identity changed')
        return value
    if path.exists():
        # A completed payload without its seal is not proof. Preserve it before
        # recomputing rather than replacing an unexplained prior output.
        orphan=path.with_name(path.name+'.unsealed-'+uuid.uuid4().hex)
        path.rename(orphan)
    attempt=uuid.uuid4().hex;started_cpu=time.process_time();started=time.monotonic()
    attempt_path=path.with_name(path.name+'.attempt-'+attempt+'.json')
    atomic(attempt_path,{'status':'RUNNING','identity':identity,'pid':os.getpid(),
                        'started_utc':datetime.now(timezone.utc).isoformat()})
    try:
        value=produce();value.update(identity=identity,actual_cpu_seconds=time.process_time()-started_cpu,
                                     actual_wall_seconds=time.monotonic()-started)
        atomic(path,value);atomic(seal,{'identity':identity,'artifact_sha256':sha(path)})
        atomic(attempt_path,{'status':'DONE','identity':identity,'actual_cpu_seconds':value['actual_cpu_seconds'],
                            'actual_wall_seconds':value['actual_wall_seconds'],'artifact_sha256':sha(path)})
        return value
    except BaseException as error:
        atomic(attempt_path,{'status':'FAILED_OR_INTERRUPTED','identity':identity,'error_type':type(error).__name__,
                            'error':str(error),'actual_cpu_seconds':time.process_time()-started_cpu,
                            'actual_wall_seconds':time.monotonic()-started})
        raise


def metric(y, prediction, classes):
    p,r,f,s=precision_recall_fscore_support(y,prediction,labels=classes,zero_division=0)
    return {'accuracy':float(accuracy_score(y,prediction)),
            'macro_f1':float(f1_score(y,prediction,labels=classes,average='macro',zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(y,prediction)),
            'correct':int(np.sum(np.asarray(y)==np.asarray(prediction))),'denominator':len(y),
            'class_order':list(map(str,classes)),'confusion_matrix':confusion_matrix(y,prediction,labels=classes).tolist(),
            'per_class':[{'label':str(c),'precision':float(pp),'recall':float(rr),'f1':float(ff),'support':int(ss)}
                         for c,pp,rr,ff,ss in zip(classes,p,r,f,s)]}


def load_inputs(features, split):
    """Reject full master inputs: this worker accepts the isolated999TRAIN pack."""
    with Path(split).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    need(len(rows)==999 and all(r.get('role')=='train' for r in rows),'Only isolated999TRAIN metadata accepted')
    need(len({r['record_id'] for r in rows})==999 and len({r['physical_sample_id'] for r in rows})==999,
         'Unique independently declared record/physical IDs required')
    need(Counter(r['species']for r in rows)=={'酸枣仁':300,'枣仁':216,'黄芪':241,'天麻':242},'Canonical corrected-kind TRAIN counts changed')
    selected={r['record_id']:r for r in rows if r['species']=='酸枣仁'}
    need(Counter(r['origin_group_id']for r in selected.values())=={c:60 for c in ['ZY03','ZY04','ZY05','ZY06','ZY07']},
         'Exactly five pure-acid source groups required')
    lock_path=Path(features).parent/'input_lock.json'
    need(lock_path.is_file(),'Frozen TRAIN-only input lock required')
    lock=read(lock_path)
    need(lock['training_records']==999 and lock['training_feature_sha256']==sha(features)
         and lock['training_split_sha256']==sha(split)
         and set(lock['training_allowlist'])=={r['record_id']for r in rows},'TRAIN input lock/hash/allowlist differs')
    with np.load(features,allow_pickle=False) as a:
        ids=a['record_ids'].astype(str);need(len(ids)==999 and len(set(ids))==999 and set(ids)=={r['record_id']for r in rows},'Only999TRAIN feature IDs accepted')
        ix=np.asarray([i for i,rid in enumerate(ids) if rid in selected]);axis=a['axis'].astype(float)
        mask=(axis>=.1)&(axis<=1.5)
        need(mask.sum()>=max(K_VALUES) and np.all(np.diff(axis[mask])>0),'Declared .1–1.5THz axis invalid')
        n=a['X_n'][ix][:,mask].astype(float);alpha=a['X_alpha'][ix][:,mask].astype(float);ids=ids[ix]
    need(n.shape==alpha.shape and np.isfinite(n).all() and np.isfinite(alpha).all(),'Paired finite source matrices required')
    y=np.asarray([selected[r]['origin_group_id']for r in ids]);physical=np.asarray([selected[r]['physical_sample_id']for r in ids])
    return n,alpha,y,ids,physical,axis[mask]


def preprocess(n, alpha, spec, axis):
    n_mode,a_mode,_=spec;blocks=[];global_indices=[];p=len(axis)
    if n_mode is not None:
        blocks.append(SpectralPreprocessor(n_mode,axis).transform(n));global_indices.extend(range(p))
    if a_mode is not None:
        blocks.append(SpectralPreprocessor(a_mode,axis).transform(alpha));global_indices.extend(range(p,2*p))
    return np.concatenate(blocks,axis=1),np.asarray(global_indices,dtype=int)


def one_ranking(z, y, runs, seed):
    selector=MCUVESelector(n_runs=runs,n_components=2,n_features_to_select=z.shape[1],
                          sample_fraction=.8,task='classification',random_state=seed)
    selector.fit(z,y)
    return selector.ranking_.copy(),selector.importances_.copy()


def fit_transform(n, alpha, y, ids, axis, spec):
    need(len(ids)==len(y)==len(n)==len(alpha) and len(set(map(str,ids)))==len(ids),'Fit/source identity mismatch')
    z,global_indices=preprocess(n,alpha,spec,axis);scaler=StandardScaler();z=scaler.fit_transform(z)
    mode=spec[2];p=len(axis);metadata={'fit_record_ids':list(map(str,ids)),
           'fit_record_ids_sha256':object_sha(list(map(str,ids))),'spec':list(spec),
           'seed':42,'sample_fraction':.8,'pls_components':2,'scaler_fit_source_only':True}
    if mode=='block20':
        n_rank,_=one_ranking(z[:,:p],y,20,42);a_rank,_=one_ranking(z[:,p:],y,20,42)
        ranks=[n_rank,a_rank];metadata.update(coefficient_draws=40,ovr_pls_fits=40*len(np.unique(y)),
                  definition='Two independent20-draw MC-UVE selectors, one per modality; Ksplit floor(n)/ceil(alpha)')
    elif mode=='rankvote5x20':
        vote=np.zeros(z.shape[1]);importance=np.zeros(z.shape[1]);seeds=[42+1009*i for i in range(5)]
        for seed in seeds:
            ranking,imp=one_ranking(z,y,20,seed);score=np.empty(len(ranking))
            score[ranking]=np.linspace(1.,0.,len(ranking));vote+=score/5;importance+=imp/5
        ranking=np.lexsort((np.arange(len(vote)),-importance,-vote));ranks=[ranking]
        metadata.update(coefficient_draws=100,ovr_pls_fits=100*len(np.unique(y)),independent_stream_seeds=seeds,
                        definition='Mean normalized within-stream Borda rank across5 independent MC20 streams; tie meanimportance then index')
    else:
        runs=100 if mode=='mc100' else 20;ranking,_=one_ranking(z,y,runs,42);ranks=[ranking]
        metadata.update(coefficient_draws=runs,ovr_pls_fits=runs*len(np.unique(y)),definition='Unchanged MC-UVE mean/stdev OVR-PLS coefficient norm')
    return {'mean':scaler.mean_,'scale':scaler.scale_,'global_indices':global_indices,'ranks':ranks,
            'spec':spec,'axis':axis,'metadata':metadata,'z_fit':z}


def chosen_indices(transform, k):
    need(type(k)is int and 1<=k<=len(transform['global_indices']),'Invalid absolute feature count')
    if transform['spec'][2]=='block20':
        p=len(transform['axis']);nn=k//2;aa=k-nn
        need(nn<=p and aa<=p,'Per-block feature budget exceeds modality width')
        return np.sort(np.concatenate((transform['ranks'][0][:nn],p+transform['ranks'][1][:aa]))).astype(int)
    return np.sort(transform['ranks'][0][:k]).astype(int)


def selected_matrix(transform, n, alpha, k, weight):
    need(np.isfinite(weight) and weight>0,'Positive finite modality weight required')
    raw,_=preprocess(n,alpha,transform['spec'],transform['axis'])
    standardized=(raw-transform['mean'])/transform['scale'];indices=chosen_indices(transform,k)
    # Weight AFTER train-fitted scaling and feature selection; never re-scale it.
    global_indices=transform['global_indices'][indices]
    weights=np.where(global_indices<len(transform['axis']),float(weight),1.)
    return standardized[:,indices]*weights,global_indices,weights


def transform_cache(folder, identity, n, alpha, y, ids, axis, spec):
    key=object_sha({'identity':identity,'fit_ids':list(map(str,ids)),'spec':spec,
                    'n':array_sha(n),'alpha':array_sha(alpha),'labels':list(map(str,y))})
    folder=Path(folder)/key;meta=folder/'metadata.json';arrays=folder/'transform.npz';seal=folder/'complete.json'
    if seal.exists():
        proof=read(seal);need(proof['identity']==key and proof['artifacts']=={'metadata.json':sha(meta),'transform.npz':sha(arrays)},'Transform cache seal changed')
        metadata=read(meta)
        with np.load(arrays,allow_pickle=False)as a:
            result={k:a[k].copy()for k in ['mean','scale','global_indices','axis']}
            result['ranks']=[a['rank%d'%i].copy()for i in range(metadata['rank_count'])]
        result.update(spec=tuple(metadata['spec']),metadata=metadata['fit_metadata'],cache_path=str(folder));return result
    folder.mkdir(parents=True,exist_ok=True)
    attempt=uuid.uuid4().hex;start=time.process_time();wall=time.monotonic()
    try:
        value=fit_transform(n,alpha,y,ids,axis,spec)
        tmp=folder/('transform.attempt-'+attempt+'.npz')
        np.savez_compressed(tmp,**{k:value[k]for k in ['mean','scale','global_indices','axis']},
                            **{'rank%d'%i:r for i,r in enumerate(value['ranks'])})
        # Never discard unsealed attempt arrays; only publish a separate final
        # snapshot after computing and checking the complete transform.
        if arrays.exists():arrays.rename(folder/('transform.unsealed-'+uuid.uuid4().hex+'.npz'))
        tmp.rename(arrays)
        atomic(meta,{'identity':key,'spec':list(spec),'rank_count':len(value['ranks']),
                     'fit_metadata':value['metadata'],'actual_cpu_seconds':time.process_time()-start,
                     'actual_wall_seconds':time.monotonic()-wall})
        atomic(seal,{'identity':key,'artifacts':{'metadata.json':sha(meta),'transform.npz':sha(arrays)}})
        value.pop('z_fit',None);value['cache_path']=str(folder);return value
    except BaseException as error:
        atomic(folder/('failed-'+attempt+'.json'),{'status':'FAILED_OR_INTERRUPTED','identity':key,
               'error_type':type(error).__name__,'error':str(error),'actual_cpu_seconds':time.process_time()-start,
               'actual_wall_seconds':time.monotonic()-wall});raise


def select_inner(rows):
    need(rows,'No inner candidates')
    groups={}
    for row in rows:
        config=row['config'];key=object_sha(config);groups.setdefault(key,{'config':config,'folds':[]})['folds'].append(row)
    summaries=[]
    for g in groups.values():
        need(sorted(x['inner_fold']for x in g['folds'])==[0,1],'Exactly two inner folds per candidate required')
        means={key:float(np.mean([r['metrics'][key]for r in g['folds']]))for key in ['macro_f1','balanced_accuracy','accuracy']}
        summaries.append({'config':g['config'],'mean_inner_metrics':means})
    def key(row):
        c=row['config'];m=row['mean_inner_metrics']
        return (-m['macro_f1'],-m['balanced_accuracy'],-m['accuracy'],c.get('k',99999),
                c.get('C',0.),GAMMAS.index(c['gamma'])if 'gamma'in c else 0,
                WEIGHTS.index(c.get('weight',1.)),c.get('hidden',0),c.get('preprocessing','raw'))
    return sorted(summaries,key=key)[0]


def svm_fit(config, train, y):
    gamma,resolved=gamma_spec(train,config['gamma'])
    model=SVC(C=config['C'],kernel='rbf',gamma=gamma,class_weight='balanced',
              decision_function_shape='ovr',break_ties=False,random_state=42)
    model.fit(train,y);return model,float(resolved)


def configurations(family):
    weights=WEIGHTS if family=='weighted_concat_nraw_asnv20'else(1.,)
    return [{'k':k,'C':C,'gamma':gamma,'weight':weight}for k in K_VALUES for C in C_VALUES
            for gamma in GAMMAS for weight in weights]


def write_gzip_csv(path, rows):
    need(rows,'No CSV rows to write');path=Path(path);tmp=path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
    with gzip.open(tmp,'wt',encoding='utf-8',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    tmp.replace(path)


def fit_auxiliary(config, train, y, query, axis):
    # All scalers and MLP early-stopping specimens come from current fit source.
    prep=SpectralPreprocessor(config['preprocessing'],axis);scaler=StandardScaler()
    z=scaler.fit_transform(prep.transform(train));q=scaler.transform(prep.transform(query))
    if config['family']=='fullalpha_shrinkLDA':model=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=np.full(len(np.unique(y)),1/len(np.unique(y))))
    else:model=MLPClassifier(hidden_layer_sizes=(config['hidden'],),alpha=.01,batch_size=32,max_iter=300,
             early_stopping=True,validation_fraction=.2,n_iter_no_change=15,random_state=42,learning_rate_init=.001)
    with warnings.catch_warnings(record=True)as caught:
        warnings.simplefilter('always',ConvergenceWarning);model.fit(z,y)
    return model.predict(q),{'model':model,'scaler':scaler,'preprocessing':config['preprocessing'],
                            'convergence_warnings':sum(issubclass(w.category,ConvergenceWarning)for w in caught),
                            'n_iter':int(model.n_iter_)if hasattr(model,'n_iter_')else None}


def run(args):
    start_cpu=time.process_time();start_wall=time.monotonic();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    n,alpha,y,ids,physical,axis=load_inputs(args.features,args.split);classes=np.unique(y)
    source,valid=sample_folds(y,physical,n_splits=5,seed=args.repeat_seed)[args.outer_fold]
    inners=sample_folds(y[source],physical[source],n_splits=2,seed=args.repeat_seed)
    manifest={'schema':'research4-matched-feature-nested-v1','source_code_sha256':sha(__file__),
      'module_sha256':{p:sha(ROOT/p)for p in SCIENTIFIC_SOURCE_FILES},'feature_file_sha256':sha(args.features),
      'split_file_sha256':sha(args.split),'repeat_seed':args.repeat_seed,'outer_fold':args.outer_fold,
      'input_lock_sha256':sha(Path(args.features).parent/'input_lock.json'),
      'model_seed':42,'frequency_interval':[.1,1.5],'axis_sha256':array_sha(axis),'features_per_modality':len(axis),
      'families':{f:list(v)for f,v in FAMILIES.items()},'auxiliary_families':list(AUXILIARY),
      'absolute_K':list(K_VALUES),'C':list(C_VALUES),'gamma':list(GAMMAS),'weights_after_scaling':list(WEIGHTS),
      'class_weight':'balanced','PLS_components':2,'MC_fraction':.8,
      'master_TRAIN_acid_ids':ids.tolist(),'source_record_ids':ids[source].tolist(),
      'outer_validation_record_ids':ids[valid].tolist(),
      'inner_partitions':[{'inner_fold':i,'fit_record_ids':ids[source[t]].tolist(),
                          'validation_record_ids':ids[source[v]].tolist()}for i,(t,v)in enumerate(inners)],
      'fit_and_selection_scope':'Only current source and inner-fit; no CAL/PRED; no prior global frequency ranking',
      'primary_K_selection':'Inner meanMacroF1 then BA/Accuracy; lowerK/C then declaredgamma/weight tie order',
      'outer_semantics':'SVC estimator.predict votes; all fixedK are ablations, not a source for primaryK selection',
      'validation_limit':'Adaptive internal development on previously used specimens, not new independent confirmation'}
    freeze(out/'job_manifest.json',manifest);identity=object_sha(manifest)
    if(out/'complete.json').exists():
        complete=read(out/'complete.json');need(complete['identity']==identity,'Completed identity mismatch')
        for name,h in complete['artifacts'].items():need(sha(out/name)==h,'Completed artifact changed')
        print(json.dumps({'status':'ALREADY_COMPLETE','identity':identity}));return
    all_inner=[];prediction_rows=[];feature_rows=[];family_results=[];locks={}
    for family,spec in FAMILIES.items():
        rows=[]
        for inner,(t,v)in enumerate(inners):
            fit=source[t];query=source[v]
            transform=transform_cache(out/'transforms',identity,n[fit],alpha[fit],y[fit],ids[fit],axis,spec)
            matrices={}
            for config in configurations(family):
                short=object_sha(config)[:20];cid=object_sha({'job':identity,'family':family,'inner':inner,'config':config})
                def produce(config=config):
                    key=(config['k'],config['weight'])
                    if key not in matrices:
                        z,_,_=selected_matrix(transform,n[fit],alpha[fit],config['k'],config['weight'])
                        q,_,_=selected_matrix(transform,n[query],alpha[query],config['k'],config['weight']);matrices[key]=(z,q)
                    z,q=matrices[key];model,resolved=svm_fit(config,z,y[fit]);pred=model.predict(q)
                    return {'config':config,'inner_fold':inner,'family':family,'metrics':metric(y[query],pred,classes),
                            'fit_ids_sha256':object_sha(ids[fit].tolist()),'query_ids_sha256':object_sha(ids[query].tolist()),
                            'transform_cache':str(Path(transform['cache_path']).relative_to(out)),'resolved_gamma':resolved}
                row=checkpoint(out/'checkpoints'/family/('inner%d-%s.json'%(inner,short)),cid,produce)
                rows.append(row);all_inner.append(row)
        per_k={str(k):select_inner([r for r in rows if r['config']['k']==k])for k in K_VALUES}
        primary=select_inner(rows)
        lock={'family':family,'fixedK_best_inner':per_k,'primary':primary,'outer_labels_used':False}
        freeze(out/'locks'/(family+'.json'),lock);locks[family]=lock
        transform=transform_cache(out/'transforms',identity,n[source],alpha[source],y[source],ids[source],axis,spec)
        for k in K_VALUES:
            choice=per_k[str(k)];config=choice['config'];eid=object_sha({'job':identity,'family':family,'outer_config':config})
            def evaluate(config=config,k=k):
                z,index,weights=selected_matrix(transform,n[source],alpha[source],k,config['weight'])
                q,_,_=selected_matrix(transform,n[valid],alpha[valid],k,config['weight'])
                model,resolved=svm_fit(config,z,y[source]);pred=model.predict(q)
                modeldir=out/'models';modeldir.mkdir(exist_ok=True)
                modelpath=modeldir/(family+'-K%d.joblib'%k)
                attempt=modelpath.with_name(modelpath.name+'.attempt-'+uuid.uuid4().hex)
                joblib.dump({'estimator':model,'selected_global_indices':index,'feature_weights_after_scaling':weights,
                             'transform_cache':str(Path(transform['cache_path']).relative_to(out)),
                             'source_ids':ids[source].tolist(),'config':config},attempt,compress=3)
                if modelpath.exists():modelpath.rename(modelpath.with_name(modelpath.name+'.unsealed-'+uuid.uuid4().hex))
                attempt.rename(modelpath)
                return {'family':family,'k':k,'config':config,'predictions':pred.tolist(),
                        'metrics':metric(y[valid],pred,classes),'selected_global_indices':index.tolist(),
                        'weights_after_scaling':weights.tolist(),'coefficient_draws':transform['metadata']['coefficient_draws'],
                        'ovr_pls_fits':transform['metadata']['ovr_pls_fits'],'resolved_gamma':resolved,
                        'source_ids_sha256':object_sha(ids[source].tolist()),'model_path':str(modelpath.relative_to(out)),
                        'model_sha256':sha(modelpath)}
            result=checkpoint(out/'outer'/family/('K%d.json'%k),eid,evaluate)
            need(sha(out/result['model_path'])==result['model_sha256'],'Outer saved model seal changed')
            is_primary=k==primary['config']['k'];family_results.append({**{z:v for z,v in result.items()if z not in ['predictions','selected_global_indices','weights_after_scaling']},
                 'is_primary_k':is_primary,'mean_inner_metrics':choice['mean_inner_metrics'],
                 'n_selected':sum(i<len(axis)for i in result['selected_global_indices']),
                 'alpha_selected':sum(i>=len(axis)for i in result['selected_global_indices'])})
            for rid,truth,pred in zip(ids[valid],y[valid],result['predictions']):
                prediction_rows.append({'family':family,'k':k,'is_primary_k':int(is_primary),'record_id':str(rid),
                  'physical_sample_id':str(rid),'y_true':str(truth),'prediction':str(pred),
                  'is_correct':int(truth==pred),'repeat_seed':args.repeat_seed,'outer_fold':args.outer_fold})
            for i,w in zip(result['selected_global_indices'],result['weights_after_scaling']):
                feature_rows.append({'family':family,'k':k,'is_primary_k':int(is_primary),'global_index':i,
                  'modality':'n'if i<len(axis)else'alpha','within_modality_index':i%len(axis),
                  'frequency_THz_export_label':float(axis[i%len(axis)]),'weight_after_scaling':w,
                  'repeat_seed':args.repeat_seed,'outer_fold':args.outer_fold})
        print(json.dumps({'event':'FAMILY_COMPLETE','family':family,'primaryK':primary['config']['k'],
                          'cpu_seconds':time.process_time()-start_cpu}),flush=True)
    # Lightweight full-alpha controls: all tuning and MLP early stopping remain
    # inside the same source, never current outer validation.
    for family in AUXILIARY:
        candidates=[{'family':family,'preprocessing':mode,**({'hidden':h}if family.endswith('MLP')else{})}
                    for mode in ('raw','snv')for h in ((32,64)if family.endswith('MLP')else(None,))]
        rows=[]
        for inner,(t,v)in enumerate(inners):
            fit=source[t];query=source[v]
            for config in candidates:
                cid=object_sha({'job':identity,'family':family,'inner':inner,'config':config})
                def produce(config=config):
                    pred,bundle=fit_auxiliary(config,alpha[fit],y[fit],alpha[query],axis)
                    return {'config':config,'inner_fold':inner,'family':family,'metrics':metric(y[query],pred,classes),
                            'fit_ids_sha256':object_sha(ids[fit].tolist()),'query_ids_sha256':object_sha(ids[query].tolist()),
                            'convergence_warnings':bundle['convergence_warnings'],'n_iter':bundle['n_iter']}
                row=checkpoint(out/'checkpoints'/family/('inner%d-%s.json'%(inner,object_sha(config)[:20])),cid,produce)
                rows.append(row);all_inner.append(row)
        primary=select_inner(rows);config=primary['config'];lock={'primary':primary,'outer_labels_used':False};freeze(out/'locks'/(family+'.json'),lock);locks[family]=lock
        def evaluate_auxiliary(config=config):
            pred,bundle=fit_auxiliary(config,alpha[source],y[source],alpha[valid],axis)
            modelpath=out/'models'/(family+'.joblib');tmp=modelpath.with_name(modelpath.name+'.attempt-'+uuid.uuid4().hex)
            joblib.dump(bundle,tmp,compress=3)
            if modelpath.exists():modelpath.rename(modelpath.with_name(modelpath.name+'.unsealed-'+uuid.uuid4().hex))
            tmp.rename(modelpath)
            return {'family':family,'k':len(axis),'config':config,'predictions':pred.tolist(),
                    'metrics':metric(y[valid],pred,classes),'model_path':str(modelpath.relative_to(out)),
                    'model_sha256':sha(modelpath),'convergence_warnings':bundle['convergence_warnings'],'n_iter':bundle['n_iter']}
        result=checkpoint(out/'outer'/family/'full.json',object_sha({'job':identity,'config':config}),evaluate_auxiliary)
        need(sha(out/result['model_path'])==result['model_sha256'],'Auxiliary model seal changed')
        family_results.append({**{z:v for z,v in result.items()if z!='predictions'},'is_primary_k':True,
                'mean_inner_metrics':primary['mean_inner_metrics'],'n_selected':0,'alpha_selected':len(axis)})
        for rid,truth,pred in zip(ids[valid],y[valid],result['predictions']):
            prediction_rows.append({'family':family,'k':len(axis),'is_primary_k':1,'record_id':str(rid),
              'physical_sample_id':str(rid),'y_true':str(truth),'prediction':str(pred),'is_correct':int(truth==pred),
              'repeat_seed':args.repeat_seed,'outer_fold':args.outer_fold})
    freeze(out/'selection_lock.json',{'identity':identity,'families':locks,'outer_labels_used':False})
    write_gzip_csv(out/'validation_predictions.csv.gz',prediction_rows);write_gzip_csv(out/'selected_features.csv.gz',feature_rows)
    inner_csv=[{'family':r['family'],'inner_fold':r['inner_fold'],'config_json':json.dumps(r['config'],sort_keys=True),
                'macro_f1':r['metrics']['macro_f1'],'balanced_accuracy':r['metrics']['balanced_accuracy'],
                'accuracy':r['metrics']['accuracy'],'fit_ids_sha256':r['fit_ids_sha256'],
                'query_ids_sha256':r['query_ids_sha256'],'actual_cpu_seconds':r['actual_cpu_seconds']}for r in all_inner]
    write_gzip_csv(out/'inner_search.csv.gz',inner_csv)
    atomic(out/'summary.json',{'schema':'research4-matched-feature-summary-v1','identity':identity,
       'scope':'acid300TRAIN_nested_development_only_not_fresh_test_not_crossbatch',
       'repeat_seed':args.repeat_seed,'outer_fold':args.outer_fold,'source_count':len(source),'validation_count':len(valid),
       'classes':classes.tolist(),'results':family_results,'actual_cpu_seconds_this_invocation':time.process_time()-start_cpu,
       'actual_wall_seconds_this_invocation':time.monotonic()-start_wall,'master_calibration_or_prediction_used':False,
       'feature_stability_definition':'Within-source 5x20 Borda-rank vote; no global outer-fold ranking reused',
       'primary_K_selection':'Only inner scores; fixedK outer rows retained as ablations, not selection candidates'})
    artifacts={str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()and p.name!='complete.json'
               and '.tmp.'not in p.name and '.unsealed-'not in p.name}
    # Include nested transform complete seals too, but not this final seal.
    for p in (out/'transforms').rglob('complete.json'):artifacts[str(p.relative_to(out))]=sha(p)
    atomic(out/'complete.json',{'status':'complete','identity':identity,'artifacts':artifacts,
                              'all_core_families':list(FAMILIES),'all_auxiliary_families':list(AUXILIARY),
                              'fixedK_not_outer_selected':True})
    print(json.dumps({'status':'DONE','identity':identity,'families':len(FAMILIES)+len(AUXILIARY),
                      'cpu_seconds':time.process_time()-start_cpu}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True);p.add_argument('--split',type=Path,required=True)
    p.add_argument('--repeat-seed',type=int,choices=REPEATS,required=True)
    p.add_argument('--outer-fold',type=int,choices=range(5),required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    def interrupted(number, frame):
        raise KeyboardInterrupt('received '+signal.Signals(number).name)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    with threadpool_limits(limits=1):run(args)


if __name__=='__main__':main()
