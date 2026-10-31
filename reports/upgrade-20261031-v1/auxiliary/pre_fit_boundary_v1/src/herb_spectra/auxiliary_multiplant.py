"""Separate four-source-group tasks using a new user-declaration overlay.

No original database, seven-class model, lineage or split is changed. Training
accepts only source features/labels; predictions are saved before truth joining.
"""
from dataclasses import dataclass
from datetime import datetime,timezone
import csv
import gzip
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
from torch import nn

from .classification import classification_metrics
from .leakage_guard import (FitScope,LeakageError,audit_pointwise_inference,
    validate_lineage_partitions,assert_modalities_aligned,instrument_mcuve_pls,
    guarded_cache_key,save_guarded_cache,load_guarded_cache)
from .selection import MCUVESelector
from .upgrade_neural import Encoder,PROFILES,row_snv,fixed_blocks,state_hash,object_hash


ROOT=Path(__file__).resolve().parents[2]
DATASETS=('huangqi','tianma')
FAMILIES=('full_svm','mcuve_svm','cnn','resnet','se','fusion')
NEURAL=FAMILIES[2:]
FUSIONS={'huangqi':('time','frequency'),'tianma':('alpha','n')}
OVERLAY='data/multiplant_database/metadata/user_declaration_overlays/20261031_huangqi_tianma_v1/overlay_manifest.json'


def sha(path):return __import__('hashlib').sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def event(out,name,**values):
    with (Path(out)/'stage_events.jsonl').open('a',encoding='utf-8') as h:
        h.write(json.dumps({'event':name,'utc':datetime.now(timezone.utc).isoformat(),**values})+'\n')


def grids():
    base=[{'C':C,'gamma_rule':g} for C in (1.,10.) for g in ('scale','scale_x0.1')]
    result={'full_svm':[{'id':f'full{i:02d}','kind':'full_svm',**p} for i,p in enumerate(base)]}
    result['mcuve_svm']=[{'id':f'mc{i:02d}','kind':'mcuve_svm',**p,'mc_runs':r,'mc_components':lv,'mc_ratio':ratio}
        for i,(r,lv,ratio,p) in enumerate((r,lv,ratio,p) for r,lv,ratio in [(10,1,.1),(20,2,.2)] for p in base)]
    for family in NEURAL:result[family]=[{**p,'kind':family} for p in PROFILES]
    return result


def default_protocol():
    overlay=json.loads((ROOT/OVERLAY).read_text());inputs={}
    for dataset in DATASETS:
        inputs[dataset]={}
        for mode in ('alpha',*FUSIONS[dataset]):
            p=ROOT/'data/multiplant_database'/dataset/(mode+'.npz')
            inputs[dataset][mode]={'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
    return {'protocol_id':'auxiliary-userdeclared-multiplant-20261031-v1','recorded_date_client':'2026-10-31',
        'overlay_manifest':OVERLAY,'overlay_manifest_sha256':sha(ROOT/OVERLAY),
        'overlay_lineage':overlay['overlay_lineage'],'overlay_lineage_sha256':overlay['overlay_lineage_sha256'],
        'annotation':overlay['annotation'],'annotation_sha256':overlay['annotation_sha256'],'inputs':inputs,
        'datasets':list(DATASETS),'class_orders':{d:overlay['datasets'][d]['class_order'] for d in DATASETS},
        'sample_counts':{'huangqi':403,'tianma':405},'outer_folds':5,'inner_folds':2,'split_seed':20261031,
        'seeds':[42,43,44],'primary_seed':42,'candidates':grids(),'fixed_alpha_n_band':[.1,1.5],
        'preprocessing':{'alpha':'same prior contiguous candidate window612, single-spectrum SNV then current-source StandardScaler',
         'n':'same prior candidate window612 then source StandardScaler','time':'fixed per-spectrum consecutive means to512 then source StandardScaler',
         'frequency':'fixed per-spectrum consecutive means to256 then source StandardScaler'},
        'fusions':{d:list(FUSIONS[d]) for d in DATASETS},'svm_class_weight':'balanced from current source only',
        'neural_selection':'Per dataset, outer fold and family separately: average2inner x3seed MacroF1, BalancedAccuracy, Accuracy, candidateID tie',
        'neural_architecture_alias':'inner_selected_neural chooses among the4neural families using inner means only; no extra fits',
        'full_svm_seeds':'all3 actually fitted but probability=False means seeds normally identical; not3independent studies',
        'mc_svm_seeds':'all3 stochastic seeds actually fitted and retained',
        'score_protocol':'save IDs/classes/predictions/scores without truth first; independent score command joins fixed truth',
        'pointwise_inference_tolerance':{'atol':1e-5,'rtol':1e-5},
        'blocked_paths':{'huangqi_alpha_n_fourclass':'Gansu n empty; cannot silently remove class',
         'tianma_allfour_fourclass':'time missing Guizhou/Anhui; cannot silently remove2classes',
         'cross_plant_transfer':'units/instrument/batch/common upstream material still unknown'},
        'scientific_status':'EXPLORATORY_USER_DECLARED_INDEPENDENCE_SOURCE_GROUPS','named_geography_externally_verified':False,
        'batch_ids_known':False,'independent_confirmation':False,'do_not_change_frozen_zaoren_or_database':True}


def read_pack(path,source=False):
    with np.load(path,allow_pickle=False) as z:payload={k:z[k] for k in z.files}
    allowed={'record_ids','X_alpha','X_n','X_time','X_frequency'}|({'source_labels'} if source else set())
    if set(payload)-allowed or source and 'source_labels' not in payload:raise LeakageError('Invalid source/inference pack schema')
    ids=payload.get('record_ids',[])
    if len(ids)!=len(set(ids.tolist())) or not len(ids):raise LeakageError('Nonempty unique sample IDs required')
    for k,x in payload.items():
        if k.startswith('X_') and (x.ndim!=2 or len(x)!=len(ids) or not np.issubdtype(x.dtype,np.number) or not np.isfinite(x).all()):
            raise LeakageError('Features must be finite paired spectra, never filenames or metadata')
    return payload


def features(pack):return {k[2:]:v for k,v in pack.items() if k.startswith('X_')}


class AuxiliaryNet(nn.Module):
    def __init__(self,kind,profile,modalities,n_classes=4):
        super().__init__();self.encoders=nn.ModuleDict({m:Encoder(profile['width'],profile['blocks'],profile['dropout'],
            residual=kind in {'resnet','se'},attention=kind=='se') for m in modalities})
        self.head=nn.Linear(16*len(modalities),n_classes)
    def forward(self,x):return self.head(torch.cat([net(x[m]) for m,net in self.encoders.items()],1))


@dataclass
class FittedAuxiliary:
    dataset_id:str
    candidate:dict
    classes:tuple
    scalers:dict
    selector:object
    estimator:object
    metadata:dict
    def transform(self,modalities):
        out={m:scaler.transform(row_snv(modalities[m]) if m=='alpha' else modalities[m]) for m,scaler in self.scalers.items()}
        if self.selector is not None:out['alpha']=out['alpha'][:,self.selector.selected_indices_]
        return out
    def predict_scores(self,modalities):
        data=self.transform(modalities)
        if self.candidate['kind'] in {'full_svm','mcuve_svm'}:return self.estimator.decision_function(data['alpha'])
        self.estimator.eval();before=state_hash(self.estimator)
        with torch.no_grad():out=self.estimator({k:torch.as_tensor(v,dtype=torch.float32) for k,v in data.items()}).cpu().numpy()
        if state_hash(self.estimator)!=before:raise LeakageError('Inference changed NN state')
        return out
    def predict(self,modalities):
        if self.candidate['kind'] in {'full_svm','mcuve_svm'}:return self.estimator.predict(self.transform(modalities)['alpha'])
        return np.asarray(self.classes)[self.predict_scores(modalities).argmax(1)]


def fit_candidate(modalities,labels,ids,dataset,candidate,seed,split_id,log_path,code_version,*,expected_source_ids):
    """Training receives no inner validation or outer test features/labels."""
    if dataset not in DATASETS:raise LeakageError('Unknown plant task; label spaces may not be mixed')
    ids=np.asarray(ids);labels=np.asarray(labels);prefix='HQ' if dataset=='huangqi' else 'TM'
    if list(ids)!=list(expected_source_ids):raise LeakageError('Actual fit IDs differ from the fixed allowed source IDs')
    classes=tuple(prefix+f'{i:02d}' for i in range(1,5))
    if tuple(np.unique(labels))!=classes or len(ids)!=len(labels):raise LeakageError('Four classes in a separate plant label space required')
    if len(ids)!=len(set(ids.tolist())):raise LeakageError('Repeated source IDs')
    chosen=FUSIONS[dataset] if candidate['kind']=='fusion' else ('alpha',)
    scope=FitScope(expected_source_ids,split_id,dataset+'-user-declaration-20261031-v1',log_path,code_version=code_version)
    standardized={};scalers={};start=time.perf_counter()
    for mode in chosen:
        X=row_snv(modalities[mode]) if mode=='alpha' else np.asarray(modalities[mode],float)
        if len(X)!=len(ids) or not np.isfinite(X).all():raise LeakageError('Source spectra must remain finite and paired')
        scaler=scope.fit(StandardScaler(),X,sample_ids=ids,component='source_scaler_'+mode)
        scalers[mode]=scaler;standardized[mode]=scaler.transform(X)
    selector=None
    if candidate['kind']=='mcuve_svm':
        selector=MCUVESelector(n_runs=candidate['mc_runs'],n_components=candidate['mc_components'],
            n_features_to_select=int(np.ceil(candidate['mc_ratio']*standardized['alpha'].shape[1])),
            sample_fraction=.8,task='classification',random_state=seed)
        with instrument_mcuve_pls(scope,standardized['alpha'],ids):
            scope.fit(selector,standardized['alpha'],labels,sample_ids=ids,component='MCUVESelector',groups=ids)
        standardized['alpha']=standardized['alpha'][:,selector.selected_indices_]
    prep_seconds=time.perf_counter()-start;start=time.perf_counter();losses=[]
    if candidate['kind'] in {'full_svm','mcuve_svm'}:
        z=standardized['alpha'];gamma='scale' if candidate['gamma_rule']=='scale' else .1/(z.shape[1]*z.var())
        estimator=SVC(C=candidate['C'],gamma=gamma,class_weight='balanced',kernel='rbf',random_state=seed,
                      probability=False,break_ties=False,decision_function_shape='ovr')
        scope.fit(estimator,z,labels,sample_ids=ids,component='SVC.fit');proof=estimator._leakage_fit_provenance
        extra={'effective_gamma':float(estimator._gamma),'source_class_weights':estimator.class_weight_.tolist(),
               'support_vectors':int(estimator.n_support_.sum()),'selected_features':z.shape[1],
               'standard_vote_preserved':True,'parameters':None,'epochs':None}
    else:
        torch.manual_seed(seed);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        estimator=AuxiliaryNet(candidate['kind'],candidate,chosen,4)
        initial=state_hash(estimator);optimizer=torch.optim.Adam(estimator.parameters(),lr=candidate['learning_rate'],weight_decay=.001)
        encoded=np.asarray([classes.index(v) for v in labels]);counts=np.bincount(encoded,minlength=4)
        weights=len(labels)/(4*counts) if candidate['class_weight']=='balanced' else np.ones(4)
        criterion=nn.CrossEntropyLoss(weight=torch.as_tensor(weights,dtype=torch.float32))
        neural_features={k:np.asarray(v,dtype=np.float32) for k,v in standardized.items()}
        tensors={k:torch.as_tensor(v,dtype=torch.float32) for k,v in neural_features.items()};target=torch.as_tensor(encoded,dtype=torch.long)
        generator=torch.Generator().manual_seed(seed)
        with scope.operation('torch_training',ids,X=neural_features,y=encoded) as proof:
            for epoch in range(candidate['epochs']):
                estimator.train();order=torch.randperm(len(ids),generator=generator);total=0.
                for left in range(0,len(ids),64):
                    selected=order[left:left+64];optimizer.zero_grad(set_to_none=True)
                    loss=criterion(estimator({k:v[selected] for k,v in tensors.items()}),target[selected]);loss.backward();optimizer.step();total+=float(loss.detach())*len(selected)
                losses.append(total/len(ids))
        estimator.eval();extra={'initial_state_sha256':initial,'fitted_state_sha256':state_hash(estimator),
            'parameters':sum(p.numel() for p in estimator.parameters()),'epochs':candidate['epochs'],
            'class_weights':weights.tolist(),'loss_curve':losses,'optimizer_initialized_fresh':True,'early_stopping':False,'selected_features':None}
    metadata={'dataset_id':dataset,'candidate':candidate,'classes':list(classes),'source_ids':ids.tolist(),'seed':seed,'split_id':split_id,
        'code_version':code_version,'source_class_counts':{c:int(np.count_nonzero(labels==c)) for c in classes},
        'raw_source_fingerprints':{k:joblib.hash(modalities[k]) for k in chosen},'scaler_statistics':{k:{'mean':s.mean_.tolist(),'variance':s.var_.tolist()} for k,s in scalers.items()},
        'transform_fit_seconds':prep_seconds,'estimator_fit_seconds':time.perf_counter()-start,'fit_scope_entries':len(scope.entries),
        'selected_indices':None if selector is None else selector.selected_indices_.tolist(),'query_truth_received':False,**extra}
    fitted=FittedAuxiliary(dataset,dict(candidate),classes,scalers,selector,estimator,metadata)
    fitted._leakage_fit_provenance=proof;return fitted


def save_fitted(model,path):
    m=model.metadata;p=model._leakage_fit_provenance
    identity=dict(dataset_version=p['dataset_version'],train_ids=m['source_ids'],split_id=m['split_id'],
        preprocessing={'raw_source_fingerprints':m['raw_source_fingerprints'],'kind':'per_spectrum_alpha_SNV_then_source_scaler'},
        selector=m['candidate'],code_version=m['code_version'],seed=m['seed'],data_fingerprint=p['data_fingerprint'])
    key=guarded_cache_key(**identity);save_guarded_cache(path,model,key)
    write_json(Path(path).with_suffix('.json'),{**m,'cache_identity':identity,'cache_key':str(key),'pipeline_sha256':sha(path)})


def load_fitted(path,expected_split=None,expected_ids=None):
    m=json.loads(Path(path).with_suffix('.json').read_text())
    if expected_split is not None and m['split_id']!=expected_split:raise LeakageError('Cross-fold weight rejected')
    if expected_ids is not None and list(expected_ids)!=m['source_ids']:raise LeakageError('Cross-source identity rejected')
    if sha(path)!=m['pipeline_sha256']:raise LeakageError('Stored pipeline changed')
    return load_guarded_cache(path,guarded_cache_key(**m['cache_identity']))


def write_predictions(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt',encoding='utf-8',newline='') as h:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        w=csv.DictWriter(h,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)


def read_predictions(path):
    with gzip.open(path,'rt',encoding='utf-8',newline='') as h:return list(csv.DictReader(h))


def read_truth(path):
    with Path(path).open(encoding='utf-8',newline='') as h:return {r['record_id']:r['label'] for r in csv.DictReader(h)}


def metrics(rows,truth,classes):
    ids=[r['record_id'] for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(truth):raise LeakageError('Duplicate or changed scoring denominator')
    value=classification_metrics(np.asarray([truth[r] for r in ids]),np.asarray([r['y_predicted'] for r in rows]),np.asarray(classes))
    matrix=np.asarray(value['confusion_matrix']);tp=matrix.diagonal();actual=matrix.sum(1);found=matrix.sum(0)
    value['per_class']={c:{'precision':float(t/p) if p else 0.,'recall':float(t/n) if n else 0.,'f1':float(2*t/(p+n)) if p+n else 0.,'support':int(n)} for c,t,p,n in zip(classes,tp,found,actual)}
    value.update(correct=int(tp.sum()),total=int(matrix.sum()));return value


def prepare(protocol_path,out):
    out=Path(out);protocol=json.loads(Path(protocol_path).read_text())
    if out.exists() and any(out.iterdir()):raise FileExistsError('Fresh output directory required')
    out.mkdir(parents=True);overlay=json.loads((ROOT/protocol['overlay_manifest']).read_text())
    for k in ['overlay_manifest','overlay_lineage','annotation']:
        if sha(ROOT/protocol[k])!=protocol[k+'_sha256']:raise LeakageError('User-declaration provenance changed')
    with (ROOT/protocol['overlay_lineage']).open(encoding='utf-8-sig',newline='') as h:lineage=list(csv.DictReader(h))
    contexts=[];transform_notes={};splits={}
    for dataset in DATASETS:
        classes=protocol['class_orders'][dataset];payloads={}
        for mode,entry in protocol['inputs'][dataset].items():
            if sha(ROOT/entry['path'])!=entry['sha256']:raise LeakageError('Frozen raw modality changed')
            with np.load(ROOT/entry['path'],allow_pickle=False) as z:payloads[mode]={k:z[k] for k in z.files}
        ids=payloads['alpha']['record_ids'].astype(str);labels=payloads['alpha']['filename_groups'].astype(str)
        if len(ids)!=protocol['sample_counts'][dataset] or tuple(np.unique(labels))!=tuple(classes):raise LeakageError('Four-source-group cohort changed')
        modalities={};transform_notes[dataset]={}
        for mode,p in payloads.items():
            if set(p['record_ids'])!=set(ids):raise LeakageError('Missing full-cohort modality; cannot silently delete classes')
            order={str(v):i for i,v in enumerate(p['record_ids'])};X=p['X'][[order[v] for v in ids]];axis=p['axis']
            if mode in {'alpha','n'}:mask=(axis>=.1)&(axis<=1.5);X=X[:,mask];axis=axis[mask];note='prior fixed candidate band0.1-1.5'
            else:X,axis,blocks=fixed_blocks(X,axis,512 if mode=='time' else 256);note={'fixed_single_spectrum_blocks':blocks}
            if not np.isfinite(X).all():raise LeakageError('Nonfinite modality requires a new declared missing-data protocol')
            modalities[mode]=X;transform_notes[dataset][mode]={'points':X.shape[1],'axis':axis.tolist(),'operation':note,'learned_statistics':False}
        assert_modalities_aligned(ids,{m:ids for m in modalities})
        rows=[r for r in lineage if r['dataset_id']==dataset];byid={r['record_id']:r for r in rows}
        if set(ids)!=set(byid) or any(byid[i]['label']!=label for i,label in zip(ids,labels,strict=True)):raise LeakageError('Overlay and raw source groups disagree')
        outer=StratifiedKFold(5,shuffle=True,random_state=protocol['split_seed']);dataset_splits=[]
        def pack(path,index,source):
            path.parent.mkdir(parents=True,exist_ok=True);v={'record_ids':ids[index],**{'X_'+m:X[index] for m,X in modalities.items()}}
            if source:v['source_labels']=labels[index]
            np.savez_compressed(path,**v);return sha(path)
        def truth(path,index):
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('w',encoding='utf-8',newline='') as h:
                w=csv.DictWriter(h,fieldnames=['record_id','label'],lineterminator='\n');w.writeheader();w.writerows({'record_id':str(ids[i]),'label':str(labels[i])} for i in index)
        for fold,(train,test) in enumerate(outer.split(ids,labels)):
            inner=StratifiedKFold(2,shuffle=True,random_state=protocol['split_seed']+fold)
            roles=[('inner',j,train[it],train[iv]) for j,(it,iv) in enumerate(inner.split(ids[train],labels[train]))]
            roles.append(('outer',None,train,test));inner_rows=[]
            for stage,number,source,inference in roles:
                sid=f'{dataset}.outer{fold:02d}.'+(f'inner{number:02d}' if stage=='inner' else 'final')
                partitions={'source':ids[source].tolist(),'inference':ids[inference].tolist()}
                if stage=='inner':partitions['forbidden_outer_test']=ids[test].tolist()
                proof=validate_lineage_partitions(rows,partitions,allow_user_declared=True)
                if proof['status']!='EXPLORATORY':raise LeakageError('User declaration must remain exploratory, never independently verified')
                if any(len(np.unique(labels[index]))!=4 for index in (source,inference)):raise LeakageError('Class missing in a saved role')
                directory=out/'inputs'/sid;sp=directory/'source.npz';ip=directory/'inference.npz';tp=out/'scoring_truth'/(sid+'.csv')
                sh=pack(sp,source,True);ih=pack(ip,inference,False);truth(tp,inference)
                context={'dataset_id':dataset,'split_id':sid,'stage':stage,'outer_fold':fold,'inner_fold':number,'source_pack':str(sp),'source_pack_sha256':sh,
                         'inference_pack':str(ip),'inference_pack_sha256':ih,'scoring_truth':str(tp),'source_ids':ids[source].tolist(),'inference_ids':ids[inference].tolist(),'lineage_status':proof['status']}
                contexts.append(context)
                if stage=='inner':inner_rows.append({'inner_fold':number,'train_ids':ids[source].tolist(),'validation_ids':ids[inference].tolist()})
            dataset_splits.append({'outer_fold':fold,'train_ids':ids[train].tolist(),'test_ids':ids[test].tolist(),'inner_folds':inner_rows})
        splits[dataset]=dataset_splits
    write_json(out/'split_manifest.json',{'protocol_id':protocol['protocol_id'],'split_seed':protocol['split_seed'],'datasets':splits,'no_seed_search':True,'independent_confirmation':False})
    prep={'created_at_utc':datetime.now(timezone.utc).isoformat(),'contexts':contexts,'transforms':transform_notes,'protocol_sha256':sha(protocol_path),
          'overlay_manifest_sha256':sha(ROOT/protocol['overlay_manifest']),'overlay_lineage_sha256':sha(ROOT/protocol['overlay_lineage']),
          'labels_excluded_from_inference_packs':True,'OS_enforced_file_isolation':False,'study_status':'EXPLORATORY_USER_DECLARED_INDEPENDENCE'}
    write_json(out/'preparation.json',prep);write_json(out/'protocol_snapshot.json',protocol);return {'contexts':len(contexts),'datasets':2,'status':prep['study_status']}


def manifest(protocol_path,out):
    names=['src/herb_spectra/auxiliary_multiplant.py','scripts/run_auxiliary_multiplant.py','src/herb_spectra/upgrade_neural.py',
           'src/herb_spectra/leakage_guard.py','src/herb_spectra/selection.py','src/herb_spectra/classification.py']
    return {'protocol_sha256':sha(protocol_path),'preparation_sha256':sha(Path(out)/'preparation.json'),
            'source_sha256':{n:sha(ROOT/n) for n in names},'versions':{k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn','torch','joblib']}}


def trial(context,family,candidate,seed,out,code_version,outer=False):
    for k in ['source_pack','inference_pack']:
        if sha(context[k])!=context[k+'_sha256']:raise LeakageError('Prepared pack changed')
    source=read_pack(context['source_pack'],True);query=read_pack(context['inference_pack'])
    directory=Path(out)/('evaluation' if outer else 'selection')/context['split_id']/family/candidate['id']/f'seed{seed}'
    if directory.exists():raise FileExistsError('Existing trial may not be overwritten')
    directory.mkdir(parents=True);start=time.perf_counter()
    fitted=fit_candidate(features(source),source['source_labels'],source['record_ids'],context['dataset_id'],candidate,seed,context['split_id'],Path(out)/'fit_scope_events.jsonl',code_version,expected_source_ids=context['source_ids'])
    if family in NEURAL:
        x=fitted.transform(features(query));n=len(query['record_ids']) if outer else min(7,len(query['record_ids']))
        audit=audit_pointwise_inference(fitted.estimator,{k:v[:n] for k,v in x.items()},batch_sizes=(1,3),atol=1e-5,rtol=1e-5)
    else:
        before=joblib.hash(fitted.estimator);x=features(query);a=fitted.predict_scores(x)
        reverse={k:v[::-1].copy() for k,v in x.items()};b=fitted.predict_scores(reverse)[::-1]
        n=len(a) if outer else min(7,len(a))
        single=np.concatenate([fitted.predict_scores({k:v[i:i+1] for k,v in x.items()}) for i in range(n)])
        chunk=np.concatenate([fitted.predict_scores({k:v[i:i+3] for k,v in x.items()}) for i in range(0,n,3)])
        if not all(np.allclose(z,a[:n],atol=1e-12,rtol=1e-12) for z in [single,chunk]) or not np.allclose(a,b,atol=1e-12,rtol=1e-12) or joblib.hash(fitted.estimator)!=before:
            raise LeakageError('SVM inference changed state, order or batch composition')
        audit={'status':'TECHNICAL_CHECK_PASS','samples_checked':n,'order_maximum_difference':float(np.max(np.abs(a-b))),
               'single_maximum_difference':float(np.max(np.abs(a[:n]-single))),'batch3_maximum_difference':float(np.max(np.abs(a[:n]-chunk))),
               'state_unchanged':True,'vote_rule':'standard_SVC.predict','atol':1e-12,'rtol':1e-12}
    path=directory/'pipeline.joblib';save_fitted(fitted,path)
    scores=fitted.predict_scores(features(query));prediction=fitted.predict(features(query))
    common={k:context[k] for k in ['dataset_id','outer_fold','inner_fold']};common.update(family=family,candidate_id=candidate['id'],seed=seed)
    rows=[{**common,'record_id':str(rid),'y_predicted':str(p),**{'score_'+c:float(v) for c,v in zip(fitted.classes,score)}} for rid,p,score in zip(query['record_ids'],prediction,scores,strict=True)]
    pred=directory/'predictions.csv.gz';write_predictions(pred,rows);write_json(directory/'inference_audit.json',audit)
    record={**common,'candidate':candidate,'pipeline':str(path),'pipeline_sha256':sha(path),'predictions':str(pred),'predictions_sha256':sha(pred),
            'source_ids':source['record_ids'].tolist(),'inference_ids':query['record_ids'].tolist(),'metadata':fitted.metadata,
            'elapsed_seconds':time.perf_counter()-start,'outer_truth_read':False,'inference_audit':audit}
    return record,rows


def selection(protocol_path,out,release_path):
    out=Path(out);protocol=json.loads(Path(protocol_path).read_text());prep=json.loads((out/'preparation.json').read_text());current=manifest(protocol_path,out)
    release=json.loads(Path(release_path).read_text())
    if not release.get('allow_source_training') or any(release.get(k)!=current[k] for k in ['protocol_sha256','preparation_sha256','source_sha256']):
        raise LeakageError('Root source authorization must bind protocol, packs and all implementation hashes')
    if (out/'selection_lock.json').exists():raise FileExistsError('Selection already locked')
    write_json(out/'run_manifest.json',current);write_json(out/'source_release_snapshot.json',release);event(out,'selection_started');records=[];predictions=[]
    code=current['source_sha256']['src/herb_spectra/auxiliary_multiplant.py']
    for context in [c for c in prep['contexts'] if c['stage']=='inner']:
        for family,candidates in protocol['candidates'].items():
            for candidate in candidates:
                for seed in protocol['seeds']:
                    record,rows=trial(context,family,candidate,seed,out,code)
                    record['metrics']=metrics(rows,read_truth(context['scoring_truth']),protocol['class_orders'][context['dataset_id']]);records.append(record);predictions.extend(rows)
        write_json(out/'selection_progress.json',{'completed_context':context['split_id'],'trials':len(records),'outer_scored':False})
        print(json.dumps({'stage':'selection','context':context['split_id'],'trials':len(records)}),flush=True)
    choices=[]
    for dataset in DATASETS:
        for fold in range(5):
            selected={}
            for family,candidates in protocol['candidates'].items():
                means=[]
                for candidate in candidates:
                    subset=[r for r in records if r['dataset_id']==dataset and r['outer_fold']==fold and r['family']==family and r['candidate_id']==candidate['id']]
                    if len(subset)!=2*len(protocol['seeds']):raise LeakageError('Missing inner fold or seed')
                    means.append({'candidate':candidate,'cases':len(subset),'mean_inner_metrics':{m:float(np.mean([r['metrics'][m] for r in subset])) for m in ['macro_f1','balanced_accuracy','accuracy']}})
                best=min(means,key=lambda r:(-r['mean_inner_metrics']['macro_f1'],-r['mean_inner_metrics']['balanced_accuracy'],-r['mean_inner_metrics']['accuracy'],r['candidate']['id']))
                selected[family]={'selected':best['candidate'],'mean_inner_metrics':best['mean_inner_metrics'],'means':means,'outer_test_used_to_select':False}
            nnchoice=min(({'family':f,**r} for f in NEURAL for r in selected[f]['means']),key=lambda r:(-r['mean_inner_metrics']['macro_f1'],-r['mean_inner_metrics']['balanced_accuracy'],-r['mean_inner_metrics']['accuracy'],r['family'],r['candidate']['id']))
            choices.append({'dataset_id':dataset,'outer_fold':fold,'choices':selected,'inner_selected_neural':{'family':nnchoice['family'],'candidate':nnchoice['candidate'],'mean_inner_metrics':nnchoice['mean_inner_metrics'],'outer_test_used_to_select':False}})
    write_json(out/'inner_metrics.json',records);write_predictions(out/'inner_predictions.csv.gz',predictions)
    lock={'created_at_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':sha(out/'run_manifest.json'),'protocol_sha256':sha(protocol_path),
          'inner_metrics_sha256':sha(out/'inner_metrics.json'),'inner_predictions_sha256':sha(out/'inner_predictions.csv.gz'),'outer_folds':choices,'inner_trials':len(records),'outer_test_used_to_select':False}
    write_json(out/'selection_lock.json',lock);event(out,'selection_locked',selection_lock_sha256=sha(out/'selection_lock.json'))
    return {'inner_trials':len(records),'inner_prediction_rows':len(predictions),'selection_lock_sha256':sha(out/'selection_lock.json'),'outer_scored':False}


def evaluation(protocol_path,out,release_path):
    out=Path(out);protocol=json.loads(Path(protocol_path).read_text());current=manifest(protocol_path,out)
    if current!=json.loads((out/'run_manifest.json').read_text()):raise LeakageError('Frozen implementation or preparation changed')
    release=json.loads(Path(release_path).read_text());expected={'selection_lock_sha256':sha(out/'selection_lock.json'),'protocol_sha256':sha(protocol_path),'manifest_sha256':sha(out/'run_manifest.json')}
    if not release.get('allow_outer_evaluation') or not release.get('released_at_utc') or any(release.get(k)!=v for k,v in expected.items()):raise LeakageError('Root final release must bind all frozen evidence')
    if (out/'outer_predictions.csv.gz').exists():raise FileExistsError('Outer evaluation already exists')
    lock=json.loads((out/'selection_lock.json').read_text());prep=json.loads((out/'preparation.json').read_text())
    if sha(out/'inner_metrics.json')!=lock['inner_metrics_sha256'] or sha(out/'inner_predictions.csv.gz')!=lock['inner_predictions_sha256']:raise LeakageError('Frozen inner evidence changed')
    write_json(out/'outer_release_snapshot.json',release);event(out,'outer_inference_started');records=[];predictions=[]
    code=current['source_sha256']['src/herb_spectra/auxiliary_multiplant.py']
    for context in [c for c in prep['contexts'] if c['stage']=='outer']:
        choice=next(c for c in lock['outer_folds'] if c['dataset_id']==context['dataset_id'] and c['outer_fold']==context['outer_fold'])
        by_role={}
        for family in FAMILIES:
            for seed in protocol['seeds']:
                record,rows=trial(context,family,choice['choices'][family]['selected'],seed,out,code,outer=True)
                records.append(record);predictions.extend(rows);by_role[family,seed]=rows
        selected=choice['inner_selected_neural']
        for seed in protocol['seeds']:
            predictions.extend([{**r,'family':'inner_selected_neural','architecture_family':selected['family']} for r in by_role[selected['family'],seed]])
        print(json.dumps({'stage':'evaluation','context':context['split_id'],'models':len(records)}),flush=True)
    # Normalize a field added only to the alias, before any truth is read.
    for r in predictions:r.setdefault('architecture_family',r['family'])
    write_json(out/'outer_model_records.json',records);write_predictions(out/'outer_predictions.csv.gz',predictions)
    event(out,'outer_inference_completed',prediction_rows=len(predictions),outer_truth_read=False)
    return {'outer_models':len(records),'prediction_rows':len(predictions),'outer_truth_read':False}


def score(protocol_path,out):
    out=Path(out);protocol=json.loads(Path(protocol_path).read_text());prep=json.loads((out/'preparation.json').read_text())
    if (out/'pooled_metrics.json').exists():raise FileExistsError('Score already recorded')
    rows=read_predictions(out/'outer_predictions.csv.gz');truth={}
    for dataset in DATASETS:
        truth[dataset]={}
        for c in prep['contexts']:
            if c['stage']=='outer' and c['dataset_id']==dataset:
                part=read_truth(c['scoring_truth'])
                if set(part)&set(truth[dataset]):raise LeakageError('Repeated outer truth denominator')
                truth[dataset].update(part)
    pooled={};summary=[]
    for dataset in DATASETS:
        pooled[dataset]={}
        for family in (*FAMILIES,'inner_selected_neural'):
            pooled[dataset][family]={}
            for seed in protocol['seeds']:
                subset=[r for r in rows if r['dataset_id']==dataset and r['family']==family and int(r['seed'])==seed]
                pooled[dataset][family][str(seed)]=metrics(subset,truth[dataset],protocol['class_orders'][dataset])
            values=pooled[dataset][family]
            summary.append({'dataset_id':dataset,'family':family,'samples_per_seed':len(truth[dataset]),'seeds':protocol['seeds'],
                **{m+'_mean':float(np.mean([v[m] for v in values.values()])) for m in ['accuracy','macro_f1','balanced_accuracy']},
                **{m+'_sample_sd':float(np.std([v[m] for v in values.values()],ddof=1)) for m in ['accuracy','macro_f1','balanced_accuracy']}})
    write_json(out/'pooled_metrics.json',pooled);write_json(out/'seed_summary.json',summary)
    write_json(out/'primary_seed42_metrics.json',{d:{f:v['42'] for f,v in families.items()} for d,families in pooled.items()})
    scored=[{**r,'y_true':truth[r['dataset_id']][r['record_id']],'is_correct':int(r['y_predicted']==truth[r['dataset_id']][r['record_id']])} for r in rows]
    write_predictions(out/'scored_outer_predictions.csv.gz',scored);event(out,'independent_scoring_completed',rows=len(rows))
    return {'datasets':2,'methods_per_dataset':7,'seeds':3,'scored_rows':len(rows),'scientific_status':'EXPLORATORY_USER_DECLARED_SOURCE_GROUPS'}
