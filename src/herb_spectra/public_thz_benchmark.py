"""Prelocked grouped-development benchmark for public ginseng THz observations.

Export groups are provenance groups, not verified biological batches. All learned
objects use the current source subset. No held-out truth enters model selection.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import csv,gzip,hashlib,importlib.metadata,json,time,warnings
from pathlib import Path
import joblib
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from .classification import classification_metrics
from .image_feature_models import ScatterPreprocessor
from .leakage_guard import (FitScope,LeakageError,CacheKey,guarded_cache_key,
    save_guarded_cache,load_guarded_cache,source_fingerprint,instrument_mcuve_pls)
from .public_thz_data import load_ginseng,DATASET_ID,STATUS
from .selection import MCUVESelector
from .spectral_models import SpectralPreprocessor
from .staged_nested_search import gamma_spec,retained_count

ROOT=Path(__file__).resolve().parents[2]
SEEDS=(42,43,44);CLASSES=('0','1');MODALITIES={'alpha':('alpha',),'n':('n',),'paired_alpha_n':('alpha','n')}
METRICS=('macro_f1','balanced_accuracy','accuracy')
PROTOCOL_ID='public-thz-ginseng-file-group-nested-20261031-v1'
SOURCE_FILES=('src/herb_spectra/public_thz_benchmark.py','scripts/run_public_thz_benchmark.py',
 'src/herb_spectra/public_thz_data.py','src/herb_spectra/leakage_guard.py','src/herb_spectra/selection.py',
 'src/herb_spectra/spectral_models.py','src/herb_spectra/image_feature_models.py',
 'src/herb_spectra/staged_nested_search.py','src/herb_spectra/classification.py')

def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def objhash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def csvwrite(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);op=gzip.open if p.suffix=='.gz' else open
 with op(p,'wt',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def event(out,name,**kw):
 with (Path(out)/'events.jsonl').open('a') as f:f.write(json.dumps({'event':name,'utc':utc(),**kw},ensure_ascii=False)+'\n')
def source_versions():return {n:sha(ROOT/n) for n in SOURCE_FILES}
def env():return {k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn','joblib']}
def fresh(p):
 p=Path(p)
 if p.exists() and any(p.iterdir()):raise FileExistsError(p)
 p.mkdir(parents=True,exist_ok=True);return p

def r1_grid():
 return [{'id':f'R1_mc{i:02d}','representation':'alpha','preprocessing':'snv','kind':'svm',
  'mc':{'runs':r,'components':l,'ratio':q},'C':10.,'gamma':'scale','class_weight':'balanced'}
  for i,(r,l,q) in enumerate((r,l,q) for r in [10,20,50] for l in [1,2,3] for q in [.1,.2,.4,.6])]
def r2_grid(mc):
 return [{'id':f'R2_sv{i:02d}','representation':'alpha','preprocessing':'snv','kind':'svm','mc':dict(mc),
  'C':c,'gamma':g,'class_weight':w} for i,(c,g,w) in enumerate((c,g,w) for c in [.01,.1,1.,10.,100.,1000.]
   for g in ['scale','auto','scale_x0.1','scale_x10'] for w in ['none','balanced'])]
def r3_grid(config):
 return [{**config,'id':'R3_'+p,'preprocessing':p} for p in ['raw','snv','sg0_snv','sg1_snv','sg2_snv','msc','linear_detrend_snv']]
def r4_grid():
 rows=[]
 for representation in MODALITIES:
  basic=[{'kind':'svm','C':c,'gamma':'scale','class_weight':'balanced','family':'full_svm'} for c in [1.,10.]]
  basic += [{'kind':'lda','family':'shrinkage_lda'}]
  basic += [{'kind':'pca_lda','components':k,'family':'pca_lda'} for k in [2,5]]
  basic += [{'kind':'plsda','components':k,'family':'plsda'} for k in [2,3]]
  basic += [{'kind':'mlp','hidden':list(h),'l2':a,'epochs':300,'family':'mlp'} for h in [(32,),(32,16)] for a in [.01,.1]]
  for i,b in enumerate(basic):rows.append({'id':f'R4_{representation}_{i:02d}','representation':representation,'preprocessing':'raw','mc':None,**b})
 return rows

def default_protocol():
 return {'protocol_id':PROTOCOL_ID,'dataset_id':DATASET_ID,'scientific_status':STATUS,
  'outer':'six leave-one-export-file-group-out folds','inner':'five leave-one-remaining-export-file-group-out folds',
  'seeds':list(SEEDS),'primary_seed':42,'classes':list(CLASSES),'R1':r1_grid(),
  'R2_base':r2_grid({'runs':10,'components':1,'ratio':.1}),
  'R2_rule':'Replace only MC settings with current-outer source-selected R1 winner',
  'R3_preprocessors':['raw','snv','sg0_snv','sg1_snv','sg2_snv','msc','linear_detrend_snv'],
  'R3_rule':'Source-selected R2 MC/SVM settings; refit all source transforms for each preprocessing',
  'R4':r4_grid(),'R4_additional_aliases':['R2_source_selected','R3_source_selected'],
  'R0':{'id':'R0_raw_alpha_svm','representation':'alpha','preprocessing':'raw','mc':None,'kind':'svm','C':1.,'gamma':'scale','class_weight':'balanced'},
  'selection':'mean15 current-outer INNER cases Macro-F1, Balanced Accuracy, Accuracy, candidate ID',
  'fixed_source_classifier_trials':11160,'source_trial_formula':'6outer x5inner x3seed x(36+48+7+33)',
  'outer_roles':['R0','R1','R2','R3','R4_policy']+[f'family_{r}_{f}' for r in MODALITIES for f in ['full_svm','shrinkage_lda','pca_lda','plsda','mlp']],
  'outer_role_seed_cases':60,'real_observations_per_role_seed':42,
  'independence_note':'42 labelled tablets; underlying plants/material ancestry and batches remain unknown; export groups are not biological batches',
  'representations':'65 alpha cm^-1 and65 dimensionless n at native0.2–1.6THz; no cross-species pooled labels',
  'alpha_negative_values':'All141 retained with QC; no selection based on prediction correctness',
  'transform_cache':'Full source IDs, values/y, split, representation/preprocessing, MC runs/LV, base selector full width, code and seed; ratios slice current-source ranking only',
  'prediction':'Native SVC.predict vote; binary margins stored as[-margin,+margin], not calibrated probabilities',
  'source_failures':'An inner failure excludes that candidate in that outer context; preserve exception before any outer score; no seed or specimen deletion',
  'no_outer_truth_for_selection':True,'all_inner_locks_before_any_outer_prediction':True,
  'fresh_confirmation':False,'no_independent_plants_assumed':True,'unknown_physical_ancestry':'EXPLORATORY, not scientific PASS',
  'CNN':'NOT_RUN;33 classical-family candidates are the fixed R4 budget','OS_truth_file_sandbox':False,
  'fit_scope':'Explicit new interfaces and actual MC internal PLS; not absolute zero-leakage proof'}

def strict_pack(path,source=False):
 with np.load(path,allow_pickle=False) as z:d={k:z[k] for k in z.files}
 allowed={'record_ids','X_alpha','X_n','axis'}|({'source_labels'} if source else set())
 if set(d)!=allowed:raise LeakageError('Exact numeric feature schema; no query truth or metadata features')
 ids=np.asarray(d['record_ids']);n=len(ids)
 if ids.dtype.kind not in 'US' or ids.ndim!=1 or n==0 or len(set(ids))!=n:raise LeakageError('Unique paired string IDs required')
 for mode in ['alpha','n']:
  x=d['X_'+mode]
  if x.dtype.kind not in 'iuf' or x.ndim!=2 or x.shape!=(n,len(d['axis'])) or not np.isfinite(x).all():raise LeakageError('Invalid numeric spectrum matrix')
 axis=d['axis']
 if axis.dtype.kind not in 'iuf' or axis.ndim!=1 or not np.isfinite(axis).all() or not np.all(np.diff(axis)>0):raise LeakageError('Invalid native frequency grid')
 if source and (d['source_labels'].shape!=(n,) or set(d['source_labels'])!=set(CLASSES)):raise LeakageError('Both source species required')
 return d

def prepare(out):
 out=fresh(out);data=load_ginseng();groups=data['export_groups'];contexts=[];splits=[]
 for group in sorted(set(groups)):
  tr=np.flatnonzero(groups!=group);te=np.flatnonzero(groups==group);inner=[]
  for held in sorted(set(groups[tr])):
   src=np.flatnonzero((groups!=group)&(groups!=held));val=np.flatnonzero(groups==held)
   inner.append((held,src,val))
  for stage,held,src,val in [('outer',None,tr,te)]+[('inner',h,s,v) for h,s,v in inner]:
   split=f'outer_{group}.'+(f'inner_{held}' if stage=='inner' else 'final')
   directory=out/'inputs'/split;directory.mkdir(parents=True)
   for role,ix in [('source',src),('query',val)]:
    d={k:data[k][ix] for k in ['record_ids','X_alpha','X_n']};d['axis']=data['axis']
    if role=='source':d['source_labels']=data['y'][ix]
    np.savez_compressed(directory/f'{role}.npz',**d)
   truth=out/'scoring_truth'/f'{split}.csv';csvwrite(truth,[{'record_id':str(i),'label':str(y)} for i,y in zip(data['record_ids'][val],data['y'][val],strict=True)])
   context={'split_id':split,'stage':stage,'outer_group':group,'inner_group':held,'source_ids':data['record_ids'][src].tolist(),'query_ids':data['record_ids'][val].tolist(),
    'source_groups':sorted(set(groups[src])),'query_groups':sorted(set(groups[val])),
    'source_pack':str((directory/'source.npz').relative_to(out)),'query_pack':str((directory/'query.npz').relative_to(out)),'truth':str(truth.relative_to(out))}
   for k in ['source_pack','query_pack','truth']:context[k+'_sha256']=sha(out/context[k])
   if set(context['source_ids'])&set(context['query_ids']) or set(context['source_groups'])&set(context['query_groups']):raise LeakageError('Export group crossed roles')
   contexts.append(context)
  splits.append({'outer_export_group':group,'source_ids':data['record_ids'][tr].tolist(),'heldout_ids':data['record_ids'][te].tolist(),'inner_groups':[h for h,_,_ in inner]})
 write(out/'protocol.json',default_protocol());write(out/'split_manifest.json',{'folds':splits,'rows_are_new_independent_plants':False})
 write(out/'preparation.json',{'created_at_utc':utc(),'contexts':contexts,'dataset_fingerprint':objhash({k:joblib.hash(v) for k,v in data.items()}),'data_status':STATUS,'label_scope':'0American/1Asian species; not geographical origin','features_or_labels_cleaned':False})
 write(out/'run_manifest.json',manifest(out));return {'contexts':len(contexts),'source_classifier_trials':11160,'outer_roles':20,'status':'PREPARED_NOT_TRAINED'}

def manifest(out):
 out=Path(out)
 return {'protocol_sha256':sha(out/'protocol.json'),'preparation_sha256':sha(out/'preparation.json'),'split_manifest_sha256':sha(out/'split_manifest.json'),'source_sha256':source_versions(),'environment':env()}

def verify_run(out):
 out=Path(out);m=read(out/'run_manifest.json')
 if manifest(out)!=m or read(out/'protocol.json')!=default_protocol():raise LeakageError('Frozen implementation, candidate budget or inputs changed')
 prep=read(out/'preparation.json')
 for c in prep['contexts']:
  for k in ['source_pack','query_pack']:
   if sha(out/c[k])!=c[k+'_sha256']:raise LeakageError('Prepared input changed')
 return prep,m

def source_release(out,release):
 _,m=verify_run(out)
 if not release.get('allow_source_training') or any(release.get(k)!=m[k] for k in ['protocol_sha256','preparation_sha256','source_sha256']):raise LeakageError('Source root gate missing or mismatched')

@dataclass
class SourceTransform:
 representation:str
 preprocessing:str
 preprocessors:dict
 scalers:dict
 selector:object
 metadata:dict
 _leakage_fit_provenance:dict
 def matrix(self,features,ratio=None):
  z=np.column_stack([self.scalers[m].transform(self.preprocessors[m].transform(features[m])) for m in MODALITIES[self.representation]])
  if self.selector is not None:
   if ratio is None:raise LeakageError('Explicit retained fraction needed')
   z=z[:,np.sort(self.selector.ranking_[:retained_count(z.shape[1],ratio)])]
  return z
 def selected(self,ratio=None):
  width=sum(s.n_features_in_ for s in self.scalers.values())
  return np.arange(width) if self.selector is None else np.sort(self.selector.ranking_[:retained_count(width,ratio)])

@dataclass
class SourceEstimator:
 kind:str
 estimator:object
 reducer:object
 classes_:np.ndarray
 metadata:dict
 def reduced(self,z):return z if self.reducer is None else self.reducer.transform(z)
 def scores(self,z):
  z=self.reduced(z)
  if self.kind=='plsda':s=self.estimator.predict(z)
  elif self.kind=='mlp':s=self.estimator.predict_proba(z)
  else:s=self.estimator.decision_function(z)
  s=np.asarray(s,dtype=float)
  if s.ndim==1:s=np.column_stack([-s,s])
  if s.shape!=(len(z),2) or not np.isfinite(s).all():raise LeakageError('Invalid class-ordered scores')
  return s
 def predict(self,z):
  z=self.reduced(z)
  return self.classes_[np.argmax(self.estimator.predict(z),axis=1)] if self.kind=='plsda' else self.estimator.predict(z)

def features(pack):return {m:pack['X_'+m] for m in ['alpha','n']}

def transform_source(pack,config,seed,context,out,cache):
 ids=pack['record_ids'];y=pack['source_labels'];axis=pack['axis'];reps=MODALITIES[config['representation']]
 if ids.tolist()!=context['source_ids']:raise LeakageError('Pack source IDs/order differ from context')
 raw=np.column_stack([pack['X_'+m] for m in reps]);mc=config.get('mc')
 selector_identity=None if mc is None else {'runs':mc['runs'],'components':mc['components'],'base_n_features_to_select':raw.shape[1]}
 code=objhash(source_versions());key=guarded_cache_key(dataset_version=DATASET_ID,train_ids=ids,split_id=context['split_id'],
  preprocessing={'representation':config['representation'],'mode':config['preprocessing'],'per_modality':True},selector=selector_identity,
  code_version=code,seed=seed,data_fingerprint=source_fingerprint(raw,y))
 path=Path(out)/'transform_cache'/f'{key}.joblib'
 if str(key) in cache:
  bundle,expected_bytes,expected_state=cache[str(key)]
  if sha(path)!=expected_bytes or joblib.hash(bundle)!=expected_state:raise LeakageError('Learned cache bytes/state changed')
  if bundle._leakage_fit_provenance.get('data_fingerprint')!=key.identity['data_fingerprint'] or bundle.metadata['cache_identity']!=key.identity:raise LeakageError('In-memory cross-source cache')
  return bundle,{'path':str(path.relative_to(out)),'sha256':sha(path),'cache_identity':key.identity,'reused':True}
 if path.exists():raise LeakageError('Unexpected existing learned cache; no resume across runs')
 scope=FitScope(context['source_ids'],context['split_id'],DATASET_ID,Path(out)/'fit_scope_events.jsonl',code_version=code)
 if ids.tolist()!=context['source_ids']:raise LeakageError('Pack source IDs/order differ from context')
 with scope.operation('transform_bundle_current_source',ids,X=raw,y=y) as proof:
  preps={};scalers={};scaled=[]
  for m in reps:
   v=pack['X_'+m];mode=config['preprocessing']
   prep=scope.fit(ScatterPreprocessor('msc'),v,sample_ids=ids,component='MSC_'+m) if mode=='msc' else SpectralPreprocessor(mode,axis)
   p=prep.transform(v);scale=scope.fit(StandardScaler(),p,sample_ids=ids,component='scaler_'+m)
   preps[m]=prep;scalers[m]=scale;scaled.append(scale.transform(p))
  z=np.column_stack(scaled);selector=None
  if mc is not None:
   selector=MCUVESelector(n_runs=mc['runs'],n_components=mc['components'],n_features_to_select=z.shape[1],sample_fraction=.8,task='classification',random_state=seed)
   with instrument_mcuve_pls(scope,z,ids):scope.fit(selector,z,y,sample_ids=ids,component='MC-UVE.full_source_ranking')
  meta={'cache_identity':key.identity,'source_ids':ids.tolist(),'source_y':y.tolist(),'selected_rank_is_source_only':True,
   'scaler_stats':{m:{'mean':s.mean_.tolist(),'var':s.var_.tolist()} for m,s in scalers.items()},'upstream_independence_known':False}
  bundle=SourceTransform(config['representation'],config['preprocessing'],preps,scalers,selector,meta,dict(proof))
 save_guarded_cache(path,bundle,key);cache[str(key)]=(bundle,sha(path),joblib.hash(bundle))
 return bundle,{'path':str(path.relative_to(out)),'sha256':sha(path),'cache_identity':key.identity,'reused':False}

def fit_estimator(z,y,ids,config,seed,context,out):
 code=objhash(source_versions());scope=FitScope(context['source_ids'],context['split_id'],DATASET_ID,Path(out)/'fit_scope_events.jsonl',code_version=code)
 reducer=None;kind=config['kind'];captured=[]
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always')
  if kind=='svm':
   gamma,_=gamma_spec(z,config['gamma']);weight=None if config['class_weight']=='none' else 'balanced'
   est=scope.fit(SVC(C=config['C'],gamma=gamma,kernel='rbf',class_weight=weight,break_ties=False,decision_function_shape='ovr',random_state=seed),z,y,sample_ids=ids,component='SVC')
  elif kind in ['lda','pca_lda']:
   if kind=='pca_lda':reducer=scope.fit(PCA(n_components=config['components'],svd_solver='full'),z,sample_ids=ids,component='PCA');z=reducer.transform(z)
   est=scope.fit(LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'),z,y,sample_ids=ids,component='shrinkage_LDA')
  elif kind=='plsda':
   classes,encoded=np.unique(y,return_inverse=True)
   est=scope.fit(PLSRegression(n_components=config['components'],scale=False),z,np.eye(len(classes))[encoded],sample_ids=ids,component='PLSDA.PLSRegression')
  elif kind=='mlp':
   est=scope.fit(MLPClassifier(hidden_layer_sizes=tuple(config['hidden']),alpha=config['l2'],max_iter=config['epochs'],early_stopping=False,
    n_iter_no_change=config['epochs'],tol=0,random_state=seed,solver='adam',learning_rate_init=.001),z,y,sample_ids=ids,component='MLPClassifier')
  else:raise LeakageError('Unknown declared classifier')
  captured=[{'category':type(w.message).__name__,'message':str(w.message)} for w in caught]
 meta={'warnings':captured,'source_ids':ids.tolist(),'source_z_fingerprint':source_fingerprint(z,y),'classes':list(CLASSES),'seed':seed,'model_kind':kind,'configuration':config}
 if kind=='mlp':meta.update(actual_epochs=int(est.n_iter_),training_loss_curve=est.loss_curve_)
 return SourceEstimator(kind,est,reducer,np.array(CLASSES),meta)

def inference_check(bundle,est,query,ratio):
 before=joblib.hash((bundle,est));x=features(query);z=bundle.matrix(x,ratio);scores=est.scores(z);pred=est.predict(z)
 rev=est.scores(bundle.matrix({k:v[::-1].copy() for k,v in x.items()},ratio))[::-1]
 single=np.concatenate([est.scores(bundle.matrix({k:v[i:i+1] for k,v in x.items()},ratio)) for i in range(len(z))])
 if not np.allclose(scores,rev,atol=1e-10,rtol=1e-10) or not np.allclose(scores,single,atol=1e-10,rtol=1e-10) or before!=joblib.hash((bundle,est)):raise LeakageError('Inference state/order/batch dependence')
 return pred,scores,{'samples':len(z),'single_maxdiff':float(np.max(np.abs(single-scores))),'reverse_maxdiff':float(np.max(np.abs(rev-scores))),'state_unchanged':True,'atol':1e-10,'rtol':1e-10}

def load_truth(path,ids):
 with Path(path).open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 d={r['record_id']:r['label'] for r in rows}
 if len(d)!=len(rows) or set(d)!=set(ids) or set(d.values())!=set(CLASSES):raise LeakageError('Scoring truth identity/class mismatch')
 return np.array([d[str(i)] for i in ids])

def trial(out,context,config,seed,cache,*,outer=False):
 out=Path(out)
 for name in ['source_pack','query_pack']:
  if sha(out/context[name])!=context[name+'_sha256']:raise LeakageError('Prepared values changed')
 src=strict_pack(out/context['source_pack'],True);query=strict_pack(out/context['query_pack'])
 if src['record_ids'].tolist()!=context['source_ids'] or query['record_ids'].tolist()!=context['query_ids']:raise LeakageError('Pack/context identity mismatch')
 start=time.perf_counter();bundle,reference=transform_source(src,config,seed,context,out,cache)
 ratio=config['mc']['ratio'] if config.get('mc') else None;z=bundle.matrix(features(src),ratio)
 est=fit_estimator(z,src['source_labels'],src['record_ids'],config,seed,context,out)
 pred,scores,audit=inference_check(bundle,est,query,ratio)
 directory=out/('evaluation_models' if outer else 'selection_models')/context['split_id']/config['id'];directory.mkdir(parents=True,exist_ok=True)
 path=directory/f'seed{seed}.joblib'
 if path.exists():raise FileExistsError(path)
 joblib.dump(est,path,compress=3)
 selected=bundle.selected(ratio)
 metadata={'config':config,'seed':seed,'context':context['split_id'],'source_ids':src['record_ids'].tolist(),'query_ids':query['record_ids'].tolist(),
  'transform':reference,'retained_indices':selected.tolist(),'retained_features':len(selected),'estimator':est.metadata,
  'inference_audit':audit,'elapsed_seconds':time.perf_counter()-start,'outer_truth_read':False}
 write(path.with_suffix('.json'),metadata)
 record={'split_id':context['split_id'],'outer_group':context['outer_group'],'inner_group':context['inner_group'],'candidate_id':config['id'],
  'config':config,'seed':seed,'estimator_path':str(path.relative_to(out)),'estimator_sha256':sha(path),'metadata_sha256':sha(path.with_suffix('.json')),
  'transform':reference,'selected_features':len(selected),'elapsed_seconds':metadata['elapsed_seconds']}
 predictions=[{'split_id':context['split_id'],'outer_group':context['outer_group'],'candidate_id':config['id'],'seed':seed,
  'record_id':str(rid),'y_predicted':str(p),'score_0':float(s[0]),'score_1':float(s[1])} for rid,p,s in zip(query['record_ids'],pred,scores,strict=True)]
 return record,predictions

def candidate_cases(out,outer_group,candidate,contexts,cache):
 records=[];preds=[];metric=[];failures=[]
 for c in contexts:
  for seed in SEEDS:
   try:
    rec,rows=trial(out,c,candidate,seed,cache)
    # INNER truth is allowed source-development supervision. Check bytes first.
    if sha(Path(out)/c['truth'])!=c['truth_sha256']:raise LeakageError('Inner source truth changed')
    y=load_truth(Path(out)/c['truth'],[r['record_id'] for r in rows]);m=classification_metrics(y,np.array([r['y_predicted'] for r in rows]),CLASSES)
    rec['metrics']=m;records.append(rec);preds.extend(rows);metric.append(m)
   except Exception as exc:
    # A reproducibility boundary failure must stop, not merely lose a candidate.
    if isinstance(exc,(LeakageError,FileExistsError)):raise
    fail={'outer_group':outer_group,'candidate':candidate,'split_id':c['split_id'],'seed':seed,'error_type':type(exc).__name__,'message':str(exc)}
    failures.append(fail)
 if failures:
  for f in failures:event(out,'candidate_failure',**f)
 return {'config':candidate,'cases':len(metric),'status':'FAILED_INNER_CANDIDATE' if failures else 'SOURCE_EVALUATED',
  'mean_inner_metrics':None if failures else {k:float(np.mean([m[k] for m in metric])) for k in METRICS},'records':records,'predictions':preds,'failures':failures}

def choose(cases):
 valid=[c for c in cases if c['status']=='SOURCE_EVALUATED' and c['cases']==15]
 if not valid:raise LeakageError('No fully evaluated source candidate')
 return sorted(valid,key=lambda c:tuple(-c['mean_inner_metrics'][k] for k in METRICS)+(c['config']['id'],))[0]

def selection(out,gate_path):
 out=Path(out);gate=read(gate_path);source_release(out,gate)
 if (out/'source_release_snapshot.json').exists():raise FileExistsError('Selection already started; preserve attempts')
 write(out/'source_release_snapshot.json',gate);event(out,'selection_start')
 prep,_=verify_run(out);outerlocks=[];allrecords=[];allpred=[];summaries=[];failed=[]
 for group in sorted({c['outer_group'] for c in prep['contexts']}):
  contexts=[c for c in prep['contexts'] if c['stage']=='inner' and c['outer_group']==group];cache={};choices={}
  for stage,candidates in [('R1',r1_grid()),('R2',None),('R3',None),('R4',r4_grid())]:
   if stage=='R2':candidates=r2_grid(choices['R1']['config']['mc'])
   if stage=='R3':candidates=r3_grid(choices['R2']['config'])
   evaluated=[]
   for config in candidates:
    c=candidate_cases(out,group,config,contexts,cache);evaluated.append(c);allrecords.extend(c['records']);allpred.extend(c['predictions']);failed.extend(c['failures'])
    summaries.append({'outer_group':group,'stage':stage,**{k:v for k,v in c.items() if k not in ['records','predictions']}})
    write_progress(out,{'outer_group':group,'stage':stage,'candidate':config['id'],'source_classifier_models':len(allrecords),'outer_evaluated':False})
   if stage!='R4':choices[stage]=strip_choice(choose(evaluated))
   else:
    families={}
    for representation in MODALITIES:
     for family in ['full_svm','shrinkage_lda','pca_lda','plsda','mlp']:
      role=f'family_{representation}_{family}';families[role]=strip_choice(choose([x for x in evaluated if x['config']['representation']==representation and x['config']['family']==family]))
    r4=choose(evaluated+[choices['R2'],choices['R3']]);choices['R4_policy']=strip_choice(r4);choices.update(families)
  outerlocks.append({'outer_group':group,'choices':choices,'outer_truth_used_to_select':False})
 write(out/'inner_records.json',allrecords);csvwrite(out/'inner_predictions.csv.gz',allpred);write(out/'candidate_summaries.json',summaries);write(out/'source_candidate_failures.json',failed)
 lock={'created_at_utc':utc(),'outer_folds':outerlocks,'source_classifier_models':len(allrecords),'source_failures':len(failed),'source_release_sha256':sha(out/'source_release_snapshot.json'),
  'run_manifest_sha256':sha(out/'run_manifest.json'),'inner_records_sha256':sha(out/'inner_records.json'),'inner_predictions_sha256':sha(out/'inner_predictions.csv.gz'),
  'candidate_summaries_sha256':sha(out/'candidate_summaries.json'),'all6outer_choices_locked':True,'outer_predictions_already_generated':False}
 write(out/'selection_lock.json',lock);event(out,'selection_completed',models=len(allrecords));return {'models':len(allrecords),'failures':len(failed),'outer_evaluated':False}

def strip_choice(case):return {k:case[k] for k in ['config','cases','status','mean_inner_metrics']}
def write_progress(out,d):
 p=Path(out)/'selection_progress.json';p.write_text(json.dumps(d,indent=2)+'\n')

def verify_selected(out):
 out=Path(out);prep,m=verify_run(out);lock=read(out/'selection_lock.json')
 checks={'run_manifest_sha256':'run_manifest.json','source_release_sha256':'source_release_snapshot.json','inner_records_sha256':'inner_records.json','inner_predictions_sha256':'inner_predictions.csv.gz','candidate_summaries_sha256':'candidate_summaries.json'}
 if any(lock.get(k)!=sha(out/v) for k,v in checks.items()) or len(lock['outer_folds'])!=6 or not lock['all6outer_choices_locked']:raise LeakageError('Frozen source selection changed')
 source_release(out,read(out/'source_release_snapshot.json'))
 records=read(out/'inner_records.json')
 if len(records)!=lock['source_classifier_models']:raise LeakageError('Incomplete source model inventory')
 # All learned source objects and cache bytes checked before evaluation/score.
 checked=set()
 for rec in records:
  for p,expected in [(out/rec['estimator_path'],rec['estimator_sha256']),((out/rec['estimator_path']).with_suffix('.json'),rec['metadata_sha256'])]:
   if p not in checked:
    if sha(p)!=expected:raise LeakageError('Source estimator or metadata changed')
    checked.add(p)
  p=out/rec['transform']['path']
  if p not in checked:
   if sha(p)!=rec['transform']['sha256']:raise LeakageError('Source transform cache changed')
   checked.add(p)
 return prep,m,lock

def verify_outer_release(out,release):
 out=Path(out)
 expected={'selection_lock_sha256':sha(out/'selection_lock.json'),'protocol_sha256':sha(out/'protocol.json'),'run_manifest_sha256':sha(out/'run_manifest.json')}
 if not release.get('allow_outer_evaluation') or not release.get('created_at_utc') or any(release.get(k)!=v for k,v in expected.items()):raise LeakageError('Root outer release must bind all source choices')

def evaluation(out,release_path):
 out=Path(out);prep,m,lock=verify_selected(out);release=read(release_path);verify_outer_release(out,release)
 if (out/'outer_release_snapshot.json').exists():raise FileExistsError('Outer evaluation already started')
 write(out/'outer_release_snapshot.json',release);event(out,'outer_prediction_start')
 records=[];predictions=[];aliases=[]
 for fold in lock['outer_folds']:
  group=fold['outer_group'];context=next(c for c in prep['contexts'] if c['outer_group']==group and c['stage']=='outer');cache={};fitted={}
  configurations={'R0':default_protocol()['R0'],**{role:c['config'] for role,c in fold['choices'].items()}}
  for role in default_protocol()['outer_roles']:
   config=configurations[role]
   for seed in SEEDS:
    # Only fully identical current-source configurations may share estimator bytes.
    identity={k:v for k,v in config.items() if k not in ['id','family']};key=(objhash(identity),seed)
    if key not in fitted:
     record,rows=trial(out,context,config,seed,cache,outer=True);fitted[key]=(record,rows);records.append(record)
    else:record,rows=fitted[key];aliases.append({'outer_group':group,'role':role,'seed':seed,'estimator_path':record['estimator_path'],'candidate_id':config['id']})
    predictions.extend({**r,'role':role,'role_candidate_id':config['id']} for r in rows)
 csvwrite(out/'outer_predictions.csv.gz',predictions);write(out/'outer_model_records.json',records);write(out/'outer_role_aliases.json',aliases)
 prediction_lock={'created_at_utc':utc(),'outer_truth_parsed':False,'prediction_sha256':sha(out/'outer_predictions.csv.gz'),
  'outer_model_records_sha256':sha(out/'outer_model_records.json'),'outer_role_aliases_sha256':sha(out/'outer_role_aliases.json'),
  'selection_lock_sha256':sha(out/'selection_lock.json'),'root_outer_release_sha256':sha(out/'outer_release_snapshot.json'),
  'run_manifest_sha256':sha(out/'run_manifest.json'),'protocol_sha256':sha(out/'protocol.json'),'prediction_rows':len(predictions),
  'actual_outer_estimators':len(records),'alias_roles':len(aliases),'expected_role_cases':360,'role_seed_pooled_cases':60,'real_observations_per_case':42}
 if len(predictions)!=2520 or len(records)+len(aliases)!=360:raise LeakageError('Incomplete role inventory')
 write(out/'prediction_lock.json',prediction_lock);event(out,'all_outer_predictions_locked',rows=len(predictions),models=len(records),aliases=len(aliases))
 return {'models':len(records),'aliases':len(aliases),'prediction_rows':len(predictions),'outer_truth_parsed':False}

def load_fitted_record(out,record):
 out=Path(out);path=out/record['estimator_path'];metadata_path=path.with_suffix('.json')
 if sha(path)!=record['estimator_sha256'] or sha(metadata_path)!=record['metadata_sha256']:raise LeakageError('Model integrity mismatch before loading')
 meta=read(metadata_path);t=record['transform'];cachepath=out/t['path']
 if sha(cachepath)!=t['sha256']:raise LeakageError('Transform integrity mismatch before loading')
 bundle=load_guarded_cache(cachepath,CacheKey(t['cache_identity']));est=joblib.load(path)
 if list(est.classes_)!=list(CLASSES) or meta['config']!=record['config']:raise LeakageError('Saved model vocabulary/config differs')
 return bundle,est

def verify_predictions_before_truth(out):
 out=Path(out);prep,m,selection=verify_selected(out);lock=read(out/'prediction_lock.json')
 expected={'prediction_sha256':'outer_predictions.csv.gz','outer_model_records_sha256':'outer_model_records.json','outer_role_aliases_sha256':'outer_role_aliases.json',
  'selection_lock_sha256':'selection_lock.json','root_outer_release_sha256':'outer_release_snapshot.json','run_manifest_sha256':'run_manifest.json','protocol_sha256':'protocol.json'}
 if not lock.get('created_at_utc') or lock.get('outer_truth_parsed') is not False or any(lock.get(k)!=sha(out/f) for k,f in expected.items()):raise LeakageError('Outer prediction lock changed')
 verify_outer_release(out,read(out/'outer_release_snapshot.json'))
 # Verify EVERY truth byte before semantically parsing even the first file.
 outer=[c for c in prep['contexts'] if c['stage']=='outer']
 for c in outer:
  if sha(out/c['truth'])!=c['truth_sha256']:raise LeakageError('Frozen outer truth bytes changed')
 records=read(out/'outer_model_records.json')
 for rec in records:
  p=out/rec['estimator_path'];t=rec['transform']
  if sha(p)!=rec['estimator_sha256'] or sha(p.with_suffix('.json'))!=rec['metadata_sha256'] or sha(out/t['path'])!=t['sha256']:raise LeakageError('Frozen outer model changed')
 if len(records)!=lock['actual_outer_estimators'] or lock['prediction_rows']!=2520:raise LeakageError('Incomplete outer results')
 return outer,lock

def score(out):
 out=Path(out)
 if (out/'scoring_manifest.json').exists():raise FileExistsError('Results already scored; do not replace')
 outer,lock=verify_predictions_before_truth(out);event(out,'independent_score_start_after_full_integrity_check')
 truth={}
 for c in outer:
  labels=load_truth(out/c['truth'],c['query_ids']);truth.update(zip(c['query_ids'],labels))
 with gzip.open(out/'outer_predictions.csv.gz','rt',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 if len(rows)!=2520:raise LeakageError('Unexpected final prediction denominator')
 groups={};folds={};joined=[]
 for r in rows:
  rid=r['record_id'];y=truth[rid];key=(r['role'],r['seed']);groups.setdefault(key,[]).append(r);folds.setdefault((r['role'],r['seed'],r['outer_group']),[]).append(r)
  joined.append({**r,'y_true':str(y),'correct':str(y)==r['y_predicted']})
 metrics={};foldmetrics=[]
 for (role,seed),items in sorted(groups.items()):
  ids=[r['record_id'] for r in items]
  if len(ids)!=42 or len(set(ids))!=42 or set(ids)!=set(truth):raise LeakageError('Each role/seed must have42 unique observation predictions')
  m=classification_metrics(np.array([truth[rid] for rid in ids]),np.array([r['y_predicted'] for r in items]),CLASSES)
  m['correct']=sum(truth[r['record_id']]==r['y_predicted'] for r in items);m['total']=42;metrics.setdefault(role,{})[seed]=m
 for (role,seed,group),items in sorted(folds.items()):
  m=classification_metrics(np.array([truth[r['record_id']] for r in items]),np.array([r['y_predicted'] for r in items]),CLASSES)
  foldmetrics.append({'role':role,'seed':int(seed),'held_export_group':group,'records':len(items),'metrics':m})
 if len(groups)!=60:raise LeakageError('Expected20roles x3seeds')
 summary=[]
 for role,values in metrics.items():
  d={'role':role,'observations_each_seed':42,'seeds':list(SEEDS)}
  for k in METRICS:
   vals=[values[str(s)][k] for s in SEEDS];d[k+'_mean']=float(np.mean(vals));d[k+'_sample_sd']=float(np.std(vals,ddof=1))
  summary.append(d)
 write(out/'pooled_metrics_by_seed.json',metrics);write(out/'fold_metrics.json',foldmetrics);write(out/'seed_summary.json',summary);csvwrite(out/'scored_outer_predictions.csv.gz',joined)
 write(out/'scoring_manifest.json',{'completed_at_utc':utc(),'prediction_lock_sha256':sha(out/'prediction_lock.json'),'prediction_sha256':lock['prediction_sha256'],
  'metrics_sha256':sha(out/'pooled_metrics_by_seed.json'),'scored_predictions_sha256':sha(out/'scored_outer_predictions.csv.gz'),'method_seed_cases':60,
  'real_observations_per_case':42,'new_independent_plants_claimed':False,'scientific_status':STATUS,'source_model_fits_in_scoring':0,'OS_file_isolation':False})
 event(out,'independent_scoring_completed',cases=60);return {'cases':60,'observations_per_case':42,'status':STATUS}
