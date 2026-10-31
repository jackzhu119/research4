"""Independent no-fit replay of all frozen ginseng source-stage artefacts."""
import collections,gzip,hashlib,io,json,sys,time
import numpy as np
import joblib
from audit_common import Audit,ROOT,OUT,read,sha,utc,write_new,verify_dataset,metrics
sys.path.insert(0,str(ROOT/'src'))
from herb_spectra.leakage_guard import source_fingerprint
B=ROOT/'reports/public-thz-20261031-v1/ginseng-benchmark'
MS=('macro_f1','balanced_accuracy','accuracy')

def metric_check(a,x,y,why):
 for m in MS:a.close(x[m],y[m],why+' '+m)
 a.equal(x['confusion_matrix'],y['confusion_matrix'],why+' confusion')
def selected_config(c):return {k:v for k,v in c.items() if k not in ['id','family']}
def choose(rows):return sorted([c for c in rows if c['status']=='SOURCE_EVALUATED' and c['cases']==15],key=lambda c:tuple(-c['mean_inner_metrics'][m] for m in MS)+(c['config']['id'],))[0]
def strip(c):return {k:c[k] for k in ['config','cases','status','mean_inner_metrics']}
def object_key(outer,candidate,split,seed):return str(outer),str(candidate),str(split),int(seed)

def main():
 start=time.monotonic();a=Audit();data,_=verify_dataset(a);p=read(B/'protocol.json');prep=read(B/'preparation.json');manifest=read(B/'run_manifest.json');lock=read(B/'selection_lock.json')
 a.check(sha(B/'run_manifest.json')=='170830f0cb4501b7f59f49f6591ecb07591c1b8327a2dd6a1ee18d21a885538a','source-review approved manifest')
 for f,h in manifest['source_sha256'].items():a.check(sha(ROOT/f)==h,'frozen dependency '+f)
 checks={'run_manifest_sha256':'run_manifest.json','source_release_sha256':'source_release_snapshot.json','inner_records_sha256':'inner_records.json','inner_predictions_sha256':'inner_predictions.csv.gz','candidate_summaries_sha256':'candidate_summaries.json'}
 for k,f in checks.items():a.check(lock[k]==sha(B/f),'full selection chain '+f)
 release=read(B/'source_release_snapshot.json');a.check(release['allow_source_training'] and all(release[k]==manifest[k] for k in ['protocol_sha256','preparation_sha256','source_sha256']),'root source-only release bound')
 a.check(lock['all6outer_choices_locked'] and lock['outer_predictions_already_generated'] is False,'source locked before any outer prediction')
 a.check(not (B/'prediction_lock.json').exists() and not (B/'outer_release_snapshot.json').exists(),'no outer evaluation at source review')
 contexts={c['split_id']:c for c in prep['contexts'] if c['stage']=='inner'};all_ids={str(r):i for i,r in enumerate(data['record_ids'])}
 packs={};
 for split,c in contexts.items():
  for role in ['source','query']:
   path=B/c[role+'_pack'];a.check(sha(path)==c[role+'_pack_sha256'],'source preparation unchanged')
   with np.load(path,allow_pickle=False) as z:packs[(split,role)]={k:z[k] for k in z.files}
 records=read(B/'inner_records.json');summaries=read(B/'candidate_summaries.json');failures=read(B/'source_candidate_failures.json')
 a.check(len(records)==lock['source_classifier_models'],'actual model inventory')
 a.check(len(records)+len(failures)==11160 and len(summaries)==744,'all source candidate attempts preserved')
 with gzip.open(B/'inner_predictions.csv.gz','rt',encoding='utf-8-sig') as f:
  import csv
  rows=list(csv.DictReader(f))
 pred_by=collections.defaultdict(list)
 for r in rows:pred_by[object_key(r['outer_group'],r['candidate_id'],r['split_id'],r['seed'])].append(r)
 a.check(len(rows)==7*len(records) and len(pred_by)==len(records),'exact seven predictions for every successful source model')
 # Complete byte verification precedes every model/cache deserialization.
 cache_refs={}
 for r in records:
  path=B/r['estimator_path'];a.check(sha(path)==r['estimator_sha256'],'all source model bytes')
  a.check(sha(path.with_suffix('.json'))==r['metadata_sha256'],'all model metadata bytes')
  ref=r['transform'];old=cache_refs.setdefault(ref['path'],ref)
  a.check(old['sha256']==ref['sha256'] and old['cache_identity']==ref['cache_identity'],'cache alias exact identity')
 for name,ref in cache_refs.items():a.check(sha(B/name)==ref['sha256'],'all shared source cache bytes')
 bundles={};full_z={};cache_mc={};raw_fp_by_split=collections.defaultdict(set);mc_map={};scope_maps={};pca_scope_maps={}
 for name,ref in cache_refs.items():
  envelope=joblib.load(B/name);blob=envelope['payload_bytes'];a.check(hashlib.sha256(blob).hexdigest()==envelope['payload_sha256'],'source cache payload before loading')
  bundle=joblib.load(io.BytesIO(blob));ident=ref['cache_identity'];split=ident['split_id'];src=packs[(split,'source')];ids=src['record_ids'];y=src['source_labels']
  a.check(envelope['identity']==ident==bundle.metadata['cache_identity'],'complete cache identity');a.equal(bundle.metadata['source_ids'],ids,'cache ordered source IDs');a.equal(bundle.metadata['source_y'],y,'cache source labels')
  mods=['alpha','n'] if bundle.representation=='paired_alpha_n' else [bundle.representation];raw=np.column_stack([src['X_'+mod] for mod in mods])
  fp=source_fingerprint(raw,y);a.check(ident['data_fingerprint']==fp==bundle._leakage_fit_provenance['data_fingerprint'],'cache actual X/y proof')
  raw_fp_by_split[split].add(fp);zs=[]
  for mod in mods:
   pre=bundle.preprocessors[mod];x=src['X_'+mod]
   if bundle.preprocessing=='msc':
    a.close(pre.reference_,x.mean(0),'MSC current-source reference')
   processed=pre.transform(x);scaler=bundle.scalers[mod]
   a.close(scaler.mean_,processed.mean(0),'current-source scaler mean');a.close(scaler.var_,processed.var(0),'current-source scaler variance')
   a.check(int(scaler.n_samples_seen_)==len(ids),'source scaler N');zs.append(scaler.transform(processed))
  z=np.column_stack(zs);full_z[name]=z;bundles[name]=bundle
  if bundle.selector is not None:
   s=bundle.selector;coef=s.coefficients_;sd=coef.std(0,ddof=1);stability=coef.mean(0)/np.maximum(sd,max(1.,float(sd.max()))*1e-12);importance=np.linalg.norm(abs(stability),axis=0)
   a.close(s.signed_stability_,stability,'MC source coefficient stability');a.close(s.importances_,importance,'MC source importance');a.equal(s.ranking_,np.argsort(-importance,kind='stable'),'MC stable ranking')
   a.check(coef.shape==(s.n_runs,2,z.shape[1]) and s.sample_unit_=='rows_assumed_independent' and s.n_training_sample_units_==len(ids),'MC target/sample dimensions')
   mc_map[(split,source_fingerprint(z,y))]=(z,y,ids);cache_mc[name]=(z,y,ids)
 per_candidate=collections.defaultdict(list);classifier_fp=collections.defaultdict(set);warnings=collections.Counter();total_seconds=0.;prediction_values=0
 for number,r in enumerate(records):
  c=contexts[r['split_id']];src=packs[(r['split_id'],'source')];q=packs[(r['split_id'],'query')];cfg=r['config'];seed=r['seed'];meta=read((B/r['estimator_path']).with_suffix('.json'));bundle=bundles[r['transform']['path']]
  a.check(meta['config']==cfg and meta['seed']==seed and meta['context']==r['split_id'],'model config/seed binding');a.equal(meta['source_ids'],c['source_ids'],'model source IDs');a.equal(meta['query_ids'],c['query_ids'],'model query IDs')
  ratio=cfg['mc']['ratio'] if cfg.get('mc') else None;z=bundle.matrix({'alpha':src['X_alpha'],'n':src['X_n']},ratio);qz=bundle.matrix({'alpha':q['X_alpha'],'n':q['X_n']},ratio)
  a.equal(meta['retained_indices'],bundle.selected(ratio),'chosen feature indices');a.check(r['selected_features']==z.shape[1]==meta['retained_features'],'retained feature count')
  est=joblib.load(B/r['estimator_path']);state=joblib.hash(est);a.check(est.metadata==meta['estimator'],'estimator persisted metadata')
  reduced=est.reduced(z);a.check(est.metadata['source_z_fingerprint']==source_fingerprint(reduced,src['source_labels']),'actual fitted source feature/label fingerprint')
  if cfg['kind']=='svm':a.close(est.estimator.support_vectors_,z[est.estimator.support_],'SVC source support vectors')
  if cfg['kind']=='pca_lda':
   a.close(est.reducer.mean_,z.mean(0),'PCA current source mean');a.close(est.reducer.components_@est.reducer.components_.T,np.eye(cfg['components']),'PCA source orthonormal components',atol=1e-10)
  if cfg['kind']=='mlp':a.check(est.estimator.random_state==seed and est.estimator.early_stopping is False and est.estimator.n_iter_==cfg['epochs'],'MLP fixed seed and epochs')
  pred=est.predict(qz);scores=est.scores(qz);rr=pred_by[object_key(r['outer_group'],r['candidate_id'],r['split_id'],seed)]
  a.equal([x['record_id'] for x in rr],c['query_ids'],'prediction route IDs cannot change label identity');a.equal([x['y_predicted'] for x in rr],pred,'all actual source predictions replayed')
  a.close([[float(x['score_0']),float(x['score_1'])] for x in rr],scores,'all actual source score vectors replayed',atol=1e-10)
  a.check(joblib.hash(est)==state,'source replay no estimator mutation')
  truth=np.array([data['y'][all_ids[rid]] for rid in c['query_ids']]);metric=metrics(truth,pred);metric_check(a,r['metrics'],metric,'independent source metric')
  per_candidate[(r['outer_group'],r['candidate_id'])].append(metric)
  total_seconds+=r['elapsed_seconds'];prediction_values+=len(pred);a.check(np.isfinite(r['elapsed_seconds']) and r['elapsed_seconds']>=0,'actual nonnegative elapsed cost')
  for w in est.metadata['warnings']:warnings[(w['category'],w['message'])]+=1
  component={'svm':'SVC','lda':'shrinkage_LDA','pca_lda':'shrinkage_LDA','plsda':'PLSDA.PLSRegression','mlp':'MLPClassifier'}[cfg['kind']]
  target=np.eye(2)[np.searchsorted(['0','1'],src['source_labels'])] if cfg['kind']=='plsda' else src['source_labels']
  classifier_fp[(r['split_id'],component)].add(source_fingerprint(reduced,target))
  if cfg['kind']=='pca_lda':classifier_fp[(r['split_id'],'PCA')].add(source_fingerprint(z))
  if number and number%2000==0:print(json.dumps({'source_models_replayed':number}),flush=True)
 for summary in summaries:
  actual=per_candidate[(summary['outer_group'],summary['config']['id'])];a.check(len(actual)==summary['cases'],'candidate case inventory')
  if summary['status']=='SOURCE_EVALUATED':
   a.check(len(actual)==15,'all five inner groups and three seeds')
   for field in MS:a.close(summary['mean_inner_metrics'][field],np.mean([m[field] for m in actual]),'candidate mean metrics')
  else:a.check(any(f['outer_group']==summary['outer_group'] and f['candidate']['id']==summary['config']['id'] for f in failures),'failed candidate preserved')
 for fold in lock['outer_folds']:
  group=fold['outer_group'];summ=[s for s in summaries if s['outer_group']==group];choices=fold['choices'];a.check(fold['outer_truth_used_to_select'] is False,'no outer truth in selection lock')
  for stage in ['R1','R2','R3']:a.check(strip(choose([s for s in summ if s['stage']==stage]))==choices[stage],'stage winner exact source metric tie rule')
  for s in [s for s in summ if s['stage']=='R2']:a.check(s['config']['mc']==choices['R1']['config']['mc'],'R2 inherits only source-chosen R1 MC')
  for s in [s for s in summ if s['stage']=='R3']:
   a.check({k:v for k,v in s['config'].items() if k not in ['id','preprocessing']}=={k:v for k,v in choices['R2']['config'].items() if k not in ['id','preprocessing']},'R3 source-chosen R2 settings frozen')
  r4=[s for s in summ if s['stage']=='R4'];a.check(strip(choose(r4+[choices['R2'],choices['R3']]))==choices['R4_policy'],'overall R4 policy source only')
  for rep in ['alpha','n','paired_alpha_n']:
   for family in ['full_svm','shrinkage_lda','pca_lda','plsda','mlp']:
    expected=strip(choose([s for s in r4 if s['config']['representation']==rep and s['config']['family']==family]));a.check(choices[f'family_{rep}_{family}']==expected,'independent per-family source winner')
 # Actual FitScope trace: every fit, including each MC PLS subsample, stays inside that current source.
 counts=collections.Counter();active_mc=None;mc_sub=None;trace_lines=0
 for line in (B/'fit_scope_events.jsonl').open():
  e=json.loads(line);trace_lines+=1;split=e['split_id'];ids=e['fit_sample_ids'];allowed=contexts[split]['source_ids'];a.check(set(ids)<=set(allowed),'every actual fit sample ID source-only')
  if e['event']!='fit_start':continue
  component=e['component'];counts[component]+=1
  if component in ['SVC','shrinkage_LDA','PCA','PLSDA.PLSRegression','MLPClassifier']:
   a.equal(ids,allowed,'classifier/reducer full source membership');a.check(e['data_fingerprint'] in classifier_fp[(split,component)],'logged classifier actual values/target fingerprint')
  elif component=='MC-UVE.full_source_ranking':active_mc=mc_map[(split,e['data_fingerprint'])]
  elif component=='MC-UVE.coefficient_standardization':
   z,y,fullids=active_mc;indices=np.array([list(fullids).index(i) for i in ids]);x=z[indices];t=np.eye(2)[np.searchsorted(['0','1'],y[indices])]
   a.check(e['data_fingerprint']==source_fingerprint(x,t),'actual MC subset source values + one-hot labels');sd=x.std(0);sd[sd<=np.finfo(float).eps]=1;mc_sub=((x-x.mean(0))/sd,t)
  elif component=='MC-UVE.PLSRegression.fit':
   x,t=mc_sub;options=[]
   for target in t.T:options.append(source_fingerprint(x,(target-target.mean())/target.std()))
   a.check(e['data_fingerprint'] in options,'each actual internal PLS normalized source subset and target')
 actual_classifiers=sum(counts[c] for c in ['SVC','shrinkage_LDA','PLSDA.PLSRegression','MLPClassifier'])
 a.check(actual_classifiers==len(records)+len(failures),'actual classifier fit trace inventory')
 out={'completed_at_utc':utc(),'status':'SOURCE_SELECTION_REPLAY_PASS_EXPLORATORY_UNKNOWN_UPSTREAM','new_audit_fits':0,'checks':a.checks,'numeric_values':a.numeric_values,'maximum_difference':a.maximum_difference,'actual_source_classifier_models':len(records),'source_failed_attempts':len(failures),'actual_source_predictions_replayed':prediction_values,'cached_transforms_replayed':len(bundles),'actual_fit_start_counts':dict(counts),'actual_fit_trace_lines':trace_lines,'sum_trial_elapsed_seconds':total_seconds,'audit_elapsed_seconds':time.monotonic()-start,'selection_lock_sha256':sha(B/'selection_lock.json'),'fit_trace_sha256':sha(B/'fit_scope_events.jsonl'),'all_source_artifact_chain_sha256':checks,'all6outer_choices_recomputed':True,'all15family_representation_winners_per_outer_recomputed':True,'main_grid_not_modified_from_G1_diagnostic':True,'outer_evaluation_performed':False,'scientific_independence_certified':False,'warnings':[{'category':k[0],'message':k[1],'count':n} for k,n in warnings.items()],'limitations':['Only source development predictions have been reviewed; root must issue a separate outer release','Published42 observations have unknown underlying root/material identities; all results remain exploratory export-group evaluation','G1 labels were exposed in a pre-fixed small diagnostic; this cannot be labelled a fresh confirmation set','MC internal one-hot fitting costs two PLS1 per repeat; actual fit counts are reported from trace','No OS-level truth sandbox or proof of absolute leakage absence'],'script_sha256':sha(__file__)}
 write_new(OUT/'selection_review.json',out);print(json.dumps({k:out[k] for k in ['status','checks','actual_source_classifier_models','actual_source_predictions_replayed','new_audit_fits']}))

if __name__=='__main__':main()
