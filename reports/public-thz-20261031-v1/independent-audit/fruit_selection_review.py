"""No-fit independent replay of every frozen fruit source-regression model."""
import collections,csv,gzip,hashlib,io,json,sys,time
import joblib
import numpy as np
from audit_common import Audit,ROOT,OUT,read,sha,utc,write_new
sys.path.insert(0,str(ROOT/'src'))
from herb_spectra.leakage_guard import source_fingerprint
B=ROOT/'outputs/public-thz-fruit-regression-20261031-v1'
INPUT=ROOT/'data/external/public_thz_20261031_v1/acquired/princeton_metasticker_fruit/prepared/fruit_fixed_Tag_time_index_v1.npz'
TASKS={'MA_Brix':('MA','y_Brix'),'PM_Brix':('PM','y_Brix'),'PR_Brix':('PR','y_Brix'),'MA_DM':('MA','y_DM')}

def metric(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float);sse=np.sum((y-p)**2);sd=float(y.std(ddof=1));rmse=float(np.sqrt(sse/len(y)))
 return {'rmse':rmse,'mae':float(np.mean(abs(y-p))),'r2':float(1-sse/np.sum((y-y.mean())**2)),'target_sample_sd':sd,'rpd':sd/rmse if rmse>0 else None,'observations':len(y)}
def check_metric(a,found,expected,why):
 for f in ['rmse','mae','r2','target_sample_sd','rpd']:
  if found[f] is None or expected[f] is None:a.check(found[f] is None and expected[f] is None,why+' '+f)
  else:a.close(found[f],expected[f],why+' '+f,atol=1e-10)
 a.check(found['observations']==expected['observations'],why+' denominator')
def choose(rows):return min([r for r in rows if r['status']=='SOURCE_EVALUATED' and r['cases']==9],key=lambda r:(r['mean_seed_metrics']['rmse'],r['mean_seed_metrics']['mae'],r['config']['id']))
def strip(c):return {k:c[k] for k in ['config','cases','status','mean_seed_metrics','pooled_metrics_per_seed']}
def key(row):return str(row['task']),int(row['outer']),str(row.get('candidate_id',row.get('config',{}).get('id'))),str(row['split_id']),int(row['seed'])

def main():
 start=time.monotonic();a=Audit();manifest=read(B/'run_manifest.json');prep=read(B/'preparation.json');lock=read(B/'selection_lock.json');protocol=read(B/'protocol.json')
 a.check(sha(B/'run_manifest.json')=='cbaad570b88803e9618c94c80720c0f0e5338e0568454992e64f6e1ff8573a11','preflight approved frozen manifest')
 for f,h in manifest['source_sha256'].items():a.check(sha(ROOT/f)==h,'frozen source '+f)
 for k,f in {'run_manifest_sha256':'run_manifest.json','source_release_sha256':'source_release_snapshot.json','inner_records_sha256':'inner_records.json','inner_predictions_sha256':'inner_predictions.csv.gz','candidate_summaries_sha256':'candidate_summaries.json','failures_sha256':'source_candidate_failures.json'}.items():a.check(lock[k]==sha(B/f),'complete source selection chain '+f)
 release=read(B/'source_release_snapshot.json');a.check(release['allow_source_training'] is True and release['run_manifest_sha256']==sha(B/'run_manifest.json'),'source root gate')
 a.check(not (B/'outer_release_snapshot.json').exists() and not (B/'prediction_lock.json').exists() and lock['outer_predictions_already_generated'] is False,'no outer execution before independent review')
 with np.load(INPUT,allow_pickle=False) as z:data={k:z[k] for k in z.files}
 id_lookup={r:i for i,r in enumerate(data['record_id'])};contexts={c['split_id']:c for c in prep['contexts'] if c['stage']=='inner'};packs={}
 for split,c in contexts.items():
  for role in ['source','query']:
   path=B/c[role+'_pack'];a.check(sha(path)==c[role+'_pack_sha256'],'actual prepared pack unchanged')
   with np.load(path,allow_pickle=False) as z:packs[(split,role)]={k:z[k] for k in z.files}
 records=read(B/'inner_records.json');summaries=read(B/'candidate_summaries.json');failures=read(B/'source_candidate_failures.json')
 a.check(len(records)==lock['source_estimator_models'] and len(records)+len(failures)==7560,'every source candidate attempt preserved')
 a.check(len(summaries)==840 and len(lock['outer_folds'])==20 and lock['all20outer_choices_locked'] is True,'all42 candidates by20 contexts locked')
 with gzip.open(B/'inner_predictions.csv.gz','rt',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 by_model=collections.defaultdict(list)
 for row in rows:by_model[key(row)].append(row)
 refs={}
 for r in records:
  path=B/r['estimator_path'];a.check(sha(path)==r['estimator_sha256'],'all source model bytes before load');a.check(sha(path.with_suffix('.json'))==r['metadata_sha256'],'all source metadata bytes')
  ref=r['transform'];prior=refs.setdefault(ref['path'],ref);a.check(prior['sha256']==ref['sha256'] and prior['cache_identity']==ref['cache_identity'],'exact source cache aliases')
 for f,ref in refs.items():a.check(sha(B/f)==ref['sha256'],'all source cache bytes before load')
 bundles={};z_full={};mc_map={};fit_fps=collections.defaultdict(set)
 for f,ref in refs.items():
  outer=joblib.load(B/f);blob=outer['payload_bytes'];a.check(hashlib.sha256(blob).hexdigest()==outer['payload_sha256'],'cache payload bytes')
  t=joblib.load(io.BytesIO(blob));ident=ref['cache_identity'];split=ident['split_id'];src=packs[(split,'source')];X=src['X'];y=src['source_targets']
  a.check(outer['identity']==ident==t.metadata['cache_identity'],'cache full identity');a.equal(t.metadata['source_ids'],src['record_ids'],'cached source membership')
  a.check(source_fingerprint(X,y)==ident['data_fingerprint']==t._leakage_fit_provenance['data_fingerprint'],'actual source X/y binding')
  if t.preprocessing=='raw':processed=X.copy()
  else:
   sd=X.std(1,keepdims=True);sd[sd<=np.finfo(float).eps]=1;processed=(X-X.mean(1,keepdims=True))/sd
  a.close(t.x_scaler.mean_,processed.mean(0),'current-source input scaler mean');a.close(t.x_scaler.var_,processed.var(0),'source input variance')
  a.close(t.y_scaler.mean_,[y.mean()],'current-source target scaler mean');a.close(t.y_scaler.var_,[y.var()],'source target variance')
  a.check(int(t.x_scaler.n_samples_seen_)==len(y)==int(t.y_scaler.n_samples_seen_),'source scaler denominators')
  Z=t.x_scaler.transform(processed);target=t.y_scaler.transform(y[:,None]).ravel();bundles[f]=t;z_full[f]=Z
  if t.selector is not None:
   s=t.selector;sd=s.coefficients_.std(0,ddof=1);stability=s.coefficients_.mean(0)/np.maximum(sd,max(1.,float(sd.max()))*1e-12);importance=np.linalg.norm(abs(stability),axis=0)
   a.close(s.signed_stability_,stability,'MC coefficient arithmetic');a.equal(s.ranking_,np.argsort(-importance,kind='stable'),'MC stable ranking')
   a.check(s.coefficients_.shape==(s.n_runs,1,128),'one continuous target one PLS per repeat')
   mc_map[(split,source_fingerprint(Z,target))]=(Z,target,src['record_ids'])
 candidate_rows=collections.defaultdict(list);sum_seconds=0.;predictions=0;warnings=collections.Counter()
 for number,r in enumerate(records):
  split=r['split_id'];c=contexts[split];src=packs[(split,'source')];q=packs[(split,'query')];config=r['config'];seed=r['seed'];t=bundles[r['transform']['path']];ratio=config['mc']['ratio'] if config.get('mc') else None
  meta=read((B/r['estimator_path']).with_suffix('.json'));a.check(meta['config']==config and meta['seed']==seed and meta['split_id']==split,'model config/seed binding');a.equal(meta['source_ids'],c['source_ids'],'model source membership');a.equal(meta['query_ids'],c['query_ids'],'model query routing')
  est=joblib.load(B/r['estimator_path']);state=joblib.hash((t,est));Z=t.matrix(src['X'],ratio);target=t.y_scaler.transform(src['source_targets'][:,None]).ravel();qz=t.matrix(q['X'],ratio)
  a.check(source_fingerprint(Z,target)==meta['estimator']['source_fingerprint'],'actual estimator source values and scaled targets')
  a.equal(meta['selected_indices'],t.indices(ratio),'source-only retained indices');a.check(len(t.indices(ratio))==r['retained_features'],'retained count')
  if config['family'] in ['full_SVR','MC_SVR']:a.close(est.support_vectors_,Z[est.support_],'SVR support vectors from source only')
  if config['family']=='MLP':a.check(est.random_state==seed and est.early_stopping is False and est.n_iter_==300,'fixed MLP training seed/epochs')
  prediction=t.y_scaler.inverse_transform(np.asarray(est.predict(qz)).reshape(-1,1)).ravel();actual=by_model[key(r)]
  a.equal([row['record_id'] for row in actual],c['query_ids'],'exact source validation routing');a.close([float(row['y_predicted']) for row in actual],prediction,'all source predictions replayed',atol=1e-9)
  a.check(joblib.hash((t,est))==state,'read-only source replay')
  candidate_rows[(r['task'],r['outer'],config['id'],seed)].extend(actual);fit_fps[(split,'fruit.'+config['family'])].add(source_fingerprint(Z,target));predictions+=len(prediction);sum_seconds+=r['elapsed_seconds']
  for w in meta['estimator']['warnings']:warnings[(w['category'],w['message'])]+=1
  if number and number%2000==0:print(json.dumps({'fruit_source_models_replayed':number}),flush=True)
 for summary in summaries:
  if summary['status']!='SOURCE_EVALUATED':
   a.check(any(f['candidate']['id']==summary['config']['id'] for f in failures),'failed candidate remains visible');continue
  a.check(summary['cases']==9,'three inner groups by three seeds')
  pooled=[]
  for seed in [42,43,44]:
   rr=candidate_rows[(summary['task'],summary['outer'],summary['config']['id'],seed)];ids=[r['record_id'] for r in rr];a.check(len(ids)==32 and len(set(ids))==32,'pooled inner denominator32 observations')
   target=TASKS[summary['task']][1];m=metric([data[target][id_lookup[rid]] for rid in ids],[float(r['y_predicted']) for r in rr]);check_metric(a,summary['pooled_metrics_per_seed'][str(seed)],m,'pooled per-seed source metric');pooled.append(m)
  for f in ['rmse','mae']:a.close(summary['mean_seed_metrics'][f],np.mean([m[f] for m in pooled]),'source mean3seed pooled metric')
 for fold in lock['outer_folds']:
  choices=fold['choices'];subset=[s for s in summaries if s['task']==fold['task'] and s['outer']==fold['outer']];a.check(fold['outer_truth_used_for_selection'] is False,'source-only family choices')
  for family in ['PLSR','full_SVR','MLP','MC_SVR']:a.check(strip(choose([s for s in subset if s['config']['family']==family]))==choices[family],'exact source family winner')
  a.check(strip(choose([choices[f] for f in ['PLSR','full_SVR','MLP','MC_SVR']]))==choices['source_policy'],'source-only policy alias')
  for s in [s for s in subset if s['round']=='R2']:
   a.check({k:v for k,v in s['config'].items() if k not in ['id','mc','family']}=={k:v for k,v in choices['full_SVR']['config'].items() if k not in ['id','mc','family']},'R2 inherits source-selected SVR settings')
 counts=collections.Counter();active=None;sub=None;lines=0
 for line in (B/'fit_scope_events.jsonl').open():
  e=json.loads(line);lines+=1;c=contexts[e['split_id']];ids=e['fit_sample_ids'];a.check(set(ids)<=set(c['source_ids']) and not set(ids)&set(c['query_ids']),'all actual fitting IDs inside current source fruits')
  if e['event']!='fit_start':continue
  component=e['component'];counts[component]+=1
  if component in ['fruit.PLSR','fruit.full_SVR','fruit.MLP','fruit.MC_SVR']:
   a.equal(ids,c['source_ids'],'source estimator full membership');a.check(e['data_fingerprint'] in fit_fps[(e['split_id'],component)],'actual regression fit values and targets')
  elif component=='fruit.MC_UVE.source_record_ranking':active=mc_map[(e['split_id'],e['data_fingerprint'])]
  elif component=='MC-UVE.coefficient_standardization':
   Z,target,sourceids=active;ix=np.array([list(sourceids).index(i) for i in ids]);X=Z[ix];y=target[ix,None];a.check(e['data_fingerprint']==source_fingerprint(X,y),'MC subset exact current-source target values')
   sd=X.std(0);sd[sd<=np.finfo(float).eps]=1;sub=((X-X.mean(0))/sd,(y[:,0]-y[:,0].mean())/y[:,0].std())
  elif component=='MC-UVE.PLSRegression.fit':a.check(e['data_fingerprint']==source_fingerprint(*sub),'actual internal MC PLS source subset normalization')
 a.check(sum(counts[c] for c in ['fruit.PLSR','fruit.full_SVR','fruit.MLP','fruit.MC_SVR'])==len(records)+len(failures),'actual regression source fit inventory')
 a.check(counts['MC-UVE.PLSRegression.fit']==10800,'actual cached continuous-target MC internal PLS budget')
 result={'completed_at_utc':utc(),'status':'FRUIT_SOURCE_SELECTION_REPLAY_PASS_LOCAL_ONLY_EXPLORATORY','checks':a.checks,'numeric_values':a.numeric_values,'maximum_difference':a.maximum_difference,'new_audit_fits':0,'actual_source_estimators':len(records),'actual_source_prediction_rows_replayed':predictions,'actual_failed_attempts':len(failures),'actual_fit_start_counts':dict(counts),'trace_lines':lines,'cached_transforms_replayed':len(bundles),'sum_trial_elapsed_seconds':sum_seconds,'audit_elapsed_seconds':time.monotonic()-start,'selection_lock_sha256':sha(B/'selection_lock.json'),'fit_trace_sha256':sha(B/'fit_scope_events.jsonl'),'all20source_folds_winners_recomputed':True,'outer_evaluation_performed':False,'local_raw_or_model_data_copied_to_git':False,'limitations':['Only source records reviewed; a root-bound outer release and independent metric replay remain required','Four timepoints are dependent within fruit; ten fruits/task not40independent plants','Engineering reflection waveform features, not verified THz absorption frequency features','No verified redistribution license; raw/derived/model/per-record artefacts remain local outputs'],'script_sha256':sha(__file__)}
 write_new(OUT/'fruit_selection_review.json',result);print(json.dumps({k:result[k] for k in ['status','checks','actual_source_estimators','actual_source_prediction_rows_replayed','new_audit_fits']}))

if __name__=='__main__':main()
