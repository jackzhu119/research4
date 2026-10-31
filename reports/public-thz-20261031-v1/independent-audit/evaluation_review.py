"""No-fit independent replay of locked main or fruit outer predictions and metrics."""
import argparse,collections,csv,gzip,hashlib,io,json,sys,time
import joblib
import numpy as np
from audit_common import Audit,ROOT,OUT,read,sha,utc,write_new,verify_dataset,metrics
from fruit_selection_review import metric as regression_metric,check_metric as check_regression_metric,TASKS,INPUT
sys.path.insert(0,str(ROOT/'src'))
from herb_spectra.leakage_guard import source_fingerprint

def csvgz(path):
 with gzip.open(path,'rt',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def cfgkey(config,fruit):return json.dumps({k:v for k,v in config.items() if k not in (['id'] if fruit else ['id','family'])},sort_keys=True)
def predkey(r,fruit):return (r['task'],int(r['outer']),r['role'],int(r['seed'])) if fruit else (r['outer_group'],r['role'],int(r['seed']))
def main(mode):
 start=time.monotonic();fruit=mode=='fruit';a=Audit();B=ROOT/('outputs/public-thz-fruit-regression-20261031-v1' if fruit else 'reports/public-thz-20261031-v1/ginseng-benchmark')
 previous=read(OUT/('fruit_selection_review.json' if fruit else 'selection_review.json'));selection=read(B/'selection_lock.json');lock=read(B/'prediction_lock.json');manifest=read(B/'run_manifest.json');prep=read(B/'preparation.json');protocol=read(B/'protocol.json')
 a.check(sha(B/'selection_lock.json')==previous['selection_lock_sha256'],'independently reviewed source lock unchanged')
 for f,h in manifest['source_sha256'].items():a.check(sha(ROOT/f)==h,'frozen implementation '+f)
 for k,f in {'prediction_sha256':'outer_predictions.csv.gz','outer_model_records_sha256':'outer_model_records.json','outer_role_aliases_sha256':'outer_role_aliases.json','selection_lock_sha256':'selection_lock.json','root_outer_release_sha256':'outer_release_snapshot.json','run_manifest_sha256':'run_manifest.json','protocol_sha256':'protocol.json'}.items():a.check(lock[k]==sha(B/f),'entire prediction chain '+f)
 release=read(B/'outer_release_snapshot.json');a.check(release['allow_outer_evaluation'] is True and release['selection_lock_sha256']==sha(B/'selection_lock.json') and release['protocol_sha256']==sha(B/'protocol.json') and release['run_manifest_sha256']==sha(B/'run_manifest.json'),'root outer release binds independently reviewed source selection')
 a.check(lock['outer_truth_parsed'] is False,'all predictions locked before scoring')
 contexts={c['split_id']:c for c in prep['contexts'] if c['stage']=='outer'};records=read(B/'outer_model_records.json');aliases=read(B/'outer_role_aliases.json')
 expected_rows=2880 if fruit else 2520;a.check(lock['prediction_rows']==expected_rows and len(records)+len(aliases)==360,'fixed role/source denominator')
 # Verify every truth file and ALL model/cache/metadata bytes BEFORE loading any weight or parsing any truth record.
 for c in contexts.values():
  for part in ['source_pack','query_pack','truth']:a.check(sha(B/c[part])==c[part+'_sha256'],'complete outer data/truth byte gate')
 for r in records:
  path=B/r['estimator_path'];a.check(sha(path)==r['estimator_sha256'],'all outer model bytes before load');a.check(sha(path.with_suffix('.json'))==r['metadata_sha256'],'all outer metadata bytes');a.check(sha(B/r['transform']['path'])==r['transform']['sha256'],'all outer cache bytes')
 if fruit:
  a.check(sha(INPUT)==protocol['input_npz_sha256'],'original fruit arrays unchanged')
  with np.load(INPUT,allow_pickle=False) as z:data={k:z[k] for k in z.files}
  idlookup={str(i):n for n,i in enumerate(data['record_id'])};truth={(task,str(r)):float(data[target][i]) for task,(kind,target) in TASKS.items() for i,r in enumerate(data['record_id']) if data['fruit_type'][i]==kind}
 else:
  data,_=verify_dataset(a);truth={str(r):str(y) for r,y in zip(data['record_ids'],data['y'],strict=True)}
 # Actual CSV truth semantics are now safe to compare against independently reconstructed originals.
 for c in contexts.values():
  with (B/c['truth']).open(encoding='utf-8-sig',newline='') as f:found=list(csv.DictReader(f))
  a.equal([row['record_id'] for row in found],c['query_ids'],'scoring file exact routing')
  if fruit:a.equal([float(row['target']) for row in found],[truth[(c['task'],rid)] for rid in c['query_ids']],'original workbook scoring truth')
  else:a.equal([row['label'] for row in found],[truth[rid] for rid in c['query_ids']],'original published scoring labels')
 rows=csvgz(B/'outer_predictions.csv.gz');a.check(len(rows)==expected_rows,'actual prediction rows')
 byrole=collections.defaultdict(list)
 for row in rows:byrole[predkey(row,fruit)].append(row)
 a.check(len(byrole)==360,'all360 export/fruit-group role/seed cases')
 roleconfigs={};modelpreds={};fits=collections.defaultdict(set);object_records={};bundles={};predictions_replayed=0;seconds=0.;retained=[]
 for fold in selection['outer_folds']:
  if fruit:foldkey=(fold['task'],fold['outer']);configurations={'dummy_mean':{'id':'dummy_mean','family':'dummy_mean','mc':None,'preprocessing':'raw'},**{role:c['config'] for role,c in fold['choices'].items()}}
  else:foldkey=(fold['outer_group'],);configurations={'R0':protocol['R0'],**{role:c['config'] for role,c in fold['choices'].items()}}
  for role,config in configurations.items():roleconfigs[(*foldkey,role)]=config
 for r in records:
  c=contexts[r['split_id']];config=r['config'];path=B/r['estimator_path'];meta=read(path.with_suffix('.json'));ref=r['transform']
  if ref['path'] not in bundles:
   envelope=joblib.load(B/ref['path']);blob=envelope['payload_bytes'];a.check(hashlib.sha256(blob).hexdigest()==envelope['payload_sha256'],'cache payload SHA before deserialize');t=joblib.load(io.BytesIO(blob));a.check(envelope['identity']==ref['cache_identity']==t.metadata['cache_identity'],'full outer source cache identity');bundles[ref['path']]=t
  t=bundles[ref['path']];est=joblib.load(path);state=joblib.hash((t,est))
  with np.load(B/c['source_pack'],allow_pickle=False) as z:source={k:z[k] for k in z.files}
  with np.load(B/c['query_pack'],allow_pickle=False) as z:query={k:z[k] for k in z.files}
  a.equal(meta['source_ids'],c['source_ids'],'outer model current-source membership');a.equal(meta['query_ids'],c['query_ids'],'outer model query routing');a.check(meta['config']==config and meta['seed']==r['seed'],'outer persisted configuration')
  ratio=config['mc']['ratio'] if config.get('mc') else None
  if fruit:
   z=t.matrix(source['X'],ratio);y=t.y_scaler.transform(source['source_targets'][:,None]).ravel();qz=t.matrix(query['X'],ratio)
   a.check(source_fingerprint(z,y)==meta['estimator']['source_fingerprint'],'outer estimator actual source values/targets');component='fruit.'+config['family'];fits[(r['split_id'],component)].add(source_fingerprint(z,y))
   a.close(t.y_scaler.mean_,[source['source_targets'].mean()],'outer source target mean');a.close(t.y_scaler.var_,[source['source_targets'].var()],'outer source target variance')
   def predict(v):return t.y_scaler.inverse_transform(np.asarray(est.predict(v)).reshape(-1,1)).ravel()
   pred=predict(qz);scores=None
   a.close(predict(qz[::-1])[::-1],pred,'reverse outer regression query');a.close(np.concatenate([predict(qz[i:i+1]) for i in range(len(qz))]),pred,'single/tail outer regression query')
   if config['family'] in ['full_SVR','MC_SVR']:a.close(est.support_vectors_,z[est.support_],'outer SVR source support vectors')
   mk=(c['task'],c['outer'],cfgkey(config,fruit),r['seed'])
  else:
   z=t.matrix({'alpha':source['X_alpha'],'n':source['X_n']},ratio);qz=t.matrix({'alpha':query['X_alpha'],'n':query['X_n']},ratio);reduced=est.reduced(z)
   a.check(source_fingerprint(reduced,source['source_labels'])==meta['estimator']['source_z_fingerprint'],'outer classifier actual source matrix/labels');component={'svm':'SVC','lda':'shrinkage_LDA','pca_lda':'shrinkage_LDA','plsda':'PLSDA.PLSRegression','mlp':'MLPClassifier'}[config['kind']]
   fit_y=np.eye(2)[np.searchsorted(['0','1'],source['source_labels'])] if config['kind']=='plsda' else source['source_labels'];fits[(r['split_id'],component)].add(source_fingerprint(reduced,fit_y))
   if config['kind']=='pca_lda':fits[(r['split_id'],'PCA')].add(source_fingerprint(z));a.close(est.reducer.mean_,z.mean(0),'outer source PCA mean')
   if config['kind']=='svm':a.close(est.estimator.support_vectors_,z[est.estimator.support_],'outer SVC source support vectors')
   pred=est.predict(qz);scores=est.scores(qz);a.close(est.scores(qz[::-1])[::-1],scores,'reverse outer class scores',atol=1e-10);a.close(np.vstack([est.scores(qz[i:i+1]) for i in range(len(qz))]),scores,'single/tail outer class scores',atol=1e-10)
   mk=(c['outer_group'],cfgkey(config,fruit),r['seed'])
  a.check(joblib.hash((t,est))==state,'outer inference no weight/state update');a.equal(meta['selected_indices' if fruit else 'retained_indices'],t.indices(ratio) if fruit else t.selected(ratio),'source-only chosen columns')
  a.check(mk not in modelpreds,'no duplicate model configuration counted as new sample');modelpreds[mk]=(pred,scores,c);object_records[r['estimator_path']]=mk
  seconds+=r['elapsed_seconds'];retained.append(r['retained_features' if fruit else 'selected_features'])
 for rolekey,rr in byrole.items():
  config=roleconfigs[rolekey[:-1]];mk=(*rolekey[:-2],cfgkey(config,fruit),rolekey[-1]);a.check(mk in modelpreds,'all role configs backed by exact current-source learned object');pred,scores,c=modelpreds[mk]
  a.equal([row['record_id'] for row in rr],c['query_ids'],'outer role/query ID alignment');a.check(all(row['role_candidate_id']==config['id'] for row in rr),'role candidate lineage')
  if fruit:a.close([float(row['y_predicted']) for row in rr],pred,'all role regression predictions replayed')
  else:a.equal([row['y_predicted'] for row in rr],pred,'all role class predictions replayed');a.close([[float(row['score_0']),float(row['score_1'])] for row in rr],scores,'all role scores replayed',atol=1e-10)
  predictions_replayed+=len(rr)
 for alias in aliases:
  rolekey=(alias['task'],alias['outer'],alias['role']) if fruit else (alias['outer_group'],alias['role']);mk=(*rolekey[:-1],cfgkey(roleconfigs[rolekey],fruit),alias['seed'])
  a.check(object_records[alias['estimator_path']]==mk,'alias shares only mathematically identical source configuration')
 pooled=collections.defaultdict(list)
 for row in rows:pooled[(row['task'],row['role'],int(row['seed'])) if fruit else (row['role'],int(row['seed']))].append(row)
 saved=read(B/'pooled_metrics_by_seed.json');foldsaved=read(B/'fold_metrics.json');summaries=read(B/'seed_summary.json');recomputed={}
 for key,rr in pooled.items():
  ids=[r['record_id'] for r in rr];a.check(len(ids)==(40 if fruit else 42) and len(set(ids))==len(ids),'pooled denominator includes each actual observation once per role/seed')
  if fruit:
   task,role,seed=key;m=regression_metric([truth[(task,rid)] for rid in ids],[float(r['y_predicted']) for r in rr]);target=next(s for s in saved if(s['task'],s['role'],s['seed'])==key);check_regression_metric(a,target,m,'independent pooled outer regression')
  else:
   role,seed=key;m=metrics([truth[rid] for rid in ids],[r['y_predicted'] for r in rr]);target=saved[role][str(seed)]
   for f in ['accuracy','macro_f1','balanced_accuracy']:a.close(target[f],m[f],'independent pooled outer classification '+f)
   a.equal(target['confusion_matrix'],m['confusion_matrix'],'independent outer confusion');a.check(target['correct']==sum(truth[r['record_id']]==r['y_predicted'] for r in rr),'exact correct count')
  recomputed[key]=m
 for f in foldsaved:
  if fruit:
   rr=byrole[(f['task'],f['outer'],f['role'],f['seed'])];m=regression_metric([truth[(f['task'],r['record_id'])] for r in rr],[float(r['y_predicted']) for r in rr]);check_regression_metric(a,f,m,'independent outer fold')
  else:
   rr=byrole[(f['held_export_group'],f['role'],f['seed'])];m=metrics([truth[r['record_id']] for r in rr],[r['y_predicted'] for r in rr])
   for name in ['accuracy','macro_f1','balanced_accuracy']:a.close(f['metrics'][name],m[name],'independent outer fold metric')
 for s in summaries:
  for name in (['rmse','mae','r2'] if fruit else ['accuracy','macro_f1','balanced_accuracy']):
   vals=[recomputed[(s['task'],s['role'],seed) if fruit else (s['role'],seed)][name] for seed in [42,43,44]]
   a.close(s[name+'_mean'],np.mean(vals),'all-seed summary mean');a.close(s[name+'_sample_sd'],np.std(vals,ddof=1),'all-seed summary sample SD')
 # Preserve source trace prefix exactly and inspect every newly appended outer fit membership.
 prefix=hashlib.sha256();counts=collections.Counter();outer_started=False;context_all={c['split_id']:c for c in prep['contexts']}
 for raw in (B/'fit_scope_events.jsonl').open('rb'):
  e=json.loads(raw);c=context_all[e['split_id']]
  if c['stage']=='inner':a.check(not outer_started,'source events precede all outer fitting');prefix.update(raw);continue
  outer_started=True;ids=e['fit_sample_ids'];a.check(set(ids)<=set(c['source_ids']) and not set(ids)&set(c['query_ids']),'every new outer fit excludes its query observations')
  if e['event']=='fit_start':
   counts[e['component']]+=1
   if (e['split_id'],e['component']) in fits:a.check(e['data_fingerprint'] in fits[(e['split_id'],e['component'])],'actual outer classifier/regressor fit values and targets')
 a.check(prefix.hexdigest()==previous['fit_trace_sha256'],'all source fit trace bytes unchanged after outer execution')
 sc=read(B/'scoring_manifest.json');a.check(sc['prediction_lock_sha256']==sha(B/'prediction_lock.json') and sc['metrics_sha256']==sha(B/'pooled_metrics_by_seed.json') and sc['source_model_fits_in_scoring']==0,'scoring chain and no scoring-time fit')
 metric_table=[dict(zip(['task','role','seed'] if fruit else ['role','seed'],key),**value) for key,value in sorted(recomputed.items())]
 metric_file=OUT/('independently_recomputed_fruit_metrics.json' if fruit else 'independently_recomputed_ginseng_metrics.json');write_new(metric_file,metric_table)
 result={'completed_at_utc':utc(),'status':'OUTER_REPLAY_AND_INDEPENDENT_METRICS_PASS_EXPLORATORY'+('_LOCAL_ONLY_FRUIT' if fruit else '_UNKNOWN_GINSENG_ANCESTRY'),'new_audit_fits':0,'checks':a.checks,'numeric_values':a.numeric_values,'maximum_difference':a.maximum_difference,'all_outer_prediction_rows_replayed':predictions_replayed,'actual_outer_models':len(records),'exact_configuration_alias_cases':len(aliases),'pooled_role_seed_cases':len(pooled),'actual_outer_fit_start_counts':dict(counts),'retained_feature_counts':sorted(set(retained)),'sum_outer_trial_elapsed_seconds':seconds,'all3seeds_included':True,'source_trace_prefix_unchanged':True,'prediction_lock_sha256':sha(B/'prediction_lock.json'),'scoring_manifest_sha256':sha(B/'scoring_manifest.json'),'pooled_metrics_sha256':sha(B/'pooled_metrics_by_seed.json'),'independently_recomputed_metrics_sha256':sha(metric_file),'audit_elapsed_seconds':time.monotonic()-start,'scientific_claims':['Exploratory development evaluation; no fresh independent confirmation','No unknown specimen ancestry, dates, batch or physical axis units invented','Aliases and multiple representations do not increase the observation/fruit denominator','No acid-jujube accuracy or transfer claim follows from this external task'],'script_sha256':sha(__file__)}
 write_new(OUT/('fruit_evaluation_review.json' if fruit else 'evaluation_review.json'),result);print(json.dumps({k:result[k] for k in ['status','checks','all_outer_prediction_rows_replayed','actual_outer_models','new_audit_fits']}))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['ginseng','fruit']);main(parser.parse_args().mode)
