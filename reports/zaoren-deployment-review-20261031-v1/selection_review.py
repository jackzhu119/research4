"""Replay75 source-development models and all logged actual training inputs."""
import argparse,csv,gzip,hashlib,json
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import review_common as c
ROOT=c.ROOT;OUT=Path(__file__).resolve().parent

def main(run):
 run=Path(run).resolve();dest=OUT/'selection_review.json'
 if dest.exists():raise FileExistsError(dest)
 from herb_spectra import deployment as d
 start=datetime.now(timezone.utc).isoformat();source=c.read(OUT/'source_review.json')
 protocol,prep,manifest=d.verify_run(run);c.check(manifest['source_sha256']==source['source_sha256'],'Source review code binding')
 lock=c.read(run/'selection_lock.json');records=c.read(run/'selection_records.json')
 for key,name in [('manifest_sha256','candidate_manifest.json'),('source_release_sha256','source_release.json'),('selection_records_sha256','selection_records.json'),('predictions_sha256','development_predictions.csv')]:c.check(lock[key]==c.sha(run/name),'Frozen selection checksum')
 c.check(lock['source_models']==len(records)==75,'Complete75source cohort')
 c.check(not(run/'deployment').exists(),'No full855refit before selection review')
 x,y,ids,axis=d.load_data();lookup={rid:i for i,rid in enumerate(ids)}
 with (run/'development_predictions.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 bycontext=defaultdict(list)
 for row in rows:bycontext[(row['candidate_id'],int(row['fold']),int(row['seed']))].append(row)
 c.check(len(rows)==12825 and len(bycontext)==75,'Development prediction cohort')
 contexts=[(cand,fold,seed) for cand in protocol['candidates'] for fold in protocol['folds'] for seed in protocol['seeds']]
 scope=c.ScopeReader(run/'fit_scope_events.jsonl',manifest['source_sha256']['src/herb_spectra/deployment.py'])
 output=[];means=defaultdict(list)
 for actual,(config,fold,seed) in zip(records,contexts,strict=True):
  key=(config['id'],fold['fold'],seed);c.check((actual['candidate_id'],actual['fold'],actual['seed'])==key,'Deterministic source trial sequence')
  tr=np.array([lookup[v] for v in fold['train_ids']]);va=np.array([lookup[v] for v in fold['validation_ids']])
  model,z,selected=c.model_record(actual,config,x[tr],y[tr],ids[tr],axis,seed)
  pre=model.bundle.scaler.transform(c.snv(x[va]))[:,model.bundle.selector.selected_indices_]
  before=__import__('joblib').hash(model)
  pred=model.estimator.predict(pre);scores=model.estimator.decision_function(pre)
  c.check(__import__('joblib').hash(model)==before,'Inference state unchanged')
  saved=bycontext[key];c.check([r['record_id'] for r in saved]==ids[va].tolist(),'Development validation ID/order')
  c.check([r['y_true_development_only'] for r in saved]==y[va].tolist(),'Truth only used for development scoring')
  c.check([r['y_predicted'] for r in saved]==pred.tolist(),'Stored native-vote labels')
  c.close(np.array([[float(r['score_'+cl]) for cl in d.CLASSES] for r in saved]),scores,'Saved decision scores',atol=0,rtol=0)
  metric=c.metrics(y[va],pred,d.CLASSES)
  for k in ['accuracy','macro_f1','balanced_accuracy']:c.close(np.array([metric[k]]),np.array([actual['metrics'][k]]),'Independent '+k)
  c.check(metric['confusion_matrix']==actual['metrics']['confusion_matrix'],'Independent confusion counts')
  means[config['id']].append(metric);output.append({'candidate_id':config['id'],'fold':fold['fold'],'seed':seed,**{k:metric[k] for k in ['accuracy','macro_f1','balanced_accuracy']}})
  scope.trial(model,config,x[tr],y[tr],ids[tr],z,selected,seed,f"development.fold{fold['fold']:02}")
 scope.finish();c.check(scope.ops==21825,'Expected actual learned operations')
 ranked=[]
 for config,saved in zip(protocol['candidates'],lock['candidate_means'],strict=True):
  own={k:float(np.mean([r[k] for r in means[config['id']]])) for k in ['macro_f1','balanced_accuracy','accuracy']}
  c.check(saved['candidate']==config and saved['cases']==len(means[config['id']])==15,'Candidate mean denominator')
  for k,v in own.items():c.close(np.array([v]),np.array([saved['mean_development_metrics'][k]]),'Candidate mean '+k)
  ranked.append((tuple(-own[k] for k in ['macro_f1','balanced_accuracy','accuracy'])+(config['id'],),config))
 winner=min(ranked,key=lambda item:item[0])[1];c.check(winner==lock['selected'],'Full deployment choice from development scores only')
 # Preserve exact source-log bytes before the separately gated three full fits.
 prefix=OUT/'selection_fit_scope_prefix.jsonl.gz';h=hashlib.sha256();size=0
 with (run/'fit_scope_events.jsonl').open('rb') as src,gzip.open(prefix,'xb') as destfile:
  for block in iter(lambda:src.read(2**20),b''):h.update(block);size+=len(block);destfile.write(block)
 c.check(d.current_manifest(run)==manifest,'Frozen sources/preparation after replay')
 result={'status':'TECHNICAL_DEPLOYMENT_SELECTION_REPLAY_PASS_EXPLORATORY','started_at_utc':start,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'checks':c.CHECKS,'maximum_numeric_difference':c.MAXIMUM,'models_replayed':75,'development_prediction_rows':12825,'candidate_mean_cases':5,'selected_config':winner,'actual_source_fit_scope_rows':scope.rows,'actual_source_learned_operations':scope.ops,'source_fit_scope_sha256':h.hexdigest(),'source_fit_scope_bytes':size,'source_fit_scope_archive_sha256':c.sha(prefix),'selection_lock_sha256':c.sha(run/'selection_lock.json'),'source_review_sha256':c.sha(OUT/'source_review.json'),'source_sha256':manifest['source_sha256'],'candidate_manifest_sha256':c.sha(run/'candidate_manifest.json'),'new_estimator_fits':0,'new_calibrator_fits':0,'all855_historically_exposed_development':True,'independent_confirmation':False,'external_accuracy':None,'review_script_sha256':c.sha(Path(__file__)),'review_common_sha256':c.sha(OUT/'review_common.py')}
 (OUT/'selection_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 with (OUT/'source_model_metrics.csv').open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
 print(json.dumps({'status':result['status'],'checks':c.CHECKS,'maximum_numeric_difference':c.MAXIMUM,'sha256':c.sha(OUT/'selection_review.json')}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);main(p.parse_args().run)
