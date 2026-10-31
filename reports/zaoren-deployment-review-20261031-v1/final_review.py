"""Independent three-model full-development refit and real serving checks.

No resubstitution accuracy is calculated; source labels bind training provenance
only. Historical copies are used solely for explicit intake/re-export tests.
"""
import argparse,csv,hashlib,json,shutil,tempfile
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
import review_common as c
ROOT=c.ROOT;OUT=Path(__file__).resolve().parent

def main(run):
 run=Path(run).resolve();bundle=run/'deployment';dest=OUT/'final_review.json'
 if dest.exists():raise FileExistsError(dest)
 from herb_spectra import deployment as d
 from herb_spectra.leakage_guard import LeakageError
 start=datetime.now(timezone.utc).isoformat();selection=c.read(OUT/'selection_review.json')
 protocol,prep,lock=d.verify_selection(run);manifest,models=d.load_bundle(bundle)
 c.check(d.code_versions()==selection['source_sha256'],'Unchanged sources after selection')
 c.check(c.sha(run/'selection_lock.json')==selection['selection_lock_sha256'],'Source selection lock unchanged')
 c.check(manifest['selected_config']==selection['selected_config']==lock['selected'],'Full model config matches development selection')
 c.check(manifest['source_count']==855 and manifest['external_accuracy'] is None,'No invented external accuracy')
 c.check(not manifest['resubstitution_accuracy_reported_as_performance'] and not manifest['historical_96p02_is_this_models_accuracy'],'Training fit not reported as performance')
 c.check(manifest['full_fit_release_sha256']==c.sha(bundle/'full_fit_release.json'),'Actual full release preserved')
 release=c.read(bundle/'full_fit_release.json')
 c.check(release['selection_lock_sha256']==c.sha(run/'selection_lock.json') and release['allow_full_fit'],'Full fit independently gated')
 x,y,ids,axis=d.load_data()
 with np.load(bundle/'history_reference.npz',allow_pickle=False) as z:
  c.check(set(z.files)=={'X_alpha','record_ids','axis'},'History QC has no target labels')
  c.check(np.array_equal(z['X_alpha'],x) and np.array_equal(z['record_ids'],ids) and np.array_equal(z['axis'],axis),'Historical reference exact canonical values')
 h=hashlib.sha256();remaining=selection['source_fit_scope_bytes']
 with (run/'fit_scope_events.jsonl').open('rb') as f:
  while remaining:
   block=f.read(min(remaining,2**20));c.check(bool(block),'Source scope truncated');h.update(block);remaining-=len(block)
 c.check(h.hexdigest()==selection['source_fit_scope_sha256'],'Original source scope prefix unchanged')
 scope=c.ScopeReader(run/'fit_scope_events.jsonl',manifest['source_sha256']['src/herb_spectra/deployment.py'])
 scope.stream.close();scope.stream=(run/'fit_scope_events.jsonl').open('rb');scope.stream.seek(selection['source_fit_scope_bytes'])
 model_reports=[]
 for seed in [42,43,44]:
  item=json.loads(json.dumps(manifest['models'][str(seed)]));item['path']=str(bundle/item['path'])
  model,z,selected=c.model_record(item,manifest['selected_config'],x,y,ids,axis,seed)
  scope.trial(model,manifest['selected_config'],x,y,ids,z,selected,seed,'full_development855')
  before=joblib.hash(model);scores=model.predict_scores(x)
  c.close(model.predict_scores(x[::-1])[::-1],scores,'Full model reverse order',atol=1e-12,rtol=1e-12)
  singles=np.vstack([model.predict_scores(x[i:i+1]) for i in range(len(x))])
  c.close(singles,scores,'Full model one-row inference',atol=1e-12,rtol=1e-12)
  chunks=np.vstack([model.predict_scores(x[i:i+13]) for i in range(0,len(x),13)])
  c.close(chunks,scores,'Full model nondividing batch13',atol=1e-12,rtol=1e-12)
  c.check(joblib.hash(model)==before,'Every model state unchanged by inference')
  model_reports.append({'seed':seed,'parameters':manifest['selected_config'],'source_samples':855,'selected_features':len(model.bundle.selector.selected_indices_),'model_sha256':item['sha256'],'pointwise_records_checked':855,'accuracy_not_calculated':True})
 scope.finish();c.check(scope.ops==1209,'Three full403-operation fits')
 # A real original spectrum under a new ID and tiny re-export roundoff remains
 # historical. It is not called a new sample and is rejected before predictions.
 diagnostic_result={}
 with tempfile.TemporaryDirectory() as td:
  t=Path(td);old=t/'known.npz';renamed=t/'renamed-roundtrip.npz'
  np.savez_compressed(old,X_alpha=x[:5],record_ids=ids[:5],axis=axis)
  np.savez_compressed(renamed,X_alpha=x[:1]+1e-14,record_ids=np.array(['ROUNDTRIP-OLD-001']),axis=axis)
  with patch('sklearn.svm.SVC.fit',side_effect=AssertionError('Serving cannot fit')):
   diagnostic_result=d.predict(bundle,old,t/'diagnostic',mode='training-diagnostic')
   try:d.predict(bundle,renamed,t/'forbidden',mode='future',confirm_compatible_measurement=True)
   except LeakageError: pass
   else:raise AssertionError('Renamed roundtrip accepted as future')
  c.check(not (t/'forbidden').exists(),'Historical rejection writes no forecast output')
  c.check(diagnostic_result['new_model_fits']==0 and diagnostic_result['accuracy'] is None and not diagnostic_result['labels_received'],'Real diagnostic explicitly unscored')
  with (t/'diagnostic/predictions.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
  c.check(len(rows)==5 and all(r['prediction_scope']=='TRAINING_DIAGNOSTIC_NOT_PERFORMANCE' and r['historical_id_match']=='True' for r in rows),'Real diagnostic scope flags')
  # Alter only a temporary copy, leaving every real bundle byte unchanged.
  copy=t/'tampered';shutil.copytree(bundle,copy)
  first=copy/manifest['models']['42']['path']
  with first.open('ab') as f:f.write(b'TAMPER_TEST_NOT_A_MODEL')
  with patch('herb_spectra.deployment.joblib.load') as loader:
   try:d.load_bundle(copy)
   except LeakageError:pass
   else:raise AssertionError('Tampered model accepted')
   c.check(not loader.called,'Tampered first model refused before deserialization')
 c.check(d.code_versions()==selection['source_sha256'],'Code remained fixed during final checks')
 result={'status':'TECHNICAL_DEPLOYMENT_FULL_REFIT_AND_SERVING_REVIEW_PASS_EXPLORATORY','started_at_utc':start,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'checks':c.CHECKS,'maximum_numeric_difference':c.MAXIMUM,'actual_full_models_replayed':3,'seeds':[42,43,44],'pointwise_records_per_full_model':855,'source_log_prefix_unchanged':True,'actual_full_fit_scope_rows':scope.rows,'actual_full_learned_operations':scope.ops,'total_source_and_full_learned_operations':selection['actual_source_learned_operations']+scope.ops,'full_bundle_manifest_sha256':c.sha(bundle/'bundle_manifest.json'),'full_release_sha256':c.sha(bundle/'full_fit_release.json'),'selection_lock_sha256':c.sha(run/'selection_lock.json'),'selection_review_sha256':c.sha(OUT/'selection_review.json'),'source_sha256':d.code_versions(),'model_reports':model_reports,'actual_diagnostic_records_checked':5,'renamed_roundtrip_future_refused':True,'tampered_model_refused_before_deserialization':True,'resubstitution_accuracy_computed':False,'external_accuracy':None,'new_estimator_fits':0,'all855_are_historically_exposed_development':True,'physical_unit_compatibility_verified':False,'unknown_group_detection_validated':False,'OS_sandbox':False,'review_script_sha256':c.sha(Path(__file__)),'review_common_sha256':c.sha(OUT/'review_common.py')}
 dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':result['status'],'checks':c.CHECKS,'maximum_numeric_difference':c.MAXIMUM,'sha256':c.sha(dest)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);main(p.parse_args().run)
