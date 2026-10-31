import csv,gzip,hashlib,json
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
import joblib,numpy as np
from sklearn.preprocessing import StandardScaler
from herb_spectra import public_thz_benchmark as b
from herb_spectra.leakage_guard import LeakageError,load_guarded_cache,CacheKey
from herb_spectra.selection import MCUVESelector

class PublicTHzBenchmarkTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  rng=np.random.default_rng(700);self.axis=np.linspace(.2,1.6,21);self.ids=np.array([f'S{i:02}' for i in range(28)])
  self.y=np.array(['0','1']*14);self.x=rng.normal(size=(28,21))+np.arange(28)[:,None]*.01
  self.pack={'record_ids':self.ids,'X_alpha':self.x,'X_n':self.x*.2+1.,'axis':self.axis,'source_labels':self.y}
  self.context={'split_id':'outer_G1.inner_G2','source_ids':self.ids.tolist()}
  self.cfg={'id':'fixture','representation':'alpha','preprocessing':'snv','kind':'svm','mc':{'runs':10,'components':1,'ratio':.2},'C':1.,'gamma':'scale','class_weight':'balanced'}
 def tearDown(self):self.tmp.cleanup()
 def test_fixed_budget_and_roles(self):
  self.assertEqual(len(b.r1_grid()),36);self.assertEqual(len(b.r2_grid(self.cfg['mc'])),48);self.assertEqual(len(b.r3_grid(self.cfg)),7);self.assertEqual(len(b.r4_grid()),33)
  p=b.default_protocol();self.assertEqual(len(p['outer_roles']),20);self.assertEqual(p['fixed_source_classifier_trials'],11160);self.assertFalse(p['fresh_confirmation'])
 def test_cached_ranking_ratios_match_original_MC_exactly_and_trace_PLS(self):
  cache={};bundle,ref=b.transform_source(self.pack,self.cfg,42,self.context,self.root,cache)
  z=bundle.scalers['alpha'].transform(bundle.preprocessors['alpha'].transform(self.x))
  for ratio in [.1,.2,.4,.6]:
   legacy=MCUVESelector(n_runs=10,n_components=1,n_features_to_select=b.retained_count(21,ratio),sample_fraction=.8,task='classification',random_state=42).fit(z,self.y)
   np.testing.assert_array_equal(bundle.matrix(b.features(self.pack),ratio),legacy.transform(z))
   np.testing.assert_array_equal(bundle.selected(ratio),legacy.selected_indices_)
  before=(self.root/'fit_scope_events.jsonl').read_bytes();again,second=b.transform_source(self.pack,{**self.cfg,'C':100.},42,self.context,self.root,cache)
  self.assertTrue(second['reused']);self.assertEqual(before,(self.root/'fit_scope_events.jsonl').read_bytes())
  events=[json.loads(line) for line in before.splitlines()];completed=[e for e in events if e['event']=='fit_completed']
  self.assertEqual(sum(e['component']=='MC-UVE.PLSRegression.fit' for e in completed),20)
  self.assertTrue(all(set(e['fit_sample_ids'])<=set(self.ids) for e in completed))
  loaded=load_guarded_cache(self.root/ref['path'],CacheKey(ref['cache_identity']));np.testing.assert_array_equal(loaded.matrix(b.features(self.pack),.2),bundle.matrix(b.features(self.pack),.2))
 def test_cache_cannot_mutate_or_cross_source(self):
  cache={};bundle,ref=b.transform_source(self.pack,self.cfg,42,self.context,self.root,cache)
  bundle.scalers['alpha'].mean_[0]+=1
  with self.assertRaisesRegex(LeakageError,'cache bytes/state'):b.transform_source(self.pack,self.cfg,42,self.context,self.root,cache)
  with self.assertRaisesRegex(LeakageError,'IDs/order'):b.transform_source(self.pack,self.cfg,42,{**self.context,'source_ids':['forbidden',*self.ids[1:].tolist()]},self.root,cache)
  bad={**self.pack,'record_ids':np.array(['forbidden',*self.ids[1:]])}
  with self.assertRaisesRegex(LeakageError,'IDs/order'):b.transform_source(bad,self.cfg,42,self.context,self.root,{})
 def test_SVC_native_vote_and_binary_scores_and_pointwise_state(self):
  bundle,_=b.transform_source(self.pack,self.cfg,42,self.context,self.root,{})
  z=bundle.matrix(b.features(self.pack),.2);est=b.fit_estimator(z,self.y,self.ids,self.cfg,42,self.context,self.root)
  # Never use a score argmax to override the fitted SVC's own vote.
  q={k:v for k,v in self.pack.items() if k!='source_labels'};pred,scores,audit=b.inference_check(bundle,est,q,.2)
  np.testing.assert_array_equal(pred,est.estimator.predict(z));self.assertEqual(scores.shape,(28,2));self.assertTrue(audit['state_unchanged'])
  margin=est.estimator.decision_function(z);np.testing.assert_array_equal(scores,np.column_stack([-margin,margin]))
 def test_all_family_fits_actual_PCA_and_PLS_source_only(self):
  z=StandardScaler().fit_transform(self.x)
  for kind,extra in [('lda',{}),('pca_lda',{'components':2}),('plsda',{'components':2}),('mlp',{'hidden':[4],'l2':.01,'epochs':3})]:
   with self.subTest(kind=kind):
    c={**self.cfg,'kind':kind,'mc':None,**extra};m=b.fit_estimator(z,self.y,self.ids,c,42,self.context,self.root)
    self.assertEqual(m.predict(z[:7]).shape,(7,));self.assertEqual(m.scores(z[:7]).shape,(7,2))
  events=[json.loads(line) for line in (self.root/'fit_scope_events.jsonl').read_text().splitlines()]
  parts={e['component'] for e in events if e['event']=='fit_completed'}
  self.assertTrue({'PCA','PLSDA.PLSRegression','MLPClassifier','shrinkage_LDA'}<=parts)
 def test_modalities_are_preprocessed_separately_not_across_concat_boundary(self):
  c={**self.cfg,'representation':'paired_alpha_n','preprocessing':'sg1_snv','mc':None}
  bundle,_=b.transform_source(self.pack,c,42,self.context,self.root,{})
  z=bundle.matrix(b.features(self.pack));self.assertEqual(z.shape,(28,42))
  a=bundle.scalers['alpha'].transform(bundle.preprocessors['alpha'].transform(self.x))
  n=bundle.scalers['n'].transform(bundle.preprocessors['n'].transform(self.pack['X_n']))
  np.testing.assert_array_equal(z,np.column_stack([a,n]))
 def test_strict_inference_schema_rejects_truth_and_metadata(self):
  q={k:v for k,v in self.pack.items() if k!='source_labels'};p=self.root/'query.npz';np.savez(p,**q)
  self.assertEqual(b.strict_pack(p)['X_alpha'].shape,(28,21))
  for key in ['y','labels','label','export_groups','X_metadata']:
   np.savez(p,**q,**{key:self.y})
   with self.assertRaises(LeakageError):b.strict_pack(p)
 def test_source_gate_cannot_use_other_code(self):
  manifest={'protocol_sha256':'p','preparation_sha256':'x','source_sha256':{'module':'c'}}
  with patch.object(b,'verify_run',return_value=({},manifest)):
   b.source_release(self.root,{'allow_source_training':True,**manifest})
   with self.assertRaises(LeakageError):b.source_release(self.root,{'allow_source_training':True,**manifest,'preparation_sha256':'different'})
 def test_score_checks_last_truth_before_any_first_semantic_parse(self):
  # Full frozen fixture, no modelling or real dataset labels in this test.
  for name in ['outer_predictions.csv.gz','outer_model_records.json','outer_role_aliases.json','selection_lock.json','outer_release_snapshot.json','run_manifest.json','protocol.json']:
   (self.root/name).write_text('[]')
  contexts=[]
  for i in range(2):
   f=self.root/f'truth{i}.csv';f.write_text('record_id,label\na,0\nb,1\n')
   contexts.append({'stage':'outer','truth':f.name,'truth_sha256':b.sha(f),'query_ids':['a','b']})
  lock={'created_at_utc':'fixture','outer_truth_parsed':False,'actual_outer_estimators':0,'prediction_rows':2520}
  keys={'prediction_sha256':'outer_predictions.csv.gz','outer_model_records_sha256':'outer_model_records.json','outer_role_aliases_sha256':'outer_role_aliases.json','selection_lock_sha256':'selection_lock.json','root_outer_release_sha256':'outer_release_snapshot.json','run_manifest_sha256':'run_manifest.json','protocol_sha256':'protocol.json'}
  lock.update({key:b.sha(self.root/name) for key,name in keys.items()});b.write(self.root/'prediction_lock.json',lock)
  with patch.object(b,'verify_selected',return_value=({'contexts':contexts},{},{})),patch.object(b,'verify_outer_release'):
   b.verify_predictions_before_truth(self.root)
   (self.root/'truth1.csv').write_text('record_id,label\na,1\nb,0\n')
   with patch.object(b,'load_truth') as read_truth:
    with self.assertRaisesRegex(LeakageError,'truth bytes'):b.score(self.root)
    read_truth.assert_not_called()
 def test_prediction_tamper_and_false_truth_lock_fail_before_read(self):
  (self.root/'prediction_lock.json').write_text(json.dumps({'created_at_utc':'fixture','outer_truth_parsed':True}))
  with patch.object(b,'verify_selected',return_value=({'contexts':[]},{},{})),patch.object(b,'load_truth') as truth:
   with self.assertRaises(LeakageError):b.score(self.root)
   truth.assert_not_called()
 def test_persisted_model_hash_verified_before_deserialization(self):
  p=self.root/'model.joblib';p.write_bytes(b'not a pickle');meta=p.with_suffix('.json');meta.write_text('{}')
  rec={'estimator_path':p.name,'estimator_sha256':'wrong','metadata_sha256':b.sha(meta)}
  with patch.object(b.joblib,'load') as load:
   with self.assertRaises(LeakageError):b.load_fitted_record(self.root,rec)
   load.assert_not_called()
 def test_choice_tie_and_failed_candidate_exclusion(self):
  good={'config':{'id':'b'},'cases':15,'status':'SOURCE_EVALUATED','mean_inner_metrics':{k:.5 for k in b.METRICS}}
  a={**good,'config':{'id':'a'}};bad={**good,'config':{'id':'first'},'status':'FAILED_INNER_CANDIDATE','cases':14}
  self.assertEqual(b.choose([good,a,bad])['config']['id'],'a')

if __name__=='__main__':unittest.main()
