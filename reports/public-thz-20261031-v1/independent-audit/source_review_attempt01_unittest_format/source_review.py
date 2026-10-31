"""Read-only, zero-fit independent review of the frozen ginseng preparation."""
import csv, hashlib, importlib.metadata, io, itertools, json
from pathlib import Path
from audit_common import Audit, ROOT, OUT, read, sha, utc, verify_dataset, write_new
import numpy as np

B=ROOT/'reports/public-thz-20261031-v1/ginseng-benchmark'
EXPECTED={
 'src/herb_spectra/public_thz_benchmark.py':'84e43169ceabd47d847ca4938dc07d78201a6bf747ef9c5acd2909d18ec3959e',
 'scripts/run_public_thz_benchmark.py':'0dbab55e56b5f978f5a1c00bbfcf02b4e8ac15037c2b85c8b1f83df6115b2117',
 'tests/test_public_thz_benchmark.py':'50e36c199b9cccb4de7266d3129f50e029c1aad0e9e596be3342b9ac7ab2f855',
 'src/herb_spectra/public_thz_data.py':'99a61593a0170ec722aa3b28762fc6f292ca69b80c5c95cf02cd78e2ac3fb948'}
FROZEN={'run_manifest.json':'170830f0cb4501b7f59f49f6591ecb07591c1b8327a2dd6a1ee18d21a885538a',
 'protocol.json':'13e4df33eab9bb0b6d48483ff93139a180d7665518a30116be212fdf5063b210',
 'preparation.json':'8d7a0fffcb2e91fcb2b15af51b84d0cc86dc12db92bff14b9059fea04c295d06',
 'split_manifest.json':'c2c84632074ab35b3dce0ddafb8f0587c38efecd09ac58a7d371dbe133e51d3c',
 'boundary_test_provenance.json':'9c39033b7f07f24990a35ede7da4ecd28657d7d8b3e8605daa430fb1256c1631'}

def main():
 a=Audit();data,dm=verify_dataset(a)
 for p,h in EXPECTED.items():a.check(sha(ROOT/p)==h,'reviewed source '+p)
 for p,h in FROZEN.items():a.check(sha(B/p)==h,'frozen artifact '+p)
 m=read(B/'run_manifest.json');p=read(B/'protocol.json');prep=read(B/'preparation.json');sp=read(B/'split_manifest.json')
 for k in ['protocol','preparation','split_manifest']:a.check(m[k+'_sha256']==sha(B/(k+'.json')),k+' manifest')
 for f,h in m['source_sha256'].items():a.check(sha(ROOT/f)==h,'full dependency '+f)
 for name,version in m['environment'].items():a.check(importlib.metadata.version(name)==version,'environment '+name)
 t=read(B/'boundary_test_provenance.json')
 for f,h in t['files_sha256'].items():a.check(sha(ROOT/f)==h,'actual tests source '+f)
 a.check(sha(B/'boundary-tests.log')==t['test_log_sha256'],'actual boundary log bytes')
 a.check(sha(B/'boundary-test-events.jsonl')==t['command_events_sha256'],'actual boundary command event bytes')
 a.check('12 passed' in (B/'boundary-tests.log').read_text(),'actual twelve tests pass')
 a.check(t['research_model_fits']==0 and t['source_models_still_not_started'],'synthetic boundary tests not research fits')
 a.equal(p['seeds'],[42,43,44],'fixed three seeds')
 a.check(p['primary_seed']==42 and p['classes']==['0','1'],'fixed primary and classes')
 a.check(len(p['R1'])==36 and len(p['R2_base'])==48 and len(p['R3_preprocessors'])==7 and len(p['R4'])==33,'fixed staged budget')
 for c,triplet in zip(p['R1'],itertools.product([10,20,50],[1,2,3],[.1,.2,.4,.6]),strict=True):
  r,l,q=triplet;a.check(c['mc']==dict(runs=r,components=l,ratio=q),'R1 exact candidate')
  a.check(c['kind']=='svm' and c['representation']=='alpha' and c['preprocessing']=='snv' and c['C']==10 and c['gamma']=='scale' and c['class_weight']=='balanced','R1 fixed surrounding model')
 for c,(C,g,w) in zip(p['R2_base'],itertools.product([.01,.1,1.,10.,100.,1000.],['scale','auto','scale_x0.1','scale_x10'],['none','balanced']),strict=True):
  a.check(c['C']==C and c['gamma']==g and c['class_weight']==w,'R2 exact candidate')
 a.equal(p['R3_preprocessors'],['raw','snv','sg0_snv','sg1_snv','sg2_snv','msc','linear_detrend_snv'],'R3 preprocessing')
 for rep in ['alpha','n','paired_alpha_n']:
  rows=[c for c in p['R4'] if c['representation']==rep]
  a.check(len(rows)==11,'eleven per representation')
  a.check({f:sum(c['family']==f for c in rows) for f in ['full_svm','shrinkage_lda','pca_lda','plsda','mlp']}=={'full_svm':2,'shrinkage_lda':1,'pca_lda':2,'plsda':2,'mlp':4},'R4 family budget')
  a.check(all(c['mc'] is None and c['preprocessing']=='raw' for c in rows),'R4 fixed raw full-width')
  a.check(sorted(c['C'] for c in rows if c['family']=='full_svm')==[1.,10.],'R4 SVM')
  a.check(sorted(c['components'] for c in rows if c['family']=='pca_lda')==[2,5],'R4 PCA')
  a.check(sorted(c['components'] for c in rows if c['family']=='plsda')==[2,3],'R4 PLS')
  a.check({(tuple(c['hidden']),c['l2'],c['epochs']) for c in rows if c['family']=='mlp'}=={(h,l,300) for h in [(32,),(32,16)] for l in [.01,.1]},'R4 MLP')
 expected_roles=['R0','R1','R2','R3','R4_policy']+[f'family_{r}_{f}' for r in ['alpha','n','paired_alpha_n'] for f in ['full_svm','shrinkage_lda','pca_lda','plsda','mlp']]
 a.equal(p['outer_roles'],expected_roles,'all twenty independently evaluated roles')
 a.check(p['fixed_source_classifier_trials']==6*5*3*(36+48+7+33)==11160,'source classifier fit count')
 a.check(p['outer_role_seed_cases']==60 and p['real_observations_per_role_seed']==42,'no modality or role pseudo replication')
 a.check(p['scientific_status']==dm['scientific_status'] and p['fresh_confirmation'] is False and p['no_independent_plants_assumed'] is True and p['OS_truth_file_sandbox'] is False,'scope limitations explicit')
 a.check(prep['features_or_labels_cleaned'] is False and 'not geographical origin' in prep['label_scope'],'unaltered species labels')
 a.check(sp['rows_are_new_independent_plants'] is False and len(sp['folds'])==6,'no invented plants')
 groups=data['export_groups'];ids=data['record_ids'];contexts=prep['contexts'];a.check(len(contexts)==36,'six outer plus thirty inner contexts')
 expected_splits=[];outer_ids=[]
 for outer in sorted(set(groups)):
  expected_splits.append((f'outer_{outer}.final','outer',outer,None))
  expected_splits.extend((f'outer_{outer}.inner_{inner}','inner',outer,inner) for inner in sorted(set(groups)) if inner!=outer)
 for c,(split,stage,outer,inner) in zip(contexts,expected_splits,strict=True):
  s=(groups!=outer) if stage=='outer' else (groups!=outer)&(groups!=inner);q=groups==(outer if stage=='outer' else inner)
  a.check((c['split_id'],c['stage'],c['outer_group'],c['inner_group'])==(split,stage,outer,inner),'ordered context identity')
  a.equal(c['source_ids'],ids[s],'context source IDs');a.equal(c['query_ids'],ids[q],'context query IDs')
  a.equal(c['source_groups'],sorted(set(groups[s])),'source groups');a.equal(c['query_groups'],sorted(set(groups[q])),'query groups')
  a.check(not set(c['source_ids'])&set(c['query_ids']) and not set(c['source_groups'])&set(c['query_groups']),'no source/query role overlap')
  a.check(not (set(c['source_ids'])|set(c['query_ids']))&set(ids[groups==outer]) if stage=='inner' else True,'outer records absent in all inner operations')
  for role,ix in [('source',s),('query',q)]:
   f=B/c[role+'_pack'];a.check(sha(f)==c[role+'_pack_sha256'],'pack SHA')
   with np.load(f,allow_pickle=False) as z:
    allowed={'record_ids','X_alpha','X_n','axis'}|({'source_labels'} if role=='source' else set())
    a.check(set(z.files)==allowed,'strict metadata whitelist')
    for k in ['record_ids','X_alpha','X_n']:a.equal(z[k],data[k][ix],'actual pack '+k)
    a.equal(z['axis'],data['axis'],'exact native axis')
    if role=='source':a.equal(z['source_labels'],data['y'][ix],'actual source supervision');a.check(set(z['source_labels'])=={'0','1'},'both source classes')
  # Only byte-compare expected truth serialization. Do not semantically parse outer truth CSV.
  text=io.StringIO(newline='');writer=csv.DictWriter(text,fieldnames=['record_id','label'],lineterminator='\n');writer.writeheader()
  writer.writerows({'record_id':str(r),'label':str(y)} for r,y in zip(ids[q],data['y'][q],strict=True))
  expected_hash=hashlib.sha256(text.getvalue().encode('utf-8-sig')).hexdigest()
  a.check(sha(B/c['truth'])==c['truth_sha256']==expected_hash,'expected truth bytes and labels bound without scoring')
  if stage=='outer':outer_ids.extend(c['query_ids'])
 a.equal(sorted(outer_ids),sorted(ids),'every observation exactly once in outer roles')
 a.check(not (B/'selection_lock.json').exists() and not (B/'fit_scope_events.jsonl').exists(),'main source models still not started at review')
 result={'completed_at_utc':utc(),'status':'SOURCE_REVIEW_PASS_FOR_FROZEN_RUN_EXPLORATORY_UNKNOWN_UPSTREAM','new_audit_fits':0,'checks':a.checks,'numeric_values':a.numeric_values,'maximum_difference':a.maximum_difference,'frozen_artifacts_sha256':FROZEN,'reviewed_source_sha256':EXPECTED,'source_classifier_fit_budget':11160,'outer_roles':20,'expected_outer_prediction_rows':2520,'scientific_independence_certified':False,'truth_scored':False,'review_scope':['Independent raw18CSV reconstruction and label/axis/lineage binding','All36 current-source/query arrays, IDs, hashes and truth byte serializations','Fixed source grid and family roles','Actual12 synthetic boundary-test log and code dependency hashes','Read-only source inspection: FitScope source-only scalers/PCA/PLS/MC internal PLS/SVC/MLP; full state cache and hash locks; binary native SVC votes; all outer prediction/truth hashes checked before scoring'],
 'limitations':['Only frozen ginseng run is approved technically; no verified root ancestry or biological batches','The independent auditor already knows public dataset labels for partition verification; no model selection or training is performed by this review','Not an OS sandbox or absolute zero-leakage proof','Standalone transform_source cache identity lacks a separate axis fingerprint. Frozen run binds all pack/axis bytes; a different axis requires a separately frozen run/API hardening','Repeated outer development evaluation is exploratory and cannot be called new independent confirmation','MC one-hot targets require two internal PLS fits per repeat; classify budget and internal PLS budget must be reported separately'], 'script_sha256':sha(__file__)}
 write_new(OUT/'source_review.json',result);print(json.dumps({k:result[k] for k in ['status','checks','numeric_values','maximum_difference','new_audit_fits']}))

if __name__=='__main__':main()
