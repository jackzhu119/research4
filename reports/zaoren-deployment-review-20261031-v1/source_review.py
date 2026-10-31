"""Independent pre-research source review; no medicinal-model fits are run.

The subprocess runs explicit synthetic unit fixtures. Review assertions bind
original frozen bytes, deployment metadata, fold memberships and real intake
values. Finite checks do not authenticate batches or prove zero leakage.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def now():return datetime.now(timezone.utc).isoformat()
def check(condition,label):
 global checks
 checks+=1
 if not condition:raise AssertionError(label)

def main(run):
 global checks
 checks=0
 for name in ['source_review.json','boundary_tests.log','boundary_test_events.json']:
  if (OUT/name).exists():raise FileExistsError('Review cannot overwrite '+name)
 start=now();run=Path(run).resolve()
 module=importlib.import_module('herb_spectra.deployment')
 io=importlib.import_module('herb_spectra.deployment_io')
 code=module.code_versions();test_paths=[ROOT/'tests/test_deployment.py',ROOT/'tests/test_deployment_io.py']
 test_sha={str(p.relative_to(ROOT)):sha(p) for p in test_paths}
 manifest=read(run/'candidate_manifest.json');prep=read(run/'preparation.json');p=read(run/'protocol.json')
 check(manifest==module.current_manifest(run),'Complete current manifest differs')
 check(code==prep['source_sha256']==manifest['source_sha256'],'Frozen source differs')
 check(sha(ROOT/module.DATA)==module.DATA_SHA==p['data_sha256'],'Canonical array binding')
 original={}
 for name in [module.SOURCE_LOCK,module.SPLITS,'data/active_workbooks.json']:
  local=(ROOT/name).read_bytes()
  stored=subprocess.check_output(['git','show','33b31e1db2e30ed893962a1eac84c1d3d1e82e65:'+name],cwd=ROOT)
  check(local==stored,'Existing frozen input changed: '+name);original[name]=sha(ROOT/name)
 x,y,ids,axis=module.load_data();lookup={rid:i for i,rid in enumerate(ids)}
 check(x.shape==(855,612) and y.shape==ids.shape==(855,) and axis.shape==(612,),'Source dimensions')
 check(dict(Counter(y))=={'ZY01':350,'ZY02':5,'ZY03':100,'ZY04':100,'ZY05':100,'ZY06':100,'ZY07':100},'Seven-class counts')
 check(prep['record_ids']==ids.tolist() and prep['axis']==axis.tolist(),'Preparation ID/axis order')
 check(p['seeds']==[42,43,44] and p['primary_seed']==42,'No best-seed choice')
 check(p['expected_source_models']==75 and p['expected_full_models']==3,'Finite frozen budget')
 check(p['all_855_are_development'] and not p['development_cv_is_independent_confirmation'],'Data exposure declaration')
 check(p['new_external_accuracy'] is None and not p['historical_96p02_is_deployment_accuracy'],'No invented external accuracy')
 lock=read(ROOT/module.SOURCE_LOCK);oldfolds=read(ROOT/module.SPLITS)
 wanted=[]
 for fold in lock['outer_folds']:
  choice=fold['choices']['R2'];check(choice['selected_using_outer_test'] is False,'Outer score used in candidate definition')
  config=json.loads(json.dumps(choice['selected']));config['id']=f"r2_source_fold{fold['outer_fold']:02}";wanted.append(config)
 check(p['candidates']==wanted==module.source_candidates(),'Five full configs from source choices only')
 expected_models=set();seen=[]
 for old,new in zip(oldfolds,p['folds'],strict=True):
  check(new=={'fold':old['fold'],'train_ids':old['train_sample_ids'],'validation_ids':old['test_sample_ids']},'Old fivefold identity/order')
  tr,va=new['train_ids'],new['validation_ids'];seen+=va
  check(len(tr)==len(set(tr))==684 and len(va)==len(set(va))==171,'Fold count or duplicate')
  check(not(set(tr)&set(va)) and set(tr)|set(va)==set(ids),'Physical record crossing')
  check(set(y[[lookup[v] for v in tr]])==set(y[[lookup[v] for v in va]])==set(module.CLASSES),'Class coverage')
  for candidate in p['candidates']:
   for seed in p['seeds']:expected_models.add((candidate['id'],new['fold'],seed))
 check(len(seen)==len(set(seen))==855 and set(seen)==set(ids),'Development validation denominator')
 check(len(expected_models)==75,'Full source model cohort')
 for rid,row in zip(ids,x,strict=True):
  check(prep['historical_spectrum_hashes'][str(rid)]==hashlib.sha256(np.asarray(row,dtype='<f8').tobytes()).hexdigest(),'Original spectrum hash')
 # Real teacher file is read without its label mapping. This also independently
 # verifies one actual2578->612 crop and pairing with the immutable database.
 intake=io.load_prediction_input(ROOT/'data/processed/zaoren_20261031/样本2_编号版.xlsx',axis)
 check(intake['X_alpha'].shape==(5,612),'Actual workbook shape')
 check(intake['qc']['axis_handling']=='explicit_612_point_slice_from_2578','Explicit full-axis crop')
 numeric=intake['X_alpha'].size+len(axis)
 for rid,row in zip(intake['record_ids'],intake['X_alpha'],strict=True):
  check(np.array_equal(row,x[lookup[rid]]),'Actual workbook array values')
 check(np.array_equal(intake['axis'],axis),'Actual workbook axis')
 # Reference matching is query-wise against the frozen source, not other query
 # rows; it is re-export detection and does not infer biological independence.
 a,b,c=module.historical_matches(np.stack([x[0],x[1]+1e-14,x[2]+.01]),['renamed0','renamed1','new2'],prep['historical_spectrum_hashes'],x)
 check(a==[False,False,False] and b==[True,False,False] and c==[True,True,False],'Renamed/re-exported source query guard')
 test_start=now();cmd=[sys.executable,'-m','unittest','discover','-s','tests','-p','test_deployment*.py','-v']
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 tested=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True)
 (OUT/'boundary_tests.log').write_text(tested.stdout+tested.stderr)
 events={'start_utc':test_start,'end_utc':now(),'argv':cmd,'returncode':tested.returncode,'unit_test_fixture_estimator_fits_only':True,'medicinal_research_estimator_fits':0,'test_source_sha256':test_sha}
 (OUT/'boundary_test_events.json').write_text(json.dumps(events,indent=2)+'\n')
 check(tested.returncode==0 and 'Ran 22 tests' in tested.stdout+tested.stderr,'Boundary unit tests')
 check(code==module.code_versions(),'Learning code changed during source review')
 check(test_sha=={str(t.relative_to(ROOT)):sha(t) for t in test_paths},'Tests changed during source review')
 check(not(run/'source_release.json').exists() and not(run/'selection_lock.json').exists(),'Review must precede research source fitting')
 result={'status':'TECHNICAL_DEPLOYMENT_PREFIT_SOURCE_REVIEW_PASS_EXPLORATORY','started_at_utc':start,'completed_at_utc':now(),'checks':checks,'real_workbook_numeric_cells_checked':numeric,'expected_source_models':75,'expected_full_models':3,'current_research_model_fits':0,'unit_tests_passed':22,'source_sha256':code,'test_source_sha256':test_sha,'candidate_manifest_sha256':sha(run/'candidate_manifest.json'),'protocol_sha256':sha(run/'protocol.json'),'preparation_sha256':sha(run/'preparation.json'),'boundary_test_log_sha256':sha(OUT/'boundary_tests.log'),'review_script_sha256':sha(Path(__file__)),'original_frozen_input_sha256':original,'all855_source_records_are_historically_exposed_development':True,'external_accuracy':None,'OS_truth_sandbox':False,'absolute_zero_leakage_proven':False,'code_review_scope':['Fresh scaler/MC selector/SVC within each current684 source; all final855 only after distinct root gate','Every MC PLS traced through existing instrumentation; no historical frequency-mask union','Native SVC.predict and non-probability decision scores','Strict label-free input schema, fixed612 axis or explicit2578 crop; no implicit resampling','Frozen-model/metadata/reference SHA, history IDs/exact values/re-export comparison','Training diagnostic explicitly in-sample, future prediction unscored; no query fit'],'uncovered':['Unknown biological ancestors and true batches','Unprovided measurement-unit/instrument compatibility','Untrusted caller can lie about physical provenance; file/API scope is not OS enforcement','Future unseen-group detection is unvalidated']}
 (OUT/'source_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':result['status'],'checks':checks,'sha256':sha(OUT/'source_review.json')}))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--run',required=True);main(a.parse_args().run)
