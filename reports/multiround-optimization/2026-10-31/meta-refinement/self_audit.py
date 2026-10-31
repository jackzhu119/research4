"""Arithmetic/lineage self-check; no fitting or project metric helpers used."""
import csv,gzip,json,hashlib
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/workspace/research4')
BASE=ROOT/'outputs/multiround-20261031/meta-refinement'
checks=0; largest=0.; rows_count=0; cases=0; models=0; meta_tasks=0

def check(value,message):
 global checks
 checks+=1
 if not value:raise AssertionError(message)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def metrics(y,p,k=7):
 cm=np.bincount(y*k+p,minlength=k*k).reshape(k,k);d=cm.diagonal();s=cm.sum(1);t=cm.sum(0)
 return {'accuracy':float(d.sum()/s.sum()),'balanced_accuracy':float(np.mean(d/s)),
         'macro_f1':float(np.mean(np.divide(2*d,s+t,out=np.zeros(k,dtype=float),where=(s+t)>0))),
         'confusion_matrix':cm.tolist(),'correct':int(d.sum()),'total':int(s.sum()),'class_counts':s.tolist(),
         'class_recall':(d/s).tolist()}

def compare(actual,expected,label):
 global largest
 if isinstance(expected,(int,float)):
  difference=abs(actual-expected);largest=max(largest,difference);check(difference<1e-12,label)
 else:check(actual==expected,label)

manifest=json.loads((BASE/'run_manifest.json').read_text())
for path,value in manifest['source_sha256'].items():check(sha(ROOT/path)==value,path)
check(sha(ROOT/'data/research/meta_refinement_search_protocol_20261031.json')==manifest['protocol_sha256'],'protocol')
check(sha(BASE/'protocol_snapshot.json')==manifest['protocol_sha256'],'snapshot')
for directory in sorted((BASE/'development').glob('episode_*/seed_*')):
 record=json.loads((directory/'metrics.json').read_text());classes=record['classes'];ids=record['query_ids'];vocab={v:i for i,v in enumerate(classes)}
 check(not(set(record['source_ids']) & set(record['pool_ids'])),'source/pool')
 check(not(set(record['source_ids']) & set(ids)),'source/query')
 check(not(set(record['pool_ids']) & set(ids)),'pool/query')
 check(sha(directory/'predictions.csv.gz')==record['predictions_sha256'],'prediction hash')
 for draw in record['draws']:
  check(len(set(draw['support_ids']))==7,'support7')
  check(set(draw['support_ids'])<=set(record['pool_ids']),'support in pool')
 groups={}
 with gzip.open(directory/'predictions.csv.gz','rt',encoding='utf-8',newline='') as f:
  for r in csv.DictReader(f):
   key=(r['strategy'],r['candidate_id'],int(r['repeat']));groups.setdefault(key,[]).append(r)
   check(int(r['is_correct'])==int(r['y_true']==r['y_predicted']),'correctness flag');rows_count+=1
 for r in record['rows']:
  key=(r['strategy'],r['candidate_id'],r['repeat']);data=groups.pop(key)
  check([x['record_id'] for x in data]==ids,'queryorder')
  y=np.asarray([vocab[x['y_true']] for x in data]);p=np.asarray([vocab[x['y_predicted']] for x in data])
  control=record['controls'][r['strategy']+':'+r['candidate_id']]
  source=np.asarray(control['source_prediction']);matched=np.asarray(control['matched_prediction'])
  for prefix,values in [('',p),('source_',source),('matched_',matched)]:
   for name,value in metrics(y,values).items():compare(r[prefix+name],value,key)
  compare(r['corrected'],int(np.sum((source!=y)&(p==y))),key)
  compare(r['new_errors'],int(np.sum((source==y)&(p!=y))),key)
  cases+=1
 check(not groups,'unmatched predictioncases')
 for meta in sorted(p for p in directory.glob('m*') if p.is_dir()):
  m=json.loads((meta/'training.json').read_text());training=m['training'];sourceids=set(m['expected']['source_ids'])
  check(training['initializer_fingerprint']==m['expected']['initial_fingerprint'],'same initializer')
  check(training['source_only_training'] and training['first_order'] and not training['equal_FLOPs'],'scope')
  for task in training['source_tasks']:
   check(len(task['support_ids'])==len(task['query_ids'])==3,'meta3way')
   check(not(set(task['support_ids'])&set(task['query_ids'])),'source task disjoint')
   check(set(task['support_ids']+task['query_ids'])<=sourceids,'source tasks sourceonly');meta_tasks+=1
  for strategy in ('fomaml','supervised'):
   p=meta/(strategy+'.pt');check(sha(p)==m['artifacts'][strategy]['sha256'],'statehash')
   state=torch.load(p,map_location='cpu',weights_only=True);digest=hashlib.sha256()
   for name,value in sorted(state.items()):
    tensor=value.detach().cpu().contiguous();digest.update(name.encode());digest.update(str(tensor.dtype).encode());digest.update(np.asarray(tensor.shape,dtype='<i8').tobytes());digest.update(tensor.numpy().tobytes())
   check(digest.hexdigest()==m['artifacts'][strategy]['fingerprint'],'statefingerprint');models+=1
report={'status':'pass','scope':'run-author self-audit; independent audit is separate','checks':checks,
        'prediction_rows':rows_count,'development_cases':cases,'source_models':models,'source_task_lists':meta_tasks,
        'maximum_numeric_difference':largest,'new_fits':0,'all_development_rows_recomputed':True}
(BASE/'development_self_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
