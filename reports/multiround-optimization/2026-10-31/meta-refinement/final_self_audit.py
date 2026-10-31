"""Run-author final arithmetic and lineage self-check; no model fitting."""
import csv,gzip,json,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/workspace/research4');BASE=ROOT/'outputs/multiround-20261031/meta-refinement'
checks=0;largest=0.;models=0;tasks=0;rowcount=0;casecount=0;wrongrandom=0

def check(value,message):
 global checks
 checks+=1
 if not value:raise AssertionError(message)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def compare(actual,expected,label):
 global largest
 if isinstance(expected,(int,float)):
  difference=abs(actual-expected);largest=max(largest,difference);check(difference<1e-12,label)
 else:check(actual==expected,label)

def metrics(y,p,k=7):
 cm=np.bincount(y*k+p,minlength=k*k).reshape(k,k);d=cm.diagonal();s=cm.sum(1);t=cm.sum(0)
 return {'accuracy':float(d.sum()/s.sum()),'balanced_accuracy':float(np.mean(d/s)),
         'macro_f1':float(np.mean(np.divide(2*d,s+t,out=np.zeros(k,dtype=float),where=(s+t)>0))),
         'confusion_matrix':cm.tolist(),'correct':int(d.sum()),'total':int(s.sum()),'class_counts':s.tolist(),
         'class_recall':(d/s).tolist()}

manifest=json.loads((BASE/'run_manifest.json').read_text())
for path,value in manifest['source_sha256'].items():check(sha(ROOT/path)==value,path)
check(sha(BASE/'selection_lock.json')==json.loads((BASE/'selection_lock_sha256.json').read_text())['sha256'],'lock unchanged')
check(sha(BASE/'global_development_gate.json')==sha(BASE.parent/'all_development_locks.json'),'global gate preserved')
with np.load(ROOT/'outputs/approved-zaoren-20261031/data-preparation/dataset.npz',allow_pickle=False) as ds:
 truthmap=dict(zip(ds['record_ids'].astype(str),ds['y'].astype(str)))
allrows=[];perseed=[]
for directory in sorted((BASE/'final').glob('seed_*')):
 record=json.loads((directory/'evaluation.json').read_text());ids=record['query_ids']
 check(len(ids)==171 and len(set(ids))==171,'fixed171')
 check(not(set(record['source_ids'])&set(record['pool_ids'])),'source/pool')
 check(not(set(record['source_ids'])&set(ids)),'source/query')
 check(not(set(record['pool_ids'])&set(ids)),'pool/query')
 check(sha(directory/'predictions.csv.gz')==record['predictions_sha256'],'pred hash')
 draws=record['draws'];check(len(draws)==100,'100draws')
 for draw in draws:
  check(len(set(draw['support_ids']))==7,'7 unique supports')
  check(set(draw['support_ids'])<=set(record['pool_ids']),'supportsubset')
  check(len({truthmap[i] for i in draw['support_ids']})==7,'7 classes')
 groups={};classes=sorted({truthmap[i] for i in ids});vocab={v:i for i,v in enumerate(classes)}
 with gzip.open(directory/'predictions.csv.gz','rt',encoding='utf-8',newline='') as f:
  for r in csv.DictReader(f):
   key=(int(r['repeat']),r['method']);groups.setdefault(key,[]).append(r)
   check(r['y_true']==truthmap[r['record_id']],'true label')
   check(int(r['is_correct'])==int(r['y_true']==r['y_predicted']),'correctflag');rowcount+=1
 for row in record['rows']:
  key=(row['repeat'],row['method']);data=groups[key]
  check([r['record_id'] for r in data]==ids,'predictionorder')
  y=np.asarray([vocab[r['y_true']] for r in data]);p=np.asarray([vocab[r['y_predicted']] for r in data])
  name=row['method'].removesuffix('_source_zero').removesuffix('_matched_zero')
  source=np.asarray([vocab[r['y_predicted']] for r in groups[(row['repeat'],name+'_source_zero')]])
  matched=np.asarray([vocab[r['y_predicted']] for r in groups[(row['repeat'],name+'_matched_zero')]])
  for prefix,values in [('',p),('source_',source),('matched_',matched)]:
   for k,v in metrics(y,values).items():compare(row[prefix+k],v,key)
  compare(row['corrected'],int(np.sum((source!=y)&(p==y))),key)
  compare(row['new_errors'],int(np.sum((source==y)&(p!=y))),key)
  allrows.append(row);casecount+=1
 for method in ('selected','diagnostic','supervised_diagnostic'):
  selected=[r for r in record['rows'] if r['method']==method]
  perseed.append({'seed':selected[0]['seed'],'method':method,'accuracy':float(np.mean([r['accuracy'] for r in selected])),
   'source_accuracy':float(np.mean([r['source_accuracy'] for r in selected])),
   'matched_accuracy':float(np.mean([r['matched_accuracy'] for r in selected]))})
 for meta in sorted(p for p in directory.glob('m*') if p.is_dir()):
  metadata=json.loads((meta/'training.json').read_text());sourceids=set(metadata['expected']['source_ids'])
  check(metadata['training']['seed']==metadata['expected']['seed']+1600,'finalmeta seed')
  for task in metadata['training']['source_tasks']:
   check(not(set(task['support_ids'])&set(task['query_ids'])),'meta task disjoint')
   check(set(task['support_ids']+task['query_ids'])<=sourceids,'meta sourceonly');tasks+=1
  for strategy in ('fomaml','supervised'):
   p=meta/(strategy+'.pt');check(sha(p)==metadata['artifacts'][strategy]['sha256'],'statehash')
   state=torch.load(p,map_location='cpu',weights_only=True);digest=hashlib.sha256()
   for name,value in sorted(state.items()):
    tensor=value.detach().cpu().contiguous();digest.update(name.encode());digest.update(str(tensor.dtype).encode());digest.update(np.asarray(tensor.shape,dtype='<i8').tobytes());digest.update(tensor.numpy().tobytes())
   check(digest.hexdigest()==metadata['artifacts'][strategy]['fingerprint'],'statefingerprint');models+=1
check(allrows==json.loads((BASE/'final_metrics.json').read_text()),'all_final_metrics')
for summary in json.loads((BASE/'summary.json').read_text()):
 cases=[r for r in allrows if r['method']==summary['method']]
 compare(summary['evaluations'],len(cases),'evaluations')
 for metric in ('accuracy','balanced_accuracy','macro_f1'):
  values=np.asarray([r[metric] for r in cases]);source=np.asarray([r['source_'+metric] for r in cases]);matched=np.asarray([r['matched_'+metric] for r in cases])
  for suffix,val in [('_mean',values.mean()),('_sd',values.std(ddof=1)),('_source_gain',(values-source).mean()),('_matched_gain',(values-matched).mean())]:
   compare(summary[metric+suffix],float(val),'summary'+metric+suffix)
 delta=np.asarray([r['accuracy']-r['source_accuracy'] for r in cases])
 for key,value in [('positive_draws',np.sum(delta>1e-12)),('unchanged_draws',np.sum(np.abs(delta)<=1e-12)),('negative_draws',np.sum(delta < -1e-12))]:compare(summary[key],int(value),'drawcounts')
 for key,field in [('mean_corrected','corrected'),('mean_new_errors','new_errors')]:compare(summary[key],float(np.mean([r[field] for r in cases])),'paired')
with (BASE/'random_predictions.csv').open(encoding='utf-8-sig',newline='') as f:
 examples=list(csv.DictReader(f))
 for r in examples:
  check(r['y_true']==truthmap[r['record_id']],'randomtruth')
  check(int(r['is_correct'])==int(r['y_true']==r['y_predicted']),'randomcorrect')
 wrongrandom=sum(r['is_correct']=='0' for r in examples)
report={'status':'pass','scope':'run-author final self-audit; independent audit is separate','checks':checks,
 'prediction_rows':rowcount,'final_cases':casecount,'final_meta_source_models':models,'source_task_lists':tasks,
 'maximum_numeric_difference':largest,'random_examples':len(examples),'random_incorrect_examples_preserved':wrongrandom,
 'new_fits':0,'per_seed_accuracy':perseed}
(BASE/'final_self_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
