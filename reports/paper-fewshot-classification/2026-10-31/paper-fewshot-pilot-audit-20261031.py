"""Independent arithmetic/lineage replay of the completed neural pilot; no fits."""
import csv,gzip,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

repo=Path.cwd();out=repo/'outputs/paper-fewshot-pilot-20261031'
def readj(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=0;maxdiff=0.
def ck(value,label):
 global checks
 checks+=1
 if not value:raise AssertionError(label)
def near(a,b,label):
 global maxdiff
 delta=abs(float(a)-float(b));maxdiff=max(maxdiff,delta)
 ck(delta<1e-12,label)
status=readj(out/'status.json');ck(status['status']=='completed','completed')
lock=readj(out/'selection_lock.json');ck(sha(out/'selection_lock.json')==status['selection_lock_sha256'],'lock_sha')
manifest=readj(out/'run_manifest.json')
for name,value in manifest['source_sha256'].items():ck(sha(repo/name)==value,'source_sha:'+name)
ck(sha(repo/'data/research/paper_neural_classification_protocol_20261031.json')==manifest['protocol_sha256'],'protocol_sha')
with np.load(repo/'outputs/approved-zaoren-20261031/data-preparation/dataset.npz',allow_pickle=False) as data:
 truth=dict(zip(data['record_ids'].tolist(),data['y'].tolist()))
plan=readj(repo/'data/processed/zaoren_threeway_20261031_v1/split_plan.json')
roles={name:{r['record_id'] for r in plan['records'] if r['role']==name} for name in ['training','calibration','prediction']}
classes=sorted(set(truth.values()));lookup={v:i for i,v in enumerate(classes)}
results=readj(out/'final_metrics.json');expected={(r['seed'],r['repeat'],r['strategy'],r['method']):r for r in results}
case_predictions={};raw_rows=0
for path in sorted((out/'final').glob('seed_*/predictions.csv.gz')):
 with gzip.open(path,'rt',encoding='utf-8',newline='') as f:
  for row in csv.DictReader(f):
   key=(int(row['seed']),int(row['repeat']),row['strategy'],row['method'])
   group=case_predictions.setdefault(key,{})
   ck(row['record_id'] in roles['prediction'],'query_membership')
   ck(row['y_true']==truth[row['record_id']],'canonical_truth')
   ck(int(row['is_correct'])==int(row['y_true']==row['y_predicted']),'is_correct')
   ck(row['record_id'] not in group,'duplicate_prediction_id')
   group[row['record_id']]=(row['y_true'],row['y_predicted']);raw_rows+=1
ck(set(case_predictions)==set(expected),'case_keys')
for key,group in case_predictions.items():
 ck(set(group)==roles['prediction'],'query_exact')
 matrix=[[0]*7 for _ in range(7)]
 for actual,predicted in group.values():matrix[lookup[actual]][lookup[predicted]]+=1
 target=expected[key];ck(matrix==target['confusion_matrix'],'cm')
 count=[sum(r) for r in matrix];predcounts=[sum(r[c] for r in matrix) for c in range(7)]
 diag=[matrix[c][c] for c in range(7)]
 values={'accuracy':sum(diag)/171,'balanced_accuracy':sum(diag[c]/count[c] for c in range(7))/7,
  'macro_f1':sum(2*diag[c]/(count[c]+predcounts[c]) if count[c]+predcounts[c] else 0 for c in range(7))/7}
 for metric,value in values.items():near(value,target[metric],metric)
 base=case_predictions[(key[0],key[1],key[2],'zero')]
 corrected=sum(base[i][0]!=base[i][1] and group[i][0]==group[i][1] for i in group)
 harmed=sum(base[i][0]==base[i][1] and group[i][0]!=group[i][1] for i in group)
 ck(corrected==target['corrected_errors'] and harmed==target['new_errors'],'paired_changes')
for path in sorted((out/'development').glob('episode_*/seed_*/training.json'))+sorted((out/'final').glob('seed_*/training.json')):
 meta=readj(path);ids=set(meta['source_ids']);ck(ids<=roles['training'],'source_only')
 records=meta['meta_training'];ref=records['supervised']['episodes']
 for name in ['prototype','fomaml']:
  ck(records[name]['episodes']==ref,'same_meta_tasks')
 for row in ref:
  support,query=set(row['support_ids']),set(row['query_ids'])
  ck(not support&query and support|query<=ids,'meta_disjoint_source')
  ck(len(support)==3 and len(query)==3,'meta_1_support_1_query_3way')
 if path.parent.parent.name=='final':
  ck(records['prototype']['policy']['eligible_classes']==list(range(7)),'final_all7_meta_eligible')
 else:ck(records['prototype']['policy']['excluded_insufficient_classes']==[1],'dev_rare_meta_excluded')
 for name,info in meta['artifacts'].items():ck(sha(path.parent/(name+'.pt'))==info['sha256'],'weight_sha')
for meta in readj(out/'final_metadata.json'):
 ck(set(meta['source_ids'])==roles['training'],'final_source_cohort')
 ck(set(meta['query_ids'])==roles['prediction'],'final_query_cohort')
 ck(meta['fingerprints_before']==meta['fingerprints_after'],'frozen_source')
 ck(len(meta['support_draws'])==100,'support100')
 for draw in meta['support_draws']:
  ids=draw['ids'];ck(len(ids)==len(set(ids))==7 and set(ids)<=roles['calibration'],'support_only')
  ck(sorted(truth[i] for i in ids)==classes,'one_each_class')
random_ids=set(readj(out/'random_query_ids.json'));ck(len(random_ids)==20,'random20')
with (out/'random_predictions.csv').open(encoding='utf-8-sig',newline='') as f:
 random_rows=list(csv.DictReader(f))
for row in random_rows:
 key=(int(row['seed']),int(row['repeat']),row['strategy'],row['method'])
 ck(row['record_id'] in random_ids and int(row['repeat'])==0,'random_membership')
 ck(case_predictions[key][row['record_id']]==(row['y_true'],row['y_predicted']),'random_replay')
record={'status':'pass','checks_evaluated':checks,'maximum_metric_difference':maxdiff,'raw_prediction_rows':raw_rows,
 'final_cases':len(case_predictions),'random_example_rows':len(random_rows),'source_model_files':len(list(out.rglob('*.pt'))),
 'scope':'Independent arithmetic and ID replay, no model fitting; user-declared anonymous labels only',
 'labels_verified_as_geographic_origins':False,'recorded_source_freeze_checks':True}
write=out/'arithmetic_and_lineage_audit.json';write.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
