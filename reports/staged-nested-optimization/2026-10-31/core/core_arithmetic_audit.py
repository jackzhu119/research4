"""Read-only arithmetic/source-statistics audit, no learning or optimization."""
from collections import defaultdict
from pathlib import Path
import csv,gzip,hashlib,json
import numpy as np
from scipy.signal import detrend,savgol_filter

ROOT=Path.cwd();RUN=ROOT/'outputs/staged-nested-20261031/core'
with np.load(ROOT/'outputs/approved-zaoren-20261031/datasets/candidate_0p1_1p5.npz',allow_pickle=False) as z:
 ids=z['groups'];X=z['X_alpha'];y=z['y'];axis=z['axis']
lookup={str(i):n for n,i in enumerate(ids)};classes=sorted(set(y.tolist()));cl={c:i for i,c in enumerate(classes)}
checks=0;maximum=0.;raw_count=0;stat_count=0;seen_stats=set()
def ck(cond,msg):
 global checks
 checks+=1
 if not cond:raise AssertionError(msg)
def hashfile(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics(cm):
 tp=np.diag(cm).astype(float);counts=cm.sum(1);predcounts=cm.sum(0)
 recall=np.divide(tp,counts,out=np.zeros(7),where=counts!=0)
 f1=np.divide(2*tp,counts+predcounts,out=np.zeros(7),where=counts+predcounts!=0)
 return {'accuracy':tp.sum()/cm.sum(),'balanced_accuracy':recall.mean(),'macro_f1':f1.mean()}
allrows=json.loads((RUN/'all_inner_metrics.json').read_text())
for path in sorted((RUN/'selection').glob('outer_*/R*/predictions.csv.gz')):
 rows=json.loads((path.parent/'metrics.json').read_text());cms=defaultdict(lambda:np.zeros((7,7),int));disagree=defaultdict(int)
 outer=int(path.parent.parent.name.split('_')[-1]);stage=path.parent.name
 specs=json.loads((RUN/'splits.json').read_text())[outer]
 with gzip.open(path,'rt',encoding='utf-8-sig',newline='') as f:
  for row in csv.DictReader(f):
   raw_count+=1;key=(int(row['inner_fold']),row['candidate_id']);i=lookup[row['record_id']]
   ck(row['y_true']==y[i],('label',path,row['record_id']))
   ck(row['record_id'] in specs['inner_folds'][key[0]]['validation_sample_ids'],('validation',path))
   ck(row['record_id'] not in specs['test_sample_ids'],('outer leaked',path))
   t,p=cl[row['y_true']],cl[row['y_predicted']];cms[key][t,p]+=1
   ck(int(row['is_correct'])==int(t==p),('flag',path))
   score=np.array([float(row['score_'+c]) for c in classes]);disagree[key]+=int(p!=score.argmax())
 for row in rows:
  key=(row['inner_fold'],row['candidate_id']);cm=cms[key];expected=row['metrics'];m=metrics(cm)
  ck(cm.tolist()==expected['confusion_matrix'],('CM',path,key))
  for name,value in m.items():
   diff=abs(value-expected[name]);maximum=max(maximum,diff);ck(diff<1e-12,('metric',path,key,name,diff))
  ck(disagree[key]==row['vote_argmax_disagreements'],('argmaxvote',path,key))
  info=row['transform_diagnostic'];stat_path=info['statistics']
  for field in ('metadata','statistics','coefficients'):
   ck(hashfile(info[field])==info[field+'_sha256'],('hash',info[field]))
  if stat_path in seen_stats:continue
  seen_stats.add(stat_path);stat_count+=1
  meta=json.loads(Path(info['metadata']).read_text());ii=np.array([lookup[i] for i in meta['source_ids']]);x=X[ii];labels=y[ii]
  ck(meta['source_ids']==specs['inner_folds'][row['inner_fold']]['train_sample_ids'],('sourceIDs',path,key))
  mode=meta['preprocessing']
  if mode=='msc':
   ref=x.mean(0);center=ref-ref.mean();slope=(x-x.mean(1,keepdims=True))@center/np.dot(center,center)
   intercept=x.mean(1)-slope*ref.mean();prepared=(x-intercept[:,None])/slope[:,None]
   ck(np.allclose(meta['msc_source_reference'],ref,atol=0,rtol=0),('MSCref',path,key))
  else:
   prepared=x.copy()
   if mode.startswith('sg'):
    prepared=savgol_filter(prepared,15,3,deriv=int(mode[2]),delta=float(np.mean(np.diff(axis))),axis=1,mode='interp')
   elif mode=='linear_detrend_snv':prepared=detrend(prepared,axis=1,type='linear')
   if mode!='raw':prepared=(prepared-prepared.mean(1,keepdims=True))/prepared.std(1,ddof=1,keepdims=True)
  with np.load(info['statistics'],allow_pickle=False) as s, np.load(info['coefficients'],allow_pickle=False) as co:
   mean=prepared.mean(0);var=prepared.var(0)
   ck(np.allclose(s['scaler_mean'],mean,rtol=1e-10,atol=1e-10),('mean',path,key))
   ck(np.allclose(s['scaler_var'],var,rtol=1e-10,atol=1e-10),('variance',path,key))
   coef=co['coefficients'];ck(len(coef)==meta['mc']['runs'],('runs',path,key))
   sd=coef.std(0,ddof=1);st=coef.mean(0)/np.maximum(sd,max(1.,float(sd.max()))*1e-12)
   imp=np.linalg.norm(np.abs(st),axis=0);rank=np.argsort(-imp,kind='stable')
   ck(np.array_equal(s['signed_stability'],st),('stability',path,key))
   ck(np.array_equal(s['importance'],imp),('importance',path,key))
   ck(np.array_equal(s['ranking'],rank),('rank',path,key))
   count=int(np.ceil(meta['mc']['ratio']*X.shape[1]));sel=np.sort(rank[:count])
   ck(np.array_equal(s['selected_indices'],sel),('selection',path,key))
   ck(np.array_equal(meta['selected_frequencies'],axis[sel]),('frequency',path,key))
   scale=np.sqrt(s['scaler_var']);scale[scale==0]=1
   z=(prepared-s['scaler_mean'])/scale;selected=z[:,sel]
  for c in classes:
   n=np.sum(labels==c);modew=row['candidate']['svm']['class_weight'];base=len(labels)/(7*n)
   if modew=='none':want=1.
   elif modew=='balanced':want=base
   else:
    power=.5 if modew=='power_0.5' else 1.5
    raw=np.array([(len(labels)/(7*np.sum(labels==k)))**power for k in classes])
    denom=sum(w*np.sum(labels==k) for k,w in zip(classes,raw))/len(labels);want=base**power/denom
   ck(abs(want-row['training_class_weights'][c])<1e-12,('classweight',path,key,c))
lock=json.loads((RUN/'selection_lock.json').read_text())
for fold in lock['outer_folds']:
 for stage,choice in fold['choices'].items():
  rows=[r for r in allrows if r['outer_fold']==fold['outer_fold'] and r['stage']==stage];groups=defaultdict(list)
  for row in rows:groups[row['candidate_id']].append(row)
  means={c:{m:float(np.mean([r['metrics'][m] for r in rs])) for m in ('macro_f1','balanced_accuracy','accuracy')} for c,rs in groups.items()}
  ck(means==choice['all_candidate_means'],('allmeans',fold['outer_fold'],stage))
  candidates=[rs[0]['candidate'] for rs in groups.values()]
  if stage=='R0':chosen=max(candidates,key=lambda c:means[c['id']]['macro_f1'])
  else:
   top=max(m['macro_f1'] for m in means.values());near=[c for c in candidates if means[c['id']]['macro_f1']>=top-1e-12]
   prep=('raw','snv','sg0_snv','sg1_snv','sg2_snv','msc','linear_detrend_snv')
   chosen=min(near,key=lambda c:(-means[c['id']]['balanced_accuracy'],-means[c['id']]['accuracy'],c['mc']['ratio'],c['mc']['components'],c['mc']['runs'],c['svm']['C'],prep.index(c['preprocessing']),c['id']))
  ck(chosen==choice['selected'],('selection',fold['outer_fold'],stage))
ck(not (RUN/'outer_predictions.csv.gz').exists(),'outer scoring happened')
result={'status':'pass','scope':'source-only selection arithmetic/source-statistics audit; no new fit','checks':checks,'raw_prediction_rows':raw_count,'inner_cases':len(allrows),'unique_saved_transforms_checked':stat_count,'maximum_metric_difference':maximum,'new_fits':0,'outer_query_scored':False,'selection_lock_sha256':hashfile(RUN/'selection_lock.json')}
(RUN/'selection_arithmetic_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
