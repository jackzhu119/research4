"""Read-only released outer predictions/model replay and historical reproduction."""
from collections import defaultdict
from pathlib import Path
import csv,gzip,hashlib,json
import joblib,numpy as np

ROOT=Path.cwd();RUN=ROOT/'outputs/staged-nested-20261031/core'
with np.load(ROOT/'outputs/approved-zaoren-20261031/datasets/candidate_0p1_1p5.npz',allow_pickle=False) as z:
 X=z['X_alpha'];y=z['y'];ids=z['groups'];axis=z['axis']
lookup={str(i):n for n,i in enumerate(ids)};classes=sorted(set(y.tolist()));cl={c:i for i,c in enumerate(classes)}
splits=json.loads((RUN/'splits.json').read_text());checks=0;maximum=0.;model_count=0;cache={}
def ck(x,msg):
 global checks
 checks+=1
 if not x:raise AssertionError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics(cm):
 tp=np.diag(cm).astype(float);total=cm.sum(1);pred=cm.sum(0)
 recall=np.divide(tp,total,out=np.zeros(7),where=total!=0);f1=np.divide(2*tp,total+pred,out=np.zeros(7),where=total+pred!=0)
 return {'accuracy':tp.sum()/cm.sum(),'balanced_accuracy':recall.mean(),'macro_f1':f1.mean()}
with gzip.open(RUN/'outer_predictions.csv.gz','rt',encoding='utf-8',newline='') as f:predictions=list(csv.DictReader(f))
groups=defaultdict(list)
for r in predictions:
 i=lookup[r['record_id']];ck(r['y_true']==y[i],('truth',r['record_id']));ck(int(r['is_correct'])==int(r['y_true']==r['y_predicted']),('flag',r['record_id']))
 ck(r['record_id'] in splits[int(r['outer_fold'])]['test_sample_ids'],('testrole',r['record_id']))
 groups[(r['method'],int(r['outer_fold']))].append(r)
rows=json.loads((RUN/'outer_metrics.json').read_text())
for row in rows:
 rs=groups[(row['method'],row['outer_fold'])];cm=np.zeros((7,7),int)
 for r in rs:cm[cl[r['y_true']],cl[r['y_predicted']]]+=1
 ck(cm.tolist()==row['confusion_matrix'],('CM',row['method'],row['outer_fold']))
 for name,value in metrics(cm).items():
  d=abs(value-row[name]);maximum=max(maximum,d);ck(d<1e-12,('metric',name))
 path=Path(row['model_path']);ck(sha(path)==row['model_sha256'],('weightSHA',path))
 if str(path) not in cache:
  model_count+=1;model=joblib.load(path);train=np.array([lookup[i] for i in splits[row['outer_fold']]['train_sample_ids']]);test=np.array([lookup[i] for i in splits[row['outer_fold']]['test_sample_ids']])
  ck(not set(train)&set(test),('overlap',path))
  ck(joblib.hash(X[train])==model.metadata['source_value_fingerprint'],('sourcevalues',path))
  ck(joblib.hash(y[train])==model.metadata['source_label_fingerprint'],('sourcelabels',path))
  before=joblib.hash(model);prediction=model.predict(X[test]);score=model.predict_scores(X[test]);ck(joblib.hash(model)==before,('mutated',path))
  prepared=model.bundle.preprocessor.transform(X[train]);scaler=model.bundle.scaler
  ck(np.allclose(scaler.mean_,prepared.mean(0),atol=1e-10,rtol=1e-10),('mean',path))
  ck(np.allclose(scaler.var_,prepared.var(0),atol=1e-10,rtol=1e-10),('var',path))
  selector=model.bundle.selector;coef=selector.coefficients_;sd=coef.std(0,ddof=1);st=coef.mean(0)/np.maximum(sd,max(1.,float(sd.max()))*1e-12)
  importance=np.linalg.norm(abs(st),axis=0);ranking=np.argsort(-importance,kind='stable');selected=np.sort(ranking[:len(selector.selected_indices_)])
  ck(np.array_equal(selector.importances_,importance),('importance',path));ck(np.array_equal(selector.selected_indices_,selected),('selection',path))
  ck(np.array_equal(axis[selected],model.metadata['selected_frequencies']),('frequency',path))
  cache[str(path)]=(prediction,score,test)
 pred,score,test=cache[str(path)]
 expected_pred=[r['y_predicted'] for r in rs];ck(np.array_equal(pred,expected_pred),('modelreplay',row['method'],row['outer_fold']))
 saved=np.array([[float(r['score_'+c]) for c in classes] for r in rs]);ck(np.allclose(score,saved,rtol=1e-13,atol=1e-13),('scorereplay',row['method']))
 ck(int(np.sum(pred!=np.asarray(classes)[score.argmax(1)]))==row['vote_argmax_disagreements'],('vote',row['method']))
for row in json.loads((RUN/'summary.json').read_text()):
 rs=[r for r in predictions if r['method']==row['method']];ck(len(rs)==855 and len(set(r['record_id'] for r in rs))==855,('coverage',row['method']))
 cm=np.zeros((7,7),int)
 for r in rs:cm[cl[r['y_true']],cl[r['y_predicted']]]+=1
 ck(cm.tolist()==row['confusion_matrix'],('aggregateCM',row['method']))
 for name,value in metrics(cm).items():
  d=abs(value-row[name]);maximum=max(maximum,d);ck(d<1e-12,('aggregate',row['method'],name))
legacy_path=ROOT/'outputs/approved-zaoren-20261031/experiments/alpha_primary/alpha_mcuve20_svc/predictions.csv'
with legacy_path.open(encoding='utf-8-sig',newline='') as f:legacy={r['sample_id']:r for r in csv.DictReader(f)}
current={r['record_id']:r for r in predictions if r['method']=='R0'}
for i,r in current.items():ck(r['y_predicted']==legacy[i]['y_predicted'] and r['y_true']==legacy[i]['y_true'],('legacyrepro',i))
result={'status':'pass','scope':'Released outer evaluation read-only replay; no fitting','checks':checks,'outer_rows':len(predictions),'method_fold_roles':len(rows),'unique_source_models_replayed':model_count,'maximum_metric_difference':maximum,'historical_R0_predictions_exactly_reproduced':True,'historical_prediction_sha256':sha(legacy_path),'new_fits':0,'selection_lock_sha256':sha(RUN/'selection_lock.json')}
(RUN/'outer_arithmetic_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
