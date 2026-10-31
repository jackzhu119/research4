"""Independent arithmetic and serialized-state checks; no estimator fit."""
import csv,hashlib,json
from pathlib import Path
import joblib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
CHECKS=0
MAXIMUM=0.

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def check(value,why):
 global CHECKS
 CHECKS+=1
 if not value:raise AssertionError(why)
def close(a,b,why,atol=1e-12,rtol=1e-12):
 global MAXIMUM
 aa,bb=np.asarray(a),np.asarray(b)
 check(aa.shape==bb.shape,why+' shape')
 delta=float(np.max(np.abs(aa-bb))) if aa.size else 0.
 MAXIMUM=max(MAXIMUM,delta)
 check(np.allclose(aa,bb,atol=atol,rtol=rtol),why)
def serial(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def afp(x):
 x=np.asarray(x);h=hashlib.sha256(serial({'shape':x.shape,'dtype':str(x.dtype)}));h.update(np.ascontiguousarray(x).tobytes());return h.hexdigest()
def fp(x,y):return hashlib.sha256(serial({'X':afp(x),'y':None if y is None else afp(y)})).hexdigest()
def snv(x):return (x-x.mean(1,keepdims=True))/x.std(1,ddof=1,keepdims=True)
def metrics(truth,pred,classes):
 lookup={v:i for i,v in enumerate(classes)};cm=np.zeros((len(classes),len(classes)),int)
 for a,b in zip(truth,pred,strict=True):cm[lookup[a],lookup[b]]+=1
 tp=np.diag(cm).astype(float);n=cm.sum(1);q=cm.sum(0)
 recall=np.divide(tp,n,out=np.zeros(len(classes)),where=n!=0)
 precision=np.divide(tp,q,out=np.zeros(len(classes)),where=q!=0)
 f1=np.divide(2*tp,n+q,out=np.zeros(len(classes)),where=(n+q)!=0)
 return {'accuracy':float(tp.sum()/cm.sum()),'macro_f1':float(f1.mean()),'balanced_accuracy':float(recall.mean()),'confusion_matrix':cm.tolist(),'precision':precision.tolist(),'recall':recall.tolist(),'f1':f1.tolist()}

def model_record(row,config,x,y,ids,axis,seed):
 item=row['model'] if 'model' in row else row
 p=Path(item['path']);p=p if p.is_absolute() else ROOT/p
 check(sha(p)==item['sha256'] and sha(p.with_suffix('.json'))==item['metadata_sha256'],'Model byte checksum')
 model=joblib.load(p);meta=read(p.with_suffix('.json'));check(meta==model.metadata,'Serialized metadata matches JSON')
 check(meta['config']==config and meta['seed']==seed,'Model config/seed')
 check(meta['source_ids']==list(ids),'Actual source identities/order')
 check(meta['source_X_sha256']==hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest(),'Source spectrum bytes')
 check(meta['source_y_sha256']==hashlib.sha256(json.dumps(y.tolist()).encode()).hexdigest(),'Source label bytes')
 check(model.estimator.break_ties is False and model.estimator.decision_function_shape=='ovr','Native SVC vote')
 check(model.estimator.probability is False and model.estimator.kernel=='rbf','No calibrated probability or changed kernel')
 check(model.classes_.tolist()==[f'ZY{i:02}' for i in range(1,8)],'Seven class order')
 check(model.estimator.C==config['svm']['C'] and model.estimator.random_state==seed,'SVC parameters')
 z0=snv(x);scaler=model.bundle.scaler
 close(scaler.mean_,z0.mean(0),'Source-only mean')
 close(scaler.var_,z0.var(0),'Source-only variance')
 close(scaler.scale_,np.sqrt(z0.var(0)),'Source-only scale')
 check(scaler.n_samples_seen_==len(x),'Scaler source denominator')
 close(model.bundle.preprocessor.transform(x),z0,'Per-row SNV',atol=0,rtol=0)
 z=scaler.transform(z0);selector=model.bundle.selector
 check(selector.n_runs==config['mc']['runs'] and selector.n_components==config['mc']['components'] and selector.random_state==seed,'MC configuration')
 coeff=selector.coefficients_;check(coeff.shape==(config['mc']['runs'],7,len(axis)),'Stored coefficient tensor')
 sd=coeff.std(0,ddof=1);stab=coeff.mean(0)/np.maximum(sd,max(1.,float(sd.max()))*1e-12)
 importance=np.linalg.norm(np.abs(stab),axis=0);rank=np.argsort(-importance,kind='stable')
 k=int(np.ceil(len(axis)*config['mc']['ratio']));selected=np.sort(rank[:k])
 close(selector.signed_stability_,stab,'Stored coefficient stability',atol=0,rtol=0)
 close(selector.importances_,importance,'MC norm importance',atol=0,rtol=0)
 check(np.array_equal(selector.ranking_,rank) and np.array_equal(selector.selected_indices_,selected),'Source-only MC ranking and selected features')
 check(meta['selected_indices']==selected.tolist() and meta['selected_frequencies']==axis[selected].tolist(),'Feature metadata')
 selected_source=z[:,selected]
 counts=np.array([(y==c).sum() for c in model.classes_],float);base=len(y)/(7*counts)
 mode=config['svm']['class_weight']
 if mode=='none':weights=np.ones(7)
 elif mode=='balanced':weights=base
 else:
  weights=base**.5;weights/=np.sum(weights*counts)/len(y)
 close(model.estimator.class_weight_,weights,'Source-only class weights')
 rule=config['svm']['gamma'];scale=1/(k*selected_source.var()) if selected_source.var() else 1.
 expected_gamma=scale*(.1 if rule=='scale_x0.1' else 1.)
 close(np.array([model.estimator._gamma]),np.array([expected_gamma]),'Source-only kernel gamma')
 return model,z,selected_source

class ScopeReader:
 def __init__(self,path,module_sha):
  self.stream=Path(path).open();self.module_sha=module_sha;self.rows=0;self.ops=0
 def event(self,event,component,ids,datafp,split):
  line=self.stream.readline();check(bool(line),'Missing fit-scope event')
  actual=json.loads(line);self.rows+=1
  check(actual['event']==event and actual['component']==component,'Actual learned operation order')
  check(actual['fit_sample_ids']==list(ids),'Every operation current-source IDs')
  check(actual['split_id']==split and actual['code_version']==self.module_sha,'Fit split/code binding')
  check(actual['data_fingerprint']==datafp,'Every actual fit X/y fingerprint')
  if event=='fit_completed':self.ops+=1
 def pair(self,component,ids,x,y,split):
  digest=fp(x,y)
  self.event('fit_start',component,ids,digest,split);self.event('fit_completed',component,ids,digest,split)
 def trial(self,model,config,x,y,ids,z,selected,seed,split):
  self.pair('standard_scaler',ids,snv(x),None,split)
  digest=fp(z,y);self.event('fit_start','MC-UVE',ids,digest,split)
  classes,encoded=np.unique(y,return_inverse=True);targets=np.eye(7)[encoded];rng=np.random.default_rng(seed)
  for _ in range(config['mc']['runs']):
   priorities=rng.random(len(y));sub=[]
   for label in range(7):
    pool=np.flatnonzero(encoded==label);size=min(len(pool),max(1,int(np.ceil(len(pool)*.8))))
    sub.extend(pool[np.argsort(priorities[pool],kind='stable')[:size]])
   sub=np.sort(sub);sx=z[sub];sy=targets[sub];sid=np.asarray(ids)[sub].tolist()
   digest_sub=fp(sx,sy);self.event('fit_start','MC-UVE.coefficient_standardization',sid,digest_sub,split)
   scale=sx.std(0);scale[scale<=np.finfo(float).eps]=1;zs=(sx-sx.mean(0))/scale
   xhash=afp(zs)
   for col in sy.T:
    target=(col-col.mean())/col.std()
    datafp=hashlib.sha256(serial({'X':xhash,'y':afp(target)})).hexdigest()
    self.event('fit_start','MC-UVE.PLSRegression.fit',sid,datafp,split)
    self.event('fit_completed','MC-UVE.PLSRegression.fit',sid,datafp,split)
   self.event('fit_completed','MC-UVE.coefficient_standardization',sid,digest_sub,split)
  self.event('fit_completed','MC-UVE',ids,digest,split)
  self.pair('SVC',ids,selected,y,split)
 def finish(self):check(not self.stream.readline(),'Unexpected extra fit-scope operation');self.stream.close()
