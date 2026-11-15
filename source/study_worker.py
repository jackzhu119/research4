#!/usr/bin/env python3
"""One real nested-training family variant/outer fold, resumable without subprocesses."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):os.environ[_k]='1'
import argparse,csv,hashlib,itertools,json,math,platform,resource,signal,sys,time,traceback,warnings
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold,RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score,confusion_matrix
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
VERSION='bounded-real-nested-worker-v1'
VARIANTS={
 'baseline':['svm_alpha_snv','lda_alpha_raw_snv','plsda_alpha_raw_snv','svm_joint_both_snv','lda_joint_both_snv','svm_n_raw','svm_n_snv'],
 'selection':['alpha_snv','joint_both_snv','joint_nraw_asnv','alpha_raw','alpha_snv_c2_k62','joint_both_snv_c2_k62','joint_nraw_asnv_c2_k62'],
 'spacing':['alpha_gap2_k62_c2','alpha_gap4_k62_c2','alpha_gap8_k62_c2','alpha_inner_gap_k62_c2'],
 'coverage':['uniform_k62','uniform_k123','uniform_k245','uniform_k490','binmean_k62','binmean_k123','binmean_k245','binmean_k490','random_k62','random_k123','random_k245','random_k490'],
 'rank_context':['true_n','shuffle_n','class_shuffle_n','true_raw_n','shuffle_raw_n','class_shuffle_raw_n','duplicate_alpha','gaussian_noise'],
 'retain_drop':['joint_both_snv','joint_nraw_asnv'],
 'amplitude':['svm_alpha_raw','svm_snv_plus_mean_sd','svm_mean_sd_only'],
 'permutation':['svm_alpha_snv','lda_alpha_raw_snv','mc_alpha_snv'],
 'material':['four_svm_alpha_snv','four_lda_alpha_raw_snv','four_plsda_alpha_raw_snv','binary_svm_alpha_snv','four_svm_alpha_snv_fixed_C10','binary_svm_alpha_snv_fixed_C10']}
SPEC={'version':VERSION,'enabled_families':list(VARIANTS),'variants':VARIANTS,'variant_interface':'job.variant is one fixed string from variants[family]; default is first listed. NEVER varies with fold/ordinal.','mc_draws_supported':[20,100,500],'single_run_variants':{'material':['four_svm_alpha_snv_fixed_C10','binary_svm_alpha_snv_fixed_C10']},
 'data_tasks':{'core300':'material_labels==酸枣仁 AND roles==train, exactly300; target group_ids','material':'fixed train/PRED roles; CAL unused; material target; material jobs only fold0'},
 'outer':{'folds':5,'splitter':'StratifiedKFold(shuffle=True,random_state=job.seed)','shared_across_all_core_variants':True},
 'inner':{'folds':5,'repetitions':2,'seed_rule':'SHA256("inner5x2|20261110|"+str(job.seed)), first8hex as uint32; same seed across family variants and folds','scoring':'mean macro-F1, then balanced accuracy, accuracy, smaller K, fewer PLS components, smaller gap, prefer full retain, lower C, gamma multiplier order0.1/1/10, raw before SNV'},
 'search':{'K':[31,62,123,184,245,367,490,612],'PLS_ranking_components':[2,4,8,12,16],'SVM_C':[.1,1.,10.,100.],'SVM_gamma_multipliers':[.1,1.,10.],'PLSDA_components':[2,4,8,12,16]},
 'fixed_controls':{'selection_*_c2_k62':'fixed components2,K62; SVM tuned','spacing':'fixed components2,K62; exact-cardinality dynamic programming; gaps2/4/8 or inner-selected1/2/4/8','rank_context':'fixed components2,K62; raw-n and SNV-n contexts distinguished; rank in [context,alpha_snv], take top62 alpha scores; final classifier onlyalpha; training-only context construction','coverage':'uniform endpoint-inclusive indices, contiguous-bin mean or one frozen random index set per seed/K (same across all outerfolds and their innerfits)','retain_drop':'choose full versus selected-alpha-only inside inner search alongside K/components/C/gamma; exact selected joint numerical gamma retained after deletion; empty alpha uses predefined training-prior DummyClassifier','amplitude':'raw alpha versus SNV alpha versus SNV plus per-spectrum raw mean/SD or only mean/SD','permutation':'fixed global permutation of core group labels before stratified nested splitting; distinct null target, never real-source metrics','material':'material reference context only; fixed roles, fold0 only, no PRED feedback; fixed_C10 variants use C10/gamma-scale and run only repeat0'},
 'checkpoints':'rank_cache per inner/current training/PLScomponent; all candidate scores atomically after each inner fold; selection.json sealed before outer refit/prediction; resume verifies unchanged job/input/spec/code hashes',
 'receipt':'cpu_receipt.json in finally, process_time seconds for current invocation; heartbeat has current and cumulative lineage CPU; return0 success,75 CPU/signal interruption,1 error',
 'chemistry':'No measured paired chemistry supplied: no chemical-prior training family is enabled. Spacing/context are structural diagnostics, not chemical validation.',
 'resource':'One process, no subprocesses, BLAS1. RLIMIT_CPU soft limit plus bounded1-second hard grace (inherited hard limit respected). Signal checkpoints retained.'}
class BudgetStop(Exception):pass
ACTIVE=None

def now():return datetime.now(timezone.utc).isoformat()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def filehash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def savez(p,**v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**v)
 tmp.replace(p)
def seedof(*v):return int(hashlib.sha256('|'.join(map(str,v)).encode()).hexdigest()[:8],16)
def snv(x):return (x-x.mean(1,keepdims=True))/np.maximum(x.std(1,ddof=1,keepdims=True),np.finfo(float).eps)
def metrics(y,p,classes):return [float(accuracy_score(y,p)),float(f1_score(y,p,labels=np.arange(classes),average='macro',zero_division=0)),float(balanced_accuracy_score(y,p))]
def stop_signal(signum,frame):raise BudgetStop('signal '+str(signum))

def spaced(scores,k,gap):
 if (k-1)*gap>=len(scores):raise ValueError('infeasible spacing')
 n=len(scores);dp=np.full((n+1,k+1),-np.inf);dp[:,0]=0;take=np.zeros((n+1,k+1),bool)
 for i in range(1,n+1):
  dp[i]=dp[i-1];val=scores[i-1]+dp[max(0,i-gap),:-1];better=val>dp[i,1:];dp[i,1:][better]=val[better];take[i,1:]=better
 i=n;j=k;out=[]
 while j:
  if take[i,j]:out.append(i-1);j-=1;i=max(0,i-gap)
  else:i-=1
 out=np.array(out[::-1],int);assert len(out)==k and np.all(np.diff(out)>=gap)
 return out

def mc_rank(z,y,components,draws,modelseed=42):
 rng=np.random.default_rng(modelseed);classes=np.unique(y);onehot=(y[:,None]==classes).astype(float);coeff=[]
 for _ in range(draws):
  priority=rng.random(len(y));ix=np.sort(np.concatenate([np.flatnonzero(y==c)[np.argsort(priority[y==c],kind='stable')[:int(np.ceil(.8*np.sum(y==c)))]] for c in classes]))
  v=z[ix];scale=v.std(0);scale[scale<=np.finfo(float).eps]=1;v=(v-v.mean(0))/scale;cc=[]
  for target in onehot[ix].T:
   sd=target.std()
   if sd<=np.finfo(float).eps:cc.append(np.zeros(z.shape[1]));continue
   model=PLSRegression(n_components=min(components,v.shape[1],len(v)-1),scale=False,max_iter=500,tol=1e-6)
   with warnings.catch_warnings():
    warnings.filterwarnings('ignore',message='y residual is constant.*');model.fit(v,(target-target.mean())/sd)
   cc.append(np.asarray(model.coef_).reshape(-1))
  coeff.append(cc)
 coeff=np.asarray(coeff);mu=coeff.mean(0);sd=coeff.std(0,ddof=1);floor=max(1.,float(sd.max()))*1e-12;scores=np.linalg.norm(abs(mu/np.maximum(sd,floor)),axis=0)
 if not np.isfinite(scores).all():raise ValueError('nonfinite ranking scores')
 return scores

class Worker:
 def __init__(self,job,out,heartbeat,resume,cpu_limit):
  self.job=job;self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True);self.hb=Path(heartbeat) if heartbeat else self.out/'heartbeat.json';self.resume=Path(resume) if resume else None;self.t0=time.monotonic();self.cpu0=0.;self.prior_cpu=0.;self.nfits=0;self.rank_fits=0;self.phase='initializing';self.inner_done=0;self.limit=cpu_limit;self.resume_manifest=None
  self.family=job['family'];self.variant=job.get('variant',VARIANTS[self.family][0]);self.seed=int(job['seed']);self.fold=int(job.get('fold',0));self.draws=int(job.get('mc_draws',20));self.testing=bool(job.get('testing',False));self.input=Path(job.get('data_path',ROOT/'inputs/current1672_paired612.npz'))
  if self.variant not in VARIANTS[self.family]:raise ValueError('unknown family/variant')
  if self.draws not in SPEC['mc_draws_supported']:raise ValueError('unsupported mc_draws')
  identity={k:v for k,v in job.items() if k not in ('resume_from','cpu_limit','attempt','attempt_id')};self.identity={'job':identity,'data_sha256':filehash(self.input),'spec_sha256':digest(SPEC),'code_sha256':filehash(__file__)};self.identity_sha=digest(self.identity)
  for base in [self.out,self.resume]:
   if base and (base/'job_frozen.json').exists():
    previous=json.loads((base/'job_frozen.json').read_text())
    if previous['identity_sha256']!=self.identity_sha:raise ValueError('saved job identity differs')
    if (base/'cpu_receipt.json').exists():self.prior_cpu=max(self.prior_cpu,float(json.loads((base/'cpu_receipt.json').read_text()).get('cpu_seconds_lineage',0)))
   if base and (base/'checkpoint.json').exists():
    cp=json.loads((base/'checkpoint.json').read_text())
    if cp['identity_sha256']!=self.identity_sha:raise ValueError('checkpoint identity differs')
    self.resume_manifest=cp
    if (base/'cpu_receipt.json').exists():self.prior_cpu=max(self.prior_cpu,float(json.loads((base/'cpu_receipt.json').read_text()).get('cpu_seconds_lineage',0)))
  protocol=ROOT/'protocol.json'
  if protocol.exists():
   frozen=json.loads(protocol.read_text());sealed=frozen.get('sealed_files',{})
   for name in ['study_worker.py','worker_spec.json']:
    if name in sealed and filehash(ROOT/name)!=sealed[name]:raise ValueError('frozen '+name+' changed')
   expected=frozen.get('input_sha256')
   if expected and expected!=self.identity['data_sha256']:raise ValueError('protocol input SHA differs')
  if job.get('data_sha256') and job['data_sha256']!=self.identity['data_sha256']:raise ValueError('job input SHA differs')
  dump(self.out/'job_frozen.json',{'identity_sha256':self.identity_sha,**self.identity});self.beat()
 def beat(self,**more):
  dump(self.hb,{'time_utc':now(),'pid':os.getpid(),'ordinal':self.job.get('ordinal'),'job_id':self.job.get('id',os.environ.get('STUDY_JOB_ID')),'attempt_id':os.environ.get('STUDY_ATTEMPT_ID'),'process_cpu_seconds':time.process_time(),'family':self.family,'variant':self.variant,'phase':self.phase,'inner_folds_complete':self.inner_done,'cpu_seconds':time.process_time()-self.cpu0,'cpu_seconds_this_attempt':time.process_time()-self.cpu0,'cpu_seconds_lineage':self.prior_cpu+time.process_time()-self.cpu0,'wall_seconds_this_attempt':time.monotonic()-self.t0,'svm_or_reference_fits':self.nfits,'PLS_coefficient_fits':self.rank_fits,**more})
 def cached_path(self,rel):
  for base in [self.out,self.resume]:
   if base and (base/rel).exists():return base/rel
  return None
 def load_data(self):
  with np.load(self.input,allow_pickle=False)as z:self.z={k:z[k].copy() for k in z.files}
  z=self.z;ids=z['record_ids'];assert len(set(ids))==len(ids);assert z['X_alpha'].shape[1]==612 and z['X_n'].shape==z['X_alpha'].shape;assert np.isfinite(z['X_alpha']).all() and np.isfinite(z['X_n']).all()
  material=self.family=='material' or self.job.get('task','core300') in ('material','material4','material_binary')
  if material:
   if self.fold!=0:raise ValueError('material fixed-role tasks must use fold0 only')
   mask=np.ones(len(ids),bool)
   if self.variant.startswith('binary'):mask=np.isin(z['material_labels'],['酸枣仁','枣仁'])
   self.indices=np.flatnonzero(mask);labels=z['material_labels'][self.indices];self.ids=ids[self.indices];self.roles=z['roles'][self.indices];self.classes,self.y=np.unique(labels,return_inverse=True);self.train=np.flatnonzero(self.roles=='train');self.test=np.flatnonzero(self.roles=='prediction');self.target_kind='fixed_role_material';assert len(set(self.ids[self.train])&set(self.ids[self.test]))==0
  else:
   mask=(z['material_labels']=='酸枣仁')&(z['roles']=='train');self.indices=np.flatnonzero(mask);self.ids=ids[self.indices];self.roles=z['roles'][self.indices];labels=z['group_ids'][self.indices];self.classes,self.y=np.unique(labels,return_inverse=True);assert len(self.ids)==300 and len(self.classes)==5 and all(np.bincount(self.y)==60);self.target_kind='source_group'
   if self.family=='permutation':self.y=np.random.default_rng(seedof('null_labels',self.seed)).permutation(self.y);self.target_kind='permuted_group_null'
   self.train,self.test=list(StratifiedKFold(5,shuffle=True,random_state=self.seed).split(self.ids,self.y))[self.fold]
  self.alpha=z['X_alpha'][self.indices];self.n=z['X_n'][self.indices];self.nclass=len(self.classes)
  insplit=RepeatedStratifiedKFold(n_splits=2 if self.testing else 5,n_repeats=1 if self.testing else 2,random_state=seedof('inner5x2',20261110,self.seed))
  self.inner=[(self.train[a],self.train[b]) for a,b in insplit.split(self.train,self.y[self.train])]
  dump(self.out/'split_frozen.json',{'identity_sha256':self.identity_sha,'target_kind':self.target_kind,'class_order':self.classes.tolist(),'outer_train_ids':self.ids[self.train].tolist(),'outer_test_ids':self.ids[self.test].tolist(),'inner':[{'fit_ids':self.ids[a].tolist(),'validation_ids':self.ids[b].tolist()} for a,b in self.inner],'outer_labels_consulted_for_model_selection':False})
 def config(self):
  f=self.family;v=self.variant;model='svm';pool='alpha_snv';representation='mc' if f in ('selection','spacing','rank_context','retain_drop') else 'full';ks=SPEC['search']['K'];comps=SPEC['search']['PLS_ranking_components'];gaps=[1];drops=['full'];pre=['fixed']
  if f=='baseline':
   model=v.split('_')[0];pool=v[len(model)+1:];pre=['raw','snv'] if pool.endswith('_raw_snv') else ['fixed'];pool=pool.replace('_raw_snv','_raw')
  elif f=='selection':
   pool=v.replace('_c2_k62','')
   if v.endswith('_c2_k62'):ks=[62];comps=[2]
  elif f=='spacing':ks=[62];comps=[2];gaps=[1,2,4,8] if 'inner' in v else [int(v.split('gap')[1].split('_')[0])]
  elif f=='coverage':representation=v.split('_')[0];ks=[int(v.split('k')[-1])];comps=[0]
  elif f=='rank_context':pool='context_'+v;ks=[62];comps=[2]
  elif f=='retain_drop':pool=v;drops=['full','alpha_only']
  elif f=='amplitude':pool=v[len('svm_'):]
  elif f=='permutation':
   if v.startswith('mc_'):representation='mc';pool='alpha_snv'
   else:model=v.split('_')[0];pool=v[len(model)+1:];pre=['raw','snv'] if pool.endswith('_raw_snv') else ['fixed'];pool=pool.replace('_raw_snv','_raw')
  elif f=='material':
   name=v.split('_',1)[1].replace('_fixed_C10','');model=name.split('_')[0];pool=name[len(model)+1:];pre=['raw','snv'] if pool.endswith('_raw_snv') else ['fixed'];pool=pool.replace('_raw_snv','_raw')
  if representation=='full':ks=[self.pool(pool,pre[0])[0].shape[1]];comps=SPEC['search']['PLSDA_components'] if model=='plsda' else [0]
  if self.testing:ks=[min(ks)];comps=[min(comps)];gaps=[min(gaps)];pre=[pre[0]]
  cs=[1.] if self.testing else SPEC['search']['SVM_C'];gs=[1.] if self.testing else SPEC['search']['SVM_gamma_multipliers']
  if v.endswith('_fixed_C10'):cs=[10.];gs=[1.]
  configs=[]
  for pp,k,comp,gap,drop in itertools.product(pre,ks,comps,gaps,drops):
   if gap>1 and (k-1)*gap>=612:continue
   if model=='svm':
    for c,g in itertools.product(cs,gs):configs.append({'preprocessing':pp,'K':k,'components':comp,'gap':gap,'drop':drop,'C':c,'gamma_multiplier':g})
   else:configs.append({'preprocessing':pp,'K':k,'components':comp,'gap':gap,'drop':drop,'C':0.,'gamma_multiplier':0.})
  return {'model':model,'pool':pool,'representation':representation},configs
 def pool(self,pool,pp):
  a=self.alpha;n=self.n
  if pool=='alpha_raw':return snv(a) if pp=='snv' else a,np.arange(612,1224)
  if pool=='alpha_snv':return snv(a),np.arange(612,1224)
  if pool=='joint_both_snv':return np.concatenate([snv(n),snv(a)],1),np.arange(1224)
  if pool=='joint_nraw_asnv':return np.concatenate([n,snv(a)],1),np.arange(1224)
  if pool=='n_raw':return n,np.arange(612)
  if pool=='n_snv':return snv(n),np.arange(612)
  if pool=='snv_plus_mean_sd':return np.column_stack([snv(a),a.mean(1),a.std(1,ddof=1)]),np.arange(614)+2000
  if pool=='mean_sd_only':return np.column_stack([a.mean(1),a.std(1,ddof=1)]),np.arange(2)+3000
  if pool.startswith('context_'):return np.concatenate([n if pool.endswith('raw_n') else snv(n),snv(a)],1),np.arange(1224)
  raise ValueError('unknown pool '+pool)
 def context_train(self,x,y,mode,stage):
  mode=mode.replace('_raw_n','_n');z=x.copy();rng=np.random.default_rng(seedof('context',self.seed,self.fold,mode,stage))
  if mode=='true_n':return z
  if mode=='shuffle_n':z[:,:612]=z[rng.permutation(len(z)),:612]
  elif mode=='class_shuffle_n':
   for cls in np.unique(y):
    ii=np.flatnonzero(y==cls);z[ii,:612]=z[rng.permutation(ii),:612]
  elif mode=='duplicate_alpha':z[:,:612]=z[:,612:]
  elif mode=='gaussian_noise':z[:,:612]=rng.normal(size=(len(z),612))
  else:raise ValueError(mode)
  return z
 def rank(self,zt,y,comp,stage,pool):
  rel=Path('rank_cache')/f'{stage}_c{comp}.npz';old=self.cached_path(rel)
  if old:
   with np.load(old,allow_pickle=False)as f:return f['importance'].copy()
  self.phase='ranking_'+stage+'_c'+str(comp);self.beat();rankx=self.context_train(zt,y,pool[len('context_'):],stage) if pool.startswith('context_') else zt
  scores=mc_rank(rankx,y,comp,self.draws);self.rank_fits+=self.draws*len(np.unique(y));savez(self.out/rel,importance=scores);self.beat();return scores
 def transformed(self,base,p,fit,query,stage,rankcache):
  x,globalids=self.pool(base['pool'],p['preprocessing']);rep=base['representation'];k=p['K'];selected=None
  if rep=='uniform':selected=np.rint(np.linspace(0,611,k)).astype(int);x=x[:,selected];globalids=globalids[selected]
  elif rep=='random':selected=np.sort(np.random.default_rng(seedof('coverage_random',self.seed,k)).choice(612,k,replace=False));x=x[:,selected];globalids=globalids[selected]
  elif rep=='binmean':x=np.column_stack([x[:,ix].mean(1) for ix in np.array_split(np.arange(612),k)]);globalids=np.arange(k)+10000
  scaler=StandardScaler();zt=scaler.fit_transform(x[fit]);zq=scaler.transform(x[query]);gamma0=1/(zt.shape[1]*zt.var()) if zt.var()>0 else 1.;info={'candidate_width':x.shape[1],'representation':rep}
  if rep=='mc':
   comp=p['components']
   if comp not in rankcache:rankcache[comp]=self.rank(zt,self.y[fit],comp,stage,base['pool'])
   scores=rankcache[comp]
   if base['pool'].startswith('context_'):selected=np.argsort(-scores[612:],kind='stable')[:k]+612
   else:selected=spaced(scores,k,p['gap']) if p['gap']>1 else np.argsort(-scores,kind='stable')[:k]
   selected=np.sort(selected);zt=zt[:,selected];zq=zq[:,selected];globalids=globalids[selected];gamma0=1/(len(selected)*zt.var()) if zt.var()>0 else 1.
   if p['drop']=='alpha_only':keep=globalids>=612;zt=zt[:,keep];zq=zq[:,keep];globalids=globalids[keep]
  info.update(selected_global_indices=globalids.tolist(),predictor_count=zt.shape[1],numeric_gamma=gamma0*p['gamma_multiplier'],scaler_mean=scaler.mean_,scaler_scale=scaler.scale_)
  return zt,zq,info
 def predict(self,base,p,zt,zq,ytrain,info):
  self.nfits+=1
  if zt.shape[1]==0:model=DummyClassifier(strategy='prior');model.fit(np.zeros((len(zt),1)),ytrain);pred=model.predict(np.zeros((len(zq),1)));scores=model.predict_proba(np.zeros((len(zq),1)));info['empty_block_policy']='training_class_prior';return pred,scores,model
  if base['model']=='svm':model=SVC(C=p['C'],gamma=info['numeric_gamma'],class_weight='balanced',kernel='rbf',random_state=42);model.fit(zt,ytrain);return model.predict(zq),model.decision_function(zq),model
  if base['model']=='lda':model=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=np.repeat(1/self.nclass,self.nclass));model.fit(zt,ytrain);return model.predict(zq),model.decision_function(zq),model
  model=PLSRegression(n_components=min(p['components'],zt.shape[1],len(zt)-1),scale=False,max_iter=500,tol=1e-6);model.fit(zt,np.eye(self.nclass)[ytrain]);scores=model.predict(zq);return scores.argmax(1),scores,model
 def execute(self):
  self.load_data();base,params=self.config();dump(self.out/'candidate_grid.json',{'identity_sha256':self.identity_sha,'base':base,'candidates':params,'inner_count':len(self.inner)})
  fold_scores=[]
  for hi,(fit,query) in enumerate(self.inner):
   rel=Path('inner_scores')/f'fold{hi}.npz';old=self.cached_path(rel)
   if old:
    with np.load(old,allow_pickle=False)as z:scores=z['scores'].copy()
    if scores.shape!=(len(params),3):raise ValueError('resumed score shape mismatch')
    fold_scores.append(scores);self.inner_done=hi+1;self.beat();continue
   self.phase='inner_'+str(hi);self.beat();scores=[];ranks={};transformcache={};predictioncache={}
   for pi,p in enumerate(params):
    key=(p['preprocessing'],p['K'],p['components'],p['gap'],p['drop'])
    if key not in transformcache:
     # Limit matrix cache: current 12-SVM grid only, ranks remain available for all K.
     transformcache={key:self.transformed(base,p,fit,query,'inner'+str(hi),ranks)}
    zt,zq,info=transformcache[key];pk=(hashlib.sha256(np.asarray(info['selected_global_indices'],dtype=np.int64).tobytes()).hexdigest(),p['preprocessing'],p['C'],info['numeric_gamma'],base['model'],p['components'] if base['model']=='plsda' else 0)
    if pk not in predictioncache:
     pred,_,_=self.predict(base,p,zt,zq,self.y[fit],info);predictioncache[pk]=metrics(self.y[query],pred,self.nclass)
    scores.append(predictioncache[pk])
    if pi%96==0:self.beat(candidate_index=pi,candidate_count=len(params))
   scores=np.asarray(scores);savez(self.out/rel,scores=scores);fold_scores.append(scores);self.inner_done=hi+1;dump(self.out/'checkpoint.json',{'identity_sha256':self.identity_sha,'inner_folds_complete':self.inner_done,'inner_total':len(self.inner),'time_utc':now(),'phase':'inner_checkpoint'});self.beat()
  mean=np.mean(fold_scores,0)
  def choice(i):
   p=params[i];m=mean[i];return (-m[1],-m[2],-m[0],p['K'],p['components'],p['gap'],p['drop']!='full',p['C'],p['gamma_multiplier'],p['preprocessing']=='snv')
  best=min(range(len(params)),key=choice);p=params[best];lock={'identity_sha256':self.identity_sha,'base':base,'chosen':p,'mean_inner_metrics':dict(zip(['accuracy','macro_f1','balanced_accuracy'],map(float,mean[best]))),'candidate_index':best,'inner_folds':len(self.inner),'outer_predictions_consulted':False,'frozen_utc':now()};dump(self.out/'selection.json',lock)
  self.phase='outer_refit_after_selection_lock';self.beat();zt,zq,info=self.transformed(base,p,self.train,self.test,'outer',{});pred,dec,model=self.predict(base,p,zt,zq,self.y[self.train],info)
  self.phase='outer_prediction_complete';m=metrics(self.y[self.test],pred,self.nclass)
  rows=[]
  for ix,pr in zip(self.test,pred):rows.append({'ordinal':self.job.get('ordinal'),'seed':self.seed,'fold':self.fold,'family':self.family,'variant':self.variant,'task':self.target_kind,'record_id':str(self.ids[ix]),'physical_sample_id':str(self.ids[ix]),'true_label':str(self.classes[self.y[ix]]),'prediction':str(self.classes[pr]),'is_correct':int(pr==self.y[ix])})
  with (self.out/'predictions.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  arrays={'selected_global_indices':np.array(info['selected_global_indices'],int),'scaler_mean':info['scaler_mean'],'scaler_scale':info['scaler_scale'],'decision_scores':np.asarray(dec),'train_record_ids':self.ids[self.train],'test_record_ids':self.ids[self.test]}
  if hasattr(model,'support_vectors_'):arrays.update(support_vectors=model.support_vectors_,dual_coef=model.dual_coef_,intercept=model.intercept_)
  elif hasattr(model,'coef_'):arrays['model_coef']=np.asarray(model.coef_)
  savez(self.out/'fitted_artifacts.npz',**arrays);dump(self.out/'inner_candidate_means.json',{'parameters':params,'mean_metrics':mean.tolist()})
  result={'status':'complete','complete':True,'job_id':self.job.get('id',os.environ.get('STUDY_JOB_ID')),'attempt_id':os.environ.get('STUDY_ATTEMPT_ID'),'identity_sha256':self.identity_sha,'job':self.job,'family':self.family,'variant':self.variant,'seed':self.seed,'fold':self.fold,'mc_draws':self.draws,'target_kind':self.target_kind,'testing_only':self.testing,'train_n':len(self.train),'test_n':len(self.test),'inner_folds':len(self.inner),'class_order':self.classes.tolist(),'classes':self.classes.tolist(),'selected_indices':info['selected_global_indices'],'candidate_dimension':info['candidate_width'],'eligible_selection_dimension':612 if self.family=='rank_context' else info['candidate_width'],'ranking_context_dimension':1224 if self.family=='rank_context' else None,'predictions':[{'record_id':row['record_id'],'physical_sample_id':row['physical_sample_id'],'y_true':row['true_label'],'prediction':row['prediction'],'is_correct':row['is_correct']} for row in rows],'chosen_parameters':p,'selected_predictor_count':info['predictor_count'],'selected_n':int(sum(0<=g<612 for g in info['selected_global_indices'])),'selected_alpha':int(sum(612<=g<1224 for g in info['selected_global_indices'])),'numeric_gamma':info['numeric_gamma'],'outer_metrics':dict(zip(['accuracy','macro_f1','balanced_accuracy'],m)),'confusion_matrix':confusion_matrix(self.y[self.test],pred,labels=np.arange(self.nclass)).tolist(),'prediction_file':'predictions.csv','selection_frozen_before_outer_prediction':True,'new_classifier_fits_this_attempt':self.nfits,'new_pls_coefficient_fits_this_attempt':self.rank_fits,'completed_utc':now()};dump(self.out/'result.json',result);self.phase='complete';self.beat();return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--job','--job-file',dest='job');parser.add_argument('--out','--output-dir',dest='out');parser.add_argument('--heartbeat-file');parser.add_argument('--resume-from');parser.add_argument('--cpu-limit',type=int,default=1800);parser.add_argument('--write-spec',action='store_true');args=parser.parse_args()
 if args.write_spec:dump(ROOT/'worker_spec.json',SPEC);print(json.dumps(SPEC));return 0
 if not args.job or not args.out:parser.error('--job and --out required')
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True);t0=time.monotonic();cpu0=0.;w=None;status='failed';exitcode=1;error=None
 signal.signal(signal.SIGXCPU,stop_signal);signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
 soft,hard=resource.getrlimit(resource.RLIMIT_CPU);cap=args.cpu_limit;hardcap=cap+1 if hard==resource.RLIM_INFINITY else min(hard,cap+1);resource.setrlimit(resource.RLIMIT_CPU,(min(cap,hardcap),hardcap))
 try:
  job=json.loads(Path(args.job).read_text());w=Worker(job,args.out,args.heartbeat_file,args.resume_from or job.get('resume_from'),args.cpu_limit)
  with threadpool_limits(limits=1):result=w.execute()
  status='complete';exitcode=0
 except BudgetStop as e:
  status='interrupted';exitcode=75;error=str(e)
 except Exception as e:
  status='failed';error=type(e).__name__+': '+str(e);dump(out/'error.json',{'error':error,'traceback':traceback.format_exc(),'time_utc':now()})
 finally:
  used=time.process_time()-cpu0;prior=w.prior_cpu if w else 0.;dump(out/'cpu_receipt.json',{'status':status,'exit_code':exitcode,'pid':os.getpid(),'job_id':w.job.get('id',os.environ.get('STUDY_JOB_ID')) if w else os.environ.get('STUDY_JOB_ID'),'attempt_id':os.environ.get('STUDY_ATTEMPT_ID'),'process_cpu_seconds':time.process_time(),'cpu_seconds':used,'cpu_seconds_this_attempt':used,'cpu_seconds_lineage':prior+used,'wall_seconds':time.monotonic()-t0,'cpu_limit':args.cpu_limit,'error':error,'closed_utc':now(),'accounting':'process_time for this process; no subprocesses','worker_version':VERSION})
  if status!='complete':dump(out/'result.json',{'status':status,'complete':False,'exit_code':exitcode,'error':error,'checkpoint_preserved':True,'selection_or_outer_score_not_used_to_stop':True})
  if w:w.phase=status;w.beat(error=error)
 print(json.dumps({'status':status,'exit_code':exitcode,'cpu_seconds':time.process_time()-cpu0,'out':str(out)}),flush=True);return exitcode
if __name__=='__main__':sys.exit(main())
