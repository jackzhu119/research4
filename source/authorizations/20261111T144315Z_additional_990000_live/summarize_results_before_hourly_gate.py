#!/usr/bin/env python3
"""Read-only scientific summaries of sealed worker results; never imports/fits worker.
Only ROOT/jobs/*/result.json is a result source. Every output file is atomic.
"""
from __future__ import annotations
import argparse,collections,fcntl,hashlib,itertools,json,math,os,random,tempfile,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix

VERSION='background-summary-v1'
GROUP=['target_kind','family','variant','mc_draws']
SUMMARY=GROUP+['state','complete_repeats','unique_records','unique_physical_ids','prediction_occurrences','mean_accuracy','accuracy_ci95_low','accuracy_ci95_high','accuracy_split_sd','mean_macro_f1','macro_f1_split_sd','mean_balanced_accuracy','balanced_accuracy_split_sd','bootstrap_draws','bootstrap_seed','bootstrap_status']
def now():return datetime.now(timezone.utc).isoformat()
def read_json(path,default=None):
 try:return json.loads(Path(path).read_text())
 except (FileNotFoundError,json.JSONDecodeError,OSError):return default

def atomic_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
 try:
  with os.fdopen(fd,'w')as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def atomic_text(path,text):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
 try:
  with os.fdopen(fd,'w')as f:f.write(text);f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def atomic_csv(path,rows,columns):
 df=pd.DataFrame(rows);df=df.reindex(columns=list(dict.fromkeys(columns+df.columns.tolist())))
 atomic_text(path,df.to_csv(index=False))
def stable(v):return v if isinstance(v,str)else json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def key_for(job,kind=None):
 family=str(job.get('family','unknown'));kind=kind or('fixed_role_material'if family=='material'else'permuted_group_null'if family=='permutation'else'source_group')
 return(kind,family,stable(job.get('variant','unknown')),int(job.get('mc_draws',20)))
def uses_mc(r):return r['family']in('selection','spacing','rank_context','retain_drop')or(r['family']=='permutation'and str(r['variant']).startswith('mc_'))
def draw_text(r):return str(r['mc_draws'])if uses_mc(r)else'n/a'
def meta(k):
 d=dict(zip(GROUP,k));d['mc_draws_applicable']=uses_mc(d);return d
def metrics(y,p,classes):
 return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,labels=classes,average='macro',zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(y,p))}
def std(v):return float(np.std(v,ddof=1))if len(v)>1 else None

def input_metadata(root):
 path=root/'inputs/current1672_paired612.npz';records={};core=set()
 if path.exists():
  with np.load(path,allow_pickle=False)as z:
   ids=z['record_ids'];groups=z['group_ids'];physical=z['physical_sample_ids']if'physical_sample_ids'in z.files else ids;materials=z['material_labels'];roles=z['roles']
   for rid,pid,g,m,r in zip(ids,physical,groups,materials,roles):
    records[str(rid)]={'physical_sample_id':str(pid),'stratum':str(g),'material':str(m),'role':str(r)}
    if str(m)=='酸枣仁'and str(r)=='train':core.add(str(rid))
 return records,core

class Bootstrap:
 def __init__(self,draws,seed):self.draws=draws;self.seed=seed;self.cache={}
 def interval(self,records):
  # Input rows already average each record's correctness over complete repeats.
  d=pd.DataFrame(records);phys=d.groupby('physical_sample_id',sort=True).agg(correct_sum=('mean_correct','sum'),record_n=('record_id','size'),strata_n=('stratum','nunique'),stratum=('stratum','first'))
  if(phys.strata_n>1).any():return None,None,'pending_physical_id_spans_strata'
  if len(phys)<2:return None,None,'pending_fewer_than_two_physical_ids'
  if phys.groupby('stratum').size().min()<2:return None,None,'pending_fewer_than_two_physical_ids_in_a_class'
  signature=tuple((str(i),str(r.stratum))for i,r in phys.iterrows())
  if signature not in self.cache:
   rng=random.Random(self.seed);labels=phys.stratum.to_numpy();blocks=[]
   for cls in sorted(set(labels)):
    ix=np.flatnonzero(labels==cls);n=len(ix)
    blocks.append(ix[np.array([[rng.randrange(n)for _ in range(n)]for _ in range(self.draws)],dtype=np.int32)])
   self.cache[signature]=np.concatenate(blocks,axis=1)
  ix=self.cache[signature];numer=phys.correct_sum.to_numpy()[ix].sum(axis=1);denom=phys.record_n.to_numpy()[ix].sum(axis=1);values=numer/denom;lo,hi=np.quantile(values,[.025,.975],method='linear')
  return float(lo),float(hi),'conditional_class_stratified_physical_id'

def validate_result(root,path,payload,job,registry,core):
 if payload.get('testing_only')or payload.get('job',{}).get('testing')or job.get('testing'):return None,'excluded_testing'
 if not payload.get('complete')or payload.get('status')!='complete':return None,'pending_'+str(payload.get('status','incomplete'))
 actual=payload.get('job',{});actual={**job,**actual}
 for field in ['family','variant','seed','fold','mc_draws']:
  if field in job and stable(actual.get(field))!=stable(job[field]):return None,'invalid_job_identity_'+field
  if field in payload and field in actual and stable(payload[field])!=stable(actual[field]):return None,'invalid_result_identity_'+field
 kind=payload.get('target_kind');k=key_for(actual,kind)
 if kind not in('source_group','permuted_group_null','fixed_role_material'):return None,'invalid_target_kind'
 if k[1]=='material'and kind!='fixed_role_material':return None,'invalid_material_target'
 if k[1]=='permutation'and kind!='permuted_group_null':return None,'invalid_null_target'
 if not payload.get('selection_frozen_before_outer_prediction',False):return None,'invalid_missing_selection_lock'
 classes=[str(x)for x in payload.get('class_order',payload.get('classes',[]))];pr=payload.get('predictions',[])
 if not classes or not pr:return None,'invalid_empty_predictions_or_classes'
 if kind!='fixed_role_material'and len(classes)!=5:return None,'invalid_source_class_count'
 ids=[str(x.get('record_id',''))for x in pr]
 if not all(ids)or len(ids)!=len(set(ids)):return None,'invalid_duplicate_or_empty_record_ids'
 if len(pr)!=int(payload.get('test_n',len(pr))):return None,'invalid_test_n'
 fold=int(actual.get('fold',payload.get('fold',-1)));seed=int(actual.get('seed',payload.get('seed',-1)));rows=[]
 for item in pr:
  rid=str(item['record_id']);true=str(item.get('y_true',item.get('true_label','')));pred=str(item.get('prediction',''));pid=str(item.get('physical_sample_id',rid));canonical=registry.get(rid,{})
  if true not in classes or pred not in classes:return None,'invalid_class_label'
  if int(item.get('is_correct',int(true==pred)))!=int(true==pred):return None,'invalid_correctness_flag'
  if canonical and pid!=canonical['physical_sample_id']:return None,'invalid_physical_id_mapping'
  if kind=='source_group'and canonical and true!=canonical['stratum']:return None,'invalid_original_source_label'
  if kind=='fixed_role_material'and canonical and(canonical['role']!='prediction'or true!=canonical['material']):return None,'invalid_material_PRED_role_or_label'
  rows.append({'record_id':rid,'physical_sample_id':pid,'y_true':true,'prediction':pred,'is_correct':int(true==pred),'stratum':canonical.get('stratum',true),'seed':seed,'repeat':actual.get('repeat'),'fold':fold})
 if kind!='fixed_role_material':
  if fold not in range(5)or len(rows)!=60:return None,'invalid_source_fold_size'
  if core and not set(ids).issubset(core):return None,'invalid_source_population'
 else:
  if fold!=0:return None,'invalid_material_fold'
  expected={rid for rid,r in registry.items()if r['role']=='prediction'and r['material']in classes}
  if expected and set(ids)!=expected:return None,'invalid_material_prediction_population'
 return {'key':k,'seed':seed,'repeat':actual.get('repeat'),'fold':fold,'classes':classes,'rows':rows,'result':payload,'job_id':str(actual.get('id',payload.get('job_id',path.parent.name))),'path':str(path.relative_to(root))},'complete'

def aggregate_group(k,units,boot,registry):
 predictions=[r for u in units for r in u['rows']];df=pd.DataFrame(predictions);classes=units[0]['classes'];seeds=sorted({u['seed']for u in units});reps=[];conf=[];perrecord=[];recurrent=[]
 for seed in seeds:
  d=df[df.seed==seed];m=metrics(d.y_true,d.prediction,classes);reps.append({**meta(k),'seed':seed,'unique_records':d.record_id.nunique(),'prediction_occurrences':len(d),**m})
  cm=confusion_matrix(d.y_true,d.prediction,labels=classes)
  for i,c in enumerate(classes):
   for j,p in enumerate(classes):conf.append({**meta(k),'seed':seed,'true_class':c,'predicted_class':p,'count':int(cm[i,j]),'row_support':int(cm[i].sum()),'row_fraction':float(cm[i,j]/cm[i].sum())if cm[i].sum()else None})
 for rid,d in df.groupby('record_id',sort=True):
  if d.physical_sample_id.nunique()!=1:raise ValueError('one record maps to multiple physical IDs')
  strata=str(d.y_true.iloc[0])if k[0]=='fixed_role_material'else registry.get(rid,{}).get('stratum')
  if strata is None:
   if d.y_true.nunique()!=1:raise ValueError('permuted labels require canonical physical class registry')
   strata=str(d.y_true.iloc[0])
  row={**meta(k),'record_id':rid,'physical_sample_id':str(d.physical_sample_id.iloc[0]),'stratum':str(strata),'complete_repeats':len(seeds),'mean_correct':float(d.is_correct.mean())};perrecord.append(row)
  eligible=d[d.y_true=='ZY07'];errors=eligible[eligible.prediction=='ZY04']
  if len(errors):recurrent.append({**meta(k),'record_id':rid,'physical_sample_id':row['physical_sample_id'],'canonical_source':row['stratum'],'complete_repeats':len(seeds),'ZY07_opportunities':len(eligible),'ZY07_to_ZY04_occurrences':len(errors),'recurrent_two_or_more':len(errors)>=2,'in_every_eligible_repeat':len(errors)==len(eligible),'error_seeds':';'.join(str(x)for x in sorted(errors.seed.tolist()))})
 lo,hi,state=boot.interval(perrecord);means={name:float(np.mean([r[name]for r in reps]))for name in['accuracy','macro_f1','balanced_accuracy']}
 summary={**meta(k),'state':'reportable_complete_repeats','complete_repeats':len(seeds),'unique_records':int(df.record_id.nunique()),'unique_physical_ids':int(df.physical_sample_id.nunique()),'prediction_occurrences':len(df),'mean_accuracy':means['accuracy'],'accuracy_ci95_low':lo,'accuracy_ci95_high':hi,'accuracy_split_sd':std([r['accuracy']for r in reps]),'mean_macro_f1':means['macro_f1'],'macro_f1_split_sd':std([r['macro_f1']for r in reps]),'mean_balanced_accuracy':means['balanced_accuracy'],'balanced_accuracy_split_sd':std([r['balanced_accuracy']for r in reps]),'bootstrap_draws':boot.draws,'bootstrap_seed':boot.seed,'bootstrap_status':state}
 return summary,reps,conf,perrecord,recurrent

def stability(k,units):
 fits=[];rows=[]
 for u in units:
  r=u['result'];indices=r.get('selected_indices')
  if indices is None:continue
  selected=set(map(int,indices));eligible=r.get('eligible_selection_dimension');context=r.get('ranking_context_dimension')
  if eligible is None:eligible=612 if k[1]=='rank_context'else r.get('candidate_dimension')
  if context is None:context=1224 if k[1]=='rank_context'else r.get('candidate_dimension')
  fits.append({'seed':u['seed'],'fold':u['fold'],'set':selected,'eligible':int(eligible)if eligible is not None else None,'context':context})
  rows.append({**meta(k),'seed':u['seed'],'fold':u['fold'],'job_id':u['job_id'],'selected_count':len(selected),'eligible_selection_dimension':eligible,'ranking_context_dimension':context,'selected_global_indices':json.dumps(sorted(selected))})
 if not fits:return [],rows
 summaries=[]
 for scope,ss in [('all_completed_fits',fits)]+[(f'within_repeat_{seed}',[x for x in fits if x['seed']==seed])for seed in sorted({x['seed']for x in fits})]:
  js=[];empty=0;within=[];between=[]
  for l,r in itertools.combinations(ss,2):
   union=l['set']|r['set']
   if not union:empty+=1;continue
   v=len(l['set']&r['set'])/len(union);js.append(v);(within if l['seed']==r['seed']else between).append(v)
  dimensions={r['eligible']for r in ss};phi=None;reason='pending_fewer_than_two_fits';D=next(iter(dimensions))if len(dimensions)==1 else None
  if len(ss)>=2:
   if D is None:reason='undefined_no_common_candidate_dimension'
   else:
    union=set().union(*(r['set']for r in ss));counts=collections.Counter(i for r in ss for i in r['set']);n=len(ss);kbar=float(np.mean([len(r['set'])for r in ss]))
    if len(union)>D:reason='undefined_transformed_features_have_no_common_candidate_universe'
    elif not(0<kbar<D):reason='undefined_all_or_no_candidates_selected'
    else:
     p=np.array(list(counts.values()),float)/n;phi=float(1-(n/(n-1)*np.sum(p*(1-p)))/(kbar*(1-kbar/D)));reason='defined_descriptive_overlapping_fits'
  summaries.append({**meta(k),'scope':scope,'fits':len(ss),'pair_count':len(ss)*(len(ss)-1)//2,'finite_jaccard_pairs':len(js),'both_empty_pairs_undefined':empty,'mean_jaccard':float(np.mean(js))if js else None,'median_jaccard':float(np.median(js))if js else None,'within_repeat_mean_jaccard':float(np.mean(within))if within else None,'across_repeat_mean_jaccard':float(np.mean(between))if between else None,'eligible_selection_dimension':D,'ranking_context_dimensions':';'.join(str(x)for x in sorted({r['context']for r in ss},key=str)),'mean_selected_count':float(np.mean([len(r['set'])for r in ss])),'nogueira':phi,'nogueira_status':reason,'interpretation':'Descriptive overlapping training fits; no independent-fit or chemical-marker claim'})
 return summaries,rows

def render_comparison(out,source,nulls,materials,fingerprint):
 prior=read_json(out/'figures_status.json',{})
 if prior.get('fingerprint')==fingerprint and all((out/p).exists()for p in prior.get('files',[])):return prior
 data=[r for r in source if r.get('mean_accuracy')is not None]
 if not data and not materials:return {'state':'pending_no_complete_results','files':[],'fingerprint':fingerprint}
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.backends.backend_pdf import PdfPages
 from matplotlib.patches import Rectangle
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,'xtick.labelsize':9,'ytick.labelsize':9,'pdf.fonttype':42,'svg.fonttype':'none'})
 # All real-source groups are included in prespecified family/variant/draw order; no best-score selection.
 rows=sorted(data,key=lambda r:(r['family'],r['variant'],r['mc_draws']))
 files=[];pages=[rows[i:i+12]for i in range(0,len(rows),12)]
 if not pages and materials:pages=[[]]
 with tempfile.TemporaryDirectory(prefix='.figure-stage-',dir=out)as temp:
  temp=Path(temp);pdf=temp/'source_accuracy_comparison.pdf'
  with PdfPages(pdf)as book:
   for pn,chunk in enumerate(pages,1):
    fig,ax=plt.subplots(figsize=(170/25.4,max(3.,.42*max(len(chunk),1)+1.3)));fig.add_artist(Rectangle((.008,.01),.984,.98,transform=fig.transFigure,fill=False,lw=.65,edgecolor='#202020'))
    if chunk:
     for j,r in enumerate(chunk):
      v=r['mean_accuracy']*100;lo=r['accuracy_ci95_low'];hi=r['accuracy_ci95_high']
      if lo is not None and hi is not None:ax.errorbar(v,j,xerr=[[v-lo*100],[hi*100-v]],fmt='o',color='#0072B2',lw=1,capsize=2,ms=4)
      else:ax.plot(v,j,'o',color='#0072B2',ms=4)
     labels=[f"{r['family']}: {r['variant']}\nMC{draw_text(r)}; R={r['complete_repeats']}"for r in chunk];ax.set(yticks=range(len(chunk)),yticklabels=labels,ylim=(len(chunk)-.5,-.5),xlim=(0,100),xlabel='OOF accuracy (%)');ax.grid(axis='x',alpha=.22,lw=.5);ax.set_axisbelow(True);ax.set_title('Source task: complete fivefold repeats only',loc='left',pad=10)
    else:ax.axis('off');ax.text(.5,.5,'Source OOF pending; material PRED results are reported separately.',ha='center',va='center',wrap=True)
    fig.subplots_adjust(left=.49,right=.96,top=.88,bottom=.14);name=f'source_accuracy_comparison_{pn:03d}.svg';fig.savefig(temp/name);book.savefig(fig);plt.close(fig);files.append(name)
  files.append(pdf.name)
  for name in files:os.replace(temp/name,out/name)
 return {'state':'ready','fingerprint':fingerprint,'files':files,'source_groups_plotted':len(rows),'scope':'Conditional class-stratified physical-ID intervals; repeated splits are not new specimens. Material/null tables remain separate.'}

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);parser.add_argument('--out',type=Path);parser.add_argument('--bootstrap-draws',type=int,default=5000);parser.add_argument('--seed',type=int,default=20261110);parser.add_argument('--skip-figures',action='store_true');args=parser.parse_args();root=args.root.resolve();out=(args.out or root/'reports').resolve();out.mkdir(parents=True,exist_ok=True)
 if args.bootstrap_draws<100:parser.error('--bootstrap-draws must be at least100')
 with (out/'.summarizer.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:print(json.dumps({'state':'already_running','out':str(out)}));return 0
  start=time.monotonic();registry,core=input_metadata(root);manifest=read_json(root/'job_manifest.json',{});all_expected={str(j['id']):j for j in manifest.get('jobs',[])if'id'in j};expected={i:j for i,j in all_expected.items()if not j.get('testing')};status=read_json(root/'status.json',{});valid=[];issues=[];excluded=0;job_rows=[];seen=set()
  # Deliberately no rglob: smoke_runtime and nested attempt/test outputs cannot enter.
  for path in sorted((root/'jobs').glob('*/result.json'))if(root/'jobs').exists()else []:
   payload=read_json(path)
   if not isinstance(payload,dict):issues.append({'path':str(path.relative_to(root)),'state':'invalid_json'});continue
   rawjob=payload.get('job',{});jid=str(rawjob.get('id',payload.get('job_id',path.parent.name)));job=expected.get(jid,rawjob)
   if all_expected.get(jid,{}).get('testing'):excluded+=1;continue
   if 'jobs'in manifest and jid not in expected:
    if payload.get('testing_only')or rawjob.get('testing'):excluded+=1;continue
    issues.append({'job_id':jid,'path':str(path.relative_to(root)),'state':'excluded_not_in_manifest'});continue
   item,state=validate_result(root,path,payload,job,registry,core);seen.add(jid)
   if state=='excluded_testing':excluded+=1;continue
   job_rows.append({'job_id':jid,'state':state,'path':str(path.relative_to(root)),**meta(item['key']if item else key_for(job))})
   if item:valid.append(item)
   elif state.startswith('invalid'):issues.append({'job_id':jid,'path':str(path.relative_to(root)),'state':state})
  for jid,job in expected.items():
   if jid not in seen:job_rows.append({'job_id':jid,'state':'pending_no_complete_result',**meta(key_for(job))})
  base={'schema':VERSION,'updated_utc':now(),'snapshot_ready':False,'study_completion_inferred':False,'manifest_jobs':len(expected),'validated_complete_result_jobs':len(valid),'excluded_testing_results':excluded,'invalid_result_jobs':len(issues),'supervisor_state':status.get('state','not_available'),'supervisor_updated_utc':status.get('updated_utc'),'observed_training_cpu_seconds':status.get('observed_training_cpu_seconds'),'cpu_budget_seconds':status.get('cpu_budget_seconds'),'supervisor_job_counts':status.get('job_counts',{}),'result_discovery':'ROOT/jobs/*/result.json only','new_model_fits':0};atomic_json(out/'progress.json',base)
  units=collections.defaultdict(list);expected_blocks=collections.defaultdict(set)
  for j in expected.values():
   k=key_for(j);expected_blocks[(k,int(j.get('seed',-1)))].add(int(j.get('fold',0)))
  for u in valid:units[(u['key'],u['seed'])].append(u)
  boot=Bootstrap(args.bootstrap_draws,args.seed);groups=collections.defaultdict(list);blocks=[]
  for pair in sorted(set(expected_blocks)|set(units),key=str):
   k,seed=pair;us=units.get(pair,[]);folds=[u['fold']for u in us];material=k[0]=='fixed_role_material';state='pending';reason='awaiting_complete_result'if material else'awaiting_all_five_outer_folds'
   if material:
    if len(us)==1 and folds==[0]:state='complete';groups[k]+=us
    elif len(us)>1:reason='duplicate_material_result'
   elif len(us)==5 and set(folds)==set(range(5)):
    allrows=[r for u in us for r in u['rows']];ids=[r['record_id']for r in allrows];class_orders={tuple(u['classes'])for u in us}
    if len(ids)!=300 or len(set(ids))!=300:reason='invalid_oof_overlap_or_size'
    elif core and set(ids)!=core:reason='invalid_oof_population'
    elif len(class_orders)!=1:reason='invalid_class_order'
    elif sorted(collections.Counter(r['y_true']for r in allrows).values())!=[60]*5:reason='invalid_oof_class_support'
    else:state='complete';groups[k]+=us
   blocks.append({**meta(k),'seed':seed,'state':state,'reason':None if state=='complete'else reason,'complete_outer_folds':len(set(folds)),'required_outer_folds':1 if material else 5,'present_folds':';'.join(map(str,sorted(set(folds))))})
  source=[];nulls=[];materials=[];rep_rows=[];conf_rows=[];record_rows=[];rec_rows=[];stab=[];selected=[]
  allkeys={k for k,_ in expected_blocks}|{k for k,_ in units}
  for k in sorted(allkeys,key=str):
   us=groups.get(k,[])
   if not us:
    row={**meta(k),'state':'pending_no_complete_repeat','complete_repeats':0}
    (materials if k[0]=='fixed_role_material'else nulls if k[0]=='permuted_group_null'else source).append(row);continue
   try:summary,reps,conf,records,recurrent=aggregate_group(k,us,boot,registry)
   except ValueError as e:issues.append({**meta(k),'state':'invalid_aggregate','reason':str(e)});continue
   if k[0]=='fixed_role_material':summary['state']='fixed_PRED_reportable';summary['interpretation']='Repeated fits reuse the same fixed PRED specimens; not source OOF or new specimens.';materials.append(summary)
   elif k[0]=='permuted_group_null':nulls.append(summary)
   else:source.append(summary)
   rep_rows+=reps;conf_rows+=conf;record_rows+=records;rec_rows+=recurrent
   if k[0]!='fixed_role_material':ss,rr=stability(k,us);stab+=ss;selected+=rr
  atomic_csv(out/'source_accuracy_summary.csv',source,SUMMARY);atomic_csv(out/'permutation_null_summary.csv',nulls,SUMMARY);atomic_csv(out/'material_PRED_summary.csv',materials,SUMMARY)
  atomic_csv(out/'completed_repeat_metrics.csv',rep_rows,GROUP+['seed','unique_records','prediction_occurrences','accuracy','macro_f1','balanced_accuracy'])
  for kind,name in [('source_group','source'),('permuted_group_null','permutation_null'),('fixed_role_material','material_PRED')]:
   atomic_csv(out/f'{name}_confusion_by_repeat.csv',[r for r in conf_rows if r['target_kind']==kind],GROUP+['seed','true_class','predicted_class','count','row_support','row_fraction'])
  atomic_csv(out/'per_record_repeat_mean_correct.csv',record_rows,GROUP+['record_id','physical_sample_id','stratum','complete_repeats','mean_correct'])
  atomic_csv(out/'ZY07_to_ZY04_recurrent_ids.csv',[r for r in rec_rows if r['target_kind']=='source_group'],GROUP+['record_id','physical_sample_id','canonical_source','complete_repeats','ZY07_opportunities','ZY07_to_ZY04_occurrences','recurrent_two_or_more','in_every_eligible_repeat','error_seeds'])
  atomic_csv(out/'selection_stability.csv',stab,GROUP+['scope','fits','pair_count','finite_jaccard_pairs','mean_jaccard','median_jaccard','eligible_selection_dimension','nogueira','nogueira_status'])
  atomic_csv(out/'selected_features_by_fit.csv',selected,GROUP+['seed','fold','job_id','selected_count','eligible_selection_dimension','ranking_context_dimension','selected_global_indices'])
  atomic_csv(out/'repeat_completion.csv',blocks,GROUP+['seed','state','reason','complete_outer_folds','required_outer_folds','present_folds']);atomic_csv(out/'job_result_index.csv',job_rows,['job_id','state','path']+GROUP);atomic_json(out/'invalid_or_excluded_results.json',issues)
  fingerprint=hashlib.sha256(json.dumps({'source':source,'materials':materials},sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
  figure={'state':'skipped_by_cli','files':[],'fingerprint':fingerprint}
  if not args.skip_figures:
   try:figure=render_comparison(out,source,nulls,materials,fingerprint)
   except Exception as e:figure={'state':'figure_failed','files':[],'error_type':type(e).__name__,'error':str(e),'fingerprint':fingerprint}
  atomic_json(out/'figures_status.json',figure)
  report=['# 后台研究结果快照','',f'更新：{now()}。此文件只读汇总训练结果，不启动训练，也不按准确率调整队列。',f'已验证完成外折/固定任务：{len(valid)}；冻结队列任务：{len(expected)}；排除测试结果：{excluded}。后台状态：{status.get("state","暂无状态文件")}。','',f'真实来源分析仅纳入完成全部5个外折的重复；当前可报告方法组数：{sum(r.get("complete_repeats",0)>0 for r in source)}。这些重复共享同一批300条核心记录，重复预测不增加独立样本。']
  ready=[r for r in source if r.get('mean_accuracy')is not None]
  if not ready:report+=['','尚无可报告的完整五折来源结果。未完成块列于repeat_completion.csv；不报告其准确率。']
  else:
   report+=['','| 方法族 / 变体 / MC | 完整重复 | 唯一记录/物理ID | 平均准确率及条件95%区间 | 划分SD |','|---|---:|---:|---:|---:|']
   for r in ready:
    ci='区间待核'if r['accuracy_ci95_low']is None else f"{100*r['accuracy_ci95_low']:.2f}–{100*r['accuracy_ci95_high']:.2f}%";sd='待至少2次完整重复'if r['accuracy_split_sd']is None else f"{100*r['accuracy_split_sd']:.2f}个百分点"
    report.append(f"| {r['family']} / {r['variant']} / {draw_text(r)} | {r['complete_repeats']} | {r['unique_records']}/{r['unique_physical_ids']} | {100*r['mean_accuracy']:.2f}% [{ci}] | {sd} |")
  report+=['','区间先对每条记录的跨重复正确性取均值，再在原来源类别内按物理ID聚类重采样。每次抽样保留该物理ID下所有记录；这是给定已训练外测预测的条件区间，不包含重采样重新拟合，不代表外部或跨批次泛化。划分SD另报；只有一次完整重复时不填零。',f'bootstrap：{args.bootstrap_draws}次，Python random.Random.randrange，种子{args.seed}；未经多重性校正。','',f'标签置换阴性对照另列permutation_null_summary.csv；可报告组数{sum(r.get("complete_repeats",0)>0 for r in nulls)}，不得与真实来源准确率混合。',f'材料固定PRED任务另列material_PRED_summary.csv；可报告组数{sum(r.get("complete_repeats",0)>0 for r in materials)}。重复模型复用相同PRED样本，不作为新增独立样本，也不与来源OOF合并。','逐组逐重复混淆、ZY07→ZY04反复错误ID、选中特征与Jaccard/Nogueira分别保存在对应CSV中。全选/空集或候选宇宙不一致时，Nogueira明确列为未定义；重叠训练折不视为独立重复。',f'图件状态：{figure["state"]}。未完成任务保持pending；本快照不据已出现结果推断整个后台研究已经完成。']
  budget_stop=read_json(root/'CPU_BUDGET_STOP.json',{})
  if budget_stop:
   base['resource_budget_stop']=budget_stop
   report+=['','停止原因：总CPU预算保护触发。管理与汇总开销计入授权上限，并预留收尾计算；这不是准确率提前停止。调度器通过既有STOP入口收尾，原始状态可能显示stopped_by_user，具体原因以CPU_BUDGET_STOP.json为准。']
  for heading,table in [('标签置换阴性对照（独立目标；不是真实来源准确率）',nulls),('材料固定PRED结果（与来源OOF分开）',materials)]:
   complete=[r for r in table if r.get('mean_accuracy')is not None]
   if complete:
    report+=['','## '+heading,'','| 方法族 / 变体 | 完整重复/拟合 | 唯一记录/物理ID | 平均准确率 |','|---|---:|---:|---:|']
    for r in complete:report.append(f"| {r['family']} / {r['variant']} | {r['complete_repeats']} | {r['unique_records']}/{r['unique_physical_ids']} | {100*r['mean_accuracy']:.2f}% |")
  atomic_text(out/'中文简报.md','\n'.join(report)+'\n');atomic_text(out/'summary_zh.md','\n'.join(report)+'\n')
  base.update(updated_utc=now(),snapshot_ready=True,complete_source_repeats=sum(r.get('complete_repeats',0)for r in source),complete_null_repeats=sum(r.get('complete_repeats',0)for r in nulls),material_fixed_role_fits=sum(r.get('complete_repeats',0)for r in materials),pending_repeat_blocks=sum(r['state']!='complete'for r in blocks),source_unique_population_records=len(core)if core else None,summary_wall_seconds=time.monotonic()-start,figure_state=figure['state'],invalid_result_jobs=len(issues));atomic_json(out/'progress.json',base);atomic_json(out/'summary_manifest.json',{'schema':VERSION,'snapshot_ready':True,'generated_utc':now(),'conditional_CI_only':True,'new_model_fits':0,'result_glob':'jobs/*/result.json','bootstrap_draws':args.bootstrap_draws,'bootstrap_seed':args.seed,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'files':[p.name for p in out.iterdir()if p.is_file()and not p.name.startswith('.')],'figure_files':figure.get('files',[])});print(json.dumps(base,ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
