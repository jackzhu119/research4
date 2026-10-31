"""Read-only post-score majority context and outer learned-transform checks."""
import collections,hashlib,io,json,sys
import joblib
import numpy as np
from audit_common import Audit,ROOT,OUT,read,sha,utc,write_new,metrics
sys.path.insert(0,str(ROOT/'src'))
from herb_spectra.leakage_guard import source_fingerprint

def outer_transforms(a,fruit):
 b=ROOT/('outputs/public-thz-fruit-regression-20261031-v1' if fruit else 'reports/public-thz-20261031-v1/ginseng-benchmark')
 audit=read(OUT/('fruit_evaluation_review.json' if fruit else 'evaluation_review.json'));a.check(audit['scoring_manifest_sha256']==sha(b/'scoring_manifest.json'),'completed independent scoring review')
 prep=read(b/'preparation.json');contexts={c['split_id']:c for c in prep['contexts']};refs={r['transform']['path']:r['transform'] for r in read(b/'outer_model_records.json')};mc={}
 for name,ref in refs.items():
  a.check(sha(b/name)==ref['sha256'],'outer learned cache bytes');value=joblib.load(b/name);blob=value['payload_bytes'];a.check(hashlib.sha256(blob).hexdigest()==value['payload_sha256'],'payload hash before loading');t=joblib.load(io.BytesIO(blob));c=contexts[ref['cache_identity']['split_id']]
  a.check(c['stage']=='outer','only outer source objects reviewed')
  with np.load(b/c['source_pack'],allow_pickle=False) as z:source={k:z[k] for k in z.files}
  a.equal(t.metadata['source_ids'],source['record_ids'],'exact actual source rows')
  if fruit:
   X=source['X'];y=source['source_targets'];processed=X.copy()
   if t.preprocessing=='snv':
    sd=X.std(1,keepdims=True);sd[sd<=np.finfo(float).eps]=1;processed=(X-X.mean(1,keepdims=True))/sd
   a.close(t.x_scaler.mean_,processed.mean(0),'outer source X mean');a.close(t.x_scaler.var_,processed.var(0),'outer source X variance')
   a.close(t.y_scaler.mean_,[y.mean()],'outer source y mean');a.close(t.y_scaler.var_,[y.var()],'outer source y variance');Z=t.x_scaler.transform(processed);target=t.y_scaler.transform(y[:,None]).ravel()
  else:
   y=source['source_labels'];mods=['alpha','n'] if t.representation=='paired_alpha_n' else [t.representation];X=np.column_stack([source['X_'+m] for m in mods]);parts=[]
   for m in mods:
    pre=t.preprocessors[m];values=source['X_'+m]
    if t.preprocessing=='msc':a.close(pre.reference_,values.mean(0),'outer source MSC reference')
    processed=pre.transform(values);scaler=t.scalers[m];a.close(scaler.mean_,processed.mean(0),'outer source scaler mean');a.close(scaler.var_,processed.var(0),'outer source scaler variance');parts.append(scaler.transform(processed))
   Z=np.column_stack(parts);target=y
  a.check(source_fingerprint(X,y)==t._leakage_fit_provenance['data_fingerprint']==ref['cache_identity']['data_fingerprint'],'actual raw outer source X/y cache lineage')
  if t.selector is not None:
   s=t.selector;sd=s.coefficients_.std(0,ddof=1);st=s.coefficients_.mean(0)/np.maximum(sd,max(1.,float(sd.max()))*1e-12);importance=np.linalg.norm(abs(st),axis=0);a.close(s.signed_stability_,st,'outer MC stability arithmetic');a.equal(s.ranking_,np.argsort(-importance,kind='stable'),'outer source-only MC ranking');mc[(c['split_id'],source_fingerprint(Z,target))]=(Z,target,source['record_ids'])
 active=None;sub=None;pls=0
 for line in (b/'fit_scope_events.jsonl').open():
  e=json.loads(line);c=contexts[e['split_id']]
  if c['stage']!='outer' or e['event']!='fit_start':continue
  component=e['component']
  if component==('fruit.MC_UVE.source_record_ranking' if fruit else 'MC-UVE.full_source_ranking'):active=mc[(e['split_id'],e['data_fingerprint'])]
  elif component=='MC-UVE.coefficient_standardization':
   Z,y,ids=active;ix=np.array([list(ids).index(r) for r in e['fit_sample_ids']]);X=Z[ix];target=y[ix,None] if fruit else np.eye(2)[np.searchsorted(['0','1'],y[ix])]
   a.check(e['data_fingerprint']==source_fingerprint(X,target),'outer actual MC subset source values/targets');sd=X.std(0);sd[sd<=np.finfo(float).eps]=1;sub=((X-X.mean(0))/sd,target)
  elif component=='MC-UVE.PLSRegression.fit':
   X,target=sub;options=[source_fingerprint(X,(t-t.mean())/t.std()) for t in target.T];a.check(e['data_fingerprint'] in options,'outer actual normalized internal PLS source fingerprint');pls+=1
 return {'cached_source_transforms':len(refs),'MC_source_rankings':len(mc),'actual_outer_internal_PLS_fits_rechecked':pls,'outer_evaluation_review_sha256':sha(OUT/('fruit_evaluation_review.json' if fruit else 'evaluation_review.json'))}

def main():
 a=Audit();g=outer_transforms(a,False);f=outer_transforms(a,True);b=ROOT/'reports/public-thz-20261031-v1/ginseng-benchmark';contexts=[c for c in read(b/'preparation.json')['contexts'] if c['stage']=='outer'];folds=[];ys=[];ps=[]
 for c in contexts:
  with np.load(b/c['source_pack'],allow_pickle=False) as z:sourcey=z['source_labels'];sourceids=z['record_ids']
  counts={cls:int(np.sum(sourcey==cls)) for cls in ['0','1']};majority=sorted(counts,key=lambda cls:(-counts[cls],cls))[0]
  # Every truth CSV was verified before semantic parsing by evaluation_review.
  import csv
  with (b/c['truth']).open(encoding='utf-8-sig') as stream:truth={r['record_id']:r['label'] for r in csv.DictReader(stream)}
  y=[truth[rid] for rid in c['query_ids']];prediction=[majority]*len(y);ys.extend(y);ps.extend(prediction)
  a.check(len(sourceids)==35 and len(y)==7,'source-only35 to query7 majority context');folds.append({'held_export_group':c['outer_group'],'source_class_counts':counts,'source_majority_class':majority,'query_records':7,'metrics':metrics(y,prediction)})
 result={'completed_at_utc':utc(),'status':'POST_SCORE_DESCRIPTIVE_BASELINE_AND_OUTER_TRANSFORMS_VERIFIED','new_audit_fits':0,'checks':a.checks,'numeric_values':a.numeric_values,'maximum_difference':a.maximum_difference,'ginseng_outer_transforms':g,'fruit_outer_transforms':f,'ginseng_source_majority':{'status':'AFTER_SCORE_DESCRIPTIVE_CONTEXT_ONLY_NOT_A_PREDECLARED_MAIN_ROLE','rule':'Each fixed35-observation outer source chooses most frequent label; ties choose lexicographically smaller class. No query label is used in that choice.','pooled_metrics':metrics(ys,ps),'folds':folds,'seeds':'No stochastic fitting; same fixed baseline for all seeds','observations':42,'source_classification_trials_added':0,'used_to_change_main_candidate_grid':False,'fresh_independent_confirmation':False},'scientific_limitations':['Source-majority is added after score as descriptive context and cannot be claimed a predeclared comparison','The n-shrinkLDA family highest on external development observations is not automatically a source-selected deployment winner','Known upstream root ancestry remains unknown; six export groups do not establish true batches'],'script_sha256':sha(__file__)}
 write_new(OUT/'postscore_checks.json',result);print(json.dumps({'checks':a.checks,'new_audit_fits':0,'ginseng_majority':result['ginseng_source_majority']['pooled_metrics']}))

if __name__=='__main__':main()
