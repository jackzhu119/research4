"""Zero-fit replay of the fixed G1 real-data C, D and H diagnostic."""
import collections,hashlib,io,json,sys
import numpy as np
import joblib
from audit_common import Audit,ROOT,OUT,read,sha,utc,verify_dataset,write_new,metrics
sys.path.insert(0,str(ROOT/'src'))
from herb_spectra.leakage_guard import source_fingerprint

B=ROOT/'reports/public-thz-20261031-v1/adversarial/prepared'
FIELDS=('accuracy','macro_f1','balanced_accuracy')

def npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def compare_metrics(a,found,expected,why):
 for f in FIELDS:a.close(found[f],expected[f],why+' '+f)
 a.equal(found['confusion_matrix'],expected['confusion_matrix'],why+' confusion')
def source_x(x):
 sd=x.std(1,ddof=1,keepdims=True);sd[sd<=np.finfo(float).eps]=1
 return (x-x.mean(1,keepdims=True))/sd
def verify_bundle(a,bundle,raw,y,ids,config):
 a.equal(bundle.metadata['source_ids'],ids,'bundle source IDs');a.equal(bundle.metadata['source_y'],y,'bundle source labels')
 a.check(bundle.representation=='alpha' and bundle.preprocessing=='snv','locked alpha SNV bundle')
 a.check(bundle._leakage_fit_provenance['data_fingerprint']==source_fingerprint(raw,y),'bundle actual raw values+y identity')
 prep=source_x(raw);scale=bundle.scalers['alpha']
 a.close(scale.mean_,prep.mean(0),'source scaler mean');a.close(scale.var_,prep.var(0),'source scaler variance')
 z=bundle.matrix({'alpha':raw,'n':raw},None) if bundle.selector is None else (prep-scale.mean_)/scale.scale_
 a.close(bundle.preprocessors['alpha'].transform(raw),prep,'independent row SNV')
 s=bundle.selector;a.check(s.n_runs==10 and s.n_components==1 and s.random_state==42 and s.sample_fraction==.8,'fixed MC params')
 a.check(s.coefficients_.shape==(10,2,65) and s.sample_unit_=='rows_assumed_independent','OVR2 coefficient shape and honest assumed grain')
 mean=s.coefficients_.mean(0);sd=s.coefficients_.std(0,ddof=1);floor=max(1.,float(sd.max()))*1e-12
 stability=mean/np.maximum(sd,floor);importance=np.linalg.norm(abs(stability),axis=0)
 a.close(s.signed_stability_,stability,'MC stability arithmetic');a.close(s.importances_,importance,'MC importance')
 a.equal(s.ranking_,np.argsort(-importance,kind='stable'),'stable MC source ranking')
 selected=np.sort(s.ranking_[:13]);a.equal(bundle.selected(.2),selected,'13 sorted source-only selected columns')
 return z[:,selected]

def main():
 a=Audit();data,_=verify_dataset(a);manifest=read(B/'frozen_manifest.json')
 a.check(sha(B/'frozen_manifest.json')=='fa9b3581b2e59440fcb53db62fd3f6ba1620f766a9cd65bbeaaf3e1b16e6f165','root approved diagnostic manifest')
 a.check(sha(B/'results.json')=='512b100d7561a181fd3c36716e06a41ea7f54b6d095eac37f7fdc372b6d68c72','result frozen before replay')
 a.check(sha(B/'diagnostic_prediction_lock.json')=='2314015bdb491400d6a62cefa6e22f76dd2706446d93b0d5570df9e0ecbc19b8','prediction lock frozen before replay')
 for f,h in manifest['source_sha256'].items():a.check(sha(ROOT/f)==h,'approved code '+f)
 for f,h in manifest['files_sha256'].items():a.check(sha(B/f)==h,'immutable inputs '+f)
 src=npz(B/'source.npz');query=npz(B/'query.npz');group=data['export_groups'];source=group!='G1';q=~source
 a.check(set(query)=={'record_ids','X_alpha','X_n','axis'},'query no truth or metadata features')
 for k in ['record_ids','X_alpha','X_n']:a.equal(src[k],data[k][source],'independent source '+k);a.equal(query[k],data[k][q],'independent query '+k)
 a.equal(src['source_labels'],data['y'][source],'source true labels');a.equal(src['source_groups'],group[source],'source fixed file groups')
 lock=read(B/'diagnostic_prediction_lock.json');result=read(B/'results.json');runs=[lock['observed'],read(B/'fits/permuted_query_truth_trap/predictions_before_scoring.json')]+lock['nulls']
 a.check(len(runs)==12 and len(lock['nulls'])==10,'all twelve prelocked fits incl ten source nulls')
 # Verify ALL model bytes and source-selection locks before any model deserialization.
 for run in runs:
  path=B/run['model_path'];a.check(sha(path)==run['model_sha256'],'all model bytes before loading')
  a.check(sha(path.parent/'source_selection_lock.json')==run['source_selection_lock_sha256'],'all inner locks before loading')
 actual=collections.Counter();recomputed=[];replayed={};all_source_checks=0
 for number,run in enumerate(runs):
  d=B/run['model_path'];directory=d.parent;y=src['source_labels'].copy()
  if number>=2:
   rng=np.random.default_rng(run['permutation_seed'])
   for g in sorted(set(src['source_groups'])):
    ix=np.flatnonzero(src['source_groups']==g);y[ix]=rng.permutation(y[ix])
  a.equal(run['source_labels'],y,'source-only deterministic group-label permutation')
  a.check(not run['outer_truth_read'] and not run['outer_metrics_present'],'saved predictions contain no outer metric')
  a.check(any(e['event']=='TRUTH_ACCESS_REJECTED' and not e['write'] for e in run['truth_access_events']),'actual protected truth read rejected')
  selection=read(directory/'source_selection_lock.json');inner=selection['inner_candidates'];a.check(len(inner)==2,'fixed two source candidates')
  bundles={};z_by_split={};source_y_by_split={};source_ids_by_split={}
  for trial in inner:
   calculated=[]
   for case in trial['cases']:
    held=case['held_source_group'];s=src['source_groups']!=held;v=~s
    a.equal(case['source_ids'],src['record_ids'][s],'inner source excludes held group');a.equal(case['validation_ids'],src['record_ids'][v],'inner validation members')
    m=metrics(y[v],case['prediction']);compare_metrics(a,case['metrics'],m,'inner metric arithmetic');calculated.append(m)
    ref=case['transform'];cache=directory/ref['path'];a.check(sha(cache)==ref['sha256'],'source cache immutable')
    if held not in bundles:
     envelope=joblib.load(cache);blob=envelope['payload_bytes'];a.check(hashlib.sha256(blob).hexdigest()==envelope['payload_sha256'],'cache payload SHA before loading')
     bundle=joblib.load(io.BytesIO(blob));bundles[held]=bundle
     a.check(envelope['identity']==ref['cache_identity']==bundle.metadata['cache_identity'],'source cache identity')
     z=verify_bundle(a,bundle,src['X_alpha'][s],y[s],src['record_ids'][s],trial['config'])
     split='adversarial.outer_G1.inner_'+held;z_by_split[split]=z;source_y_by_split[split]=y[s];source_ids_by_split[split]=src['record_ids'][s]
    a.equal(case['retained_indices'],bundles[held].selected(.2),'inner selected columns')
   for f in FIELDS:a.close(trial['mean_inner_metrics'][f],np.mean([m[f] for m in calculated]),'inner mean '+f)
  winner=sorted(inner,key=lambda c:tuple(-c['mean_inner_metrics'][f] for f in ['macro_f1','balanced_accuracy','accuracy'])+(c['config']['id'],))[0]
  a.check(winner['config']==selection['config']==run['chosen_config'],'winner selected only from current source inner records')
  objects=joblib.load(d);bundle,est=objects['bundle'],objects['estimator'];state=joblib.hash((bundle,est));a.check(state==run['training_state_hash'],'full model weight state')
  z=verify_bundle(a,bundle,src['X_alpha'],y,src['record_ids'],run['chosen_config']);split='adversarial.outer_G1.final';z_by_split[split]=z;source_y_by_split[split]=y;source_ids_by_split[split]=src['record_ids']
  a.check(est.metadata['source_z_fingerprint']==source_fingerprint(z,y),'final SVC actual source feature/label identity')
  a.equal(est.estimator.support_vectors_,z[est.estimator.support_],'final SVC support vectors current source')
  qz=bundle.matrix({'alpha':query['X_alpha'],'n':query['X_n']},.2);pred=est.estimator.predict(qz);margin=est.estimator.decision_function(qz);scores=np.column_stack([-margin,margin])
  a.equal(pred,run['predictions'],'native SVC exact prediction replay');a.close(scores,run['scores'],'score replay')
  a.equal(est.predict(qz),pred,'wrapper native vote');a.close(est.scores(qz),scores,'binary class-margin order')
  a.close(np.vstack([est.scores(qz[i:i+1]) for i in range(7)]),scores,'tail-safe single-record prediction');a.close(est.scores(qz[::-1])[::-1],scores,'reversed query order')
  a.check(joblib.hash((bundle,est))==state,'replay does not update model')
  if number==0:
   changed=query['X_alpha'].copy();changed[0,0]+=.01
   zm=bundle.matrix({'alpha':changed,'n':query['X_n']},.2)
   a.close(est.scores(zm)[1:],scores[1:],'one query perturbation leaves other six scores unchanged');a.equal(est.predict(zm)[1:],pred[1:],'other six predictions unchanged')
   a.check(joblib.hash((bundle,est))==state,'postfit D model state unchanged')
  replayed[run['run']]={'state':state,'config':run['chosen_config'],'indices':bundle.selected(.2).tolist(),'prediction':pred.tolist(),'scores':scores.tolist()}
  records=[json.loads(line) for line in (directory/'fit_scope_events.jsonl').read_text().splitlines()]
  for record in records:
   ids=record['fit_sample_ids'];allowed=source_ids_by_split[record['split_id']]
   a.check(set(ids)<=set(allowed) and not set(ids)&set(query['record_ids']),'every actual fit source role')
   if record['event']=='fit_start':
    actual[record['component']]+=1
    if record['component']=='SVC':
     a.equal(ids,allowed,'each classifier full current source IDs')
     a.check(record['data_fingerprint']==source_fingerprint(z_by_split[record['split_id']],source_y_by_split[record['split_id']]),'SVC logged actual matrix and labels')
    all_source_checks+=1
  recomputed.append(metrics(data['y'][q],pred))
 a.check(replayed['observed']==replayed['permuted_query_truth_trap'],'actual label trap retrains produce identical learned state/selection/predictions')
 compare_metrics(a,result['observed_metrics'],recomputed[0],'observed outer arithmetic')
 majority=sorted(['0','1'],key=lambda c:(-int(np.sum(src['source_labels']==c)),c))[0]
 compare_metrics(a,result['majority_metrics'],metrics(data['y'][q],np.repeat(majority,7)),'source-selected actual G1 majority')
 for f in FIELDS:
  values=[m[f] for m in recomputed[2:]];found=result['null_summary'][f]
  a.close(found['null_values'],values,'all null metrics '+f);a.close(found['null_mean'],np.mean(values),'null mean '+f)
  a.close(found['empirical_pvalue_plus_one'],(1+sum(v>=recomputed[0][f] for v in values))/11,'null plus-one p '+f)
 a.check(actual['SVC']==132 and actual['MC-UVE.PLSRegression.fit']==1440,'actual classifier and internal OVR PLS fit counts')
 output={'completed_at_utc':utc(),'status':'ISOLATION_AND_REPLAY_PASS_WITH_DECLARED_PLS_COST_DISCREPANCY','checks':a.checks,'numeric_values':a.numeric_values,'maximum_difference':a.maximum_difference,'new_audit_fits':0,'actual_classifier_fits':actual['SVC'],'actual_internal_PLS_fits':actual['MC-UVE.PLSRegression.fit'],'declared_internal_PLS_fits':720,'cost_discrepancy':'Frozen plan counted one PLS per MC repeat; binary one-hot _coefficients fits two PLS1 models. Actual1440, not720; frozen records preserved. This is a budget reporting failure, not evidence of label leakage.','actual_source_fit_starts_reviewed':all_source_checks,'observed':recomputed[0],'majority':result['majority_metrics'],'nulls':recomputed[2:],'query_label_trap_state_equal':True,'single_query_postfit_other_six_equal':True,'full_D_source_retraining':False,'scientific_status':'EXPLORATORY_G1_ONLY_UNKNOWN_PLANT_ANCESTRY_NOT_CONFIRMATION','selection_or_main_grid_changed_from_diagnostic':False,'limitations':['Only one outer export group, two MC-SVM candidates and seed42; cannot certify all main stages/families','Inner cached transforms and prediction metrics replayed; diagnostic does not save each inner SVC weight object, so inner predictions are not independently rerun through those absent objects','Named Python I/O hooks and actual read refusals are not an OS sandbox','Ten nulls give p-value resolution1/11 and do not establish generalization','All losses retained; G1 observed3/7 is below actual source-majority4/7 and null mean3.1/7'], 'input_hashes':{'prediction_lock':sha(B/'diagnostic_prediction_lock.json'),'results':sha(B/'results.json'),'manifest':sha(B/'frozen_manifest.json')},'script_sha256':sha(__file__)}
 write_new(OUT/'adversarial_review.json',output);print(json.dumps({k:output[k] for k in ['status','checks','new_audit_fits','actual_classifier_fits','actual_internal_PLS_fits']}))

if __name__=='__main__':main()
