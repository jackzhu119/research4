from pathlib import Path
import json, hashlib, itertools, gzip, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

ROOT=Path(__file__).resolve().parents[3]
parser=argparse.ArgumentParser(description='Recompute revision summaries from immutable saved predictions and masks; no training.')
parser.add_argument('--evidence-dir',type=Path,default=ROOT/'outputs/revision_20261116/evidence')
parser.add_argument('--output-dir',type=Path,default=ROOT/'paper/revision_20261116_r2/analysis')
args=parser.parse_args()
BASE=args.evidence_dir/'5594d9c3/reports/focused_followup_20261116'
V4=ROOT/'paper/revision_20261115/Supplementary_Data_S1/derived_checks/verified_v4_sources/runs'
OUT=args.output_dir;OUT.mkdir(parents=True,exist_ok=True)
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
core=read(ROOT/'paper/current/manifest.json')
for row in core['files']:assert sha(ROOT/'paper/current'/row['path'])==row['sha256'],row['path']
books=read(ROOT/'paper/current/inputs/active_workbooks.json')['workbooks']
for row in books:assert sha(ROOT/row['path'])==row['sha256']

v4=pd.concat([pd.read_csv(p) for p in V4.glob('*/validation_predictions.csv.gz')],ignore_index=True)
primary=v4[v4.is_primary_k.eq(1)].copy()
bridge=pd.read_csv(BASE/'e1/oof_predictions.csv.gz');bridge['family']='fullalpha_snv_svm_bridge';bridge['k']=612
bridge['is_correct']=bridge.y_true.eq(bridge.prediction).astype(int)
assert len(bridge)==900 and bridge.record_id.nunique()==300
allpred=pd.concat([primary,bridge],ignore_index=True)
def metrics(frame):
    return dict(accuracy=accuracy_score(frame.y_true,frame.prediction),
      balanced_accuracy=balanced_accuracy_score(frame.y_true,frame.prediction),
      macro_f1=f1_score(frame.y_true,frame.prediction,average='macro',zero_division=0))
rows=[];foldrows=[]
for family,group in allpred.groupby('family'):
    assert len(group)==900 and group.record_id.nunique()==300
    repeats=[]
    for seed,part in group.groupby('repeat_seed'):
        assert len(part)==300 and part.record_id.nunique()==300
        repeats.append(dict(repeat_seed=int(seed),**metrics(part)))
    for (seed,fold),part in group.groupby(['repeat_seed','outer_fold']):
        assert len(part)==60
        foldrows.append(dict(family=family,repeat_seed=int(seed),outer_fold=int(fold),**metrics(part)))
    rows.append(dict(family=family,n_physical=300,predictions=900,repeated_partitions=3,
      mean_K=float(group.groupby(['repeat_seed','outer_fold']).k.first().mean()),
      **{key:float(np.mean([x[key] for x in repeats])) for key in ['accuracy','balanced_accuracy','macro_f1']},
      repetition_accuracies=[x['accuracy'] for x in repeats]))
models=pd.DataFrame(rows);models.to_csv(OUT/'primary_models.csv',index=False)
pd.DataFrame(foldrows).to_csv(OUT/'primary_fold_metrics.csv',index=False)
allpred.to_csv(OUT/'primary_predictions.csv.gz',index=False)

frozen=read(BASE/'e1/frozen_sources.json')
assert len(frozen['contexts'])==15
for context in frozen['contexts']:
    seed=context['repeat_seed'];fold=context['outer_fold']
    old=read(V4/f'matched_r{seed}_f{fold}'/'job_manifest.json')
    for key in ['source_record_ids','outer_validation_record_ids','inner_partitions']:
        assert old[key]==context[key],(seed,fold,key)
    assert not set(old['source_record_ids'])&set(old['outer_validation_record_ids'])
gamma_checks=0
for file in sorted((BASE/'e1/folds').glob('*_inner.json')):
    data=read(file);assert len(data)==24
    for inner in [0,1]:
        part=[r for r in data if r['inner_fold']==inner]
        assert len({(r['config']['C'],r['config']['gamma']) for r in part})==12
        assert len({round(r['numeric_gamma'],14) for r in part})==3
        for r in part:
            factor={'scale':1.,'scale_x0.1':.1,'scale_x10':10.}[r['config']['gamma']]
            assert np.isclose(r['numeric_gamma'],factor/612,rtol=1e-12)
            gamma_checks+=1
assert gamma_checks==360
assert int(pd.read_csv(BASE/'e1/outer_parameters.csv').fits.sum())==375

# Conditional paired resampling: one unit is a physical specimen, retaining
# all three repeated predictions and preserving each recorded source count.
# This is a retrospective descriptive analysis, not a new confirmation cohort.
ids=sorted(bridge.record_id.unique());seeds=sorted(bridge.repeat_seed.unique())
reference=allpred[allpred.family.eq('alpha_snv20')].pivot(index='record_id',columns='repeat_seed',values='is_correct').loc[ids,seeds].to_numpy()
labels=allpred[allpred.family.eq('alpha_snv20')].drop_duplicates('record_id').set_index('record_id').loc[ids].y_true.to_numpy()
rng=np.random.default_rng(20261116);weights=np.zeros((10000,300),dtype=np.int16)
for label in sorted(set(labels)):
    ix=np.flatnonzero(labels==label);weights[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=10000)
effects=[]
for family in ['fullalpha_snv_svm_bridge','fullalpha_shrinkLDA','alpha_raw20','alpha_snv100','concat_both_snv20']:
    other=allpred[allpred.family.eq(family)].pivot(index='record_id',columns='repeat_seed',values='is_correct').loc[ids,seeds].to_numpy()
    per_id=(reference-other).mean(1);draws=weights@per_id/300*100
    differences=reference.mean(0)-other.mean(0)
    effects.append(dict(reference='alpha_snv20',comparator=family,difference_pp=float(per_id.mean()*100),
       ci_low_pp=float(np.quantile(draws,.025)),ci_high_pp=float(np.quantile(draws,.975)),bootstrap_draws=10000,
       seed=20261116,unit='source-stratified physical specimen; all 3 predictions retained',
       repeat_differences_pp=(100*differences).tolist(),interpretation='Conditional on saved fits; retrospective, unadjusted; not between-batch or retraining uncertainty'))
pd.DataFrame(effects).to_csv(OUT/'paired_primary_effects.csv',index=False)

features=pd.concat([pd.read_csv(p) for p in V4.glob('*/selected_features.csv.gz')],ignore_index=True)
features['coordinate']=np.where(features.family.str.startswith(('alpha_','n_')),features.within_modality_index,features.global_index).astype(int)
def stability(sets,p):
    z=np.zeros((len(sets),p))
    for i,s in enumerate(sets):z[i,list(s)]=1
    k=z.sum(1);phat=z.mean(0);den=(k.mean()/p)*(1-k.mean()/p)
    phi=1-np.var(z,axis=0,ddof=1).mean()/den
    j=[len(a&b)/len(a|b) for a,b in itertools.combinations(sets,2)]
    return dict(mask_count=len(sets),universe_p=p,mean_K=float(k.mean()),Jaccard=float(np.mean(j)),Nogueira_Phi=float(phi)),phat
stable=[];probs=[]
for family in sorted(features.family.unique()):
    p=612 if family.startswith(('alpha_','n_')) else 1224
    for budget in ['primary',62]:
        sub=features[(features.family==family)&(features.is_primary_k.eq(1) if budget=='primary' else features.k.eq(budget))]
        sets=[set(x.coordinate) for _,x in sub.groupby(['repeat_seed','outer_fold'])]
        result,phat=stability(sets,p);stable.append(dict(family=family,budget=budget,**result))
        if family=='alpha_snv20' and budget=='primary':probs=[dict(coordinate=i,selection_probability=float(v)) for i,v in enumerate(phat)]
pd.DataFrame(stable).to_csv(OUT/'v4_stability.csv',index=False);pd.DataFrame(probs).to_csv(OUT/'v4_primary_selection_probability.csv',index=False)
cross=[]
for key,g in features[(features.family=='alpha_snv20')&features.k.eq(62)].groupby(['repeat_seed','outer_fold']):
    h=features[(features.family=='concat_both_snv20')&features.k.eq(62)&features.repeat_seed.eq(key[0])&features.outer_fold.eq(key[1])]
    assert len(h)==62 and h.modality.eq('alpha').all()
    a,b=set(g.within_modality_index),set(h.within_modality_index)
    cross.append(dict(repeat_seed=int(key[0]),outer_fold=int(key[1]),intersection=len(a&b),jaccard=len(a&b)/len(a|b)))
pd.DataFrame(cross).to_csv(OUT/'cross_pipeline_K62.csv',index=False)

fixed=[];saved=read(BASE/'e2/fixed_train_summary.json')['summaries']
for context in [0,1,2]:
    data=[read(p) for p in sorted((BASE/'e2/mc').glob(f'context{context}_*/result.json'))]
    assert len(data)==30 and len({r['outer_train_ids_sha256'] for r in data})==1 and len({r['MC_seed'] for r in data})==30
    for method in ['alpha_only/no_gap','alpha_only/fixed_gap4','true_n_context/no_gap','true_n_context/fixed_gap4']:
        sets=[set(r['masks_local_indices'][method]) for r in data]
        value,_=stability(sets,612)
        matched=[v for v in saved if v['method']==method+'/train_context'+str(context) and v['space']=='alpha' and 'grid_group_width' not in v]
        assert len(matched)==1
        assert np.isclose(value['Jaccard'],matched[0]['exact_Jaccard']['mean'])
        assert np.isclose(value['Nogueira_Phi'],matched[0]['Nogueira_Phi'])
        fixed.append(dict(context=context,method=method,variation='MC seed variation with training specimens fixed',**value))
pd.DataFrame(fixed).to_csv(OUT/'fixed_train_stability.csv',index=False)

crossed=[]
for folder in sorted((BASE/'e3').glob('context[0-9]*')):
    design=read(folder/'locked_design.json');p=design['parameters_actual_numeric_gamma']
    frame=pd.read_csv(folder/'specimen_predictions.csv');assert len(frame)==60
    assert frame.AA.eq(frame.archived_AA).all() and frame.BB.eq(frame.archived_BB).all()
    values={cell:accuracy_score(frame.true_label,frame[cell]) for cell in ['AA','AB','BA','BB']}
    assert frame.AA.eq(frame.AB).all() and frame.BA.eq(frame.BB).all()
    assert p['A']['C']==p['B']['C']==100
    assert np.isclose(p['A']['gamma'],p['B']['gamma'],rtol=1e-13)
    crossed.append(dict(context=design['context_index'],C_A=p['A']['C'],C_B=p['B']['C'],gamma_A=p['A']['gamma'],gamma_B=p['B']['gamma'],
      **values,mask_effect_pp=100*(values['BA']-values['AA']),parameter_contrast='same C and numerically equal gamma; not separately identifiable'))
assert len(crossed)==15
pd.DataFrame(crossed).to_csv(OUT/'crossed_masks.csv',index=False)

lowo=[]
for task in ['binary_kinds','four_kinds']:
    base=ROOT/'paper/current/results/kind_lowo_20261115'/task
    frame=pd.read_csv(base/'oof_predictions.csv')
    # Locate names without altering saved labels or physical specimen IDs.
    truth=next(c for c in ['y_true','true_label','true_kind'] if c in frame)
    pred=next(c for c in ['prediction','predicted_label','predicted_kind'] if c in frame)
    original=read(base/'summary.json')
    row=dict(task=task,n_records=len(frame),n_correct=int(frame[truth].eq(frame[pred]).sum()),accuracy=accuracy_score(frame[truth],frame[pred]),
      balanced_accuracy=balanced_accuracy_score(frame[truth],frame[pred]),macro_f1=f1_score(frame[truth],frame[pred],average='macro',zero_division=0))
    lowo.append(row)
pd.DataFrame(lowo).to_csv(OUT/'lowo_metrics.csv',index=False)
bug=read(BASE/'e4/benchmarks/context0_baseline_svm/inner_candidate_means.json')
for c in [.1,1.,10.,100.]:
    ix=[i for i,r in enumerate(bug['parameters']) if r['C']==c]
    assert len(ix)==3 and np.allclose(np.asarray(bug['mean_metrics'])[ix],np.asarray(bug['mean_metrics'])[ix[0]],rtol=0,atol=0)
source=(args.evidence_dir/'93a607c8/source/study_worker.py').read_text()
assert "key=(p['preprocessing'],p['K'],p['components'],p['gap'],p['drop'])" in source
assert "numeric_gamma=gamma0*p['gamma_multiplier']" in source
assert "zt,zq,info=transformcache[key]" in source

receipt=dict(status='PASS',core_allowlisted_files_verified=len(core['files']),raw_workbooks_verified=len(books),
  e1_prediction_rows=len(bridge),e1_independent_specimens=300,e1_exact_outer_and_inner_id_sets=15,e1_inner_candidate_gamma_values_verified=gamma_checks,
  primary_models=rows,primary_paired_effects=effects,v4_primary_stability=next(v for v in stable if v['family']=='alpha_snv20' and v['budget']=='primary'),
  cross_pipeline_K62_Jaccard=float(pd.DataFrame(cross).jaccard.mean()),fixed_train_rank_tasks=90,fixed_train_stability=fixed,
  E3_crossed_contexts=15,E3_mask_difference_pp=float(np.mean([r['mask_effect_pp'] for r in crossed])),E3_parameters_informative=False,
  material_LOWO=lowo,gamma_defect_confirmed_in_expanded_worker=True,gamma_defect_absent_from_verified_E1_dispatch=True,
  scientific_models_trained_in_this_revision=0,bootstrap_policy='10,000 post hoc source-stratified physical-ID draws, seed20261116; saved predictions fixed; conditional and unadjusted')
(OUT/'EVIDENCE_AUDIT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ['primary_models','primary_paired_effects','fixed_train_stability']},ensure_ascii=False,indent=2))
print('E1 paired effect:',effects[0])
