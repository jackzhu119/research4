from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request,hashlib,json,subprocess
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'data/active_workbooks.json').exists()); OUT=Path(__file__).resolve().parent.parent; A=Path(__file__).resolve().parent; F=ROOT/'paper/revision_20261115/figures'
UPLOAD=OUT/'uploaded_original/research4_Applied_Spectroscopy/evidence/original_snapshots'
PIN='b051733989161517a68c4a60cce75766880c95f5'
BASE='data/research/evidence_validation_20261108_v4'
receipt=[]
rels=[f'{BASE}/snapshots/00330/{n}' for n in ('metrics.csv','summary.json')]
rels += [f'{BASE}/runs/matched_r{s}_f{f}/{n}' for s in (42,20261031,20261108) for f in range(5) for n in ('validation_predictions.csv.gz','selected_features.csv.gz','summary.json','job_manifest.json')]
def verify(rel):
 local=UPLOAD/rel
 target=A/'verified_v4_sources'/rel.removeprefix(BASE+'/')
 target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists(): data=target.read_bytes()
 else:
  data=urllib.request.urlopen(f'https://raw.githubusercontent.com/jackzhu119/research4/{PIN}/{rel}',timeout=35).read()
  target.write_bytes(data)
 if local.exists(): assert data==local.read_bytes(),f'UPLOAD DIFF {rel}'
 git_data=subprocess.check_output(['git','-C',str(ROOT),'show',f'{PIN}:{rel}'])
 assert data==git_data,f'GIT DIFF {rel}'
 return {'source_commit':PIN,'source_path':rel,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'uploaded_bytes_equal':True if local.exists() else None,'pinned_git_bytes_equal':True,'verified_copy':str(target.relative_to(OUT))}
with ThreadPoolExecutor(max_workers=8) as pool: receipt=list(pool.map(verify,rels))
# The accepted current view is the controlling source. Uploaded calculations are reusable only after byte equality above.
for name in ('metrics.csv','summary.json'):
 assert (ROOT/f'paper/current/results/v4_330/{name}').read_bytes()==(A/f'verified_v4_sources/snapshots/00330/{name}').read_bytes()
P=pd.concat([pd.read_csv(A/f'verified_v4_sources/runs/matched_r{s}_f{f}/validation_predictions.csv.gz') for s in (42,20261031,20261108) for f in range(5)],ignore_index=True)
S=pd.concat([pd.read_csv(A/f'verified_v4_sources/runs/matched_r{s}_f{f}/selected_features.csv.gz') for s in (42,20261031,20261108) for f in range(5)],ignore_index=True)
M=pd.read_csv(ROOT/'paper/current/results/v4_330/metrics.csv')
rows=[]
for (family,k),g in P.groupby(['family','k']):
 vals=[]
 for seed,v in g.groupby('repeat_seed'):
  assert len(v)==300 and v.record_id.nunique()==300
  vals.append((accuracy_score(v.y_true,v.prediction), f1_score(v.y_true,v.prediction,average='macro'),balanced_accuracy_score(v.y_true,v.prediction)))
 mean=np.mean(vals,axis=0)
 check=M[(M.method==family)&(M.selection_scope=='fixed_k')&(M.k.astype(str)==str(k))]
 if len(check): assert np.allclose(mean,check.iloc[0][['accuracy','macro_f1','balanced_accuracy']].astype(float),atol=1e-12)
 rows.append({'family':family,'k':int(k),'n_unique_specimens':300,'partition_repeats':3,'outer_fits':15,'accuracy':mean[0],'macro_f1':mean[1],'balanced_accuracy':mean[2]})
metrics=pd.DataFrame(rows);metrics.to_csv(A/'feature_budget_all_fixedK_metrics.csv',index=False)
# Primary metric replays are separate from pre-specified fixed K sensitivities.
pri=[]
for family,g in P[P.is_primary_k==1].groupby('family'):
 vals=[]
 for seed,v in g.groupby('repeat_seed'):
  assert len(v)==300 and v.record_id.nunique()==300
  vals.append((accuracy_score(v.y_true,v.prediction),f1_score(v.y_true,v.prediction,average='macro'),balanced_accuracy_score(v.y_true,v.prediction)))
 mean=np.mean(vals,axis=0)
 check=M[(M.method==family)&(M.selection_scope=='primary')]
 assert len(check)==1,(family,check.shape)
 assert np.allclose(mean,check.iloc[0][['accuracy','macro_f1','balanced_accuracy']].astype(float),atol=1e-12)
 pri.append({'family':family,'accuracy':mean[0],'macro_f1':mean[1],'balanced_accuracy':mean[2]})
pd.DataFrame(pri).to_csv(A/'feature_primary_replayed.csv',index=False)
budgets=[31,62,123,184,245]
mod=[]; overlaps=[]; paired=[]
rng=np.random.default_rng(20261109)
ids=sorted(P.physical_sample_id.unique());assert len(ids)==300
indices=rng.integers(0,300,size=(5000,300))
for k in budgets:
 sel=S[(S.family=='concat_both_snv20')&(S.k==k)]
 assert len(sel)==15*k
 a=int((sel.modality=='alpha').sum());n=int((sel.modality=='n').sum());assert a+n==15*k
 for (seed,fold),v in sel.groupby(['repeat_seed','outer_fold']):
  other=S[(S.family=='alpha_snv20')&(S.k==k)&(S.repeat_seed==seed)&(S.outer_fold==fold)]
  sa=set(v.loc[v.modality=='alpha','within_modality_index']);sb=set(other.within_modality_index)
  overlaps.append({'k':k,'repeat_seed':int(seed),'outer_fold':int(fold),'alpha_intersection':len(sa&sb),'alpha_union':len(sa|sb),'alpha_subset_jaccard':len(sa&sb)/len(sa|sb)})
 j=np.mean([o['alpha_subset_jaccard'] for o in overlaps if o['k']==k])
 mod.append({'k':k,'outer_refits':15,'total_feature_occurrences':15*k,'alpha_occurrences':a,'n_occurrences':n,'n_fraction':n/(15*k),'mean_alpha_subset_jaccard':j})
 g=P[(P.k==k)&P.family.isin(['alpha_snv20','concat_both_snv20'])]
 wide=g.pivot(index=['physical_sample_id','repeat_seed'],columns='family',values='is_correct').reset_index()
 delta=wide.assign(delta=wide.concat_both_snv20-wide.alpha_snv20).groupby('physical_sample_id').delta.mean().reindex(ids).to_numpy()
 boot=delta[indices].mean(axis=1)
 base=metrics[(metrics.family=='alpha_snv20')&(metrics.k==k)].iloc[0]
 joint=metrics[(metrics.family=='concat_both_snv20')&(metrics.k==k)].iloc[0]
 paired.append({'k':k,'alpha_accuracy':float(base.accuracy),'joint_pool_accuracy':float(joint.accuracy),'joint_minus_alpha_pp':float(delta.mean()*100),'cluster_bootstrap_95_low_pp':float(np.quantile(boot,.025)*100),'cluster_bootstrap_95_high_pp':float(np.quantile(boot,.975)*100),'draws':5000,'seed':20261109,'specimen_clusters':300,'repeats_per_cluster':3,'multiplicity_adjusted':False})
mod=pd.DataFrame(mod);mod.to_csv(A/'feature_modality_composition.csv',index=False)
pd.DataFrame(overlaps).to_csv(A/'feature_subset_overlap_by_refit.csv',index=False)
paired=pd.DataFrame(paired);paired.to_csv(A/'feature_budget_paired_bootstrap.csv',index=False)
(A/'feature_source_verification.json').write_text(json.dumps({'status':'PASS','accepted_view':'paper/current/results/v4_330','source_pin':PIN,'verified_source_files':len(receipt),'current_summary_and_metrics_byte_equal':True,'records':receipt,'model_fits_performed':0,'bootstrap_scope':'exploratory paired saved predictions, 300 physical specimen clusters retaining 3 repeats; no multiplicity adjustment'},ensure_ascii=False,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
colors={'alpha_snv20':'#0072B2','concat_both_snv20':'#D55E00','alpha_raw20':'#009E73','n_raw20':'#777777'}
labels={'alpha_snv20':'SNV absorption','concat_both_snv20':'Joint SNV candidate pool','alpha_raw20':'Raw absorption','n_raw20':'Raw refractive index'}
fig,ax=plt.subplots(1,2,figsize=(7.3,3.0),gridspec_kw={'width_ratios':[1.35,1]})
for fam in colors:
 d=metrics[metrics.family==fam].sort_values('k');ax[0].plot(d.k,d.accuracy*100,'o-',lw=1.35,ms=4,color=colors[fam],label=labels[fam])
ax[0].set(xlabel='Total retained predictors K',ylabel='Mean accuracy (%)',xticks=budgets,ylim=(65,97));ax[0].legend(frameon=False,fontsize=7,loc='lower right');ax[0].text(0,1.04,'A',transform=ax[0].transAxes,weight='bold',fontsize=12)
y=paired.joint_minus_alpha_pp.to_numpy(); lo=paired.cluster_bootstrap_95_low_pp.to_numpy();hi=paired.cluster_bootstrap_95_high_pp.to_numpy()
ax[1].errorbar(paired.k,y,yerr=np.vstack([y-lo,hi-y]),fmt='o',color='#D55E00',capsize=3,lw=1.2);ax[1].axhline(0,color='#777777',ls='--',lw=.9)
ax[1].set(xlabel='Total retained predictors K',ylabel='Joint pool minus absorption (pp)',xticks=budgets);ax[1].text(0,1.04,'B',transform=ax[1].transAxes,weight='bold',fontsize=12)
fig.tight_layout(w_pad=1.6)
for ext in ['png','pdf','svg']:fig.savefig(F/f'fig2_feature_budget.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(7.3,2.85),gridspec_kw={'width_ratios':[1.1,1]})
x=np.arange(len(mod));nf=mod.n_fraction.to_numpy()*100
ax[0].bar(x,100-nf,color='#0072B2',label='Absorption');ax[0].bar(x,nf,bottom=100-nf,color='#CC79A7',label='Refractive index')
for i,r in mod.iterrows():ax[0].text(i,102,f"{int(r.n_occurrences)}/{int(r.total_feature_occurrences)}",ha='center',fontsize=7)
ax[0].set(xticks=x,xticklabels=budgets,xlabel='Total retained predictors K',ylabel='Feature occurrences (%)',ylim=(0,112));ax[0].legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.27),ncol=2);ax[0].text(0,1.04,'A',transform=ax[0].transAxes,weight='bold',fontsize=12)
ax[1].plot(mod.k,mod.mean_alpha_subset_jaccard,'o-',color='#009E73');ax[1].set(xlabel='Total retained predictors K',ylabel='Absorption-subset Jaccard',xticks=budgets,ylim=(0,1));ax[1].text(0,1.04,'B',transform=ax[1].transAxes,weight='bold',fontsize=12)
fig.tight_layout(w_pad=1.6)
for ext in ['png','pdf','svg']:fig.savefig(F/f'fig3_selected_modalities.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)
print('VERIFIED',len(receipt));print(paired.to_string(index=False));print(mod.to_string(index=False))
