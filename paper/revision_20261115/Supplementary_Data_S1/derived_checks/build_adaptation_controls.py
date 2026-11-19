#!/usr/bin/env python3
"""Build manuscript controls figure directly from current allowlisted results."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'data/active_workbooks.json').exists())
CURRENT=ROOT/'paper/current'
OUT=Path(__file__).resolve().parent.parent
A=Path(__file__).resolve().parent;F=ROOT/'paper/revision_20261115/figures'
A.mkdir(parents=True,exist_ok=True);F.mkdir(parents=True,exist_ok=True)
manifest=json.loads((CURRENT/'manifest.json').read_text())
allow={r['path']:r for r in manifest['files']}
used=[]
def verified(relative):
    p=CURRENT/relative
    r=allow[relative]
    assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],relative
    if relative not in used:used.append(relative)
    return p

def curves(b):
    relative=f'results/B{b}/results_summary.json'
    d=json.loads(verified(relative).read_text())
    return relative,pd.DataFrame(d['curves_macro_over_jobs' if b==1 else 'curves_macro_over_job_draws'])

specs=[
 ('Source only',1,None,'identity',0,.8),
 ('True support labels: full refit',3,'G02','true_support',10,.8388888888888889),
 ('Frozen scaler and kernel scale',2,'F02','scaler_frozen_gamma_fixed',10,.8377777777777777),
 ('Unlabeled support statistics',3,'G02','unlabeled_statistics',10,.8011111111111111),
 ('Cyclically incorrect support labels',3,'G02','cyclic_support',10,.7333333333333333),
 ('Normalized total loss mass',4,'H01','source_support_normalized_mass',10,.8377777777777777),
]
rows=[]
for label,b,family,method,k,expected in specs:
    path,d=curves(b)
    q=d[(d.experiment=='E03')&(d.method==method)&(d.k==k)]
    if family:q=q[q.family==family]
    assert len(q)==1,(label,len(q))
    r=q.iloc[0]
    assert abs(float(r.accuracy)-expected)<1e-12,(label,r.accuracy)
    tr_path=f'tables/B{b}/paired_transitions.csv.gz'
    t=pd.read_csv(verified(tr_path),low_memory=False)
    t=t[(t.experiment=='E03')&(t.method==method)&(t.k==k)&(t.scope=='ALL')]
    if family:t=t[t.family==family]
    if b==3:t=t[t.comparison==method+'_minus_source_identity']
    if b==2:t=t[t.comparison=='scaler_frozen_gamma_fixed_minus_source_baseline']
    if b==4:t=t[t.comparison=='source_support_normalized_mass_minus_source_C10']
    if k:
        assert len(t)==1,(label,'transition rows',len(t))
        tr=t.iloc[0]
        assert int(tr.n_predictions)==900 and int(tr.n_physical_samples)==300
        assert abs(float(tr.after_accuracy)-float(r.accuracy))<1e-12
        assert abs(float(tr.before_accuracy)-.8)<1e-12
    rows.append(dict(label=label,round=f'B{b}',family=family or '',experiment='E03',method=method,k_per_class=k,source_n=190,support_pool_n=50,query_per_context=60,independent_specimens=300,partition_seeds=3,outer_folds=5,repeated_query_predictions=900,accuracy=float(r.accuracy),accuracy_percent=100*float(r.accuracy),macro_f1=float(r.macro_f1),delta_accuracy_pp=100*(float(r.accuracy)-.8),summary_path=path,summary_source_commit=allow[path]['source_commit'],paired_transitions_path=tr_path,source_row_key=f'E03|{family or ""}|{method}|K{k}'))

# B1 and B3 full refits are the same endpoint; explicitly verify the cross-round reuse.
p,b1=curves(1)
q=b1[(b1.experiment=='E03')&(b1.method=='full_pipeline_retrain')&(b1.k==10)]
assert len(q)==1 and abs(float(q.iloc[0].accuracy)-rows[1]['accuracy'])<1e-12
b1.to_csv(A/'B1_learning_curves_current.csv',index=False)
p,b5=curves(5)
acq=b5[b5.family=='I02'].copy()
assert len(acq)>0
acq.to_csv(A/'B5_acquisition_curves_current.csv',index=False)
# All I02 endpoint rows preserve labels used and observed class coverage for the supplement.
p='tables/B5/metrics.csv.gz'
metrics=pd.read_csv(verified(p),low_memory=False)
acq_metrics=metrics[metrics.family=='I02'].copy()
assert len(acq_metrics)>0
acq_metrics.to_csv(A/'B5_acquisition_metrics_current.csv.gz',index=False,compression={'method':'gzip','mtime':0})
df=pd.DataFrame(rows);df.to_csv(A/'adaptation_controls.csv',index=False)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'axes.titlesize':12,'svg.fonttype':'none','pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(8.5,4.4),constrained_layout=True)
y=np.arange(len(df))[::-1]
colors=['#697681','#1366A2','#287E8E','#7C8790','#B65040','#287E8E']
ax.axvline(80,color='#7A8790',lw=1.1,ls=(0,(4,3)),zorder=1)
for i,(r,yy) in enumerate(zip(rows,y)):
    ax.plot([80,r['accuracy_percent']],[yy,yy],color=colors[i],lw=1.2,alpha=.65,zorder=2)
    ax.scatter(r['accuracy_percent'],yy,s=62,color=colors[i],edgecolor='white',lw=.65,zorder=3)
    left=r['accuracy_percent']<80
    ax.text(r['accuracy_percent']+(-.30 if left else .30),yy,f"{r['accuracy_percent']:.2f}%",va='center',ha='right' if left else 'left',fontsize=10,color=colors[i],fontweight='bold' if i==1 else 'normal',bbox=dict(facecolor='white',edgecolor='none',pad=1.3))
ax.set_yticks(y,df.label)
ax.set_xlim(70.0,87.2);ax.set_ylim(-.65,5.65)
ax.set_xticks([72,76,80,84]);ax.set_xlim(70.0,87.2)
ax.set_xlabel('Internal query accuracy (%)')
ax.set_title('Support-label and preprocessing controls',loc='left',pad=18,fontweight='bold')
ax.grid(axis='x',color='#E8ECEF',lw=.7)
ax.set_axisbelow(True)
ax.tick_params(axis='y',length=0,pad=10)
ax.tick_params(axis='x',color='#BAC3C9')
for s in ['left','top','right']:ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color('#BAC3C9')
fig.patch.set_facecolor('white');ax.set_facecolor('white')
for ext in ['png','pdf','svg']:fig.savefig(F/f'fig4_adaptation_controls.{ext}',dpi=300,facecolor='white',bbox_inches='tight')
plt.close(fig)
caption=("Figure 4. Support-label and preprocessing controls for the five-source Suanzaoren task. "
"Points show archived internal query accuracies, with the dashed line marking the source-only model (80.00%). "
"The full-refit endpoint with true support labels (83.89%; B1/B3) is compared with frozen scaler and kernel scale (83.78%; B2), "
"unlabeled support statistics (80.11%; B3), cyclically incorrect support labels (73.33%; B3), and normalized total loss mass (83.78%; B4). "
"Each context uses 190 source specimens, a 50-specimen support pool and 60 query specimens; K=10 per class supplies 50 additional labels where applicable. "
"Three partition seeds and five outer folds reuse the same 300 master-TRAIN specimens (900 repeated query predictions). "
"B1 and B3 full-refit values refer to the same archived endpoint and are not independent replications. "
"The unlabeled-statistics condition does not use support labels in classifier fitting. "
"All results are exploratory internal comparisons on previously exposed specimens; no new-batch evaluation is represented. "
"No confidence interval is plotted.")
receipt={'caption':caption,'generated_files':[f'figures/fig4_adaptation_controls.{e}' for e in ['png','pdf','svg']],'source_manifest_sha256':hashlib.sha256((CURRENT/'manifest.json').read_bytes()).hexdigest(),'source_files':[{**allow[p],'relative_to':'paper/current'} for p in used],'controls_rows':len(df),'B1_learning_curve_rows':len(b1),'B5_acquisition_curve_rows':len(acq),'B5_acquisition_metric_rows':len(acq_metrics),'checks':['Every input hash matches current allowlist','Each plotted summary endpoint is unique and equals expected archived value','Each nonzero-K paired endpoint has 900 repeated predictions on 300 physical specimens','All controls have source-only accuracy 0.8','B1/B3 full-refit endpoint equality verified'],'no_new_model_fits':True}
(A/'fig4_caption_and_sources.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
(A/'fig4_caption.txt').write_text(caption+'\n')
print(json.dumps({'figure':str(F/'fig4_adaptation_controls.png'),'controls':len(df),'B1_rows':len(b1),'B5_curves':len(acq),'B5_metric_rows':len(acq_metrics)},ensure_ascii=False))
