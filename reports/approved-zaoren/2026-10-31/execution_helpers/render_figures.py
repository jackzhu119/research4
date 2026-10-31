"""Render prespecified descriptive plots from archived predictions; no fitting."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ARCHIVE=Path(__file__).resolve().parents[1]
FIGURES=ARCHIVE/'figures'
FIGURES.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
with (ARCHIVE/'independent-audit/pooled_metrics.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
assert len(rows)==13
labels={
'alpha_primary/alpha_dummy':'alpha / Majority class',
'alpha_primary/alpha_svc':'alpha / SVM',
'alpha_primary/alpha_plsda':'alpha / PLS-DA',
'alpha_primary/alpha_mcuve20_svc':'alpha / MC20 + SVM',
'alpha_primary/alpha_cars_svc':'alpha / CARS-style + SVM',
'modality_controls/n_svc':'n / SVM',
'modality_controls/n_plsda':'n / PLS-DA',
'modality_controls/concat_svc':'n + alpha / SVM',
'modality_controls/concat_plsda':'n + alpha / PLS-DA',
'stability_controls/alpha_mcuve100_svc':'alpha / MC100 + SVM',
'stability_controls/alpha_stable_svc':'alpha / Repeated MC + SVM',
'full_export_sensitivity/alpha_svc':'Full-range alpha / SVM [sensitivity]',
'full_export_sensitivity/alpha_plsda':'Full-range alpha / PLS-DA [sensitivity]'}
blue='#245B83';orange='#C77728';colors=[orange if r['case'].startswith('full_export') else blue for r in rows]
y=np.arange(len(rows))
fig,(left,right)=plt.subplots(1,2,figsize=(15.8,8.6),sharey=True)
for ax,metric,title in [(left,'macro_f1','Macro F1 (primary)'),(right,'balanced_accuracy','Balanced accuracy')]:
    values=[float(r[metric]) for r in rows]
    ax.barh(y,values,color=colors,height=.66)
    ax.set_xlim(0,1);ax.set_xticks(np.linspace(0,1,6));ax.set_xlabel('Score')
    ax.set_title(title,pad=11,fontsize=13);ax.grid(axis='x',alpha=.22,zorder=0);ax.set_axisbelow(True)
    ax.axhline(10.5,color='#999999',lw=.8,linestyle='--')
    for position,value in zip(y,values):
        if value>.88:ax.text(value-.014,position,f'{value:.3f}',ha='right',va='center',fontsize=10,color='white')
        else:ax.text(value+.012,position,f'{value:.3f}',ha='left',va='center',fontsize=10,color='#202020')
left.set_yticks(y,[labels[r['case']] for r in rows],fontsize=10.5);left.invert_yaxis()
fig.suptitle('Acid jujube: anonymous origin-group internal validation',fontsize=16,y=.982)
fig.text(.5,.94,'855 user-declared independent samples | 7 groups | 5 outer folds / 2 inner folds | one source batch per group',ha='center',fontsize=10.5)
fig.legend(handles=[Patch(color=blue,label='0.1-1.5 THz engineering candidate'),Patch(color=orange,label='Full exported range: sensitivity only')],loc='lower center',bbox_to_anchor=(.6,.065),ncol=2,frameon=False,fontsize=10)
fig.text(.5,.035,'No cross-batch validation or chemical causality is established. ZY02 has n=5. No inferential intervals are shown.',ha='center',fontsize=10)
fig.subplots_adjust(left=.29,right=.975,top=.87,bottom=.145,wspace=.16)
for suffix in ['png','pdf']:fig.savefig(FIGURES/f'metric_comparison.{suffix}',dpi=190,facecolor='white')
plt.close(fig)

for case,title,name in [('experiments/alpha_primary/alpha_svc/result.json','Predefined alpha-SVM confusion matrix','confusion_alpha_svm'),
                        ('experiments/modality_controls/concat_svc/result.json','Predefined n+alpha-SVM confusion matrix','confusion_concat_svm')]:
    result=json.loads((ARCHIVE/case).read_text());metrics=result['metrics'];classes=metrics['class_order'];counts=np.asarray(metrics['confusion_matrix'],dtype=int)
    totals=counts.sum(axis=1);assert counts.sum()==855 and dict(zip(classes,totals.tolist()))==metrics['class_counts']
    norm=counts/totals[:,None]
    fig,ax=plt.subplots(figsize=(8.7,7.6));image=ax.imshow(norm,cmap='Blues',vmin=0,vmax=1,aspect='equal')
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j,i,f'{counts[i,j]}\n{100*norm[i,j]:.1f}%',ha='center',va='center',fontsize=9,color='white' if norm[i,j]>.55 else '#182733')
    ax.set_xticks(np.arange(len(classes)),classes,fontsize=10)
    ax.set_yticks(np.arange(len(classes)),[f'{c} (n={n})' for c,n in zip(classes,totals)],fontsize=10)
    ax.set_xlabel('Predicted origin group');ax.set_ylabel('True origin group')
    ax.set_title(title,fontsize=14,pad=15)
    bar=fig.colorbar(image,ax=ax,fraction=.045,pad=.04);bar.set_label('Fraction within true class');bar.set_ticks(np.linspace(0,1,6))
    ax.set_xticks(np.arange(-.5,len(classes),1),minor=True);ax.set_yticks(np.arange(-.5,len(classes),1),minor=True)
    ax.grid(which='minor',color='white',lw=.8);ax.tick_params(which='minor',bottom=False,left=False)
    fig.text(.5,.055,'0.1-1.5 THz candidate | original OOF counts and row percentages | ZY02 denominator = 5',ha='center',fontsize=9.3)
    fig.text(.5,.028,'Anonymous groups; one source batch per group. These are prespecified models, not selected by test performance.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.97))
    for suffix in ['png','pdf']:fig.savefig(FIGURES/f'{name}.{suffix}',dpi=190,facecolor='white')
    plt.close(fig)
metadata={'matplotlib_version':matplotlib.__version__,'source':'Archived OOF predictions and independently reconciled metrics',
          'comparison_cases':13,'comparison_xlimits':[0,1],'comparison_primary_metric':'macro_f1','comparison_secondary_metric':'balanced_accuracy',
          'confusion_models_prespecified':['candidate alpha-SVM','candidate n+alpha-SVM'],'confusion_cell_text':'original count and within-true-class percentage',
          'ZY02_denominator':5,'statistical_intervals_or_significance_claims':False,
          'scope':'Anonymous origin-group, single-source-batch internal recognition; not cross-batch generalization or chemical causality'}
(ARCHIVE/'figures/figure_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'figures':6,'matplotlib':matplotlib.__version__,'directory':str(FIGURES)}))
