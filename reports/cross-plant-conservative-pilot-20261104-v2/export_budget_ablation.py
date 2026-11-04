"""Actual, paired development results at two registered maximum-epoch budgets."""
from pathlib import Path
import csv
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/research4-conservative-pilot-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

r=Path(__file__).resolve().parent
previous=r.with_name('cross-plant-conservative-pilot-20261104-v1')
v1=json.loads((previous/'metrics.json').read_text());v2=json.loads((r/'metrics.json').read_text())
p=json.loads((r/'protocol.json').read_text());arms=p['arms'];seeds=p['seeds']
def csvwrite(path,rows):
    if path.exists():raise FileExistsError(path)
    with path.open('w',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
summary=[];change=[]
for maximum,v in [(48,v1),(96,v2)]:
    for arm in arms:
        m=v['mean'][arm];a=v['mean']['A']
        summary.append({'maximum_source_epoch_budget':maximum,'arm':arm,'accuracy_percent':100*m['accuracy'],
            'macro_f1':m['macro_f1'],'balanced_accuracy_percent':100*m['balanced_accuracy'],
            'accuracy_delta_vs_same_budget_A_pp':100*(m['accuracy']-a['accuracy']),
            'unique_target_samples':855,'seeds':'42/43/44','confirmation_status':'DEVELOPMENT_ONLY'})
for arm in arms:
    before=v1['mean'][arm];after=v2['mean'][arm]
    change.append({'arm':arm,'accuracy_max48_percent':100*before['accuracy'],'accuracy_max96_percent':100*after['accuracy'],
        'accuracy_change_pp':100*(after['accuracy']-before['accuracy']),
        'accuracy_relative_change_percent':100*(after['accuracy']/before['accuracy']-1),
        'macro_f1_change':after['macro_f1']-before['macro_f1'],
        'balanced_accuracy_change_pp':100*(after['balanced_accuracy']-before['balanced_accuracy']),
        'cause_under_comparison':'maximum source-selected target epoch budget, not added external observations'})
csvwrite(r/'two_budget_model_comparison.csv',summary)
csvwrite(r/'epoch_budget_changes.csv',change)
names=['Scratch','Target-source + L2-SP','Huangqi/Tianma + L2-SP'];colors=['#356ca5','#65a395','#c16b55']
fig,axes=plt.subplots(1,3,figsize=(12.6,4.3),layout='constrained')
for ax,key,label,scale in zip(axes,['accuracy','macro_f1','balanced_accuracy'],['Accuracy (%)','Macro-F1','Balanced accuracy (%)'],[100,1,100]):
    for arm,name,color in zip(arms,names,colors):
        means=[scale*v['mean'][arm][key] for v in (v1,v2)]
        sd=[scale*np.std([v['per_seed'][f'{s}.{arm}'][key] for s in seeds],ddof=1) for v in (v1,v2)]
        ax.errorbar([48,96],means,yerr=sd,marker='o',color=color,capsize=4,label=name)
    ax.set_xticks([48,96]);ax.set_xlabel('Maximum source epoch budget');ax.set_ylabel(label);ax.grid(alpha=.2)
    ax.set_ylim((50,100) if scale==100 else (.4,1))
axes[0].legend(fontsize=8,loc='lower right')
fig.suptitle('Longer source-selected training improves all arms; external transfer remains below scratch',fontsize=11)
fig.text(.5,-.035,'Same 855 exposed targets and seeds42/43/44. Bars show model-seed SD, not independent-sample uncertainty.',ha='center',fontsize=8)
for extension in ('png','pdf'):fig.savefig(r/('epoch_budget_ablation.'+extension),dpi=200,bbox_inches='tight')
plt.close(fig)
lock=json.loads((r/'source_epoch_lock.json').read_text())
fig,ax=plt.subplots(figsize=(9.2,5),layout='constrained')
for choice in lock['choices']:
    curve=choice['source_curve'];x=[a['epoch'] for a in curve];y=[a['macro_f1'] for a in curve]
    line=ax.plot(x,y,label=f"Outer {choice['outer_fold']}: selected {choice['epochs']}",linewidth=1.4)[0]
    selected=choice['source_selected_metrics'];ax.scatter([choice['epochs']],[selected['macro_f1']],color=line.get_color(),s=25)
ax.axvline(48,color='gray',linestyle='--',label='First registered maximum48')
ax.set_xlabel('Source-only candidate epoch');ax.set_ylabel('Scratch source OOF Macro-F1 (mean of seeds)')
ax.set_title('Source-only convergence boundary check; no outer truth in epoch selection',fontsize=11)
ax.legend(fontsize=8,loc='lower right');ax.grid(alpha=.2)
for extension in ('png','pdf'):fig.savefig(r/('source_epoch_curves.'+extension),dpi=200,bbox_inches='tight')
plt.close(fig)
print(json.dumps({'status':'EXPORTED_REAL_TWO_BUDGET_COMPARISON','changes':change}))
