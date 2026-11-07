"""Read-only summary and figures from this completed, registered pilot."""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/research4-conservative-pilot-matplotlib')
REPORT=Path(__file__).resolve().parent
ROOT=REPORT.parents[1]
sys.path.insert(0,str(ROOT/'src'))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def csvwrite(p,rows):
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    import numpy as np
    import torch
    from herb_spectra import cross_plant_transfer as old
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if read(REPORT/'completion.json')['status']!='DONE_DEVELOPMENT_ONLY':
        raise ValueError('Do not summarize incomplete seeds/folds as a complete experiment')
    metrics=read(REPORT/'metrics.json');protocol=read(REPORT/'protocol.json')
    arms=protocol['arms'];seeds=protocol['seeds'];labels=protocol['classes']
    rows=[]
    for arm in arms:
        m=metrics['mean'][arm];baseline=metrics['mean']['A']
        rows.append({'arm':arm,'accuracy':m['accuracy'],'accuracy_percent':100*m['accuracy'],
            'macro_f1':m['macro_f1'],'balanced_accuracy':m['balanced_accuracy'],
            'accuracy_seed_sample_SD_pp':100*np.std([metrics['per_seed'][f'{s}.{arm}']['accuracy'] for s in seeds],ddof=1),
            'accuracy_delta_vs_A_pp':100*(m['accuracy']-baseline['accuracy']),
            'accuracy_relative_change_vs_A_percent':100*(m['accuracy']/baseline['accuracy']-1),
            'macro_f1_delta_vs_A':m['macro_f1']-baseline['macro_f1'],
            'balanced_accuracy_delta_vs_A_pp':100*(m['balanced_accuracy']-baseline['balanced_accuracy']),
            'unique_target_samples':855,'model_seeds':3,'status':'DEVELOPMENT_NOT_NEW_CONFIRMATION'})
    csvwrite(REPORT/'model_comparison.csv',rows)
    csvwrite(REPORT/'paired_corrected_harmed.csv',metrics['paired_changes'])
    perclass=[];matrices=[];costs=[];replayed=[]
    prep=read(old.ROOT/'reports/cross-plant-deep-transfer-20261031-v1/neural/preparation.json')
    outer=[c for c in prep['contexts'] if c['stage']=='outer']
    oldroot=old.ROOT/'reports/cross-plant-deep-transfer-20261031-v1/neural'
    # Reconstruct scores from the actual saved weights and source-fitted scaler;
    # no fit, backward pass or optimizer is created here.
    for seed in seeds:
        for arm in arms:
            value=metrics['per_seed'][f'{seed}.{arm}']
            for label,entry in value['per_class'].items():
                perclass.append({'seed':seed,'arm':arm,'class':label,**entry})
            for i,label in enumerate(labels):
                for j,pred in enumerate(labels):
                    matrices.append({'seed':seed,'arm':arm,'true_class':label,'predicted_class':pred,
                        'count':value['confusion_matrix'][i][j]})
            for c in outer:
                d=REPORT/'outer_models'/c['split_id']/f'seed{seed}'/arm
                training=read(d/'training.json')
                if sha(d/'last-state.pt')!=training['model_sha256'] or sha(d/'predictions.npz')!=training['prediction_sha256']:
                    raise ValueError('Saved fitted state or predictions changed')
                with np.load(oldroot/c['query_pack'],allow_pickle=False) as q:
                    x=old.row_snv(q['X']);ids=q['record_ids']
                with np.load(d/'target-scaler.npz',allow_pickle=False) as scaler:
                    x=((x-scaler['mean'])/scaler['scale']).astype(np.float32)
                model=old.TransferNet(protocol['architecture'],seed)
                model.load_state_dict(torch.load(d/'last-state.pt',map_location='cpu',weights_only=True));model.eval()
                before=old.state_hash(model)
                with torch.no_grad():scores=model({'alpha':torch.as_tensor(x)}).cpu().numpy()
                if old.state_hash(model)!=before:raise ValueError('Replay changed learned state')
                with np.load(d/'predictions.npz',allow_pickle=False) as a:
                    if not np.array_equal(ids,a['record_ids']):raise ValueError('Replay row order differs')
                    difference=float(np.max(np.abs(scores-a['scores'])))
                    if not np.allclose(scores,a['scores'],atol=1e-5,rtol=1e-5):raise ValueError('Native replay differs')
                    if not np.array_equal(scores.argmax(1),a['scores'].argmax(1)):raise ValueError('Replay class changed')
                replayed.append({'outer_fold':c['outer_fold'],'seed':seed,'arm':arm,'rows':len(ids),
                    'maximum_logit_difference':difference,'model_state_unchanged':True,
                    'model_sha256':training['model_sha256'],'prediction_sha256':training['prediction_sha256']})
                costs.append({'outer_fold':c['outer_fold'],'seed':seed,'arm':arm,
                    'epochs':training['epochs'],'target_steps':training['target_steps'],
                    'pretraining_steps':sum(training['pretraining_pool_steps'].values()),
                    'actual_CPU_seconds_inside_fit_excludes_setup':training['actual_CPU_seconds_inside_fit'],
                    'model_parameters':sum(p.numel() for p in model.parameters()),
                    'concurrent_DFT_no_wall_speedup_claim':True})
    csvwrite(REPORT/'per_class_metrics.csv',perclass)
    csvwrite(REPORT/'confusion_matrix_counts.csv',matrices)
    csvwrite(REPORT/'outer_computational_cost.csv',costs)
    save(REPORT/'native_model_replay.json',{'status':'PASS','new_training_fits':0,'quantum_calculations':0,
        'pipeline_count':45,'prediction_count':sum(v['rows'] for v in replayed),
        'maximum_logit_difference':max(v['maximum_logit_difference'] for v in replayed),
        'absolute_and_relative_tolerance':1e-5,'replays':replayed,
        'export_source_sha256':sha(__file__),'input_metrics_sha256':sha(REPORT/'metrics.json')})
    names=['Scratch','Target-source\npretrain + L2-SP','Huangqi + Tianma\npretrain + L2-SP']
    colors=['#356ca5','#65a395','#c16b55']
    fig,axes=plt.subplots(1,3,figsize=(12.6,4.4),layout='constrained')
    for ax,key,title,scale in zip(axes,['accuracy','macro_f1','balanced_accuracy'],
            ['Accuracy (%)','Macro-F1','Balanced accuracy (%)'],[100,1,100]):
        samples=np.asarray([[metrics['per_seed'][f'{s}.{a}'][key]*scale for s in seeds] for a in arms])
        ax.bar(range(3),samples.mean(1),yerr=samples.std(1,ddof=1),capsize=4,color=colors,alpha=.85)
        for i in range(3):ax.scatter(np.full(3,i)+[-.07,0,.07],samples[i],c='#333333',s=20,zorder=3)
        ax.set_xticks(range(3),names,fontsize=8);ax.set_ylabel(title);ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True);ax.set_ylim(0,100 if scale==100 else 1)
    fig.suptitle('Registered development pilot: 855 exposed target samples, three fixed model seeds',fontsize=11)
    fig.text(.5,-.025,'Error bars: model-seed sample SD, not biological confidence intervals. Same folds and target epochs within this pilot.',ha='center',fontsize=8)
    for extension in ('png','pdf'):fig.savefig(REPORT/('model_metrics.'+extension),dpi=200,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    maximum=max(max(max(row) for row in metrics['per_seed'][f'42.{a}']['confusion_matrix']) for a in arms)
    for ax,arm,name in zip(axes,arms,names):
        matrix=np.asarray(metrics['per_seed'][f'42.{arm}']['confusion_matrix'])
        image=ax.imshow(matrix,vmin=0,vmax=maximum,cmap='Blues')
        for i in range(7):
            for j in range(7):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=7,color='white' if matrix[i,j]>.6*maximum else 'black')
        ax.set_xticks(range(7),labels,rotation=45,fontsize=8);ax.set_yticks(range(7),labels,fontsize=8)
        ax.set_xlabel('Predicted');ax.set_ylabel('True');ax.set_title(name,fontsize=10)
    fig.colorbar(image,ax=axes,shrink=.75,label='Prediction count (N=855 per arm)')
    fig.suptitle('Seed 42: first predeclared seed; all other seeds retained in CSV/JSON',fontsize=11,y=1.12)
    for extension in ('png','pdf'):fig.savefig(REPORT/('confusion_seed42.'+extension),dpi=200,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'status':'EXPORTED_FROM_REAL_COMPLETED_RESULTS','native_replay_max_difference':max(v['maximum_logit_difference'] for v in replayed),'comparison':rows}))


if __name__=='__main__':main()
