"""Freeze a finite, evidence-led TRAIN-only experiment matrix; never fit models."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/research/evidence_validation_20261108_v4'
RUN='evidence-validation-20261108-v4'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()


def freeze(path,value):
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Existing frozen plan differs: '+str(path))
    else:path.write_text(text)


def main():
    auth=json.loads((BASE/'user_authorization_NEW_90000.json').read_text())
    assert auth['new_total_cpu_seconds']==90000
    features=BASE/'train_only_features.npz';split=BASE/'train_only_split.csv';lock=BASE/'input_lock.json'
    locked=json.loads(lock.read_text());assert sha(features)==locked['training_feature_sha256']
    assert sha(split)==locked['training_split_sha256'] and locked['training_records']==999
    workers={name:ROOT/'scripts'/filename for name,filename in {
       'matched_classical':'research4_matched_feature_validation_20261108_v4.py',
       'neural':'research4_representation_transfer_validation_20261108_v4.py',
       'admission':'research4_evidence_admission_20261108_v4.py',
       'resources':'research4_evidence_resources_20261108_v4.py',
       'summary':'research4_compact_evidence_20261108_v4.py',
       'supervisor':'species_background_budget.py',
       'inherited_neural':'species_revision_background_neural.py'}.items()}
    workers['closed_cache_advice']=ROOT/'scripts/research4_closed_cache_advice_20261108.py'
    assert all(p.is_file()for p in workers.values())
    admission={'schema':'research4-evidence-admission-v1','run_id':RUN,
        'authorization_observed_utc':auth['authorization_observed_utc'],'deadline_utc':auth['deadline_utc'],
        'old_state_path':'/workspace/research4-relaunch/data/research/species_relaunch_20261108_v2/control/state.json',
        'old_outer_state_path':'/workspace/research4-relaunch/data/research/species_relaunch_20261108_v2/resource-control/state.json',
        'approved_worker_refs':[{'path':str(workers[k]),'sha256':sha(workers[k])}for k in ['matched_classical','neural']],
        'closed_cache_advice_ref':{'path':str(workers['closed_cache_advice']),'sha256':sha(workers['closed_cache_advice'])},
        'note':'Read old queue identities only. Never signal old processes or modify its source, ledgers or Git.'}
    freeze(BASE/'admission_config.json',admission)
    sources=sorted((ROOT/'src/herb_spectra').glob('*.py'))
    protocol={'schema':'research4-evidence-driven-validation-v3','run_id':RUN,'cpu_budget_seconds':88000,
        'grant_id':'evidence-validation-20261108-v3','same_grant_recovery':True,
        'prior_execution_carry':{'path':str(BASE/'prior_execution_carry.json'),'sha256':sha(BASE/'prior_execution_carry.json')},
        'authorized_total_cpu_seconds':90000,'preparation_reserved_cpu_seconds':1000,'controller_reserved_cpu_seconds':1000,
        'authorization_record':{'path':str(BASE/'user_authorization_NEW_90000.json'),'sha256':sha(BASE/'user_authorization_NEW_90000.json')},
        'input_lock_sha256':sha(lock),'training_allowlist':locked['training_allowlist'],'training_records':999,
        'master_roles_unchanged':True,'CAL334_PRED339_excluded':True,'pure_acid_TRAIN':300,
        'expected_fold_ids':[0,1,2,3,4],'expected_repeat_seeds':[20261031,20261108,42],'expected_model_seeds':[42,43,44],
        'inner_folds':2,'worker_code_sha256':{str(p.relative_to(ROOT)):sha(p)for p in workers.values()},
        'module_code_sha256':{str(p.relative_to(ROOT)):sha(p)for p in sources},
        'runtime_versions':{k:importlib.metadata.version(k)for k in ['numpy','scipy','scikit-learn','torch','joblib']},
        'hypotheses':[
          'Apparent n+alpha benefit may be absolute-K budget confounding: equal-K alpha vs global/block fusion.',
          'n deterioration may arise from SNV discarding absolute-level information: matched raw/SNV modality controls.',
          'Stability ranking benefit must be separated from computation: independent5xMC20 vs oneMC100, sameK.',
          'NN/SVM gaps are confounded by source190vs240 and resolution128vs512; hold the frequency interval fixed.',
          'Matched480-step transfer benefit may depend on auxiliary herb: HQ-only/TM-only/both with independent origin heads.'],
        'classical':{'jobs':15,'K':[31,62,123,184,245],'MC_fixed':20,'PLS_components_fixed':2,
            'MC_budget_controls':[100,'5independentx20'],'primary_K_selected_by_inner_only':True,
            'fixedK_outer_curves_are_ablation_not_parameter_selection':True,'auxiliary_fair_baselines':['shrinkLDA','MLP']},
        'neural':{'jobs':315,'representations':['alpha128','alpha512'],'same_frequency_range_THz':[.21,1.49],
            'source_counts':[190,240],'auxiliary':['target_only','target_extra','hq_only','tm_only','hq_tm'],
            'target':'pure_acid_origins','profile':'p24_e192_full','pretraining_steps':480,
            'existing_128190_A_Aextra_HQTM':'Reference already run; three conditions excluded from new fits',
            'matched_target_update_controls':'144epochs for source240 vs192epochs/source190 =576updates; alpha128/512 with target_only/hq_tm only',
            'source240_support':'None; the50 formerly reserved TRAIN supports become source. No fewshot claim.',
            'calibration':'Only alpha512/source190/hq_tm: fixed source-only/identity/traditional/weak controls; not real cross-batch.'},
        'recommendation_policy':'Complete repeated5fold pooledOOF groups only; maximize inner macroF1 for per-family selection. Never use CAL/PRED.',
        'paired_uncertainty_unit':'300unique physical sampleIDs; repeated folds/seeds/draws are not new specimens',
        'data_exposure_limit':'Exploratory internal validation on already developed material; new family/profile choices are not external confirmation',
        'old96_0234':'Historical mixed-kind source-group result, not pure-acid origin accuracy',
        'quantum_prior':'NOT_USED; current branches do not establish reliable chemical assignments or IR intensities',
        'stop_rule':'Finite matrix or any resource ceiling. No work added to burn unused grant; no resets/refunds of previous grants.',
        'summary_policy':'Compact streamed counts/tables, summaryJSON at most2MiB; no repeated unrolled support-draw giantJSON.',
        'publication_policy':'New independent Git directory/branch; only closed small evidence, latest compact summary first; no main merges.'}
    freeze(BASE/'protocol.json',protocol)
    shared=[str(features),str(split),str(lock),str(BASE/'protocol.json'),str(BASE/'admission_config.json'),
            *map(str,sources),str(workers['inherited_neural']),str(workers['admission']),str(workers['closed_cache_advice'])]
    def science(identifier,kind,args,output_name):
        out=BASE/'runs'/identifier
        command=[sys.executable,'-B',str(workers[kind]),'--features',str(features),'--split',str(split),*args,'--output',str(out)]
        wrapped=[sys.executable,'-B',str(workers['admission']),'--root',str(BASE),'--output',str(BASE/'admission'/identifier),'--',*command]
        return {'id':identifier,'kind':kind,'command':wrapped,'cwd':str(ROOT),
            'inputs':[*shared,str(workers[kind])],
            'outputs':[str(out/'complete.json'),str(out/'summary.json'),str(out/output_name)],'max_retries':0}
    classic=[science('matched_r%d_f%d'%(repeat,fold),'matched_classical',
                    ['--repeat-seed',str(repeat),'--outer-fold',str(fold)],'validation_predictions.csv.gz')
             for repeat in [20261031,20261108,42]for fold in range(5)]
    conditions=[(rep,count,aux)for rep in ['alpha128','alpha512']for count in [190,240]
                for aux in ['target_only','target_extra','hq_only','tm_only','hq_tm']
                if not(rep=='alpha128'and count==190 and aux in ['target_only','target_extra','hq_tm'])]
    priority=[('alpha128',240,'target_only'),('alpha512',190,'target_only'),('alpha512',240,'target_only'),
              ('alpha128',190,'hq_only'),('alpha128',190,'tm_only')]
    conditions=priority+[c for c in conditions if c not in priority]
    conditions=[(*case,192)for case in conditions]+[(rep,240,aux,144)for rep in ['alpha128','alpha512']for aux in ['target_only','hq_tm']]
    neural=[]
    for rep,count,aux,epochs in conditions:
        for fold in range(5):
            for seed in [42,43,44]:
                args=['--representation',rep,'--source-count',str(count),'--auxiliary',aux,
                      '--outer-fold',str(fold),'--seed',str(seed),'--target-epochs',str(epochs)]
                if(rep,count,aux)==('alpha512',190,'hq_tm'):args+=['--with-calibration']
                neural.append(science('representation_%s_n%d_%s_e%d_f%d_s%d'%(rep,count,aux,epochs,fold,seed),
                                      'neural',args,'predictions.csv.gz'))
    assert len(classic)==15 and len(neural)==315
    ordered=classic[:5]+neural[:15]+classic[5:]+neural[15:]
    points={1,5,20,30,*range(40,len(ordered)+1,10)}
    jobs=[]
    for i,job in enumerate(ordered,1):
        jobs.append(job)
        if i in points:
            for kind in ['summary','backup']:
                out=BASE/('snapshots'if kind=='summary'else'backups')/('%05d'%i)
                command=[sys.executable,'-B',str(workers['summary']),
                         'summarize'if kind=='summary'else'backup','--root',str(BASE),'--output',str(out)]
                if kind=='backup':command+=['--execute']
                names=['summary.json','metrics.csv','REPORT.md']if kind=='summary'else['receipt.json','complete.json']
                jobs.append({'id':'evidence_%s_%05d'%(kind,i),'kind':kind,'command':command,'cwd':str(ROOT),
                    'inputs':[str(BASE/'protocol.json'),str(BASE/'jobs.json'),str(lock),str(workers['summary'])],
                    'outputs':[str(out/name)for name in names],'max_retries':0 if kind=='backup'else 1})
    manifest={'run_id':RUN,'cpu_budget_seconds':88000,'authorized_total_cpu_seconds':90000,
              'protocol_path':str(BASE/'protocol.json'),'protocol_sha256':sha(BASE/'protocol.json'),'jobs':jobs}
    freeze(BASE/'jobs.json',manifest)
    print(json.dumps({'status':'PREPARED_NOT_LAUNCHED','science_jobs':330,'classic':15,'neural':315,
                     'summaries':len(points),'backups':len(points),'all_jobs':len(jobs),'manifest_sha256':sha(BASE/'jobs.json')}))


if __name__=='__main__':main()
