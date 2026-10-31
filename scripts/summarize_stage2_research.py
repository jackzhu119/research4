"""Export stage-two exploratory evidence without mirroring raw spectral inputs."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from herb_spectra.benchmark import evaluation_metrics, write_json

REPO=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text())


def copy_if(source,target):
    source=Path(source)
    if source.exists():
        target=Path(target);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classification',type=Path,required=True)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--transfer-root',type=Path,default=REPO/'outputs/research-stage2')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plots',action='store_true')
    args=parser.parse_args();out=args.output
    if out.exists() and any(out.iterdir()):parser.error('Choose a new/empty output directory')
    rows=read(args.classification/'results.json')
    if len(rows)!=200 or any(r['status']!='pass' for r in rows):raise ValueError('Incomplete200-case classification experiment')
    specs={r['id']:r for r in read(args.spec)['datasets']}
    out.mkdir(parents=True,exist_ok=True);groups=defaultdict(list);predictions=[]
    for row in rows:
        spec=specs[row['id']];path=args.classification/row['id']/row['predictions']
        with path.open() as handle:values=list(csv.DictReader(handle))
        ids=[v['sample_id'] for v in values];truth=np.asarray([float(v['y_true']) for v in values]);pred=np.asarray([float(v['y_predicted']) for v in values])
        with np.load(spec['data_path'],allow_pickle=False) as data:
            if ids!=data['test_ids'].astype(str).tolist() or not np.array_equal(truth,data['y_test']):raise ValueError('Saved test identities or labels changed')
        metrics=evaluation_metrics(truth,pred,'classification',np.unique(truth))
        for k,v in metrics.items():
            if not np.isclose(v,row['metrics'][k],rtol=1e-10,atol=1e-10):raise ValueError('Classification metric mismatch')
        groups[(row['dataset'],row['model'])].append((row,ids,truth,pred))
        predictions.extend({'dataset':row['dataset'],'model':row['model'],'fold':row['fold'],'row_id':i,'y_true':float(t),'y_predicted':float(p)} for i,t,p in zip(ids,truth,pred))
    pooled=[]
    for (dataset,model),values in sorted(groups.items()):
        if sorted(v[0]['fold'] for v in values)!=[1,2,3,4,5]:raise ValueError('Missing or duplicate outer fold')
        ids=[i for v in values for i in v[1]]
        if len(ids)!=len(set(ids)):raise ValueError('A source spectrum was tested more than once')
        truth=np.concatenate([v[2] for v in values]);pred=np.concatenate([v[3] for v in values])
        pooled.append({'dataset':dataset,'model':model,'n_unique_test_spectra':len(ids),
                       'metrics':evaluation_metrics(truth,pred,'classification',np.unique(truth)),
                       'fold_accuracy_sd_ddof1':float(np.std([v[0]['metrics']['accuracy'] for v in values],ddof=1)),
                       'selected_features_mean':float(np.mean([v[0]['selected_feature_count'] for v in values])),
                       'fit_seconds_sum':float(sum(v[0]['fit_seconds'] for v in values)),
                       'interpretation':'Descriptive row-level5fold benchmark; SD is not CI or batch-generalization evidence'})
    write_json(out/'classification_pooled.json',pooled)
    with (out/'classification_pooled.csv').open('w',newline='') as f:
        flat=[{k:v for k,v in r.items() if k not in {'metrics','interpretation'}}|r['metrics'] for r in pooled]
        w=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');w.writeheader();w.writerows(flat)
    with (out/'classification_predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(predictions[0]),lineterminator='\n');w.writeheader();w.writerows(predictions)
    for name in ['results.json','manifest.json','independent_review.json','metadata_correction.json',
                 'execution_results_before_scope_correction.json.gz']:
        copy_if(args.classification/name,out/'classification'/name)
    code_dir=args.classification/'execution_code'
    if code_dir.exists():shutil.copytree(code_dir,out/'classification/execution_code')
    inner={key:read(args.classification/key/'inner_folds.json') for key in specs}
    write_json(out/'classification/inner_folds.json',inner)
    transfers=[]
    for name in ['pharma','wheat_A','wheat_B']:
        source=args.transfer_root/('transfer_'+name+'_final');manifest=read(source/'manifest.json')
        if not manifest['all_selected_passed'] or manifest['actual_cases']!=2400:raise ValueError('Incomplete transfer experiment:'+name)
        target=out/('transfer_'+name);target.mkdir()
        for filename in ['manifest.json','summary.json','summary.csv','results.json','query_predictions.npz','split_audit.json','source_metadata.json','independent_review.json']:
            copy_if(source/filename,target/filename)
        for representation in ['full','mcuve','mcuve100','stable_mcuve']:
            copy_if(source/representation/'source_model.json',target/representation/'source_model.json')
        transfers.extend({'dataset':name,**r} for r in read(source/'summary.json'))
    write_json(out/'transfer_summary.json',transfers)
    assets={
        'inputs/herb_audit.json':args.spec.parent/'audit.json',
        'inputs/herb_source_registry.json':args.spec.parent/'source_registry.json',
        'inputs/pharma_audit.json':args.transfer_root/'pharma/audit.json',
        'inputs/wheat_audit.json':args.transfer_root/'wheat/audit.json',
        'inputs/wheat_source_registry.json':args.transfer_root/'wheat/source_registry.json',
        'inputs/wheat_rebuild_validation.json':args.transfer_root/'wheat_rebuild_validation.json',
        'inputs/qsi_array_audit.json':args.transfer_root/'qsi_array_audit.json',
        'inputs/AMM_array_audit.json':args.transfer_root/'AMM_array_audit.json',
        'unit_tests.log':REPO/'outputs/research-stage2/unit_tests.log',
        'method_audit.md':args.transfer_root/'method_audit.md',
    }
    for name,source in assets.items():copy_if(source,out/name)
    summary={'created_at_utc':datetime.now(timezone.utc).isoformat(),'classification_scenarios':200,
             'transfer_evaluation_scenarios':7200,'all_selected_paths_passed':True,
             'source_catalog_records':34,'catalog_records_are_not_independent_cohorts':True,
             'transfer_scenarios_are_repeated_calibration_evaluations_not7200_independent_source_trainings':True,
             'wheat_A_B_same150_target_records':True,'new_herb_batch_generalization_verified':False,
             'new_method_novelty_established':False,'raw_spectral_arrays_mirrored':False}
    if args.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        summary['matplotlib_version']=matplotlib.__version__
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for ax,name,title in zip(axes,['pharma','wheat_A','wheat_B'],['Tablets: instrument/sample-pool shift','Wheat: new instrument A / new samples','Wheat: new instrument B / same new samples']):
            for method,label in [('none','Source model'),('bias_unshrunk','Bias correction'),('affine_shrinkage','Anchored affine'),('source_plus_support','Joint PLS head')]:
                chosen=sorted([r for r in transfers if r['dataset']==name and r['representation']=='full' and r['method']==method],key=lambda x:x['shots'])
                ax.plot([r['shots'] for r in chosen],[r['metric_mean']['rmse'] for r in chosen],marker='o',label=label)
            ax.set_title(title,fontsize=9);ax.set_xlabel('Labeled target records');ax.grid(alpha=.2)
            ax.set_ylabel('RMSE (mg/tablet)' if name=='pharma' else 'RMSE (source label units unspecified)')
        axes[-1].legend(fontsize=8);fig.suptitle('Full-spectrum PLS: mean across20 support draws; shared query within each repeat')
        fig.tight_layout();fig.savefig(out/'calibration_curves.png',dpi=200);fig.savefig(out/'calibration_curves.pdf');plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        methods=['svc','plsda','mcuve20_svc','mcuve100_svc','stable_mcuve_svc']
        for ax,dataset in zip(axes.flat,sorted(set(r['dataset'] for r in pooled))):
            items={r['model']:r for r in pooled if r['dataset']==dataset}
            ax.bar(range(5),[items[m]['metrics']['balanced_accuracy']*100 for m in methods],color=['#2878a5','#348c65','#ae7d35','#8e66a3','#b45767'])
            ax.set_xticks(range(5),['SVM','PLS-DA','MC20\n+ SVM','MC100\n+ SVM','Stable MC\n+ SVM'],fontsize=8)
            ax.set_title(dataset);ax.set_ylim(0,100);ax.set_ylabel('Balanced accuracy (%)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        fig.suptitle('New herb source rows: five held-out folds; exact duplicates collapsed')
        fig.tight_layout();fig.savefig(out/'stability_classification.png',dpi=200);fig.savefig(out/'stability_classification.pdf');plt.close(fig)
    code=['src/herb_spectra/stability.py','src/herb_spectra/calibration.py','scripts/run_stability_research.py','scripts/run_transfer_research.py','scripts/prepare_stage2_herb_benchmarks.py','scripts/prepare_transfer_benchmark.py','scripts/prepare_wheat_transfer.py','scripts/summarize_stage2_research.py']
    write_json(out/'current_code_sha256.json',[{'path':p,'sha256':hashlib.sha256((REPO/p).read_bytes()).hexdigest()} for p in code])
    write_json(out/'stage_summary.json',summary)
    compressed=[]
    for path in sorted(out.rglob('*.json')):
        if path.stat().st_size<100_000 or path.name in {'classification_pooled.json','transfer_summary.json'}:continue
        raw=path.read_bytes();saved=path.with_suffix('.json.gz');saved.write_bytes(gzip.compress(raw,mtime=0));path.unlink()
        compressed.append({'original_path':str(path.relative_to(out)),'saved_path':str(saved.relative_to(out)),
                           'original_sha256':hashlib.sha256(raw).hexdigest(),'original_bytes':len(raw)})
    write_json(out/'compressed_artifacts.json',compressed)
    write_json(out/'files_sha256.json',[{'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(out.rglob('*')) if p.is_file() and p.name!='files_sha256.json'])
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
