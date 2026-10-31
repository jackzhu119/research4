"""Recompute pooled out-of-fold metrics and export inspectable benchmark evidence.

Each source spectrum must occur once per target/model in the five held-out folds.
Pooled R2 is recomputed from all held-out predictions, never averaged across folds.
Fold SD describes variation between these splits, not a confidence interval.
"""
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
from sklearn.metrics import classification_report, confusion_matrix

from herb_spectra.benchmark import evaluation_metrics, write_json


REPO = Path(__file__).resolve().parents[1]
LABELS = {'dummy': 'Dummy', 'svc': 'SVM', 'plsda': 'PLS-DA',
          'cars_svc': 'CARS-style + SVM', 'mcuve_svc': 'MC-UVE + SVM',
          'pls': 'PLS', 'svr': 'SVR', 'cars_pls': 'CARS-style + PLS',
          'mcuve_pls': 'MC-UVE + PLS'}


def read(path):return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classification', type=Path, required=True)
    parser.add_argument('--regression', type=Path, required=True)
    parser.add_argument('--octave', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plots', action='store_true', help='Export static scientific figures; requires matplotlib')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):parser.error('Choose a new/empty output directory')
    specs = {s['id']: s for s in read(args.spec)['datasets']}
    groups = defaultdict(list); case_counts = {}; all_predictions = []; quality = {}; warnings = {}
    args.output.mkdir(parents=True, exist_ok=True)
    for backend, source in [('python_classification', args.classification),
                            ('python_regression', args.regression), ('octave', args.octave)]:
        rows = read(source/'results.json')
        if any(r['status'] != 'pass' for r in rows):raise ValueError('Failure in '+backend)
        case_counts[backend] = len(rows)
        target = args.output/backend; target.mkdir()
        shutil.copyfile(source/'results.json', target/'results.json')
        for name in ['manifest.json', 'run_metadata.json', 'timeout_process_group_check.json']:
            if (source/name).exists():shutil.copyfile(source/name, target/name)
        seen = set()
        for row in rows:
            native = backend == 'octave'
            spec_id = row['dataset_id'] if native else row['id']
            spec = specs[spec_id]
            model = row['method'] if native else row['model']
            if (spec_id, model) in seen:raise ValueError('Duplicate case')
            seen.add((spec_id, model))
            case = source/row['id'] if native else source/spec_id
            prediction_path = case/'predictions.csv' if native else case/row['predictions']
            with prediction_path.open() as handle:predictions = list(csv.DictReader(handle))
            ids = [r['test_id' if native else 'sample_id'] for r in predictions]
            truth = np.asarray([float(r['y_true']) for r in predictions])
            predicted = np.asarray([float(r['y_pred' if native else 'y_predicted']) for r in predictions])
            with np.load(spec['data_path'], allow_pickle=False) as array:
                if ids != array['test_ids'].astype(str).tolist() or not np.array_equal(truth, array['y_test']):
                    raise ValueError('Prediction identities/targets differ from held-out NPZ')
            classes = np.unique(truth) if spec['task']=='classification' else None
            recomputed = evaluation_metrics(truth, predicted, spec['task'], classes)
            for key, value in recomputed.items():
                native_key = 'bias' if native and key=='bias_pred_minus_true' else key
                if not isinstance(value, (int, float)) or native_key not in row['metrics']:continue
                if not np.isclose(value, row['metrics'][native_key], rtol=1e-10, atol=1e-10):
                    raise ValueError('Stored metric does not match saved predictions: '+key)
            group = (backend, spec['dataset'], model)
            groups[group].append((row, spec, ids, truth, predicted, recomputed))
            for key, true, pred in zip(ids, truth, predicted):
                all_predictions.append({'backend': backend, 'dataset': spec['dataset'],
                                        'target_name': spec['target_name'], 'model': model,
                                        'fold': spec['fold'], 'row_hash': key,
                                        'y_true': float(true), 'y_predicted': float(pred)})
            if native:
                exported = target/row['id'];exported.mkdir()
                for name in ['check.m', 'adaptation.json', 'run.log']:
                    shutil.copyfile(case/name, exported/name)
            elif spec_id not in quality:
                quality[spec_id] = {'quality': read(case/'quality.json'),
                                    'inner_folds': read(case/'inner_folds.json')}
            if not native:
                warnings[spec_id+'/'+model] = read(case/model/'warnings.json')
    summary = []
    for (backend, dataset, model), values in sorted(groups.items()):
        values.sort(key=lambda x: x[1]['fold'])
        if [v[1]['fold'] for v in values] != [1,2,3,4,5]:raise ValueError('Missing official outer fold')
        spec = values[0][1]
        ids = [key for v in values for key in v[2]]
        if len(ids) != len(set(ids)):raise ValueError('A spectrum was tested more than once')
        truth = np.concatenate([v[3] for v in values]); predicted = np.concatenate([v[4] for v in values])
        classes = np.unique(truth) if spec['task']=='classification' else None
        metrics = evaluation_metrics(truth, predicted, spec['task'], classes)
        numeric = [k for k,v in values[0][5].items() if isinstance(v, (int,float))]
        item = {'backend': backend, 'dataset': dataset, 'herb': spec['herb'], 'task': spec['task'],
                'target_name': spec['target_name'], 'target_unit': spec['target_unit'],
                'model': model, 'oof_rows': len(truth), 'folds': 5,
                'each_spectrum_tested_once': True, 'pooled_oof_metrics': metrics,
                'fold_mean': {k:float(np.mean([v[5][k] for v in values])) for k in numeric},
                'fold_sd_ddof1': {k:float(np.std([v[5][k] for v in values],ddof=1)) for k in numeric},
                'selected_features_mean': float(np.mean([v[0]['n_selected' if backend=='octave' else 'selected_feature_count'] for v in values])),
                'limitation': 'Source spectrum hashes, not verified physical samples or independent batches; outer training sets overlap'}
        if classes is not None:
            item['class_names'] = spec['class_names']
            item['confusion_matrix'] = confusion_matrix(truth,predicted,labels=classes).tolist()
            item['per_class_metrics'] = classification_report(truth,predicted,labels=classes,output_dict=True,zero_division=0)
        summary.append(item)
    if case_counts != {'python_classification':50,'python_regression':250,'octave':30}:
        raise ValueError('Incomplete preregistered benchmark case coverage')
    write_json(args.output/'pooled_metrics.json', summary)
    write_json(args.output/'quality_and_inner_splits.json', quality)
    write_json(args.output/'model_warnings.json', warnings)
    write_json(args.output/'prepared_spec.json', {'datasets':list(specs.values())})
    with (args.output/'all_held_out_predictions.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(all_predictions[0]));writer.writeheader();writer.writerows(all_predictions)
    flat=[]
    for row in summary:
        flat.append({**{k:row[k] for k in ['backend','dataset','target_name','target_unit','model','oof_rows']},
                     **{k:v for k,v in row['pooled_oof_metrics'].items() if isinstance(v,(int,float))}})
    with (args.output/'pooled_metrics.csv').open('w',newline='') as handle:
        fields=list(dict.fromkeys(k for r in flat for k in r));writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader();writer.writerows(flat)
    copied = {'source_manifest.json':REPO/'data/external/ultrair_sources.json',
              'source_metadata.json':REPO/'data/external/ultrair_metadata.json',
              'source_array_audit.json':REPO/'outputs/external-discovery/herb/ultrair/array_audit.json',
              'preparation_independent_review.json':REPO/'outputs/external-discovery/herb/preparation_independent_review.json',
              'classification_independent_review.json':REPO/'outputs/external-classification-independent-review.json',
              'regression_independent_review.json':REPO/'outputs/external-regression-independent-review.json',
              'unit_tests.log':REPO/'outputs/external-benchmark-unit-tests.log',
              'additional_discovery.md':REPO/'outputs/external-discovery/public/discovery.md',
              'additional_provenance.json':REPO/'outputs/external-discovery/public/provenance.json',
              'additional_downloads.json':REPO/'outputs/external-discovery/public/downloads.json'}
    for name,source in copied.items():
        if source.exists():shutil.copyfile(source,args.output/name)
    code_paths=['scripts/prepare_external_benchmarks.py','scripts/run_external_benchmarks.py',
                'scripts/runtime/run_external_octave.py','scripts/summarize_external_benchmarks.py',
                'src/herb_spectra/benchmark.py','src/herb_spectra/selection.py','src/herb_spectra/validation.py']
    write_json(args.output/'code_sha256.json',[{'path':p,'sha256':hashlib.sha256((REPO/p).read_bytes()).hexdigest()} for p in code_paths])
    info={'exported_at_utc':datetime.now(timezone.utc).isoformat(),'case_counts':case_counts,
          'all_cases_passed':True,'pooled_model_target_groups':len(summary),
          'physical_sample_or_batch_independence_verified':False,'absolute_concentration_prediction_verified':False,
          'fold_sd_is_not_confidence_interval':True,'source_commit':next(iter(specs.values()))['source_commit']}
    if args.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        info['matplotlib_version']=matplotlib.__version__
        fig,axes=plt.subplots(1,2,figsize=(12,4.2),sharey=True)
        order=['svc','plsda','cars_svc','mcuve_svc']
        for ax,herb,name in zip(axes,['jyh','syh'],['Jinyinhua: 4 origins, n=120','Shanyinhua: 5 origins, n=150']):
            subset={r['model']:r for r in summary if r['task']=='classification' and r['herb']==herb}
            ax.bar(np.arange(4),[subset[m]['pooled_oof_metrics']['accuracy']*100 for m in order],
                   yerr=[subset[m]['fold_sd_ddof1']['accuracy']*100 for m in order],capsize=4,
                   color=['#2b6f9c','#24846a','#af7434','#8c639f'])
            ax.axhline(subset['dummy']['pooled_oof_metrics']['accuracy']*100,color='#666',linestyle='--',label='Majority baseline')
            ax.set_xticks(np.arange(4),[LABELS[m].replace(' + ','\n+ ') for m in order],fontsize=9)
            ax.set_title(name);ax.set_ylim(0,100);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        axes[0].set_ylabel('Held-out accuracy (%)');axes[1].legend(loc='lower right',fontsize=8)
        fig.suptitle('Public FTIR benchmark: official five outer folds; error bars = fold SD')
        fig.tight_layout();fig.savefig(args.output/'classification_accuracy.png',dpi=200);fig.savefig(args.output/'classification_accuracy.pdf');plt.close(fig)
        regression=[r for r in summary if r['backend']=='python_regression']
        tasks=sorted(set(r['dataset'] for r in regression));models=['pls','svr','cars_pls','mcuve_pls']
        matrix=np.asarray([[next(r for r in regression if r['dataset']==d and r['model']==m)['pooled_oof_metrics']['r2'] for m in models] for d in tasks])
        names=[next(r for r in regression if r['dataset']==d) for d in tasks]
        fig,ax=plt.subplots(figsize=(9,7));im=ax.imshow(matrix,cmap='RdYlGn',vmin=-1,vmax=1,aspect='auto')
        ax.set_xticks(np.arange(4),[LABELS[m] for m in models]);ax.set_yticks(np.arange(len(tasks)),[r['herb'].upper()+': '+r['target_name'] for r in names])
        for i in range(len(tasks)):
            for j in range(4):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=9)
        ax.set_title('Pooled held-out R²: LC-MS relative abundance (a.u.)\nColor scale clipped to [-1, 1]; numbers retain actual values')
        fig.colorbar(im,ax=ax,label='R²');fig.tight_layout();fig.savefig(args.output/'regression_r2.png',dpi=200);fig.savefig(args.output/'regression_r2.pdf');plt.close(fig)
    write_json(args.output/'benchmark_summary.json',info)
    compressed=[]
    for path in sorted(args.output.rglob('*.json')):
        if path.stat().st_size < 100_000 or path.name=='pooled_metrics.json':continue
        raw=path.read_bytes();saved=path.with_suffix('.json.gz')
        saved.write_bytes(gzip.compress(raw,mtime=0));path.unlink()
        compressed.append({'original_path':str(path.relative_to(args.output)),
                           'saved_path':str(saved.relative_to(args.output)),
                           'original_bytes':len(raw),'original_sha256':hashlib.sha256(raw).hexdigest()})
    write_json(args.output/'compressed_artifacts.json',compressed)
    write_json(args.output/'files_sha256.json',[{'path':str(p.relative_to(args.output)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
               for p in sorted(args.output.rglob('*')) if p.is_file() and p.name!='files_sha256.json'])
    print(json.dumps(info,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
