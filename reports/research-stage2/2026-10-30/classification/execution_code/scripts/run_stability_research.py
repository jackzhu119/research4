"""Compare fixed spectral classification paths and repeated-variable selection.

Official train/test partitions are supplied by a source-backed NPZ spec. Every
preprocessor, scaler, selector and classifier is refitted inside training-only
three-fold model selection. Frequencies/Jaccard describe resampling of training
rows; without physical IDs they are not proof of sample/batch independence.

Stable MC-UVE retains the same count as ordinary MC-UVE. The 100-run MC-UVE is
an additional-effort comparator: ten stable-selector tie-break coefficient fits
and different sampling mean that effective compute is not exactly identical.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import warnings

import numpy as np
from joblib import Memory
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from herb_spectra.benchmark import (PLSDAClassifier, evaluation_metrics,
                                    training_folds, validate_arrays, write_json)
from herb_spectra.selection import CARSStyleSelector, MCUVESelector
from herb_spectra.stability import StableVariableSelector
from herb_spectra.validation import RowPreprocessor


REPO = Path(__file__).resolve().parents[1]
MODEL_NAMES = ('svc', 'plsda', 'cars_svc', 'cars_plsda',
               'mcuve20_svc', 'mcuve20_plsda', 'mcuve100_svc', 'mcuve100_plsda',
               'stable_mcuve_svc', 'stable_mcuve_plsda')
CODE_PATHS = ('scripts/run_stability_research.py', 'src/herb_spectra/benchmark.py',
              'src/herb_spectra/stability.py', 'src/herb_spectra/selection.py',
              'src/herb_spectra/validation.py')


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_research_pipeline(name, features, min_inner_rows, modes, seed=42, memory=None):
    if name not in MODEL_NAMES:raise ValueError('Unknown classification research model: '+name)
    if not modes or not set(modes).issubset({'raw', 'snv'}):
        raise ValueError('preprocess_modes must be a nonempty list of raw/snv')
    selector = 'passthrough'; retained = features; budget = None
    if name.startswith('cars_'):
        retained = min(2, features)
        selector = CARSStyleSelector(n_runs=12, n_components=2,
                                     min_features=retained, cv=3,
                                     task='classification', random_state=seed)
        budget = {'method':'independent_cars_style', 'n_runs':12,
                  'n_components':2, 'internal_cv':3,
                  'final_feature_count':'selected by internal training search'}
    elif name.startswith(('mcuve20_', 'mcuve100_', 'stable_mcuve_')):
        retained = min(features, max(5, int(np.ceil(0.1*features))))
        if name.startswith('stable_mcuve_'):
            selector = StableVariableSelector(base_selector='mcuve',
                         n_features_to_select=retained, n_resamples=10,
                         sample_fraction=.8, inner_runs=10, n_components=2,
                         task='classification', random_state=seed)
            budget = {'method':'repeated_mcuve_frequency', 'n_resamples':10,
                      'inner_runs':10, 'n_components':2, 'outer_sample_fraction':.8,
                      'base_coefficient_resamples':100, 'tie_break_coefficient_fits':10,
                      'retained_feature_count':retained}
        else:
            runs = 100 if name.startswith('mcuve100_') else 20
            selector = MCUVESelector(n_runs=runs, n_components=2,
                         n_features_to_select=retained, task='classification', random_state=seed)
            budget = {'method':'independent_mcuve', 'n_runs':runs, 'n_components':2,
                      'base_coefficient_resamples':runs, 'tie_break_coefficient_fits':0,
                      'retained_feature_count':retained}
    grid = {'preprocess__mode':list(modes)}
    if name.endswith('svc'):
        model = SVC(class_weight='balanced', gamma='scale')
        grid['model__C'] = [.1, 1., 10., 100.]
    else:
        model = PLSDAClassifier()
        maximum = min(min_inner_rows-1, retained)
        grid['model__n_components'] = [n for n in [1,2,3,5,8] if n<=maximum]
        if not grid['model__n_components']:raise ValueError('No feasible PLS components')
    pipeline = Pipeline([('preprocess',RowPreprocessor(modes[0])),
                         ('scale',StandardScaler()),('selector',selector),('model',model)],
                        memory=memory)
    return pipeline, grid, budget


def run_dataset(spec, output, seed=42, models=None):
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    if spec.get('task')!='classification':raise ValueError('Only classification tasks are supported')
    data_hash = sha256(spec['data_path'])
    if spec.get('data_sha256') and spec['data_sha256']!=data_hash:
        raise ValueError('Prepared NPZ hash differs from source spec')
    with np.load(spec['data_path'],allow_pickle=False) as data:
        arrays = {key:data[key] for key in data.files}
    X,y,test_X,test_y,train_ids,test_ids,hashes,audit = validate_arrays(arrays,'classification')
    audit['provided_id_scope'] = spec.get('identity_scope','unspecified by source manifest')
    folds = training_folds(X,y,'classification',hashes,seed=seed,n_splits=3)
    write_json(output/'quality.json',audit)
    write_json(output/'inner_folds.json',[{'train_ids':train_ids[t].tolist(),
                                          'validation_ids':train_ids[v].tolist()} for t,v in folds])
    chosen = list(MODEL_NAMES) if models is None else list(models)
    if not chosen or len(chosen)!=len(set(chosen)) or not set(chosen).issubset(MODEL_NAMES):
        raise ValueError('Select unique supported model names')
    code_hashes = {path:sha256(REPO/path) for path in CODE_PATHS}
    classes = np.unique(y); results = []
    for name in chosen:
        print(f"Running {spec['id']} / {name}",flush=True)
        case = output/name; case.mkdir(exist_ok=True)
        row = {k:spec[k] for k in ['id','dataset','fold','herb','task','source_url','source_scope',
                                   'source_commit','source_dataset','target_name','target_unit',
                                   'input_transform','split_policy','identity_scope','class_names'] if k in spec}
        row.update({'model':name,'status':'fail','seed':seed,'data_sha256':data_hash,
                    'code_sha256':code_hashes,'train_rows':len(X),'test_rows':len(test_X),
                    'input_features':X.shape[1],'inner_cv_folds':len(folds),
                    'implementation':'Repository independent CARS-style/MC-UVE and repeated-selection baseline; not native archived MATLAB reproduction',
                    'comparison_scope':'Same data splits and classifier grids; matching final MC-UVE dimensions, not identical effective compute',
                    'selection_diagnostics_scope':'Training-row resampling only; no p-values, FDR control or batch robustness claim',
                    'cache_scope':'Per-case content-hashed training arrays and parameters; no query fit or fitted-selector reuse across different training folds'})
        start=time.monotonic()
        try:
            cache = Memory(location=case/'.cache',verbose=0)
            pipeline,grid,budget=make_research_pipeline(name,X.shape[1],min(len(t) for t,_ in folds),
                             spec.get('preprocess_modes',['raw','snv']),seed,cache)
            row['selector_budget']=budget
            search=GridSearchCV(pipeline,grid,cv=folds,n_jobs=1,scoring='balanced_accuracy',
                                error_score='raise',return_train_score=False)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                search.fit(X,y)
                row['fit_seconds']=round(time.monotonic()-start,6)
                prediction_start=time.monotonic()
                predicted=np.asarray(search.predict(test_X)).reshape(-1)
                row['predict_seconds']=round(time.monotonic()-prediction_start,6)
            warning_counts=Counter((str(v.category.__name__),str(v.message)) for v in caught)
            write_json(case/'warnings.json',[{'category':k[0],'message':k[1],'count':n}
                                             for k,n in warning_counts.items()])
            row['warning_count']=len(caught)
            row['best_params']=search.best_params_
            row['candidate_count']=len(search.cv_results_['params'])
            row['downstream_cv_fit_count']=row['candidate_count']*len(folds)
            row['best_inner_cv_score']=float(search.best_score_)
            write_json(case/'cv_candidates.json',[{'params':p,'mean_validation_balanced_accuracy':float(mean),
                    'sd_validation_balanced_accuracy':float(sd)} for p,mean,sd in zip(
                        search.cv_results_['params'],search.cv_results_['mean_test_score'],search.cv_results_['std_test_score'])])
            selector=search.best_estimator_.named_steps['selector']
            indices=np.arange(X.shape[1]) if isinstance(selector,str) else selector.selected_indices_
            row['selected_feature_count']=len(indices)
            row['selected_feature_indices_zero_based']=indices.tolist()
            if isinstance(selector,StableVariableSelector):
                diagnostics={'training_unit_count':selector.training_unit_count_,
                    'sample_unit':selector.sample_unit_,'frequency':selector.frequency_.tolist(),
                    'mean_coefficient_importance':selector.mean_coefficient_importance_.tolist(),
                    'base_selected_feature_counts':selector.base_selected_feature_counts_.tolist(),
                    'resample_unit_indices':[a.tolist() for a in selector.resample_unit_indices_],
                    'resample_seeds':selector.resample_seeds_.tolist(),
                    'pairwise_jaccard':selector.pairwise_jaccard_.tolist(),
                    'mean_pairwise_jaccard':selector.mean_pairwise_jaccard_,
                    'scope':row['selection_diagnostics_scope']}
                write_json(case/'stability_diagnostics.json',diagnostics)
                row['mean_training_resample_jaccard']=selector.mean_pairwise_jaccard_
            row['metrics']=evaluation_metrics(test_y,predicted,'classification',classes)
            row['class_order']=classes.tolist()
            row['confusion_matrix']=confusion_matrix(test_y,predicted,labels=classes).tolist()
            row['per_class_metrics']=classification_report(test_y,predicted,labels=classes,output_dict=True,zero_division=0)
            with (case/'predictions.csv').open('w',newline='') as handle:
                writer=csv.writer(handle);writer.writerow(['sample_id','y_true','y_predicted'])
                writer.writerows(zip(test_ids.tolist(),test_y.tolist(),predicted.tolist()))
            row['predictions']=str((case/'predictions.csv').relative_to(output))
            row['status']='pass'
        except Exception as exc:
            row['error']=f'{type(exc).__name__}: {exc}'
            row['elapsed_seconds']=round(time.monotonic()-start,6)
        finally:
            shutil.rmtree(case/'.cache',ignore_errors=True)
        write_json(case/'result.json',row);results.append(row)
        write_json(output/'results.json',results)
        print(f"Completed {spec['id']} / {name}: {row['status']}",flush=True)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--models',nargs='+',choices=MODEL_NAMES,
                        help='Explicit subset for engineering checks; omit for ten-model research protocol')
    args=parser.parse_args()
    if args.seed<0:parser.error('seed must be nonnegative')
    if args.output.exists() and any(args.output.iterdir()):parser.error('Choose an empty/new output directory')
    content=json.loads(args.spec.read_text());specs=content['datasets'] if isinstance(content,dict) else content
    if not isinstance(specs,list) or not specs:parser.error('Spec must contain a nonempty task list')
    ids=[s['id'] for s in specs]
    if len(ids)!=len(set(ids)):parser.error('Dataset/fold IDs must be unique')
    if any(Path(v).name!=v or v in {'.','..'} for v in ids):parser.error('IDs must be plain directory names')
    if any(s.get('task')!='classification' for s in specs):parser.error('Use a classification-only spec')
    args.output.mkdir(parents=True,exist_ok=True)
    manifest={'started_at_utc':datetime.now(timezone.utc).isoformat(),'seed':args.seed,
              'spec_path':str(args.spec.resolve()),'spec_sha256':sha256(args.spec),
              'selected_models':list(MODEL_NAMES) if args.models is None else args.models,
              'full_ten_model_protocol':args.models is None,
              'code_sha256':{p:sha256(REPO/p) for p in CODE_PATHS},
              'versions':{p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','joblib']},
              'datasets':[]}
    results=[]
    for raw in specs:
        spec=dict(raw);path=Path(spec['data_path'])
        spec['data_path']=str((path if path.is_absolute() else args.spec.resolve().parent/path).resolve())
        try:
            if not spec.get('source_url'):raise ValueError('Each task needs source_url')
            data_hash=sha256(spec['data_path'])
            manifest['datasets'].append({'id':spec['id'],'data_path':spec['data_path'],'sha256':data_hash})
            results.extend(run_dataset(spec,args.output/spec['id'],args.seed,args.models))
        except Exception as exc:
            results.append({'id':spec['id'],'model':'_dataset_validation','status':'fail',
                            'task':'classification','error':f'{type(exc).__name__}: {exc}'})
        write_json(args.output/'results.json',results);write_json(args.output/'manifest.json',manifest)
    manifest['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    expected=len(specs)*len(manifest['selected_models'])
    manifest['expected_cases']=expected;manifest['completed_cases']=len(results)
    manifest['all_selected_passed']=len(results)==expected and all(r['status']=='pass' for r in results)
    write_json(args.output/'manifest.json',manifest)
    print(json.dumps({'all_selected_passed':manifest['all_selected_passed'],
                      'cases':len(results),'expected_cases':expected,'output':str(args.output)},indent=2))
    return 0 if manifest['all_selected_passed'] else 1


if __name__=='__main__':sys.exit(main())
