"""Train-only variable selection and small-support cross-instrument regression.

All source parameters are selected using source-only inner CV. Target supports
are nested prefixes of a pre-drawn pool; the remaining query is identical for
every shot count and representation in the same repeat. Query never fits a
transform, selector, source parameter, calibration coefficient or regularizer.
This is an engineering comparison, not a novelty or independence claim.
"""
import argparse
from collections import defaultdict
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

from joblib import Memory
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array

from herb_spectra.benchmark import evaluation_metrics, write_json
from herb_spectra.calibration import ShrinkageAffineCalibrator
from herb_spectra.selection import MCUVESelector
from herb_spectra.stability import StableVariableSelector
from herb_spectra.validation import RowPreprocessor


REPO = Path(__file__).resolve().parents[1]
REPRESENTATIONS = ('full', 'mcuve', 'mcuve100', 'stable_mcuve')
METHODS = ('none', 'bias_unshrunk', 'affine_unshrunk', 'affine_shrinkage',
           'target_support_only', 'source_plus_support')


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_input(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {name:data[name] for name in data.files}
    names = ['X_source','y_source','source_ids','X_target','y_target','target_ids']
    if any(name not in arrays for name in names):raise ValueError('Missing transfer arrays: '+', '.join(names))
    for domain in ['source','target']:
        arrays['X_'+domain] = check_array(arrays['X_'+domain], dtype=float, ensure_min_samples=6 if domain=='source' else 3)
        y = np.asarray(arrays['y_'+domain],dtype=float)
        ids = np.asarray(arrays[domain+'_ids']).astype(str)
        if y.ndim!=1 or len(y)!=len(arrays['X_'+domain]) or not np.isfinite(y).all():
            raise ValueError('One finite regression target per '+domain+' row is required')
        if ids.ndim!=1 or len(ids)!=len(y) or len(set(ids))!=len(ids):
            raise ValueError('Unique IDs are required for each '+domain+' record')
        if any(v.strip().lower() in {'','nan','none'} for v in ids):raise ValueError('Missing record ID')
        arrays['y_'+domain],arrays[domain+'_ids']=y,ids
    if arrays['X_source'].shape[1]!=arrays['X_target'].shape[1]:raise ValueError('Source and target feature grids differ')
    if set(arrays['source_ids']) & set(arrays['target_ids']):raise ValueError('Source records overlap target pool')
    if np.std(arrays['y_source']) <= np.finfo(float).eps:raise ValueError('Source target has no variation')
    rowhashes = {}
    for domain in ['source','target']:
        rowhashes[domain]=[hashlib.sha256(np.asarray(v,dtype='<f8').tobytes()).hexdigest() for v in arrays['X_'+domain]]
        if len(set(rowhashes[domain]))!=len(rowhashes[domain]):raise ValueError('Duplicate spectra in '+domain+'; aggregate/review record identities first')
    if set(rowhashes['source']) & set(rowhashes['target']):raise ValueError('Exactly identical spectrum crosses source/target')
    return arrays


def support_query_splits(n_target, shots, repeats=20, seed=42):
    shots = list(shots)
    if not shots or len(set(shots))!=len(shots) or any(not isinstance(k,(int,np.integer)) or isinstance(k,bool) or k<1 for k in shots):
        raise ValueError('shots must contain unique positive integers')
    if not isinstance(repeats,(int,np.integer)) or isinstance(repeats,bool) or repeats<1:
        raise ValueError('repeats must be a positive integer')
    maximum=max(shots)
    if n_target-maximum<2:raise ValueError('At least two untouched target query records are required')
    rng=np.random.default_rng(seed)
    result=[]
    for repeat in range(repeats):
        order=rng.permutation(n_target)
        result.append({'repeat':repeat,'support_pool':order[:maximum], 'query':order[maximum:]})
    return result


def source_search(X,y,ids,representation,output,seed):
    folds=list(KFold(3,shuffle=True,random_state=seed).split(X))
    retained=min(X.shape[1],max(5,int(np.ceil(X.shape[1]*.1))))
    if representation=='full':selector='passthrough';retained=X.shape[1]
    elif representation in {'mcuve','mcuve100'}:
        selector=MCUVESelector(n_runs=100 if representation=='mcuve100' else 20,n_components=2,n_features_to_select=retained,
                               task='regression',random_state=seed)
    else:
        selector=StableVariableSelector(base_selector='mcuve',n_features_to_select=retained,
                                         n_resamples=10,inner_runs=10,n_components=2,
                                         task='regression',random_state=seed)
    components=[k for k in [1,2,3,5,8] if k<=min(retained,min(len(t)-1 for t,_ in folds))]
    pipeline=Pipeline([('preprocess',RowPreprocessor()),('scale',StandardScaler()),
                       ('selector',selector),('model',PLSRegression(scale=False))],
                      memory=Memory(output/'.cache',verbose=0))
    search=GridSearchCV(pipeline,{'preprocess__mode':['raw','snv'],'model__n_components':components},
                        cv=folds,scoring='neg_mean_squared_error',n_jobs=1,error_score='raise')
    started=time.monotonic()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');search.fit(X,y)
        model=search.best_estimator_
        record={'status':'pass','representation':representation,'best_params':search.best_params_,
                'best_source_cv_neg_mse':float(search.best_score_),'fit_seconds':time.monotonic()-started,
                'candidate_count':len(search.cv_results_['params']),'cv_fits':3*len(search.cv_results_['params']),
                'source_rows':len(X),'input_features':X.shape[1],'selected_feature_count':retained,
                'selector_budget':{'mcuve_runs':100 if representation=='mcuve100' else 20,'stable_resamples':10,'stable_inner_runs':10,
                                    'selector_components':2,'retained_rule':'ceil(10% of source features), at least5'},
                'inner_folds':[{'train_ids':ids[t].tolist(),'validation_ids':ids[v].tolist()} for t,v in folds],
                'warnings':[{'category':v.category.__name__,'message':str(v.message)} for v in caught],
                'comparison_scope':'Same source partitions and downstream grid; selector work adds unequal computational cost'}
        fitted=model.named_steps['selector']
        record['selected_feature_indices_zero_based']=(list(range(X.shape[1])) if isinstance(fitted,str) else fitted.selected_indices_.tolist())
        if not isinstance(fitted,str) and hasattr(fitted,'frequency_'):
            record.update({'frequency':fitted.frequency_.tolist(),
                           'mean_resample_pairwise_jaccard':fitted.mean_pairwise_jaccard_,
                           'frequency_note':'Resampling diagnostic, not p-value or proven target-domain stability'})
        write_json(output/'source_model.json',record)
        return model,record
    finally:
        shutil.rmtree(output/'.cache',ignore_errors=True)


def fit_frozen_head(source_model, Z_source, y_source, Z_support, y_support, Z_query, joint=False):
    """Refit only PLS head; source-fitted representation remains unchanged."""
    if joint:
        X=np.r_[Z_source,Z_support];y=np.r_[y_source,y_support]
    else:X,y=Z_support,y_support
    if len(y)==1:
        return np.full(len(Z_query),float(y[0])),{'head':'support_mean','n_components':0,'training_rows':1}
    components=min(int(source_model.named_steps['model'].n_components),len(y)-1,X.shape[1])
    head=PLSRegression(n_components=components,scale=False).fit(X,y)
    return np.asarray(head.predict(Z_query)).reshape(-1),{'head':'PLS','n_components':components,
                                                        'training_rows':len(y),'joint':joint}


def evaluate_method(method, pred_support, y_support, pred_query, source_scale,
                    source_model, Z_source, y_source, Z_support, Z_query):
    if method=='none':return pred_query.copy(),{'calibration':'none'}
    if method in {'bias_unshrunk','affine_unshrunk','affine_shrinkage'}:
        calibration=ShrinkageAffineCalibrator(
            regularization=1.0 if method=='affine_shrinkage' else 0.0,
            source_y_scale=source_scale,mode='bias' if method=='bias_unshrunk' else 'affine')
        calibration.fit(pred_support,y_support)
        return calibration.predict(pred_query),{'calibration':calibration.fit_mode_,
               'slope':calibration.slope_,'intercept':calibration.intercept_,
               'regularization':calibration.regularization,'source_y_scale':source_scale}
    return fit_frozen_head(source_model,Z_source,y_source,Z_support,y_support,Z_query,
                           joint=method=='source_plus_support')


def aggregate_results(rows):
    passed=[r for r in rows if r['status']=='pass'];groups=defaultdict(list)
    for row in passed:groups[(row['representation'],row['shots'],row['method'])].append(row)
    summaries=[]
    for (representation,shots,method),cases in sorted(groups.items()):
        baselines={(r['repeat']):r for r in passed if r['representation']==representation and r['shots']==shots and r['method']=='none'}
        numeric=['rmse','mae','r2','bias_pred_minus_true','rpd']
        defined={k:[r['metrics'][k] for r in cases if r['metrics'][k] is not None] for k in numeric}
        means={k:float(np.mean(v)) if v else None for k,v in defined.items()}
        sd={k:float(np.std(v,ddof=1)) if len(v)>1 else (0.0 if v else None) for k,v in defined.items()}
        delta=np.asarray([r['metrics']['rmse']-baselines[r['repeat']]['metrics']['rmse'] for r in cases])
        summaries.append({'representation':representation,'shots':shots,'method':method,'repeats':len(cases),
                          'metric_mean':means,'metric_sd_ddof1':sd,
                          'paired_rmse_difference_vs_none_mean':float(delta.mean()),
                          'paired_rmse_difference_vs_none_sd_ddof1':float(delta.std(ddof=1)) if len(delta)>1 else 0.0,
                          'paired_rmse_improvement_fraction':float(np.mean(delta<0)),
                          'paired_rmse_difference_direction':'negative means lower RMSE than source model on same repeat/query',
                          'uncertainty_scope':'Repeated support draws with overlapping queries; SD is not confidence interval or independent cohort evidence'})
    return summaries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=20)
    parser.add_argument('--shots',type=int,nargs='+',default=[1,3,5,10,20])
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args();out=args.output
    if out.exists() and any(out.iterdir()):parser.error('Choose an empty/new output directory')
    out.mkdir(parents=True,exist_ok=True)
    manifest={'started_at_utc':datetime.now(timezone.utc).isoformat(),
              'input':str(args.input.resolve()),'input_sha256':sha256(args.input),
              'metadata_sha256':sha256(args.metadata),'seed':args.seed,'repeats':args.repeats,'shots':args.shots,
              'versions':{name:importlib.metadata.version(name) for name in ['numpy','scipy','scikit-learn']},
              'scope':'Source-target regression transfer engineering benchmark; domain meaning follows source metadata, not filename initials; not medicinal-herb validation or method novelty',
              'records_are_independent_physical_samples_verified':False,
              'source_plus_support_head_components':'Source chosen count, capped by total source+support rows and selected features',
              'target_only_head_components':'Source chosen count capped at shots-1; one-shot is support mean',
              'head_baseline_scope':'target_support_only and source_plus_support freeze the source-selected preprocessing, source scaler and selector, and refit only the PLS head; they are not whole-pipeline target/joint retraining',
              'all_selected_passed':False}
    code=['scripts/run_transfer_research.py','src/herb_spectra/calibration.py','src/herb_spectra/stability.py',
          'src/herb_spectra/selection.py','src/herb_spectra/validation.py']
    manifest['code_sha256']=[{'path':p,'sha256':sha256(REPO/p)} for p in code]
    rows=[];prediction_values=[];prediction_case_indices=[]
    try:
        arrays=validate_input(args.input)
        metadata=json.loads(args.metadata.read_text());write_json(out/'source_metadata.json',metadata)
        manifest['source_domain_from_metadata']=metadata.get('source_domain','See source metadata and NPZ instrument fields')
        manifest['target_domain_from_metadata']=metadata.get('target_domains',{}).get(args.input.name,'See source metadata and NPZ instrument fields')
        Xs,ys,source_ids=arrays['X_source'],arrays['y_source'],arrays['source_ids']
        Xt,yt,target_ids=arrays['X_target'],arrays['y_target'],arrays['target_ids']
        splits=support_query_splits(len(yt),args.shots,args.repeats,args.seed)
        manifest.update({'source_rows':len(ys),'target_pool_rows':len(yt),'query_rows_per_repeat':len(yt)-max(args.shots),
                         'features':Xs.shape[1],'source_target_id_overlap':0,'source_target_exact_spectrum_overlap':0,
                         'source_instrument':str(arrays.get('source_instrument','unspecified')),
                         'target_instrument':str(arrays.get('target_instrument','unspecified')),
                         'target_unit':str(arrays.get('target_unit','source-units-unspecified')),
                         'shots_unit':'total labelled target records, not per-class shots',
                         'support_pool_rule':'One permutation per repeat; prefixes of max-shot pool; all other records form common query'})
        write_json(out/'split_audit.json',[{'repeat':s['repeat'],
                   'support_pool_ids':target_ids[s['support_pool']].tolist(),'query_ids':target_ids[s['query']].tolist(),
                   'support_ids_by_shots':{str(k):target_ids[s['support_pool'][:k]].tolist() for k in args.shots},
                   'source_support_query_id_overlap':0} for s in splits])
        query_truth=np.asarray([yt[s['query']] for s in splits]);query_indices=np.asarray([s['query'] for s in splits])
        for representation in REPRESENTATIONS:
            directory=out/representation;directory.mkdir()
            print('Source-only search:',representation,flush=True)
            try:
                source_model,source_record=source_search(Xs,ys,source_ids,representation,directory,args.seed)
                frozen=source_model[:-1]
                Zs,Zt=frozen.transform(Xs),frozen.transform(Xt)
                target_predictions=np.asarray(source_model.predict(Xt)).reshape(-1)
                source_scale=float(np.std(ys))
                for split in splits:
                    repeat=split['repeat'];query=split['query']
                    for shots in args.shots:
                        support=split['support_pool'][:shots]
                        for method in METHODS:
                            row={'representation':representation,'repeat':repeat,'shots':shots,'method':method,
                                 'status':'fail','source_components':source_record['best_params']['model__n_components'],
                                 'query_rows':len(query),'source_selected_features':source_record['selected_feature_count']}
                            started=time.monotonic()
                            try:
                                with warnings.catch_warnings(record=True) as caught:
                                    warnings.simplefilter('always')
                                    predicted,parameters=evaluate_method(method,target_predictions[support],yt[support],
                                        target_predictions[query],source_scale,source_model,Zs,ys,Zt[support],Zt[query])
                                predicted=np.asarray(predicted,dtype=float).reshape(-1)
                                if predicted.shape!=(len(query),) or not np.isfinite(predicted).all():raise ValueError('Invalid query predictions')
                                row.update({'status':'pass','metrics':evaluation_metrics(yt[query],predicted,'regression'),
                                            'parameters':parameters,'warning_count':len(caught),
                                            'warnings':[str(v.message) for v in caught],
                                            'prediction_row':len(prediction_values)})
                                prediction_values.append(predicted);prediction_case_indices.append(len(rows))
                            except Exception as exc:row['error']=f'{type(exc).__name__}: {exc}'
                            row['fit_predict_seconds']=time.monotonic()-started;rows.append(row)
                    print(representation,'repeat',repeat+1,'/',args.repeats,'done',flush=True)
                write_json(out/'results.json',rows)
            except Exception as exc:
                rows.append({'representation':representation,'method':'_source_fit','status':'fail',
                             'error':f'{type(exc).__name__}: {exc}'})
                write_json(out/'results.json',rows)
        np.savez_compressed(out/'query_predictions.npz',predictions=np.asarray(prediction_values),
                            result_row_indices=np.asarray(prediction_case_indices,dtype=int),
                            query_y_true_by_repeat=query_truth,query_indices_by_repeat=query_indices,
                            target_ids=target_ids,source_ids=source_ids,
                            support_pool_indices_by_repeat=np.asarray([s['support_pool'] for s in splits]))
        summary=aggregate_results(rows);write_json(out/'summary.json',summary)
        with (out/'summary.csv').open('w',newline='') as handle:
            columns=['representation','shots','method','repeats','rmse_mean','rmse_sd','mae_mean','r2_mean',
                     'paired_rmse_difference_vs_none_mean','paired_rmse_improvement_fraction']
            writer=csv.DictWriter(handle,fieldnames=columns,lineterminator='\n');writer.writeheader()
            for row in summary:
                writer.writerow({k:row.get(k) for k in columns if k in row} | {
                    'rmse_mean':row['metric_mean']['rmse'],'rmse_sd':row['metric_sd_ddof1']['rmse'],
                    'mae_mean':row['metric_mean']['mae'],'r2_mean':row['metric_mean']['r2']})
        expected=len(REPRESENTATIONS)*args.repeats*len(args.shots)*len(METHODS)
        manifest.update({'expected_cases':expected,'actual_cases':len(rows),
                         'all_selected_passed':len(rows)==expected and all(r['status']=='pass' for r in rows),
                         'prediction_file_sha256':sha256(out/'query_predictions.npz')})
    except Exception as exc:
        manifest['error']=f'{type(exc).__name__}: {exc}'
    manifest['finished_at_utc']=datetime.now(timezone.utc).isoformat();write_json(out/'manifest.json',manifest)
    print(json.dumps({'all_selected_passed':manifest['all_selected_passed'],'cases':len(rows),'output':str(out)},indent=2))
    return 0 if manifest['all_selected_passed'] else 1


if __name__=='__main__':sys.exit(main())
