"""Train-only model selection and held-out evaluation for external spectra.

The Python selectors are the repository's independent, explicitly labelled
implementations. This module does not claim native execution or exact numerical
reproduction of the archived MATLAB programs. Dataset identity, physical-sample
relationships and upstream processing still require a separate source audit.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
import warnings

import numpy as np
from joblib import Memory
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .selection import CARSStyleSelector, MCUVESelector
from .validation import RowPreprocessor


CLASSIFICATION_MODELS = ('dummy', 'svc', 'plsda', 'cars_svc', 'mcuve_svc')
REGRESSION_MODELS = ('dummy', 'pls', 'svr', 'cars_pls', 'mcuve_pls')


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Unsupported JSON value: {type(value).__name__}')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, default=json_value, ensure_ascii=False,
                                    indent=2, allow_nan=False) + '\n')


class PLSDAClassifier(ClassifierMixin, BaseEstimator):
    """One-hot PLS2 with argmax; class vocabulary is learned from train only.

    This is an explicit multiclass adaptation, not the archived binary plslda.m.
    Scores need not be probabilities and no probability calibration is claimed.
    """
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=float)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError('PLS-DA requires at least two training classes')
        if not 1 <= self.n_components <= min(X.shape[1], len(X) - 1):
            raise ValueError('PLS component count exceeds training dimensions')
        self.n_features_in_ = X.shape[1]
        self.model_ = PLSRegression(n_components=self.n_components, scale=False)
        self.model_.fit(X, np.eye(len(self.classes_))[encoded])
        return self

    def decision_function(self, X):
        check_is_fitted(self, ['model_', 'classes_'])
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Feature grid differs from training')
        return self.model_.predict(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


def _row_hashes(X):
    values = np.asarray(X, dtype='<f8').copy()
    values[values == 0] = 0.0  # Canonicalize signed zeros.
    return np.asarray([hashlib.sha256(row.tobytes()).hexdigest() for row in values])


def validate_arrays(arrays, task):
    """Validate held-out arrays and reject exact cross-partition leakage."""
    if task not in {'classification', 'regression'}:
        raise ValueError('task must be classification or regression')
    required = ['X_train', 'y_train', 'X_test', 'y_test']
    if any(name not in arrays for name in required):
        raise ValueError('NPZ must contain ' + ', '.join(required))
    X_train = check_array(arrays['X_train'], dtype=float, ensure_min_samples=6)
    X_test = check_array(arrays['X_test'], dtype=float, ensure_min_samples=2)
    y_train, y_test = np.asarray(arrays['y_train']), np.asarray(arrays['y_test'])
    if y_train.ndim != 1 or y_test.ndim != 1:
        raise ValueError('Each task must contain a single one-dimensional target')
    if len(y_train) != len(X_train) or len(y_test) != len(X_test):
        raise ValueError('Feature and target row counts differ')
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError('Train and test feature grids differ')
    if task == 'regression':
        y_train, y_test = y_train.astype(float), y_test.astype(float)
        if not np.isfinite(y_train).all() or not np.isfinite(y_test).all():
            raise ValueError('Regression targets must be finite measured values')
        if np.std(y_train) <= np.finfo(float).eps:
            raise ValueError('Regression training target is constant')
    else:
        if any(v is None or str(v).strip().lower() in {'', 'nan', 'none'}
               for v in np.r_[y_train, y_test]):
            raise ValueError('Missing classification label')
        if len(np.unique(y_train)) < 2:
            raise ValueError('Classification training set has fewer than two classes')
        if not set(np.unique(y_test)).issubset(set(np.unique(y_train))):
            raise ValueError('Test has a class absent from training')
    train_hashes, test_hashes = _row_hashes(X_train), _row_hashes(X_test)
    overlap = set(train_hashes) & set(test_hashes)
    if overlap:
        raise ValueError(f'Exact spectra overlap train/test: {len(overlap)} unique rows')
    train_ids = np.asarray(arrays.get('train_ids', [f'train-{i}' for i in range(len(X_train))])).astype(str)
    test_ids = np.asarray(arrays.get('test_ids', [f'test-{i}' for i in range(len(X_test))])).astype(str)
    if train_ids.ndim != 1 or test_ids.ndim != 1 or len(train_ids) != len(X_train) or len(test_ids) != len(X_test):
        raise ValueError('One ID per row is required')
    if set(train_ids) & set(test_ids):
        raise ValueError('Provided sample IDs overlap train/test')
    if len(np.unique(train_ids)) != len(train_ids) or len(np.unique(test_ids)) != len(test_ids):
        raise ValueError('Repeated sample IDs: aggregate physical-sample scans before benchmarking')
    audit = {'train_rows': len(X_train), 'test_rows': len(X_test),
             'features': X_train.shape[1], 'all_values_finite': True,
             'train_test_exact_duplicate_rows': 0, 'train_test_id_overlap': 0,
             'train_duplicate_feature_rows': len(X_train) - len(np.unique(train_hashes)),
             'test_duplicate_feature_rows': len(X_test) - len(np.unique(test_hashes)),
             'constant_train_features': int(np.count_nonzero(np.std(X_train, axis=0) <= np.finfo(float).eps)),
             'constant_train_spectra': int(np.count_nonzero(np.std(X_train, axis=1) <= np.finfo(float).eps)),
             'constant_test_spectra': int(np.count_nonzero(np.std(X_test, axis=1) <= np.finfo(float).eps)),
             'physical_sample_identity': 'provided row/sample IDs; semantics require source audit' if 'train_ids' in arrays else 'not supplied; rows assumed independent',
             'limitation': 'Exact-row checks do not establish that near-duplicate scans or batches are independent'}
    if task == 'classification':
        audit['train_class_counts'] = {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))}
        audit['test_class_counts'] = {str(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))}
    else:
        audit['train_target_range'] = [float(y_train.min()), float(y_train.max())]
        audit['test_target_range'] = [float(y_test.min()), float(y_test.max())]
        audit['test_target_std_ddof1'] = float(np.std(y_test, ddof=1))
    return X_train, y_train, X_test, y_test, train_ids, test_ids, train_hashes, audit


def training_folds(X, y, task, hashes, seed=42, n_splits=3):
    if task == 'classification':
        if min(np.unique(y, return_counts=True)[1]) < n_splits:
            raise ValueError('Not enough training rows per class for inner CV')
        for value in np.unique(hashes):
            if len(np.unique(y[hashes == value])) > 1:
                raise ValueError('Identical training spectra have conflicting labels')
        folds = list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, hashes))
        for train, valid in folds:
            if set(np.unique(y[train])) != set(np.unique(y)) or set(np.unique(y[valid])) != set(np.unique(y)):
                raise ValueError('An inner fold is missing a class; collect samples or reduce folds')
    elif len(np.unique(hashes)) != len(hashes):
        # Avoid repeated exact spectra crossing folds in regression as well.
        from sklearn.model_selection import GroupKFold
        folds = list(GroupKFold(n_splits).split(X, y, hashes))
    else:
        folds = list(KFold(n_splits, shuffle=True, random_state=seed).split(X))
    for train, valid in folds:
        if set(hashes[train]) & set(hashes[valid]):
            raise ValueError('Exact training duplicates cross an inner fold')
    return folds


def model_pipeline(name, task, features, min_inner_rows, modes, seed=42, memory=None):
    """Build whole train-only pipeline; caching reuses identical training fits."""
    if not modes or not set(modes).issubset({'raw', 'snv'}):
        raise ValueError('preprocess_modes must be a nonempty list of raw/snv')
    supported = CLASSIFICATION_MODELS if task == 'classification' else REGRESSION_MODELS
    if name not in supported:
        raise ValueError(f'{name} is not a {task} model')
    selector = 'passthrough'
    retained = features
    if name.startswith('cars_'):
        retained = min(2, features)
        selector = CARSStyleSelector(n_runs=12, n_components=2, min_features=retained,
                                     cv=3, task=task, random_state=seed)
    elif name.startswith('mcuve_'):
        retained = min(features, max(5, int(np.ceil(0.1 * features))))
        selector = MCUVESelector(n_runs=20, n_components=2,
                                 n_features_to_select=retained, task=task, random_state=seed)
    max_pls = min(min_inner_rows - 1, retained)
    components = [v for v in [1, 2, 3, 5, 8] if v <= max_pls]
    grid = {'preprocess__mode': list(modes)}
    if name == 'dummy':
        model = DummyClassifier(strategy='most_frequent') if task == 'classification' else DummyRegressor(strategy='mean')
        grid = {}
    elif name in {'svc', 'cars_svc', 'mcuve_svc'}:
        model = SVC(class_weight='balanced', gamma='scale')
        grid['model__C'] = [0.1, 1.0, 10.0, 100.0]
    elif name == 'plsda':
        model = PLSDAClassifier()
        grid['model__n_components'] = components
    elif name in {'pls', 'cars_pls', 'mcuve_pls'}:
        model = PLSRegression(scale=False)
        grid['model__n_components'] = components
    elif name == 'svr':
        model = TransformedTargetRegressor(regressor=SVR(gamma='scale'), transformer=StandardScaler())
        grid['model__regressor__C'] = [0.1, 1.0, 10.0, 100.0]
    pipe = Pipeline([('preprocess', RowPreprocessor('raw' if name == 'dummy' else modes[0])),
                     ('scale', StandardScaler()), ('selector', selector), ('model', model)], memory=memory)
    return pipe, grid


def evaluation_metrics(y_true, predicted, task, classes=None):
    if task == 'classification':
        return {'accuracy': float(accuracy_score(y_true, predicted)),
                'balanced_accuracy': float(balanced_accuracy_score(y_true, predicted)),
                'macro_f1': float(f1_score(y_true, predicted, labels=classes, average='macro', zero_division=0)),
                'macro_precision': float(precision_score(y_true, predicted, labels=classes, average='macro', zero_division=0)),
                'macro_recall': float(recall_score(y_true, predicted, labels=classes, average='macro', zero_division=0)),
                'cohen_kappa': _finite_or_none(cohen_kappa_score(y_true, predicted, labels=classes)),
                'mcc': float(matthews_corrcoef(y_true, predicted))}
    rmse = float(np.sqrt(mean_squared_error(y_true, predicted)))
    sd = float(np.std(y_true, ddof=1))
    return {'rmse': rmse, 'mae': float(mean_absolute_error(y_true, predicted)),
            'r2': float(r2_score(y_true, predicted)) if sd > 0 else None,
            'bias_pred_minus_true': float(np.mean(predicted - y_true)),
            'rpd': sd / rmse if rmse > 0 else None,
            'rpd_definition': 'held-out target standard deviation (ddof=1) / held-out RMSE',
            'rpd_note': 'undefined/infinite under zero RMSE' if rmse == 0 else None}


def _finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def benchmark_dataset(spec, output, models=None, seed=42):
    """Fit one dataset's fixed split and persist every selected model's result."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    data_sha256 = hashlib.sha256(Path(spec['data_path']).read_bytes()).hexdigest()
    if spec.get('data_sha256') and spec['data_sha256'] != data_sha256:
        raise ValueError('Prepared NPZ hash differs from dataset spec')
    with np.load(spec['data_path'], allow_pickle=False) as values:
        arrays = {k: values[k] for k in values.files}
    X, y, test_X, test_y, train_ids, test_ids, hashes, audit = validate_arrays(arrays, spec['task'])
    audit['provided_id_scope'] = spec.get('identity_scope', 'not specified by dataset manifest')
    folds = training_folds(X, y, spec['task'], hashes, seed)
    write_json(output/'quality.json', audit)
    write_json(output/'inner_folds.json', [{'train_ids': train_ids[t].tolist(), 'validation_ids': train_ids[v].tolist()}
                                         for t, v in folds])
    supported = CLASSIFICATION_MODELS if spec['task'] == 'classification' else REGRESSION_MODELS
    chosen = list(supported) if models is None else [m for m in models if m in supported]
    if not chosen:
        raise ValueError('No requested models apply to this task')
    results = []
    for name in chosen:
        print(f"Running {spec['id']} / {name}", flush=True)
        case = output/name; case.mkdir(exist_ok=True)
        row = {k: spec[k] for k in ['id', 'dataset', 'fold', 'herb', 'task', 'source_url', 'source_scope',
                                   'source_commit', 'source_dataset', 'target_name', 'target_unit',
                                   'target_column', 'input_transform', 'split_policy', 'identity_scope',
                                   'class_names'] if k in spec}
        row.update({'model': name, 'status': 'fail', 'seed': seed,
                    'implementation': 'scikit-learn pipeline; repository independent CARS-style/MC-UVE implementations, not archived MATLAB native reproduction',
                    'train_rows': len(X), 'test_rows': len(test_X), 'input_features': X.shape[1],
                    'inner_cv_folds': len(folds), 'data_sha256': data_sha256,
                    'selector_budget': ({'n_runs': 12, 'pls_components': 2, 'internal_cv': 3}
                                        if name.startswith('cars_') else
                                        {'n_runs': 20, 'pls_components': 2, 'retained_feature_policy': 'min(p,max(5,ceil(0.1*p)))'}
                                        if name.startswith('mcuve_') else None),
                    'comparison_scope': 'shared data partitions and downstream parameter grid; selector extra fits mean computational budgets are not identical',
                    'target_scaling': 'training-only StandardScaler, restored to source units' if name == 'svr' else 'no external target scaling'})
        start = time.monotonic()
        try:
            cache = Memory(location=case/'.cache', verbose=0)
            pipeline, grid = model_pipeline(name, spec['task'], X.shape[1], min(len(t) for t, _ in folds),
                                            spec.get('preprocess_modes', ['raw', 'snv']), seed, cache)
            if name == 'dummy':
                estimator = pipeline
            else:
                estimator = GridSearchCV(pipeline, grid, cv=folds, n_jobs=1,
                                         scoring='balanced_accuracy' if spec['task'] == 'classification' else 'neg_mean_squared_error',
                                         error_score='raise', return_train_score=False)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                estimator.fit(X, y)
                row['fit_seconds'] = round(time.monotonic() - start, 6)
                pred_start = time.monotonic()
                predicted = np.asarray(estimator.predict(test_X)).reshape(-1)
                row['predict_seconds'] = round(time.monotonic() - pred_start, 6)
            warning_counts = Counter((str(v.category.__name__), str(v.message)) for v in caught)
            write_json(case/'warnings.json', [{'category': k[0], 'message': k[1], 'count': v}
                                             for k, v in warning_counts.items()])
            row['warning_count'] = len(caught)
            fitted = estimator if name == 'dummy' else estimator.best_estimator_
            row['best_params'] = {} if name == 'dummy' else estimator.best_params_
            row['candidate_count'] = 1 if name == 'dummy' else len(estimator.cv_results_['params'])
            row['downstream_cv_fit_count'] = 0 if name == 'dummy' else row['candidate_count'] * len(folds)
            row['best_inner_cv_score'] = None if name == 'dummy' else float(estimator.best_score_)
            row['inner_score_definition'] = 'balanced accuracy' if spec['task'] == 'classification' else 'negative MSE in original target units'
            selector = fitted.named_steps['selector']
            indices = np.arange(X.shape[1]) if isinstance(selector, str) else selector.selected_indices_
            row['selected_feature_count'] = len(indices)
            row['selected_feature_indices_zero_based'] = indices.tolist()
            classes = np.unique(y) if spec['task'] == 'classification' else None
            row['metrics'] = evaluation_metrics(test_y, predicted, spec['task'], classes)
            if spec['task'] == 'classification':
                row['class_order'] = classes.tolist()
                row['confusion_matrix'] = confusion_matrix(test_y, predicted, labels=classes).tolist()
                row['per_class_metrics'] = classification_report(test_y, predicted, labels=classes,
                                                                  output_dict=True, zero_division=0)
            with (case/'predictions.csv').open('w', newline='') as handle:
                writer = csv.writer(handle); writer.writerow(['sample_id', 'y_true', 'y_predicted'])
                writer.writerows(zip(test_ids.tolist(), test_y.tolist(), predicted.tolist()))
            row['predictions'] = str((case/'predictions.csv').relative_to(output))
            row['status'] = 'pass'
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
            row['elapsed_seconds'] = round(time.monotonic() - start, 6)
        finally:
            shutil.rmtree(case/'.cache', ignore_errors=True)
        write_json(case/'result.json', row)
        results.append(row); write_json(output/'results.json', results)
        print(f"Completed {spec['id']} / {name}: {row['status']}", flush=True)
    return results
