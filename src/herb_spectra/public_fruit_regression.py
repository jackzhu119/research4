"""Predeclared grouped regression on local public-fruit engineering waveforms.

This is an exploratory method check, not a medicinal-plant THz-TDS experiment.
The upstream licence, time unit and target assay methods remain unverified.
Every fruit's four dates stay together. Raw/derived data, fitted models, trace
and per-record predictions stay in an ignored local run directory.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .leakage_guard import (CacheKey, FitScope, LeakageError, guarded_cache_key,
                            instrument_mcuve_pls, load_guarded_cache,
                            save_guarded_cache, source_fingerprint)
from .selection import MCUVESelector

ROOT = Path(__file__).resolve().parents[2]
DATASET_ID = 'princeton-fruit-fixed-Tag-time-index-20261031-v1'
PROTOCOL_ID = 'public-fruit-grouped-regression-20261031-v1'
STATUS = 'EXPLORATORY_FRUIT_GROUPED_ENGINEERING_REGRESSION'
INPUT = ROOT / ('data/external/public_thz_20261031_v1/acquired/'
                'princeton_metasticker_fruit/prepared/fruit_fixed_Tag_time_index_v1.npz')
INPUT_SHA256 = 'b3c5d8b6272038377cfdfe272a6d084895b9817325fdd9ec34bab6c9db1ec27b'
SEEDS = (42, 43, 44)
TASKS = {'MA_Brix': ('MA', 'y_Brix'), 'PM_Brix': ('PM', 'y_Brix'),
         'PR_Brix': ('PR', 'y_Brix'), 'MA_DM': ('MA', 'y_DM')}
ROLES = ('dummy_mean', 'PLSR', 'full_SVR', 'MLP', 'MC_SVR', 'source_policy')
SOURCE_FILES = ('src/herb_spectra/public_fruit_regression.py',
                'scripts/run_public_fruit_regression.py',
                'src/herb_spectra/leakage_guard.py', 'src/herb_spectra/selection.py')
NPZ_KEYS = {'X', 'y_Brix', 'y_DM', 'record_id', 'groups', 'fruit_type',
            'timestamp', 'condition', 'sample_index', 'source_member'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def objhash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def csvwrite(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or not rows:
        raise LeakageError('Prediction/trace files must be nonempty and immutable')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wt', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def event(out, name, **values):
    with (Path(out) / 'events.jsonl').open('a', encoding='utf8') as handle:
        handle.write(json.dumps({'event': name, 'utc': utc(), **values}) + '\n')


def source_versions():
    return {name: sha(ROOT / name) for name in SOURCE_FILES}


def environment():
    return {name: importlib.metadata.version(name)
            for name in ('numpy', 'scipy', 'scikit-learn', 'joblib')}


def local_directory(path):
    """Require raw/derived/model artefacts below the existing ignored outputs/."""
    path = Path(path).resolve()
    if path == ROOT / 'outputs' or ROOT / 'outputs' not in path.parents:
        raise LeakageError('Licence-unknown raw/derived artefacts must stay local under outputs/')
    return path


def block_means(X):
    """Fixed 128 consecutive index blocks; no data-fitted or physical frequency axis."""
    X = np.asarray(X)
    if X.dtype.kind not in 'iuf' or X.ndim != 2 or X.shape[1] != 7000 or not np.isfinite(X).all():
        raise LeakageError('Expected finite 7000-point numeric waveforms only')
    return np.column_stack([X[:, block].mean(axis=1)
                            for block in np.array_split(np.arange(7000), 128)])


def pointwise_preprocess(X, mode):
    X = np.asarray(X, dtype=float)
    if mode == 'raw':
        return X.copy()
    if mode != 'snv':
        raise LeakageError('Undeclared preprocessing')
    sd = X.std(axis=1, keepdims=True)
    sd[sd <= np.finfo(float).eps] = 1.0
    return (X - X.mean(axis=1, keepdims=True)) / sd


def load_input(path=INPUT, *, expected_sha256=INPUT_SHA256):
    """Strict archived interface. Optional checksum is for explicit unit fixtures."""
    if sha(path) != expected_sha256:
        raise LeakageError('Frozen upstream fruit NPZ bytes differ')
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if set(data) != NPZ_KEYS:
        raise LeakageError('Exact upstream schema required; metadata cannot become features')
    X = data['X']
    if X.shape != (120, 7000) or X.dtype != np.dtype('float64') or not np.isfinite(X).all():
        raise LeakageError('Expected verbatim 120x7000 finite float64 waveforms')
    if data['sample_index'].shape != (7000,) or not np.array_equal(data['sample_index'], np.arange(7000)):
        raise LeakageError('Unverified time units must not replace integer sample index')
    for key in ('record_id', 'groups', 'fruit_type', 'timestamp', 'condition', 'source_member'):
        if data[key].shape != (120,) or data[key].dtype.kind not in 'US' or any(not str(v).strip() for v in data[key]):
            raise LeakageError('Explicit nonempty string provenance required: ' + key)
    if len(set(data['record_id'])) != 120 or set(data['condition']) != {'Tag'}:
        raise LeakageError('Unique fixed-condition observations required')
    if set(data['fruit_type']) != {'MA', 'PM', 'PR'}:
        raise LeakageError('Three separate declared fruit types required')
    expected_times = {'2023-05-13T21:00:00', '2023-05-18T21:00:00',
                      '2023-05-23T21:00:00', '2023-05-28T21:00:00'}
    for kind in ('MA', 'PM', 'PR'):
        ix = np.flatnonzero(data['fruit_type'] == kind)
        if len(ix) != 40 or len(set(data['groups'][ix])) != 10:
            raise LeakageError('Ten persistent fruits and 40 records per task required')
        for group in set(data['groups'][ix]):
            rows = np.flatnonzero(data['groups'] == group)
            if len(rows) != 4 or set(data['fruit_type'][rows]) != {kind} or set(data['timestamp'][rows]) != expected_times:
                raise LeakageError('Every fruit must have exactly the same four fixed dates and one species')
    for key in ('y_Brix', 'y_DM'):
        if data[key].dtype != np.dtype('float64') or data[key].shape != (120,):
            raise LeakageError('Targets must be one float64 reading per observation')
    if not np.isfinite(data['y_Brix']).all():
        raise LeakageError('Missing Brix target in fixed cohort')
    mango = data['fruit_type'] == 'MA'
    if not np.isfinite(data['y_DM'][mango]).all() or not np.isnan(data['y_DM'][~mango]).all():
        raise LeakageError('DM exists only for MA; unavailable targets may not be invented')
    return data


def split_indices(data):
    """Return all fixed outer/inner indices; labels are never passed to GroupKFold."""
    result = []
    for task, (kind, _) in TASKS.items():
        population = np.flatnonzero(data['fruit_type'] == kind)
        groups = data['groups'][population]
        for outer, (train, test) in enumerate(GroupKFold(5).split(population, groups=groups)):
            source, query = population[train], population[test]
            result.append((task, outer, None, source, query))
            for inner, (subtrain, validation) in enumerate(GroupKFold(3).split(source, groups=data['groups'][source])):
                result.append((task, outer, inner, source[subtrain], source[validation]))
    return result


def r1_grid():
    rows = []
    for preprocessing in ('raw', 'snv'):
        for components in (1, 2, 3):
            rows.append({'family': 'PLSR', 'preprocessing': preprocessing, 'components': components})
        for C in (1., 10., 100.):
            for gamma in ('scale', 'scale_x0.1'):
                for epsilon in (.05, .1):
                    rows.append({'family': 'full_SVR', 'preprocessing': preprocessing,
                                 'C': C, 'gamma': gamma, 'epsilon': epsilon})
        for alpha in (1., 10.):
            rows.append({'family': 'MLP', 'preprocessing': preprocessing, 'hidden': [16],
                         'alpha': alpha, 'epochs': 300})
    return [{'id': f'R1_{i:02d}', 'mc': None, **row} for i, row in enumerate(rows)]


def r2_grid(svr):
    if svr.get('family') != 'full_SVR' or svr.get('mc') is not None:
        raise LeakageError('R2 must inherit current outer-source R1 full SVR')
    rows = []
    for runs in (10, 20):
        for components in (1, 2):
            for ratio in (.25, .5):
                rows.append({**svr, 'id': f'R2_{len(rows):02d}', 'family': 'MC_SVR',
                             'mc': {'runs': runs, 'components': components, 'ratio': ratio}})
    return rows


def default_protocol():
    return {
        'protocol_id': PROTOCOL_ID, 'dataset_id': DATASET_ID, 'scientific_status': STATUS,
        'input_npz_sha256': INPUT_SHA256, 'tasks': {k: list(v) for k, v in TASKS.items()},
        'seeds': list(SEEDS), 'outer': 'GroupKFold5 by persistent fruit ID; 8fruit source/2fruit query',
        'inner': 'GroupKFold3 among current outer source8fruits; validation3/3/2 fruits',
        'features': 'Fixed 128 consecutive blocks np.array_split(arange(7000),128), each waveform mean per block',
        'feature_axis': 'engineering block index only; no verified time or THz frequency unit',
        'preprocessing_order': 'fixed block means; optional pointwise SNV; current-source X StandardScaler',
        'targets': 'Current-source y StandardScaler for every estimator and MC-UVE; inverse-transform predictions',
        'target_units': 'Excel-reported Brix or DM numeric reading units; assay method/unit not confirmed',
        'R1': r1_grid(), 'R2_template': r2_grid(next(x for x in r1_grid() if x['family'] == 'full_SVR')),
        'R2_rule': 'Replace only SVR preprocessor/C/gamma/epsilon with this outer-source selected R1 full SVR',
        'selection': 'Per candidate pool32 current-source OOF observations separately per seed; mean3seed RMSE then MAE then candidate ID',
        'R2_policy': 'Compare source-selected PLSR/full_SVR/MLP/MC_SVR only on same inner pooled predictions',
        'MC_sampling': 'Source observation rows only; four dates are dependent. No claim MC rows are independent fruits.',
        'source_candidates_per_outer': 42, 'source_estimator_fit_budget': 7560,
        'source_formula': '4tasks*5outer*3inner*3seeds*(34R1+8R2)',
        'MC_PLScall_budget_with_transform_cache': 10800,
        'outer_roles': list(ROLES), 'outer_role_seed_cases': 360,
        'outer_actual_estimator_fit_upper_bound': 300,
        'policy_is_alias_not_outer_selected': True, 'all20outer_choices_locked_before_outer_inference': True,
        'all_outer_truth_bytes_hashed_before_any_truth_parse': True,
        'source_cache': 'Exact data/targets/order/split/code/seed/preprocessing and MC LV/runs; ratios share source ranking only',
        'source_failures': 'Save failures; exclude incomplete candidate; never remove a fruit or select seed by query scores',
        'MLP': 'Adam lr.001,300iterations,tol0,n_iter_no_change300,early_stoppingFalse; all seeds',
        'prediction_boundary': 'Separate source fitting, truth-free query inference, immutable prediction lock then scoring',
        'OS_truth_file_sandbox': False, 'redistribution': 'Raw/derived,models,per-record predictions,traces LOCAL_ONLY license unresolved',
        'fresh_confirmation': False, 'cross_time_or_new_batch_claim': False,
        'medicinal_THZ_TDS_transfer_claim': False, 'reference_normalization': 'NOT_USED: Background/Metal identical',
        'cost_estimate': '7560 small128-feature estimator fits including720 MLP300epoch fits; 10800 cached MC PLS fits; wall time NOT_MEASURED before gate',
    }


def prepare(out):
    out = local_directory(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Prepare requires a fresh local directory')
    out.mkdir(parents=True, exist_ok=True)
    data = load_input()
    X = block_means(data['X'])
    contexts = []
    for task, outer, inner, source, query in split_indices(data):
        target = TASKS[task][1]
        split = f'{task}.outer{outer}.' + ('final' if inner is None else f'inner{inner}')
        directory = out / 'inputs' / split
        directory.mkdir(parents=True)
        for role, ix in (('source', source), ('query', query)):
            pack = {'X': X[ix], 'record_ids': data['record_id'][ix], 'feature_index': np.arange(128)}
            if role == 'source':
                pack['source_targets'] = data[target][ix]
            np.savez_compressed(directory / f'{role}.npz', **pack)
        truth = out / 'scoring_truth' / f'{split}.csv'
        csvwrite(truth, [{'record_id': str(i), 'target': float(y)}
                         for i, y in zip(data['record_id'][query], data[target][query], strict=True)])
        context = {'split_id': split, 'task': task, 'outer': outer, 'inner': inner,
                   'stage': 'outer' if inner is None else 'inner',
                   'source_ids': data['record_id'][source].tolist(), 'query_ids': data['record_id'][query].tolist(),
                   'source_groups': sorted(set(data['groups'][source])), 'query_groups': sorted(set(data['groups'][query])),
                   'source_group_by_id': dict(zip(data['record_id'][source], data['groups'][source])),
                   'query_group_by_id': dict(zip(data['record_id'][query], data['groups'][query])),
                   'source_pack': str((directory / 'source.npz').relative_to(out)),
                   'query_pack': str((directory / 'query.npz').relative_to(out)), 'truth': str(truth.relative_to(out))}
        validate_context(context)
        for key in ('source_pack', 'query_pack', 'truth'):
            context[key + '_sha256'] = sha(out / context[key])
        contexts.append(context)
    write(out / 'protocol.json', default_protocol())
    write(out / 'preparation.json', {'created_at_utc': utc(), 'input_path': str(INPUT.relative_to(ROOT)),
                                   'input_sha256': INPUT_SHA256, 'contexts': contexts,
                                   'records_per_task': 40, 'fruit_groups_per_task': 10,
                                   'datasets_not_pooled': True, 'same_four_dates_overlap_train_query': True,
                                   'data_status': STATUS, 'license_verified': False,
                                   'feature_matrix_sha256': source_fingerprint(X)})
    write(out / 'run_manifest.json', run_manifest(out))
    return {'status': 'PREPARED_NOT_TRAINED', 'contexts': len(contexts),
            'source_estimator_fit_budget': 7560, 'local_output': str(out)}


def validate_context(context):
    source, query = context['source_ids'], context['query_ids']
    if not source or not query or len(set(source)) != len(source) or len(set(query)) != len(query) or set(source) & set(query):
        raise LeakageError('Record crossed roles or duplicated')
    sg, qg = context['source_group_by_id'], context['query_group_by_id']
    if set(sg) != set(source) or set(qg) != set(query):
        raise LeakageError('Every record must map to a frozen fruit ID')
    if set(sg.values()) & set(qg.values()) or sorted(set(sg.values())) != context['source_groups'] or sorted(set(qg.values())) != context['query_groups']:
        raise LeakageError('Same persistent fruit crossed source/query roles')
    if any(list(sg.values()).count(g) != 4 for g in set(sg.values())) or any(list(qg.values()).count(g) != 4 for g in set(qg.values())):
        raise LeakageError('All four dates must follow their fruit')
    if context['stage'] == 'outer' and (len(source), len(query)) != (32, 8):
        raise LeakageError('Outer denominator changed')
    if context['stage'] == 'inner' and ((len(source), len(query)) not in ((20, 12), (24, 8))):
        raise LeakageError('Inner denominator changed')
    return True


def run_manifest(out):
    return {'protocol_sha256': sha(out / 'protocol.json'), 'preparation_sha256': sha(out / 'preparation.json'),
            'source_sha256': source_versions(), 'environment': environment()}


def verify_run(out):
    out = local_directory(out)
    saved = read(out / 'run_manifest.json')
    if saved != run_manifest(out) or read(out / 'protocol.json') != default_protocol():
        raise LeakageError('Frozen implementation, budget, environment or protocol changed')
    preparation = read(out / 'preparation.json')
    if preparation['input_sha256'] != INPUT_SHA256 or sha(INPUT) != INPUT_SHA256:
        raise LeakageError('Frozen upstream input changed')
    contexts = preparation['contexts']
    if len(contexts) != 80 or len({c['split_id'] for c in contexts}) != 80:
        raise LeakageError('All four task/five outer/three inner contexts required')
    # Byte integrity checks parse no truth records and complete before any trial.
    for context in contexts:
        validate_context(context)
        for name in ('source_pack', 'query_pack', 'truth'):
            if sha(out / context[name]) != context[name + '_sha256']:
                raise LeakageError('Frozen context bytes changed: ' + context['split_id'])
    return preparation, saved


def verify_source_release(out, release):
    _, manifest = verify_run(out)
    if release.get('allow_source_training') is not True or not release.get('created_at_utc'):
        raise LeakageError('Root source release required')
    if release.get('run_manifest_sha256') != sha(Path(out) / 'run_manifest.json') or any(release.get(k) != manifest[k] for k in ('protocol_sha256', 'preparation_sha256', 'source_sha256')):
        raise LeakageError('Root source release does not bind this prepared implementation')


def strict_pack(path, *, source=False):
    with np.load(path, allow_pickle=False) as archive:
        pack = {key: archive[key] for key in archive.files}
    expected = {'record_ids', 'X', 'feature_index'} | ({'source_targets'} if source else set())
    if set(pack) != expected:
        raise LeakageError('Only explicit engineering X, IDs and source targets accepted; no query truth or metadata')
    ids = pack['record_ids']
    X = pack['X']
    if ids.dtype.kind not in 'US' or ids.ndim != 1 or not len(ids) or len(set(ids)) != len(ids) or any(not str(i).strip() for i in ids):
        raise LeakageError('Unique explicit record IDs required')
    if X.dtype.kind not in 'iuf' or X.shape != (len(ids), 128) or not np.isfinite(X).all():
        raise LeakageError('Feature matrix must be128 engineering amplitudes, excluding metadata')
    if pack['feature_index'].dtype.kind not in 'iu' or not np.array_equal(pack['feature_index'], np.arange(128)):
        raise LeakageError('Fixed engineering block index required')
    if source:
        y = pack['source_targets']
        if y.dtype.kind not in 'iuf' or y.shape != (len(ids),) or not np.isfinite(y).all() or np.std(y) <= np.finfo(float).eps:
            raise LeakageError('Finite, varying measured source targets required')
    return pack


@dataclass
class SourceTransform:
    preprocessing: str
    x_scaler: object
    y_scaler: object
    selector: object
    metadata: dict
    _leakage_fit_provenance: dict

    def indices(self, ratio=None):
        if self.selector is None:
            if ratio is not None:
                raise LeakageError('Unexpected MC retention ratio')
            return np.arange(128)
        if ratio not in (.25, .5):
            raise LeakageError('Predeclared MC retained fractions required')
        return np.sort(self.selector.ranking_[:int(np.ceil(128 * ratio))])

    def matrix(self, X, ratio=None):
        return self.x_scaler.transform(pointwise_preprocess(X, self.preprocessing))[:, self.indices(ratio)]


def transform_source(pack, config, seed, context, out, cache):
    ids, X, y = pack['record_ids'], pack['X'], pack['source_targets']
    if ids.tolist() != context['source_ids']:
        raise LeakageError('Source order differs from context')
    mc = config.get('mc')
    key = guarded_cache_key(dataset_version=DATASET_ID, train_ids=ids, split_id=context['split_id'],
                            preprocessing={'mode': config['preprocessing'], 'fixed_width': 128,
                                           'target_scale': 'StandardScaler_current_source'},
                            selector=None if mc is None else {'runs': mc['runs'], 'components': mc['components'], 'ranking_width': 128},
                            code_version=objhash(source_versions()), seed=seed,
                            data_fingerprint=source_fingerprint(X, y))
    path = Path(out) / 'transform_cache' / f'{key}.joblib'
    if str(key) in cache:
        bundle, expected_hash, expected_state = cache[str(key)]
        if sha(path) != expected_hash or joblib.hash(bundle) != expected_state or bundle.metadata['cache_identity'] != key.identity:
            raise LeakageError('Learned cache changed')
        return bundle, {'path': str(path.relative_to(out)), 'sha256': expected_hash, 'cache_identity': key.identity, 'reused': True}
    if path.exists():
        raise FileExistsError('Unexpected prior transform; attempts are immutable')
    scope = FitScope(context['source_ids'], context['split_id'], DATASET_ID, Path(out) / 'fit_scope_events.jsonl', code_version=objhash(source_versions()))
    with scope.operation('fruit_transform_bundle', ids, X=X, y=y) as proof:
        transformed = pointwise_preprocess(X, config['preprocessing'])
        xs = scope.fit(StandardScaler(), transformed, sample_ids=ids, component='fruit.X_StandardScaler')
        ys = scope.fit(StandardScaler(), y[:, None], sample_ids=ids, component='fruit.y_StandardScaler')
        Z, target = xs.transform(transformed), ys.transform(y[:, None]).ravel()
        selector = None
        if mc is not None:
            selector = MCUVESelector(n_runs=mc['runs'], n_components=mc['components'],
                                     n_features_to_select=128, sample_fraction=.8, task='regression', random_state=seed)
            with instrument_mcuve_pls(scope, Z, ids):
                scope.fit(selector, Z, target, sample_ids=ids, component='fruit.MC_UVE.source_record_ranking')
        metadata = {'cache_identity': key.identity, 'source_ids': ids.tolist(),
                    'x_mean': xs.mean_.tolist(), 'x_variance': xs.var_.tolist(),
                    'y_mean': ys.mean_.tolist(), 'y_variance': ys.var_.tolist(),
                    'selector_sampling_unit': 'dependent timepoint records inside current source fruits, not independent fruits'}
        bundle = SourceTransform(config['preprocessing'], xs, ys, selector, metadata, dict(proof))
    save_guarded_cache(path, bundle, key)
    cache[str(key)] = (bundle, sha(path), joblib.hash(bundle))
    return bundle, {'path': str(path.relative_to(out)), 'sha256': sha(path), 'cache_identity': key.identity, 'reused': False}


def fit_estimator(Z, y, ids, config, seed, context, out):
    scope = FitScope(context['source_ids'], context['split_id'], DATASET_ID, Path(out) / 'fit_scope_events.jsonl', code_version=objhash(source_versions()))
    family = config['family']
    if family == 'PLSR':
        estimator = PLSRegression(n_components=config['components'], scale=False)
    elif family in ('full_SVR', 'MC_SVR'):
        variance = float(np.var(Z))
        scale = 1. / (Z.shape[1] * variance) if variance else 1.
        gamma = scale * (.1 if config['gamma'] == 'scale_x0.1' else 1.)
        if config['gamma'] not in ('scale', 'scale_x0.1'):
            raise LeakageError('Undeclared gamma')
        estimator = SVR(C=config['C'], epsilon=config['epsilon'], gamma=gamma, kernel='rbf')
    elif family == 'MLP':
        estimator = MLPRegressor(hidden_layer_sizes=tuple(config['hidden']), alpha=config['alpha'],
                                 max_iter=config['epochs'], early_stopping=False, n_iter_no_change=config['epochs'],
                                 tol=0., learning_rate_init=.001, random_state=seed, solver='adam')
    elif family == 'dummy_mean':
        estimator = DummyRegressor(strategy='mean')
    else:
        raise LeakageError('Unknown regression family')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        scope.fit(estimator, Z, y, sample_ids=ids, component='fruit.' + family)
    metadata = {'source_ids': ids.tolist(), 'source_fingerprint': source_fingerprint(Z, y),
                'warnings': [{'category': type(w.message).__name__, 'message': str(w.message)} for w in caught],
                'seed': seed, 'config': config}
    if family == 'MLP':
        metadata['actual_epochs'] = int(estimator.n_iter_)
        metadata['training_loss_curve'] = estimator.loss_curve_
    return estimator, metadata


def inference_check(transform, estimator, X, ratio):
    before = joblib.hash((transform, estimator))
    def predict(values):
        output = np.asarray(estimator.predict(transform.matrix(values, ratio)), dtype=float).reshape(-1, 1)
        result = transform.y_scaler.inverse_transform(output).ravel()
        if result.shape != (len(values),) or not np.isfinite(result).all():
            raise LeakageError('Invalid reported-unit regression prediction')
        return result
    predictions = predict(X)
    reverse = predict(X[::-1])[::-1]
    single = np.concatenate([predict(X[i:i + 1]) for i in range(len(X))])
    order = np.random.default_rng(20261031).permutation(len(X))
    permuted = np.empty_like(predictions)
    permuted[order] = predict(X[order])
    for other in (reverse, single, permuted):
        if not np.allclose(predictions, other, atol=1e-9, rtol=1e-9):
            raise LeakageError('Query batch or order changes predictions')
    if before != joblib.hash((transform, estimator)):
        raise LeakageError('Inference changed learned state')
    return predictions, {'state_unchanged': True, 'single_maxdiff': float(np.max(np.abs(predictions-single))),
                         'reverse_maxdiff': float(np.max(np.abs(predictions-reverse))),
                         'permutation_maxdiff': float(np.max(np.abs(predictions-permuted))), 'atol': 1e-9, 'rtol': 1e-9}


def trial(out, context, config, seed, cache, *, outer=False):
    out = Path(out)
    validate_context(context)
    for name in ('source_pack', 'query_pack'):
        if sha(out / context[name]) != context[name + '_sha256']:
            raise LeakageError('Prepared input changed')
    source = strict_pack(out / context['source_pack'], source=True)
    query = strict_pack(out / context['query_pack'])
    if source['record_ids'].tolist() != context['source_ids'] or query['record_ids'].tolist() != context['query_ids']:
        raise LeakageError('Source/query identity or order mismatch')
    start = time.perf_counter()
    transform, reference = transform_source(source, config, seed, context, out, cache)
    ratio = config['mc']['ratio'] if config.get('mc') else None
    Z = transform.matrix(source['X'], ratio)
    y = transform.y_scaler.transform(source['source_targets'][:, None]).ravel()
    estimator, model_metadata = fit_estimator(Z, y, source['record_ids'], config, seed, context, out)
    predictions, audit = inference_check(transform, estimator, query['X'], ratio)
    directory = out / ('evaluation_models' if outer else 'selection_models') / context['split_id'] / config['id']
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'seed{seed}.joblib'
    if path.exists():
        raise FileExistsError(path)
    joblib.dump(estimator, path, compress=3)
    elapsed = time.perf_counter() - start
    metadata = {'config': config, 'seed': seed, 'split_id': context['split_id'],
                'source_ids': source['record_ids'].tolist(), 'query_ids': query['record_ids'].tolist(),
                'transform': reference, 'selected_indices': transform.indices(ratio).tolist(),
                'retained_features': len(transform.indices(ratio)), 'estimator': model_metadata,
                'inference_audit': audit, 'elapsed_seconds': elapsed, 'outer_truth_parsed': False}
    write(path.with_suffix('.json'), metadata)
    record = {'task': context['task'], 'outer': context['outer'], 'inner': context['inner'],
              'split_id': context['split_id'], 'config': config, 'seed': seed,
              'estimator_path': str(path.relative_to(out)), 'estimator_sha256': sha(path),
              'metadata_sha256': sha(path.with_suffix('.json')), 'transform': reference,
              'retained_features': metadata['retained_features'], 'elapsed_seconds': elapsed}
    rows = [{'task': context['task'], 'outer': context['outer'], 'inner': context['inner'],
             'split_id': context['split_id'], 'candidate_id': config['id'], 'seed': seed,
             'record_id': str(rid), 'y_predicted': float(pred)}
            for rid, pred in zip(query['record_ids'], predictions, strict=True)]
    return record, rows


def load_truth(path, expected_ids):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['record_id', 'target']:
            raise LeakageError('Exact scoring truth schema required')
        rows = list(reader)
    values = {row['record_id']: float(row['target']) for row in rows}
    if len(values) != len(rows) or set(values) != set(expected_ids) or not all(np.isfinite(v) for v in values.values()):
        raise LeakageError('Truth identity, denominator or numeric value differs')
    return np.array([values[str(rid)] for rid in expected_ids])


def regression_metrics(y_true, prediction):
    y_true, prediction = np.asarray(y_true, dtype=float), np.asarray(prediction, dtype=float)
    if y_true.ndim != 1 or y_true.shape != prediction.shape or len(y_true) < 2 or not np.isfinite(y_true).all() or not np.isfinite(prediction).all():
        raise LeakageError('Finite paired regression vectors required')
    sd = float(np.std(y_true, ddof=1))
    rmse = float(np.sqrt(mean_squared_error(y_true, prediction)))
    return {'rmse': rmse, 'mae': float(mean_absolute_error(y_true, prediction)),
            'r2': float(r2_score(y_true, prediction)), 'target_sample_sd': sd,
            'rpd': sd / rmse if rmse > 0 else None, 'observations': len(y_true)}


def candidate_cases(out, candidate, contexts, cache):
    records, predictions, failures = [], [], []
    for context in contexts:
        for seed in SEEDS:
            try:
                record, rows = trial(out, context, candidate, seed, cache)
                records.append(record)
                predictions.extend(rows)
            except Exception as exc:
                if isinstance(exc, (LeakageError, FileExistsError)):
                    raise
                failure = {'split_id': context['split_id'], 'candidate': candidate,
                           'seed': seed, 'error_type': type(exc).__name__, 'message': str(exc)}
                failures.append(failure)
                event(out, 'candidate_failure', **failure)
    seed_metrics = {}
    if not failures:
        # Inner truth belongs to outer SOURCE. Verify every inner file before parsing any.
        for context in contexts:
            if sha(Path(out) / context['truth']) != context['truth_sha256']:
                raise LeakageError('Inner source truth changed')
        truth = {}
        for context in contexts:
            truth.update(zip(context['query_ids'], load_truth(Path(out) / context['truth'], context['query_ids']), strict=True))
        for seed in SEEDS:
            rows = [r for r in predictions if r['seed'] == seed]
            ids = [r['record_id'] for r in rows]
            if len(ids) != 32 or len(set(ids)) != 32 or set(ids) != set(truth):
                raise LeakageError('Each inner candidate/seed must pool exactly32 current-source predictions')
            seed_metrics[str(seed)] = regression_metrics([truth[i] for i in ids], [r['y_predicted'] for r in rows])
    means = None if failures else {key: float(np.mean([seed_metrics[str(seed)][key] for seed in SEEDS])) for key in ('rmse', 'mae')}
    return {'config': candidate, 'cases': len(records), 'status': 'FAILED_INNER_CANDIDATE' if failures else 'SOURCE_EVALUATED',
            'mean_seed_metrics': means, 'pooled_metrics_per_seed': seed_metrics,
            'records': records, 'predictions': predictions, 'failures': failures}


def choose(cases):
    valid = [case for case in cases if case['status'] == 'SOURCE_EVALUATED' and case['cases'] == 9]
    if not valid:
        raise LeakageError('No complete source-only candidate for required family')
    return min(valid, key=lambda case: (case['mean_seed_metrics']['rmse'], case['mean_seed_metrics']['mae'], case['config']['id']))


def strip_case(case):
    return {key: case[key] for key in ('config', 'cases', 'status', 'mean_seed_metrics', 'pooled_metrics_per_seed')}


def selection(out, release_path):
    out = local_directory(out)
    release = read(release_path)
    verify_source_release(out, release)
    if (out / 'source_release_snapshot.json').exists():
        raise FileExistsError('Selection already started; preserve attempt')
    write(out / 'source_release_snapshot.json', release)
    preparation, _ = verify_run(out)
    event(out, 'source_selection_start')
    locks, records, predictions, summaries, failures = [], [], [], [], []
    for task in TASKS:
        for outer in range(5):
            contexts = [c for c in preparation['contexts'] if c['task'] == task and c['outer'] == outer and c['stage'] == 'inner']
            cache, evaluated = {}, []
            for config in r1_grid():
                case = candidate_cases(out, config, contexts, cache)
                evaluated.append(case)
                records.extend(case['records']); predictions.extend(case['predictions']); failures.extend(case['failures'])
                summaries.append({'task': task, 'outer': outer, 'round': 'R1', **strip_case(case)})
                event(out, 'source_candidate_completed', task=task, outer=outer, candidate=config['id'], status=case['status'], models=len(records))
            choices = {family: strip_case(choose([c for c in evaluated if c['config']['family'] == family])) for family in ('PLSR', 'full_SVR', 'MLP')}
            r2 = []
            for config in r2_grid(choices['full_SVR']['config']):
                case = candidate_cases(out, config, contexts, cache)
                r2.append(case)
                records.extend(case['records']); predictions.extend(case['predictions']); failures.extend(case['failures'])
                summaries.append({'task': task, 'outer': outer, 'round': 'R2', **strip_case(case)})
                event(out, 'source_candidate_completed', task=task, outer=outer, candidate=config['id'], status=case['status'], models=len(records))
            choices['MC_SVR'] = strip_case(choose(r2))
            choices['source_policy'] = strip_case(choose(list(choices.values())))
            locks.append({'task': task, 'outer': outer, 'choices': choices, 'outer_truth_used_for_selection': False})
    write(out / 'inner_records.json', records)
    csvwrite(out / 'inner_predictions.csv.gz', predictions)
    write(out / 'candidate_summaries.json', summaries)
    write(out / 'source_candidate_failures.json', failures)
    lock = {'created_at_utc': utc(), 'outer_folds': locks, 'source_estimator_models': len(records),
            'source_failures': len(failures), 'all20outer_choices_locked': True,
            'outer_predictions_already_generated': False}
    for key, name in (('run_manifest_sha256', 'run_manifest.json'), ('source_release_sha256', 'source_release_snapshot.json'),
                      ('inner_records_sha256', 'inner_records.json'), ('inner_predictions_sha256', 'inner_predictions.csv.gz'),
                      ('candidate_summaries_sha256', 'candidate_summaries.json'), ('failures_sha256', 'source_candidate_failures.json')):
        lock[key] = sha(out / name)
    write(out / 'selection_lock.json', lock)
    event(out, 'all_source_choices_locked', models=len(records), failures=len(failures))
    return {'models': len(records), 'failures': len(failures), 'outer_evaluated': False}


def verify_records(out, records):
    checked = set()
    for record in records:
        for path, expected in ((out / record['estimator_path'], record['estimator_sha256']),
                               ((out / record['estimator_path']).with_suffix('.json'), record['metadata_sha256']),
                               (out / record['transform']['path'], record['transform']['sha256'])):
            if path not in checked:
                if sha(path) != expected:
                    raise LeakageError('Learned source artifact changed')
                checked.add(path)


def verify_selected(out):
    out = local_directory(out)
    preparation, manifest = verify_run(out)
    lock = read(out / 'selection_lock.json')
    checks = {'run_manifest_sha256': 'run_manifest.json', 'source_release_sha256': 'source_release_snapshot.json',
              'inner_records_sha256': 'inner_records.json', 'inner_predictions_sha256': 'inner_predictions.csv.gz',
              'candidate_summaries_sha256': 'candidate_summaries.json', 'failures_sha256': 'source_candidate_failures.json'}
    if any(lock.get(key) != sha(out / path) for key, path in checks.items()) or not lock.get('all20outer_choices_locked'):
        raise LeakageError('Frozen source selection changed')
    if {(f['task'], f['outer']) for f in lock['outer_folds']} != {(t, o) for t in TASKS for o in range(5)} or len(lock['outer_folds']) != 20:
        raise LeakageError('All20 unique outer choices required')
    verify_source_release(out, read(out / 'source_release_snapshot.json'))
    records = read(out / 'inner_records.json')
    if len(records) != lock['source_estimator_models']:
        raise LeakageError('Source estimator inventory incomplete')
    verify_records(out, records)
    return preparation, manifest, lock


def verify_outer_release(out, release):
    expected = {'selection_lock_sha256': sha(Path(out) / 'selection_lock.json'),
                'protocol_sha256': sha(Path(out) / 'protocol.json'), 'run_manifest_sha256': sha(Path(out) / 'run_manifest.json')}
    if release.get('allow_outer_evaluation') is not True or not release.get('created_at_utc') or any(release.get(k) != v for k, v in expected.items()):
        raise LeakageError('Root outer release must bind all source choices')


def evaluation(out, release_path):
    out = local_directory(out)
    preparation, _, selection_lock = verify_selected(out)
    release = read(release_path)
    verify_outer_release(out, release)
    if (out / 'outer_release_snapshot.json').exists():
        raise FileExistsError('Outer evaluation already started')
    write(out / 'outer_release_snapshot.json', release)
    event(out, 'outer_prediction_start')
    records, predictions, aliases = [], [], []
    for fold in selection_lock['outer_folds']:
        context = next(c for c in preparation['contexts'] if c['task'] == fold['task'] and c['outer'] == fold['outer'] and c['stage'] == 'outer')
        cache, fitted = {}, {}
        configurations = {'dummy_mean': {'id': 'dummy_mean', 'family': 'dummy_mean', 'mc': None, 'preprocessing': 'raw'},
                          **{role: choice['config'] for role, choice in fold['choices'].items()}}
        for role in ROLES:
            config = configurations[role]
            for seed in SEEDS:
                key = (objhash({k: v for k, v in config.items() if k != 'id'}), seed)
                if key not in fitted:
                    record, rows = trial(out, context, config, seed, cache, outer=True)
                    fitted[key] = record, rows
                    records.append(record)
                else:
                    record, rows = fitted[key]
                    aliases.append({'task': fold['task'], 'outer': fold['outer'], 'role': role, 'seed': seed,
                                    'estimator_path': record['estimator_path'], 'candidate_id': config['id']})
                predictions.extend({**row, 'role': role, 'role_candidate_id': config['id']} for row in rows)
    if len(records) + len(aliases) != 360 or len(predictions) != 2880:
        raise LeakageError('Four tasks/six roles/three seeds/40 observations required')
    csvwrite(out / 'outer_predictions.csv.gz', predictions)
    write(out / 'outer_model_records.json', records)
    write(out / 'outer_role_aliases.json', aliases)
    lock = {'created_at_utc': utc(), 'outer_truth_parsed': False, 'prediction_rows': len(predictions),
            'actual_outer_estimators': len(records), 'alias_role_cases': len(aliases)}
    for key, name in (('prediction_sha256', 'outer_predictions.csv.gz'), ('outer_model_records_sha256', 'outer_model_records.json'),
                      ('outer_role_aliases_sha256', 'outer_role_aliases.json'), ('selection_lock_sha256', 'selection_lock.json'),
                      ('root_outer_release_sha256', 'outer_release_snapshot.json'), ('run_manifest_sha256', 'run_manifest.json'),
                      ('protocol_sha256', 'protocol.json')):
        lock[key] = sha(out / name)
    write(out / 'prediction_lock.json', lock)
    event(out, 'all_outer_predictions_locked', models=len(records), rows=len(predictions), aliases=len(aliases))
    return {'models': len(records), 'aliases': len(aliases), 'prediction_rows': len(predictions), 'outer_truth_parsed': False}


def verify_predictions_before_truth(out):
    out = local_directory(out)
    preparation, _, _ = verify_selected(out)
    lock = read(out / 'prediction_lock.json')
    checks = {'prediction_sha256': 'outer_predictions.csv.gz', 'outer_model_records_sha256': 'outer_model_records.json',
              'outer_role_aliases_sha256': 'outer_role_aliases.json', 'selection_lock_sha256': 'selection_lock.json',
              'root_outer_release_sha256': 'outer_release_snapshot.json', 'run_manifest_sha256': 'run_manifest.json',
              'protocol_sha256': 'protocol.json'}
    if lock.get('outer_truth_parsed') is not False or not lock.get('created_at_utc') or any(lock.get(key) != sha(out / name) for key, name in checks.items()):
        raise LeakageError('Prediction lock changed or truth already joined')
    verify_outer_release(out, read(out / 'outer_release_snapshot.json'))
    outer_contexts = [c for c in preparation['contexts'] if c['stage'] == 'outer']
    # verify_run already hashed all80 truth files; repeat all20 outer hashes before any parse.
    for context in outer_contexts:
        if sha(out / context['truth']) != context['truth_sha256']:
            raise LeakageError('Frozen outer truth bytes changed')
    records = read(out / 'outer_model_records.json')
    verify_records(out, records)
    if len(records) != lock['actual_outer_estimators'] or lock['prediction_rows'] != 2880:
        raise LeakageError('Incomplete outer model/prediction inventory')
    return outer_contexts, lock


def score(out):
    out = local_directory(out)
    if (out / 'scoring_manifest.json').exists():
        raise FileExistsError('Immutable results already scored')
    contexts, lock = verify_predictions_before_truth(out)
    event(out, 'score_start_after_all_truth_hash_checks')
    truth, context_by_record = {}, {}
    for context in contexts:
        for rid, value in zip(context['query_ids'], load_truth(out / context['truth'], context['query_ids']), strict=True):
            key = context['task'], rid
            if key in truth:
                raise LeakageError('Observation occurs in multiple outer folds for same task')
            truth[key] = value
            context_by_record[key] = context
    with gzip.open(out / 'outer_predictions.csv.gz', 'rt', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 2880:
        raise LeakageError('Prediction denominator changed')
    pooled, byfold, joined = {}, {}, []
    for row in rows:
        task, rid, role, seed = row['task'], row['record_id'], row['role'], int(row['seed'])
        if (task, rid) not in truth or role not in ROLES or seed not in SEEDS:
            raise LeakageError('Unexpected task, record, role or seed')
        context = context_by_record[(task, rid)]
        if int(row['outer']) != context['outer'] or row['split_id'] != context['split_id']:
            raise LeakageError('OOF prediction assigned to incorrect fruit fold')
        pooled.setdefault((task, role, seed), []).append(row)
        byfold.setdefault((task, role, seed, context['outer']), []).append(row)
        prediction = float(row['y_predicted'])
        joined.append({**row, 'y_true': float(truth[(task, rid)]), 'error': prediction-truth[(task, rid)],
                       'fruit_id': context['query_group_by_id'][rid]})
    if len(pooled) != 72 or len(byfold) != 360:
        raise LeakageError('Missing task/family/seed or outer fold')
    metrics, fold_metrics = [], []
    for (task, role, seed), items in sorted(pooled.items()):
        ids = [row['record_id'] for row in items]
        expected = {rid for t, rid in truth if t == task}
        if len(ids) != 40 or len(set(ids)) != 40 or set(ids) != expected:
            raise LeakageError('Every role/seed must predict40 unique observations from10 fruits')
        metrics.append({'task': task, 'role': role, 'seed': seed, 'fruit_groups': 10,
                        **regression_metrics([truth[(task, rid)] for rid in ids], [float(r['y_predicted']) for r in items])})
    for (task, role, seed, outer), items in sorted(byfold.items()):
        if len(items) != 8 or len({r['record_id'] for r in items}) != 8:
            raise LeakageError('Every outer fold must contain8 predictions from2 fruits')
        fold_metrics.append({'task': task, 'role': role, 'seed': seed, 'outer': outer, 'fruit_groups': 2,
                             **regression_metrics([truth[(task, r['record_id'])] for r in items], [float(r['y_predicted']) for r in items])})
    summaries = []
    for task in TASKS:
        for role in ROLES:
            current = [m for m in metrics if m['task'] == task and m['role'] == role]
            row = {'task': task, 'role': role, 'observations_per_seed': 40, 'fruit_groups': 10,
                   'seeds': list(SEEDS), 'target_units': 'Excel reported numeric reading units'}
            for name in ('rmse', 'mae', 'r2'):
                values = [m[name] for m in current]
                row[name+'_mean'] = float(np.mean(values))
                row[name+'_sample_sd'] = float(np.std(values, ddof=1))
            summaries.append(row)
    write(out / 'pooled_metrics_by_seed.json', metrics)
    write(out / 'fold_metrics.json', fold_metrics)
    write(out / 'seed_summary.json', summaries)
    csvwrite(out / 'scored_outer_predictions.csv.gz', joined)
    write(out / 'scoring_manifest.json', {'completed_at_utc': utc(), 'prediction_lock_sha256': sha(out / 'prediction_lock.json'),
                                        'prediction_sha256': lock['prediction_sha256'],
                                        'metrics_sha256': sha(out / 'pooled_metrics_by_seed.json'),
                                        'scored_predictions_sha256': sha(out / 'scored_outer_predictions.csv.gz'),
                                        'case_count': 72, 'observations_per_task_seed': 40, 'fruit_groups_per_task': 10,
                                        'source_model_fits_in_scoring': 0, 'scientific_status': STATUS,
                                        'unit_assay_license_verified': False, 'OS_file_isolation': False})
    event(out, 'score_complete', cases=72)
    return {'cases': 72, 'observations_per_task_seed': 40, 'status': STATUS}
