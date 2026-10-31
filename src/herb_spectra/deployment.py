"""Versioned anonymous seven-group deployment, separate from research OOF scores.

All 855 historical specimens are development data. Five preserved partitions
are reused only for selecting a deployment configuration. No new external score
is inferred from this process or from full-development refitting.
"""
from __future__ import annotations
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .classification import classification_metrics
from .leakage_guard import FitScope, LeakageError, instrument_mcuve_pls
from .selection import MCUVESelector
from .spectral_models import SpectralPreprocessor
from .staged_nested_search import (StagedSVM, TransformBundle, class_weight_spec,
                                  gamma_spec, retained_count)

ROOT = Path(__file__).resolve().parents[2]
DATA = 'data/multiplant_database/zaoren/aligned_modalities.npz'
DATA_SHA = '6d1a6917b0f97ab431ec91e78621292b04ddc5cba56a02c0c9c641b2d0035647'
SOURCE_LOCK = 'reports/staged-nested-optimization/2026-10-31/core/selection_lock.json'
SPLITS = 'reports/staged-nested-optimization/2026-10-31/core/splits.json'
CLASSES = tuple(f'ZY{i:02}' for i in range(1, 8))
SEEDS = (42, 43, 44)
PROTOCOL_ID = 'zaoren-anonymous-full-development-deployment-20261031-v1'


def utc(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
def spectrum_hash(row):
    return hashlib.sha256(np.asarray(row, dtype='<f8').tobytes(order='C')).hexdigest()

def fresh(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError('A fresh output directory is required')
    path.mkdir(parents=True, exist_ok=True)
    return path

def code_versions():
    names = ['src/herb_spectra/deployment.py', 'scripts/run_zaoren_deployment.py',
             'src/herb_spectra/deployment_io.py', 'src/herb_spectra/leakage_guard.py',
             'src/herb_spectra/classification.py', 'src/herb_spectra/selection.py',
             'src/herb_spectra/spectral_models.py', 'src/herb_spectra/staged_nested_search.py',
             'src/herb_spectra/image_feature_models.py']
    return {name: sha(ROOT / name) for name in names}

def environment():
    return {'python': platform.python_version(), 'packages': {name: importlib.metadata.version(name)
            for name in ['numpy', 'scipy', 'scikit-learn', 'joblib', 'openpyxl']}}

def load_data():
    if sha(ROOT / DATA) != DATA_SHA: raise LeakageError('Canonical source data changed')
    with np.load(ROOT / DATA, allow_pickle=False) as data:
        x, y, ids, axis = [data[k].copy() for k in ('X_alpha', 'y', 'record_ids', 'axis_alpha')]
    if (x.shape != (855, 612) or axis.shape != (612,) or not np.isfinite(x).all()
            or len(set(ids)) != 855 or set(y) != set(CLASSES)):
        raise LeakageError('Canonical seven-group development data shape changed')
    return x, y, ids, axis

def source_candidates():
    lock = read(ROOT / SOURCE_LOCK)
    candidates = []
    for item in lock['outer_folds']:
        choice = item['choices']['R2']
        if choice['selected_using_outer_test']: raise LeakageError('Candidate used historical outer score')
        config = json.loads(json.dumps(choice['selected']))
        config['id'] = f"r2_source_fold{item['outer_fold']:02}"
        if config['preprocessing'] != 'snv': raise LeakageError('Only frozen SNV candidates are permitted')
        candidates.append(config)
    if len(candidates) != 5: raise LeakageError('Expected five source-selected historical configurations')
    return candidates

def prepare(run):
    run = fresh(run); x, y, ids, axis = load_data()
    source_splits = read(ROOT / SPLITS)
    folds = []
    for item in source_splits:
        train, val = item['train_sample_ids'], item['test_sample_ids']
        if set(train) & set(val) or set(train) | set(val) != set(ids):
            raise LeakageError('Development partition identity mismatch')
        folds.append({'fold': item['fold'], 'train_ids': train, 'validation_ids': val})
    if len(folds) != 5 or sorted(v for f in folds for v in f['validation_ids']) != sorted(ids):
        raise LeakageError('Each development record must appear in one validation fold')
    protocol = {'protocol_id': PROTOCOL_ID, 'data': DATA, 'data_sha256': DATA_SHA,
        'candidate_source': SOURCE_LOCK, 'candidate_source_sha256': sha(ROOT / SOURCE_LOCK),
        'partition_source': SPLITS, 'partition_source_sha256': sha(ROOT / SPLITS),
        'candidates': source_candidates(), 'seeds': list(SEEDS), 'primary_seed': 42,
        'folds': folds, 'expected_source_models': 75, 'expected_full_models': 3,
        'selection': 'mean15 development cases Macro-F1, Balanced Accuracy, Accuracy, candidate ID',
        'all_855_are_development': True, 'development_cv_is_independent_confirmation': False,
        'new_external_accuracy': None, 'historical_96p02_is_deployment_accuracy': False,
        'future_unknown_group_detection_validated': False,
        'scientific_status': 'EXPLORATORY_SINGLE_SOURCE_BATCH_ANONYMOUS_GROUPS'}
    write(run / 'protocol.json', protocol)
    preparation = {'created_at_utc': utc(), 'records': 855, 'features': 612,
        'classes': list(CLASSES), 'class_counts': {c: int(np.sum(y == c)) for c in CLASSES},
        'record_ids': ids.tolist(), 'axis': axis.tolist(), 'source_sha256': code_versions(),
        'environment': environment(), 'protocol_sha256': sha(run / 'protocol.json'),
        'historical_spectrum_hashes': {str(r): spectrum_hash(v) for r, v in zip(ids, x, strict=True)},
        'data_sha256': DATA_SHA, 'physical_independence_basis': 'user_declaration',
        'no_new_query_data_used': True, 'OS_enforced_file_isolation': False}
    write(run / 'preparation.json', preparation)
    write(run / 'candidate_manifest.json', current_manifest(run))
    return {'status': 'PREPARED_NOT_TRAINED', 'planned_source_fits': 75, 'planned_full_fits': 3}

def current_manifest(run):
    run = Path(run)
    return {'protocol_sha256': sha(run / 'protocol.json'),
            'preparation_sha256': sha(run / 'preparation.json'),
            'source_sha256': code_versions(), 'environment': environment()}

def verify_run(run):
    run = Path(run); manifest = read(run / 'candidate_manifest.json')
    if current_manifest(run) != manifest: raise LeakageError('Frozen implementation, environment or preparation drifted')
    protocol = read(run / 'protocol.json')
    if sha(ROOT / SOURCE_LOCK) != protocol['candidate_source_sha256'] or sha(ROOT / SPLITS) != protocol['partition_source_sha256']:
        raise LeakageError('Historical source-selection provenance changed')
    return protocol, read(run / 'preparation.json'), manifest

def verify_source_gate(run, gate):
    _, _, manifest = verify_run(run)
    if not gate.get('allow_source_training') or any(gate.get(k) != manifest[k]
            for k in ('protocol_sha256', 'preparation_sha256', 'source_sha256')):
        raise LeakageError('Root source gate must bind frozen protocol, preparation and code')

def fit_model(config, x, y, ids, axis, seed, *, split_id, log_path, expected_source_ids):
    x = np.asarray(x, dtype=float); y = np.asarray(y); ids = np.asarray(ids, dtype=str)
    if (set(ids) != set(expected_source_ids) or len(ids) != len(expected_source_ids)
            or len(set(ids)) != len(ids) or len(x) != len(ids) or len(y) != len(ids)):
        raise LeakageError('Actual model source IDs differ from allowed source IDs')
    if config['preprocessing'] != 'snv': raise LeakageError('Deployment candidate must use SNV')
    scope = FitScope(expected_source_ids, split_id, PROTOCOL_ID, log_path,
                     code_version=sha(Path(__file__)))
    prep = SpectralPreprocessor('snv', axis)
    prepared = prep.transform(x)
    scaler = scope.fit(StandardScaler(), prepared, sample_ids=ids, component='standard_scaler')
    scaled = scaler.transform(prepared); mc = config['mc']
    selector = MCUVESelector(n_runs=mc['runs'], n_components=mc['components'],
        n_features_to_select=retained_count(x.shape[1], mc['ratio']), sample_fraction=.8,
        task='classification', random_state=seed)
    with instrument_mcuve_pls(scope, scaled, ids):
        scope.fit(selector, scaled, y, sample_ids=ids, component='MC-UVE')
    selected = selector.transform(scaled); svm = config['svm']
    weight, weights = class_weight_spec(y, svm['class_weight'])
    gamma, numeric_gamma = gamma_spec(selected, svm['gamma'])
    estimator = scope.fit(SVC(C=svm['C'], kernel='rbf', gamma=gamma, class_weight=weight,
        break_ties=False, decision_function_shape='ovr', random_state=seed),
        selected, y, sample_ids=ids, component='SVC')
    metadata = {'config': config, 'seed': seed, 'split_id': split_id, 'source_ids': ids.tolist(),
        'source_X_sha256': hashlib.sha256(np.asarray(x, dtype='<f8').tobytes()).hexdigest(),
        'source_y_sha256': hashlib.sha256(json.dumps(y.tolist()).encode()).hexdigest(),
        'selected_indices': selector.selected_indices_.tolist(),
        'selected_frequencies': np.asarray(axis)[selector.selected_indices_].tolist(),
        'selected_features': len(selector.selected_indices_), 'effective_gamma': float(estimator._gamma),
        'declared_gamma_numeric': numeric_gamma, 'training_class_weights': weights,
        'scaler_mean': scaler.mean_.tolist(), 'scaler_variance': scaler.var_.tolist(),
        'prediction_rule': 'SVC standard vote, break_ties=False', 'scores_are_probabilities': False,
        'fit_scope': 'current_allowed_source_only', 'classes': estimator.classes_.tolist()}
    return StagedSVM(config, TransformBundle(prep, scaler, selector, metadata), estimator, metadata)

def save_model(model, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_suffix('.json').exists(): raise FileExistsError(path)
    joblib.dump(model, path, compress=3)
    write(path.with_suffix('.json'), model.metadata)
    return {'path': str(path), 'sha256': sha(path), 'metadata_sha256': sha(path.with_suffix('.json'))}

def select(run, gate_path):
    run = Path(run); gate = read(gate_path); verify_source_gate(run, gate)
    if (run / 'source_release.json').exists(): raise FileExistsError('Selection was already started; preserve failed attempts')
    write(run / 'source_release.json', gate)
    protocol, prep, manifest = verify_run(run); x, y, ids, axis = load_data()
    indices = {v: i for i, v in enumerate(ids)}; records = []; prediction_rows = []
    for candidate in protocol['candidates']:
        for fold in protocol['folds']:
            tr = np.asarray([indices[v] for v in fold['train_ids']]); va = np.asarray([indices[v] for v in fold['validation_ids']])
            for seed in protocol['seeds']:
                start = time.perf_counter(); split = f"development.fold{fold['fold']:02}"
                model = fit_model(candidate, x[tr], y[tr], ids[tr], axis, seed,
                    split_id=split, log_path=run / 'fit_scope_events.jsonl', expected_source_ids=ids[tr])
                predicted = model.predict(x[va]); scores = model.predict_scores(x[va])
                path = run / 'selection' / candidate['id'] / f"fold{fold['fold']:02}" / f'seed{seed}.joblib'
                saved = save_model(model, path)
                metric = classification_metrics(y[va], predicted, CLASSES)
                records.append({'candidate_id': candidate['id'], 'fold': fold['fold'], 'seed': seed,
                    'metrics': metric, 'model': saved, 'elapsed_seconds': time.perf_counter() - start})
                prediction_rows.extend({'candidate_id': candidate['id'], 'fold': fold['fold'], 'seed': seed,
                    'record_id': str(rid), 'y_true_development_only': str(truth), 'y_predicted': str(label),
                    **{f'score_{c}': float(s) for c, s in zip(CLASSES, row, strict=True)}}
                    for rid, truth, label, row in zip(ids[va], y[va], predicted, scores, strict=True))
                write(run / f"progress_{candidate['id']}_fold{fold['fold']}_seed{seed}.json", {'completed': len(records), 'planned': 75})
    means = []
    for config in protocol['candidates']:
        rows = [r for r in records if r['candidate_id'] == config['id']]
        means.append({'candidate': config, 'mean_development_metrics': {k: float(np.mean([r['metrics'][k] for r in rows]))
                     for k in ('macro_f1', 'balanced_accuracy', 'accuracy')}, 'cases': len(rows)})
    winner = sorted(means, key=lambda row: tuple(-row['mean_development_metrics'][k]
        for k in ('macro_f1', 'balanced_accuracy', 'accuracy')) + (row['candidate']['id'],))[0]
    write(run / 'selection_records.json', records)
    write_csv(run / 'development_predictions.csv', prediction_rows)
    lock = {'created_at_utc': utc(), 'selected': winner['candidate'], 'candidate_means': means,
        'source_models': len(records), 'manifest_sha256': sha(run / 'candidate_manifest.json'),
        'source_release_sha256': sha(run / 'source_release.json'),
        'selection_records_sha256': sha(run / 'selection_records.json'),
        'predictions_sha256': sha(run / 'development_predictions.csv'),
        'development_metrics_are_confirmation': False, 'external_query_labels_received': False,
        'primary_seed': 42, 'seeds': list(SEEDS), 'full_development_training_allowed': False}
    write(run / 'selection_lock.json', lock)
    return {'status': 'SOURCE_SELECTION_COMPLETE_NO_FULL_FIT', 'models': len(records), 'selected': winner['candidate']}

def write_csv(path, rows):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(rows)

def verify_selection(run):
    run = Path(run); protocol, prep, manifest = verify_run(run); lock = read(run / 'selection_lock.json')
    pairs = {'manifest_sha256': run / 'candidate_manifest.json', 'source_release_sha256': run / 'source_release.json',
             'selection_records_sha256': run / 'selection_records.json', 'predictions_sha256': run / 'development_predictions.csv'}
    if any(lock[k] != sha(p) for k, p in pairs.items()) or lock['source_models'] != 75:
        raise LeakageError('Frozen source selection evidence changed or incomplete')
    records = read(run / 'selection_records.json')
    if len(records) != 75: raise LeakageError('Incomplete source model records')
    for row in records:
        item = row['model']; path = Path(item['path'])
        if sha(path) != item['sha256'] or sha(path.with_suffix('.json')) != item['metadata_sha256']:
            raise LeakageError('Saved source model changed')
    return protocol, prep, lock

def fit_full(run, release_path):
    run = Path(run); protocol, prep, lock = verify_selection(run); release = read(release_path)
    if (not release.get('allow_full_fit') or release.get('selection_lock_sha256') != sha(run / 'selection_lock.json')
            or release.get('run_manifest_sha256') != sha(run / 'candidate_manifest.json')):
        raise LeakageError('Full refit release must bind independently reviewed selection')
    bundle = fresh(run / 'deployment'); write(bundle / 'full_fit_release.json', release)
    x, y, ids, axis = load_data(); models = {}
    for seed in protocol['seeds']:
        start = time.perf_counter()
        model = fit_model(lock['selected'], x, y, ids, axis, seed, split_id='full_development855',
                          log_path=run / 'fit_scope_events.jsonl', expected_source_ids=ids)
        item = save_model(model, bundle / f'model_seed{seed}.joblib')
        item['path'] = Path(item['path']).name
        item['elapsed_seconds'] = time.perf_counter() - start
        models[str(seed)] = item
    np.savez_compressed(bundle / 'history_reference.npz', X_alpha=x, record_ids=ids, axis=axis)
    manifest = {'created_at_utc': utc(), 'protocol_id': PROTOCOL_ID, 'classes': list(CLASSES),
        'axis': axis.tolist(), 'selected_config': lock['selected'], 'source_count': 855,
        'models': models, 'primary_seed': 42, 'seeds': list(SEEDS), 'source_sha256': code_versions(),
        'environment': environment(), 'historical_spectrum_hashes': prep['historical_spectrum_hashes'],
        'selection_lock_sha256': sha(run / 'selection_lock.json'),
        'full_fit_release_sha256': sha(bundle / 'full_fit_release.json'),
        'history_reference_sha256': sha(bundle / 'history_reference.npz'),
        'historical_roundtrip_matching': {'atol': 1e-12, 'rtol': 1e-12, 'rule': 'all612 coordinates; re-export QC, not biological identity proof'},
        'external_accuracy': None, 'resubstitution_accuracy_reported_as_performance': False,
        'historical_96p02_is_this_models_accuracy': False, 'unknown_group_detection_validated': False,
        'scientific_status': 'FROZEN_DEPLOYMENT_REQUIRES_FUTURE_VALIDATION'}
    write(bundle / 'bundle_manifest.json', manifest)
    return {'status': 'FULL_DEVELOPMENT_MODELS_FROZEN', 'models': 3, 'primary_seed': 42, 'external_accuracy': None}

def load_bundle(bundle):
    bundle = Path(bundle); manifest = read(bundle / 'bundle_manifest.json')
    if manifest['source_sha256'] != code_versions() or manifest['environment'] != environment():
        raise LeakageError('Deployment implementation or environment differs from frozen bundle')
    if manifest['classes'] != list(CLASSES) or manifest['primary_seed'] != 42:
        raise LeakageError('Deployment class order or primary seed changed')
    if sha(bundle / 'full_fit_release.json') != manifest['full_fit_release_sha256']:
        raise LeakageError('Full-fit release changed')
    if sha(bundle / 'history_reference.npz') != manifest['history_reference_sha256']:
        raise LeakageError('Frozen historical QC reference changed')
    models = {}
    for seed, item in manifest['models'].items():
        path = bundle / item['path']
        if sha(path) != item['sha256'] or sha(path.with_suffix('.json')) != item['metadata_sha256']:
            raise LeakageError('Deployment model integrity mismatch')
        model = joblib.load(path)
        if list(model.classes_) != list(CLASSES) or model.estimator.break_ties:
            raise LeakageError('Model class order or standard voting changed')
        models[seed] = model
    if set(models) != {'42', '43', '44'}: raise LeakageError('All frozen seeds required')
    return manifest, models

def historical_matches(x, ids, historical, reference):
    x = np.asarray(x, dtype=float); reference = np.asarray(reference, dtype=float)
    if reference.ndim != 2 or x.ndim != 2 or reference.shape[1] != x.shape[1]:
        raise LeakageError('Frozen historical QC reference shape mismatch')
    values = set(historical.values())
    id_flags = [str(r) in historical for r in ids]
    exact_flags = [spectrum_hash(row) in values for row in x]
    # Each query is compared independently to frozen development reference only.
    # This checks numerical re-exports, and makes no biological duplicate claim.
    roundtrip_flags = [bool(np.any(np.all(np.isclose(reference, row[None, :],
                              atol=1e-12, rtol=1e-12), axis=1))) for row in x]
    return id_flags, exact_flags, roundtrip_flags


def predict(bundle, input_path, output, *, mode='future', confirm_compatible_measurement=False):
    from .deployment_io import load_prediction_input
    manifest, models = load_bundle(bundle)
    if mode not in {'future', 'training-diagnostic'}: raise ValueError('Unknown prediction mode')
    if mode == 'future' and not confirm_compatible_measurement:
        raise LeakageError('Confirm matching absorption definition, units and measurement conditions')
    data = load_prediction_input(input_path, np.asarray(manifest['axis']))
    x, ids = data['X_alpha'], data['record_ids']; historical = manifest['historical_spectrum_hashes']
    with np.load(Path(bundle) / 'history_reference.npz', allow_pickle=False) as reference:
        if not np.array_equal(reference['axis'], np.asarray(manifest['axis'])) or set(reference['record_ids']) != set(historical):
            raise LeakageError('Historical reference axis or IDs changed')
        matched_ids, matched_values, matched_roundtrip = historical_matches(x, ids, historical, reference['X_alpha'])
    if mode == 'future' and any(a or b or c for a, b, c in zip(matched_ids, matched_values, matched_roundtrip, strict=True)):
        raise LeakageError('Historical spectra/IDs cannot be presented as new samples')
    before = {s: joblib.hash(m) for s, m in models.items()}
    primary = models['42']; predictions = primary.predict(x); scores = primary.predict_scores(x)
    alternatives = {s: m.predict(x) for s, m in models.items()}
    # Actual state and batch/order checks use fitted objects, never fit().
    reverse = primary.predict_scores(x[::-1])[::-1]
    singles = np.vstack([primary.predict_scores(x[i:i+1]) for i in range(len(x))])
    if not np.allclose(scores, reverse, atol=1e-12, rtol=1e-12) or not np.allclose(scores, singles, atol=1e-12, rtol=1e-12):
        raise LeakageError('Pointwise inference changed under order/batch')
    if before != {s: joblib.hash(m) for s, m in models.items()}: raise LeakageError('Inference modified model state')
    result = []
    for i, (rid, label, row) in enumerate(zip(ids, predictions, scores, strict=True)):
        result.append({'record_id': str(rid), 'predicted_group': str(label), 'primary_seed': 42,
            'historical_id_match': matched_ids[i], 'historical_value_match': matched_values[i],
            'historical_roundtrip_match': matched_roundtrip[i],
            'negative_alpha_cells': int(np.sum(x[i] < 0)),
            'prediction_scope': 'TRAINING_DIAGNOSTIC_NOT_PERFORMANCE' if mode == 'training-diagnostic' else 'UNSCORED_FUTURE_INPUT',
            **{f'seed{s}_prediction': str(v[i]) for s, v in alternatives.items()},
            **{f'score_{c}': float(v) for c, v in zip(CLASSES, row, strict=True)}})
    output = fresh(output); write_csv(output / 'predictions.csv', result)
    report = {'created_at_utc': utc(), 'mode': mode, 'records': len(result), 'input_sha256': data['source_sha256'],
        'bundle_manifest_sha256': sha(Path(bundle) / 'bundle_manifest.json'), 'predictions_sha256': sha(output / 'predictions.csv'),
        'class_order': list(CLASSES), 'qc': data['qc'], 'labels_received': False, 'new_model_fits': 0,
        'scores_are_probabilities': False, 'accuracy': None, 'pointwise_state_checks_passed': True,
        'user_attested_measurement_compatibility': bool(confirm_compatible_measurement),
        'measurement_compatibility_verified': False,
        'unknown_group_detection_validated': False}
    write(output / 'prediction_lock.json', report)
    return report

def template(bundle, output):
    from .deployment_io import write_prediction_template
    manifest, _ = load_bundle(bundle)
    return write_prediction_template(output, np.asarray(manifest['axis']))
