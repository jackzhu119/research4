#!/usr/bin/env python3
"""Continuously develop a matched ProtoNet on the frozen acid-jujube TRAIN only.

Checkpoints are a schedule, never a score-driven selection rule. This local
worker has no CPU, wall-time, score, retry or plateau stopping threshold.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    from herb_spectra.meta_learning import EpisodicEmbedding, prototype_probabilities
except ModuleNotFoundError:
    from meta_learning import EpisodicEmbedding, prototype_probabilities

GROUPS = ('ZY03', 'ZY04', 'ZY05', 'ZY06', 'ZY07')
SHOTS = (1, 3, 5, 10)
STOP_REQUESTED = False


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    return canonical_hash({'shape': list(value.shape), 'dtype': str(value.dtype),
                           'bytes': hashlib.sha256(value.tobytes()).hexdigest()})


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def validate_master(root):
    base = root / 'data/research/species_revision_20261108'
    registry_path = root / 'data/active_workbooks.json'
    metadata_path = root / 'data/metadata_template.csv'
    manifest = base / 'split_manifest.csv'
    master = json.loads((base / 'protocol.json').read_text())
    registry = json.loads(registry_path.read_text())
    for path, expected in ((registry_path, master['active_registry_sha256']),
                           (manifest, master['manifest_sha256']),
                           (metadata_path, master['metadata_sha256'])):
        if file_hash(path) != expected:
            raise ValueError('Frozen master input changed: ' + str(path))
    frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    if len(frame) != 1672 or not frame.record_id.is_unique or not frame.physical_sample_id.is_unique:
        raise ValueError('Frozen master must contain 1672 distinct record and physical identities')
    if frame.physical_sample_id.eq('').any():
        raise ValueError('Missing independent physical identity')
    if frame.role.value_counts().to_dict() != {'train': 999, 'prediction': 339, 'calibration': 334}:
        raise ValueError('Frozen master roles differ')
    workbook_hashes = {}
    for entry in registry['workbooks']:
        path = root / entry['path']
        if file_hash(path) != entry['sha256'] or entry['sha256'] != master['workbook_hashes'][entry['path']]:
            raise ValueError('Active workbook changed: ' + entry['path'])
        workbook_hashes[entry['path']] = entry['sha256']
    selected = frame.loc[frame.role.eq('train') & frame.origin_group_id.isin(GROUPS)].copy()
    selected = selected.sort_values('record_id').reset_index(drop=True)
    if len(selected) != 300 or not selected.species.eq('酸枣仁').all():
        raise ValueError('Require exactly 300 pure acid-jujube master TRAIN records')
    if selected.origin_group_id.value_counts().to_dict() != {group: 60 for group in GROUPS}:
        raise ValueError('Require sixty TRAIN samples per acid-jujube origin')
    return frame, selected, {'registry_sha256': file_hash(registry_path),
                             'manifest_sha256': file_hash(manifest),
                             'metadata_sha256': file_hash(metadata_path),
                             'workbook_sha256': workbook_hashes}


def load_inputs(root, output, cache_path=None, allow_prepare=True):
    """Read labels for identities; model arrays are whitelisted to TRAIN immediately."""
    frame, selected, provenance = validate_master(root)
    candidates = ([cache_path] if cache_path else []) + [
        root / 'reports/b_mechanism_20261110/.cache/features.npz',
        root / 'data/research/species_revision_20261108/features.npz',
        output / '.cache/features.npz']
    cache = next((Path(path) for path in candidates if path is not None and Path(path).is_file()), None)
    if cache is None:
        if not allow_prepare:
            raise ValueError('Read-only verification requires an existing audited feature cache')
        destination = output / '.cache'
        subprocess.run([sys.executable, str(root / 'scripts/prepare_species_revision_features.py'),
                        '--root', str(root), '--output', str(destination)], cwd=root, check=True)
        cache = destination / 'features.npz'
    audit_path = cache.parent / 'feature_audit.json'
    if not audit_path.is_file():
        raise ValueError('Feature cache requires its source reconstruction audit')
    audit = json.loads(audit_path.read_text())
    if (file_hash(cache) != audit['features_sha256']
            or provenance['registry_sha256'] != audit['registry_sha256']):
        raise ValueError('Feature cache bytes or active registry differ from cache audit')
    with np.load(cache, allow_pickle=False) as payload:
        ids = payload['record_ids'].astype(str)
        lookup = {identifier: index for index, identifier in enumerate(ids)}
        if len(ids) != len(lookup) or set(ids) != set(frame.record_id):
            raise ValueError('Feature/manifest record identities differ')
        indices = [lookup[identifier] for identifier in selected.record_id]
        arrays = {name: np.array(payload[name][indices], dtype=np.float64) for name in ('X_alpha', 'X_n')}
        axis = np.array(payload['axis'], dtype=np.float64)
        cached_species = payload['species'][indices].astype(str)
        cached_groups = payload['origin_group_ids'][indices].astype(str)
    if not np.array_equal(cached_species, selected.species) or not np.array_equal(cached_groups, selected.origin_group_id):
        raise ValueError('TRAIN feature labels disagree with current master')
    if axis.shape != (612,) or not np.all(np.diff(axis) > 0):
        raise ValueError('Require existing 612-point original-coordinate comparison')
    if any(value.shape != (300, 612) or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Invalid TRAIN-only model arrays')
    provenance.update(feature_cache_sha256=file_hash(cache), feature_audit_sha256=file_hash(audit_path),
                      train_array_sha256={key: array_hash(value) for key, value in arrays.items()},
                      axis_sha256=array_hash(axis),
                      whitelist_policy='Only these 300 acid TRAIN rows enter any model or new evaluation')
    return selected, arrays, axis, provenance


def build_design(frame, seeds=(20261031,)):
    """Exact E03 roles from the existing engine, restricted to chosen fixed seeds."""
    frame = frame.sort_values('record_id').reset_index(drop=True)
    jobs = []
    for seed in seeds:
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (inside, query) in enumerate(folds.split(frame, frame.origin_group_id)):
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), fold, 20261110]))
            remaining = frame.iloc[inside]
            source_ids, support_ids = [], []
            for group in GROUPS:
                order = rng.permutation(sorted(remaining.loc[remaining.origin_group_id.eq(group), 'record_id'].astype(str))).tolist()
                support_ids.extend(order[:10])
                source_ids.extend(order[10:])
            job = {'job_id': f'E03_s{seed}_f{fold}', 'seed': int(seed), 'fold': fold,
                   'source_ids': source_ids, 'support_ids': support_ids,
                   'query_ids': frame.iloc[query].record_id.tolist(), 'classes': list(GROUPS)}
            validate_job(job, frame)
            jobs.append(job)
    return jobs


def validate_job(job, frame):
    indexed = frame.set_index('record_id', drop=False)
    roles = [job[role + '_ids'] for role in ('source', 'support', 'query')]
    sets = [set(ids) for ids in roles]
    if any(len(ids) != len(unique) for ids, unique in zip(roles, sets)) or any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Source/Support/Query identities overlap')
    if tuple(map(len, roles)) != (190, 50, 60):
        raise ValueError('Expected exact inherited E03 source190/support50/query60')
    for role, per_class in (('source', 38), ('support', 10), ('query', 12)):
        side = indexed.loc[job[role + '_ids']]
        if not side.role.eq('train').all() or not side.species.eq('酸枣仁').all():
            raise ValueError('Only pure acid-jujube master TRAIN may enter these jobs')
        if side.origin_group_id.value_counts().to_dict() != {group: per_class for group in GROUPS}:
            raise ValueError('Invalid inherited role class counts')
    if not indexed.loc[sum(roles, []), 'physical_sample_id'].is_unique:
        raise ValueError('Physical samples overlap')


def support_draws(job, labels, draws=20):
    result = []
    candidates = np.asarray(job['support_ids'])
    support_labels = np.asarray([labels[value] for value in candidates])
    for draw in range(draws):
        rng = np.random.default_rng(np.random.SeedSequence([job['seed'], job['fold'], 20261111, draw]))
        orders = {label: rng.permutation(candidates[support_labels == label]).tolist() for label in range(5)}
        for k in SHOTS:
            # K10 is the same complete set for every draw, so fit it once.
            if k == 10 and draw:
                continue
            result.append({'k': k, 'draw': draw,
                           'ids': list(job['support_ids']) if k == 10 else sum([orders[label][:k] for label in range(5)], [])})
    return result


def snv(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    scale = matrix.std(axis=1, ddof=1, keepdims=True)
    if (scale == 0).any() or not np.isfinite(scale).all():
        raise ValueError('SNV received a constant or invalid spectrum')
    return (matrix - matrix.mean(axis=1, keepdims=True)) / scale


def representation(arrays, name):
    if name == 'alpha':
        return snv(arrays['X_alpha'])
    if name == 'n':
        return snv(arrays['X_n'])
    if name == 'fusion':
        return np.concatenate([snv(arrays['X_n']), snv(arrays['X_alpha'])], axis=1)
    raise ValueError('Unknown representation')


def metric_values(truth, predicted):
    return {'accuracy': float(accuracy_score(truth, predicted)),
            'macro_f1': float(f1_score(truth, predicted, labels=np.arange(5), average='macro', zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(truth, predicted)),
            'recall': recall_score(truth, predicted, labels=np.arange(5), average=None, zero_division=0).tolist(),
            'confusion_matrix': confusion_matrix(truth, predicted, labels=np.arange(5)).tolist()}


def svc_fit(X, y, scaler=None, gamma='scale'):
    scaler = StandardScaler().fit(X) if scaler is None else scaler
    model = SVC(C=10.0, gamma=gamma, class_weight=None, decision_function_shape='ovr',
                break_ties=True, random_state=42).fit(scaler.transform(X), y)
    return scaler, model


def scores_for_svc(bundle, X):
    scaler, model = bundle
    values = model.decision_function(scaler.transform(X))
    if not np.array_equal(model.classes_, np.arange(5)):
        raise ValueError('SVC class column order differs')
    return np.asarray(values, dtype=np.float64)


class Cell:
    def __init__(self, job, name, matrix, frame, config, axis=None):
        self.job, self.name, self.config = job, name, config
        self.index = {value: index for index, value in enumerate(frame.record_id)}
        self.labels = {row.record_id: GROUPS.index(row.origin_group_id) for row in frame.itertuples()}
        self.matrix = matrix
        self.axis = np.arange(matrix.shape[1] // (2 if name == 'fusion' else 1), dtype=float) if axis is None else np.asarray(axis, dtype=float)
        self.source_ids = job['source_ids']
        self.X_source = self.take(self.source_ids)
        self.y_source = self.y(self.source_ids)
        self.X_query = self.take(job['query_ids'])
        self.scaler = StandardScaler().fit(self.X_source)
        self.pca = PCA(n_components=32, svd_solver='full', random_state=42).fit(self.scaler.transform(self.X_source))
        self.Z_source = self.encode(self.X_source)
        self.Z_query = self.encode(self.X_query)
        self.draws = support_draws(job, self.labels, config['support_draws'])
        self.cell_id = name + '__' + job['job_id']
        self.baseline_metadata = {}
        self.fingerprint = canonical_hash({'job': job, 'representation': name, 'config': config,
                                           'X_source': array_hash(self.X_source),
                                           'scaler_mean': array_hash(self.scaler.mean_),
                                           'scaler_scale': array_hash(self.scaler.scale_),
                                           'pca_components': array_hash(self.pca.components_)})

    def take(self, ids):
        return self.matrix[[self.index[value] for value in ids]]

    def y(self, ids):
        return np.asarray([self.labels[value] for value in ids], dtype=np.int64)

    def encode(self, X):
        return self.pca.transform(self.scaler.transform(X))

    def make_model(self):
        return EpisodicEmbedding(32, 5, embedding_dim=self.config['embedding_dim'],
                                 learning_rate=self.config['learning_rate'], random_state=42,
                                 weight_decay=self.config['weight_decay'])

    def models(self, state):
        if state is None:
            return self.make_model(), self.make_model()
        return (EpisodicEmbedding.from_state_dict(state['episodic']),
                EpisodicEmbedding.from_state_dict(state['supervised']))

    def endpoint(self, method, k, draw, scores, ids, native_prediction=None):
        scores = np.asarray(scores)
        if scores.shape != (len(self.job['query_ids']), 5) or not np.isfinite(scores).all():
            raise ValueError('Invalid saved query score matrix')
        # Outer query truth is consulted only after completed predictions exist.
        prediction = scores.argmax(axis=1) if native_prediction is None else np.asarray(native_prediction, dtype=np.int64)
        truth = self.y(self.job['query_ids'])
        return {'endpoint_id': f'{method}__k{k}__d{draw}', 'method': method, 'k': k, 'draw': draw,
                'support_ids': ids,
                'decision_rule': 'score_argmax' if native_prediction is None else 'native_ovo_vote',
                **metric_values(truth, prediction)}, prediction, scores

    def evaluate(self, models=None, baseline=False):
        endpoints, predictions, scores = [], [], []
        def record(method, k, draw, values, ids, native_prediction=None):
            endpoint, prediction, score = self.endpoint(method, k, draw, values, ids, native_prediction)
            endpoints.append(endpoint)
            predictions.append(prediction)
            scores.append(score)
        if baseline:
            frozen = self.make_model()
            source = svc_fit(self.X_source, self.y_source)
            record('source_only_svm', 0, 0, scores_for_svc(source, self.X_query), [])
            source[1].break_ties = False
            native = source[1].predict(source[0].transform(self.X_query))
            source[1].break_ties = True
            record('source_native_svm', 0, 0, scores_for_svc(source, self.X_query), [], native)
            record('pca_prototype', 0, 0, prototype_probabilities(self.Z_source, self.y_source, self.Z_query, 5), [])
            record('random_encoder_prototype', 0, 0,
                   frozen.predict_proba(self.Z_query, self.Z_source, self.y_source), [])
            if self.config.get('strong_control', False):
                from meta_strong_control import fit_source_control, refit_control
                strong = fit_source_control(self.X_source, self.y_source, seed=self.job['seed'])
                support_ids = list(self.job['support_ids'])
                combined_X = np.concatenate([self.X_source, self.take(support_ids)])
                combined_y = np.concatenate([self.y_source, self.y(support_ids)])
                strong_refit = refit_control(strong, combined_X, combined_y)
                for label, model, k, ids in (('source', strong, 0, []),
                                             ('full_support_refit', strong_refit, 10, support_ids)):
                    values = model.scores(self.X_query)
                    record('mc20_tuned_' + label + '_argmax', k, 0, values, ids)
                    record('mc20_tuned_' + label + '_native', k, 0, values, ids, model.predict(self.X_query))
                self.baseline_metadata['strong_mc20_source'] = strong.metadata
                self.baseline_metadata['strong_mc20_full_support_refit'] = strong_refit.metadata
                for metadata in self.baseline_metadata.values():
                    metadata['selected_feature_coordinates'] = [
                        {'input_index': index,
                         'modality': ('n' if index < len(self.axis) else 'alpha') if self.name == 'fusion' else self.name,
                         'workbook_Thz_coordinate': float(self.axis[index % len(self.axis)])}
                        for index in metadata['selected_feature_indices']]
        else:
            for method, model in zip(('episodic_prototype', 'supervised_prototype'), models):
                record(method, 0, 0, model.predict_proba(self.Z_query, self.Z_source, self.y_source), [])
        for draw in self.draws:
            ids, k, number = draw['ids'], draw['k'], draw['draw']
            X, y = self.take(ids), self.y(ids)
            Z = self.encode(X)
            if baseline:
                record('pca_prototype', k, number, prototype_probabilities(Z, y, self.Z_query, 5), ids)
                record('random_encoder_prototype', k, number, frozen.predict_proba(self.Z_query, Z, y), ids)
                record('support_only_svm', k, number, scores_for_svc(svc_fit(X, y), self.X_query), ids)
                combined_X = np.concatenate([self.X_source, X])
                combined_y = np.concatenate([self.y_source, y])
                record('full_pipeline_refit_svm', k, number,
                       scores_for_svc(svc_fit(combined_X, combined_y), self.X_query), ids)
                record('frozen_coordinates_refit_svm', k, number,
                       scores_for_svc(svc_fit(combined_X, combined_y, source[0], source[1]._gamma), self.X_query), ids)
            else:
                for method, model in zip(('episodic_prototype', 'supervised_prototype'), models):
                    record(method, k, number, model.predict_proba(self.Z_query, Z, y), ids)
        return endpoints, np.asarray(predictions, dtype=np.uint8), np.asarray(scores, dtype=np.float64)


def seal_evaluation(directory, cell, step, endpoints, predictions, scores):
    if (directory / 'COMPLETE.json').exists():
        previous = verify_evaluation(directory, cell)
        with np.load(directory / 'predictions.npz', allow_pickle=False) as saved:
            equivalent = (np.array_equal(saved['predictions'], predictions)
                          and np.array_equal(saved['scores'], scores)
                          and list(saved['endpoint_ids']) == [value['endpoint_id'] for value in endpoints])
        if previous['training_steps'] != step or not equivalent:
            raise ValueError('Replayed evaluation differs from immutable sealed evidence; original preserved')
        return previous
    directory.mkdir(parents=True, exist_ok=True)
    payload = {'cell_id': cell.cell_id, 'training_steps': step, 'evaluated_at_utc': now(),
               'cell_hash': cell.fingerprint, 'query_ids': cell.job['query_ids'],
               'query_truth': cell.y(cell.job['query_ids']).tolist(), 'classes': list(GROUPS),
               'endpoints': endpoints, 'query_scope': 'master TRAIN internal development only',
               'selection_policy': 'All scheduled snapshots reported; no metric-driven selection'}
    if step == 0 and cell.baseline_metadata:
        payload['model_metadata'] = cell.baseline_metadata
    atomic_npz(directory / 'predictions.npz', predictions=predictions, scores=scores,
               query_ids=np.asarray(cell.job['query_ids']),
               true_labels=cell.y(cell.job['query_ids']),
               endpoint_ids=np.asarray([value['endpoint_id'] for value in endpoints]))
    payload['predictions_sha256'] = file_hash(directory / 'predictions.npz')
    atomic_json(directory / 'metrics.json', payload)
    atomic_json(directory / 'COMPLETE.json', {'metrics_sha256': file_hash(directory / 'metrics.json'),
                                             'predictions_sha256': payload['predictions_sha256'],
                                             'cell_hash': cell.fingerprint, 'training_steps': step})
    return payload


def verify_evaluation(directory, cell):
    seal = json.loads((directory / 'COMPLETE.json').read_text())
    if (seal['cell_hash'] != cell.fingerprint
            or seal['metrics_sha256'] != file_hash(directory / 'metrics.json')
            or seal['predictions_sha256'] != file_hash(directory / 'predictions.npz')):
        raise ValueError('Evaluation seal differs from saved artifacts')
    metrics = json.loads((directory / 'metrics.json').read_text())
    with np.load(directory / 'predictions.npz', allow_pickle=False) as saved:
        if list(saved['query_ids']) != cell.job['query_ids'] or not np.array_equal(saved['true_labels'], cell.y(cell.job['query_ids'])):
            raise ValueError('Saved query identity/truth differs')
        if len(metrics['endpoints']) != len(saved['predictions']):
            raise ValueError('Saved scores and predictions disagree')
        identifiers = [item['endpoint_id'] for item in metrics['endpoints']]
        if len(identifiers) != len(set(identifiers)) or identifiers != list(saved['endpoint_ids']):
            raise ValueError('Saved endpoint identities differ')
        for endpoint, predicted, score in zip(metrics['endpoints'], saved['predictions'], saved['scores']):
            if endpoint['decision_rule'] == 'score_argmax' and not np.array_equal(score.argmax(axis=1), predicted):
                raise ValueError('Saved score argmax and predictions disagree')
            recalculated = metric_values(saved['true_labels'], predicted)
            for key in ('accuracy', 'macro_f1', 'balanced_accuracy'):
                if not np.isclose(endpoint[key], recalculated[key], rtol=0, atol=1e-12):
                    raise ValueError('Independently recomputed metric differs')
            if endpoint['recall'] != recalculated['recall'] or endpoint['confusion_matrix'] != recalculated['confusion_matrix']:
                raise ValueError('Independently recomputed per-class metric differs')
    return metrics


def _read_checkpoint(path, cell):
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    payload = saved['payload']
    if saved['sha256'] != canonical_hash(payload) or payload['cell_hash'] != cell.fingerprint:
        raise ValueError('Checkpoint contents or input fingerprint changed')
    models = cell.models(payload)
    if models[0].step != models[1].step or models[0].step != payload['training_steps']:
        raise ValueError('Matched optimizer step counts differ')
    return payload


def load_checkpoint(path, cell, recover=False):
    """Recover a corrupt latest from one verified generation, preserving evidence."""
    previous_path = path.with_name('checkpoint.previous.json')
    if not path.exists() and previous_path.exists():
        if not recover:
            raise ValueError('Latest checkpoint is absent while a previous generation exists')
        payload = _read_checkpoint(previous_path, cell)
        atomic_json(path, {'sha256': canonical_hash(payload), 'payload': payload})
        atomic_json(path.with_name('checkpoint_recovery.json'), {
            'recovered_at_utc': now(), 'error': 'Latest checkpoint absent',
            'previous_sha256': file_hash(previous_path), 'restored_training_steps': payload['training_steps']})
        return payload
    try:
        return _read_checkpoint(path, cell)
    except (OSError, ValueError, KeyError, TypeError) as error:
        if not recover:
            raise
        previous = previous_path
        payload = _read_checkpoint(previous, cell)
        if payload is None:
            raise ValueError('Latest checkpoint failed and no verified previous generation exists') from error
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        quarantine = path.with_name('checkpoint.corrupt.' + stamp + '.json')
        os.replace(path, quarantine)
        atomic_json(path, {'sha256': canonical_hash(payload), 'payload': payload})
        atomic_json(path.with_name('checkpoint_recovery.json'), {
            'recovered_at_utc': now(), 'error': repr(error), 'preserved_corrupt_checkpoint': str(quarantine),
            'previous_sha256': file_hash(previous), 'restored_training_steps': payload['training_steps']})
        return payload


def save_checkpoint(path, cell, episodic, supervised, cpu_seconds, training):
    payload = {'schema_version': 1, 'cell_hash': cell.fingerprint, 'cell_id': cell.cell_id,
               'training_steps': episodic.step, 'saved_at_utc': now(),
               'episodic': episodic.state_dict(), 'supervised': supervised.state_dict(),
               'accumulated_training_cpu_seconds': cpu_seconds, 'last_training_chunks': training}
    if episodic.step != supervised.step:
        raise ValueError('Cannot commit unmatched training budgets')
    if path.exists():
        # Verify before rotating; a corrupt current file never replaces a good previous.
        previous = _read_checkpoint(path, cell)
        atomic_json(path.with_name('checkpoint.previous.json'),
                    {'sha256': canonical_hash(previous), 'payload': previous})
    atomic_json(path, {'sha256': canonical_hash(payload), 'payload': payload})
    return payload


def scheduled_evaluation(step, chunk_steps):
    count = step // chunk_steps
    return step > 0 and step % chunk_steps == 0 and count & (count - 1) == 0


def signal_stop(signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('reports/meta_learning_continuous_20261111'))
    parser.add_argument('--feature-cache', type=Path)
    parser.add_argument('--representations', nargs='+', choices=('alpha', 'n', 'fusion'), default=['alpha'])
    parser.add_argument('--outer-seeds', nargs='+', type=int, default=[20261031])
    parser.add_argument('--chunk-steps', type=int, default=250)
    parser.add_argument('--support-draws', type=int, default=20)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--once', action='store_true', help='Run one complete cycle for explicit verification only')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    args.output = args.output.resolve() if args.output.is_absolute() else args.root / args.output
    if args.chunk_steps <= 0 or args.support_draws <= 0:
        parser.error('Chunk size and support draws must be positive; they are work units, not run stop thresholds')
    if len(args.representations) != len(set(args.representations)) or len(args.outer_seeds) != len(set(args.outer_seeds)):
        parser.error('Repeated representation or seed')
    return args


def science_code_hashes(root, paths):
    result = {}
    for item in paths:
        path = Path(item).resolve()
        try:
            key = path.relative_to(root.resolve()).as_posix()
        except ValueError:
            key = path.name
        if key in result:
            raise ValueError('Duplicate scientific code identity: ' + key)
        result[key] = file_hash(path)
    return result


def prepare(args, readonly=False):
    sys.path[:0] = [str(args.root / 'src'), str(args.root / 'scripts')]
    import meta_strong_control
    import herb_spectra.selection as selection_module
    import herb_spectra.staged_nested_search as search_module
    frame, arrays, axis, inputs = load_inputs(args.root, args.output, args.feature_cache,
                                             allow_prepare=not readonly)
    jobs = build_design(frame, args.outer_seeds)
    config = {'chunk_steps': args.chunk_steps, 'support_draws': args.support_draws,
              'representations': args.representations, 'outer_seeds': args.outer_seeds,
              'embedding_dim': 16, 'learning_rate': 0.001, 'weight_decay': 0.0001,
              'training_support_per_class': 5, 'training_query_per_class': 5,
              'pca_components': 32, 'model_seed': 42, 'temperature': 1.0,
              'strong_control': True}
    module_path = Path(sys.modules[EpisodicEmbedding.__module__].__file__).resolve()
    protocol = {'protocol_id': 'train-only-episodic-protonet-20261111-v1', 'inputs': inputs,
                'config': config, 'design': jobs,
                'frequency_axis': axis.tolist(),
                'science_code_sha256': science_code_hashes(args.root, [__file__, module_path,
                    meta_strong_control.__file__, selection_module.__file__, search_module.__file__]),
                'runtime': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'pandas')},
                'compute_policy': {'authorized_cpu_hours': 900000, 'actual_reserved_cpu_hours': None,
                                   'cpu_stop_limit': None, 'wall_stop_limit': None, 'metric_stop_limit': None,
                                   'scope': 'Current local host lifetime; no platform uptime guarantee'},
                'interpretation': 'Known-origin, same-collection internal representation development; not new-batch or unseen-class validation',
                'snapshot_policy': 'Latest and previous optimizer checkpoints every chunk; detailed evaluation at chunk counts 1,2,4,8,..., all retained; never choose checkpoint using outer Query',
                'support_policy': '20 fixed nested support draws; identical complete K10 pool evaluated once',
                'strong_control': {'algorithm': 'Source-internal-2fold SNV/MC-UVE20/tuned SVM',
                    'k_candidates': [31, 62, 123, 184, 245], 'C_candidates': [0.1, 1, 10, 100],
                    'gamma_candidates': ['scale', 'scale_x0.1', 'scale_x10'],
                    'endpoints': 'source K0 and all-Support K10 full-Pipeline refit, each native OVO vote and OVR argmax',
                    'query_policy': 'No outer Query is passed to fitting/selection; K10 freezes Source-chosen K/C/gamma',
                    'historical_scope': 'Method-family control within E03 Source190; historical 240-train 93.4444% score is not reused'},
                'fixed_svm': {'C': 10.0, 'class_weight': None, 'gamma': 'scale from fitted source coordinates',
                              'feature_selector': None, 'representation': 'full-spectrum, modality-block SNV',
                              'scope': 'New matched full-spectrum controls; no historical score or source model reuse is claimed',
                              'decision_rule': 'OVR score argmax with break_ties=True; source native OVO vote is also reported separately'}}
    protocol['input_hash'] = canonical_hash(protocol)
    path = args.output / 'frozen_protocol.json'
    whitelist_path = args.output / 'train_whitelist.json'
    whitelist = {'record_ids': frame.record_id.tolist(), 'master_role': 'train',
                 'selected_records': 300, 'input_hash': protocol['input_hash']}
    for destination, expected in ((path, protocol), (whitelist_path, whitelist)):
        if destination.exists():
            if json.loads(destination.read_text()) != expected:
                raise ValueError('Frozen input changed; use a new output directory instead of mixing results: ' + destination.name)
        elif readonly:
            raise ValueError('Read-only verification requires existing ' + destination.name)
        else:
            atomic_json(destination, expected)
    cells = [Cell(job, name, representation(arrays, name), frame, config, axis=axis)
             for name in args.representations for job in jobs]
    return protocol, cells


def execute(args, protocol, cells):
    args.output.mkdir(parents=True, exist_ok=True)
    startup = time.monotonic()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, signal_stop)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    iteration = 0
    while not STOP_REQUESTED:
        iteration += 1
        for cell in cells:
            if STOP_REQUESTED:
                break
            directory = args.output / 'cells' / cell.cell_id
            directory.mkdir(parents=True, exist_ok=True)
            state_path = directory / 'checkpoint.json'
            checkpoint = load_checkpoint(state_path, cell, recover=True)
            models = cell.models(checkpoint)
            step = models[0].step
            atomic_json(args.output / 'execution_state.json', {
                'status': 'RUNNING', 'current_cell': cell.cell_id, 'current_training_steps': step,
                'heartbeat_at_utc': now(), 'input_hash': protocol['input_hash'],
                'process_pid': os.getpid(), 'quality_cpu_wall_stop_limit': None,
                'process_cpu_seconds': time.process_time(), 'process_wall_seconds': time.monotonic() - startup,
                'completed_cycles_this_process': iteration - 1})
            baseline = directory / 'baseline'
            if not (baseline / 'COMPLETE.json').is_file():
                endpoints, predictions, scores = cell.evaluate(baseline=True)
                seal_evaluation(baseline, cell, 0, endpoints, predictions, scores)
            verify_evaluation(baseline, cell)
            # A checkpoint may have committed before evaluation or before process exit.
            if scheduled_evaluation(step, args.chunk_steps):
                pending = directory / 'snapshots' / f'step_{step:012d}'
                if not (pending / 'COMPLETE.json').exists():
                    values = cell.evaluate(models=models)
                    seal_evaluation(pending, cell, step, *values)
                verify_evaluation(pending, cell)
            cpu = time.process_time()
            training = []
            for mode, model in zip(('episodic', 'supervised'), models):
                training.append(model.train_chunk(cell.Z_source, cell.y_source,
                    steps=args.chunk_steps, support_per_class=5, query_per_class=5, mode=mode))
            accumulated = (checkpoint or {}).get('accumulated_training_cpu_seconds', 0.0) + time.process_time() - cpu
            checkpoint = save_checkpoint(state_path, cell, *models, accumulated, training)
            step = models[0].step
            evaluated = scheduled_evaluation(step, args.chunk_steps)
            if evaluated:
                snapshot = directory / 'snapshots' / f'step_{step:012d}'
                values = cell.evaluate(models=models)
                seal_evaluation(snapshot, cell, step, *values)
                verify_evaluation(snapshot, cell)
            atomic_json(directory / 'latest_training.json', {
                'at_utc': now(), 'training_steps': step, 'cell_hash': cell.fingerprint,
                'accumulated_training_cpu_seconds': accumulated, 'training_chunks': training,
                'evaluation_saved_this_chunk': evaluated, 'quality_cpu_wall_stop_limit': None})
            if evaluated:
                print(json.dumps({'at_utc': now(), 'cell': cell.cell_id, 'training_steps': step,
                                  'checkpoint_sha256': file_hash(state_path),
                                  'training_cpu_seconds': accumulated,
                                  'episodic_training_loss': training[0]['mean_training_loss'],
                                  'supervised_training_loss': training[1]['mean_training_loss'],
                                  'status': 'CHECKPOINT_SAVED', 'evaluation_saved': evaluated}, ensure_ascii=False), flush=True)
        if args.once:
            break
    atomic_json(args.output / 'execution_state.json', {
        'status': 'EXPLICIT_SINGLE_CYCLE_DONE' if args.once else 'SIGNAL_CHECKPOINTED',
        'input_hash': protocol['input_hash'], 'process_pid': os.getpid(),
        'heartbeat_at_utc': now(), 'quality_cpu_wall_stop_limit': None,
        'process_cpu_seconds': time.process_time(), 'process_wall_seconds': time.monotonic() - startup})


def main(argv=None):
    args = parse_args(argv)
    if args.verify:
        protocol, cells = prepare(args, readonly=True)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        lock = (args.output / 'worker.lock').open('a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, cells = prepare(args)
    if args.prepare:
        print(json.dumps({'status': 'PREPARED_NO_FIT', 'cells': len(cells), 'input_hash': protocol['input_hash']}), flush=True)
    elif args.verify:
        counts = {'baselines': 0, 'snapshots': 0, 'checkpoints': 0}
        for cell in cells:
            directory = args.output / 'cells' / cell.cell_id
            if (directory / 'checkpoint.json').exists():
                load_checkpoint(directory / 'checkpoint.json', cell)
                counts['checkpoints'] += 1
            if (directory / 'baseline/COMPLETE.json').exists():
                verify_evaluation(directory / 'baseline', cell)
                counts['baselines'] += 1
            for snapshot in sorted((directory / 'snapshots').glob('step_*')):
                if (snapshot / 'COMPLETE.json').exists():
                    verify_evaluation(snapshot, cell)
                    counts['snapshots'] += 1
        print(json.dumps({'status': 'VERIFIED', **counts}), flush=True)
    else:
        execute(args, protocol, cells)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
