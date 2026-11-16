"""Atomic artifacts and independent verification for the frozen B follow-up."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), ensure_ascii=False,
                                    indent=2, allow_nan=False) + '\n', encoding='utf-8')


def endpoint_id(method, k=0, draw=0):
    return f'{method}__k{k}__d{draw}'


def metrics(true, predicted, classes):
    return {'n_query': len(true), 'n_unique_query': len(true),
            'accuracy': float(accuracy_score(true, predicted)),
            'macro_f1': float(f1_score(true, predicted, labels=classes, average='macro', zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(true, predicted)),
            'recall_json': json.dumps(dict(zip(classes, recall_score(true, predicted, labels=classes,
                                                          average=None, zero_division=0).tolist())), ensure_ascii=False),
            'confusion_matrix_json': json.dumps(confusion_matrix(true, predicted, labels=classes).tolist()),
            'classes_json': json.dumps(list(classes), ensure_ascii=False)}


def validate_roles(job, metadata):
    indexed = metadata.set_index('record_id', drop=False)
    if not metadata.record_id.is_unique or not metadata.physical_sample_id.is_unique:
        raise ValueError('Record/physical IDs must be unique')
    parts = [job[name + '_ids'] for name in ('source', 'support', 'query')]
    groups = [set(ids) for ids in parts]
    if any(len(ids) != len(unique) for ids, unique in zip(parts, groups)):
        raise ValueError('Repeated ID within role')
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError('Source/Support/Query roles overlap')
    ids = set.union(*groups)
    if not ids.issubset(indexed.index) or not indexed.loc[sorted(ids), 'role'].eq('train').all():
        raise ValueError('Only current master TRAIN may be used')
    if sorted(indexed.loc[job['source_ids'], job['label_column']].unique()) != job['classes']:
        raise ValueError('Source class vocabulary differs')
    for group in ('ZY01', 'ZY02'):
        if not metadata.loc[metadata.origin_group_id.eq(group), 'species'].eq('枣仁').all():
            raise ValueError('Current kind semantics violated')
    for group in ('ZY03', 'ZY04', 'ZY05', 'ZY06', 'ZY07'):
        if not metadata.loc[metadata.origin_group_id.eq(group), 'species'].eq('酸枣仁').all():
            raise ValueError('Current kind semantics violated')
    return True


def load_parent(job, oldroot):
    oldroot = Path(oldroot)
    parent_id = job.get('parent_job_id')
    if not parent_id or Path(parent_id).name != parent_id:
        raise ValueError('Missing/invalid historical parent identifier')
    path = oldroot / 'jobs' / parent_id
    protocol = json.loads((oldroot / 'frozen_protocol.json').read_text())
    original = next((value for value in protocol['design'] if value['job_id'] == parent_id), None)
    if original is None:
        raise ValueError('Historical parent is outside frozen protocol')
    result = json.loads((path / 'result.json').read_text())
    seal = json.loads((path / 'COMPLETE.json').read_text())
    if result['input_hash'] != protocol['input_hash'] or seal['input_hash'] != protocol['input_hash']:
        raise ValueError('Historical input hash changed')
    if seal['result_sha256'] != file_hash(path / 'result.json'):
        raise ValueError('Historical result hash changed')
    if result['job_hash'] != canonical_hash(original) or seal['job_hash'] != result['job_hash']:
        raise ValueError('Historical job changed')
    for key in ('source_ids', 'support_ids', 'query_ids', 'classes', 'label_column'):
        if job[key] != original[key]:
            raise ValueError('Reused historical sample/label definition differs: ' + key)
    for name, expected in result['artifact_hashes'].items():
        if Path(name).name != name or file_hash(path / name) != expected:
            raise ValueError('Historical artifact changed: ' + name)
    return result, joblib.load(path / 'source_model.joblib'), path


class TaskWriter:
    def __init__(self, job, X, metadata, axis, output, input_hash, oldroot):
        validate_roles(job, metadata)
        self.job = job
        self.X = np.asarray(X, dtype=float)
        self.metadata = metadata.reset_index(drop=True).copy()
        self.indexed = self.metadata.set_index('record_id', drop=False)
        self.axis = np.asarray(axis, dtype=float)
        self.output, self.oldroot, self.input_hash = Path(output), Path(oldroot), input_hash
        if (self.X.ndim != 2 or len(self.X) != len(metadata) or not np.isfinite(self.X).all()
                or self.axis.shape != (self.X.shape[1],) or not (np.diff(self.axis) > 0).all()):
            raise ValueError('Feature alignment, finite values or axis invalid')
        self.lookup = {value: index for index, value in enumerate(self.metadata.record_id)}
        self.classes = np.asarray(job['classes'])
        self.destination = self.output / 'jobs' / job['job_id']
        if self.destination.exists():
            raise FileExistsError('Refusing to overwrite sealed task')
        self.temp = self.output / 'jobs' / f'.{job["job_id"]}.tmp.{os.getpid()}'
        self.temp.parent.mkdir(parents=True, exist_ok=True)
        if self.temp.exists():
            shutil.rmtree(self.temp)
        self.temp.mkdir()
        self.query_X = self.take(job['query_ids'])
        self._rows, self._metrics, self._endpoints = [], [], []
        self.start_cpu, self.start_wall = time.process_time(), time.monotonic()

    def take(self, ids):
        return self.X[[self.lookup[sample] for sample in ids]]

    def labels(self, ids):
        return self.indexed.loc[ids, self.job['label_column']].to_numpy()

    def save_json(self, name, value):
        if Path(name).name != name:
            raise ValueError('Artifact name must be local')
        write_json(self.temp / name, value)

    def save_model(self, name, model):
        if Path(name).name != name:
            raise ValueError('Artifact name must be local')
        joblib.dump(model, self.temp / name, compress=3)

    def record(self, method, k, scores, pred=None, draw=0, support_ids=None,
               decision_rule='score_argmax', **extra):
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (len(self.job['query_ids']), len(self.classes)) or not np.isfinite(scores).all():
            raise ValueError('Invalid query scores')
        predicted = self.classes[scores.argmax(axis=1)] if pred is None else np.asarray(pred)
        if predicted.shape != (len(scores),) or not set(predicted).issubset(self.classes):
            raise ValueError('Invalid predicted labels')
        # Labels first join only after the caller has generated its predictions.
        true = self.labels(self.job['query_ids'])
        identifier = endpoint_id(method, k, draw)
        if identifier in {item['endpoint_id'] for item in self._endpoints}:
            raise ValueError('Duplicate method/K/draw endpoint')
        support_ids = [] if support_ids is None else list(support_ids)
        if not set(support_ids).issubset(self.job['support_ids']):
            raise ValueError('Support drawn from outside frozen candidate pool')
        context = {key: self.job.get(key) for key in ('job_id', 'family', 'experiment', 'parent_job_id',
                                                      'target_origin', 'seed', 'coverage', 'ratio', 'fold')}
        common = {**context, 'endpoint_id': identifier, 'method': method, 'k': k, 'draw': draw,
                  'decision_rule': decision_rule, 'total_support_labels': len(support_ids)}
        self._endpoints.append(json_safe({**common, 'support_ids': support_ids, **extra}))
        self._metrics.append(json_safe({**common, **metrics(true, predicted, self.classes.tolist()), **extra}))
        margin = ((scores[:, 1] - scores[:, 0]) / 2 if len(self.classes) == 2 else
                  np.sort(scores, axis=1)[:, -1] - np.sort(scores, axis=1)[:, -2])
        for index, record_id in enumerate(self.job['query_ids']):
            self._rows.append({**common, 'record_id': record_id,
                               'physical_sample_id': self.indexed.at[record_id, 'physical_sample_id'],
                               'origin_group_id': self.indexed.at[record_id, 'origin_group_id'],
                               'true_label': true[index], 'pred_label': predicted[index],
                               'margin': float(margin[index]), 'scores_json': json.dumps(scores[index].tolist())})
        return identifier

    def finish(self, expected_endpoints, new_model_fits=0, **extra):
        expected = list(expected_endpoints)
        actual = [item['endpoint_id'] for item in self._endpoints]
        if len(expected) != len(set(expected)) or set(actual) != set(expected):
            raise ValueError('Incomplete frozen endpoint matrix')
        pd.DataFrame(self._rows).to_csv(self.temp / 'predictions.csv.gz', index=False,
                                       compression={'method': 'gzip', 'mtime': 0})
        pd.DataFrame(self._metrics).to_csv(self.temp / 'metrics.csv', index=False)
        self.save_json('endpoints.json', self._endpoints)
        self.save_json('split_ids.json', self.job)
        result = {**self.job, **json_safe(extra), 'status': 'DONE', 'input_hash': self.input_hash,
                  'job_hash': canonical_hash(self.job), 'expected_endpoints': expected,
                  'new_model_fits': int(new_model_fits),
                  'cpu_seconds': time.process_time() - self.start_cpu,
                  'wall_seconds': time.monotonic() - self.start_wall,
                  'metric_rows': len(self._metrics), 'prediction_rows': len(self._rows),
                  'artifact_hashes': {p.name: file_hash(p) for p in sorted(self.temp.iterdir()) if p.is_file()}}
        write_json(self.temp / 'result.json', result)
        write_json(self.temp / 'COMPLETE.json', {'input_hash': self.input_hash,
                                               'job_hash': canonical_hash(self.job),
                                               'result_sha256': file_hash(self.temp / 'result.json')})
        os.rename(self.temp, self.destination)
        return result


def verify_task(path, input_hash=None, expected_job=None, expected_endpoints=None):
    path = Path(path)
    result = json.loads((path / 'result.json').read_text())
    seal = json.loads((path / 'COMPLETE.json').read_text())
    if input_hash is not None and result['input_hash'] != input_hash:
        raise ValueError('Scientific input/code hash changed')
    if seal['input_hash'] != result['input_hash'] or seal['result_sha256'] != file_hash(path / 'result.json'):
        raise ValueError('Task seal or result changed')
    split = json.loads((path / 'split_ids.json').read_text())
    if canonical_hash(split) != result['job_hash'] or result['job_hash'] != seal['job_hash']:
        raise ValueError('Frozen task split changed')
    if expected_job is not None and result['job_hash'] != canonical_hash(expected_job):
        raise ValueError('Task differs from frozen design')
    for name, expected in result['artifact_hashes'].items():
        if Path(name).name != name or file_hash(path / name) != expected:
            raise ValueError('Task artifact changed: ' + name)
    predictions = pd.read_csv(path / 'predictions.csv.gz', keep_default_na=False)
    measured = pd.read_csv(path / 'metrics.csv', keep_default_na=False)
    expected = result['expected_endpoints'] if expected_endpoints is None else list(expected_endpoints)
    if (len(expected) != len(set(expected)) or len(measured) != len(expected)
            or set(measured.endpoint_id) != set(expected) or set(predictions.endpoint_id) != set(expected)
            or set(result['expected_endpoints']) != set(expected)):
        raise ValueError('Incomplete endpoint coverage')
    endpoint_meta = json.loads((path / 'endpoints.json').read_text())
    if {item['endpoint_id'] for item in endpoint_meta} != set(expected):
        raise ValueError('Incomplete endpoint split metadata')
    for identifier, rows in predictions.groupby('endpoint_id', sort=False):
        if (len(rows) != len(result['query_ids']) or not rows.record_id.is_unique
                or not rows.physical_sample_id.is_unique or set(rows.record_id) != set(result['query_ids'])):
            raise ValueError('Query IDs not complete and mutually aligned')
        metric = measured.loc[measured.endpoint_id.eq(identifier)].iloc[0]
        fresh = metrics(rows.true_label, rows.pred_label, result['classes'])
        for name in ('accuracy', 'macro_f1', 'balanced_accuracy'):
            if not np.isclose(float(metric[name]), fresh[name], rtol=0, atol=1e-12):
                raise ValueError('Independent metric mismatch: ' + name)
        for name in ('recall_json', 'confusion_matrix_json', 'classes_json'):
            if json.loads(metric[name]) != json.loads(fresh[name]):
                raise ValueError('Independent metric mismatch: ' + name)
        if rows.decision_rule.eq('score_argmax').all():
            scores = np.asarray([json.loads(value) for value in rows.scores_json])
            if not np.array_equal(np.asarray(result['classes'])[scores.argmax(axis=1)], rows.pred_label.to_numpy()):
                raise ValueError('Score argmax and saved predictions disagree')
    return result
