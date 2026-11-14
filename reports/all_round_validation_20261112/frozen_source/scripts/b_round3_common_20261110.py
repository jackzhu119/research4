"""Read-only evidence resolution for the third frozen B mechanism study."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from b_followup_common_20261110 import (TaskWriter, canonical_hash, endpoint_id, file_hash,
                                      json_safe, load_parent, metrics, validate_roles,
                                      verify_task, write_json)


def load_round2_bundle(round2root, job_id):
    root = Path(round2root)
    if not job_id or Path(job_id).name != job_id:
        raise ValueError('Invalid second-round evidence identifier')
    protocol = json.loads((root / 'frozen_protocol.json').read_text())
    definition = next((job for job in protocol['design'] if job['job_id'] == job_id), None)
    if definition is None:
        raise ValueError('Parent is outside the second-round frozen protocol')
    path = root / 'jobs' / job_id
    result = json.loads((path / 'result.json').read_text())
    seal = json.loads((path / 'COMPLETE.json').read_text())
    if result['input_hash'] != protocol['input_hash'] or seal['input_hash'] != protocol['input_hash']:
        raise ValueError('Second-round input identity mismatch')
    if result['job_hash'] != canonical_hash(definition) or seal['job_hash'] != result['job_hash']:
        raise ValueError('Second-round task definition mismatch')
    if seal['result_sha256'] != file_hash(path / 'result.json'):
        raise ValueError('Second-round result seal mismatch')
    if canonical_hash(json.loads((path / 'split_ids.json').read_text())) != result['job_hash']:
        raise ValueError('Second-round role lock mismatch')
    for name, expected in result['artifact_hashes'].items():
        if Path(name).name != name or file_hash(path / name) != expected:
            raise ValueError('Second-round artifact mismatch: ' + name)
    return result, path


def check_parent_roles(job, result):
    for name in ('source_ids', 'support_ids', 'query_ids', 'classes', 'label_column'):
        if job[name] != result[name]:
            raise ValueError('Reused endpoint has a different sample/label definition: ' + name)


def _model(path, result, name):
    if name not in result['artifact_hashes'] or file_hash(path / name) != result['artifact_hashes'][name]:
        raise ValueError('Model is absent from verified parent evidence: ' + name)
    return joblib.load(path / name)


def load_true_refit(job, k, oldroot, round2root):
    """Resolve the exact true-label, refitted-Scaler/fixed-gamma old endpoint."""
    if k not in (1, 3, 5, 10):
        raise ValueError('Unsupported fixed Support budget')
    identifier = 'F02_' + job['parent_job_id']
    result, path = load_round2_bundle(round2root, identifier)
    check_parent_roles(job, result)
    name = f'scaler_refit_gamma_fixed_k{k}.joblib'
    if name in result['artifact_hashes']:
        model = _model(path, result, name)
        provenance = {'round': 2, 'job_id': identifier, 'artifact': name,
                      'artifact_sha256': result['artifact_hashes'][name]}
    else:
        if k not in result.get('exact_gamma_reuse_k', []):
            raise ValueError('No declared exact-gamma reuse for the requested endpoint')
        old_result, source, oldpath = load_parent(job, oldroot)
        oldname = f'retrain_k{k}.joblib'
        model = _model(oldpath, old_result, oldname)
        if float(model.estimator_._gamma) != float(source.estimator_._gamma):
            raise ValueError('Historical numeric gamma is not exactly the source gamma')
        provenance = {'round': 1, 'job_id': job['parent_job_id'], 'artifact': oldname,
                      'artifact_sha256': old_result['artifact_hashes'][oldname],
                      'qualification_job': identifier, 'reason': 'EXACT_EFFECTIVE_GAMMA_MATCH'}
    return model, provenance


def load_weight_model(job, setting, oldroot, round2root):
    """Resolve the balanced/None model without refitting equivalent objectives."""
    if setting not in ('balanced', 'none'):
        raise ValueError('Unknown class-weight setting')
    identifier = job.get('phase2_job_id')
    result, path = load_round2_bundle(round2root, identifier)
    check_parent_roles(job, result)
    if setting == 'none' and job['ratio'] == 'B':
        model, provenance = load_weight_model(job, 'balanced', oldroot, round2root)
        if not np.array_equal(model.estimator_.class_weight_, np.ones(len(job['classes']))):
            raise ValueError('Equivalent class-weight reuse is not justified')
        return model, {**provenance, 'requested_class_weight': 'none',
                       'reason': 'EQUAL_EFFECTIVE_CLASS_WEIGHTS'}
    if setting == 'balanced' and job.get('parent_job_id'):
        old_result, model, oldpath = load_parent(job, oldroot)
        return model, {'round': 1, 'job_id': job['parent_job_id'], 'artifact': 'source_model.joblib',
                       'artifact_sha256': old_result['artifact_hashes']['source_model.joblib'],
                       'qualification_job': identifier, 'requested_class_weight': setting}
    name = f'class_weight_{setting}.joblib'
    model = _model(path, result, name)
    return model, {'round': 2, 'job_id': identifier, 'artifact': name,
                   'artifact_sha256': result['artifact_hashes'][name], 'requested_class_weight': setting}
