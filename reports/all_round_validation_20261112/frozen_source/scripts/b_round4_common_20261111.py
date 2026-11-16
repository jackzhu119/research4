"""Read-only historical evidence resolution for the fourth B mechanism study."""
from __future__ import annotations

import joblib

from b_followup_common_20261110 import (TaskWriter, canonical_hash, endpoint_id, file_hash,
                                      json_safe, load_parent, metrics, validate_roles,
                                      verify_task, write_json)
from b_round3_common_20261110 import (check_parent_roles, load_round2_bundle,
                                     load_weight_model)


def load_round3_bundle(round3root, job_id):
    # Both sealed generations use the same protocol, role and artifact format.
    try:
        return load_round2_bundle(round3root, job_id)
    except ValueError as error:
        raise ValueError('Third-round evidence verification failed: ' + str(error)) from error


def load_frozen_refit(job, k, oldroot, round2root):
    """Resolve Source+Support C10 with original Source scaler/numeric gamma."""
    if k not in (1, 3, 5, 10):
        raise ValueError('Unexpected frozen Support budget')
    identifier = 'F02_' + job['parent_job_id']
    result, path = load_round2_bundle(round2root, identifier)
    check_parent_roles(job, result)
    name = f'scaler_frozen_gamma_fixed_k{k}.joblib'
    if name not in result['artifact_hashes'] or file_hash(path / name) != result['artifact_hashes'][name]:
        raise ValueError('The exact historical frozen-coordinate endpoint is absent or changed')
    model = joblib.load(path / name)
    _, source, _ = load_parent(job, oldroot)
    if (joblib.hash(model.scaler_) != joblib.hash(source.scaler_)
            or float(model.estimator_._gamma) != float(source.estimator_._gamma)
            or float(model.estimator_.C) != 10.0):
        raise ValueError('Historical endpoint does not keep Source coordinates and C10')
    return model, {'round': 2, 'job_id': identifier, 'artifact': name,
                   'artifact_sha256': result['artifact_hashes'][name],
                   'reason': 'EXACT_FROZEN_SCALER_FIXED_GAMMA_C10_ENDPOINT'}
