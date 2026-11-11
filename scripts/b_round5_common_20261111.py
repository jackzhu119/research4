"""Verified historical endpoint access for the fifth frozen B study."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from b_followup_common_20261110 import (TaskWriter, canonical_hash, endpoint_id, file_hash,
                                      json_safe, load_parent, metrics, validate_roles,
                                      verify_task, write_json)
from b_round3_common_20261110 import check_parent_roles, load_round2_bundle
from b_round4_common_20261111 import load_frozen_refit


def load_bundle(root, job_id):
    """Rounds two and later share this immutable job/role/artifact format."""
    try:
        return load_round2_bundle(root, job_id)
    except ValueError as error:
        raise ValueError('Historical endpoint verification failed: ' + str(error)) from error


def load_scores(root, job_id, method, k, draw, query_ids):
    """Read a verified endpoint; no historical Query truth is returned."""
    result, path = load_bundle(root, job_id)
    if set(query_ids) != set(result['query_ids']) or len(query_ids) != len(result['query_ids']):
        raise ValueError('Historical endpoint has different Query identities')
    identifier = endpoint_id(method, k, draw)
    metadata = {item['endpoint_id']: item for item in json.loads((path / 'endpoints.json').read_text())}
    if identifier not in metadata:
        raise ValueError('Requested endpoint is absent from historical metadata')
    predictions = pd.read_csv(path / 'predictions.csv.gz', keep_default_na=False)
    rows = predictions.loc[predictions.endpoint_id.eq(identifier)]
    if (len(rows) != len(query_ids) or not rows.record_id.is_unique
            or set(rows.record_id) != set(query_ids)):
        raise ValueError('Historical endpoint Query rows are incomplete')
    rows = rows.set_index('record_id').loc[query_ids]
    scores = np.asarray([json.loads(value) for value in rows.scores_json], dtype=float)
    predicted = rows.pred_label.to_numpy()
    if scores.shape != (len(query_ids), len(result['classes'])) or not np.isfinite(scores).all():
        raise ValueError('Historical score dimensions or finite values disagree')
    if rows.decision_rule.eq('score_argmax').all():
        if not np.array_equal(np.asarray(result['classes'])[scores.argmax(axis=1)], predicted):
            raise ValueError('Historical score-argmax endpoint does not reproduce')
    provenance = {'historical_job_id': job_id, 'historical_endpoint_id': identifier,
                  'historical_result_sha256': file_hash(path / 'result.json'),
                  'historical_prediction_sha256': result['artifact_hashes']['predictions.csv.gz'],
                  'historical_input_hash': result['input_hash']}
    return scores, predicted, metadata[identifier], provenance
