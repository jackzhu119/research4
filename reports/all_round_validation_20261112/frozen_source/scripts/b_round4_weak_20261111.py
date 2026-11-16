"""H03: zero-fit audit of weak score motion and decision-boundary activation.

All score arrays and fitted adapters come from sealed previous experiments.
The crossing calculation takes no labels and chooses no blend. Its sufficient
margin bound applies to these finite saved score vectors, not to future data.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from b_followup_common_20261110 import TaskWriter, endpoint_id, file_hash


ACTUAL_BLEND = 0.05
F01_METHOD = "raw_weak_lambda1000_blend005"
G03_METHOD = "weak_lambda1000_blend005"


def f01_episodes():
    return [(k, draw) for draw in (0, *range(101, 111)) for k in (1, 3, 5, 10)
            if k != 10 or draw == 0]


def g03_episodes():
    return [(10, q * 1000 + seed) for q in (1, 2, 4, 10)
            for seed in ((201, 202, 203, 204, 205) if q != 10 else (201,))]


def expected_endpoints(job=None):
    result = [endpoint_id("identity"), endpoint_id("svc_predict")]
    result += [endpoint_id("f01_weak", k, draw) for k, draw in f01_episodes()]
    result += [endpoint_id("g03_weak", k, draw) for k, draw in g03_episodes()]
    if len(result) != 52 or len(set(result)) != 52:
        raise AssertionError("Frozen weak audit endpoint matrix changed")
    return result


def fit_count(job=None):
    return 0


def adapter_count(job=None):
    return 0


def crossing_diagnostics(source_scores, weak_scores, actual_blend=ACTUAL_BLEND):
    """Describe an affine score path, without labels or parameter selection.

    ``first_crossing_blend`` is the first nonnegative boundary on the path
    ``source + t * (weak-source)/actual_blend``. On a boundary, argmax uses
    the first class index. A higher-index rival therefore switches just after
    the boundary, while a lower-index rival can switch at equality. Infinity
    is retained when no competitor can overtake the original winning class.
    """
    source = np.asarray(source_scores, dtype=float)
    weak = np.asarray(weak_scores, dtype=float)
    if (source.ndim != 2 or source.shape != weak.shape or source.shape[1] < 2
            or not len(source) or not np.isfinite(source).all() or not np.isfinite(weak).all()):
        raise ValueError("Scores must be matching nonempty finite N x C arrays")
    if not np.isfinite(actual_blend) or actual_blend <= 0:
        raise ValueError("The recorded actual blend must be finite and positive")
    n_rows, n_classes = source.shape
    index = np.arange(n_rows)
    winners = source.argmax(axis=1)
    weak_winners = weak.argmax(axis=1)
    correction = weak - source
    direction = correction / float(actual_blend)
    sorted_source = np.sort(source, axis=1)
    gap = sorted_source[:, -1] - sorted_source[:, -2]
    correction_norm = np.max(np.abs(correction), axis=1)
    direction_norm = np.max(np.abs(direction), axis=1)
    changed = winners != weak_winners
    score_changed = np.any(correction != 0, axis=1)
    certified = gap > 2.0 * correction_norm

    deficits = source[index, winners, None] - source
    relative_slope = direction - direction[index, winners, None]
    rivals = np.arange(n_classes)[None, :] != winners[:, None]
    overtaking = rivals & (relative_slope > 0)
    crossings = np.full(source.shape, np.inf)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        crossings[overtaking] = deficits[overtaking] / relative_slope[overtaking]
    first = crossings.min(axis=1)
    finite = np.isfinite(first)
    candidate = np.where(finite, crossings.argmin(axis=1), -1)
    boundary_tie_switch = np.zeros(n_rows, dtype=bool)
    for row in np.flatnonzero(finite):
        simultaneous = np.flatnonzero(crossings[row] == first[row])
        boundary_tie_switch[row] = bool(np.any(simultaneous < winners[row]))

    path_prediction = (source + float(actual_blend) * direction).argmax(axis=1)
    # This tolerance describes floating arithmetic near a crossing, and never
    # discards an endpoint or controls whether the queue runs.
    roundoff = 1e-12 * np.maximum(1.0, np.abs(first))
    definitely_before = finite & (first < actual_blend - roundoff)
    definitely_after = (~finite) | (first > actual_blend + roundoff)
    implication_consistent = ~((definitely_before & ~changed) | (definitely_after & changed))
    return {
        "source_winner_index": winners,
        "weak_winner_index": weak_winners,
        "source_top2_gap": gap,
        "actual_correction_linf": correction_norm,
        "path_direction_linf": direction_norm,
        "score_changed": score_changed,
        "decision_changed": changed,
        "score_moved_decision_unchanged": score_changed & ~changed,
        "strict_margin_certificate": certified,
        "certificate_violation": certified & changed,
        "first_crossing_blend": first,
        "first_crossing_finite": finite,
        "crossing_competitor_index": candidate,
        "crossing_boundary_switches_at_tie": boundary_tie_switch,
        "first_crossing_le_actual_blend": finite & (first <= actual_blend),
        "first_crossing_le_one": finite & (first <= 1.0),
        "path_prediction_consistent": path_prediction == weak_winners,
        "crossing_implication_consistent": implication_consistent,
    }


def _rows(frame, method, k, draw, query_ids, classes, *, score_rule=True):
    selected = frame.loc[frame.method.eq(method) & frame.k.eq(k) & frame.draw.eq(draw)]
    if len(selected) != len(query_ids) or not selected.record_id.is_unique:
        raise ValueError("Saved weak endpoint has incomplete or duplicate Query IDs")
    if set(selected.record_id) != set(query_ids):
        raise ValueError("Saved weak endpoint differs from the frozen Query")
    block = selected.set_index("record_id", drop=False).loc[query_ids]
    scores = np.asarray([json.loads(value) for value in block.scores_json], dtype=float)
    predicted = block.pred_label.to_numpy()
    if score_rule and not np.array_equal(np.asarray(classes)[scores.argmax(axis=1)], predicted):
        raise ValueError("Saved score-based weak prediction does not reproduce its scores")
    return block, scores, predicted


def _check_roles(job, historical):
    for field in ("source_ids", "support_ids", "query_ids", "classes", "label_column"):
        if job[field] != historical[field]:
            raise ValueError("Weak audit parent has different frozen roles: " + field)


def _parameter_diagnostics(adapter, n_classes):
    if float(adapter.get("blend", ACTUAL_BLEND)) != ACTUAL_BLEND:
        raise ValueError("Saved weak adapter has a different actual blend")
    coefficient = np.asarray(adapter["coef"], dtype=float)
    intercept = np.asarray(adapter["intercept"], dtype=float)
    if coefficient.shape != (n_classes, n_classes) or intercept.shape != (n_classes,):
        raise ValueError("Saved weak adapter class dimensions differ")
    return {
        "adapter_parameter_changed": bool(not np.array_equal(coefficient, np.eye(n_classes))
                                           or not np.array_equal(intercept, np.zeros(n_classes))),
        "adapter_coefficient_correction_frobenius": float(np.linalg.norm(coefficient - np.eye(n_classes))),
        "adapter_intercept_l2": float(np.linalg.norm(intercept)),
        "adapter_regularization": float(adapter["regularization"]),
    }


def execute(job, X, metadata, axis, output, input_hash, oldroot, round2root, round3root):
    """Seal 50 saved weak-score audits and two matched zero-support controls."""
    from b_round4_common_20261111 import load_round2_bundle, load_round3_bundle

    writer = TaskWriter(job, X, metadata, axis, output, input_hash, oldroot)
    parent = job["parent_job_id"]
    second_result, second_path = load_round2_bundle(round2root, "F01_" + parent)
    third_result, third_path = load_round3_bundle(round3root, "G03_" + parent)
    _check_roles(job, second_result)
    _check_roles(job, third_result)
    second_predictions = pd.read_csv(second_path / "predictions.csv.gz", keep_default_na=False)
    third_predictions = pd.read_csv(third_path / "predictions.csv.gz", keep_default_na=False)
    second_endpoints = {item["endpoint_id"]: item for item in json.loads((second_path / "endpoints.json").read_text())}
    third_endpoints = {item["endpoint_id"]: item for item in json.loads((third_path / "endpoints.json").read_text())}
    second_parameters = {item["endpoint_id"]: item for item in json.loads((second_path / "adapter_parameters.json").read_text())}
    third_parameter_rows = json.loads((third_path / "adapter_parameters.json").read_text())
    third_parameters = {(item["method"], item["draw"]): item for item in third_parameter_rows}
    third_objective_parameters = {(item["method"], item["objective_key"]): item for item in third_parameter_rows}
    classes = job["classes"]
    _, source_scores, source_pred = _rows(second_predictions, "identity", 0, 0, job["query_ids"], classes)
    _, third_source_scores, _ = _rows(third_predictions, "identity", 0, 0, job["query_ids"], classes)
    if not np.array_equal(source_scores, third_source_scores):
        raise ValueError("Saved phases do not use exactly the same frozen source scores")
    _, native_scores, native_pred = _rows(second_predictions, "svc_predict", 0, 0,
                                          job["query_ids"], classes, score_rule=False)
    writer.record("identity", 0, source_scores, pred=source_pred, support_ids=[],
                  saved_prediction_reused=True, new_adapter_fit=False, new_model_fit=False)
    writer.record("svc_predict", 0, native_scores, pred=native_pred, support_ids=[],
                  decision_rule="SVC.predict", saved_prediction_reused=True,
                  new_adapter_fit=False, new_model_fit=False)

    certificate_rows, mechanism_rows, reused = [], [], []
    cases = [(2, "f01_weak", F01_METHOD, k, draw) for k, draw in f01_episodes()]
    cases += [(3, "g03_weak", G03_METHOD, k, draw) for k, draw in g03_episodes()]
    for historical_round, method, historical_method, k, draw in cases:
        frame, path, endpoints = ((second_predictions, second_path, second_endpoints)
                                  if historical_round == 2 else
                                  (third_predictions, third_path, third_endpoints))
        saved_id = endpoint_id(historical_method, k, draw)
        if saved_id not in endpoints:
            raise ValueError("Missing frozen weak endpoint metadata")
        endpoint = endpoints[saved_id]
        block, weak_scores, weak_pred = _rows(frame, historical_method, k, draw, job["query_ids"], classes)
        diagnostic = crossing_diagnostics(source_scores, weak_scores, ACTUAL_BLEND)
        if historical_round == 2:
            parameter_record = second_parameters[saved_id]
            adapter = parameter_record["adapter"]
            if parameter_record["support_ids"] != endpoint["support_ids"]:
                raise ValueError("Saved weak adapter has different Support IDs")
            noise_rate, q, noise_seed = 0.0, 0, 0
        else:
            parameter_record = third_parameters.get((historical_method, draw))
            if parameter_record is None:
                parameter_record = third_objective_parameters[(historical_method, endpoint["objective_key"])]
            adapter = parameter_record["adapter"]
            noise_rate = float(endpoint["actual_noise_rate"])
            q, noise_seed = int(endpoint["noise_q_per_class"]), int(endpoint["noise_seed"])
        parameters = _parameter_diagnostics(adapter, len(classes))
        reproduced = source_scores @ np.asarray(adapter["coef"]) + np.asarray(adapter["intercept"])
        if not np.allclose(reproduced, weak_scores, rtol=0, atol=1e-12):
            raise ValueError("Saved weak parameters no longer reproduce the sealed scores")
        # Query labels join only after all crossing and margin calculations.
        truth = writer.labels(job["query_ids"])
        if not np.array_equal(block.true_label.to_numpy(), truth):
            raise ValueError("Saved Query truth differs from current frozen metadata")
        extras = {
            "case_type": "true_support_reordering" if historical_round == 2 else "artificial_support_label_noise",
            "source_evidence_round": historical_round,
            "actual_noise_rate": noise_rate, "noise_q_per_class": q, "noise_seed": noise_seed,
            "actual_blend": ACTUAL_BLEND, "support_labels_used": len(endpoint["support_ids"]),
            "saved_prediction_reused": True, "new_adapter_fit": False, "new_model_fit": False,
            **parameters,
        }
        identifier = writer.record(method, k, weak_scores, pred=weak_pred, draw=draw,
                                   support_ids=endpoint["support_ids"], **extras)
        context = {"job_id": job["job_id"], "family": job["family"], "experiment": job["experiment"],
                   "parent_job_id": parent, "endpoint_id": identifier, "method": method, "k": k, "draw": draw,
                   **extras}
        for index, record_id in enumerate(job["query_ids"]):
            certificate_rows.append({**context, "record_id": record_id,
                                     "physical_sample_id": writer.indexed.at[record_id, "physical_sample_id"],
                                     "origin_group_id": writer.indexed.at[record_id, "origin_group_id"],
                                     "true_label": truth[index], "source_pred_label": source_pred[index],
                                     "weak_pred_label": weak_pred[index],
                                     "source_correct": bool(source_pred[index] == truth[index]),
                                     "weak_correct": bool(weak_pred[index] == truth[index]),
                                     **{name: values[index] for name, values in diagnostic.items()}})
        finite_crossings = diagnostic["first_crossing_blend"][diagnostic["first_crossing_finite"]]
        mechanism_rows.append({
            **context, "n_query": len(truth), "score_changed_count": int(diagnostic["score_changed"].sum()),
            "decision_changed_count": int(diagnostic["decision_changed"].sum()),
            "score_moved_decision_unchanged_count": int(diagnostic["score_moved_decision_unchanged"].sum()),
            "certified_count": int(diagnostic["strict_margin_certificate"].sum()),
            "certificate_violation_count": int(diagnostic["certificate_violation"].sum()),
            "crossing_le_actual_count": int(diagnostic["first_crossing_le_actual_blend"].sum()),
            "crossing_le_one_count": int(diagnostic["first_crossing_le_one"].sum()),
            "finite_crossing_count": int(diagnostic["first_crossing_finite"].sum()),
            "path_inconsistency_count": int((~diagnostic["path_prediction_consistent"]).sum()),
            "crossing_implication_inconsistency_count": int((~diagnostic["crossing_implication_consistent"]).sum()),
            "min_finite_crossing_blend": float(finite_crossings.min()) if len(finite_crossings) else None,
            "maximum_actual_correction_linf": float(diagnostic["actual_correction_linf"].max()),
            "minimum_source_top2_gap": float(diagnostic["source_top2_gap"].min()),
        })
        reused.append({"endpoint_id": identifier, "source_round": historical_round,
                       "source_job_id": path.name, "source_endpoint_id": saved_id,
                       "source_result_sha256": file_hash(path / "result.json"),
                       "source_predictions_sha256": file_hash(path / "predictions.csv.gz"),
                       "source_adapters_sha256": file_hash(path / "adapter_parameters.json")})

    certificates = pd.DataFrame(certificate_rows)
    mechanisms = pd.DataFrame(mechanism_rows)
    certificates.to_csv(writer.temp / "weak_certificates.csv", index=False)
    mechanisms.to_csv(writer.temp / "weak_mechanism.csv", index=False)
    totals = {
        "weak_endpoints": len(mechanism_rows), "query_rows": len(certificate_rows),
        "new_classifier_fits": 0, "new_adapter_fits": 0, "actual_blend": ACTUAL_BLEND,
        "score_changed_rows": int(certificates.score_changed.sum()),
        "decision_changed_rows": int(certificates.decision_changed.sum()),
        "strictly_certified_rows": int(certificates.strict_margin_certificate.sum()),
        "certificate_violations": int(certificates.certificate_violation.sum()),
        "path_prediction_inconsistencies": int((~certificates.path_prediction_consistent).sum()),
        "interpretation": "Finite saved-score margin and affine-path audit, not a guarantee for new samples or evidence of useful adaptation.",
        "first_crossing_scope": "Only the first boundary is derived; t>1 extrapolates beyond the full lambda1000 correction and is not a proposed method.",
        "selection_policy": "No Query-label-based blend choice or parameter search; labels join after label-free diagnostics.",
        "K10_zero_noise_deduplication": "G03 zero-noise weak endpoint excluded because it repeats the F01 K10 canonical endpoint.",
        "tie_rule": "score argmax chooses first class; a higher-index rival switches just after its first tie boundary.",
    }
    writer.save_json("weak_mechanism.json", {"summary": totals, "endpoints": mechanism_rows})
    writer.save_json("reused_prediction_provenance.json", reused)
    return writer.finish(expected_endpoints(job), new_model_fits=0, new_adapter_fits=0,
                         saved_weak_endpoints_reused=50, saved_source_controls_reused=2,
                         weak_audit=totals)
