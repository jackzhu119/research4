"""G01: loss normalization and intercept-prior calibration mechanisms.

This module reuses frozen spectral sources and the exact prior sum-loss,
penalized-intercept Anchor endpoints.  The three new cells are a fixed 2x2
mechanism experiment, not a query-selected regularization search.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from b_followup_common_20261110 import TaskWriter, endpoint_id, file_hash, load_parent
from b_followup_calibration_20261110 import support_plan
from b_mechanism_engine_20261110 import native_predict


DRAW_VALUES = (0, 101, 102)
K_VALUES = (1, 3, 5, 10)
METHODS = (
    ("sum_penalized", "sum", True),
    ("mean_penalized", "mean", True),
    ("sum_free_intercept", "sum", False),
    ("mean_free_intercept", "mean", False),
)


def episodes():
    """Ten distinct declared support episodes; the complete K10 pool occurs once."""
    return [(draw, k) for draw in DRAW_VALUES for k in K_VALUES if draw == 0 or k < 10]


def expected_endpoints(job=None):
    result = [endpoint_id("identity"), endpoint_id("svc_predict")]
    result.extend(endpoint_id(method, k, draw)
                  for draw, k in episodes() for method, _, _ in METHODS)
    if len(result) != 42 or len(set(result)) != 42:
        raise AssertionError("Frozen G01 endpoint contract changed")
    return result


def fit_count(job=None):
    return 0


def adapter_count(job=None):
    return 30


def fit_adapter(support_scores, labels, classes, *, loss="sum", penalize_intercept=True):
    """Fit an identity-anchored residual ridge using labelled Support only.

    ``loss='mean'`` with a unit prior is implemented as sum loss with lambda=N.
    The penalty acts on the *correction* from identity.  A free intercept is
    represented by a zero last penalty diagonal, not an unpenalized classifier.
    No query arrays or query labels are accepted by this fitting interface.
    """
    scores = np.asarray(support_scores, dtype=float)
    labels = np.asarray(labels)
    classes = list(classes)
    if (scores.ndim != 2 or scores.shape != (len(labels), len(classes))
            or not len(labels) or not np.isfinite(scores).all()):
        raise ValueError("Support scores and labels must be finite and aligned")
    if len(classes) < 2 or set(labels.tolist()) != set(classes):
        raise ValueError("Each fixed source class must occur in Support")
    if loss not in {"sum", "mean"} or not isinstance(penalize_intercept, (bool, np.bool_)):
        raise ValueError("Unknown frozen loss/intercept condition")
    effective_lambda = 1.0 if loss == "sum" else float(len(labels))
    one_hot = np.asarray([[float(label == cls) for cls in classes] for label in labels])
    design = np.column_stack([scores, np.ones(len(scores))])
    penalty = np.eye(len(classes) + 1) * effective_lambda
    if not penalize_intercept:
        penalty[-1, -1] = 0.0
    gram = design.T @ design
    delta = np.linalg.solve(gram + penalty, design.T @ (one_hot - scores))
    coefficient = np.eye(len(classes)) + delta[:-1]
    intercept = delta[-1]
    if not np.isfinite(coefficient).all() or not np.isfinite(intercept).all():
        raise ValueError("Nonfinite calibrated parameters")
    return {
        "coef": coefficient.tolist(), "intercept": intercept.tolist(),
        "identity": False, "loss": loss, "prior_strength": 1.0,
        "effective_lambda": effective_lambda,
        "regularization": effective_lambda,
        "penalize_intercept": bool(penalize_intercept),
        "penalty_diagonal": np.diag(penalty).tolist(),
        "support_count": len(labels),
        "objective": ("sum(||onehot - scores - Z*delta||^2) + delta^T P delta"),
        "mean_loss_equivalence": "unit mean-loss prior equals sum-loss lambda=N",
        "intercept_prior": "zero correction" if penalize_intercept else "unpenalized correction",
    }


def adapter_geometry(support_scores, adapter, *, loss, penalize_intercept,
                     source_query_scores=None, calibrated_query_scores=None):
    """Describe the fixed estimator without using Query values to fit it."""
    scores = np.asarray(support_scores, dtype=float)
    n, classes = scores.shape
    design = np.column_stack([scores, np.ones(n)])
    effective_lambda = 1.0 if loss == "sum" else float(n)
    penalty = np.eye(classes + 1) * effective_lambda
    if not penalize_intercept:
        penalty[-1, -1] = 0.0
    gram = design.T @ design
    system = gram + penalty
    coefficient = np.asarray(adapter["coef"], dtype=float)
    intercept = np.asarray(adapter["intercept"], dtype=float)
    correction = coefficient - np.eye(classes)
    off_diagonal = correction.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    effective_df = float(np.trace(np.linalg.solve(system, gram)))
    result = {
        "loss": loss, "intercept_penalized": bool(penalize_intercept),
        "support_count": n, "effective_lambda": effective_lambda,
        "intercept_penalty": effective_lambda if penalize_intercept else 0.0,
        "design_rank": int(np.linalg.matrix_rank(design)),
        "design_columns": classes + 1,
        "regularized_condition": float(np.linalg.cond(system)),
        "effective_df_per_output": effective_df,
        "effective_df_all_outputs": classes * effective_df,
        "nominal_parameters": classes * (classes + 1),
        "correction_frobenius": float(np.linalg.norm(correction)),
        "off_diagonal_frobenius": float(np.linalg.norm(off_diagonal)),
        "intercept_l2": float(np.linalg.norm(intercept)),
        "a": None, "b": None, "affine_max_error": None,
        "a_positive": None, "a_nonpositive_separate": None,
        "rank_unchanged": None, "threshold": None, "threshold_equivalence": None,
    }
    if classes == 2 and source_query_scores is not None and calibrated_query_scores is not None:
        source_query_scores = np.asarray(source_query_scores)
        calibrated_query_scores = np.asarray(calibrated_query_scores)
        old_margin = (source_query_scores[:, 1] - source_query_scores[:, 0]) / 2
        new_margin = (calibrated_query_scores[:, 1] - calibrated_query_scores[:, 0]) / 2
        a = float((coefficient[0, 0] - coefficient[1, 0]
                   - coefficient[0, 1] + coefficient[1, 1]) / 2)
        b = float((intercept[1] - intercept[0]) / 2)
        threshold = -b / a if a > 0 else None
        result.update({
            "a": a, "b": b,
            "affine_max_error": float(np.max(np.abs(new_margin - (a * old_margin + b)))),
            "a_positive": a > 0, "a_nonpositive_separate": a <= 0,
            "rank_unchanged": bool(np.array_equal(np.argsort(old_margin, kind="stable"),
                                                  np.argsort(new_margin, kind="stable"))),
            "threshold": threshold,
            "threshold_equivalence": (bool(np.array_equal(old_margin > threshold, new_margin > 0))
                                      if threshold is not None else None),
        })
    return result


def _historical_endpoint(predictions, parameters, k, draw, query_ids):
    identifier = endpoint_id("raw_anchor_lambda1", k, draw)
    subset = predictions.loc[predictions.endpoint_id.eq(identifier)]
    if len(subset) != len(query_ids) or not subset.record_id.is_unique:
        raise ValueError("Missing or duplicated reusable raw Anchor endpoint")
    subset = subset.set_index("record_id").loc[query_ids]
    parameter = parameters.get(identifier)
    if parameter is None:
        raise ValueError("Reusable raw Anchor parameters are missing")
    return (np.asarray([json.loads(value) for value in subset.scores_json]),
            subset.pred_label.to_numpy(), parameter)


def execute(job, X, metadata, axis, output, input_hash, oldroot, round2root):
    """Evaluate one G01 context, with exactly thirty new ridge calibrations."""
    from b_round3_common_20261110 import load_round2_bundle

    writer = TaskWriter(job, X, metadata, axis, output, input_hash, oldroot)
    old_result, source, source_path = load_parent(job, oldroot)
    round2_job_id = job.get("phase2_job_id", "F01_" + job["parent_job_id"])
    round2_result, round2_path = load_round2_bundle(round2root, round2_job_id)
    for key in ("source_ids", "support_ids", "query_ids", "classes", "label_column"):
        if round2_result[key] != job[key]:
            raise ValueError("G01 roles differ from the frozen reusable second-round endpoint")
    parent_predictions = pd.read_csv(round2_path / "predictions.csv.gz", keep_default_na=False)
    parent_parameters = {item["endpoint_id"]: item for item in
                         json.loads((round2_path / "adapter_parameters.json").read_text())}
    source_before = joblib.hash(source)
    query_scores = source.predict_scores(writer.query_X)
    pool_scores = source.predict_scores(writer.take(job["support_ids"]))
    lookup = dict(zip(job["support_ids"], pool_scores))
    classes = writer.classes.tolist()
    writer.record("identity", 0, query_scores, support_ids=[], adapter_reused=True,
                  support_labels_used=0, new_adapter_fit=False, source_parent_job=job["parent_job_id"])
    writer.record("svc_predict", 0, query_scores, pred=native_predict(source, writer.query_X),
                  support_ids=[], decision_rule="SVC.predict", adapter_reused=True,
                  support_labels_used=0, new_adapter_fit=False, source_parent_job=job["parent_job_id"])
    saved_parameters, geometry_rows, support_locks = [], [], {}
    new_adapters, reused_adapters = 0, 0
    for draw, k in episodes():
        support_ids = support_plan(job, metadata, draw, k)
        support_scores = np.asarray([lookup[sample] for sample in support_ids])
        support_y = writer.labels(support_ids)
        support_locks[f"draw{draw}_k{k}"] = support_ids
        for method, loss, penalize_intercept in METHODS:
            start_cpu, start_wall = time.process_time(), time.monotonic()
            reused = method == "sum_penalized"
            if reused:
                scores, prediction, parameter = _historical_endpoint(
                    parent_predictions, parent_parameters, k, draw, job["query_ids"])
                if parameter["support_ids"] != support_ids:
                    raise ValueError("Reused calibration Support IDs changed")
                if parameter.get("diagnostic_label_copy", False):
                    raise ValueError("A copied-label diagnostic cannot be reused as genuine Support")
                if parameter["fit_labels_diagnostic_copy"] != support_y.tolist():
                    raise ValueError("Reused genuine Support label sequence changed")
                adapter = dict(parameter["adapter"])
                calculated = query_scores @ np.asarray(adapter["coef"]) + adapter["intercept"]
                if not np.allclose(scores, calculated, atol=1e-12, rtol=0):
                    raise ValueError("Historical adapter no longer reproduces its sealed scores")
                if not np.array_equal(np.asarray(classes)[scores.argmax(axis=1)], prediction):
                    raise ValueError("Historical raw Anchor uses a different prediction rule")
                reused_adapters += 1
            else:
                adapter = fit_adapter(support_scores, support_y, classes,
                                      loss=loss, penalize_intercept=penalize_intercept)
                scores = query_scores @ np.asarray(adapter["coef"]) + adapter["intercept"]
                prediction = np.asarray(classes)[scores.argmax(axis=1)]
                new_adapters += 1
            geometry = adapter_geometry(support_scores, adapter, loss=loss,
                                        penalize_intercept=penalize_intercept,
                                        source_query_scores=query_scores,
                                        calibrated_query_scores=scores)
            identifier = writer.record(
                method, k, scores, pred=prediction, draw=draw, support_ids=support_ids,
                loss=loss, intercept_penalized=penalize_intercept,
                effective_lambda=geometry["effective_lambda"],
                support_labels_used=len(support_ids), adapter_reused=reused,
                new_adapter_fit=not reused,
                adapter_cpu_seconds=time.process_time() - start_cpu,
                adapter_wall_seconds=time.monotonic() - start_wall,
                effective_df_per_output=geometry["effective_df_per_output"],
                source_parent_job=job["parent_job_id"])
            geometry.update(endpoint_id=identifier, job_id=job["job_id"], family="G01",
                            experiment=job["experiment"], method=method, k=k, draw=draw,
                            adapter_reused=reused, new_adapter_fit=not reused)
            if len(classes) == 2:
                # Labels enter evaluation only, after the fitted prediction above.
                query_y = writer.labels(job["query_ids"])
                geometry["auc_before"] = float(roc_auc_score(
                    query_y == classes[1], (query_scores[:, 1] - query_scores[:, 0]) / 2))
                geometry["auc_after"] = float(roc_auc_score(
                    query_y == classes[1], (scores[:, 1] - scores[:, 0]) / 2))
            geometry_rows.append(geometry)
            saved_parameters.append({
                "endpoint_id": identifier, "method": method, "k": k, "draw": draw,
                "support_ids": support_ids, "fit_labels": support_y.tolist(),
                "loss": loss, "intercept_penalized": penalize_intercept,
                "adapter_reused": reused, "adapter": adapter,
                "parent_endpoint_id": endpoint_id("raw_anchor_lambda1", k, draw) if reused else None,
            })
    if (new_adapters, reused_adapters) != (30, 10):
        raise AssertionError("Frozen G01 calibration accounting changed")
    if joblib.hash(source) != source_before:
        raise ValueError("G01 calibration changed the frozen spectral source")
    pd.DataFrame(geometry_rows).to_csv(writer.temp / "geometry.csv", index=False)
    writer.save_json("adapter_parameters.json", saved_parameters)
    writer.save_json("support_plan.json", {"episodes": episodes(), "plans": support_locks,
                                          "k10_policy": "single genuine-label full-pool endpoint"})
    writer.save_json("source_reuse.json", {
        "parent_job_id": job["parent_job_id"], "source_hash": old_result["source_hash"],
        "phase2_job_id": round2_job_id,
        "phase2_result_sha256": file_hash(round2_path / "result.json"),
        "source_model_unchanged": True, "support_only_fitted_adapters": True,
    })
    return writer.finish(expected_endpoints(job), new_model_fits=0,
                         new_adapter_fits=new_adapters, reused_adapters=reused_adapters,
                         reused_source_models=1, source_model_unchanged=True,
                         source_hash=old_result["source_hash"],
                         round2_result_sha256=file_hash(round2_path / "result.json"))
