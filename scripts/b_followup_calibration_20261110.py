"""F01: fixed-source score geometry and support-order calibration controls.

The source spectral pipelines are reused from the sealed first B study.  This
module never trains a spectral classifier.  Positive global score units and
per-row centering preserve the K=0 argmax boundary; all fitted normalization
uses source scores only.  Query labels enter only evaluation after prediction.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from b_mechanism_engine_20261110 import calibrate_scores, native_predict
from b_followup_common_20261110 import TaskWriter, endpoint_id, load_parent


DRAW_VALUES = (0, *range(101, 111))
K_VALUES = (1, 3, 5, 10)
TRUE_METHODS = (
    "raw_anchor_lambda1", "raw_bias_only", "raw_weak_lambda1000_blend005",
    "diagonal_anchor_lambda1", "scale01_anchor", "scale10_anchor",
    "rowcenter_source_rms_anchor",
)
SHUFFLE_METHOD = "shuffled_anchor_lambda1"
REUSABLE_METHODS = {
    "raw_anchor_lambda1": "anchor_lambda1",
    "raw_bias_only": "bias_only",
    "raw_weak_lambda1000_blend005": "weak_lambda1000_blend005",
}


def expected_endpoints(job=None):
    """Complete preregistered endpoint list, including the single K10 pool."""
    result = [endpoint_id("identity", 0, 0), endpoint_id("svc_predict", 0, 0)]
    for draw in DRAW_VALUES:
        for k in K_VALUES:
            methods = TRUE_METHODS + (SHUFFLE_METHOD,) if k < 10 or draw == 0 else (SHUFFLE_METHOD,)
            result.extend(endpoint_id(method, k, draw) for method in methods)
    if len(result) != 284 or len(set(result)) != len(result):
        raise AssertionError("F01 endpoint plan changed")
    return result


def support_plan(job, metadata, draw, k):
    """Nested IDs from each fixed ten-member class pool; no spectra are read."""
    if draw not in DRAW_VALUES or k not in (0, *K_VALUES):
        raise ValueError("Support draw/K is outside the frozen F01 plan")
    frame = metadata.set_index("record_id", drop=False)
    result = []
    for class_index, label in enumerate(job["classes"]):
        members = [sample for sample in job["support_ids"]
                   if frame.at[sample, job["label_column"]] == label]
        if len(members) != 10 or len(set(members)) != 10:
            raise ValueError("F01 needs ten unique support candidates per class")
        if draw:
            rng = np.random.default_rng(np.random.SeedSequence([draw, class_index, 20261110]))
            members = np.asarray(members)[rng.permutation(10)].tolist()
        result.extend(members[:k])
    return result


def diagnostic_labels(labels, draw, k):
    """Shuffle an explicit copy, preserving the class histogram and input array."""
    values = np.asarray(labels).copy()
    rng = np.random.default_rng(np.random.SeedSequence([draw, k, 20261110, 913]))
    result = values[rng.permutation(len(values))]
    if sorted(result.tolist()) != sorted(values.tolist()):
        raise AssertionError("Shuffled support histogram changed")
    return result


def rowcenter_source_rms(source_scores):
    """Return a fixed row-centering/global-source-RMS linear score transform."""
    source_scores = np.asarray(source_scores, dtype=float)
    if source_scores.ndim != 2 or source_scores.shape[1] < 2 or not np.isfinite(source_scores).all():
        raise ValueError("Source scores must be a finite N x C matrix")
    count = source_scores.shape[1]
    centering = np.eye(count) - np.ones((count, count)) / count
    centered = source_scores @ centering
    rms = float(np.sqrt(np.mean(centered ** 2)))
    effective_rms = rms if rms > 0 else 1.0
    return centering / effective_rms, rms, effective_rms


def fit_diagonal_anchor(support_scores, labels, classes, penalty=1.0):
    """Per-class anchored slope/intercept ridge with no cross-class weights."""
    support_scores = np.asarray(support_scores, dtype=float)
    labels = np.asarray(labels)
    classes = list(classes)
    if support_scores.shape != (len(labels), len(classes)) or not len(labels):
        raise ValueError("Diagonal calibration support dimensions disagree")
    if set(labels.tolist()) != set(classes):
        raise ValueError("All fixed source classes must occur in support")
    one_hot = np.asarray([[float(label == cls) for cls in classes] for label in labels])
    coefficient = np.eye(len(classes))
    intercept = np.zeros(len(classes))
    for j in range(len(classes)):
        design = np.column_stack([support_scores[:, j], np.ones(len(labels))])
        delta = np.linalg.solve(design.T @ design + penalty * np.eye(2),
                                design.T @ (one_hot[:, j] - support_scores[:, j]))
        coefficient[j, j] += delta[0]
        intercept[j] = delta[1]
    return {"method": "diagonal_anchor_lambda1", "coef": coefficient.tolist(),
            "intercept": intercept.tolist(), "regularization": float(penalty),
            "identity": False, "formula": "separate ridge residual on [own_class_score,1] per class"}


def fit_score_method(method, support_scores, labels, query_scores, classes, transform=None):
    """Fit using support labels only; this signature cannot receive Query labels."""
    support_scores, query_scores = np.asarray(support_scores), np.asarray(query_scores)
    if method == "diagonal_anchor_lambda1":
        adapter = fit_diagonal_anchor(support_scores, labels, classes)
        return query_scores @ np.asarray(adapter["coef"]) + adapter["intercept"], adapter
    if method in REUSABLE_METHODS:
        return calibrate_scores(REUSABLE_METHODS[method], support_scores, labels, query_scores, classes)
    if method == SHUFFLE_METHOD:
        return calibrate_scores("anchor_lambda1", support_scores, labels, query_scores, classes)
    if method not in {"scale01_anchor", "scale10_anchor", "rowcenter_source_rms_anchor"}:
        raise ValueError("Unknown F01 score method")
    if transform is None:
        raise ValueError("A frozen score transform is required")
    scores, adapter = calibrate_scores("anchor_lambda1", support_scores @ transform,
                                       labels, query_scores @ transform, classes)
    adapter["score_transform"] = np.asarray(transform).tolist()
    return scores, adapter


def _geometry(support_scores, adapter, method, *, input_transform, source_rms,
              reused, query_scores, calibrated_scores, baseline_boundary):
    count = support_scores.shape[1]
    transform = np.asarray(input_transform)
    fitted_scores = support_scores @ transform
    penalty = float(adapter.get("regularization", 1.0))
    design = np.column_stack([fitted_scores, np.ones(len(fitted_scores))])
    coefficient = np.asarray(adapter["coef"])
    intercept = np.asarray(adapter["intercept"])
    delta = coefficient - np.eye(count)
    off_diagonal = delta.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    full_parameters = count * (count + 1)
    if method == "diagonal_anchor_lambda1":
        model_parameters = 2 * count
        regularized_condition = max(float(np.linalg.cond(
            np.column_stack([fitted_scores[:, j], np.ones(len(fitted_scores))]).T @
            np.column_stack([fitted_scores[:, j], np.ones(len(fitted_scores))]) + penalty * np.eye(2)))
            for j in range(count))
    elif method == "raw_bias_only":
        model_parameters, regularized_condition = count, 1.0
    else:
        model_parameters = full_parameters
        regularized_condition = float(np.linalg.cond(design.T @ design + penalty * np.eye(count + 1)))
    effective_coefficient = transform @ coefficient
    result = {
        "support_count": len(support_scores), "design_rank": int(np.linalg.matrix_rank(design)),
        "design_columns": design.shape[1], "regularized_condition": regularized_condition,
        "fitted_parameter_count": model_parameters,
        "full_anchor_parameter_count": full_parameters,
        "correction_frobenius": float(np.linalg.norm(delta)),
        "off_diagonal_frobenius": float(np.linalg.norm(off_diagonal)),
        "diagonal_delta_frobenius": float(np.linalg.norm(np.diag(delta))),
        "intercept_l2": float(np.linalg.norm(intercept)),
        "source_rms": source_rms,
        "input_scale": 0.1 if method == "scale01_anchor" else 10.0 if method == "scale10_anchor" else 1.0,
        "K0_boundary_preserved": bool(baseline_boundary),
        "adapter_reused": bool(reused), "new_adapter_fit": not bool(reused),
        "a": None, "b": None, "affine_max_error": None,
        "a_positive": None, "a_nonpositive_separate": None,
        "rank_unchanged": None, "threshold": None, "threshold_equivalence": None,
    }
    if count == 2:
        source_margin = (query_scores[:, 1] - query_scores[:, 0]) / 2
        new_margin = (calibrated_scores[:, 1] - calibrated_scores[:, 0]) / 2
        a = float((-effective_coefficient[0, 1] + effective_coefficient[1, 1]
                   + effective_coefficient[0, 0] - effective_coefficient[1, 0]) / 2)
        b = float((intercept[1] - intercept[0]) / 2)
        threshold = -b / a if a > 0 else None
        # Comparisons use argmax's first-class tie rule, matching strict >0.
        equivalent = (bool(np.array_equal(source_margin > threshold, new_margin > 0))
                      if threshold is not None else None)
        result.update(a=a, b=b, affine_max_error=float(np.max(np.abs(new_margin - (a * source_margin + b)))),
                      a_positive=a > 0, a_nonpositive_separate=a <= 0,
                      rank_unchanged=bool(np.array_equal(np.argsort(source_margin, kind="stable"),
                                                         np.argsort(new_margin, kind="stable"))),
                      threshold=threshold, threshold_equivalence=equivalent)
    return result


def _old_endpoint(frame, old_method, k, query_ids, classes):
    block = frame.loc[frame.method.eq(old_method) & frame.k.eq(k)].set_index("record_id").loc[query_ids]
    scores = np.asarray([json.loads(text) for text in block.scores_json], dtype=float)
    prediction = block.pred_label.to_numpy()
    if not np.array_equal(np.asarray(classes)[np.argmax(scores, axis=1)], prediction):
        raise ValueError("Reused score endpoint prediction rule differs")
    return scores, prediction


def execute(job, X, metadata, axis, output, input_hash, oldroot):
    """Run one frozen source-model F01 bundle, with 284 endpoint evaluations."""
    writer = TaskWriter(job, X, metadata, axis, output, input_hash, oldroot)
    old_result, source, parent_dir = load_parent(job, oldroot)
    classes = writer.classes.tolist()
    source_before = joblib.hash(source)
    source_scores = source.predict_scores(writer.take(job["source_ids"]))
    query_scores = source.predict_scores(writer.query_X)
    support_scores_all = source.predict_scores(writer.take(job["support_ids"]))
    score_lookup = dict(zip(job["support_ids"], support_scores_all))
    transformation, rms, effective_rms = rowcenter_source_rms(source_scores)
    identity_transform = np.eye(len(classes))
    transforms = {"scale01_anchor": 0.1 * identity_transform,
                  "scale10_anchor": 10.0 * identity_transform,
                  "rowcenter_source_rms_anchor": transformation}
    raw_argmax = np.argmax(query_scores, axis=1)
    for transform in transforms.values():
        if not np.array_equal(np.argmax(query_scores @ transform, axis=1), raw_argmax):
            raise ValueError("Prespecified score transform changed the K0 decision boundary")
        if not np.array_equal(np.argmax(source_scores @ transform, axis=1), np.argmax(source_scores, axis=1)):
            raise ValueError("Prespecified score transform changed source K0 decisions")
    old_predictions = pd.read_csv(parent_dir / "predictions.csv", keep_default_na=False)
    old_adapters = json.loads((parent_dir / "calibration_parameters.json").read_text())
    reusable_adapters = {(item["method"], int(item["k"])): item for item in old_adapters}
    writer.record("identity", 0, query_scores, draw=0, support_ids=[],
                  adapter_reused=True, new_adapter_fit=False, new_model_fit=False,
                  support_labels_used=0, allocated_support_labels=0,
                  source_parent_job=job["parent_job_id"])
    writer.record("svc_predict", 0, query_scores, pred=native_predict(source, writer.query_X),
                  draw=0, support_ids=[], decision_rule="SVC.predict", adapter_reused=True,
                  new_adapter_fit=False, new_model_fit=False, support_labels_used=0,
                  allocated_support_labels=0, source_parent_job=job["parent_job_id"])
    all_parameters, plans, geometry_rows = [], {}, []
    new_adapter_fits, reused_adapters = 0, 0
    for draw in DRAW_VALUES:
        for k in K_VALUES:
            ids = support_plan(job, metadata, draw, k)
            support_scores = np.asarray([score_lookup[sample] for sample in ids])
            support_y = writer.labels(ids)
            plans[f"draw{draw}_k{k}"] = ids
            methods = TRUE_METHODS + (SHUFFLE_METHOD,) if k < 10 or draw == 0 else (SHUFFLE_METHOD,)
            for method in methods:
                transformed = transforms.get(method, identity_transform)
                fitted_y = diagnostic_labels(support_y, draw, k) if method == SHUFFLE_METHOD else support_y.copy()
                permuted_labels = int(np.count_nonzero(fitted_y != support_y))
                reused = draw == 0 and method in REUSABLE_METHODS
                start_cpu, start_wall = time.process_time(), time.monotonic()
                if reused:
                    old_method = REUSABLE_METHODS[method]
                    adapter = dict(reusable_adapters[(old_method, k)])
                    scores, pred = _old_endpoint(old_predictions, old_method, k, job["query_ids"], classes)
                    expected_ids = json.loads(old_predictions.loc[
                        old_predictions.method.eq(old_method) & old_predictions.k.eq(k), "support_ids_json"].iloc[0])
                    if expected_ids != ids:
                        raise ValueError("Canonical support IDs do not match the reusable historical endpoint")
                    if not np.allclose(scores, query_scores @ np.asarray(adapter["coef"]) + adapter["intercept"],
                                       rtol=0, atol=1e-12):
                        raise ValueError("Reused adapter no longer reproduces the sealed historical scores")
                    reused_adapters += 1
                else:
                    scores, adapter = fit_score_method(method, support_scores, fitted_y, query_scores,
                                                        classes, transform=transformed)
                    pred = np.asarray(classes)[np.argmax(scores, axis=1)]
                    new_adapter_fits += 1
                adapter_cpu = time.process_time() - start_cpu
                adapter_wall = time.monotonic() - start_wall
                id_ = writer.record(method, k, scores, pred=pred, draw=draw, support_ids=ids,
                                    adapter_reused=reused, new_adapter_fit=not reused, new_model_fit=False,
                                    support_labels_used=len(ids), allocated_support_labels=len(ids),
                                    diagnostic_label_copy=method == SHUFFLE_METHOD,
                                    support_labels_changed=permuted_labels,
                                    adapter_cpu_seconds=adapter_cpu, adapter_wall_seconds=adapter_wall,
                                    source_parent_job=job["parent_job_id"])
                geometry = _geometry(support_scores, adapter, method, input_transform=transformed,
                                     source_rms=rms, reused=reused, query_scores=query_scores,
                                     calibrated_scores=scores, baseline_boundary=True)
                geometry.update(endpoint_id=id_, family="F01", job_id=job["job_id"],
                                parent_job_id=job["parent_job_id"], experiment=job["experiment"],
                                method=method, k=k, draw=draw,
                                diagnostic_label_copy=method == SHUFFLE_METHOD,
                                support_labels_changed=permuted_labels,
                                effective_source_rms=effective_rms)
                if len(classes) == 2:
                    # Evaluation only: no fit above receives these Query labels.
                    true = writer.labels(job["query_ids"])
                    geometry["auc_before"] = float(roc_auc_score(true == classes[1],
                        (query_scores[:, 1] - query_scores[:, 0]) / 2))
                    geometry["auc_after"] = float(roc_auc_score(true == classes[1], (scores[:, 1] - scores[:, 0]) / 2))
                geometry_rows.append(geometry)
                all_parameters.append({"endpoint_id": id_, "method": method, "k": k, "draw": draw,
                                       "support_ids": ids, "fit_labels_diagnostic_copy": fitted_y.tolist(),
                                       "diagnostic_label_copy": method == SHUFFLE_METHOD,
                                       "support_labels_changed": permuted_labels,
                                       "adapter_reused": reused, "adapter": adapter,
                                       "input_transform": transformed.tolist(),
                                       "parent_adapter_path": str(parent_dir / "calibration_parameters.json") if reused else None})
    if joblib.hash(source) != source_before:
        raise ValueError("Calibration mutated the frozen spectral source model")
    if (new_adapter_fits, reused_adapters) != (270, 12):
        raise AssertionError("Frozen F01 adapter accounting changed")
    pd.DataFrame(geometry_rows).to_csv(writer.temp / "geometry.csv", index=False)
    writer.save_json("adapter_parameters.json", all_parameters)
    writer.save_json("support_plan.json", {"draws": list(DRAW_VALUES), "k_values": list(K_VALUES),
                                          "plans": plans, "k10_true_label_policy": "single full-pool endpoint at canonical draw0"})
    writer.save_json("source_reuse.json", {"parent_job_id": job["parent_job_id"],
                                         "parent_input_hash": old_result["input_hash"],
                                         "source_hash": old_result["source_hash"],
                                         "source_model_unchanged": True,
                                         "source_rms_only_from_source_samples": len(job["source_ids"]),
                                         "source_rms": rms, "effective_source_rms": effective_rms})
    return writer.finish(expected_endpoints=expected_endpoints(job), new_model_fits=0,
                         new_adapter_fits=new_adapter_fits, reused_adapters=reused_adapters,
                         reused_source_models=1, source_model_unchanged=True,
                         source_parent_job=job["parent_job_id"], source_hash=old_result["source_hash"])
