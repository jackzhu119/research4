"""G04: zero-fit class-weight, margin and SVC box-constraint audit.

Every Query endpoint is copied from the sealed second-round F03 artifact.
Existing models are evaluated, never fitted.  The bound tolerance below is a
descriptive numerical diagnostic; it never selects a model or stops training.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel

from b_followup_common_20261110 import TaskWriter, endpoint_id, file_hash
from b_round3_common_20261110 import load_round2_bundle, load_weight_model


METHODS = (
    "class_weight_balanced", "class_weight_balanced_argmax",
    "class_weight_none", "class_weight_none_argmax",
)
BOUND_RTOL = 1e-7
BOUND_ATOL = 1e-9


def expected_endpoints(job=None):
    return [endpoint_id(method) for method in METHODS]


def fit_count(job=None):
    return 0


def adapter_count(job=None):
    return 0


def _rank_correlation(left, right, function):
    """Undefined rank correlation remains missing, never fabricated as one."""
    left, right = np.asarray(left), np.asarray(right)
    if np.unique(left).size < 2 or np.unique(right).size < 2:
        return None
    value = float(function(left, right).statistic)
    return value if np.isfinite(value) else None


def _binary_scores(model, X):
    scores = np.asarray(model.predict_scores(X), dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 2 or not np.isfinite(scores).all():
        raise ValueError("G04 requires finite aligned binary source scores")
    return scores, (scores[:, 1] - scores[:, 0]) / 2


def audit_model(model, X_source, y_source, source_ids, X_query, y_query, setting):
    """Evaluate a fixed SVC and its effective per-observation upper bounds."""
    if len(model.classes_) != 2:
        raise ValueError("G04 is the frozen binary class-weight audit")
    estimator = model.estimator_
    source_ids = list(source_ids)
    X_source, y_source = np.asarray(X_source), np.asarray(y_source)
    if len(X_source) != len(source_ids) or len(y_source) != len(source_ids):
        raise ValueError("Source IDs, labels and spectra do not align")
    source_scores, source_margin = _binary_scores(model, X_source)
    query_scores, query_margin = _binary_scores(model, X_query)
    classes = model.classes_.tolist()
    if set(y_source.tolist()) != set(classes) or not set(np.asarray(y_query).tolist()).issubset(classes):
        raise ValueError("SVC vocabulary differs from evaluation labels")
    native = estimator.predict(model._scaled(X_source))
    query_native = estimator.predict(model._scaled(X_query))
    positive = y_source == classes[1]
    signed_margin = np.where(positive, 1.0, -1.0) * source_margin
    hinge = np.maximum(0.0, 1.0 - signed_margin)
    weights = dict(zip(classes, np.asarray(estimator.class_weight_, dtype=float)))
    bounds = float(estimator.C) * np.asarray([weights[label] for label in y_source], dtype=float)
    if not np.isfinite(bounds).all() or not np.all(bounds > 0):
        raise ValueError("Invalid effective class-weight box bounds")
    support = np.asarray(estimator.support_, dtype=int)
    if len(set(support.tolist())) != len(support) or np.any(support < 0) or np.any(support >= len(source_ids)):
        raise ValueError("SVC support indices do not refer to the frozen Source")
    beta = np.zeros(len(source_ids), dtype=float)
    beta[support] = np.asarray(estimator.dual_coef_[0], dtype=float)
    alpha = np.abs(beta)
    gaps = bounds - alpha
    ratios = alpha / bounds
    close = np.isclose(alpha[support], bounds[support], rtol=BOUND_RTOL, atol=BOUND_ATOL)
    violations = alpha > bounds + BOUND_ATOL + BOUND_RTOL * bounds
    tolerance = BOUND_ATOL + BOUND_RTOL * bounds[support]
    # All arrays use the existing learned coordinate system and numeric gamma.
    K = rbf_kernel(estimator.support_vectors_, gamma=float(estimator._gamma))
    support_beta = np.asarray(estimator.dual_coef_[0], dtype=float)
    norm_squared = float(support_beta @ K @ support_beta)
    primal = float(0.5 * norm_squared + bounds @ hinge)
    dual = float(alpha.sum() - 0.5 * norm_squared)
    query_binary = np.asarray(y_query) == classes[1]
    auc = float(roc_auc_score(query_binary, query_margin)) if np.unique(query_binary).size == 2 else None
    summary = {
        "analysis": "model_box_constraints", "setting": setting,
        "source_samples": len(source_ids), "n_support_vectors": len(support),
        "max_alpha_over_C": float(ratios.max()),
        "min_support_bound_gap": float(gaps[support].min()) if len(support) else None,
        "near_upper_bound_sv_count": int(close.sum()),
        "box_violation_count": int(violations.sum()),
        "all_support_coefficients_strictly_below_bounds": bool(np.all(gaps[support] > 0)),
        "all_support_coefficients_below_bounds_with_numerical_margin": bool(np.all(gaps[support] > tolerance)),
        "bound_diagnostic_rtol": BOUND_RTOL, "bound_diagnostic_atol": BOUND_ATOL,
        "source_hinge_loss_sum": float(hinge.sum()),
        "source_weighted_hinge_loss_sum": float(bounds @ hinge),
        "source_training_error_count": int(np.count_nonzero(native != y_source)),
        "minimum_source_signed_margin": float(signed_margin.min()),
        "mean_source_signed_margin": float(signed_margin.mean()),
        "query_auc": auc, "effective_gamma": float(estimator._gamma),
        "C": float(estimator.C), "effective_class_weights_json": json.dumps(weights, ensure_ascii=False),
        "rkhs_norm_squared": norm_squared, "primal_objective": primal,
        "dual_objective": dual, "primal_dual_gap": primal - dual,
        "interpretation": "Numerical Source optimum diagnostics only; the tolerance does not select or stop an experiment.",
    }
    sample_rows = [{
        "setting": setting, "record_id": sample, "true_label": str(y_source[index]),
        "source_pred_label": str(native[index]), "source_margin": float(source_margin[index]),
        "signed_margin": float(signed_margin[index]), "hinge_loss": float(hinge[index]),
        "support_vector": bool(index in support), "signed_dual_coefficient": float(beta[index]),
        "alpha": float(alpha[index]), "C_i": float(bounds[index]),
        "alpha_over_C_i": float(ratios[index]), "bound_gap": float(gaps[index]),
        "near_upper_bound": bool(index in support and np.isclose(alpha[index], bounds[index], rtol=BOUND_RTOL, atol=BOUND_ATOL)),
    } for index, sample in enumerate(source_ids)]
    return {
        "summary": summary, "samples": sample_rows, "beta": beta,
        "support_ids": {source_ids[index] for index in support},
        "query_scores": query_scores, "query_margin": query_margin,
        "query_native": query_native,
    }


def compare_models(balanced, unweighted, balanced_audit, unweighted_audit, y_query):
    before, after = balanced_audit, unweighted_audit
    y_query = np.asarray(y_query)
    before_correct, after_correct = before["query_native"] == y_query, after["query_native"] == y_query
    margin_before, margin_after = before["query_margin"], after["query_margin"]
    source_coordinate_equal = (
        np.array_equal(balanced.scaler_.mean_, unweighted.scaler_.mean_)
        and np.array_equal(balanced.scaler_.scale_, unweighted.scaler_.scale_)
        and joblib.hash(balanced.preprocessor_) == joblib.hash(unweighted.preprocessor_)
    )
    beta_difference = np.abs(before["beta"] - after["beta"])
    max_score_difference = float(np.max(np.abs(before["query_scores"] - after["query_scores"])))
    same_beta = np.array_equal(before["beta"], after["beta"])
    same_intercept = np.array_equal(balanced.estimator_.intercept_, unweighted.estimator_.intercept_)
    support_union = before["support_ids"] | after["support_ids"]
    both_interior = bool(
        before["summary"]["all_support_coefficients_below_bounds_with_numerical_margin"]
        and after["summary"]["all_support_coefficients_below_bounds_with_numerical_margin"]
    )
    same_kernel = source_coordinate_equal and float(balanced.estimator_._gamma) == float(unweighted.estimator_._gamma)
    return {
        "analysis": "weight_pair", "setting": "none_minus_balanced",
        "query_labels_identical": bool(np.array_equal(before["query_native"], after["query_native"])),
        "query_scores_exact_equal": bool(np.array_equal(before["query_scores"], after["query_scores"])),
        "max_query_score_difference": max_score_difference,
        "max_query_margin_difference": float(np.max(np.abs(margin_after - margin_before))),
        "mean_absolute_query_margin_difference": float(np.mean(np.abs(margin_after - margin_before))),
        "query_margin_sign_flips": int(np.count_nonzero((margin_before > 0) != (margin_after > 0))),
        "query_margin_spearman": _rank_correlation(margin_before, margin_after, spearmanr),
        "query_margin_kendall": _rank_correlation(margin_before, margin_after, kendalltau),
        "query_rank_unchanged": bool(np.array_equal(np.argsort(margin_before, kind="stable"), np.argsort(margin_after, kind="stable"))),
        "query_auc_before": before["summary"]["query_auc"], "query_auc_after": after["summary"]["query_auc"],
        "scaler_equal": source_coordinate_equal,
        "effective_gamma_equal": float(balanced.estimator_._gamma) == float(unweighted.estimator_._gamma),
        "kernel_coordinates_equal": same_kernel,
        "dual_coefficients_exact_equal": bool(same_beta),
        "max_dual_coefficient_difference": float(beta_difference.max()),
        "intercept_exact_equal": bool(same_intercept),
        "max_intercept_difference": float(np.max(np.abs(balanced.estimator_.intercept_ - unweighted.estimator_.intercept_))),
        "support_set_equal": before["support_ids"] == after["support_ids"],
        "support_overlap_count": len(before["support_ids"] & after["support_ids"]),
        "support_union_count": len(support_union),
        "support_jaccard": len(before["support_ids"] & after["support_ids"]) / len(support_union) if support_union else None,
        "both_weighted_optima_interior": both_interior,
        "inactive_bounds_same_solution": bool(same_kernel and same_beta and same_intercept and both_interior),
        "wrong_to_correct": int(np.count_nonzero(~before_correct & after_correct)),
        "correct_to_wrong": int(np.count_nonzero(before_correct & ~after_correct)),
        "interpretation": "Equal coefficients with inactive box bounds can explain weight insensitivity in these frozen Source contexts; this is not a general ineffectiveness claim.",
    }


def _saved_endpoint(predictions, method, query_ids, classes):
    rows = predictions.loc[predictions.method.eq(method) & predictions.k.eq(0) & predictions.draw.eq(0)]
    if len(rows) != len(query_ids) or not rows.record_id.is_unique or set(rows.record_id) != set(query_ids):
        raise ValueError("Historical Query endpoint is incomplete or changed")
    rows = rows.set_index("record_id").loc[query_ids]
    scores = np.asarray([json.loads(value) for value in rows.scores_json], dtype=float)
    predicted = rows.pred_label.to_numpy()
    if scores.shape != (len(query_ids), len(classes)) or not np.isfinite(scores).all():
        raise ValueError("Historical Query scores are invalid")
    if not set(predicted).issubset(classes):
        raise ValueError("Historical prediction vocabulary changed")
    return rows, scores, predicted


def execute(job, X, metadata, axis, output, input_hash, oldroot, round2root):
    writer = TaskWriter(job, X, metadata, axis, output, input_hash, oldroot)
    previous_result, previous_path = load_round2_bundle(round2root, job["phase2_job_id"])
    for key in ("source_ids", "support_ids", "query_ids", "classes", "label_column"):
        if previous_result[key] != job[key]:
            raise ValueError("G04 differs from historical F03 role/label definition: " + key)
    previous_predictions = pd.read_csv(previous_path / "predictions.csv.gz", keep_default_na=False)
    models, provenance, audits = {}, [], {}
    # Reproduce saved predictions with existing models before evaluation joins labels.
    for setting in ("balanced", "none"):
        model, origin = load_weight_model(job, setting, oldroot, round2root)
        if model.classes_.tolist() != job["classes"]:
            raise ValueError("Historical model vocabulary changed")
        before_hash = joblib.hash(model)
        models[setting] = (model, before_hash)
        provenance.append({"setting": setting, **origin})
        current_scores, _ = _binary_scores(model, writer.query_X)
        current_native = model.estimator_.predict(model._scaled(writer.query_X))
        for suffix in ("", "_argmax"):
            method = f"class_weight_{setting}" + suffix
            rows, scores, predicted = _saved_endpoint(previous_predictions, method, job["query_ids"], job["classes"])
            if not np.allclose(current_scores, scores, rtol=0, atol=1e-12):
                raise ValueError("Historical model no longer reproduces saved Query scores")
            expected_prediction = model.classes_[current_scores.argmax(axis=1)] if suffix else current_native
            if not np.array_equal(expected_prediction, predicted):
                raise ValueError("Historical model no longer reproduces saved Query decisions")
            writer.record(method, 0, scores, pred=predicted, draw=0, support_ids=[],
                          decision_rule="score_argmax" if suffix else "SVC.predict",
                          reused_model=True, saved_prediction_reused=True,
                          new_classifier_fits=0, new_adapter_fit=False,
                          source_phase2_job=job["phase2_job_id"],
                          saved_prediction_sha256=previous_result["artifact_hashes"]["predictions.csv.gz"])
    # Labels only evaluate models that were already trained and predicted above.
    source_X, source_y = writer.take(job["source_ids"]), writer.labels(job["source_ids"])
    query_y = writer.labels(job["query_ids"])
    for setting, (model, _) in models.items():
        audits[setting] = audit_model(model, source_X, source_y, job["source_ids"], writer.query_X, query_y, setting)
    pair = compare_models(models["balanced"][0], models["none"][0], audits["balanced"], audits["none"], query_y)
    context = {key: job.get(key) for key in ("job_id", "family", "experiment", "phase2_job_id", "target_origin", "seed", "ratio", "coverage")}
    mechanism_rows = [{**context, **audits[setting]["summary"]} for setting in ("balanced", "none")]
    mechanism_rows.append({**context, **pair})
    pd.DataFrame(mechanism_rows).to_csv(writer.temp / "margin_mechanism.csv", index=False)
    sample_rows = [{**context, **value} for setting in ("balanced", "none") for value in audits[setting]["samples"]]
    pd.DataFrame(sample_rows).to_csv(writer.temp / "source_sample_diagnostics.csv", index=False)
    pairs = []
    for index, sample in enumerate(job["query_ids"]):
        left, right = audits["balanced"], audits["none"]
        pairs.append({**context, "record_id": sample,
                      "physical_sample_id": writer.indexed.at[sample, "physical_sample_id"],
                      "true_label": query_y[index],
                      "balanced_margin": float(left["query_margin"][index]),
                      "none_margin": float(right["query_margin"][index]),
                      "margin_delta": float(right["query_margin"][index] - left["query_margin"][index]),
                      "balanced_prediction": left["query_native"][index],
                      "none_prediction": right["query_native"][index]})
    pd.DataFrame(pairs).to_csv(writer.temp / "query_margin_pairs.csv", index=False)
    writer.save_json("margin_mechanism.json", {"models": {s: audits[s]["summary"] for s in audits}, "comparison": pair,
                                               "new_fits": 0, "diagnostic_tolerance_is_not_a_stopping_rule": True})
    writer.save_json("reused_model_provenance.json", provenance)
    for model, before_hash in models.values():
        if joblib.hash(model) != before_hash:
            raise ValueError("Historical model changed during zero-fit audit")
    return writer.finish(expected_endpoints=expected_endpoints(job), new_model_fits=0,
                         new_adapter_fits=0, reused_prediction_endpoints=4,
                         parent_result_sha256=file_hash(previous_path / "result.json"),
                         source_models_unchanged=True, margin_comparison=pair)
