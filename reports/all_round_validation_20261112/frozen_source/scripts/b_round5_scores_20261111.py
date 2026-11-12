"""I03/I04: anchored pairwise-score information and ESS-aware fixed shrinkage.

Every output remains an adaptation of the same frozen class-aligned OVR source
scores. Score feature bases are fitted on Source only, never Support or Query.
The ESS rule is a prespecified heuristic, not a Query-selected regularization.
"""
from __future__ import annotations

import copy
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from b_followup_common_20261110 import TaskWriter, endpoint_id, file_hash, load_parent
from b_followup_calibration_20261110 import support_plan
from b_mechanism_engine_20261110 import native_predict
from b_round4_support_20261111 import (adapter_geometry, class_balanced_weights,
                                     composition_plan, fit_residual_adapter,
                                     support_compositions)


FEATURE_METHODS = ("ovr_trace1", "ovo_trace1", "ovo_pca_trace1")
I03_METHODS = ("raw_mean_anchor", *FEATURE_METHODS)
I04_METHODS = ("weighted_lambda1", "weighted_mean", "weighted_ess")


def episodes():
    return [(draw, k) for draw in (0, 101, 102) for k in (1, 3, 5, 10)
            if draw == 0 or k != 10]


def expected_endpoints(job):
    result = [endpoint_id("identity"), endpoint_id("svc_predict")]
    if job["family"] == "I03":
        result += [endpoint_id(method, k, draw) for draw, k in episodes() for method in I03_METHODS]
    elif job["family"] == "I04":
        for composition in support_compositions(job):
            for draw in (0, 101, 102):
                encoded = composition["composition_index"] * 1000 + draw
                result += [endpoint_id(method, -1, encoded) for method in I04_METHODS]
    else:
        raise ValueError("Unknown score/ESS family")
    return result


def fit_count(job=None):
    return 0


def adapter_count(job):
    if job["family"] == "I03":
        return 30
    if job["family"] == "I04":
        return 2 * 3 * sum(not item["balanced_composition"] for item in support_compositions(job))
    raise ValueError("Unknown score/ESS family")


def source_score_features(source, X):
    """Return immutable-source OVR output scores and public-API OVO features."""
    ovr = np.asarray(source.predict_scores(X), dtype=float)
    if source.reducer_ is not None or source.candidate["model"] != "svc":
        raise ValueError("Pairwise features require the frozen unreduced SVC source")
    estimator = copy.deepcopy(source.estimator_)
    estimator.decision_function_shape = "ovo"
    ovo = np.asarray(estimator.decision_function(source._scaled(X)), dtype=float)
    if ovo.ndim == 1:
        ovo = ovo[:, None]
    expected = len(source.classes_) * (len(source.classes_) - 1) // 2
    if ovo.shape != (len(ovr), expected) or not np.isfinite(ovo).all():
        raise ValueError("Unexpected finite Source OVO margin dimensions")
    return ovr, ovo


def fit_feature_basis(source, X_source, mode):
    """Fit only Source score centering, global trace scaling, and optional PCA."""
    if mode not in FEATURE_METHODS:
        raise ValueError("Unknown prespecified score feature basis")
    ovr, ovo = source_score_features(source, X_source)
    raw = ovr if mode == "ovr_trace1" else ovo
    pca = None
    if mode == "ovo_pca_trace1":
        components = min(len(source.classes_), raw.shape[1], len(raw))
        pca = PCA(n_components=components, svd_solver="full").fit(raw)
        raw = pca.transform(raw)
    center = raw.mean(axis=0)
    centered = raw - center
    squared_rms = float(np.mean(np.sum(centered ** 2, axis=1)))
    scale = float(np.sqrt(squared_rms)) if squared_rms > 0 else 1.0
    return {"mode": mode, "center": center, "scale": scale, "pca": pca,
            "source_samples": len(raw), "feature_count": raw.shape[1],
            "original_ovo_features": ovo.shape[1],
            "source_trace_before_scaling": squared_rms,
            "zero_source_variance_fallback": squared_rms == 0,
            "source_class_order": source.classes_.tolist(),
            "fit_scope": "Source scores only"}


def transform_feature_basis(source, X, basis):
    ovr, ovo = source_score_features(source, X)
    values = ovr if basis["mode"] == "ovr_trace1" else ovo
    if basis["pca"] is not None:
        values = basis["pca"].transform(values)
    result = (values - basis["center"]) / basis["scale"]
    if result.shape != (len(ovr), basis["feature_count"]) or not np.isfinite(result).all():
        raise ValueError("Score feature transformation produced invalid values")
    return result


def fit_feature_residual(features, scores, labels, classes, regularization=None):
    """Fit anchored residuals from features; labels never include Query values."""
    features, scores = np.asarray(features, dtype=float), np.asarray(scores, dtype=float)
    labels, classes = np.asarray(labels), list(classes)
    if (features.ndim != 2 or len(features) != len(labels)
            or scores.shape != (len(labels), len(classes)) or not len(labels)
            or not np.isfinite(features).all() or not np.isfinite(scores).all()
            or set(labels.tolist()) != set(classes)):
        raise ValueError("Residual feature/score/Support label dimensions disagree")
    penalty = float(len(labels) if regularization is None else regularization)
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("A finite positive ridge penalty is required")
    index = {label: position for position, label in enumerate(classes)}
    target = np.eye(len(classes))[[index[label] for label in labels]]
    design = np.column_stack([features, np.ones(len(labels))])
    gram = design.T @ design
    system = gram + penalty * np.eye(design.shape[1])
    delta = np.linalg.solve(system, design.T @ (target - scores))
    return {"feature_coef": delta[:-1].tolist(), "intercept": delta[-1].tolist(),
            "regularization": penalty, "effective_lambda": penalty,
            "support_count": len(labels), "feature_count": features.shape[1],
            "effective_df_per_output": float(np.trace(np.linalg.solve(system, gram))),
            "regularized_condition": float(np.linalg.cond(system)),
            "design_rank": int(np.linalg.matrix_rank(design)),
            "correction_frobenius": float(np.linalg.norm(delta[:-1])),
            "intercept_l2": float(np.linalg.norm(delta[-1])),
            "output_anchor": "unchanged frozen Source class-aligned OVR scores",
            "objective": "sum(||onehot-S0-[Source_feature,1]delta||^2)+N*||delta||^2; intercept penalized"}


def ess_penalty(weights):
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or not len(weights) or not np.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("ESS weights must be a nonempty positive finite vector")
    mass = float(weights.sum())
    n = len(weights)
    if not np.isclose(mass, n, atol=1e-12, rtol=0):
        raise ValueError("The frozen ESS comparison requires sum of weights=N")
    ess = float(mass ** 2 / np.dot(weights, weights))
    return {"support_count": n, "total_support_weight": mass,
            "ESS": ess, "effective_support_weight_ess": ess,
            "mean_lambda": float(n), "ess_lambda": float(n * n / ess)}


def _load_bundle(root, identifier):
    from b_round5_common_20261111 import load_bundle
    return load_bundle(root, identifier)


def _check_roles(job, old):
    for field in ("source_ids", "support_ids", "query_ids", "classes", "label_column"):
        if job[field] != old[field]:
            raise ValueError("Reused score evidence has different sample/label definition: " + field)


def _history(result, path):
    frame = pd.read_csv(path / "predictions.csv.gz", keep_default_na=False)
    parameters = {item["endpoint_id"]: item for item in json.loads((path / "adapter_parameters.json").read_text())}
    endpoints = {item["endpoint_id"]: item for item in json.loads((path / "endpoints.json").read_text())}
    return frame, parameters, endpoints


def _old_endpoint(history, method, k, draw, query_ids, ids, labels):
    frame, parameters, endpoints = history
    identifier = endpoint_id(method, k, draw)
    if identifier not in parameters or identifier not in endpoints:
        raise ValueError("Exact historical calibration endpoint is absent")
    parameter, endpoint = parameters[identifier], endpoints[identifier]
    if endpoint["support_ids"] != ids or parameter["support_ids"] != ids:
        raise ValueError("Historical calibration Support ID order differs")
    saved_labels = parameter.get("fit_labels", parameter.get("support_labels"))
    if saved_labels != labels.tolist():
        raise ValueError("Historical calibration Support label order differs")
    rows = frame.loc[frame.endpoint_id.eq(identifier)]
    if len(rows) != len(query_ids) or not rows.record_id.is_unique or set(rows.record_id) != set(query_ids):
        raise ValueError("Historical Query endpoint is not exactly aligned")
    rows = rows.set_index("record_id").loc[query_ids]
    adapter = dict(parameter["adapter"])
    if "weights" in parameter and "weights" not in adapter:
        adapter["weights"] = parameter["weights"]
    return (np.asarray([json.loads(text) for text in rows.scores_json]),
            rows.pred_label.to_numpy(), adapter, identifier)


def _record_source(writer, source):
    scores = source.predict_scores(writer.query_X)
    writer.record("identity", 0, scores, support_ids=[], new_model_fit=False, new_adapter_fit=False,
                  support_labels_used=0)
    writer.record("svc_predict", 0, scores, pred=native_predict(source, writer.query_X),
                  support_ids=[], decision_rule="SVC.predict", new_model_fit=False,
                  new_adapter_fit=False, support_labels_used=0)
    return scores


def _motion(source_scores, scores):
    delta = scores - source_scores
    return {"score_moved_count": int(np.count_nonzero(np.any(delta != 0, axis=1))),
            "decision_changed_count": int(np.count_nonzero(scores.argmax(axis=1) != source_scores.argmax(axis=1))),
            "maximum_score_correction": float(np.max(np.abs(delta)))}


def _i03(writer, job, source, round3root):
    old_id = "G01_" + job["parent_job_id"]
    old_result, old_path = _load_bundle(round3root, old_id)
    _check_roles(job, old_result)
    historical = _history(old_result, old_path)
    source_scores = _record_source(writer, source)
    bases = {name: fit_feature_basis(source, writer.take(job["source_ids"]), name) for name in FEATURE_METHODS}
    source_features = {name: transform_feature_basis(source, writer.take(job["source_ids"]), basis)
                       for name, basis in bases.items()}
    pool_features = {name: transform_feature_basis(source, writer.take(job["support_ids"]), basis)
                     for name, basis in bases.items()}
    query_features = {name: transform_feature_basis(source, writer.query_X, basis)
                      for name, basis in bases.items()}
    pool_scores = source.predict_scores(writer.take(job["support_ids"]))
    pool_lookup = {sample: index for index, sample in enumerate(job["support_ids"])}
    parameters, geometry, support_locks, binary_checks = [], [], {}, []
    fitted = 0
    for draw, k in episodes():
        ids = support_plan(job, writer.metadata, draw, k)
        positions = [pool_lookup[sample] for sample in ids]
        support_y = writer.labels(ids)
        scores_support = pool_scores[positions]
        support_locks[f"draw{draw}_k{k}"] = ids
        baseline, prediction, old_adapter, old_endpoint = _old_endpoint(
            historical, "mean_penalized", k, draw, job["query_ids"], ids, support_y)
        if float(old_adapter["regularization"]) != len(ids):
            raise ValueError("Historical mean-Anchor does not use sum lambda=N")
        reconstructed = source_scores @ old_adapter["coef"] + old_adapter["intercept"]
        if not np.allclose(baseline, reconstructed, atol=1e-12, rtol=0):
            raise ValueError("Historical mean-Anchor scores no longer reproduce")
        writer.record("raw_mean_anchor", k, baseline, pred=prediction, draw=draw, support_ids=ids,
                      adapter_reused=True, new_adapter_fit=False, new_model_fit=False,
                      support_labels_used=len(ids), effective_lambda=float(len(ids)),
                      feature_basis="historical raw OVR", output_anchor_preserved=True,
                      source_input_fit_scope="Source only", historical_endpoint_id=old_endpoint)
        output_scores = {}
        for method in FEATURE_METHODS:
            started = time.process_time()
            adapter = fit_feature_residual(pool_features[method][positions], scores_support,
                                           support_y, job["classes"])
            scores = source_scores + query_features[method] @ adapter["feature_coef"] + adapter["intercept"]
            elapsed = time.process_time() - started
            fitted += 1
            output_scores[method] = scores
            detail = {key: adapter[key] for key in ("regularization", "effective_lambda", "feature_count",
                      "effective_df_per_output", "regularized_condition", "design_rank",
                      "correction_frobenius", "intercept_l2")}
            detail.update(_motion(source_scores, scores))
            id_ = writer.record(method, k, scores, draw=draw, support_ids=ids,
                                adapter_reused=False, new_adapter_fit=True, new_model_fit=False,
                                support_labels_used=len(ids), output_anchor_preserved=True,
                                source_input_fit_scope="Source only", adapter_cpu_seconds=elapsed, **detail)
            parameters.append({"endpoint_id": id_, "method": method, "k": k, "draw": draw,
                               "support_ids": ids, "support_labels": support_y.tolist(), "adapter": adapter})
            geometry.append({"endpoint_id": id_, "method": method, "k": k, "draw": draw,
                             "support_count": len(ids), "source_feature_trace": float(np.mean(np.sum(source_features[method] ** 2, axis=1))),
                             **detail})
        if len(job["classes"]) == 2:
            reference = source_features["ovr_trace1"] @ source_features["ovr_trace1"].T
            for method in ("ovo_trace1", "ovo_pca_trace1"):
                other = source_features[method] @ source_features[method].T
                binary_checks.append({"draw": draw, "k": k, "comparison": method + "_minus_ovr_trace1",
                                      "source_gram_max_absolute_difference": float(np.max(np.abs(other - reference))),
                                      "query_score_max_absolute_difference": float(np.max(np.abs(output_scores[method] - output_scores["ovr_trace1"]))),
                                      "query_prediction_disagreements": int(np.count_nonzero(output_scores[method].argmax(axis=1) != output_scores["ovr_trace1"].argmax(axis=1))),
                                      "interpretation": "Algebraically equivalent normalized binary Gram embeddings; floating differences measured, not three independent experiments"})
    if fitted != 30:
        raise ValueError("I03 fitting count changed")
    writer.save_model("score_feature_basis.joblib", bases)
    basis_info = {}
    for name, basis in bases.items():
        description = {key: value for key, value in basis.items() if key != "pca"}
        pca = basis["pca"]
        description["pca"] = None if pca is None else {"components": pca.components_.tolist(),
            "mean": pca.mean_.tolist(), "explained_variance": pca.explained_variance_.tolist(),
            "n_samples": pca.n_samples_}
        basis_info[name] = description
    writer.save_json("score_feature_basis.json", basis_info)
    writer.save_json("adapter_parameters.json", parameters)
    writer.save_json("support_locks.json", support_locks)
    writer.save_json("binary_equivalence.json", {"binary_task": len(job["classes"]) == 2,
                                                "comparisons": binary_checks,
                                                "classification_model_fits": 0})
    pd.DataFrame(geometry).to_csv(writer.temp / "score_geometry.csv", index=False)
    writer.save_json("historical_reuse.json", {"job_id": old_id,
        "result_sha256": file_hash(old_path / "result.json"),
        "raw_mean_anchor_endpoints_reused": 10,
        "output_K0_policy": "Original class-aligned Source OVR scores retained for every feature basis"})
    if joblib.hash(source) != writer.source_initial_hash:
        raise ValueError("I03 changed the frozen Source before sealing")
    return writer.finish(expected_endpoints(job), new_model_fits=0, new_adapter_fits=fitted,
                         source_model_unchanged=True, reused_adapter_endpoints=10,
                         source_feature_basis_fits=3, source_only_PCA_fits=1,
                         binary_equivalent_feature_negative_controls=len(job["classes"]) == 2)


def _i04(writer, job, source, round3root, round4root):
    prior_id = "H04_" + job["parent_job_id"]
    prior_result, prior_path = _load_bundle(round4root, prior_id)
    mean_id = "G01_" + job["parent_job_id"]
    mean_result, mean_path = _load_bundle(round3root, mean_id)
    _check_roles(job, prior_result)
    _check_roles(job, mean_result)
    historical, historical_mean = _history(prior_result, prior_path), _history(mean_result, mean_path)
    plan = composition_plan(job, writer.metadata)
    if job.get("composition_plan") is not None and job["composition_plan"] != plan:
        raise ValueError("Frozen composition ID/quantity plan changed")
    source_scores = _record_source(writer, source)
    pool_scores = source.predict_scores(writer.take(job["support_ids"]))
    score_lookup = dict(zip(job["support_ids"], pool_scores))
    parameters, geometry, reuse = [], [], []
    fitted = 0
    for episode in plan["episodes"]:
        ids = episode["support_ids"]
        labels = writer.labels(ids)
        observed = [int(np.count_nonzero(labels == label)) for label in job["classes"]]
        if observed != episode["counts"]:
            raise ValueError("Frozen Support composition labels changed")
        support_scores = np.asarray([score_lookup[sample] for sample in ids])
        weights = class_balanced_weights(labels, job["classes"])
        penalty = ess_penalty(weights)
        balanced = episode["balanced_composition"]
        baseline, baseline_pred, baseline_adapter, baseline_id = _old_endpoint(
            historical, "anchor_class_balanced", -1, episode["draw"], job["query_ids"], ids, labels)
        if float(baseline_adapter["regularization"]) != 1.0:
            raise ValueError("Historical weighted baseline is not lambda1")
        if not np.array_equal(np.asarray(baseline_adapter["weights"]), weights):
            raise ValueError("Historical weighted baseline uses different class weights")
        for method in I04_METHODS:
            reused = method == "weighted_lambda1" or balanced
            if method == "weighted_lambda1":
                scores, predicted, adapter, reuse_id = baseline, baseline_pred, baseline_adapter, baseline_id
                reuse_job = prior_id
                requested_lambda = 1.0
            else:
                requested_lambda = penalty["mean_lambda"] if method == "weighted_mean" else penalty["ess_lambda"]
                if balanced:
                    if not np.array_equal(weights, np.ones(len(ids))) or requested_lambda != len(ids):
                        raise ValueError("Balanced ESS/mean objective equivalence is not exact")
                    scores, predicted, adapter, reuse_id = _old_endpoint(
                        historical_mean, "mean_penalized", 5, episode["draw_seed"], job["query_ids"], ids, labels)
                    if float(adapter["regularization"]) != requested_lambda:
                        raise ValueError("Balanced mean-Anchor reuse has a different penalty")
                    reuse_job = mean_id
                else:
                    adapter = fit_residual_adapter(support_scores, labels, job["classes"], kind="anchor",
                                                   class_balanced=True, regularization=requested_lambda)
                    scores = source_scores @ adapter["coef"] + adapter["intercept"]
                    predicted = writer.classes[scores.argmax(axis=1)]
                    fitted += 1
                    reuse_id = reuse_job = None
            reconstructed = source_scores @ adapter["coef"] + adapter["intercept"]
            if not np.allclose(scores, reconstructed, rtol=0, atol=1e-12):
                raise ValueError("ESS endpoint does not reproduce its saved/fitted parameters")
            detail = adapter_geometry(support_scores, weights, adapter, "anchor")
            detail.update(_motion(source_scores, scores))
            id_ = writer.record(method, -1, scores, pred=predicted, draw=episode["draw"], support_ids=ids,
                                composition_id=episode["composition_id"], composition_index=episode["composition_index"],
                                draw_seed=episode["draw_seed"], balanced_composition=balanced,
                                support_counts_json=json.dumps(dict(zip(job["classes"], episode["counts"])), ensure_ascii=False),
                                fixed_total_support=len(ids), k_per_class_not_applicable=True,
                                support_labels_used=len(ids), new_model_fit=False, new_adapter_fit=not reused,
                                adapter_reused=reused, regularization=requested_lambda,
                                mean_lambda=penalty["mean_lambda"], ess_lambda=penalty["ess_lambda"],
                                penalty_rule=method, **detail)
            parameters.append({"endpoint_id": id_, "method": method, "support_ids": ids,
                               "support_labels": labels.tolist(), "weights": weights.tolist(),
                               "adapter": adapter, "adapter_reused": reused})
            geometry.append({"endpoint_id": id_, "method": method,
                             "composition_id": episode["composition_id"], "draw_seed": episode["draw_seed"],
                             "balanced_composition": balanced, "support_count": len(ids),
                             "regularization": requested_lambda, "mean_lambda": penalty["mean_lambda"],
                             "ess_lambda": penalty["ess_lambda"], "penalty_rule": method, **detail})
            if reused:
                reuse.append({"endpoint_id": id_, "historical_job_id": reuse_job,
                              "historical_endpoint_id": reuse_id,
                              "reason": "exact fixed-ID weighted lambda1 baseline" if method == "weighted_lambda1"
                              else "balanced unit weights imply ESS=N and both penalties equal historical sum lambda=N"})
    if fitted != adapter_count(job):
        raise ValueError("I04 adapter fitting count changed")
    writer.save_json("adapter_parameters.json", parameters)
    writer.save_json("composition_plan.json", plan)
    writer.save_json("reused_adapter_provenance.json", reuse)
    pd.DataFrame(geometry).to_csv(writer.temp / "ess_geometry.csv", index=False)
    if joblib.hash(source) != writer.source_initial_hash:
        raise ValueError("I04 changed the frozen Source before sealing")
    return writer.finish(expected_endpoints(job), new_model_fits=0, new_adapter_fits=fitted,
                         source_model_unchanged=True, reused_adapter_endpoints=len(reuse),
                         ESS_shrinkage_status="prespecified heuristic; no Query selection",
                         unchanged_support_weights_and_ids=True)


def execute(job, X, metadata, axis, output, input_hash, oldroot, round2root, round3root, round4root):
    writer = TaskWriter(job, X, metadata, axis, output, input_hash, oldroot)
    _, source, _ = load_parent(job, oldroot)
    source_before = joblib.hash(source)
    writer.source_initial_hash = source_before
    if job["family"] == "I03":
        result = _i03(writer, job, source, round3root)
    elif job["family"] == "I04":
        result = _i04(writer, job, source, round3root, round4root)
    else:
        raise ValueError("Unknown fifth-round score family")
    if joblib.hash(source) != source_before:
        raise ValueError("The frozen Source classifier or coordinate state changed")
    return result
