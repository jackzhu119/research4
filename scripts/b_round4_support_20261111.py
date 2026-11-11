"""H04: fixed-total Support composition and matched weighted score adaptation.

Only the frozen Support pool supplies adaptation labels.  Class balancing is an
explicit sampling correction with fixed total weight, not a class-posterior or
new-batch claim.  Balanced K5 endpoints are reused from sealed round-two scores.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd

from b_followup_common_20261110 import TaskWriter, endpoint_id, file_hash, load_parent
from b_followup_calibration_20261110 import support_plan
from b_mechanism_engine_20261110 import native_predict


DRAW_SEEDS = (0, 101, 102)
METHODS = ("anchor_unweighted", "anchor_class_balanced",
           "bias_unweighted", "bias_class_balanced")


def support_compositions(job):
    """Predeclared class counts; their order fixes the endpoint identifiers."""
    n_classes = len(job["classes"])
    if n_classes == 2:
        counts = [[1, 9], [3, 7], [5, 5], [7, 3], [9, 1]]
        names = ["binary_1_9", "binary_3_7", "balanced", "binary_7_3", "binary_9_1"]
    elif n_classes == 5:
        counts = [[5] * 5]
        names = ["balanced"]
        for index, label in enumerate(job["classes"]):
            value = [6] * 5
            value[index] = 1
            counts.append(value)
            names.append("scarce_" + str(label))
    else:
        raise ValueError("H04 permits the fixed binary or five-origin task only")
    return [{"composition_index": index, "composition_id": name,
             "counts": value, "balanced_composition": len(set(value)) == 1,
             "fixed_total_support": int(sum(value))}
            for index, (name, value) in enumerate(zip(names, counts))]


def support_ids(job, metadata, counts, draw_seed):
    """Take per-class prefixes of precisely the round-two locked pool order."""
    if draw_seed not in DRAW_SEEDS or len(counts) != len(job["classes"]):
        raise ValueError("Support composition/draw is outside the frozen plan")
    if any(isinstance(n, bool) or int(n) != n or n < 1 or n > 10 for n in counts):
        raise ValueError("Every class must have one through ten Support specimens")
    ordered = support_plan(job, metadata, draw_seed, 10)
    indexed = metadata.set_index("record_id", drop=False)
    result = []
    for label, count in zip(job["classes"], counts):
        members = [sample for sample in ordered if indexed.at[sample, job["label_column"]] == label]
        if len(members) != 10:
            raise ValueError("A fixed ten-member candidate pool is required per class")
        result.extend(members[:int(count)])
    if len(result) != sum(counts) or len(set(result)) != len(result):
        raise ValueError("Invalid or repeated Support identity")
    return result


def composition_plan(job, metadata):
    episodes = []
    for composition in support_compositions(job):
        for draw_seed in DRAW_SEEDS:
            episodes.append({**composition, "draw_seed": draw_seed,
                             "draw": composition["composition_index"] * 1000 + draw_seed,
                             "support_ids": support_ids(job, metadata, composition["counts"], draw_seed)})
    return {"schema_version": 1, "episodes": episodes, "draw_seeds": list(DRAW_SEEDS),
            "support_total": 10 if len(job["classes"]) == 2 else 25,
            "new_adapter_fits": adapter_count(job),
            "support_order": "exact round-two per-class Support order and deterministic redraw"}


def expected_endpoints(job):
    result = [endpoint_id("identity"), endpoint_id("svc_predict")]
    for composition in support_compositions(job):
        for draw_seed in DRAW_SEEDS:
            draw = composition["composition_index"] * 1000 + draw_seed
            result.extend(endpoint_id(method, -1, draw) for method in METHODS)
    return result


def fit_count(job=None):
    return 0


def adapter_count(job):
    unbalanced = sum(not composition["balanced_composition"] for composition in support_compositions(job))
    return unbalanced * len(DRAW_SEEDS) * len(METHODS)


def class_balanced_weights(labels, classes):
    """Inverse class-frequency weights whose total equals the actual pool size."""
    values = np.asarray(labels)
    classes = list(classes)
    if not len(values) or len(classes) != len(set(classes)) or set(values.tolist()) != set(classes):
        raise ValueError("Support must contain every and only the frozen class labels")
    counts = {label: int(np.count_nonzero(values == label)) for label in classes}
    weights = np.asarray([len(values) / (len(classes) * counts[label]) for label in values], dtype=float)
    if not np.isfinite(weights).all() or not np.isclose(weights.sum(), len(values), atol=1e-12, rtol=0):
        raise ValueError("Support weights must be finite with fixed total mass")
    return weights


def fit_residual_adapter(support_scores, support_y, classes, *, kind="anchor",
                         class_balanced=False, regularization=1.0):
    """Fit a standard anchored weighted residual ridge, without Query inputs."""
    scores = np.asarray(support_scores, dtype=float)
    labels = np.asarray(support_y)
    classes = list(classes)
    if (scores.shape != (len(labels), len(classes)) or not len(labels)
            or not np.isfinite(scores).all() or set(labels.tolist()) != set(classes)):
        raise ValueError("Support scores/labels do not align with the frozen vocabulary")
    if kind not in ("anchor", "bias") or regularization <= 0 or not np.isfinite(regularization):
        raise ValueError("Unknown adapter or invalid positive ridge penalty")
    vocabulary = {label: index for index, label in enumerate(classes)}
    target = np.eye(len(classes))[[vocabulary[label] for label in labels]]
    residual = target - scores
    weights = class_balanced_weights(labels, classes) if class_balanced else np.ones(len(labels))
    if kind == "bias":
        coefficient = np.eye(len(classes))
        intercept = (weights[:, None] * residual).sum(axis=0) / (weights.sum() + regularization)
    else:
        design = np.column_stack([scores, np.ones(len(scores))])
        gram = design.T @ (weights[:, None] * design)
        rhs = design.T @ (weights[:, None] * residual)
        delta = np.linalg.solve(gram + regularization * np.eye(len(classes) + 1), rhs)
        coefficient, intercept = np.eye(len(classes)) + delta[:-1], delta[-1]
    return {"kind": kind, "class_balanced": bool(class_balanced),
            "regularization": float(regularization), "coef": coefficient.tolist(),
            "intercept": intercept.tolist(), "weights": weights.tolist(),
            "total_support_weight": float(weights.sum()),
            "effective_support_weight_ess": float(weights.sum() ** 2 / np.dot(weights, weights)),
            "objective": "sum(w_i * ||onehot_i-score_i-[score_i,1]delta||^2) + lambda*||delta||^2; all parameters penalized"}


def adapter_geometry(support_scores, weights, adapter, kind):
    scores, weights = np.asarray(support_scores, dtype=float), np.asarray(weights, dtype=float)
    penalty = float(adapter.get("regularization", 1.0))
    coefficient, intercept = np.asarray(adapter["coef"]), np.asarray(adapter["intercept"])
    design = np.column_stack([scores, np.ones(len(scores))])
    if kind == "bias":
        condition, df = 1.0, float(weights.sum() / (weights.sum() + penalty))
    else:
        gram = design.T @ (weights[:, None] * design)
        system = gram + penalty * np.eye(design.shape[1])
        condition = float(np.linalg.cond(system))
        df = float(np.trace(np.linalg.solve(system, gram)))
    correction = coefficient - np.eye(scores.shape[1])
    off_diagonal = correction.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    return {"design_rank": int(np.linalg.matrix_rank(design)),
            "regularized_condition": condition, "effective_df_per_output": df,
            "correction_frobenius": float(np.linalg.norm(correction)),
            "off_diagonal_frobenius": float(np.linalg.norm(off_diagonal)),
            "intercept_l2": float(np.linalg.norm(intercept)),
            "total_support_weight": float(weights.sum()),
            "effective_support_weight_ess": float(weights.sum() ** 2 / np.dot(weights, weights))}


def _reused_endpoint(predictions, parameters, old_method, draw_seed, query_ids, ids, labels):
    old_key = endpoint_id(old_method, 5, draw_seed)
    parameter = parameters.get(old_key)
    if parameter is None:
        raise ValueError("Balanced K5 historical adapter is absent")
    if parameter["support_ids"] != ids or parameter["fit_labels_diagnostic_copy"] != labels.tolist():
        raise ValueError("Balanced K5 reuse has different Support IDs or labels")
    if parameter.get("diagnostic_label_copy"):
        raise ValueError("A diagnostic-label adapter cannot replace the true-label baseline")
    block = predictions.loc[predictions.endpoint_id.eq(old_key)].set_index("record_id").loc[query_ids]
    if len(block) != len(query_ids) or not block.index.is_unique:
        raise ValueError("Historical K5 Query rows are incomplete")
    scores = np.asarray([json.loads(value) for value in block.scores_json], dtype=float)
    predicted = block.pred_label.to_numpy()
    return scores, predicted, dict(parameter["adapter"]), old_key


def execute(job, X, metadata, axis, output, input_hash, oldroot, round2root, round3root):
    from b_round4_common_20261111 import load_round2_bundle

    writer = TaskWriter(job, X, metadata, axis, output, input_hash, oldroot)
    old_result, source, parent_path = load_parent(job, oldroot)
    source_before = joblib.hash(source)
    phase2_id = "F01_" + job["parent_job_id"]
    phase2_result, phase2_path = load_round2_bundle(round2root, phase2_id)
    for key in ("source_ids", "support_ids", "query_ids", "classes", "label_column"):
        if phase2_result[key] != job[key]:
            raise ValueError("Historical score baseline has a different sample definition")
    if source.classes_.tolist() != job["classes"]:
        raise ValueError("Source class vocabulary changed")
    plan = composition_plan(job, metadata)
    if job.get("composition_plan") is not None and job["composition_plan"] != plan:
        raise ValueError("Frozen fixed-total Support composition plan changed")
    query_scores = source.predict_scores(writer.query_X)
    support_scores = source.predict_scores(writer.take(job["support_ids"]))
    score_lookup = dict(zip(job["support_ids"], support_scores))
    predictions = pd.read_csv(phase2_path / "predictions.csv.gz", keep_default_na=False)
    parameter_list = json.loads((phase2_path / "adapter_parameters.json").read_text())
    parameters = {item["endpoint_id"]: item for item in parameter_list}
    writer.record("identity", 0, query_scores, support_ids=[], new_model_fit=False,
                  new_adapter_fit=False, support_labels_used=0)
    writer.record("svc_predict", 0, query_scores, pred=native_predict(source, writer.query_X),
                  support_ids=[], decision_rule="SVC.predict", new_model_fit=False,
                  new_adapter_fit=False, support_labels_used=0)
    new_adapters = 0
    adapters, geometry, reuse = [], [], []
    for episode in plan["episodes"]:
        ids = episode["support_ids"]
        scores = np.asarray([score_lookup[sample] for sample in ids])
        true_support = writer.labels(ids)
        if [int(np.count_nonzero(true_support == label)) for label in job["classes"]] != episode["counts"]:
            raise ValueError("Support class counts differ from the frozen composition")
        for method in METHODS:
            weighted = method.endswith("class_balanced")
            kind = "anchor" if method.startswith("anchor") else "bias"
            weights = class_balanced_weights(true_support, job["classes"]) if weighted else np.ones(len(ids))
            start_cpu, start_wall = time.process_time(), time.monotonic()
            reused = episode["balanced_composition"]
            old_endpoint = None
            if reused:
                if not np.array_equal(weights, np.ones(len(ids))):
                    raise ValueError("Balanced/ordinary residual objectives are not exactly equivalent")
                old_method = "raw_anchor_lambda1" if kind == "anchor" else "raw_bias_only"
                new_scores, predicted, adapter, old_endpoint = _reused_endpoint(
                    predictions, parameters, old_method, episode["draw_seed"], job["query_ids"], ids, true_support)
                reconstructed = query_scores @ np.asarray(adapter["coef"]) + np.asarray(adapter["intercept"])
                if not np.allclose(new_scores, reconstructed, atol=1e-12, rtol=0):
                    raise ValueError("Reused balanced K5 adapter no longer reconstructs its saved scores")
                if not np.array_equal(writer.classes[new_scores.argmax(axis=1)], predicted):
                    raise ValueError("Reused balanced K5 endpoint has a different decision rule")
                reuse.append({"method": method, "draw": episode["draw"],
                              "historical_endpoint_id": old_endpoint, "historical_job_id": phase2_id,
                              "reason": "EXACT_BALANCED_K5_ENDPOINT; unit weights also prove weighted objective equivalence"})
            else:
                adapter = fit_residual_adapter(scores, true_support, job["classes"],
                                                kind=kind, class_balanced=weighted, regularization=1.0)
                new_scores = query_scores @ np.asarray(adapter["coef"]) + np.asarray(adapter["intercept"])
                predicted = writer.classes[new_scores.argmax(axis=1)]
                new_adapters += 1
            detail = adapter_geometry(scores, weights, adapter, kind)
            id_ = writer.record(method, -1, new_scores, pred=predicted, draw=episode["draw"],
                                support_ids=ids, composition_index=episode["composition_index"],
                                composition_id=episode["composition_id"], draw_seed=episode["draw_seed"],
                                support_counts_json=json.dumps(dict(zip(job["classes"], episode["counts"])), ensure_ascii=False),
                                balanced_composition=episode["balanced_composition"],
                                fixed_total_support=episode["fixed_total_support"],
                                k_per_class_not_applicable=True, support_labels_used=len(ids),
                                support_class_balancing=weighted, adapter_reused=reused,
                                new_adapter_fit=not reused, new_model_fit=False,
                                regularization=1.0, adapter_cpu_seconds=time.process_time() - start_cpu,
                                adapter_wall_seconds=time.monotonic() - start_wall, **detail)
            adapters.append({"endpoint_id": id_, "method": method, "support_ids": ids,
                             "support_labels": true_support.tolist(), "weights": weights.tolist(),
                             "adapter": adapter, "adapter_reused": reused,
                             "historical_endpoint_id": old_endpoint})
            geometry.append({"endpoint_id": id_, "method": method,
                             "composition_id": episode["composition_id"], "draw_seed": episode["draw_seed"],
                             "balanced_composition": episode["balanced_composition"],
                             "support_count": len(ids), **detail})
    if new_adapters != adapter_count(job):
        raise ValueError("H04 adapter fitting count differs from the frozen matrix")
    if joblib.hash(source) != source_before:
        raise ValueError("H04 mutated the frozen spectral source model")
    writer.save_json("composition_plan.json", plan)
    writer.save_json("adapter_parameters.json", adapters)
    writer.save_json("reused_adapter_provenance.json", reuse)
    writer.save_json("source_reuse.json", {
        "parent_job_id": job["parent_job_id"], "source_model_sha256": old_result["source_hash"],
        "historical_score_job": phase2_id, "historical_result_sha256": file_hash(phase2_path / "result.json"),
        "historical_scores_sha256": phase2_result["artifact_hashes"]["predictions.csv.gz"],
        "historical_adapter_parameters_sha256": phase2_result["artifact_hashes"]["adapter_parameters.json"],
        "source_model_unchanged": True})
    pd.DataFrame(geometry).to_csv(writer.temp / "support_geometry.csv", index=False)
    return writer.finish(expected_endpoints(job), new_model_fits=0, new_adapter_fits=new_adapters,
                         source_model_unchanged=True, reused_adapter_endpoints=len(reuse),
                         unique_historical_adapters_reused=len({item["historical_endpoint_id"] for item in reuse}),
                         fixed_support_total=plan["support_total"],
                         interpretation="Fixed total labeled Support composition; weighted and unweighted residual fits share IDs and Query; labels are unchanged")
