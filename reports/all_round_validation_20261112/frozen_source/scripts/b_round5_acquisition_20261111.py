"""I02: label-blind ranking inside an already stratified Support candidate pool.

Ranking receives only frozen Source coordinates/scores and opaque hash keys.
Candidate labels are revealed after all rankings have been fixed. Missing class
coverage causes a prespecified score-adapter Identity fallback, never resampling
or queue termination. It does not make acquisition of the original stratified
candidate pool label-free, and policy effects also change class composition.
"""
from __future__ import annotations

import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd

from b_followup_common_20261110 import (TaskWriter, canonical_hash, endpoint_id,
                                      file_hash, load_parent, validate_roles)
from b_followup_models_20261110 import nested_support, native_predict
from b_round3_regularization_20261110 import fit_adapter
from b_round4_models_20261111 import fit_source_fixedcoords


POLICIES = (
    ("random_601", "random", 601),
    ("random_602", "random", 602),
    ("random_603", "random", 603),
    ("uncertainty", "uncertainty", 0),
    ("diversity", "diversity", 0),
)
BUDGET_MULTIPLIERS = (1, 3, 5, 10)
METHODS = ("full_refit", "guarded_mean_anchor")


def opaque_keys(record_ids, salt="round5"):
    """Remove semantic identifier prefixes before the ranking routine sees keys."""
    values = [str(value) for value in record_ids]
    if len(values) != len(set(values)) or any(not value for value in values):
        raise ValueError("Candidate identifiers must be nonempty and unique")
    return [hashlib.sha256((str(salt) + "\0" + value).encode()).hexdigest()
            for value in values]


def label_blind_order(candidate_features, source_scores, source_centroid,
                      opaque_keys, policy, seed=0):
    """Rank candidate rows without candidate labels, Query or semantic IDs.

    Randomness is a deterministic hash priority, so row permutation or the
    original class-grouped workbook order cannot change the selected identities.
    Diversity starts farthest from the Source centroid, then maximizes distance
    from its already selected points. Both deterministic policies resolve ties
    using only the supplied opaque keys.
    """
    values = np.asarray(candidate_features, dtype=float)
    scores = np.asarray(source_scores, dtype=float)
    centroid = np.asarray(source_centroid, dtype=float)
    keys = [str(value) for value in opaque_keys]
    if (values.ndim != 2 or not len(values) or scores.ndim != 2
            or len(scores) != len(values) or scores.shape[1] < 2
            or centroid.shape != (values.shape[1],)
            or len(keys) != len(values) or len(set(keys)) != len(keys)
            or not all(keys) or not np.isfinite(values).all()
            or not np.isfinite(scores).all() or not np.isfinite(centroid).all()):
        raise ValueError("Finite candidate features, Source scores, centroid and unique opaque keys are required")
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("Acquisition seed must be a nonnegative integer")
    if policy == "random":
        priorities = [hashlib.sha256((str(int(seed)) + "\0" + value).encode()).hexdigest()
                      for value in keys]
        return sorted(range(len(values)), key=lambda index: (priorities[index], keys[index]))
    if policy == "uncertainty":
        ordered = np.sort(scores, axis=1)
        gaps = ordered[:, -1] - ordered[:, -2]
        return sorted(range(len(values)), key=lambda index: (float(gaps[index]), keys[index]))
    if policy != "diversity":
        raise ValueError("Unknown frozen acquisition policy")
    differences = values - centroid
    distances = np.einsum("ij,ij->i", differences, differences)
    remaining, chosen = set(range(len(values))), []
    while remaining:
        index = min(remaining, key=lambda candidate: (-float(distances[candidate]), keys[candidate]))
        chosen.append(index)
        remaining.remove(index)
        differences = values - values[index]
        updated = np.einsum("ij,ij->i", differences, differences)
        distances = np.minimum(distances, updated)
    return chosen


def expected_endpoints(job=None):
    result = [endpoint_id("identity"), endpoint_id("svc_predict")]
    result += [endpoint_id(method, -1, policy_index * 1000 + multiplier)
               for policy_index in range(len(POLICIES))
               for multiplier in BUDGET_MULTIPLIERS for method in METHODS]
    if len(result) != 42 or len(set(result)) != 42:
        raise AssertionError("Frozen I02 endpoint matrix changed")
    return result


def fit_count(job):
    return int(job["acquisition_plan"]["new_model_fits"])


def adapter_count(job):
    return int(job["acquisition_plan"]["new_adapter_fits"])


def acquisition_plan(job, X, metadata, axis, oldroot, round2root, round3root, round4root):
    """Fix every acquisition identity before revealing its Support labels.

    Historical evidence access here obtains only the frozen source model. All
    policies are run before the first candidate label lookup. Outer Query rows
    are not indexed, scored, summarized or passed to the selectors.
    """
    validate_roles(job, metadata)
    frame = metadata.reset_index(drop=True)
    values = np.asarray(X, dtype=float)
    axis = np.asarray(axis, dtype=float)
    if (values.ndim != 2 or len(values) != len(frame) or not np.isfinite(values).all()
            or axis.shape != (values.shape[1],) or not np.isfinite(axis).all()
            or not np.all(np.diff(axis) > 0)):
        raise ValueError("Acquisition feature rows/axis must match the metadata")
    indexed = frame.set_index("record_id", drop=False)
    row = {identifier: index for index, identifier in enumerate(frame.record_id)}
    parent, source, _ = load_parent(job, oldroot)
    before = joblib.hash(source)
    if (float(source.estimator_.C) != 10.0
            or not np.array_equal(source.estimator_.class_weight_, np.ones(len(job["classes"])))):
        raise ValueError("Acquisition contexts require frozen C10 and exactly unit Source class weights")
    pool_ids = list(job["support_ids"])
    source_X = values[[row[identifier] for identifier in job["source_ids"]]]
    pool_X = values[[row[identifier] for identifier in pool_ids]]
    source_scaled = source._scaled(source_X)
    pool_scaled = source._scaled(pool_X)
    pool_scores = source.predict_scores(pool_X)
    centroid = source_scaled.mean(axis=0)
    keys = opaque_keys(pool_ids, salt="round5_acquisition:" + job["parent_job_id"])
    # Labels are deliberately unavailable in this block. Store all rankings
    # first; the label reveal for audit/adapter eligibility follows below.
    rankings = []
    for name, family, seed in POLICIES:
        order = label_blind_order(pool_scaled, pool_scores, centroid, keys, family, seed)
        rankings.append((name, family, seed, [pool_ids[index] for index in order]))

    classes = list(job["classes"])
    labels_after_ranking = indexed.loc[pool_ids, job["label_column"]].astype(str).to_dict()
    canonical = {k: nested_support(job, indexed, k) for k in BUDGET_MULTIPLIERS}
    episodes, new_models, new_adapters = [], set(), set()
    for policy_index, (name, family, seed, order) in enumerate(rankings):
        for multiplier in BUDGET_MULTIPLIERS:
            budget = len(classes) * multiplier
            selected = order[:budget]
            selected_set = set(selected)
            ids = [identifier for identifier in pool_ids if identifier in selected_set]
            if len(ids) != budget or len(selected_set) != budget:
                raise ValueError("Acquisition did not select its exact fixed label budget")
            counts = {label: sum(labels_after_ranking[identifier] == label for identifier in ids)
                      for label in classes}
            missing = [label for label, count in counts.items() if not count]
            reuse_k = next((k for k, old_ids in canonical.items() if ids == old_ids), None)
            objective = canonical_hash({"source_ids": job["source_ids"], "support_ids": ids,
                                        "class_weight": None, "C": 10.0,
                                        "numeric_gamma": float(source.estimator_._gamma)})
            if reuse_k is None:
                new_models.add(objective)
                if not missing:
                    new_adapters.add(objective)
            episodes.append({
                "policy_index": policy_index, "policy": name, "policy_family": family,
                "policy_seed": seed, "budget_multiplier": multiplier,
                "nominal_shots_per_class": multiplier, "budget": budget,
                "total_pool_size": len(pool_ids), "draw": policy_index * 1000 + multiplier,
                "selected_ids": selected, "adaptation_ids": ids,
                "support_counts": counts, "observed_support_classes": [label for label in classes if counts[label]],
                "observed_support_class_count": len(classes) - len(missing),
                "missing_support_classes": missing, "all_support_classes_present": not missing,
                "guard_fallback_identity": bool(missing),
                "reuse_k": reuse_k, "objective_key": objective,
            })
    if joblib.hash(source) != before:
        raise ValueError("Planning changed the frozen Source model")
    return {
        "schema_version": 1, "episodes": episodes,
        "policies": [{"name": name, "family": family, "seed": seed} for name, family, seed in POLICIES],
        "budget_multipliers": list(BUDGET_MULTIPLIERS), "candidate_pool_ids": pool_ids,
        "source_model_sha256": parent["source_hash"],
        "source_numeric_gamma": float(source.estimator_._gamma),
        "source_centroid_sha256": hashlib.sha256(np.ascontiguousarray(centroid).tobytes()).hexdigest(),
        "candidate_features_sha256": hashlib.sha256(np.ascontiguousarray(pool_scaled).tobytes()).hexdigest(),
        "candidate_scores_sha256": hashlib.sha256(np.ascontiguousarray(pool_scores).tobytes()).hexdigest(),
        "new_model_fits": len(new_models), "new_adapter_fits": len(new_adapters),
        "selection_uses_candidate_true_labels": False, "selection_uses_query": False,
        "labels_revealed_after_all_rankings": True,
        "canonical_training_order": "filter the original Support candidate ID order by the acquired set",
        "objective_reuse": "exact ordered acquired IDs; canonical K endpoints reused; no outcome-based reuse",
        "scope": "label-blind ranking conditional on a previously class-stratified candidate pool",
    }


def _historical_mean(job, k, query_ids, support_ids, support_labels, query_scores, round3root):
    from b_round5_common_20261111 import load_bundle, load_scores
    identifier = "G01_" + job["parent_job_id"]
    result, path = load_bundle(round3root, identifier)
    for field in ("source_ids", "support_ids", "query_ids", "classes", "label_column"):
        if result[field] != job[field]:
            raise ValueError("Historical mean Anchor uses different outer roles")
    scores, predicted, endpoint, provenance = load_scores(
        round3root, identifier, "mean_penalized", k, 0, query_ids)
    if endpoint["support_ids"] != support_ids:
        raise ValueError("Historical mean Anchor has a different selected Support set")
    parameters = json.loads((path / "adapter_parameters.json").read_text())
    row = next((item for item in parameters if item["method"] == "mean_penalized"
                and int(item["k"]) == k and int(item["draw"]) == 0), None)
    if row is None or row["support_ids"] != support_ids or row["fit_labels"] != support_labels.tolist():
        raise ValueError("Historical mean Anchor parameters/labels do not match acquisition")
    adapter = row["adapter"]
    if float(adapter["effective_lambda"]) != len(support_ids):
        raise ValueError("Historical mean Anchor has a different unit mean-loss prior")
    reconstructed = query_scores @ np.asarray(adapter["coef"]) + adapter["intercept"]
    if not np.allclose(scores, reconstructed, atol=1e-12, rtol=0):
        raise ValueError("Historical mean Anchor no longer reconstructs saved scores")
    if not np.array_equal(np.asarray(job["classes"])[scores.argmax(axis=1)], predicted):
        raise ValueError("Historical mean Anchor has a different score decision rule")
    return scores, predicted, adapter, provenance


def execute(job, X, metadata, axis, output, input_hash, oldroot, round2root, round3root, round4root):
    from b_round5_common_20261111 import load_frozen_refit

    writer = TaskWriter(job, X, metadata, axis, output, input_hash, oldroot)
    plan = acquisition_plan(job, X, metadata, axis, oldroot, round2root, round3root, round4root)
    if job.get("acquisition_plan") != plan:
        raise ValueError("Frozen label-blind acquisition identities or eligibility changed")
    old_result, source, parent_path = load_parent(job, oldroot)
    source_before = joblib.hash(source)
    query_scores = source.predict_scores(writer.query_X)
    source_y = writer.labels(job["source_ids"])
    source_X = writer.take(job["source_ids"])
    writer.record("identity", 0, query_scores, support_ids=[],
                  support_labels_used=0, new_model_fit=False, new_adapter_fit=False)
    writer.record("svc_predict", 0, query_scores, pred=native_predict(source, writer.query_X),
                  support_ids=[], decision_rule="SVC.predict", support_labels_used=0,
                  new_model_fit=False, new_adapter_fit=False)
    model_cache, adapter_cache = {}, {}
    parameters, provenance, coverage_rows = [], [], []
    new_models = new_adapters = 0
    for episode in plan["episodes"]:
        ids, objective = episode["adaptation_ids"], episode["objective_key"]
        support_y = writer.labels(ids)
        actual_counts = {label: int(np.count_nonzero(support_y == label)) for label in job["classes"]}
        if actual_counts != episode["support_counts"]:
            raise ValueError("Observed Support coverage differs from its frozen post-ranking audit")
        observed = len(set(support_y.tolist()))
        reuse_k = episode["reuse_k"]
        is_new_model = objective not in model_cache and reuse_k is None
        is_new_adapter = (objective not in adapter_cache and reuse_k is None
                          and episode["all_support_classes_present"])
        model_cpu = model_wall = adapter_cpu = adapter_wall = 0.0
        if objective not in model_cache:
            started_cpu, started_wall = time.process_time(), time.monotonic()
            if reuse_k is not None:
                model, reused = load_frozen_refit(job, int(reuse_k), oldroot, round2root)
                if not np.array_equal(model.estimator_.class_weight_, np.ones(len(job["classes"]))):
                    raise ValueError("Historical balanced refit is not equivalent to fixed unit class weights")
                provenance.append({"method": "full_refit", "objective_key": objective,
                                   "draw": episode["draw"], **reused})
            else:
                fit_ids = list(job["source_ids"]) + ids
                model = fit_source_fixedcoords(writer.take(fit_ids), np.concatenate([source_y, support_y]),
                                                source, C=10.0, class_weight=None)
                writer.save_model("acquisition_" + objective[:24] + ".joblib", model)
                new_models += 1
            if (float(model.estimator_.C) != 10.0
                    or float(model.estimator_._gamma) != float(source.estimator_._gamma)
                    or joblib.hash(model.scaler_) != joblib.hash(source.scaler_)
                    or not np.array_equal(model.estimator_.class_weight_, np.ones(len(job["classes"])))):
                raise ValueError("Acquisition refit changed fixed Source coordinates, C or unit weights")
            model_cache[objective] = (model.predict_scores(writer.query_X), native_predict(model, writer.query_X), episode["draw"])
            model_cpu, model_wall = time.process_time() - started_cpu, time.monotonic() - started_wall
        if objective not in adapter_cache:
            started_cpu, started_wall = time.process_time(), time.monotonic()
            if not episode["all_support_classes_present"]:
                score, prediction, adapter = query_scores.copy(), writer.classes[query_scores.argmax(axis=1)], None
            elif reuse_k is not None:
                score, prediction, adapter, reused = _historical_mean(
                    job, int(reuse_k), job["query_ids"], ids, support_y, query_scores, round3root)
                provenance.append({"method": "guarded_mean_anchor", "objective_key": objective,
                                   "draw": episode["draw"], **reused})
            else:
                support_scores = source.predict_scores(writer.take(ids))
                adapter = fit_adapter(support_scores, support_y, job["classes"], loss="mean", penalize_intercept=True)
                score = query_scores @ np.asarray(adapter["coef"]) + adapter["intercept"]
                prediction = writer.classes[score.argmax(axis=1)]
                new_adapters += 1
            adapter_cache[objective] = (score, prediction, adapter, episode["draw"])
            adapter_cpu, adapter_wall = time.process_time() - started_cpu, time.monotonic() - started_wall
            if adapter is not None:
                parameters.append({"objective_key": objective, "draw": episode["draw"],
                                   "support_ids": ids, "fit_labels": support_y.tolist(),
                                   "adapter": adapter, "adapter_reused": reuse_k is not None})
        common = {key: episode[key] for key in ("policy", "policy_family", "policy_seed", "policy_index",
                  "budget", "nominal_shots_per_class", "total_pool_size", "objective_key",
                  "observed_support_class_count", "all_support_classes_present")}
        common.update(support_counts_json=json.dumps(actual_counts, ensure_ascii=False),
                      missing_support_classes_json=json.dumps(episode["missing_support_classes"], ensure_ascii=False),
                      selected_ids_json=json.dumps(episode["selected_ids"], ensure_ascii=False),
                      acquisition_order_label_blind=True, candidate_pool_previously_stratified=True,
                      k_per_class_not_applicable=True, budget_labels_observed=len(ids),
                      reuse_k=reuse_k, class_weight_setting="fixed_unit_weights")
        full_score, full_pred, origin_draw = model_cache[objective]
        writer.record("full_refit", -1, full_score, pred=full_pred, draw=episode["draw"],
                      support_ids=ids, decision_rule="SVC.predict", **common,
                      support_labels_used=len(ids), labels_used_in_classifier_fit=len(ids), labels_used_in_adapter_fit=0,
                      classifier_source_label_count=len(job["source_ids"]),
                      labeled_training_n=len(job["source_ids"]) + len(ids),
                      statistics_training_n=len(job["source_ids"]),
                      labels_observed_for_guard=0, guard_fallback_identity=False, fallback_reason="",
                      new_model_fit=bool(is_new_model), new_adapter_fit=False, reused_model=not is_new_model,
                      reused_adapter=False, reused_from_draw=origin_draw,
                      fit_cpu_seconds=model_cpu if is_new_model else 0.0,
                      fit_wall_seconds=model_wall if is_new_model else 0.0,
                      effective_gamma=float(source.estimator_._gamma), C=10.0)
        adapter_score, adapter_pred, adapter, origin_draw = adapter_cache[objective]
        fallback = not episode["all_support_classes_present"]
        writer.record("guarded_mean_anchor", -1, adapter_score, pred=adapter_pred,
                      draw=episode["draw"], support_ids=ids, **common,
                      support_labels_used=0 if fallback else len(ids), labels_used_in_classifier_fit=0,
                      classifier_source_label_count=0, labeled_training_n=0,
                      statistics_training_n=len(job["source_ids"]),
                      labels_used_in_adapter_fit=0 if fallback else len(ids), labels_observed_for_guard=len(ids),
                      guard_fallback_identity=fallback,
                      fallback_reason="FALLBACK_IDENTITY_MISSING_SUPPORT_CLASSES" if fallback else "",
                      new_model_fit=False, new_adapter_fit=bool(is_new_adapter), reused_model=False,
                      reused_adapter=bool(not is_new_adapter and not fallback), reused_from_draw=origin_draw,
                      effective_lambda=len(ids) if adapter is not None else None,
                      fit_cpu_seconds=adapter_cpu if is_new_adapter else 0.0,
                      fit_wall_seconds=adapter_wall if is_new_adapter else 0.0)
        coverage_rows.append({**common, "draw": episode["draw"], "guard_fallback_identity": fallback,
                              "selected_ids": json.dumps(episode["selected_ids"], ensure_ascii=False),
                              "adaptation_ids": json.dumps(ids, ensure_ascii=False)})
    if new_models != fit_count(job) or new_adapters != adapter_count(job):
        raise ValueError("Acquisition fit/reuse counts differ from the frozen unique-objective plan")
    if joblib.hash(source) != source_before:
        raise ValueError("Acquisition changed the frozen Source model")
    writer.save_json("acquisition_plan.json", plan)
    writer.save_json("adapter_parameters.json", parameters)
    writer.save_json("reused_model_provenance.json", provenance)
    writer.save_json("training_id_locks.json", {"source_ids": job["source_ids"],
                     "canonical_pool_order": job["support_ids"],
                     "episodes": [{"draw": item["draw"], "selected_ids": item["selected_ids"],
                                    "adaptation_ids": item["adaptation_ids"]} for item in plan["episodes"]],
                     "statistics_fit_ids": job["source_ids"], "class_weight_setting": None})
    pd.DataFrame(coverage_rows).to_csv(writer.temp / "acquisition_coverage.csv", index=False)
    return writer.finish(expected_endpoints(job), new_model_fits=new_models,
                         new_adapter_fits=new_adapters, source_model_unchanged=True,
                         unique_acquired_objectives=len(model_cache),
                         identity_fallback_episodes=sum(item["guard_fallback_identity"] for item in plan["episodes"]),
                         original_source_sha256=old_result["source_hash"],
                         acquisition_label_blind=True,
                         interpretation="Label-blind ranking inside a previously stratified fixed pool; policy also changes realized class composition")
