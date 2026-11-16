"""I01: select an adaptation policy in honest holdouts inside outer Source.

No outer Support label or Query value is used to choose the policy. Inner
preprocessing is fitted from the inner Source, not inherited from the outer
source model. This is a small, frozen source-domain policy, not a search over
outer-query performance or an unbiased claim about its best inner score.
"""
from __future__ import annotations

import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd

from herb_spectra.spectral_models import fit_source_candidate
from b_followup_common_20261110 import TaskWriter, endpoint_id, load_parent, metrics
from b_followup_models_20261110 import native_predict, nested_support
from b_round4_models_20261111 import fit_source_fixedcoords
from b_mechanism_engine_20261110 import CANDIDATE, calibrate_scores
from b_round3_regularization_20261110 import fit_adapter


EPISODE_SEEDS = (501, 502, 503)
K_VALUES = (1, 3, 5, 10)
CANDIDATE_METHODS = ("identity", "bias", "mean_anchor", "full_refit")
OUTER_METHODS = ("bias", "mean_anchor", "full_refit", "source_selected")


def expected_endpoints(job=None):
    return [endpoint_id("identity"), endpoint_id("svc_predict")] + [
        endpoint_id(method, k) for k in K_VALUES for method in OUTER_METHODS]


def fit_count(job=None):
    return len(EPISODE_SEEDS) * (1 + len(K_VALUES))


def adapter_count(job=None):
    return len(EPISODE_SEEDS) * len(K_VALUES) * 2


def inner_plan(job, metadata):
    """Lock disjoint pseudo-roles using only the original Source IDs/labels."""
    frame = metadata.set_index("record_id", drop=False)
    if not frame.index.is_unique or not metadata.physical_sample_id.is_unique:
        raise ValueError("Record and physical sample IDs must be unique")
    source_ids = list(job["source_ids"])
    if len(source_ids) != len(set(source_ids)) or not set(source_ids).issubset(frame.index):
        raise ValueError("Outer Source IDs must be unique and known")
    if not frame.loc[source_ids, "role"].eq("train").all():
        raise ValueError("Only master TRAIN may form source-policy episodes")
    if set(source_ids) & (set(job["support_ids"]) | set(job["query_ids"])):
        raise ValueError("Outer Source overlaps outer Support or Query")
    n_classes = len(job["classes"])
    if n_classes == 5:
        required, source_n, validation_n = 38, 20, 8
    elif n_classes == 2:
        required, source_n, validation_n = 120, 90, 20
    else:
        raise ValueError("The frozen selector supports binary and five-origin contexts")
    class_members = {}
    for label in job["classes"]:
        members = sorted(sample for sample in source_ids if frame.at[sample, job["label_column"]] == label)
        if len(members) != required:
            raise ValueError("Source-only selector has an unexpected per-class Source count")
        class_members[label] = members
    if sum(map(len, class_members.values())) != len(source_ids):
        raise ValueError("Outer Source includes an unexpected class")
    episodes = []
    parent = job.get("parent_job_id") or job["job_id"]
    for seed in EPISODE_SEEDS:
        parts = {"inner_source_ids": [], "inner_support_ids": [], "inner_validation_ids": []}
        for label in job["classes"]:
            raw = hashlib.sha256(f"round5-selector:{parent}:{seed}:{label}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(raw[:8], "big"))
            order = rng.permutation(class_members[label]).tolist()
            parts["inner_source_ids"].extend(order[:source_n])
            parts["inner_support_ids"].extend(order[source_n:source_n + 10])
            parts["inner_validation_ids"].extend(order[source_n + 10:])
        role_sets = [set(parts[name]) for name in parts]
        if (any(role_sets[i] & role_sets[j] for i in range(3) for j in range(i + 1, 3))
                or set.union(*role_sets) != set(source_ids)):
            raise ValueError("Inner roles must partition the outer Source exactly")
        if len(parts["inner_validation_ids"]) != validation_n * n_classes:
            raise ValueError("Unexpected source-only validation budget")
        episodes.append({"episode_seed": seed, **parts,
                         "source_counts": {label: source_n for label in job["classes"]},
                         "support_counts": {label: 10 for label in job["classes"]},
                         "validation_counts": {label: validation_n for label in job["classes"]}})
    return {"schema_version": 1, "episode_seeds": list(EPISODE_SEEDS), "episodes": episodes,
            "all_roles_within_outer_source": True,
            "outer_support_query_excluded": True,
            "inner_source_samples": source_n * n_classes,
            "inner_validation_samples_per_episode": validation_n * n_classes,
            "transport_limitation": "Inner training sets are smaller and remain Source-domain; policy transfer to the outer Support/Query domain is not guaranteed."}


def nested_inner_support(pool_ids, indexed, label_column, classes, k):
    if k not in K_VALUES:
        raise ValueError("K lies outside the frozen inner episode")
    selected = []
    for label in classes:
        members = [sample for sample in pool_ids if indexed.at[sample, label_column] == label]
        if len(members) != 10:
            raise ValueError("Each inner Support class must contain ten candidates")
        selected.extend(members[:k])
    return selected


def select_policy(inner_metrics):
    """Choose only from Source-validation metrics with a fixed simple tie rule."""
    rows = pd.DataFrame(inner_metrics).copy()
    required = {"episode_seed", "k", "method", "macro_f1"}
    if rows.empty or not required.issubset(rows.columns):
        raise ValueError("Source-validation metrics are incomplete")
    if (not set(rows["method"]).issubset(CANDIDATE_METHODS)
            or not set(rows["k"]).issubset(K_VALUES)
            or not np.isfinite(rows["macro_f1"].astype(float)).all()
            or rows.duplicated(["episode_seed", "k", "method"]).any()):
        raise ValueError("Invalid source-validation candidate metrics")
    decisions = {}
    for k, block in rows.groupby("k", sort=True):
        if set(block.method) != set(CANDIDATE_METHODS):
            raise ValueError("All four frozen candidates must be compared at a budget")
        episode_sets = [set(block.loc[block.method.eq(method), "episode_seed"]) for method in CANDIDATE_METHODS]
        if not episode_sets[0] or any(values != episode_sets[0] for values in episode_sets[1:]):
            raise ValueError("Source-validation candidate episodes do not match")
        means = {method: float(block.loc[block.method.eq(method), "macro_f1"].mean())
                 for method in CANDIDATE_METHODS}
        rounded = {method: round(value, 12) for method, value in means.items()}
        selected = max(CANDIDATE_METHODS, key=lambda method: (rounded[method], -CANDIDATE_METHODS.index(method)))
        decisions[str(int(k))] = {
            "selected_method": selected,
            "mean_validation_macro_f1_by_method": means,
            "rounded_selection_scores": rounded,
            "selection_score": means[selected],
            "tie_priority": list(CANDIDATE_METHODS),
            "episode_seeds": sorted(episode_sets[0]),
            "selection_scope": "outer_Source_only_inner_holdouts",
            "not_an_unbiased_best_inner_score": True,
        }
    return decisions


def _score_prediction(model, X):
    scores = model.predict_scores(X)
    return scores, model.classes_[scores.argmax(axis=1)]


def _outer_scores(root, parent_job, method, k, query_ids):
    from b_round5_common_20261111 import load_scores
    return load_scores(root, parent_job, method, k, 0, query_ids)


def execute(job, X, metadata, axis, output, input_hash, oldroot, round2root, round3root, round4root):
    """Fit the source-only policy, freeze its choices, then load outer outcomes."""
    from b_round5_common_20261111 import load_frozen_refit

    writer = TaskWriter(job, X, metadata, axis, output, input_hash, oldroot)
    plan = inner_plan(job, metadata)
    if job.get("inner_plan") is not None and job["inner_plan"] != plan:
        raise ValueError("Frozen source-only inner role plan changed")
    validation_rows, validation_metrics = [], []
    adapter_records, fit_locks, model_costs = [], [], []
    new_models = new_adapters = 0
    classes = job["classes"]

    for episode in plan["episodes"]:
        seed = episode["episode_seed"]
        source_ids = episode["inner_source_ids"]
        validation_ids = episode["inner_validation_ids"]
        pool_ids = episode["inner_support_ids"]
        started_cpu, started_wall = time.process_time(), time.monotonic()
        inner_source = fit_source_candidate(CANDIDATE, writer.take(source_ids), writer.labels(source_ids), axis, seed=42)
        new_models += 1
        writer.save_model(f"inner_source_s{seed}.joblib", inner_source)
        model_costs.append({"episode_seed": seed, "kind": "inner_source", "k": 0,
                           "cpu_seconds": time.process_time() - started_cpu,
                           "wall_seconds": time.monotonic() - started_wall})
        if inner_source.classes_.tolist() != classes:
            raise ValueError("Inner source class vocabulary changed")
        before = joblib.hash(inner_source)
        validation_X = writer.take(validation_ids)
        base_scores = inner_source.predict_scores(validation_X)
        pool_scores = inner_source.predict_scores(writer.take(pool_ids))
        support_lookup = dict(zip(pool_ids, pool_scores))
        for k in K_VALUES:
            support_ids = nested_inner_support(pool_ids, writer.indexed, job["label_column"], classes, k)
            y_support = writer.labels(support_ids)
            support_scores = np.asarray([support_lookup[sample] for sample in support_ids])
            score_bias, bias = calibrate_scores("bias_only", support_scores, y_support, base_scores, classes)
            mean = fit_adapter(support_scores, y_support, classes, loss="mean", penalize_intercept=True)
            score_mean = base_scores @ np.asarray(mean["coef"]) + mean["intercept"]
            new_adapters += 2
            adapter_records.extend([{"episode_seed": seed, "k": k, "method": "bias", "support_ids": support_ids, "adapter": bias},
                                    {"episode_seed": seed, "k": k, "method": "mean_anchor", "support_ids": support_ids, "adapter": mean}])
            fit_ids = source_ids + support_ids
            started_cpu, started_wall = time.process_time(), time.monotonic()
            full = fit_source_fixedcoords(writer.take(fit_ids), writer.labels(fit_ids), inner_source, C=10.0, class_weight="balanced")
            new_models += 1
            writer.save_model(f"inner_full_s{seed}_k{k}.joblib", full)
            model_costs.append({"episode_seed": seed, "kind": "inner_full_refit", "k": k,
                               "cpu_seconds": time.process_time() - started_cpu,
                               "wall_seconds": time.monotonic() - started_wall})
            score_full = full.predict_scores(validation_X)
            candidates = {
                "identity": (base_scores, np.asarray(classes)[base_scores.argmax(axis=1)], "score_argmax"),
                "bias": (score_bias, np.asarray(classes)[score_bias.argmax(axis=1)], "score_argmax"),
                "mean_anchor": (score_mean, np.asarray(classes)[score_mean.argmax(axis=1)], "score_argmax"),
                "full_refit": (score_full, native_predict(full, validation_X), "SVC.predict"),
            }
            # Only inner validation labels join here; no outer Query is used.
            validation_truth = writer.labels(validation_ids)
            for method, (scores, predicted, rule) in candidates.items():
                values = metrics(validation_truth, predicted, classes)
                validation_metrics.append({"episode_seed": seed, "k": k, "method": method,
                                           "decision_rule": rule, **values})
                for index, record_id in enumerate(validation_ids):
                    validation_rows.append({"episode_seed": seed, "k": k, "method": method,
                                            "decision_rule": rule, "record_id": record_id,
                                            "physical_sample_id": writer.indexed.at[record_id, "physical_sample_id"],
                                            "origin_group_id": writer.indexed.at[record_id, "origin_group_id"],
                                            "true_label": validation_truth[index], "pred_label": predicted[index],
                                            "scores_json": json.dumps(scores[index].tolist())})
            fit_locks.append({"episode_seed": seed, "k": k, "fresh_transform_fit_ids": source_ids,
                              "inner_source_classifier_label_ids": source_ids,
                              "inner_full_classifier_label_ids": fit_ids,
                              "calibrator_label_ids": support_ids, "validation_only_ids": validation_ids})
        if joblib.hash(inner_source) != before:
            raise ValueError("Adaptation changed the inner frozen source model")

    decisions = select_policy(validation_metrics)
    if set(decisions) != {str(k) for k in K_VALUES}:
        raise ValueError("Source-only policy choices are incomplete")
    writer.save_json("inner_plan.json", plan)
    writer.save_json("source_policy.json", decisions)
    writer.save_json("inner_training_id_locks.json", fit_locks)
    writer.save_json("inner_adapter_parameters.json", adapter_records)
    writer.save_json("inner_model_costs.json", model_costs)
    pd.DataFrame(validation_metrics).to_csv(writer.temp / "inner_validation_metrics.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(writer.temp / "inner_validation_predictions.csv.gz", index=False,
                                        compression={"method": "gzip", "mtime": 0})

    # Policy choices are complete before even loading the historical outer
    # outcomes, or evaluating the outer frozen model on its Query.
    _, source, parent_path = load_parent(job, oldroot)
    source_before = joblib.hash(source)
    source_scores = source.predict_scores(writer.query_X)
    source_argmax = writer.classes[source_scores.argmax(axis=1)]
    source_native = native_predict(source, writer.query_X)
    writer.record("identity", 0, source_scores, pred=source_argmax, support_ids=[],
                  support_labels_used=0, new_model_fit=False, new_adapter_fit=False)
    writer.record("svc_predict", 0, source_scores, pred=source_native, support_ids=[],
                  decision_rule="SVC.predict", support_labels_used=0, new_model_fit=False, new_adapter_fit=False)
    provenance, selected_records = [], []
    for k in K_VALUES:
        ids = nested_support(job, writer.indexed, k)
        bias_scores, bias_pred, bias_endpoint, bias_provenance = _outer_scores(
            round2root, "F01_" + job["parent_job_id"], "raw_bias_only", k, job["query_ids"])
        mean_scores, mean_pred, mean_endpoint, mean_provenance = _outer_scores(
            round3root, "G01_" + job["parent_job_id"], "mean_penalized", k, job["query_ids"])
        if bias_endpoint["support_ids"] != ids or mean_endpoint["support_ids"] != ids:
            raise ValueError("Reused outer adapter has a different canonical Support sequence")
        full, full_provenance = load_frozen_refit(job, k, oldroot, round2root)
        if (joblib.hash(full.scaler_) != joblib.hash(source.scaler_)
                or float(full.estimator_._gamma) != float(source.estimator_._gamma)):
            raise ValueError("Reused outer full refit changed Source coordinates")
        full_scores = full.predict_scores(writer.query_X)
        outer_candidates = {
            "identity": (source_scores, source_argmax, "score_argmax"),
            "bias": (np.asarray(bias_scores), np.asarray(bias_pred), "score_argmax"),
            "mean_anchor": (np.asarray(mean_scores), np.asarray(mean_pred), "score_argmax"),
            "full_refit": (full_scores, native_predict(full, writer.query_X), "SVC.predict"),
        }
        for method in ("bias", "mean_anchor", "full_refit"):
            scores, predicted, rule = outer_candidates[method]
            writer.record(method, k, scores, pred=predicted, support_ids=ids,
                          decision_rule=rule, reused_model=method == "full_refit",
                          reused_adapter=method != "full_refit", support_labels_used=len(ids),
                          allocated_support_labels=len(ids), new_model_fit=False, new_adapter_fit=False,
                          policy_scope="fixed_historical_candidate")
        choice = decisions[str(k)]["selected_method"]
        scores, predicted, rule = outer_candidates[choice]
        observed_ids = [] if choice == "identity" else ids
        writer.record("source_selected", k, scores, pred=predicted, support_ids=observed_ids,
                      decision_rule=rule, selected_method=choice, chosen_decision_rule=rule,
                      support_labels_used=len(observed_ids), allocated_support_labels=len(ids),
                      selection_score=decisions[str(k)]["selection_score"],
                      policy_scope="outer_Source_only_inner_holdouts",
                      inner_source_samples=plan["inner_source_samples"],
                      inner_validation_samples=plan["inner_validation_samples_per_episode"],
                      new_model_fit=False, new_adapter_fit=False, candidate_prediction_reused=True,
                      selection_before_outer_prediction_load=True)
        provenance.extend([{"k": k, "method": "bias", **bias_provenance},
                           {"k": k, "method": "mean_anchor", **mean_provenance},
                           {"k": k, "method": "full_refit", **full_provenance}])
        selected_records.append({"k": k, **decisions[str(k)], "allocated_support_ids": ids,
                                 "used_support_ids": observed_ids, "decision_rule": rule})
    if joblib.hash(source) != source_before:
        raise ValueError("Policy application changed the outer frozen source")
    if new_models != fit_count(job) or new_adapters != adapter_count(job):
        raise ValueError("Source-only policy fitting counts differ from the frozen plan")
    writer.save_json("outer_reuse_provenance.json", provenance)
    writer.save_json("outer_selection_application.json", selected_records)
    writer.save_json("source_reuse.json", {"parent_job_id": job["parent_job_id"],
                                          "parent_result_path": str(parent_path / "result.json"),
                                          "source_model_unchanged": True})
    return writer.finish(expected_endpoints(job), new_model_fits=new_models,
                         new_adapter_fits=new_adapters, source_model_unchanged=True,
                         policy_selection_scope="outer_Source_only_inner_holdouts",
                         decisions_frozen_before_outer_prediction_loads=True,
                         inner_validation_prediction_rows=len(validation_rows),
                         inner_validation_metric_rows=len(validation_metrics),
                         selection_tie_priority=list(CANDIDATE_METHODS),
                         interpretation="Source-only policy with fresh inner preprocessing; smaller inner training and Source-domain validation limit transfer, and best inner score is not unbiased performance")
