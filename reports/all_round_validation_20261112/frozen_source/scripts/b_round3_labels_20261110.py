"""G02/G03: unlabeled statistics and explicitly copied support-label controls.

No source labels or metadata are corrupted. The dose controls are artificial,
class-balanced cyclic perturbations, not a claim about current label quality.
All training interfaces accept only the named labeled/statistics arrays; query
labels join at the shared writer after a prediction has already been produced.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from herb_spectra.spectral_models import FittedSpectralSource
from b_mechanism_engine_20261110 import calibrate_scores
from b_followup_common_20261110 import (
    TaskWriter, canonical_hash, endpoint_id, file_hash, load_parent, verify_task,
)
from b_followup_models_20261110 import diagnostics, fit_model, native_predict, nested_support


K_VALUES = (1, 3, 5, 10)
NOISE_Q = (1, 2, 4)
NOISE_SEEDS = (201, 202, 203, 204, 205)
G03_METHODS = ("raw_anchor_lambda1", "weak_lambda1000_blend005", "full_pipeline_fixedgamma")


def cyclic_labels(labels, classes, changed_ids=None, ids=None):
    """Return a copied label array with an explicit next-class mapping.

    If a mask is supplied it must change the same number in every class, so the
    complete support label histogram is unchanged. Inputs are never mutated.
    """
    classes = list(classes)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("Cyclic diagnostic needs a unique multiclass vocabulary")
    original = np.asarray(labels, dtype=str)
    if not set(original).issubset(classes):
        raise ValueError("Support label outside frozen class vocabulary")
    copied = original.copy()
    mapping = {label: classes[(index + 1) % len(classes)] for index, label in enumerate(classes)}
    if changed_ids is None:
        mask = np.ones(len(original), dtype=bool)
    else:
        if ids is None or len(ids) != len(original) or len(ids) != len(set(ids)):
            raise ValueError("Diagnostic mask requires aligned unique support IDs")
        changed_ids = set(changed_ids)
        if not changed_ids.issubset(ids):
            raise ValueError("Diagnostic mask is outside support pool")
        mask = np.asarray([record in changed_ids for record in ids])
        altered_counts = [int(np.count_nonzero(mask & (original == label))) for label in classes]
        if len(set(altered_counts)) != 1:
            raise ValueError("Cyclic diagnostic must change equal counts per class")
    copied[mask] = [mapping[label] for label in original[mask]]
    if sorted(copied.tolist()) != sorted(original.tolist()):
        raise ValueError("Diagnostic changed the support class histogram")
    if int(np.count_nonzero(copied != original)) != int(mask.sum()):
        raise ValueError("Frozen cyclic mask did not change all requested labels")
    return copied


def noise_plan(job, metadata):
    """Freeze realizable masks and exact-objective reuse without reading spectra."""
    indexed = metadata.set_index("record_id", drop=False)
    support_ids = nested_support(job, indexed, 10)
    labels = indexed.loc[support_ids, job["label_column"]].astype(str).to_numpy()
    classes = list(job["classes"])
    episodes = []
    seen = {}
    for q in (*NOISE_Q, 10):
        for seed in (NOISE_SEEDS if q != 10 else NOISE_SEEDS[:1]):
            selected = []
            for class_index, label in enumerate(classes):
                members = [sample for sample, truth in zip(support_ids, labels) if truth == label]
                if len(members) != 10:
                    raise ValueError("G03 requires exactly ten support candidates per class")
                # A fixed per-class permutation is shared by every dose at a
                # seed; q=1/2/4 are nested masks, not different error samples.
                rng = np.random.default_rng(np.random.SeedSequence([20261110, seed, class_index, 731]))
                selected.extend(np.asarray(members)[rng.permutation(10)[:q]].tolist())
            copied = cyclic_labels(labels, classes, selected, support_ids)
            key = canonical_hash({"support_ids": support_ids, "copied_support_labels": copied.tolist()})
            encoded_draw = q * 1000 + seed
            reused_from = seen.get(key)
            episode = {
                "noise_q_per_class": q, "noise_seed": seed, "draw": encoded_draw,
                "actual_noise_rate": q / 10, "support_ids": list(support_ids),
                "changed_support_ids": selected, "copied_support_labels": copied.tolist(),
                "objective_key": key, "reused_from_draw": reused_from,
                "diagnostic_labels_changed_count": q * len(classes),
            }
            episodes.append(episode)
            seen.setdefault(key, encoded_draw)
    unique_partial = {item["objective_key"] for item in episodes if item["noise_q_per_class"] != 10}
    return {
        "schema_version": 1, "episodes": episodes,
        "support_ids": support_ids, "true_support_labels": labels.tolist(),
        "class_mapping": {label: classes[(index + 1) % len(classes)] for index, label in enumerate(classes)},
        "new_model_fits": len(unique_partial), "new_adapter_fits": 2 * len(seen),
        "dose_masks_nested_within_seed": True,
        "objective_deduplication": "exact ordered support IDs and copied label vector, never Query outcomes",
    }


def fit_count(job):
    if job["family"] == "G02":
        return 8
    if job["family"] == "G03":
        return int(job["noise_plan"]["new_model_fits"])
    raise ValueError("Unknown label-mechanism family")


def adapter_count(job):
    if job["family"] == "G02":
        return 0
    if job["family"] == "G03":
        return int(job["noise_plan"]["new_adapter_fits"])
    raise ValueError("Unknown label-mechanism family")


def expected_endpoints(job):
    if job["family"] == "G02":
        return [endpoint_id("identity")] + [
            endpoint_id(method, k) for k in K_VALUES
            for method in ("unlabeled_statistics", "true_support", "cyclic_support")
        ]
    if job["family"] == "G03":
        return [endpoint_id("identity"), endpoint_id("svc_predict")] + [endpoint_id(method, 10) for method in G03_METHODS] + [
            endpoint_id(method, 10, q * 1000 + seed)
            for q in (*NOISE_Q, 10)
            for seed in (NOISE_SEEDS if q != 10 else NOISE_SEEDS[:1])
            for method in G03_METHODS
        ]
    raise ValueError("Unknown label-mechanism family")


def fit_with_statistics(X_labeled, y_labeled, X_statistics, axis, reference):
    """Fit Source-only labels with Source+unlabeled-Support spectral statistics."""
    X_labeled = np.asarray(X_labeled, dtype=float)
    X_statistics = np.asarray(X_statistics, dtype=float)
    labels = np.asarray(y_labeled, dtype=str)
    if (X_labeled.ndim != 2 or X_statistics.ndim != 2
            or X_labeled.shape[1] != X_statistics.shape[1]
            or len(X_labeled) != len(labels) or not np.isfinite(X_labeled).all()
            or not np.isfinite(X_statistics).all()
            or X_labeled.shape[1] != len(axis)):
        raise ValueError("Labeled/statistics feature matrices do not align")
    preprocessor = copy.deepcopy(reference.preprocessor_)
    scaler = StandardScaler().fit(preprocessor.transform(X_statistics))
    values = scaler.transform(preprocessor.transform(X_labeled))
    estimator = SVC(C=10.0, kernel="rbf", gamma=float(reference.estimator_._gamma),
                    class_weight="balanced", break_ties=True,
                    decision_function_shape="ovr", random_state=42).fit(values, labels)
    classes = np.unique(labels)
    if not np.array_equal(estimator.classes_, classes):
        raise ValueError("Source-only classifier class vocabulary changed")
    classes.setflags(write=False)
    candidate = {"id": "round3_source_labels_unlabeled_support_statistics", "preprocessing": "snv",
                 "model": "svc", "C": 10.0, "class_weight": "balanced",
                 "scaler_fit_scope": "source_plus_unlabeled_support", "gamma_mode": "fixed"}
    metadata = {"fit_scope": "source_labels_only", "source_samples": len(labels),
                "statistics_samples": len(X_statistics), "class_order": classes.tolist(),
                "effective_gamma": float(estimator._gamma), "support_labels_used": 0,
                "input_features": X_labeled.shape[1]}
    return FittedSpectralSource(candidate, preprocessor, scaler, estimator, classes, metadata)


def _record_source(writer, source, *, calibration_control=False):
    scores = source.predict_scores(writer.query_X)
    writer.record("identity", 0, scores,
                  pred=None if calibration_control else native_predict(source, writer.query_X), support_ids=[],
                  decision_rule="score_argmax" if calibration_control else "SVC.predict", reused_model=True,
                  support_labels_used=0, actual_noise_rate=0.0, noise_q_per_class=0,
                  support_noise_fraction=0.0, total_labeled_training_noise_fraction=0.0,
                  noise_seed=0, diagnostic_labels_changed_count=0,
                  artificial_label_copy=False, new_model_fit=False, new_adapter_fit=False)
    if calibration_control:
        writer.record("svc_predict", 0, scores,
                      pred=native_predict(source, writer.query_X), support_ids=[],
                      decision_rule="SVC.predict", reused_model=True,
                      support_labels_used=0, actual_noise_rate=0.0, noise_q_per_class=0,
                      support_noise_fraction=0.0, total_labeled_training_noise_fraction=0.0,
                      noise_seed=0, diagnostic_labels_changed_count=0,
                      artificial_label_copy=False, new_model_fit=False, new_adapter_fit=False)


def _true_refit(job, k, oldroot, round2root):
    # Kept local so metadata-only plan building does not require the root helper.
    from b_round3_common_20261110 import load_true_refit
    return load_true_refit(job, k, oldroot, round2root)


def _g02(writer, job, axis, oldroot, round2root):
    old_result, source, parent_path = load_parent(job, oldroot)
    initial_hash = joblib.hash(source)
    _record_source(writer, source)
    fit_locks, model_provenance, copies = {}, [], {}
    new_fits = 0
    for k in K_VALUES:
        support_ids = nested_support(job, writer.indexed, k)
        fit_ids = list(job["source_ids"]) + support_ids
        support_truth = writer.labels(support_ids)
        corrupted = cyclic_labels(support_truth, job["classes"])
        source_y = writer.labels(job["source_ids"])
        true_model, provenance = _true_refit(job, k, oldroot, round2root)
        model_provenance.append({"method": "true_support", "k": k, **provenance})
        copies[str(k)] = {"support_ids": support_ids, "true_labels": support_truth.tolist(),
                          "copied_labels": corrupted.tolist(),
                          "changed_count": len(support_ids), "actual_noise_rate": 1.0}
        fit_locks[str(k)] = {
            "scaler_statistics_ids": fit_ids, "unlabeled_statistics_labeled_ids": job["source_ids"],
            "true_support_labeled_ids": fit_ids, "cyclic_support_labeled_ids": fit_ids,
        }
        started_cpu, started_wall = time.process_time(), time.monotonic()
        unlabeled = fit_with_statistics(writer.take(job["source_ids"]), source_y,
                                       writer.take(fit_ids), axis, source)
        unlabeled_cost = (time.process_time() - started_cpu, time.monotonic() - started_wall)
        writer.save_model(f"unlabeled_statistics_k{k}.joblib", unlabeled)
        started_cpu, started_wall = time.process_time(), time.monotonic()
        noisy = fit_model(writer.take(fit_ids), np.concatenate([source_y, corrupted]), axis,
                          reference=source, scaler_mode="refit", gamma_mode="fixed",
                          class_weight="balanced")
        noisy_cost = (time.process_time() - started_cpu, time.monotonic() - started_wall)
        writer.save_model(f"cyclic_support_k{k}.joblib", noisy)
        new_fits += 2
        for method, model, reused in (("unlabeled_statistics", unlabeled, False),
                                      ("true_support", true_model, True),
                                      ("cyclic_support", noisy, False)):
            if model.classes_.tolist() != job["classes"]:
                raise ValueError("G02 class vocabulary changed")
            if float(model.estimator_._gamma) != float(source.estimator_._gamma):
                raise ValueError("G02 must keep numeric source gamma fixed")
            if joblib.hash(model.scaler_) != joblib.hash(true_model.scaler_):
                raise ValueError("G02 conditions require identical unlabeled statistics")
            writer.record(method, k, model.predict_scores(writer.query_X),
                          pred=native_predict(model, writer.query_X), support_ids=support_ids,
                          decision_rule="SVC.predict", reused_model=reused,
                          new_model_fit=not reused, new_adapter_fit=False,
                          support_labels_used=0 if method == "unlabeled_statistics" else len(support_ids),
                          labeled_training_n=len(job["source_ids"]) if method == "unlabeled_statistics" else len(fit_ids),
                          statistics_training_n=len(fit_ids),
                          actual_noise_rate=1.0 if method == "cyclic_support" else 0.0,
                          support_noise_fraction=1.0 if method == "cyclic_support" else 0.0,
                          total_labeled_training_noise_fraction=len(support_ids) / len(fit_ids) if method == "cyclic_support" else 0.0,
                          source_training_labels_corrupted=0,
                          noise_q_per_class=k if method == "cyclic_support" else 0,
                          noise_seed=0, artificial_label_copy=method == "cyclic_support",
                          diagnostic_labels_changed_count=len(support_ids) if method == "cyclic_support" else 0,
                          fit_cpu_seconds=unlabeled_cost[0] if method == "unlabeled_statistics" else noisy_cost[0] if method == "cyclic_support" else 0.0,
                          fit_wall_seconds=unlabeled_cost[1] if method == "unlabeled_statistics" else noisy_cost[1] if method == "cyclic_support" else 0.0,
                          **diagnostics(model, reference=source))
    if joblib.hash(source) != initial_hash:
        raise ValueError("G02 changed the frozen source")
    writer.save_json("training_id_locks.json", fit_locks)
    writer.save_json("diagnostic_label_copies.json", copies)
    writer.save_json("reused_model_provenance.json", model_provenance)
    return writer.finish(expected_endpoints(job), new_model_fits=new_fits,
                         new_adapter_fits=0, source_model_unchanged=True,
                         interpretation="Source-only labels plus coherent refitted statistics versus additional true/artificial labels")


def _old_scores(parent_path, method, query_ids):
    predictions = pd.read_csv(parent_path / "predictions.csv", keep_default_na=False)
    rows = predictions.loc[predictions.method.eq(method) & predictions.k.eq(10)].set_index("record_id").loc[query_ids]
    return np.asarray([json.loads(item) for item in rows.scores_json]), rows.pred_label.to_numpy()


def _g02_cyclic_model(job, output, input_hash):
    identifier = "G02_" + job["parent_job_id"]
    path = Path(output) / "jobs" / identifier
    result = verify_task(path, input_hash=input_hash)
    if result["family"] != "G02" or result["parent_job_id"] != job["parent_job_id"]:
        raise ValueError("G03 label control has the wrong G02 dependency")
    for name in ("source_ids", "support_ids", "query_ids", "classes", "label_column"):
        if result[name] != job[name]:
            raise ValueError("G02/G03 role definitions disagree")
    name = "cyclic_support_k10.joblib"
    if name not in result["artifact_hashes"] or file_hash(path / name) != result["artifact_hashes"][name]:
        raise ValueError("G02 cyclic control model is not sealed")
    return joblib.load(path / name), {"job_id": identifier, "artifact": name,
                                    "artifact_sha256": result["artifact_hashes"][name],
                                    "result_sha256": file_hash(path / "result.json")}


def _g03(writer, job, axis, oldroot, round2root):
    plan = noise_plan(job, writer.metadata)
    if job.get("noise_plan") != plan:
        raise ValueError("Frozen label-contamination plan changed")
    old_result, source, parent_path = load_parent(job, oldroot)
    initial_hash = joblib.hash(source)
    _record_source(writer, source, calibration_control=True)
    support_ids = plan["support_ids"]
    support_truth = writer.labels(support_ids)
    source_y = writer.labels(job["source_ids"])
    fit_ids = list(job["source_ids"]) + support_ids
    query_scores = source.predict_scores(writer.query_X)
    support_scores = source.predict_scores(writer.take(support_ids))
    true_model, true_provenance = _true_refit(job, 10, oldroot, round2root)
    true_key = canonical_hash({"support_ids": support_ids, "copied_support_labels": support_truth.tolist()})
    for method in G03_METHODS:
        if method == "full_pipeline_fixedgamma":
            score, pred = true_model.predict_scores(writer.query_X), native_predict(true_model, writer.query_X)
            rule = "SVC.predict"
        else:
            old_method = "anchor_lambda1" if method == "raw_anchor_lambda1" else method
            score, pred = _old_scores(parent_path, old_method, job["query_ids"])
            rule = "score_argmax"
        writer.record(method, 10, score, pred=pred, support_ids=support_ids,
                      decision_rule=rule, actual_noise_rate=0.0, noise_q_per_class=0,
                      support_noise_fraction=0.0, total_labeled_training_noise_fraction=0.0,
                      noise_seed=0, objective_key=true_key,
                      diagnostic_labels_changed_count=0, artificial_label_copy=False,
                      reused_model=method == "full_pipeline_fixedgamma",
                      reused_adapter=method != "full_pipeline_fixedgamma",
                      new_model_fit=False, new_adapter_fit=False, support_labels_used=len(support_ids))
    cyclic_model, cyclic_provenance = _g02_cyclic_model(job, writer.output, writer.input_hash)
    cache_models, cache_adapters, adapter_parameters = {}, {}, []
    reused_records = [{"method": "full_pipeline_fixedgamma", "noise_q_per_class": 0, **true_provenance},
                      {"method": "full_pipeline_fixedgamma", "noise_q_per_class": 10, **cyclic_provenance}]
    new_fits = new_adapters = 0
    for episode in plan["episodes"]:
        q, seed, draw = (episode[name] for name in ("noise_q_per_class", "noise_seed", "draw"))
        key = episode["objective_key"]
        copied = np.asarray(episode["copied_support_labels"])
        if not np.array_equal(copied, cyclic_labels(support_truth, job["classes"],
                                                   episode["changed_support_ids"], support_ids)):
            raise ValueError("Frozen diagnostic copied labels do not reproduce")
        if key not in cache_models:
            if q == 10:
                model, model_new = cyclic_model, False
                model_cost = (0.0, 0.0)
            else:
                started_cpu, started_wall = time.process_time(), time.monotonic()
                model = fit_model(writer.take(fit_ids), np.concatenate([source_y, copied]), axis,
                                  reference=source, scaler_mode="refit", gamma_mode="fixed",
                                  class_weight="balanced")
                model_cost = (time.process_time() - started_cpu, time.monotonic() - started_wall)
                writer.save_model(f"noise_{key[:16]}.joblib", model)
                new_fits += 1
                model_new = True
            if (float(model.estimator_._gamma) != float(source.estimator_._gamma)
                    or joblib.hash(model.scaler_) != joblib.hash(true_model.scaler_)):
                raise ValueError("Label-dose controls changed fixed gamma or spectral statistics")
            cache_models[key] = (model.predict_scores(writer.query_X), native_predict(model, writer.query_X), model_cost)
        else:
            model_new = False
        for method in G03_METHODS:
            started, started_wall = time.process_time(), time.monotonic()
            if method == "full_pipeline_fixedgamma":
                scores, predicted, model_cost = cache_models[key]
                new_adapter = False
                rule = "SVC.predict"
                new_model = model_new
                fit_cpu, fit_wall = model_cost if new_model else (0.0, 0.0)
            else:
                cache_key = (method, key)
                new_adapter = cache_key not in cache_adapters
                if new_adapter:
                    old_method = "anchor_lambda1" if method == "raw_anchor_lambda1" else method
                    scores, adapter = calibrate_scores(old_method, support_scores, copied,
                                                        query_scores, job["classes"])
                    cache_adapters[cache_key] = scores
                    adapter_parameters.append({"method": method, "objective_key": key, "draw": draw,
                                               "noise_q_per_class": q, "noise_seed": seed,
                                               "adapter": adapter})
                    new_adapters += 1
                scores = cache_adapters[cache_key]
                predicted = writer.classes[scores.argmax(axis=1)]
                rule, new_model = "score_argmax", False
                fit_cpu = time.process_time() - started if new_adapter else 0.0
                fit_wall = time.monotonic() - started_wall if new_adapter else 0.0
            writer.record(method, 10, scores, pred=predicted, draw=draw, support_ids=support_ids,
                          decision_rule=rule, actual_noise_rate=q / 10, noise_q_per_class=q,
                          support_noise_fraction=q / 10,
                          total_labeled_training_noise_fraction=episode["diagnostic_labels_changed_count"] / len(fit_ids),
                          source_training_labels_corrupted=0,
                          noise_seed=seed, diagnostic_labels_changed_count=episode["diagnostic_labels_changed_count"],
                          objective_key=key, artificial_label_copy=True,
                          reused_model=method == "full_pipeline_fixedgamma" and not new_model,
                          reused_adapter=method != "full_pipeline_fixedgamma" and not new_adapter,
                          new_model_fit=bool(new_model), new_adapter_fit=bool(new_adapter),
                          support_labels_used=len(support_ids), label_histogram_unchanged=True,
                          fit_cpu_seconds=fit_cpu, fit_wall_seconds=fit_wall)
    if new_fits != fit_count(job) or new_adapters != adapter_count(job):
        raise ValueError("Actual label-dose fits differ from frozen objective-deduplicated counts")
    if joblib.hash(source) != initial_hash:
        raise ValueError("G03 changed the frozen source")
    writer.save_json("noise_plan.json", plan)
    writer.save_json("adapter_parameters.json", adapter_parameters)
    writer.save_json("reused_model_provenance.json", reused_records)
    writer.save_json("training_id_locks.json", {"training_ids": fit_ids, "statistics_ids": fit_ids,
                                               "source_labels_unchanged": True, "support_copies_only": True})
    return writer.finish(expected_endpoints(job), new_model_fits=new_fits,
                         new_adapter_fits=new_adapters, source_model_unchanged=True,
                         artificial_label_noise=True, diagnostic_dose_not_iid_field_errors=True,
                         unique_noisy_label_objectives=len(cache_models),
                         G02_dependency_job_id="G02_" + job["parent_job_id"])


def execute(job, X, metadata, axis, output, input_hash, oldroot, round2root):
    writer = TaskWriter(job, X, metadata, axis, output, input_hash, oldroot)
    if job["family"] == "G02":
        return _g02(writer, job, axis, oldroot, round2root)
    if job["family"] == "G03":
        return _g03(writer, job, axis, oldroot, round2root)
    raise ValueError("Only G02/G03 are implemented by this module")
