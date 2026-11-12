"""Fixed model-side mechanisms for the second B study.

F02 separates fitted spectral scales from the data-dependent RBF gamma. F03
crosses class weighting with the frozen source-coverage/ratio sampling design.
F04 measures a nested label-budget curve. Historical exact endpoints are loaded
only after their seals and hashes have been checked, and are never fitted again.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from herb_spectra.spectral_models import FittedSpectralSource, SpectralPreprocessor
from b_followup_common_20261110 import TaskWriter, endpoint_id, load_parent


K_VALUES = (1, 3, 5, 10)
SCALER_GAMMA_METHODS = (
    ("scaler_frozen_gamma_fixed", "frozen", "fixed"),
    ("scaler_frozen_gamma_scale", "frozen", "scale"),
    ("scaler_refit_gamma_fixed", "refit", "fixed"),
    ("scaler_refit_gamma_scale", "refit", "scale"),
)
LEARNING_BUDGETS = (50, 100, 150, 190, 240)


def expected_endpoints(job):
    family = job["family"]
    if family == "F02":
        return [endpoint_id("identity")] + [
            endpoint_id(method, k) for k in K_VALUES
            for method, _, _ in SCALER_GAMMA_METHODS
        ]
    if family == "F03":
        return [endpoint_id(method) for method in (
            "class_weight_balanced", "class_weight_balanced_argmax",
            "class_weight_none", "class_weight_none_argmax",
        )]
    if family == "F04":
        return [endpoint_id(f"source_n{n:03d}") for n in LEARNING_BUDGETS]
    raise ValueError(f"Unknown model-side family: {family}")


def fit_count(job):
    if job["family"] == "F02":
        reused = job.get("exact_gamma_reuse_k", [])
        if len(set(reused)) != len(reused) or not set(reused).issubset(K_VALUES):
            raise ValueError("Invalid frozen exact-gamma reuse plan")
        return 12 - len(reused)
    if job["family"] == "F03":
        if job["ratio"] == "B":
            return 0 if job.get("parent_job_id") else 1
        return 1 if job.get("parent_job_id") else 2
    if job["family"] == "F04":
        return 3
    raise ValueError("Unknown model-side family")


def source_hashes(model):
    """Hash learned preprocessing separately from the classifier."""
    return {
        "preprocessor_hash": joblib.hash(model.preprocessor_),
        "scaler_hash": joblib.hash(model.scaler_),
        "scaler_mean_sha256": hashlib.sha256(
            np.ascontiguousarray(model.scaler_.mean_).tobytes()).hexdigest(),
        "scaler_scale_sha256": hashlib.sha256(
            np.ascontiguousarray(model.scaler_.scale_).tobytes()).hexdigest(),
    }


def fit_model(X_train, y_train, axis, *, reference=None,
              scaler_mode="refit", gamma_mode="scale", class_weight="balanced"):
    """Training-only SVM fit; no support/query roles or labels are accepted.

    ``reference`` supplies frozen statistics and, for gamma_mode=fixed, the
    source model's actual numeric gamma. Neither its classifier nor its fitted
    transforms are changed. ``gamma='scale'`` deliberately recomputes gamma on
    this endpoint's training matrix, even when the scaler itself is frozen.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=str)
    axis = np.asarray(axis, dtype=float)
    if (X_train.ndim != 2 or len(X_train) != len(y_train)
            or not np.isfinite(X_train).all() or X_train.shape[1] != len(axis)):
        raise ValueError("Training spectra/labels/axis do not align")
    if not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0):
        raise ValueError("Invalid spectral axis")
    if scaler_mode not in {"frozen", "refit"} or gamma_mode not in {"fixed", "scale"}:
        raise ValueError("Unknown prespecified scaler/gamma condition")
    if class_weight not in {"balanced", None}:
        raise ValueError("Unknown prespecified class weighting")
    if (scaler_mode == "frozen" or gamma_mode == "fixed") and reference is None:
        raise ValueError("A frozen source is required for this condition")
    classes = np.unique(y_train)
    if len(classes) < 2:
        raise ValueError("At least two training classes are required")
    preprocessor = (copy.deepcopy(reference.preprocessor_) if reference is not None
                    else SpectralPreprocessor("snv", axis))
    preprocessed = preprocessor.transform(X_train)
    if scaler_mode == "frozen":
        scaler = copy.deepcopy(reference.scaler_)
        values = scaler.transform(preprocessed)
    else:
        scaler = StandardScaler()
        values = scaler.fit_transform(preprocessed)
    gamma = (float(reference.estimator_._gamma) if gamma_mode == "fixed" else "scale")
    estimator = SVC(C=10.0, kernel="rbf", gamma=gamma,
                    class_weight=class_weight, break_ties=True,
                    decision_function_shape="ovr", random_state=42)
    estimator.fit(values, y_train)
    if not np.array_equal(estimator.classes_, classes):
        raise ValueError("Classifier class order changed")
    classes.setflags(write=False)
    candidate = {"id": "b_followup_fixed_snv_rbf_svm_c10", "preprocessing": "snv",
                 "model": "svc", "C": 10.0, "class_weight": class_weight,
                 "scaler_mode": scaler_mode, "gamma_mode": gamma_mode}
    metadata = {
        "fit_scope": "explicit_training_ids_only", "source_samples": len(y_train),
        "class_order": classes.tolist(), "input_features": X_train.shape[1],
        "class_counts": {str(c): int(np.count_nonzero(y_train == c)) for c in classes},
        "C": 10.0, "gamma_argument": gamma, "effective_gamma": float(estimator._gamma),
        "class_weight": class_weight, "scaler_mode": scaler_mode,
        "gamma_mode": gamma_mode, "preprocessing": "SNV ddof=1",
    }
    return FittedSpectralSource(candidate, preprocessor, scaler, estimator,
                                classes, metadata)


def native_predict(model, X):
    return model.estimator_.predict(model._scaled(X))


def diagnostics(model, *, reference=None):
    result = {
        **source_hashes(model), "effective_gamma": float(model.estimator_._gamma),
        "class_weight_json": json.dumps(dict(zip(model.classes_.tolist(),
                                                   model.estimator_.class_weight_.tolist())),
                                          ensure_ascii=False, sort_keys=True),
        "n_support_vectors": int(model.estimator_.support_.size),
        "n_support_vectors_json": json.dumps(model.estimator_.n_support_.tolist()),
    }
    if reference is not None:
        result["source_effective_gamma"] = float(reference.estimator_._gamma)
        result["scaler_identical_to_source"] = (
            joblib.hash(model.scaler_) == joblib.hash(reference.scaler_))
        result["gamma_identical_to_source"] = (
            float(model.estimator_._gamma) == float(reference.estimator_._gamma))
    return result


def nested_support(job, indexed, k):
    """The exact old candidate-pool order, one nested prefix per class."""
    if k not in K_VALUES:
        raise ValueError("Unexpected support budget")
    result = []
    for label in job["classes"]:
        members = [sample for sample in job["support_ids"]
                   if indexed.at[sample, job["label_column"]] == label]
        if len(members) != 10:
            raise ValueError("Frozen candidate pool must contain ten samples per class")
        result.extend(members[:k])
    return result


def learning_ids(job, indexed, budget):
    """Nested class-balanced subsets of the original Source, then full pool."""
    if budget not in LEARNING_BUDGETS or len(job["classes"]) != 5:
        raise ValueError("F04 requires the frozen five-origin budgets")
    if budget == 240:
        return job["source_ids"] + nested_support(job, indexed, 10)
    if budget == 190:
        return list(job["source_ids"])
    per_class = budget // 5
    result = []
    for label in job["classes"]:
        members = [sample for sample in job["source_ids"]
                   if indexed.at[sample, job["label_column"]] == label]
        if len(members) != 38:
            raise ValueError("F04 source must contain 38 samples per class")
        result.extend(members[:per_class])
    if len(result) != budget or len(set(result)) != budget:
        raise ValueError("Invalid nested learning-curve IDs")
    return result


def _source_identity(writer, source):
    scores = source.predict_scores(writer.query_X)
    writer.record("identity", 0, scores, pred=native_predict(source, writer.query_X),
                  support_ids=[], decision_rule="SVC.predict",
                  reused_model=True, new_classifier_fits=0,
                  training_n=len(writer.job["source_ids"]), **diagnostics(source, reference=source))


def _load_old_model(parent_dir, name):
    # load_parent has just verified every parent artifact against its seal.
    # This additional check prevents reading a model that was not in the seal.
    result = json.loads((parent_dir / "result.json").read_text())
    if name not in result["artifact_hashes"]:
        raise ValueError("Requested reuse model is absent from the verified parent receipt")
    path = parent_dir / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != result["artifact_hashes"][name]:
        raise ValueError("Historical model bytes changed before reuse")
    return joblib.load(path)


def gamma_reuse_plan(job, oldroot):
    """Return exact old-gamma equalities; never deduplicate near equal floats."""
    _, source, parent_dir = load_parent(job, oldroot)
    gamma = float(source.estimator_._gamma)
    return [k for k in K_VALUES if float(
        _load_old_model(parent_dir, f"retrain_k{k}.joblib").estimator_._gamma) == gamma]


def _f02(writer, job, axis, oldroot):
    old_result, source, parent_dir = load_parent(job, oldroot)
    old_hash = joblib.hash(source)
    _source_identity(writer, source)
    training_locks, new_fits, provenance = {}, 0, []
    planned_gamma_reuse = set(job.get("exact_gamma_reuse_k", []))
    # The plan is an input to the frozen protocol, not inferred from Query scores.
    if planned_gamma_reuse != set(gamma_reuse_plan(job, oldroot)):
        raise ValueError("Exact numeric gamma equality differs from the frozen reuse plan")
    for k in K_VALUES:
        support_ids = nested_support(job, writer.indexed, k)
        fit_ids = job["source_ids"] + support_ids
        training_locks[str(k)] = fit_ids
        for method, scaler_mode, gamma_mode in SCALER_GAMMA_METHODS:
            started_cpu, started_wall = time.process_time(), time.monotonic()
            reused_full = scaler_mode == "refit" and gamma_mode == "scale"
            reused_gamma = scaler_mode == "refit" and gamma_mode == "fixed" and k in planned_gamma_reuse
            reused = reused_full or reused_gamma
            if reused:
                artifact = f"retrain_k{k}.joblib"
                model = _load_old_model(parent_dir, artifact)
                origin = {"method": method, "k": k, "artifact": artifact,
                          "parent_job_id": job["parent_job_id"],
                          "artifact_sha256": old_result["artifact_hashes"][artifact],
                          "reuse_reason": "EXACT_EFFECTIVE_GAMMA_MATCH" if reused_gamma else "EXACT_HISTORICAL_ENDPOINT"}
                provenance.append(origin)
            else:
                model = fit_model(writer.take(fit_ids), writer.labels(fit_ids), axis,
                                  reference=source, scaler_mode=scaler_mode,
                                  gamma_mode=gamma_mode, class_weight="balanced")
                writer.save_model(f"{method}_k{k}.joblib", model)
                new_fits += 1
            if model.classes_.tolist() != job["classes"]:
                raise ValueError("F02 reused/fitted class vocabulary changed")
            score = model.predict_scores(writer.query_X)
            prediction = native_predict(model, writer.query_X)
            details = diagnostics(model, reference=source)
            if scaler_mode == "frozen" and not details["scaler_identical_to_source"]:
                raise ValueError("Frozen source scaler changed")
            if gamma_mode == "fixed" and not details["gamma_identical_to_source"]:
                raise ValueError("Frozen numeric source gamma changed")
            writer.record(method, k, score, pred=prediction,
                          support_ids=support_ids, decision_rule="SVC.predict",
                          scaler_mode=scaler_mode, gamma_mode=gamma_mode,
                          reused_model=reused, new_classifier_fits=int(not reused),
                          training_n=len(fit_ids), total_support_labels=len(support_ids),
                          fit_cpu_seconds=time.process_time() - started_cpu,
                          fit_wall_seconds=time.monotonic() - started_wall, **details)
    if joblib.hash(source) != old_hash:
        raise ValueError("Historical source model changed during the factorial")
    writer.save_json("training_id_locks.json", training_locks)
    writer.save_json("reused_model_provenance.json", provenance)
    return writer.finish(expected_endpoints=expected_endpoints(job), new_model_fits=new_fits,
                         reused_models=5 + len(planned_gamma_reuse), source_model_unchanged=True,
                         reused_model_endpoints=5 + len(planned_gamma_reuse),
                         unique_historical_models_reused=5,
                         parent_result_sha256=hashlib.sha256((parent_dir / "result.json").read_bytes()).hexdigest())


def _f03(writer, job, axis, oldroot):
    source, old_result, parent_dir = None, None, None
    if job.get("parent_job_id"):
        old_result, source, parent_dir = load_parent(job, oldroot)
    new_fits, provenance, balanced_model = 0, [], None
    for name, weight in (("class_weight_balanced", "balanced"), ("class_weight_none", None)):
        started_cpu, started_wall = time.process_time(), time.monotonic()
        reused_historical = weight == "balanced" and source is not None
        equal_weight_reuse = weight is None and job["ratio"] == "B"
        reused = reused_historical or equal_weight_reuse
        if reused_historical:
            model = source
            provenance.append({"method": name, "artifact": "source_model.joblib",
                               "parent_job_id": job["parent_job_id"],
                               "artifact_sha256": old_result["artifact_hashes"]["source_model.joblib"],
                               "reuse_reason": "EXACT_HISTORICAL_ENDPOINT"})
        elif equal_weight_reuse:
            if balanced_model is None or not np.array_equal(
                    balanced_model.estimator_.class_weight_, np.ones(len(job["classes"]))):
                raise ValueError("Balanced/None objective equality requires exactly unit class weights")
            counts = writer.labels(job["source_ids"])
            if len(set(np.unique(counts, return_counts=True)[1])) != 1:
                raise ValueError("Balanced/None objective equality requires equal training class counts")
            model = balanced_model
            provenance.append({"method": name, "reused_from_method": "class_weight_balanced",
                               "parent_job_id": job.get("parent_job_id"),
                               "reuse_reason": "EQUAL_EFFECTIVE_CLASS_WEIGHTS",
                               "effective_class_weights": model.estimator_.class_weight_.tolist()})
        else:
            model = fit_model(writer.take(job["source_ids"]), writer.labels(job["source_ids"]),
                              axis, class_weight=weight)
            writer.save_model(f"{name}.joblib", model)
            new_fits += 1
        if weight == "balanced":
            balanced_model = model
        if model.classes_.tolist() != job["classes"]:
            raise ValueError("F03 class vocabulary changed")
        scores = model.predict_scores(writer.query_X)
        native = native_predict(model, writer.query_X)
        native_score_disagreements = int(np.count_nonzero(
            native != model.classes_[np.argmax(scores, axis=1)]))
        detail = {**diagnostics(model), "class_weight_setting": "balanced" if weight else "none",
                  "reused_model": reused, "training_n": len(job["source_ids"]),
                  "equal_effective_class_weight_reuse": equal_weight_reuse,
                  "native_argmax_disagreements": native_score_disagreements,
                  "fit_cpu_seconds": time.process_time() - started_cpu,
                  "fit_wall_seconds": time.monotonic() - started_wall}
        writer.record(name, 0, scores, pred=native, support_ids=[],
                      decision_rule="SVC.predict", new_classifier_fits=int(not reused), **detail)
        writer.record(name + "_argmax", 0, scores, support_ids=[],
                      decision_rule="score_argmax", new_classifier_fits=0, **detail)
    writer.save_json("reused_model_provenance.json", provenance)
    return writer.finish(expected_endpoints=expected_endpoints(job), new_model_fits=new_fits,
                         reused_models=int(source is not None) + int(job["ratio"] == "B"),
                         reused_model_endpoints=int(source is not None) + int(job["ratio"] == "B"),
                         unique_historical_models_reused=int(source is not None),
                         source_origin_counts=writer.indexed.loc[job["source_ids"], "origin_group_id"].value_counts().to_dict())


def _f04(writer, job, axis, oldroot):
    old_result, source, parent_dir = load_parent(job, oldroot)
    old_hash = joblib.hash(source)
    new_fits, training_locks, provenance = 0, {}, []
    for budget in LEARNING_BUDGETS:
        started_cpu, started_wall = time.process_time(), time.monotonic()
        fit_ids = learning_ids(job, writer.indexed, budget)
        training_locks[str(budget)] = fit_ids
        if budget == 190:
            model, reused, artifact = source, True, "source_model.joblib"
        elif budget == 240:
            artifact = "retrain_k10.joblib"
            model, reused = _load_old_model(parent_dir, artifact), True
        else:
            model, reused, artifact = fit_model(writer.take(fit_ids), writer.labels(fit_ids), axis), False, None
            writer.save_model(f"source_n{budget:03d}.joblib", model)
            new_fits += 1
        if reused:
            provenance.append({"training_n": budget, "artifact": artifact,
                               "parent_job_id": job["parent_job_id"],
                               "artifact_sha256": old_result["artifact_hashes"][artifact]})
        if model.classes_.tolist() != job["classes"]:
            raise ValueError("F04 class vocabulary changed")
        scores = model.predict_scores(writer.query_X)
        writer.record(f"source_n{budget:03d}", 0, scores,
                      pred=native_predict(model, writer.query_X), support_ids=[],
                      decision_rule="SVC.predict", training_n=budget,
                      labels_used=budget, support_pool_labels_used=50 if budget == 240 else 0,
                      learning_curve_k_not_applicable=True, reused_model=reused,
                      new_classifier_fits=int(not reused),
                      fit_cpu_seconds=time.process_time() - started_cpu,
                      fit_wall_seconds=time.monotonic() - started_wall,
                      **diagnostics(model, reference=source))
    if joblib.hash(source) != old_hash:
        raise ValueError("Historical source changed during learning-curve evaluation")
    writer.save_json("training_id_locks.json", training_locks)
    writer.save_json("reused_model_provenance.json", provenance)
    return writer.finish(expected_endpoints=expected_endpoints(job), new_model_fits=new_fits,
                         reused_models=2, source_model_unchanged=True)


def execute(job, X, metadata, axis, output, input_hash, oldroot):
    """Execute exactly one family bundle; the caller handles sealed resumes."""
    writer = TaskWriter(job, X, metadata, axis, output, input_hash, oldroot)
    if job["family"] == "F02":
        return _f02(writer, job, axis, oldroot)
    if job["family"] == "F03":
        return _f03(writer, job, axis, oldroot)
    if job["family"] == "F04":
        return _f04(writer, job, axis, oldroot)
    raise ValueError("Only F02, F03 and F04 are supported by the model module")
