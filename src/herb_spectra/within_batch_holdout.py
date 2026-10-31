"""Retrospective same-batch train/calibration/prediction experiment.

This deliberately does not relax the genuine cross-batch protocol. Prediction
truth is returned separately from the features and never enters the fitting
API. Existing samples have already been explored in earlier CV experiments.
"""
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np

from .calibration import AnchorScoreCalibrator
from .classification import (class_scores, classification_metrics, sample_folds,
                             validate_dataset)
from .classification_transfer import fit_source


ROLES = ("training", "calibration", "prediction")
SCOPE = "retrospective_same_batch_internal_holdout; not independent batch confirmation"


class HoldoutProtocolError(ValueError):
    """An explicit partition or annotation prerequisite was not satisfied."""


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def model_fingerprint(model):
    """Fingerprint fitted arrays and parameters before and after adaptation."""
    return joblib.hash(model, hash_name="sha1")


def _counts(labels):
    return {str(c): int(np.count_nonzero(labels == c)) for c in np.unique(labels)}


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise HoldoutProtocolError(f"{name} must be a positive integer")
    return int(value)


def prepare_holdout(arrays, plan, *, frequency_min=.1, frequency_max=1.5):
    """Validate a frozen plan against reviewed arrays, then separate truth.

    The plan is checked using the original metadata once. After this boundary,
    the prediction payload contains no origin labels or annotation flags.
    """
    data = validate_dataset(arrays)
    version = np.asarray(arrays.get("dataset_version", ""))
    if version.shape != () or not str(version) or str(version) != plan.get("dataset_version"):
        raise HoldoutProtocolError("NPZ and split plan need the same explicit dataset_version")
    if len(set(data["groups"].tolist())) != len(data["y"]):
        raise HoldoutProtocolError("This protocol requires one explicitly independent physical sample per record")
    records = plan.get("records", [])
    if not isinstance(records, list) or len(records) != len(data["y"]):
        raise HoldoutProtocolError("Split plan must cover every input record exactly once")
    identifiers = [row.get("record_id") for row in records]
    if len(set(identifiers)) != len(identifiers) or set(identifiers) != set(data["record_ids"].tolist()):
        raise HoldoutProtocolError("Split plan has duplicate, missing or unknown record IDs")
    lookup = {str(identifier): i for i, identifier in enumerate(data["record_ids"])}
    role_indices = {role: [] for role in ROLES}
    for row in records:
        index = lookup[row["record_id"]]
        role = row.get("role")
        if role not in ROLES:
            raise HoldoutProtocolError("Each plan record needs an explicit supported role")
        if (row.get("physical_sample_id") != str(data["groups"][index])
                or row.get("origin_group_id") != str(data["y"][index])):
            raise HoldoutProtocolError("Split plan disagrees with explicit physical-sample or origin annotations")
        role_indices[role].append(index)
    if any(not values for values in role_indices.values()):
        raise HoldoutProtocolError("All three roles must be nonempty")
    if not np.isfinite([frequency_min, frequency_max]).all() or frequency_min >= frequency_max:
        raise HoldoutProtocolError("Provide a finite ordered frequency interval")
    mask = (data["axis"] >= frequency_min) & (data["axis"] <= frequency_max)
    if mask.sum() < 2:
        raise HoldoutProtocolError("Requested fixed frequency interval has fewer than two points")
    prepared = {}
    classes = set(data["y"].tolist())
    for role, values in role_indices.items():
        idx = np.asarray(sorted(values), dtype=int)
        if set(data["y"][idx].tolist()) != classes:
            raise HoldoutProtocolError(f"{role} does not cover every known class")
        item = {name: data[name][idx].copy() for name in
                ("record_ids", "groups", "batch_ids", "labels_verified", "y")}
        item.update(X_alpha=data["X_alpha"][idx][:, mask].copy(),
                    X_n=data["X_n"][idx][:, mask].copy(), axis=data["axis"][mask].copy(),
                    sample_ids=data["groups"][idx].copy())
        prepared[role] = item
    expected_counts = plan.get("counts_by_origin")
    if expected_counts is not None:
        actual = {str(label): {role: int(np.count_nonzero(data["y"][values] == label))
                              for role, values in role_indices.items()}
                  for label in np.unique(data["y"])}
        if actual != expected_counts:
            raise HoldoutProtocolError("Split plan declared counts disagree with its record assignments")
    truth = {str(identifier): str(label) for identifier, label in
             zip(prepared["prediction"]["record_ids"], prepared["prediction"].pop("y"))}
    prepared["prediction"].pop("labels_verified")
    metadata = {"dataset_version": str(version), "scope": SCOPE,
        "split_seed": plan.get("split_seed"), "fixed_frequency_interval": [frequency_min, frequency_max],
        "actual_frequency_range": [float(data["axis"][mask][0]), float(data["axis"][mask][-1])],
        "feature_count": int(mask.sum()), "original_feature_indices": np.flatnonzero(mask).tolist(),
        "roles": {role: {"sample_count": len(values),
                           "origin_counts": _counts(data["y"][values]),
                           "sample_ids": data["groups"][values].tolist()}
                  for role, values in role_indices.items()},
        "prediction_ids_encode_origin": True,
        "identifier_policy": "Identifiers and paths are never model features; files are not blinded",
        "batch_policy": "No invented batch IDs; same-batch roles are not three source batches"}
    if "thickness_mm" in arrays:
        thickness = np.asarray(arrays["thickness_mm"], dtype=float)
        if thickness.shape != (len(data["y"]),):
            raise HoldoutProtocolError("Thickness metadata shape differs from input records")
        metadata["thickness_mm_counts_by_role"] = {
            role: {str(float(value)): int(np.count_nonzero(thickness[values] == value))
                   for value in np.unique(thickness[values][np.isfinite(thickness[values])])}
            for role, values in role_indices.items()}
        metadata["nonfinite_thickness_by_role"] = {
            role: int(np.count_nonzero(~np.isfinite(thickness[values])))
            for role, values in role_indices.items()}
    return prepared, truth, metadata


def support_draws(calibration, classes, *, shots=(1,), repeats=20, seed=42):
    """Draw only from the dedicated calibration pool; no query argument exists."""
    repeats = _positive_integer(repeats, "repeats")
    shots = tuple(_positive_integer(k, "shots") for k in shots)
    if not shots or len(set(shots)) != len(shots):
        raise HoldoutProtocolError("Choose nonempty unique shot counts")
    pools = {str(c): np.flatnonzero(calibration["y"] == c) for c in classes}
    if any(not len(indices) for indices in pools.values()):
        raise HoldoutProtocolError("Calibration must contain all source classes")
    unavailable = {int(k): "Insufficient independent calibration samples per class: "
                   + ", ".join(f"{c}={len(indices)}" for c, indices in pools.items())
                   for k in shots if any(len(indices) < k for indices in pools.values())}
    rng = np.random.default_rng(seed)
    draws = []
    for repeat in range(repeats):
        orders = {c: rng.permutation(indices) for c, indices in pools.items()}
        for k in sorted(shots):
            if k not in unavailable:
                selected = np.concatenate([orders[str(c)][:k] for c in classes])
                draws.append({"repeat": repeat, "shots_per_class": int(k), "indices": selected,
                              "sample_ids": calibration["groups"][selected].tolist()})
    return draws, unavailable


def _validate_feature_roles(training, calibration, prediction):
    source, support = validate_dataset(training), validate_dataset(calibration)
    for key in ("record_ids", "groups"):
        sets = [set(np.asarray(item[key]).tolist()) for item in (source, support, prediction)]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise HoldoutProtocolError(f"Physical roles overlap in {key}")
    if "y" in prediction:
        raise HoldoutProtocolError("Prediction fitting payload must not contain truth labels")
    for item in (support, prediction):
        if not np.array_equal(source["axis"], item["axis"]):
            raise HoldoutProtocolError("All roles require the same fixed frequency grid")
    if set(source["y"].tolist()) != set(support["y"].tolist()):
        raise HoldoutProtocolError("Training and calibration need the same class vocabulary")
    X_query = np.asarray(prediction["X_alpha"], dtype=float)
    if (X_query.ndim != 2 or X_query.shape != (len(prediction["record_ids"]), len(source["axis"]))
            or not np.isfinite(X_query).all()):
        raise HoldoutProtocolError("Prediction features must be a finite paired-grid matrix")
    source["sample_ids"] = source["groups"].copy()
    return source, support, X_query


def fit_frozen_predictions(training, calibration, prediction, *, selectors=("full", "mcuve20"),
        shots=(1,), repeats=20, seed=42, draw_seed=42, source_cv_splits=3,
        regularization=1., output_dir=None, source_grid=None):
    """Fit source and score-only adaptation without any prediction truth.

    Fixed lambda is an engineering default declared before evaluation, not a
    value optimized using final prediction labels. Source refitting stops after
    training-only CV and final source fit. Only an outer score map is fitted.
    """
    if (not selectors or len(set(selectors)) != len(selectors)
            or not set(selectors).issubset({"full", "mcuve20"})):
        raise HoldoutProtocolError("Choose unique full/mcuve20 source selectors")
    if not np.isfinite(regularization) or regularization <= 0:
        raise HoldoutProtocolError("A positive predeclared anchor regularization is required")
    source, support, X_query = _validate_feature_roles(training, calibration, prediction)
    classes = np.unique(source["y"])
    draws, unavailable = support_draws(support, classes, shots=shots, repeats=repeats, seed=draw_seed)
    # Preflight before fitting; do not silently reduce folds or discard a class.
    sample_folds(source["y"], source["groups"], source_cv_splits, seed)
    output = Path(output_dir) if output_dir is not None else None
    if output is not None:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Choose a fresh experiment directory")
        output.mkdir(parents=True, exist_ok=True)
    bundle = {"scope": SCOPE, "class_order": classes.tolist(),
        "prediction_ids": np.asarray(prediction["record_ids"]).tolist(),
        "configuration": {"selectors": list(selectors), "shots_per_class": list(shots),
            "repeats": repeats, "model_seed": seed, "support_draw_seed": draw_seed,
            "source_cv_splits": source_cv_splits, "fixed_anchor_regularization": float(regularization),
            "regularization_selection": "predeclared engineering constant; no final-query tuning",
            "representation": "alpha", "source_tie_policy": "argmax_ovr_scores",
            "primary_example": "repeat 0, fixed in advance; all repeats reported"},
        "unavailable_shots": unavailable,
        "support_draws": [{k: v for k, v in row.items() if k != "indices"} for row in draws],
        "models": []}
    pool_sizes = _counts(support["y"])
    bundle["not_evaluable_uniform_shots"] = {
        k: {"status": "not_evaluable", "calibration_counts_by_class": pool_sizes,
            "reason": "Every class must provide K distinct calibration samples; query never shrinks"}
        for k in (3, 5, 10) if any(size < k for size in pool_sizes.values())}
    for selector in selectors:
        started = time.monotonic()
        case = output / selector if output is not None else None
        if case is not None:
            case.mkdir()
        pipeline, fit_info = fit_source(source, model="svc", selector=selector, cv_splits=source_cv_splits,
                                       seed=seed, grid=source_grid,
                                       memory=str(case / ".cache") if case is not None else None)
        if not np.array_equal(pipeline.classes_, classes):
            raise HoldoutProtocolError("Unexpected class-score column order")
        frozen_hash = model_fingerprint(pipeline)
        artifact = {}
        if case is not None:
            path = case / "source_model.joblib"
            joblib.dump(pipeline, path, compress=3)
            artifact = {"path": str(path), "sha256": _digest(path.read_bytes()),
                        "trust_policy": "Local generated model; never load untrusted joblib files"}
        query_scores = class_scores(pipeline, X_query)
        calibration_scores = class_scores(pipeline, support["X_alpha"])
        selector_step = pipeline.named_steps["selector"]
        selected = (np.arange(source["X_alpha"].shape[1]) if isinstance(selector_step, str)
                    else np.flatnonzero(selector_step.get_support()))
        outcomes = [{"method": "source_only", "shots_per_class": 0, "repeat": 0,
                     "predicted": classes[np.argmax(query_scores, axis=1)].tolist(),
                     "support_sample_ids": []}]
        for draw in draws:
            indices = draw["indices"]
            calibrator = AnchorScoreCalibrator(classes=classes, regularization=regularization)
            calibrator.fit(calibration_scores[indices], support["y"][indices])
            outcomes.append({"method": "anchor_scores", "shots_per_class": draw["shots_per_class"],
                "repeat": draw["repeat"], "predicted": calibrator.predict(query_scores).tolist(),
                "support_sample_ids": draw["sample_ids"],
                "support_counts_by_class": calibrator.support_counts_by_class_,
                "calibration_coefficients": calibrator.coef_.tolist(),
                "calibration_intercept": calibrator.intercept_.tolist()})
        after_hash = model_fingerprint(pipeline)
        if frozen_hash != after_hash:
            raise AssertionError("Frozen source model changed during prediction or calibration")
        artifact_unchanged = None
        if artifact:
            artifact_unchanged = artifact["sha256"] == _digest(Path(artifact["path"]).read_bytes())
            if not artifact_unchanged:
                raise AssertionError("Frozen model artifact changed")
        bundle["models"].append({"selector": selector, "source_fit": fit_info,
            "selected_feature_indices": selected.tolist(), "selected_feature_count": len(selected),
            "source_model_fingerprint_before": frozen_hash, "source_model_fingerprint_after": after_hash,
            "source_model_unchanged": True, "model_artifact": artifact,
            "model_artifact_unchanged": artifact_unchanged, "outcomes": outcomes,
            "fit_predict_seconds": time.monotonic() - started})
    return bundle


def score_prediction_bundle(bundle, truth):
    """Evaluate already-generated predictions. No fitting happens here."""
    identifiers = bundle["prediction_ids"]
    if set(identifiers) != set(truth) or len(identifiers) != len(truth):
        raise HoldoutProtocolError("Prediction truth must match the frozen query IDs exactly")
    classes = np.asarray(bundle["class_order"])
    labels = np.asarray([truth[identifier] for identifier in identifiers])
    if not set(labels.tolist()).issubset(set(classes.tolist())):
        raise HoldoutProtocolError("Prediction truth contains an unknown class")
    results, summaries, prediction_rows = [], [], []
    for model in bundle["models"]:
        baseline = np.asarray(model["outcomes"][0]["predicted"])
        baseline_metrics = classification_metrics(labels, baseline, classes)
        for outcome in model["outcomes"]:
            predicted = np.asarray(outcome["predicted"])
            metrics = classification_metrics(labels, predicted, classes)
            gain = {f"{metric}_gain": float(metrics[metric] - baseline_metrics[metric])
                    for metric in ("accuracy", "balanced_accuracy", "macro_f1")}
            corrected = int(np.count_nonzero((baseline != labels) & (predicted == labels)))
            damaged = int(np.count_nonzero((baseline == labels) & (predicted != labels)))
            row = {"selector": model["selector"], "method": outcome["method"],
                "shots_per_class": outcome["shots_per_class"], "repeat": outcome["repeat"],
                "metrics": metrics, **gain,
                "accuracy_gain_percentage_points": 100 * gain["accuracy_gain"],
                "paired_corrected_errors": corrected, "paired_new_errors": damaged,
                "paired_changed_predictions": int(np.count_nonzero(predicted != baseline)),
                "correct_predictions": int(np.count_nonzero(predicted == labels)),
                "query_count": len(labels)}
            results.append(row)
            for identifier, label, prediction in zip(identifiers, labels, predicted):
                prediction_rows.append({"selector": model["selector"], "method": outcome["method"],
                    "shots_per_class": outcome["shots_per_class"], "repeat": outcome["repeat"],
                    "record_id": identifier, "y_true": str(label), "y_predicted": str(prediction)})
        for method, k in sorted({(r["method"], r["shots_per_class"]) for r in results
                                 if r["selector"] == model["selector"]}):
            cases = [r for r in results if r["selector"] == model["selector"]
                     and r["method"] == method and r["shots_per_class"] == k]
            summary = {"selector": model["selector"], "method": method, "shots_per_class": k,
                       "evaluations": len(cases)}
            for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                values = [r["metrics"][metric] for r in cases]
                summary.update({f"{metric}_{name}": float(operation(values)) for name, operation in
                                [("mean", np.mean), ("sd", np.std), ("min", np.min), ("max", np.max)]})
            summary["accuracy_gain_percentage_points_mean"] = float(np.mean([
                r["accuracy_gain_percentage_points"] for r in cases]))
            summary["positive_gain_draws"] = sum(r["accuracy_gain"] > 0 for r in cases)
            summary["zero_gain_draws"] = sum(r["accuracy_gain"] == 0 for r in cases)
            summary["negative_gain_draws"] = sum(r["accuracy_gain"] < 0 for r in cases)
            summaries.append(summary)
    return {"scope": SCOPE, "rows": results, "summary": summaries,
            "prediction_rows": prediction_rows,
            "uncertainty_scope": "Repeated supports share one query and the same scarce-class sample; SD is descriptive, not independent-batch uncertainty"}
