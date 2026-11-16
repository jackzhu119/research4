"""Known-class, new-batch few-shot classification with frozen target queries.

The three roles must contain disjoint physical specimens and real batch IDs.
Development queries select calibration strength; final-target query labels are
only passed to scoring. This is an engineering comparison of existing methods,
not evidence that calibration or a new algorithm improves medicinal spectra.
"""
from dataclasses import dataclass
import hashlib
from itertools import combinations
import time

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .calibration import AnchorScoreCalibrator
from .classification import (aggregate_samples, build_pipeline, class_scores,
                             classification_metrics, parameter_grid,
                             representation, sample_folds, validate_dataset)


METHODS = ("source_only", "support_only", "source_plus_support_full_retrain",
           "frozen_representation_head", "anchor_scores")


class TransferProtocolError(ValueError):
    """A scientific input or partition prerequisite is not satisfied."""
    def __init__(self, issues):
        self.issues = tuple(issues if isinstance(issues, (tuple, list)) else [str(issues)])
        super().__init__("; ".join(self.issues))


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise TransferProtocolError(f"{name} must be an integer >= {minimum}")
    return int(value)


def prepare_domains(source, development, target):
    """Validate roles before fitting any model or preprocessing statistics."""
    raw = {name: validate_dataset(data) for name, data in
           [("source", source), ("development", development), ("target", target)]}
    issues = []
    reference_axis = raw["source"]["axis"]
    reference_classes = set(raw["source"]["y"].tolist())
    for role, data in raw.items():
        missing = sum(str(value).strip().lower() in
                      {"", "nan", "none", "null", "unknown", "unverified", "未知", "待补"}
                      for value in data["batch_ids"])
        if missing:
            issues.append(f"{role}: {missing} records lack verified batch IDs; record/file IDs are not batches")
        if not np.array_equal(reference_axis, data["axis"]):
            issues.append(f"{role}: frequency grid differs from the source")
        if set(data["y"].tolist()) != reference_classes:
            issues.append(f"{role}: must contain the same known classes as source, not a single-origin new class")
    for left, right in combinations(raw, 2):
        for key, label in [("groups", "physical samples"), ("record_ids", "record IDs")]:
            if set(raw[left][key]) & set(raw[right][key]):
                issues.append(f"{left}/{right}: overlapping {label}")
        known_left = {str(v).strip() for v in raw[left]["batch_ids"] if str(v).strip()}
        known_right = {str(v).strip() for v in raw[right]["batch_ids"] if str(v).strip()}
        if known_left & known_right:
            issues.append(f"{left}/{right}: overlapping batch IDs")
        def paired_hashes(data):
            paired = np.column_stack([data["X_alpha"], data["X_n"]]).astype("<f8")
            paired[paired == 0] = 0.
            return {hashlib.sha256(row.tobytes()).hexdigest() for row in paired}
        if paired_hashes(raw[left]) & paired_hashes(raw[right]):
            issues.append(f"{left}/{right}: identical paired spectra across roles require provenance review")
    if issues:
        raise TransferProtocolError(issues)
    return {role: aggregate_samples(data) for role, data in raw.items()}


def fixed_partition(y, sample_ids, query_fraction=.5, minimum_query_per_class=10, seed=42):
    """Reserve query once; larger K never changes this partition."""
    y, sample_ids = np.asarray(y), np.asarray(sample_ids)
    minimum = _integer(minimum_query_per_class, "minimum_query_per_class")
    if not 0 < query_fraction < 1:
        raise TransferProtocolError("query_fraction must lie strictly between zero and one")
    if y.ndim != 1 or sample_ids.shape != y.shape or len(set(sample_ids.tolist())) != len(y):
        raise TransferProtocolError("Partition requires one unique physical-sample ID per row")
    rng = np.random.default_rng(seed)
    query, pool = [], []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        count = max(minimum, int(np.ceil(len(indices) * query_fraction)))
        if count >= len(indices):
            raise TransferProtocolError(f"Class {label}: cannot reserve {count} query samples and a support pool")
        order = rng.permutation(indices)
        query.extend(order[:count]); pool.extend(order[count:])
    return {"query": np.asarray(sorted(query), dtype=int),
            "support_pool": np.asarray(sorted(pool), dtype=int)}


def checked_partition(partition, size):
    """Accept an explicitly frozen partition without consulting query labels."""
    result = {}
    for name in ("query", "support_pool"):
        values = np.asarray(partition[name])
        if (values.ndim != 1 or values.dtype.kind not in "iu" or not len(values)
                or len(set(values.tolist())) != len(values)
                or np.any(values < 0) or np.any(values >= size)):
            raise TransferProtocolError(f"Invalid frozen {name} indices")
        result[name] = values.astype(int)
    if set(result["query"]) & set(result["support_pool"]):
        raise TransferProtocolError("Support pool overlaps the fixed query")
    if set(result["query"]) | set(result["support_pool"]) != set(range(size)):
        raise TransferProtocolError("Frozen query and support pool must cover the role's physical samples")
    return result


def nested_support_draws(y, partition, shots=(1, 3, 5, 10), repeats=20, seed=42):
    """Nested K-per-class supports; every repeat and K shares one query."""
    y = np.asarray(y)
    repeats = _integer(repeats, "repeats")
    shots = tuple(_integer(value, "shots_per_class") for value in shots)
    if not shots or len(set(shots)) != len(shots):
        raise TransferProtocolError("shots must be nonempty and unique")
    partition = checked_partition(partition, len(y))
    classes = np.unique(y)
    pools = {str(label): partition["support_pool"][y[partition["support_pool"]] == label]
             for label in classes}
    unavailable = {k: f"Each class needs {k} independent support samples; available: "
                   + ", ".join(f"{label}={len(pool)}" for label, pool in pools.items())
                   for k in shots if any(len(pool) < k for pool in pools.values())}
    rng = np.random.default_rng(seed)
    draws = []
    for repeat in range(repeats):
        ordered = {label: rng.permutation(pool) for label, pool in pools.items()}
        for k in sorted(shots):
            if k in unavailable:
                continue
            support = np.concatenate([ordered[str(label)][:k] for label in classes])
            draws.append({"repeat": repeat, "shots_per_class": k, "support": support,
                          "query": partition["query"].copy()})
    return draws, unavailable


def fit_source(data, *, modality="alpha", model="svc", selector="full", cv_splits=3,
               cars_cv=3, seed=42, grid=None, memory=None):
    X, blocks = representation(data, modality)
    try:
        folds = sample_folds(data["y"], data["groups"], n_splits=cv_splits, seed=seed)
    except ValueError as exc:
        raise TransferProtocolError(f"Source CV is not feasible: {exc}") from exc
    if selector == "cars":
        for train, _ in folds:
            if any(np.count_nonzero(data["y"][train] == label) < cars_cv for label in np.unique(data["y"])):
                raise TransferProtocolError("Source inner training folds cannot support the requested CARS internal CV")
    pipeline = build_pipeline(model=model, selector=selector, block_sizes=blocks,
                              random_state=seed, cars_cv=cars_cv, memory=memory)
    if model == "svc":
        # Phase-C source and anchored scores must use the same tie policy.
        # Do not change the separate, already frozen phase-A/B factory.
        pipeline.set_params(model__break_ties=True, model__decision_function_shape="ovr")
    parameters = parameter_grid(model, folds, X.shape, selector=selector) if grid is None else grid
    if model == "svc":
        for name, required in [("model__break_ties", True), ("model__decision_function_shape", "ovr")]:
            if name in parameters and any(value != required for value in parameters[name]):
                raise TransferProtocolError("Source grid cannot override the phase-C class-score tie policy")
    def score_macro_f1(estimator, values, labels):
        predicted = score_predictions(estimator, values)
        return classification_metrics(labels, predicted, estimator.classes_)["macro_f1"]
    search = GridSearchCV(pipeline, parameters, scoring=score_macro_f1, cv=folds,
                          n_jobs=1, error_score="raise")
    start = time.monotonic(); search.fit(X, data["y"])
    return search.best_estimator_, {"best_params": search.best_params_,
        "best_inner_macro_f1": float(search.best_score_), "source_samples": len(X),
        "source_sample_ids": data["sample_ids"].tolist(), "fit_seconds": time.monotonic() - start,
        "tuning_scope": "source_only", "candidate_count": len(search.cv_results_["params"]),
        "tie_policy": "argmax_ovr_scores" if model == "svc" else "argmax_plsda_scores",
        "exact_tie_resolution": "first_class_in_source_class_order",
        "source_svc_break_ties": bool(search.best_estimator_.get_params()["model__break_ties"]) if model == "svc" else None,
        "inner_folds": [{"train_sample_ids": data["sample_ids"][train].tolist(),
                         "validation_sample_ids": data["sample_ids"][validation].tolist()}
                        for train, validation in folds]}


def choose_anchor_regularization(source_pipeline, X_development, y_development, partition,
                                 shots=(1, 3, 5, 10), repeats=20,
                                 regularizations=(.1, 1., 10., 100.), seed=42):
    """Tune only using development supports and development query labels."""
    values = tuple(float(value) for value in regularizations)
    if not values or len(set(values)) != len(values) or any(not np.isfinite(v) or v <= 0 for v in values):
        raise TransferProtocolError("Anchor regularizations must be unique positive finite values")
    draws, unavailable = nested_support_draws(y_development, partition, shots, repeats, seed)
    scores = class_scores(source_pipeline, X_development)
    classes = np.asarray(source_pipeline.classes_)
    trials, selected = [], {}
    for k in sorted(set(draw["shots_per_class"] for draw in draws)):
        candidates = []
        for value in values:
            outcomes = []
            for draw in (d for d in draws if d["shots_per_class"] == k):
                support, query = draw["support"], draw["query"]
                calibrator = AnchorScoreCalibrator(classes=classes, regularization=value)
                calibrator.fit(scores[support], np.asarray(y_development)[support])
                prediction = calibrator.predict(scores[query])
                metrics = classification_metrics(np.asarray(y_development)[query], prediction, classes)
                outcomes.append(metrics["macro_f1"])
            entry = {"shots_per_class": k, "regularization": value,
                     "mean_development_macro_f1": float(np.mean(outcomes)), "repeats": len(outcomes)}
            candidates.append(entry); trials.append(entry)
        winner = max(candidates, key=lambda row: (row["mean_development_macro_f1"], row["regularization"]))
        selected[k] = winner["regularization"]
    return selected, {"selection_scope": "development_only", "trials": trials,
                      "unavailable_shots": unavailable,
                      "tie_break": "largest_regularization_on_equal_development_macro_f1"}


def score_predictions(source_pipeline, X):
    """Source prediction uses the exact argmax rule used by anchored scores."""
    return np.asarray(source_pipeline.classes_)[np.argmax(class_scores(source_pipeline, X), axis=1)]


@dataclass
class ScoreArgmaxClassifier:
    source_pipeline: object

    def predict(self, X):
        return score_predictions(self.source_pipeline, X)

    @property
    def named_steps(self):
        return self.source_pipeline.named_steps

    @property
    def classes_(self):
        return self.source_pipeline.classes_


@dataclass
class FrozenHeadClassifier:
    source_pipeline: object
    head: object

    def predict(self, X):
        return self.head.predict(self.source_pipeline[:-1].transform(X))


@dataclass
class AnchoredScoreClassifier:
    source_pipeline: object
    calibrator: object

    def predict(self, X):
        return self.calibrator.predict(class_scores(self.source_pipeline, X))


def fit_adaptation(method, source_pipeline, X_source, y_source, X_support, y_support,
                   *, regularization=None, seed=42):
    """Fit without query X or query y; return a predictor and its training scope."""
    if method not in METHODS:
        raise TransferProtocolError(f"Unknown method: {method}")
    if method == "source_only":
        return ScoreArgmaxClassifier(source_pipeline), {"training_scope": "source_only", "transforms_refit": False,
                                 "training_samples": len(X_source)}
    classes = np.asarray(source_pipeline.classes_)
    if set(np.asarray(y_support).tolist()) != set(classes.tolist()):
        raise TransferProtocolError("Every known class must occur in the labelled support set")
    if method == "source_plus_support_full_retrain":
        fitted = clone(source_pipeline)
        fitted.fit(np.concatenate([X_source, X_support]), np.concatenate([y_source, y_support]))
        return ScoreArgmaxClassifier(fitted), {"training_scope": "source_plus_support_complete_pipeline_fixed_source_hyperparameters",
                        "transforms_refit": True, "training_samples": len(X_source) + len(X_support)}
    if method == "support_only":
        fitted = make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight="balanced",
                                    max_iter=1000, random_state=seed))
        fitted.fit(X_support, y_support)
        return fitted, {"training_scope": "target_support_only_raw_feature_scaler_and_logistic",
                        "transforms_refit": True, "training_samples": len(X_support)}
    if method == "frozen_representation_head":
        head = LogisticRegression(C=1., class_weight="balanced", max_iter=1000, random_state=seed)
        head.fit(source_pipeline[:-1].transform(X_support), y_support)
        return FrozenHeadClassifier(source_pipeline, head), {
            "training_scope": "support_only_logistic_head_on_frozen_source_representation",
            "transforms_refit": False, "training_samples": len(X_support)}
    if regularization is None:
        raise TransferProtocolError("Anchor strength must have been selected on development data")
    calibrator = AnchorScoreCalibrator(classes=classes, regularization=regularization)
    calibrator.fit(class_scores(source_pipeline, X_support), y_support)
    return AnchoredScoreClassifier(source_pipeline, calibrator), {
        "training_scope": "support_only_source_score_residual_ridge",
        "transforms_refit": False, "training_samples": len(X_support),
        "regularization": float(regularization), "scores_are_probabilities": False}


def run_classification_transfer(source, development, target, *, modality="alpha", model="svc",
        selector="full", shots=(1, 3, 5, 10), repeats=20, query_fraction=.5,
        minimum_query_per_class=10, source_cv_splits=3, cars_cv=3,
        regularizations=(.1, 1., 10., 100.), seed=42, source_grid=None,
        methods=METHODS, development_partition=None, target_partition=None, memory=None):
    """Return reproducible sample predictions and paired adaptation comparisons."""
    if not methods or len(set(methods)) != len(methods) or not set(methods).issubset(METHODS):
        raise TransferProtocolError("Choose unique supported adaptation methods")
    domains = prepare_domains(source, development, target)
    source, development, target = (domains[name] for name in ("source", "development", "target"))
    dev_partition = (fixed_partition(development["y"], development["sample_ids"], query_fraction,
                                    minimum_query_per_class, seed + 1)
                     if development_partition is None else checked_partition(development_partition, len(development["y"])))
    final_partition = (fixed_partition(target["y"], target["sample_ids"], query_fraction,
                                      minimum_query_per_class, seed + 2)
                       if target_partition is None else checked_partition(target_partition, len(target["y"])))
    pipeline, source_metadata = fit_source(source, modality=modality, model=model, selector=selector,
        cv_splits=source_cv_splits, cars_cv=cars_cv, seed=seed, grid=source_grid, memory=memory)
    X_source, _ = representation(source, modality)
    X_dev, _ = representation(development, modality)
    X_target, _ = representation(target, modality)
    choices, calibration_metadata = choose_anchor_regularization(pipeline, X_dev, development["y"],
        dev_partition, shots, repeats, regularizations, seed + 3)
    draws, unavailable = nested_support_draws(target["y"], final_partition, shots, repeats, seed + 4)
    classes = np.asarray(pipeline.classes_); query = final_partition["query"]
    baseline_prediction = score_predictions(pipeline, X_target[query])
    baseline_metrics = classification_metrics(target["y"][query], baseline_prediction, classes)
    results, prediction_rows = [], []

    def record(method, k, repeat, predicted, fit_metadata, seconds):
        metrics = classification_metrics(target["y"][query], predicted, classes)
        row = {"method": method, "shots_per_class": k, "repeat": repeat, "status": "pass",
               "metrics": metrics, "fit_and_predict_seconds": float(seconds), "fit": fit_metadata,
               "macro_f1_gain_vs_source": float(metrics["macro_f1"] - baseline_metrics["macro_f1"]),
               "balanced_accuracy_gain_vs_source": float(metrics["balanced_accuracy"] - baseline_metrics["balanced_accuracy"])}
        results.append(row)
        for index, prediction in zip(query, predicted):
            prediction_rows.append({"method": method, "shots_per_class": k, "repeat": repeat,
                "sample_id": str(target["sample_ids"][index]), "y_true": str(target["y"][index]),
                "y_predicted": str(prediction), "input_record_ids": list(target["input_record_ids"][index])})

    record("source_only", 0, 0, baseline_prediction,
           {"training_scope": "source_only", "transforms_refit": False, "training_samples": len(X_source)}, 0.)
    for k, reason in sorted(unavailable.items()):
        for method in methods:
            results.append({"method": method, "shots_per_class": k, "status": "not_evaluable", "reason": reason})
    manifest_draws = []
    for draw in draws:
        support, k, repeat = draw["support"], draw["shots_per_class"], draw["repeat"]
        manifest_draws.append({"repeat": repeat, "shots_per_class": k,
                               "support_sample_ids": target["sample_ids"][support].tolist()})
        for method in methods:
            if method == "anchor_scores" and k not in choices:
                results.append({"method": method, "shots_per_class": k, "repeat": repeat,
                    "status": "not_evaluable", "reason": "No feasible development episode for selecting anchor strength at this K"})
                continue
            started = time.monotonic()
            try:
                fitted, metadata = fit_adaptation(method, pipeline, X_source, source["y"],
                    X_target[support], target["y"][support], regularization=choices.get(k), seed=seed)
                predicted = np.asarray(fitted.predict(X_target[query]))
                record(method, k, repeat, predicted, metadata, time.monotonic() - started)
            except (ValueError, np.linalg.LinAlgError) as exc:
                results.append({"method": method, "shots_per_class": k, "repeat": repeat,
                                "status": "fail", "error": str(exc)})
    summaries = []
    for method, k in sorted({(row["method"], row["shots_per_class"]) for row in results}):
        passed = [row for row in results if row["method"] == method and row["shots_per_class"] == k and row["status"] == "pass"]
        if not passed:
            continue
        summaries.append({"method": method, "shots_per_class": k, "evaluations": len(passed),
            "macro_f1_mean": float(np.mean([r["metrics"]["macro_f1"] for r in passed])),
            "macro_f1_sd": float(np.std([r["metrics"]["macro_f1"] for r in passed])),
            "balanced_accuracy_mean": float(np.mean([r["metrics"]["balanced_accuracy"] for r in passed])),
            "macro_f1_gain_mean": float(np.mean([r["macro_f1_gain_vs_source"] for r in passed]))})
    return {"status": "completed", "task": "known_class_cross_batch_classification",
        "source": source_metadata, "anchor_selection": calibration_metadata,
        "selected_anchor_regularization_by_shots": choices, "classes": classes.tolist(),
        "configuration": {"modality": modality, "model": model, "selector": selector,
            "shots_per_class": list(shots), "repeats": repeats, "query_fraction": query_fraction,
            "minimum_query_per_class": minimum_query_per_class, "source_cv_splits": source_cv_splits,
            "cars_internal_cv": cars_cv, "seed": seed},
        "domains": {role: {"physical_samples": len(data["y"]),
                              "batch_ids": np.unique(data["batch_ids"]).tolist()}
                    for role, data in domains.items()},
        "partitions": {"development_query_ids": development["sample_ids"][dev_partition["query"]].tolist(),
                       "development_support_pool_ids": development["sample_ids"][dev_partition["support_pool"]].tolist(),
                       "target_query_ids": target["sample_ids"][query].tolist(),
                       "target_support_pool_ids": target["sample_ids"][final_partition["support_pool"]].tolist(),
                       "target_support_draws": manifest_draws},
        "uncertainty_scope": "Support-draw SD on one fixed query is descriptive, not an independent-batch confidence interval",
        "target_query_policy": "No target-query labels used to fit, tune or select adaptation; no unlabelled query distribution fitting",
        "rows": results, "summary": summaries, "prediction_rows": prediction_rows}
