"""Reviewed sample-level classification and shared few-shot pipeline helpers.

Record identifiers are never substituted for missing physical-sample groups.
The caller must establish the scientific meaning of the supplied annotations.
All supervised transformations are fitted within each training partition.
"""
import csv
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, recall_score)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_array, check_is_fitted

from .benchmark import PLSDAClassifier
from .selection import CARSStyleSelector, MCUVESelector
from .stability import StableVariableSelector


REQUIRED_ARRAYS = ("X_alpha", "X_n", "axis", "record_ids", "y", "groups",
                   "batch_ids", "labels_verified")
DEFAULT_METHODS = ("dummy", "svc", "plsda", "mcuve20_svc", "cars_svc")
METHODS = {"dummy": ("dummy", "full"), "svc": ("svc", "full"),
           "plsda": ("plsda", "full"), "mcuve20_svc": ("svc", "mcuve20"),
           "cars_svc": ("svc", "cars"), "mcuve100_svc": ("svc", "mcuve100"),
           "stable_svc": ("svc", "stable")}


class DatasetValidationError(ValueError):
    def __init__(self, issues):
        self.issues = tuple(issues)
        super().__init__("; ".join(self.issues))


def _missing(value):
    return str(value).strip().lower() in {"", "nan", "none", "null"}


def validate_dataset(arrays):
    """Return normalized arrays or reject missing/unverified annotations.

    A documented user statement may establish a one-record/one-specimen mapping,
    so groups may equal record_ids. Missing groups are never filled from IDs.
    Unknown batch IDs may be blank for within-dataset CV, never for transfer.
    """
    missing = [name for name in REQUIRED_ARRAYS if name not in arrays]
    if missing:
        raise DatasetValidationError(["Missing required arrays: " + ", ".join(missing)])
    result, issues = {}, []
    for name in ("X_alpha", "X_n"):
        try:
            result[name] = check_array(arrays[name], dtype=float, ensure_min_samples=2,
                                       ensure_min_features=2)
        except (TypeError, ValueError) as exc:
            issues.append(f"{name} must be a finite records-by-frequency matrix: {exc}")
    if issues:
        raise DatasetValidationError(issues)
    n, p = result["X_alpha"].shape
    if result["X_n"].shape != (n, p):
        issues.append("X_n and X_alpha must have the same paired record and feature shape")
    try:
        axis = np.asarray(arrays["axis"], dtype=float)
        if axis.shape != (p,) or not np.isfinite(axis).all() or not (np.diff(axis) > 0).all():
            issues.append("axis must be finite, strictly increasing, and match spectral columns")
        result["axis"] = axis
    except (TypeError, ValueError):
        issues.append("axis must be a numeric one-dimensional frequency vector")
    for name in ("record_ids", "y", "groups", "batch_ids"):
        values = np.asarray(arrays[name])
        if values.ndim != 1 or len(values) != n or values.dtype.kind not in "US":
            issues.append(f"{name} requires one Unicode/string value per record")
            continue
        values = np.asarray([str(v).strip() for v in values], dtype=str)
        if name != "batch_ids" and any(_missing(v) for v in values):
            message = ("Missing physical-sample groups; supply verified groups, not an ID fallback"
                       if name == "groups" else f"Missing {name}")
            issues.append(message)
        if name == "batch_ids":
            values = np.asarray(["" if _missing(v) else v for v in values])
        result[name] = values
    verified = np.asarray(arrays["labels_verified"])
    if (verified.dtype != np.dtype(bool)
            or verified.shape not in [(), (n,)] or not verified.all()):
        issues.append("labels_verified must be explicit Boolean true, scalar or one per record")
    result["labels_verified"] = np.ones(n, dtype=bool)
    if issues:
        raise DatasetValidationError(issues)
    if len(set(result["record_ids"].tolist())) != n:
        issues.append("record_ids must be unique")
    if len(np.unique(result["y"])) < 2:
        issues.append("Classification requires at least two origin groups")
    for group in np.unique(result["groups"]):
        mask = result["groups"] == group
        if len(np.unique(result["y"][mask])) != 1:
            issues.append(f"Physical sample {group} has conflicting origin labels")
        known_batches = {v for v in result["batch_ids"][mask] if v}
        if len(known_batches) > 1:
            issues.append(f"Physical sample {group} has conflicting known batch IDs")
    if issues:
        raise DatasetValidationError(issues)
    return result


def load_reviewed_dataset(path):
    """Load numeric/string NPZ data without pickle, then enforce review gates."""
    try:
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
    except (OSError, ValueError) as exc:
        raise DatasetValidationError([f"Cannot load reviewed NPZ: {exc}"]) from exc
    return validate_dataset(arrays)


def aggregate_samples(arrays):
    """Average repeat spectra within supplied physical groups before any split."""
    data = validate_dataset(arrays)
    samples = np.unique(data["groups"])
    alpha, refractive, labels, batches, records, counts, provenance = [], [], [], [], [], [], []
    for group in samples:
        mask = data["groups"] == group
        alpha.append(data["X_alpha"][mask].mean(axis=0))
        refractive.append(data["X_n"][mask].mean(axis=0))
        labels.append(data["y"][mask][0])
        known = sorted({v for v in data["batch_ids"][mask] if v})
        batches.append(known[0] if known else "")
        identifiers = tuple(data["record_ids"][mask].tolist())
        records.append(identifiers[0]); counts.append(len(identifiers)); provenance.append(identifiers)
    return {"X_alpha": np.asarray(alpha), "X_n": np.asarray(refractive),
            "axis": data["axis"].copy(), "record_ids": np.asarray(records),
            "y": np.asarray(labels), "groups": samples, "sample_ids": samples.copy(),
            "batch_ids": np.asarray(batches), "labels_verified": np.ones(len(samples), dtype=bool),
            "scan_counts": np.asarray(counts), "input_record_ids": provenance}


def representation(data, name="alpha"):
    if name == "alpha":
        X = np.asarray(data["X_alpha"])
        return X, (X.shape[1],)
    if name == "n":
        X = np.asarray(data["X_n"])
        return X, (X.shape[1],)
    if name == "concat":
        n, alpha = np.asarray(data["X_n"]), np.asarray(data["X_alpha"])
        return np.column_stack([n, alpha]), (n.shape[1], alpha.shape[1])
    raise ValueError("representation must be alpha, n, or concat")


class BlockRowPreprocessor(TransformerMixin, BaseEstimator):
    """Raw spectra or per-row SNV performed separately within each modality."""
    def __init__(self, mode="raw", block_sizes=None):
        self.mode = mode
        self.block_sizes = block_sizes

    def fit(self, X, y=None):
        X = check_array(X)
        if self.mode not in {"raw", "snv"}:
            raise ValueError("mode must be raw or snv")
        sizes = (X.shape[1],) if self.block_sizes is None else tuple(self.block_sizes)
        if (not sizes or any(not isinstance(v, (int, np.integer)) or v < 2 for v in sizes)
                or sum(sizes) != X.shape[1]):
            raise ValueError("block_sizes must cover the features with blocks of at least two points")
        self.n_features_in_ = X.shape[1]
        self.block_sizes_ = sizes
        return self

    def transform(self, X):
        check_is_fitted(self, ["n_features_in_", "block_sizes_"])
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature grid differs from training")
        result = X.copy()
        if self.mode == "snv":
            start = 0
            for width in self.block_sizes_:
                block = result[:, start:start + width]
                sd = block.std(axis=1, ddof=1, keepdims=True)
                if np.any(sd <= np.finfo(float).eps) or not np.isfinite(sd).all():
                    raise ValueError("SNV is undefined for a constant modality spectrum")
                result[:, start:start + width] = (block - block.mean(axis=1, keepdims=True)) / sd
                start += width
        return result


def build_pipeline(model="svc", selector="full", block_sizes=None, random_state=42,
                   cars_cv=3, memory=None):
    if selector == "full":
        selection = "passthrough"
    elif selector in {"mcuve20", "mcuve100", "stable"}:
        if block_sizes is None:
            raise ValueError("Provide block_sizes so the selector feature budget is explicit")
        features = sum(block_sizes)
        retained = min(features, max(5, int(np.ceil(.1 * features))))
        if selector == "stable":
            selection = StableVariableSelector(base_selector="mcuve", n_features_to_select=retained,
                n_resamples=10, sample_fraction=.8, inner_runs=10, n_components=2,
                task="classification", random_state=random_state)
        else:
            selection = MCUVESelector(n_runs=20 if selector == "mcuve20" else 100,
                n_components=2, n_features_to_select=retained, task="classification",
                random_state=random_state)
    elif selector == "cars":
        selection = CARSStyleSelector(n_runs=12, n_components=2, min_features=2,
            sample_fraction=.8, cv=cars_cv, task="classification", random_state=random_state)
    else:
        raise ValueError("Unknown selector")
    if model == "svc":
        estimator = SVC(class_weight="balanced", gamma="scale")
    elif model == "plsda":
        estimator = PLSDAClassifier()
    elif model == "dummy":
        if selector != "full":
            raise ValueError("The dummy reference does not use supervised selection")
        estimator = DummyClassifier(strategy="most_frequent")
    else:
        raise ValueError("model must be svc, plsda, or dummy")
    return Pipeline([("preprocess", BlockRowPreprocessor(block_sizes=block_sizes)),
                     ("scale", StandardScaler()), ("selector", selection),
                     ("model", estimator)], memory=memory)


def parameter_grid(model, inner_folds, x_shape, selector="full"):
    grid = {"preprocess__mode": ["raw", "snv"]}
    if model == "svc":
        grid["model__C"] = [.1, 1., 10., 100.]
    elif model == "plsda":
        bound = min(x_shape[1], min(len(train) - 1 for train, _ in inner_folds))
        if selector == "cars":
            bound = min(bound, 2)
        elif selector in {"mcuve20", "mcuve100", "stable"}:
            bound = min(bound, max(5, int(np.ceil(.1 * x_shape[1]))))
        grid["model__n_components"] = [v for v in [1, 2, 3, 5, 8] if v <= bound]
        if not grid["model__n_components"]:
            raise ValueError("No feasible PLS-DA latent-variable candidate")
    elif model == "dummy":
        grid = {"preprocess__mode": ["raw"]}
    else:
        raise ValueError("Unknown model")
    return grid


def class_scores(fitted_pipeline, X):
    """Return scores in fitted_pipeline.classes_ order; these are not probabilities."""
    check_is_fitted(fitted_pipeline)
    classes = np.asarray(fitted_pipeline.classes_)
    if hasattr(fitted_pipeline, "decision_function"):
        values = np.asarray(fitted_pipeline.decision_function(X), dtype=float)
        if values.ndim == 1 and len(classes) == 2:
            values = np.column_stack([-values, values])
    elif hasattr(fitted_pipeline, "predict_proba"):
        values = np.asarray(fitted_pipeline.predict_proba(X), dtype=float)
    else:
        raise ValueError("Fitted classifier does not expose class-aligned scores")
    if values.shape != (len(X), len(classes)) or not np.isfinite(values).all():
        raise ValueError("Class-score shape or values are invalid")
    return values


def classification_metrics(y, prediction, classes=None):
    classes = np.unique(y) if classes is None else np.asarray(classes)
    recalls = recall_score(y, prediction, labels=classes, average=None, zero_division=0)
    return {"accuracy": float(accuracy_score(y, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, labels=classes, average="macro", zero_division=0)),
            "class_recall": {str(label): float(value) for label, value in zip(classes, recalls)},
            "class_counts": {str(label): int(np.count_nonzero(np.asarray(y) == label)) for label in classes},
            "confusion_matrix": confusion_matrix(y, prediction, labels=classes).tolist(),
            "class_order": classes.tolist()}


def sample_folds(y, groups, n_splits=3, seed=42):
    """Stratify already aggregated physical samples without splitting a group."""
    y, groups = np.asarray(y), np.asarray(groups)
    if (not isinstance(n_splits, (int, np.integer)) or isinstance(n_splits, bool)
            or n_splits < 2):
        raise ValueError("n_splits must be an integer of at least two")
    if len(y) != len(groups) or len(set(groups.tolist())) != len(groups):
        raise ValueError("sample_folds requires exactly one row per physical sample")
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() < n_splits:
        raise ValueError("Not enough independent samples per class for the configured folds")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result = list(splitter.split(np.zeros((len(y), 1)), y))
    for train, test in result:
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("A physical sample crosses a fold")
        if set(y[train]) != set(classes) or set(y[test]) != set(classes):
            raise ValueError("A configured fold is missing an origin class")
    return result


def _selector_preflight(selector, y, outer_folds, inner_folds, cars_cv):
    if selector == "cars" and (not isinstance(cars_cv, (int, np.integer))
                               or isinstance(cars_cv, bool) or cars_cv < 2):
        return "CARS internal CV must be explicitly configured as an integer >= 2"
    for number, ((train, _), inner) in enumerate(zip(outer_folds, inner_folds)):
        for inner_number, (inner_train, _) in enumerate(inner):
            labels = y[train][inner_train]
            if selector != "full" and len(labels) < 4:
                return f"Outer fold {number}, inner fold {inner_number}: selector needs >=4 training sample units"
            if selector == "cars":
                values, counts = np.unique(labels, return_counts=True)
                if counts.min() < cars_cv:
                    detail = {str(k): int(v) for k, v in zip(values, counts)}
                    return (f"Outer fold {number}, inner fold {inner_number}: CARS internal cv={cars_cv} "
                            f"requires at least {cars_cv} independent samples per class; counts={detail}")
    return None


def _write_json(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    default=convert, allow_nan=False) + "\n", encoding="utf-8")


def run_grouped_classification(arrays, *, representations=("alpha",), methods=DEFAULT_METHODS,
                               outer_splits=3, inner_splits=3, cars_cv=3, seed=42,
                               split_seed=20261031, output_dir=None, preflight_only=False,
                               identity_scope="user-declared physical-sample mapping; not independently established by code"):
    """Run a fixed shared outer/inner split matrix on physical-sample means.

    This evaluates within-dataset sample recognition. It does not hold out
    batches, establish independent sampling provenance, or test domain transfer.
    Insufficient configured folds/selector CV are reported, never reduced.
    """
    data = aggregate_samples(arrays)
    representations, methods = tuple(representations), tuple(methods)
    if (not representations or len(set(representations)) != len(representations)
            or not set(representations).issubset({"alpha", "n", "concat"})):
        raise ValueError("Select unique alpha/n/concat representations")
    if not methods or len(set(methods)) != len(methods) or not set(methods).issubset(METHODS):
        raise ValueError("Select unique supported classification methods")
    output = Path(output_dir) if output_dir is not None else None
    if output is not None:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Choose a fresh output directory; existing experiments are not overwritten")
        output.mkdir(parents=True, exist_ok=True)
    y, groups, classes = data["y"], data["groups"], np.unique(data["y"])
    report = {"status": "pending", "scope": "within-dataset physical-sample CV; not batch-held-out validation",
              "identity_scope": identity_scope,
              "aggregation": "mean of paired spectra per supplied physical sample before splitting",
              "input_records": len(arrays["y"]), "physical_sample_count": len(y),
              "class_order": classes.tolist(),
              "class_counts": {str(c): int(np.count_nonzero(y == c)) for c in classes},
              "configuration": {"representations": representations, "methods": methods,
                  "outer_splits": outer_splits, "inner_splits": inner_splits,
                  "cars_internal_cv": cars_cv, "model_seed": seed, "split_seed": split_seed,
                  "selection_scoring": "macro_f1", "preprocessing_candidates": ["raw", "snv"],
                  "svc_C": [.1, 1., 10., 100.], "svc_gamma": "scale",
                  "svc_class_weight": "balanced", "pls_components": [1, 2, 3, 5, 8],
                  "preflight_only": preflight_only}, "results": []}
    try:
        outer = sample_folds(y, groups, outer_splits, split_seed)
        inners = [sample_folds(y[train], groups[train], inner_splits, split_seed)
                  for train, _ in outer]
    except ValueError as exc:
        report.update(status="not_evaluable", reason=str(exc), models_not_run=True)
        if output is not None:
            _write_json(output / "summary.json", report)
        return report
    split_records = []
    for number, ((train, test), inner) in enumerate(zip(outer, inners)):
        split_records.append({"fold": number, "train_sample_ids": groups[train].tolist(),
            "test_sample_ids": groups[test].tolist(),
            "inner_folds": [{"train_sample_ids": groups[train][t].tolist(),
                             "validation_sample_ids": groups[train][v].tolist()} for t, v in inner]})
    report["split_file"] = "splits.json" if output is not None else None
    if output is not None:
        _write_json(output / "splits.json", split_records)
        _write_json(output / "sample_aggregation.json", [
            {"sample_id": str(group), "input_record_ids": records,
             "scan_count": int(count), "batch_id": str(batch)}
            for group, records, count, batch in zip(groups, data["input_record_ids"],
                                                  data["scan_counts"], data["batch_ids"])])
    for rep in representations:
        X, blocks = representation(data, rep)
        for method in methods:
            model_name, selector_name = METHODS[method]
            result = {"representation": rep, "method": method, "status": "pending",
                      "sample_count": len(y), "input_feature_count": X.shape[1],
                      "block_sizes": blocks, "folds": []}
            reason = _selector_preflight(selector_name, y, outer, inners, cars_cv)
            if reason is not None:
                result.update(status="not_evaluable", reason=reason, models_not_run=True)
                report["results"].append(result)
                continue
            if preflight_only:
                result.update(status="ready", models_not_run=True)
                report["results"].append(result)
                continue
            print(f"Running {rep}/{method}", flush=True)
            start = time.monotonic()
            prediction = np.empty(len(y), dtype=y.dtype)
            scores = np.empty((len(y), len(classes)), dtype=float)
            assignments = np.empty(len(y), dtype=int)
            case = output / f"{rep}_{method}" if output is not None else None
            if case is not None:
                case.mkdir()
            try:
                for number, ((train, test), inner) in enumerate(zip(outer, inners)):
                    print(f"  outer fold {number + 1}/{len(outer)}", flush=True)
                    cache = str(case / ".cache") if case is not None else None
                    pipeline = build_pipeline(model_name, selector_name, blocks, seed,
                                              cars_cv=cars_cv, memory=cache)
                    grid = parameter_grid(model_name, inner, X[train].shape, selector_name)
                    search = GridSearchCV(pipeline, grid, cv=inner, n_jobs=1,
                                          scoring="f1_macro", error_score="raise")
                    fit_start = time.monotonic()
                    with warnings.catch_warnings(record=True) as caught:
                        search.fit(X[train], y[train])
                    fit_seconds = time.monotonic() - fit_start
                    predict_start = time.monotonic()
                    prediction[test] = search.predict(X[test])
                    if not np.array_equal(search.best_estimator_.classes_, classes):
                        raise AssertionError("Fitted class order differs from the fixed vocabulary")
                    scores[test] = class_scores(search.best_estimator_, X[test])
                    predict_seconds = time.monotonic() - predict_start
                    assignments[test] = number
                    selector = search.best_estimator_.named_steps["selector"]
                    selected = (np.arange(X.shape[1]) if isinstance(selector, str)
                                else selector.selected_indices_)
                    record = {"fold": number, "best_params": search.best_params_,
                        "best_inner_macro_f1": float(search.best_score_),
                        "candidate_count": len(search.cv_results_["params"]),
                        "train_samples": len(train), "test_samples": len(test),
                        "selected_feature_count": len(selected),
                        "selected_feature_indices": np.asarray(selected).tolist(),
                        "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                        "metrics": classification_metrics(y[test], prediction[test], classes),
                        "warnings": sorted(set(f"{w.category.__name__}: {w.message}" for w in caught))}
                    if hasattr(selector, "mean_pairwise_jaccard_"):
                        record["training_resample_jaccard"] = float(selector.mean_pairwise_jaccard_)
                        record["stability_scope"] = "training resampling diagnostic; not batch robustness"
                    result["folds"].append(record)
                result.update(status="completed", metrics=classification_metrics(y, prediction, classes),
                              total_seconds=time.monotonic() - start,
                              mean_selected_features=float(np.mean([f["selected_feature_count"]
                                                                    for f in result["folds"]])))
                if case is not None:
                    with (case / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                        writer = csv.writer(handle, lineterminator="\n")
                        writer.writerow(["sample_id", "representative_record_id", "batch_id", "scan_count",
                                         "y_true", "y_predicted", "outer_fold", *["score_" + str(c) for c in classes]])
                        writer.writerows([str(groups[i]), str(data["record_ids"][i]), str(data["batch_ids"][i]),
                                          int(data["scan_counts"][i]), str(y[i]), str(prediction[i]),
                                          int(assignments[i]), *scores[i].tolist()] for i in range(len(y)))
                    _write_json(case / "result.json", result)
            except (ValueError, RuntimeError, AssertionError, np.linalg.LinAlgError) as exc:
                result.update(status="failed", reason=f"{type(exc).__name__}: {exc}",
                              total_seconds=time.monotonic() - start)
                if case is not None:
                    _write_json(case / "result.json", result)
            report["results"].append(result)
            if output is not None:
                _write_json(output / "summary.json", report)
    statuses = {row["status"] for row in report["results"]}
    report["status"] = ("ready" if preflight_only and statuses == {"ready"}
                        else "completed" if statuses == {"completed"}
                        else "partial" if statuses & {"completed", "ready"}
                        else "not_evaluable" if statuses == {"not_evaluable"} else "failed")
    report["models_not_run"] = preflight_only or not any(r["status"] == "completed" for r in report["results"])
    if output is not None:
        _write_json(output / "summary.json", report)
    return report
