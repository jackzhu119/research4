"""Guarded building blocks for the separately registered supplemental study.

This module does not discover workbooks or create partitions. It accepts the
protocol preparer's numeric packs, reuses the historical selector, and keeps
prediction independent of the truth file. Supplemental results remain
exploratory/retrospective unless a separate independent study establishes more.
"""
from dataclasses import dataclass
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .calibration import AnchorScoreCalibrator
from .classification import classification_metrics
from .image_feature_models import ScatterPreprocessor
from .leakage_guard import (FitScope, LeakageError, array_fingerprint,
                            instrument_mcuve_pls, source_fingerprint)
from .paper_c_calibration import fit_limited, candidate_grid as limited_grid
from .selection import MCUVESelector
from .spectral_models import SpectralPreprocessor
from .staged_nested_search import class_weight_spec, gamma_spec, retained_count


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def write_new_json(path, value):
    """Refuse accidental replacement of a protocol, prediction or result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _vector(pack, key, n):
    values = np.asarray(pack[key])
    if values.shape != (n,) or values.dtype.kind not in "US":
        raise LeakageError(f"{key} must be a Unicode/string vector matching rows")
    if any(not str(v).strip() for v in values):
        raise LeakageError(f"{key} contains a missing value")
    return values.astype(str)


def validate_pack(pack, *, role, expected_axis_sha256=None):
    required = {"X", "axis", "record_ids", "entity_ids", "batch_ids", "legacy_match_ids"}
    if not required.issubset(pack):
        raise LeakageError("Numeric role pack lacks required provenance fields")
    if any(np.asarray(v).dtype.kind == "O" for v in pack.values()):
        raise LeakageError("Object/pickled arrays are forbidden")
    x = np.asarray(pack["X"])
    axis = np.asarray(pack["axis"])
    if (x.ndim != 2 or not len(x) or x.dtype.kind not in "fiu" or x.shape[1] < 2
            or not np.isfinite(x).all() or axis.dtype.kind not in "fiu"
            or axis.shape != (x.shape[1],) or not np.isfinite(axis).all()
            or not np.all(np.diff(axis) > 0)):
        raise LeakageError("Invalid finite samples-by-frequency matrix or increasing axis")
    ids = _vector(pack, "record_ids", len(x))
    entities = _vector(pack, "entity_ids", len(x))
    _vector(pack, "batch_ids", len(x))
    legacy = _vector(pack, "legacy_match_ids", len(x))
    if len(set(ids)) != len(ids) or len(set(entities)) != len(entities):
        raise LeakageError("Canonical record/entity identifiers are not unique")
    for item in legacy:
        try:
            decoded = json.loads(str(item))
        except (ValueError, TypeError) as exc:
            raise LeakageError("legacy_match_ids must contain JSON lists") from exc
        if not isinstance(decoded, list) or any(not isinstance(v, str) or not v for v in decoded):
            raise LeakageError("legacy_match_ids must contain JSON string lists")
    label_keys = set(pack) & {"y", "labels", "source_labels", "origin_group_ids"}
    if role == "query" and label_keys:
        raise LeakageError("Query prediction packs must not carry truth labels")
    if role != "query":
        if not ({"y", "labels"} & label_keys):
            raise LeakageError("Source/support labels are required")
        labels = _vector(pack, "y" if "y" in pack else "labels", len(x))
        if "y" in pack and "labels" in pack and not np.array_equal(labels, pack["labels"]):
            raise LeakageError("Pack label aliases disagree")
    axis_hash = array_fingerprint(axis)
    if expected_axis_sha256 is not None and axis_hash != expected_axis_sha256:
        raise LeakageError("Frequency axis differs from the frozen protocol")
    return {"role": role, "rows": len(x), "features": x.shape[1],
            "record_ids_sha256": array_fingerprint(ids), "axis_sha256": axis_hash}


def load_role_pack(path, *, role, expected_sha256=None, expected_axis_sha256=None):
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise LeakageError("Numeric pack bytes differ from the frozen protocol")
    with np.load(path, allow_pickle=False) as archive:
        pack = {k: archive[k] for k in archive.files}
    validate_pack(pack, role=role, expected_axis_sha256=expected_axis_sha256)
    return pack


def pack_labels(pack):
    return np.asarray(pack["y"] if "y" in pack else pack["labels"]).astype(str)


def preflight_role_packs(packs):
    if set(packs) != {"source", "support", "query"}:
        raise LeakageError("Exactly source/support/query role packs are required")
    reports = {role: validate_pack(pack, role=role) for role, pack in packs.items()}
    axis = packs["source"]["axis"]
    classes = np.unique(pack_labels(packs["source"]))
    if len(classes) < 2:
        raise LeakageError("At least two source classes are required")
    observed = {key: {} for key in ("record_ids", "entity_ids", "legacy", "alpha_row")}
    for role, pack in packs.items():
        if not np.array_equal(axis, pack["axis"]):
            raise LeakageError("Role packs have different physical frequency axes")
        if role != "query" and set(pack_labels(pack)) - set(classes):
            raise LeakageError("Support vocabulary contains a class absent from source")
        for i, row in enumerate(pack["X"]):
            tokens = {"record_ids": [str(pack["record_ids"][i])],
                      "entity_ids": [str(pack["entity_ids"][i])],
                      "legacy": json.loads(str(pack["legacy_match_ids"][i])),
                      "alpha_row": [array_fingerprint(np.asarray(row, dtype=np.float64))]}
            for kind, values in tokens.items():
                for token in values:
                    if token in observed[kind]:
                        previous = observed[kind][token]
                        if previous != (role, i):
                            raise LeakageError(f"Repeated {kind} identity in {previous[0]} and {role}")
                    observed[kind][token] = (role, i)
    return {"status": "PASS_API_PREFLIGHT", "roles": reports, "classes": classes.tolist(),
            "confirmation_eligible": False,
            "interpretation": "EXPLORATORY; API checks do not establish biological independence",
            "support_counts": {str(c): int(np.sum(pack_labels(packs["support"]) == c)) for c in classes}}


@dataclass
class SourceTransform:
    preprocessor: object
    scaler: object
    selector: object
    metadata: dict

    def matrix(self, X, ratio=None):
        z = self.scaler.transform(self.preprocessor.transform(np.asarray(X, dtype=float)))
        if self.selector is not None:
            indices = self.selector.ranking_[:retained_count(z.shape[1], ratio)]
            return z[:, indices]
        return z


@dataclass
class SourceModel:
    transform: SourceTransform
    estimator: object
    config: dict
    metadata: dict

    @property
    def classes_(self):
        return self.estimator.classes_

    def matrix(self, X):
        return self.transform.matrix(X, (self.config.get("mc") or {}).get("ratio"))

    def predict(self, X):
        # Preserve SVC's historical voting rule; it can differ from OVR argmax.
        return self.estimator.predict(self.matrix(X))

    def scores(self, X):
        z = self.matrix(X)
        scores = (self.estimator.predict_proba(z) if self.config["family"] == "mlp"
                  else self.estimator.decision_function(z))
        scores = np.asarray(scores, dtype=float)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        if scores.shape != (len(z), len(self.classes_)) or not np.isfinite(scores).all():
            raise LeakageError("Model scores differ from its frozen source vocabulary")
        return scores


def fit_source_model(config, X, y, axis, identifiers, *, seed, dataset_version,
                     split_id, log_path=None, code_version, physical_quantity,
                     cache=None):
    """Source-only API: no support/query arrays, labels, or paths are accepted.

    A cache is scoped by actual ordered source values/labels, IDs, axis and seed.
    MC-UVE computes one original full ranking; fixed retained ratios only slice
    that ranking. Every underlying legacy PLS fit is instrumented.
    """
    started = time.perf_counter()
    x, labels = np.asarray(X, dtype=np.float64), np.asarray(y).astype(str)
    ids, axis = np.asarray(identifiers).astype(str), np.asarray(axis, dtype=float)
    if (x.ndim != 2 or labels.shape != (len(x),) or ids.shape != (len(x),)
            or len(set(ids)) != len(ids) or len(np.unique(labels)) < 2
            or axis.shape != (x.shape[1],) or not np.isfinite(x).all()
            or not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0)):
        raise LeakageError("Invalid source-only fit inputs")
    config = json.loads(json.dumps(config, allow_nan=False))
    if not isinstance(physical_quantity, str) or not physical_quantity.strip():
        raise LeakageError("Physical quantity and unit verification status must be explicit")
    family = config["family"]
    if family not in {"mcuve_svm", "svm", "shrinkage_lda", "mlp"}:
        raise LeakageError("Unregistered model family")
    scope = FitScope(ids, split_id, dataset_version, log_path, code_version=code_version)
    mc = config.get("mc")
    if (family == "mcuve_svm") != (mc is not None):
        raise LeakageError("MC-UVE configuration disagrees with classifier family")
    cache = {} if cache is None else cache
    identity = {"source": source_fingerprint(x, labels), "ids": array_fingerprint(ids),
                "axis": array_fingerprint(axis), "seed": int(seed), "split_id": split_id,
                "dataset": dataset_version, "code": code_version,
                "physical_quantity": physical_quantity,
                "preprocessing": config["preprocessing"],
                "mc": None if mc is None else {k: mc[k] for k in ("runs", "components")}}
    key = object_sha256(identity)
    reused = key in cache
    if reused:
        transform, state_hash = cache[key]
        if joblib.hash(transform) != state_hash or transform.metadata["cache_identity"] != identity:
            raise LeakageError("Cached source transform changed")
        scope._record("cache_reused", component="source_transform", cache_identity=identity,
                      fit_sample_ids=ids.tolist())
    else:
        mode = config["preprocessing"]
        prep = (scope.fit(ScatterPreprocessor("msc"), x, sample_ids=ids, component="MSC")
                if mode == "msc" else SpectralPreprocessor(mode, axis))
        prepared = prep.transform(x)
        scale = scope.fit(StandardScaler(), prepared, sample_ids=ids, component="StandardScaler")
        z = scale.transform(prepared)
        selector = None
        if mc is not None:
            selector = MCUVESelector(n_runs=int(mc["runs"]), n_components=int(mc["components"]),
                n_features_to_select=z.shape[1], sample_fraction=.8,
                task="classification", random_state=int(seed))
            with instrument_mcuve_pls(scope, z, ids):
                scope.fit(selector, z, labels, sample_ids=ids, component="MC-UVE.original_ranking")
        transform = SourceTransform(prep, scale, selector, {"cache_identity": identity})
        cache[key] = (transform, joblib.hash(transform))
    z = transform.matrix(x, None if mc is None else mc["ratio"])
    if family in {"mcuve_svm", "svm"}:
        weight, weights = class_weight_spec(labels, config.get("class_weight", "balanced"))
        gamma, _ = gamma_spec(z, config["gamma"])
        estimator = SVC(C=float(config["C"]), gamma=gamma, kernel="rbf", class_weight=weight,
                        break_ties=False, decision_function_shape="ovr", random_state=int(seed), cache_size=64)
        fit_kwargs = {}
    elif family == "shrinkage_lda":
        priors = (np.full(len(np.unique(labels)), 1 / len(np.unique(labels)))
                  if config.get("priors", "empirical") == "uniform" else None)
        estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=config["shrinkage"], priors=priors)
        fit_kwargs, weights = {}, None
    else:
        epochs = int(config.get("max_iter", 300))
        estimator = MLPClassifier(hidden_layer_sizes=tuple(config["hidden"]), alpha=float(config["alpha"]),
            solver="adam", learning_rate_init=.001, batch_size=64, max_iter=epochs,
            early_stopping=False, tol=0., n_iter_no_change=epochs + 1, random_state=int(seed))
        fit_kwargs, weights = {}, None
        if config.get("class_weight", "none") == "balanced":
            _, weights = class_weight_spec(labels, "balanced")
            fit_kwargs["sample_weight"] = np.array([weights[str(c)] for c in labels])
    before_fit = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scope.fit(estimator, z, labels, sample_ids=ids, component=family, **fit_kwargs)
    selected = (np.arange(z.shape[1]) if mc is None
                else transform.selector.ranking_[:retained_count(len(axis), mc["ratio"])])
    metadata = {"config": config, "seed": int(seed), "source_ids": ids.tolist(),
        "allowed_fit_ids": list(scope.allowed_ids), "source_fingerprint": source_fingerprint(x, labels),
        "axis_sha256": array_fingerprint(axis), "code_version": code_version,
        "physical_quantity": physical_quantity,
        "selected_indices": selected.tolist(), "selected_frequencies": axis[selected].tolist(),
        "selected_features": len(selected), "transform_cache_reused": reused,
        "class_order": estimator.classes_.tolist(), "class_weights": weights,
        "prediction_rule": "native estimator.predict; SVC standard vote, break_ties=False",
        "score_kind": "probability" if family == "mlp" else "uncalibrated_decision_score",
        "estimator_fit_seconds": time.perf_counter() - before_fit,
        "fit_total_seconds": time.perf_counter() - started,
        "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught]}
    if hasattr(estimator, "_gamma"):
        metadata["effective_gamma"] = float(estimator._gamma)
    if hasattr(estimator, "n_iter_"):
        metadata["actual_iterations"] = np.asarray(estimator.n_iter_).tolist()
    if hasattr(estimator, "loss_curve_"):
        metadata["training_loss_curve"] = list(estimator.loss_curve_)
    return SourceModel(transform, estimator, config, metadata)


def predict_numeric(model, X, record_ids):
    """No ground truth accepted; ensure inference is pointwise and immutable."""
    x, ids = np.asarray(X, dtype=float), np.asarray(record_ids).astype(str)
    if len(x) != len(ids) or len(set(ids)) != len(ids):
        raise LeakageError("Inference IDs must uniquely match numeric rows")
    before = joblib.hash(model)
    predicted, scores = model.predict(x), model.scores(x)
    if len(x) > 1:
        reversed_scores = model.scores(x[::-1])[::-1]
        first = model.scores(x[:1])
        if not np.allclose(scores, reversed_scores, rtol=1e-10, atol=1e-10):
            raise LeakageError("Inference depends on query ordering")
        if not np.allclose(scores[:1], first, rtol=1e-10, atol=1e-10):
            raise LeakageError("Inference depends on query batch statistics")
    if before != joblib.hash(model):
        raise LeakageError("Inference changed fitted model state")
    return {"record_ids": ids, "predictions": predicted, "scores": scores,
            "classes": model.classes_.copy(),
            "native_vote_vs_argmax_disagreements": int(np.sum(predicted != model.classes_[scores.argmax(1)]))}


def apply_changed_scores(base_predictions, scores, corrected, classes):
    """A zero correction preserves native SVC votes, including OVR ties."""
    before, after = np.asarray(scores, dtype=float), np.asarray(corrected, dtype=float)
    base, classes = np.asarray(base_predictions), np.asarray(classes)
    if (before.ndim != 2 or after.shape != before.shape or before.shape[1] != len(classes)
            or base.shape != (len(before),) or not np.isfinite(after).all()):
        raise LeakageError("Correction dimensions or values are invalid")
    changed = np.any(after != before, axis=1)
    predictions = base.copy()
    predictions[changed] = classes[after[changed].argmax(1)]
    return predictions


def calibrate_scores(method, source_scores, classes, query_scores, query_native,
                     *, support_scores=None, support_labels=None, support_native=None,
                     limited_candidate=None, fit_scope=None, support_ids=None):
    """Calibration accepts support truth only, never query truth.

    Parameters are registered before query prediction. Missing/insufficient
    evidence is an explicit no-op, rather than a claim of calibration benefit.
    """
    if method not in {"none", "traditional", "conservative", "zero_label_control"}:
        raise LeakageError("Unknown registered calibration method")
    scores, classes = np.asarray(query_scores, dtype=float), np.asarray(classes)
    native_query = np.asarray(query_native)
    reference = np.asarray(source_scores, dtype=float)
    if (classes.ndim != 1 or len(classes) < 2 or len(set(classes)) != len(classes)
            or scores.ndim != 2 or not len(scores) or scores.shape[1] != len(classes)
            or not np.isfinite(scores).all() or native_query.shape != (len(scores),)
            or set(native_query) - set(classes) or reference.ndim != 2
            or not len(reference) or reference.shape[1] != len(classes)
            or not np.isfinite(reference).all()):
        raise LeakageError("Invalid frozen class vocabulary/source/query scores")
    result = {"method": method, "predictions": np.asarray(query_native).copy(),
              "scores": scores.copy(), "status": "NO_OP", "reason": "no_calibration"}
    if method == "none":
        return result
    if support_scores is None or not len(support_scores):
        result["reason"] = "zero_support"
        return result
    if support_labels is None or method == "zero_label_control":
        result["reason"] = "no_support_labels"
        return result
    support, labels = np.asarray(support_scores, dtype=float), np.asarray(support_labels)
    if support.shape != (len(labels), len(classes)) or not np.isfinite(support).all():
        raise LeakageError("Invalid support scores/labels")
    if set(labels) - set(classes):
        raise LeakageError("Support has unknown classes")
    if set(labels) != set(classes):
        result["reason"] = "incomplete_support_class_coverage"
        return result
    if fit_scope is not None and (support_ids is None or len(support_ids) != len(support)):
        raise LeakageError("Calibrator fits need explicit support IDs in their fit scope")
    operation = (fit_scope.operation(method + ".support_only_fit", support_ids, X=support, y=labels)
                 if fit_scope is not None else nullcontext())
    if method == "traditional":
        with operation:
            adapter = AnchorScoreCalibrator(classes, regularization=1.).fit(support, labels)
        corrected = adapter.transform(scores)
    else:
        if limited_candidate not in limited_grid():
            raise LeakageError("Conservative parameters must be from the registered finite grid")
        if support_native is None:
            raise LeakageError("Conservative support evidence needs native base predictions")
        native = np.asarray(support_native)
        if native.shape != labels.shape:
            raise LeakageError("Native support predictions are misaligned")
        errors = int(np.sum(native != labels))
        if errors < limited_candidate["minimum_support_errors"]:
            result.update(reason="insufficient_native_support_error_evidence", support_error_count=errors)
            return result
        with operation:
            adapter = fit_limited(source_scores, classes, limited_candidate, support, labels)
        if adapter.ridge is None:
            result.update(reason=adapter.rejection_reason, support_error_count=errors)
            return result
        corrected = adapter.predict_scores(scores)
        result.update(cap=adapter.cap, gate_margin=adapter.gate_margin,
                      support_error_count=errors, applied_blend=limited_candidate["blend"])
    result.update(status="APPLIED", reason="", scores=corrected,
                  predictions=apply_changed_scores(query_native, scores, corrected, classes),
                  changed_score_rows=int(np.sum(np.any(corrected != scores, axis=1))),
                  max_absolute_score_correction=float(np.max(np.abs(corrected - scores))))
    return result


def score_predictions(truth_ids, truth_labels, prediction, *, baseline=None):
    """Separate scoring boundary: no fit/model object accepted here."""
    ids, truth = np.asarray(truth_ids).astype(str), np.asarray(truth_labels).astype(str)
    expected = np.asarray(prediction["record_ids"]).astype(str)
    if (ids.ndim != 1 or expected.ndim != 1 or len(set(ids)) != len(ids)
            or len(expected) != len(ids) or len(set(expected)) != len(expected)
            or set(ids) != set(expected) or truth.shape != ids.shape):
        raise LeakageError("Truth identity does not match the frozen prediction")
    lookup = dict(zip(ids, truth, strict=True))
    truth = np.array([lookup[i] for i in expected])
    pred, classes = np.asarray(prediction["predictions"]), np.asarray(prediction["classes"])
    if pred.shape != truth.shape or classes.ndim != 1 or len(set(classes)) != len(classes):
        raise LeakageError("Prediction dimensions/class vocabulary are invalid")
    if set(truth) - set(classes) or set(pred) - set(classes):
        raise LeakageError("Prediction or truth includes unknown classes")
    metrics = classification_metrics(truth, pred, classes)
    p, r, f, n = precision_recall_fscore_support(truth, pred, labels=classes, zero_division=0)
    metrics["per_class"] = {str(c): {"precision": float(a), "recall": float(b),
                                     "f1": float(d), "support": int(e)}
                             for c, a, b, d, e in zip(classes, p, r, f, n, strict=True)}
    metrics.update(total=len(truth), correct=int(np.sum(truth == pred)))
    if baseline is not None:
        base = np.asarray(baseline)
        if base.shape != truth.shape:
            raise LeakageError("Baseline predictions are misaligned")
        metrics.update(corrected_errors=int(np.sum((base != truth) & (pred == truth))),
                       introduced_errors=int(np.sum((base == truth) & (pred != truth))),
                       changed_predictions=int(np.sum(base != pred)))
    return metrics
