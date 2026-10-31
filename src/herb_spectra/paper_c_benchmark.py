"""Paper-C same-fold classifier comparison and conditional R2 ablations.

Every new classifier is tuned separately inside each original outer training
partition. Historical R1/R2 choices are source-only conditional controls, not
new global hyperparameters selected from exposed OOF scores.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import gzip
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import warnings

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .classification import classification_metrics
from .spectral_models import SpectralPreprocessor
from .staged_nested_search import (
    class_weight_spec, digest, fit_transform_bundle, gamma_spec,
    object_digest, prepare_protocol_roles, retained_count, svm_grid, write_json,
)


FAMILIES = ("full_svm", "shrinkage_lda", "mlp")
CONTROLS = ("R2_reproduction", "R2_full_features", "R2_random_K",
            "R2_no_C_gamma_search", "R2_no_class_weight", "R2_fixed_MC20_LV2_K62")
SOURCE_FILES = ("src/herb_spectra/paper_c_benchmark.py", "scripts/run_paper_c_benchmark.py",
                "src/herb_spectra/staged_nested_search.py", "src/herb_spectra/selection.py",
                "src/herb_spectra/classification.py", "src/herb_spectra/spectral_models.py")


def utc():
    return datetime.now(timezone.utc).isoformat()


def grids():
    result = {"full_svm": [dict(id=c["id"], family="full_svm", **{k:c[k] for k in ("C", "gamma", "class_weight")}) for c in svm_grid()]}
    result["shrinkage_lda"] = [dict(id=f"lda{i:02d}", family="shrinkage_lda", shrinkage=s, priors=p)
        for i, (s, p) in enumerate((s, p) for s in ("auto", .01, .1, .5, 1.) for p in ("empirical", "uniform"))]
    result["mlp"] = [dict(id=f"mlp{i:02d}", family="mlp", hidden=h, alpha=a, class_weight=w)
        for i, (h, a, w) in enumerate((h, a, w) for h in (64, 128) for a in (.001, .1) for w in ("none", "balanced"))]
    return result


def random_indices(features, count, seed):
    if not 1 <= count <= features:
        raise ValueError("Random control count outside feature range")
    return np.sort(np.random.default_rng(seed).choice(features, count, replace=False))


@dataclass
class FittedBenchmark:
    candidate: dict
    preprocessor: object
    scaler: object
    indices: np.ndarray
    estimator: object
    metadata: dict
    mc_bundle: object = None

    def transform(self, X):
        if self.mc_bundle is not None:
            return self.mc_bundle.transform(X)
        return self.scaler.transform(self.preprocessor.transform(X))[:, self.indices]

    def predict(self, X):
        return self.estimator.predict(self.transform(X))

    def scores(self, X):
        z = self.transform(X)
        if self.candidate["family"] == "mlp":
            return self.estimator.predict_proba(z)
        return self.estimator.decision_function(z)


def fit_candidate(candidate, X, y, axis, seed=42, cache=None):
    """All fitted statistics come from X/y; no validation arrays accepted."""
    X, y, axis = np.asarray(X), np.asarray(y), np.asarray(axis)
    if X.ndim != 2 or len(y) != len(X) or len(axis) != X.shape[1] or not np.isfinite(X).all():
        raise ValueError("Invalid source matrix, labels or axis")
    cache = {} if cache is None else cache
    before = time.perf_counter()
    family, mc = candidate["family"], candidate.get("mc")
    if mc is not None:
        bundle, z, _ = fit_transform_bundle(X, y, X[:1], axis, "snv", mc, seed, cache)
        prep, scaler, indices = bundle.preprocessor, bundle.scaler, bundle.selector.selected_indices_
    else:
        key = ("full_transform", joblib.hash(X), joblib.hash(y))
        if key not in cache:
            prep = SpectralPreprocessor("snv", axis)
            scaler = StandardScaler()
            cache[key] = (prep, scaler, scaler.fit_transform(prep.transform(X)))
        prep, scaler, full = cache[key]
        indices = (random_indices(X.shape[1], candidate["random_K"], seed)
                   if "random_K" in candidate else np.arange(X.shape[1]))
        z, bundle = full[:, indices], None
    transform_seconds = time.perf_counter() - before
    classes = np.unique(y)
    extra = {}
    if family == "shrinkage_lda":
        priors = np.full(len(classes), 1 / len(classes)) if candidate["priors"] == "uniform" else None
        estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=candidate["shrinkage"], priors=priors)
    elif family == "mlp":
        estimator = MLPClassifier(hidden_layer_sizes=(candidate["hidden"],), activation="relu", solver="adam",
            alpha=candidate["alpha"], batch_size=64, learning_rate_init=.001, max_iter=200,
            shuffle=True, random_state=seed, early_stopping=False, tol=0., n_iter_no_change=201)
    else:
        weight, weights = class_weight_spec(y, candidate["class_weight"])
        gamma, numeric = gamma_spec(z, candidate["gamma"])
        estimator = SVC(C=candidate["C"], kernel="rbf", gamma=gamma, class_weight=weight,
            break_ties=False, decision_function_shape="ovr", random_state=seed)
        extra = {"class_weights":weights, "declared_gamma_numeric":numeric}
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if family == "mlp" and candidate["class_weight"] == "balanced":
            _, weights = class_weight_spec(y, "balanced")
            sample_weight = np.asarray([weights[str(c)] for c in y])
            estimator.fit(z, y, sample_weight=sample_weight)
            extra["class_weights"] = weights
        else:
            estimator.fit(z, y)
    fit_seconds = time.perf_counter() - start
    if not np.array_equal(estimator.classes_, classes):
        raise AssertionError("Fitted class vocabulary changed")
    extra.update({"effective_gamma":float(estimator._gamma)} if hasattr(estimator, "_gamma") else {})
    if hasattr(estimator, "n_support_"):
        extra.update({"support_vectors":int(estimator.n_support_.sum()), "support_vectors_per_class":estimator.n_support_.tolist()})
    if family == "shrinkage_lda":
        extra.update({"fitted_class_priors":estimator.priors_.tolist(), "coefficient_parameters":int(estimator.coef_.size+estimator.intercept_.size)})
    if family == "mlp":
        extra.update({"epochs":int(estimator.n_iter_), "training_loss_curve":list(estimator.loss_curve_),
                      "trainable_parameters":int(sum(a.size for a in estimator.coefs_ + estimator.intercepts_)),
                      "early_stopping":False, "max_iter":200, "training_sample_weight":"balanced" if candidate["class_weight"] == "balanced" else "none"})
    meta = {"candidate":candidate, "source_samples":len(X), "source_fingerprint":joblib.hash(X),
            "source_label_fingerprint":joblib.hash(y), "class_order":classes.tolist(),
            "source_class_counts":{str(c):int(np.count_nonzero(y==c)) for c in classes},
            "scaler_mean":scaler.mean_.tolist(), "scaler_variance":scaler.var_.tolist(),
            "selected_indices":indices.tolist(), "selected_frequencies":axis[indices].tolist(),
            "input_features":X.shape[1], "selected_features":len(indices), "model_seed":seed,
            "transform_seconds":transform_seconds, "estimator_fit_seconds":fit_seconds,
            "warnings":[{"category":w.category.__name__, "message":str(w.message)} for w in caught],
            "fit_scope":"current_training_partition_only", "query_statistics_fitted":False,
            "score_kind":"probability" if family == "mlp" else "uncalibrated_decision_score", **extra}
    return FittedBenchmark(dict(candidate), prep, scaler, indices.copy(), estimator, meta, bundle)


def metrics(y, prediction, classes):
    value = classification_metrics(y, prediction, classes)
    precision, recall, f1, support = precision_recall_fscore_support(y, prediction, labels=classes, zero_division=0)
    value["per_class"] = {str(c):{"precision":float(p), "recall":float(r), "f1":float(f), "support":int(n)}
                          for c, p, r, f, n in zip(classes, precision, recall, f1, support)}
    value["correct"] = int(np.sum(np.asarray(y) == prediction))
    value["total"] = len(y)
    return value


def select_candidate(rows, candidates):
    means = {}
    for candidate in candidates:
        found = [r for r in rows if r["candidate_id"] == candidate["id"]]
        if len(found) != 2 or any(r["status"] != "ok" for r in found):
            continue
        means[candidate["id"]] = {m:float(np.mean([r["metrics"][m] for r in found]))
                                  for m in ("macro_f1", "balanced_accuracy", "accuracy")}
    if not means:
        raise ValueError("No candidate completed both inner folds")
    # Fixed candidate ordering provides deterministic final tie resolution.
    selected = min((c for c in candidates if c["id"] in means),
        key=lambda c:(-means[c["id"]]["macro_f1"], -means[c["id"]]["balanced_accuracy"],
                      -means[c["id"]]["accuracy"], c["id"]))
    return {"selected":selected, "mean_inner_metrics":means[selected["id"]], "all_candidate_means":means,
            "selected_using_outer_test":False}


def controls(choice, features=612):
    original = {"family":"conditional_svm", **choice["svm"], "mc":choice["mc"]}
    result = {name:dict(original, id=name) for name in CONTROLS}
    result["R2_full_features"].pop("mc")
    result["R2_random_K"].pop("mc")
    result["R2_random_K"]["random_K"] = retained_count(features, choice["mc"]["ratio"])
    result["R2_no_C_gamma_search"].update(C=10., gamma="scale")
    result["R2_no_class_weight"]["class_weight"] = "none"
    result["R2_fixed_MC20_LV2_K62"]["mc"] = {"runs":20, "components":2, "ratio":.1}
    return result


def _manifest(root, protocol_path, protocol):
    return {"protocol_sha256":digest(protocol_path), "input_sha256":digest(root/protocol["input"]),
        "historical_selection_lock_sha256":digest(root/protocol["historical_selection_lock"]),
        "source_sha256":{s:digest(root/s) for s in SOURCE_FILES},
        "python":platform.python_version(),
        "versions":{name:importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib")}}


def _event(output, kind, **details):
    with (output/"stage_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event":kind, "utc":utc(), **details})+"\n")


def run(protocol_path, output, *, stage="selection", release_path=None):
    root = Path(__file__).resolve().parents[2]
    protocol_path, output = Path(protocol_path), Path(output)
    protocol = json.loads(protocol_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(root, protocol_path, protocol)
    manifest_path = output/"run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Frozen source, input or protocol changed")
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
        (output/"protocol_snapshot.json").write_bytes(protocol_path.read_bytes())
    if manifest["input_sha256"] != protocol["input_sha256"]:
        raise ValueError("Unexpected input version")
    if manifest["historical_selection_lock_sha256"] != protocol["historical_selection_lock_sha256"]:
        raise ValueError("Historical per-outer source choices changed")
    data, specs = prepare_protocol_roles(root/protocol["input"], root/protocol["historical_protocol"], output)
    classes = np.unique(data["y"])
    historical = json.loads((root/protocol["historical_selection_lock"]).read_text())
    lockpath = output/"selection_lock.json"
    _event(output, "stage_start", stage=stage, manifest_sha256=digest(manifest_path))
    started = time.perf_counter()
    if stage == "selection":
        if lockpath.exists():
            raise ValueError("Completed run is immutable; use a new protocol/run for changed work")
        all_rows, locks = [], []
        with gzip.open(output/"inner_predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["outer_fold", "inner_fold", "family", "candidate_id", "record_id", "y_true", "y_predicted", "is_correct"], lineterminator="\n")
            writer.writeheader()
            for spec in specs:
                choices = {}
                fold_rows = []
                for inner_no, inner in enumerate(spec["inner_folds"]):
                    train, valid = inner["train_indices"], inner["validation_indices"]
                    assert not (set(train)&set(valid) or set(train)&set(spec["test_indices"]) or set(valid)&set(spec["test_indices"]))
                    cache = {}
                    for family, candidates in protocol["candidate_grids"].items():
                        for candidate in candidates:
                            before = time.perf_counter()
                            row = {"outer_fold":spec["outer_fold"], "inner_fold":inner_no, "family":family,
                                "candidate_id":candidate["id"], "candidate":candidate,
                                "source_ids":data["groups"][train].tolist(), "validation_ids":data["groups"][valid].tolist()}
                            try:
                                model = fit_candidate(candidate, data["X_alpha"][train], data["y"][train], data["axis"], protocol["model_seed"], cache)
                                prediction = model.predict(data["X_alpha"][valid])
                                row.update(status="ok", metrics=metrics(data["y"][valid], prediction, classes), model_metadata=model.metadata)
                                for rid, truth, predicted in zip(data["groups"][valid], data["y"][valid], prediction):
                                    writer.writerow({"outer_fold":spec["outer_fold"], "inner_fold":inner_no, "family":family,
                                        "candidate_id":candidate["id"], "record_id":str(rid), "y_true":str(truth),
                                        "y_predicted":str(predicted), "is_correct":int(truth==predicted)})
                            except Exception as exc:
                                row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                            row["seconds"] = time.perf_counter()-before
                            fold_rows.append(row)
                for family, candidates in protocol["candidate_grids"].items():
                    choices[family] = select_candidate([r for r in fold_rows if r["family"] == family], candidates)
                historical_choice = historical["outer_folds"][spec["outer_fold"]]["choices"]["R2"]["selected"]
                locks.append({"outer_fold":spec["outer_fold"], "choices":choices,
                              "conditional_controls":controls(historical_choice, data["X_alpha"].shape[1])})
                all_rows.extend(fold_rows)
                write_json(output/"selection"/f"outer_{spec['outer_fold']:02d}.json", fold_rows)
                write_json(output/"status.json", {"status":"selection", "completed_folds":len(locks), "inner_evaluations":len(all_rows), "outer_test_scored":False})
                print(json.dumps({"stage":"selection", "outer_fold":spec["outer_fold"], "candidates_completed":len(fold_rows),
                                  "choices":{k:v["selected"] for k,v in choices.items()}}), flush=True)
        write_json(output/"all_inner_metrics.json", all_rows)
        write_json(lockpath, {"created_at_utc":utc(), "outer_folds":locks, "inner_evaluations":len(all_rows),
            "manifest_sha256":digest(manifest_path), "inner_metrics_sha256":digest(output/"all_inner_metrics.json"),
            "inner_predictions_sha256":digest(output/"inner_predictions.csv.gz"), "outer_test_used_to_select":False})
        write_json(output/"status.json", {"status":"selection_completed", "seconds":time.perf_counter()-started,
            "selection_lock_sha256":digest(lockpath), "inner_evaluations":len(all_rows), "failed_candidates":sum(r["status"]!="ok" for r in all_rows), "outer_test_scored":False})
        _event(output, "selection_lock_written", selection_lock_sha256=digest(lockpath))
        return json.loads((output/"status.json").read_text())
    if stage != "evaluation" or release_path is None:
        raise ValueError("Evaluation requires explicit release file")
    lock = json.loads(lockpath.read_text())
    release = json.loads(Path(release_path).read_text())
    for key, value in {"selection_lock_sha256":digest(lockpath), "manifest_sha256":digest(manifest_path), "protocol_sha256":digest(protocol_path)}.items():
        if release.get(key) != value:
            raise ValueError("Evaluation release does not bind frozen selection/protocol/source: "+key)
    if release.get("allow_outer_evaluation") is not True or not release.get("released_at_utc"):
        raise ValueError("Explicit dated release required")
    if lock["manifest_sha256"] != digest(manifest_path) or lock["inner_metrics_sha256"] != digest(output/"all_inner_metrics.json") or lock["inner_predictions_sha256"] != digest(output/"inner_predictions.csv.gz"):
        raise ValueError("Frozen selection evidence changed")
    if (output/"outer_predictions.csv.gz").exists():
        raise ValueError("Outer evaluation already exists and may not be overwritten")
    (output/"release_snapshot.json").write_bytes(Path(release_path).read_bytes())
    rows, all_predictions, model_manifest = [], [], []
    fields = ["method", "outer_fold", "record_id", "y_true", "y_predicted", "is_correct"] + ["score_"+str(c) for c in classes]
    with gzip.open(output/"outer_predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for spec, fold_lock in zip(specs, lock["outer_folds"], strict=True):
            train, test = spec["train_indices"], spec["test_indices"]
            assert spec["outer_fold"] == fold_lock["outer_fold"]
            methods = {k:v["selected"] for k,v in fold_lock["choices"].items()} | fold_lock["conditional_controls"]
            for method, candidate in methods.items():
                model = fit_candidate(candidate, data["X_alpha"][train], data["y"][train], data["axis"], protocol["model_seed"])
                before = time.perf_counter()
                prediction = model.predict(data["X_alpha"][test])
                predict_seconds = time.perf_counter()-before
                scores = model.scores(data["X_alpha"][test])
                model_path = output/"models"/f"outer_{spec['outer_fold']:02d}"/(method+".joblib")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, model_path, compress=3)
                info = {"method":method, "outer_fold":spec["outer_fold"], "candidate":candidate,
                    "source_ids":data["groups"][train].tolist(), "test_ids":data["groups"][test].tolist(),
                    "model_metadata":model.metadata, "prediction_seconds":predict_seconds,
                    "model_path":str(model_path), "model_sha256":digest(model_path), "model_bytes":model_path.stat().st_size,
                    "metrics":metrics(data["y"][test], prediction, classes)}
                rows.append(info)
                write_json(model_path.with_suffix(".json"), info)
                model_manifest.append({k:info[k] for k in ("method", "outer_fold", "model_path", "model_sha256", "model_bytes")})
                for rid, truth, predicted, score in zip(data["groups"][test], data["y"][test], prediction, scores):
                    record = {"method":method, "outer_fold":spec["outer_fold"], "record_id":str(rid), "y_true":str(truth),
                        "y_predicted":str(predicted), "is_correct":int(truth==predicted),
                        **{"score_"+str(c):float(v) for c,v in zip(classes,score)}}
                    writer.writerow(record)
                    all_predictions.append(record)
            print(json.dumps({"stage":"evaluation", "outer_fold":spec["outer_fold"], "methods_completed":len(methods)}), flush=True)
    pooled = {}
    for method in (*FAMILIES, *CONTROLS):
        found = [r for r in all_predictions if r["method"] == method]
        assert len(found) == len({r["record_id"] for r in found}) == 855
        pooled[method] = metrics(np.asarray([r["y_true"] for r in found]), np.asarray([r["y_predicted"] for r in found]), classes)
        method_rows = [r for r in rows if r["method"] == method]
        pooled[method]["cost"] = {"outer_transform_seconds":sum(r["model_metadata"]["transform_seconds"] for r in method_rows),
            "outer_estimator_fit_seconds":sum(r["model_metadata"]["estimator_fit_seconds"] for r in method_rows),
            "outer_predict_seconds":sum(r["prediction_seconds"] for r in method_rows),
            "model_bytes_sum":sum(r["model_bytes"] for r in method_rows),
            "mean_selected_features":float(np.mean([r["model_metadata"]["selected_features"] for r in method_rows])),
            "inner_search_seconds":sum(r["seconds"] for r in json.loads((output/"all_inner_metrics.json").read_text()) if r["family"] == method)}
    write_json(output/"outer_metrics.json", rows)
    write_json(output/"pooled_metrics.json", pooled)
    write_json(output/"models_manifest.json", model_manifest)
    write_json(output/"status.json", {"status":"completed", "seconds":time.perf_counter()-started,
        "outer_method_fold_roles":len(rows), "outer_predictions":len(all_predictions), "selection_lock_sha256":digest(lockpath)})
    _event(output, "evaluation_completed", outer_predictions_sha256=digest(output/"outer_predictions.csv.gz"))
    return pooled
