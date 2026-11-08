"""One resumable classical development job restricted to master TRAIN specimens.

This worker never fits or selects with the master calibration/prediction pools.
Its outer validation folds are development folds within the existing TRAIN pool,
and do not constitute a fresh blinded test or a real cross-batch experiment.
The budget supervisor, not this worker, decides which jobs may start.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from herb_spectra.benchmark import PLSDAClassifier
from herb_spectra.classification import classification_metrics, sample_folds
from herb_spectra.spectral_models import SpectralPreprocessor
from herb_spectra.staged_nested_search import (
    _prefix_selector, choose_candidate, class_weight_spec, fit_source_candidate,
    gamma_spec, historical_grid, mc_grid, svm_grid,
)

BASE = ROOT / "data/research/species_revision_20261108"
TASKS = ("acid_origins", "huangqi_origins", "tianma_origins", "four_kinds", "binary_kinds")
REPRESENTATIONS = ("alpha", "n", "n_alpha")
REPEAT_SEEDS = (20261031, 20261108, 42)
HISTORICAL_COMMIT = "3fc996e99cce8ff5799b5a2af9b565f982dfc899"
SOURCE_FILES = (
    "src/herb_spectra/staged_nested_search.py", "src/herb_spectra/selection.py",
    "src/herb_spectra/spectral_models.py", "src/herb_spectra/benchmark.py",
    "src/herb_spectra/classification.py",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def checked_freeze(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Existing frozen job differs: {path}")
    else:
        atomic_json(path, value)


def source_manifest():
    return {str(path): digest(ROOT / path) for path in SOURCE_FILES}


class BlockPreprocessor:
    """Apply the original per-spectrum transform separately to n and alpha."""
    def __init__(self, mode, axis, representation):
        if representation not in REPRESENTATIONS or mode not in ("raw", "snv"):
            raise ValueError("Unsupported frozen representation or preprocessing")
        self.mode = mode
        self.representation = representation
        self.axis = np.asarray(axis, dtype=float)
        self.block = SpectralPreprocessor(mode, self.axis)
        self.n_features_in_ = len(self.axis) * (2 if representation == "n_alpha" else 1)

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("Preprocessor received invalid spectral matrix")
        if self.representation != "n_alpha":
            return self.block.transform(X)
        width = len(self.axis)
        return np.concatenate((self.block.transform(X[:, :width]),
                               self.block.transform(X[:, width:])), axis=1)


@dataclass
class BlockFittedModel:
    candidate: dict
    preprocessor: BlockPreprocessor
    scaler: StandardScaler
    estimator: object
    selector: object = None

    @property
    def classes_(self):
        return self.estimator.classes_

    def transform(self, X):
        value = self.scaler.transform(self.preprocessor.transform(X))
        return value if self.selector is None else self.selector.transform(value)

    def predict(self, X):
        return self.estimator.predict(self.transform(X))

    def predict_scores(self, X):
        return self.estimator.decision_function(self.transform(X))


def fit_candidate(candidate, X, y, axis, representation, cache=None):
    """Reuse the original MC coefficient stream and SVC parameter semantics."""
    if "mc" in candidate and representation != "n_alpha":
        return fit_source_candidate(candidate, X, y, axis, seed=42,
                                    cache={} if cache is None else cache)
    preprocessor = BlockPreprocessor(candidate["preprocessing"], axis, representation)
    scaler = StandardScaler()
    values = scaler.fit_transform(preprocessor.transform(X))
    selector = None
    if "mc" in candidate:
        selector = _prefix_selector(values, y, candidate["mc"], 42,
                                    {} if cache is None else cache)
        values = selector.transform(values)
    if candidate.get("model") == "plsda":
        estimator = PLSDAClassifier(n_components=int(candidate["n_components"]))
    else:
        svm = candidate["svm"]
        weight, _ = class_weight_spec(y, svm["class_weight"])
        gamma, _ = gamma_spec(values, svm["gamma"])
        estimator = SVC(C=svm["C"], kernel="rbf", gamma=gamma, class_weight=weight,
                        decision_function_shape="ovr", break_ties=False, random_state=42)
    estimator.fit(values, y)
    return BlockFittedModel(dict(candidate), preprocessor, scaler, estimator, selector)


def model_scores(model, X):
    values = np.asarray(model.predict_scores(X), dtype=float)
    if values.ndim == 1:
        if len(model.classes_) != 2:
            raise ValueError("One-dimensional scores with nonbinary classes")
        values = np.column_stack((-values, values))
    if values.shape != (len(X), len(model.classes_)) or not np.isfinite(values).all():
        raise ValueError("Class-score columns are invalid")
    return values


def selected_indices(model, p):
    if hasattr(model, "bundle"):
        return model.bundle.selector.selected_indices_.astype(int)
    if model.selector is not None:
        return model.selector.selected_indices_.astype(int)
    return np.arange(p, dtype=int)


def feature_annotations(indices, axis, representation):
    width = len(axis)
    rows = []
    for index in indices:
        index = int(index)
        if representation == "n_alpha":
            modality, within = ("n", index) if index < width else ("alpha", index - width)
        else:
            modality, within = representation, index
        rows.append({"feature_index": index, "modality": modality,
                     "within_modality_index": int(within), "frequency_THz_export_label": float(axis[within])})
    return rows


def full_svm_grid():
    return [{"id": f"full_{i:02d}", "model": "svc", "preprocessing": preprocessing,
             "svm": {"C": C, "gamma": "scale", "class_weight": "balanced"}}
            for i, (C, preprocessing) in enumerate((c, mode) for c in (.1, 1., 10., 100.)
                                                   for mode in ("raw", "snv"))]


def pls_grid():
    return [{"id": f"pls_{i:02d}", "model": "plsda", "preprocessing": preprocessing,
             "n_components": components}
            for i, (components, preprocessing) in enumerate((n, mode) for n in (1, 2, 3, 5, 8)
                                                            for mode in ("raw", "snv"))]


def non_mc_choice(rows, candidates):
    """Macro-F1 primary, BA/accuracy secondary, then the declared simple order."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["candidate_id"]].append(row["metrics"])
    means = {c["id"]: {m: float(np.mean([r[m] for r in groups[c["id"]]]))
                        for m in ("macro_f1", "balanced_accuracy", "accuracy")}
             for c in candidates}
    if any(len(groups[c["id"]]) != 2 for c in candidates):
        raise ValueError("Incomplete two-fold source selection")
    best = max(row["macro_f1"] for row in means.values())
    eligible = [c for c in candidates if means[c["id"]]["macro_f1"] >= best - 1e-12]
    selected = min(eligible, key=lambda c: (-means[c["id"]]["balanced_accuracy"],
                                           -means[c["id"]]["accuracy"],
                                           c.get("n_components", 0),
                                           c.get("svm", {}).get("C", 0),
                                           ("raw", "snv").index(c["preprocessing"]), c["id"]))
    return {"selected": selected, "mean_inner_metrics": means[selected["id"]],
            "all_candidate_means": means, "selected_using_outer_test": False,
            "tie_rule": "macroF1 then BA/accuracy, fewer PLS components/lower C, raw then SNV, candidate ID"}


def resume_stage(path, receipt_path, manifest_hash, candidates):
    if not Path(path).exists() or not Path(receipt_path).exists():
        return None
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    if receipt != {"artifact_sha256": digest(path), "job_manifest_sha256": manifest_hash}:
        raise ValueError(f"Partial stage checkpoint integrity failure: {path}")
    stage = json.loads(Path(path).read_text(encoding="utf-8"))
    if stage["candidates"] != candidates or stage["job_manifest_sha256"] != manifest_hash:
        raise ValueError(f"Partial stage parameters changed: {path}")
    return stage


def select_stage(name, candidates, X, y, ids, axis, representation, folds,
                 out, manifest_hash, *, historical=False, mc=True):
    path, receipt_path = out / f"{name}_inner.json", out / f"{name}_inner.receipt.json"
    previous = resume_stage(path, receipt_path, manifest_hash, candidates)
    if previous is not None:
        return previous["selection"]
    rows = []
    for fold_number, (train, valid) in enumerate(folds):
        if set(ids[train]) & set(ids[valid]):
            raise AssertionError("Physical specimen crosses inner roles")
        cache = {"__coefficient_budget__": max(c["mc"]["runs"] for c in candidates)} if mc else {}
        for candidate in candidates:
            before = time.process_time()
            fitted = fit_candidate(candidate, X[train], y[train], axis, representation, cache)
            pred = fitted.predict(X[valid])
            rows.append({"inner_fold": fold_number, "candidate_id": candidate["id"],
                         "metrics": classification_metrics(y[valid], pred, np.unique(y)),
                         "selected_feature_indices": selected_indices(fitted, X.shape[1]).tolist(),
                         "fit_cpu_seconds": time.process_time() - before})
    choice = choose_candidate(rows, candidates, historical=historical) if mc else non_mc_choice(rows, candidates)
    payload = {"stage": name, "job_manifest_sha256": manifest_hash,
               "candidates": candidates, "selection": choice, "inner_results": rows,
               "inner_folds": [{"fit_record_ids": ids[t].tolist(), "validation_record_ids": ids[v].tolist()}
                               for t, v in folds],
               "all_ids_are_master_train": True, "outer_validation_used_to_select": False,
               "master_calibration_or_prediction_used": False}
    atomic_json(path, payload)
    atomic_json(receipt_path, {"artifact_sha256": digest(path), "job_manifest_sha256": manifest_hash})
    print(json.dumps({"event": "selection_complete", "stage": name,
                      "candidate": choice["selected"], "inner_metrics": choice["mean_inner_metrics"]}), flush=True)
    return choice


def load_master_train(features, split, task, representation):
    """Only TRAIN-row target metadata is consulted; other labels are unused."""
    with Path(split).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    train_rows = [row for row in rows if row["role"] == "train"]
    if len({row["physical_sample_id"] for row in rows}) != len(rows):
        raise ValueError("Master split does not have unique independent physical specimens")
    if task == "binary_kinds":
        train_rows = [row for row in train_rows if row["species"] in ("枣仁", "酸枣仁")]
    elif task != "four_kinds":
        kind = {"acid_origins": "酸枣仁", "huangqi_origins": "黄芪", "tianma_origins": "天麻"}[task]
        train_rows = [row for row in train_rows if row["species"] == kind]
    if not train_rows:
        raise ValueError("Task has no master training specimens")
    # Preserve the reviewed feature archive's row order, not workbook identity.
    with np.load(features, allow_pickle=False) as archive:
        record_ids = archive["record_ids"].astype(str)
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("Feature archive has duplicate record IDs")
        included = {row["record_id"]: row for row in train_rows}
        indices = np.asarray([i for i, value in enumerate(record_ids) if value in included], dtype=int)
        if len(indices) != len(included):
            raise ValueError("Master TRAIN identifiers not completely covered by feature archive")
        ids = record_ids[indices]
        axis = archive["axis"].astype(float)
        mask = (axis >= .1) & (axis <= 1.5)
        if mask.sum() < 2 or np.any(np.diff(axis[mask]) <= 0):
            raise ValueError("Prespecified frequency axis invalid")
        # No task labels are read from the NPZ. Only selected TRAIN rows reach
        # the returned matrix, estimators, selector, or candidate-selection API.
        if representation == "alpha":
            X = archive["X_alpha"][indices][:, mask].astype(float)
        elif representation == "n":
            X = archive["X_n"][indices][:, mask].astype(float)
        else:
            X = np.concatenate((archive["X_n"][indices][:, mask],
                                archive["X_alpha"][indices][:, mask]), axis=1).astype(float)
        axis = axis[mask]
    y = np.asarray([included[value]["species" if task in ("binary_kinds", "four_kinds") else "origin_group_id"]
                    for value in ids])
    physical = np.asarray([included[value]["physical_sample_id"] for value in ids])
    if not np.isfinite(X).all() or len(np.unique(y)) < 2 or len(set(physical)) != len(physical):
        raise ValueError("Invalid source-only spectra, classes, or physical IDs")
    return X, y, ids, physical, axis


def run_job(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    X, y, ids, physical, axis = load_master_train(args.features, args.split, args.task, args.representation)
    outer_folds = sample_folds(y, physical, n_splits=5, seed=args.repeat_seed)
    source, validation = outer_folds[args.outer_fold]
    if set(ids[source]) & set(ids[validation]):
        raise AssertionError("Outer TRAIN-development roles overlap")
    inner_folds = sample_folds(y[source], physical[source], n_splits=2, seed=args.repeat_seed)
    manifest = {
        "worker_protocol": "training-only-classical-nested-development-v1",
        "scope": "development_only_within_master_TRAIN_not_final_prediction_not_cross_batch",
        "task": args.task, "representation": args.representation,
        "outer_fold": args.outer_fold, "outer_folds": 5, "inner_folds": 2,
        "outer_split_seed": args.repeat_seed, "inner_split_seed": args.repeat_seed,
        "fixed_model_seed": 42, "declared_repeat_split_seeds": list(REPEAT_SEEDS),
        "features_sha256": digest(args.features), "master_split_sha256": digest(args.split),
        "worker_sha256": digest(__file__), "historical_algorithm_commit": HISTORICAL_COMMIT,
        "historical_module_sha256": source_manifest(),
        "frequency_interval_inclusive": [.1, 1.5], "frequency_points_per_modality": len(axis),
        "representation_blocks": ["n", "alpha"] if args.representation == "n_alpha" else [args.representation],
        "snv_policy": "separate per-spectrum SNV within each modality; never SNV across mixed n/alpha units",
        "source_record_ids": ids[source].tolist(), "outer_validation_record_ids": ids[validation].tolist(),
        "master_training_record_ids": ids.tolist(),
        "class_order": np.unique(y).tolist(),
        "master_train_class_counts": {str(c): int(np.sum(y == c)) for c in np.unique(y)},
        "candidate_grids": {"R0": historical_grid(), "R1": mc_grid(), "R2": svm_grid(),
                            "fullSVM": full_svm_grid(), "PLSDA": pls_grid()},
        "selection_policy": "2fold source macroF1; exact historical R0 first-tie and R1/R2 tie rules; fullSVM/PLS simpler-first declared tie rule",
        "prediction_policy": "save estimator vote and fixed-class score argmax; scores are not probabilities",
        "master_calibration_or_prediction_used_for_fit_or_selection": False,
        "support_calibration_status": "not_run_in_this_source_only_job; master calibration untouched",
    }
    checked_freeze(out / "job_manifest.json", manifest)
    manifest_hash = digest(out / "job_manifest.json")
    if (out / "complete.json").exists():
        done = json.loads((out / "complete.json").read_text(encoding="utf-8"))
        if done["job_manifest_sha256"] != manifest_hash:
            raise ValueError("Completed job belongs to a different manifest")
        for name, expected in done["artifacts_sha256"].items():
            if digest(out / name) != expected:
                raise ValueError(f"Completed job artifact changed: {name}")
        print(json.dumps({"status": "already_complete", "job_done": str(out / "complete.json")}), flush=True)
        return done
    if args.plan_only:
        print(json.dumps({"status": "planned", "job_manifest_sha256": manifest_hash,
                          "source_count": len(source), "development_validation_count": len(validation)}), flush=True)
        return manifest
    started_wall, started_cpu = time.monotonic(), time.process_time()
    sx, sy, si = X[source], y[source], ids[source]
    choices = {}
    choices["R0"] = select_stage("R0", historical_grid(), sx, sy, si, axis, args.representation,
                                  inner_folds, out, manifest_hash, historical=True)
    r1 = [{"id": candidate["id"], "preprocessing": "snv",
           "mc": {key: candidate[key] for key in ("runs", "components", "ratio")},
           "svm": {"C": 10., "gamma": "scale", "class_weight": "balanced"}}
          for candidate in mc_grid()]
    first = select_stage("R1", r1, sx, sy, si, axis, args.representation, inner_folds, out, manifest_hash)
    r2 = [{"id": candidate["id"], "preprocessing": "snv", "mc": first["selected"]["mc"],
           "svm": {key: candidate[key] for key in ("C", "gamma", "class_weight")}}
          for candidate in svm_grid()]
    choices["R2"] = select_stage("R2", r2, sx, sy, si, axis, args.representation, inner_folds, out, manifest_hash)
    choices["fullSVM"] = select_stage("fullSVM", full_svm_grid(), sx, sy, si, axis, args.representation,
                                       inner_folds, out, manifest_hash, mc=False)
    choices["PLSDA"] = select_stage("PLSDA", pls_grid(), sx, sy, si, axis, args.representation,
                                     inner_folds, out, manifest_hash, mc=False)
    # Every source method is locked before the outer development labels score it.
    checked_freeze(out / "selection_lock.json", {"job_manifest_sha256": manifest_hash,
                                                "choices": choices, "outer_validation_used": False,
                                                "master_calibration_or_prediction_used": False})
    result, prediction_rows = [], []
    for method in ("R0", "R2", "fullSVM", "PLSDA"):
        model = fit_candidate(choices[method]["selected"], sx, sy, axis, args.representation,
                              {"__coefficient_budget__": choices[method]["selected"].get("mc", {}).get("runs", 0)})
        classes = np.asarray(model.classes_)
        scores = model_scores(model, X[validation])
        vote, argmax = model.predict(X[validation]), classes[np.argmax(scores, axis=1)]
        indices = selected_indices(model, X.shape[1])
        model_result = {"method": method, "selected_candidate": choices[method]["selected"],
                        "mean_inner_metrics": choices[method]["mean_inner_metrics"],
                        "selected_feature_indices": indices.tolist(),
                        "selected_feature_annotations": feature_annotations(indices, axis, args.representation),
                        "class_order": classes.tolist(),
                        "outer_metrics_vote": classification_metrics(y[validation], vote, classes),
                        "outer_metrics_score_argmax": classification_metrics(y[validation], argmax, classes),
                        "vote_argmax_disagreements": int(np.sum(vote != argmax))}
        result.append(model_result)
        for index, predicted, score_pred, values in zip(validation, vote, argmax, scores):
            prediction_rows.append({"task": args.task, "representation": args.representation,
                                    "repeat_seed": args.repeat_seed, "outer_fold": args.outer_fold,
                                    "method": method, "record_id": str(ids[index]),
                                    "physical_sample_id": str(physical[index]), "y_true": str(y[index]),
                                    "prediction_vote": str(predicted), "prediction_score_argmax": str(score_pred),
                                    "is_correct_vote": int(y[index] == predicted),
                                    "is_correct_score_argmax": int(y[index] == score_pred),
                                    "class_order_json": json.dumps(classes.tolist(), ensure_ascii=False),
                                    "scores_json": json.dumps(values.tolist(), allow_nan=False)})
    summary = {"job_manifest_sha256": manifest_hash, "task": args.task,
               "representation": args.representation, "repeat_seed": args.repeat_seed,
               "outer_fold": args.outer_fold, "master_train_count": len(X),
               "source_count": len(source), "development_validation_count": len(validation),
               "scope": "source_only_development_within_master_train",
               "master_calibration_or_prediction_used": False, "results": result,
               "elapsed_cpu_seconds_this_invocation": time.process_time() - started_cpu,
               "elapsed_wall_seconds_this_invocation": time.monotonic() - started_wall,
               "no_final_accuracy_claim": "These are nested training-development scores, not new independent test results."}
    temporary = out / f"validation_predictions.csv.gz.tmp.{os.getpid()}"
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(prediction_rows)
    os.replace(temporary, out / "validation_predictions.csv.gz")
    atomic_json(out / "summary.json", summary)
    artifacts = sorted([path.name for path in out.iterdir()
                        if path.is_file() and path.name != "complete.json" and ".tmp." not in path.name])
    done = {"status": "complete", "job_manifest_sha256": manifest_hash,
            "artifacts_sha256": {name: digest(out / name) for name in artifacts},
            "elapsed_cpu_seconds_this_invocation": summary["elapsed_cpu_seconds_this_invocation"],
            "elapsed_wall_seconds_this_invocation": summary["elapsed_wall_seconds_this_invocation"],
            "all_four_methods_complete": True}
    atomic_json(out / "complete.json", done)
    print(json.dumps({"status": "complete", "cpu_seconds": summary["elapsed_cpu_seconds_this_invocation"],
                      "methods": {r["method"]: r["outer_metrics_vote"]["accuracy"] for r in result}}), flush=True)
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    parser.add_argument("--outer-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--repeat-seed", type=int, choices=REPEAT_SEEDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=BASE / "features.npz")
    parser.add_argument("--split", type=Path, default=BASE / "split_manifest.csv")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run_job(args)


if __name__ == "__main__":
    main()
