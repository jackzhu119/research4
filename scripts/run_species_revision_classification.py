"""Reuse historical source-only SVM/MC-UVE and frozen calibration controls.

The master specimen split is read, never regenerated. Historical ZY specimens
have already been exposed during development: this is retrospective internal
holdout evidence, not an independent batch or blinded external validation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from herb_spectra.classification import classification_metrics, sample_folds
from herb_spectra.conservative_calibration import build_context, fit_candidate
from herb_spectra.spectral_models import SpectralPreprocessor
from herb_spectra.staged_nested_search import (choose_candidate, fit_source_candidate,
                                              historical_grid, mc_grid, svm_grid)

BASE = ROOT / "data/research/species_revision_20261108"
HISTORICAL_COMMIT = "3fc996e99cce8ff5799b5a2af9b565f982dfc899"
MC20 = {"preprocessing": "snv", "mc": {"runs": 20, "components": 2, "ratio": .1},
        "svm": {"C": 10., "gamma": "scale", "class_weight": "balanced"}, "id": "fixed_MC20"}
CALIBRATORS = [
    {"id": "legacy_anchor_lambda1", "family": "legacy_anchor", "regularization": 1.},
    {"id": "weak_anchor_lambda1000_blend005", "family": "damped_anchor", "regularization": 1000., "blend": .05},
]
TASKS = ("binary_kinds", "acid_origins", "four_kinds", "huangqi_origins", "tianma_origins", "jujube_origins")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def freeze(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen configuration differs: {path}")
    else:
        write_json(path, value)


def score_matrix(model, values):
    scores = (model.predict_scores(values) if hasattr(model, "predict_scores")
              else model.decision_function(values))
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = np.column_stack((-scores, scores))
    if scores.shape != (len(values), len(model.classes_)) or not np.isfinite(scores).all():
        raise ValueError("Invalid fixed-order source scores")
    return scores


def representation(model, values):
    return (model.bundle.transform(values) if hasattr(model, "bundle")
            else model[:-1].transform(values))


def fit_model(kind, candidate, X, y, axis, cache=None):
    if kind == "full_svm":
        model = Pipeline([("snv", FunctionTransformer(SpectralPreprocessor("snv", axis).transform,
                                                     validate=False)),
                          ("scaler", StandardScaler()),
                          ("svm", SVC(C=10., gamma="scale", class_weight="balanced",
                                      decision_function_shape="ovr", break_ties=False,
                                      random_state=42))])
        return model.fit(X, y)
    return fit_source_candidate(candidate, X, y, axis, seed=42,
                                cache={} if cache is None else cache)


def inner_select(X, y, ids, axis, candidates, output, historical=False):
    """Reuse the exact historical two-fold candidate tie-breaking rule."""
    folds = sample_folds(y, ids, n_splits=2, seed=20261031)
    rows = []
    for fold, (train, valid) in enumerate(folds):
        cache = {"__coefficient_budget__": max(c["mc"]["runs"] for c in candidates)}
        for candidate in candidates:
            started = time.monotonic()
            model = fit_source_candidate(candidate, X[train], y[train], axis, seed=42, cache=cache)
            pred = model.predict(X[valid])
            rows.append({"inner_fold": fold, "candidate_id": candidate["id"],
                         "metrics": classification_metrics(y[valid], pred, np.unique(y)),
                         "candidate": candidate, "seconds": time.monotonic() - started,
                         "selected_indices": model.bundle.selector.selected_indices_.tolist()})
    choice = choose_candidate(rows, candidates, historical=historical)
    write_json(output, {"selection": choice, "inner_results": rows,
                        "train_ids": ids.tolist(), "folds": [
                            {"train": ids[t].tolist(), "validation": ids[v].tolist()}
                            for t, v in folds], "prediction_data_used": False,
                        "calibration_data_used": False})
    return choice["selected"]


def task_arrays(name, arrays, roles):
    species = arrays["species"].astype(str)
    group_key = "origin_group_ids" if "origin_group_ids" in arrays else "origin_group_id"
    origins = arrays[group_key].astype(str)
    if name == "binary_kinds":
        mask, y = np.isin(species, ["枣仁", "酸枣仁"]), species
    elif name == "four_kinds":
        mask, y = np.ones(len(species), dtype=bool), species
    else:
        kind = {"acid_origins": "酸枣仁", "huangqi_origins": "黄芪", "tianma_origins": "天麻", "jujube_origins": "枣仁"}[name]
        mask, y = species == kind, origins
    indices = np.flatnonzero(mask)
    return {role: indices[roles[indices] == role] for role in ("train", "calibration", "prediction")}, y


def run_task(name, arrays, roles, X, axis, output):
    out = output / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        saved = json.loads((out / "complete.json").read_text())
        if saved["summary_sha256"] != digest(out / "summary.json"):
            raise ValueError("Completed task summary has changed")
        return json.loads((out / "summary.json").read_text())
    role, labels = task_arrays(name, arrays, roles)
    train, cal, query = (role[r] for r in ("train", "calibration", "prediction"))
    ids = arrays["record_ids"].astype(str)
    classes = np.unique(labels[train])
    if any(set(labels[ix]) != set(classes) for ix in (cal, query)):
        raise ValueError("Each frozen role must cover every class")
    if any(set(ids[a]) & set(ids[b]) for a, b in ((train, cal), (train, query), (cal, query))):
        raise ValueError("Physical specimen crossed roles")
    main = name in ("binary_kinds", "acid_origins")
    configs = {"full_svm": None, "fixed_MC20": MC20}
    choices = {}
    if main:
        configs["R0_reselected"] = inner_select(X[train], labels[train], ids[train], axis,
                                              historical_grid(), out / "R0_inner_selection.json", True)
        candidates = [{"id": m["id"], "preprocessing": "snv",
                       "mc": {k: m[k] for k in ("runs", "components", "ratio")},
                       "svm": {"C": 10., "gamma": "scale", "class_weight": "balanced"}}
                      for m in mc_grid()]
        selected_mc = inner_select(X[train], labels[train], ids[train], axis, candidates,
                                   out / "R1_inner_selection.json")
        candidates = [{"id": s["id"], "preprocessing": "snv", "mc": selected_mc["mc"],
                       "svm": {k: s[k] for k in ("C", "gamma", "class_weight")}}
                      for s in svm_grid()]
        configs["R2_reselected"] = inner_select(X[train], labels[train], ids[train], axis, candidates,
                                              out / "R2_inner_selection.json")
    freeze(out / "selection_lock.json", {"configs": configs, "train_ids": ids[train].tolist(),
                                          "calibration_used_to_select": False,
                                          "prediction_used_to_select": False})
    result_rows, records = [], []

    def add(model_name, method, pred, *, shot="all", repeat=0, support=None, params=None):
        metric = classification_metrics(labels[query], pred, classes)
        result_rows.append({"model": model_name, "method": method, "shots_per_class": shot,
                            "repeat": repeat, "support_count": 0 if support is None else len(support),
                            "metrics": metric, "parameters": params})
        for idx, predicted in zip(query, pred):
            records.append({"task": name, "model": model_name, "method": method,
                            "shots_per_class": shot, "repeat": repeat, "record_id": str(ids[idx]),
                            "y_true": str(labels[idx]), "y_predicted": str(predicted),
                            "is_correct": int(labels[idx] == predicted)})

    majority = min(classes, key=lambda c: (-int(np.sum(labels[train] == c)), str(c)))
    add("dummy_majority", "source_only", np.full(len(query), majority), shot=0)
    rng = np.random.default_rng(42)
    shots, unsupported = [], {}
    if main:
        for repeat in range(20):
            permutations = {str(c): rng.permutation(cal[labels[cal] == c]) for c in classes}
            for k in (1, 3, 5, 10):
                if any(len(v) < k for v in permutations.values()):
                    unsupported[str(k)] = "Insufficient independent calibration specimens per class"
                    continue
                shots.append((k, repeat, np.concatenate([permutations[str(c)][:k] for c in classes])))
    shots.append(("all", 0, cal))
    write_json(out / "support_draws.json", [
        {"shots_per_class": k, "repeat": r, "record_ids": ids[s].tolist()} for k, r, s in shots])
    for kind, candidate in configs.items():
        print(json.dumps({"task": name, "event": "fit_source", "model": kind}), flush=True)
        started = time.monotonic()
        model = fit_model(kind, candidate, X[train], labels[train], axis)
        source_fingerprint = joblib.hash(model)
        source_pred = model.predict(X[query])
        add(kind, "source_only", source_pred, shot=0, params=candidate)
        source_scores = score_matrix(model, X[train])
        query_scores, cal_scores = score_matrix(model, X[query]), score_matrix(model, X[cal])
        query_features, cal_features = representation(model, X[query]), representation(model, X[cal])
        context = build_context(representation(model, X[train]), source_scores, labels[train], classes, seed=42)
        add(kind, "source_score_argmax", classes[np.argmax(query_scores, axis=1)], shot=0,
            params={"reason": "matched score-rule control for anchor corrections; standard SVC voting is source_only"})
        cal_lookup = {int(v): i for i, v in enumerate(cal)}
        for shot, repeat, support in shots:
            local = np.asarray([cal_lookup[int(v)] for v in support])
            for adapter_config in CALIBRATORS:
                adapter = fit_candidate(context, adapter_config, cal_features[local], cal_scores[local], labels[support])
                corrected = adapter.predict_scores(query_features, query_scores)
                pred = classes[np.argmax(corrected, axis=1)]
                add(kind, adapter_config["id"], pred, shot=shot, repeat=repeat, support=support,
                    params=adapter_config)
        if joblib.hash(model) != source_fingerprint:
            raise AssertionError("Calibration mutated the source model")
        # All-pool refitting is a separate, explicitly named matched control.
        combined = np.concatenate((train, cal))
        refit = fit_model(kind, candidate, X[combined], labels[combined], axis)
        add(kind, "train_plus_calibration_full_retrain", refit.predict(X[query]), support=cal, params=candidate)
        joblib.dump(model, out / (kind + "_source.joblib"), compress=3)
        joblib.dump(refit, out / (kind + "_train_plus_calibration.joblib"), compress=3)
        write_json(out / (kind + "_fit_receipt.json"), {
            "seconds": time.monotonic() - started, "source_fingerprint": source_fingerprint,
            "source_ids": ids[train].tolist(), "full_refit_ids": ids[combined].tolist(),
            "prediction_ids": ids[query].tolist(), "config": candidate,
            "source_metadata": getattr(model, "metadata", {}),
            "model_features_contain_ids_or_labels": False,
            "source_vote_vs_argmax_disagreements": int(np.sum(source_pred != classes[np.argmax(query_scores, axis=1)]))})
    grouped = defaultdict(list)
    for row in result_rows:
        grouped[(row["model"], row["method"], str(row["shots_per_class"]))].append(row)
    aggregates = []
    for (model, method, k), rows in grouped.items():
        aggregates.append({"model": model, "method": method, "shots_per_class": k, "repeats": len(rows),
                           **{m + "_mean": float(np.mean([r["metrics"][m] for r in rows]))
                              for m in ("accuracy", "macro_f1", "balanced_accuracy")},
                           "accuracy_sd_support_draws": float(np.std([r["metrics"]["accuracy"] for r in rows], ddof=0))})
    summary = {"task": name, "scope": "retrospective_internal_specimen_holdout_not_cross_batch",
               "roles": {r: {"count": len(ix), "class_counts": {str(c): int(np.sum(labels[ix] == c)) for c in classes}}
                         for r, ix in role.items()}, "classes": classes.tolist(),
               "training_cv": "2-fold source-only; historical R0 and sequential R1/R2 grids" if main else "fixed historical parameters; no new tuning",
               "tiny_class_warning": "Three new jujube origin groups have only one independent specimen in each role; rates are descriptive and unstable" if name == "jujube_origins" else None,
               "unsupported_shots": unsupported, "selection_lock_sha256": digest(out / "selection_lock.json"),
               "all_results": result_rows, "aggregates": aggregates,
               "calibration_gain_reference": "Anchor correction must be compared to source_score_argmax; full retrain to standard-vote source_only",
               "prediction_never_used_for_selection": True,
               "historical_exposure": "Old855 already used in prior algorithm development; preserving roles does not create a new blind test"}
    write_json(out / "summary.json", summary)
    with gzip.open(out / "predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(records)
    write_json(out / "complete.json", {"summary_sha256": digest(out / "summary.json"),
                                       "predictions_sha256": digest(out / "predictions.csv.gz"),
                                       "method_selected_using_prediction": False})
    print(json.dumps({"task": name, "event": "complete", "all_pool_results": [
        r for r in aggregates if r["shots_per_class"] in ("all", "0")]}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=BASE / "features.npz")
    parser.add_argument("--split", type=Path, default=BASE / "split_manifest.csv")
    parser.add_argument("--output", type=Path, default=BASE / "classification")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.features, allow_pickle=False) as f:
        arrays = {key: f[key] for key in f.files}
    with args.split.open(encoding="utf-8-sig", newline="") as handle:
        split = list(csv.DictReader(handle))
    by_id = {r["record_id"]: r for r in split}
    ids = arrays["record_ids"].astype(str)
    if len(by_id) != len(ids) or set(ids) != set(by_id):
        raise ValueError("Feature IDs and frozen master split disagree")
    if len(set(r["physical_sample_id"] for r in split)) != len(split):
        raise ValueError("One independently declared physical specimen required per row")
    roles = np.asarray([by_id[i]["role"] for i in ids])
    axis = arrays["axis"].astype(float)
    mask = (axis >= .1) & (axis <= 1.5)
    X, axis = arrays["X_alpha"][:, mask].astype(float), axis[mask]
    if len(axis) < 2 or not np.isfinite(X).all():
        raise ValueError("Invalid prespecified alpha window")
    frozen = {"historical_algorithm_commit": HISTORICAL_COMMIT,
              "feature_sha256": digest(args.features), "split_sha256": digest(args.split),
              "script_sha256": digest(__file__), "tasks": args.tasks,
              "frequency_interval_inclusive": [.1, 1.5], "actual_frequency_points": len(axis),
              "split_policy": "inherit master manifest without moving any specimen",
              "inner_cv": 2, "inner_split_seed": 20261031, "model_seed": 42,
              "main_tasks": ["binary_kinds", "acid_origins"],
              "R0_grid": historical_grid(), "R1_mc_grid": mc_grid(), "R2_svm_grid": svm_grid(),
              "secondary_fixed_MC20": MC20, "calibrators": CALIBRATORS,
              "support_shots_main": [1, 3, 5, 10], "support_repeats_main": 20,
              "source_vote_rule": "SVC.predict; break_ties=False",
              "calibration_score_rule": "fixed-order score argmax with its own zero-support control",
              "full_refit_policy": "same locked source hyperparameters; re-fit every transformation on train+cal only",
              "final_prediction_used_for_selection": False,
              "no_external_or_cross_batch_claim": True}
    freeze(args.output / "frozen_config.json", frozen)
    results = []
    with threadpool_limits(limits=1):
        for task in args.tasks:
            results.append(run_task(task, arrays, roles, X, axis, args.output))
    write_json(args.output / "summary.json", {"frozen_config_sha256": digest(args.output / "frozen_config.json"),
                                              "tasks": results})


if __name__ == "__main__":
    main()
