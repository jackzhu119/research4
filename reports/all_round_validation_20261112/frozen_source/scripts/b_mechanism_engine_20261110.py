"""Frozen E01--E03 design and atomic, resumable scientific task execution.

No query labels are accepted by fitting helpers. Query labels are joined only
after all source/calibrated/retrained predictions have been produced. This is
an internal development experiment, not independent new-batch validation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold

from herb_spectra.calibration import AnchorScoreCalibrator
from herb_spectra.spectral_models import fit_source_candidate


K_VALUES = (0, 1, 3, 5, 10)
BINARY_SEEDS = (42, 43, 44)
OUTER_SEEDS = (20261031, 20261108, 42)
ACID_ORIGINS = ("ZY03", "ZY04", "ZY05", "ZY06", "ZY07")
RATIOS = {"A": (60, 180), "B": (120, 120), "C": (180, 60)}
CANDIDATE = {"id": "fixed_snv_rbf_svm_c10", "preprocessing": "snv",
             "model": "svc", "C": 10.0}
CALIBRATION_METHODS = ("identity", "bias_only", "anchor_lambda1",
                       "weak_lambda1000_blend005", "full_pipeline_retrain")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _metadata(metadata):
    frame = metadata.copy().reset_index(drop=True)
    required = {"record_id", "physical_sample_id", "origin_group_id", "species", "role"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing metadata columns: {sorted(required - set(frame.columns))}")
    if frame["record_id"].isna().any() or not frame["record_id"].is_unique:
        raise ValueError("record_id must be nonmissing and unique")
    if frame["physical_sample_id"].isna().any() or not frame["physical_sample_id"].is_unique:
        raise ValueError("physical_sample_id must be nonmissing and unique")
    for column in required:
        frame[column] = frame[column].astype(str)
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"Empty {column}")
    expected = {"ZY01": "枣仁", "ZY02": "枣仁", **{g: "酸枣仁" for g in ACID_ORIGINS}}
    for group, species in expected.items():
        if not frame.loc[frame.origin_group_id.eq(group), "species"].eq(species).all():
            raise ValueError(f"Current species semantics violated for {group}")
    return frame


def _permuted(values, rng):
    return rng.permutation(sorted(str(v) for v in values)).tolist()


def nested_support(job, metadata, k):
    if k not in K_VALUES:
        raise ValueError("K is outside frozen design")
    frame = _metadata(metadata).set_index("record_id", drop=False)
    result = []
    for label in job["classes"]:
        members = [sample for sample in job["support_ids"]
                   if frame.at[sample, job["label_column"]] == label]
        if len(members) != 10:
            raise ValueError("Each support candidate class must contain exactly 10 IDs")
        result.extend(members[:k])
    return result


def validate_job(job, metadata):
    frame = _metadata(metadata).set_index("record_id", drop=False)
    roles = [job[f"{name}_ids"] for name in ("source", "support", "query")]
    sets = [set(ids) for ids in roles]
    if any(len(ids) != len(unique) for ids, unique in zip(roles, sets)):
        raise ValueError("Repeated record ID within a role")
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Source, Support and Query IDs must be disjoint")
    all_ids = set.union(*sets)
    if not all_ids.issubset(frame.index):
        raise ValueError("Unknown sample ID")
    if not frame.loc[list(all_ids), "role"].str.lower().eq("train").all():
        raise ValueError("Only master TRAIN may enter the new experiment")
    if not frame.loc[list(all_ids), "physical_sample_id"].is_unique:
        raise ValueError("Physical samples overlap across experiment roles")
    source = frame.loc[job["source_ids"]]
    support = frame.loc[job["support_ids"]]
    query = frame.loc[job["query_ids"]]
    labels = job["label_column"]
    if sorted(source[labels].unique()) != job["classes"]:
        raise ValueError("Source class vocabulary mismatch")
    counts = source[labels].value_counts().to_dict()
    if counts != job["source_class_counts"]:
        raise ValueError("Frozen source class counts mismatch")
    if any(int((support[labels] == label).sum()) != 10 for label in job["classes"]):
        raise ValueError("Support pool must have ten samples per class")
    if job["experiment"] == "E01":
        acid_n, zaoren_n = RATIOS[job["ratio"]]
        if len(source) != 240 or counts != {"枣仁": zaoren_n, "酸枣仁": acid_n}:
            raise ValueError("E01 source count or class ratio mismatch")
        if len(query) != 40 or any(int((query.species == s).sum()) != 20 for s in job["classes"]):
            raise ValueError("E01 query must contain 20+20 samples")
        included = int(source.origin_group_id.eq(job["target_origin"]).sum())
        if included != 20 * job["coverage"]:
            raise ValueError("E01 target-source coverage mismatch")
        if not source.loc[source.species.eq("枣仁"), "origin_group_id"].eq("ZY01").all():
            raise ValueError("E01 zaoren Source must use ZY01")
        for side in (support, query):
            if not side.loc[side.species.eq("枣仁"), "origin_group_id"].eq("ZY01").all():
                raise ValueError("E01 zaoren Support/Query must use ZY01")
            if not side.loc[side.species.eq("酸枣仁"), "origin_group_id"].eq(job["target_origin"]).all():
                raise ValueError("E01 acid Support/Query must use target origin")
    elif job["experiment"] == "E03":
        if (len(source), len(support), len(query)) != (190, 50, 60):
            raise ValueError("E03 roles must be 190/50/60")
        if not frame.loc[list(all_ids), "species"].eq("酸枣仁").all():
            raise ValueError("E03 permits pure acid-jujube TRAIN only")
        if any(int((query.origin_group_id == g).sum()) != 12 for g in ACID_ORIGINS):
            raise ValueError("E03 Query must have twelve samples per class")
    else:
        raise ValueError("Unknown experiment")
    return True


def build_design(metadata):
    """Freeze all 105 source models, without looking at spectral/query values."""
    frame = _metadata(metadata)
    train = frame.loc[frame.role.str.lower().eq("train")]
    jobs = []
    for target in ACID_ORIGINS:
        target_ids = train.loc[train.origin_group_id.eq(target), "record_id"]
        zaoren_ids = train.loc[train.origin_group_id.eq("ZY01"), "record_id"]
        other_ids = train.loc[train.origin_group_id.isin(set(ACID_ORIGINS) - {target}), "record_id"]
        if (len(target_ids), len(zaoren_ids), len(other_ids)) != (60, 210, 240):
            raise ValueError(f"INFEASIBLE {target}: expected target60/ZY01210/other-acid240; "
                             f"got {len(target_ids)}/{len(zaoren_ids)}/{len(other_ids)}")
        for seed in BINARY_SEEDS:
            rng = np.random.default_rng(seed)
            target_order = _permuted(target_ids, rng)
            zaoren_order = _permuted(zaoren_ids, rng)
            other_order = _permuted(other_ids, rng)
            query_ids = zaoren_order[:20] + target_order[:20]
            support_ids = zaoren_order[20:30] + target_order[20:30]
            for ratio, (acid_n, zaoren_n) in RATIOS.items():
                for coverage in (0, 1):
                    source_ids = (zaoren_order[30:30 + zaoren_n]
                                  + other_order[:acid_n - 20 * coverage]
                                  + (target_order[30:50] if coverage else []))
                    job = {"job_id": f"E01_{target}_s{seed}_r{ratio}_c{coverage}",
                           "experiment": "E01", "target_origin": target,
                           "seed": seed, "coverage": coverage, "ratio": ratio, "fold": None,
                           "source_ids": source_ids, "support_ids": support_ids.copy(),
                           "query_ids": query_ids.copy(), "classes": ["枣仁", "酸枣仁"],
                           "label_column": "species", "source_class_counts": {"枣仁": zaoren_n, "酸枣仁": acid_n}}
                    validate_job(job, frame)
                    jobs.append(job)
    pure = train.loc[train.origin_group_id.isin(ACID_ORIGINS)].sort_values("record_id").reset_index(drop=True)
    if len(pure) != 300 or not pure.groupby("origin_group_id").size().eq(60).all():
        raise ValueError("INFEASIBLE E03: require 300 pure acid TRAIN, 60 per origin")
    for seed in OUTER_SEEDS:
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (inside, query) in enumerate(folds.split(pure, pure.origin_group_id)):
            rng = np.random.default_rng(np.random.SeedSequence([seed, fold, 20261110]))
            remaining = pure.iloc[inside]
            source_ids, support_ids = [], []
            for group in ACID_ORIGINS:
                order = _permuted(remaining.loc[remaining.origin_group_id.eq(group), "record_id"], rng)
                support_ids.extend(order[:10])
                source_ids.extend(order[10:])
            job = {"job_id": f"E03_s{seed}_f{fold}", "experiment": "E03",
                   "target_origin": "ZY03-ZY07", "seed": seed, "coverage": None,
                   "ratio": None, "fold": fold, "source_ids": source_ids,
                   "support_ids": support_ids, "query_ids": pure.iloc[query].record_id.tolist(),
                   "classes": list(ACID_ORIGINS), "label_column": "origin_group_id",
                   "source_class_counts": {g: 38 for g in ACID_ORIGINS}}
            validate_job(job, frame)
            jobs.append(job)
    return jobs


def fit_source(X_source, y_source, axis, seed=42):
    """Source-only interface; neither support nor query labels are accepted."""
    return fit_source_candidate(CANDIDATE, X_source, y_source, axis, seed=seed)


def native_predict(source, X):
    return source.estimator_.predict(source._scaled(X))


def calibrate_scores(method, support_scores, support_y, query_scores, classes):
    """Support-only calibration; zero support always gives exact source scores."""
    query_scores = np.asarray(query_scores)
    count = len(support_y)
    if method == "identity" or count == 0:
        return query_scores.copy(), {"method": method, "identity": True,
                                     "coef": np.eye(len(classes)).tolist(),
                                     "intercept": np.zeros(len(classes)).tolist()}
    support_scores = np.asarray(support_scores)
    if method == "bias_only":
        vocabulary = {label: index for index, label in enumerate(classes)}
        one_hot = np.eye(len(classes))[[vocabulary[v] for v in support_y]]
        bias = (one_hot - support_scores).sum(axis=0) / (count + 1.0)
        return query_scores + bias, {"method": method, "identity": False,
                                    "regularization": 1.0, "coef": np.eye(len(classes)).tolist(),
                                    "intercept": bias.tolist(),
                                    "formula": "sum(onehot-support_score)/(N+lambda), lambda=1"}
    if method not in {"anchor_lambda1", "weak_lambda1000_blend005"}:
        raise ValueError("Unknown frozen score correction")
    penalty = 1.0 if method == "anchor_lambda1" else 1000.0
    blend = 1.0 if method == "anchor_lambda1" else 0.05
    adapter = AnchorScoreCalibrator(classes, regularization=penalty).fit(support_scores, support_y)
    coefficient = np.eye(len(classes)) + blend * (adapter.coef_ - np.eye(len(classes)))
    intercept = blend * adapter.intercept_
    return query_scores @ coefficient + intercept, {
        "method": method, "identity": False, "regularization": penalty, "blend": blend,
        "coef": coefficient.tolist(), "intercept": intercept.tolist()}


def _metrics(true, pred, classes):
    return {"n_query": len(true), "n_unique_query": len(true),
            "accuracy": float(accuracy_score(true, pred)),
            "macro_f1": float(f1_score(true, pred, labels=classes, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
            "recall_json": json.dumps(dict(zip(classes, recall_score(true, pred, labels=classes,
                                                       average=None, zero_division=0).tolist())), ensure_ascii=False),
            "confusion_matrix_json": json.dumps(confusion_matrix(true, pred, labels=classes).tolist()),
            "classes_json": json.dumps(classes, ensure_ascii=False)}


def _job_context(job):
    return {key: job[key] for key in ("job_id", "experiment", "target_origin", "seed", "coverage", "ratio", "fold")}


def execute_job(job, X, metadata, outdir, input_hash, resume=True, axis=None):
    """Fit a source model and all fixed adaptations, then atomically seal task.

    ``X`` rows must match ``metadata`` exactly. ``axis`` is the verified alpha
    axis from the feature cache. A synthetic integer axis is for unit tests only.
    Runtime, accuracy and CPU are recorded and never used as stopping rules.
    """
    frame = _metadata(metadata)
    validate_job(job, frame)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) != len(frame) or not np.isfinite(X).all():
        raise ValueError("Feature rows must match metadata and contain finite values")
    axis = np.arange(X.shape[1], dtype=float) if axis is None else np.asarray(axis, dtype=float)
    if axis.shape != (X.shape[1],) or not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0):
        raise ValueError("Invalid feature axis")
    base = Path(outdir) / "jobs"
    base.mkdir(parents=True, exist_ok=True)
    destination = base / job["job_id"]
    if destination.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite completed task {destination}")
        return verify_job(destination, input_hash, expected_job=job)
    temporary = base / f".{job['job_id']}.tmp.{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    start_cpu, start_wall = time.process_time(), time.monotonic()
    rows = {sample: i for i, sample in enumerate(frame.record_id)}
    indexed = frame.set_index("record_id", drop=False)
    get_X = lambda ids: X[[rows[s] for s in ids]]
    source_y = indexed.loc[job["source_ids"], job["label_column"]].to_numpy()
    fit_cpu, fit_wall = time.process_time(), time.monotonic()
    source = fit_source(get_X(job["source_ids"]), source_y, axis, seed=42)
    source_cpu = time.process_time() - fit_cpu
    source_wall = time.monotonic() - fit_wall
    joblib.dump(source, temporary / "source_model.joblib", compress=3)
    source_before = joblib.hash(source)
    classes = source.classes_.tolist()
    if classes != job["classes"]:
        raise ValueError("Fitted class ordering differs from design")
    query_X = get_X(job["query_ids"])
    query_scores = source.predict_scores(query_X)
    native = native_predict(source, query_X)
    prediction_sets, costs, adapters = [], [], []
    source_margin = ((query_scores[:, 1] - query_scores[:, 0]) / 2 if len(classes) == 2
                     else np.sort(query_scores, axis=1)[:, -1] - np.sort(query_scores, axis=1)[:, -2])
    for k in K_VALUES:
        support_ids = nested_support(job, frame, k)
        support_y = indexed.loc[support_ids, job["label_column"]].to_numpy()
        support_scores = (source.predict_scores(get_X(support_ids)) if support_ids
                          else np.empty((0, len(classes))))
        support_allocation = len(support_ids)
        for method in ("identity", "svc_predict", "decision_sign") + CALIBRATION_METHODS[1:]:
            if method == "decision_sign" and len(classes) != 2:
                continue
            method_cpu, method_wall = time.process_time(), time.monotonic()
            adapter = None
            if method == "full_pipeline_retrain":
                if k:
                    fit_ids = job["source_ids"] + support_ids
                    model = fit_source(get_X(fit_ids), indexed.loc[fit_ids, job["label_column"]].to_numpy(), axis, seed=42)
                    scores = model.predict_scores(query_X)
                    pred = native_predict(model, query_X)
                    joblib.dump(model, temporary / f"retrain_k{k}.joblib", compress=3)
                else:
                    scores, pred = query_scores.copy(), native.copy()
                rule = "SVC.predict"
            elif method == "svc_predict":
                scores, pred, rule = query_scores.copy(), native.copy(), "SVC.predict"
            elif method == "decision_sign":
                scores = query_scores.copy()
                pred = np.asarray(classes)[(source_margin > 0).astype(int)]
                rule = "decision_function>0; first class on zero"
            else:
                scores, adapter = calibrate_scores(method, support_scores, support_y, query_scores, classes)
                pred, rule = np.asarray(classes)[np.argmax(scores, axis=1)], "argmax_class_aligned_scores; first class on tie"
            elapsed_cpu, elapsed_wall = time.process_time() - method_cpu, time.monotonic() - method_wall
            cost = {"method": method, "k": k, "cpu_seconds": elapsed_cpu, "wall_seconds": elapsed_wall}
            costs.append(cost)
            if adapter is not None:
                adapters.append({"k": k, **adapter})
            prediction_sets.append({"method": method, "k": k, "scores": scores,
                                    "pred": pred, "decision_rule": rule,
                                    "total_support_labels": support_allocation,
                                    "support_ids": support_ids, "adapter": adapter, **cost})
    if joblib.hash(source) != source_before:
        raise AssertionError("Frozen source model was mutated by adaptation")
    # Query labels first enter here, after all training/adaptation has completed.
    true = indexed.loc[job["query_ids"], job["label_column"]].to_numpy()
    prediction_rows, metric_rows, mechanism_rows = [], [], []
    context = _job_context(job)
    for value in prediction_sets:
        common = {**context, "method": value["method"], "k": value["k"],
                  "total_support_labels": value["total_support_labels"],
                  "support_labels_used": (value["total_support_labels"] if value["method"]
                                          not in {"identity", "svc_predict", "decision_sign"} else 0),
                  "decision_rule": value["decision_rule"]}
        metric_rows.append({**common, **_metrics(true, value["pred"], classes),
                            "cpu_seconds": value["cpu_seconds"], "wall_seconds": value["wall_seconds"]})
        margins = ((value["scores"][:, 1] - value["scores"][:, 0]) / 2 if len(classes) == 2
                   else np.sort(value["scores"], axis=1)[:, -1] - np.sort(value["scores"], axis=1)[:, -2])
        for index, sample in enumerate(job["query_ids"]):
            prediction_rows.append({**common, "record_id": sample,
                                    "physical_sample_id": indexed.at[sample, "physical_sample_id"],
                                    "origin_group_id": indexed.at[sample, "origin_group_id"],
                                    "true_label": true[index], "pred_label": value["pred"][index],
                                    "source_margin": float(source_margin[index]), "margin": float(margins[index]),
                                    "source_scores_json": json.dumps(query_scores[index].tolist()),
                                    "scores_json": json.dumps(value["scores"][index].tolist()),
                                    "support_ids_json": json.dumps(value["support_ids"], ensure_ascii=False)})
        if len(classes) == 2 and value["adapter"] is not None:
            coefficient = np.asarray(value["adapter"]["coef"])
            intercept = np.asarray(value["adapter"]["intercept"])
            a = float((-coefficient[0, 1] + coefficient[1, 1]
                       + coefficient[0, 0] - coefficient[1, 0]) / 2)
            b = float((intercept[1] - intercept[0]) / 2)
            affine_error = float(np.max(np.abs(margins - (a * source_margin + b))))
            auc_before = float(roc_auc_score(true == classes[1], source_margin))
            auc_after = float(roc_auc_score(true == classes[1], margins))
            threshold = -b / a if a > 0 else None
            equivalent = (bool(np.array_equal(np.asarray(classes)[(source_margin > threshold).astype(int)], value["pred"]))
                          if threshold is not None else None)
            mechanism_rows.append({**common, "a": a, "b": b, "a_positive": a > 0,
                                   "affine_max_error": affine_error,
                                   "rank_unchanged": bool(np.array_equal(np.argsort(source_margin, kind="stable"), np.argsort(margins, kind="stable"))),
                                   "auc_before": auc_before, "auc_after": auc_after,
                                   "threshold": threshold, "threshold_equivalence": equivalent,
                                   "a_nonpositive_separate": a <= 0})
    pd.DataFrame(prediction_rows).to_csv(temporary / "predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(temporary / "metrics.csv", index=False)
    mechanism_columns = list(context) + ["method", "k", "total_support_labels", "decision_rule", "a", "b", "a_positive", "affine_max_error", "rank_unchanged", "auc_before", "auc_after", "threshold", "threshold_equivalence", "a_nonpositive_separate"]
    pd.DataFrame(mechanism_rows, columns=mechanism_columns).to_csv(temporary / "mechanisms.csv", index=False)
    write_json(temporary / "calibration_parameters.json", adapters)
    write_json(temporary / "split_ids.json", {"job": job, "nested_support": {str(k): nested_support(job, frame, k) for k in K_VALUES}})
    source_composition = indexed.loc[job["source_ids"], "origin_group_id"].value_counts().to_dict()
    result = {**job, "status": "DONE", "input_hash": input_hash, "job_hash": canonical_hash(job),
              "source_model_unchanged": True, "source_origin_counts": source_composition,
              "candidate": CANDIDATE, "source_metadata": source.metadata,
              "cpu_seconds": time.process_time() - start_cpu,
              "wall_seconds": time.monotonic() - start_wall,
              "source_cpu_seconds": source_cpu, "source_wall_seconds": source_wall,
              "costs": costs, "scientific_model_fits": 5, "metric_rows": len(metric_rows),
              "prediction_rows": len(prediction_rows), "source_hash": file_hash(temporary / "source_model.joblib"),
              "artifact_hashes": {p.name: file_hash(p) for p in sorted(temporary.iterdir()) if p.is_file()}}
    write_json(temporary / "result.json", result)
    write_json(temporary / "COMPLETE.json", {"result_sha256": file_hash(temporary / "result.json"),
                                              "input_hash": input_hash, "job_hash": canonical_hash(job)})
    os.rename(temporary, destination)
    return result


def verify_job(path, input_hash, expected_job=None):
    """Independently reconstruct all metrics and validate sealed task bytes."""
    path = Path(path)
    result = json.loads((path / "result.json").read_text())
    marker = json.loads((path / "COMPLETE.json").read_text())
    if result["input_hash"] != input_hash or marker["input_hash"] != input_hash:
        raise ValueError("Input hash changed; refusing incorrect resume")
    if marker["result_sha256"] != file_hash(path / "result.json"):
        raise ValueError("Task result hash mismatch")
    if expected_job is not None and result["job_hash"] != canonical_hash(expected_job):
        raise ValueError("Frozen task definition changed; refusing incorrect resume")
    if marker["job_hash"] != result["job_hash"]:
        raise ValueError("Task marker job hash mismatch")
    for name, expected in result["artifact_hashes"].items():
        if Path(name).name != name or file_hash(path / name) != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    predictions = pd.read_csv(path / "predictions.csv", keep_default_na=False)
    metrics = pd.read_csv(path / "metrics.csv", keep_default_na=False)
    split_job = json.loads((path / "split_ids.json").read_text())["job"]
    if canonical_hash(split_job) != result["job_hash"] or any(result.get(key) != value for key, value in split_job.items()):
        raise ValueError("Saved task metadata differs from the frozen split definition")
    if len(predictions) != result["prediction_rows"] or len(metrics) != result["metric_rows"]:
        raise ValueError("Saved task endpoint row counts differ from the receipt")
    methods = {"identity", "svc_predict", "bias_only", "anchor_lambda1",
               "weak_lambda1000_blend005", "full_pipeline_retrain"}
    if len(result["classes"]) == 2:
        methods.add("decision_sign")
    expected_endpoints = {(method, k) for method in methods for k in K_VALUES}
    actual_endpoints = set(zip(metrics.method, metrics.k))
    if actual_endpoints != expected_endpoints or len(metrics) != len(expected_endpoints):
        raise ValueError("Frozen method/K endpoints are incomplete or duplicated")
    if set(zip(predictions.method, predictions.k)) != expected_endpoints:
        raise ValueError("Frozen prediction endpoints are incomplete")
    physical_ids = set(predictions.loc[predictions.method.eq("identity") & predictions.k.eq(0), "physical_sample_id"])
    for (_, k), block in predictions.groupby(["method", "k"]):
        if len(block) != len(result["query_ids"]) or block.physical_sample_id.nunique() != len(block):
            raise ValueError("Incomplete or duplicate task query predictions")
        if set(block.record_id) != set(result["query_ids"]) or set(block.physical_sample_id) != physical_ids:
            raise ValueError("Method query IDs differ from the frozen common query")
        method = block.iloc[0]["method"]
        metric = metrics.loc[metrics.method.eq(method) & metrics.k.eq(k)]
        if len(metric) != 1:
            raise ValueError("Metrics missing or duplicated")
        fresh = _metrics(block.true_label.to_numpy(), block.pred_label.to_numpy(), result["classes"])
        for column in ("accuracy", "macro_f1", "balanced_accuracy"):
            if not np.isclose(float(metric.iloc[0][column]), fresh[column], atol=1e-12, rtol=0):
                raise ValueError(f"Independent metric mismatch: {method}/{k}/{column}")
        for column in ("recall_json", "confusion_matrix_json", "classes_json"):
            if json.loads(metric.iloc[0][column]) != json.loads(fresh[column]):
                raise ValueError(f"Independent metric mismatch: {column}")
    reference = predictions.loc[predictions.method.eq("identity") & predictions.k.eq(0)].set_index("record_id")
    native = predictions.loc[predictions.method.eq("svc_predict") & predictions.k.eq(0)].set_index("record_id")
    for (method, k), block in predictions.groupby(["method", "k"]):
        aligned = block.set_index("record_id").loc[reference.index]
        if not np.array_equal(aligned.true_label.to_numpy(), reference.true_label.to_numpy()):
            raise ValueError("Matched endpoint true labels differ")
        if method not in {"svc_predict", "full_pipeline_retrain"}:
            scores = np.asarray([json.loads(value) for value in aligned.scores_json])
            reconstructed = np.asarray(result["classes"])[np.argmax(scores, axis=1)]
            if not np.array_equal(reconstructed, aligned.pred_label.to_numpy()):
                raise ValueError("Saved score argmax does not reproduce predictions")
    for method in methods:
        zero = predictions.loc[predictions.method.eq(method) & predictions.k.eq(0)].set_index("record_id").loc[reference.index]
        control = native if method in {"svc_predict", "full_pipeline_retrain"} else reference
        if not np.array_equal(zero.pred_label.to_numpy(), control.pred_label.to_numpy()):
            raise ValueError("Zero-support method differs from its matched decision control")
    return result
