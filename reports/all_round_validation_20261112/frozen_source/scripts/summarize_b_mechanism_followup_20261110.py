#!/usr/bin/env python3
"""Summarize sealed follow-up experiments without fitting any models.

All intervals resample physical specimens, carrying their repeated predictions
together. Support redraws and new split seeds are internal repetitions, not new
specimens, source populations, or batches.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports/b_mechanism_followup_20261110"
REPORT = ROOT / "docs/B_MECHANISM_FOLLOWUP_RESULTS_20261110.md"
FAMILIES = {"F01": 45, "F02": 45, "F03": 240, "F04": 15}
F01_TRUE = (
    "raw_anchor_lambda1", "raw_bias_only", "raw_weak_lambda1000_blend005",
    "diagonal_anchor_lambda1", "scale01_anchor", "scale10_anchor",
    "rowcenter_source_rms_anchor",
)
LIMITS = (
    "本轮为依据上一轮结果预先冻结的探索性补充。仅使用master TRAIN；样品已有历史曝光。"
    "支持重排和新增划分种子不增加独立实物数。描述性95%区间按physical_sample_id聚类，"
    "不覆盖来源总体、批次总体、协议选择或长期开发偏倚。来源和批次混杂，不能据此宣称真实跨批次泛化或确认性因果机制。"
)
CONTEXT = ("family", "experiment", "parent_job_id", "target_origin", "seed", "ratio", "coverage", "fold")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".summary-", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_text(path, json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, frame, columns=None):
    if frame.empty and not len(frame.columns):
        frame = pd.DataFrame(columns=columns or ["status", "reason"])
    atomic_text(path, frame.to_csv(index=False))


def frozen_design(output):
    path = Path(output) / "frozen_protocol.json"
    if not path.is_file():
        return [], None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for field in ("design", "jobs", "tasks"):
        if isinstance(payload.get(field), list):
            return payload[field], payload.get("input_hash")
    return [], payload.get("input_hash")


def independent_metrics(rows, classes):
    """Independent confusion-count calculation, deliberately not sklearn's path."""
    vocabulary = {str(label): index for index, label in enumerate(classes)}
    try:
        actual = np.fromiter((vocabulary[str(x)] for x in rows.true_label), dtype=int)
        predicted = np.fromiter((vocabulary[str(x)] for x in rows.pred_label), dtype=int)
    except KeyError as error:
        raise ValueError(f"Unknown label in predictions: {error}") from error
    count = len(classes)
    cm = np.bincount(actual * count + predicted, minlength=count * count).reshape(count, count)
    actual_n, predicted_n = cm.sum(axis=1), cm.sum(axis=0)
    diagonal = cm.diagonal()
    recall = np.divide(diagonal, actual_n, out=np.zeros(count, dtype=float), where=actual_n > 0)
    f1 = np.divide(2 * diagonal, actual_n + predicted_n, out=np.zeros(count, dtype=float), where=(actual_n + predicted_n) > 0)
    return {
        "n_query": len(rows), "n_predictions": len(rows),
        "n_physical_samples": int(rows.physical_sample_id.nunique()),
        "correct": int(diagonal.sum()), "accuracy": float(diagonal.sum() / len(rows)),
        "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall[actual_n > 0].mean()),
        "class_recall_json": json.dumps(dict(zip(map(str, classes), recall.tolist())), ensure_ascii=False, sort_keys=True),
        "confusion_matrix_json": json.dumps(cm.tolist()),
        "classes_json": json.dumps(list(map(str, classes)), ensure_ascii=False),
    }


def load_jobs(output):
    """Read complete hash-sealed bundles, then reconstruct each endpoint metric."""
    output = Path(output)
    design, input_hash = frozen_design(output)
    design_by_id = {str(item["job_id"]): item for item in design}
    jobs, frames, metric_rows, diagnostics, manifest = [], [], [], [], []
    checks = 0
    for path in sorted((output / "jobs").glob("*/result.json")):
        directory = path.parent
        if directory.name.startswith(".") or not (directory / "COMPLETE.json").is_file():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        job = result.get("job", result)
        merged = {**job, **result}
        marker = json.loads((directory / "COMPLETE.json").read_text(encoding="utf-8"))
        if result.get("status") != "DONE" or merged.get("job_id") != directory.name:
            raise ValueError(f"Invalid task identity or completion status: {directory}")
        if marker.get("result_sha256") != sha256(path):
            raise ValueError(f"Sealed result hash mismatch: {path}")
        if marker.get("input_hash") != result.get("input_hash") or (input_hash and input_hash != result.get("input_hash")):
            raise ValueError(f"Frozen input hash mismatch: {directory}")
        if marker.get("job_hash") != result.get("job_hash"):
            raise ValueError(f"Task definition seal mismatch: {directory}")
        if design and merged["job_id"] not in design_by_id:
            raise ValueError(f"Unplanned task: {directory}")
        if design:
            frozen_job = design_by_id[merged["job_id"]]
            expected_hash = hashlib.sha256(json.dumps(frozen_job, ensure_ascii=False, sort_keys=True,
                                                      separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            if result.get("job_hash") != expected_hash:
                raise ValueError(f"Task differs from frozen design: {directory}")
        for name, expected in result.get("artifact_hashes", {}).items():
            artifact = directory / name
            if Path(name).name != name or not artifact.is_file() or sha256(artifact) != expected:
                raise ValueError(f"Artifact hash mismatch: {artifact}")
        prediction_path = directory / "predictions.csv.gz"
        if not prediction_path.is_file():
            prediction_path = directory / "predictions.csv"
        rows = pd.read_csv(prediction_path, keep_default_na=False,
                           dtype={"record_id": str, "physical_sample_id": str, "true_label": str, "pred_label": str})
        saved_metrics = pd.read_csv(directory / "metrics.csv", keep_default_na=False)
        required = {"endpoint_id", "method", "k", "draw", "record_id", "physical_sample_id", "true_label", "pred_label"}
        if not required.issubset(rows):
            raise ValueError(f"Prediction columns missing: {required - set(rows)}")
        if rows.duplicated(["endpoint_id", "physical_sample_id"]).any():
            raise ValueError(f"Duplicate physical ID in an endpoint: {directory}")
        if saved_metrics.endpoint_id.duplicated().any() or set(saved_metrics.endpoint_id) != set(rows.endpoint_id):
            raise ValueError(f"Endpoint coverage differs between metrics and predictions: {directory}")
        expected_endpoints = result.get("expected_endpoint_ids", result.get("expected_endpoints"))
        if expected_endpoints and set(expected_endpoints) != set(rows.endpoint_id):
            raise ValueError(f"Declared endpoint coverage incomplete: {directory}")
        classes = merged["classes"]
        query_ids = set(map(str, merged["query_ids"]))
        contexts = {key: merged.get(key) for key in CONTEXT}
        for key, value in contexts.items():
            rows[key] = value
        rows["job_id"] = merged["job_id"]
        rows["k"] = rows.k.astype(int)
        rows["draw"] = rows.draw.astype(int)
        rows["seed"] = pd.to_numeric(rows.seed, errors="coerce").fillna(-1).astype(int)
        rows["fold"] = pd.to_numeric(rows.fold, errors="coerce").fillna(-1).astype(int)
        rows["coverage"] = pd.to_numeric(rows.coverage, errors="coerce").fillna(-1).astype(int)
        rows["ratio"] = rows.ratio.replace({None: "NA", "": "NA"}).fillna("NA")
        if rows.groupby("physical_sample_id").true_label.nunique().max() != 1:
            raise ValueError(f"A physical specimen has inconsistent true labels: {directory}")
        saved_by_endpoint = saved_metrics.set_index("endpoint_id")
        for endpoint, block in rows.groupby("endpoint_id", sort=False):
            if len(block) != len(query_ids) or set(block.record_id) != query_ids:
                raise ValueError(f"Query denominator changed: {directory}/{endpoint}")
            calculated = independent_metrics(block, classes)
            existing = saved_by_endpoint.loc[endpoint].to_dict()
            for column in ("accuracy", "macro_f1", "balanced_accuracy"):
                if not np.isclose(float(existing[column]), calculated[column], atol=1e-12, rtol=0):
                    raise ValueError(f"Independent metric mismatch: {directory}/{endpoint}/{column}")
                checks += 1
            for candidate in ("recall_json", "class_recall_json"):
                if candidate in existing:
                    if json.loads(existing[candidate]) != json.loads(calculated["class_recall_json"]):
                        raise ValueError(f"Independent recall mismatch: {directory}/{endpoint}")
                    checks += 1
            for column in ("confusion_matrix_json", "classes_json"):
                if column in existing:
                    if json.loads(existing[column]) != json.loads(calculated[column]):
                        raise ValueError(f"Independent matrix/vocabulary mismatch: {directory}/{endpoint}/{column}")
                    checks += 1
            combined = {**existing, **contexts, "job_id": merged["job_id"], "endpoint_id": endpoint,
                        "method": str(block.method.iloc[0]), "k": int(block.k.iloc[0]),
                        "draw": int(block.draw.iloc[0]), **calculated}
            if merged["family"] == "F04":
                combined["source_budget"] = int(existing.get("training_n", "".join(ch for ch in combined["method"] if ch.isdigit())))
            metric_rows.append(combined)
        for name in ("geometry.csv", "diagnostics.csv", "mechanisms.csv"):
            candidate = directory / name
            if candidate.is_file():
                try:
                    frame = pd.read_csv(candidate, keep_default_na=False)
                except pd.errors.EmptyDataError:
                    continue
                if not frame.empty:
                    for key, value in {**contexts, "job_id": merged["job_id"]}.items():
                        frame[key] = value
                    frame["analysis"] = "saved_" + name.split(".")[0]
                    diagnostics.append(frame)
        jobs.append(merged)
        frames.append(rows)
        manifest.append({**contexts, **{key: merged.get(key) for key in
                         ("job_id", "input_hash", "job_hash", "cpu_seconds", "wall_seconds", "new_model_fits", "new_adapter_fits")},
                         "status": "DONE", "result_sha256": sha256(path),
                         "result_path": str(path.relative_to(output)), "metric_rows": len(saved_metrics), "prediction_rows": len(rows)})
    return (jobs, pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
            pd.DataFrame(metric_rows), pd.concat(diagnostics, ignore_index=True) if diagnostics else pd.DataFrame(),
            pd.DataFrame(manifest), checks)


def pair(left, right, keys):
    """Match endpoints explicitly; a repeated baseline is supplied by the caller."""
    columns = keys + ["true_label", "pred_label"]
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError(f"Ambiguous paired comparison on {keys}")
    joined = left[columns].merge(right[columns], on=keys, how="outer", suffixes=("_before", "_after"),
                                 indicator=True, validate="one_to_one")
    if not joined._merge.eq("both").all():
        raise ValueError("Comparison has incomplete matched predictions")
    if not joined.true_label_before.eq(joined.true_label_after).all():
        raise ValueError("Matched true labels disagree")
    joined["before_correct"] = joined.true_label_before.eq(joined.pred_label_before).astype(int)
    joined["after_correct"] = joined.true_label_after.eq(joined.pred_label_after).astype(int)
    joined["delta"] = joined.after_correct - joined.before_correct
    return joined.drop(columns="_merge")


def repeat_baseline(baseline, right):
    """Repeat a locked zero-label endpoint to the right's explicit draws only."""
    keys = ["job_id", "physical_sample_id"]
    ref = baseline.drop(columns=["draw"], errors="ignore")
    if ref.duplicated(keys).any():
        raise ValueError("Locked baseline is not unique per task/specimen")
    layout = right[["job_id", "physical_sample_id", "draw"]].drop_duplicates()
    return layout.merge(ref, on=keys, how="left", validate="many_to_one")


def cluster_interval(rows, n_boot=1000, seed=20261110):
    groups = rows.groupby("physical_sample_id", sort=True).delta.agg(["sum", "count"])
    if groups.empty:
        return None, None
    values, weights = groups["sum"].to_numpy(), groups["count"].to_numpy()
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, n_boot, 100):
        picks = rng.integers(0, len(groups), size=(min(100, n_boot - start), len(groups)))
        estimates.extend((values[picks].sum(axis=1) / weights[picks].sum(axis=1)).tolist())
    lower, upper = np.quantile(estimates, [.025, .975])
    return float(lower), float(upper)


def transition_record(rows, info, n_boot):
    recalls = {str(label): float(group.after_correct.mean() - group.before_correct.mean())
               for label, group in rows.groupby("true_label_before")}
    before, after = rows.before_correct.to_numpy(), rows.after_correct.to_numpy()
    low, high = cluster_interval(rows, n_boot=n_boot)
    shared = {**info, "n_predictions": len(rows), "n_physical_samples": int(rows.physical_sample_id.nunique()),
              "sampling_structure": "paired internal repeats; all rows for one physical_sample_id kept in each cluster"}
    result = {**shared, "before_accuracy": float(before.mean()), "after_accuracy": float(after.mean()),
              "delta_accuracy": float((after - before).mean()),
              "wrong_to_correct": int(((before == 0) & (after == 1)).sum()),
              "correct_to_wrong": int(((before == 1) & (after == 0)).sum()),
              "recall_deltas_json": json.dumps(recalls, ensure_ascii=False, sort_keys=True),
              "recall_tradeoff": any(v > 0 for v in recalls.values()) and any(v < 0 for v in recalls.values()),
              "ci_lower": low, "ci_upper": high}
    interval = {**shared, "effect": "paired_delta_accuracy", "estimate": result["delta_accuracy"],
                "ci_lower": low, "ci_upper": high, "confidence_level": .95, "n_bootstrap": n_boot,
                "resampling_unit": "physical_sample_id", "interval_scope": LIMITS}
    return result, interval


def comparison_tables(predictions, n_boot=1000):
    transitions, intervals, interactions = [], [], []
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def add(left, right, info, keys, scopes=False):
        if left.empty or right.empty:
            return None
        common = set(left.job_id) & set(right.job_id) if "job_id" in keys else None
        if common is not None:
            left, right = left[left.job_id.isin(common)], right[right.job_id.isin(common)]
        if left.empty or right.empty:
            return None
        matched = pair(left, right, keys)
        for scope, block in [("ALL", matched)] + ([(str(g), b) for g, b in matched.groupby("target_origin")]
                                                   if scopes and "target_origin" in matched else []):
            rec, ci = transition_record(block, {**info, "scope": scope}, n_boot)
            transitions.append(rec)
            intervals.append(ci)
        return matched

    def add_interaction(first, second, info, keys, scopes=False):
        if first is None or second is None or first.empty or second.empty:
            return
        joined = first[keys + ["delta"]].merge(second[keys + ["delta"]], on=keys,
                                                suffixes=("_first", "_second"), how="outer",
                                                indicator=True, validate="one_to_one")
        if not joined._merge.eq("both").all():
            raise ValueError("Interaction has incomplete matched cells")
        joined["delta"] = joined.delta_second - joined.delta_first
        scoped = [("ALL", joined)]
        if scopes and "target_origin" in joined:
            scoped += [(str(origin), group) for origin, group in joined.groupby("target_origin")]
        for scope, block in scoped:
            low, high = cluster_interval(block, n_boot=n_boot)
            item = {**info, "scope": scope, "effect": "difference_of_differences_accuracy",
                    "estimate": float(block.delta.mean()), "ci_lower": low, "ci_upper": high,
                    "confidence_level": .95, "n_bootstrap": n_boot, "n_predictions": len(block),
                    "n_physical_samples": int(block.physical_sample_id.nunique()),
                    "resampling_unit": "physical_sample_id", "interval_scope": LIMITS}
            intervals.append(item)
            interactions.append({**item, "analysis": "factor_interaction"})

    def matched_if_complete(left, right, keys):
        if left.empty or right.empty:
            return None
        return pair(left, right, keys)

    # F01: comparisons use an explicit locked baseline, never a selected redraw.
    f01 = predictions[predictions.family.eq("F01")]
    for experiment, data in f01.groupby("experiment"):
        identity = data[data.method.eq("identity") & data.k.eq(0)]
        keys = ["job_id", "draw", "physical_sample_id"]
        for method in F01_TRUE + ("shuffled_anchor_lambda1",):
            for k in (1, 3, 5, 10):
                right = data[data.method.eq(method) & data.k.eq(k)]
                if right.empty:
                    continue
                left = repeat_baseline(identity, right)
                add(left, right, {"family": "F01", "experiment": experiment, "comparison": method + "_minus_identity",
                                 "method": method, "k": k}, keys)
                if method != "raw_anchor_lambda1":
                    anchor = data[data.method.eq("raw_anchor_lambda1") & data.k.eq(k)]
                    if k == 10 and method == "shuffled_anchor_lambda1":
                        anchor = repeat_baseline(anchor, right)
                    add(anchor, right, {"family": "F01", "experiment": experiment,
                                       "comparison": method + "_minus_raw_anchor_lambda1", "method": method, "k": k}, keys)
        native = data[data.method.eq("svc_predict") & data.k.eq(0)]
        add(native, identity, {"family": "F01", "experiment": experiment,
                              "comparison": "score_argmax_minus_native_svc_zero_support", "method": "identity", "k": 0}, keys)

    # F02/F04 have method names deliberately discoverable from fixed endpoint metadata.
    f02 = predictions[predictions.family.eq("F02")]
    for experiment, data in f02.groupby("experiment"):
        baseline = data[data.method.eq("identity") & data.k.eq(0)]
        methods = sorted(set(data.method) - {"identity", "svc_predict"})
        keys = ["job_id", "draw", "physical_sample_id"]
        for k in (1, 3, 5, 10):
            for method in methods:
                right = data[data.method.eq(method) & data.k.eq(k)]
                add(repeat_baseline(baseline, right), right,
                    {"family": "F02", "experiment": experiment, "comparison": method + "_minus_source_baseline",
                     "method": method, "k": k}, keys)
            # Full pair set is small and frozen, retaining negative as well as positive effects.
            for index, earlier in enumerate(methods):
                for later in methods[index + 1:]:
                    add(data[data.method.eq(earlier) & data.k.eq(k)], data[data.method.eq(later) & data.k.eq(k)],
                        {"family": "F02", "experiment": experiment, "comparison": later + "_minus_" + earlier,
                         "method": later, "reference_method": earlier, "k": k}, keys)
            fixed_scaler_effect = matched_if_complete(
                data[data.method.eq("scaler_frozen_gamma_fixed") & data.k.eq(k)],
                data[data.method.eq("scaler_refit_gamma_fixed") & data.k.eq(k)], keys)
            scale_scaler_effect = matched_if_complete(
                data[data.method.eq("scaler_frozen_gamma_scale") & data.k.eq(k)],
                data[data.method.eq("scaler_refit_gamma_scale") & data.k.eq(k)], keys)
            add_interaction(fixed_scaler_effect, scale_scaler_effect,
                            {"family": "F02", "experiment": experiment,
                             "comparison": "scaler_refit_x_gamma_recalculation", "k": k}, keys)

    f03 = predictions[predictions.family.eq("F03")]
    if not f03.empty:
        # Native endpoints only for weight/coverage/ratio effects. Argmax is an interface diagnostic.
        methods = sorted(f03.method.unique())
        balanced = next((m for m in methods if "balanced" in m and "argmax" not in m and "score" not in m), None)
        unweighted = next((m for m in methods if any(x in m.lower() for x in ("none", "unweighted"))
                           and "argmax" not in m and "score" not in m), None)
        native = f03[f03.method.isin([m for m in (balanced, unweighted) if m])]
        for seed_set, seeds in (("all42_49", set(range(42, 50))), ("historical42_44", {42, 43, 44}), ("new45_49", set(range(45, 50)))):
            data = native[native.seed.isin(seeds)]
            keys = ["seed", "target_origin", "physical_sample_id"]
            for ratio in ("A", "B", "C"):
                for coverage in (0, 1):
                    cell = data[data.ratio.eq(ratio) & data.coverage.eq(coverage)]
                    if balanced and unweighted:
                        add(cell[cell.method.eq(balanced)], cell[cell.method.eq(unweighted)],
                            {"family": "F03", "experiment": "E01", "comparison": "unweighted_minus_balanced",
                             "ratio": ratio, "coverage": coverage, "seed_set": seed_set, "k": 0}, keys, scopes=True)
            for method in (balanced, unweighted):
                if not method:
                    continue
                sub = data[data.method.eq(method)]
                cells = sub[["seed", "target_origin", "ratio", "coverage"]].drop_duplicates()
                complete = cells.groupby(["seed", "target_origin"]).size()
                complete = complete[complete.eq(6)].index
                sub = sub.set_index(["seed", "target_origin"]).loc[lambda frame: frame.index.isin(complete)].reset_index()
                for ratio in ("A", "B", "C"):
                    cell = sub[sub.ratio.eq(ratio)]
                    add(cell[cell.coverage.eq(0)], cell[cell.coverage.eq(1)],
                        {"family": "F03", "experiment": "E01", "comparison": "coverage_1_minus_0",
                         "method": method, "ratio": ratio, "seed_set": seed_set, "k": 0}, keys, scopes=True)
                for coverage in (0, 1):
                    cell = sub[sub.coverage.eq(coverage)]
                    for earlier, later in (("A", "B"), ("A", "C"), ("B", "C")):
                        add(cell[cell.ratio.eq(earlier)], cell[cell.ratio.eq(later)],
                            {"family": "F03", "experiment": "E01", "comparison": f"ratio_{later}_minus_{earlier}",
                             "method": method, "coverage": coverage, "seed_set": seed_set, "k": 0}, keys, scopes=True)
            # Difference-of-differences requires complete 2x3 cells for both weights.
            if balanced and unweighted:
                counts = data[["seed", "target_origin", "ratio", "coverage", "method"]].drop_duplicates().groupby(["seed", "target_origin"]).size()
                complete = counts[counts.eq(12)].index
                full = data.set_index(["seed", "target_origin"]).loc[lambda frame: frame.index.isin(complete)].reset_index()
                for ratio in ("A", "B", "C"):
                    effects = []
                    for method in (balanced, unweighted):
                        cell = full[full.method.eq(method) & full.ratio.eq(ratio)]
                        effects.append(matched_if_complete(cell[cell.coverage.eq(0)], cell[cell.coverage.eq(1)], keys))
                    add_interaction(effects[0], effects[1],
                                    {"family": "F03", "experiment": "E01", "comparison": "class_weight_x_coverage",
                                     "ratio": ratio, "seed_set": seed_set, "k": 0}, keys, scopes=True)
                for coverage in (0, 1):
                    for earlier, later in (("A", "B"), ("A", "C"), ("B", "C")):
                        effects = []
                        for method in (balanced, unweighted):
                            cell = full[full.method.eq(method) & full.coverage.eq(coverage)]
                            effects.append(matched_if_complete(cell[cell.ratio.eq(earlier)], cell[cell.ratio.eq(later)], keys))
                        add_interaction(effects[0], effects[1],
                                        {"family": "F03", "experiment": "E01", "comparison": f"class_weight_x_ratio_{later}_minus_{earlier}",
                                         "coverage": coverage, "seed_set": seed_set, "k": 0}, keys, scopes=True)

    f04 = predictions[predictions.family.eq("F04")]
    if not f04.empty:
        methods = sorted(f04.method.unique())
        def budget(name):
            digits = "".join(ch for ch in name if ch.isdigit())
            return int(digits) if digits else -1
        methods.sort(key=budget)
        baseline = next((m for m in methods if budget(m) == 190), None)
        keys = ["job_id", "draw", "physical_sample_id"]
        if baseline:
            for method in methods:
                add(f04[f04.method.eq(baseline)], f04[f04.method.eq(method)],
                    {"family": "F04", "experiment": "E03", "comparison": method + "_minus_source190",
                     "method": method, "source_budget": budget(method), "k": 0}, keys)
        for earlier, later in zip(methods[:-1], methods[1:]):
            add(f04[f04.method.eq(earlier)], f04[f04.method.eq(later)],
                {"family": "F04", "experiment": "E03", "comparison": later + "_minus_" + earlier,
                 "method": later, "source_budget": budget(later), "k": 0}, keys)
    return pd.DataFrame(transitions), pd.DataFrame(intervals), pd.DataFrame(interactions)


def mechanism_table(predictions, metrics, diagnostics):
    rows = diagnostics.to_dict("records") if not diagnostics.empty else []
    if metrics.empty:
        return pd.DataFrame(rows)
    f01 = metrics[metrics.family.eq("F01")]
    for (experiment, method, k), group in f01.groupby(["experiment", "method", "k"]):
        by_job = group.groupby("job_id").accuracy.agg(["mean", "std", "min", "max", "count"])
        rows.append({"analysis": "support_redraw_variability", "family": "F01", "experiment": experiment,
                     "method": method, "k": int(k), "n_jobs": len(by_job), "endpoint_rows": len(group),
                     "mean_accuracy_over_job_draws": float(group.accuracy.mean()),
                     "mean_within_job_draw_sd": float(by_job["std"].mean()),
                     "minimum_accuracy": float(group.accuracy.min()), "maximum_accuracy": float(group.accuracy.max()),
                     "min_draws_per_job": int(by_job["count"].min()), "max_draws_per_job": int(by_job["count"].max()),
                     "interpretation": "K10真标签完整池只保留一个端点；只有一个draw时SD未定义，不声称抽样稳定"})
    f02 = metrics[metrics.family.eq("F02")]
    if "gamma_identical_to_source" in f02:
        for (experiment, method, k), group in f02[~f02.method.eq("identity")].groupby(["experiment", "method", "k"]):
            equal = group.gamma_identical_to_source.astype(str).str.lower().eq("true")
            actual = pd.to_numeric(group.get("effective_gamma", pd.Series(index=group.index, dtype=float)), errors="coerce")
            original = pd.to_numeric(group.get("source_effective_gamma", pd.Series(index=group.index, dtype=float)), errors="coerce")
            difference = (actual - original).abs()
            rows.append({"analysis": "gamma_factor_identifiability", "family": "F02", "experiment": experiment,
                         "method": method, "k": int(k), "endpoint_rows": len(group),
                         "exact_source_gamma_count": int(equal.sum()), "different_source_gamma_count": int((~equal).sum()),
                         "maximum_absolute_gamma_difference": float(difference.max()),
                         "interpretation": "逐列StandardScaler使scale gamma通常等于1/特征数；完全相同的因子并未产生独立干预，ULP级差异不是生物学机制证据"})
    for (family, experiment, method, k, origin), group in predictions.groupby(["family", "experiment", "method", "k", "origin_group_id"]):
        rows.append({"analysis": "per_query_origin_performance", "family": family, "experiment": experiment,
                     "method": method, "k": int(k), "query_origin_group_id": origin,
                     "n_predictions": len(group), "n_physical_samples": int(group.physical_sample_id.nunique()),
                     "accuracy": float(group.true_label.eq(group.pred_label).mean())})
    f04 = metrics[metrics.family.eq("F04")]
    for job, group in f04.groupby("job_id"):
        if "source_budget" in group:
            group = group.sort_values("source_budget")
        else:
            group = group.sort_values("method")
        rows.append({"analysis": "learning_curve_monotonicity", "family": "F04", "job_id": job,
                     "methods_json": json.dumps(group.method.tolist()), "accuracy_json": json.dumps(group.accuracy.tolist()),
                     "monotone_nondecreasing": bool((np.diff(group.accuracy.to_numpy()) >= -1e-12).all())})
    return pd.DataFrame(rows)


def make_figures(output, metrics):
    if metrics.empty:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target = Path(output) / "figures"
    target.mkdir(parents=True, exist_ok=True)
    names = []
    f01 = metrics[metrics.family.eq("F01")]
    for experiment, group in f01.groupby("experiment"):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for method, line in group.groupby("method"):
            if method in ("identity", "svc_predict"):
                continue
            curve = line.groupby("k").accuracy.mean().sort_index()
            ax.plot(curve.index, curve.to_numpy(), marker="o", label=method)
        baseline = group[group.method.eq("identity")].accuracy.mean()
        ax.axhline(baseline, color="black", linestyle="--", label="locked identity")
        ax.set(xlabel="Support labels per class", ylabel="Accuracy (internal repeated predictions)", title=f"F01 {experiment}: all fixed methods and redraws", xticks=[1, 3, 5, 10], ylim=(0, 1.02))
        ax.legend(fontsize=7, loc="best"); ax.grid(alpha=.2); fig.tight_layout()
        path = target / f"f01_{experiment.lower()}_calibration.png"
        fig.savefig(path, dpi=160); plt.close(fig); names.append(str(path.relative_to(output)))
    f02 = metrics[metrics.family.eq("F02")]
    for experiment, group in f02.groupby("experiment"):
        fig, ax = plt.subplots(figsize=(9, 5))
        for method, line in group.groupby("method"):
            if method in ("identity", "svc_predict"):
                continue
            curve = line.groupby("k").accuracy.mean().sort_index()
            ax.plot(curve.index, curve.to_numpy(), marker="o", label=method)
        ax.set(xlabel="Support labels per class", ylabel="Accuracy", title=f"F02 {experiment}: scaler / gamma refit factors", xticks=[1, 3, 5, 10], ylim=(0, 1.02))
        ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
        path = target / f"f02_{experiment.lower()}_refit_factors.png"
        fig.savefig(path, dpi=160); plt.close(fig); names.append(str(path.relative_to(output)))
    f03 = metrics[metrics.family.eq("F03")]
    native_methods = [m for m in f03.method.unique() if "argmax" not in m and "score" not in m]
    if native_methods:
        fig, ax = plt.subplots(figsize=(8, 5))
        for (method, coverage), line in f03[f03.method.isin(native_methods)].groupby(["method", "coverage"]):
            curve = line.groupby("ratio").accuracy.mean().reindex(["A", "B", "C"])
            ax.plot(["60:180", "120:120", "180:60"], curve.to_numpy(), marker="o", label=f"{method}, coverage={coverage}")
        ax.set(xlabel="Source acid : jujube count", ylabel="Accuracy", title="F03: class weight and target-origin coverage", ylim=(0, 1.02))
        ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
        path = target / "f03_weights_coverage_ratio.png"
        fig.savefig(path, dpi=160); plt.close(fig); names.append(str(path.relative_to(output)))
    f04 = metrics[metrics.family.eq("F04")]
    if not f04.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if "source_budget" in f04:
            curve = f04.groupby("source_budget").accuracy.mean().sort_index()
            ax.plot(curve.index, curve.to_numpy(), marker="o")
            ax.set_xticks([50, 100, 150, 190, 240])
        else:
            curve = f04.groupby("method").accuracy.mean().sort_index()
            ax.plot(curve.index, curve.to_numpy(), marker="o")
        ax.set(xlabel="Training specimens (five balanced origins)", ylabel="Accuracy", title="F04: fixed nested learning curve", ylim=(0, 1.02))
        ax.grid(alpha=.2); fig.tight_layout()
        path = target / "f04_learning_curve.png"
        fig.savefig(path, dpi=160); plt.close(fig); names.append(str(path.relative_to(output)))
    return names


def report_text(summary, curves, transitions):
    lines = ["# B方案第二轮：校准退化、重训分解与来源覆盖稳定性", "",
             f"状态：**{summary['status']}**，已封存{summary['completed_jobs']}/{summary['expected_jobs']}个任务包。",
             "本报告只读取完整封存预测；不拟合模型、不修改原105任务的代码、数据、标签或结果。", "",
             f"新增完整分类器拟合{summary['new_model_fits']}次；已封存科研任务CPU {summary['cpu_seconds']:.6f}秒。重复使用的旧模型端点不计为新训练；控制、核验、失败尝试及推送成本另见后台账本。",
             f"逐样品预测行数{summary['prediction_rows']}，涉及唯一Query实物{summary['physical_query_samples']}；两者不得互换。", "",
             "## 状态及设计", "", "|实验块|内容|完成/计划|状态|", "|---|---|---:|---|"]
    labels = {"F01": "分数几何、支持重排及乱标诊断", "F02": "Scaler × gamma 重训分解", "F03": "类别权重 × 来源覆盖 × 类别比例", "F04": "五来源固定嵌套学习曲线"}
    for family, record in summary["family_task_counts"].items():
        lines.append(f"|{family}|{labels.get(family,family)}|{record['completed']}/{record['planned']}|{record['status']}|")
    lines.extend(["", "当前有效数据仍为18份工作簿、1672份用户声明独立实物。本轮仅在master TRAIN构造角色，不使用原CAL/PRED搜索模型。ZY01/ZY02为枣仁、ZY03—ZY07为酸枣仁；原来源/批次混杂不变。", "",
                  "## F01：支持抽样和分数几何", "",
                  "复用15个五来源以及预先指定30个类别平衡二分类模型。原始Support顺序作为draw0，再固定101—110十个重排种子；每类K嵌套取1/3/5/10。真标签K10完整池仅保留一次，重排不是新的实物或新真标签端点。乱标仅作用于显式诊断副本，保持样品ID与类别计数。", "",
                  "比较原始Anchor、Bias、弱Anchor、对角Anchor、0.1/10倍分数和逐样品行中心化＋Source全局RMS归一化。正比例与行中心化保留原始K0类别边界；归一化统计量只来自Source。完整Anchor包含跨类耦合，对角方法是降低自由度的明确对照。分数不是概率。", "",
                  "## 全部冻结方法的描述性结果", "", "下表按每任务、每个合法draw的端点等权汇总；不从这些结果选择最优配置作为确认性结论。", "",
                  "|块|原任务|方法|K/类|平均Accuracy|平均Macro-F1|平均Balanced Accuracy|端点数|",
                  "|---|---|---|---:|---:|---:|---:|---:|"])
    if not curves.empty:
        for row in curves.to_dict("records"):
            lines.append(f"|{row['family']}|{row['experiment']}|{row['method']}|{row['k']}|{row['accuracy']:.6f}|{row['macro_f1']:.6f}|{row['balanced_accuracy']:.6f}|{row['endpoint_count']}|")
    lines.extend(["", "## F02：重训收益的Scaler与gamma分解", "",
                  "保持Source、Support、Query和支持标签预算完全相同，分别冻结/重估StandardScaler、冻结原源模型数值gamma/按新增训练侧重新计算gamma。C固定10，类别权重balanced。原Scaler重估＋gamma重算的端点与上一轮完整重训完全相同，优先复用，不重复训练。", "",
                  "即使冻结Scaler，gamma='scale'也会随训练集合改变，因此不能把两者混为同一因素。逐任务保存有效gamma与Scaler哈希，四个条件的全部配对差值见paired_transitions.csv。对于重新逐列StandardScaler后的完整612点输入，scale gamma通常等于1/612；严格相同的数值与等价目标端点直接复用，不构成独立干预。审计中的ULP级差异也不能解释为生物学或稳健的核宽度机制。", "",
                  "## 配对效应及描述性不确定性", "", "以下先列每块全体Query对齐的主要对照；每来源、逐类Recall代价、全部方法差值与区间保存在CSV。", "",
                  "|块|原任务|比较|K/类|ΔAccuracy百分点及95%区间|纠错/误伤|重复预测/实物|",
                  "|---|---|---|---:|---:|---:|---:|"])
    if not transitions.empty:
        selected = transitions[transitions.scope.eq("ALL")]
        selected = selected[(selected.family.isin(["F01", "F02"]) & selected.comparison.str.endswith(("_minus_identity", "_minus_source_baseline"))) |
                            (selected.family.eq("F04") & selected.comparison.str.endswith("_minus_source190")) |
                            (selected.family.eq("F03") & selected.comparison.eq("unweighted_minus_balanced") & selected.get("seed_set", pd.Series(index=selected.index, dtype=str)).eq("new45_49"))]
        for row in selected.to_dict("records"):
            condition = row['comparison']
            if row["family"] == "F03":
                condition += f" ratio={row.get('ratio')} coverage={row.get('coverage')}"
            lines.append(f"|{row['family']}|{row['experiment']}|{condition}|{int(row.get('k',0))}|{100*row['delta_accuracy']:+.3f} [{100*row['ci_lower']:+.3f},{100*row['ci_upper']:+.3f}]|{row['wrong_to_correct']}/{row['correct_to_wrong']}|{row['n_predictions']}/{row['n_physical_samples']}|")
    lines.extend(["", "## F03：权重、来源覆盖与比例", "",
                  "五个目标酸枣仁来源×种子42—49×三种比例×两种覆盖，每个条件固定240个Source与20+20 Query，比对balanced与无类别权重。42—44中已有balanced源模型直接复用；45—49是预先指定的新划分重复，并非新样品。原生SVC预测用于因子效应，score argmax接口诊断单列。", "",
                  "分别报告旧种子42—44、新种子45—49以及全部42—49的对照，并保留每来源结果。覆盖样品始终等量替换其他酸枣仁来源；不能称跨批次或任意未知来源泛化。", "",
                  "## F04：标签预算与学习曲线", "",
                  "原15个五来源外折中固定Query，在原Source每来源38份的锁定顺序中取10/20/30份，再比较原190份以及Source190＋全部Support50＝240份。190与240端点分别复用上一轮源模型和K10完整重训。所有预算平衡且按每类嵌套抽样；固定同一SVM参数，不依据Query选训练量。", "",
                  "学习曲线各折是否单调和每步纠错/误伤全部保存；预算变化同时改变标签数量与训练统计，不据此主张独立因果机制。", "",
                  "## 局限与未运行项目", "", LIMITS, "",
                  "二分类与五来源的样品组成、类别数和标签预算不同，准确率不得直接相减解释为类别数的单因素作用。新增实验依据已曝光结果提出，本轮不是独立确认集。所有正、负、零效应均保留，不因成绩低而更换Query、种子、方法或指标。", "",
                  "真实新批次评价继续为**NOT_RUN_NEEDS_NEW_BATCH_DATA**。上一轮已交付元数据模板、导入验证器及五来源优先450份新样品的采集起点；目前未采集，不生成真实batch_id或补造评价。", "",
                  "## 复现", "", "```bash", "python scripts/run_b_mechanism_followup_20261110.py --resume",
                  "python scripts/summarize_b_mechanism_followup_20261110.py --resume",
                  "python scripts/summarize_b_mechanism_followup_20261110.py --verify", "```", "",
                  "summary_verification.json记录独立混淆计数复算、任务与工件哈希核验。训练中途的PARTIAL报告只反映已封存子集。", ""])
    return "\n".join(lines)


def summarize(output, *, verify=False, n_boot=1000, report_path=None):
    output = Path(output)
    jobs, predictions, metrics, diagnostics, manifest, checked = load_jobs(output)
    if verify:
        path = output / "metrics.csv"
        if not path.is_file():
            raise ValueError("No saved metrics.csv; summarize first")
        saved = pd.read_csv(path, keep_default_na=False)
        if len(saved) != len(metrics):
            raise ValueError("Root metric count differs from sealed endpoints")
        if not metrics.empty:
            joined = saved.merge(metrics, on=["job_id", "endpoint_id"], suffixes=("_saved", "_recomputed"), how="outer", validate="one_to_one", indicator=True)
            if not joined._merge.eq("both").all():
                raise ValueError("Root metric endpoint identities differ")
            for column in ("accuracy", "macro_f1", "balanced_accuracy", "correct", "n_predictions", "n_physical_samples"):
                if not np.allclose(pd.to_numeric(joined[column + "_saved"]), pd.to_numeric(joined[column + "_recomputed"]), atol=1e-12, rtol=0):
                    raise ValueError(f"Root independent metric mismatch: {column}")
                checked += len(joined)
            for column in ("class_recall_json", "confusion_matrix_json", "classes_json"):
                if not all(json.loads(a) == json.loads(b) for a, b in zip(joined[column + "_saved"], joined[column + "_recomputed"])):
                    raise ValueError(f"Root independent categorical metric mismatch: {column}")
                checked += len(joined)
        receipt = {"status": "DONE", "sealed_jobs": len(jobs), "metric_rows": len(metrics), "checks": checked,
                   "artifact_hashes_checked": True, "independent_confusion_count_recomputation": True,
                   "timestamp_utc": datetime.now(timezone.utc).isoformat()}
        write_json(output / "summary_verification.json", receipt)
        return receipt
    transitions, intervals, interactions = comparison_tables(predictions, n_boot=n_boot)
    mechanisms = mechanism_table(predictions, metrics, diagnostics)
    if not interactions.empty:
        mechanisms = pd.concat([mechanisms, interactions], ignore_index=True)
    design, _ = frozen_design(output)
    expected_counts = {family: sum(item.get("family") == family for item in design) for family in FAMILIES} if design else FAMILIES.copy()
    expected_total = sum(expected_counts.values())
    state = lambda completed, planned: "DONE" if completed == planned and planned else ("PARTIAL" if completed else "NOT_RUN")
    counts = {family: {"completed": sum(job.get("family") == family for job in jobs), "planned": planned}
              for family, planned in expected_counts.items()}
    for record in counts.values():
        record["status"] = state(record["completed"], record["planned"])
    curves = (metrics.groupby(["family", "experiment", "method", "k"], dropna=False)
              .agg(accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"), balanced_accuracy=("balanced_accuracy", "mean"), endpoint_count=("endpoint_id", "count"))
              .reset_index()) if not metrics.empty else pd.DataFrame()
    summary = {"status": state(len(jobs), expected_total), "completed_jobs": len(jobs), "expected_jobs": expected_total,
               "family_task_counts": counts, "new_model_fits": sum(int(job.get("new_model_fits", 0)) for job in jobs),
               "new_adapter_fits": sum(int(job.get("new_adapter_fits", 0)) for job in jobs),
               "cpu_seconds": sum(float(job.get("cpu_seconds", 0)) for job in jobs),
               "wall_seconds": sum(float(job.get("wall_seconds", 0)) for job in jobs),
               "prediction_rows": len(predictions), "metric_rows": len(metrics),
               "physical_query_samples": int(predictions.physical_sample_id.nunique()) if not predictions.empty else 0,
               "uncertainty": LIMITS, "e05_status": "NOT_RUN_NEEDS_NEW_BATCH_DATA",
               "previous_results_modified": False, "no_new_fits_by_summarizer": True,
               "timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "curves_macro_over_job_draws": curves.to_dict("records")}
    write_csv(output / "run_manifest.csv", manifest)
    write_csv(output / "metrics.csv", metrics, columns=["job_id", "endpoint_id", "family", "method", "k", "draw", "accuracy", "macro_f1", "balanced_accuracy"])
    write_csv(output / "paired_transitions.csv", transitions)
    write_csv(output / "uncertainty.csv", intervals)
    write_csv(output / "mechanism_analysis.csv", mechanisms)
    summary["figures"] = make_figures(output, metrics)
    write_json(output / "results_summary.json", summary)
    write_json(output / "status.json", {key: summary[key] for key in
               ("status", "completed_jobs", "expected_jobs", "family_task_counts", "new_model_fits", "cpu_seconds", "e05_status", "timestamp_utc")})
    atomic_text(report_path or REPORT, report_text(summary, curves, transitions))
    return clean(summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "--outdir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.bootstrap <= 0:
        parser.error("--bootstrap must be positive")
    if args.dry_run:
        jobs, predictions, metrics, _, _, checked = load_jobs(args.output)
        design, _ = frozen_design(args.output)
        print(json.dumps({"status": "NOT_RUN", "sealed_jobs": len(jobs), "prediction_rows": len(predictions),
                          "metric_rows": len(metrics), "expected_jobs": len(design) if design else sum(FAMILIES.values()),
                          "checks": checked, "new_fits": 0}, ensure_ascii=False))
        return 0
    print(json.dumps(summarize(args.output, verify=args.verify, n_boot=args.bootstrap, report_path=args.report), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, AssertionError, KeyError) as error:
        print(f"INTEGRITY_ERROR: {error}", file=sys.stderr)
        raise SystemExit(42)
