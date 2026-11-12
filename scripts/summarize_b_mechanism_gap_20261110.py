#!/usr/bin/env python3
"""Read sealed B-mechanism predictions; independently recompute and describe them.

This program never fits a model. Bootstrap resampling uses physical sample IDs,
keeping every repeated prediction belonging to a selected physical sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports/b_mechanism_20261110"
KS = (0, 1, 3, 5, 10)
CORRECTIONS = ("bias_only", "anchor_lambda1", "weak_lambda1000_blend005", "full_pipeline_retrain")
KEYS = ["job_id", "method", "k"]
PAIR_KEYS = ["seed", "fold", "target_origin", "physical_sample_id"]
LIMITS = (
    "所有比较为探索性内部开发；历史样品已曝光。区间按physical_sample_id聚类重采样，"
    "只描述当前样品重复预测的不确定性，不包含来源总体、批次总体、模型选择或长期开发偏倚。"
    "仅五个酸枣仁来源，来源与批次混杂；不能据此作真实跨批次因果或外部泛化结论。"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".summary-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, frame: pd.DataFrame, columns=None) -> None:
    if frame.empty and len(frame.columns) == 0:
        frame = pd.DataFrame(columns=columns or ["status", "reason"])
    atomic_text(path, frame.to_csv(index=False))


def finite(value):
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, np.generic):
        return finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def calculate_metrics(rows: pd.DataFrame, classes) -> dict:
    """Only true/predicted labels enter the independent recomputation."""
    y, p = rows.true_label.astype(str), rows.pred_label.astype(str)
    labels = [str(x) for x in classes]
    unknown = (set(y) | set(p)) - set(labels)
    if unknown:
        raise ValueError(f"Labels outside the frozen class list: {unknown}")
    recalls = recall_score(y, p, labels=labels, average=None, zero_division=0)
    return {
        "n_predictions": len(rows), "n_physical_samples": rows.physical_sample_id.nunique(),
        "correct": int((y == p).sum()), "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "class_recall_json": json.dumps(dict(zip(labels, recalls.tolist())), ensure_ascii=False, sort_keys=True),
        "confusion_matrix_json": json.dumps(confusion_matrix(y, p, labels=labels).tolist()),
        "classes_json": json.dumps(labels, ensure_ascii=False),
    }


def load_jobs(output: Path):
    """Accept only complete, hashed engine bundles; partial folders never count."""
    jobs, frames, mechanisms, manifest = [], [], [], []
    for result_path in sorted((output / "jobs").glob("*/result.json")):
        folder = result_path.parent
        if folder.name.startswith(".") or not (folder / "COMPLETE.json").is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "DONE" or result.get("job_id") != folder.name:
            raise ValueError(f"Task completion identity invalid: {folder}")
        marker = json.loads((folder / "COMPLETE.json").read_text(encoding="utf-8"))
        if marker["result_sha256"] != sha256(result_path):
            raise ValueError(f"Sealed result hash mismatch: {result_path}")
        if marker["input_hash"] != result["input_hash"] or marker["job_hash"] != result["job_hash"]:
            raise ValueError(f"Sealed identity mismatch: {result_path}")
        for name, expected in result.get("artifact_hashes", {}).items():
            artifact = folder / name
            if not artifact.is_file() or sha256(artifact) != expected:
                raise ValueError(f"Task artifact hash mismatch: {artifact}")
        rows = pd.read_csv(folder / "predictions.csv", dtype={"physical_sample_id": str, "true_label": str, "pred_label": str})
        for col in ("fold", "target_origin", "coverage", "ratio", "seed", "experiment", "job_id"):
            if col not in rows:
                rows[col] = result.get(col)
        rows["fold"] = rows["fold"].fillna(-1)
        rows["target_origin"] = rows["target_origin"].fillna("ALL")
        rows["coverage"] = pd.to_numeric(rows["coverage"], errors="coerce").fillna(-1).astype(int)
        rows["ratio"] = rows["ratio"].fillna("NA")
        rows["k"] = rows["k"].astype(int)
        if rows.duplicated(KEYS + ["physical_sample_id"]).any():
            raise ValueError(f"Duplicate query ID in method/K: {folder}")
        query_ids = set(map(str, result["query_ids"]))
        for _, group in rows.groupby(["method", "k"], sort=False):
            if set(group.physical_sample_id) != query_ids:
                raise ValueError(f"Prediction denominator differs from frozen Query: {folder}")
        frames.append(rows)
        mech_path = folder / "mechanisms.csv"
        if mech_path.is_file() and mech_path.stat().st_size > 1:
            try:
                mech = pd.read_csv(mech_path)
            except pd.errors.EmptyDataError:
                mech = pd.DataFrame()
            if not mech.empty:
                for col in ("job_id", "experiment", "target_origin", "seed", "coverage", "ratio", "fold"):
                    mech[col] = result.get(col)
                mechanisms.append(mech)
        jobs.append(result)
        manifest.append({k: result.get(k) for k in (
            "job_id", "experiment", "target_origin", "seed", "coverage", "ratio", "fold",
            "input_hash", "cpu_seconds", "wall_seconds", "source_cpu_seconds", "source_wall_seconds"
        )} | {"status": "DONE", "result_sha256": sha256(result_path), "result_path": str(result_path.relative_to(output))})
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return jobs, frame, pd.concat(mechanisms, ignore_index=True) if mechanisms else pd.DataFrame(), pd.DataFrame(manifest)


def metrics_table(jobs, predictions):
    table = []
    for result in jobs:
        subset = predictions[predictions.job_id == result["job_id"]]
        costs = {(row["method"], int(row["k"])): row for row in result.get("costs", [])}
        for (method, k), rows in subset.groupby(["method", "k"], sort=True):
            info = {col: result.get(col) for col in ("job_id", "experiment", "target_origin", "seed", "coverage", "ratio", "fold")}
            cost = costs.get((method, int(k)), {})
            used = int(rows.support_labels_used.iloc[0]) if "support_labels_used" in rows else (0 if method in ("identity", "svc_predict", "decision_sign") else int(rows.total_support_labels.iloc[0]))
            table.append(info | {"method": method, "k": int(k), "total_support_labels": int(rows.total_support_labels.iloc[0]),
                                 "support_labels_used": used, "cpu_seconds": float(cost.get("cpu_seconds", 0)),
                                 "wall_seconds": float(cost.get("wall_seconds", 0)),
                                 "source_cpu_seconds": result.get("source_cpu_seconds"), "source_wall_seconds": result.get("source_wall_seconds")}
                         | calculate_metrics(rows, result["classes"]))
    return pd.DataFrame(table)


def paired(left, right, keys):
    columns = keys + ["true_label", "pred_label"]
    a, b = left[columns], right[columns]
    if a.duplicated(keys).any() or b.duplicated(keys).any():
        raise ValueError(f"Comparison is not one-to-one on {keys}")
    merged = a.merge(b, on=keys, suffixes=("_before", "_after"), how="outer", indicator=True, validate="one_to_one")
    if not (merged._merge == "both").all():
        raise ValueError("Matched comparison has missing predictions; refusing to silently drop samples")
    if not (merged.true_label_before == merged.true_label_after).all():
        raise ValueError("True labels differ across a matched comparison")
    merged["before_correct"] = (merged.true_label_before == merged.pred_label_before).astype(int)
    merged["after_correct"] = (merged.true_label_after == merged.pred_label_after).astype(int)
    merged["delta"] = merged.after_correct - merged.before_correct
    return merged.drop(columns="_merge")


def cluster_interval(rows: pd.DataFrame, n_boot=1000, seed=20261110):
    """A paired row-weighted difference; entire physical-ID clusters resampled."""
    groups = rows.groupby("physical_sample_id", sort=True).delta.agg(["sum", "count"])
    if groups.empty:
        return None, None
    values, weights = groups["sum"].to_numpy(), groups["count"].to_numpy()
    rng = np.random.default_rng(seed)
    # Bounded batches avoid allocating a large bootstrapping matrix.
    estimates = []
    for start in range(0, n_boot, 100):
        draws = rng.integers(0, len(groups), size=(min(100, n_boot - start), len(groups)))
        estimates.extend((values[draws].sum(axis=1) / weights[draws].sum(axis=1)).tolist())
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def transition_record(rows, info, n_boot):
    before, after = rows.before_correct.to_numpy(), rows.after_correct.to_numpy()
    recalls = {}
    for label, group in rows.groupby("true_label_before"):
        recalls[str(label)] = float(group.after_correct.mean() - group.before_correct.mean())
    rec = info | {
        "n_predictions": len(rows), "n_physical_samples": rows.physical_sample_id.nunique(),
        "before_accuracy": float(before.mean()), "after_accuracy": float(after.mean()),
        "delta_accuracy": float((after - before).mean()),
        "wrong_to_correct": int(((before == 0) & (after == 1)).sum()),
        "correct_to_wrong": int(((before == 1) & (after == 0)).sum()),
        "recall_deltas_json": json.dumps(recalls, ensure_ascii=False, sort_keys=True),
        "recall_tradeoff": bool(any(x > 0 for x in recalls.values()) and any(x < 0 for x in recalls.values())),
        "sampling_structure": "matched predictions; repeated seeds/folds retained within physical_sample_id",
    }
    low, high = cluster_interval(rows, n_boot=n_boot)
    uncertainty = info | {
        "effect": "paired_delta_accuracy", "estimate": rec["delta_accuracy"],
        "ci_lower": low, "ci_upper": high, "confidence_level": .95,
        "n_bootstrap": n_boot, "n_predictions": len(rows),
        "n_physical_samples": rows.physical_sample_id.nunique(),
        "resampling_unit": "physical_sample_id", "interval_scope": LIMITS,
    }
    return rec, uncertainty


def contrasts(predictions: pd.DataFrame, n_boot=1000):
    transitions, intervals, interaction = [], [], []
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def add(left, right, info, keys, scope=True):
        if left.empty or right.empty:
            return
        matched = paired(left, right, keys)
        rec, interval = transition_record(matched, info, n_boot)
        transitions.append(rec)
        intervals.append(interval)
        if scope and "target_origin" in matched and matched.target_origin.nunique() > 1:
            for origin, group in matched.groupby("target_origin", sort=True):
                rec, interval = transition_record(group, info | {"scope": str(origin)}, n_boot)
                transitions.append(rec)
                intervals.append(interval)
        return matched

    base = predictions[(predictions.method == "svc_predict") & (predictions.k == 0) & (predictions.experiment == "E01")]
    # While a queue is running, only complete six-condition factorial blocks
    # contribute to E01 contrasts; no partially available cell is silently used.
    if not base.empty:
        cells = base[["seed", "target_origin", "coverage", "ratio"]].drop_duplicates()
        complete = cells.groupby(["seed", "target_origin"]).size()
        complete = complete[complete == 6].index
        base = base.set_index(["seed", "target_origin"]).loc[lambda f: f.index.isin(complete)].reset_index()
    for ratio in sorted(base.ratio.unique()):
        sub = base[base.ratio == ratio]
        add(sub[sub.coverage == 0], sub[sub.coverage == 1],
            {"experiment": "E01", "comparison": "coverage_1_minus_0", "ratio": ratio, "method": "svc_predict", "k": 0, "scope": "ALL"}, PAIR_KEYS)
    for coverage in sorted(base.coverage.unique()):
        sub = base[base.coverage == coverage]
        for earlier, later in (("A", "B"), ("A", "C"), ("B", "C")):
            add(sub[sub.ratio == earlier], sub[sub.ratio == later],
                {"experiment": "E01", "comparison": f"ratio_{later}_minus_{earlier}", "coverage": int(coverage), "method": "svc_predict", "k": 0, "scope": "ALL"}, PAIR_KEYS)
    # Difference of differences from matched per-sample correctness, without extra fits.
    for earlier, later in (("A", "B"), ("A", "C"), ("B", "C")):
        if set((earlier, later)).issubset(set(base.ratio)) and set(base.coverage) == {0, 1}:
            first = paired(base[(base.ratio == earlier) & (base.coverage == 0)], base[(base.ratio == earlier) & (base.coverage == 1)], PAIR_KEYS)
            second = paired(base[(base.ratio == later) & (base.coverage == 0)], base[(base.ratio == later) & (base.coverage == 1)], PAIR_KEYS)
            joined = first[PAIR_KEYS + ["delta"]].merge(second[PAIR_KEYS + ["delta"]], on=PAIR_KEYS, suffixes=("_first", "_second"), validate="one_to_one")
            joined["delta"] = joined.delta_second - joined.delta_first
            for scope, group in [("ALL", joined)] + [(str(o), g) for o, g in joined.groupby("target_origin")]:
                low, high = cluster_interval(group, n_boot)
                info = {"experiment": "E01", "comparison": f"coverage_x_ratio_{later}_minus_{earlier}", "scope": scope,
                        "effect": "difference_of_differences_accuracy", "estimate": float(group.delta.mean()), "ci_lower": low, "ci_upper": high,
                        "n_predictions": len(group), "n_physical_samples": group.physical_sample_id.nunique(), "resampling_unit": "physical_sample_id",
                        "n_bootstrap": n_boot, "confidence_level": .95, "interval_scope": LIMITS}
                intervals.append(info)
                interaction.append(info)

    for experiment, sub in predictions.groupby("experiment"):
        keys = ["job_id", "physical_sample_id", "target_origin"]
        for method in CORRECTIONS:
            for k in KS:
                right = sub[(sub.method == method) & (sub.k == k)]
                baseline_method = "svc_predict" if method == "full_pipeline_retrain" else "identity"
                left = sub[(sub.method == baseline_method) & (sub.k == 0)]
                if right.empty:
                    continue
                common_jobs = set(right.job_id) & set(left.job_id)
                add(left[left.job_id.isin(common_jobs)], right[right.job_id.isin(common_jobs)],
                    {"experiment": experiment, "comparison": f"{method}_minus_{baseline_method}", "method": method, "k": k, "scope": "ALL"}, keys)
                # Full refit increment over each frozen correction, same label budget.
                if method == "full_pipeline_retrain" and k > 0:
                    for frozen in CORRECTIONS[:-1]:
                        left2 = sub[(sub.method == frozen) & (sub.k == k) & (sub.job_id.isin(common_jobs))]
                        add(left2, right[right.job_id.isin(common_jobs)],
                            {"experiment": experiment, "comparison": f"full_pipeline_retrain_minus_{frozen}", "method": method, "k": k, "scope": "ALL"}, keys)
        # Interface-only zero-label changes are explicitly reported.
        native = sub[(sub.method == "svc_predict") & (sub.k == 0)]
        identity = sub[(sub.method == "identity") & (sub.k == 0)]
        add(native, identity, {"experiment": experiment, "comparison": "score_argmax_minus_native_svc_zero_support", "method": "identity", "k": 0, "scope": "ALL"}, keys)
    return pd.DataFrame(transitions), pd.DataFrame(intervals), pd.DataFrame(interaction)


def mechanism_table(predictions, affine, metrics, interaction):
    rows = []
    if not affine.empty:
        for rec in affine.to_dict("records"):
            rec["analysis"] = "binary_affine_and_threshold_check"
            rec["interpretation"] = "a<=0须单独解释，不能归为保持方向的阈值移动" if float(rec.get("a", 1)) <= 0 else "a>0时应核验排序、AUROC和阈值等价"
            rows.append(rec)
    if not predictions.empty:
        for (experiment, method, k, origin), group in predictions.groupby(["experiment", "method", "k", "origin_group_id"]):
            rows.append({"analysis": "per_query_origin_performance", "experiment": experiment, "method": method,
                         "k": int(k), "query_origin_group_id": origin, "n_predictions": len(group),
                         "n_physical_samples": group.physical_sample_id.nunique(),
                         "accuracy": float((group.true_label == group.pred_label).mean()),
                         "interpretation": "按查询来源分别报告；重复预测行数与独立实物ID数分开"})
        for (experiment, method, k), group in predictions.groupby(["experiment", "method", "k"]):
            if "margin" not in group or experiment != "E01":
                continue
            margin = pd.to_numeric(group.margin, errors="coerce")
            # Fixed bins avoid choosing cutpoints from query outcomes.
            bins = pd.cut(margin.abs(), [0., .25, .5, 1., 2., np.inf], include_lowest=True)
            for margin_bin, rows_in_bin in group.groupby(bins, observed=True):
                if rows_in_bin.empty:
                    continue
                rows.append({"analysis": "absolute_margin_error_relationship", "experiment": experiment, "method": method, "k": int(k),
                             "margin_bin": str(margin_bin), "n_predictions": len(rows_in_bin),
                             "n_physical_samples": rows_in_bin.physical_sample_id.nunique(),
                             "error_probability": float((rows_in_bin.true_label != rows_in_bin.pred_label).mean()),
                             "interpretation": "分数绝对值与错误率的描述性关系；不是已校准概率，也不是因果机制证明"})
    if not metrics.empty:
        for (job, method), group in metrics.groupby(["job_id", "method"]):
            group = group.sort_values("k")
            differences = np.diff(group.accuracy.to_numpy())
            rows.append({"analysis": "k_shot_monotonicity_per_job", "job_id": job, "experiment": group.experiment.iloc[0], "method": method,
                         "k_values_json": json.dumps(group.k.tolist()), "accuracies_json": json.dumps(group.accuracy.tolist()),
                         "monotone_nondecreasing": bool(np.all(differences >= -1e-12)), "decrease_steps": int((differences < -1e-12).sum()),
                         "total_support_labels_json": json.dumps(group.total_support_labels.tolist())})
        for (experiment, method), group in metrics.groupby(["experiment", "method"]):
            curve = group.groupby("k").accuracy.mean().sort_index()
            rows.append({"analysis": "k_shot_monotonicity_macro_over_jobs", "experiment": experiment, "method": method,
                         "k_values_json": json.dumps(curve.index.tolist()), "accuracies_json": json.dumps(curve.tolist()),
                         "monotone_nondecreasing": bool(np.all(np.diff(curve.to_numpy()) >= -1e-12)),
                         "interpretation": "按任务等权均值，不能视为新的独立样品分母"})
    if not interaction.empty:
        rows.extend((interaction.assign(analysis="coverage_ratio_interaction")).to_dict("records"))
    return pd.DataFrame(rows)


def expected_count(output):
    protocol_path = output / "frozen_protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        for key in ("jobs", "design", "tasks"):
            if isinstance(protocol.get(key), list):
                return len(protocol[key])
        for key in ("expected_jobs", "n_jobs", "task_count"):
            if isinstance(protocol.get(key), int):
                return protocol[key]
    return 105


def make_figures(output, predictions):
    if predictions.empty:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target = output / "figures"
    target.mkdir(exist_ok=True)
    created = []
    # English labels use installed fonts reliably; Chinese interpretation is in the report.
    for experiment, group in predictions.groupby("experiment"):
        group = group[group.method.isin(("identity",) + CORRECTIONS)]
        group = group.assign(correct=(group.true_label == group.pred_label).astype(float))
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, line in group.groupby("method", sort=True):
            curve = line.groupby("k").correct.mean().sort_index()
            ax.plot(curve.index, curve.to_numpy(), marker="o", label=method)
        ax.set(xlabel="Support labels per class (K)", ylabel="Accuracy (repeated predictions)", title=f"{experiment}: fixed K-shot methods")
        ax.set_xticks(KS)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.tight_layout()
        path = target / f"{experiment.lower()}_k_shot_curves.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        created.append(str(path.relative_to(output)))
    base = predictions[(predictions.experiment == "E01") & (predictions.method == "svc_predict") & (predictions.k == 0)].copy()
    if not base.empty:
        base["correct"] = (base.true_label == base.pred_label).astype(float)
        fig, ax = plt.subplots(figsize=(7, 5))
        for coverage, line in base.groupby("coverage"):
            curve = line.groupby("ratio").correct.mean().reindex(["A", "B", "C"])
            ax.plot(["60:180", "120:120", "180:60"], curve, marker="o", label=f"Target origin coverage={coverage}")
        ax.set(xlabel="Source acid : jujube count (total=240)", ylabel="Accuracy", title="E01: coverage by class ratio")
        ax.set_ylim(0, 1.02)
        ax.legend()
        ax.grid(alpha=.2)
        fig.tight_layout()
        path = target / "e01_coverage_ratio.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        created.append(str(path.relative_to(output)))
    return created


def report_text(summary, metrics, transitions, intervals):
    state = summary["status"]
    lines = ["# B方案：少样本校准、来源覆盖与决策边界补充研究", "", f"状态：**{state}**。本报告仅由当前已完整封存的逐样品预测生成，不新增模型拟合。", "",
             f"已完成{summary['completed_jobs']}/{summary['expected_jobs']}个任务包；其中E01/E02={summary['e01_e02_jobs']}，E03={summary['e03_jobs']}。每个E01包共用一套冻结Source模型及Support/Query，包含全部K与方法；任务数不等于独立样品数。",
             f"实际已封存模型拟合{summary['scientific_model_fits']}次，科学计算CPU={summary['cpu_seconds']:.6f}秒，墙钟成本之和={summary['wall_seconds']:.6f}秒。失败/未封存尝试若存在，另见后台监督器账本；不将其隐入已封存CPU。", "",
             "## 数据、设计与分母", "", "当前有效清单18份工作簿、1672份用户声明独立实物；本研究仅在master TRAIN内构造角色。ZY01/ZY02为枣仁，ZY03—ZY07为酸枣仁。固定CAL/PRED不用于本轮模型搜索。", "",
             "E01固定Source=240，酸枣仁:枣仁=A 60:180、B 120:120、C 180:60；目标酸来源覆盖0或20份替换。每个目标来源/种子保持相同20+20 Query及每类10份Support池。E03纯酸五来源300份TRAIN，三重复五外折，每折190 Source、50 Support候选、60 Query。K=0/1/3/5/10每类，二分类总标签2K，五分类总标签5K。", "",
             "固定alpha、逐样品SNV、训练侧StandardScaler及C=10/gamma=scale/class_weight=balanced RBF-SVM。完整Pipeline重训重新拟合训练侧预处理及分类器；冻结分数校准另列。", "",
             "## E01：来源覆盖与类别比例", ""]
    selected = transitions[(transitions.get("experiment", pd.Series(dtype=str)) == "E01") & (transitions.get("scope", pd.Series(dtype=str)) == "ALL")] if not transitions.empty else pd.DataFrame()
    if not selected.empty:
        selected = selected[selected.comparison.str.startswith(("coverage_", "ratio_"))]
        lines.extend(["|比较|条件|准确率变化及描述性95%区间（百分点）|纠错/误伤|重复预测/实物ID|", "|---|---|---:|---:|---:|"])
        for row in selected.to_dict("records"):
            candidates = intervals[(intervals.comparison == row["comparison"]) & (intervals.scope == "ALL")]
            for key in ("ratio", "coverage"):
                if key in candidates and pd.notna(row.get(key)):
                    candidates = candidates[candidates[key] == row[key]]
            ci = ""
            if len(candidates) == 1:
                ci_row = candidates.iloc[0]
                ci = f" [{100*ci_row.ci_lower:+.3f},{100*ci_row.ci_upper:+.3f}]"
            lines.append(f"|{row['comparison']}|ratio={row.get('ratio','')} coverage={row.get('coverage','')}|{100*row['delta_accuracy']:+.3f}{ci}|{row['wrong_to_correct']}/{row['correct_to_wrong']}|{row['n_predictions']}/{row['n_physical_samples']}|")
    else:
        lines.append("尚无完整配对条件；不以不完整对照推断效应。")
    lines.extend(["", "各目标酸枣仁来源单独结果见paired_transitions.csv（scope=ZY03—ZY07），覆盖×比例差中之差见mechanism_analysis.csv和uncertainty.csv。覆盖条件下的Source/Query含同来源，不称未知来源或跨批次泛化。", "", "## E02/E03：K-shot校准与完整重训", ""])
    if not metrics.empty:
        lines.extend(["下表按每个已完成任务等权汇总全部K；全部负结果同样保留。每行的准确率均为描述性内部结果。", "", "|任务|方法|K每类|总Support标签|平均Accuracy|平均Macro-F1|平均Balanced Accuracy|", "|---|---|---:|---:|---:|---:|---:|"])
        useful = metrics[metrics.method.isin(("identity",) + CORRECTIONS)]
        for (exp, method, k), group in useful.groupby(["experiment", "method", "k"], sort=True):
            lines.append(f"|{exp}|{method}|{k}|{int(group.total_support_labels.iloc[0])}|{group.accuracy.mean():.6f}|{group.macro_f1.mean():.6f}|{group.balanced_accuracy.mean():.6f}|")
    lines.extend(["", "表中总Support为该K条件分配的候选标签预算；Identity/SVC诊断实际不消耗支持标签。逐方法实际标签用量及CPU/墙钟成本另见metrics.csv的support_labels_used/cpu_seconds/wall_seconds，源模型拟合成本单独列出，不重复累加。", "",
                  "二分类Identity使用二列score argmax；SVC.predict与decision_function符号作为接口诊断单列。完整重训的零支持基准匹配原生SVC；分数校准匹配score Identity。接口更换造成的差异不归为校准收益。", ""])
    if not transitions.empty:
        calibrations = transitions[(transitions.scope == "ALL") & transitions.comparison.isin([f"{m}_minus_{'svc_predict' if m == 'full_pipeline_retrain' else 'identity'}" for m in CORRECTIONS])]
        lines.extend(["配对校准变化（全部K、正负结果均列；区间仅描述当前实物样品）：", "", "|任务|方法|K每类|ΔAccuracy及95%区间（百分点）|纠错/误伤|逐类Recall有相反方向变化|", "|---|---|---:|---:|---:|---|"])
        for row in calibrations.to_dict("records"):
            ci_rows = intervals[(intervals.scope == "ALL") & (intervals.comparison == row["comparison"]) & (intervals.experiment == row["experiment"]) & (intervals.k == row["k"])]
            ci = ""
            if len(ci_rows) == 1:
                ci = f" [{100*ci_rows.iloc[0].ci_lower:+.3f},{100*ci_rows.iloc[0].ci_upper:+.3f}]"
            lines.append(f"|{row['experiment']}|{row['method']}|{row['k']}|{100*row['delta_accuracy']:+.3f}{ci}|{row['wrong_to_correct']}/{row['correct_to_wrong']}|{'是' if row['recall_tradeoff'] else '否'}|")
    lines.extend(["",
                  "## E04：机制和不确定性", "", "逐任务二分类仿射系数a、b、排序、AUROC、阈值等价核验见mechanism_analysis.csv；a>0且核验通过才支持保持排序的阈值移动解释，a<=0单独报告。margin分箱固定为0/.25/.5/1/2/∞，只描述错误率关系，不将margin写为校准概率。", "",
                  f"当前已保存二分类仿射检查{summary.get('binary_affine_checks', 0)}项，其中a<=0为{summary.get('affine_nonpositive_checks', 0)}项；a>0但阈值等价核验未通过为{summary.get('positive_a_threshold_failures', 0)}项。保持正方向并不证明新的生物学机制，阈值等价核验需与排序/AUROC及纠错、误伤一起解释。", "",
                  "K曲线的非单调、完整重训增量、逐类别Recall变化与纠错/误伤均保留。paired_transitions.csv记录配对分母和Recall权衡；uncertainty.csv提供实物ID聚类的描述性95%区间。跨种子同一个ID的所有相关预测一并重采样，未将重复行当独立观测。", "", LIMITS, "",
                  "二分类和五来源任务的类别数、来源组成、训练分母及总Support标签预算不同，不能直接相减准确率并归因于类别数。未依据Query表现改样品、K、种子、指标或超参数，也不从多重比较挑单项正结果作确认性结论。", "",
                  "## E05：新批次采集建议", "", "**NOT_RUN_NEEDS_NEW_BATCH_DATA**。现有实验不能评价真实新批次。优先补齐ZY03—ZY07五个酸枣仁来源的独立新批次，同时纳入枣仁对照来源；每个来源至少3个独立批次、每批至少30份经标签核验的独立实物（10 Support+20 Query），五酸来源基础规划450份。此数用于来源/批次重复与可执行Support预算，是设计起点而非效能保证；二分类可另补相应枣仁新批次至少90份。最终数量须结合先验最小重要效应、类内/批次内相关及预算在见Query前做效能规划。", "",
                  "真实批次ID、采样/测量日期、仪器、厚度、含水率、制样及标签依据由采集方填写；缺失保持缺失。先验证元数据、谱值、样品身份与Support/Query互斥，再用旧批次模型和预先指定的校准方法评价。详见new_batch_protocol.md及元数据验证接口。", "",
                  "## 状态及复现", "", f"E00见preflight.json；E01/E02、E03完成数以上述封存包为准，E04随其完整性{'DONE' if state == 'DONE' else 'PARTIAL'}；E05为NOT_RUN_NEEDS_NEW_BATCH_DATA。缺失与失败任务见status.json及后台日志。", "",
                  "```bash", "python scripts/run_b_mechanism_gap_20261110.py --resume", "python scripts/run_b_mechanism_gap_20261110.py --verify", "python scripts/summarize_b_mechanism_gap_20261110.py --verify", "```", "",
                  "metrics.csv由逐样品真值/预测独立复算；job工件哈希、结果清单和汇总验证回执用于复核。训练中途报告只反映已封存子集，不能当完整矩阵结论。", ""])
    return "\n".join(lines)


def summarize(output: Path, *, verify=False, n_boot=1000, report_path=None):
    jobs, predictions, affine, manifest = load_jobs(output)
    metrics = metrics_table(jobs, predictions) if jobs else pd.DataFrame()
    checked = 0
    if verify:
        saved_path = output / "metrics.csv"
        if not saved_path.is_file():
            raise ValueError("No saved metrics.csv to verify; run summarizer first")
        saved = pd.read_csv(saved_path)
        if len(saved) != len(metrics):
            raise ValueError("Saved metrics row count does not match sealed predictions")
        joined = saved.merge(metrics, on=KEYS, suffixes=("_saved", "_recomputed"), how="outer", validate="one_to_one", indicator=True)
        if not (joined._merge == "both").all():
            raise ValueError("Saved metrics method/K identities differ")
        for col in ("accuracy", "macro_f1", "balanced_accuracy", "correct", "n_predictions", "n_physical_samples"):
            if not np.allclose(joined[f"{col}_saved"], joined[f"{col}_recomputed"], atol=1e-12, rtol=0):
                raise ValueError(f"Independent metric mismatch: {col}")
            checked += len(joined)
        for col in ("class_recall_json", "confusion_matrix_json", "classes_json"):
            if not all(json.loads(a) == json.loads(b) for a, b in zip(joined[f"{col}_saved"], joined[f"{col}_recomputed"])):
                raise ValueError(f"Independent categorical metric mismatch: {col}")
            checked += len(joined)
        verification = {"status": "DONE", "sealed_jobs": len(jobs), "metric_rows": len(metrics), "checks": checked,
                        "independent_recomputation": True, "artifact_hashes_checked": True, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
        write_json(output / "summary_verification.json", verification)
        return verification

    transitions, intervals, interaction = contrasts(predictions, n_boot=n_boot)
    mechanisms = mechanism_table(predictions, affine, metrics, interaction)
    expected = expected_count(output)
    summary = {
        "status": "DONE" if len(jobs) == expected else "PARTIAL", "completed_jobs": len(jobs), "expected_jobs": expected,
        "e01_e02_jobs": sum(job["experiment"] == "E01" for job in jobs), "e03_jobs": sum(job["experiment"] == "E03" for job in jobs),
        "cpu_seconds": sum(float(job.get("cpu_seconds", 0)) for job in jobs),
        "wall_seconds": sum(float(job.get("wall_seconds", 0)) for job in jobs),
        "scientific_model_fits": sum(int(job.get("scientific_model_fits", 0)) for job in jobs),
        "prediction_rows": len(predictions), "physical_query_samples": int(predictions.physical_sample_id.nunique()) if jobs else 0,
        "e05_status": "NOT_RUN_NEEDS_NEW_BATCH_DATA", "uncertainty": LIMITS,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "no_new_fits": True,
    }
    if not affine.empty:
        slopes = pd.to_numeric(affine.a, errors="raise")
        threshold_ok = affine.threshold_equivalence.astype(str).str.lower().eq("true")
        summary.update({"binary_affine_checks": len(affine), "affine_nonpositive_checks": int((slopes <= 0).sum()),
                        "positive_a_threshold_failures": int(((slopes > 0) & ~threshold_ok).sum())})
    if not metrics.empty:
        grouped = metrics.groupby(["experiment", "method", "k"])[["accuracy", "macro_f1", "balanced_accuracy"]].mean().reset_index()
        summary["curves_macro_over_jobs"] = grouped.to_dict("records")
    write_csv(output / "run_manifest.csv", manifest)
    write_csv(output / "metrics.csv", metrics, columns=KEYS + ["accuracy", "macro_f1", "balanced_accuracy"])
    write_csv(output / "paired_transitions.csv", transitions)
    write_csv(output / "mechanism_analysis.csv", mechanisms)
    write_csv(output / "uncertainty.csv", intervals)
    summary["figures"] = make_figures(output, predictions)
    write_json(output / "results_summary.json", finite(summary))
    # Preserve supervisor-specific fields, but never mark incomplete jobs DONE.
    status_path = output / "status.json"
    existing = json.loads(status_path.read_text()) if status_path.exists() else {}
    existing.update({k: summary[k] for k in ("status", "completed_jobs", "expected_jobs", "cpu_seconds", "timestamp_utc")})
    task_status = lambda done, planned: "DONE" if done == planned else ("PARTIAL" if done else "NOT_RUN")
    existing["experiments"] = {"E00": "DONE" if (output / "preflight.json").exists() else "NOT_RUN", "E01_E02": task_status(summary["e01_e02_jobs"], 90), "E03": task_status(summary["e03_jobs"], 15), "E04": summary["status"] if jobs else "NOT_RUN", "E05": "NOT_RUN_NEEDS_NEW_BATCH_DATA"}
    existing["experiment_task_counts"] = {"E01_E02": {"completed": summary["e01_e02_jobs"], "planned": 90}, "E03": {"completed": summary["e03_jobs"], "planned": 15}}
    write_json(status_path, finite(existing))
    atomic_text(report_path or ROOT / "docs/B_MECHANISM_GAP_RESULTS_20261110.md", report_text(summary, metrics, transitions, intervals))
    return finite(summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "--outdir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Idempotently rebuild summaries from sealed jobs; never refit")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    if args.dry_run:
        jobs, predictions, _, _ = load_jobs(args.output)
        print(json.dumps({"status": "NOT_RUN", "sealed_jobs": len(jobs), "prediction_rows": len(predictions), "expected_jobs": expected_count(args.output), "fits": 0}, ensure_ascii=False))
        return
    print(json.dumps(summarize(args.output, verify=args.verify, n_boot=args.bootstrap, report_path=args.report), ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, AssertionError) as error:
        # Immutable input/artifact/metric violations require investigation;
        # exit 42 tells the supervisor not to retry the same invalid evidence.
        print(f"INTEGRITY_ERROR: {error}", file=sys.stderr)
        raise SystemExit(42)
