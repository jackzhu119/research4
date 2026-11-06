#!/usr/bin/env python3
"""Conditional uncertainty and feature-set diagnostics of existing predictions.

No model fitting, tuning, label editing, inference, or quantum calculations.
All source artifacts are read-only; a fresh output directory is required.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import resource
import signal
import time

ROOT = Path(__file__).resolve().parents[1]
STAGED = Path("reports/staged-nested-optimization/2026-10-31")
METHODS = ["R0", "R1", "R2", "R3", "R4"]
METRICS = ["accuracy", "macro_f1", "balanced_accuracy"]
PAIRS = [("R1", "R0"), ("R2", "R0"), ("R3", "R0"),
         ("R4", "R0"), ("R2", "R1"), ("R3", "R2"), ("R4", "R2")]
WARNINGS = [
    "All 855 specimens are historically exposed development data, not a new confirmation cohort.",
    "Intervals condition on the saved predictions and observed class counts. They exclude model-training, tuning, seed, and feature-selection uncertainty.",
    "OOF models share overlapping training specimens. Specimen bootstrap and nominal McNemar independence are approximations, not a validated nested-CV significance test.",
    "ZY02 has only five independent specimens. Stratification preserves five draws, not five newly independent observations; minority uncertainty remains poorly resolved.",
    "Fold feature sets use overlapping source samples and different selected feature counts. Their overlap is descriptive, not a test of independent replication.",
    "The random-feature null is uniform independent subsets with matched cardinalities; it ignores adjacent-frequency correlation and is not a chemical or performance null.",
    "No held-out score is used to select a new model, feature, hyperparameter, or seed in this analysis.",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(rel, sources):
    path = ROOT / rel
    if path.stat().st_size > 2_000_000:
        raise ValueError(f"Unexpected large JSON: {rel}")
    sources[str(rel)] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return json.loads(path.read_text())


def read_predictions(rel, sources):
    path = ROOT / rel
    sources[str(rel)] = {"sha256": sha(path), "bytes": path.stat().st_size}
    with gzip.open(path, "rt", newline="") as handle:
        return [r for r in csv.DictReader(handle) if r["method"] in METHODS]


def metrics_from_confusion(np, confusion):
    diag = np.diagonal(confusion, axis1=-2, axis2=-1)
    true_n = confusion.sum(axis=-1)
    predicted_n = confusion.sum(axis=-2)
    total = true_n.sum(axis=-1)
    recall = np.divide(diag, true_n, out=np.zeros_like(diag, dtype=float), where=true_n != 0)
    f1 = np.divide(2 * diag, true_n + predicted_n,
                   out=np.zeros_like(diag, dtype=float), where=(true_n + predicted_n) != 0)
    return np.stack([diag.sum(axis=-1) / total, f1.mean(axis=-1), recall.mean(axis=-1)], axis=-1)


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n))


def exact_uniform_jaccard(p, ka, kb):
    denominator = math.comb(p, kb)
    low, high = max(0, kb - (p - ka)), min(ka, kb)
    return sum((t / (ka + kb - t)) *
               (math.comb(ka, t) * math.comb(p - ka, kb - t) / denominator)
               for t in range(low, high + 1))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Empty table: {path}")
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def feature_sets(sources, rows_by_method, axis):
    result = {}
    for rel in sorted((ROOT / "models/staged_nested_20261031/core").glob("outer_*/*.json")):
        # Small metadata only; never load the joblib model.
        raw = json.loads(rel.read_text())
        methods = [r["method"] for r in raw.get("roles", []) if r["method"] in METHODS]
        if not methods:
            continue
        raw = read_json(rel.relative_to(ROOT), sources)
        for method in methods:
            key = method, int(raw["outer_fold"])
            if key in result:
                raise ValueError(f"Duplicate method/fold metadata: {key}")
            result[key] = (raw["model_metadata"], raw["source_ids"],
                           raw["outer_query_ids"], str(rel.relative_to(ROOT)), method)
    for fold in range(5):
        subset = [x for x in rows_by_method["R4"] if int(x["outer_fold"]) == fold]
        families = {x["family"] for x in subset}
        if len(families) != 1:
            raise ValueError("R4 family is not fixed within outer fold")
        family = next(iter(families))
        rel = Path(f"models/staged_nested_20261031/combinations/evaluation/outer_{fold:02d}/source_checkpoint.json")
        raw = read_json(rel, sources)
        metadata = raw["artifacts"][family]["metadata"]
        result["R4", fold] = (metadata, metadata["source_ids"],
                               [x["record_id"] for x in subset], str(rel), family)
    records, point_rows = {}, []
    for (method, fold), (meta, source, query, rel, family) in sorted(result.items()):
        selected = meta.get("selected_indices")
        if not selected or len(selected) != len(set(selected)):
            raise ValueError(f"Missing/nonunique features: {method}/{fold}")
        expected_query = {x["record_id"] for x in rows_by_method[method] if int(x["outer_fold"]) == fold}
        if set(query) != expected_query or len(source) != 684 or set(source) & set(query):
            raise ValueError("Feature source/query lineage mismatch")
        if any(i < 0 or i >= len(axis) for i in selected):
            raise ValueError("Feature index outside original axis")
        frequencies = meta.get("selected_frequencies")
        if frequencies is None or len(frequencies) != len(selected):
            raise ValueError("Selected-frequency provenance absent")
        if any(abs(axis[i] - f) > 1e-12 for i, f in zip(selected, frequencies)):
            raise ValueError("Axis/feature frequencies mismatch")
        records[method, fold] = {"indices": set(selected), "source_ids": set(source), "family": family}
        for i in selected:
            point_rows.append({"method": method, "outer_fold": fold, "family": family,
                               "feature_index_zero_based": i, "frequency_THz": axis[i],
                               "source_metadata_path": rel})
    if set(records) != set(itertools.product(METHODS, range(5))):
        raise ValueError("Not exactly 25 saved method/fold feature sets")
    return records, point_rows


def run(output, iterations, seed):
    started = datetime.now(timezone.utc).isoformat()
    cpu_start, wall_start = time.process_time(), time.monotonic()
    if output.exists():
        raise ValueError("Fresh output required; refusing to overwrite")
    import numpy as np
    sources = {}
    rows = read_predictions(STAGED / "core/outer_predictions.csv.gz", sources)
    rows += read_predictions(STAGED / "combinations/outer_predictions.csv.gz", sources)
    by_method = {m: sorted((r for r in rows if r["method"] == m), key=lambda r: r["record_id"]) for m in METHODS}
    ids = [r["record_id"] for r in by_method["R0"]]
    classes = sorted({r["y_true"] for r in by_method["R0"]})
    if len(ids) != 855 or len(set(ids)) != 855 or len(classes) != 7:
        raise ValueError("Unexpected dataset universe")
    labels = [r["y_true"] for r in by_method["R0"]]
    folds = [r["outer_fold"] for r in by_method["R0"]]
    for method, records in by_method.items():
        if [r["record_id"] for r in records] != ids or [r["y_true"] for r in records] != labels or [r["outer_fold"] for r in records] != folds:
            raise ValueError(f"Pairing/label/fold mismatch: {method}")
        if any((r["y_true"] == r["y_predicted"]) != (r["is_correct"] == "1") for r in records):
            raise ValueError("Saved correctness field mismatch")
    lookup = {c: i for i, c in enumerate(classes)}
    y = np.array([lookup[c] for c in labels], dtype=int)
    preds = {m: np.array([lookup[r["y_predicted"]] for r in records], dtype=int) for m, records in by_method.items()}
    counts = {c: labels.count(c) for c in classes}
    if counts != {"ZY01": 350, "ZY02": 5, **{f"ZY{i:02d}": 100 for i in range(3, 8)}}:
        raise ValueError("Class count drift")
    rng = np.random.default_rng(seed)
    draws = np.concatenate([rng.choice(np.flatnonzero(y == k), size=(iterations, counts[c]), replace=True)
                            for k, c in enumerate(classes)], axis=1)
    if not all(np.all((y[draws] == k).sum(axis=1) == counts[c]) for k, c in enumerate(classes)):
        raise ValueError("Bootstrap lost class strata")
    offsets = np.arange(iterations)[:, None] * 49
    points, boots, metric_rows, fold_rows = {}, {}, [], []
    saved = read_json(STAGED / "core/summary.json", sources) + read_json(STAGED / "combinations/summary.json", sources)
    references = {x["method"]: x for x in saved}
    for m in METHODS:
        codes = y * 7 + preds[m]
        cm = np.bincount(codes, minlength=49).reshape(7, 7)
        points[m] = metrics_from_confusion(np, cm)
        bcm = np.bincount((codes[draws] + offsets).ravel(), minlength=iterations * 49).reshape(iterations, 7, 7)
        boots[m] = metrics_from_confusion(np, bcm)
        for k, metric in enumerate(METRICS):
            if abs(points[m][k] - references[m][metric]) > 1e-12:
                raise ValueError(f"Historical metric replay mismatch: {m}/{metric}")
            lo, hi = np.quantile(boots[m][:, k], [.025, .975])
            metric_rows.append({"method": m, "metric": metric, "estimate": float(points[m][k]),
                                "conditional_percentile_CI95_low": float(lo), "conditional_percentile_CI95_high": float(hi),
                                "samples": len(y), "bootstrap_replicates": iterations,
                                "correct_samples": int(np.trace(cm))})
        for f in range(5):
            use = np.array([int(x) == f for x in folds])
            fcm = np.bincount(codes[use], minlength=49).reshape(7, 7)
            fm = metrics_from_confusion(np, fcm)
            fold_rows.append({"method": m, "outer_fold": f, "samples": int(use.sum()),
                              "correct": int(np.trace(fcm)), **{k: float(v) for k, v in zip(METRICS, fm)}})
    delta_rows, nominal_rows = [], []
    for m, baseline in PAIRS:
        for k, metric in enumerate(METRICS):
            lo, hi = np.quantile(boots[m][:, k] - boots[baseline][:, k], [.025, .975])
            delta_rows.append({"method": m, "baseline": baseline, "metric": metric,
                               "delta_percentage_points": float(100 * (points[m][k] - points[baseline][k])),
                               "conditional_CI95_low_percentage_points": float(100 * lo),
                               "conditional_CI95_high_percentage_points": float(100 * hi)})
        a, z = preds[m] == y, preds[baseline] == y
        corrected, harmed = int((a & ~z).sum()), int((~a & z).sum())
        nominal_rows.append({"method": m, "baseline": baseline, "baseline_wrong_method_correct": corrected,
                             "baseline_correct_method_wrong": harmed, "both_correct": int((a & z).sum()),
                             "both_wrong": int((~a & ~z).sum()), "nominal_exact_two_sided_p": exact_mcnemar(corrected, harmed)})
    running = 0.0
    for rank, row in enumerate(sorted(nominal_rows, key=lambda r: r["nominal_exact_two_sided_p"])):
        running = max(running, (len(nominal_rows) - rank) * row["nominal_exact_two_sided_p"])
        row["nominal_Holm_adjusted_p_seven_comparisons"] = min(1.0, running)
    index = read_json(Path("reports/paper-c/2026-10-31/best-candidate/candidate_index.json"), sources)
    axis = index["frequency_axis"]
    records, selected_rows = feature_sets(sources, by_method, axis)
    feature_rows, feature_summary = [], []
    for method in METHODS:
        method_rows = []
        for fa, fb in itertools.combinations(range(5), 2):
            a, b = records[method, fa], records[method, fb]
            sa, sb = a["indices"], b["indices"]
            ka, kb = len(sa), len(sb)
            inter, union = len(sa & sb), len(sa | sb)
            expected_inter = ka * kb / len(axis)
            ej = exact_uniform_jaccard(len(axis), ka, kb)
            row = {"method": method, "outer_fold_a": fa, "outer_fold_b": fb,
                   "family_a": a["family"], "family_b": b["family"], "features_a": ka, "features_b": kb,
                   "intersection": inter, "union": union, "jaccard": inter / union,
                   "uniform_matched_size_expected_intersection": expected_inter,
                   "uniform_matched_size_exact_expected_jaccard": ej,
                   "jaccard_minus_uniform_expectation": inter / union - ej,
                   "chance_adjusted_overlap": (inter - expected_inter) / (min(ka, kb) - expected_inter),
                   "source_ID_intersection": len(a["source_ids"] & b["source_ids"]),
                   "source_ID_union": len(a["source_ids"] | b["source_ids"])}
            feature_rows.append(row); method_rows.append(row)
        feature_summary.append({"method": method, "pair_count": 10,
                                "mean_selected_features": float(np.mean([len(records[method, f]["indices"]) for f in range(5)])),
                                **{k: float(np.mean([r[k] for r in method_rows])) for k in ["jaccard", "uniform_matched_size_exact_expected_jaccard", "jaccard_minus_uniform_expectation", "chance_adjusted_overlap"]}})
    output.mkdir(parents=True)
    for name, data in [("metric_confidence_intervals.csv", metric_rows), ("paired_delta_confidence_intervals.csv", delta_rows),
                       ("nominal_mcnemar_seven_prespecified_pairs.csv", nominal_rows), ("outer_fold_metrics.csv", fold_rows),
                       ("feature_pairwise_overlap.csv", feature_rows), ("feature_overlap_summary.csv", feature_summary),
                       ("selected_features_by_fold.csv", selected_rows)]:
        write_csv(output / name, data)
    plot_status = "NOT_RUN_MATPLOTLIB_UNAVAILABLE"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric in zip(axes, ["accuracy", "macro_f1"]):
            data = [r for r in metric_rows if r["metric"] == metric]
            val = np.array([r["estimate"] * 100 for r in data])
            low = np.array([r["conditional_percentile_CI95_low"] * 100 for r in data])
            high = np.array([r["conditional_percentile_CI95_high"] * 100 for r in data])
            ax.errorbar(np.arange(5), val, yerr=np.stack([val-low, high-val]), fmt="o", capsize=4)
            ax.set_xticks(np.arange(5), METHODS); ax.set_ylabel(metric + " (%)")
            ax.grid(axis="y", alpha=.25)
        figure.suptitle("Saved 855-sample OOF predictions: conditional stratified bootstrap")
        figure.text(.5, .01, "95% percentile intervals; fixed predictions; excludes training/selection uncertainty; not independent confirmation", ha="center", fontsize=8)
        figure.tight_layout(rect=[0,.06,1,.94])
        figure.savefig(output / "conditional_oof_intervals.png", dpi=170)
        figure.savefig(output / "conditional_oof_intervals.pdf")
        plt.close(figure)
        plot_status = "DONE_ACTUAL_STANDARD_MATPLOTLIB_PLOT"
    except ImportError:
        pass
    summary = {"schema": "research4-existing-prediction-evidence-v1", "status": "DONE_CONDITIONAL_DIAGNOSTICS_NOT_NEW_MODEL_VALIDATION",
               "started_at_utc": started, "ended_at_utc": datetime.now(timezone.utc).isoformat(),
               "source_worker_sha256": sha(__file__), "source_files": sources,
               "protocol": "Original855 5outer/2inner staged nested optimization; identical specimen/label/fold alignment verified",
               "samples": 855, "class_counts": counts, "methods": METHODS, "bootstrap_seed": seed,
               "bootstrap_replicates": iterations, "confidence_interval": "class-stratified paired percentile 95%; fixed observed class counts",
               "prespecified_comparisons": PAIRS, "familywise_adjustment": "Holm across seven nominal two-sided exact McNemar p values",
               "plot_status": plot_status, "recomputed_metrics_match_historical_1e_12": True,
               "unique_feature_set_records": len(records), "historical_frequency_axis_features": len(axis),
               "class_strata_counts_preserved_each_replicate": True,
               "models_fitted": 0, "new_predictions_generated": 0, "quantum_evaluations": 0,
               "raw_spectra_labels_weights_modified": False, "limitations": WARNINGS,
               "actual_process_CPU_seconds": time.process_time()-cpu_start,
               "actual_wall_seconds": time.monotonic()-wall_start,
               "environment": {"python": platform.python_version(), "numpy": np.__version__}}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 已保存855份OOF预测的条件不确定性与特征稳定性分析", "",
             "实际执行：固定种子20261106、2000次按真实类别分层的配对bootstrap；没有重新训练或选模。",
             "所有五阶段逐样本ID、标签和外层fold一致；Accuracy、Macro-F1、Balanced Accuracy全部与历史汇总重算一致。", "",
             "这些是**给定已保存预测的条件区间**。855份数据已有历史曝光；外层模型共享训练样品，区间不覆盖训练、选参和特征选择不确定性。", "",
             "| 阶段 | Accuracy | Macro-F1 | Balanced Accuracy |", "|---|---:|---:|---:|"]
    for m in METHODS:
        lines.append(f"| {m} | {points[m][0]*100:.4f}% | {points[m][1]:.4f} | {points[m][2]*100:.4f}% |")
    lines += ["", "区间及配对差值见CSV；McNemar为描述性的名义精确检验，七个预先写定比较进行Holm调整。不能将其当作独立批次确认或完整嵌套CV重训检验。", "",
              "特征稳定性直接读取25份真实外层训练元数据，不加载模型权重。不同保留数量采用匹配集合大小的均匀随机子集超几何期望作参照；该参照忽略邻近频率相关性。外层训练集合重叠，特征稳定性仅作描述。", "",
              "ZY02仅5份，所有bootstrap保持5次该类抽样，无法补足生物样本或跨批次证据。R4为每折内层选定的组合家族，家族可以不同。", "",
              "## 源文件与复现", "", "全部输入及SHA-256、实际CPU/wall、环境见summary.json。输出目录要求不存在，防止覆盖历史产物。", "",
              "```sh", "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=.runtime/quantum-pilot-site:.runtime/quantum-upgrade-site .runtime/bin/python -B scripts/research4_existing_model_evidence_20261106.py --output reports/cloud-followup-20261106-v1/model-evidence", "```", "",
              "不能生成真正独立新批次、标准品或HPLC/LC-MS数据；这些实验仍未执行。"]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    index = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in output.iterdir() if p.is_file()}
    (output / "artifact_index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "output": str(output), "CPU_seconds": summary["actual_process_CPU_seconds"],
                      "wall_seconds": summary["actual_wall_seconds"], "source_sha256": summary["source_worker_sha256"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()
    if args.iterations != 2000 or args.seed != 20261106:
        raise ValueError("Frozen diagnostic protocol requires 2000 replicates and seed20261106")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "reports/cloud-followup-20261106-v1"):
        raise ValueError("Independent followup report path required")
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    signal.alarm(180)
    run(output, args.iterations, args.seed)


if __name__ == "__main__":
    main()
