#!/usr/bin/env python3
"""Paper figures from authenticated, already-computed 72h ML tables only.

No learned model, training input array, inference, quantum module or Git action
is loaded. All summaries remain conditional retrospective diagnostics.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT/"reports/research72h-20261107-v1"
INPUTS = {
    "evidence": (PREFIX/"initial-ml-evidence-001", "99cd2181e5b2a9ed6becd4f77436df6e2a55b59f0a6e2d9dbfae24010ddf7f2c"),
    "robustness": (PREFIX/"frozen-robustness/results-v1", "3338cd8b6802691e6ca016b1b29adbfa36caf1680ebb09408b6674172feba414"),
}
METRICS = ("accuracy", "macro_f1", "balanced_accuracy")
LABELS = {"full_svm": "Full-spectrum SVM", "shrinkage_lda": "Shrinkage LDA", "mlp": "Small MLP",
          "strong_mlp_200": "Strong MLP: 200 epochs", "strong_mlp_200_plus100_source_updates": "Strong MLP: +100 source updates",
          "R2_full_features": "Ablation: no MC", "R2_random_K": "Ablation: random matched K",
          "R2_no_C_gamma_search": "Ablation: fixed C/gamma", "R2_no_class_weight": "Ablation: no class weights",
          "R2_fixed_MC20_LV2_K62": "Ablation: fixed MC20/LV2/K62"}
CAPTION = "Fixed exposed 855-sample OOF predictions; one source lot per anonymous group. No new specimens or independent-batch confirmation."


def sha(file):
    h = hashlib.sha256()
    with Path(file).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read_json(file):
    return json.loads(Path(file).read_text())


def csv_rows(file):
    with Path(file).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def authenticate(directory, expected):
    index = directory/"artifact_index.json"
    if sha(index) != expected:
        raise ValueError("Actual completed input artifact index changed")
    records = read_json(index)
    for relative, value in records.items():
        file = directory/relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or file.is_symlink() or not file.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Artifact path escapes its completed run")
        expected_hash = value if isinstance(value, str) else value["sha256"]
        expected_bytes = None if isinstance(value, str) else value["bytes"]
        if sha(file) != expected_hash or (expected_bytes is not None and file.stat().st_size != expected_bytes):
            raise ValueError("Completed scientific artifact changed: "+relative)
    return {"path": str(index.relative_to(ROOT)), "sha256": expected, "verified_artifacts": len(records)}


def validate_noise(curves, raw):
    wanted = {(level, metric) for level in (0., .001, .005, .01) for metric in METRICS}
    observed = set()
    for row in curves:
        level, metric = float(row["noise_level"]), row["metric"]
        key = level, metric
        if key in observed or key not in wanted:
            raise ValueError("Unexpected/duplicate noise-curve row")
        observed.add(key)
        cases = [r for r in raw if r["kind"] == "noise" and float(r["level"]) == level]
        values = [float(r[metric]) for r in cases]
        expected_seeds = {""} if level == 0 else {"42", "43", "44"}
        if {r["seed"] for r in cases} != expected_seeds or len(values) != len(expected_seeds):
            raise ValueError("Noise seeds are not the actual fixed three-seed schedule")
        expected = {"mean": sum(values)/len(values), "minimum": min(values), "maximum": max(values)}
        if int(row["seed_count"]) != len(values) or any(not math.isfinite(float(row[k])) or abs(float(row[k])-v) > 1e-12 for k, v in expected.items()):
            raise ValueError("Noise summary does not reconstruct from actual completed cases")
    if observed != wanted:
        raise ValueError("Missing noise-curve rows")


def selected_pairs(rows):
    selected = [r for r in rows if r["method"] == "R2_reproduction" and r["metric"] == "macro_f1"]
    if len(selected) != 10 or {r["baseline"] for r in selected} != set(LABELS):
        raise ValueError("Expected all ten fixed strong-control/ablation pairs")
    for row in selected:
        values = [float(row[key]) for key in ("delta_percentage_points", "conditional_low_percentage_points", "conditional_high_percentage_points")]
        if not all(math.isfinite(v) for v in values) or values[1] > values[2] or int(row["bootstrap_replicates"]) != 2000:
            raise ValueError("Invalid actual conditional range")
    return sorted(selected, key=lambda r: list(LABELS).index(r["baseline"]))


def sensitivity_ranges(rows, pairs, kind):
    result = []
    for pair in pairs:
        subset = [r for r in rows if r["method"] == pair["method"] and r["baseline"] == pair["baseline"] and r["metric"] == "macro_f1"]
        if len(subset) != 5 or len({r["omitted"] for r in subset}) != 5 or any(r["omission_type"] != kind for r in subset):
            raise ValueError("Each sensitivity range needs five separate actual omissions")
        values = [float(r["delta_percentage_points"]) for r in subset]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Nonfinite sensitivity metric")
        result.append({"baseline": pair["baseline"], "omission_type": kind, "minimum_delta_pp": min(values),
                       "maximum_delta_pp": max(values), "full_sample_delta_pp": float(pair["delta_percentage_points"]),
                       "omissions": 5, "interpretation": "fixed-prediction deletion sensitivity; not a confidence interval"})
    return result


def write_csv(file, rows):
    with file.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def save(figure, output, name):
    figure.savefig(output/(name+".png"), dpi=220)
    figure.savefig(output/(name+".pdf"))


def figures(output, noise, pairs, rare, folds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4))
    for ax, metric, title in zip(axes, METRICS, ("Accuracy", "Macro-F1 × 100", "Balanced accuracy")):
        rows = sorted([r for r in noise if r["metric"] == metric], key=lambda r: float(r["noise_level"]))
        x = [float(r["noise_level"]) for r in rows]
        ax.plot(x, [100*float(r["mean"]) for r in rows], "o-", color="#0072B2")
        ax.fill_between(x, [100*float(r["minimum"]) for r in rows], [100*float(r["maximum"]) for r in rows], color="#0072B2", alpha=.18)
        ax.set(xlabel="Noise SD / source RMS bandwise SD", ylabel=title+" (%)"); ax.grid(alpha=.2)
    fig.subplots_adjust(bottom=.28, wspace=.32)
    fig.text(.5, .1, "Shading: min–max over three fixed corruption seeds; baseline has one saved replay.\nSynthetic full-export-axis perturbation; not biological variation or a confidence interval.", ha="center", fontsize=8)
    fig.text(.5, .02, CAPTION, ha="center", fontsize=7)
    save(fig, output, "noise_robustness"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, row in enumerate(pairs):
        ax.hlines(i, float(row["conditional_low_percentage_points"]), float(row["conditional_high_percentage_points"]), color="#0072B2", linewidth=2)
        ax.plot(float(row["delta_percentage_points"]), i, "o", color="#D55E00")
    ax.set_yticks(range(len(pairs)), [LABELS[r["baseline"]] for r in pairs]); ax.invert_yaxis(); ax.axvline(0, color="gray", linestyle="--")
    ax.set_xlabel("R2 − comparator: macro-F1 percentage points"); ax.grid(axis="x", alpha=.2)
    fig.subplots_adjust(left=.34, right=.97, bottom=.18)
    fig.text(.5, .07, "Ranges: paired class-stratified percentile bootstrap, conditional on fixed saved predictions.\nOverlapping OOF training sets; no training/selection uncertainty or significance decision.", ha="center", fontsize=8)
    fig.text(.5, .015, CAPTION, ha="center", fontsize=7)
    save(fig, output, "paired_macro_f1"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, records, title in zip(axes, (rare, folds), ("Omit each of five ZY02 specimens", "Omit each of five outer query folds")):
        for i, row in enumerate(records):
            ax.hlines(i, row["minimum_delta_pp"], row["maximum_delta_pp"], color="#0072B2", linewidth=2)
            ax.plot(row["full_sample_delta_pp"], i, "o", color="#D55E00")
        ax.set_title(title); ax.axvline(0, color="gray", linestyle="--"); ax.grid(axis="x", alpha=.2)
        ax.set_xlabel("R2 − comparator: macro-F1 percentage points")
    axes[0].set_yticks(range(len(pairs)), [LABELS[r["baseline"]] for r in pairs]); axes[0].invert_yaxis()
    fig.subplots_adjust(left=.28, right=.98, bottom=.18, wspace=.15)
    fig.text(.5, .07, "Lines: minimum–maximum after each separate omission; points: original full-855 delta.\nSaved predictions are not refitted. Omission ranges are not confidence intervals.", ha="center", fontsize=8)
    fig.text(.5, .015, CAPTION, ha="center", fontsize=7)
    save(fig, output, "rare_class_and_fold_sensitivity"); plt.close(fig)


def run(output):
    cpu, wall = time.process_time(), time.monotonic()
    evidence, robustness = INPUTS["evidence"][0], INPUTS["robustness"][0]
    verified = {name: authenticate(*entry) for name, entry in INPUTS.items()}
    e, r = read_json(evidence/"summary.json"), read_json(robustness/"summary.json")
    if e["status"] != "COMPLETE" or e["specimens"] != 855 or e["new_model_fits"] != 0 or r["cases"] != 18 or r["historical_baseline_correct"] != 821 or r["models_fitted"] != 0:
        raise ValueError("Expected actual completed 855-sample no-fit inputs")
    replay = read_json(robustness/"baseline_replay.json")
    if replay["identical_predictions"] != 855 or replay["scores_max_abs_difference"] != 0:
        raise ValueError("Actual exact baseline replay is required")
    noise = csv_rows(robustness/"noise_level_curves.csv"); validate_noise(noise, csv_rows(robustness/"metric_summary.csv"))
    pairs = selected_pairs(csv_rows(evidence/"paired_conditional_ranges.csv"))
    rare = sensitivity_ranges(csv_rows(evidence/"rare_class_single_deletion.csv"), pairs, "single_ZY02_specimen")
    folds = sensitivity_ranges(csv_rows(evidence/"outer_fold_omission.csv"), pairs, "outer_query_fold")
    output = Path(output).resolve()
    if not output.is_relative_to(PREFIX) or output == PREFIX or output.exists():
        raise ValueError("Fresh output child under the new report prefix required")
    output.mkdir(parents=True)
    figures(output, noise, pairs, rare, folds)
    write_csv(output/"noise_curve_data.csv", noise); write_csv(output/"paired_macro_f1_data.csv", pairs)
    write_csv(output/"omission_sensitivity_data.csv", rare+folds)
    with (output/"README.md").open("x", encoding="utf-8") as handle:
        handle.write("# 已完成ML实测的论文图表\n\n本目录只读取并验证已完成产物的SHA，不训练、不推理、不进行量子计算。\n\n")
        handle.write("- 噪声曲线阴影是同855样品在三个固定合成噪声种子下的最小—最大范围，不是生物样品置信区间。\n")
        handle.write("- 配对Macro-F1区间条件于保存的OOF预测；不包括重新训练、选参或特征选择的不确定性。\n")
        handle.write("- ZY02逐样品删除和外折省略图不重训模型，其范围只是影响诊断。\n")
        handle.write("- 每匿名产地组只有一来源批次，全部855份均已历史曝光；结果不构成新批次确认，也不能证明化学机制。\n")
    result = {"status": "DONE_AUTHENTICATED_ML_PAPER_FIGURES", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "output_directory": str(output), "new_model_fits": 0, "new_model_inferences": 0, "new_quantum_calculations": 0,
              "verified_inputs": verified, "cpu_seconds": time.process_time()-cpu, "wall_seconds": time.monotonic()-wall,
              "figure_pairs": 3, "source_sha256": sha(__file__), "limitations": [CAPTION, "All intervals/ranges are conditional/descriptive, never independent biological confirmation."]}
    (output/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    index = {str(p.relative_to(output)): {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()}
    (output/"artifact_index.json").write_text(json.dumps(index, indent=2)+"\n")
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze-only", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); run(args.output)


if __name__ == "__main__":
    main()
