"""Render exportable research figures from saved, independently checked results."""
import csv
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/research/species_revision_20261108"
font_path = subprocess.check_output(["fc-match", "-f", "%{file}", "Noto Serif CJK SC"], text=True).strip()
font_manager.fontManager.addfont(font_path)
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams.update({"font.family": font_name, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "axes.titleweight": "bold"})


def main():
    classification = json.loads((BASE / "classification/summary.json").read_text())
    tasks = {r["task"]: r for r in classification["tasks"]}
    acid = tasks["acid_origins"]
    selected = [r for r in acid["aggregates"] if r["model"] == "R2_reselected"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    names = ["全谱SVM", "固定MC20", "历史R0分支", "历史R2分支"]
    models = ["full_svm", "fixed_MC20", "R0_reselected", "R2_reselected"]
    source, refit = [], []
    for model in models:
        rows = [r for r in acid["all_results"] if r["model"] == model]
        source.append(next(r["metrics"]["accuracy"] for r in rows if r["method"] == "source_only"))
        refit.append(next(r["metrics"]["accuracy"] for r in rows if r["method"] == "train_plus_calibration_full_retrain"))
    y = np.arange(4)
    axes[0].barh(y + .17, source, height=.3, color="#245d83", label="300份训练")
    axes[0].barh(y - .17, refit, height=.3, color="#57a59a", label="训练＋100份校准完整重训")
    for i, (a, b) in enumerate(zip(source, refit)):
        axes[0].text(a+.007, i+.17, f"{a:.0%}", va="center", fontsize=9)
        axes[0].text(b+.007, i-.17, f"{b:.0%}", va="center", fontsize=9)
    axes[0].set(yticks=y, yticklabels=names, xlim=(0, 1.05), title="酸枣仁五来源组：固定100份预测")
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -.08), frameon=False, fontsize=8)

    for method, label, color, marker in [
        ("legacy_anchor_lambda1", "原λ=1锚定", "#c65c4d", "o"),
        ("weak_anchor_lambda1000_blend005", "弱校准λ=1000／混合0.05", "#245d83", "s")]:
        values = [next(r["accuracy_mean"] for r in selected
                       if r["method"] == method and r["shots_per_class"] == str(k)) for k in [1, 3, 5, 10]]
        axes[1].plot([0, 1, 3, 5, 10], [.87, *values], marker=marker, color=color, label=label)
    axes[1].axhline(.87, color="#777777", linestyle="--", linewidth=1, label="R2源模型87%")
    axes[1].set(ylim=(0, 1), xticks=[0, 1, 3, 5, 10], xlabel="每类校准样品K（20次支持抽样均值）",
                title="冻结R2模型：少样本分数校准未改善")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(loc="lower right", frameon=False, fontsize=8)

    with (BASE / "transfer/paired_source_transfer.csv").open(newline="") as f:
        transfer = list(csv.DictReader(f))
    arms = ["A", "Aextra", "B"]
    for seed, color in [(42, "#245d83"), (43, "#57a59a"), (44, "#c65c4d")]:
        values = [float(next(r["accuracy"] for r in transfer if r["arm"] == arm and int(r["seed"]) == seed)) for arm in arms]
        axes[2].plot(range(3), values, "o-", color=color, alpha=.8, label=f"种子{seed}")
    means = [np.mean([float(r["accuracy"]) for r in transfer if r["arm"] == arm]) for arm in arms]
    axes[2].plot(range(3), means, "D--", color="#222222", linewidth=2, label="三种子均值")
    for i, m in enumerate(means):
        axes[2].annotate(f"{m:.2%}", (i, m), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)
    axes[2].set(ylim=(0, 1), xticks=range(3), xticklabels=["仅目标训练", "额外目标预训练", "黄芪＋天麻预训练"],
                title="跨物种表示迁移：B−A平均−2个百分点")
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].tick_params(axis="x", labelsize=8)
    axes[2].legend(loc="lower left", frameon=False, fontsize=8)
    fig.suptitle("种类拆分后的实验复核｜内部固定留出，非独立新批次确认", fontsize=14)
    out = BASE / "figures"
    out.mkdir(exist_ok=True)
    for extension in ["png", "svg"]:
        path = out / ("species_revision_results." + extension)
        fig.savefig(path, dpi=180, facecolor="white")
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(out / "species_revision_results.png")


if __name__ == "__main__":
    main()
