#!/usr/bin/env python3
"""Render paper-inspired classification results from frozen, adjacent JSON files.

Run: python figures/plot.py [--report REPORT_DIR] [--output FIGURE_DIR]
No model training or statistical confidence interval estimation is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#4477AA"
TEAL = "#228833"
GREY = "#777777"
RED = "#AA3377"


def read_json(root: Path, name: str):
    return json.loads((root / name).read_text(encoding="utf-8"))


def base_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300,
    })


def decorate(ax, ylabel):
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)


def label_bars(ax, bars, *, fontsize=9):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 1.15, f"{h:.2f}",
                ha="center", va="bottom", fontsize=fontsize)


def save(fig, output: Path, stem: str):
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{suffix}", facecolor="white")
    plt.close(fig)


def strategy_figure(report: Path, output: Path):
    summary = read_json(report, "summary.json")
    lock = read_json(report, "selection_lock.json")["choices"]
    records = {(r["strategy"], r["method"]): r for r in summary}
    strategies = ["head_ft", "full_ft", "prototype", "fomaml_ft"]
    names = ["Head-only fine-tuning", "Full fine-tuning", "Episodic ProtoNet", "First-order MAML + FT"]
    for s in strategies:
        for method in ("zero", "selected", "diagnostic"):
            r = records[s, method]
            assert r["evaluations"] == 300 and r["seeds"] == [42, 43, 44]
        if lock[s]["fallback_to_zero"]:
            assert records[s, "zero"]["accuracy_mean"] == records[s, "selected"]["accuracy_mean"]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.subplots_adjust(left=.08, right=.98, bottom=.30, top=.83)
    x = np.arange(len(strategies))
    w = .235
    zero = [100 * records[s, "zero"]["accuracy_mean"] for s in strategies]
    selected = [100 * records[s, "selected"]["accuracy_mean"] for s in strategies]
    diagnostic = [100 * records[s, "diagnostic"]["accuracy_mean"] for s in strategies]
    bars0 = ax.bar(x - w, zero, w, color=BLUE, label="Own zero-support control")
    bars1 = ax.bar(x, selected, w, color=TEAL, label="Development-locked selection")
    bars2 = ax.bar(x + w, diagnostic, w, color="#EEEEEE", edgecolor=GREY,
                   hatch="////", label="Diagnostic candidate (includes rejected results)")
    for bars in (bars0, bars1, bars2):
        label_bars(ax, bars)
    # This is contextual benchmark accuracy from the existing classical experiment,
    # not one of the neural draw means or a denominator to combine with them.
    svm_accuracy_reference = 88.89
    ax.axhline(svm_accuracy_reference, color=RED, linestyle="--", linewidth=1.4)
    ax.text(3.49, svm_accuracy_reference + 1.1,
            f"Existing classical SVM reference: {svm_accuracy_reference:.2f}%",
            color=RED, ha="right", va="bottom", fontsize=10)
    labels = [name + ("\nselected = zero (fallback)" if lock[s]["fallback_to_zero"]
                     else "\nselected = calibrated") for s, name in zip(strategies, names)]
    ax.set_xticks(x, labels)
    decorate(ax, "Fixed-query classification accuracy (%)")
    ax.set_xlim(-.6, 3.6)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2,
              frameon=False, fontsize=9)
    fig.suptitle("Paper-inspired classification pilot: selected and diagnostic outcomes", y=.97, fontsize=14)
    fig.text(.08, .19, "Bars: means over 3 seeds × 100 support draws; each draw labels 1 specimen per class (7 total).", fontsize=10)
    fig.text(.08, .145, "Each strategy uses its own zero-support control; differences between source models are not calibration gains.", fontsize=9)
    fig.text(.08, .10, "Single-batch retrospective evaluation; the same 171 query specimens are reused in all 300 non-independent cases.", fontsize=9)
    fig.text(.08, .055, "No statistical confidence intervals. Paper-inspired classification adaptation; the original benchmark studied regression.", fontsize=9)
    save(fig, output, "01_strategy_accuracy")


def fomaml_figure(report: Path, output: Path):
    summary = read_json(report, "summary.json")
    per_seed = sorted([r for r in read_json(report, "per_seed_summary.json")
                       if r["strategy"] == "fomaml_ft"], key=lambda r: r["seed"])
    diagnostics = read_json(report, "fomaml_class_diagnostic.json")
    prep = read_json(report, "preparation.json")
    selected_config = read_json(report, "selection_lock.json")["choices"]["fomaml_ft"]["selected"]
    controls = {(r["strategy"], r["method"]): r for r in summary}
    assert [r["seed"] for r in per_seed] == [42, 43, 44]
    before = [100 * r["zero_accuracy"] for r in per_seed]
    after = [100 * r["selected_accuracy"] for r in per_seed]
    before.append(100 * controls["fomaml_ft", "zero"]["accuracy_mean"])
    after.append(100 * controls["fomaml_ft", "selected"]["accuracy_mean"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1, 1.55]})
    fig.subplots_adjust(left=.06, right=.98, bottom=.30, top=.82, wspace=.24)
    x = np.arange(4)
    w = .34
    b0 = axes[0].bar(x - w/2, before, w, color=BLUE, label="Zero-support control")
    b1 = axes[0].bar(x + w/2, after, w, color=TEAL, label="Selected calibration")
    label_bars(axes[0], b0, fontsize=8)
    label_bars(axes[0], b1, fontsize=8)
    for i, (a, b) in enumerate(zip(before, after)):
        axes[0].text(i, max(a, b) + 7, f"+{b-a:.2f} pp", ha="center", fontsize=9, color=TEAL)
    axes[0].set_xticks(x, [f"Seed {r['seed']}" for r in per_seed] + ["Mean"])
    axes[0].set_title("A  All three seeds retained")
    decorate(axes[0], "Fixed-query accuracy (%)")

    classes = sorted(diagnostics["zero"]["recall_mean"])
    counts = prep["roles"]["prediction"]["origin_counts"]
    assert counts["ZY02"] == 1
    r0 = [100 * diagnostics["zero"]["recall_mean"][c] for c in classes]
    r1 = [100 * diagnostics["selected"]["recall_mean"][c] for c in classes]
    x2 = np.arange(len(classes))
    axes[1].bar(x2 - w/2, r0, w, color=BLUE)
    axes[1].bar(x2 + w/2, r1, w, color=TEAL)
    axes[1].set_xticks(x2, [f"{c}\nn={counts[c]}" for c in classes])
    axes[1].set_title("B  Recall by anonymous origin (300-case mean)")
    decorate(axes[1], "Per-class recall (%)")
    axes[1].annotate("0% before and after", xy=(1, 0), xytext=(1.9, 16),
                     ha="center", fontsize=9, color=RED,
                     arrowprops={"arrowstyle": "->", "color": RED, "lw": 1})
    for i in [4, 5, 6]:
        axes[1].text(i, max(r0[i], r1[i]) + 3,
                     f"{r1[i]-r0[i]:+.2f} pp", ha="center", fontsize=9,
                     color=TEAL if r1[i] >= r0[i] else RED)

    fig.suptitle("First-order MAML: calibration improves the mean but leaves class-specific failures", y=.97, fontsize=14)
    fig.legend(handles=[b0, b1], labels=["Zero-support control", "Development-locked calibration"],
               loc="upper center", bbox_to_anchor=(.5, .92), ncol=2, frameon=False)
    fig.text(.06, .20, f"Selected support-only SGD: learning rate {selected_config['learning_rate']:g}, {selected_config['steps']} steps; 1 labelled specimen per class (7 total) for each draw.", fontsize=10)
    fig.text(.06, .15, "Single-batch retrospective evaluation; the same 171 query specimens are reused in 3 seeds × 100 draws (300 non-independent cases).", fontsize=9)
    fig.text(.06, .10, "Class n counts unique fixed-query specimens, not repeated predictions; ZY02 has only one query specimen and remains unrecognized.", fontsize=9)
    fig.text(.06, .055, f"Mean accuracy {after[-1]:.2f}% remains below the existing classical SVM reference (88.89%). No statistical confidence intervals.", fontsize=9)
    save(fig, output, "02_fomaml_seed_and_origin")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base_style()
    strategy_figure(args.report, args.output)
    fomaml_figure(args.report, args.output)
    print(f"Wrote 2 PNG and 2 PDF figures under {args.output}")


if __name__ == "__main__":
    main()
