"""Plot all predeclared support draws on the same fixed prediction specimens."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("figures"))
    args = parser.parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6.3))
    labels = ["SVM (612 points)", "MC-UVE20 + SVM (62 points)"]
    for i, selector in enumerate(["full", "mcuve20"]):
        rows = [r for r in data["rows"] if r["selector"] == selector]
        source = next(r for r in rows if r["method"] == "source_only")
        values = np.asarray([100*r["metrics"]["accuracy"] for r in rows if r["method"] == "anchor_scores"])
        baseline = 100*source["metrics"]["accuracy"]
        ax.bar(i-.19, baseline, width=.34, color="#24587b", label="Frozen baseline" if i == 0 else None)
        ax.bar(i+.19, values.mean(), width=.34, color="#d89a45", yerr=values.std(), capsize=5,
               label="1-shot score adjustment: mean and SD" if i == 0 else None)
        ax.scatter(i+.19+np.linspace(-.10,.10,len(values)), values, color="#714c1e", s=21,
                   zorder=3, label="All 20 support draws" if i == 0 else None)
        ax.text(i-.19, baseline+2, f"{baseline:.2f}%", ha="center", fontsize=11)
        ax.text(i+.19, 98, f"Mean change: {values.mean()-baseline:+.2f} pp", ha="center", fontsize=11)
    ax.set_xticks([0,1], labels)
    ax.set_ylim(0,105)
    ax.set_ylabel("Accuracy on the same 171 specimens (%)")
    ax.set_title("Frozen models and 1-shot score adjustment", loc="left", pad=16, fontsize=15)
    ax.grid(axis="y", alpha=.22)
    ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=.94, fontsize=9)
    fig.text(.12,.02,"Retrospective within-batch split; 7 support specimens per draw.\n"
             "ZY02 has 1 query specimen. Draw SD is not an independent-batch confidence interval.", fontsize=9)
    fig.tight_layout(rect=(0,.09,1,1))
    for suffix in ("png","pdf"):
        fig.savefig(args.output / f"accuracy_comparison.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
