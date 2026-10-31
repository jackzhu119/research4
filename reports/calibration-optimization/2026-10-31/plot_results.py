"""Plot locked development-selected methods and their fixed-query controls."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path(__file__).with_name("summary.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("figures"))
    args = parser.parse_args()
    rows = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    def find(scope, selector, k, method):
        return next(r for r in rows if r["scope"] == scope and r["selector"] == selector
                    and r["shots_per_class"] == k and r["method"] == method)

    fig, axes = plt.subplots(1,2,figsize=(11,5.8),sharey=True)
    methods = [("source_identity","Frozen source","#24587b"),
               ("legacy_lambda1","Old correction","#bb684b"),
               ("development_selected","Development-selected","#3a8864")]
    for ax, selector, name in zip(axes,["full","mcuve20"],["SVM (612 points)","MC-UVE20 + SVM"]):
        for i,(method,label,color) in enumerate(methods):
            r=find("seven_class_all",selector,1,method)
            mean=100*r["accuracy_mean"]
            ax.bar(i,mean,color=color,width=.65,yerr=100*r["accuracy_sd"],capsize=4)
            ax.text(i,mean+100*r["accuracy_sd"]+1.8,f"{mean:.2f}%",ha="center",fontsize=10)
        ax.set_xticks([0,1,2],["Source","Old correction","Selected"])
        ax.set_title(name)
        ax.set_ylim(0,105)
        ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Accuracy on the same 171 specimens (%)")
    fig.suptitle("Seven-class 1-shot results: methods selected before query evaluation",fontsize=12)
    fig.text(.09,.02,"100 support draws on one fixed query. Bars show mean and draw SD, not batch uncertainty.\n"
             "Retrospective within-batch exploration; ZY02 has only one query specimen.",fontsize=9)
    fig.tight_layout(rect=(0,.1,1,.94))
    for suffix in ("png","pdf"):fig.savefig(args.output/f"seven_class_comparison.{suffix}",dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,5.8),sharey=True)
    for ax,selector,name in zip(axes,["full","mcuve20"],["SVM (612 points)","MC-UVE20 + SVM"]):
        for method,label,color in methods:
            data=[find("six_class_abundant",selector,k,method) for k in [1,3,5,10]]
            ax.errorbar([1,3,5,10],[100*r["accuracy_mean"] for r in data],
                yerr=[100*r["accuracy_sd"] for r in data],marker="o",color=color,capsize=3,label=label)
        ax.set_title(name);ax.set_xticks([1,3,5,10]);ax.set_xlabel("Labelled calibration specimens per class")
        ax.set_ylim(0,105);ax.grid(alpha=.2);ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Accuracy on the same 170 specimens (%)")
    axes[1].legend(loc="lower right",frameon=True,fontsize=9)
    fig.suptitle("Six-class auxiliary study: calibration sample count",fontsize=13)
    fig.text(.09,.02,"ZY02 excluded only in this auxiliary task. These are not seven-class results.\n"
             "100 nested support draws; the 170 query specimens remain fixed for every K.",fontsize=9)
    fig.tight_layout(rect=(0,.1,1,.94))
    for suffix in ("png","pdf"):fig.savefig(args.output/f"six_class_learning_curve.{suffix}",dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
