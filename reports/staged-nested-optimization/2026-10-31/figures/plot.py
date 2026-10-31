"""Plot the frozen 855-specimen staged nested-CV results; no model fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


COLORS = {"historical": "#7d8893", "anchor": "#9bb8d1", "mc": "#388b7a", "svm": "#365f91",
          "prep": "#c28a38", "combined": "#8465a3", "diagnostic": "#a7b4c0", "cnn": "#d59b71",
          "priority": "#3f9587", "control": "#56799c"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_oof(row):
    counts = row["class_counts"]
    if sum(counts.values()) != 855 or counts.get("ZY02") != 5:
        raise ValueError("The plotted result must be the pooled855 seven-class OOF evaluation")
    matrix = np.asarray(row["confusion_matrix"])
    if matrix.shape != (7, 7) or matrix.sum() != 855:
        raise ValueError("Expected one pooled855 confusion matrix per method")
    if not np.isclose(np.trace(matrix)/855, row["accuracy"], atol=1e-14, rtol=0):
        raise ValueError("Recorded accuracy and pooled OOF confusion matrix differ")


def dual_plot(rows, labels, roles, title, output, notes, *, legend_roles, reference=None):
    for row in rows:
        check_oof(row)
    fig, axes = plt.subplots(1, 2, figsize=(22.8, 11.5), sharey=True)
    for axis, metric, heading in zip(axes, ("accuracy", "macro_f1"),
                                   ("A  Pooled OOF accuracy", "B  Pooled seven-class macro F1")):
        for i, (row, role) in enumerate(zip(rows, roles)):
            value = row[metric]
            axis.barh(i, value*100, height=.66, color=COLORS[role], edgecolor="#52606d", linewidth=.8,
                      hatch="///" if role in {"diagnostic", "cnn"} else None)
            text = f"{value*100:.2f}%" if metric == "accuracy" else f"{value:.4f}"
            axis.text(value*100+.75, i, text, va="center", fontsize=12,
                      fontweight="bold" if role in {"svm", "combined", "priority"} else "normal")
        if reference is not None:
            axis.axvline(reference[metric]*100, color=COLORS["svm"], linestyle="--", linewidth=1.1, alpha=.65)
        axis.set_xlim(0, 100)
        axis.set_xticks(np.arange(0, 101, 20))
        axis.set_xlabel("Accuracy (%)" if metric == "accuracy" else "Macro F1 (value × 100)", fontsize=13)
        axis.set_title(heading, loc="left", fontsize=16, pad=12)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis="x", color="#dde3e9", linewidth=.8)
        axis.tick_params(labelsize=11)
    axes[0].set_yticks(np.arange(len(rows)), labels, fontsize=11.5)
    axes[0].invert_yaxis()
    fig.suptitle(title, fontsize=21, y=.975)
    handles = [Patch(facecolor=COLORS[role], edgecolor="#52606d", hatch="///" if role in {"diagnostic", "cnn"} else None,
                     label=label) for role, label in legend_roles]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.605, .925), ncol=2, frameon=False, fontsize=12)
    fig.subplots_adjust(left=.285, right=.965, top=.825, bottom=.30, wspace=.19)
    for i, note in enumerate(notes):
        fig.text(.045, .225-i*.032, note, fontsize=11.2, ha="left", va="top")
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix("."+suffix), dpi=180, facecolor="white")
    plt.close(fig)


def stage_comparison(core, combinations, output):
    methods = ("R0", "fixed_snv_C10_MC20", "R1", "R2", "R3")
    rows = [core[m] for m in methods] + [combinations["R4"]]
    labels = [
        "R0: historical nested baseline\nraw/SNV × 4 C settings",
        "Fixed SNV / C10 / MC20 anchor\n20 MC resamples, LV2, retained ratio 0.10",
        "R1: MC-UVE parameter search\n36 prespecified settings; fixed SNV / C10",
        "R2: SVM parameter search\n96 settings; inherits each outer R1 choice",
        "R3: preprocessing search\n7 recipes; inherits each outer R1 / R2 choice",
        "R4: combination-family search\n7 families selected within each outer training set",
    ]
    roles = ["historical", "anchor", "mc", "svm", "prep", "combined"]
    anchor, r1, r2 = core["fixed_snv_C10_MC20"], core["R1"], core["R2"]
    notes = [
        f"Controlled MC comparison: fixed SNV/C10/MC20 anchor {anchor['accuracy']*100:.2f}% → R1 {r1['accuracy']*100:.2f}%; R1−R0 is not a pure MC-only effect.",
        f"R2 has the highest pooled scores among the prespecified routes ({r2['accuracy']*100:.2f}%; macro F1 {r2['macro_f1']:.4f}); it is not a final-score winner substituted into later stages.",
        "Standard SVC vote (break_ties=False), not OVR-score argmax; one OOF prediction per 855 specimens on matched 5 outer / 2 inner folds; inner-only choices.",
        "Macro F1 is computed from the pooled 855 OOF predictions, not the arithmetic mean of five fold F1 values. No confidence intervals are shown.",
        "One batch per anonymous origin group; historically used data. Five folds are not independent new batches; no cross-batch or algorithm-novelty claim.",
        "This 855-specimen nested-CV comparison must not be ranked against the earlier 171-specimen holdout experiments; both axes start at zero."
    ]
    dual_plot(rows, labels, roles, "Four staged nested-CV searches with a separate fixed MC anchor", output,
              notes, legend_roles=[("historical", "Historical searched baseline"), ("anchor", "Fixed control for the MC effect"),
                                   ("mc", "R1 MC search"), ("svm", "R2 prespecified route"),
                                   ("prep", "R3 preprocessing search"), ("combined", "R4 inner-selected family")])
    return [{"method":r["method"], "accuracy":r["accuracy"], "macro_f1":r["macro_f1"]} for r in rows]


def preprocessing_comparison(core, protocol, output):
    order = protocol["R3"]["preprocessors"]
    rows = [core["R3_diagnostic_"+mode] for mode in order]
    titles = {"raw":"Raw spectrum", "snv":"SNV (matched R2 control)", "sg0_snv":"SG smoothing + SNV",
              "sg1_snv":"SG first derivative + SNV\nrequested priority probe", "sg2_snv":"SG second derivative + SNV",
              "msc":"MSC", "linear_detrend_snv":"Linear detrending + SNV"}
    labels = [titles[mode] for mode in order]
    roles = ["svm" if mode == "snv" else "priority" if mode == "sg1_snv" else "diagnostic" for mode in order]
    snv, sg1 = core["R3_diagnostic_snv"], core["R3_diagnostic_sg1_snv"]
    if not np.isclose(snv["accuracy"], core["R2"]["accuracy"], rtol=0, atol=1e-14):
        raise ValueError("SNV preprocessing diagnostic no longer matches R2")
    notes = [
        f"SG1+SNV priority probe did not exceed SNV: accuracy {sg1['accuracy']*100:.2f}% vs {snv['accuracy']*100:.2f}%; macro F1 {sg1['macro_f1']:.4f} vs {snv['macro_f1']:.4f}.",
        "These seven recipes were fixed before outer scoring. In every outer fold they share that fold's R1 MC and R2 SVM settings; source transforms/selectors are refitted.",
        "SNV reproduces the R2 route. The formal R3 pipeline uses its own inner-selected recipe per outer fold; outer diagnostics are not used to overwrite that choice.",
        "Each recipe uses standard SVC vote (break_ties=False), one OOF prediction per 855 specimens, matched 5 outer / 2 inner folds; pooled Macro F1.",
        "Single-batch retrospective anonymous-group data; five folds are not independent batches. No CIs, no extrapolation to new batches or algorithm-novelty claim.",
        "No comparison with earlier 171-sample holdout rankings. Dashed lines mark the matched SNV/R2 scores; both axes start at zero."
    ]
    dual_plot(rows, labels, roles, "Preprocessing ablation under the same per-outer MC and SVM choices", output,
              notes, legend_roles=[("svm", "Matched SNV / R2 control"), ("priority", "Requested SG1 + SNV probe"),
                                   ("diagnostic", "Other prespecified preprocessing diagnostic")], reference=snv)
    return [{"method":r["method"], "accuracy":r["accuracy"], "macro_f1":r["macro_f1"]} for r in rows]


def combination_comparison(core, combinations, protocol, output):
    families = protocol["families"]
    rows = [combinations["R4_diagnostic_"+family] for family in families] + [combinations["R4"]]
    ordinary = combinations["R4_diagnostic_ordinary_mcuve"]
    if (ordinary["confusion_matrix"] != core["R3"]["confusion_matrix"]
            or not np.isclose(ordinary["macro_f1"], core["R3"]["macro_f1"], rtol=0, atol=1e-14)):
        raise ValueError("Ordinary MC-UVE no longer exactly matches R3")
    k = core["R1"]["mean_selected_features"]
    family_titles = {
        "ordinary_mcuve":f"Ordinary MC-UVE + SVM\nmatched R3 control; mean K={k:g}",
        "budget_mcuve":f"Budget MC-UVE + SVM\n10× MC resampling count; mean K={k:g}",
        "stable_mcuve":f"Stable MC-UVE + SVM\nmatched mean K={k:g}",
        "stable_cars":f"Stable CARS-style + SVM\nmatched mean K={k:g}",
        "stable_joint":f"Joint stable MC/CARS + SVM\nmatched mean K={k:g}",
        "cnn_softmax":"CNN classifier head\n612 ordered inputs → 32-D embedding",
        "cnn_embedding_svm":"CNN embedding + SVM\n612 ordered inputs → 32-D embedding",
    }
    labels = [family_titles[family] for family in families] + ["R4: inner-selected family\nfamily may differ between outer folds"]
    roles = ["control" if f == "ordinary_mcuve" else "cnn" if f.startswith("cnn_") else "diagnostic" for f in families] + ["combined"]
    notes = [
        f"The first five traditional selectors inherit K separately from each outer fold's R1 choice (mean K={k:g}), as well as its R2 SVM and R3 preprocessing choices.",
        "Ordinary MC-UVE is exactly the R3 control. Budget MC-UVE matches 10× resampling count, not equal time/FLOPs; stable families use source-only resamples.",
        "CNN paths use all 612 ordered spectral inputs and a 32-dimensional embedding; they are not matched-dimension controls for the K-variable selectors.",
        "Each diagnostic and the formal R4 pipeline predict each of the 855 specimens once out of fold. R4 family selection is inside each outer training set.",
        "SVM heads use standard SVC vote; CNN head uses argmax. Same 5 outer / 2 inner folds, pooled Macro F1. Historical single-batch data; no independent batches or CIs.",
        "Diagnostics do not replace the inner-selected family after outer scoring. Do not rank these results against prior 171-holdout scores; both axes start at zero."
    ]
    dual_plot(rows, labels, roles, "Combination-family diagnostics with matched traditional feature counts", output,
              notes, legend_roles=[("control", "Exact ordinary MC-UVE / R3 control"),
                                   ("diagnostic", "Prespecified traditional-family diagnostic"),
                                   ("cnn", "CNN diagnostic: different input / feature dimensions"),
                                   ("combined", "Formal per-outer inner-selected R4")], reference=ordinary)
    return [{"method":r["method"], "accuracy":r["accuracy"], "macro_f1":r["macro_f1"]} for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--run-root", type=Path, default=here.parent)
    parser.add_argument("--output", type=Path, default=here)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    core_path, combo_path = args.run_root/"core/summary.json", args.run_root/"combinations/summary.json"
    core_protocol_path = args.run_root/"core/protocol_snapshot.json"
    combo_protocol_path = args.run_root/"combinations/protocol_snapshot.json"
    core, combinations = ({r["method"]:r for r in read(p)} for p in (core_path,combo_path))
    core_protocol, combo_protocol = read(core_protocol_path), read(combo_protocol_path)
    assert core_protocol["sample_count"] == combo_protocol["samples"] == 855
    assert core_protocol["outer_splits"] == combo_protocol["outer_splits"] == 5
    assert core_protocol["inner_splits"] == combo_protocol["inner_splits"] == 2
    plt.rcParams.update({"font.family":"DejaVu Sans", "pdf.fonttype":42, "ps.fonttype":42})
    rows = {
        "stage_comparison":stage_comparison(core,combinations,args.output/"stage_comparison"),
        "preprocessing_comparison":preprocessing_comparison(core,core_protocol,args.output/"preprocessing_comparison"),
        "combination_comparison":combination_comparison(core,combinations,combo_protocol,args.output/"combination_comparison"),
    }
    result = {"status":"pass", "new_model_fits":0, "sample_count_per_method":855,
        "metric_scope":"pooled5outer OOF; 2inner selection; historical within-batch data",
        "sources":{str(p):sha(p) for p in (core_path,combo_path,core_protocol_path,combo_protocol_path)},
        "script_sha256":sha(Path(__file__)), "plots":rows,
        "limits":["No CIs; folds are not independent batches.","R1 MC effect uses the fixedSNV/C10/MC20 anchor, not R0.",
            "R2 is a prespecified route, not an outer-score-selected replacement.",
            "Traditional selector K is matched per outer; CNN612input/32embedding is a distinct control.",
            "No ranking against the earlier171-specimen holdout results.","No verified algorithm novelty or cross-batch claim."]}
    (args.output/"plot_validation.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"pass","plots":3,"files":6,"new_model_fits":0}))


if __name__ == "__main__":
    main()
