"""Plot frozen image-guided evidence; does not fit or choose any model."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


COLORS = {"reference": "#4e7199", "selected": "#c09137", "diagnostic": "#a4b1bd", "mlp": "#408b79"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def feature_directory(run, repo):
    records = read_json(run / "all_development_locks.json")["rounds"]
    record = next(r for r in records if r["name"] == "features")
    lock = Path(record["selection_lock"])
    lock = lock if lock.is_absolute() else repo / lock
    if sha(lock) != record["selection_lock_sha256"]:
        raise ValueError("Global feature selection lock changed")
    return lock.parent


def style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="#dce2e8", linewidth=.8)
    axis.tick_params(labelsize=11)


def feature_plot(run, repo, output):
    source = feature_directory(run, repo)
    summary = read_json(source / "summary.json")
    by_method = {r["method"]: r for r in summary}
    selected = read_json(source / "selection_lock.json")["choice"]["selected"]
    reference = by_method["reference"]["candidate"]
    data = [by_method["reference"], by_method["selected"],
            by_method["family_cars_svm"], by_method["family_mcuve_svm"],
            by_method["family_noise_uve_svm"], by_method["family_spa_svm"],
            by_method["family_pca_lda"]]
    fusion = read_json(run / "fusion/summary.json")
    old = next(r for r in fusion if r["method"] == "mlp_sup100_raw_softmax")
    assert by_method["family_full_svm"]["candidate"] == reference
    assert by_method["family_lda"]["candidate"] == selected
    labels = []
    for row in data:
        candidate = row["candidate"]
        prep = candidate["preprocessing"].upper()
        family = candidate["family"]
        title = {"full_svm": "full SVM", "lda": "shrinkage LDA", "cars_svm": "CARS-style + SVM",
                 "mcuve_svm": "MC-UVE + SVM", "noise_uve_svm": "noise-UVE + SVM",
                 "spa_svm": "single-chain SPA-style + SVM", "pca_lda": "PCA + shrinkage LDA"}[family]
        state = "reference" if row["method"] == "reference" else "selected" if row["method"] == "selected" else "preselected diagnostic"
        count = row["selected_features_mean"]
        feature_note = f"{int(count)} original frequencies" if float(count).is_integer() else f"{count:.1f} mean original frequencies"
        if family == "pca_lda":
            feature_note += f" -> {candidate['components']} PCs"
        labels.append(f"{prep} + {title}\n{state} | {feature_note}")
    labels.append("Previous MLP + 100 supervised updates\nfixed source-model reference")
    data.append(old)
    y = np.arange(len(data))
    styles = ["reference", "selected"] + ["diagnostic"]*5 + ["mlp"]
    fig, axes = plt.subplots(1, 2, figsize=(22.5, 12), sharey=True)
    for axis, metric, title in zip(axes, ("accuracy_mean", "macro_f1_mean"),
                                  ("A  Fixed-query accuracy", "B  Seven-class macro F1")):
        for index, (row, role) in enumerate(zip(data, styles)):
            value = row[metric]
            axis.barh(index, value*100, color=COLORS[role], edgecolor="#52606d", linewidth=.9,
                      height=.68, hatch="///" if role == "diagnostic" else None)
            label = f"{value*100:.2f}%" if metric == "accuracy_mean" else f"{value:.4f}"
            color = "#a93658" if role == "selected" and metric == "macro_f1_mean" else "#222222"
            axis.text(value*100+1., index, label, va="center", fontsize=12, color=color, fontweight="bold" if role == "selected" else "normal")
        axis.set_xlim(0, 100)
        axis.set_xticks(np.arange(0, 101, 20))
        axis.set_title(title, loc="left", fontsize=16, pad=13)
        axis.set_xlabel("Accuracy (%)" if metric == "accuracy_mean" else "Macro F1 (value × 100)", fontsize=13)
        style(axis)
    axes[0].set_yticks(y, labels, fontsize=11)
    axes[0].invert_yaxis()
    axes[1].axvline(old["macro_f1_mean"]*100, color=COLORS["mlp"], linestyle="--", linewidth=1.1, alpha=.75)
    fig.suptitle("Image-guided feature/classifier comparisons: accuracy and class balance", fontsize=21, y=.98)
    fig.legend(handles=[Patch(facecolor=COLORS["reference"], edgecolor="#52606d", label="Reference"),
                        Patch(facecolor=COLORS["selected"], edgecolor="#52606d", label="Development-selected feature model"),
                        Patch(facecolor=COLORS["diagnostic"], edgecolor="#52606d", hatch="///", label="Development-preselected family diagnostic"),
                        Patch(facecolor=COLORS["mlp"], edgecolor="#52606d", label="Previous fixed MLP source reference")],
               loc="upper center", bbox_to_anchor=(.61,.94), ncol=2, frameon=False, fontsize=12)
    fig.subplots_adjust(left=.265, right=.965, top=.84, bottom=.25, wspace=.18)
    notes = [
        "The selected LDA has higher accuracy (92.40%) but lower macro F1 (0.8352) than the previous MLP source (90.25%; 0.8787).",
        "All plotted family candidates were selected in source-only development. Diagnostic final scores do not replace the locked selected model.",
        "Counts average retained original frequencies across seeds; PCA-LDA uses 612 original inputs mapped to 40 source-fitted principal components.",
        "Means of three recorded seeds (42/43/44), same 171 query specimens; one batch per group; retrospective analysis on historically used data.",
        "No target-support labels, no independent new batches, no confidence intervals. Both horizontal axes start at zero."
    ]
    for i, note in enumerate(notes):
        fig.text(.055, .185-i*.032, note, fontsize=11.5, ha="left", va="top")
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"feature_comparison.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)
    return {"feature_summary": str(source/"summary.json"), "feature_summary_sha256": sha(source/"summary.json"),
            "source_reference_summary_sha256": sha(run/"fusion/summary.json"),
            "rows": [{"label":label,"accuracy":row["accuracy_mean"],"macro_f1":row["macro_f1_mean"]} for label,row in zip(labels,data)]}


def probability_plot(run, output):
    path = run/"probability/summary.json"
    summary = read_json(path)
    providers = ("svm", "mlp", "mlp_sup100")
    pairs = []
    for provider in providers:
        raw = next(r for r in summary if r["provider"] == provider and r["kind"] == "raw")
        calibrated = next(r for r in summary if r["provider"] == provider and r["kind"] == "temperature")
        assert abs(raw["accuracy_mean"]-calibrated["accuracy_mean"]) < 1e-15
        assert abs(raw["macro_f1_mean"]-calibrated["macro_f1_mean"]) < 1e-15
        assert raw["target_support_labels_used"] == calibrated["target_support_labels_used"] == 0
        pairs.append((raw, calibrated))
    labels = ["SVM", "MLP\n200 epochs", "MLP + 100\nsupervised updates"]
    fig, axes = plt.subplots(1,3,figsize=(20,10.7))
    x = np.arange(3)
    for axis, key, title in zip(axes, ("nll_mean", "brier_mean", "ece10_mean"),
                               ("A  Negative log-likelihood", "B  Multiclass Brier score", "C  ECE, 10 confidence bins")):
        raw = np.array([p[0][key] for p in pairs]); calibrated = np.array([p[1][key] for p in pairs])
        for values, offset, color in ((raw,-.2,"#4e7199"),(calibrated,.2,"#408b79")):
            axis.bar(x+offset,values,width=.37,color=color,edgecolor="#52606d",linewidth=.8)
            for loc,value in zip(x+offset,values):
                axis.text(loc,value+max(raw.max(),calibrated.max())*.035,f"{value:.4f}",ha="center",va="bottom",fontsize=11)
        axis.set_xticks(x,labels,fontsize=12)
        axis.set_ylim(0,1.21*max(raw.max(),calibrated.max()))
        axis.set_ylabel("Lower is better",fontsize=12)
        axis.set_title(title,loc="left",fontsize=15,pad=12)
        axis.spines[["top","right"]].set_visible(False)
        axis.set_axisbelow(True);axis.grid(axis="y",color="#dce2e8",linewidth=.8)
        axis.tick_params(axis="y",labelsize=11)
    fig.suptitle("Source-OOF temperature scaling improves probability reliability, not class decisions",fontsize=20,y=.975)
    fig.legend(handles=[Patch(facecolor="#4e7199",label="Raw softmax normalization (uncalibrated reference)"),
                        Patch(facecolor="#408b79",label="Temperature fitted on source-only OOF records")],
               loc="upper center",bbox_to_anchor=(.51,.925),ncol=2,frameon=False,fontsize=12)
    fig.subplots_adjust(left=.065,right=.975,top=.82,bottom=.31,wspace=.27)
    accuracies = ", ".join(f"{provider}: {pair[0]['accuracy_mean']*100:.2f}%" for provider,pair in zip(("SVM","MLP","MLP+100"),pairs))
    notes = [
        "Positive temperature preserves argmax: accuracy and macro F1 are unchanged for each provider.",
        f"Unchanged accuracies — {accuracies}. Formal fusion fallback has SVM weight 0; 90.25% is the single MLP source accuracy.",
        "Raw softmax of SVM margins/MLP logits is a normalized reference, not a claim of an already calibrated posterior.",
        "Brier is the sample mean of the sum of squared errors over all seven classes (not divided by seven). ECE uses ten equal-width bins.",
        "Only source OOF labels fit temperatures; zero target-support labels and no query statistics are fitted.",
        "Same 171 historical query specimens; three source-OOF seed contexts are not independent batches. Single-batch retrospective results; no CIs."
    ]
    for i,note in enumerate(notes):
        fig.text(.055,.245-i*.034,note,fontsize=11.4,ha="left",va="top")
    for suffix in ("png","pdf"):
        fig.savefig(output/f"probability_reliability.{suffix}",dpi=180,facecolor="white")
    plt.close(fig)
    return {"probability_summary":str(path),"probability_summary_sha256":sha(path),
            "rows":[{"provider":provider,"raw":raw,"temperature":calibrated} for provider,(raw,calibrated) in zip(providers,pairs)]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    here=Path(__file__).resolve().parent
    parser.add_argument("--run-root",type=Path,default=here.parent)
    parser.add_argument("--output",type=Path,default=here)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[3]
    args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,"ps.fonttype":42})
    source={"status":"pass","new_model_fits":0,
            "feature":feature_plot(args.run_root,repo,args.output),
            "probability":probability_plot(args.run_root,args.output)}
    (args.output/"plot_validation.json").write_text(json.dumps(source,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"pass","figures":4,"new_model_fits":0}))


if __name__=="__main__":
    main()
