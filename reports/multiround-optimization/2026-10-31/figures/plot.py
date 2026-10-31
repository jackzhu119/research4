"""Plot locked source-model results and rejected calibration diagnostics.

Only existing summary JSON is read. No training, adaptation, metric selection,
confidence interval estimation, or rewriting of experimental evidence occurs.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


HERE = Path(__file__).resolve().parent
RUN = HERE.parent


def read(relative):
    return json.loads((RUN/relative).read_text(encoding="utf-8"))


def row(rows, name):
    return next(item for item in rows if item["method"] == name)


def setup():
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11,
        "axes.titlesize":13, "axes.labelsize":12, "axes.spines.top":False,
        "axes.spines.right":False, "pdf.fonttype":42, "ps.fonttype":42,
        "savefig.facecolor":"white"})


def save(fig, name):
    for extension in ("png", "pdf"):
        fig.savefig(HERE/(name+"."+extension), dpi=260)
    plt.close(fig)


def label_bars(ax, bars, values, formatter):
    for bar,value in zip(bars,values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.2,
                formatter(value), ha="center", va="bottom", fontsize=10)


def source_models(svm, mlp, meta, control):
    reference = row(svm,"source_original")
    retained = row(meta,"selected_source_zero")
    meta500 = row(meta,"diagnostic_source_zero")
    supervised500 = row(meta,"supervised_diagnostic_source_zero")
    records = [reference,mlp,retained,meta500,supervised500]
    labels = ["Reference\nSNV + SVM", "R1 source\nMLP 64-32",
              "R4 retained\n100 FO updates", "Diagnostic\n500 FO updates",
              "Diagnostic control\n500 supervised\nupdates"]
    colors = ["#4b719e", "#449471", "#c18a36", "#698e9c", "#929aa4"]
    fig,axes = plt.subplots(1,2,figsize=(16,8.5))
    fig.subplots_adjust(left=.065,right=.985,bottom=.31,top=.83,wspace=.16)
    fig.suptitle("Source-model improvements and supervised controls", fontsize=20, y=.965)
    x = np.arange(len(records))
    for ax,metric,title,ylabel in [(axes[0],"accuracy_mean","A  Fixed-query accuracy","Accuracy (%)"),
                                    (axes[1],"macro_f1_mean","B  Seven-class macro F1","Macro F1 (value × 100)")]:
        values = [r[metric] for r in records]
        bars = ax.bar(x,np.asarray(values)*100,color=colors,width=.68,edgecolor="#3b4653",linewidth=.75,zorder=3)
        for index in (3,4):bars[index].set_hatch("///")
        label_bars(ax,bars,values,(lambda value:f"{value*100:.2f}%") if metric=="accuracy_mean" else (lambda value:f"{value:.4f}"))
        ax.set_ylim(0,100);ax.set_yticks(np.arange(0,101,20));ax.set_ylabel(ylabel)
        ax.set_xticks(x,labels,fontsize=9.4);ax.set_title(title,loc="left",pad=12)
        ax.grid(axis="y",color="#dce1e6",linewidth=.8,zorder=0)
    fig.legend(handles=[Patch(facecolor="#d1d8df",edgecolor="#3b4653",label="Reference / retained source"),
                        Patch(facecolor="#d1d8df",edgecolor="#3b4653",hatch="///",label="Preselected diagnostic; not the retained source")],
               loc="upper center",bbox_to_anchor=(.5,.91),ncol=2,frameon=False,fontsize=11)
    pairwise = control["pairwise_disagreement"]
    identical = all(r["different_predictions"]==0 for r in pairwise if
                    (r["first"],r["second"]) in {("m000_fomaml","m000_supervised"),("m002_fomaml","m002_supervised")})
    compared = [r for r in pairwise if (r["first"],r["second"]) in
                {("m000_fomaml","m000_supervised"),("m002_fomaml","m002_supervised")}]
    assert len(compared)==6 and identical
    notes = ["All source models use 513 training specimens and zero target-calibration labels; fixed query n = 171.",
             "Neural bars average seeds 42 / 43 / 44; SVM is the fixed reference. Diagnostic status was locked using development data.",
             "FO and matched supervised source models make identical query classifications at both budgets for all 3 seeds; logits differ.",
             "Single-batch retrospective study; the same historical query specimens are reused. Source gains are not calibration gains.",
             "Axes start at zero. No statistical confidence intervals; random seeds are not independent acquisition batches."]
    for i,note in enumerate(notes):fig.text(.065,.205-i*.037,note,ha="left",va="top",fontsize=10.6)
    save(fig,"source_models")


def calibration_effects(svm, anchored, meta):
    configs = [(row(svm,"source_original"),row(svm,"family_support_svm")),
               (row(anchored,"source_zero"),row(anchored,"diagnostic")),
               (row(meta,"diagnostic_source_zero"),row(meta,"diagnostic")),
               (row(meta,"supervised_diagnostic_source_zero"),row(meta,"supervised_diagnostic"))]
    labels = ["Support-weighted SVM\ndiagnostic", "Anchored MLP fine-tuning\nrejected diagnostic",
              "500-update FO + fine-tuning\nrejected diagnostic", "500-update supervised + fine-tuning\nmatched diagnostic"]
    source = np.asarray([pair[0]["accuracy_mean"] for pair in configs])
    adapted = np.asarray([pair[1]["accuracy_mean"] for pair in configs])
    assert np.isclose(row(anchored,"diagnostic_matched_zero")["accuracy_mean"],source[1])
    assert np.isclose(row(meta,"diagnostic_matched_zero")["accuracy_mean"],source[2])
    assert np.isclose(row(meta,"supervised_diagnostic_matched_zero")["accuracy_mean"],source[3])
    fig,ax = plt.subplots(figsize=(15,8.3));fig.subplots_adjust(left=.075,right=.985,bottom=.32,top=.81)
    fig.suptitle("Few-shot label effects: retain matched zero-support controls",fontsize=20,y=.96)
    x = np.arange(len(configs));width=.32
    before=ax.bar(x-width/2,source*100,width,color="#4b719e",edgecolor="#35404d",linewidth=.75,
                  label="Source / matched zero-support",zorder=3)
    after=ax.bar(x+width/2,adapted*100,width,color="#b7c8b7",edgecolor="#485b48",hatch="///",linewidth=.75,
                 label="Preselected calibration diagnostic",zorder=3)
    label_bars(ax,before,source,lambda value:f"{value*100:.3f}%")
    label_bars(ax,after,adapted,lambda value:f"{value*100:.3f}%")
    for i,gain in enumerate((adapted-source)*100):
        ax.text(i,97.2,f"{gain:+.3f} pp",ha="center",va="center",fontsize=11,
                color="#277540" if gain>0 else "#a64e70" if gain<0 else "#4b5563")
    ax.set_ylim(0,100);ax.set_yticks(np.arange(0,101,20));ax.set_ylabel("Fixed-query accuracy (%)")
    ax.set_xticks(x,labels,fontsize=10.2);ax.grid(axis="y",color="#dce1e6",linewidth=.8,zorder=0)
    fig.legend(handles=[before,after],labels=["Source / matched zero-support","Preselected calibration diagnostic"],
               loc="upper center",bbox_to_anchor=(.5,.9),ncol=2,frameon=False,fontsize=11)
    notes = ["Each labelled-support draw uses 1 independent specimen per anonymous class (7 labels total); the query stays fixed at 171 specimens.",
             "No support-based calibration passed the full development gate in R2 / R3 / R4: all retained their zero-support source predictor.",
             "The 500-update meta rows are diagnostics. R4 retained the 100-update source, with 90.2534% accuracy and no target calibration.",
             "Neural values average 3 seeds × 100 support draws; SVM uses 100 draws. These repeated cases are not independent samples or batches.",
             "Single-batch retrospective results on historically explored data. Axes start at zero; no statistical confidence intervals."]
    for i,note in enumerate(notes):fig.text(.075,.215-i*.038,note,ha="left",va="top",fontsize=10.6)
    save(fig,"calibration_effects")


def main():
    setup();svm=read("support-svm/summary.json");mlp=read("neural-base/summary.json")
    meta=read("meta-refinement/summary.json");anchored=read("anchored-ft/summary.json")
    control=read("meta-refinement/source_control_diagnostic.json")
    source_models(svm,mlp,meta,control);calibration_effects(svm,anchored,meta)
    print("Saved source_models.png/pdf and calibration_effects.png/pdf from existing JSON only")


if __name__ == "__main__":
    main()
