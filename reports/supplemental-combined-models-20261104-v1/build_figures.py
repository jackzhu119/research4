#!/usr/bin/env python3
"""Combine already-audited, fixed-query metrics; never load spectra or fit models."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import resource
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLASSIC = ROOT / "reports/supplemental-spectra-independent-audit-20261104-v1"
NEURAL = ROOT / "reports/supplemental-multirepresentation-independent-audit-20261104-v2"
SCORE = ROOT / "reports/supplemental-multirepresentation-20261104-v2/score"
TASKS = ["zaoren_legacy7", "zaoren_expanded10", "huangqi_legacy4", "tianma_legacy4"]
COUNTS = dict(zip(TASKS, [171, 174, 82, 83]))
SOURCES = dict(zip(TASKS, [513, 516, 241, 242]))
CLASSES = dict(zip(TASKS, [7, 10, 4, 4]))
TITLES = dict(zip(TASKS, ["Acid jujube: 7 groups", "Acid jujube: 10 groups", "Huangqi: 4 groups", "Gastrodia: 4 groups"]))
MODELS = ["mcuve_svm", "svm", "shrinkage_lda", "mlp", "alpha_single", "alpha_capacity3", "alpha_n_time"]
LABELS = dict(zip(MODELS, ["MC-UVE + SVM", "SNV-SVM", "Shrinkage LDA", "MLP", "ResNet alpha", "ResNet alpha x3", "ResNet alpha+n+time"]))
SEEDS = [42, 43, 44]
INPUTS = [CLASSIC / "metrics_recomputed.json.gz", CLASSIC / "native_model_comparison.csv",
          CLASSIC / "paired_legacy171_A_vs_B.json", NEURAL / "all_metrics_recomputed.json",
          NEURAL / "old171_A_vs_B_paired.csv", NEURAL / "summary.json",
          NEURAL / "audit_stage_closed.json", SCORE / "model_comparison.csv"]


def read_json(p):
    with (gzip.open(p, "rt", encoding="utf-8") if p.suffix == ".gz" else p.open(encoding="utf-8")) as f:
        return json.load(f)


def read_csv(p):
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(p, rows):
    with p.open("x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def ref(p):
    return {"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}


def mean(x):
    return statistics.mean(x)


def close(a, b):
    assert math.isclose(float(a), float(b), abs_tol=1e-12, rel_tol=1e-12), (a, b)


def build(out):
    started = datetime.now(timezone.utc).isoformat()
    if out.exists():
        assert out == Path(__file__).resolve().parent and {p.name for p in out.iterdir()} == {Path(__file__).name}, "Fresh output required"
    else:
        out.mkdir(parents=True)
    assert read_json(NEURAL / "summary.json")["status"] == "PASS_INDEPENDENT_POST_Q_NUMERICAL_AUDIT"
    assert read_json(NEURAL / "audit_stage_closed.json")["status"] == "EXITED_HEALTHY"
    native = [r for r in read_json(INPUTS[0]) if r["method"] == "none" and r["shot"] == 0 and r["stratum"] == "all"]
    nn = [r for r in read_json(NEURAL / "all_metrics_recomputed.json") if r["subset"] == "overall"]
    assert len(native) == 48 and len(nn) == 36
    raw = []
    for r in native:
        raw.append({"task_id": r["task"], "model": r["family"], "seed": int(r["seed"]), "N": r["total"], "correct": r["correct"],
                    "accuracy_percent": 100*r["accuracy"], "macro_f1": r["macro_f1"], "balanced_accuracy_percent": 100*r["balanced_accuracy"], "source_type": "independently_recomputed_classical_native_no_calibration"})
    for r in nn:
        raw.append({"task_id": r["task_id"], "model": r["arm"], "seed": int(r["seed"]), "N": r["N"], "correct": r["correct"],
                    **{k: r[k] for k in ("accuracy_percent", "macro_f1", "balanced_accuracy_percent")}, "source_type": "independently_recomputed_neural_no_calibration"})
    assert len(raw) == 84 and len({(r['task_id'], r['model'], r['seed']) for r in raw}) == 84
    means = []
    classical_check = {(r["task"], r["model"]): r for r in read_csv(CLASSIC / "native_model_comparison.csv")}
    neural_check = {(r["task_id"], r["arm"]): r for r in read_csv(SCORE / "model_comparison.csv") if r["subset"] == "overall"}
    for task in TASKS:
        for model in MODELS:
            rows = sorted((r for r in raw if r["task_id"] == task and r["model"] == model), key=lambda r: r["seed"])
            assert [r["seed"] for r in rows] == SEEDS and {r["N"] for r in rows} == {COUNTS[task]}
            for r in rows:
                close(r["accuracy_percent"], 100*r["correct"]/r["N"])
            v = {"task_id": task, "model": model, "display_label": LABELS[model], "query_N_per_seed": COUNTS[task], "source_N": SOURCES[task], "class_count": CLASSES[task], "seeds": "42,43,44", "independent_seed_cohorts": False}
            for metric in ("accuracy_percent", "macro_f1", "balanced_accuracy_percent"):
                values = [r[metric] for r in rows]
                v[metric+"_mean"] = mean(values); v[metric+"_sample_sd"] = statistics.stdev(values)
            v["protocol"] = "fixed_source_support_query_roles; source_only_selection; no_calibration; exposed_development"
            v["error_bar_definition"] = "sample_SD_across_3_seeds_same_query_subjects_not_CI"
            if model in MODELS[:4]:
                check = classical_check[(task, model)]
                close(v["accuracy_percent_mean"], check["accuracy_percent"]); close(v["macro_f1_mean"], check["macro_f1"])
                close(v["balanced_accuracy_percent_mean"], check["balanced_accuracy_percent"])
                close(statistics.pstdev(r["accuracy_percent"] for r in rows), check["accuracy_population_sd_pp"])
                v["source_reported_SD_definition"] = "population_SD; combined_sample_SD_recomputed_from_actual_seeds"
            else:
                check = neural_check[(task, model)]
                for metric in ("accuracy_percent", "macro_f1", "balanced_accuracy_percent"):
                    close(v[metric+"_mean"], check[metric+"_mean"])
                    close(v[metric+"_sample_sd"], check[metric+"_sampleSD"])
                v["source_reported_SD_definition"] = "sample_SD"
            means.append(v)
    assert len(means) == 28
    write_csv(out/"model_comparison.csv", means); write_csv(out/"per_seed_metrics.csv", raw)

    paired = []
    for r in read_json(CLASSIC / "paired_legacy171_A_vs_B.json"):
        paired.append({"model": r["family"], "seed": int(r["seed"]), "N": 171, "A_correct": r["A_correct"], "B_old_correct": r["B_old_subset_correct"], "corrected": r["A_wrong_to_B_correct"], "harmed": r["A_correct_to_B_wrong"], "accuracy_delta_pp": r["accuracy_gain_pp"]})
    for r in read_csv(NEURAL / "old171_A_vs_B_paired.csv"):
        paired.append({"model": r["arm"], "seed": int(r["seed"]), "N": int(r["N"]), **{k:int(r[k]) for k in ("A_correct", "B_old_correct", "corrected", "harmed")}, "accuracy_delta_pp": float(r["accuracy_delta_pp"])})
    assert len(paired) == 21 and len({(r['model'], r['seed']) for r in paired}) == 21
    pairmeans = []
    for model in MODELS:
        z = sorted((r for r in paired if r["model"] == model), key=lambda r:r["seed"])
        assert [r["seed"] for r in z] == SEEDS and {r["N"] for r in z} == {171}
        for r in z:
            assert r["B_old_correct"]-r["A_correct"] == r["corrected"]-r["harmed"]
            close(100*(r["B_old_correct"]-r["A_correct"])/171, r["accuracy_delta_pp"])
        a, b = [100*r["A_correct"]/171 for r in z], [100*r["B_old_correct"]/171 for r in z]
        pairmeans.append({"model":model,"display_label":LABELS[model],"same_old_query_N":171,"seeds":"42,43,44", "A7_accuracy_percent_mean":mean(a),"A7_accuracy_percent_sample_sd":statistics.stdev(a),"B10_old171_accuracy_percent_mean":mean(b),"B10_old171_accuracy_percent_sample_sd":statistics.stdev(b),"accuracy_delta_pp_mean":mean(r["accuracy_delta_pp"] for r in z),"accuracy_delta_pp_sample_sd":statistics.stdev(r["accuracy_delta_pp"] for r in z),"corrected_mean":mean(r["corrected"] for r in z),"harmed_mean":mean(r["harmed"] for r in z),"interpretation":"B_adds_3_source_specimens_AND_3_classes;not_causal_data_only_gain;not_fresh_confirmation"})
    write_csv(out/"paired_old171_accuracy.csv", pairmeans); write_csv(out/"paired_old171_per_seed.csv", paired)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    colors = ["#527BA0"]*4+["#BE8E4F"]*2+["#268778"]
    plt.rcParams.update({"font.size":9,"axes.titlesize":11,"pdf.fonttype":42,"ps.fonttype":42})
    for metric, label, filename, maximum in [("accuracy_percent","Accuracy (%)","accuracy_comparison",108),("macro_f1","Macro-F1","macro_f1_comparison",1.08)]:
        fig, axs = plt.subplots(2,2,figsize=(13,8.5))
        for ax,task in zip(axs.flat,TASKS):
            z = [next(r for r in means if r["task_id"]==task and r["model"]==m) for m in MODELS]
            values=[r[metric+"_mean"] for r in z]; errors=[r[metric+"_sample_sd"] for r in z]
            ax.barh(range(7),values,xerr=errors,color=colors,error_kw={"elinewidth":1,"capsize":2,"ecolor":"#333333"},height=.65)
            ax.set_yticks(range(7),[LABELS[m] for m in MODELS]);ax.invert_yaxis();ax.set_xlim(0,maximum)
            ax.set_title(TITLES[task]+f"  |  query N={COUNTS[task]}",loc="left",fontweight="bold")
            ax.set_xlabel(label);ax.grid(axis="x",alpha=.18);ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
            for y,(v,e) in enumerate(zip(values,errors)):
                ax.text(v+e+maximum*.012,y,f"{v:.2f}",va="center",fontsize=8)
        fig.suptitle("Fixed-query pipeline comparison\nMean +/- sample SD across seeds 42, 43, 44; repeated subjects, development evaluation",fontsize=12,y=.99)
        fig.legend(handles=[Patch(color=colors[0],label="Classical pipelines"),Patch(color=colors[4],label="ResNet alpha / control"),Patch(color=colors[6],label="ResNet alpha+n+time")],loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,.005))
        fig.tight_layout(rect=[0,.045,1,.94],h_pad=2.2,w_pad=3)
        for ext in ['png','pdf']:fig.savefig(out/(filename+'.'+ext),dpi=220)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5.4));x=list(range(7));width=.34
    a=[r['A7_accuracy_percent_mean'] for r in pairmeans];b=[r['B10_old171_accuracy_percent_mean'] for r in pairmeans]
    ax.bar([v-width/2 for v in x],a,width,yerr=[r['A7_accuracy_percent_sample_sd'] for r in pairmeans],label='A: 7-class model',color='#527BA0',capsize=2)
    ax.bar([v+width/2 for v in x],b,width,yerr=[r['B10_old171_accuracy_percent_sample_sd'] for r in pairmeans],label='B: 10-class model, old171 only',color='#BE8E4F',capsize=2)
    ax.set_xticks(x,[LABELS[m] for m in MODELS],rotation=16,ha='right');ax.set_ylabel('Accuracy (%)');ax.set_ylim(0,108)
    ax.set_title('Same 171 old query specimens: paired A/B model comparison',fontweight='bold')
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False);ax.legend(loc='lower left',frameon=False)
    fig.text(.5,.02,'B adds three source specimens AND three classes. This does not isolate a causal data benefit or establish new confirmation.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.065,1,.98])
    for ext in ['png','pdf']:fig.savefig(out/('paired_old171_accuracy.'+ext),dpi=220)
    plt.close(fig)
    note='''# Combined fixed-query model figures

Actual independently audited metrics only; no fitting, spectra, labels or model weights were loaded by this plotting script. Four task-specific comparisons share the registered source/support/query memberships within each task. Pipelines have different source-search budgets and preprocessing; this is not an isolated classifier or feature contribution experiment.

- `model_comparison.csv`: 28 pipeline/task summaries. `per_seed_metrics.csv`: all84 actual seed rows.
- `accuracy_comparison.png/.pdf` and `macro_f1_comparison.png/.pdf`: all seven no-calibration pipelines on each fixed query.
- `paired_old171_accuracy.csv`: 7 model summaries; `paired_old171_per_seed.csv`:21 paired seed rows; `paired_old171_accuracy.png/.pdf`: same old171 queries, A seven-class versus B ten-class models. B adds three source specimens and three classes, so the difference is not a causal estimate of additional-data benefit.

All error bars are **sample SD over the same three seeds**, not confidence intervals and not independent new subject cohorts. The classical original table used population SD; this build derives common sample SD directly from the three independently recomputed seed metrics and checks the original means/population SD. Neural sample SD and all means are checked against the original score table. All per-seed accuracy values are checked against actual correct/N; paired net gains are checked against corrected minus harmed counts.

These are exposed development cohorts. The old96.0234% five-outer/two-inner result is a separate protocol and is not plotted as a directly comparable line. New ten-class query groups have only one specimen each, and repeated seeds do not increase that independent N. Anonymous labels do not verify botanical identity, named geography or cross-batch performance. C-versus-single neural changes include extra alpha representation/resolution as well as network capacity; the matched multi-representation effect is M-versus-C.

Reproduce into a **new** directory with the existing CPU Python and installed plotting libraries:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/workspace/research4/.runtime/quantum-pilot-site:/workspace/research4/.runtime/quantum-upgrade-site /workspace/research4/.runtime/bin/python reports/supplemental-combined-models-20261104-v1/build_figures.py --output reports/supplemental-combined-models-REPRODUCTION
```
'''
    (out/'README.md').write_text(note,encoding='utf-8')
    manifest={"status":"DONE_ACTUAL_AUDITED_METRICS_ONLY_NO_FITS","started_at_utc":started,"finished_at_utc":datetime.now(timezone.utc).isoformat(),"input_refs":[ref(p) for p in INPUTS],"build_script":ref(Path(__file__).resolve()),"summary_rows":28,"seed_rows":84,"paired_summary_rows":7,"paired_seed_rows":21,"all_mean_SD_correct_N_and_pair_checks":"PASS","seed_SD_definition":"sample SD (n=3), same physical queries; not confidence intervals","protocol_caveats":"Different pipelines/search budgets; no-calibration; exposed development; historical96.0234%not_directly_comparable","fits":0,"raw_spectra_or_weights_read":False,"process_CPU_seconds":resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,"outputs":[ref(p) for p in sorted(out.iterdir()) if p.is_file() and p.name not in ('source_manifest.json','build_figures.py')]}
    with (out/'source_manifest.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:manifest[k] for k in ('status','summary_rows','seed_rows','paired_summary_rows','paired_seed_rows','process_CPU_seconds')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU,(60,65))
    build(a.output.resolve())
