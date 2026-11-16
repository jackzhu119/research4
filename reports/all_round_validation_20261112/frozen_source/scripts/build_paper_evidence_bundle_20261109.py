#!/usr/bin/env python3
"""Archive saved research evidence and render paper tables; never fit a model.

Build requires the original completed worktrees. --verify is portable and uses
only the standard library and the committed bundle. Source files stay unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REL = Path("reports/paper-evidence-20261109")
V4_REL = Path("data/research/evidence_validation_20261108_v4")
OLD_REL = Path("data/research/species_relaunch_20261108_v2")
V4_SHA = "71a303e089c7c7b7b2557d62372aaceda44e2f91"
OLD_SHA = "8a388b8b4d8a5baf9f2e4321e76455c2c0b49eca"
TAIL_SHA = "df3e12656e2177cfcf0a298e66bdedc5c524cb74"
BASE_URL = "https://github.com/jackzhu119/research4/blob/"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def write_csv(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def safe_relative(base, value):
    path = base / value
    if path.resolve().is_relative_to(base.resolve()) and not Path(value).is_absolute():
        return path
    raise ValueError(f"Unsafe bundle path: {value}")


def freeze_manifest(repo):
    roots = [repo / REL, repo / "docs/paper-writing-20261109"]
    extra = [repo / "README.md", repo / "PAPER_WRITING_START_HERE.md", repo / "docs/RESEARCH4_EXPERIMENT_AUDIT.md",
             repo / "docs/RESEARCH4_PAPER_C_MASTER_PLAN.md", repo / "docs/RESEARCH4_PAPER_OUTLINE.md",
             repo / "scripts/build_paper_evidence_bundle_20261109.py",
             repo / "scripts/export_paper_tables_xlsx_20261109.py"]
    excluded = {"bundle_manifest.json", "bundle_verification.json", "git_publication_receipt.json"}
    files = sorted(set(extra + [p for r in roots for p in r.rglob("*")
                                if p.is_file() and "__pycache__" not in p.parts]))
    rows = [{"path": str(p.relative_to(repo)), "bytes": p.stat().st_size, "sha256": digest(p)}
            for p in files if p.exists() and p.name not in excluded]
    write_json(repo / REL / "bundle_manifest.json", {
        "schema": "research4-paper-evidence-manifest-v1", "files": rows,
        "excluded_self_referential_files": sorted(excluded),
        "scope": "Saved results and writing guidance; no new model fits or quantum calculation",
    })


def verify(repo):
    manifest = read(repo / REL / "bundle_manifest.json")
    for row in manifest["files"]:
        p = safe_relative(repo, row["path"])
        assert p.stat().st_size == row["bytes"], row["path"]
        assert digest(p) == row["sha256"], row["path"]
    s = read(repo / REL / "sources/current_v4/summary.json")
    cm_checked = 0
    support_averages_checked = 0
    for row in s["results"]:
        assert row["status"] == "complete"
        assert row["complete_repeats"] == row["expected_repeats"]
        assert len(row["repeats"]) == 3
        for metric in ["accuracy", "macro_f1", "balanced_accuracy"]:
            mean = sum(rep["metrics"][metric] for rep in row["repeats"]) / 3
            assert abs(mean - row["mean_metrics"][metric]) < 1e-12, (row["method"], metric)
        for rep in row["repeats"]:
            m = rep["metrics"]
            if "confusion_matrix" not in m:
                assert row["calibration"] in {"legacy_lambda1", "weak_lambda1000_blend005"}
                assert m["denominator_per_support_draw"] == 300
                assert abs(m["correct_mean_over_support_draws"] / 300 - m["accuracy"]) < 1e-12
                support_averages_checked += 1
                continue
            cm = m["confusion_matrix"]
            total = sum(map(sum, cm))
            correct = sum(cm[i][i] for i in range(len(cm)))
            assert total == m["denominator"] == 300
            assert correct == m["correct"]
            assert abs(correct / total - m["accuracy"]) < 1e-12
            f1s, recalls = [], []
            for i in range(len(cm)):
                tp, actual, predicted = cm[i][i], sum(cm[i]), sum(r[i] for r in cm)
                f1s.append(2 * tp / (actual + predicted) if actual + predicted else 0.)
                recalls.append(tp / actual if actual else 0.)
            assert abs(sum(f1s) / len(cm) - m["macro_f1"]) < 1e-12
            assert abs(sum(recalls) / len(cm) - m["balanced_accuracy"]) < 1e-12
            cm_checked += 1
    with (repo / REL / "tables/scientific_job_register.csv").open(encoding="utf-8-sig") as f:
        jobs = list(csv.DictReader(f))
    assert len(jobs) == 1635
    assert len({(x["queue"], x["job_id"]) for x in jobs}) == 1635
    assert all(x["status"] == "completed" and x["receipt_hash_verified"] == "True" for x in jobs)
    result = {"status": "PASS", "checked_utc": datetime.now(timezone.utc).isoformat(),
              "hashed_files": len(manifest["files"]), "scientific_jobs": len(jobs),
              "confusion_matrices_recomputed": cm_checked,
              "calibration_support_average_counts_checked": support_averages_checked,
              "metric_check": "independent confusion-matrix accuracy/Macro-F1/balanced-accuracy recomputation",
              "scope": "Portable saved-evidence validation, not model retraining or external confirmation"}
    write_json(repo / REL / "bundle_verification.json", result)
    print(json.dumps(result, ensure_ascii=False))


def build(repo, legacy, runtime):
    out = repo / REL
    provenance = []

    def copy(src, dest, commit=None, original=None):
        src, dst = Path(src), out / "sources" / dest
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            shutil.copyfile(src, dst)
            transport = "local_saved_file"
        elif commit and original:
            data = subprocess.run(["git", "-C", str(repo), "show", commit + ":" + original],
                                  capture_output=True, check=True).stdout
            dst.write_bytes(data)
            transport = "immutable_git_blob_missing_from_sparse_checkout"
        else:
            raise FileNotFoundError(src)
        provenance.append({"bundle_path": str(dst.relative_to(repo)), "original_local_path": str(src),
                           "original_repository_path": original, "source_commit": commit,
                           "source_url": BASE_URL + commit + "/" + original if commit and original else None,
                           "sha256": digest(dst), "bytes": dst.stat().st_size, "transport": transport})

    for name in ["summary.json", "metrics.csv", "REPORT.md", "complete.json"]:
        original = str(V4_REL / "snapshots/00330" / name)
        copy(repo / original, "current_v4/" + name, V4_SHA, original)
        original = "reports/00480/" + name
        copy(runtime / original, "legacy_final/" + name, TAIL_SHA, original)
    for name in ["protocol.json", "input_lock.json", "train_only_split.csv", "jobs.json",
                 "prior_execution_carry.json", "user_authorization_NEW_90000.json", "preparation_cpu.json"]:
        original = str(V4_REL / name)
        copy(repo / original, "current_v4/" + name, V4_SHA, original)
    for source, dest in [(runtime / "work/state.json", "legacy_final/authoritative_closed_state.json"),
                         (runtime / "work/control/state.json", "legacy_final/tail_job_state.json"),
                         (runtime / "evidence/prior_inner_closed.json", "legacy_final/prior_job_state.json"),
                         (runtime / "publications/00480/receipt.json", "legacy_final/github_final_receipt.json"),
                         (runtime / "original_manifest.json", "legacy_final/original_manifest.json"),
                         (repo / V4_REL / "resource-control/state.json", "current_v4/authoritative_closed_state.json"),
                         (repo / V4_REL / "control/state.json", "current_v4/job_state.json"),
                         (repo / V4_REL / "backups/00330/receipt.json", "current_v4/github_final_receipt.json")]:
        copy(source, dest)
    for name in ["data/active_workbooks.json", "data/CURRENT_DATASET.md",
                 "data/research/species_revision_20261108/split_manifest.csv",
                 "data/research/species_revision_20261108/split_summary.json",
                 "data/research/species_revision_20261108/protocol.json",
                 "data/research/species_revision_20261108/feature_audit.json",
                 "data/research/species_revision_20261108/independent_results_validation.json",
                 "data/research/species_revision_20261108/historical_relabel_audit.json",
                 "data/research/species_revision_20261108/classification/summary.json",
                 "data/research/species_revision_20261108/classification/metrics_overview.csv",
                 "data/research/species_revision_20261108/transfer/result.json"]:
        copy(repo / name, "data_and_fixed_split/" + name, V4_SHA, name)
    write_json(out / "source_provenance.json", provenance)
    remote = subprocess.run(["git", "-C", str(repo), "ls-remote", "--heads", "origin"],
                            capture_output=True, text=True, check=True, timeout=45)
    write_csv(out / "tables/branch_inventory.csv", [{"branch": line.split()[1].removeprefix("refs/heads/"),
              "commit": line.split()[0], "observed_utc": datetime.now(timezone.utc).isoformat()}
              for line in remote.stdout.splitlines()])

    all_jobs = []
    prior = read(runtime / "evidence/prior_inner_closed.json")
    tail = read(runtime / "work/control/state.json")
    v4state = read(repo / V4_REL / "control/state.json")
    for queue, root, manifest, states in [
        ("species_relaunch_v2", legacy, read(runtime / "original_manifest.json"), {**prior["jobs"], **tail["jobs"]}),
        ("evidence_validation_v4", repo, read(repo / V4_REL / "jobs.json"), v4state["jobs"])]:
        for job in manifest["jobs"]:
            if job["kind"] not in {"classical", "neural", "matched_classical"}:
                continue
            js = states[job["id"]]
            assert js["status"] == "completed"
            if queue == "evidence_validation_v4":
                rp = repo / V4_REL / "control/receipts" / (job["id"] + ".json")
            else:
                ctl = runtime / "work/control" if job["id"] in tail["jobs"] else legacy / OLD_REL / "control"
                rp = ctl / "receipts" / (job["id"] + ".json")
            assert digest(rp) == js["receipt_sha256"]
            receipt = read(rp)
            outputs = receipt["output_hashes"]
            for f, sha in outputs.items():
                assert digest(f) == sha, f
            summary_path = next(Path(f) for f in outputs if Path(f).name == "summary.json")
            s = read(summary_path)
            all_jobs.append({"queue": queue, "job_id": job["id"], "family": job["kind"],
                "status": js["status"], "exit_code": js.get("exit_code"), "attempts": js.get("attempts"),
                "started_unix": js.get("started_unix"), "finished_unix": js.get("finished_unix"),
                "process_cpu_seconds": s.get("process_cpu_seconds", s.get("elapsed_cpu_seconds", s.get("actual_cpu_seconds_this_invocation"))),
                "receipt_sha256": digest(rp), "receipt_hash_verified": True,
                "summary_path": str(summary_path.relative_to(root)), "summary_sha256": digest(summary_path),
                "command_json": json.dumps(job["command"], ensure_ascii=False),
                "direct_output_hashes_json": json.dumps(outputs, ensure_ascii=False),
                "scope": "previously_exposed_MASTER_TRAIN_internal_development"})
    write_csv(out / "tables/scientific_job_register.csv", all_jobs)

    v4 = read(out / "sources/current_v4/summary.json")
    legacy_report = read(out / "sources/legacy_final/summary.json")
    combined = []
    per_class, matrices = [], []
    for row in v4["results"]:
        combined.append({"protocol": "acid300_TRAIN_nested_3x5" if row["kind"] == "matched_classical" else "acid300_TRAIN_neural_5fold_3seeds",
                         **{k: v for k, v in row.items() if k not in {"repeats", "mean_metrics", "expected_repeats", "complete_repeats"}},
                         **row["mean_metrics"], "source": "sources/current_v4/summary.json"})
        key = "|".join(str(row[k]) for k in ["kind", "method", "selection_scope", "k", "representation", "source_count", "auxiliary", "target_epochs", "calibration", "shots_per_class"])
        for rep in row["repeats"]:
            m = rep["metrics"]
            for c in m.get("per_class", []):
                per_class.append({"setting": key, "repeat_seed": rep["seed"], **c})
            labels = m.get("class_order", [])
            for i, line in enumerate(m.get("confusion_matrix", [])):
                for j, count in enumerate(line):
                    matrices.append({"setting": key, "repeat_seed": rep["seed"], "true_label": labels[i], "predicted_label": labels[j], "count": count,
                                     "unit": "one_prediction_per_physical_specimen_per_repeat"})
    for row in legacy_report["results"]:
        combined.append({"protocol": "legacy_TRAIN_neural_source_only_fivefold_3seeds", **{k: v for k, v in row.items() if k not in {"per_seed", "mean_metrics", "mean_per_class", "complete_seeds", "expected_seeds"}},
                         **row["mean_metrics"], "source": "sources/legacy_final/summary.json"})
    write_csv(out / "tables/all_final_model_settings.csv", combined)
    write_csv(out / "tables/current_per_class_by_repeat.csv", per_class)
    write_csv(out / "tables/current_confusion_counts_by_repeat.csv", matrices)
    write_csv(out / "tables/paired_classical_effects.csv", [{**x, "cluster_bootstrap_accuracy_delta_95_interval": json.dumps(x["cluster_bootstrap_accuracy_delta_95_interval"])} for x in v4["paired_classical_effects"]])
    selection_rows = []
    for f in sorted((repo / V4_REL / "runs").glob("matched_*/summary.json")):
        s = read(f)
        for r in s["results"]:
            selection_rows.append({"repeat_seed": s["repeat_seed"], "outer_fold": s["outer_fold"], "family": r["family"],
                "k": r["k"], "is_primary_k": r["is_primary_k"], "config": json.dumps(r["config"], ensure_ascii=False),
                "n_selected": r.get("n_selected"), "alpha_selected": r.get("alpha_selected"),
                "recorded_candidate_cpu_seconds": r.get("actual_cpu_seconds"),
                "timing_scope": "candidate_saved_stage_not_full_end_to_end",
                "source": str(f.relative_to(repo))})
    write_csv(out / "tables/feature_and_parameter_selection.csv", selection_rows)
    render(out, v4)
    write_json(out / "build_record.json", {"status": "DONE", "built_utc": datetime.now(timezone.utc).isoformat(),
               "new_model_fits": 0, "new_quantum_calculations": 0, "scientific_jobs_indexed": len(all_jobs),
               "current_result_settings": len(v4["results"]), "legacy_neural_base_settings": len(legacy_report["results"]),
               "note": "Historical sources and archive manifests are added by separate evidence audits before final manifest freeze."})


def render(out, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    def save(fig, name):
        for ext in ["png", "pdf", "svg"]:
            path = figs / (name + "." + ext)
            fig.savefig(path, dpi=200, bbox_inches="tight")
            if ext == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        plt.close(fig)
    classic = [r for r in summary["results"] if r["kind"] == "matched_classical" and r["selection_scope"] == "primary"]
    methods = ["alpha_raw20", "alpha_snv20", "alpha_snv100", "alpha_snv_stability5x20", "concat_both_snv20", "fullalpha_shrinkLDA", "fullalpha_MLP"]
    labels = ["Raw + MC20", "SNV + MC20", "SNV + MC100", "SNV + stable MC", "n + alpha SNV", "Shrinkage LDA", "MLP"]
    rows = [next(r for r in classic if r["method"] == name) for name in methods]
    fig, ax = plt.subplots(figsize=(9, 4.6)); x = np.arange(len(rows))
    for offset, metric, label in [(-.16,"accuracy","Accuracy"),(.16,"macro_f1","Macro-F1")]:
        ax.bar(x+offset,[r["mean_metrics"][metric]*100 for r in rows],width=.30,label=label)
    ax.set_xticks(x, labels, rotation=25, ha="right");ax.set_ylim(0,100);ax.set_ylabel("Percent / F1 x 100")
    ax.set_title("Acid-jujube TRAIN development: 3 x 5 nested CV, 300 physical specimens");ax.legend();fig.tight_layout();save(fig,"01_classical_primary_comparison")
    fig, ax = plt.subplots(figsize=(7.5,4.5))
    for name in ["alpha_raw20","alpha_snv20","alpha_snv100","concat_both_snv20","n_raw20","n_snv20"]:
        rr=sorted([r for r in summary["results"] if r["method"]==name and r["selection_scope"]=="fixed_k"],key=lambda r:int(r["k"]))
        ax.plot([int(r["k"]) for r in rr],[100*r["mean_metrics"]["accuracy"] for r in rr],marker="o",label=name)
    ax.set_xlabel("Fixed feature budget K");ax.set_ylabel("Accuracy (%)");ax.set_title("Outer fixed-K ablations (not a rule for choosing K)");ax.legend(fontsize=8,ncol=2);fig.tight_layout();save(fig,"02_feature_budget_ablation")
    best=next(r for r in classic if r["method"]=="alpha_snv20")
    fig, axes=plt.subplots(1,3,figsize=(12,4))
    for ax,rep in zip(axes,best["repeats"]):
        cm=np.array(rep["metrics"]["confusion_matrix"]);im=ax.imshow(cm,vmin=0,vmax=60,cmap="Blues")
        for i in range(5):
            for j in range(5):ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>30 else "black",fontsize=8)
        ax.set_xticks(range(5),["ZY03","ZY04","ZY05","ZY06","ZY07"],rotation=45);ax.set_yticks(range(5),["ZY03","ZY04","ZY05","ZY06","ZY07"])
        ax.set_xlabel("Predicted");ax.set_ylabel("True");ax.set_title(f"Partition seed {rep['seed']}\nn=300, correct={rep['metrics']['correct']}")
    fig.suptitle("SNV + MC-UVE20 + SVM: repeated specimens, not n=900");fig.tight_layout();save(fig,"03_primary_confusion_by_repeat")
    nn=[r for r in summary["results"] if r["kind"]=="neural" and r["method"]=="neural" and r["calibration"]=="source_only" and r["source_count"]==240]
    fig,axes=plt.subplots(1,2,figsize=(9,4),sharey=True)
    for ax,rep in zip(axes,["alpha128","alpha512"]):
        for epochs,offset in [(144,-.17),(192,.17)]:
            rr=[next(r for r in nn if r["representation"]==rep and r["target_epochs"]==epochs and r["auxiliary"]==aux) for aux in ["target_only","hq_tm"]]
            ax.bar(np.arange(2)+offset,[100*r["mean_metrics"]["accuracy"] for r in rr],width=.32,label=f"{epochs} target epochs")
        ax.set_xticks([0,1],["Target-only","HQ + TM pretrain"]);ax.set_title(rep);ax.set_ylim(0,100);ax.legend(fontsize=8)
    axes[0].set_ylabel("Accuracy (%)");fig.suptitle("Transfer controls: same 240 source / 60 query in each fold");fig.tight_layout();save(fig,"04_transfer_representation_budget")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--legacy",type=Path,default=Path("/workspace/research4-relaunch"))
    p.add_argument("--runtime",type=Path,default=Path("/workspace/research4-runtime-20261108"))
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--build",action="store_true")
    g.add_argument("--freeze-manifest",action="store_true")
    g.add_argument("--verify",action="store_true")
    a=p.parse_args()
    if a.build: build(a.repo,a.legacy,a.runtime)
    elif a.freeze_manifest: freeze_manifest(a.repo)
    else: verify(a.repo)


if __name__=="__main__":
    main()
