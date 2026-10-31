"""Inventory the starting committed research state without changing old data."""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "reports/paper-c/2026-10-31/repository-state"
WORKFLOWS = [
    ("runtime", "runtime/2026-10-30", "docs/RUNTIME_USABILITY_REPORT.md", "软件/原算法兼容性，不是酸枣仁855成绩"),
    ("external_benchmarks", "external-benchmarks/2026-10-30", "docs/EXTERNAL_BENCHMARK_REPORT.md", "外部Raman/SERS谱行分类及相对丰度回归；不是THz跨植物迁移"),
    ("stage2", "research-stage2/2026-10-30", "docs/RESEARCH_STAGE2_REPORT.md", "外部谱行分类与药片/小麦回归校准；任务和模态独立"),
    ("approved_zaoren", "approved-zaoren/2026-10-31", "docs/ZAOREN_APPROVED_EXECUTION_REPORT.md", "855五外折/二内折，含α主区与敏感性诊断；旧90.41基线来源"),
    ("threeway", "zaoren-threeway/2026-10-31", "docs/ZAOREN_THREEWAY_REPORT.md", "513训练/171校准池/171预测；不能混入855 OOF表"),
    ("calibration_exploration", "calibration-optimization/2026-10-31", "docs/CALIBRATION_OPTIMIZATION_REPORT.md", "源内开发24/6门禁；七类1-shot与不含ZY02六类辅助分开"),
    ("representations", "representation-optimization/2026-10-31", "docs/REPRESENTATION_AND_LATENT_REPORT.md", "既定171预测；源表示改进与支持标签适配收益分开"),
    ("paper_neural", "paper-fewshot-classification/2026-10-31", "docs/PAPER_FEWSHOT_CLASSIFICATION_REPORT.md", "NIR/MIR回归论文启发的分类改写；FT/ProtoNet/FOMAML，非原文精确复现"),
    ("multiround", "multiround-optimization/2026-10-31", "docs/MULTIROUND_OPTIMIZATION_REPORT.md", "原171协议：MLP源监督优化、SVM适配、锚定微调和FOMAML，保留零支持控制"),
    ("image_guided", "image-guided-optimization/2026-10-31", "docs/IMAGE_GUIDED_OPTIMIZATION_REPORT.md", "原171协议：特征/概率/融合；MLP90.25为三种子平均，LDA92.40不是855成绩"),
    ("staged_nested", "staged-nested-optimization/2026-10-31", "docs/STAGED_NESTED_OPTIMIZATION_REPORT.md", "855相同5/2分区，R0—R4逐折选择，全部选择锁定后统一外测"),
    ("accuracy_gains", "staged-accuracy-gains/2026-10-31", "scripts/export_staged_accuracy_gains.py", "已有855 OOF整数正确数的派生增幅；新增拟合0"),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads((DEST / "repository_snapshot.json").read_text())
    head = snapshot["starting_head"]
    listing = subprocess.check_output(["git", "ls-tree", "-r", "-l", "-z", head], cwd=ROOT)
    tracked = []
    for item in listing.split(b"\0"):
        if not item:
            continue
        fields, name = item.split(b"\t", 1)
        mode, kind, blob, size = fields.decode().split()
        tracked.append({"path":name.decode(), "git_blob":blob, "bytes":int(size), "kind":kind})
    workflow_rows = []
    for run_id, relative, document, scope in WORKFLOWS:
        prefix = "reports/"+relative+"/"
        files = [p for p in tracked if p["path"].startswith(prefix)]
        indices = [p for p in files if Path(p["path"]).name in {
            "summary.json", "stage_summary.json", "benchmark_summary.json", "run_manifest.json",
            "source_manifest.json", "selection_lock.json", "outer_evaluation_release.json", "files_sha256.json"}]
        workflow_rows.append({"run_id":run_id,"report_directory":prefix.rstrip("/"),"document":document,
                              "scope":scope,"committed_files":len(files),"committed_bytes":sum(p["bytes"] for p in files),
                              "evidence_indices":[{"path":p["path"],"sha256":digest(ROOT/p["path"])} for p in indices]})
    provenance = [p for p in tracked if p["path"].startswith(("data/raw/","data/archive/","references/"))]
    archive_files = [p for p in tracked if Path(p["path"]).suffix.lower() in {".zip",".rar",".7z"}]
    branch_containment = []
    names = subprocess.check_output(["git","for-each-ref","--format=%(refname:short)","refs/heads","refs/remotes"],cwd=ROOT,text=True).splitlines()
    for name in names:
        branch_containment.append({"branch":name,"commits_not_in_starting_research":int(subprocess.check_output(
            ["git","rev-list","--count",head+".."+name],cwd=ROOT,text=True))})
    result = {"observed_at_utc":datetime.now(timezone.utc).isoformat(),"starting_commit":head,
              "tracked_files":len(tracked),"tracked_bytes":sum(p["bytes"] for p in tracked),
              "file_suffix_counts":dict(Counter(Path(p["path"]).suffix.lower() or "[none]" for p in tracked)),
              "branch_containment":branch_containment,"historical_workflows":workflow_rows,
              "original_provenance_file_count":len(provenance),"archive_files":archive_files,
              "current_input_registry":"data/active_workbooks.json","current_dataset_note":"data/CURRENT_DATASET.md",
              "raw_inventory_is_not_model_input_discovery":True,"new_model_fits":0,
              "algorithm_review":"docs/ALGORITHM_REVIEW.md","originality_review":"docs/ALGORITHM_ORIGINALITY_AND_LITERATURE.md"}
    (DEST/"research_inventory.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    with (DEST/"historical_experiment_ledger.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        keys=["run_id","report_directory","document","scope","committed_files","committed_bytes"]
        writer=csv.DictWriter(handle,fieldnames=keys,lineterminator="\n");writer.writeheader()
        writer.writerows({k:r[k] for k in keys} for r in workflow_rows)
    (DEST/"committed_provenance_inventory.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"starting_commit":head,"historical_workflows":len(workflow_rows),"tracked_files":len(tracked),
                      "archives":len(archive_files),"branches_contained":all(r["commits_not_in_starting_research"]==0 for r in branch_containment)},ensure_ascii=False))


if __name__ == "__main__":
    main()
