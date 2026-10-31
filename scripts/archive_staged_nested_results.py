"""Copy verified nested-search compact evidence and fold-specific models."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def archive(run_root, report, models):
    run_root, report, models = map(Path, (run_root, report, models))
    report.mkdir(parents=True, exist_ok=True); models.mkdir(parents=True, exist_ok=True)
    release = json.loads((run_root / "outer_evaluation_release.json").read_text())
    model_files, tracks = [], []
    for record in release["rounds"]:
        run = Path(record["selection_lock"]).parent
        if json.loads((run / "status.json").read_text())["status"] != "completed":
            raise ValueError("Incomplete outer evaluation")
        for key, path in (("selection_lock_sha256", run / "selection_lock.json"),
                          ("protocol_sha256", Path(record["protocol"])),
                          ("run_manifest_sha256", run / "run_manifest.json")):
            if sha(path) != record[key]:
                raise ValueError(f"Changed frozen artifact: {path}")
        source = run / "compact_archive"
        shutil.copytree(source, report / record["name"], dirs_exist_ok=True)
        for original in source.rglob("*"):
            if original.is_file():
                assert sha(original) == sha(report / record["name"] / original.relative_to(source))
        source_models = run / "deliverable_models"
        shutil.copytree(source_models, models / record["name"], dirs_exist_ok=True)
        for original in sorted(source_models.rglob("*")):
            if original.is_file():
                copied = models / record["name"] / original.relative_to(source_models)
                assert sha(original) == sha(copied)
                model_files.append({"original": str(original), "archived": str(copied),
                                    "sha256": sha(original), "bytes": original.stat().st_size})
        tracks.append({"name": record["name"], "original_run": str(run),
                       "archive": str(report / record["name"]), "selection_lock_sha256": record["selection_lock_sha256"]})
    for original in sorted(run_root.iterdir()):
        if original.is_file() and original.suffix in (".json", ".jsonl", ".py", ".log"):
            shutil.copy2(original, report / original.name)
    for name in ("tables", "figures"):
        if (run_root / name).exists():
            shutil.copytree(run_root / name, report / name, dirs_exist_ok=True)
    protocol_dir = report / "protocols"; protocol_dir.mkdir(exist_ok=True)
    for original in sorted(Path("data/research").glob("staged_nested*20261031.json")):
        shutil.copy2(original, protocol_dir / original.name)
    write(models / "manifest.json", {"artifacts": model_files,
        "scope": "Five outer-fold training partitions of684; not a single855-trained deployment model",
        "core_unique_pipelines": 48, "combination_unique_pipelines": 35,
        "combination_shared_cnn_state_dicts": 5})
    write(report / "archive_validation.json", {"status": "pass", "tracks": tracks,
        "model_artifacts_verified": len(model_files), "new_model_fits": 0,
        "raw_mc_coefficient_tensors_in_cloud_cache_only": True,
        "exact_coefficient_summary_and_original_hashes_preserved": True,
        "release_sha256": sha(run_root / "outer_evaluation_release.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("outputs/staged-nested-20261031"))
    parser.add_argument("--report", type=Path, default=Path("reports/staged-nested-optimization/2026-10-31"))
    parser.add_argument("--models", type=Path, default=Path("models/staged_nested_20261031"))
    args = parser.parse_args()
    archive(args.run_root, args.report, args.models)
