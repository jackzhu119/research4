"""Archive image-guided results and exact saved predictions without cache bloat."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def archive(run_root, report, models):
    run_root, report, models = map(Path, (run_root, report, models))
    report.mkdir(parents=True, exist_ok=True); models.mkdir(parents=True, exist_ok=True)
    release = json.loads((run_root / "all_development_locks.json").read_text())
    model_files, archives = [], []
    for record in release["rounds"]:
        run = Path(record["selection_lock"]).parent
        if sha(run / "selection_lock.json") != record["selection_lock_sha256"]:
            raise ValueError("Changed development lock")
        if json.loads((run / "status.json").read_text())["status"] != "completed":
            raise ValueError("Final execution incomplete")
        target = report / record["name"]
        if record["name"] == "features":
            shutil.copytree(run / "compact_archive", target, dirs_exist_ok=True)
            source_models = run / "deliverable_models"
            shutil.copytree(source_models, models / "features", dirs_exist_ok=True)
            for original in sorted(source_models.rglob("*")):
                if original.is_file():
                    copied = models / "features" / original.relative_to(source_models)
                    model_files.append({"original": str(original), "archived": str(copied), "sha256": sha(original)})
        else:
            target.mkdir(parents=True, exist_ok=True)
            originals, development = {}, []
            for original in sorted(run.rglob("*")):
                if not original.is_file() or original.suffix == ".partial":
                    continue
                relative = original.relative_to(run)
                originals[str(relative)] = {"sha256": sha(original), "bytes": original.stat().st_size}
                if original.suffix in (".pt", ".joblib"):
                    if relative.parts[0] == "final":
                        copied = models / record["name"] / relative
                        copied.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(original, copied)
                        model_files.append({"original": str(original), "archived": str(copied), "sha256": sha(original)})
                    continue
                if relative.parts[0] == "development":
                    development.append(original)
                else:
                    copied = target / relative
                    copied.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(original, copied)
            packed = target / "development_evidence.tar.gz"
            with packed.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode="w") as tar:
                    for original in development:
                        tar.add(original, arcname=str(original.relative_to(run)), recursive=False)
            with tarfile.open(packed, "r:gz") as tar:
                for member in tar.getmembers():
                    stream = tar.extractfile(member)
                    assert stream is not None
                    assert hashlib.sha256(stream.read()).hexdigest() == originals[member.name]["sha256"]
            dump(target / "original_run_files.json", originals)
            dump(target / "archive_description.json", {"original_run": str(run), "development_files": len(development),
                "tar_member_bytes_verified": True, "development_binary_caches_omitted": True,
                "final_binary_caches_separate": str(models / record["name"])})
        archives.append({"name": record["name"], "original_run": str(run), "archive": str(target),
            "selection_lock_sha256": record["selection_lock_sha256"]})
    for original in sorted(run_root.iterdir()):
        if original.is_file() and original.suffix in (".json", ".jsonl", ".py", ".log"):
            shutil.copy2(original, report / original.name)
    for name in ("tables", "figures"):
        if (run_root / name).exists():
            shutil.copytree(run_root / name, report / name, dirs_exist_ok=True)
    protocol_dir = report / "protocols"; protocol_dir.mkdir(exist_ok=True)
    for original in sorted(Path("data/research").glob("image_*20261031*.json")):
        shutil.copy2(original, protocol_dir / original.name)
    for item in model_files:
        assert sha(Path(item["archived"])) == item["sha256"]
    dump(models / "manifest.json", {"artifacts": model_files,
        "feature_final_full_pipelines": 21, "probability_binary_caches_are_oof_models_not_full_source_predictors": True,
        "fusion_has_no_new_source_binary_weights": True})
    dump(report / "archive_validation.json", {"status": "pass", "tracks": archives,
        "model_artifacts_verified": len(model_files), "new_model_fits": 0,
        "release_sha256": sha(run_root / "all_development_locks.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("outputs/image-guided-20261031"))
    parser.add_argument("--report", type=Path, default=Path("reports/image-guided-optimization/2026-10-31"))
    parser.add_argument("--models", type=Path, default=Path("models/image_guided_20261031"))
    arguments = parser.parse_args()
    archive(arguments.run_root, arguments.report, arguments.models)
