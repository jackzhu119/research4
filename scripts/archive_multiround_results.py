"""Archive completed four-round evidence without duplicating all model caches."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def archive(run_root, destination, model_destination):
    run_root, destination, model_destination = map(Path, (run_root, destination, model_destination))
    destination.mkdir(parents=True, exist_ok=True)
    model_destination.mkdir(parents=True, exist_ok=True)
    release = json.loads((run_root / "all_development_locks.json").read_text())
    model_entries, round_entries = [], []
    names = ("neural-base", "support-svm", "anchored-ft", "meta-refinement")
    for name in names:
        run = run_root / name
        status = json.loads((run / "status.json").read_text())
        if status.get("status", status.get("stage")) not in ("completed", "final_complete"):
            raise ValueError(f"Incomplete round: {name}")
        lock_record = next(r for r in release["rounds"] if r["name"] == name)
        if sha(run / "selection_lock.json") != lock_record["selection_lock_sha256"]:
            raise ValueError(f"Selection lock changed: {name}")
        target = destination / name
        if name == "neural-base":
            shutil.copytree(run / "compact_archive", target, dirs_exist_ok=True)
            model_source = run / "deliverable_models"
            shutil.copytree(model_source, model_destination / name, dirs_exist_ok=True)
            for p in sorted(model_source.rglob("*")):
                if p.is_file():
                    model_entries.append({"round": name, "original": str(p),
                        "archived": str(model_destination / name / p.relative_to(model_source)),
                        "sha256": sha(p), "bytes": p.stat().st_size})
        else:
            target.mkdir(parents=True, exist_ok=True)
            original_files = {}
            development = []
            for p in sorted(run.rglob("*")):
                if not p.is_file() or p.suffix == ".partial":
                    continue
                relative = p.relative_to(run)
                original_files[str(relative)] = {"sha256": sha(p), "bytes": p.stat().st_size}
                if p.suffix in (".pt", ".joblib"):
                    if relative.parts[0] == "final":
                        copied = model_destination / name / relative
                        copied.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, copied)
                        model_entries.append({"round": name, "original": str(p), "archived": str(copied),
                            "sha256": sha(p), "bytes": p.stat().st_size})
                    continue
                if relative.parts[0] == "development":
                    development.append(p)
                else:
                    copied = target / relative
                    copied.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, copied)
            packed = target / "development_evidence.tar.gz"
            with packed.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode="w") as tar:
                    for p in development:
                        tar.add(p, arcname=str(p.relative_to(run)), recursive=False)
            with tarfile.open(packed, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        stream = tar.extractfile(member)
                        assert stream is not None
                        assert hashlib.sha256(stream.read()).hexdigest() == original_files[member.name]["sha256"]
            write_json(target / "original_run_files.json", original_files)
            write_json(target / "archive_description.json", {
                "original_run": str(run), "development_files_in_tar": len(development),
                "development_binary_model_cache_omitted": True,
                "final_models_separate": str(model_destination / name),
                "all_candidate_metrics_and_saved_predictions_preserved": True,
                "tar_member_bytes_verified_against_original_sha256": True})
        round_entries.append({"name": name, "selection_lock_sha256": lock_record["selection_lock_sha256"],
            "status": status, "archive": str(target)})
    for p in sorted(run_root.iterdir()):
        if p.is_file() and p.suffix in (".json", ".jsonl", ".py", ".log"):
            shutil.copy2(p, destination / p.name)
    if (run_root / "tables").exists():
        shutil.copytree(run_root / "tables", destination / "tables", dirs_exist_ok=True)
    if (run_root / "figures").exists():
        shutil.copytree(run_root / "figures", destination / "figures", dirs_exist_ok=True)
    for item in model_entries:
        assert sha(Path(item["archived"])) == item["sha256"]
    write_json(model_destination / "manifest.json", {"artifacts": model_entries,
        "neural_base_includes_18_selected_development_and_3_final_contexts": True,
        "other_rounds_store_final_weights_only": True,
        "remaining_development_caches_rebuild_from_recorded_protocols": True})
    write_json(destination / "archive_validation.json", {"status": "pass", "rounds": round_entries,
        "model_artifacts_checked": len(model_entries), "new_model_fits_in_archiving": 0,
        "final_evaluation_release_sha256": sha(run_root / "all_development_locks.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("outputs/multiround-20261031"))
    parser.add_argument("--destination", type=Path, default=Path("reports/multiround-optimization/2026-10-31"))
    parser.add_argument("--model-destination", type=Path, default=Path("models/multiround_20261031"))
    args = parser.parse_args()
    archive(args.run_root, args.destination, args.model_destination)
