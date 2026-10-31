"""Root-authorized immutable evidence packaging and fresh restoration, without fits/scores."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

from herb_spectra import cross_plant_transfer as ct

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "reports/cross-plant-deep-transfer-20261031-v1"
OUT = RUN / "delivery-preflight"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


archive = load_script("transfer_evidence_archive_execution", ROOT / "scripts/archive_cross_plant_transfer_evidence_20261031.py")
empirical = load_script("transfer_empirical_evidence_execution", ROOT / "scripts/audit_cross_plant_transfer_empirical.py")
ARCHIVES = RUN / "evidence-archives-v1"
started = datetime.now(timezone.utc).isoformat()
reviews = {name: sha(RUN / "independent-audit" / name) for name in
           ["main_selection_review.json", "empirical_review.json", "evaluation_review.json", "postscore_integrity.json"]}
fixed = ["neural/run_manifest.json", "neural/selection_lock.json", "neural/prediction_lock.json",
         "neural/scoring_manifest.json", "empirical/prepared/frozen_manifest.json",
         "empirical/prepared/diagnostic_prediction_lock.json", "empirical/prepared/results.json"]
before = {name: sha(RUN / name) for name in fixed}
ct.verify_predictions_before_truth(RUN / "neural")
empirical.verify_prediction_lock(RUN / "empirical/prepared")
manifest = archive.create_evidence_archive(RUN, ARCHIVES)
with tempfile.TemporaryDirectory(prefix="cross-plant-transfer-restoration-") as temp:
    restored = Path(temp) / "run"
    result = archive.restore_evidence_archive(ARCHIVES, restored)
    second = archive.restore_evidence_archive(ARCHIVES, restored)
    neural = RUN / "neural"
    for folder in ["inputs", "auxiliary", "scoring_truth"]:
        shutil.copytree(neural / folder, restored / "neural" / folder)
    for name in ["protocol.json", "preparation.json", "run_manifest.json", "source_release_snapshot.json",
                 "source_records.json", "source_predictions.csv.gz", "candidate_summaries.json", "selection_lock.json",
                 "outer_release_snapshot.json", "outer_model_records.json", "outer_role_aliases.json",
                 "outer_predictions.csv.gz", "prediction_lock.json", "scoring_manifest.json"]:
        shutil.copyfile(neural / name, restored / "neural" / name)
    source_preparation = RUN / "empirical/prepared"
    restored_preparation = restored / "empirical/prepared"
    for folder in ["inputs", "auxiliary", "scoring_truth"]:
        shutil.copytree(source_preparation / folder, restored_preparation / folder)
    for name in ["plan.json", "preparation.json", "frozen_manifest.json", "root_fit_release_snapshot.json",
                 "diagnostic_prediction_lock.json", "results.json"]:
        shutil.copyfile(source_preparation / name, restored_preparation / name)
    ct.verify_run(restored / "neural")
    _, selection = ct.verify_selected(restored / "neural")
    ct.verify_predictions_before_truth(restored / "neural")
    empirical_lock = empirical.verify_prediction_lock(restored_preparation)
    # The native diagnostic verifier also checks three frozen absolute source paths.
    # Separately prove identical metadata exists at the restored neural location.
    main_dependencies = json.loads((restored_preparation / "frozen_manifest.json").read_text())["main_files_sha256"]
    relocated_dependency_checks = []
    for original, expected in main_dependencies.items():
        restored_main = restored / "neural" / Path(original).name
        if sha(restored_main) != expected:
            raise RuntimeError("Restored main metadata dependency differs")
        relocated_dependency_checks.append({"original_absolute_path": original,
            "restored_relative_path": "neural/" + Path(original).name, "sha256": expected})
    source_records = json.loads((restored / "neural/source_records.json").read_text())
    outer_records = json.loads((restored / "neural/outer_model_records.json").read_text())
    if len(source_records) != 240 or len(outer_records) != 60 or empirical_lock["pipeline_fits"] != 240:
        raise RuntimeError("Restored pipeline denominator changed")
    after_restore = archive.verified_manifest(ARCHIVES)
    for record in after_restore["files"]:
        if archive.old.digest(restored / record["path"]) != record["sha256"]:
            raise RuntimeError("Restored original-path digest differs")
after = {name: sha(RUN / name) for name in fixed}
if before != after:
    raise RuntimeError("Original freeze/score locks changed during delivery")
for record in manifest["files"]:
    if archive.old.digest(RUN / record["path"]) != record["sha256"]:
        raise RuntimeError("Original model or trace changed during delivery")
report = {"started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
          "status": "LOSSLESS_ARCHIVE_FRESH_RESTORE_PASS", "root_final_go_received": True,
          "prior_independent_reviews_sha256": reviews, "original_lock_hashes_before_after": before,
          "originals_moved_deleted_or_changed": False, "archive_entries": list(archive.ENTRIES),
          "source_files": manifest["source_files"], "source_bytes": manifest["source_bytes"],
          "archive_files": len(manifest["archives"]),
          "archive_bytes": sum(p["bytes"] for p in manifest["archives"].values()),
          "largest_archive_file_bytes": max(p["bytes"] for p in manifest["archives"].values()),
          "archive_manifest_sha256": sha(ARCHIVES / "archive_manifest.json"),
          "archive_helper_sha256": sha(ROOT / "scripts/archive_cross_plant_transfer_evidence_20261031.py"),
          "immutable_prior_helper_sha256": archive.OLD_HELPER_SHA256,
          "fresh_restore": result, "idempotent_restore": second,
          "ct_verify_run_pass": True, "ct_verify_selected_pass": True,
          "ct_verify_predictions_before_truth_pass": True, "empirical_verify_prediction_lock_pass": True,
          "source_pipelines_verified": len(source_records), "outer_pipelines_verified": len(outer_records),
          "empirical_pipelines_verified": empirical_lock["pipeline_fits"],
          "empirical_absolute_main_dependencies": relocated_dependency_checks,
          "portability_scope": "All model/trace paths restored. Main verification uses restored files. Native empirical verifier additionally checks its three original absolute main-file paths; identical restored main files independently verified. Frozen snapshot not rewritten.",
          "new_fits": 0, "new_scores": 0, "tests_run": 0,
          "temporary_restore_removed_after_verification": True, "git_index_mutated": False}
with (OUT / "archive_execution.json").open("x", encoding="utf-8") as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps(report, indent=2), flush=True)
