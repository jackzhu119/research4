"""One-shot root-authorized packaging; no training or score recomputation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone

from herb_spectra.public_thz_benchmark import verify_run, verify_selected, verify_outer_release

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "reports/public-thz-20261031-v1/ginseng-benchmark"
OUT = ROOT / "reports/public-thz-20261031-v1/delivery-preflight"
SCRIPT = ROOT / "scripts/archive_public_thz_evidence_20261031.py"
spec = importlib.util.spec_from_file_location("archive_delivery", SCRIPT)
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

ENTRIES = ("selection_models", "evaluation_models", "transform_cache", "fit_scope_events.jsonl")
ARCHIVES = RUN / "artifact-archives-v1"
started = datetime.now(timezone.utc).isoformat()
review_paths = [ROOT / "reports/public-thz-20261031-v1/independent-audit" / name
                for name in ["selection_review.json", "evaluation_review.json"]]
reviews = {str(p.relative_to(ROOT)): sha(p) for p in review_paths}
before = {name: sha(RUN / name) for name in ["run_manifest.json", "selection_lock.json",
          "prediction_lock.json", "scoring_manifest.json", "fit_scope_events.jsonl"]}
verify_selected(RUN)
manifest = archive.create_archive(RUN, ARCHIVES, entries=ENTRIES)
with tempfile.TemporaryDirectory(prefix="public-thz-ginseng-restore-") as temp:
    restored = Path(temp) / "ginseng-benchmark"
    result = archive.restore_archive(ARCHIVES, restored)
    second = archive.restore_archive(ARCHIVES, restored)
    for dirname in ["inputs", "scoring_truth"]:
        shutil.copytree(RUN / dirname, restored / dirname)
    fixed_files = ["protocol.json", "preparation.json", "split_manifest.json", "run_manifest.json",
                   "source_release_snapshot.json", "selection_lock.json", "inner_records.json",
                   "inner_predictions.csv.gz", "candidate_summaries.json", "prediction_lock.json",
                   "outer_model_records.json", "outer_role_aliases.json", "outer_release_snapshot.json",
                   "outer_predictions.csv.gz", "pooled_metrics_by_seed.json", "scoring_manifest.json"]
    for name in fixed_files:
        shutil.copyfile(RUN / name, restored / name)
    verify_run(restored)
    _, _, lock = verify_selected(restored)
    verify_outer_release(restored, json.loads((restored / "outer_release_snapshot.json").read_text()))
    checked_models = 0
    for record in json.loads((restored / "outer_model_records.json").read_text()):
        for path, expected in [(record["estimator_path"], record["estimator_sha256"]),
                               (str(Path(record["estimator_path"]).with_suffix(".json")), record["metadata_sha256"]),
                               (record["transform"]["path"], record["transform"]["sha256"])]:
            if sha(restored / path) != expected:
                raise RuntimeError("Restored outer model hash mismatch")
        checked_models += 1
    restored_fitlog_sha = sha(restored / "fit_scope_events.jsonl")
after = {name: sha(RUN / name) for name in before}
if before != after or restored_fitlog_sha != before["fit_scope_events.jsonl"]:
    raise RuntimeError("Original frozen evidence changed during packaging")
report = {"started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
          "status": "LOSSLESS_ARCHIVE_AND_FRESH_RESTORE_PASS", "helper_sha256": sha(SCRIPT),
          "root_go_received": True, "prior_independent_reviews_sha256": reviews,
          "original_locks_before_and_after": before, "originals_unchanged": True,
          "archive_manifest_sha256": sha(ARCHIVES / "archive_manifest.json"),
          "archived_entries": list(ENTRIES), "source_files": manifest["source_files"],
          "source_bytes": manifest["source_bytes"], "archive_files": len(manifest["archives"]),
          "archive_bytes": sum(v["bytes"] for v in manifest["archives"].values()),
          "max_archive_file_bytes": max(v["bytes"] for v in manifest["archives"].values()),
          "restore_result": result, "idempotent_restore_result": second,
          "restored_verify_run_pass": True, "restored_verify_selected_pass": True,
          "restored_source_classifier_models": lock["source_classifier_models"],
          "restored_outer_model_files_verified": checked_models,
          "new_model_fits": 0, "new_scores": 0, "unknown_license_data_packaged": False,
          "temporary_restore_removed_after_validation": True}
with (OUT / "archive_execution.json").open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
    f.write("\n")
print(json.dumps(report, indent=2), flush=True)
