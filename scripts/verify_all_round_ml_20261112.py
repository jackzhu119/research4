#!/usr/bin/env python3
"""Recheck fixed saved ML evidence in disposable copies; never train a model.

Every historical validator writes only inside its disposable snapshot. Durable
receipts and logs go to reports/all_round_validation_20261112/ml. COMPLETE means
the listed executable checks passed, not independent-batch or paper qualification.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import resource
import subprocess
import sys
import tarfile
import tempfile
import traceback
from datetime import datetime, timezone

B_COMMIT = "647f069c1ed4d56b098689571f43074b581a0cc0"
BUNDLE_COMMIT = "b051733989161517a68c4a60cce75766880c95f5"
META_COMMIT = "2d5062d79d760682eaeaf61f02f56d78a02f29ce"
BACKFILL_COMMIT = "f38113e51072f0a99c864b2e6dae79a9169ff93c"
ROUNDS = (
    ("b1", "b_mechanism_20261110", "gap_20261110", 105),
    ("b2", "b_mechanism_followup_20261110", "followup_20261110", 345),
    ("b3", "b_mechanism_round3_20261110", "round3_20261110", 375),
    ("b4", "b_mechanism_round4_20261111", "round4_20261111", 375),
    ("b5", "b_mechanism_round5_20261111", "round5_20261111", 180),
)
REL = Path("reports/all_round_validation_20261112/ml")
LIMITATIONS = [
    "All B comparisons remain exploratory internal TRAIN-only evidence on previously exposed specimens.",
    "Real new-batch evaluation remains NOT_RUN_NEEDS_NEW_BATCH_DATA; CPU work cannot supply missing batches.",
    "The portable paper bundle checks saved hashes, counts and confusion matrices, not every original model replay.",
    "Raw saved-prediction replay validates prior metrics; it does not rerun all trained models or create independent evidence.",
    "Continuous meta-learning is not a finished training round: only its initial 45-checkpoint backup is verified here.",
    "No molecular IR calculation or chemical peak assignment is performed by this ML verifier.",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".verification-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def git_bytes(root, commit, path):
    return subprocess.check_output(["git", "-C", str(root), "show", commit + ":" + path])


def materialize(root, commit, paths, target):
    """Stream only named immutable Git subtrees, rejecting links and unsafe paths."""
    command = ["git", "-C", str(root), "archive", "--format=tar", commit, "--", *paths]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    count, size = 0, 0
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Unsafe archive member: " + member.name)
                path = target.joinpath(*relative.parts)
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise ValueError("Unexpected nonregular Git member: " + member.name)
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, path.open("wb") as destination:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        destination.write(block)
                count += 1
                size += member.size
        process.stdout.close()
        error = process.stderr.read().decode("utf-8", errors="replace")
        code = process.wait()
        if code:
            raise RuntimeError("git archive failed: " + error)
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        process.stderr.close()
    return {"commit": commit, "materialized_files": count, "materialized_bytes": size}


def run_validator(command, log, cwd):
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    # Historic verifiers use assert for numerical certificates; keep them active.
    env.pop("PYTHONOPTIMIZE", None)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Validator exit {result.returncode}; see {log}")


def check_roles(output, manifest, expected):
    """Check the frozen sample identity contract as well as the metric receipts."""
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    role_field = next((name for name in ("role", "split", "assignment") if name in rows[0]), None)
    if role_field is None:
        raise ValueError("Cannot identify master TRAIN assignment field")
    allowed = {row["record_id"] for row in rows if row[role_field].lower() in ("train", "training")}
    if len(allowed) != 999:
        raise ValueError("Master TRAIN whitelist must contain 999 distinct records")
    protocol = json.loads((output / "frozen_protocol.json").read_text())
    design = next((protocol[name] for name in ("design", "jobs", "tasks")
                   if isinstance(protocol.get(name), list)), None)
    if design is None or len(design) != expected:
        raise ValueError("Frozen design count differs from expected completed matrix")
    planned = {job["job_id"] for job in design}
    if len(planned) != expected:
        raise ValueError("Duplicate frozen task identity")
    completed = {path.parent.name for path in (output / "jobs").glob("*/COMPLETE.json")}
    if completed != planned:
        raise ValueError("Completed task identities differ from the complete frozen design")
    checked = 0
    for path in (output / "jobs").glob("*/result.json"):
        result = json.loads(path.read_text())
        job = {**result.get("job", {}), **result}
        roles = [job.get(name) for name in ("source_ids", "support_ids", "query_ids")]
        if any(not isinstance(part, list) for part in roles):
            raise ValueError("Missing frozen role IDs in " + str(path))
        sets = [set(part) for part in roles]
        if any(len(part) != len(unique) for part, unique in zip(roles, sets)):
            raise ValueError("Duplicate specimen identity within role")
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError("Source/Support/Query overlap")
        if not set.union(*sets).issubset(allowed):
            raise ValueError("Historical CAL/PRED entered this TRAIN-only task")
        checked += 1
    if checked != expected:
        raise ValueError("Result count differs from frozen design")
    return {"master_train_records": 999, "role_disjointness_tasks_checked": checked,
            "all_roles_in_master_train": True, "planned_and_completed_identity_sets_equal": True}


def verify_round(root, durable, stage, item):
    name, folder, slug, expected = item
    script = "scripts/summarize_b_mechanism_" + slug + ".py"
    paths = ["reports/" + folder, script,
             "scripts/summarize_b_mechanism_followup_20261110.py",
             "data/research/species_revision_20261108/split_manifest.csv"]
    snapshot = stage / name
    snapshot.mkdir()
    materialized = materialize(root, B_COMMIT, sorted(set(paths)), snapshot)
    output = snapshot / "reports" / folder
    roles = check_roles(output, snapshot / paths[-1], expected)
    old_receipt = json.loads((output / "summary_verification.json").read_text())
    run_validator([sys.executable, str(snapshot / script), "--output", str(output), "--verify"],
                  durable / (name + ".log"), snapshot)
    fresh = json.loads((output / "summary_verification.json").read_text())
    if fresh.get("status") != "DONE" or fresh.get("sealed_jobs") != expected:
        raise ValueError("Independent verifier did not cover the complete declared matrix")
    for field in ("metric_rows", "checks"):
        if fresh.get(field) != old_receipt.get(field):
            raise ValueError("Independent check coverage changed: " + field)
    source = snapshot / script
    receipt = {"status": "COMPLETE", "component": name, "source_commit": B_COMMIT,
               "historical_validator": script, "historical_validator_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "historical_artifact_hashes_and_metric_recomputation": fresh,
               "sample_identity_contract": roles, "materialization": materialized,
               "no_model_fitted": True, "historical_outputs_modified": False,
               "scientific_scope": "Exploratory internal validation; new batches NOT_RUN_NEEDS_NEW_BATCH_DATA",
               "verified_at_utc": now()}
    write_json(durable / (name + ".json"), receipt)
    return receipt


def verify_bundle(root, durable, stage):
    manifest_path = "reports/paper-evidence-20261109/bundle_manifest.json"
    manifest = json.loads(git_bytes(root, BUNDLE_COMMIT, manifest_path))
    paths = [entry["path"] for entry in manifest["files"]] + [manifest_path]
    snapshot = stage / "paper_bundle"
    snapshot.mkdir()
    materialized = materialize(root, BUNDLE_COMMIT, paths, snapshot)
    script = snapshot / "scripts/build_paper_evidence_bundle_20261109.py"
    run_validator([sys.executable, str(script), "--repo", str(snapshot), "--verify"],
                  durable / "paper_bundle.log", snapshot)
    fresh = json.loads((snapshot / "reports/paper-evidence-20261109/bundle_verification.json").read_text())
    if (fresh.get("status") != "PASS" or fresh.get("hashed_files") != 213
            or fresh.get("scientific_jobs") != 1635
            or fresh.get("confusion_matrices_recomputed") != 318
            or fresh.get("calibration_support_average_counts_checked") != 48):
        raise ValueError("Portable evidence bundle verification coverage differs")
    receipt = {"status": "COMPLETE", "source_commit": BUNDLE_COMMIT,
               "historical_validator_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
               "historical_verification": fresh, "materialization": materialized,
               "no_model_fitted": True, "historical_outputs_modified": False,
               "scope": "Saved evidence/transport and confusion matrices, not complete raw-model replay or external confirmation",
               "verified_at_utc": now()}
    write_json(durable / "paper_bundle.json", receipt)
    return receipt


def verify_meta_backup(root, durable, stage):
    snapshot = stage / "meta_backup"
    snapshot.mkdir()
    script_path = "scripts/restore_meta_learning_launch_backup.py"
    backup_path = "data/research/meta_learning_20261111/launch_backup"
    materialized = materialize(root, META_COMMIT, [script_path, backup_path], snapshot)
    restored = snapshot / "restored"
    run_validator([sys.executable, str(snapshot / script_path), "--backup", str(snapshot / backup_path),
                   "--output", str(restored)], durable / "meta_backup.log", snapshot)
    manifest = json.loads((snapshot / backup_path / "manifest.json").read_text())
    checkpoints = list(restored.glob("runs/*/cells/*/checkpoint.json"))
    if len(manifest["files"]) != 51 or len(checkpoints) != 45:
        raise ValueError("Initial meta-learning backup coverage differs")
    receipt = {"status": "COMPLETE", "source_commit": META_COMMIT,
               "backup_files_both_compressed_and_uncompressed_sha256_verified": 51,
               "initial_checkpoints": 45, "materialization": materialized,
               "historical_validator_sha256": hashlib.sha256((snapshot / script_path).read_bytes()).hexdigest(),
               "scope": "Initial recoverable backup only; not completed training or verification of missing live snapshots",
               "training_status": "NOT_CERTIFIED_COMPLETE", "no_model_fitted": True,
               "historical_outputs_modified": False, "verified_at_utc": now()}
    write_json(durable / "meta_backup.json", receipt)
    return receipt


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_species_predictions(root, durable, stage):
    """Restore all 1635 jobs and reunite the 591+524+190 old predictions."""
    snapshot = stage / "species"
    snapshot.mkdir()
    archive_rel = "reports/paper-evidence-20261109/archives"
    materialized = materialize(root, BUNDLE_COMMIT, [archive_rel], snapshot)
    archive = snapshot / archive_rel
    restore = archive / "restore_archive.py"
    small = snapshot / "small"
    completion = snapshot / "completion"
    commands = [
        ("species_small_restore", [sys.executable, str(restore), "--manifest", str(archive / "manifest.json"), "--output", str(small)]),
        ("species_prediction_completion_restore", [sys.executable, str(restore), "--manifest", str(archive / "legacy-prediction-completion/manifest.json"), "--output", str(completion)]),
    ]
    transport = {}
    for label, command in commands:
        log = durable / (label + ".log")
        run_validator(command, log, snapshot)
        transport[label] = json.loads(log.read_text())
        if transport[label].get("status") != "PASS":
            raise ValueError("Archive restoration failed: " + label)
    materialize(root, B_COMMIT,
                ["data/research/species_revision_20261108/split_manifest.csv"], small)

    coverage = json.loads(gzip.decompress((archive / "legacy-prediction-completion/all-1305-prediction-coverage.json.gz").read_bytes()))
    if len(coverage["files"]) != 1305:
        raise ValueError("Legacy prediction coverage is not 1305")
    grouped = {}
    for entry in coverage["files"]:
        source = next((source for source in entry["already_remote"] if source["storage"] == "git_blob"), None)
        if source is not None:
            grouped.setdefault(source["commit"], []).append(source["path"])
    direct = snapshot / "direct"
    direct.mkdir()
    for commit, paths in grouped.items():
        # Each selected path is fixed by the hash-bound historical coverage table.
        materialize(root, commit, paths, direct)
    backfill = snapshot / "backfill_source"
    backfill.mkdir()
    materialize(root, BACKFILL_COMMIT, ["manifest.json", "restore_backfill.py", "archives"], backfill)
    backfill_restored = snapshot / "backfill_restored"
    log = durable / "species_backfill_restore.log"
    run_validator([sys.executable, str(backfill / "restore_backfill.py"), "--manifest", str(backfill / "manifest.json"),
                   "--output", str(backfill_restored)], log, snapshot)
    transport["species_backfill_restore"] = json.loads(log.read_text())
    if transport["species_backfill_restore"].get("status") != "PASS":
        raise ValueError("Prior 190-prediction backfill restoration failed")
    by_job = {}
    for source_root in (completion, direct, backfill_restored):
        for path in source_root.rglob("validation_predictions.csv.gz"):
            if path.parent.name in by_job:
                raise ValueError("Duplicate recovered legacy prediction job: " + path.parent.name)
            by_job[path.parent.name] = path
    if len(by_job) != 1305:
        raise ValueError(f"Recovered legacy prediction jobs: {len(by_job)}, expected 1305")
    unified = snapshot / "unified_legacy_predictions"
    for entry in coverage["files"]:
        job = PurePosixPath(entry["path"]).parent.name
        path = by_job[job]
        if path.stat().st_size != entry["bytes"] or file_sha(path) != entry["sha256"]:
            raise ValueError("Recovered legacy prediction differs from frozen coverage hash: " + job)
        destination = unified / job / "validation_predictions.csv.gz"
        destination.parent.mkdir(parents=True)
        os.link(path, destination)
    helper = root / "scripts/verify_saved_species_predictions_20261112.py"
    command = [sys.executable, str(helper), "--small-root", str(small), "--legacy-prediction-roots", str(unified),
               "--output", str(durable / "species_predictions")]
    run_validator(command, durable / "species_predictions.log", snapshot)
    scientific = json.loads((durable / "species_predictions/summary.json").read_text())
    if scientific.get("status") != "COMPLETE":
        raise ValueError("All-job saved prediction replay did not pass")
    receipt = {"status": "COMPLETE", "source_commit": BUNDLE_COMMIT,
               "backfill_commit": BACKFILL_COMMIT, "materialization": materialized,
               "legacy_prediction_files_sha256_checked": 1305, "transport_verification": transport,
               "scientific_validation": scientific, "helper_sha256": file_sha(helper),
               "no_model_fitted": True, "historical_outputs_modified": False,
               "verified_at_utc": now()}
    write_json(durable / "species_all_predictions.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    root = args.root.resolve()
    durable = root / REL
    durable.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "research4-all-round-ml-verification-v1", "status": "RUNNING",
               "all_requested_checks_passed": False,
               "started_at_utc": now(), "source_commits": [B_COMMIT, BUNDLE_COMMIT, META_COMMIT],
               "verification_scope": "B1-5 full sealed predictions and frozen roles; all 1305 legacy and 330 V4 saved predictions; portable paper bundle; initial meta-learning backup",
               "components": {}, "limitations": LIMITATIONS, "paper_ready_certificate": False,
               "new_batch_evaluation": "NOT_RUN_NEEDS_NEW_BATCH_DATA", "no_model_fitted": True}
    write_json(durable / "summary.json", summary)
    tasks = [(item[0], lambda stage, item=item: verify_round(root, durable, stage, item)) for item in ROUNDS]
    tasks += [("paper_bundle", lambda stage: verify_bundle(root, durable, stage)),
              ("meta_backup", lambda stage: verify_meta_backup(root, durable, stage)),
              ("species_all_predictions", lambda stage: verify_species_predictions(root, durable, stage))]
    for name, action in tasks:
        print(json.dumps({"component": name, "status": "RUNNING", "timestamp_utc": now()}), flush=True)
        try:
            with tempfile.TemporaryDirectory(prefix="research4-ml-verify-") as temporary:
                receipt = action(Path(temporary))
            summary["components"][name] = {"status": "COMPLETE", "receipt": str((durable / (name + ".json")).relative_to(root))}
        except Exception as error:
            receipt = {"status": "FAILED", "component": name, "error_type": type(error).__name__,
                       "error": str(error), "traceback": traceback.format_exc(), "observed_at_utc": now()}
            write_json(durable / (name + ".failure.json"), receipt)
            summary["components"][name] = {"status": "FAILED", "error": str(error)}
        write_json(durable / "summary.json", summary)
        print(json.dumps({"component": name, **summary["components"][name]}, ensure_ascii=False), flush=True)
    summary["status"] = "COMPLETE" if all(value["status"] == "COMPLETE" for value in summary["components"].values()) else "FAILED"
    summary["all_requested_checks_passed"] = summary["status"] == "COMPLETE"
    summary["finished_at_utc"] = now()
    own, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    summary["cpu_seconds"] = own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime
    write_json(durable / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
