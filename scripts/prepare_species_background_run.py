"""Freeze a CPU-bounded extension using only the established training pool.

This prepares files and job commands; it never launches training by itself.
The previously scored calibration/prediction pools are not exported to workers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data/research/species_revision_20261108"
BASE = ROOT / "data/research/species_background_20261108"
PROFILES = ("p12_e96_full", "p12_e192_full", "p12_e384_full",
            "p24_e96_full", "p24_e192_full", "p24_e384_full",
            "p12_e192_frozen", "p24_e192_frozen")
TARGETS = ("suanzaoren", "huangqi", "tianma")
TASKS = ("acid_origins", "huangqi_origins", "tianma_origins", "binary_kinds", "four_kinds")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def freeze_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen preparation differs: {path}")
    else:
        json_write(path, value)


def prepare_inputs():
    BASE.mkdir(parents=True, exist_ok=True)
    split_path = OLD / "split_manifest.csv"
    feature_path = OLD / "features.npz"
    protocol = json.loads((OLD / "protocol.json").read_text())
    audit = json.loads((OLD / "feature_audit.json").read_text())
    if digest(split_path) != protocol["manifest_sha256"] or digest(feature_path) != audit["features_sha256"]:
        raise ValueError("Original frozen feature/split bytes have changed")
    with split_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    training = sorted((row for row in rows if row["role"] == "train"), key=lambda r: r["record_id"])
    allowlist = [row["record_id"] for row in training]
    forbidden = {r["record_id"] for r in rows if r["role"] != "train"}
    if len(training) != 999 or len(set(allowlist)) != 999 or forbidden.intersection(allowlist):
        raise ValueError("Expected exact disjoint 999-record historical training pool")
    if len({r["physical_sample_id"] for r in training}) != len(training):
        raise ValueError("Training physical samples must be unique")
    feature_output = BASE / "train_only_features.npz"
    split_output = BASE / "train_only_split.csv"
    if not feature_output.exists():
        with np.load(feature_path, allow_pickle=False) as original:
            identifiers = original["record_ids"].astype(str)
            lookup = {rid: i for i, rid in enumerate(identifiers)}
            indices = np.asarray([lookup[rid] for rid in allowlist])
            arrays = {key: original[key][indices] if original[key].ndim and original[key].shape[0] == len(identifiers)
                      else original[key] for key in original.files}
        if arrays["record_ids"].astype(str).tolist() != allowlist:
            raise ValueError("Training-only cache order differs")
        np.savez_compressed(feature_output, **arrays)
    with np.load(feature_output, allow_pickle=False) as check:
        if check["record_ids"].astype(str).tolist() != allowlist:
            raise ValueError("Existing training-only cache identity drift")
    if not split_output.exists():
        with split_output.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(training)
    with split_output.open(encoding="utf-8-sig", newline="") as stream:
        if list(csv.DictReader(stream)) != training:
            raise ValueError("Training-only metadata differs from exact master subset")
    lock = {"master_split_sha256": digest(split_path), "master_features_sha256": digest(feature_path),
            "training_feature_sha256": digest(feature_output), "training_split_sha256": digest(split_output),
            "training_records": len(training), "training_allowlist": allowlist,
            "excluded_master_calibration_records": 334, "excluded_master_prediction_records": 339,
            "policy": "Only master TRAIN rows exported. No final calibration/prediction spectra or truth in worker snapshots."}
    freeze_json(BASE / "input_lock.json", lock)
    return feature_output, split_output, lock


def prepare_jobs(features, split, input_lock):
    workers = {"classical": ROOT / "scripts/species_revision_background_classical.py",
               "neural": ROOT / "scripts/species_revision_background_neural.py",
               "summary": ROOT / "scripts/summarize_species_background.py"}
    supervisor = ROOT / "scripts/species_background_budget.py"
    for path in [*workers.values(), supervisor]:
        if not path.is_file():
            raise ValueError(f"Worker not ready: {path}")
    sources = sorted((ROOT / "src/herb_spectra").glob("*.py"))
    shared_inputs = [str(features), str(split), str(BASE / "input_lock.json"),
                     *map(str, sources)]
    classical = []
    for repeat in (20261031, 20261108, 42):
        for representation in ("alpha", "n", "n_alpha"):
            for fold in range(5):
                for task in TASKS:
                    identifier = f"classical_{task}_{representation}_r{repeat}_f{fold}"
                    out = BASE / "runs" / identifier
                    classical.append({"id": identifier, "kind": "classical",
                        "command": [sys.executable, str(workers["classical"]), "--features", str(features),
                                    "--split", str(split), "--task", task, "--representation", representation,
                                    "--outer-fold", str(fold), "--repeat-seed", str(repeat), "--output", str(out)],
                        "cwd": str(ROOT), "inputs": [*shared_inputs, str(workers["classical"])],
                        "outputs": [str(out / name) for name in ("complete.json", "summary.json", "validation_predictions.csv.gz")],
                        "max_retries": 1})
    neural = []
    for steps in (120, 480, 1920):
        for profile in PROFILES:
            for fold in range(5):
                for target in TARGETS:
                    for seed in (42, 43, 44):
                        identifier = f"neural_{target}_{profile}_pre{steps}_f{fold}_s{seed}"
                        out = BASE / "runs" / identifier
                        neural.append({"id": identifier, "kind": "neural",
                            "command": [sys.executable, str(workers["neural"]), "--features", str(features),
                                        "--split", str(split), "--target", target, "--profile", profile,
                                        "--outer-fold", str(fold), "--seed", str(seed), "--pretrain-steps", str(steps),
                                        "--cache-root", str(BASE / "cache"), "--output", str(out)],
                            "cwd": str(ROOT), "inputs": [*shared_inputs, str(workers["neural"])],
                            "outputs": [str(out / name) for name in ("complete.json", "summary.json", "validation_predictions.csv.gz")],
                            "max_retries": 1})
    # Start both inherited algorithm families promptly, then finish baseline coverage
    # before increasing profiles, repetitions and pretraining computation.
    first = [classical[0], neural[0]]
    used = {j["id"] for j in first}
    baseline_classical = [j for j in classical if "_r20261031_" in j["id"]]
    baseline_neural = [j for j in neural if "_p12_e96_full_pre120_" in j["id"]]
    ordered = [*first]
    for group in (baseline_classical, baseline_neural, classical, neural):
        for job in group:
            if job["id"] not in used:
                ordered.append(job)
                used.add(job["id"])
    if len(ordered) != 1305:
        raise AssertionError("Unexpected frozen experiment count")
    return ordered, workers, supervisor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-only", action="store_true")
    args = parser.parse_args()
    features, split, lock = prepare_inputs()
    if args.inputs_only:
        print(json.dumps({"training_records": lock["training_records"], "features": str(features), "split": str(split)}, ensure_ascii=False))
        return
    jobs, workers, supervisor = prepare_jobs(features, split, lock)
    protocol = {"run_id": "species-background-20261108-v1", "cpu_budget_seconds": 90000,
        "authorization": "User: 在后台运行吗，我授权 90000 秒 cpu 时，不要中断在后台自行运行",
        "budget_unit": "aggregate CPU core-seconds; 90000 seconds = 25 CPU core-hours, not a wall-time guarantee",
        "stop_rule": "Finish all predeclared jobs or stop at the CPU budget; no artificial work to consume remaining budget.",
        "master_roles_unchanged": True, "training_records": 999, "training_allowlist": lock["training_allowlist"],
        "input_lock_sha256": digest(BASE / "input_lock.json"),
        "expected_fold_ids": [0, 1, 2, 3, 4], "inner_folds": 2,
        "classical": {"jobs": 225, "tasks": list(TASKS), "representations": ["alpha", "n", "n_alpha"],
                      "partition_repeat_seeds": [20261031, 20261108, 42], "model_seed": 42,
                      "methods": ["historical_R0", "historical_R2", "full_SVM", "PLS_DA"]},
        "neural": {"jobs": 1080, "targets": list(TARGETS), "profiles": list(PROFILES),
                   "pretraining_steps": [120, 480, 1920], "model_seeds": [42, 43, 44], "partition_seed": 20261031,
                   "arms": ["A", "Aextra", "B", "matched_SVM"],
                   "support": "10 per class reserved inside each training-only outer fold; K1/3/5/10 nested draws",
                   "auxiliary_data": "Other two herb kinds, master TRAIN only; independent origin-group heads",
                   "cache": "Reuse identical A and matched SVM fits across pretraining budgets, with hash-checked checkpoints"},
        "selection_policy": "Only complete training-internal cross-validation groups may inform recommendations. No adaptive use of master calibration/prediction.",
        "inference_limits": ["historical training records already exposed in earlier development", "no new batches or instruments", "development CV is not external confirmation", "seeds and support repeats are not additional independent samples"],
        "planned_experiment_jobs": len(jobs), "planned_snapshot_jobs": 13,
        "automatic_network_publication": False,
        "runtime_versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "torch", "joblib")},
        "worker_code_sha256": {str(path.relative_to(ROOT)): digest(path) for path in [*workers.values(), supervisor]}}
    freeze_json(BASE / "protocol.json", protocol)
    # Periodic immutable snapshots retain a partial readout even when the budget
    # is reached before the final summarization. Each has its own output path.
    def summary_job(identifier, destination):
        return {"id": identifier, "kind": "summary",
                "command": [sys.executable, str(workers["summary"]), "--root", str(BASE), "--output", str(destination)],
                "cwd": str(ROOT), "inputs": [str(BASE / "protocol.json"), str(BASE / "input_lock.json"),
                                                  str(BASE / "jobs.json"), str(workers["summary"])],
                "outputs": [str(destination / "results_summary.json"), str(destination / "BACKGROUND_RESULTS.md")],
                "max_retries": 1}

    queue = []
    snapshot_points = {120, *range(200, 1306, 100)}
    for count, job in enumerate(jobs, 1):
        queue.append(job)
        if count in snapshot_points:
            queue.append(summary_job(f"summarize_snapshot_{count:05d}", BASE / "snapshots" / f"{count:05d}"))
    queue.append(summary_job("summarize_training_only_development", BASE / "results"))
    if len(queue) != 1319:
        raise AssertionError("Expected 1305 experiments, 13 snapshots and one final readout")
    manifest = {"run_id": protocol["run_id"], "cpu_budget_seconds": 90000, "jobs": jobs,
                "protocol_path": str(BASE / "protocol.json"), "protocol_sha256": digest(BASE / "protocol.json")}
    manifest["jobs"] = queue
    freeze_json(BASE / "jobs.json", manifest)
    print(json.dumps({"jobs": len(queue), "experiment_jobs": len(jobs), "cpu_budget_seconds": 90000,
                      "manifest": str(BASE / "jobs.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
