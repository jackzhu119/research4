"""Prepare a NEW finite CPU grant without resetting any earlier runtime.

No model is fitted or inferred here. The only numerical preparation exports
the existing master TRAIN subset; workers never receive master CAL/PRED rows.
The scientific workers, candidate grids and master roles remain unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "scripts/prepare_species_background_run.py"
ORIGINAL_SHA = "a694ad839b94dbdb308db0a8ad0ce3ae49e582a5a58e84aac65634169a55392e"
OLD = ROOT / "data/research/species_revision_20261108"
BASE = ROOT / "data/research/species_relaunch_20261108_v2"
RUN_ID = "species-relaunch-20261108-v2"
AUTHORIZED_TOTAL = 90000
INNER_CAP = 88000
PREPARATION_RESERVE = 1900
WATCH_RESERVE = 100


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(p.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(p)


def freeze(path, value):
    p = Path(path)
    if p.exists():
        require(read_json(p) == value, "Existing relaunch preparation differs: " + str(p))
    else:
        atomic_json(p, value)


def original_preparer():
    require(digest(ORIGINAL) == ORIGINAL_SHA, "Original preparation source changed")
    spec = importlib.util.spec_from_file_location("species_relaunch_original_preparer", ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT, module.OLD, module.BASE = ROOT, OLD, BASE
    return module


def validate_authorization(record):
    amounts = [record[k] for k in ("authorized_cpu_seconds", "authorized_total_cpu_seconds") if k in record]
    require(amounts and all(type(value) is int and value == AUTHORIZED_TOTAL for value in amounts),
            "This NEW grant must explicitly authorize 90000 CPU core-seconds")
    require(INNER_CAP + PREPARATION_RESERVE + WATCH_RESERVE == AUTHORIZED_TOTAL,
            "All finite allocations must fit inside the authorized total")


def fingerprint(array):
    value = np.ascontiguousarray(array)
    require(value.dtype.kind != "O", "Object arrays are not admissible spectral evidence")
    return {"shape": list(value.shape), "dtype": value.dtype.str,
            "payload_sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def select_training(rows):
    require(len(rows) == 1672 and len({r["record_id"] for r in rows}) == 1672,
            "Exact canonical 1672-record master split required")
    roles = {name: [r for r in rows if r["role"] == name] for name in ("train", "calibration", "prediction")}
    require({name: len(value) for name, value in roles.items()} ==
            {"train": 999, "calibration": 334, "prediction": 339}, "Frozen master role counts changed")
    training = sorted(roles["train"], key=lambda row: row["record_id"])
    require(len({r["physical_sample_id"] for r in training}) == 999
            and all(r["physical_sample_id"] for r in training), "Independent TRAIN physical IDs required")
    counts = {kind: sum(row["species"] == kind for row in training)
              for kind in ("枣仁", "酸枣仁", "黄芪", "天麻")}
    require(counts == {"枣仁": 216, "酸枣仁": 300, "黄芪": 241, "天麻": 242},
            "Corrected kind counts in TRAIN changed")
    source_ids = {r["record_id"] for r in training}
    require(not source_ids & {r["record_id"] for name in ("calibration", "prediction") for r in roles[name]},
            "Master holdout IDs entered the source pool")
    return training


def prepare_inputs(features):
    """Reconstruct input provenance honestly; never change the old audit."""
    features = Path(features).resolve()
    split_path = OLD / "split_manifest.csv"
    old_protocol = read_json(OLD / "protocol.json")
    old_audit = read_json(OLD / "feature_audit.json")
    require(digest(split_path) == old_protocol["manifest_sha256"], "Master split bytes changed")
    with split_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    training = select_training(rows)
    registry_path = ROOT / "data/active_workbooks.json"
    registry = read_json(registry_path)
    require(registry["version"] == "2026-11-08-confirmed-origin-kind-v1"
            and registry["total_active_records"] == 1672 and len(registry["workbooks"]) == 18,
            "Current user-confirmed 18-workbook registry required")
    audited = {row["path"]: row["sha256"] for row in old_audit["workbook_checks"]}
    raw_records = []
    for row in registry["workbooks"]:
        raw_path = ROOT / row["path"]
        require(audited.get(row["path"]) == row["sha256"] == digest(raw_path),
                "Workbook bytes differ from both canonical registry and prior extraction audit")
        raw_records.append({"path": row["path"], "sha256": row["sha256"],
                            "species": row["species"], "record_count": row["record_count"]})
    with np.load(features, allow_pickle=False) as original:
        identifiers = original["record_ids"].astype(str)
        require(len(identifiers) == 1672 and len(set(identifiers)) == 1672
                and set(identifiers) == {r["record_id"] for r in rows}, "Feature/master identities differ")
        lookup = {rid: i for i, rid in enumerate(identifiers)}
        indices = np.asarray([lookup[row["record_id"]] for row in training])
        original_fingerprints = {}
        arrays = {}
        for key in original.files:
            value = original[key]
            require(value.dtype.kind != "O" and (value.dtype.kind not in "fc" or np.isfinite(value).all()),
                    "Finite non-object input arrays required")
            original_fingerprints[key] = fingerprint(value)
            arrays[key] = value[indices].copy() if value.ndim and value.shape[0] == len(identifiers) else value.copy()
    require(arrays["record_ids"].astype(str).tolist() == [row["record_id"] for row in training],
            "Training pack order differs")
    feature_output, split_output = BASE / "train_only_features.npz", BASE / "train_only_split.csv"
    BASE.mkdir(parents=True, exist_ok=True)
    if feature_output.exists():
        with np.load(feature_output, allow_pickle=False) as existing:
            require(set(existing.files) == set(arrays) and
                    all(fingerprint(existing[key]) == fingerprint(value) for key, value in arrays.items()),
                    "Existing TRAIN-only cache drift")
    else:
        np.savez_compressed(feature_output, **arrays)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(training)
    split_bytes = ("\ufeff" + stream.getvalue()).encode("utf-8")
    if split_output.exists():
        require(split_output.read_bytes() == split_bytes, "Existing TRAIN-only metadata differs")
    else:
        split_output.write_bytes(split_bytes)
    feature_sha = digest(features)
    derived = {"run_id": RUN_ID, "scope": "new reconstruction from exact original raw workbooks, no learned transforms",
               "master_features_path": str(features), "master_features_sha256": feature_sha,
               "historical_feature_audit_path": str(OLD / "feature_audit.json"),
               "historical_feature_audit_sha256": digest(OLD / "feature_audit.json"),
               "historical_features_sha256": old_audit["features_sha256"],
               "cache_byte_equivalent_to_historical": feature_sha == old_audit["features_sha256"],
               "old_cache_array_equality": "IMPLIED_BY_EXACT_FILE_SHA" if feature_sha == old_audit["features_sha256"]
                                            else "NOT_VERIFIED_OLD_CACHE_UNAVAILABLE",
               "original_workbooks_sha256_verified": raw_records,
               "registry_sha256": digest(registry_path), "master_split_sha256": digest(split_path),
               "extractor_sha256": digest(ROOT / "scripts/prepare_species_revision_features.py"),
               "new_master_array_fingerprints": original_fingerprints,
               "new_training_array_fingerprints": {k: fingerprint(v) for k, v in arrays.items()},
               "master_roles_reassigned": False, "old_audit_modified": False,
               "all_1672_read_for_packaging_not_fitting": True,
               "worker_export_policy": "Only999TRAIN arrays/metadata exported; CAL334/PRED339 not present in worker input"}
    freeze(BASE / "derived_source_feature_audit.json", derived)
    lock = {"master_split_sha256": digest(split_path), "master_features_sha256": feature_sha,
            "training_feature_sha256": digest(feature_output), "training_split_sha256": digest(split_output),
            "training_records": 999, "training_allowlist": [row["record_id"] for row in training],
            "excluded_master_calibration_records": 334, "excluded_master_prediction_records": 339,
            "derived_source_feature_audit_sha256": digest(BASE / "derived_source_feature_audit.json"),
            "policy": "Only canonical master TRAIN exported. Packaging reads are not model fitting or selection."}
    freeze(BASE / "input_lock.json", lock)
    return feature_output, split_output, lock


def reorder_jobs(jobs):
    """Keep every fixed scientific job, in complete 5-fold or 15-job groups."""
    require(len(jobs) == 1305 and len({job["id"] for job in jobs}) == 1305, "Original finite scientific matrix required")
    classic, neural = [], []
    for job in jobs:
        if job["kind"] == "classical":
            argv = job["command"]
            key = (argv[argv.index("--task") + 1], argv[argv.index("--representation") + 1],
                   int(argv[argv.index("--repeat-seed") + 1]), int(argv[argv.index("--outer-fold") + 1]))
            classic.append((key, job))
        elif job["kind"] == "neural":
            argv = job["command"]
            key = (argv[argv.index("--target") + 1], int(argv[argv.index("--pretrain-steps") + 1]),
                   argv[argv.index("--profile") + 1], int(argv[argv.index("--outer-fold") + 1]),
                   int(argv[argv.index("--seed") + 1]))
            neural.append((key, job))
        else:
            raise ValueError("Unexpected scientific job kind")
    require(len(classic) == 225 and len(neural) == 1080, "Scientific family counts changed")
    tasks = {name: i for i, name in enumerate(("acid_origins", "huangqi_origins", "tianma_origins", "binary_kinds", "four_kinds"))}
    representations = {name: i for i, name in enumerate(("alpha", "n", "n_alpha"))}
    repeats = {value: i for i, value in enumerate((20261031, 20261108, 42))}
    targets = {name: i for i, name in enumerate(("suanzaoren", "huangqi", "tianma"))}
    profiles = {name: i for i, name in enumerate(("p12_e96_full", "p12_e192_full", "p12_e384_full",
                                               "p24_e96_full", "p24_e192_full", "p24_e384_full",
                                               "p12_e192_frozen", "p24_e192_frozen"))}
    expected_classic = {(task, representation, repeat, fold) for task in tasks
                        for representation in representations for repeat in repeats for fold in range(5)}
    expected_neural = {(target, steps, profile, fold, seed) for target in targets
                       for steps in (120, 480, 1920) for profile in profiles
                       for fold in range(5) for seed in (42, 43, 44)}
    require({key for key, _ in classic} == expected_classic
            and {key for key, _ in neural} == expected_neural,
            "Exact fixed scientific parameter, fold and seed matrix required")
    classic.sort(key=lambda row: (tasks[row[0][0]], repeats[row[0][2]], representations[row[0][1]], row[0][3]))
    neural.sort(key=lambda row: (targets[row[0][0]], row[0][1], profiles[row[0][2]], row[0][3], row[0][4]))
    # A complete acid baseline fold group and a complete neural seed/fold group
    # run first. This detects either runtime's failures promptly without changing
    # the predeclared matrix or selecting subsequent work from observed scores.
    initial_classic = classic[:5]
    initial_neural = neural[:15]
    return ([job for _, job in initial_classic] + [job for _, job in initial_neural]
            + [job for _, job in classic[5:]] + [job for _, job in neural[15:]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=OLD / "features.npz")
    parser.add_argument("--authorization", type=Path, default=BASE / "user_authorization_NEW_90000.json")
    parser.add_argument("--inputs-only", action="store_true")
    args = parser.parse_args()
    authorization = args.authorization.resolve()
    validate_authorization(read_json(authorization))
    features, split, lock = prepare_inputs(args.features)
    if args.inputs_only:
        print(json.dumps({"status": "TRAIN_ONLY_PACK_PREPARED_NO_MODELS_FITTED", "records": 999,
                          "features": str(features), "split": str(split)})); return
    original = original_preparer()
    original_jobs, workers, supervisor = original.prepare_jobs(features, split, lock)
    workers["backup"] = ROOT / "scripts/species_relaunch_backup_20261108.py"
    require(workers["backup"].is_file(), "Reviewed bounded result-backup worker is required")
    ordered = reorder_jobs(original_jobs)
    source_protocol = read_json(ROOT / "data/research/species_background_20261108/protocol.json")
    protocol = {**source_protocol, "run_id": RUN_ID, "cpu_budget_seconds": INNER_CAP,
                "authorized_total_cpu_seconds": AUTHORIZED_TOTAL,
                "preparation_reserved_cpu_seconds": PREPARATION_RESERVE,
                "watchdog_reserved_cpu_seconds": WATCH_RESERVE,
                "authorization": "NEW user authorization: 重新发布进程…授权你90000秒cpu时在后台自动执行",
                "authorization_record": {"path": str(authorization), "sha256": digest(authorization)},
                "prior_grants_status": "UNRECONCILED_NO_REFUND_NO_CARRY",
                "old_runtime_restored": False,
                "replication_scope": "NEW independently budgeted replication of the declared TRAIN-only matrix; no claim to recover unknown old-host runtime.",
                "original_final_holdout_experiments_repeated": False,
                "training_allowlist": lock["training_allowlist"], "input_lock_sha256": digest(BASE / "input_lock.json"),
                "derived_source_feature_audit_sha256": digest(BASE / "derived_source_feature_audit.json"),
                "runtime_versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "torch", "joblib")},
                "worker_code_sha256": {str(path.relative_to(ROOT)): digest(path) for path in [*workers.values(), supervisor]},
                "new_preparer_sha256": digest(__file__), "unchanged_original_preparer_sha256": ORIGINAL_SHA,
                "queue_priority": "First5 acid alpha classical folds; next15 acid pre120/p12_e96 full seeded folds; then remaining acid classical, other classical, and remaining acid/other neural complete groups. No score-adaptive scheduling.",
                "stop_rule": "Finish finite matrix or cap. No artificial work to burn leftover allocation. Failed/partial/unstarted groups remain visible.",
                "automatic_network_publication": "Reviewed filtered Git result backups after immutable summaries; never main; no raw, live, unsealed or oversized cache publication; small sealed TRAIN-only model artifacts allowed"}
    snapshot_points = {5, 10, *range(25, len(ordered) + 1, 25)}
    protocol["planned_snapshot_jobs"] = len(snapshot_points)
    protocol["planned_backup_jobs"] = len(snapshot_points) + 1
    protocol["planned_experiment_jobs"] = len(ordered)
    protocol_path = BASE / "protocol.json"
    freeze(protocol_path, protocol)
    queue = []
    def summary(identifier, destination):
        return {"id": identifier, "kind": "summary",
                "command": [sys.executable, str(workers["summary"]), "--root", str(BASE), "--output", str(destination)],
                "cwd": str(ROOT), "inputs": [str(protocol_path), str(BASE / "input_lock.json"), str(BASE / "jobs.json"), str(workers["summary"])],
                "outputs": [str(destination / "results_summary.json"), str(destination / "BACKGROUND_RESULTS.md")], "max_retries": 1}
    def backup(identifier, destination):
        return {"id": identifier, "kind": "backup",
                "command": [sys.executable, str(workers["backup"]), "--root", str(BASE),
                            "--output", str(destination), "--execute"],
                "cwd": str(ROOT),
                "inputs": [str(workers["backup"]), str(protocol_path), str(BASE / "input_lock.json"), str(BASE / "jobs.json")],
                "outputs": [str(destination / "receipt.json"), str(destination / "complete.json")],
                "max_retries": 0}
    for index, job in enumerate(ordered, 1):
        queue.append(job)
        if index in snapshot_points:
            queue.append(summary("new_grant_snapshot_%05d" % index, BASE / "snapshots" / ("%05d" % index)))
            queue.append(backup("new_grant_backup_%05d" % index, BASE / "backups" / ("%05d" % index)))
    queue.append(summary("new_grant_final_summary", BASE / "results"))
    queue.append(backup("new_grant_final_backup", BASE / "backups" / "final"))
    manifest = {"run_id": RUN_ID, "cpu_budget_seconds": INNER_CAP,
                "authorized_total_cpu_seconds": AUTHORIZED_TOTAL,
                "authorization_record": {"path": str(authorization), "sha256": digest(authorization)},
                "prior_grants_status": "UNRECONCILED_NO_REFUND_NO_CARRY",
                "protocol_path": str(protocol_path), "protocol_sha256": digest(protocol_path), "jobs": queue}
    freeze(BASE / "jobs.json", manifest)
    print(json.dumps({"status": "NEW_GRANT_QUEUE_PREPARED_NOT_LAUNCHED", "jobs": len(queue),
                      "experiments": len(ordered), "snapshots": len(snapshot_points),
                      "backups": len(snapshot_points) + 1,
                      "inner_cap": INNER_CAP, "authorized_total": AUTHORIZED_TOTAL,
                      "manifest": str(BASE / "jobs.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
