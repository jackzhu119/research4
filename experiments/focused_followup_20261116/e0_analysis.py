#!/usr/bin/env python3
"""E0: inspect archived evidence and resample specimens, without fitting models.

The resampling unit is one declared physical specimen. Its complete vector of
predictions over methods and outer repeats is retained on every draw. No score
is used to select repeats, methods, retry policies, or subsequent experiments.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from datetime import datetime, timezone

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / ".cache/focused_followup_20261116"
DEFAULT_REPORT = ROOT / "reports/focused_followup_20261116"
SOURCE_COMMIT = "93a607c8ce206a5bbbff22538233bdd0929bd04d"
ARCHIVE_COMMIT = "7a145aaab9be8a8d969452352b9c8e4a9697c6f1"
DATA_SHA = "76bde1b77584f9a6aba0e97cbb4fee2cb702f2da727fd26084c1f16209217b73"
DATA_VERSION = "2026-11-08-confirmed-origin-kind-v1"
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20261116
COMPARISONS = (
    ("selection/alpha_snv", "baseline/svm_alpha_snv"),
    ("selection/alpha_snv", "baseline/lda_alpha_raw_snv"),
    ("spacing/alpha_inner_gap_k62_c2", "selection/alpha_snv_c2_k62"),
    ("spacing/alpha_inner_gap_k62_c2", "coverage/uniform_k62"),
    ("rank_context/true_n", "rank_context/shuffle_n"),
    ("spacing/alpha_inner_gap_k62_c2", "selection/alpha_snv"),
)
METHODS = tuple(dict.fromkeys(method for pair in COMPARISONS for method in pair))
LIMITATIONS = (
    "Six explanatory comparisons were locked after archived outcomes were visible; "
    "they are not prospective preregistered confirmation experiments.",
    "Intervals are fixed archived-prediction paired resampling intervals "
    "(固定存档预测的配对重采样区间), conditional on these fitted models and specimens. "
    "They exclude model-development, historical-selection, and new-batch uncertainty.",
    "Outer repeats and repeated predictions are dependent descriptions, not additional "
    "independent specimens. No independent-sample t test or noninferiority/equivalence claim is made.",
    "Physical independence follows the frozen user declarations, not a sampling-log audit. "
    "Batch identifiers remain unknown where unrecorded; a workbook is not assigned a new batch ID.",
    "The 300 existing TRAIN specimens have been exposed in historical model development. "
    "Neither repeat count nor total completed-job count establishes external validation.",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def load(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        return json.load(stream)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = fieldnames or (list(rows[0]) if rows else ["status"])
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_split(jobdir):
    for name in ("split_frozen.json", "split_frozen.json.gz"):
        path = jobdir / name
        if path.is_file():
            return load(path), path
    raise FileNotFoundError(str(jobdir / "split_frozen.json[.gz]"))


def metrics_from_confusion(matrix):
    """Return accuracy, BA, macro-F1, and source recall from one confusion matrix."""
    matrix = np.asarray(matrix, dtype=float)
    tp = np.diag(matrix)
    rows, cols = matrix.sum(1), matrix.sum(0)
    recall = np.divide(tp, rows, out=np.zeros_like(tp), where=rows != 0)
    f1 = np.divide(2 * tp, rows + cols, out=np.zeros_like(tp), where=(rows + cols) != 0)
    return np.array([tp.sum() / matrix.sum(), recall.mean(), f1.mean()]), recall


def confusion(y, pred, classes):
    return np.bincount(y * classes + pred, minlength=classes * classes).reshape(classes, classes)


def frozen_specimens():
    split = csv_rows(ROOT / "paper/current/inputs/split_manifest.csv")
    metadata = {r["record_id"]: r for r in csv_rows(ROOT / "paper/current/inputs/metadata.csv")}
    core = [r for r in split if r["species"] == "酸枣仁" and r["role"].lower() == "train"]
    if len(core) != 300 or Counter(r["origin_group_id"] for r in core) != Counter({
            "ZY03": 60, "ZY04": 60, "ZY05": 60, "ZY06": 60, "ZY07": 60}):
        raise ValueError("Frozen current-paper core300 task differs from requested task")
    for row in core:
        meta = metadata[row["record_id"]]
        if row["physical_sample_id"] != meta["physical_sample_id"] or not row["physical_sample_id"]:
            raise ValueError("Frozen record-to-specimen mapping is inconsistent")
    if len({r["physical_sample_id"] for r in core}) != 300:
        raise ValueError("core300 is not 300 distinct frozen physical specimen IDs")
    return core, metadata


def verify_input(background, core):
    path = background / "inputs/current1672_paired612.npz"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    actual = sha_file(path)
    if actual != DATA_SHA:
        raise ValueError("Pinned background input SHA256 mismatch")
    with np.load(path, allow_pickle=False) as values:
        mask = (values["material_labels"] == "酸枣仁") & (values["roles"] == "train")
        ids = values["record_ids"][mask].astype(str).tolist()
        labels = values["group_ids"][mask].astype(str).tolist()
        if set(ids) != {r["record_id"] for r in core} or len(ids) != 300:
            raise ValueError("Pinned background IDs do not equal frozen corrected TRAIN specimens")
        expected = {r["record_id"]: r["origin_group_id"] for r in core}
        if any(expected[i] != y for i, y in zip(ids, labels)):
            raise ValueError("Pinned background source labels differ from frozen corrected labels")
    return ids, labels, actual


def archive_manifest(background):
    path = background / "snapshot_source/job_manifest.json"
    if not path.exists():
        raise FileNotFoundError(str(path))
    manifest = load(path)
    jobs = manifest["jobs"]
    by_id = {r["id"]: r for r in jobs}
    if len(by_id) != len(jobs):
        raise ValueError("Duplicate IDs in frozen source job manifest")
    return manifest, by_id


def queue_counts(background):
    path = background / "snapshot_source/queue.sqlite3"
    if not path.exists():
        return {"status": "BLOCKED_MISSING_ARTIFACT", "missing": str(path)}
    with sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True) as connection:
        rows = connection.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status").fetchall()
        counts = {str(k): int(v) for k, v in rows}
        attempts = connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    return {"status": "REUSED_VERIFIED", "jobs_by_status": counts, "attempt_count": attempts,
            "independent_specimens": 300, "completed_jobs_are_not_independent_sample_size": True}


def task_inventory(background, report, manifest_jobs):
    """Distinguish scheduled jobs, completed jobs, recoverable files, and repeats."""
    database = background / "snapshot_source/queue.sqlite3"
    statuses = {}
    if database.is_file():
        with sqlite3.connect("file:" + str(database) + "?mode=ro", uri=True) as connection:
            statuses = dict(connection.execute("SELECT id,status FROM jobs").fetchall())
    missing_completed = []
    fields = ["job_id", "ordinal", "method", "mc_draws_scheduled", "repeat", "seed", "fold",
              "source_queue_status", "result_recovered", "included_in_E0_comparison_methods"]
    with gzip.open(report / "source_task_completion_inventory.csv.gz", "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for uid, job in sorted(manifest_jobs.items(), key=lambda item: item[1].get("ordinal", 2**63 - 1)):
            method = job["family"] + "/" + job["variant"]
            present = (background / "jobs" / uid / "result.json").is_file()
            status = statuses.get(uid, "unknown_missing_queue_record")
            writer.writerow({"job_id": uid, "ordinal": job.get("ordinal"), "method": method,
                             "mc_draws_scheduled": job.get("mc_draws"), "repeat": job.get("repeat"),
                             "seed": job.get("seed"), "fold": job.get("fold"),
                             "source_queue_status": status, "result_recovered": present,
                             "included_in_E0_comparison_methods": method in METHODS})
            if method in METHODS and status == "completed" and not present:
                missing_completed.append({"job_id": uid, "method": method, "mc_draws": job.get("mc_draws"),
                                          "repeat": job.get("repeat"), "fold": job.get("fold"),
                                          "reason": "SQLite completed job lacks recovered per-specimen result.json",
                                          "status": "BLOCKED_MISSING_ARTIFACT", "recovery_commit": ARCHIVE_COMMIT,
                                          "recovery_path": "background_study_20261110/jobs/" + uid + "/result.json"})
    return missing_completed


def candidate_ranges(method, protocol):
    """Map the fixed worker's published search to the actual method family."""
    search = protocol.get("worker_spec", {}).get("search", {})
    family, variant = method.split("/", 1)
    answer = {"K": search.get("K"), "PLS_components": search.get("PLS_ranking_components"),
              "C": search.get("SVM_C"), "gamma_multipliers": search.get("SVM_gamma_multipliers"),
              "gap": [1], "preprocessing": ["fixed"]}
    if family == "baseline":
        answer.update(K=[612], PLS_components=[0])
        if variant.startswith("lda_"):
            answer.update(C=[0.0], gamma_multipliers=[0.0], preprocessing=["raw", "snv"])
    elif family == "selection" and variant.endswith("_c2_k62"):
        answer.update(K=[62], PLS_components=[2])
    elif family == "spacing":
        answer.update(K=[62], PLS_components=[2], gap=[1, 2, 4, 8] if "inner" in variant else [int(variant.split("gap")[1].split("_")[0])])
    elif family == "coverage":
        answer.update(K=[int(variant.split("k")[-1])], PLS_components=[0])
    elif family == "rank_context":
        answer.update(K=[62], PLS_components=[2])
    else:
        return {"published_global_search": search, "method_specific_range": "see sealed worker config", "method": method}
    answer["selection_order"] = "macro_f1,balanced_accuracy,accuracy,then published complexity/order"
    return answer


def identity_audit(result, jobdir, manifest_job, metadata, ids, labels, protocol):
    """Verify actual saved IDs and seals; no seed-based split reconstruction."""
    split, split_path = load_split(jobdir)
    selection_path, frozen_path = jobdir / "selection.json", jobdir / "job_frozen.json"
    selection, frozen = load(selection_path), load(frozen_path)
    identity = {k: frozen[k] for k in ("job", "data_sha256", "spec_sha256", "code_sha256")}
    expected_identity = digest(identity)
    seals = [result.get("identity_sha256"), split.get("identity_sha256"),
             selection.get("identity_sha256"), frozen.get("identity_sha256")]
    if any(x != expected_identity for x in seals):
        raise ValueError("Result/selection/split/job identity seals differ")
    if frozen["data_sha256"] != DATA_SHA:
        raise ValueError("Job input SHA differs from pinned core input")
    expected_code = protocol.get("sealed_files", {}).get("study_worker.py")
    if expected_code and frozen["code_sha256"] != expected_code:
        raise ValueError("Job code SHA differs from fixed source protocol")
    if protocol.get("worker_spec") and frozen["spec_sha256"] != digest(protocol["worker_spec"]):
        raise ValueError("Job worker specification SHA differs from fixed source protocol")
    for key in ("id", "repeat", "seed", "fold", "family", "variant", "mc_draws", "ordinal"):
        if key in manifest_job and result["job"].get(key) != manifest_job[key]:
            raise ValueError("Manifest/result job mismatch: " + key)
        if key in result and key in manifest_job and result[key] != manifest_job[key]:
            raise ValueError("Result top-level/manifest mismatch: " + key)
    frozen_job = {k: v for k, v in result["job"].items()
                  if k not in ("resume_from", "cpu_limit", "attempt", "attempt_id")}
    if frozen["job"] != frozen_job:
        raise ValueError("Saved frozen job differs from result job")
    if selection.get("chosen") != result.get("chosen_parameters"):
        raise ValueError("Saved selection parameters differ from final result")
    if selection.get("outer_predictions_consulted") is not False or not result.get(
            "selection_frozen_before_outer_prediction"):
        raise ValueError("Missing training-only selection lock assertion")
    if result.get("testing_only") or result.get("target_kind") != "source_group":
        raise ValueError("Result belongs to another task/testing mode")
    train, test = split["outer_train_ids"], split["outer_test_ids"]
    if len(train) != 240 or len(test) != 60 or len(set(train)) != 240 or len(set(test)) != 60:
        raise ValueError("Outer fold is not 240 unique TRAIN / 60 unique test records")
    if set(train) & set(test) or set(train) | set(test) != set(ids):
        raise ValueError("Outer train/test record separation or task universe differs")
    physical = lambda values: {metadata[i]["physical_sample_id"] for i in values}
    if physical(train) & physical(test):
        raise ValueError("Outer train/test physical specimens overlap")
    inner = split.get("inner", [])
    if len(inner) != 10 or selection.get("inner_folds") != 10 or result.get("inner_folds") != 10:
        raise ValueError("Source protocol is not inner5x2")
    for innerfold in inner:
        fit, val = innerfold["fit_ids"], innerfold["validation_ids"]
        if (len(fit) != len(set(fit)) or len(val) != len(set(val))
                or set(fit) & set(val) or set(fit) | set(val) != set(train)
                or physical(fit) & physical(val)):
            raise ValueError("Invalid or leaking saved inner fold IDs")
    expected = dict(zip(ids, labels))
    predictions = result.get("predictions", [])
    pred_ids = [r["record_id"] for r in predictions]
    if len(pred_ids) != 60 or len(set(pred_ids)) != 60 or set(pred_ids) != set(test):
        raise ValueError("Predictions do not cover the actual heldout fold exactly once")
    classes = sorted(set(labels))
    for row in predictions:
        if row["physical_sample_id"] != metadata[row["record_id"]]["physical_sample_id"]:
            raise ValueError("Prediction physical ID differs from frozen declared mapping")
        if row["y_true"] != expected[row["record_id"]] or row["prediction"] not in classes:
            raise ValueError("Prediction true/source labels differ from corrected task")
        if int(row["is_correct"]) != int(row["y_true"] == row["prediction"]):
            raise ValueError("Incorrect saved prediction correctness flag")
    return {"train": train, "test": test, "train_hash": digest(train), "test_hash": digest(test),
            "inner_hash": digest(inner), "split_path": str(split_path),
            "selection_sha256": sha_file(selection_path), "job_identity_sha256": expected_identity,
            "predictions": predictions}


def make_registry(background, report, ids, labels, metadata, manifest_jobs, protocol):
    """Stream registry rows; keep only compact comparison evidence in memory."""
    groups = defaultdict(dict)
    duplicates, missing, tally = [], [], Counter()
    fields = ["task", "data_version", "input_sha256", "code_commit", "protocol_id", "method",
              "mc_draws_scheduled", "ranking_mc_draws", "repeat", "seed", "fold", "sample_role",
              "candidate_ranges", "final_parameters", "numeric_gamma", "result_path", "result_sha256",
              "job_id", "ordinal", "completion_status", "audit_status", "evidence_limitations"]
    registry_path = report / "evidence_registry.csv.gz"
    with gzip.open(registry_path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result_path in sorted((background / "jobs").glob("*/result.json")):
            result = load(result_path)
            job = result.get("job", {})
            method = str(result.get("family", job.get("family", "unknown"))) + "/" + str(
                result.get("variant", job.get("variant", "unknown")))
            family = method.split("/", 1)[0]
            draws = int(result.get("mc_draws", job.get("mc_draws", 20)))
            repeat = int(job.get("repeat", -1))
            seed, fold = int(result.get("seed", job.get("seed", -1))), int(result.get("fold", job.get("fold", -1)))
            uid = result.get("job_id") or job.get("id") or result_path.parent.name
            ordinal = int(job.get("ordinal", manifest_jobs.get(uid, {}).get("ordinal", 2**63 - 1)))
            complete = result.get("complete") is True and result.get("status") == "complete"
            audit_status = "NOT_IN_PRIMARY_COMPARISON"
            evidence = None
            if method in METHODS and complete:
                try:
                    if uid not in manifest_jobs:
                        raise ValueError("Result job absent from fixed manifest")
                    evidence = identity_audit(result, result_path.parent, manifest_jobs[uid], metadata,
                                              ids, labels, protocol)
                    audit_status = "REUSED_VERIFIED"
                except (KeyError, ValueError, FileNotFoundError, TypeError) as error:
                    audit_status = "BLOCKED_MISSING_ARTIFACT" if isinstance(error, FileNotFoundError) else "INVALID_EVIDENCE"
                    missing.append({"job_id": uid, "method": method, "mc_draws": draws, "repeat": repeat,
                                    "fold": fold, "reason": str(error), "status": audit_status,
                                    "recovery_commit": ARCHIVE_COMMIT,
                                    "recovery_path": "background_study_20261110/jobs/" + uid})
            tally[(method, draws, audit_status)] += 1
            writer.writerow({
                "task": result.get("target_kind", job.get("task", "unknown")), "data_version": DATA_VERSION,
                "input_sha256": DATA_SHA, "code_commit": SOURCE_COMMIT,
                "protocol_id": "thz-mechanism-background-protocol-v1; inner5x2", "method": method,
                "mc_draws_scheduled": draws,
                "ranking_mc_draws": draws if family in ("selection", "spacing", "rank_context", "retain_drop", "permutation") else "none",
                "repeat": repeat, "seed": seed, "fold": fold, "sample_role": "existing TRAIN-only source OOF" if result.get("target_kind") == "source_group" else result.get("target_kind"),
                "candidate_ranges": json.dumps(candidate_ranges(method, protocol), separators=(",", ":")),
                "final_parameters": json.dumps(result.get("chosen_parameters", {}), separators=(",", ":")),
                "numeric_gamma": result.get("numeric_gamma"), "result_path": str(result_path.relative_to(ROOT)) if result_path.is_relative_to(ROOT) else str(result_path),
                "result_sha256": sha_file(result_path), "job_id": uid, "ordinal": ordinal,
                "completion_status": "COMPLETED" if complete else result.get("status", "unknown"),
                "audit_status": audit_status,
                "evidence_limitations": "existing exposed specimens; batch IDs unknown where unrecorded; repeat/job count is not independent N",
            })
            if evidence is not None:
                key = (method, draws, repeat)
                if fold in groups[key]:
                    duplicates.append({"method": method, "mc_draws": draws, "repeat": repeat, "fold": fold,
                                       "job_ids": [groups[key][fold]["job_id"], uid]})
                    groups[key][fold]["duplicate"] = True
                else:
                    groups[key][fold] = {**evidence, "job_id": uid, "seed": seed, "fold": fold,
                                         "repeat": repeat, "method": method, "draws": draws, "ordinal": ordinal,
                                         "parameters": result.get("chosen_parameters"), "numeric_gamma": result.get("numeric_gamma")}
    return groups, missing, duplicates, [{"method": key[0], "mc_draws": key[1], "audit_status": key[2], "jobs": value}
                                         for key, value in sorted(tally.items())]


def complete_groups(groups, ids):
    complete, incomplete = {}, []
    for key, folds in groups.items():
        reason = None
        if set(folds) != set(range(5)):
            reason = "missing outer folds " + str(sorted(set(range(5)) - set(folds)))
        elif any(f.get("duplicate") for f in folds.values()):
            reason = "duplicate method/repeat/fold identity"
        else:
            predicted = [p["record_id"] for f in folds.values() for p in f["predictions"]]
            if len(predicted) != 300 or len(set(predicted)) != 300 or set(predicted) != set(ids):
                reason = "OOF specimen coverage is not exactly once over all five folds"
            elif len({f["seed"] for f in folds.values()}) != 1:
                reason = "outer folds use different seeds within one repeat"
        if reason:
            incomplete.append({"method": key[0], "mc_draws": key[1], "repeat": key[2],
                               "present_folds": sorted(folds), "reason": reason})
        else:
            complete[key] = folds
    return complete, incomplete


def paired_repeats(complete, draws):
    """A no-MC baseline can be paired with a draw family only on identical IDs."""
    candidate = None
    for method in METHODS:
        actual_draws = draws if method.split("/")[0] not in ("baseline", "coverage") else 20
        repeats = {key[2] for key in complete if key[:2] == (method, actual_draws)}
        candidate = repeats if candidate is None else candidate & repeats
    good, rejected = [], []
    for repeat in candidate or set():
        reference = complete[(METHODS[0], draws, repeat)]
        failures = []
        for method in METHODS:
            actual_draws = draws if method.split("/")[0] not in ("baseline", "coverage") else 20
            group = complete[(method, actual_draws, repeat)]
            for fold in range(5):
                a, b = reference[fold], group[fold]
                if (a["seed"], a["train_hash"], a["test_hash"], a["inner_hash"]) != (
                        b["seed"], b["train_hash"], b["test_hash"], b["inner_hash"]):
                    failures.append({"method": method, "fold": fold, "job_id": b["job_id"]})
        if failures:
            rejected.append({"repeat": repeat, "mc_draws": draws, "reason": "actual saved split-ID hashes differ", "failures": failures})
        else:
            good.append(repeat)
    good.sort(key=lambda repeat: min(complete[(METHODS[0], draws, repeat)][f]["ordinal"] for f in range(5)))
    return good, rejected


def prediction_cube(complete, repeats, ids, labels, draws):
    classes = sorted(set(labels))
    index = {v: i for i, v in enumerate(ids)}
    class_index = {v: i for i, v in enumerate(classes)}
    truth = np.array([class_index[y] for y in labels], dtype=np.int64)
    cube = np.empty((len(METHODS), len(repeats), len(ids)), dtype=np.uint8)
    for mi, method in enumerate(METHODS):
        actual_draws = draws if method.split("/")[0] not in ("baseline", "coverage") else 20
        for ri, repeat in enumerate(repeats):
            for ev in complete[(method, actual_draws, repeat)].values():
                for row in ev["predictions"]:
                    cube[mi, ri, index[row["record_id"]]] = class_index[row["prediction"]]
    return cube, truth, classes


def common_contexts(complete, repeats, report):
    contexts = []
    for repeat in repeats[:3]:
        for fold in range(5):
            ev = complete[(METHODS[0], 20, repeat)][fold]
            contexts.append({"context_index": len(contexts), "repeat": repeat, "seed": ev["seed"], "fold": fold,
                             "outer_train_ids": ev["train"], "outer_test_ids": ev["test"],
                             "outer_train_ids_sha256": ev["train_hash"], "outer_test_ids_sha256": ev["test_hash"],
                             "methods": {method: complete[(method, 20, repeat)][fold]["job_id"] for method in METHODS}})
    value = {"protocol_id": "background-inner5x2", "order": "minimum frozen manifest ordinal; not ranked by scores",
             "first3_repeat_ids": repeats[:3], "common_repeats": repeats, "contexts": contexts,
             "MC_seed_list": list(range(2026111600, 2026111630))}
    dump(report / "common_contexts.json", value)
    return value


def bootstrap_weights(labels, report):
    labels = np.asarray(labels)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = np.zeros((BOOTSTRAP_REPLICATES, len(labels)), dtype=np.uint16)
    for cls in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == cls)
        weights[:, indices] = rng.multinomial(len(indices), np.ones(len(indices)) / len(indices),
                                             size=BOOTSTRAP_REPLICATES)
    path = report / "specimen_bootstrap_weights.npz"
    np.savez_compressed(path, weights=weights)
    dump(report / "bootstrap_definition.json", {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES,
         "unit": "frozen declared physical specimen", "strata": "true source group",
         "weights_shared_across_all_methods_and_comparisons": True,
         "retains_complete_repeated_prediction_vector": True,
         "accuracy": "weighted mean of each specimen's correctness averaged across complete repeats",
         "balanced_accuracy": "mean across true sources of weighted specimen correctness averaged across repeats",
         "macro_f1": "point estimate: compute from each repeat's complete confusion matrix, then average repeats; no per-sample F1",
         "interval": "percentile 2.5%/97.5%, numpy linear quantile, conditional on fixed archived predictions",
         "weights_sha256": sha_file(path)})
    return weights


def analyze_predictions(complete, repeats, ids, labels, report, draws=20, bootstrap=True):
    report.mkdir(parents=True, exist_ok=True)
    cube, truth, classes = prediction_cube(complete, repeats, ids, labels, draws)
    nclass, nrepeat = len(classes), len(repeats)
    score = np.empty((len(METHODS), nrepeat, 3), float)
    recall = np.empty((len(METHODS), nrepeat, nclass), float)
    confusion_sum = np.zeros((len(METHODS), nclass, nclass), np.int64)
    for mi in range(len(METHODS)):
        for ri in range(nrepeat):
            cm = confusion(truth, cube[mi, ri], nclass)
            score[mi, ri], recall[mi, ri] = metrics_from_confusion(cm)
            confusion_sum[mi] += cm
    suffix = "" if draws == 20 else "_mc" + str(draws)
    correct = (cube == truth[None, None, :]).mean(1)
    weights = bootstrap_weights(labels, report) if bootstrap else None
    method_rows, effect_rows, repeat_rows, fold_rows, error_rows, source_rows = [], [], [], [], [], []
    source_effect_rows, discordance_rows = [], []
    metrics = ("accuracy", "balanced_accuracy", "macro_f1")
    for mi, method in enumerate(METHODS):
        method_rows.append({"method": method, "ranking_mc_draws": "none" if method.split("/")[0] in ("baseline", "coverage") else draws,
                            "n_complete_repeats": nrepeat, "independent_specimens": len(ids),
                            **dict(zip(metrics, map(float, score[mi].mean(0))))})
        for ci, source in enumerate(classes):
            source_rows.append({"method": method, "source": source, "n_specimens": int(np.sum(truth == ci)),
                                "mean_recall_over_repeats": float(recall[mi, :, ci].mean())})
            for cj, predicted in enumerate(classes):
                error_rows.append({"method": method, "true_source": source, "predicted_source": predicted,
                                   "total_prediction_events": int(confusion_sum[mi, ci, cj]),
                                   "mean_count_per_complete_repeat": float(confusion_sum[mi, ci, cj] / nrepeat),
                                   "prediction_events_are_not_independent_specimens": True})
    for pair_index, (left, right) in enumerate(COMPARISONS, 1):
        li, ri = METHODS.index(left), METHODS.index(right)
        differences = score[li] - score[ri]
        specimen_diff = correct[li] - correct[ri]
        for ci, source in enumerate(classes):
            rdifference = recall[li, :, ci] - recall[ri, :, ci]
            source_effect_rows.append({"comparison": pair_index, "left": left, "right": right,
                                      "source": source, "left_recall": float(recall[li, :, ci].mean()),
                                      "right_recall": float(recall[ri, :, ci].mean()),
                                      "recall_difference": float(rdifference.mean()),
                                      "repeat_wins": int(np.sum(rdifference > 1e-12)),
                                      "repeat_ties": int(np.sum(abs(rdifference) <= 1e-12)),
                                      "repeat_losses": int(np.sum(rdifference < -1e-12))})
            mask = truth == ci
            left_correct = cube[li][:, mask] == truth[None, mask]
            right_correct = cube[ri][:, mask] == truth[None, mask]
            discordance_rows.append({"comparison": pair_index, "left": left, "right": right,
                                     "source": source, "both_correct_prediction_events": int(np.sum(left_correct & right_correct)),
                                     "left_only_correct_prediction_events": int(np.sum(left_correct & ~right_correct)),
                                     "right_only_correct_prediction_events": int(np.sum(~left_correct & right_correct)),
                                     "both_wrong_prediction_events": int(np.sum(~left_correct & ~right_correct)),
                                     "events_are_not_independent_specimens": True})
        if bootstrap:
            acc_draws = weights @ specimen_diff / len(ids)
            ba_draws = sum((weights[:, truth == ci] @ specimen_diff[truth == ci]) / np.sum(truth == ci)
                           for ci in range(nclass)) / nclass
        for metric_index, metric in enumerate(metrics):
            diff = differences[:, metric_index]
            samples = acc_draws if metric == "accuracy" and bootstrap else ba_draws if metric == "balanced_accuracy" and bootstrap else None
            ci = np.quantile(samples, [.025, .975]).tolist() if samples is not None else [None, None]
            effect_rows.append({"comparison": pair_index, "left": left, "right": right, "metric": metric,
                                "difference": float(diff.mean()), "difference_percentage_points": float(100 * diff.mean()),
                                "interval_low": ci[0], "interval_high": ci[1],
                                "interval_low_percentage_points": None if ci[0] is None else 100 * ci[0],
                                "interval_high_percentage_points": None if ci[1] is None else 100 * ci[1],
                                "interval_label": "固定存档预测的配对重采样区间" if samples is not None else "point estimate only",
                                "repeat_min_difference": float(diff.min()), "repeat_q25_difference": float(np.quantile(diff, .25)),
                                "repeat_median_difference": float(np.median(diff)), "repeat_q75_difference": float(np.quantile(diff, .75)),
                                "repeat_max_difference": float(diff.max()), "repeat_wins": int(np.sum(diff > 1e-12)),
                                "repeat_ties": int(np.sum(abs(diff) <= 1e-12)), "repeat_losses": int(np.sum(diff < -1e-12)),
                                "n_paired_repeats_descriptive": nrepeat, "n_independent_specimens": len(ids)})
        for repeat_index, repeat in enumerate(repeats):
            for metric_index, metric in enumerate(metrics):
                repeat_rows.append({"comparison": pair_index, "left": left, "right": right, "repeat": repeat,
                                    "metric": metric, "left_value": float(score[li, repeat_index, metric_index]),
                                    "right_value": float(score[ri, repeat_index, metric_index]),
                                    "difference": float(differences[repeat_index, metric_index])})
            ld = draws if left.split("/")[0] not in ("baseline", "coverage") else 20
            rd = draws if right.split("/")[0] not in ("baseline", "coverage") else 20
            for fold in range(5):
                left_ev, right_ev = complete[(left, ld, repeat)][fold], complete[(right, rd, repeat)][fold]
                by_id = {row["record_id"]: row for row in right_ev["predictions"]}
                lt = np.array([classes.index(p["y_true"]) for p in left_ev["predictions"]])
                lp = np.array([classes.index(p["prediction"]) for p in left_ev["predictions"]])
                rp = np.array([classes.index(by_id[p["record_id"]]["prediction"]) for p in left_ev["predictions"]])
                lm, _ = metrics_from_confusion(confusion(lt, lp, nclass))
                rm, _ = metrics_from_confusion(confusion(lt, rp, nclass))
                for metric_index, metric in enumerate(metrics):
                    fold_rows.append({"comparison": pair_index, "left": left, "right": right, "repeat": repeat,
                                      "seed": left_ev["seed"], "fold": fold, "metric": metric,
                                      "left_value": float(lm[metric_index]), "right_value": float(rm[metric_index]),
                                      "difference": float(lm[metric_index] - rm[metric_index]),
                                      "outer_train_ids_sha256": left_ev["train_hash"], "outer_test_ids_sha256": left_ev["test_hash"]})
    for effect in effect_rows:
        fd = np.array([row["difference"] for row in fold_rows if row["comparison"] == effect["comparison"] and row["metric"] == effect["metric"]])
        effect.update(fold_min_difference=float(fd.min()), fold_q25_difference=float(np.quantile(fd, .25)),
                      fold_median_difference=float(np.median(fd)), fold_q75_difference=float(np.quantile(fd, .75)),
                      fold_max_difference=float(fd.max()), fold_wins=int(np.sum(fd > 1e-12)),
                      fold_ties=int(np.sum(abs(fd) <= 1e-12)), fold_losses=int(np.sum(fd < -1e-12)),
                      fold_differences_are_descriptive_and_dependent=True)
    write_csv(report / ("method_metrics" + suffix + ".csv"), method_rows)
    write_csv(report / ("paired_effects" + suffix + ".csv"), effect_rows)
    write_csv(report / ("repeat_differences" + suffix + ".csv.gz"), repeat_rows)
    write_csv(report / ("fold_differences" + suffix + ".csv.gz"), fold_rows)
    write_csv(report / ("source_recalls" + suffix + ".csv"), source_rows)
    write_csv(report / ("error_distributions" + suffix + ".csv"), error_rows)
    write_csv(report / ("source_paired_effects" + suffix + ".csv"), source_effect_rows)
    write_csv(report / ("paired_error_discordance" + suffix + ".csv"), discordance_rows)
    if bootstrap:
        np.savez_compressed(report / "paired_prediction_cube.npz", predictions=cube, truth=truth,
                            record_ids=np.asarray(ids), physical_sample_ids=np.asarray(ids),
                            classes=np.asarray(classes), methods=np.asarray(METHODS), repeats=np.asarray(repeats))
        dump(report / "specimen_vector_manifest.json", {"record_ids": ids, "source_labels": labels,
             "repeats": repeats, "methods": list(METHODS), "sha256": sha_file(report / "paired_prediction_cube.npz"),
             "shape_method_repeat_specimen": list(cube.shape), "independent_specimens": len(ids),
             "saved_prediction_count_is_not_independent_sample_size": True})
        make_figure(report, effect_rows)
    return {"mc_draws": draws, "complete_common_repeats": nrepeat, "methods": method_rows, "effects": effect_rows}


def make_figure(report, effects):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in effects if r["metric"] == "accuracy"]
    y = np.arange(len(rows))
    point = np.array([r["difference_percentage_points"] for r in rows])
    lower = np.array([r["interval_low_percentage_points"] for r in rows])
    upper = np.array([r["interval_high_percentage_points"] for r in rows])
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.errorbar(point, y, xerr=np.vstack((np.maximum(0, point - lower), np.maximum(0, upper - point))),
                fmt="o", color="#145a73", capsize=4)
    ax.axvline(0, color="#777777", linewidth=1)
    ax.set_yticks(y, [str(r["comparison"]) + ". " + r["left"] + "\nvs " + r["right"] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy difference (percentage points); fixed-prediction paired resampling interval")
    ax.set_title("Six retrospective explanatory comparisons; 300 existing specimens")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(report / "paired_accuracy_effects.png", dpi=170)
    fig.savefig(report / "paired_accuracy_effects.svg")
    plt.close(fig)


def lowo_analysis(report):
    """Keep all workbook successes and failure scenarios separate from core300."""
    source = ROOT / "paper/current/results/kind_lowo_20261115"
    manifest = load(ROOT / "paper/current/manifest.json")
    allowed = {row["path"]: row for row in manifest["files"]}
    outcomes, errors, summaries, audit = [], [], {}, []
    for task in ("binary_kinds", "four_kinds"):
        taskdir = source / task
        for name in ("oof_predictions.csv", "fold_metrics.csv", "fold_splits.json", "summary.json"):
            path = taskdir / name
            key = path.relative_to(ROOT / "paper/current").as_posix()
            if key not in allowed or sha_file(path) != allowed[key]["sha256"]:
                raise ValueError("LOWO file is absent from allowlist or SHA differs: " + key)
        rows = csv_rows(taskdir / "oof_predictions.csv")
        splits = load(taskdir / "fold_splits.json")
        seen = [r["physical_sample_id"] for r in rows]
        if len(set(seen)) != len(seen):
            raise ValueError("LOWO physical specimen appears more than once: " + task)
        for split in splits:
            tr, te = set(split["train_physical_sample_ids"]), set(split["test_physical_sample_ids"])
            if tr & te:
                raise ValueError("LOWO physical train/test overlap: " + split["fold_id"])
            fold_rows = [r for r in rows if r["fold_id"] == split["fold_id"]]
            if set(r["physical_sample_id"] for r in fold_rows) != te:
                raise ValueError("LOWO fold predictions do not match saved test IDs")
        summaries[task] = load(taskdir / "summary.json")
        classes = summaries[task]["metrics"]["class_order"]
        truth = np.array([classes.index(r["true_kind"]) for r in rows])
        predicted = np.array([classes.index(r["predicted_kind"]) for r in rows])
        pooled, _ = metrics_from_confusion(confusion(truth, predicted, len(classes)))
        expected = np.array([summaries[task]["metrics"][k] for k in ("accuracy", "balanced_accuracy", "macro_f1")])
        if not np.allclose(pooled, expected, atol=1e-12, rtol=0):
            raise ValueError("LOWO archived pooled metrics differ from actual OOF predictions")
        for row in csv_rows(taskdir / "fold_metrics.csv"):
            actual = [r for r in rows if r["fold_id"] == row["fold_id"]]
            if len(actual) != int(row["n_test"]) or sum(r["true_kind"] == r["predicted_kind"] for r in actual) != int(row["n_correct"]):
                raise ValueError("LOWO workbook counts differ from actual OOF predictions")
            outcomes.append({"task": task, **row, "outcome": "contains_errors" if int(row["n_correct"]) < int(row["n_test"]) else "all_correct"})
        for row in rows:
            if row["true_kind"] != row["predicted_kind"]:
                errors.append({"task": task, **row})
        audit.append({"task": task, "n_specimens": len(rows), "n_workbooks": len(splits),
                      "each_specimen_has_one_oof_prediction": True, "actual_physical_ID_separation_verified": True})
    write_csv(report / "lowo_all_workbook_outcomes.csv", outcomes)
    write_csv(report / "lowo_all_error_records.csv.gz", errors)
    dump(report / "lowo_summary.json", {"status": "REUSED_VERIFIED", "tasks": summaries, "audit": audit,
         "limitations": ["Existing previously exposed collection; no newly procured independent-batch validation.",
                         "Workbook holdout is a recorded-source stress test; unknown batch IDs remain unknown.",
                         "Fold true-class recall equals accuracy because each workbook has one kind; pooled BA/macro-F1 use all task classes.",
                         "All workbooks and errors retained, including small workbooks; not pooled with the five-source core300 task."]})
    return audit


def execute(source_root, report_root):
    report = report_root / "e0"
    report.mkdir(parents=True, exist_ok=True)
    start_cpu, start_wall = time.process_time(), time.monotonic()
    out = {"module": "E0", "started_utc": now(), "new_classifier_fits": 0, "new_PLS_fits": 0,
           "limitations": list(LIMITATIONS), "comparisons": [list(p) for p in COMPARISONS],
           "method_key_mapping": {m: m for m in METHODS}, "no_score_adaptive_changes": True}
    try:
        out["lowo_audit"] = lowo_analysis(report)
        background = source_root if (source_root / "jobs").exists() else source_root / "background"
        receipt = load(background / "recovery_status.json")
        if receipt.get("complete") is not True:
            raise ValueError("Archive recovery has not completed member/shard/gzip verification")
        core, metadata = frozen_specimens()
        ids, labels, input_hash = verify_input(background, core)
        manifest, manifest_jobs = archive_manifest(background)
        protocol_path = background / "snapshot_source/protocol.json"
        protocol = load(protocol_path)
        out["queue_inventory"] = queue_counts(background)
        out["frozen_manifest_jobs"] = len(manifest["jobs"])
        out["input_sha256"] = input_hash
        out["input_specimens"] = len(ids)
        write_csv(report / "specimen_entity_roles.csv", [{"record_id": uid,
             "physical_specimen_id": metadata[uid]["physical_sample_id"],
             "physical_specimen_basis": metadata[uid]["physical_sample_id_basis"],
             "pellet_id": "unknown", "technical_scan_id": "unknown",
             "metadata_scan_id": metadata[uid].get("scan_id", "unknown"),
             "batch_id": metadata[uid].get("batch_id") or "unknown",
             "source_group": metadata[uid]["origin_group_id"], "role": "train",
             "source_workbook": metadata[uid]["source_workbook"],
             "a_workbook_is_not_assigned_a_new_batch_ID": True} for uid in ids])
        # The pinned background worker explicitly serialized the original core
        # declaration as physical_sample_id=record_id; verify before storing it.
        if any(metadata[uid]["physical_sample_id"] != uid for uid in ids):
            raise ValueError("Pinned worker's record=physical mapping differs from frozen core metadata")
        missing_completed = task_inventory(background, report, manifest_jobs)
        groups, missing, duplicates, tally = make_registry(background, report, ids, labels, metadata,
                                                          manifest_jobs, protocol)
        missing = missing_completed + missing
        complete, incomplete = complete_groups(groups, ids)
        repeats, unpaired = paired_repeats(complete, 20)
        dump(report / "missing_artifacts.json", {"missing_or_invalid": missing, "duplicate_identities": duplicates,
             "incomplete_repeats": incomplete, "unpaired_saved_splits": unpaired,
             "restore_from_commit": ARCHIVE_COMMIT, "not_retrained": True})
        write_csv(report / "evidence_inventory.csv", tally)
        out["complete_method_repeat_counts"] = [{"method": m, "mc_draws": d,
             "complete_repeats": sum(1 for key in complete if key[:2] == (m, d))}
             for m, d in sorted({key[:2] for key in complete})]
        out["common_MC20_repeat_ids"] = repeats
        out["common_MC20_repeat_count"] = len(repeats)
        out["expected_approximate_count_330_is_not_an_acceptance_threshold"] = True
        common_contexts(complete, repeats, report)
        if not repeats:
            raise ValueError("No verified complete paired MC20 repeats; specimen bootstrap is blocked")
        out["primary_analysis"] = analyze_predictions(complete, repeats, ids, labels, report)
        out["separate_MC_convergence"] = []
        for draws in (100, 500):
            comparison_repeats, rejected = paired_repeats(complete, draws)
            if comparison_repeats:
                # Pair MC20/100/500 on the same repeat IDs; never mix counts in the main table.
                common_with20 = [r for r in comparison_repeats if r in repeats]
                if common_with20:
                    out["separate_MC_convergence"].append(analyze_predictions(
                        complete, common_with20, ids, labels, report, draws=draws, bootstrap=False))
                    analyze_predictions(complete, common_with20, ids, labels,
                                        report / ("mc20_matched_to_" + str(draws)), draws=20, bootstrap=False)
            dump(report / ("mc" + str(draws) + "_pairing.json"), {"common_repeat_ids": comparison_repeats,
                 "rejected_saved_splits": rejected, "noMC_baselines_reused_only_on_matching_actual_IDs": True})
        out["status"] = "COMPLETED" if not missing and not duplicates and not unpaired else "BLOCKED_MISSING_ARTIFACT"
        out["missing_or_invalid_job_count"] = len(missing)
        out["incomplete_method_repeat_count"] = len(incomplete)
        out["note"] = "Primary analysis uses only the explicitly listed common complete verified repeats; exclusions and unfinished tasks are retained in inventories."
    except (FileNotFoundError, KeyError, ValueError, sqlite3.Error) as error:
        out["status"] = "BLOCKED_MISSING_ARTIFACT"
        out["reason"] = str(error)
        out["recovery_source"] = {"source_commit": SOURCE_COMMIT, "archive_commit": ARCHIVE_COMMIT,
                                  "restore_script": "experiments/focused_followup_20261116/recovery.py"}
    out["cpu_seconds_this_attempt"] = time.process_time() - start_cpu
    out["wall_seconds_this_attempt"] = time.monotonic() - start_wall
    out["finished_utc"] = now()
    dump(report / "summary.json", out)
    dump(report_root / "task_results/E0.json", {"status": out["status"], "module": "E0",
         "summary": "e0/summary.json", "new_classifier_fits": 0, "new_PLS_fits": 0,
         "common_complete_repeats": out.get("common_MC20_repeat_count", 0), "reason": out.get("reason")})
    lines = ["# E0 evidence and paired archived-prediction analysis", "", "Status: " + out["status"], "",
             "This module fitted no classifier or PLS model.", ""]
    if out.get("reason"):
        lines.extend(["Blocked evidence: " + out["reason"], ""])
    if "primary_analysis" in out:
        lines.extend(["Verified common complete MC20 repeats: " + str(out["common_MC20_repeat_count"]) +
                      "; independent declared physical specimens: 300.", "",
                      "|Comparison|Accuracy difference, pp|Fixed archived-prediction paired resampling interval, pp|", "|---|---:|---:|"])
        for row in out["primary_analysis"]["effects"]:
            if row["metric"] == "accuracy":
                lines.append("|" + str(row["comparison"]) + ": " + row["left"] + " vs " + row["right"] + "|" +
                             f"{row['difference_percentage_points']:.4f}|[{row['interval_low_percentage_points']:.4f}, {row['interval_high_percentage_points']:.4f}]|")
        lines.append("")
    lines.extend(["- " + limit for limit in LIMITATIONS])
    lines.extend(["", "LOWO binary/four-kind results, all heldout-workbook outcomes, and all misclassified records are separate files. "
                  "The five-source analysis does not combine them.", "",
                  "Source artifacts, identities, exact saved split IDs, final training-selected parameters, "
                  "and missing evidence are recorded in evidence_registry.csv.gz and missing_artifacts.json.", ""])
    (report / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=Path(os.environ.get("FOCUSED_SOURCE_ROOT", DEFAULT_CACHE)))
    parser.add_argument("--report-root", type=Path,
                        default=Path(os.environ.get("FOCUSED_RESULT_ROOT", DEFAULT_REPORT)))
    args = parser.parse_args()
    result = execute(args.source_root.resolve(), args.report_root.resolve())
    print(json.dumps({"module": "E0", "status": result["status"],
                      "common_complete_repeats": result.get("common_MC20_repeat_count", 0)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
