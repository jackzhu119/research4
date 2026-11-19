#!/usr/bin/env python3
"""Recompute corrected-kind classification by leave-one-workbook-out validation.

This is a new, fixed-parameter validation protocol using all records. It does
not reassign or replace the historical TRAIN/CAL/PRED roles, and is not a
prospective experiment on newly acquired material. Only the active registry
is used to discover spectra. Existing papers and sealed inputs are read only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform
import posixpath
import re
import subprocess
import warnings
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import openpyxl
from openpyxl.utils.cell import column_index_from_string
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports/kind_lowo_20261115"
BASELINE_COMMIT = "0156e2189d13456d1faf05ebfd53d86eeeb4f5e7"
VERSION = "2026-11-08-confirmed-origin-kind-v1"
KIND_COUNTS = {"枣仁": 364, "酸枣仁": 500, "黄芪": 403, "天麻": 405}
TASKS = {
    "four_kinds": {"labels": ["枣仁", "酸枣仁", "黄芪", "天麻"], "folds": 18, "records": 1672},
    "binary_kinds": {"labels": ["枣仁", "酸枣仁"], "folds": 10, "records": 864},
}
FROZEN_HASHES = {
    "paper/current/inputs/active_workbooks.json": "dd9ec7728590dde2574f64b9664148465c85d7b618311494c613c32b4d879f95",
    "paper/current/inputs/metadata.csv": "fee5e3053c7958da8f0ddadd01de9994efe9669c671e3c1aed72a6cd01639ff1",
    "paper/current/inputs/split_manifest.csv": "0767e3165984a4e0e4ae782d69fdea6f410bfe36bce05ec72f8eadca982317d4",
    "paper/current/inputs/feature_audit.json": "a5a3a6424af44c2a700c0b27e5f711c386ebd6b725050f5ea27dc2fcfab588b3",
}
MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_json(path, value):
    Path(path).write_bytes(json_bytes(value))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    require(bool(rows), f"No rows for {path}")
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def input_hashes():
    paths = ["AGENTS.md", "paper/current/README.md", "paper/current/manifest.json",
             "data/active_workbooks.json", "paper/current/protocols/fixed_classification.json",
             *FROZEN_HASHES]
    values = {path: sha(ROOT / path) for path in paths}
    for path, expected in FROZEN_HASHES.items():
        require(values[path] == expected, f"Sealed input changed: {path}")
    manifest = json.loads((ROOT / "paper/current/manifest.json").read_text())
    allow = {row["path"]: row for row in manifest["files"]}
    for path in FROZEN_HASHES:
        relative = str(Path(path).relative_to("paper/current"))
        require(allow[relative]["sha256"] == values[path], f"Current allowlist mismatch: {path}")
    return values


def read_band(book, member, strings, columns):
    """Stream actual XML addresses, ignoring incorrect uploaded XLSX dimensions."""
    wanted = [1, *columns]
    wanted_set = set(wanted)
    headers = None
    axis, spectra, row_numbers = [], [], []
    with book.open(member) as stream:
        for _, row in ET.iterparse(stream, events=("end",)):
            if row.tag != MAIN + "row":
                continue
            cells = {}
            for cell in row:
                match = re.match(r"([A-Z]+)", cell.get("r", ""))
                require(match is not None, "Unaddressed worksheet cell")
                col = column_index_from_string(match.group(1))
                if col not in wanted_set:
                    continue
                value = cell.find(MAIN + "v")
                if cell.get("t") == "inlineStr":
                    text = "".join(node.text or "" for node in cell.iter(MAIN + "t"))
                elif cell.get("t") == "s":
                    text = strings[int(value.text)]
                else:
                    text = None if value is None else value.text
                cells[col] = text
            number = int(row.get("r"))
            if number == 1:
                headers = [cells.get(col) for col in wanted]
            elif cells.get(1) is not None:
                coordinate = float(cells[1])
                if coordinate > 1.5:
                    row.clear()
                    break
                if coordinate >= 0.1:
                    require(all(cells.get(col) is not None for col in columns), "Missing spectral cell")
                    axis.append(coordinate)
                    spectra.append([float(cells[col]) for col in columns])
                    row_numbers.append(number)
            row.clear()
    values = np.asarray(spectra, dtype=np.float64).T
    frequency = np.asarray(axis, dtype=np.float64)
    require(headers is not None and values.shape == (len(columns), 612), "Expected 612 absorption coordinates")
    require(np.isfinite(values).all() and np.isfinite(frequency).all(), "Nonfinite spectral value")
    require(np.all(np.diff(frequency) > 0), "Nonincreasing frequency axis")
    return headers, frequency, values, row_numbers


def load_features():
    registry = json.loads((ROOT / "data/active_workbooks.json").read_text())
    frozen = json.loads((ROOT / "paper/current/inputs/active_workbooks.json").read_text())
    require(registry["version"] == frozen["version"] == VERSION, "Dataset version mismatch")
    entries = registry["workbooks"]
    require(len(entries) == 18 and all(w["status"] == "active" for w in entries), "Expected 18 active workbooks")
    fixed = {row["path"]: row for row in frozen["workbooks"]}
    metadata = read_csv(ROOT / "paper/current/inputs/metadata.csv")
    splits = read_csv(ROOT / "paper/current/inputs/split_manifest.csv")
    require(len(metadata) == len(splits) == 1672, "Expected 1672 frozen records")
    meta = {row["record_id"]: row for row in metadata}
    split = {row["record_id"]: row for row in splits}
    require(len(meta) == len(split) == 1672 and set(meta) == set(split), "Duplicate/missing record IDs")
    require(len({row["physical_sample_id"] for row in metadata}) == 1672, "Repeated physical sample ID")
    require(Counter(row["role"] for row in splits) == {"train": 999, "calibration": 334, "prediction": 339},
            "Historical roles changed")
    require({row["source_workbook"] for row in metadata} == {row["path"] for row in entries}, "Registry/metadata paths disagree")
    all_records, matrices, audit = [], [], []
    frequency = None
    for entry in entries:
        path = entry["path"]
        require(all(entry[key] == fixed[path][key] for key in
                    ("sha256", "record_count", "origin_group_id", "species", "origin")), "Active/frozen registry disagreement")
        actual_hash = sha(ROOT / path)
        require(actual_hash == entry["sha256"], f"Workbook hash mismatch: {path}")
        records = sorted((row for row in metadata if row["source_workbook"] == path),
                         key=lambda row: column_index_from_string(row["excel_column"]))
        require(len(records) == entry["record_count"], f"Workbook record count: {path}")
        columns = [column_index_from_string(row["excel_column"]) for row in records]
        require(len(columns) == len(set(columns)), "Duplicate metadata column")
        for row in records:
            for key in ("physical_sample_id", "source_workbook", "excel_column", "scan_header", "species", "origin_group_id"):
                require(row[key] == split[row["record_id"]][key], f"Frozen split mismatch: {row['record_id']}/{key}")
            require(row["species"] == entry["species"] and row["origin_group_id"] == entry["origin_group_id"], "Frozen labels disagree")
        with ZipFile(ROOT / path) as book:
            targets = {node.get("Id"): node.get("Target") for node in ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))}
            sheets = {}
            for node in ET.fromstring(book.read("xl/workbook.xml")).find(MAIN + "sheets"):
                target = targets[node.get(REL + "id")]
                sheets[node.get("name")] = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
            strings = []
            if "xl/sharedStrings.xml" in book.namelist():
                strings = ["".join(node.text or "" for node in item.iter(MAIN + "t"))
                           for item in ET.fromstring(book.read("xl/sharedStrings.xml"))]
            ah, aa, alpha, ar = read_band(book, sheets["吸收系数"], strings, columns)
            nh, na, refractive, nr = read_band(book, sheets["折射率"], strings, columns)
        require(ah == nh and ah[1:] == [row["scan_header"] for row in records], "Paired headers disagree")
        require(np.array_equal(aa, na) and ar == nr, "Paired coordinates disagree")
        if frequency is None:
            frequency = aa
        require(np.array_equal(aa, frequency), "Workbook coordinates differ; interpolation prohibited")
        matrices.append(alpha)
        for row in records:
            all_records.append({key: row[key] for key in ("record_id", "physical_sample_id", "source_workbook", "excel_column", "scan_header", "origin_group_id", "species", "origin")} |
                               {"historical_role": split[row["record_id"]]["role"], "dataset_version": VERSION})
        audit.append({"path": path, "origin_group_id": entry["origin_group_id"], "species": entry["species"],
                      "records": len(records), "sha256": actual_hash, "matches_active_registry": True,
                      "matches_frozen_registry": True, "paired_headers_and_coordinates": True,
                      "frequency_count": len(aa), "frequency_min_thz": float(aa[0]), "frequency_max_thz": float(aa[-1]),
                      "alpha_array_sha256": digest_bytes(np.ascontiguousarray(alpha, dtype="<f8").tobytes()),
                      "paired_refractive_array_sha256": digest_bytes(np.ascontiguousarray(refractive, dtype="<f8").tobytes())})
        print(f"Loaded {entry['origin_group_id']}: {len(records)} records, 612 paired coordinates", flush=True)
    require(Counter(row["species"] for row in all_records) == KIND_COUNTS, "Kind denominators changed")
    X = np.concatenate(matrices)
    require(X.shape == (1672, 612), "Unexpected feature matrix")
    fingerprint = [{key: item[key] for key in ("path", "sha256", "records")} for item in audit]
    dataset = {"dataset_version": VERSION, "records": 1672, "workbooks": 18, "kind_counts": KIND_COUNTS,
               "dataset_sha256": digest_bytes(json_bytes(fingerprint)), "dataset_sha256_definition": "SHA256 of sorted-key UTF-8 JSON list of active-registry-order {path,sha256,records}",
               "alpha_matrix_sha256": digest_bytes(np.ascontiguousarray(X, dtype="<f8").tobytes()),
               "axis_sha256": digest_bytes(np.ascontiguousarray(frequency, dtype="<f8").tobytes()),
               "model_predictors": "612 absorption values only; paired refractive-index values checked but not predictors",
               "frequency_policy": "Original shared 0.1 <= f <= 1.5 THz coordinates; no interpolation",
               "workbook_checks": audit, "frozen_roles_preserved": dict(Counter(row["historical_role"] for row in all_records))}
    return X, frequency, all_records, dataset


def snv(X):
    values = np.asarray(X, dtype=np.float64)
    deviation = values.std(axis=1, ddof=1, keepdims=True)
    require(np.isfinite(values).all() and np.all(deviation > np.finfo(float).eps), "Undefined SNV")
    return (values - values.mean(axis=1, keepdims=True)) / deviation


def pooled(y, prediction, labels):
    matrix = confusion_matrix(y, prediction, labels=labels)
    supports = matrix.sum(axis=1)
    recalls = np.diag(matrix) / supports
    return {"accuracy": float(accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, labels=labels, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "n_correct": int(np.trace(matrix)), "n_records": int(matrix.sum()), "class_order": labels,
            "confusion_counts": matrix.tolist(), "confusion_row_normalized": (matrix / supports[:, None]).tolist(),
            "kind_recall": [{"kind": label, "n_records": int(n), "n_correct": int(matrix[i, i]), "recall": float(recall)}
                            for i, (label, n, recall) in enumerate(zip(labels, supports, recalls))]}


def planned_folds(records):
    """Freeze complete deterministic ID partitions before the first model fit."""
    plans = {}
    for name, config in TASKS.items():
        eligible = [row for row in records if row["species"] in config["labels"]]
        books = list(dict.fromkeys(row["source_workbook"] for row in eligible))
        plans[name] = [{"fold_id": f"{name}_fold_{number:02d}", "heldout_workbook": heldout,
                        "train_record_ids": [row["record_id"] for row in eligible if row["source_workbook"] != heldout],
                        "test_record_ids": [row["record_id"] for row in eligible if row["source_workbook"] == heldout]}
                       for number, heldout in enumerate(books, 1)]
    return plans


def analyze_task(name, config, X, records, out, protocol_sha):
    labels = config["labels"]
    selected = np.asarray([i for i, row in enumerate(records) if row["species"] in labels])
    task_records = [records[i] for i in selected]
    features = X[selected]
    y = np.asarray([row["species"] for row in task_records])
    groups = np.asarray([row["source_workbook"] for row in task_records])
    ids = np.asarray([row["record_id"] for row in task_records])
    physical_ids = np.asarray([row["physical_sample_id"] for row in task_records])
    workbooks = list(dict.fromkeys(groups.tolist()))
    require(len(task_records) == config["records"] and len(workbooks) == config["folds"], "Task denominator mismatch")
    prediction = np.empty(len(y), dtype=object)
    majority_prediction = np.empty(len(y), dtype=object)
    tests = np.zeros(len(y), dtype=int)
    folds, fold_rows, oof_rows, fit_params = [], [], [], {}
    for fold_number, heldout in enumerate(workbooks, 1):
        train = np.flatnonzero(groups != heldout)
        test = np.flatnonzero(groups == heldout)
        require(not set(ids[train]) & set(ids[test]), "Record ID leakage")
        require(not set(physical_ids[train]) & set(physical_ids[test]), "Physical ID leakage")
        train_books = list(dict.fromkeys(groups[train].tolist()))
        require(heldout not in train_books and set(train_books) | {heldout} == set(workbooks), "Workbook leakage/omission")
        require(len(set(y[test])) == 1 and set(y[train]) == set(labels), "Test kind missing from training")
        class_counts = Counter(y[train])
        class_workbooks = {label: len({groups[i] for i in train if y[i] == label}) for label in labels}
        require(all(class_workbooks.values()), "Kind has no training workbook")
        majority = max(labels, key=lambda label: class_counts[label])
        model = Pipeline([("snv", FunctionTransformer(snv, validate=False)), ("scaler", StandardScaler()),
                          ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced",
                                      decision_function_shape="ovr", break_ties=False, random_state=42))])
        with threadpool_limits(limits=1):
            model.fit(features[train], y[train])
            pred = model.predict(features[test])
        scaler = model.named_steps["scaler"]
        train_snv = snv(features[train])
        require(int(scaler.n_samples_seen_) == len(train), "Scaler fit count is not training count")
        require(np.allclose(scaler.mean_, train_snv.mean(axis=0), rtol=1e-13, atol=1e-14), "Scaler means not training-only")
        require(np.allclose(scaler.var_, train_snv.var(axis=0), rtol=1e-13, atol=1e-14), "Scaler variance not training-only")
        prediction[test] = pred
        majority_prediction[test] = majority
        tests[test] += 1
        fold_id = f"{name}_fold_{fold_number:02d}"
        group_id = task_records[int(test[0])]["origin_group_id"]
        true_kind = str(y[test[0]])
        n_correct = int(np.sum(pred == y[test]))
        accuracy = n_correct / len(test)
        baseline_correct = int(np.sum(y[test] == majority))
        fold_rows.append({"task": name, "fold_id": fold_id, "heldout_group": group_id,
                          "heldout_workbook": heldout, "true_kind": true_kind,
                          "n_train": len(train), "n_test": len(test), "n_correct": n_correct,
                          "accuracy": accuracy, "true_kind_recall": accuracy,
                          "training_majority_kind": majority, "majority_n_correct": baseline_correct,
                          "majority_accuracy": baseline_correct / len(test)})
        split = {"fold_id": fold_id, "heldout_group": group_id, "heldout_workbook": heldout,
                 "true_kind": true_kind, "train_workbooks": train_books, "test_workbooks": [heldout],
                 "train_record_ids": ids[train].tolist(), "test_record_ids": ids[test].tolist(),
                 "train_physical_sample_ids": physical_ids[train].tolist(), "test_physical_sample_ids": physical_ids[test].tolist(),
                 "train_kind_counts": dict(class_counts), "train_workbooks_per_kind": class_workbooks,
                 "train_historical_role_counts": dict(Counter(task_records[i]["historical_role"] for i in train)),
                 "test_historical_role_counts": dict(Counter(task_records[i]["historical_role"] for i in test)),
                 "record_id_disjoint": True, "physical_id_disjoint": True, "workbook_disjoint": True,
                 "all_task_kinds_represented_in_training": True, "scaler_fit_n": int(scaler.n_samples_seen_),
                 "scaler_mean_and_variance_match_training_only": True,
                 "scaler_mean_sha256": digest_bytes(scaler.mean_.astype("<f8").tobytes()),
                 "scaler_scale_sha256": digest_bytes(scaler.scale_.astype("<f8").tobytes()),
                 "fitted_gamma": float(model.named_steps["svm"]._gamma),
                 "training_majority_kind": majority, "training_majority_tie_rule": "First in declared class order",
                 "fit_protocol_sha256": protocol_sha}
        folds.append(split)
        fit_params[fold_id + "_scaler_mean"] = scaler.mean_
        fit_params[fold_id + "_scaler_scale"] = scaler.scale_
        fit_params[fold_id + "_scaler_var"] = scaler.var_
        fit_params[fold_id + "_gamma"] = np.asarray(model.named_steps["svm"]._gamma)
        for local, predicted in zip(test, pred):
            row = task_records[int(local)]
            oof_rows.append({"task": name, "fold_id": fold_id, **row, "true_kind": row["species"],
                             "predicted_kind": str(predicted), "correct": int(predicted == row["species"]),
                             "training_majority_prediction": majority, "majority_correct": int(majority == row["species"])})
        print(f"{name} {group_id}: {n_correct}/{len(test)} = {accuracy:.6f}; training majority {majority}", flush=True)
    require(np.all(tests == 1), "Every task record must be tested exactly once")
    require(len(oof_rows) == len(set(row["record_id"] for row in oof_rows)) == config["records"], "OOF coverage failed")
    summary = {"task": name, "protocol": "leave-one-workbook-out", "fit_protocol_sha256": protocol_sha,
               "n_folds": len(folds), "n_workbooks": len(folds), "all_records_tested_exactly_once": True,
               "metrics": pooled(y, prediction, labels), "training_majority_baseline": pooled(y, majority_prediction, labels),
               "equal_weight_workbook_mean_accuracy": float(np.mean([row["accuracy"] for row in fold_rows])),
               "equal_weight_workbook_mean_majority_accuracy": float(np.mean([row["majority_accuracy"] for row in fold_rows])),
               "fold_metric_note": "Every heldout workbook has one true kind; fold accuracy equals its true-kind recall. Macro-F1 and balanced accuracy are computed only after pooling all out-of-workbook predictions with all task classes.",
               "per_workbook": fold_rows}
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "oof_predictions.csv", oof_rows)
    write_csv(out / "fold_metrics.csv", fold_rows)
    write_json(out / "fold_splits.json", folds)
    write_json(out / "summary.json", summary)
    write_csv(out / "kind_recall.csv", summary["metrics"]["kind_recall"])
    for metric, filename in (("confusion_counts", "pooled_confusion.csv"), ("confusion_row_normalized", "pooled_confusion_normalized.csv")):
        write_csv(out / filename, [{"true_kind": label, **dict(zip(labels, values))}
                                  for label, values in zip(labels, summary["metrics"][metric])])
    np.savez_compressed(out / "fit_parameters.npz", **fit_params)
    return summary


def verify_existing(out):
    receipt = json.loads((out / "verification.json").read_text())
    require(receipt["script_sha256"] == sha(__file__), "Analysis source hash changed")
    for path, value in receipt["input_sha256"].items():
        require(sha(ROOT / path) == value, f"Protected input changed: {path}")
    for item in receipt["output_files"]:
        require(sha(out / item["path"]) == item["sha256"], f"Output hash changed: {item['path']}")
    dataset = json.loads((out / "dataset_audit.json").read_text())
    for item in dataset["workbook_checks"]:
        require(sha(ROOT / item["path"]) == item["sha256"], f"Workbook changed: {item['path']}")
    for task, config in TASKS.items():
        predictions = read_csv(out / task / "oof_predictions.csv")
        require(len(predictions) == len({row["record_id"] for row in predictions}) == config["records"], "OOF ID coverage failed")
        actual = pooled([row["true_kind"] for row in predictions], [row["predicted_kind"] for row in predictions], config["labels"])
        require(actual == json.loads((out / task / "summary.json").read_text())["metrics"], "OOF pooled metrics differ")
    print("PASS: source/input/output SHA256, all active workbooks, OOF coverage and pooled metrics", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    require(out.is_relative_to(ROOT / "reports"), "Output must be beneath reports/")
    if args.verify_only:
        verify_existing(out)
        return
    protected = input_hashes()
    baseline_sources = []
    for path in ("scripts/run_species_revision_classification.py", "src/herb_spectra/spectral_models.py", "scripts/prepare_species_revision_features.py"):
        raw = subprocess.check_output(["git", "-C", str(ROOT), "show", BASELINE_COMMIT + ":" + path])
        baseline_sources.append({"commit": BASELINE_COMMIT, "path": path, "sha256": digest_bytes(raw)})
    out.mkdir(parents=True, exist_ok=True)
    protocol = {"protocol_id": "kind-lowo-fixed-full-snv-rbf-svc-20261115", "script_sha256": sha(__file__),
                "new_validation_protocol": True, "dataset_version": VERSION, "tasks": TASKS,
                "split_rule": "For each task, hold out each complete workbook once; fit on all other eligible workbooks. Original TRAIN/CAL/PRED roles are retained only as provenance, with all roles included in this new protocol.",
                "preprocessing": ["Per-spectrum SNV using sample standard deviation (ddof=1)", "StandardScaler fit only on the training workbooks in the current fold"],
                "features": "612 absorption coordinates at original paired frequency points in the inclusive 0.1-1.5 THz interval; no interpolation, selection or metadata predictors",
                "estimator": {"class": "sklearn.svm.SVC", "kernel": "rbf", "C": 10.0, "gamma": "scale", "class_weight": "balanced", "random_state": 42, "decision_function_shape": "ovr", "break_ties": False},
                "hyperparameter_search": False, "parameters_inherited_from": "Existing corrected-kind full_svm baseline; no tuning under this validation protocol",
                "common_baseline_source": baseline_sources, "input_sha256": protected,
                "reporting": "Retain every fold and both tasks, including errors. Report pooled accuracy/macro-F1/balanced accuracy and kind recalls, per-workbook true-kind recall, equal-weight workbook mean accuracy and per-fold training-majority baseline. No per-fold macro-F1 for single-true-kind workbooks.",
                "interpretation": "Retrospective workbook-group validation on existing corrected spectra; holding out a workbook does not establish prospective, independent procurement, acquisition-date, instrument or new-batch generalization.",
                "software": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__, "openpyxl": openpyxl.__version__}}
    write_json(out / "protocol.json", protocol)  # Written before any classifier fit.
    protocol_sha = sha(out / "protocol.json")
    X, axis, records, dataset = load_features()
    write_json(out / "dataset_audit.json", dataset)
    write_csv(out / "record_index.csv", records)
    write_csv(out / "feature_axis.csv", [{"feature_index": i, "frequency_thz": float(value)} for i, value in enumerate(axis)])
    np.savez_compressed(out / "features.npz", X_alpha=X, axis=axis, record_ids=np.asarray([row["record_id"] for row in records]))
    plans = planned_folds(records)
    write_json(out / "planned_folds.json", plans)
    summaries = {name: analyze_task(name, config, X, records, out / name, protocol_sha) for name, config in TASKS.items()}
    for name in TASKS:
        actual = json.loads((out / name / "fold_splits.json").read_text())
        require([{key: fold[key] for key in ("fold_id", "heldout_workbook", "train_record_ids", "test_record_ids")}
                 for fold in actual] == plans[name], "Executed folds differ from pre-fit plan")
    write_json(out / "summary.json", {"protocol_id": protocol["protocol_id"], "dataset_sha256": dataset["dataset_sha256"],
                                      "tasks": summaries, "no_original_role_reassignments": True,
                                      "no_model_selection_under_lowo": True, "all_positive_and_negative_outcomes_retained": True})
    (out / "README.md").write_text("# Kind classification: leave-one-workbook-out\n\n"
        "New retrospective validation protocol on all corrected records: four kinds, 18 folds/1672 specimens; Zaoren versus Suanzaoren, 10 folds/864 specimens.\n\n"
        "Each complete workbook is tested once while all other eligible workbooks train the fixed full-spectrum SNV (ddof=1), training-only StandardScaler and C=10 RBF SVC baseline. There is no hyperparameter search. The existing TRAIN/CAL/PRED roles are preserved as provenance, not reused as this protocol's partition.\n\n"
        "All folds, mistakes, per-record OOF predictions, explicit train/test IDs, training class/workbook counts, scaler parameters and majority-class controls are retained. A fold has one true kind, so fold accuracy is true-kind recall; pooled macro-F1 and balanced accuracy use the whole task. Equal-weight workbook mean accuracy gives each workbook one vote. Workbook grouping is not proof of prospective or independently acquired batch performance.\n\n"
        "Recompute: `python scripts/validate_kind_lowo_20261115.py`\n\n"
        "Verify existing outputs: `python scripts/validate_kind_lowo_20261115.py --verify-only`\n", encoding="utf-8")
    require(protected == input_hashes(), "A protected source changed during analysis")
    files = [{"path": str(path.relative_to(out)), "bytes": path.stat().st_size, "sha256": sha(path)}
             for path in sorted(out.rglob("*")) if path.is_file() and path.name != "verification.json"]
    write_json(out / "verification.json", {"status": "PASS", "script_sha256": sha(__file__),
               "input_sha256": protected, "dataset_sha256": dataset["dataset_sha256"], "output_files": files,
               "checks": ["18 original workbook hashes match active and frozen registries", "1672 unique record and physical IDs; frozen labels and original roles retained", "612 exact paired coordinates; numeric absorption predictors only", "Every task record tested exactly once", "Complete workbook, record ID and physical ID separation in every fold", "Every kind retains at least one training workbook", "SNV ddof=1 and fold-training-only scaler means/variances verified", "Fixed existing classifier parameters; no tuning or outcome filtering"]})
    verify_existing(out)
    print(json.dumps({name: {**value["metrics"], "equal_weight_workbook_mean_accuracy": value["equal_weight_workbook_mean_accuracy"]}
                      for name, value in summaries.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="Workbook contains no default style")
    main()
