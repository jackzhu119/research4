#!/usr/bin/env python3
"""Independently recount saved legacy-1305 and V4-330 predictions, without fitting.

Only new audit outputs are written. Saved producer code, models, predictions and
metrics remain untouched. Repeated support draws are never counted as new people,
plants, physical specimens or batches. COMPLETE describes saved-output checks,
not external biological or chemical validation.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time


EXPECTED = {"legacy1305": 1305, "v4_330": 330}
METRICS = ("accuracy", "macro_f1", "balanced_accuracy")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def label_for(identifier, task):
    group = identifier.split("-", 1)[0]
    if task in ("binary_kinds", "four_kinds"):
        if group in ("ZY01", "ZY02") or group.startswith("ZR"):
            return "枣仁"
        if group in ("ZY03", "ZY04", "ZY05", "ZY06", "ZY07"):
            return "酸枣仁"
        if group.startswith("HQ"):
            return "黄芪"
        if group.startswith("TM"):
            return "天麻"
        raise ValueError("Unrecognized current specimen ID " + identifier)
    return group


def metrics(matrix, labels):
    n = sum(map(sum, matrix))
    require(n > 0, "Empty prediction setting")
    per_class = []
    for i, name in enumerate(labels):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        per_class.append(dict(label=name, precision=precision, recall=recall,
                              f1=f1, support=support))
    correct = sum(matrix[i][i] for i in range(len(labels)))
    return dict(n=n, denominator=n, correct=correct, accuracy=correct / n,
                macro_f1=sum(row["f1"] for row in per_class) / len(labels),
                balanced_accuracy=sum(row["recall"] for row in per_class
                                      if row["support"]) / sum(row["support"] > 0 for row in per_class),
                class_order=labels, confusion_matrix=matrix, per_class=per_class,
                class_counts={row["label"]: row["support"] for row in per_class},
                class_recall={row["label"]: row["recall"] for row in per_class})


def compare(actual, saved, context):
    differences = []
    for key in ("n", "denominator", "correct", "confusion_matrix", "class_order",
                "class_counts", "class_recall", "per_class", *METRICS):
        if key not in saved:
            continue
        def close(a, b):
            if isinstance(b, dict):
                return isinstance(a, dict) and all(k in a and close(a[k], v) for k, v in b.items())
            if isinstance(b, list):
                return isinstance(a, list) and len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
            if isinstance(b, (int, float)) and not isinstance(b, bool):
                return isinstance(a, (int, float)) and math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 2e-12
            return a == b
        if not close(actual[key], saved[key]):
            differences.append(context + ":" + key)
    require(not differences, "Saved metric mismatch " + ", ".join(differences))


def string_ids(values, context):
    require(isinstance(values, list), "Missing ID list " + context)
    require(all(isinstance(x, str) and x for x in values), "Invalid ID " + context)
    require(len(values) == len(set(values)), "Duplicate physical ID " + context)
    return set(values)


def flatten_ids(value):
    if isinstance(value, dict):
        return [item for nested in value.values() for item in flatten_ids(nested)]
    if isinstance(value, list):
        return [item for nested in value for item in flatten_ids(nested)]
    return [value] if isinstance(value, str) else []


def file_index(roots):
    indexed = defaultdict(list)
    names = ("validation_predictions.csv.gz", "predictions.csv.gz")
    for root in roots:
        for name in names:
            for path in root.rglob(name):
                if path not in indexed[(path.parent.name, name)]:
                    indexed[(path.parent.name, name)].append(path)
    return indexed


def saved_file(directory, name, index):
    local = directory / name
    candidates = ([local] if local.is_file() else []) + index.get((directory.name, name), [])
    require(bool(candidates), "Missing " + directory.name + "/" + name)
    return candidates[0], candidates


def audit_job(suite, directory, index, master_train):
    summary = read(directory / "summary.json")
    complete = read(directory / "complete.json")
    require(complete.get("status") == "complete", "Producer did not seal complete")
    is_legacy_classic = suite == "legacy1305" and directory.name.startswith("classical_")
    is_matched = suite == "v4_330" and directory.name.startswith("matched_")
    is_neural = suite == "legacy1305" and directory.name.startswith("neural_")
    kind = "classical" if is_legacy_classic else "matched" if is_matched else "neural" if is_neural else "representation"
    declaration_name = "protocol.json" if is_neural else "job_manifest.json"
    declaration = read(directory / declaration_name)
    task = declaration.get("task", "acid_origins")
    prediction_name = "predictions.csv.gz" if kind == "representation" else "validation_predictions.csv.gz"
    prediction, candidates = saved_file(directory, prediction_name, index)
    artifacts = complete.get("artifacts_sha256", complete.get("artifacts", {}))
    checked_hashes = {}
    relevant = ("summary.json", declaration_name, "metrics_by_setting.json", "validation_truth.json",
                "query_truth.json", "support_draws.json", "prediction_lock.json", prediction_name)
    for name in relevant:
        p = prediction if name == prediction_name else directory / name
        if p.is_file():
            checked_hashes[name] = sha(p)
            if name in artifacts:
                require(checked_hashes[name] == artifacts[name], "Producer SHA mismatch " + name)
    require(prediction_name in artifacts, "Prediction file has no producer completion hash")
    for duplicate in candidates[1:]:
        require(sha(duplicate) == checked_hashes[prediction_name], "Conflicting prediction copies")

    if kind in ("classical", "matched"):
        source = string_ids(declaration["source_record_ids"], "source")
        support = set()
        query = string_ids(declaration["outer_validation_record_ids"], "query")
        labels = summary.get("classes", declaration.get("class_order"))
        require(isinstance(labels, list), "Missing class order")
    else:
        source = string_ids(declaration["source_ids"], "source")
        support = string_ids(declaration["support_ids"], "support")
        query = string_ids(declaration.get("validation_ids", declaration.get("query_ids")), "query")
        labels = summary.get("classes", ["ZY03", "ZY04", "ZY05", "ZY06", "ZY07"])
    require(source and query and not source & query and not source & support and not support & query,
            "Source/support/query physical IDs overlap or an essential role is empty")
    target_all = source | support | query
    if master_train is not None:
        require(target_all <= master_train, "Non-TRAIN physical ID accessed")
        require(set(flatten_ids(declaration.get("auxiliary_ids", []))) <= master_train,
                "Non-TRAIN auxiliary ID accessed")
    require(not summary.get("master_calibration_or_prediction_used", False), "Master CAL/PRED used")
    require(summary.get("no_master_CAL_PRED_used", True), "Master CAL/PRED used")
    for partition in declaration.get("inner_partitions", []):
        if isinstance(partition, dict):
            train = set(partition.get("train_ids", partition.get("train", [])))
            validation = set(partition.get("validation_ids", partition.get("validation", [])))
            if train or validation:
                require(not train & validation and train | validation <= source, "Inner split overlap or leakage")

    truth = {identifier: label_for(identifier, task) for identifier in query}
    truth_name = "validation_truth.json" if is_neural else "query_truth.json"
    if kind in ("neural", "representation"):
        saved_truth = read(directory / truth_name)
        require(saved_truth == truth, "Saved truths do not match frozen query IDs/current task labels")
    require(len(labels) == len(set(labels)) and set(truth.values()) <= set(labels), "Invalid class order")
    lookup = {name: i for i, name in enumerate(labels)}
    support_draws_checked = 0
    limitations = []
    if is_neural:
        draws = read(directory / "support_draws.json")
        expected_draws = {(r, k) for r in range(declaration["support_repeats"]) for k in declaration["shots"]}
        drawn = {}
        support_order = declaration["support_ids"]
        for draw in draws:
            key = (int(draw["repeat"]), int(draw["shots"]))
            require(key not in drawn, "Repeated support-draw key")
            ids = string_ids(draw["support_ids"], "support draw")
            require(ids <= support, "Draw includes source/query physical IDs")
            require([support_order[i] for i in draw["indices"]] == draw["support_ids"], "Support index/ID mismatch")
            counts = Counter(label_for(x, task) for x in ids)
            require(counts == {label: key[1] for label in labels}, "Support draw is not K independent IDs per class")
            drawn[key] = ids
        require(set(drawn) == expected_draws, "Missing or extra support draws")
        for repeat in range(declaration["support_repeats"]):
            previous = set()
            for k in sorted(declaration["shots"]):
                require(previous <= drawn[(repeat, k)], "Support draws are not nested")
                previous = drawn[(repeat, k)]
        support_draws_checked = len(draws)
    elif kind == "representation" and declaration.get("with_calibration"):
        limitations.append("V4 producer does not save per-draw support IDs; only frozen source/support/query isolation can be checked.")

    expected = {}
    if kind == "classical":
        for row in summary["results"]:
            for semantics in ("vote", "score_argmax"):
                expected[(row["method"], semantics)] = row["outer_metrics_" + semantics]
    elif kind == "matched":
        for row in summary["results"]:
            expected[(row["family"], int(row["k"]))] = row["metrics"]
    elif kind == "neural":
        saved_settings = read(directory / "metrics_by_setting.json")
        for row in saved_settings:
            expected[(row["arm"], row["method"], int(row["repeat"]), int(row["shots"]))] = row
        require(len(expected) == len(saved_settings), "Duplicate saved metric setting")
        arms = declaration["arms"] + ["SVM_reference"]
        declared_settings = {(arm, "source_only", -1, 0) for arm in arms}
        declared_settings.update((arm, calibrator["id"], repeat, shot)
                                 for arm in arms for calibrator in declaration["calibrators"]
                                 for repeat in range(declaration["support_repeats"])
                                 for shot in declaration["shots"])
        require(set(expected) == declared_settings, "Saved settings omit a declared arm/calibrator/repeat/K")
    else:
        for row in summary["metrics"]:
            expected[(row["model"], row["setting"], int(row["repeat"]), int(row["shots"]))] = row
        require(len(expected) == len(summary["metrics"]), "Duplicate saved metric setting")
        declared_settings = {(model, "source_only", -1, 0) for model in ("neural", "matched_SVM")}
        if declaration.get("with_calibration"):
            declared_settings.update((model, "identity_zero_support", -1, 0) for model in ("neural", "matched_SVM"))
            declared_settings.update((model, method, repeat, shot)
                                     for model in ("neural", "matched_SVM")
                                     for method in ("legacy_lambda1", "weak_lambda1000_blend005")
                                     for repeat in range(20) for shot in (1, 3, 5, 10))
        require(set(expected) == declared_settings, "Saved V4 settings omit a declared model/calibrator/repeat/K")
    require(expected, "No saved settings")
    # One bounded job is held in memory; whole raw CSVs are never materialized.
    cells = {key: {"matrix": [[0] * len(labels) for _ in labels], "ids": set(), "predictions": {}}
             for key in expected}
    rows = 0
    scores_not_saved = 0
    with gzip.open(prediction, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            rid = row["record_id"]
            require(rid in truth, "Prediction outside frozen query")
            if "physical_sample_id" in row:
                require(row["physical_sample_id"] == rid, "Physical specimen mapping changed")
            if "y_true" in row:
                require(row["y_true"] == truth[rid], "Prediction truth disagrees with frozen task")
            if kind == "classical":
                values = [((row["method"], s), row["prediction_" + s]) for s in ("vote", "score_argmax")]
                require(json.loads(row["class_order_json"]) == labels, "CSV class order mismatch")
                scores = json.loads(row["scores_json"])
                require(len(scores) == len(labels) and all(math.isfinite(x) for x in scores), "Invalid decision scores")
                require(labels[max(range(len(scores)), key=scores.__getitem__)] == row["prediction_score_argmax"], "Score argmax mismatch")
                for s in ("vote", "score_argmax"):
                    require(int(row["is_correct_" + s]) == int(row["prediction_" + s] == truth[rid]), "Correctness flag mismatch")
            elif kind == "matched":
                values = [((row["family"], int(row["k"])), row["prediction"])]
                require(int(row["is_correct"]) == int(row["prediction"] == truth[rid]), "Correctness flag mismatch")
            else:
                a, m = ("arm", "method") if kind == "neural" else ("model", "setting")
                values = [((row[a], row[m], int(row["repeat"]), int(row["shots"])), row["predicted"])]
                if kind == "representation" and not row.get("scores_json"):
                    require(row["setting"] != "source_only", "Source decision scores are missing")
                    scores_not_saved += 1
                else:
                    scores = json.loads(row["scores_json"]) if "scores_json" in row else [float(row["score_" + label]) for label in labels]
                    require(len(scores) == len(labels) and all(math.isfinite(x) for x in scores), "Nonfinite/missing decision scores")
            for key, predicted in values:
                require(key in cells, "Undeclared prediction setting")
                require(predicted in lookup, "Prediction outside declared classes")
                cell = cells[key]
                require(rid not in cell["ids"], "Duplicate physical specimen within setting")
                cell["ids"].add(rid)
                cell["predictions"][rid] = predicted
                cell["matrix"][lookup[truth[rid]]][lookup[predicted]] += 1
    recalculated = {}
    for key, cell in cells.items():
        require(cell["ids"] == query, "Missing query IDs in setting " + str(key))
        value = metrics(cell["matrix"], labels)
        compare(value, expected[key], str(key))
        if kind == "representation" and "corrected" in expected[key]:
            baseline = cells[(key[0], "source_only", -1, 0)]["predictions"]
            corrected = sum(baseline[rid] != truth[rid] and pred == truth[rid] for rid, pred in cell["predictions"].items())
            damaged = sum(baseline[rid] == truth[rid] and pred != truth[rid] for rid, pred in cell["predictions"].items())
            require(corrected == expected[key]["corrected"] and damaged == expected[key]["damaged"], "Corrected/damaged recount differs")
        recalculated[key] = value
    if is_neural:
        for saved in summary["metrics"]:
            subset = [value for key, value in recalculated.items()
                      if (key[0], key[1], key[3]) == (saved["arm"], saved["method"], int(saved["shots"]))]
            require(len(subset) == saved["support_repeats"], "Summary support-repeat count mismatch")
            require(saved["n_validation"] == len(query), "Summary query denominator mismatch")
            compare({name: sum(row[name] for row in subset) / len(subset) for name in METRICS},
                    {name: saved[name] for name in METRICS}, "summary mean")
    if "prediction_rows" in summary:
        require(rows == summary["prediction_rows"], "Summary CSV row count mismatch")
    fold = declaration.get("outer_fold", declaration.get("fold"))
    seed = declaration.get("outer_split_seed", declaration.get("repeat_seed", declaration.get("seed")))
    group_id = directory.name.rsplit("_f", 1)[0] if kind != "classical" else directory.name.rsplit("_f", 1)[0]
    # Neural/representation names carry seed after fold; retain it explicitly.
    if kind in ("neural", "representation"):
        group_id += "_s" + str(seed)
    result = dict(suite=suite, job_id=directory.name, kind=kind, status="PASS", prediction_file=str(prediction),
                  hashes=checked_hashes, rows=rows, settings=len(expected), query_physical_specimens=len(query),
                  source_physical_specimens=len(source), support_physical_specimens=len(support),
                  support_draws_checked=support_draws_checked, prediction_rows_without_saved_scores=scores_not_saved,
                  limitations=limitations)
    return result, (group_id, int(fold), query, target_all, labels), recalculated


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-root", type=Path, required=True)
    parser.add_argument("--legacy-prediction-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.process_time()
    index = file_index(args.legacy_prediction_roots)
    master_files = list(args.small_root.rglob("split_manifest.csv"))
    master_train = None
    if master_files:
        with master_files[0].open(encoding="utf-8-sig", newline="") as handle:
            records = list(csv.DictReader(handle))
        require(len(records) == 1672 and len({row["record_id"] for row in records}) == 1672,
                "Frozen split must contain1672 unique specimens")
        require(all(row["physical_sample_id"] == row["record_id"] for row in records),
                "Physical specimen mapping differs from the frozen independent-ID declaration")
        master_train = {row["record_id"] for row in records
                        if row.get("role", row.get("split", "")).lower() in ("train", "training")}
        require(len(master_train) == 999, "Frozen master split must identify exactly999 TRAIN specimens")
    counts, failures, outputs, oof = Counter(), [], {}, {}
    progress = args.output / "progress.json"
    with (args.output / "jobs.jsonl").open("w", encoding="utf-8") as joblog, (args.output / "settings.jsonl").open("w", encoding="utf-8") as settinglog:
        for suite, expected_n in EXPECTED.items():
            runroot = args.small_root / suite / "runs"
            directories = sorted(p for p in runroot.iterdir() if p.is_dir()) if runroot.is_dir() else []
            counts[suite + "_discovered"] = len(directories)
            if len(directories) != expected_n:
                failures.append(dict(suite=suite, error="Expected exactly" + str(expected_n) + " jobs; found" + str(len(directories))))
            for directory in directories:
                try:
                    result, group, values = audit_job(suite, directory, index, master_train)
                    group_id, fold, query, target_all, labels = group
                    groupkey = (suite, group_id)
                    pooled = oof.setdefault(groupkey, dict(folds=set(), ids=set(), all=target_all, labels=labels, matrices={}))
                    require(fold not in pooled["folds"] and not query & pooled["ids"], "Repeated or overlapping outer validation fold")
                    require(pooled["all"] == target_all and pooled["labels"] == labels, "OOF target universe/class order changed")
                    pooled["folds"].add(fold)
                    pooled["ids"].update(query)
                    for key, value in values.items():
                        matrix = pooled["matrices"].setdefault(key, [[0] * len(labels) for _ in labels])
                        for i, row in enumerate(value["confusion_matrix"]):
                            for j, count in enumerate(row):
                                matrix[i][j] += count
                        settinglog.write(json.dumps(dict(suite=suite, job_id=directory.name, setting=key, metrics=value), ensure_ascii=False) + "\n")
                    counts[suite + "_passed"] += 1
                    counts["prediction_rows"] += result["rows"]
                    counts["settings_recounted"] += result["settings"]
                    counts["support_draws_checked"] += result["support_draws_checked"]
                except (OSError, ValueError, KeyError, TypeError, IndexError, EOFError, json.JSONDecodeError) as error:
                    result = dict(suite=suite, job_id=directory.name, status="FAIL", error=type(error).__name__ + ": " + str(error))
                    failures.append(result)
                joblog.write(json.dumps(result, ensure_ascii=False) + "\n")
                joblog.flush()
                write(progress, dict(status="RUNNING", counts=dict(counts), failures=len(failures), last_job=directory.name))
        with (args.output / "pooled_oof.jsonl").open("w", encoding="utf-8") as pooledlog:
            for (suite, group_id), pooled in sorted(oof.items()):
                ok = pooled["folds"] == set(range(5)) and pooled["ids"] == pooled["all"]
                if not ok:
                    failures.append(dict(suite=suite, group=group_id, error="Incomplete fivefold physical specimen coverage"))
                    continue
                counts["complete_oof_procedures"] += 1
                for key, matrix in pooled["matrices"].items():
                    value = metrics(matrix, pooled["labels"])
                    if value["n"] != len(pooled["ids"]):
                        failures.append(dict(suite=suite, group=group_id, setting=key,
                                             error="Pooled setting does not cover every OOF specimen once"))
                        continue
                    pooledlog.write(json.dumps(dict(suite=suite, group=group_id, setting=key, metrics=value), ensure_ascii=False) + "\n")
    if master_train is None:
        failures.append(dict(error="Missing frozen split_manifest.csv; master TRAIN allowlist could not be independently checked"))
    passed = not failures and all(counts[s + "_passed"] == n for s, n in EXPECTED.items())
    for path in sorted(args.output.iterdir()):
        if path.suffix == ".jsonl":
            outputs[path.name] = dict(sha256=sha(path), bytes=path.stat().st_size)
    result = dict(schema="research4-saved-species-independent-prediction-audit-20261112-v1",
                  status="COMPLETE" if passed else "INCOMPLETE_OR_FAILED", counts=dict(counts), failures=failures,
                  expected_jobs=EXPECTED, training_data_only=master_train is not None and passed,
                  master_split_file=str(master_files[0]) if master_files else None,
                  master_split_sha256=sha(master_files[0]) if master_files else None,
                  source_code_sha256=sha(__file__), output_artifacts=outputs,
                  process_cpu_seconds=time.process_time()-started,
                  completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  limitations=["Saved-output reproducibility audit, not independent external-batch or botanical authentication.",
                               "No models fitted, no threshold or candidate selected, no historical producer artifacts modified.",
                               "Support repetitions, model seeds and overlapping training sets do not increase physical specimen counts.",
                               "V4 calibration does not retain per-draw IDs, so their exact selection cannot be independently replayed from this archive."])
    write(args.output / "summary.json", result)
    write(progress, dict(status=result["status"], counts=dict(counts), failures=len(failures)))
    print(json.dumps({k: result[k] for k in ("status", "counts", "process_cpu_seconds")}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
