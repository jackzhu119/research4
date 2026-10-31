"""Independent neural pilot audit: no project metric/selection helpers or fits."""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "outputs/paper-fewshot-pilot-20261031"
OUTPUT = REPO / "outputs/independent_paper_pilot_audit.json"
checks = 0
maximum_difference = 0.0
failures = []


def check(condition, label):
    global checks
    checks += 1
    if not condition:
        failures.append(label)
        raise AssertionError(label)


def close(a, b, label, tolerance=1e-12):
    global maximum_difference
    delta = abs(float(a) - float(b))
    maximum_difference = max(maximum_difference, delta)
    check(delta <= tolerance, label)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str(tensor.dtype).encode())
        result.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        result.update(tensor.numpy().tobytes())
    return result.hexdigest()


def key(row):
    return int(row["seed"]), int(row["repeat"]), row["strategy"], row["method"]


def metrics(truth, predicted):
    matrix = [[0] * 7 for _ in range(7)]
    for actual, prediction in zip(truth, predicted):
        matrix[int(actual)][int(prediction)] += 1
    actual_count = [sum(row) for row in matrix]
    predicted_count = [sum(matrix[i][j] for i in range(7)) for j in range(7)]
    correct = sum(matrix[i][i] for i in range(7))
    recalls = [matrix[i][i] / actual_count[i] if actual_count[i] else 0. for i in range(7)]
    f1 = [2 * matrix[i][i] / (actual_count[i] + predicted_count[i])
          if actual_count[i] + predicted_count[i] else 0. for i in range(7)]
    return {
        "accuracy": correct / len(truth),
        "balanced_accuracy": statistics.fmean(recalls[i] for i in range(7) if actual_count[i]),
        "macro_f1": statistics.fmean(f1),
        "confusion_matrix": matrix,
        "class_recall": recalls,
        "class_counts": actual_count,
        "correct": correct,
        "total": len(truth),
    }


protocol_path = REPO / "data/research/paper_neural_classification_protocol_20261031.json"
protocol = read(protocol_path)
plan_path = REPO / "data/processed/zaoren_threeway_20261031_v1/split_plan.json"
plan = read(plan_path)
manifest = read(RUN / "run_manifest.json")
status = read(RUN / "status.json")
for relative, expected in manifest["source_sha256"].items():
    check(sha(REPO / relative) == expected, "source SHA " + relative)
check(sha(protocol_path) == manifest["protocol_sha256"], "protocol SHA")
check(sha(plan_path) == manifest["plan_sha256"], "split SHA")
source_npz = REPO / "outputs/approved-zaoren-20261031/data-preparation/dataset.npz"
check(sha(source_npz) == manifest["input_sha256"], "numerical input SHA")
with np.load(source_npz, allow_pickle=False) as data:
    ids = [str(v) for v in data["record_ids"]]
    true_by_id = dict(zip(ids, [str(v) for v in data["y"]]))
    physical_by_id = dict(zip(ids, [str(v) for v in data["groups"]]))
classes = sorted(set(true_by_id.values()))
vocabulary = {label: i for i, label in enumerate(classes)}
roles = {role: {row["record_id"] for row in plan["records"] if row["role"] == role}
         for role in ["training", "calibration", "prediction"]}
query_ids = sorted(roles["prediction"])
index = {identifier: i for i, identifier in enumerate(query_ids)}
truth = np.asarray([vocabulary[true_by_id[v]] for v in query_ids])
check(len(query_ids) == 171, "171 frozen query specimens")
check(set(classes) == {"ZY01", "ZY02", "ZY03", "ZY04", "ZY05", "ZY06", "ZY07"}, "seven anonymous classes")

stored_cases = {key(row): row for row in read(RUN / "final_metrics.json")}
predictions = {}
candidate_by_case = {}
seen = set()
raw_rows = 0
files = sorted((RUN / "final").glob("seed_*/predictions.csv.gz"))
check(len(files) == 3, "three seed CSV files")
for path in files:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_rows += 1
            case = key(row)
            identifier = row["record_id"]
            check(identifier in roles["prediction"], "prediction ID is reserved query")
            check(row["y_true"] == true_by_id[identifier], "CSV truth matches canonical NPZ")
            check(row["y_predicted"] in vocabulary, "prediction is a known class")
            check(int(row["is_correct"]) == int(row["y_true"] == row["y_predicted"]), "is_correct exact")
            sample_key = case + (identifier,)
            check(sample_key not in seen, "no duplicate sample in a case")
            seen.add(sample_key)
            if case not in predictions:
                predictions[case] = np.full(len(query_ids), -1, dtype=int)
                candidate_by_case[case] = row["candidate_id"]
            check(candidate_by_case[case] == row["candidate_id"], "case candidate fixed")
            predictions[case][index[identifier]] = vocabulary[row["y_predicted"]]
check(raw_rows == 615600 == status["prediction_rows"], "615600 raw prediction rows")
check(len(predictions) == 3600 == status["final_evaluations"], "3600 final cases")
check(set(predictions) == set(stored_cases), "all metric cases covered")
recomputed = {}
for case, predicted in predictions.items():
    check(np.all(predicted >= 0), "every fixed query ID scored")
    actual = metrics(truth, predicted)
    saved = stored_cases[case]
    check(actual["confusion_matrix"] == saved["confusion_matrix"], "confusion matrix")
    check(actual["class_counts"] == saved["class_counts"], "class counts")
    check(actual["correct"] == saved["correct"] and actual["total"] == saved["total"], "correct/total")
    check(candidate_by_case[case] == saved["candidate_id"], "metric candidate matches prediction")
    for name in ["accuracy", "balanced_accuracy", "macro_f1"]:
        close(actual[name], saved[name], name)
    for left, right in zip(actual["class_recall"], saved["class_recall"]):
        close(left, right, "class recall")
    baseline = predictions[(case[0], case[1], case[2], "zero")]
    base_metrics = metrics(truth, baseline)
    for name in ["accuracy", "balanced_accuracy", "macro_f1"]:
        actual[name + "_gain"] = actual[name] - base_metrics[name]
        close(actual[name + "_gain"], saved[name + "_gain"], "paired " + name)
    corrected = int(np.count_nonzero((baseline != truth) & (predicted == truth)))
    harmed = int(np.count_nonzero((baseline == truth) & (predicted != truth)))
    check(corrected == saved["corrected_errors"], "corrected errors")
    check(harmed == saved["new_errors"], "new errors")
    close((corrected-harmed)/len(truth), actual["accuracy_gain"], "paired count arithmetic")
    recomputed[case] = actual

for seed in protocol["seeds"]:
    for strategy in ["head_ft", "full_ft", "prototype", "fomaml_ft"]:
        reference = predictions[(seed, 0, strategy, "zero")]
        for repeat in range(100):
            check(np.array_equal(predictions[(seed, repeat, strategy, "zero")], reference),
                  "frozen zero predictor independent of support draw")

summaries = read(RUN / "summary.json")
check(len(summaries) == 12, "12 combined summaries")
per_seed = []
for saved in summaries:
    selected = [(case, value) for case, value in recomputed.items()
                if case[2] == saved["strategy"] and case[3] == saved["method"]]
    check(len(selected) == 300 == saved["evaluations"], "all three seeds and 100 draws")
    check(sorted({case[0] for case, _ in selected}) == protocol["seeds"], "no seed selection")
    for name in ["accuracy", "balanced_accuracy", "macro_f1"]:
        values = [value[name] for _, value in selected]
        gains = [value[name + "_gain"] for _, value in selected]
        for suffix, actual in [
            ("mean", statistics.fmean(values)),
            ("sd", statistics.pstdev(values)),
            ("min", min(values)),
            ("max", max(values)),
        ]:
            close(actual, saved[name + "_" + suffix], "summary " + name + "_" + suffix)
        close(statistics.fmean(gains), saved[name + "_gain_mean"], "summary gain")
    gain_values = [value["accuracy_gain"] for _, value in selected]
    check(sum(v > 0 for v in gain_values) == saved["positive_draws"], "positive draws")
    check(sum(v == 0 for v in gain_values) == saved["unchanged_draws"], "unchanged draws")
    check(sum(v < 0 for v in gain_values) == saved["negative_draws"], "negative draws")
    if saved["strategy"] == "fomaml_ft":
        for seed in protocol["seeds"]:
            cases = [value for case, value in selected if case[0] == seed]
            per_seed.append({"seed": seed, "method": saved["method"],
                             "accuracy_percent": 100 * statistics.fmean(r["accuracy"] for r in cases),
                             "gain_percentage_points": 100 * statistics.fmean(r["accuracy_gain"] for r in cases)})

dev_rows = read(RUN / "all_development_metrics.json")
check(len(dev_rows) == 2790, "2790 development comparisons")
grid = read(RUN / "candidate_grid.json")
check(len(grid) == 31, "31 parameter configurations")
check(hashlib.sha256(json.dumps(grid, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
      == protocol["candidate_grid_sha256"], "candidate grid SHA")
groups = defaultdict(list)
for row in dev_rows:
    groups[(row["strategy"], row["candidate_id"], row["stage"])].append(row)
averages = {
    k: {name: statistics.fmean(row[name] for row in values)
        for name in ["accuracy", "balanced_accuracy", "macro_f1",
                     "matched_accuracy", "matched_balanced_accuracy", "matched_macro_f1"]}
    for k, values in groups.items()
}
lock = read(RUN / "selection_lock.json")
independent_choices = {}
for strategy in ["head_ft", "full_ft", "prototype", "fomaml_ft"]:
    configs = [c for c in grid if c["strategy"] == strategy]
    ranked = sorted(configs, key=lambda c: (
        -averages[(strategy, c["id"], "selection")]["accuracy"],
        c.get("blend", 1.), c.get("steps", 0), c.get("learning_rate", 0.), c["id"]))
    def passes(c, stage):
        value = averages[(strategy, c["id"], stage)]
        rule = protocol["selection"]
        return (value["accuracy"]-value["matched_accuracy"] > rule["minimum_accuracy_gain"]
                and value["macro_f1"]-value["matched_macro_f1"] >= -rule["max_macro_f1_loss"]
                and value["balanced_accuracy"]-value["matched_balanced_accuracy"] >= -rule["max_balanced_accuracy_loss"])
    eligible = [c for c in ranked if passes(c, "selection")]
    diagnostic = eligible[0] if eligible else ranked[0]
    choice = diagnostic if eligible and passes(diagnostic, "gate") else None
    check(choice == lock["choices"][strategy]["selected"], "independent selected " + strategy)
    check(diagnostic == lock["choices"][strategy]["diagnostic"], "independent diagnostic " + strategy)
    independent_choices[strategy] = {
        "selected": None if choice is None else choice["id"],
        "diagnostic": diagnostic["id"],
        "eligible_selection_candidates": len(eligible),
    }

training_files = sorted((RUN / "development").glob("episode_*/seed_*/training.json"))
training_files += sorted((RUN / "final").glob("seed_*/training.json"))
check(len(training_files) == 21, "18 development and three final initializers")
meta_episode_manifests = 0
for path in training_files:
    info = read(path)
    source_ids = set(info["source_ids"])
    final = "final" in path.parts
    check(source_ids <= roles["training"], "all neural sources from original training role")
    check(len(source_ids) == (513 if final else 307), "source count")
    check(len(info["supervised_loss"]) == 40, "40 initializer epochs")
    check(info["source_optimization_budget"]["equal_FLOPs"] is False, "no equal FLOPs claim")
    shared = info["meta_training"]["supervised"]["episodes"]
    for strategy in ["supervised", "prototype", "fomaml"]:
        training = info["meta_training"][strategy]
        check(training["episodes"] == shared and len(shared) == 100, "same 100 source tasks")
        expected = [0, 1, 2, 3, 4, 5, 6] if final else [0, 2, 3, 4, 5, 6]
        check(training["policy"]["eligible_classes"] == expected, "dynamic source eligibility")
        check(training["first_order"] == (strategy == "fomaml"), "first-order designation")
        for episode in training["episodes"]:
            support = set(episode["support_ids"])
            query = set(episode["query_ids"])
            check(len(support) == 3 and len(query) == 3 and not support & query,
                  "source episode disjoint three support/three query")
            check(support | query <= source_ids, "source episode physical membership")
            check(not {physical_by_id[v] for v in support} & {physical_by_id[v] for v in query},
                  "source episode physical-group disjointness")
            check(Counter(vocabulary[true_by_id[v]] for v in support) == Counter(episode["classes"]),
                  "source episode one support per class")
            check(Counter(vocabulary[true_by_id[v]] for v in query) == Counter(episode["classes"]),
                  "source episode one query per class")
            meta_episode_manifests += 1
    for name, artifact in info["artifacts"].items():
        model_path = REPO / artifact["path"]
        check(sha(model_path) == artifact["sha256"], "saved CPU state SHA")
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        check(fingerprint(state) == artifact["fingerprint"], "independent saved-state fingerprint")
    if not final:
        record = read(path.with_name("record.json"))
        sets = {name: set(values) for name, values in record["partition"].items()}
        check(set.union(*sets.values()) == roles["training"], "complete development outer-training partition")
        check(all(not sets[a] & sets[b] for a, b in [
            ("source", "support_pool"), ("source", "query"), ("support_pool", "query")]),
            "disjoint development roles")
        for draw in record["support_draws"]:
            support = set(draw["ids"])
            check(support <= sets["support_pool"] and not support & sets["query"], "development calibration support")
            check(Counter(true_by_id[v] for v in support) == Counter(classes), "balanced development seven-class support")

metadata = read(RUN / "final_metadata.json")
check(len(metadata) == 3, "all final seeds preserved")
for entry in metadata:
    check(set(entry["source_ids"]) == roles["training"], "513 original source IDs")
    check(set(entry["query_ids"]) == roles["prediction"], "171 fixed prediction IDs")
    check(entry["fingerprints_before"] == entry["fingerprints_after"], "frozen final weights")
    check(len(entry["support_draws"]) == 100, "100 target support draws")
    for draw in entry["support_draws"]:
        support = set(draw["ids"])
        check(len(support) == 7 and support <= roles["calibration"], "seven dedicated support labels")
        check(not support & roles["prediction"] and not support & roles["training"], "target-role isolation")
        check(Counter(true_by_id[v] for v in support) == Counter(classes), "one target support per class")
    path = RUN / "final" / ("seed_" + str(entry["seed"])) / "predictions.csv.gz"
    check(sha(path) == entry["prediction_sha256"], "final CSV SHA")
    training = read(path.with_name("training.json"))
    for name in ["supervised", "prototype", "fomaml"]:
        check(entry["fingerprints_before"][name] == training["artifacts"][name]["fingerprint"],
              "same saved and adapted source state")
check(all(entry["support_draws"] == metadata[0]["support_draws"] for entry in metadata),
      "all seeds share identical target support draws")

random_ids = read(RUN / "random_query_ids.json")
expected_ids = np.random.default_rng(protocol["final"]["random_query_seed"]).choice(
    np.asarray(sorted(query_ids)), 20, replace=False).tolist()
check(random_ids == expected_ids, "random query IDs predeclared independently")
random_count = 0
random_errors = 0
with (RUN / "random_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
    for row in csv.DictReader(handle):
        random_count += 1
        case = key(row)
        check(row["record_id"] in random_ids and int(row["repeat"]) == 0, "random display membership")
        prediction = int(predictions[case][index[row["record_id"]]])
        check(row["y_predicted"] == classes[prediction], "random prediction matches full CSV")
        check(row["y_true"] == true_by_id[row["record_id"]], "random provided truth")
        check(int(row["is_correct"]) == int(row["y_true"] == row["y_predicted"]), "random correctness")
        random_errors += int(row["is_correct"]) == 0
check(random_count == 20 * 3 * 4 * 3 == 720, "all random rows, no correct-only filter")
check(sha(RUN / "selection_lock.json") == status["selection_lock_sha256"], "final choice lock SHA")
events = [json.loads(line) for line in (REPO / "outputs/paper-fewshot-pilot-20261031.log").read_text().splitlines()
          if line.startswith("{")]
lock_index = next(i for i, row in enumerate(events) if row.get("stage") == "choices_locked")
check(lock_index < min(i for i, row in enumerate(events) if row.get("stage") == "final_seed_completed"),
      "selection lock precedes final completion events")
check(status["paper_exact_reproduction"] is False and status["cross_batch_validation"] is False,
      "adaptation and within-batch status explicit")
check(status["query_labels_used_to_fit"] is False and status["query_statistics_fitted"] is False,
      "recorded query isolation; also checked in source review")

result = {
    "status": "pass",
    "auditor": "Independent /root/fewshot_paper_recheck_round2; no project metrics or selection helpers used",
    "checks_evaluated": checks,
    "failed_checks": failures,
    "prediction_files": len(files),
    "prediction_rows": raw_rows,
    "final_cases": len(predictions),
    "summary_rows": len(summaries),
    "development_rows": len(dev_rows),
    "maximum_numeric_difference": maximum_difference,
    "independently_recomputed_choices": independent_choices,
    "training_metadata_files_checked": len(training_files),
    "source_meta_episode_manifests_checked": meta_episode_manifests,
    "random_example_rows": random_count,
    "random_incorrect_rows_preserved": random_errors,
    "fomaml_per_seed_results": per_seed,
    "source_review": "outputs/paper-fewshot-pilot-review-20261031.json",
    "provenance": {
        "protocol_sha256": sha(protocol_path),
        "selection_lock_sha256": sha(RUN / "selection_lock.json"),
        "audit_script_sha256": sha(Path(__file__)),
        "source_sha256": manifest["source_sha256"],
    },
    "limits": [
        "No model fitting, no parameter/seed selection, no labels changed.",
        "Real uploaded spectra with anonymous groups; not independently verified geographic origins.",
        "Compact GroupNorm ResNet and first-order MAML are explicit classification adaptations, not exact author regression reproduction.",
        "Development meta episodes use six eligible source classes, final source uses all seven; rare class remains in supervised initialization.",
        "Same source task IDs and 100 continuation/meta outer updates, not equal FLOPs or identical objectives.",
        "Repeated target supports share fixed 171 query specimens; seeds/draws are not independent new batches.",
        "FO-MAML gains compare its own zero-support predictor; compare separately with the stronger teacher SVM baseline.",
        "Chronology uses recorded execution events and source/state hashes, not external execution attestation.",
    ],
}
OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: result[k] for k in ["status", "checks_evaluated", "prediction_rows", "final_cases",
                                        "maximum_numeric_difference", "random_example_rows",
                                        "random_incorrect_rows_preserved", "fomaml_per_seed_results"]},
                 ensure_ascii=False))

