"""Read-only arithmetic/lineage self-audit of the SVM parameter-adaptation run."""
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/multiround-20261031/support-svm"
checks = 0
maximum_difference = 0.


def check(value, label):
    global checks
    checks += 1
    if not value:
        raise AssertionError(label)


def close(actual, expected, label):
    global maximum_difference
    difference = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
    maximum_difference = max(maximum_difference, difference)
    check(difference < 1e-12, label)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(truth, predicted, classes):
    lookup = {str(c): i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for actual, guessed in zip(truth, predicted):
        cm[lookup[actual], lookup[guessed]] += 1
    diagonal = cm.diagonal()
    true_count, pred_count = cm.sum(1), cm.sum(0)
    recall = diagonal/true_count
    f1 = np.divide(2*diagonal, true_count+pred_count, out=np.zeros(len(classes)), where=true_count+pred_count > 0)
    return {"accuracy": diagonal.sum()/cm.sum(), "balanced_accuracy": recall.mean(),
            "macro_f1": f1.mean(), "confusion_matrix": cm, "class_recall": recall,
            "class_counts": true_count, "correct": diagonal.sum(), "total": cm.sum()}


with np.load(ROOT/"outputs/approved-zaoren-20261031/data-preparation/dataset.npz", allow_pickle=False) as arrays:
    original_labels = dict(zip(arrays["record_ids"].tolist(), arrays["y"].tolist()))
plan = read(ROOT/"data/processed/zaoren_threeway_20261031_v1/split_plan.json")
role_ids = {role: {r["record_id"] for r in plan["records"] if r["role"] == role}
            for role in ("training", "calibration", "prediction")}
manifest = read(RUN/"run_manifest.json")
for path, digest in manifest["source_sha256"].items():
    check(sha(ROOT/path) == digest, "executed code hash")
check(sha(ROOT/"data/research/support_svm_search_protocol_20261031.json") == manifest["protocol_sha256"], "protocol hash")
check(sha(ROOT/"data/processed/zaoren_threeway_20261031_v1/split_plan.json") == manifest["plan_sha256"], "split hash")
stage_rows = []
prediction_rows = 0
final_rows = 0
for directory in [*sorted((RUN/"development").glob("episode_*")), *([RUN/"final"] if (RUN/"final/metadata.json").exists() else [])]:
    metadata = read(directory/"metadata.json")
    final = metadata["stage"] == "final"
    source = set(metadata["source_ids"]); pool = set(metadata["support_pool_ids"]); query = set(metadata["query_ids"])
    check(not (source & pool or source & query or pool & query), "physical roles disjoint")
    check(source == role_ids["training"] if final else source | pool | query == role_ids["training"], "source roles restricted")
    if final:
        check(pool == role_ids["calibration"] and query == role_ids["prediction"], "fixed target roles")
    close(metadata["zero_refit_max_score_difference"], 0., "zero scores reproduce")
    check(metadata["source_fingerprint_before"] == metadata["source_fingerprint_after"], "source remains frozen")
    check(metadata["context_fingerprint_before"] == metadata["context_fingerprint_after"], "frozen transformed context")
    check(not metadata["final_query_labels_used_for_fit"] and not metadata["final_query_statistics_fitted"], "target boundary assertions")
    check(sha(directory/"predictions.csv.gz") == metadata["predictions_sha256"], "prediction SHA")
    check(sha(directory/"metrics.json") == metadata["metrics_sha256"], "metric SHA")
    for draw in metadata["draws"]:
        support = draw["support_ids"]
        check(len(set(support)) == 7 and set(support) <= pool, "7 unique supports from pool")
        check(len({original_labels[i] for i in support}) == 7, "one support for every class")
    original_weights = {c: len(source)/(7*sum(original_labels[i] == c for i in source)) for c in metadata["class_order"]}
    for c, value in original_weights.items():
        close(metadata["fixed_parameters"]["class_weight"][c], value, "classweight source only")
    grouped = defaultdict(list)
    with gzip.open(directory/"predictions.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            check(row["y_true"] == original_labels[row["record_id"]], "canonical original label")
            check(int(row["is_correct"]) == (row["y_true"] == row["y_predicted"]), "correctness flag")
            check(row["record_id"] in query, "query-only predictions")
            grouped[(int(row["repeat"]), row["method"])].append(row)
            prediction_rows += 1
            final_rows += int(final)
    metrics = read(directory/"metrics.json")
    for row in metrics:
        values = grouped[(row["repeat"], row["method"])]
        check({r["record_id"] for r in values} == query and len(values) == len(query), "complete per-case query IDs")
        calculated = scores([r["y_true"] for r in values], [r["y_predicted"] for r in values], metadata["class_order"])
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            close(row[metric], calculated[metric], "metric recomputed")
            close(row[metric+"_gain_vs_source"], row[metric]-metadata["source_baseline_metrics"][metric], "own-source gain")
            close(row["matched_"+metric], metadata["matched_zero_metrics"][metric], "matched zero")
        if final:
            for key in ("confusion_matrix", "class_recall", "class_counts", "correct", "total"):
                close(row[key], calculated[key], "detailed metric recomputed")
        else:
            stage_rows.append(row)
aggregate = read(RUN/"development_aggregate.json")
for stage in ("selection", "gate"):
    for identifier, values in aggregate[stage].items():
        rows = [r for r in stage_rows if r["stage"] == stage and r["candidate_id"] == identifier]
        check(len(rows) == values["evaluations"], "aggregate count")
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "matched_accuracy", "matched_balanced_accuracy", "matched_macro_f1"):
            close(values[metric], np.mean([r[metric] for r in rows]), "development mean")
grid = read(RUN/"candidate_grid.json")
identity = grid[0]


def allowed(candidate, stage):
    candidate, base = aggregate[stage][candidate["id"]], aggregate[stage][identity["id"]]
    return (candidate["accuracy"]-base["accuracy"] > .001
        and candidate["macro_f1"]-base["macro_f1"] >= -.005
        and candidate["balanced_accuracy"]-base["balanced_accuracy"] >= -.005
        and candidate["accuracy"]-candidate["matched_accuracy"] > .001
        and candidate["macro_f1"]-candidate["matched_macro_f1"] >= -.005
        and candidate["balanced_accuracy"]-candidate["matched_balanced_accuracy"] >= -.005)


def best(candidates):
    top = max(aggregate["selection"][c["id"]]["accuracy"] for c in candidates)
    return min([c for c in candidates if aggregate["selection"][c["id"]]["accuracy"] >= top-.001],
               key=lambda c: (c["blend"], c["support_weight"], c["id"]))


eligible = [c for c in grid if allowed(c, "selection")]
proposed = best(eligible) if eligible else identity
selected = proposed if proposed != identity and allowed(proposed, "gate") else identity
diagnostic = best(grid[1:])
lock = read(RUN/"selection_lock.json")
check(lock["choice"]["proposed"] == proposed, "independent proposed selection")
check(lock["choice"]["selected"] == selected, "independent gated selection")
check(lock["choice"]["diagnostic"] == diagnostic, "independent diagnostic selection")
check(lock["source_only_development"] and lock["final_query_not_scored_at_lock"], "lock boundary")
result = {"status": "pass", "scope": "self-audit; read-only arithmetic/lineage; no new fits",
    "checks_evaluated": checks, "maximum_numeric_difference": maximum_difference,
    "development_prediction_rows": prediction_rows-final_rows, "final_prediction_rows": final_rows,
    "development_cases_recomputed": len(stage_rows), "selection_lock_sha256": sha(RUN/"selection_lock.json"),
    "final_query_scored": final_rows > 0, "selected": selected, "diagnostic": diagnostic,
    "script_sha256": sha(Path(__file__)), "model_fit_calls": 0}
write = RUN/("arithmetic_and_lineage_audit.json" if final_rows else "development_arithmetic_audit.json")
write.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
print(json.dumps(result, ensure_ascii=False))
