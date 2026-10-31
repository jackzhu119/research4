"""Independent-formula numerical audit of the recorded fusion CSV files; no fitting."""
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
NPZ = ROOT / "outputs/approved-zaoren-20261031/data-preparation/dataset.npz"
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "nll", "brier", "ece")
checked = 0
maximum = 0.


def check(value, message):
    global checked
    checked += 1
    if not value:
        raise AssertionError(message)


def close(a, b, message):
    global maximum
    delta = abs(float(a)-float(b))
    maximum = max(maximum, delta)
    check(delta <= 1e-12, (message, a, b))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def formulas(y, predicted, p):
    cm = np.zeros((7, 7), dtype=int)
    for actual, inferred in zip(y, predicted):
        cm[actual, inferred] += 1
    sizes, calls, diagonal = cm.sum(1), cm.sum(0), cm.diagonal()
    recall = np.divide(diagonal, sizes, out=np.zeros(7), where=sizes > 0)
    f1 = np.divide(2*diagonal, sizes+calls, out=np.zeros(7), where=sizes+calls > 0)
    nll = float(np.mean([-np.log(max(1e-15, row[actual])) for actual, row in zip(y, p)]))
    brier = float(np.mean([sum((row[k]-int(k == actual))**2 for k in range(7)) for actual, row in zip(y, p)]))
    confidence = p.max(1)
    ece = 0.
    for index in range(10):
        mask = (confidence >= index/10) & (confidence < (index+1)/10 if index < 9 else confidence <= 1.)
        if mask.any():
            ece += mask.sum()/len(y)*abs((predicted[mask] == y[mask]).mean()-confidence[mask].mean())
    return {"accuracy": diagonal.sum()/len(y), "balanced_accuracy": recall[sizes > 0].mean(),
        "macro_f1": f1.mean(), "nll": nll, "brier": brier, "ece": ece,
        "confusion_matrix": cm.tolist(), "class_recall": recall.tolist(),
        "class_counts": sizes.tolist(), "correct": int(diagonal.sum()), "total": len(y)}


def key(row):
    return (str(row.get("episode") if row.get("episode") is not None else ""),
            int(row["seed"]), row["method"], row["candidate_id"])


def audit():
    arrays = np.load(NPZ, allow_pickle=False)
    truth = dict(zip(arrays["record_ids"].tolist(), arrays["y"].tolist()))
    cases, total_rows, classmaps = {}, 0, {}
    for phase in ("development", "final"):
        for path in sorted((RUN/phase).glob("**/predictions.csv.gz")):
            record = json.loads((path.parent/"evaluation.json").read_text())
            classes = record["class_order"]
            vocab = {c:i for i,c in enumerate(classes)}
            grouped = defaultdict(list)
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    check(row["record_id"] in truth and truth[row["record_id"]] == row["y_true"], "canonical truth")
                    check(int(row["target_support_labels_used"]) == 0, "no target labels")
                    check(int(row["is_correct"]) == int(row["y_true"] == row["y_predicted"]), "correctness")
                    p = np.asarray([float(row["p_"+c]) for c in classes])
                    check(np.isfinite(p).all() and np.all(p >= 0) and np.all(p <= 1), "finite probability")
                    close(p.sum(), 1., "probability normalization")
                    check(classes[int(p.argmax())] == row["y_predicted"], "probability argmax")
                    grouped[key(row)].append((row, p))
                    total_rows += 1
            saved = {key(row):row for row in record["rows"]}
            check(set(grouped) == set(saved), "all CSV cases saved")
            for k, members in grouped.items():
                y = np.asarray([vocab[r["y_true"]] for r,p in members])
                prediction = np.asarray([vocab[r["y_predicted"]] for r,p in members])
                p = np.asarray([p for r,p in members])
                value = formulas(y, prediction, p)
                for name in METRICS:
                    close(value[name], saved[k][name], (k, name))
                for name in ("confusion_matrix", "class_counts", "correct", "total"):
                    check(value[name] == saved[k][name], (k, name))
                check(np.allclose(value["class_recall"], saved[k]["class_recall"], atol=1e-14, rtol=0), "class recalls")
                cases[k] = {**saved[k], **value}
                classmaps[k] = {r["record_id"]:r["y_predicted"] for r,p in members}
            check(record["predictions_sha256"] == sha(path), "CSV hash")
            check(record["context_fingerprint_before"] == record["context_fingerprint_after"], "source context freeze")
            check(not set(record["source_ids"]) & set(record["query_ids"]), "physical role isolation")
            check(record["source_models_fitted_by_fusion"] == 0 and not record["query_statistics_fitted"], "no fitting")
    grid = json.loads((RUN/"candidate_grid.json").read_text())
    dev = json.loads((RUN/"all_development_metrics.json").read_text())
    check(len(dev) == 360, "18 contexts by20 candidates")
    dev_group = defaultdict(list)
    for row in dev:
        independent = cases[key(row)]
        for metric in METRICS:
            close(independent[metric], row[metric], "whole development table")
        dev_group[(row["stage"], row["candidate_id"])].append(independent)
    means = {stage: {c["id"]: {**{m:float(np.mean([r[m] for r in dev_group[(stage,c["id"])]])) for m in METRICS},
        "evaluations": len(dev_group[(stage,c["id"])])} for c in grid} for stage in ("selection", "gate")}
    policy = json.loads((RUN/"protocol_snapshot.json").read_text())["selection"]
    interior = [c for c in grid if 0 < c["svm_weight"] < 1]
    def endpoint(candidate, weight):
        return next(c["id"] for c in grid if c["mlp_provider"] == candidate["mlp_provider"] and
            c["probability"] == candidate["probability"] and c["svm_weight"] == weight)
    def allowed(candidate, stage):
        value = means[stage][candidate["id"]]
        return all(value["macro_f1"]-means[stage][endpoint(candidate,w)]["macro_f1"] > policy["minimum_macro_f1_gain"]
            and value["accuracy"]-means[stage][endpoint(candidate,w)]["accuracy"] >= -policy["max_accuracy_loss"]
            and value["balanced_accuracy"]-means[stage][endpoint(candidate,w)]["balanced_accuracy"] >= -policy["max_balanced_accuracy_loss"]
            for w in (0., 1.))
    def best(pool):
        top = max(means["selection"][c["id"]]["macro_f1"] for c in pool)
        near = [c for c in pool if means["selection"][c["id"]]["macro_f1"] >= top-policy["tie_tolerance"]]
        return min(near, key=lambda c:(-means["selection"][c["id"]]["accuracy"],abs(c["svm_weight"]-.5),c["mlp_provider"],c["probability"],c["id"]))
    eligible = [c for c in interior if allowed(c,"selection")]
    proposed = best(eligible) if eligible else None
    passed = proposed is not None and allowed(proposed,"gate")
    fallback = next(c for c in grid if c["mlp_provider"] == "mlp_sup100" and c["probability"] == "source_oof_temperature" and c["svm_weight"] == 0.)
    expected = {"selected":proposed if passed else fallback,"proposed":proposed,"diagnostic":best(interior),
        "eligible_selection_candidates":len(eligible),"passed_gate":passed,
        "fallback_to_prespecified_mlp_sup100":not passed,"selected_using_final_query":False}
    check(expected == json.loads((RUN/"selection_lock.json").read_text())["choice"], "independent selection")
    final_count = 0
    if (RUN/"final_metrics.json").exists():
        final = json.loads((RUN/"final_metrics.json").read_text())
        final_count = len(final)
        check(final_count == 84, "fixed finalcase count")
        final_group = defaultdict(list)
        for row in final:
            independent = cases[key(row)]
            for metric in METRICS:close(row[metric], independent[metric], "whole finaltable")
            final_group[row["method"]].append(independent)
        for row in json.loads((RUN/"summary.json").read_text()):
            values = final_group[row["method"]]
            for metric in METRICS:
                close(row[metric+"_mean"], np.mean([r[metric] for r in values]), "summarymean")
                close(row[metric+"_sd"], np.std([r[metric] for r in values],ddof=1), "summarysd")
    result = {"status":"pass","checks_evaluated":checked,"maximum_numeric_difference":maximum,
        "prediction_rows":total_rows,"development_cases":len(dev),"final_cases":final_count,
        "independently_recomputed_choice":expected,"new_fits":0,"script_sha256":sha(Path(__file__)),
        "limits":["Same-batch historical anonymous labels; not externally verified origins.",
            "Score providers and OOF fitting audited upstream; this audit rechecks fusion arithmetic and records."]}
    destination = RUN/("arithmetic_audit.json" if final_count else "development_arithmetic_audit.json")
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))


if __name__ == "__main__":
    audit()
