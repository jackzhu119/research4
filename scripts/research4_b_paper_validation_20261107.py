#!/usr/bin/env python3
"""Bounded, proof-gated diagnostics of existing Paper-C OOF predictions.

No fitting, hyperparameter selection, new predictions, or quantum calculations.
The seven historical McNemar comparisons are replayed with their original Holm
family, not presented as seven new experiments. New outputs are marginal recall
intervals and descriptive record/class/fold error agreement.
"""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
import time
from collections import Counter
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
REQUEST_SCHEMA = "research4-b-paper-validation-request-v1"
SCHEMA = "research4-b-paper-validation-result-v1"
METHODS = ("R0", "R1", "R2", "R3", "R4")
CLASSES = tuple(f"ZY{i:02d}" for i in range(1, 8))
COUNTS = dict(zip(CLASSES, (350, 5, 100, 100, 100, 100, 100)))
PAIRS = (("R1", "R0"), ("R2", "R0"), ("R3", "R0"), ("R4", "R0"),
         ("R2", "R1"), ("R3", "R2"), ("R4", "R2"))
LIMITATIONS = [
    "All 855 specimens are historically exposed development data from one lot per origin; no new independent cohort, cross-batch test, or model validation is created.",
    "Clopper-Pearson exact refers to the binomial calculation. Nominal 95% intervals are marginal per class/method, not simultaneous, and assume independent Bernoulli trials. Shared OOF training sets and historical selection prevent a guaranteed generalization-coverage interpretation.",
    "Saved OOF predictions condition these diagnostics; training, tuning, seed, feature-selection, and model-selection uncertainty are not included.",
    "The seven exact McNemar/Holm results replay the previously published seven-comparison family. They are nominal conditional diagnostics with overlapping OOF training, not a new confirmatory significance test.",
    "Error-set overlaps and class/fold tables are descriptive; no sample, model, seed, feature, or threshold is selected from these results.",
    "No perturbation robustness, learning curve, independent batch, instrument, moisture, thickness, chemical assay, or standard-compound THz measurement is performed by this worker.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def path(value):
    p = Path(value)
    if not p.is_absolute():
        p = ROOT / p
    p = p.resolve()
    require(p.is_relative_to(ROOT), "Only the original B repository is permitted")
    return p


def checked(record):
    p = path(record["path"])
    require(p.is_file() and p.stat().st_size <= 5 * 1024**2, "Bounded actual input file required")
    require(sha(p) == record["sha256"], "Input SHA256 differs")
    if "bytes" in record:
        require(p.stat().st_size == record["bytes"], "Input byte count differs")
    return p


def ref(p):
    p = path(p)
    return {"path": str(p.relative_to(ROOT)), "sha256": sha(p), "bytes": p.stat().st_size}


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(p, data):
    with p.open("x") as h:
        json.dump(data, h, ensure_ascii=False, indent=2, allow_nan=False)
        h.write("\n")


def write_csv(p, rows):
    require(rows, "CSV rows required")
    with p.open("x", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def binomial_cdf(k, n, probability):
    if k < 0:
        return 0.
    if k >= n:
        return 1.
    if probability <= 0:
        return 1.
    if probability >= 1:
        return 0.
    terms = [math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)
             + i*math.log(probability)+(n-i)*math.log1p(-probability)
             for i in range(k+1)]
    highest = max(terms)
    return min(1., math.exp(highest)*math.fsum(math.exp(v-highest) for v in terms))


def recall_interval(successes, total, alpha=.05):
    """Two-sided marginal Clopper-Pearson binomial interval by inversion."""
    require(type(successes) is int and type(total) is int and 0 <= successes <= total and total > 0,
            "Valid binomial successes/total required")
    lower, upper = 0., 1.
    if successes:
        low, high = 0., 1.
        for _ in range(64):
            middle = (low+high)/2
            # P(X>=k)=P(n-X<=n-k) avoids 1-CDF cancellation.
            tail = binomial_cdf(total-successes, total, 1-middle)
            if tail < alpha/2:
                low = middle
            else:
                high = middle
        lower = (low+high)/2
    if successes < total:
        low, high = 0., 1.
        for _ in range(64):
            middle = (low+high)/2
            if binomial_cdf(successes, total, middle) > alpha/2:
                low = middle
            else:
                high = middle
        upper = (low+high)/2
    return lower, upper


def exact_mcnemar(b, c):
    return min(1., 2*binomial_cdf(min(b, c), b+c, .5)) if b+c else 1.


def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted, running = [0.]*len(values), 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., (len(values)-rank)*values[index]))
        adjusted[index] = running
    return adjusted


def validate_request(request):
    require(request.get("schema") == REQUEST_SCHEMA, "Wrong paper request schema")
    require(request.get("operation") == "existing_prediction_diagnostics", "Only predeclared prediction diagnostics allowed")
    require(request.get("methods") == list(METHODS) and request.get("class_counts") == COUNTS,
            "Frozen methods/class support required")
    require(request.get("threads") == 2, "Same B two-thread allocation required")
    require(request.get("nominal_confidence_level") == .95, "Predeclared marginal 95% intervals required")
    required = ("core_predictions", "combinations_predictions", "prior_summary", "prior_mcnemar", "splits", "input_provenance")
    require(set(request.get("inputs", {})) == set(required), "Explicit frozen input set required")
    for record in request["inputs"].values():
        checked(record)
    validate_provenance(request["inputs"])


def validate_provenance(inputs):
    provenance = json.loads(checked(inputs["input_provenance"]).read_text())
    require(provenance.get("schema") == "research4-b-paper-input-provenance-v1", "Wrong paper input provenance schema")
    source_commit = provenance.get("source_commit", "")
    require(len(source_commit) == 40 and all(c in "0123456789abcdef" for c in source_commit), "Exact source commit required")
    records = provenance.get("files", [])
    lookup = {str(path(record["local"])): record for record in records}
    expected = {str(path(record["path"])): record for key, record in inputs.items() if key != "input_provenance"}
    require(len(lookup) == len(records) == len(expected) and set(lookup) == set(expected), "Provenance file identities differ from request inputs")
    for local, request_record in expected.items():
        record = lookup[local]
        require(record["sha256"] == request_record["sha256"] and record["bytes"] == request_record["bytes"], "Provenance input digest/size differs")
        require(record.get("git_source_commit") == source_commit, "Provenance source commits differ")
        blob = record.get("git_blob", "")
        require(len(blob) == 40 and all(c in "0123456789abcdef" for c in blob), "Exact source Git blob required")
        source_path = Path(record["git_source_path"])
        require(not source_path.is_absolute() and ".." not in source_path.parts, "Repository-relative source path required")


def load_rows(request):
    methods = {m: {} for m in METHODS}
    for key, allowed in (("core_predictions", set(METHODS)-{"R4"}), ("combinations_predictions", {"R4"})):
        with gzip.open(checked(request["inputs"][key]), "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                method = row["method"]
                if method not in allowed:
                    continue
                rid = row["record_id"]
                require(rid not in methods[method], "Duplicate method/record prediction")
                require(row["y_true"] in CLASSES and row["y_predicted"] in CLASSES, "Anonymous registered classes required")
                correct = int(row["y_true"] == row["y_predicted"])
                require(int(row["is_correct"]) == correct, "Saved correctness differs from labels")
                methods[method][rid] = {"record_id": rid, "truth": row["y_true"], "prediction": row["y_predicted"],
                                        "fold": int(row["outer_fold"]), "correct": correct}
    reference = methods["R2"]
    require(len(reference) == 855, "Exactly 855 unique held-out observations required")
    for method, records in methods.items():
        require(set(records) == set(reference), "Method record identities differ")
        require(dict(Counter(r["truth"] for r in records.values())) == COUNTS, "Class supports differ")
        require(dict(Counter(r["fold"] for r in records.values())) == dict.fromkeys(range(5), 171), "Expected five 171-observation outer folds")
        for rid, row in records.items():
            require((row["truth"], row["fold"]) == (reference[rid]["truth"], reference[rid]["fold"]), "Truth/fold alignment differs")
    splits = json.loads(checked(request["inputs"]["splits"]).read_text())
    require(len(splits) == 5, "Saved five outer folds required")
    heldout = set()
    for split in splits:
        train, test = set(split["train_sample_ids"]), set(split["test_sample_ids"])
        require(len(train) == 684 and len(test) == 171 and not train & test and train | test == set(reference), "Saved fold partition invalid")
        require(not heldout & test, "Outer test identities repeat")
        heldout |= test
        require(test == {rid for rid, row in reference.items() if row["fold"] == split["fold"]}, "Prediction folds differ from saved source/test partition")
    return methods


def metrics(records):
    rows = list(records)
    total = len(rows)
    require(total > 0, "Nonempty record scope required")
    class_rows, f1s, recalls = [], [], []
    for label in CLASSES:
        support = sum(row["truth"] == label for row in rows)
        predicted = sum(row["prediction"] == label for row in rows)
        tp = sum(row["truth"] == label and row["prediction"] == label for row in rows)
        recall = tp/support if support else None
        precision = tp/predicted if predicted else 0.
        f1 = 2*tp/(support+predicted) if support+predicted else 0.
        if recall is not None:
            recalls.append(recall)
            f1s.append(f1)
        class_rows.append({"class": label, "support": support, "true_positive": tp, "predicted_count": predicted,
                           "recall": recall, "precision": precision, "F1": f1})
    return {"N": total, "correct": sum(row["correct"] for row in rows),
            "accuracy": sum(row["correct"] for row in rows)/total,
            "balanced_accuracy": math.fsum(recalls)/len(recalls), "macro_F1": math.fsum(f1s)/len(f1s)}, class_rows


def analyze(request, output):
    methods = load_rows(request)
    ids = sorted(methods["R2"])
    prior = json.loads(checked(request["inputs"]["prior_summary"]).read_text())
    require(prior["samples"] == 855 and prior["class_counts"] == COUNTS and prior["models_fitted"] == 0,
            "Prior evidence provenance differs")
    for name, expected in (("core_predictions", "reports/staged-nested-optimization/2026-10-31/core/outer_predictions.csv.gz"),
                           ("combinations_predictions", "reports/staged-nested-optimization/2026-10-31/combinations/outer_predictions.csv.gz")):
        require(prior["source_files"][expected]["sha256"] == request["inputs"][name]["sha256"], "New inputs differ from prior evidence input bytes")
    overall, intervals, fold_metrics, confusion = [], [], [], []
    for method in METHODS:
        metric, per_class = metrics(methods[method].values())
        overall.append({"method": method, **metric})
        for row in per_class:
            lower, upper = recall_interval(row["true_positive"], row["support"])
            intervals.append({"method": method, **row, "nominal_marginal_CP95_lower": lower, "nominal_marginal_CP95_upper": upper})
        for fold in range(5):
            subset = [r for r in methods[method].values() if r["fold"] == fold]
            metric, per_class = metrics(subset)
            for row in per_class:
                fold_metrics.append({"method": method, "outer_fold": fold, **metric, **row})
        counts = Counter((r["truth"], r["prediction"]) for r in methods[method].values())
        confusion.extend({"method": method, "truth": t, "prediction": p, "count": counts[t, p]} for t in CLASSES for p in CLASSES)
    expected_correct = (773, 791, 821, 816, 812)
    require(tuple(r["correct"] for r in overall) == expected_correct, "Historical correct-count qualification differs")
    mcnemar = []
    for method, baseline in PAIRS:
        transitions = Counter((methods[baseline][rid]["correct"], methods[method][rid]["correct"]) for rid in ids)
        b, c = transitions[0, 1], transitions[1, 0]
        mcnemar.append({"method": method, "baseline": baseline, "baseline_wrong_method_correct": b,
                        "baseline_correct_method_wrong": c, "both_correct": transitions[1, 1], "both_wrong": transitions[0, 0],
                        "nominal_exact_two_sided_p": exact_mcnemar(b, c)})
    for row, adjusted in zip(mcnemar, holm([r["nominal_exact_two_sided_p"] for r in mcnemar])):
        row["nominal_Holm_adjusted_p_seven_comparisons"] = adjusted
    with checked(request["inputs"]["prior_mcnemar"]).open(newline="") as h:
        old_rows = list(csv.DictReader(h))
    require(len(old_rows) == len(mcnemar), "Original seven-comparison family required")
    for new, old in zip(mcnemar, old_rows):
        require((new["method"], new["baseline"]) == (old["method"], old["baseline"]), "Original pair order/family differs")
        for key, value in new.items():
            if key not in ("method", "baseline"):
                require(abs(float(old[key])-value) <= 1e-12, "Independent historical McNemar/Holm replay mismatch")
    agreement = []
    for left, right in itertools.combinations(METHODS, 2):
        for scope, subset in [("all", ids)]+[(label, [rid for rid in ids if methods["R2"][rid]["truth"] == label]) for label in CLASSES]:
            a = {rid for rid in subset if not methods[left][rid]["correct"]}
            b = {rid for rid in subset if not methods[right][rid]["correct"]}
            agreement.append({"left": left, "right": right, "scope": scope, "N": len(subset), "left_errors": len(a),
                              "right_errors": len(b), "both_errors": len(a & b), "either_error": len(a | b),
                              "error_Jaccard": len(a & b)/len(a | b) if a | b else None,
                              "different_predicted_labels": sum(methods[left][rid]["prediction"] != methods[right][rid]["prediction"] for rid in subset)})
    error_records = []
    for rid in ids:
        row = methods["R2"][rid]
        wrong = [method for method in METHODS if not methods[method][rid]["correct"]]
        error_records.append({"record_id": rid, "truth": row["truth"], "outer_fold": row["fold"], "wrong_method_count": len(wrong),
                              "wrong_methods": ";".join(wrong), **{method+"_prediction": methods[method][rid]["prediction"] for method in METHODS}})
    outputs = {"method_metrics.csv": overall, "marginal_class_recall_intervals.csv": intervals,
               "fold_class_metrics.csv": fold_metrics, "confusion_counts.csv": confusion,
               "historical_seven_mcnemar_replay.csv": mcnemar, "error_set_agreement.csv": agreement,
               "record_error_consistency.csv": error_records}
    for filename, rows in outputs.items():
        write_csv(output/filename, rows)
    minority = next(row for row in intervals if row["method"] == "R2" and row["class"] == "ZY02")
    return {"status": "DONE_CONDITIONAL_PREDICTION_DIAGNOSTICS", "models_fitted": 0, "new_predictions_generated": 0,
            "quantum_evaluations": 0, "samples": 855, "methods": list(METHODS), "class_counts": COUNTS,
            "historical_correct_counts_replayed": dict(zip(METHODS, expected_correct)),
            "majority_class_only_baseline": {"majority_class": "ZY01", "accuracy": 350/855, "balanced_accuracy": 1/7,
                                             "scope": "Descriptive fixed majority-rule comparator; not fitted or selected from new evaluation"},
            "R2_ZY02_nominal_recall_interval": minority, "historical_seven_McNemar_Holm_replay_matches_1e_12": True,
            "new_analyses": ["35 marginal class-recall binomial intervals", "class/fold confusion and error consistency", "pairwise error-set overlaps"],
            "reused_existing_analyses": ["Historical seven McNemar/Holm comparisons replayed; not counted as newly designed tests"],
            "persistent_wrong_all_five_methods": sum(row["wrong_method_count"] == 5 for row in error_records),
            "correct_all_five_methods": sum(row["wrong_method_count"] == 0 for row in error_records),
            "limitations": LIMITATIONS,
            "中文结论": "本次只补充既有855份开发样本的少数类不确定性和错误一致性描述，重放原七项配对检验；没有新模型或新独立验证。ZY02仅5份，整体96.0234%不能代替该类的召回率与证据不足。",
            "output_artifacts": [ref(output/name) for name in outputs]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    request_path = checked({"path": args.request, "sha256": args.request_sha256})
    request = json.loads(request_path.read_text())
    validate_request(request)
    if not args.execute:
        print(json.dumps({"status": "NOT_RUN_CHECK_ONLY", "request": ref(request_path), "models_fitted": 0, "quantum_evaluations": 0}))
        return
    supervisor_path = ROOT/"scripts/research4_b_validation_supervisor_20261107.py"
    spec = importlib.util.spec_from_file_location("research4_b_validation_supervisor_paper", supervisor_path)
    supervisor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervisor)
    proof = supervisor.verify_resource_supervision(expected_worker_path=SELF, expected_threads=2)
    started, wall, cpu = utc(), time.monotonic(), time.process_time()
    output = path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), "Unique empty paper output directory required")
    result = analyze(request, output)
    result.update(schema=SCHEMA, operation=request["operation"], request=ref(request_path), source_worker=ref(SELF),
                  resource_supervision=proof, output_directory=str(output), started_at_utc=started, completed_at_utc=utc(),
                  actual_process_CPU_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-wall,
                  original_science_modified=False, classifier_modified=False, IR_intensities_status="NOT_COMPUTED",
                  frozen_inputs=request["inputs"])
    save(output/"result.json", result)
    print(json.dumps({"status": result["status"], "samples": 855, "models_fitted": 0, "quantum_evaluations": 0}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED_PRESERVED", "reason": str(error)}), file=sys.stderr)
        raise SystemExit(2)
