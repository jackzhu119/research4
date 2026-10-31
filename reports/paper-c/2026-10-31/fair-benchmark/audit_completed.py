"""Read-only count/identity/source-statistic/model replay audit; never fits."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_predictions(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def evidence_bytes(path):
    path = Path(path)
    if path.exists():
        return path.read_bytes()
    return gzip.decompress(path.with_name(path.name+".gz").read_bytes())


def independently_count(rows, classes):
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    mapping = {c:i for i,c in enumerate(classes)}
    for row in rows:
        cm[mapping[row["y_true"]], mapping[row["y_predicted"]]] += 1
    tp = cm.diagonal()
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    recall = tp / support
    precision = np.divide(tp, predicted, out=np.zeros(len(classes)), where=predicted>0)
    f1 = np.divide(2*tp, support+predicted, out=np.zeros(len(classes)), where=support+predicted>0)
    return {"accuracy":float(tp.sum()/cm.sum()), "balanced_accuracy":float(recall.mean()),
            "macro_f1":float(f1.mean()), "correct":int(tp.sum()), "total":int(cm.sum()),
            "confusion_matrix":cm.tolist(), "per_class":{c:{"precision":float(p), "recall":float(r), "f1":float(f), "support":int(n)}
                for c,p,r,f,n in zip(classes, precision, recall, f1, support)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()
    run, protocol_path = args.run, args.protocol
    protocol = json.loads(protocol_path.read_text())
    manifest = json.loads((run/"run_manifest.json").read_text())
    lock = json.loads((run/"selection_lock.json").read_text())
    arrays = np.load(protocol["input"], allow_pickle=False)
    ids, y, X = arrays["groups"].astype(str), arrays["y"].astype(str), arrays["X_alpha"]
    index = {rid:i for i,rid in enumerate(ids)}
    truth_by_id = dict(zip(ids, y))
    classes = sorted(set(y))
    checks, maximum = 0, 0.

    def exact(a, b):
        nonlocal checks
        assert a == b, (a, b)
        checks += 1

    def numeric(a, b, tolerance=1e-11):
        nonlocal checks, maximum
        a, b = np.asarray(a), np.asarray(b)
        difference = float(np.max(np.abs(a-b))) if a.size else 0.
        assert difference <= tolerance, difference
        maximum = max(maximum, difference)
        checks += a.size

    exact(sha(protocol_path), manifest["protocol_sha256"])
    exact(sha(protocol["input"]), manifest["input_sha256"])
    exact(sha(protocol["historical_protocol"]), protocol["historical_protocol_sha256"])
    exact(sha(protocol["historical_selection_lock"]), protocol["historical_selection_lock_sha256"])
    for path, expected in manifest["source_sha256"].items():
        exact(sha(path), expected)
    exact(sha(run/"run_manifest.json"), lock["manifest_sha256"])
    exact(hashlib.sha256(evidence_bytes(run/"all_inner_metrics.json")).hexdigest(), lock["inner_metrics_sha256"])
    exact(sha(run/"inner_predictions.csv.gz"), lock["inner_predictions_sha256"])
    splits = json.loads((run/"splits.json").read_text())
    inner_rows = json.loads(evidence_bytes(run/"all_inner_metrics.json"))
    grouped = defaultdict(list)
    for row in read_predictions(run/"inner_predictions.csv.gz"):
        key = (int(row["outer_fold"]), int(row["inner_fold"]), row["family"], row["candidate_id"])
        exact(row["y_true"], truth_by_id[row["record_id"]])
        grouped[key].append(row)
    completed = 0
    for row in inner_rows:
        outer, inner = row["outer_fold"], row["inner_fold"]
        source, valid = row["source_ids"], row["validation_ids"]
        exact(set(source), set(splits[outer]["inner_folds"][inner]["train_sample_ids"]))
        exact(set(valid), set(splits[outer]["inner_folds"][inner]["validation_sample_ids"]))
        exact(bool(set(source)&set(valid)), False)
        exact(bool(set(source+valid)&set(splits[outer]["test_sample_ids"])), False)
        if row["status"] != "ok":
            continue
        found = grouped[(outer, inner, row["family"], row["candidate_id"])]
        exact(len(found), 342)
        exact(set(r["record_id"] for r in found), set(valid))
        counted = independently_count(found, classes)
        for key in ("accuracy", "macro_f1", "balanced_accuracy", "correct", "total", "confusion_matrix"):
            numeric(counted[key], row["metrics"][key])
        src = X[[index[i] for i in source]]
        snv = (src-src.mean(axis=1, keepdims=True))/src.std(axis=1, ddof=1, keepdims=True)
        numeric(snv.mean(axis=0), row["model_metadata"]["scaler_mean"])
        numeric(snv.var(axis=0), row["model_metadata"]["scaler_variance"])
        if row["family"] == "mlp":
            exact(row["model_metadata"]["epochs"], 200)
            exact(len(row["model_metadata"]["training_loss_curve"]), 200)
        completed += 1
    for outer in lock["outer_folds"]:
        for family, choice in outer["choices"].items():
            candidates = protocol["candidate_grids"][family]
            means = {}
            for c in candidates:
                found = [r for r in inner_rows if r["outer_fold"] == outer["outer_fold"] and r["family"] == family and r["candidate_id"] == c["id"]]
                if len(found) == 2 and all(r["status"] == "ok" for r in found):
                    means[c["id"]] = {m:float(np.mean([r["metrics"][m] for r in found])) for m in ("macro_f1","balanced_accuracy","accuracy")}
            chosen = min((c for c in candidates if c["id"] in means), key=lambda c:(-means[c["id"]]["macro_f1"],-means[c["id"]]["balanced_accuracy"],-means[c["id"]]["accuracy"],c["id"]))
            exact(chosen, choice["selected"])
            exact(means, choice["all_candidate_means"])
    replayed, original_matched = 0, 0
    if not args.selection_only:
        outer_rows = json.loads((run/"outer_metrics.json").read_text())
        predictions = read_predictions(run/"outer_predictions.csv.gz")
        by_method = defaultdict(list)
        by_fold_method = defaultdict(list)
        for row in predictions:
            exact(row["y_true"], truth_by_id[row["record_id"]])
            by_method[row["method"]].append(row)
            by_fold_method[(int(row["outer_fold"]), row["method"])].append(row)
        pooled = json.loads((run/"pooled_metrics.json").read_text())
        for method, records in by_method.items():
            exact(len(records), 855)
            exact(len(set(r["record_id"] for r in records)), 855)
            counted = independently_count(records, classes)
            for key, value in counted.items():
                if key == "per_class":
                    for c in classes:
                        for subkey in value[c]:
                            numeric(value[c][subkey], pooled[method][key][c][subkey])
                else:
                    numeric(value, pooled[method][key])
        for row in outer_rows:
            path = Path(row["model_path"])
            if not path.exists():
                path = run/"models"/f"outer_{row['outer_fold']:02d}"/(row["method"]+".joblib")
            before = (sha(path), path.stat().st_mtime_ns)
            exact(before[0], row["model_sha256"])
            model = joblib.load(path)
            source = row["source_ids"]
            test = row["test_ids"]
            exact(set(source), set(splits[row["outer_fold"]]["train_sample_ids"]))
            exact(set(test), set(splits[row["outer_fold"]]["test_sample_ids"]))
            found = by_fold_method[(row["outer_fold"], row["method"])]
            exact([r["record_id"] for r in found], test)
            xquery = X[[index[i] for i in test]]
            exact(model.predict(xquery).tolist(), [r["y_predicted"] for r in found])
            numeric(model.scores(xquery), [[float(r["score_"+c]) for c in classes] for r in found])
            exact((sha(path), path.stat().st_mtime_ns), before)
            replayed += 1
        original = {r["record_id"]:r for r in read_predictions(Path(protocol["historical_selection_lock"]).parent/"outer_predictions.csv.gz") if r["method"] == "R2"}
        for row in by_method["R2_reproduction"]:
            old = original[row["record_id"]]
            exact(row["y_predicted"], old["y_predicted"])
            numeric([float(row["score_"+c]) for c in classes], [float(old["score_"+c]) for c in classes])
            original_matched += 1
    result = {"status":"pass", "checks":int(checks), "maximum_numeric_difference":maximum,
        "inner_cases":len(inner_rows), "completed_inner_cases":completed, "inner_raw_predictions":sum(map(len, grouped.values())),
        "outer_model_replays":replayed, "original_R2_labels_scores_matched":original_matched,
        "new_model_fits":0, "selection_only":args.selection_only}
    outfile = run/("selection_arithmetic_audit.json" if args.selection_only else "completed_arithmetic_audit.json")
    outfile.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
