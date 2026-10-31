"""Read-only replay of all archived strong-MLP source contexts; zero fits."""
import csv
import gzip
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from herb_spectra.neural_base_search import DenseClassifier, _snv, model_scores, transform_features
from herb_spectra.paper_c_strong_mlp import METHODS, load_context
from herb_spectra.paper_fewshot_classification import state_fingerprint
from herb_spectra.staged_nested_search import digest


ROOT = Path(__file__).resolve().parents[4]
REPORT = Path(__file__).resolve().parent
DATA = ROOT / "outputs/approved-zaoren-20261031/datasets/candidate_0p1_1p5.npz"


def counts_metrics(truth, prediction, classes):
    matrix = np.zeros((7, 7), dtype=int)
    for actual, found in zip(truth, prediction, strict=True):
        matrix[classes.index(str(actual)), classes.index(str(found))] += 1
    tp = matrix.diagonal(); n, p = matrix.sum(1), matrix.sum(0)
    recall = np.divide(tp, n, out=np.zeros(7), where=n > 0)
    f1 = np.divide(2 * tp, n + p, out=np.zeros(7), where=n + p > 0)
    return {"correct": int(tp.sum()), "accuracy": float(tp.sum() / matrix.sum()), "macro_f1": float(f1.mean()),
            "balanced_accuracy": float(recall.mean()), "confusion_matrix": matrix.tolist()}


def main():
    torch.set_num_threads(1)
    with np.load(DATA, allow_pickle=False) as handle:
        x, y, ids = handle["X_alpha"], handle["y"], handle["record_ids"]
    positions = {str(rid): i for i, rid in enumerate(ids)}
    classes = sorted(set(y.tolist()))
    manifest = json.loads((REPORT / "run_manifest.json").read_text())
    checks, maximum = 0, 0.

    def check(condition):
        nonlocal checks
        assert condition
        checks += 1

    def close(left, right, atol=1e-10):
        nonlocal checks, maximum
        left, right = np.asarray(left), np.asarray(right)
        check(left.shape == right.shape)
        difference = float(np.max(np.abs(left - right))) if left.size else 0.
        maximum = max(maximum, difference)
        check(difference <= atol)

    check(digest(DATA) == manifest["input_sha256"])
    for relative, expected in manifest["source_sha256"].items():
        check(digest(ROOT / relative) == expected)
    prediction_rows = {}
    for phase, filename in [("selection", "inner_predictions.csv.gz"), ("models", "outer_predictions.csv.gz")]:
        with gzip.open(REPORT / filename, "rt", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            key = (phase, int(row["outer_fold"]), int(row["seed"]), int(row.get("inner_fold", -1)), row["method"], row["record_id"])
            check(key not in prediction_rows)
            prediction_rows[key] = row
    contexts, task_count, replayed = 0, 0, 0
    for phase in ("selection", "models"):
        for path in sorted((REPORT / phase).rglob("training.json")):
            prep, models, metadata = load_context(path.parent)
            source_ids = metadata["source_ids"]
            query_ids = metadata["validation_ids"] if phase == "selection" else metadata["heldout_ids"]
            train = [positions[rid] for rid in source_ids]
            query = [positions[rid] for rid in query_ids]
            check(set(source_ids).isdisjoint(query_ids))
            check(metadata["source_fingerprint"] == joblib.hash(x[train]))
            check(metadata["label_fingerprint"] == joblib.hash(y[train]))
            check(metadata["class_order"] == classes)
            check(metadata["source_samples"] == (342 if phase == "selection" else 684))
            check(len(query) == (342 if phase == "selection" else 171))
            check(metadata["base_epochs"] == len(metadata["base_losses"]) == 200)
            check(metadata["continuation_updates_not_epochs"] == 100)
            check(metadata["continuation_seed"] == metadata["model_seed"] + 1600)
            check(metadata["parameter_count"] == 41543)
            check(metadata["base_fingerprint"] == metadata["base_fingerprint_after_continuation"])
            torch.manual_seed(metadata["model_seed"])
            initial = DenseClassifier(612, [64, 32], 7)
            check(state_fingerprint(initial) == metadata["initial_fingerprint"])
            close(prep["scaler"].mean_, _snv(x[train]).mean(0))
            close(prep["scaler"].var_, _snv(x[train]).var(0))
            for task in metadata["continuation"]["source_tasks"]:
                check(set(task["support_ids"]).isdisjoint(task["query_ids"]))
                check(set(task["support_ids"] + task["query_ids"]).issubset(source_ids))
                task_count += 1
            features = transform_features(prep, x[query])
            for method, model in models.items():
                scores = model_scores(model, features)
                predicted = np.asarray(classes)[scores.argmax(1)]
                for row_number, rid in enumerate(query_ids):
                    key = (phase, metadata["outer_fold"], metadata["model_seed"], metadata.get("inner_fold", -1), method, rid)
                    saved = prediction_rows[key]
                    check(saved["y_predicted"] == predicted[row_number])
                    check(saved["y_true"] == y[positions[rid]])
                    close(scores[row_number], [float(saved["score_" + c]) for c in classes], atol=1e-6)
                    replayed += 1
            contexts += 1
    pooled = json.loads((REPORT / "pooled_metrics_by_seed.json").read_text())
    outer = [row for key, row in prediction_rows.items() if key[0] == "models"]
    changes, paired = [], {}
    for seed in (42, 43, 44):
        indexed = {method: {row["record_id"]: row for row in outer if int(row["seed"]) == seed and row["method"] == method} for method in METHODS}
        for method, records in indexed.items():
            check(set(records) == set(positions))
            recalculated = counts_metrics([row["y_true"] for row in records.values()], [row["y_predicted"] for row in records.values()], classes)
            for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
                close(recalculated[metric], pooled[method][str(seed)][metric])
            check(recalculated["confusion_matrix"] == pooled[method][str(seed)]["confusion_matrix"])
        corrected, harmed = 0, 0
        for rid in sorted(positions):
            before, after = indexed[METHODS[0]][rid], indexed[METHODS[1]][rid]
            old_good, new_good = before["y_true"] == before["y_predicted"], after["y_true"] == after["y_predicted"]
            corrected += (not old_good) and new_good
            harmed += old_good and (not new_good)
            changes.append({"seed": seed, "record_id": rid, "y_true": before["y_true"], "base_prediction": before["y_predicted"],
                            "continued_prediction": after["y_predicted"], "corrected": int((not old_good) and new_good),
                            "harmed": int(old_good and (not new_good))})
        paired[str(seed)] = {"corrected": corrected, "harmed": harmed, "net_correct": corrected - harmed,
                             "accuracy_change_percentage_points": 100 * (corrected - harmed) / 855,
                             "macro_f1_change": pooled[METHODS[1]][str(seed)]["macro_f1"] - pooled[METHODS[0]][str(seed)]["macro_f1"],
                             "target_support_labels_used": 0}
    check(contexts == 45)
    check(task_count == 4500)
    check(replayed == len(prediction_rows) == 25650)
    with (REPORT / "continuation_paired_changes.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(changes[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(changes)
    (REPORT / "continuation_paired_summary.json").write_text(json.dumps(paired, ensure_ascii=False, indent=2) + "\n")
    result = {"status": "pass", "checks": checks, "maximum_numeric_difference": maximum, "source_contexts": contexts,
              "source_tasks": task_count, "predictions_replayed": replayed, "new_model_fits": 0,
              "primary_seed": 42, "all_seeds": [42, 43, 44], "source_implementation_hashes_checked": True,
              "audit_script_sha256": digest(Path(__file__)), "pooled_metrics_by_seed_sha256": digest(REPORT / "pooled_metrics_by_seed.json")}
    (REPORT / "replay_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
