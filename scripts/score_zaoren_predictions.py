"""Score locked future ZY01--ZY07 predictions with separately supplied truth.

This standard-library-only program does not import a model or fit anything.
Prediction mode and file integrity do not establish physical-sample/batch
independence. Synthetic test fixtures are not new medicinal-plant experiments.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

CLASSES = tuple(f"ZY{i:02d}" for i in range(1, 8))


class ScoringInputError(ValueError):
    """The supplied prediction/truth pair is ineligible for future scoring."""


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def _csv_rows(path, *, exact_header=None, required=()):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames
        if not headers or len(headers) != len(set(headers)):
            raise ScoringInputError("CSV requires nonempty, unique column headers")
        if exact_header is not None and headers != list(exact_header):
            raise ScoringInputError("Truth CSV requires exactly record_id,label columns in that order")
        if not set(required).issubset(headers):
            raise ScoringInputError("Prediction CSV requires record_id and predicted_group")
        rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ScoringInputError("CSV must contain complete, nonempty prediction/truth rows")
    ids = [row["record_id"] for row in rows]
    if any(not rid or rid != rid.strip() for rid in ids) or len(ids) != len(set(ids)):
        raise ScoringInputError("record_id values must be nonempty, exact and unique")
    return rows


def _read_labels(path):
    rows = _csv_rows(path, exact_header=("record_id", "label"))
    if any(row["label"] not in CLASSES for row in rows):
        raise ScoringInputError("Truth labels must be exactly ZY01 through ZY07")
    return {row["record_id"]: row["label"] for row in rows}


def _read_predictions(path):
    rows = _csv_rows(path, required=("record_id", "predicted_group"))
    for row in rows:
        if row["predicted_group"] not in CLASSES:
            raise ScoringInputError("Predicted groups must be exactly ZY01 through ZY07")
        for name in ("historical_id_match", "historical_value_match"):
            if name in row and row[name].casefold() not in {"false", "0"}:
                raise ScoringInputError("Historical or unknown matching status is not future evaluation")
        if "prediction_scope" in row and row["prediction_scope"] != "UNSCORED_FUTURE_INPUT":
            raise ScoringInputError("Historical/training prediction scope cannot be scored as future data")
    return rows


def _metrics(rows, truth):
    index = {label: i for i, label in enumerate(CLASSES)}
    confusion = [[0 for _ in CLASSES] for _ in CLASSES]
    joined = []
    for row in rows:
        actual, prediction = truth[row["record_id"]], row["predicted_group"]
        confusion[index[actual]][index[prediction]] += 1
        joined.append({"record_id": row["record_id"], "label": actual,
                       "predicted_group": prediction, "correct": actual == prediction})
    support = [sum(row) for row in confusion]
    predicted = [sum(row[j] for row in confusion) for j in range(len(CLASSES))]
    per_class = {}
    for i, label in enumerate(CLASSES):
        tp = confusion[i][i]
        per_class[label] = {"precision": tp/predicted[i] if predicted[i] else 0.,
                            "recall": tp/support[i] if support[i] else 0.,
                            "f1": 2*tp/(support[i]+predicted[i]) if support[i]+predicted[i] else 0.,
                            "support": support[i], "predicted_count": predicted[i]}
    observed = [label for label in CLASSES if per_class[label]["support"]]
    missing = [label for label in CLASSES if not per_class[label]["support"]]
    correct = sum(confusion[i][i] for i in range(len(CLASSES)))
    result = {"samples": len(rows), "correct": correct, "accuracy": correct/len(rows),
              "macro_f1": sum(per_class[label]["f1"] for label in CLASSES)/len(CLASSES),
              "balanced_accuracy": sum(per_class[label]["recall"] for label in observed)/len(observed),
              "macro_f1_class_policy": "Fixed seven classes, zero F1 for unsupported classes",
              "balanced_accuracy_class_policy": "Mean recall over classes with true support",
              "class_order": list(CLASSES), "per_class": per_class, "confusion_matrix": confusion,
              "class_coverage": {"observed_true_classes": observed, "missing_true_classes": missing,
                                 "observed_predicted_classes": [label for i, label in enumerate(CLASSES) if predicted[i]],
                                 "true_classes_observed": len(observed), "all_seven_true_classes_present": not missing},
              "partial_class_coverage": bool(missing),
              "partial_class_coverage_note": (
                  "Some true classes are absent. This subset, including a single specimen, is not directly comparable to complete seven-class evaluation."
                  if missing else "All seven true classes are represented; batch and physical independence still require separate evidence.")}
    return result, joined


def score_predictions(prediction_dir, labels_csv, output_dir):
    """Join exact IDs only after validating the frozen future prediction lock.

    Existing output directories are rejected, even if empty. No labels are
    accepted by the prediction/model path: this is a separate scoring process.
    """
    prediction_dir, labels_csv, output_dir = Path(prediction_dir), Path(labels_csv), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    lock_path, prediction_path = prediction_dir/"prediction_lock.json", prediction_dir/"predictions.csv"
    lock_hash = _digest(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(lock, dict):
        raise ScoringInputError("Prediction lock must be a JSON object")
    if lock.get("mode") != "future" or lock.get("labels_received") is not False:
        raise ScoringInputError("Only locked, label-free future predictions are eligible")
    if type(lock.get("new_model_fits")) is not int or lock["new_model_fits"] != 0:
        raise ScoringInputError("Prediction must perform exactly zero new model fits")
    if lock.get("class_order") != list(CLASSES):
        raise ScoringInputError("Prediction class order must be the fixed seven anonymous groups")
    prediction_hash = _digest(prediction_path)
    if prediction_hash != lock.get("predictions_sha256"):
        raise ScoringInputError("Prediction CSV SHA256 differs from its lock")
    predictions = _read_predictions(prediction_path)
    if type(lock.get("records")) is not int or lock["records"] != len(predictions):
        raise ScoringInputError("Prediction row count differs from its lock")
    labels_hash = _digest(labels_csv)
    truth = _read_labels(labels_csv)
    predicted_ids = {row["record_id"] for row in predictions}
    if set(truth) != predicted_ids:
        raise ScoringInputError("Prediction and truth IDs must match exactly, with no missing or extra sample")
    result, joined = _metrics(predictions, truth)
    # Freeze byte identity throughout semantic parsing/joining before any output.
    if _digest(prediction_path) != prediction_hash or _digest(lock_path) != lock_hash or _digest(labels_csv) != labels_hash:
        raise ScoringInputError("An input changed during scoring")
    result.update({"created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "prediction_mode": "future", "predictions_sha256": prediction_hash,
                   "prediction_lock_sha256": lock_hash, "labels_sha256": labels_hash,
                   "metrics_from_supplied_labels": True, "new_model_fits": 0,
                   "independent_confirmation_established": False,
                   "physical_sample_independence_verified": False, "batch_independence_verified": False,
                   "evidence_status": "SUPPLIED_TRUTH_SCORED_AFTER_LOCKED_FUTURE_PREDICTIONS",
                   "evidence_note": "Future prediction mode and file integrity are not verification of biological ancestry, actual new batches or label authenticity."})
    output_dir.mkdir(parents=True, exist_ok=False)
    joined_path = output_dir/"joined_predictions.csv"
    with joined_path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["record_id", "label", "predicted_group", "correct"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(joined)
    result["joined_predictions_sha256"] = _digest(joined_path)
    with (output_dir/"metrics.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Directory containing predictions.csv and prediction_lock.json")
    parser.add_argument("--labels", type=Path, required=True, help="Separate CSV: record_id,label")
    parser.add_argument("--output", type=Path, required=True, help="A new output directory")
    args = parser.parse_args()
    result = score_predictions(args.predictions, args.labels, args.output)
    print(json.dumps({"samples": result["samples"], "correct": result["correct"],
                      "accuracy": result["accuracy"], "macro_f1": result["macro_f1"],
                      "balanced_accuracy": result["balanced_accuracy"],
                      "partial_class_coverage": result["partial_class_coverage"],
                      "output": str(args.output), "new_model_fits": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
