"""Replay held-out R2 predictions or predict compatible, genuinely new spectra.

The five archived models are research fold models. Their future majority vote
has no measured external accuracy and must not be scored on their source data.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "reports/paper-c/2026-10-31/best-candidate/candidate_index.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spectrum_sha256(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes(order="C")).hexdigest()


def validate_input(values, axis, record_ids, expected_axis):
    x, actual, expected = np.asarray(values, dtype=float), np.asarray(axis, dtype=float), np.asarray(expected_axis, dtype=float)
    ids = np.asarray(record_ids, dtype=str)
    if actual.ndim != 1 or actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("Frequency axis dimensions or values are invalid")
    if not np.allclose(actual, expected, rtol=0, atol=1e-12):
        raise ValueError("Frequency axis/order must match the archived 612-point axis; no implicit interpolation")
    if x.ndim != 2 or x.shape[1] != len(expected) or len(x) == 0:
        raise ValueError("Expected a nonempty N by 612 absorption-coefficient matrix")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite absorption coefficients")
    if ids.ndim != 1 or len(ids) != len(x) or len(set(ids)) != len(ids) or any(not s.strip() for s in ids):
        raise ValueError("Record IDs must be nonempty, unique and match the sample count")
    return x, ids


def verify_models(index, repo_root=ROOT):
    for relative, expected in index.get("source_code_sha256", {}).items():
        if sha256(Path(repo_root) / relative) != expected:
            raise ValueError(f"Frozen model implementation SHA256 mismatch: {relative}")
    models = {}
    for fold in index["folds"]:
        path = Path(repo_root) / fold["model_path"]
        if sha256(path) != fold["model_sha256"]:
            raise ValueError(f"Model SHA256 mismatch: {fold['model_path']}")
        model = joblib.load(path)
        if list(model.classes_) != index["classes"]:
            raise ValueError("Model class order differs from candidate index")
        if model.estimator.break_ties or model.metadata["prediction_rule"] != "SVC standard vote, break_ties=False":
            raise ValueError("Archived standard SVC voting rule changed")
        models[fold["outer_fold"]] = model
    return models


def aggregate_votes(votes, classes):
    counts = {label: sum(v == label for v in votes) for label in classes}
    highest = max(counts.values())
    tied = [label for label in classes if counts[label] == highest]
    return tied[0], highest, tied


def predict_records(index, values, axis, record_ids, *, mode="heldout-replay", confirm_compatible_measurement=False, repo_root=ROOT):
    x, ids = validate_input(values, axis, record_ids, index["frequency_axis"])
    if mode not in {"heldout-replay", "future-ensemble"}:
        raise ValueError("Unknown prediction mode")
    heldout_fold = {}
    all_source_ids = set()
    for fold in index["folds"]:
        source = set(fold["source_record_ids"])
        query = set(fold["heldout_record_ids"])
        if source & query:
            raise ValueError("Fold source and held-out IDs overlap")
        all_source_ids.update(source)
        for rid in query:
            if rid in heldout_fold:
                raise ValueError("A record belongs to multiple held-out folds")
            heldout_fold[rid] = fold["outer_fold"]
    if mode == "future-ensemble":
        if not confirm_compatible_measurement:
            raise ValueError("Confirm the same absorption-coefficient definition, units and measurement compatibility")
        if any(rid in all_source_ids or rid in heldout_fold for rid in ids):
            raise ValueError("Known historical records cannot be evaluated with the full fold ensemble")
        historical_values = set(index.get("historical_record_value_sha256", {}).values())
        if any(spectrum_sha256(row) in historical_values for row in x):
            raise ValueError("A historical spectrum cannot become a new independent sample by changing its ID")
    elif any(rid not in heldout_fold for rid in ids):
        raise ValueError("Held-out replay accepts only IDs in the original 855-record partition")
    elif "historical_record_value_sha256" in index:
        for rid, values_row in zip(ids, x, strict=True):
            if spectrum_sha256(values_row) != index["historical_record_value_sha256"][rid]:
                raise ValueError("Held-out spectrum values no longer match the historical record ID")
    models = verify_models(index, repo_root)
    result = []
    if mode == "heldout-replay":
        for fold_id, model in sorted(models.items()):
            selected = [i for i, rid in enumerate(ids) if heldout_fold[rid] == fold_id]
            if not selected:
                continue
            # Preserve the original SVC.predict vote. Do not use score argmax.
            predicted = model.predict(x[selected])
            for i, label in zip(selected, predicted, strict=True):
                result.append({"record_id": str(ids[i]), "predicted_origin_group": str(label),
                               "outer_fold": fold_id, "source_seen_by_this_model": False})
    else:
        by_fold = {fold_id: model.predict(x) for fold_id, model in sorted(models.items())}
        for i, rid in enumerate(ids):
            votes = [str(labels[i]) for labels in by_fold.values()]
            label, count, tied = aggregate_votes(votes, index["classes"])
            result.append({"record_id": str(rid), "predicted_origin_group": label,
                           "agreeing_fold_models": count, "vote_fraction_not_probability": count / len(votes),
                           "fold_disagreement": len(set(votes)) > 1, "tied_classes": ";".join(tied),
                           **{f"fold_{fold_id}_prediction": str(labels[i]) for fold_id, labels in by_fold.items()}})
    order = {rid: i for i, rid in enumerate(ids)}
    return sorted(result, key=lambda row: order[row["record_id"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--input", type=Path, required=True, help="NPZ with X_alpha, axis, record_ids; labels are ignored")
    parser.add_argument("--output", type=Path, required=True, help="New CSV path; an existing result is never overwritten")
    parser.add_argument("--mode", choices=("heldout-replay", "future-ensemble"), default="heldout-replay")
    parser.add_argument("--confirm-compatible-measurement", action="store_true")
    args = parser.parse_args()
    index = json.loads(args.index.read_text(encoding="utf-8"))
    with np.load(args.input, allow_pickle=False) as data:
        rows = predict_records(index, data["X_alpha"], data["axis"], data["record_ids"],
                               mode=args.mode, confirm_compatible_measurement=args.confirm_compatible_measurement)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"mode": args.mode, "records": len(rows), "output": str(args.output),
                      "index_sha256": sha256(args.index), "input_sha256": sha256(args.input),
                      "new_model_fits": 0, "probabilities_calibrated": False,
                      "future_ensemble_external_accuracy": None, "unknown_origin_detection_validated": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
