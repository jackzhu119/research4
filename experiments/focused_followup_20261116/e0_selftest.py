#!/usr/bin/env python3
"""Analytical invariants for E0, using synthetic predictions and no model fits."""
from pathlib import Path
import tempfile

import numpy as np

import e0_analysis as e0


def main():
    with tempfile.TemporaryDirectory(prefix="focused-e0-check-") as directory:
        report = Path(directory)
        ids = [str(i) for i in range(10)]
        labels = ["a"] * 5 + ["b"] * 5
        weights = e0.bootstrap_weights(labels, report)
        assert np.all(weights[:, :5].sum(1) == 5)
        assert np.all(weights[:, 5:].sum(1) == 5)
        assert np.array_equal(weights, e0.bootstrap_weights(labels, report))
        observed, _ = e0.metrics_from_confusion(np.array([[5, 0], [5, 0]]))
        assert np.allclose(observed, [.5, .5, 1 / 3], atol=1e-12, rtol=0)
        complete = {}
        for method_index, method in enumerate(e0.METHODS):
            for repeat in range(2):
                folds = {}
                for fold in range(5):
                    rows = []
                    for i in (fold, fold + 5):
                        pred = "a" if repeat == 0 and method_index == 0 else labels[i]
                        rows.append({"record_id": ids[i], "physical_sample_id": ids[i],
                                     "y_true": labels[i], "prediction": pred,
                                     "is_correct": int(pred == labels[i])})
                    folds[fold] = {"predictions": rows, "seed": repeat,
                                   "train_hash": "same-actual-train-ids", "test_hash": str(fold)}
                complete[method, 20, repeat] = folds
        result = e0.analyze_predictions(complete, [0, 1], ids, labels, report)
        # A collapsed first repeat and perfect second repeat have mean macro-F1
        # 2/3; pooling the repeated predictions first would give another value.
        assert abs(result["methods"][0]["macro_f1"] - 2 / 3) < 1e-12
        accuracy = [row for row in result["effects"] if row["metric"] == "accuracy"]
        assert len(accuracy) == 6
        assert abs(accuracy[0]["difference"] + .25) < 1e-12
        # Every specimen retains both repeat predictions, so the stratified
        # paired contrast is exactly -0.25 on all 10,000 bootstrap draws.
        assert abs(accuracy[0]["interval_low"] + .25) < 1e-12
        assert abs(accuracy[0]["interval_high"] + .25) < 1e-12
        assert accuracy[0]["repeat_losses"] == 1 and accuracy[0]["repeat_ties"] == 1
        assert accuracy[0]["fold_losses"] == 5 and accuracy[0]["fold_ties"] == 5
        print("PASS: shared deterministic source-stratified specimen weights; complete prediction vectors; "
              "repeatwise macro-F1; paired effect intervals; fold/repeat sensitivity counts. No model fits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
