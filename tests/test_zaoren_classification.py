import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from herb_spectra.classification import (BlockRowPreprocessor, DatasetValidationError,
    aggregate_samples, build_pipeline, class_scores, representation,
    run_grouped_classification, validate_dataset)


def fixture(counts=(15, 5), repeats=1, points=8):
    rng = np.random.default_rng(83)
    labels = np.concatenate([np.repeat(f"ZY{i + 1:02d}", n) for i, n in enumerate(counts)])
    samples = np.asarray([f"physical-{i:03d}" for i in range(len(labels))])
    alpha = rng.normal(size=(len(labels), points))
    alpha[:, :2] += np.arange(len(labels))[:, None] / max(1, len(labels))
    n = alpha * .2 + rng.normal(size=alpha.shape)
    return {"X_alpha": np.repeat(alpha, repeats, axis=0), "X_n": np.repeat(n, repeats, axis=0),
            "axis": np.linspace(.1, 1.5, points),
            "record_ids": np.asarray([f"record-{i:03d}" for i in range(len(labels) * repeats)]),
            "y": np.repeat(labels, repeats), "groups": np.repeat(samples, repeats),
            "batch_ids": np.repeat("", len(labels) * repeats),
            "labels_verified": np.ones(len(labels) * repeats, dtype=bool)}


class ZaorenClassificationTests(unittest.TestCase):
    def test_review_gate_never_falls_back_to_record_ids(self):
        data = fixture()
        data["groups"][:] = ""
        with self.assertRaisesRegex(DatasetValidationError, "physical-sample groups"):
            validate_dataset(data)
        data = fixture(); data["labels_verified"] = np.asarray(False)
        with self.assertRaisesRegex(DatasetValidationError, "labels_verified"):
            validate_dataset(data)
        # An explicit reviewed one-record/one-specimen mapping is allowed.
        data = fixture(); data["groups"] = data["record_ids"].copy()
        self.assertEqual(len(validate_dataset(data)["groups"]), 20)

    def test_repeats_are_aggregated_and_conflicts_rejected(self):
        data = fixture(repeats=2)
        data["X_alpha"][1] += 2
        grouped = aggregate_samples(data)
        self.assertEqual(len(grouped["y"]), 20)
        np.testing.assert_allclose(grouped["X_alpha"][0], data["X_alpha"][:2].mean(axis=0))
        self.assertEqual(grouped["input_record_ids"][0], ("record-000", "record-001"))
        self.assertTrue(np.all(grouped["scan_counts"] == 2))
        conflicting = fixture(repeats=2); conflicting["y"][1] = "ZY02"
        with self.assertRaisesRegex(DatasetValidationError, "conflicting origin"):
            aggregate_samples(conflicting)
        conflicting = fixture(repeats=2)
        conflicting["batch_ids"] = np.asarray(["batch-A", "batch-B"] * 20)
        with self.assertRaisesRegex(DatasetValidationError, "conflicting known batch"):
            aggregate_samples(conflicting)

    def test_concat_snv_treats_modalities_separately(self):
        data = aggregate_samples(fixture())
        X, blocks = representation(data, "concat")
        np.testing.assert_array_equal(X[:, :8], data["X_n"])
        np.testing.assert_array_equal(X[:, 8:], data["X_alpha"])
        transformed = BlockRowPreprocessor("snv", blocks).fit_transform(X)
        for block in (transformed[:, :8], transformed[:, 8:]):
            np.testing.assert_allclose(block.mean(axis=1), 0, atol=1e-14)
            np.testing.assert_allclose(block.std(axis=1, ddof=1), 1, atol=1e-14)

    def test_configured_cars_cv_is_rejected_not_silently_reduced(self):
        data = fixture()
        report = run_grouped_classification(data, methods=("svc", "cars_svc"),
            outer_splits=3, inner_splits=3, cars_cv=3, preflight_only=True)
        cases = {r["method"]: r for r in report["results"]}
        self.assertEqual(cases["svc"]["status"], "ready")
        self.assertEqual(cases["cars_svc"]["status"], "not_evaluable")
        self.assertIn("cv=3", cases["cars_svc"]["reason"])
        valid = run_grouped_classification(data, methods=("svc", "cars_svc"),
            outer_splits=5, inner_splits=2, cars_cv=2, preflight_only=True)
        self.assertEqual(valid["status"], "ready")
        self.assertEqual(valid["class_counts"]["ZY02"], 5)

    def test_binary_scores_follow_class_order_without_refitting_scaler(self):
        data = aggregate_samples(fixture())
        X, blocks = representation(data)
        for model in ("svc", "plsda"):
            pipe = build_pipeline(model, block_sizes=blocks).fit(X, data["y"])
            original_mean = pipe.named_steps["scale"].mean_.copy()
            score = class_scores(pipe, X)
            self.assertEqual(score.shape, (20, 2))
            np.testing.assert_array_equal(pipe.classes_[score.argmax(axis=1)], pipe.predict(X))
            class_scores(pipe, X + 100)
            np.testing.assert_array_equal(original_mean, pipe.named_steps["scale"].mean_)

    def test_five_by_two_actual_synthetic_matrix_writes_sample_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            data = fixture(repeats=2)
            report = run_grouped_classification(data,
                methods=("dummy", "svc", "plsda", "mcuve20_svc", "cars_svc"),
                outer_splits=5, inner_splits=2, cars_cv=2,
                output_dir=output, identity_scope="synthetic fixtures only")
            self.assertEqual(report["status"], "completed", report)
            self.assertEqual(report["input_records"], 40)
            self.assertEqual(report["physical_sample_count"], 20)
            splits = json.loads((output / "splits.json").read_text())
            for fold in splits:
                self.assertFalse(set(fold["train_sample_ids"]) & set(fold["test_sample_ids"]))
            for result in report["results"]:
                self.assertEqual(len(result["folds"]), 5)
                self.assertEqual(result["metrics"]["class_counts"]["ZY02"], 5)
                predictions = output / f"alpha_{result['method']}" / "predictions.csv"
                self.assertEqual(len(predictions.read_text(encoding="utf-8-sig").splitlines()), 21)
            with self.assertRaises(FileExistsError):
                run_grouped_classification(data, output_dir=output)

    def test_cli_unverified_fixture_is_blocked_without_models(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "input.npz"; output = Path(temporary) / "output"
            data = fixture(); data["labels_verified"][:] = False
            np.savez(source, **data)
            completed = subprocess.run([sys.executable, str(root / "scripts/run_zaoren_classification.py"),
                "--input", str(source), "--output", str(output)], cwd=root,
                capture_output=True, text=True)
            self.assertEqual(completed.returncode, 2)
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(report["status"], "blocked")
            self.assertTrue(report["models_not_run"])
            self.assertFalse((output / "splits.json").exists())


if __name__ == "__main__":
    unittest.main()
