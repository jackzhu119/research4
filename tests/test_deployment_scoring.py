"""Synthetic scoring fixtures only; these are not new THz measurements."""
import csv
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1]/"scripts/score_zaoren_predictions.py"
SPEC = importlib.util.spec_from_file_location("deployment_scoring_under_test", SCRIPT)
SCORER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DeploymentScoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.sequence = 0

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, predictions=None, labels=None, lock_updates=None):
        self.sequence += 1
        base = self.directory/f"case{self.sequence}"
        pred = base/"predictions"; pred.mkdir(parents=True)
        predictions = predictions or [{"record_id": "NEW-001", "predicted_group": "ZY01"}]
        fields = ["record_id", "predicted_group", "historical_id_match", "historical_value_match", "prediction_scope"]
        rows = [{"historical_id_match": False, "historical_value_match": False,
                 "prediction_scope": "UNSCORED_FUTURE_INPUT", **row} for row in predictions]
        with (pred/"predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        lock = {"mode": "future", "labels_received": False, "new_model_fits": 0,
                "class_order": list(SCORER.CLASSES), "records": len(rows),
                "predictions_sha256": digest(pred/"predictions.csv")}
        lock.update(lock_updates or {})
        (pred/"prediction_lock.json").write_text(json.dumps(lock))
        truth = base/"labels.csv"
        if labels is None:
            labels = [{"record_id": row["record_id"], "label": row["predicted_group"]} for row in rows]
        self.write_labels(truth, labels)
        return pred, truth, base/"scored"

    @staticmethod
    def write_labels(path, rows, fields=("record_id", "label")):
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

    def test_known_seven_by_seven_confusion_arithmetic_and_truth_join(self):
        expected = [[2, 1, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 2, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 0, 1]]
        predictions, labels = [], []
        for i, row in enumerate(expected):
            for j, count in enumerate(row):
                for _ in range(count):
                    rid = f"NEW-{len(labels):03d}"
                    predictions.append({"record_id": rid, "predicted_group": SCORER.CLASSES[j]})
                    labels.append({"record_id": rid, "label": SCORER.CLASSES[i]})
        pred, truth, output = self.fixture(predictions, labels[::-1])  # order may differ, ID join must be exact
        result = SCORER.score_predictions(pred, truth, output)
        self.assertEqual(result["confusion_matrix"], expected)
        self.assertEqual(result["correct"], 9)
        self.assertEqual(result["samples"], 13)
        self.assertAlmostEqual(result["accuracy"], 9/13)
        self.assertAlmostEqual(result["macro_f1"], float((4*Fraction(2, 3)+Fraction(1, 2)+Fraction(4, 5)+1)/7))
        self.assertAlmostEqual(result["balanced_accuracy"], float((Fraction(2, 3)+3*Fraction(1, 2)+3)/7))
        self.assertEqual(result["per_class"]["ZY01"]["support"], 3)
        self.assertAlmostEqual(result["per_class"]["ZY01"]["precision"], 2/3)
        self.assertFalse(result["partial_class_coverage"])
        self.assertFalse(result["independent_confirmation_established"])
        self.assertEqual(result["new_model_fits"], 0)
        self.assertEqual(result["labels_sha256"], digest(truth))
        with (output/"joined_predictions.csv").open(encoding="utf-8-sig", newline="") as f:
            joined = list(csv.DictReader(f))
        self.assertEqual(sum(row["correct"] == "True" for row in joined), 9)
        self.assertEqual([row["record_id"] for row in joined], [row["record_id"] for row in predictions])
        self.assertEqual(json.loads((output/"metrics.json").read_text())["confusion_matrix"], expected)

    def test_single_future_specimen_partial_coverage_is_explicit(self):
        pred, truth, output = self.fixture()
        result = SCORER.score_predictions(pred, truth, output)
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["balanced_accuracy"], 1)
        self.assertAlmostEqual(result["macro_f1"], 1/7)
        self.assertTrue(result["partial_class_coverage"])
        self.assertEqual(result["class_coverage"]["true_classes_observed"], 1)
        self.assertEqual(len(result["class_coverage"]["missing_true_classes"]), 6)
        self.assertIn("not directly comparable", result["partial_class_coverage_note"])

    def test_prediction_tamper_rejected_before_any_truth_parse(self):
        pred, truth, output = self.fixture()
        with (pred/"predictions.csv").open("a") as f:
            f.write("NEW-002,ZY02,False,False,UNSCORED_FUTURE_INPUT\n")
        with mock.patch.object(SCORER, "_read_labels") as parse:
            with self.assertRaisesRegex(SCORER.ScoringInputError, "SHA256"):
                SCORER.score_predictions(pred, truth, output)
            parse.assert_not_called()
        self.assertFalse(output.exists())

    def test_training_replay_labels_or_fitting_cannot_be_scored_as_future(self):
        for change in [{"mode": "training-diagnostic"}, {"mode": "heldout-replay"},
                       {"labels_received": True}, {"labels_received": 0},
                       {"new_model_fits": 1}, {"new_model_fits": False}]:
            with self.subTest(change=change):
                pred, truth, output = self.fixture(lock_updates=change)
                with mock.patch.object(SCORER, "_read_labels") as parse:
                    with self.assertRaises(SCORER.ScoringInputError):
                        SCORER.score_predictions(pred, truth, output)
                    parse.assert_not_called()
                self.assertFalse(output.exists())

    def test_future_lock_does_not_override_historical_row_flags(self):
        for change in [{"historical_id_match": True}, {"historical_value_match": True},
                       {"historical_id_match": "unknown"},
                       {"prediction_scope": "TRAINING_DIAGNOSTIC_NOT_PERFORMANCE"}]:
            with self.subTest(change=change):
                pred, truth, output = self.fixture(predictions=[{"record_id": "NEW-001", "predicted_group": "ZY01", **change}])
                with self.assertRaises(SCORER.ScoringInputError):
                    SCORER.score_predictions(pred, truth, output)

    def test_truth_duplicate_missing_extra_and_unknown_labels_fail(self):
        predictions = [{"record_id": "NEW-001", "predicted_group": "ZY01"},
                       {"record_id": "NEW-002", "predicted_group": "ZY02"}]
        cases = [[{"record_id": "NEW-001", "label": "ZY01"}]*2,
                 [{"record_id": "NEW-001", "label": "ZY01"}],
                 [{"record_id": f"NEW-{i:03d}", "label": "ZY01"} for i in [1, 2, 3]],
                 [{"record_id": "NEW-001", "label": "ZY01"}, {"record_id": "NEW-002", "label": "ZY08"}],
                 [{"record_id": "NEW-001", "label": " ZY01"}, {"record_id": "NEW-002", "label": "ZY02"}]]
        for i, labels in enumerate(cases):
            with self.subTest(case=i):
                pred, truth, output = self.fixture(predictions, labels)
                with self.assertRaises(SCORER.ScoringInputError):
                    SCORER.score_predictions(pred, truth, output)
                self.assertFalse(output.exists())

    def test_prediction_duplicates_unknown_classes_and_counts_fail(self):
        cases = [[{"record_id": "NEW-001", "predicted_group": "ZY01"}]*2,
                 [{"record_id": "NEW-001", "predicted_group": "ZY08"}]]
        for rows in cases:
            with self.subTest(rows=rows):
                pred, truth, output = self.fixture(rows)
                with self.assertRaises(SCORER.ScoringInputError):
                    SCORER.score_predictions(pred, truth, output)
        for change in [{"records": 2}, {"class_order": list(reversed(SCORER.CLASSES))}]:
            with self.subTest(change=change):
                pred, truth, output = self.fixture(lock_updates=change)
                with self.assertRaises(SCORER.ScoringInputError):
                    SCORER.score_predictions(pred, truth, output)

    def test_truth_schema_is_exact_and_no_existing_output_is_overwritten(self):
        pred, truth, output = self.fixture()
        self.write_labels(truth, [{"record_id": "NEW-001", "label": "ZY01", "batch": "unknown"}], fields=("record_id", "label", "batch"))
        with self.assertRaisesRegex(SCORER.ScoringInputError, "exactly"):
            SCORER.score_predictions(pred, truth, output)
        self.write_labels(truth, [{"record_id": "NEW-001", "label": "ZY01"}])
        output.mkdir()
        marker = output/"untouched.txt"; marker.write_text("do not overwrite")
        with self.assertRaises(FileExistsError):
            SCORER.score_predictions(pred, truth, output)
        self.assertEqual(marker.read_text(), "do not overwrite")

    def test_cli_scores_synthetic_fixture_without_loading_models(self):
        pred, truth, output = self.fixture()
        completed = subprocess.run([sys.executable, str(SCRIPT), "--predictions", str(pred),
                                    "--labels", str(truth), "--output", str(output)],
                                   check=True, capture_output=True, text=True)
        result = json.loads(completed.stdout)
        self.assertEqual(result["samples"], 1)
        self.assertEqual(result["new_model_fits"], 0)
        self.assertTrue(result["partial_class_coverage"])
        self.assertEqual(sorted(path.name for path in output.iterdir()), ["joined_predictions.csv", "metrics.json"])


if __name__ == "__main__":
    unittest.main()
