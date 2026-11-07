"""Finite synthetic checks for the additive, no-fitting evidence battery."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SPEC = importlib.util.spec_from_file_location("ml_evidence_20261107", Path(__file__).resolve().parents[1] / "scripts/run_ml_evidence_battery_20261107.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MLEvidenceBatteryTests(unittest.TestCase):
    def test_local_cpu_limits_cannot_raise_supervisor_hard_limit(self):
        with patch.object(MODULE.resource, "getrlimit", return_value=(1680, 1800)), \
             patch.object(MODULE.resource, "setrlimit") as setter, \
             patch.object(MODULE.signal, "alarm"):
            MODULE.apply_local_resource_limits()
        setter.assert_called_once_with(MODULE.resource.RLIMIT_CPU, (1800, 1800))

    def test_local_cpu_limits_preserve_smaller_inherited_budget(self):
        with patch.object(MODULE.resource, "getrlimit", return_value=(300, 400)), \
             patch.object(MODULE.resource, "setrlimit") as setter, \
             patch.object(MODULE.signal, "alarm"):
            MODULE.apply_local_resource_limits()
        setter.assert_called_once_with(MODULE.resource.RLIMIT_CPU, (400, 400))

    def test_metrics_independent_hand_computation(self):
        matrix = np.array([[3, 1], [2, 4]])
        observed = MODULE.metric_array(matrix)
        np.testing.assert_allclose(observed, [.7, ((6/9)+(8/11))/2, (.75+4/6)/2])

    def test_all_classes_required_in_every_sensitivity_subset(self):
        with self.assertRaisesRegex(ValueError, "every true class"):
            MODULE.metric_array(np.array([[3, 0], [0, 0]]))

    def test_paired_identical_predictions_have_exactly_zero_delta(self):
        y = np.array([0]*9+[1]*3+[2]*6)
        prediction = y.copy()
        prediction[[1, 10]] = [1, 0]
        boot = MODULE.bootstrap_metrics(y, {"a": prediction, "b": prediction.copy()}, 71, 42, 11)
        np.testing.assert_array_equal(boot["a"], boot["b"])
        self.assertEqual(boot["a"].shape, (71, 3))

    def test_bootstrap_determinism_and_perfect_model(self):
        y = np.array([0]*8+[1]*4+[2]*3)
        first = MODULE.bootstrap_metrics(y, {"perfect": y}, 47, 98, 10)["perfect"]
        second = MODULE.bootstrap_metrics(y, {"perfect": y}, 47, 98, 10)["perfect"]
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, np.ones((47, 3)))

    def test_alignment_by_identity_survives_row_shuffle(self):
        truth = {"sample-b": "ZY01", "sample-a": "ZY02"}
        membership = {"sample-b": 3, "sample-a": 4}
        rows = [{"method": "m", "record_id": name, "outer_fold": str(membership[name]), "y_true": label,
                 "y_predicted": label, "is_correct": "1"} for name, label in truth.items()]
        observed = MODULE.align_rows(rows[::-1], ("m",), truth, membership)
        np.testing.assert_array_equal(observed["m"], [1, 0])

    def test_duplicate_missing_and_wrong_fold_are_rejected(self):
        truth, membership = {"a": "ZY01", "b": "ZY02"}, {"a": 0, "b": 1}
        rows = [{"method": "m", "record_id": name, "outer_fold": fold, "y_true": truth[name], "y_predicted": truth[name]}
                for name, fold in membership.items()]
        for invalid in (rows+[rows[0]], rows[:1], [dict(rows[0], outer_fold=3), rows[1]]):
            with self.assertRaises(ValueError):
                MODULE.align_rows(invalid, ("m",), truth, membership)

    def test_predeclared_seed_does_not_pool_or_select_other_seeds(self):
        truth, membership = {"a": "ZY01", "b": "ZY02"}, {"a": 0, "b": 1}
        rows = [{"method": "m", "record_id": name, "outer_fold": fold, "y_true": truth[name],
                 "y_predicted": truth[name] if seed == 42 else "ZY07", "seed": seed}
                for seed in (42, 43, 44) for name, fold in membership.items()]
        observed = MODULE.align_rows(rows, ("m",), truth, membership, seed=42)
        np.testing.assert_array_equal(observed["m"], [0, 1])

    def test_incorrect_saved_correctness_flag_is_rejected(self):
        row = {"method": "m", "record_id": "a", "outer_fold": 0, "y_true": "ZY01", "y_predicted": "ZY01", "is_correct": "0"}
        with self.assertRaisesRegex(ValueError, "correctness"):
            MODULE.align_rows([row], ("m",), {"a": "ZY01"}, {"a": 0})

    def test_fresh_output_cannot_overwrite_or_escape_new_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allowed = root / MODULE.PREFIX / "new"
            self.assertEqual(MODULE.fresh_output(allowed, root), allowed)
            allowed.mkdir(parents=True)
            for bad in (allowed, root/"reports/old", root/MODULE.PREFIX):
                with self.assertRaises(ValueError):
                    MODULE.fresh_output(bad, root)

    def test_additive_battery_does_not_repeat_old_stage_comparisons(self):
        self.assertEqual(len(MODULE.PAIRS), 11)
        self.assertEqual(len(MODULE.METHODS), 11)
        self.assertNotIn(("R2", "R0"), MODULE.PAIRS)
        self.assertEqual(MODULE.REPLICATES, 2000)
        self.assertEqual(MODULE.SEED, 20261107)

    def test_protocol_freezes_input_and_source_bytes_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("data.npz", "script.py", "test.py"):
                (root/name).write_bytes(name.encode())
            output = root / MODULE.PREFIX / "protocol"
            with patch.multiple(MODULE, ROOT=root, INPUTS=(Path("data.npz"),),
                                SCRIPT=Path("script.py"), TEST=Path("test.py")):
                with contextlib.redirect_stdout(io.StringIO()):
                    MODULE.freeze(output)
                protocol = output / "protocol.json"
                self.assertEqual(MODULE.verify_protocol(protocol)["status"], "FROZEN_NOT_RUN")
                (root/"data.npz").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    MODULE.verify_protocol(protocol)

    def test_rehashed_protocol_cannot_silently_change_scientific_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("data.npz", "script.py", "test.py"):
                (root/name).write_bytes(name.encode())
            output = root / MODULE.PREFIX / "protocol"
            with patch.multiple(MODULE, ROOT=root, INPUTS=(Path("data.npz"),),
                                SCRIPT=Path("script.py"), TEST=Path("test.py")):
                with contextlib.redirect_stdout(io.StringIO()):
                    MODULE.freeze(output)
                protocol = output / "protocol.json"
                value = json.loads(protocol.read_text())
                value["bootstrap_seed"] += 1
                protocol.write_text(json.dumps(value))
                (output/"protocol.sha256").write_text(MODULE.sha(protocol))
                with self.assertRaisesRegex(ValueError, "scientific settings"):
                    MODULE.verify_protocol(protocol)

    def test_sensitivity_keeps_original_class_vocabulary_and_no_refits(self):
        names = [f"{label}-{i}" for label in MODULE.CLASSES for i in range(5)]
        y = np.repeat(np.arange(7), 5)
        folds = np.tile(np.arange(5), 7)
        predictions = {method: y.copy() for method in MODULE.METHODS}
        predictions["full_svm"][5] = 0
        tables, points = MODULE.diagnostic_tables(names, y, folds, predictions, 17, 32)
        self.assertEqual(len(tables["rare_class_single_deletion.csv"]), 5*11*3)
        self.assertEqual(len(tables["outer_fold_omission.csv"]), 5*11*3)
        rare_rows = [r for r in tables["rare_class_single_deletion.csv"]
                     if r["baseline"] == "full_svm" and r["metric"] == "accuracy"]
        omitted_error = next(r for r in rare_rows if r["omitted"] == "ZY02-0")
        self.assertEqual(omitted_error["delta_percentage_points"], 0)
        self.assertTrue(any(r["delta_percentage_points"] > 0 for r in rare_rows))
        self.assertEqual(points["R2_reproduction"][0], 1)


if __name__ == "__main__":
    unittest.main()
