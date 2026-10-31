"""No research-model fitting: adversarial access and immutable-membership fixtures."""
from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_public_thz_adversarial.py"
SPEC = importlib.util.spec_from_file_location("public_thz_adversarial_under_test", SCRIPT)
adv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adv)


def query_fixture():
    rng = np.random.default_rng(991)
    return {"record_ids": np.array(["Q1", "Q2", "Q3", "Q4"]),
            "X_alpha": rng.normal(size=(4, 65)), "X_n": rng.normal(size=(4, 65)),
            "axis": np.linspace(.2, 1.6, 65)}


class PublicTHzAdversarialBoundaryTests(unittest.TestCase):
    def test_query_truth_and_metadata_injection_rejected(self):
        for key in ("y", "source_labels", "label", "export_groups", "origin", "file_name"):
            pack = query_fixture()
            pack[key] = np.array(["0", "1", "0", "1"])
            with self.subTest(key=key), self.assertRaises(adv.LeakageError):
                adv.query_schema(pack)

    def test_query_object_numeric_injection_and_axis_error_rejected(self):
        pack = query_fixture()
        pack["X_alpha"] = pack["X_alpha"].astype(object)
        with self.assertRaises(adv.LeakageError):
            adv.query_schema(pack)
        pack = query_fixture()
        pack["axis"] = pack["axis"][::-1]
        with self.assertRaises(adv.LeakageError):
            adv.query_schema(pack)

    def test_valid_query_route_has_no_label_field(self):
        pack = query_fixture()
        self.assertIs(adv.query_schema(pack), pack)

    def test_group_permutation_preserves_fixed_members_and_class_counts(self):
        groups = np.repeat(np.array(["G2", "G3", "G4", "G5", "G6"]), 7)
        ids = np.array([f"S{i:02d}" for i in range(35)])
        labels = np.array((["0", "0", "0", "0", "1", "1", "1"] * 5))
        old_ids, old_groups = ids.copy(), groups.copy()
        changed = 0
        for seed in range(1000, 1010):
            result = adv.shuffle_source_labels(labels, groups, seed)
            changed += int(not np.array_equal(result, labels))
            for group in set(groups):
                ix = groups == group
                self.assertEqual(sorted(result[ix]), sorted(labels[ix]))
                np.testing.assert_array_equal(ids[ix], old_ids[old_groups == group])
        self.assertEqual(changed, 10)
        np.testing.assert_array_equal(groups, old_groups)
        np.testing.assert_array_equal(ids, old_ids)

    def test_real_named_python_truth_reads_rejected_during_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query_truth.csv"
            path.write_text("record_id,label\nQ1,0\n")
            events = []
            with adv.truth_file_boundary([path], events):
                for reader in (lambda: open(path).read(), lambda: path.read_text(), lambda: io.open(path).read(), lambda: os.open(path, os.O_RDONLY)):
                    with self.assertRaises(adv.LeakageError):
                        reader()
            self.assertEqual(len(events), 4)
            self.assertEqual(path.read_text(), "record_id,label\nQ1,0\n")

    def test_scoring_reads_allowed_but_real_writes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query_truth.csv"
            path.write_text("frozen\n")
            events = []
            with adv.truth_file_boundary([path], events, scoring_read=True):
                self.assertEqual(path.read_text(), "frozen\n")
                for writer in (lambda: path.write_text("changed"), lambda: open(path, "a"), lambda: os.open(path, os.O_RDWR)):
                    with self.assertRaises(adv.LeakageError):
                        writer()
            self.assertEqual(path.read_text(), "frozen\n")
            self.assertEqual(sum(e["event"] == "TRUTH_ACCESS_REJECTED" for e in events), 3)

    def test_boundary_hooks_restore_after_exception_and_do_not_deny_source(self):
        original = io.open
        with tempfile.TemporaryDirectory() as directory:
            truth, source = Path(directory) / "truth.csv", Path(directory) / "source.csv"
            truth.write_text("truth")
            source.write_text("source")
            with self.assertRaises(RuntimeError):
                with adv.truth_file_boundary([truth], []):
                    self.assertEqual(source.read_text(), "source")
                    raise RuntimeError("fixture exception")
        self.assertIs(io.open, original)

    def test_gate_denies_missing_or_false_root_authorization_before_any_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            adv.write(out / "frozen_manifest.json", {"files_sha256": {}, "source_sha256": {}})
            adv.write(out / "gate.json", {"allow_adversarial_source_training": False})
            with patch.object(adv, "fit_one_run", side_effect=AssertionError("No fit may start")) as fit:
                with self.assertRaises(adv.LeakageError):
                    adv.run(out, out / "gate.json")
                fit.assert_not_called()

    def test_frozen_budget_and_single_row_perturbation(self):
        plan = adv.protocol()
        self.assertEqual(plan["classifier_fit_budget"]["total"], 132)
        self.assertEqual(plan["MC_PLS_fit_budget"]["total"], 720)
        self.assertEqual(plan["expected_retained_features"], 13)
        pack = query_fixture()
        changed = {key: value.copy() for key, value in pack.items()}
        changed["X_alpha"][0, 0] += .01
        self.assertFalse(np.array_equal(pack["X_alpha"][0], changed["X_alpha"][0]))
        np.testing.assert_array_equal(pack["X_alpha"][1:], changed["X_alpha"][1:])
        np.testing.assert_array_equal(pack["X_n"], changed["X_n"])


if __name__ == "__main__":
    unittest.main()
