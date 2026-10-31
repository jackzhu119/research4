"""Protocol isolation tests on synthetic spectra, not herbal performance."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from herb_spectra.classification_transfer import fit_source as original_fit_source
from herb_spectra.within_batch_holdout import (HoldoutProtocolError, fit_frozen_predictions,
    prepare_holdout, score_prediction_bundle, support_draws)


def fixture():
    rng = np.random.default_rng(20)
    labels = np.asarray(["A"] * 15 + ["B"] * 5 + ["C"] * 15)
    identifiers = np.asarray([f"sample-{i:03d}" for i in range(len(labels))])
    X = rng.normal(size=(len(labels), 10)) + np.asarray([{"A": -1, "B": 0, "C": 1}[c] for c in labels])[:, None]
    arrays = {"X_alpha": X, "X_n": X / 3 + 2, "axis": np.linspace(.1, 1.5, 10),
        "record_ids": identifiers, "groups": identifiers.copy(), "y": labels,
        "labels_verified": np.ones(len(labels), dtype=bool),
        "batch_ids": np.repeat("", len(labels)), "dataset_version": np.asarray("fixture-v1"),
        "thickness_mm": np.repeat(1.5, len(labels))}
    records = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        train, cal = int(.6 * len(indices)), int(.2 * len(indices))
        roles = ["training"] * train + ["calibration"] * cal + ["prediction"] * (len(indices) - train - cal)
        for index, role in zip(indices, roles):
            records.append({"record_id": str(identifiers[index]),
                "physical_sample_id": str(identifiers[index]), "origin_group_id": str(label), "role": role})
    return arrays, {"dataset_version": "fixture-v1", "split_seed": 5, "records": records}


SMALL_GRID = {"preprocess__mode": ["raw"], "model__C": [1.]}


class WithinBatchHoldoutTests(unittest.TestCase):
    def test_plan_is_complete_group_disjoint_and_truth_separated(self):
        arrays, plan = fixture()
        roles, truth, info = prepare_holdout(arrays, plan)
        self.assertEqual([len(roles[r]["record_ids"]) for r in ("training", "calibration", "prediction")], [21, 7, 7])
        self.assertNotIn("y", roles["prediction"])
        self.assertEqual(set(truth), set(roles["prediction"]["record_ids"]))
        self.assertEqual(info["thickness_mm_counts_by_role"]["prediction"], {"1.5": 7})
        bad = deepcopy(plan); bad["records"][0] = bad["records"][1].copy()
        with self.assertRaisesRegex(HoldoutProtocolError, "duplicate, missing"):
            prepare_holdout(arrays, bad)
        bad = deepcopy(plan); bad["records"].pop()
        with self.assertRaisesRegex(HoldoutProtocolError, "cover every"):
            prepare_holdout(arrays, bad)

    def test_missing_verified_labels_groups_and_conflicting_plan_rejected(self):
        arrays, plan = fixture()
        bad = deepcopy(arrays); bad["labels_verified"][0] = False
        with self.assertRaisesRegex(ValueError, "labels_verified"):
            prepare_holdout(bad, plan)
        bad = deepcopy(arrays); bad["groups"][0] = ""
        with self.assertRaisesRegex(ValueError, "Missing physical"):
            prepare_holdout(bad, plan)
        bad = deepcopy(plan); bad["records"][0]["physical_sample_id"] = "someone-else"
        with self.assertRaisesRegex(HoldoutProtocolError, "disagrees"):
            prepare_holdout(arrays, bad)
        bad = deepcopy(plan); bad["dataset_version"] = "different"
        with self.assertRaisesRegex(HoldoutProtocolError, "dataset_version"):
            prepare_holdout(arrays, bad)

    def test_unavailable_shots_preserve_every_class_and_fixed_prediction(self):
        arrays, plan = fixture()
        roles, _, _ = prepare_holdout(arrays, plan)
        draws, unavailable = support_draws(roles["calibration"], np.asarray(["A", "B", "C"]),
            shots=(1, 3, 5, 10), repeats=3, seed=42)
        self.assertEqual(set(unavailable), {3, 5, 10})
        self.assertEqual(len(draws), 3)
        for row in draws:
            self.assertEqual(len(row["indices"]), 3)
            self.assertEqual(set(roles["calibration"]["y"][row["indices"]]), {"A", "B", "C"})
            self.assertFalse(set(row["sample_ids"]) & set(roles["prediction"]["groups"]))

    def test_frozen_model_and_query_truth_permutation_only_change_scoring(self):
        arrays, plan = fixture()
        roles, truth, _ = prepare_holdout(arrays, plan)
        bundle = fit_frozen_predictions(**roles, selectors=("full",), repeats=2, source_grid=SMALL_GRID)
        model = bundle["models"][0]
        self.assertTrue(model["source_model_unchanged"])
        self.assertEqual(model["source_model_fingerprint_before"], model["source_model_fingerprint_after"])
        self.assertEqual(bundle["class_order"], ["A", "B", "C"])
        snapshot = deepcopy(bundle)
        first = score_prediction_bundle(bundle, truth)
        changed = {identifier: {"A": "B", "B": "C", "C": "A"}[value] for identifier, value in truth.items()}
        second = score_prediction_bundle(bundle, changed)
        self.assertEqual(bundle, snapshot)
        self.assertEqual([r["y_predicted"] for r in first["prediction_rows"]],
                         [r["y_predicted"] for r in second["prediction_rows"]])
        self.assertNotEqual(first["rows"][0]["metrics"], second["rows"][0]["metrics"])

    def test_query_feature_changes_do_not_change_source_or_calibrator_fit(self):
        arrays, plan = fixture()
        roles, _, _ = prepare_holdout(arrays, plan)
        first = fit_frozen_predictions(**roles, selectors=("full",), repeats=2, source_grid=SMALL_GRID)
        changed = deepcopy(roles)
        changed["prediction"]["X_alpha"] = changed["prediction"]["X_alpha"] * 31 + 1000
        second = fit_frozen_predictions(**changed, selectors=("full",), repeats=2, source_grid=SMALL_GRID)
        left, right = first["models"][0], second["models"][0]
        self.assertEqual(left["source_fit"]["best_params"], right["source_fit"]["best_params"])
        self.assertEqual(left["source_model_fingerprint_before"], right["source_model_fingerprint_before"])
        for a, b in zip(left["outcomes"][1:], right["outcomes"][1:]):
            assert_allclose(a["calibration_coefficients"], b["calibration_coefficients"], rtol=0, atol=0)
            assert_allclose(a["calibration_intercept"], b["calibration_intercept"], rtol=0, atol=0)

    def test_only_training_arrays_enter_source_fit_and_id_cannot_be_a_feature(self):
        arrays, plan = fixture()
        roles, _, _ = prepare_holdout(arrays, plan)
        with patch("herb_spectra.within_batch_holdout.fit_source", wraps=original_fit_source) as fitted:
            fit_frozen_predictions(**roles, selectors=("full", "mcuve20"), repeats=1, source_grid=SMALL_GRID)
        self.assertEqual(fitted.call_count, 2)
        for call in fitted.call_args_list:
            passed = call.args[0]
            assert_array_equal(passed["X_alpha"], roles["training"]["X_alpha"])
            self.assertEqual(set(passed["sample_ids"]), set(roles["training"]["groups"]))
        bad = deepcopy(roles); bad["prediction"]["y"] = np.asarray(["A"] * 7)
        with self.assertRaisesRegex(HoldoutProtocolError, "must not contain truth"):
            fit_frozen_predictions(**bad, selectors=("full",), repeats=1, source_grid=SMALL_GRID)

    def test_role_overlap_rejected_before_source_model_fit(self):
        arrays, plan = fixture()
        roles, _, _ = prepare_holdout(arrays, plan)
        roles["prediction"]["groups"][0] = roles["training"]["groups"][0]
        with patch("herb_spectra.within_batch_holdout.fit_source") as fitted:
            with self.assertRaisesRegex(HoldoutProtocolError, "overlap"):
                fit_frozen_predictions(**roles, selectors=("full",), repeats=1)
        fitted.assert_not_called()


if __name__ == "__main__":
    unittest.main()
