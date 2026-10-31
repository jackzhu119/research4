import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SPEC = importlib.util.spec_from_file_location("predict_paper_c_candidate", Path(__file__).resolve().parents[1] / "scripts/predict_paper_c_candidate.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CandidateTests(unittest.TestCase):
    def test_axis_order_and_nonfinite_are_rejected(self):
        axis = np.linspace(0.1, 1.5, 612)
        for supplied, x in [(axis[::-1], np.ones((1, 612))), (axis, np.full((1, 612), np.nan))]:
            with self.assertRaises(ValueError):
                MODULE.validate_input(x, supplied, ["new"], axis)

    def test_hash_is_checked_before_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "model.joblib").write_bytes(b"changed")
            index = {"folds": [{"model_path": "model.joblib", "model_sha256": "0" * 64}]}
            with patch.object(MODULE.joblib, "load") as loader, self.assertRaisesRegex(ValueError, "SHA256"):
                MODULE.verify_models(index, tmp)
            loader.assert_not_called()

    def test_heldout_uses_model_predict_not_score_argmax(self):
        class VoteModel:
            def predict(self, values):
                return np.array(["ZY02"] * len(values))

            def predict_scores(self, values):
                raise AssertionError("Scores must not replace the original SVC vote")

        index = {"frequency_axis": [0.1, 0.2], "classes": ["ZY01", "ZY02"],
                 "folds": [{"outer_fold": 0, "source_record_ids": ["source"], "heldout_record_ids": ["query"]}]}
        with patch.object(MODULE, "verify_models", return_value={0: VoteModel()}):
            result = MODULE.predict_records(index, [[2., 1.]], [0.1, 0.2], ["query"])
        self.assertEqual(result[0]["predicted_origin_group"], "ZY02")
        self.assertFalse(result[0]["source_seen_by_this_model"])

    def test_future_mode_rejects_historical_source_and_requires_measurement_confirmation(self):
        index = {"frequency_axis": [0.1, 0.2], "classes": ["ZY01", "ZY02"],
                 "folds": [{"outer_fold": 0, "source_record_ids": ["source"], "heldout_record_ids": ["query"]}]}
        with self.assertRaisesRegex(ValueError, "Confirm"):
            MODULE.predict_records(index, [[2., 1.]], [0.1, 0.2], ["new"], mode="future-ensemble")
        with self.assertRaisesRegex(ValueError, "historical"):
            MODULE.predict_records(index, [[2., 1.]], [0.1, 0.2], ["source"], mode="future-ensemble", confirm_compatible_measurement=True)

    def test_future_vote_ties_are_explicit_and_deterministic(self):
        label, count, tied = MODULE.aggregate_votes(["ZY02", "ZY01", "ZY02", "ZY01", "ZY03"], ["ZY01", "ZY02", "ZY03"])
        self.assertEqual((label, count, tied), ("ZY01", 2, ["ZY01", "ZY02"]))

    def test_renaming_a_historical_spectrum_does_not_create_a_new_sample(self):
        index = {"frequency_axis": [0.1, 0.2], "classes": ["ZY01", "ZY02"],
                 "historical_record_value_sha256": {"query": MODULE.spectrum_sha256([2., 1.])},
                 "folds": [{"outer_fold": 0, "source_record_ids": ["source"], "heldout_record_ids": ["query"]}]}
        with self.assertRaisesRegex(ValueError, "changing its ID"):
            MODULE.predict_records(index, [[2., 1.]], [0.1, 0.2], ["renamed"],
                                   mode="future-ensemble", confirm_compatible_measurement=True)


if __name__ == "__main__":
    unittest.main()
