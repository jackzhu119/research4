import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("ml_figures", Path(__file__).resolve().parents[1]/"scripts/analyze_research72h_ml_evidence_20261107.py")
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


class MLPaperFigureTests(unittest.TestCase):
    def noise(self):
        raw, curves = [], []
        for level in (0., .001, .005, .01):
            for seed in ([""] if level == 0 else ["42", "43", "44"]):
                raw.append({"kind": "noise", "level": str(level), "seed": seed, **{m: ".8" for m in M.METRICS}})
            for metric in M.METRICS:
                curves.append({"noise_level": str(level), "metric": metric, "seed_count": "1" if level == 0 else "3",
                               "mean": ".8", "minimum": ".8", "maximum": ".8"})
        return curves, raw

    def test_noise_range_reconstructs_actual_seeds(self):
        curves, raw = self.noise(); M.validate_noise(curves, raw)
        raw[-1]["seed"] = "best_seed"
        with self.assertRaisesRegex(ValueError, "three-seed"):
            M.validate_noise(curves, raw)

    def test_modified_curve_or_duplicate_noise_row_is_rejected(self):
        curves, raw = self.noise()
        for wrong in (curves+[curves[0]], [dict(curves[0], mean=".9")]+curves[1:]):
            with self.assertRaises(ValueError):
                M.validate_noise(wrong, raw)

    def test_only_complete_ten_predeclared_pairs_are_plot_candidates(self):
        rows = [{"method": "R2_reproduction", "baseline": name, "metric": "macro_f1", "bootstrap_replicates": "2000",
                 "delta_percentage_points": "2", "conditional_low_percentage_points": "-1", "conditional_high_percentage_points": "4"}
                for name in M.LABELS]
        self.assertEqual(len(M.selected_pairs(rows)), 10)
        with self.assertRaises(ValueError):
            M.selected_pairs(rows[:-1])

    def test_omission_ranges_are_five_actual_deletions_not_intervals(self):
        pair = {"method": "R2_reproduction", "baseline": "full_svm", "delta_percentage_points": "2"}
        rows = [{"method": pair["method"], "baseline": pair["baseline"], "metric": "macro_f1", "omission_type": "single_ZY02_specimen",
                 "omitted": str(i), "delta_percentage_points": str(i)} for i in range(5)]
        actual = M.sensitivity_ranges(rows, [pair], "single_ZY02_specimen")[0]
        self.assertEqual((actual["minimum_delta_pp"], actual["maximum_delta_pp"]), (0., 4.))
        self.assertIn("not a confidence interval", actual["interpretation"])
        with self.assertRaises(ValueError):
            M.sensitivity_ranges(rows[:4]+[rows[0]], [pair], "single_ZY02_specimen")

    def test_completed_artifact_hash_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); file = root/"data.csv"; file.write_text("id,value\na,1\n")
            index = root/"artifact_index.json"; index.write_text(json.dumps({"data.csv": {"sha256": M.sha(file), "bytes": file.stat().st_size}}))
            index_hash = M.sha(index)
            with patch.object(M, "ROOT", root):
                self.assertEqual(M.authenticate(root, index_hash)["verified_artifacts"], 1)
                file.write_text("id,value\na,2\n")
                with self.assertRaisesRegex(ValueError, "changed"):
                    M.authenticate(root, index_hash)

    def test_robustness_hash_only_index_has_its_actual_distinct_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); file = root/"data.csv"; file.write_text("id,value\na,1\n")
            index = root/"artifact_index.json"; index.write_text(json.dumps({"data.csv": M.sha(file)}))
            with patch.object(M, "ROOT", root):
                self.assertEqual(M.authenticate(root, M.sha(index))["verified_artifacts"], 1)


if __name__ == "__main__":
    unittest.main()
