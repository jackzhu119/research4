"""Tiny synthetic regression of the ORIGINAL locked peak implementation.

Only invented arrays are used. No medicinal spectra, NPZ, model, or QC job.
"""
from dataclasses import asdict, replace
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from herb_spectra.experimental_peaks import analyze_source_peaks, LOCKED_PEAK_PROTOCOL
from herb_spectra.leakage_guard import FitScope, LeakageError
SPEC = importlib.util.spec_from_file_location("species_peak_regression_worker", ROOT / "scripts/research4_species_correction_analysis_20261108.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class OriginalLockedPeaks(unittest.TestCase):
    def setUp(self):
        self.axis = np.linspace(.1, 1.5, 101)
        self.ids = ["synthetic-A", "synthetic-B", "synthetic-C"]
        self.values = np.zeros((3, 101)); self.values[:, 50] = 1
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def scope(self, ids=None):
        return FitScope(ids or self.ids, "synthetic-locked-peaks", "synthetic-only",
                        Path(self.tmp.name) / "audit.jsonl", code_version="original-module-regression")

    def test_01_exact_original_parameter_snapshot(self):
        self.assertEqual(asdict(LOCKED_PEAK_PROTOCOL), M.PEAK_PARAMETERS)
        self.assertEqual(M.PEAK_PARAMETERS["minimum_frequency_distance_thz"], .03)

    def test_02_prominence_uses_raw_mean_ptp(self):
        result = analyze_source_peaks(self.values, self.axis, self.ids, self.scope())
        self.assertEqual(result["raw_source_mean_ptp"], 1)
        self.assertEqual(result["required_prominence"], .05)
        self.assertNotAlmostEqual(result["required_prominence"], .05 * float(np.ptp(result["smoothed_source_mean"])))

    def test_03_distance_is_inherited_ceil_conversion(self):
        result = analyze_source_peaks(self.values, self.axis, self.ids, self.scope())
        expected = max(1, int(math.ceil(.03 / float(np.median(np.diff(self.axis))))))
        self.assertEqual(result["minimum_distance_samples"], expected)

    def test_04_nonuniform_measured_axis_rejected(self):
        axis = self.axis.copy(); axis[20] += .001
        with self.assertRaises(LeakageError):
            analyze_source_peaks(self.values, axis, self.ids, self.scope())

    def test_05_numerical_floor_removes_roundoff_peak(self):
        values = np.ones((3, 101)); values[:, 50] += np.finfo(float).eps
        result = analyze_source_peaks(values, self.axis, self.ids, self.scope())
        self.assertEqual(result["status"], "NO_PEAKS")
        self.assertEqual(result["candidate_count"], 0)

    def test_06_changed_parameters_rejected(self):
        changed = replace(LOCKED_PEAK_PROTOCOL, prominence_fraction_of_raw_mean_ptp=.06)
        with self.assertRaises(LeakageError):
            analyze_source_peaks(self.values, self.axis, self.ids, self.scope(), protocol=changed)

    def test_07_nontraining_scope_membership_rejected(self):
        with self.assertRaises(LeakageError):
            analyze_source_peaks(self.values, self.axis, self.ids, self.scope(["other", "synthetic-B", "synthetic-C"]))

    def test_08_original_values_and_proof_preserved(self):
        before = self.values.copy()
        result = analyze_source_peaks(self.values, self.axis, self.ids, self.scope())
        self.assertTrue(np.array_equal(before, self.values))
        self.assertEqual(result["source_sample_ids"], self.ids)
        self.assertFalse(result["labels_used"])
        self.assertFalse(result["target_spectra_received"])
        self.assertFalse(result["classification_prior_used"])
        self.assertTrue(result["fit_provenance"])


if __name__ == "__main__":
    unittest.main()
