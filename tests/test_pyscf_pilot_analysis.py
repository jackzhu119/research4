"""Synthetic guard tests only; no DFT, SCF, gradient or real report export."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_pyscf_spinosin_pilot_20261031.py"
spec = importlib.util.spec_from_file_location("pyscf_report_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def fixture(identity="spinosin"):
    count, atoms = (219, 75) if identity == "spinosin" else (3, 3)
    values = np.linspace(5, 500, count)
    return {"identity": identity, "atoms": atoms, "status": "DONE", "geometry_optimized": True,
            "optimizer_converged": True, "scf_converged": True, "final_gradient_thresholds_passed": True,
            "frequencies_status": "DONE", "frequency_validation_passed": True,
            "all_frequencies_real_positive": True, "normal_modes_complete": True,
            "normal_mode_count": count, "expected_normal_mode_count": count,
            "harmonic_analysis_excludes_translations_and_rotations": True,
            "frequencies_cm1": values.tolist(), "frequencies_thz": (values * analysis.CM1_TO_THZ).tolist(),
            "imaginary_frequency_count": 0, "IR_intensities_status": "NOT_COMPUTED"}


class PySCFReportGuardTests(unittest.TestCase):
    def test_accepts_complete_synthetic_shape_only(self):
        self.assertTrue(analysis.valid_modes(fixture(), "spinosin"))

    def test_water_cannot_be_used_as_herbal_theory(self):
        water = fixture("water")
        self.assertTrue(analysis.valid_modes(water, "water"))
        self.assertFalse(analysis.valid_modes(water, "spinosin"))

    def test_incomplete_or_unoptimized_frequencies_not_accepted(self):
        for key, value in [("geometry_optimized", False), ("optimizer_converged", False),
                           ("status", "PARTIAL"), ("normal_mode_count", 218),
                           ("IR_intensities_status", "FAKE_COMPUTED")]:
            record = fixture()
            record[key] = value
            self.assertFalse(analysis.valid_modes(record, "spinosin"), key)

    def test_zero_imaginary_nonfinite_wrong_units_rejected(self):
        for number in [0, -2, np.nan]:
            record = fixture()
            record["frequencies_cm1"][0] = number
            self.assertFalse(analysis.valid_modes(record, "spinosin"))
        record = fixture()
        record["frequencies_thz"][0] += 1
        self.assertFalse(analysis.valid_modes(record, "spinosin"))

    def test_blocked_comparison_keeps_missing_numeric_values(self):
        observations = [{"experimental_frequency_thz": 1.08, "experimental_wavenumber_cm1": 1.08 / analysis.CM1_TO_THZ}]
        rows = analysis.comparison_rows(observations, [])
        self.assertIsNone(rows[0]["theoretical_frequency_thz"])
        self.assertIsNone(rows[0]["absolute_difference_thz"])
        self.assertFalse(rows[0]["chemical_attribution_verified"])
        modes = [{"phase": "synthetic", "mode": i, "frequency_thz": .9 + i * .1, "frequency_cm1": (.9 + i * .1) / analysis.CM1_TO_THZ} for i in [1, 2]]
        pairs = analysis.comparison_rows(observations, modes)
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(row["comparison_status"] == "DESCRIPTIVE_FREQUENCY_PAIR_ONLY" for row in pairs))

    def test_running_budget_and_inconsistent_cpu_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = root / "cpu_budget.json"
            value = {"phases": [{"status": "RUNNING"}], "consumed_cpu_seconds": 0, "user_total_cpu_cap_seconds": 3600}
            ledger.write_text(json.dumps(value))
            release = {"allow_analysis_export": True, "created_at_utc": "synthetic", "cpu_budget_sha256": analysis.sha(ledger)}
            with self.assertRaises(ValueError):
                analysis.authorize(root, release)
            value["phases"] = [{"status": "EXITED", "actual_cpu_seconds": 10}]
            ledger.write_text(json.dumps(value)); release["cpu_budget_sha256"] = analysis.sha(ledger)
            with self.assertRaises(ValueError):
                analysis.authorize(root, release)

    def test_experimental_unit_conversion_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "peaks.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["outer_fold", "candidate_number", "frequency_index", "frequency_thz", "wavenumber_cm1"])
                writer.writerow([0, 1, 20, 1.08, 1.08 / analysis.CM1_TO_THZ])
            self.assertEqual(len(analysis.experimental_rows(path)), 1)
            path.write_text(path.read_text().replace(str(1.08 / analysis.CM1_TO_THZ), "99"))
            with self.assertRaises(ValueError):
                analysis.experimental_rows(path)

    def test_mode_file_cardinality_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            np.save(root / "normal_modes.npy", np.zeros((2, 3, 3)))
            with self.assertRaises(ValueError):
                analysis.modes_from_phase(root, fixture("water"))

    def test_gated_full_synthetic_report_preserves_blocked_theory(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            root = project / "pilot"
            phase = root / "water-optfreq"
            phase.mkdir(parents=True)
            result = fixture("water")
            analysis.write(phase / "result.json", result)
            np.save(phase / "normal_modes.npy", np.eye(9)[:3].reshape(3, 3, 3))
            np.save(phase / "hessian_hartree_per_bohr2.npy", np.eye(9).reshape(3, 3, 3, 3).transpose(0, 2, 1, 3))
            with (phase / "frequencies.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["frequency_cm1", "frequency_thz"])
                writer.writerows(zip(result["frequencies_cm1"], result["frequencies_thz"]))
            ledger = {"phases": [{"phase": "synthetic_water", "command": ["synthetic", "--output", str(phase)],
                                  "actual_cpu_seconds": 0, "elapsed_wall_seconds": 0, "exit_code": 0, "status": "EXITED"}],
                      "consumed_cpu_seconds": 0, "user_total_cpu_cap_seconds": 3600}
            analysis.write(root / "cpu_budget.json", ledger)
            peaks = project / "reports/upgrade-20261031-v1/source-peaks/experimental_candidates_by_fold.csv"
            peaks.parent.mkdir(parents=True)
            with peaks.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["outer_fold", "candidate_number", "frequency_index", "frequency_thz", "wavenumber_cm1"])
                writer.writerow([0, 1, 20, 1.08, 1.08 / analysis.CM1_TO_THZ])
            gate = project / "release.json"
            analysis.write(gate, {"allow_analysis_export": True, "created_at_utc": "synthetic", "cpu_budget_sha256": analysis.sha(root / "cpu_budget.json")})
            old_root = analysis.ROOT
            analysis.ROOT = project
            try:
                summary = analysis.run(root, project / "analysis", gate)
            finally:
                analysis.ROOT = old_root
            self.assertEqual(summary["validated_spinosin_frequency_rows"], 0)
            self.assertEqual(summary["water_validated_frequency_rows"], 3)
            self.assertFalse(summary["chemical_attribution_verified"])
            self.assertTrue((project / "analysis/water_mode_positions.pdf").exists())
            with (project / "analysis/experiment_theory_pairs.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["theoretical_frequency_thz"], "")


if __name__ == "__main__":
    unittest.main()
