"""Tiny generic eigenvector fixtures only; never read target modes or run QM."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/analyze_spinosin_group_participation_20261104.py"
before = set(sys.modules)
spec = importlib.util.spec_from_file_location("pure_group_participation", SOURCE)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
quantum_imports = {name for name in set(sys.modules)-before if name == "pyscf" or name.startswith("pyscf.")}


def groups():
    return [{"group_id": f"part{i}", "family": analysis.PARTITION,
             "indices_zero_based": [i], "indices_one_based": [i+1]} for i in range(4)]+[
        {"group_id": "overlap01", "family": analysis.OVERLAP, "indices_zero_based": [0, 1], "indices_one_based": [1, 2]},
        {"group_id": "overlap12", "family": analysis.OVERLAP, "indices_zero_based": [1, 2], "indices_one_based": [2, 3]}]


def qualified_checks():
    return {"schema": "research4-V5-independent-modes-analysis-v1", "status": "DONE_COMPUTED_MODES_AUDITED",
            "case": "Spinosin", "atoms": 75, "normal_modes": 219, "qualified_for_mechanism": True,
            "all_frequencies_positive": True, "imaginary_frequency_count": 0, "zero_frequency_count": 0,
            "Spinosin_target_status": "ACTUAL_CLOSED_FREQUENCIES_AUDITED", "analysis_source_sha256": analysis.ANALYZER_SHA,
            "consumer_sha256": analysis.CONSUMER_SHA, "snapshot_manifest_sha256": analysis.SNAPSHOT_SHA,
            "IR_intensities_status": "NOT_COMPUTED", "PySCF_imported": False, "new_quantum_evaluations": 0,
            "chemical_attribution_verified": False, "theory_or_source_peaks_enter_classifier": False,
            "source_MF_unchanged_verified": True}


class GroupParticipationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pure-group-fixture-", dir=ROOT / "reports")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def test_import_has_no_quantum_engine(self):
        self.assertEqual(quantum_imports, set())

    def test_mass_weighting_is_not_unweighted_displacement(self):
        modes = np.zeros((1, 4, 3)); modes[0, :2, 0] = 1
        result, total = analysis.participation(modes, np.arange(1, 5), groups())
        np.testing.assert_allclose(result[0], [1/3, 2/3, 0, 0, 1, 2/3], atol=1e-15)
        np.testing.assert_allclose(total, 1)

    def test_all_atoms_equal_displacement_mass_fraction_and_overlaps_separate(self):
        result, total = analysis.participation(np.ones((1, 4, 3)), np.arange(1, 5), groups())
        np.testing.assert_allclose(result[0], [.1, .2, .3, .4, .3, .5])
        self.assertAlmostEqual(total[0], 1)
        self.assertAlmostEqual(result.sum(), 1.8)  # Never require overlapping groups to sum to one.

    def test_sign_and_per_mode_global_scaling_invariance(self):
        rng = np.random.default_rng(17)
        modes = rng.normal(size=(3, 4, 3))
        expected, _ = analysis.participation(modes, np.arange(1, 5), groups())
        actual, _ = analysis.participation(modes*np.array([-1, 1e150, 1e-150])[:, None, None], np.arange(1, 5), groups())
        np.testing.assert_allclose(actual, expected, atol=2e-16)

    def test_rotation_within_degenerate_subspace_changes_individual_mode_shares(self):
        modes = np.zeros((2, 4, 3)); modes[0, 0, 0] = 1; modes[1, 1, 0] = 1
        old, _ = analysis.participation(modes, np.ones(4), groups())
        rotated = np.einsum("ij,jab->iab", [[.6, .8], [-.8, .6]], modes)
        new, _ = analysis.participation(rotated, np.ones(4), groups())
        self.assertFalse(np.allclose(old, new))
        np.testing.assert_allclose(new[:, :4].sum(axis=0), old[:, :4].sum(axis=0))

    def test_zero_nonfinite_complex_and_wrongshape_rejected(self):
        good = np.ones((1, 4, 3))
        invalid = [np.zeros_like(good), good+np.nan, good+np.inf, good.astype(complex), np.ones((1, 4, 2)), np.ones((4, 3))]
        for modes in invalid:
            with self.subTest(shape=modes.shape), self.assertRaises(ValueError):
                analysis.participation(modes, np.ones(4), groups())

    def test_nonpositive_wrongshape_and_complex_masses_rejected(self):
        for masses in ([1, 2, 3], [1, 2, 3, 0], [1, -2, 3, 4], [1, 2, 3, np.nan], np.ones(4, complex)):
            with self.subTest(masses=masses), self.assertRaises(ValueError):
                analysis.participation(np.ones((1, 4, 3)), masses, groups())

    def test_duplicate_range_noninteger_and_onebased_errors_rejected(self):
        for indices, onebased in (([0, 0], [1, 1]), ([4], [5]), ([.0], [1]), ([0], [0])):
            candidate = groups(); candidate[0]["indices_zero_based"] = indices; candidate[0]["indices_one_based"] = onebased
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                analysis.validate_groups(candidate, 4)

    def test_partition_overlap_gap_unknown_family_duplicate_id_rejected(self):
        invalid = []
        candidate = groups(); candidate[1].update(indices_zero_based=[0], indices_one_based=[1]); invalid.append(candidate)
        invalid.append(groups()[1:])
        candidate = groups(); candidate[0]["family"] = "OTHER"; invalid.append(candidate)
        candidate = groups(); candidate[1]["group_id"] = "part0"; invalid.append(candidate)
        for candidate in invalid:
            with self.assertRaises(ValueError):
                analysis.validate_groups(candidate, 4)

    def test_real_static_map_has_22_groups_four_partitions_75_identity_atoms_no_arrays(self):
        hashes = {}
        with patch.object(analysis.np, "load", side_effect=AssertionError("No target arrays")):
            actual, symbols = analysis.static_groups(hashes)
        self.assertEqual(len(actual), 22); self.assertEqual(len(symbols), 75)
        self.assertEqual(sum(g["family"] == analysis.PARTITION for g in actual), 4)
        self.assertEqual(len(hashes), 4)

    def test_metadata_minimum_and_case_gate_strict_types(self):
        analysis.validate_analysis_checks(qualified_checks())
        failures = {"qualified_for_mechanism": [False, 1], "case": ["water_software_control"],
                    "normal_modes": [218], "imaginary_frequency_count": [1], "zero_frequency_count": [1],
                    "all_frequencies_positive": [False], "status": ["NOT_RUN"], "PySCF_imported": [True],
                    "source_MF_unchanged_verified": [False], "analysis_source_sha256": ["0"*64]}
        for key, values in failures.items():
            for value in values:
                candidate = qualified_checks(); candidate[key] = value
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "BLOCKED"):
                    analysis.validate_analysis_checks(candidate)

    def invalid_export(self, checks):
        (self.base / "checks.json").write_text(json.dumps(checks))
        data = {"analysis_source_sha256": analysis.ANALYZER_SHA,
                "files": [{"path": "checks.json", "sha256": analysis.sha(self.base / "checks.json"), "bytes": (self.base / "checks.json").stat().st_size}]}
        (self.base / "artifact_index.json").write_text(json.dumps(data))
        return analysis.sha(self.base / "artifact_index.json")

    def test_blocked_analysis_cannot_read_arrays_or_invoke_consumer(self):
        candidate = qualified_checks(); candidate["qualified_for_mechanism"] = False
        digest = self.invalid_export(candidate)
        with patch.object(analysis.np, "load", side_effect=AssertionError("No target arrays")), \
             patch.object(analysis, "load_frozen", side_effect=AssertionError("No consumer until qualified")), \
             self.assertRaisesRegex(ValueError, "BLOCKED"):
            analysis.analyze(self.base, digest, self.base / "output")
        self.assertFalse((self.base / "output").exists())

    def test_incomplete_original_result_fails_before_arrays(self):
        source = self.base / "result.json"; source.write_text(json.dumps({"status": "PARTIAL"}))
        candidate = qualified_checks(); candidate.update(source_result_sha256=analysis.sha(source), input_hashes={str(source): {"sha256": analysis.sha(source)}})
        digest = self.invalid_export(candidate)
        with patch.object(analysis.np, "load", side_effect=AssertionError("No target arrays")), \
             self.assertRaisesRegex(ValueError, "Incomplete original"):
            analysis.analyze(self.base, digest, self.base / "output")

    def test_wrong_artifact_index_sha_rejected_before_arrays(self):
        self.invalid_export(qualified_checks())
        with patch.object(analysis.np, "load", side_effect=AssertionError("No target arrays")), \
             self.assertRaisesRegex(ValueError, "Input SHA"):
            analysis.analyze(self.base, "0"*64, self.base / "output")

    def closure_fixture(self):
        # Deliberately synthetic metadata needed to reach a mocked failing gate.
        # No target eigenvectors, target frequencies or numerical results are read.
        result = {"schema": "research4-resumable-full-rks-hessian-result-v1", "status": "FREQUENCIES_COMPUTED",
                  "frequencies_status": "DONE", "snapshot_manifest_sha256": analysis.SNAPSHOT_SHA,
                  "atoms": 75, "normal_mode_count": 219, "expected_normal_mode_count": 219,
                  "IR_intensities_status": "NOT_COMPUTED", "frequencies_cm1": [1.] * 219,
                  "frequencies_thz": [analysis.CM1_TO_THZ] * 219, "source_worker": {}, "resource_supervision": {},
                  "phase": "SYNTHETIC_UNIT_FIXTURE_NOT_DFT", "boot_id": "synthetic-boot",
                  "gradient_gate": {"result": {"path": "synthetic-no-file", "sha256": analysis.GRADIENT_SHA}}}
        for key in ("complete_global_Hessian", "complete_RHF_helper", "normal_modes_complete", "stable_MF_state_unchanged",
                    "no_new_SCF_performed", "all_frequencies_real_positive", "stationary_minimum_verified"):
            result[key] = True
        path = self.base / "result.json"; path.write_text(json.dumps(result))
        closure = {"producer_result_sha256": analysis.sha(path)}
        candidate = qualified_checks()
        candidate.update(source_result_sha256=analysis.sha(path), producer_closure=closure,
                         input_hashes={str(path): {"sha256": analysis.sha(path)}})
        digest = self.invalid_export(candidate)
        return digest, closure

    def test_unhealthy_original_producer_fails_before_any_array_read(self):
        digest, _ = self.closure_fixture()
        def reject(*args, **kwargs):
            raise ValueError("synthetic unhealthy producer")
        consumer = SimpleNamespace(EXECUTION_SCHEMA="fixture", verify_closed_execution=reject)
        with patch.object(analysis, "load_frozen", return_value=consumer), \
             patch.object(analysis.np, "load", side_effect=AssertionError("No target arrays")), \
             self.assertRaisesRegex(ValueError, "unhealthy producer"):
            analysis.analyze(self.base, digest, self.base / "output")

    def test_invalid_fresh_gradient_fails_before_any_array_read(self):
        digest, closure = self.closure_fixture()
        def reject(*args, **kwargs):
            self.assertEqual(args[-1], "synthetic-boot")
            raise ValueError("synthetic invalid gradient")
        consumer = SimpleNamespace(EXECUTION_SCHEMA="fixture", verify_closed_execution=lambda *a, **kw: closure,
                                   qualify_gradient_receipt=reject)
        with patch.object(analysis, "load_frozen", return_value=consumer), \
             patch.object(analysis.np, "load", side_effect=AssertionError("No target arrays")), \
             self.assertRaisesRegex(ValueError, "invalid gradient"):
            analysis.analyze(self.base, digest, self.base / "output")

    def test_preflight_is_not_run_has_no_participation_csv_or_target_arrays(self):
        output = self.base / "preflight"
        with patch.object(analysis.np, "load", side_effect=AssertionError("No target arrays")):
            result = analysis.preflight(output)
        self.assertEqual(result["status"], "NOT_RUN")
        self.assertFalse(result["participation_values_computed"])
        self.assertFalse(result["target_arrays_read"])
        self.assertEqual(list(output.glob("*.csv")), [])
        self.assertEqual({p.name for p in output.iterdir()}, {"preflight.json", "README.md", "artifact_index.json"})
        with self.assertRaisesRegex(ValueError, "overwrite"):
            analysis.preflight(output)


if __name__ == "__main__":
    unittest.main()
