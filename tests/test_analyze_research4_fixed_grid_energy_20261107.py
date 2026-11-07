"""Pure metadata/scalar tests; no PySCF, Git or quantum calls."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/analyze_research4_fixed_grid_energy_20261107.py"
spec = importlib.util.spec_from_file_location("local_energy_analysis_test", SOURCE)
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


class EnergyReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.path = self.base / "result.json"
        energies = {"fresh-zero": -76., "plus": -76. + 3e-8, "minus": -76. + 2e-8}
        self.value = {"schema": M.SCHEMA, "status": "ENERGY_CURVATURE_COMPUTED", "case": "Spinosin",
            "method": "FIXED_ORIGINAL_QUADRATURE", "predetermined_mode_index_1based": 1,
            "predetermined_step_bohr": M.STEP, "all_three_fresh_energies_completed": True,
            "completed_points": list(M.NAMES), "stable_MF_state_unchanged": True,
            "same_fresh_zero_MF_and_density_preserved_for_both_signs": True,
            "source_worker": M.ref(M.WORKER), "snapshot_manifest_sha256": "a" * 64,
            "points": {}, "energies": energies.copy(), "energies_hartree": energies.copy()}
        for name, sign in zip(M.NAMES, (0, 1, -1)):
            directory = self.base / name; directory.mkdir()
            point = {"schema": M.POINT_SCHEMA, "status": M.POINT_SUCCESS, "energy_completed": True,
                "SCF_performed": True, "scf_converged": True, "source_worker": self.value["source_worker"],
                "source_snapshot_manifest_sha256": "a" * 64, "parent_output_directory": str(self.base),
                "energy_hartree": energies[name], "displacement": {"mode_index_1based": 1,
                    "coordinate_index_0based": None, "sign": sign, "step_bohr": 0. if sign == 0 else M.STEP,
                    "direction_normalization": "EUCLIDEAN_CARTESIAN_UNIT_NORM"},
                "final_SCF_endpoint_audit": {"qualified": True, "certification_policy": M.POLICY,
                    "actual_native_final_extra_after_accepted_main_same_kernel": True,
                    "current_full_energy_hartree": energies[name],
                    "full_consecutive_density_energy_delta_abs_hartree": 1e-13,
                    "full_current_orbital_gradient_norm": 1e-11, "native_kernel_converged": False}}
            path = directory / "native_result.json"; path.write_text(json.dumps(point))
            self.value["points"][name] = {"result": M.ref(path), "status": M.POINT_SUCCESS,
                "energy_completed": True, "energy_hartree": energies[name]}
        numerator = math.fsum((energies["plus"], energies["minus"], -2 * energies["fresh-zero"]))
        curvature = numerator / M.STEP**2
        self.value["central_energy_analysis"] = {"Ezero_hartree": energies["fresh-zero"],
            "Eplus_hartree": energies["plus"], "Eminus_hartree": energies["minus"], "step_bohr": M.STEP,
            "compensated_energy_numerator_hartree": numerator,
            "central_energy_curvature_Eh_per_Bohr2": curvature,
            "analytic_directional_curvature_Eh_per_Bohr2": -.0375,
            "central_energy_minus_analytic_curvature_Eh_per_Bohr2": curvature + .0375}
        self.path.write_text(json.dumps(self.value))

    def tearDown(self):
        self.temp.cleanup()

    def test_real_three_children_replay_compensated_curvature_and_keep_native_false(self):
        points, comparison = M.validate_triplet(self.value, self.path)
        self.assertEqual(len(points), 3)
        self.assertFalse(points["plus"]["native_kernel_converged"])
        self.assertEqual(comparison["central_energy_curvature_Eh_per_Bohr2"],
                         self.value["central_energy_analysis"]["central_energy_curvature_Eh_per_Bohr2"])

    def test_partial_status_and_missing_triplet_rejected(self):
        for mutation in ("status", "missing", "complete_flag"):
            value = copy.deepcopy(self.value)
            if mutation == "status": value["status"] = "FAILED_INCOMPLETE_ENERGY_TRIPLET"
            if mutation == "missing": del value["points"]["minus"]
            if mutation == "complete_flag": value["all_three_fresh_energies_completed"] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                M.validate_triplet(value, self.path)

    def test_reported_energy_numerator_curvature_and_difference_mismatch_rejected(self):
        for key in ("Ezero_hartree", "compensated_energy_numerator_hartree",
                    "central_energy_curvature_Eh_per_Bohr2", "central_energy_minus_analytic_curvature_Eh_per_Bohr2"):
            value = copy.deepcopy(self.value); value["central_energy_analysis"][key] += .001
            with self.subTest(key=key), self.assertRaises(ValueError):
                M.validate_triplet(value, self.path)

    def test_child_hash_change_and_parent_energy_change_rejected(self):
        value = copy.deepcopy(self.value); value["energies_hartree"]["plus"] += .1
        with self.assertRaises(ValueError): M.validate_triplet(value, self.path)
        path = self.base / "plus/native_result.json"; path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"): M.validate_triplet(self.value, self.path)

    def test_wrong_sign_duplicate_reference_and_uncertified_endpoint_rejected(self):
        value = copy.deepcopy(self.value); value["points"]["minus"]["result"] = value["points"]["plus"]["result"]
        with self.assertRaises(ValueError): M.validate_triplet(value, self.path)
        path = self.base / "plus/native_result.json"; point = json.loads(path.read_text())
        point["final_SCF_endpoint_audit"]["qualified"] = False; path.write_text(json.dumps(point))
        self.value["points"]["plus"]["result"] = M.ref(path)
        with self.assertRaisesRegex(ValueError, "certificate"): M.validate_triplet(self.value, self.path)

    def test_unclosed_or_unreviewed_producer_cannot_authenticate(self):
        value = copy.deepcopy(self.value); value["source_worker"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            M.authenticate_closed(value, self.path)

    def test_bad_top_result_hash_rejected_before_any_artifact_is_written(self):
        output = self.base / "analysis"
        with self.assertRaises(ValueError): M.analyze(self.path, "0" * 64, output)
        self.assertFalse(output.exists())


if __name__ == "__main__": unittest.main()
