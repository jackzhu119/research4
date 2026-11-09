"""Minimum labels require numeric evidence; warnings never control execution."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import tempfile
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "scripts/pyscf_minimum_repair_20261109.py"
spec = importlib.util.spec_from_file_location("minimum_repair_tests", SOURCE)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class MinimumRepairTests(unittest.TestCase):
    def config(self):
        return worker.settings(SimpleNamespace(grid_level=3, conv_tol=1e-10, conv_grad=1e-9))

    def pair(self, directory, update=None):
        from pyscf.data import elements
        cfg = self.config()
        symbols = ["O", "H", "H"]
        coordinates = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        masses = np.array([elements.MASSES[elements.charge(s)] for s in symbols])
        _, _, basis, _ = worker.vibrational_modes(np.eye(9), coordinates, masses)
        roots = np.repeat(np.sqrt(masses), 3)
        matrix = (basis @ np.diag([.1, .2, .3]) @ basis.T) * roots[:, None] * roots[None, :]
        coordinate_ref = worker.array_ref(directory / "coordinates.npy", coordinates)
        paths = []
        for index, step in enumerate((.001, .0005)):
            rec = worker.result("HESSIAN_COMPUTED", cfg)
            rec.update(coordinates_bohr=coordinate_ref, symbols=symbols,
                       center_density_sha256="a" * 64, step_bohr=step,
                       raw_hessian=worker.array_ref(directory / ("h%d.npy" % index), matrix),
                       center_optimizer_converged=True,
                       center_stationarity_settings_match=True,
                       center_strict_stationarity_quality=True,
                       center_nuclear_gradient_rms=1e-6, center_nuclear_gradient_max=2e-6,
                       center_native_pyscf_converged=True,
                       center_fresh_orbital_gradient_norm=1e-11,
                       center_fresh_energy_recheck_difference_hartree=0.,
                       native_unconverged_points=0,
                       maximum_point_fresh_orbital_gradient_norm=1e-11,
                       point_fresh_orbital_residual_unavailable_count=0,
                       maximum_point_energy_recheck_difference_hartree=0.,
                       point_energy_recheck_unavailable_count=0,
                       expected_points=18, actual_finite_points=18,
                       center_quality_warnings=["FORMER_1E12_ENERGY_LABEL_DID_NOT_PASS"],
                       points_with_quality_warnings=[{"point": "c000-plus", "warnings": ["HISTORICAL_DIAGNOSTIC_WARNING"]}])
            if update:
                update(rec, index)
            path = directory / ("h%d.json" % index)
            worker.save(path, rec); paths.append(str(path))
        return paths

    def analyze(self, directory, update=None):
        paths = self.pair(directory, update)
        return worker.stage_analyze(SimpleNamespace(hessians=paths, output=directory / "analysis"), self.config())

    def test_all_numeric_evidence_can_verify_minimum_despite_warning_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.analyze(Path(tmp))
        self.assertEqual(rec["status"], "ANALYSIS_COMPUTED")
        self.assertTrue(rec["minimum_verified"])
        self.assertTrue(all(item["passed"] for item in rec["minimum_criteria"].values()))

    def test_nuclear_nonstationarity_labels_false_without_aborting_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.analyze(Path(tmp), lambda record, index: record.update(center_nuclear_gradient_max=8e-5))
        self.assertEqual(rec["status"], "ANALYSIS_COMPUTED")
        self.assertFalse(rec["minimum_verified"])
        self.assertTrue(any(not item["passed"] for item in rec["minimum_criteria"].values()))

    def test_point_fock_residual_above_numeric_bound_labels_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.analyze(Path(tmp), lambda record, index: record.update(maximum_point_fresh_orbital_gradient_norm=4e-7))
        self.assertEqual(rec["status"], "ANALYSIS_COMPUTED")
        self.assertFalse(rec["minimum_verified"])

    def test_missing_point_fock_evidence_cannot_verify_minimum(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.analyze(Path(tmp), lambda record, index: record.update(point_fresh_orbital_residual_unavailable_count=1))
        self.assertFalse(rec["minimum_verified"])

    def test_another_method_center_gradient_cannot_certify_this_hessian(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.analyze(Path(tmp), lambda record, index: record.update(center_stationarity_settings_match=False))
        self.assertEqual(rec["status"], "ANALYSIS_COMPUTED")
        self.assertFalse(rec["minimum_verified"])

    def test_two_identical_displacement_steps_are_not_step_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.analyze(Path(tmp), lambda record, index: record.update(step_bohr=.001))
        self.assertFalse(rec["minimum_verified"])

    def test_hessians_from_different_methods_cannot_certify_a_paired_minimum(self):
        def alter_second_method(record, index):
            if index == 1:
                record["settings"] = dict(record["settings"], grid_level=5)
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.analyze(Path(tmp), alter_second_method)
        self.assertEqual(rec["status"], "ANALYSIS_COMPUTED")
        self.assertFalse(rec["minimum_verified"])
        self.assertFalse(rec["minimum_criteria"]["two_distinct_step_hessians_same_method"]["passed"])

    def test_hessians_with_different_center_density_do_not_fabricate_paired_modes(self):
        def alter_second_center(record, index):
            if index == 1:
                record["center_density_sha256"] = "b" * 64
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.analyze(Path(tmp), alter_second_center)
        self.assertEqual(rec["status"], "ANALYSIS_PARTIAL")
        self.assertFalse(rec["minimum_verified"])
        self.assertNotIn("signed_frequencies_cm1", rec)

    def test_optimization_passes_the_requested_stricter_geometry_targets(self):
        from pyscf.geomopt import addons, geometric_solver
        cfg = self.config()
        molecule = SimpleNamespace()
        point = {"symbols": ["O", "H", "H"], "settings": cfg,
                 "coordinates_bohr": {"path": "not-read", "sha256": "a" * 64},
                 "density": {"path": "not-read", "sha256": "b" * 64},
                 "gradient": {"path": "not-read", "sha256": "c" * 64},
                 "native_pyscf_converged": True, "quality_warnings": [],
                 "nuclear_gradient_rms": 1e-6, "nuclear_gradient_max": 2e-6,
                 "diagnostics": {"fresh_orbital_gradient_norm": 1e-11,
                                 "energy_recheck_difference_hartree": 0.}}
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(worker, "input_molecule", return_value=(molecule, np.eye(2), {})), \
             mock.patch.object(addons, "as_pyscf_method", return_value=object()), \
             mock.patch.object(geometric_solver, "kernel", return_value=(True, molecule)) as optimize, \
             mock.patch.object(worker, "evaluate_point", return_value=(point, np.eye(2), np.zeros((3, 3)))):
            worker.stage_optimize(SimpleNamespace(output=Path(tmp), source_point=None,
                                  opt_energy=1e-8, opt_grms=3e-6, opt_gmax=1e-5), cfg)
        self.assertEqual(optimize.call_args.kwargs["convergence_energy"], 1e-8)
        self.assertEqual(optimize.call_args.kwargs["convergence_grms"], 3e-6)
        self.assertEqual(optimize.call_args.kwargs["convergence_gmax"], 1e-5)


if __name__ == "__main__":
    unittest.main()
