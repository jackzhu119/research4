"""Pure/mocked worker guards; no quantum calculations in this test module."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location("pilot_worker", Path(__file__).parents[1] / "scripts/pyscf_pilot_worker_20261031.py")
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class PilotWorkerTests(unittest.TestCase):
    def test_xyz_composition_and_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "water.xyz"
            path.write_text("3\nfixture only\nO 0 0 0\nH 0 0 1\nH 1 0 0\n")
            symbols, xyz = worker.read_xyz(path, worker.FORMULAS["water"])
            self.assertEqual(symbols, ["O", "H", "H"])
            self.assertEqual(xyz.shape, (3, 3))
            with self.assertRaises(ValueError):
                worker.read_xyz(path, worker.FORMULAS["spinosin"])
            path.write_text("3\nfixture only\nO 0 0 0\nH 0 0 .01\nH 1 0 0\n")
            with self.assertRaises(ValueError):
                worker.read_xyz(path, worker.FORMULAS["water"])

    def test_gradient_thresholds_use_atom_norms(self):
        # All components individually small, but one atom vector exceeds gmax.
        gradient = np.asarray([[.0003, .0003, .0003], [0, 0, 0], [0, 0, 0]])
        result = worker.stationary_checks(gradient)
        self.assertFalse(result["final_gradient_thresholds_passed"])
        self.assertGreater(result["atom_max_hartree_per_bohr"], .00045)
        self.assertTrue(worker.stationary_checks(np.zeros((3, 3)))["final_gradient_thresholds_passed"])

    def test_mode_count_signed_frequency_units_and_no_ir(self):
        result = worker.frequencies_payload([-10., 1000., 2000.], np.ones((3, 3, 3)), 3)
        self.assertEqual(result["imaginary_frequency_count"], 1)
        self.assertEqual(result["IR_intensities_status"], "NOT_COMPUTED")
        self.assertAlmostEqual(result["frequencies_thz"][0], -.299792458)
        self.assertFalse(result["all_frequencies_real_positive"])
        with self.assertRaises(ValueError):
            worker.frequencies_payload([1, 2], np.ones((2, 3, 3)), 3)
        with self.assertRaises(ValueError):
            worker.frequencies_payload([1, 2, np.nan], np.ones((3, 3, 3)), 3)

    def test_frequency_requires_stationary_same_method_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "optimized.xyz"
            path.write_text("fixture bytes only")
            settings = {k: v for k, v in zip(("xc", "basis", "grid_level", "dispersion", "density_fitting", "auxbasis"),
                                            ("PBE", "def2-svp", 1, "NONE", True, "jfit"))}
            result = {"status": "DONE", "geometry_optimized": True, "optimizer_converged": True,
                      "scf_converged": True, "final_gradient_thresholds_passed": True,
                      "optimized_xyz_sha256": worker.sha(path), "settings": settings}
            worker.require_optimized_result(result, path, settings)
            for key in ("geometry_optimized", "optimizer_converged", "scf_converged", "final_gradient_thresholds_passed"):
                bad = dict(result, **{key: False})
                with self.assertRaises(ValueError):
                    worker.require_optimized_result(bad, path, settings)
            with self.assertRaises(ValueError):
                worker.require_optimized_result(result, path, dict(settings, grid_level=2))
            path.write_text("changed")
            with self.assertRaises(ValueError):
                worker.require_optimized_result(result, path, settings)

    def test_xyz_output_roundtrip_and_bad_modes(self):
        text = worker.xyz_text(["O", "H", "H"], [[0, 0, 0], [0, 0, 1], [1, 0, 0]], "fixture")
        self.assertTrue(text.startswith("3\nfixture\n"))
        with self.assertRaises(ValueError):
            worker.xyz_text(["O"], [[0, 0, np.nan]], "bad")
        with self.assertRaises(ValueError):
            worker.frequencies_payload(np.zeros(219), np.zeros((219, 74, 3)), 75)


if __name__ == "__main__":
    unittest.main()
