"""Continuity and scientific-label checks; no large quantum calculations."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import tempfile
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "scripts/pyscf_continuous_science_20261108_v3.py"
spec = importlib.util.spec_from_file_location("continuous_science_v3_tests", SOURCE)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class ContinuousScienceTests(unittest.TestCase):
    def config(self):
        return worker.settings(SimpleNamespace(grid_level=3, conv_tol=1e-10, conv_grad=1e-7))

    def test_central_gradient_columns_keep_signed_curvature(self):
        rng = np.random.default_rng(771)
        matrix = rng.normal(size=(9, 9)); matrix = .5 * (matrix + matrix.T)
        matrix[0, 0] = -4.
        center = rng.normal(size=9)
        h = .001
        plus = np.array([(center + h * matrix[:, i]).reshape(3, 3) for i in range(9)])
        minus = np.array([(center - h * matrix[:, i]).reshape(3, 3) for i in range(9)])
        np.testing.assert_allclose(worker.assemble_hessian(plus, minus, h), matrix, atol=1e-12)

    def test_nonfinite_or_incomplete_gradients_never_make_a_complete_hessian(self):
        plus = np.zeros((9, 3, 3)); minus = np.zeros_like(plus)
        plus[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            worker.assemble_hessian(plus, minus, .001)
        with self.assertRaises(ValueError):
            worker.assemble_hessian(np.zeros((8, 3, 3)), np.zeros((8, 3, 3)), .001)

    def test_projection_removes_six_mass_weighted_external_modes(self):
        coordinates = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        masses = np.array([16., 1., 1.])
        _, _, basis, rank = worker.vibrational_modes(np.eye(9), coordinates, masses)
        self.assertEqual(rank, 6)
        self.assertEqual(basis.shape, (9, 3))
        weights = np.repeat(np.sqrt(masses), 3)
        expected = np.array([-2., 3., 4.])
        hessian = (basis @ np.diag(expected) @ basis.T) * weights[:, None] * weights[None, :]
        frequencies, values, vectors, rank = worker.vibrational_modes(hessian, coordinates, masses)
        np.testing.assert_allclose(values, expected, atol=1e-12)
        self.assertEqual(int(np.sum(frequencies < 0)), 1)
        translation = np.zeros((3, 3)); translation[:, 0] = np.sqrt(masses)
        np.testing.assert_allclose(translation.ravel() @ vectors, 0., atol=1e-12)

    def test_finite_unconverged_state_produces_gradient_and_warning(self):
        molecule = SimpleNamespace(natm=2, atom_coords=lambda: np.zeros((2, 3)),
                                   atom_symbol=lambda i: "He")
        molecule.copy = lambda: molecule
        mf = SimpleNamespace(converged=False, mo_coeff=np.eye(2), kernel=lambda **kw: -1.,
                             make_rdm1=lambda: np.eye(2), get_hcore=lambda: np.eye(2),
                             get_veff=lambda *a: np.eye(2), energy_tot=lambda *a: -1.,
                             get_fock=lambda *a, **kw: np.eye(2), get_ovlp=lambda: np.eye(2),
                             get_grad=lambda *a: np.array([3e-5]))
        mf.newton = mock.Mock(side_effect=ValueError("bounded Newton failure"))
        mf.nuc_grad_method = lambda: SimpleNamespace(grid_response=False, kernel=lambda: np.ones((2, 3)) * 1e-4)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(worker, "make_scf", return_value=mf):
            point, density, gradient = worker.calculate_attempt(molecule, Path(tmp), self.config(), None, True, False)
            self.assertEqual(point["status"], "POINT_COMPUTED")
            self.assertFalse(point["native_pyscf_converged"])
            self.assertIn("NATIVE_SCF_UNCONVERGED_FINITE_STATE_USED", point["quality_warnings"])
            self.assertTrue(np.isfinite(gradient).all())
            self.assertEqual(gradient.shape, (2, 3))

    def test_point_resume_does_not_require_displacement_density(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            gradient = worker.array_ref(out / "gradient.npy", np.ones((3, 3)))
            density = worker.array_ref(out / "local-only/density.npy", np.eye(3))
            worker.save(out / "result.json", {"status": "POINT_COMPUTED", "input_fingerprint": "fixed",
                                              "gradient": gradient, "density": density})
            Path(density["path"]).unlink()
            with mock.patch.object(worker, "fingerprint", return_value=("fixed", {})), \
                 mock.patch.object(worker, "calculate_attempt") as compute:
                point, saved_density, actual = worker.evaluate_point(None, out, self.config(), save_density=False)
            compute.assert_not_called()
            self.assertIsNone(saved_density)
            np.testing.assert_array_equal(actual, np.ones((3, 3)))

    def test_orphan_attempt_cannot_be_relabelled_with_different_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); attempt = out / "attempt-001-1"
            worker.save(attempt / "input.json", {"input_fingerprint": "old-coordinate-or-density"})
            worker.save(attempt / "result.json", {"status": "POINT_COMPUTED"})
            with mock.patch.dict("os.environ", {"R4_ATTEMPT": "1"}), \
                 mock.patch.object(worker, "fingerprint", return_value=("new-coordinate-or-density", {})), \
                 mock.patch.object(worker, "calculate_attempt") as compute:
                with self.assertRaises(ValueError):
                    worker.evaluate_point(None, out, self.config())
            compute.assert_not_called()

    def test_all_finite_warning_points_still_form_the_complete_hessian(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); geometry = out / "geometry.json"; geometry.write_text("{}")
            coordinates = worker.array_ref(out / "coordinates.npy", np.zeros((3, 3)))
            record = {"coordinates_bohr": coordinates, "density": {"sha256": "same-zero"},
                      "optimizer_converged": False, "quality_warnings": ["not stationary"]}
            def evaluate(mol, path, cfg, dm0=None, **kwargs):
                calls.append(path.name)
                point = {"status": "POINT_COMPUTED", "native_pyscf_converged": False,
                         "quality_warnings": ["unconverged finite gradient"]}
                worker.save(path / "result.json", point)
                return point, None, np.zeros((3, 3))
            with mock.patch.object(worker, "load_source", return_value=(record, ["O", "H", "H"], np.zeros((3, 3)), np.eye(2))), \
                 mock.patch.object(worker, "new_molecule", return_value=None), \
                 mock.patch.object(worker, "evaluate_point", side_effect=evaluate):
                result = worker.stage_hessian(SimpleNamespace(geometry=geometry, step=.001, output=out / "hessian"), self.config())
            self.assertEqual(len(calls), 18)
            self.assertEqual(result["status"], "HESSIAN_COMPUTED")
            self.assertEqual(result["actual_finite_points"], 18)
            self.assertEqual(result["native_unconverged_points"], 18)
            self.assertFalse(result["center_strict_stationarity_quality"])
            np.testing.assert_array_equal(worker.read_ref(result["raw_hessian"]), np.zeros((9, 9)))

    def test_hard_point_error_has_only_two_recorded_attempts(self):
        calls = []
        def fail(mol, out, cfg, dm0, nuclear_gradient, save_density):
            calls.append(out)
            worker.save(out / "result.json", {"status": "POINT_ERROR", "error": "real point failure"})
            raise ValueError("real point failure")
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(worker, "fingerprint", return_value=("fixed", {})), \
             mock.patch.object(worker, "calculate_attempt", side_effect=fail):
            with self.assertRaises(ValueError):
                worker.evaluate_point(None, Path(tmp), self.config())
            self.assertEqual(len(calls), 2)
            record = json.loads((Path(tmp) / "result.json").read_text())
            self.assertEqual(record["status"], "POINT_ERROR")
            self.assertEqual(len(record["failed_attempts"]), 2)

    def test_hessian_attempts_every_coordinate_after_a_hard_error(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); geometry = out / "geometry.json"; geometry.write_text("{}")
            coordinates = worker.array_ref(out / "coordinates.npy", np.zeros((3, 3)))
            record = {"coordinates_bohr": coordinates, "density": {"sha256": "same-zero"},
                      "optimizer_converged": False, "quality_warnings": ["not stationary"]}
            def evaluate(mol, path, cfg, dm0=None, **kwargs):
                calls.append(path.name)
                if path.name == "c000-plus":
                    raise ValueError("point exhausted attempts")
                point = {"status": "POINT_COMPUTED", "native_pyscf_converged": False,
                         "quality_warnings": ["unconverged finite gradient"]}
                worker.save(path / "result.json", point)
                return point, None, np.zeros((3, 3))
            with mock.patch.object(worker, "load_source", return_value=(record, ["O", "H", "H"], np.zeros((3, 3)), np.eye(2))), \
                 mock.patch.object(worker, "new_molecule", return_value=None), \
                 mock.patch.object(worker, "evaluate_point", side_effect=evaluate):
                result = worker.stage_hessian(SimpleNamespace(geometry=geometry, step=.001, output=out / "hessian"), self.config())
            self.assertEqual(len(calls), 18)
            self.assertEqual(result["status"], "HESSIAN_PARTIAL")
            self.assertEqual(result["actual_finite_points"], 17)
            self.assertEqual(result["native_unconverged_points"], 17)
            self.assertNotIn("raw_hessian", result)

    def test_partial_hessian_analysis_records_partial_without_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); path = out / "partial.json"
            worker.save(path, {"status": "HESSIAN_PARTIAL", "actual_finite_points": 17})
            rec = worker.stage_analyze(SimpleNamespace(hessians=[str(path)], output=out / "analysis"), self.config())
            self.assertEqual(rec["status"], "ANALYSIS_PARTIAL")
            self.assertFalse(rec["minimum_verified"])
            self.assertNotIn("signed_frequencies_cm1", rec)

    def test_array_hash_tampering_remains_a_hard_data_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = worker.array_ref(Path(tmp) / "array.npy", np.ones((3, 3)))
            np.save(ref["path"], np.zeros((3, 3)))
            with self.assertRaises(ValueError):
                worker.read_ref(ref)

    def test_prepare_source_copies_finite_state_without_quantum_recalculation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            coords = np.arange(9, dtype=float).reshape(3, 3)
            density = np.eye(2)
            gradient = np.ones((3, 3)) * 1e-5
            source = {"status": "ELECTRONIC_POINT_CERTIFIED", "settings": self.config(),
                      "symbols": ["O", "H", "H"],
                      "coordinates_bohr": worker.array_ref(out / "old-coordinates.npy", coords),
                      "density": worker.array_ref(out / "old-density.npy", density),
                      "gradient": worker.array_ref(out / "old-gradient.npy", gradient),
                      "nuclear_gradient_rms": 1e-5, "nuclear_gradient_max": 1e-5,
                      "native_pyscf_converged": True}
            source_path = out / "source.json"; worker.save(source_path, source)
            args = SimpleNamespace(source_point=source_path, xyz=None, output=out / "prepared")
            with mock.patch.object(worker, "new_molecule", return_value=SimpleNamespace(atom_coords=lambda: coords)), \
                 mock.patch.object(worker, "evaluate_point") as compute:
                rec = worker.stage_prepare(args, self.config())
            compute.assert_not_called()
            self.assertEqual(rec["status"], "GEOMETRY_AVAILABLE")
            self.assertFalse(rec["optimizer_converged"])
            self.assertFalse(rec["minimum_verified"])
            self.assertEqual(rec["derivative_settings"], source["settings"])
            np.testing.assert_array_equal(worker.read_ref(rec["coordinates_bohr"]), coords)
            np.testing.assert_array_equal(worker.read_ref(rec["density"]), density)
            np.testing.assert_array_equal(worker.read_ref(rec["gradient"]), gradient)

    def test_grid5_hessian_continues_but_does_not_reuse_grid3_stationarity(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); geometry = out / "geometry.json"; geometry.write_text("{}")
            coords = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
            record = {"coordinates_bohr": worker.array_ref(out / "coordinates.npy", coords),
                      "density": {"sha256": "same-zero"}, "optimizer_converged": True,
                      "quality_warnings": [], "derivative_settings": self.config(),
                      "nuclear_gradient_rms": 0., "nuclear_gradient_max": 0.}
            actual_cfg = dict(self.config(), grid_level=5)
            def evaluate(mol, path, cfg, dm0=None, **kwargs):
                point = {"status": "POINT_COMPUTED", "native_pyscf_converged": True, "quality_warnings": []}
                worker.save(path / "result.json", point)
                return point, None, np.zeros((3, 3))
            with mock.patch.object(worker, "load_source", return_value=(record, ["O", "H", "H"], coords, np.eye(2))), \
                 mock.patch.object(worker, "new_molecule", return_value=None), \
                 mock.patch.object(worker, "evaluate_point", side_effect=evaluate):
                rec = worker.stage_hessian(SimpleNamespace(geometry=geometry, step=.001, output=out / "hessian"), actual_cfg)
            self.assertEqual(rec["status"], "HESSIAN_COMPUTED")
            self.assertEqual(rec["actual_finite_points"], 18)
            self.assertFalse(rec["center_stationarity_settings_match"])
            self.assertIn("CENTER_GRADIENT_METHOD_DIFFERS_FROM_HESSIAN_METHOD", rec["center_quality_warnings"])


if __name__ == "__main__":
    unittest.main()
