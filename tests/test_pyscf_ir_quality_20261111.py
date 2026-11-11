"""Independent unit, invariance, and deliberate-corruption tests for IR quality."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pyscf_ir_quality_20261111.py"
spec = importlib.util.spec_from_file_location("ir_quality_test_target", SCRIPT)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


class IRQualityTests(unittest.TestCase):
    def setUp(self):
        self.masses = np.array([1.0, 4.0])
        self.modes = np.eye(6)[:2] / np.sqrt(np.repeat(self.masses, 3))
        self.derivative = np.arange(18, dtype=float).reshape(6, 3) / 50
        self.steps = np.array([0.004, 0.002, 0.001])

    def test_prefactor_has_independent_correct_absolute_units(self):
        self.assertAlmostEqual(q.prefactor_si(), 974.8801098, places=6)
        self.assertLess(abs(q.prefactor_si() / q.prefactor_atomic() - 1), 2e-9)
        # One point charge e with mass 4 amu, unit Cartesian direction.
        modes = np.array([[0.5, 0.0, 0.0]])
        result = q.common_frame_derivative_comparison([np.eye(3)] * 2,
                    [0.002, 0.001], modes, [4])
        self.assertTrue(result["qualified"])
        self.assertAlmostEqual(result["ir_intensity_series_km_mol"][0][0],
                               974.8801098 / 4, places=6)

    def test_mass_normalization_rejects_euclidean_renormalization(self):
        modes = np.eye(6)[3:5] / 2
        self.assertTrue(q.validate_mode_normalization(modes, self.masses)["qualified"])
        self.assertFalse(q.validate_mode_normalization(modes * 2, self.masses)["qualified"])
        self.assertFalse(q.validate_mode_normalization(np.vstack([modes[0], modes[0]]), self.masses)["qualified"])

    def test_common_frame_exact_and_quadratic_series(self):
        correction = np.ones((6, 3)) * 0.3
        series = [self.derivative + h*h*correction for h in self.steps]
        result = q.common_frame_derivative_comparison(series, self.steps,
                                                     self.modes, self.masses)
        self.assertTrue(result["qualified"], result)
        rich = result["richardson"]
        self.assertFalse(rich["is_rigorous_error_bound"])
        self.assertTrue(rich["three_step_diagnostic_available"])
        np.testing.assert_allclose(rich["extrapolated_projected_derivatives"],
                                  self.modes @ self.derivative, atol=1e-14)
        np.testing.assert_allclose(rich["observed_difference_norm_ratios"], [4, 4], atol=1e-9)

    def test_same_intensity_opposite_dipole_vector_fails(self):
        result = q.common_frame_derivative_comparison(
            [self.derivative, -self.derivative], [0.002, 0.001], self.modes, self.masses)
        self.assertFalse(result["qualified"])
        self.assertIn("projected_derivative_vectors_not_converged", result["failures"])
        np.testing.assert_allclose(result["mode_intensity_absolute_differences_km_mol"], 0)

    def test_weak_peaks_use_absolute_error_floor(self):
        d = np.zeros((2, 6, 3))
        d[1, 0, 0] = 1e-5
        result = q.common_frame_derivative_comparison(d, [0.002, 0.001], self.modes, self.masses)
        self.assertTrue(result["qualified"], result)
        self.assertTrue(all(result["weak_peaks"]))

    def test_nonmonotonic_three_step_series_is_not_silent_success(self):
        ds = [self.derivative + z for z in [0.01, -0.01, 0.01]]
        result = q.common_frame_derivative_comparison(ds, self.steps, self.modes, self.masses)
        self.assertFalse(result["qualified"])
        self.assertTrue(all(result["richardson"]["non_asymptotic_or_noise_suspected"]))

    def test_steps_are_sorted_and_must_be_distinct_positive(self):
        ds = np.array([self.derivative] * 3)
        result = q.common_frame_derivative_comparison(ds, [0.001, 0.004, 0.002], self.modes, self.masses)
        self.assertEqual(result["steps_bohr_descending"], self.steps.tolist())
        for hs in [[0.001, 0.001], [0, 0.001], [-1, 0.001]]:
            self.assertFalse(q.common_frame_derivative_comparison(ds[:2], hs, self.modes, self.masses)["qualified"])

    def test_direct_normal_coordinate_path_and_unit_corruption(self):
        projected = self.modes @ self.derivative
        direct = np.array([projected + h*h*0.1 for h in self.steps])
        result = q.directional_comparison(projected, direct, self.steps)
        self.assertTrue(result["qualified"], result)
        result = q.directional_comparison(projected, direct * 0.529177, self.steps)
        self.assertFalse(result["qualified"])
        # Squaring alone would not detect a sign error in the direct path.
        self.assertFalse(q.directional_comparison(projected, -direct, self.steps)["qualified"])

    def test_degenerate_rotation_and_sign_do_not_change_group_result(self):
        a = self.modes
        theta = 0.71
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        b = rot @ a
        da = a @ self.derivative
        db = rot @ da
        result = q.degenerate_subspace_comparison(a, b, da, db, self.masses)
        self.assertTrue(result["qualified"], result)
        self.assertGreater(np.max(np.abs(np.sum(da**2, axis=1) - np.sum(db**2, axis=1))), 1e-4)
        self.assertTrue(q.degenerate_subspace_comparison(a, -a, da, -da, self.masses)["qualified"])
        self.assertFalse(q.degenerate_subspace_comparison(a, b, da, db[:, ::-1], self.masses)["qualified"])

    def test_wrong_matched_subspace_is_detected(self):
        other = np.eye(6)[3:5] / 2
        da = self.modes @ self.derivative
        result = q.degenerate_subspace_comparison(self.modes, other, da, da, self.masses)
        self.assertFalse(result["qualified"])
        self.assertIn("mode_subspaces_do_not_match", result["failures"])

    def test_total_dipole_point_charges_obey_charged_rigid_identities(self):
        coords = np.array([[1., 2., 3.], [-0.3, 0.5, 0.1]])
        charges = np.array([1.3, -0.3])
        d = charges[:, None, None] * np.eye(3)[None]
        mu = charges @ coords
        for origin in [np.zeros(3), np.array([0.2, 0.3, -0.4])]:
            result = q.rigid_body_residuals(d, coords, mu, charges.sum(), origin_bohr=origin)
            self.assertTrue(result["qualified"], result)
        wrong = q.rigid_body_residuals(d, coords, mu, 0)
        self.assertFalse(wrong["qualified"])
        # Nuclear/electronic pieces separately have charges +8 and -7;
        # requiring neutral electronic-only derivatives would be erroneous.
        nuclear = np.array([5., 3.])[:, None, None] * np.eye(3)[None]
        electronic = d - nuclear
        self.assertFalse(q.rigid_body_residuals(electronic, coords, mu, 1)["qualified"])

    def test_rigid_identities_are_covariant_under_whole_rotation(self):
        coords = np.array([[1., 0., 0.], [0., 1., 0.]])
        charges = np.array([0.2, -0.2])
        d = charges[:, None, None] * np.eye(3)[None]
        mu = charges @ coords
        rot = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        dr = np.einsum("ij,ajk,lk->ail", rot, d, rot)
        result = q.rigid_body_residuals(dr, coords @ rot.T, rot @ mu, 0)
        self.assertTrue(result["qualified"], result)

    def test_nan_missing_or_incomplete_evidence_cannot_pass(self):
        good = np.array([self.derivative] * 2)
        bad = good.copy(); bad[0, 0, 0] = np.nan
        for values in [None, [], bad, good[:1]]:
            self.assertFalse(q.common_frame_derivative_comparison(values, [0.002, 0.001], self.modes, self.masses)["qualified"])
        self.assertFalse(q.directional_comparison([[np.nan, 0, 0]], [[[0, 0, 0]]]*2, [0.002, 0.001])["qualified"])
        self.assertFalse(q.rigid_body_residuals(None, [[0, 0, 0]], [0, 0, 0], 0)["qualified"])


if __name__ == "__main__":
    unittest.main()
