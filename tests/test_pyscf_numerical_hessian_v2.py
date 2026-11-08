"""Scientific checks use exact synthetic gradients; no large quantum work."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import unittest
import tempfile

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "scripts/pyscf_full_numerical_hessian_20261108_v2.py"
spec = importlib.util.spec_from_file_location("numeric_worker", SOURCE)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class HessianTests(unittest.TestCase):
    def test_central_gradients_preserve_order_and_signed_curvature(self):
        rng = np.random.default_rng(27)
        raw = rng.normal(size=(9, 9))
        expected = (raw + raw.T) / 2
        zero = rng.normal(size=9)
        step = .001
        plus = np.array([(zero + step * expected[:, i]).reshape(3, 3) for i in range(9)])
        minus = np.array([(zero - step * expected[:, i]).reshape(3, 3) for i in range(9)])
        actual = worker.assemble_hessian(plus, minus, step)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_mass_projection_has_exact_dimension_and_signed_modes(self):
        # A 3-atom nonlinear model has 3 vibrations after projecting all 6
        # actual mass-weighted translations and rotations.
        coords = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        masses = np.array([16., 1., 1.])
        _, _, complement, rank = worker.vibrational_modes(np.eye(9), coords, masses)
        self.assertEqual(rank, 6)
        self.assertEqual(complement.shape, (9, 3))
        rootmass = np.repeat(np.sqrt(masses), 3)
        hessian = (complement @ np.diag([-2., 3., 4.]) @ complement.T) * rootmass[:, None] * rootmass[None, :]
        freq, eigenvalues, vectors, _ = worker.vibrational_modes(hessian, coords, masses)
        np.testing.assert_allclose(eigenvalues, [-2., 3., 4.], atol=1e-12)
        self.assertEqual(np.sum(freq < 0), 1)
        np.testing.assert_allclose(vectors.T @ vectors, np.eye(3), atol=1e-12)
        # Translation is excluded using sqrt(m), not unweighted vectors.
        translation = np.zeros((3, 3)); translation[:, 0] = np.sqrt(masses)
        np.testing.assert_allclose(translation.ravel() @ vectors, 0., atol=1e-12)

    def test_formula_and_incomplete_hessian_rejected(self):
        worker.require_formula(["C"] * 7 + ["H"] * 6 + ["O"] * 4)
        with self.assertRaises(ValueError):
            worker.require_formula(["C"] * 7 + ["H"] * 6 + ["O"] * 3)
        with self.assertRaises(ValueError):
            worker.assemble_hessian(np.zeros((8, 3, 3)), np.zeros((8, 3, 3)), .001)

    def test_checkpoint_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.npy"
            ref = worker.save_array(path, np.ones((3, 3)))
            np.save(path, np.zeros((3, 3)))
            with self.assertRaises(ValueError):
                worker.read_ref(ref)

    def test_all_coordinates_attempted_after_one_failure_no_full_claim(self):
        optimum = {"symbols": ["C"] * 7 + ["H"] * 6 + ["O"] * 4}
        calls = []
        def evaluate(mol, path, density):
            calls.append(path.name)
            if path.name == "c000-plus":
                raise ValueError("real scientific failure")
            return {}, None, np.zeros((17, 3))
        with tempfile.TemporaryDirectory() as tmp:
            optpath = Path(tmp) / "opt.json"; optpath.write_text("{}")
            args = SimpleNamespace(optimization=optpath, output=Path(tmp) / "out", step=.001)
            with mock.patch.object(worker, "load_optimization", return_value=(optimum, np.zeros((17, 3)), np.eye(2))), \
                 mock.patch.object(worker, "new_molecule", return_value=None), \
                 mock.patch.object(worker, "evaluate_point", side_effect=evaluate):
                result = worker.stage_hessian(args)
            self.assertEqual(len(calls), 102)
            self.assertEqual(result["actual_qualified_points"], 101)
            self.assertEqual(result["status"], "NUMERICAL_HESSIAN_PARTIAL")
            self.assertNotIn("raw_hessian", result)

    def test_positive_modes_below_step_uncertainty_do_not_certify_minimum(self):
        from pyscf.data import elements
        rng = np.random.default_rng(123)
        symbols = ["C"] * 7 + ["H"] * 6 + ["O"] * 4
        coords = rng.normal(size=(17, 3))
        masses = np.array([elements.MASSES[elements.charge(x)] for x in symbols])
        _, _, q, _ = worker.vibrational_modes(np.eye(51), coords, masses)
        roots = np.repeat(np.sqrt(masses), 3)
        coarse_values = np.ones(45); coarse_values[0] = 1e-8
        fine_values = coarse_values.copy(); fine_values[1] += 2e-7
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            coordref = worker.save_array(out / "coords.npy", coords)
            paths = []
            for index, values in enumerate((coarse_values, fine_values)):
                matrix = (q @ np.diag(values) @ q.T) * roots[:, None] * roots[None, :]
                record = worker.base_result("NUMERICAL_HESSIAN_COMPLETE")
                record.update(step_bohr=(.001, .0005)[index], optimization_sha256="same",
                              coordinates_bohr=coordref, symbols=symbols,
                              all_points_same_zero_density_sha256="same",
                              raw_hessian=worker.save_array(out / ("h%d.npy" % index), matrix))
                path = out / ("h%d.json" % index); worker.save_json(path, record); paths.append(str(path))
            result = worker.stage_analyze(SimpleNamespace(hessians=paths, output=out / "analysis"))
            self.assertEqual(result["imaginary_counts"], [0, 0])
            self.assertTrue(result["numerical_step_consistency_passed"])
            self.assertFalse(result["minimum_eigenvalue_exceeds_step_uncertainty_margin"])
            self.assertFalse(result["minimum_verified"])
            self.assertEqual(result["vibrational_modes"], 45)

    def test_newton_preconditioner_disables_ordinary_callback_and_preserves_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = SimpleNamespace(check_convergence=lambda env: True)
            seed = SimpleNamespace(check_convergence=lambda env: True, _scf=base,
                                   ah_max_cycle=40, converged=False,
                                   mo_coeff=np.eye(2), mo_occ=np.array([2., 0.]), mo_energy=np.ones(2),
                                   kernel=lambda **kw: -1., make_rdm1=lambda: np.eye(2))
            base.newton = lambda: seed
            with mock.patch.object(worker, "make_scf", return_value=base):
                density, record = worker.run_newton_seed(None, Path(tmp), np.eye(2))
            self.assertIsNone(base.check_convergence)
            self.assertIsNone(seed.check_convergence)
            self.assertEqual(seed.max_cycle, 30)
            self.assertEqual(seed.max_cycle_inner, 12)
            self.assertFalse(record["native_seed_converged"])
            self.assertEqual(record["status"], "UNCERTIFIED_INITIAL_GUESS")

    def test_only_true_previous_endpoint_seeds_second_certification(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            mol = SimpleNamespace(copy=lambda: SimpleNamespace(atom_coords=lambda: np.zeros((1, 3)), atom_symbol=lambda i: "He", natm=1))
            first_density = np.eye(2)
            actual_endpoint = np.eye(2) * 2
            seen = []
            def certify(actual_mol, path, density, index):
                seen.append(np.array(density, copy=True))
                record = {"native_pyscf_converged": index == 2, "energy_hartree": -1.,
                          "strict_iteration_checks": [{"passed": index == 2}],
                          "fresh_full_fock_orbital_gradient_norm": 0., "fresh_energy_hartree": -1.,
                          "final_energy_recheck_difference_hartree": 0.,
                          "electronic_certification_passed": index == 2}
                return SimpleNamespace(chkfile=str(out / "absent.chk")), actual_endpoint, record
            with mock.patch.object(worker, "point_fingerprint", return_value=("fixed", {})), \
                 mock.patch.object(worker, "run_newton_seed", return_value=(first_density, {"native_seed_converged": False})), \
                 mock.patch.object(worker, "strict_certification_call", side_effect=certify):
                result, _, _ = worker.evaluate_point(mol, out, nuclear_gradient=False)
            self.assertEqual(result["actual_SCF_calls_started"], 3)
            self.assertEqual(len(seen), 2)
            np.testing.assert_array_equal(seen[0], first_density)
            np.testing.assert_array_equal(seen[1], actual_endpoint)
            self.assertFalse(result["certification_calls"][0]["native_pyscf_converged"])
            self.assertTrue(result["electronic_certification_passed"])

    def test_failed_saved_point_never_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            worker.save_json(out / "result.json", {"input_fingerprint": "fixed", "status": "POINT_FAILED"})
            with mock.patch.object(worker, "point_fingerprint", return_value=("fixed", {})), \
                 mock.patch.object(worker, "make_scf") as calculate:
                with self.assertRaisesRegex(ValueError, "cannot automatically retry"):
                    worker.evaluate_point(None, out)
                calculate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
