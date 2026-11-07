"""v6 recovery tests, including an installed CIAH matrix-only regression.

The CIAH test has a manufactured 16-dimensional SPD matrix. It creates no
molecule, SCF, Fock matrix, quadrature grid or quantum calculation.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
SOURCE = SCRIPTS / "research4_b_validation_recovery_worker_v6_20261107.py"
V5_SOURCE = SCRIPTS / "research4_b_validation_recovery_worker_20261107.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load(SOURCE, "validation_recovery_v6_worker_test_subject")
prior = load(SCRIPTS / "test_research4_b_validation_recovery_worker_20261107.py", "recovery_v6_prior_v5_tests")
prior.worker = worker
prior.SOURCE = SOURCE
prior.prior.worker = worker
prior.prior.SOURCE = SOURCE
prior.prior.prior.worker = worker
prior.prior.prior.original_tests.worker = worker
prior.prior.prior.recovery_tests.worker = worker
prior.prior.prior.recovery_tests.RECOVERY = SOURCE


class AHControlTests(unittest.TestCase):
    setUp = prior.SolverTests.setUp
    tearDown = prior.SolverTests.tearDown
    fake_mf = prior.SolverTests.fake_mf
    run_solver = prior.SolverTests.run_solver

    def test_newton_receives_strict_inner_ah_tolerance_and_records_it_separately(self):
        mf = self.fake_mf()
        original_newton = mf.newton
        observed_tolerances = []
        def observed_newton():
            solver = original_newton()
            solver.ah_conv_tol = 1e-12
            original_kernel = solver.kernel
            def observed_kernel(**kwargs):
                observed_tolerances.append(solver.ah_conv_tol)
                return original_kernel(**kwargs)
            solver.kernel = observed_kernel
            return solver
        mf.newton = observed_newton
        _, state, events = self.run_solver(mf)
        self.assertEqual(observed_tolerances, [1e-22])
        controls = state["SCF_solver_controls"]
        self.assertTrue(controls["Newton_AH_used"])
        self.assertEqual(controls["Newton_AH_conv_tol"], 1e-22)
        self.assertEqual(controls["previous_v5_Newton_AH_conv_tol"], 1e-12)
        self.assertEqual(controls["stock_Newton_AH_conv_tol_before_override"], 1e-12)
        self.assertEqual(controls["Newton_AH_lindep"], 1e-24)
        transition = next(value for value in state["SCF_solver_transitions"] if value["solver"] == "stock_Newton_AH")
        self.assertEqual(transition["ah_conv_tol"], 1e-22)
        start_event = next(value for event, value in events if event == "checkpoint_Newton_AH_started")
        self.assertEqual(start_event["ah_conv_tol"], 1e-22)
        self.assertEqual(state["SCF_final_state"]["internal_orbital_gradient_tolerance"], 1e-9)
        self.assertEqual(state["SCF_final_state"]["final_fresh_Fock_qualification_threshold"], 1e-8)

    def test_method_settings_and_qualification_are_identical_to_v5(self):
        functions = lambda path: {node.name: node for node in ast.parse(path.read_text()).body
                                  if isinstance(node, ast.FunctionDef)}
        v5, v6 = functions(V5_SOURCE), functions(SOURCE)
        for name in ("settings_for", "configure_scf_tolerance", "independent_scf_qualification", "point_group_key",
                     "load_warmstart", "checkpoint_probe"):
            self.assertEqual(ast.dump(v5[name], include_attributes=False),
                             ast.dump(v6[name], include_attributes=False), name)
        source = {"source": {"settings": {"grid_level": 3, "scf_conv_tol": 1e-12}}}
        settings = worker.settings_for({"operation": "point"}, source)
        self.assertNotIn("ah_conv_tol", settings)
        self.assertNotIn("Newton_AH_conv_tol", settings)

    def test_successful_cdiis_honestly_reports_that_inner_ah_was_not_used(self):
        mf = self.fake_mf(cdiis_converged=True)
        _, state, events = self.run_solver(mf)
        self.assertFalse(state["SCF_solver_controls"]["Newton_AH_used"])
        self.assertEqual(mf.newton_calls, [])
        self.assertEqual([value["solver"] for value in state["SCF_solver_transitions"]], ["stock_CDIIS"])
        self.assertFalse(any(event == "checkpoint_Newton_AH_started" for event, _ in events))


class InstalledCIAHRegressionTests(unittest.TestCase):
    @staticmethod
    def problem():
        rng = np.random.default_rng(7)
        q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
        matrix = (q * np.linspace(.2, 3., 16)) @ q.T
        gradient = rng.normal(size=16)
        # Fix the small-gradient scale explicitly, independently of RNG norm.
        gradient *= 2.64664e-8 / np.linalg.norm(gradient)
        return matrix, gradient

    @staticmethod
    def first_terminated_step(matrix, gradient, *, tol, lindep=None):
        from pyscf.soscf import ciah
        kwargs = {"tol": tol, "max_cycle": 20}
        if lindep is not None:
            kwargs["lindep"] = lindep
        precondition = lambda x, eigenvalue: x / (np.diag(matrix) - eigenvalue)
        for ended, iteration, eigenvalue, step, h_step, reported_residual, metric_minimum in ciah.davidson_cc(
                matrix.__matmul__, lambda: gradient, precondition, gradient, **kwargs):
            if ended:
                # This is the independently evaluated linear response
                # residual, rather than CIAH's augmented-eigenvector residual.
                physical_residual = float(np.linalg.norm(matrix @ step + gradient))
                return step, physical_residual, iteration, metric_minimum
        raise AssertionError("Manufactured CIAH problem never returned a terminated step")

    def test_strict_inner_ah_tolerance_reduces_actual_response_residual(self):
        matrix, gradient = self.problem()
        self.assertAlmostEqual(float(np.linalg.norm(gradient)), 2.64664e-8, places=20)
        self.assertGreater(float(np.linalg.eigvalsh(matrix).min()), 0.)
        stock_step, stock_residual, stock_iterations, _ = self.first_terminated_step(
            matrix, gradient, tol=float(gradient @ gradient), lindep=1e-24)
        strict_step, strict_residual, strict_iterations, _ = self.first_terminated_step(
            matrix, gradient, tol=1e-22, lindep=1e-24)
        self.assertEqual(stock_iterations, 1)
        self.assertGreater(stock_residual, 1e-8)
        self.assertLess(stock_residual, 2e-8)
        self.assertGreater(strict_iterations, stock_iterations)
        self.assertLess(strict_residual, 5e-12)
        self.assertLess(strict_residual / stock_residual, 1e-3)
        exact_step = np.linalg.solve(matrix, -gradient)
        self.assertLess(np.linalg.norm(strict_step - exact_step), np.linalg.norm(stock_step - exact_step))

    def test_default_linear_dependence_cutoff_can_return_zero_step_for_a_small_seed(self):
        matrix, gradient = self.problem()
        gradient *= 1e-8
        step, residual, iterations, metric_minimum = self.first_terminated_step(
            matrix, gradient, tol=float(gradient @ gradient))
        self.assertEqual(iterations, 1)
        np.testing.assert_array_equal(step, np.zeros(16))
        self.assertAlmostEqual(residual / np.linalg.norm(gradient), 1., places=13)
        self.assertLess(metric_minimum, 1e-14)


def suite():
    result = unittest.TestSuite()
    result.addTests(prior.suite())
    result.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(AHControlTests))
    result.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(InstalledCIAHRegressionTests))
    return result


if __name__ == "__main__":
    raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(suite()).wasSuccessful() else 1)
