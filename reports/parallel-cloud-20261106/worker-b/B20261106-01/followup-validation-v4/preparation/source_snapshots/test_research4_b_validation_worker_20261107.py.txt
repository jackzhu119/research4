"""Validation-worker algebra/provenance tests; no quantum kernels execute."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
SOURCE = SCRIPTS / "research4_b_validation_worker_20261107.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load(SOURCE, "validation_worker_test_subject")
prior = load(SCRIPTS / "test_research4_b_rootcause_worker_20261107.py", "validation_prior_test_cases")
prior.worker = worker
prior.original_tests.worker = worker
prior.recovery_tests.worker = worker
prior.recovery_tests.RECOVERY = SOURCE


class ValidationTests(unittest.TestCase):
    def test_grid5_and_explicit_dual_controls_are_validated(self):
        request = {"schema": worker.REQUEST_SCHEMA, "operation": "point", "case": "water",
                   "direction": {"type": "mode", "index": 1}, "step_bohr": .0005, "sign": 1,
                   "grid_level": 5, "scf_conv_tol": 1e-12, "scf_conv_tol_grad": 1e-9,
                   "small_rho_cutoff": 0, "gradient_response_controls": [False, True]}
        worker.validate_request(request)
        for invalid in ([True], [False, False], [False, 1]):
            with self.assertRaisesRegex(ValueError, "gradient controls"):
                worker.validate_request({**request, "gradient_response_controls": invalid})
        with self.assertRaisesRegex(ValueError, "moving-grid"):
            worker.validate_request({**request, "operation": "fixed_grid_point"})
        with self.assertRaisesRegex(ValueError, "zero small-rho"):
            worker.validate_request({**request, "small_rho_cutoff": 1e-7})

    def test_scf_qualification_uses_an_independent_unmixed_fock(self):
        coords, weights = np.array([[0., 0., 0.]]), np.array([1.])
        density, hcore, veff = np.diag([2., 0.]), np.eye(2), np.full((2, 2), .1)
        class MF:
            grids = SimpleNamespace(coords=coords, weights=weights)
            converged = True
            conv_tol_grad = 1e-9
            mo_coeff, mo_occ = np.eye(2), np.array([2., 0.])
            mol = object()
            make_rdm1 = lambda self: density
            get_hcore = lambda self: hcore
            def get_veff(self, molecule, d):
                self.assert_density = d.copy()
                return veff
            def get_grad(self, coeff, occupation, fock):
                self.assert_fock = fock.copy()
                return np.array([self.final_norm])
        mf = MF(); mf.final_norm = 2e-9
        record = worker.independent_scf_qualification(mf, {}, np, lambda *a, **k: None)
        self.assertTrue(record["qualified"])
        np.testing.assert_array_equal(mf.assert_density, density)
        np.testing.assert_array_equal(mf.assert_fock, hcore + veff)
        mf.final_norm = 2e-8
        record = worker.independent_scf_qualification(mf, {}, np, lambda *a, **k: None)
        self.assertFalse(record["qualified"])
        self.assertEqual(record["status"], "SCF_NOT_QUALIFIED")

    def test_explicit_krylov_cutoff_reaches_bound_function_and_is_restored(self):
        seen = []
        def krylov(aop, b, lindep=1e-13, callback=None):
            seen.append(lindep)
            callback(3, [b], [aop(b)])
            return b
        lib = SimpleNamespace(krylov=krylov)
        def solve(fx, energies, occupation, h1, s1, **kwargs):
            return lib.krylov(fx, h1), np.eye(1)
        cphf = SimpleNamespace(lib=lib, solve=solve)
        request = {"cphf_tol": 1e-13, "cphf_max_cycle": 200, "cphf_krylov_lindep": 1e-24}
        _, _, record = worker.controlled_cphf_solve(cphf, lambda x: x, np.ones(2), np.array([2., 0.]),
                            np.ones((2, 1)), np.ones((2, 1)), request, lambda *a, **k: None)
        self.assertEqual(seen, [1e-24])
        self.assertIs(lib.krylov, krylov)
        self.assertEqual(record["stock_krylov_lindep"], 1e-13)
        self.assertEqual(record["krylov_iteration_callbacks"], 1)
        def failed(*args, **kwargs):
            raise RuntimeError("manufactured solve failure")
        cphf.solve = failed
        with self.assertRaisesRegex(RuntimeError, "manufactured"):
            worker.controlled_cphf_solve(cphf, None, None, None, None, None, request, lambda *a, **k: None)
        self.assertIs(lib.krylov, krylov)

    def test_true_quadrature_observation_handles_sorting_and_zero_padding(self):
        from pyscf.grad import rks as rks_gradient
        coords = np.array([[3., 1., 0.], [1., 0., 0.], [2., 0., 0.]])
        weights = np.array([.1, .2, .3])
        def generated(grid):
            yield coords[:1], weights[:1], None
            yield coords[1:], weights[1:], None
        mf = SimpleNamespace(grids=SimpleNamespace(coords=np.vstack([coords[::-1], [1e4] * 3]),
                                                  weights=np.r_[weights[::-1], 0.]))
        class Gradient:
            def kernel(self):
                for _ in rks_gradient.grids_response_cc(mf.grids):
                    pass
                return np.zeros((1, 3))
        with patch.object(rks_gradient, "grids_response_cc", generated):
            _, record = worker.true_gradient_quadrature(Gradient(), mf, np)
            self.assertTrue(record["nonzero_quadrature_sets_numerically_match"])
            self.assertEqual(record["generated_grid_points"], 3)
            self.assertEqual(record["SCF_grid_points_including_padding"], 4)
            self.assertIs(rks_gradient.grids_response_cc, generated)
            mf.grids.weights[0] += .01
            _, record = worker.true_gradient_quadrature(Gradient(), mf, np)
            self.assertFalse(record["nonzero_quadrature_sets_numerically_match"])
            class Failed:
                def kernel(self):
                    raise RuntimeError("manufactured gradient failure")
            with self.assertRaisesRegex(RuntimeError, "manufactured"):
                worker.true_gradient_quadrature(Failed(), mf, np)
            self.assertIs(rks_gradient.grids_response_cc, generated)

    def test_previous_direction_and_response_algebra_is_unchanged(self):
        before = ast.parse((SCRIPTS / "research4_b_rootcause_worker_20261107.py").read_text())
        after = ast.parse(SOURCE.read_text())
        functions = lambda tree: {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        old, new = functions(before), functions(after)
        for name in ("unit_direction", "central_difference", "load_case", "original_worker", "case_postprocess",
                     "overlap_directional_derivative", "response_equation_metrics", "density_directional_derivative",
                     "initialize_platform_processor_without_fork", "supervise"):
            self.assertEqual(ast.dump(old[name], include_attributes=False), ast.dump(new[name], include_attributes=False), name)

    def test_point_has_one_scf_call_and_response_has_no_scf_or_full_hessian_call(self):
        functions = {node.name: node for node in ast.parse(SOURCE.read_text()).body if isinstance(node, ast.FunctionDef)}
        kernels = [node for node in ast.walk(functions["point"]) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == "kernel"
                   and isinstance(node.func.value, ast.Name) and node.func.value.id == "mf"]
        self.assertEqual(len(kernels), 1)
        for name in ("response_audit", "controlled_cphf_solve"):
            calls = [node.func.attr for node in ast.walk(functions[name])
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
            self.assertNotIn("kernel", calls)
            self.assertNotIn("gen_hop", calls)
            self.assertNotIn("partial_hess_elec", calls)


class ValidationAggregateTests(prior.RootcauseProvenanceTests):
    def dual(self, binding, sign, h):
        path = worker.checked(binding); value = worker.read(path)
        # The primary gradient has cG=2, the full-response gradient and common
        # energy have cG=cE=4. This catches accidental reuse of gradient.npy.
        gradient = np.array([[4 * sign * h, 0., 0.]])
        worker.save_array(path.parent / "gradient_grid_response_true.npy", gradient, np)
        value["arrays"]["gradient_grid_response_true.npy"] = worker.ref(path.parent / "gradient_grid_response_true.npy")
        value["energy_hartree"] = -75. + 2 * (sign * h) ** 2
        value["gradient_observations"] = {
            "false": {"gradient_grid_response": False, "array": value["arrays"]["gradient.npy"]},
            "true": {"gradient_grid_response": True, "array": value["arrays"]["gradient_grid_response_true.npy"]}}
        value["gradient_controls_share_one_SCF_energy_and_density"] = True
        worker.save(path, value)
        return worker.ref(path)

    def test_both_gradient_curvatures_use_one_new_matching_energy_center(self):
        points = [self.synthetic_point("center", 0, 0, -75., [0., 0., 0.])]
        for label, h in (("small", .001), ("large", .005)):
            points.extend([self.dual(self.synthetic_point(label + "minus", -1, h, 0., [-2 * h, 0., 0.]), -1, h),
                           self.dual(self.synthetic_point(label + "plus", 1, h, 0., [2 * h, 0., 0.]), 1, h)])
        output = self.root / "aggregate"; output.mkdir(); state = {}
        worker.aggregate({"points": points}, state, output, lambda *a, **k: None, [])
        self.assertEqual(len(state["paired_comparisons"]), 4)
        for row in state["paired_comparisons"]:
            self.assertEqual(row["center_point_result"]["sha256"], points[0]["sha256"])
            self.assertAlmostEqual(row["gradient_FD_curvature_Eh_per_Bohr2"], 4. if row["gradient_grid_response"] else 2.)
            self.assertAlmostEqual(row["energy_FD_curvature_Eh_per_Bohr2"], 4., places=7)
        self.assertEqual(len(state["double_step_comparisons"]), 2)
        self.assertEqual({row["gradient_grid_response"] for row in state["double_step_comparisons"]}, {False, True})

    def test_unqualified_scf_is_excluded_even_if_legacy_status_claims_computed(self):
        binding = self.synthetic_point("unqualified", -1, .001, -75., [0., 0., 0.])
        path = worker.checked(binding); value = worker.read(path); value["scf_qualified"] = False
        worker.save(path, value)
        output = self.root / "aggregate"; output.mkdir(); state = {}
        worker.aggregate({"points": [worker.ref(path)]}, state, output, lambda *a, **k: None, [])
        self.assertEqual(len(state["excluded_points"]), 1)
        self.assertEqual(state["paired_comparisons"], [])

    def test_rebuilt_energy_never_falls_back_to_original_loose_scf_energy(self):
        points = [self.synthetic_point("minus", -1, .001, -75., [-.002, 0., 0.], tightened=False),
                  self.synthetic_point("plus", 1, .001, -75., [.002, 0., 0.], tightened=False)]
        output = self.root / "aggregate"; output.mkdir(); state = {}
        worker.aggregate({"points": points}, state, output, lambda *a, **k: None, [])
        self.assertIsNone(state["paired_comparisons"][0]["center_energy_source"])
        self.assertNotIn("energy_FD_curvature_Eh_per_Bohr2", state["paired_comparisons"][0])


def suite():
    result = unittest.TestSuite(); loader = unittest.defaultTestLoader
    for cls in (prior.original_tests.NumericalTests, prior.original_tests.ProvenanceTests, prior.NoForkMetadataTests):
        result.addTests(loader.loadTestsFromTestCase(cls))
    for name in prior.RootcauseAlgebraTests.__dict__:
        if name.startswith("test_") and name not in ("test_source_scientific_direction_and_difference_functions_remain_identical",
                "test_response_audit_contains_no_scf_or_full_hessian_execution"):
            result.addTest(prior.RootcauseAlgebraTests(name))
    result.addTests(prior.RootcauseProvenanceTests(name) for name in prior.RootcauseProvenanceTests.__dict__ if name.startswith("test_"))
    result.addTests(loader.loadTestsFromTestCase(ValidationTests))
    result.addTests(ValidationAggregateTests(name) for name in ValidationAggregateTests.__dict__ if name.startswith("test_"))
    return result


if __name__ == "__main__":
    raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(suite()).wasSuccessful() else 1)
