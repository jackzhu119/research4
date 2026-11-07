"""Fixed-grid and response algebra/provenance tests; no quantum kernels run."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
SOURCE = SCRIPTS / "research4_b_rootcause_worker_20261107.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load(SOURCE, "rootcause_worker_tests")
original_tests = load(SCRIPTS / "test_research4_b_diagnostic_worker_20261106.py", "rootcause_original_tests")
original_tests.worker = worker
recovery_tests = load(SCRIPTS / "test_research4_b_diagnostic_worker_recovery_20261107.py", "rootcause_metadata_tests")
recovery_tests.worker = worker
recovery_tests.RECOVERY = SOURCE


class NoForkMetadataTests(unittest.TestCase):
    setUp = recovery_tests.RecoveryMetadataTests.setUp
    tearDown = recovery_tests.RecoveryMetadataTests.tearDown
    test_processor_resolution_survives_cache_invalidation_without_subprocess = recovery_tests.RecoveryMetadataTests.test_processor_resolution_survives_cache_invalidation_without_subprocess
    test_reported_processor_value_is_an_observed_local_value = recovery_tests.RecoveryMetadataTests.test_reported_processor_value_is_an_observed_local_value
    test_real_cold_numerical_imports_cannot_start_a_subprocess = recovery_tests.RecoveryMetadataTests.test_real_cold_numerical_imports_cannot_start_a_subprocess
    test_failed_proof_never_initializes_platform_metadata = recovery_tests.RecoveryMetadataTests.test_failed_proof_never_initializes_platform_metadata


class RootcauseAlgebraTests(unittest.TestCase):
    def test_grid_content_hash_is_order_sensitive_and_dtype_canonical(self):
        coords = np.arange(12, dtype=float).reshape(4, 3)
        weights = np.arange(4, dtype=float) + .1
        digest = worker.grid_digest(coords, weights, np)
        self.assertEqual(digest, worker.grid_digest(coords.astype(np.float32), weights, np))
        self.assertNotEqual(digest, worker.grid_digest(coords[::-1], weights[::-1], np))
        with self.assertRaisesRegex(ValueError, "grid arrays"):
            worker.grid_digest(coords, np.ones(3), np)

    def test_fixed_grid_rebuilds_molecule_specific_mask_without_changing_coordinates(self):
        class Grid:
            def make_mask(self, molecule, coordinates):
                self.mask_molecule = molecule
                self.mask_coordinates = coordinates.copy()
                return np.full((2, 3), molecule.mask_value, dtype=np.uint8)
        mf = SimpleNamespace(grids=Grid(), small_rho_cutoff=1e-7)
        molecule = SimpleNamespace(mask_value=17)
        coords, weights = np.arange(12.).reshape(4, 3), np.arange(4.)
        expected = worker.grid_digest(coords, weights, np)
        self.assertEqual(worker.configure_fixed_grid(mf, molecule, coords, weights, np), expected)
        self.assertIs(mf.grids.mol, molecule)
        self.assertIs(mf.grids.mask_molecule, molecule)
        self.assertIs(mf.grids.screen_index, mf.grids.non0tab)
        np.testing.assert_array_equal(mf.grids.non0tab, np.full((2, 3), 17))
        np.testing.assert_array_equal(mf.grids.coords, coords)
        self.assertFalse(np.shares_memory(coords, mf.grids.coords))
        self.assertFalse(np.shares_memory(weights, mf.grids.weights))
        self.assertEqual(mf.small_rho_cutoff, 1e-7)

    def test_cphf_residual_detects_an_independently_injected_defect(self):
        rng = np.random.default_rng(2718)
        occupation = np.array([2, 2, 0, 0])
        energies = np.array([-1.1, -.6, .2, .8])
        s1 = rng.normal(size=(4, 2)) * .01
        u = rng.normal(size=(4, 2)) * .1
        u[:2] = -.5 * s1[:2]
        induced = rng.normal(size=(4, 2)) * .01
        gap = energies[:, None] - energies[:2]
        h1 = s1 * energies[:2] - gap * u - induced
        exact = worker.response_equation_metrics(u, induced, h1, s1, energies, occupation, np)
        self.assertLess(exact["virtual_equation_max_abs_Eh"], 1e-14)
        self.assertLess(exact["occupied_normalization_max_abs"], 1e-14)
        wrong = h1.copy(); wrong[3, 1] += .125
        defect = worker.response_equation_metrics(u, induced, wrong, s1, energies, occupation, np)
        self.assertAlmostEqual(defect["virtual_equation_max_abs_Eh"], .125)

    def test_ao_density_derivative_matches_orbital_finite_difference(self):
        rng = np.random.default_rng(141)
        coeff, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        occupied = coeff[:, :2]
        u = rng.normal(size=(5, 2)) * .1
        u[:2] = np.array([[0, .1], [-.1, 0]])
        d1 = worker.density_directional_derivative(coeff, occupied, u, np)
        h = 1e-5
        plus, minus = occupied + h * coeff @ u, occupied - h * coeff @ u
        fd = (2 * plus @ plus.T - 2 * minus @ minus.T) / (2 * h)
        np.testing.assert_allclose(d1, fd, atol=2e-11, rtol=1e-9)
        self.assertLess(float(np.max(np.abs(d1 - d1.T))), 1e-14)
        self.assertLess(abs(float(np.trace(d1))), 1e-14)

    def test_uniform_translation_has_zero_overlap_derivative(self):
        rng = np.random.default_rng(0)
        derivative = rng.normal(size=(3, 4, 4))
        derivative -= derivative.transpose(0, 2, 1)
        class Molecule:
            nao_nr = lambda self: 4
            intor = lambda self, *a, **k: derivative
            aoslice_by_atom = lambda self: [(0, 1, 0, 2), (1, 2, 2, 4)]
        q = np.zeros((2, 3)); q[:, 0] = 1 / np.sqrt(2)
        np.testing.assert_allclose(worker.overlap_directional_derivative(Molecule(), q, np), np.zeros((4, 4)), atol=1e-14)

    def test_source_scientific_direction_and_difference_functions_remain_identical(self):
        before = ast.parse((SCRIPTS / "research4_b_diagnostic_worker_recovery_20261107.py").read_text())
        after = ast.parse(SOURCE.read_text())
        functions = lambda tree: {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        old, new = functions(before), functions(after)
        for name in ("unit_direction", "central_difference", "load_case", "original_worker", "point", "case_postprocess",
                     "initialize_platform_processor_without_fork", "supervise"):
            self.assertEqual(ast.dump(old[name], include_attributes=False), ast.dump(new[name], include_attributes=False), name)

    def test_response_audit_contains_no_scf_or_full_hessian_execution(self):
        tree = ast.parse(SOURCE.read_text())
        response = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "response_audit")
        calls = [node.func.attr for node in ast.walk(response)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertNotIn("kernel", calls)
        self.assertNotIn("gen_hop", calls)
        self.assertNotIn("partial_hess_elec", calls)
        self.assertIn("make_h1", calls)
        self.assertIn("solve", calls)

    def test_fixed_center_cannot_be_displaced_or_fixed_point_undisplaced(self):
        base = {"schema": worker.REQUEST_SCHEMA, "case": "water", "scf_conv_tol": 1e-12}
        worker.validate_request({**base, "operation": "fixed_grid_center"})
        worker.validate_request({**base, "operation": "fixed_grid_point", "direction": {"type": "mode", "index": 1},
                                 "step_bohr": .0005, "sign": -1})
        with self.assertRaisesRegex(ValueError, "center must"):
            worker.validate_request({**base, "operation": "fixed_grid_center", "step_bohr": .001, "sign": 1})
        with self.assertRaisesRegex(ValueError, "nonzero"):
            worker.validate_request({**base, "operation": "fixed_grid_point", "direction": {"type": "mode", "index": 1},
                                     "step_bohr": 0, "sign": 0})


class RootcauseProvenanceTests(original_tests.ProvenanceTests):
    # Inherited original tests are loaded separately. This class only adds
    # tests requiring their synthetic, source-bound finite-difference fixtures.
    def fixed(self, binding, center=None, *, grid="g" * 64):
        path = worker.checked(binding)
        record = worker.read(path)
        is_center = record["sign"] == 0
        record.update(operation="fixed_grid_center" if is_center else "fixed_grid_point",
                      status="FIXED_GRID_CENTER_COMPUTED" if is_center else "POINT_COMPUTED",
                      grid_semantics="FIXED_CENTER_CARTESIAN_QUADRATURE", fixed_grid_content_sha256=grid,
                      grid_consistency_limitation="Fixed absolute quadrature test")
        if center is not None:
            record["center_result"] = center
        worker.save(path, record)
        return worker.ref(path)

    def test_fixed_points_use_their_actual_center_and_expose_csv_semantics(self):
        center = self.fixed(self.synthetic_point("fixed_center", 0, 0, -75., [0., 0., 0.]))
        minus = self.fixed(self.synthetic_point("fixed_minus", -1, .01, -74.9999, [-.02, 0., 0.]), center)
        plus = self.fixed(self.synthetic_point("fixed_plus", 1, .01, -74.9999, [.02, 0., 0.]), center)
        # A rebuilt-grid center at the same recorded numerical settings has a
        # deliberately incorrect energy and must never enter the fixed group.
        rebuilt = self.synthetic_point("rebuilt_center", 0, 0, -99., [0., 0., 0.])
        output = self.root / "aggregate"; output.mkdir(); state = {}
        worker.aggregate({"points": [minus, plus, center, rebuilt]}, state, output, lambda *a, **k: None, [])
        row = state["paired_comparisons"][0]
        self.assertEqual(row["center_point_result"]["sha256"], center["sha256"])
        self.assertAlmostEqual(row["energy_FD_curvature_Eh_per_Bohr2"], 2., places=8)
        self.assertEqual(row["grid_semantics"], "FIXED_CENTER_CARTESIAN_QUADRATURE")
        csv_header = (output / "directional_comparisons.csv").read_text().splitlines()[0]
        for column in ("grid_semantics", "fixed_grid_content_sha256", "direction_type", "mode_index"):
            self.assertIn(column, csv_header)

    def test_fixed_grid_never_falls_back_to_original_or_another_center(self):
        wrong_center = self.fixed(self.synthetic_point("wrong_center", 0, 0, -99., [0., 0., 0.]))
        correct_center = self.fixed(self.synthetic_point("correct_center", 0, 0, -75., [0., 0., 0.]))
        minus = self.fixed(self.synthetic_point("minus_fixed", -1, .01, -74.9999, [-.02, 0., 0.], tightened=False), correct_center)
        plus = self.fixed(self.synthetic_point("plus_fixed", 1, .01, -74.9999, [.02, 0., 0.], tightened=False), correct_center)
        # All recorded numerical settings match the original here. The old E0
        # fallback is still invalid because absolute grid identity differs.
        output = self.root / "aggregate"; output.mkdir(); state = {}
        worker.aggregate({"points": [minus, plus, wrong_center]}, state, output, lambda *a, **k: None, [])
        row = state["paired_comparisons"][0]
        self.assertIsNone(row["center_energy_source"])
        self.assertNotIn("energy_FD_curvature_Eh_per_Bohr2", row)

    def test_fixed_grid_id_and_center_hash_are_both_part_of_pairing(self):
        original = {"case": "water", "settings": {"scf_conv_tol": 1e-12},
                    "original_frequency_result": {"sha256": "f" * 64}, "direction": {"type": "mode", "index": 1}}
        fixed = {**original, "grid_semantics": "FIXED_CENTER_CARTESIAN_QUADRATURE", "fixed_grid_content_sha256": "g" * 64,
                 "center_result": {"sha256": "c" * 64}}
        self.assertNotEqual(worker.point_group_key(original), worker.point_group_key(fixed))
        self.assertNotEqual(worker.point_group_key(fixed), worker.point_group_key({**fixed, "fixed_grid_content_sha256": "h" * 64}))
        self.assertNotEqual(worker.point_group_key(fixed), worker.point_group_key({**fixed, "center_result": {"sha256": "d" * 64}}))

    def test_cli_accepts_hash_bound_center_and_signed_response_sources(self):
        args = worker.parser().parse_args(["--request", "request.json", "--request-sha256", "f" * 64,
             "--output", "reports/result", "--center-result", "center.json", "--center-result-sha256", "c" * 64,
             "--minus-result", "minus.json", "--minus-result-sha256", "a" * 64,
             "--plus-result", "plus.json", "--plus-result-sha256", "b" * 64])
        self.assertEqual(args.center_result, Path("center.json"))
        self.assertEqual(args.minus_result_sha256, "a" * 64)
        self.assertEqual(args.plus_result_sha256, "b" * 64)


def suite():
    result = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    result.addTests(loader.loadTestsFromTestCase(original_tests.NumericalTests))
    result.addTests(loader.loadTestsFromTestCase(original_tests.ProvenanceTests))
    result.addTests(loader.loadTestsFromTestCase(NoForkMetadataTests))
    result.addTests(loader.loadTestsFromTestCase(RootcauseAlgebraTests))
    # Avoid rerunning inherited original cases, which are already in this suite.
    result.addTests(RootcauseProvenanceTests(name) for name in RootcauseProvenanceTests.__dict__ if name.startswith("test_"))
    return result


if __name__ == "__main__":
    raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(suite()).wasSuccessful() else 1)
