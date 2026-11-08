"""Negative-subspace algebra and provenance tests; no molecular kernels run."""
from __future__ import annotations

import ast
import argparse
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
SOURCE = SCRIPTS / "research4_b_negative_subspace_worker_20261108.py"
FROZEN_V6 = SCRIPTS / "research4_b_validation_recovery_worker_v6_20261107.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load(SOURCE, "negative_subspace_worker_test_subject")
prior = load(SCRIPTS / "test_research4_b_validation_recovery_worker_v6_20261107.py", "subspace_prior_v6_tests")


def rebind(module, seen=None):
    seen = set() if seen is None else seen
    if id(module) in seen:
        return
    seen.add(id(module))
    if hasattr(module, "worker"):
        module.worker = worker
    if hasattr(module, "SOURCE"):
        module.SOURCE = SOURCE
    if hasattr(module, "RECOVERY"):
        module.RECOVERY = SOURCE
    for name in ("prior", "original_tests", "recovery_tests"):
        if hasattr(module, name):
            rebind(getattr(module, name), seen)


rebind(prior)


class V7CompatibilityTests(unittest.TestCase):
    def test_method_settings_and_scf_qualification_remain_identical_to_frozen_v6(self):
        functions = lambda path: {node.name: node for node in ast.parse(path.read_text()).body
                                  if isinstance(node, ast.FunctionDef)}
        v6, v7 = functions(FROZEN_V6), functions(SOURCE)
        for name in ("settings_for", "configure_scf_tolerance", "independent_scf_qualification",
                     "load_warmstart", "checkpoint_probe", "unit_direction", "central_difference"):
            self.assertEqual(ast.dump(v6[name], include_attributes=False),
                             ast.dump(v7[name], include_attributes=False), name)

    def test_different_frozen_candidates_cannot_be_paired_but_can_share_the_same_energy_center(self):
        point = {"case": "protocatechuic_acid", "settings": {"grid_level": 3},
                 "original_frequency_result": {"sha256": "f" * 64},
                 "direction": {"type": "subspace_minimum", "index": 1},
                 "direction_resolution": {"subspace_result": {"sha256": "a" * 64}}}
        other = {**point, "direction_resolution": {"subspace_result": {"sha256": "b" * 64}}}
        self.assertNotEqual(worker.point_group_key(point), worker.point_group_key(other))
        self.assertEqual(worker.point_group_key(point, False), worker.point_group_key(other, False))


class ProjectedSubspaceTests(unittest.TestCase):
    @staticmethod
    def problem():
        rng = np.random.default_rng(20261108)
        basis, _ = np.linalg.qr(rng.normal(size=(9, 9)))
        hessian = (basis * np.array([-3., -1., .2, .7, 1., 2., 3., 4., 5.])) @ basis.T
        columns = rng.normal(size=(9, 3))
        columns[:, 1] += 1.5 * columns[:, 0]
        columns[:, 2] += .6 * columns[:, 0] - .3 * columns[:, 1]
        columns /= np.linalg.norm(columns, axis=0)
        return hessian, columns

    def test_generalized_eigenproblem_matches_direct_orthonormal_subspace_projection(self):
        hessian, columns = self.problem()
        derivative = hessian @ columns
        arrays, metrics = worker.projected_subspace(columns, derivative, np)
        self.assertTrue(metrics["qualified"])
        orthonormal, _ = np.linalg.qr(columns)
        expected = np.linalg.eigvalsh(orthonormal.T @ hessian @ orthonormal)
        np.testing.assert_allclose(arrays["generalized_eigenvalues"], expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(arrays["G"], columns.T @ columns, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(arrays["Kraw"], columns.T @ derivative, rtol=1e-14, atol=1e-14)
        ordinary = np.linalg.eigvalsh(arrays["Ksym"])
        self.assertGreater(np.linalg.norm(ordinary - expected), .05)
        eigenvectors, eigenvalues = arrays["generalized_eigenvectors"], arrays["generalized_eigenvalues"]
        residual = arrays["Ksym"] @ eigenvectors - (arrays["G"] @ eigenvectors) * eigenvalues
        self.assertLess(float(np.linalg.norm(residual)), 1e-11)
        np.testing.assert_allclose(eigenvectors.T @ arrays["G"] @ eigenvectors, np.eye(3), atol=1e-12)

    def test_selected_cartesian_direction_is_normalized_and_has_a_deterministic_phase(self):
        hessian, columns = self.problem()
        arrays, metrics = worker.projected_subspace(columns, hessian @ columns, np)
        direction = arrays["selected_direction"]
        coefficients = arrays["selected_coefficients"]
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1., places=13)
        self.assertAlmostEqual(float(coefficients @ arrays["G"] @ coefficients), 1., places=12)
        np.testing.assert_allclose(columns @ coefficients, direction, atol=1e-12)
        self.assertGreater(direction[int(np.argmax(np.abs(direction)))], 0.)
        self.assertAlmostEqual(float(direction @ hessian @ direction),
                               float(arrays["generalized_eigenvalues"][0]), places=11)
        flipped, _ = worker.projected_subspace(-columns, -hessian @ columns, np)
        np.testing.assert_allclose(flipped["selected_direction"], direction, atol=1e-12)

    def test_nonsymmetric_response_keeps_raw_matrix_and_reports_symmetrization(self):
        hessian, columns = self.problem()
        gram = columns.T @ columns
        skew = np.array([[0., .2, -.1], [-.2, 0., .3], [.1, -.3, 0.]])
        derivative = hessian @ columns + columns @ np.linalg.solve(gram, skew)
        arrays, metrics = worker.projected_subspace(columns, derivative, np)
        expected_symmetric = columns.T @ hessian @ columns
        np.testing.assert_allclose(arrays["Kraw"], expected_symmetric + skew, atol=1e-12)
        np.testing.assert_allclose(arrays["Ksym"], expected_symmetric, atol=1e-12)
        self.assertGreater(metrics["antisymmetric_norm"], .1)
        self.assertGreater(metrics["antisymmetric_relative_norm"], 0.)
        np.testing.assert_array_equal(arrays["D"], derivative)

    def test_rank_deficiency_and_poor_gram_condition_prevent_selection(self):
        hessian, columns = self.problem()
        duplicate = columns.copy(); duplicate[:, 1] = duplicate[:, 0]
        nearly_duplicate = columns.copy()
        nearly_duplicate[:, 1] = columns[:, 0] + 1e-6 * columns[:, 1]
        nearly_duplicate[:, 1] /= np.linalg.norm(nearly_duplicate[:, 1])
        for invalid in (duplicate, nearly_duplicate):
            with self.subTest(rank_deficient=np.array_equal(invalid[:, 0], invalid[:, 1])):
                arrays, metrics = worker.projected_subspace(invalid, hessian @ invalid, np)
                self.assertFalse(metrics["qualified"])
                self.assertNotIn("selected_direction", arrays)
                self.assertNotIn("generalized_eigenvalues", arrays)
                for name in ("Q", "D", "G", "Kraw", "Ksym"):
                    self.assertTrue(np.isfinite(arrays[name]).all())


class SubspaceProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="v7-subspace-unit-", dir=worker.ROOT / "reports")
        self.directory = Path(self.temp.name)
        self.case, self.h = "protocatechuic_acid", .001
        self.hessian, _ = ProjectedSubspaceTests.problem()
        self.false_hessian = self.hessian + 3 * np.diag(np.linspace(-1., 1., 9))
        rng = np.random.default_rng(91)
        mass_basis, _ = np.linalg.qr(rng.normal(size=(9, 3)))
        self.masses = np.array([1., 12., 16.])
        raw_modes = (mass_basis / np.repeat(np.sqrt(self.masses), 3)[:, None]).T.reshape(3, 3, 3)
        self.geometry = np.array([[0., 0., 0.], [1.5, 0., 0.], [0., 2., 0.]])
        original_settings = {"xc": "PBE", "basis": "def2-svp", "grid_level": 1,
                             "gradient_grid_response": False, "scf_conv_tol": 1e-8}
        source_values = {"geometry_bohr.npy": self.geometry, "normal_modes.npy": raw_modes,
                         "masses_amu.npy": self.masses, "frequencies_cm1.npy": np.array([-3., -2., -1.]),
                         "hessian_used.npy": self.hessian.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3)}
        source_refs = {}
        for name, value in source_values.items():
            worker.save_array(self.directory / name, value, np)
            source_refs[name] = worker.ref(self.directory / name)
        self.frequency = self.directory / "frequency.json"
        worker.save(self.frequency, {"manufactured_source": True}, exclusive=True)
        self.source = {"source": {"case": self.case, "settings": original_settings, "arrays": source_refs},
                       "source_ref": worker.ref(self.frequency), "arrays": source_values,
                       "geometry": {"atom_symbols": ["H", "C", "O"],
                                    "coordinates_angstrom": (self.geometry * worker.BOHR_TO_ANGSTROM).tolist()},
                       "cartesian_hessian": self.hessian, "case": {"name": self.case}}
        self.point_request = {"schema": worker.REQUEST_SCHEMA, "operation": "point", "case": self.case,
                              "grid_level": 3, "gradient_grid_response": False, "scf_conv_tol": 1e-12,
                              "scf_conv_tol_grad": 1e-9, "small_rho_cutoff": 0, "direct_scf_tol": 1e-15,
                              "frequency_result": worker.ref(self.frequency), "source_worker": worker.ref(worker.ORIGINAL)}
        self.settings = worker.settings_for(self.point_request, self.source)
        self.columns = np.column_stack([worker.unit_direction({"type": "mode", "index": index}, raw_modes, np).ravel()
                                        for index in (1, 2, 3)])
        self.energy, self.slope = -100., np.linspace(-.003, .003, 9)
        self.center = self.make_point("center", 1, 0)
        self.points = [self.make_point(f"mode{index:02d}-{'minus' if sign < 0 else 'plus'}", index, sign)
                       for index in (1, 2, 3) for sign in (-1, 1)]
        self.request = {**self.point_request, "operation": "subspace_aggregate", "step_bohr": self.h,
                        "mode_indices": [1, 2, 3], "points": self.points}
        self.args = argparse.Namespace(center_result=worker.checked(self.center), center_result_sha256=self.center["sha256"])
        self.output_count = 0

    def tearDown(self):
        self.temp.cleanup()

    def make_point(self, label, mode_index, sign):
        directory = self.directory / label; directory.mkdir()
        q = self.columns[:, mode_index - 1].reshape(3, 3)
        displacement = sign * self.h * q.ravel()
        energy = self.energy + self.slope @ displacement + .5 * displacement @ self.hessian @ displacement
        gradient_true = (self.slope + self.hessian @ displacement).reshape(3, 3)
        gradient_false = (self.slope + self.false_hessian @ displacement).reshape(3, 3)
        values = {"direction.npy": q, "geometry_bohr.npy": self.geometry + displacement.reshape(3, 3),
                  "gradient.npy": gradient_false, "gradient_grid_response_true.npy": gradient_true,
                  "predicted_Hq.npy": (self.hessian @ q.ravel()).reshape(3, 3)}
        arrays = {}
        for name, value in values.items():
            worker.save_array(directory / name, value, np); arrays[name] = worker.ref(directory / name)
        observations = {"false": {"gradient_grid_response": False, "array": arrays["gradient.npy"]},
                        "true": {"gradient_grid_response": True, "array": arrays["gradient_grid_response_true.npy"],
                                 "regenerated_quadrature": {"nonzero_quadrature_sets_numerically_match": False,
                                                            "limitation": "Manufactured unproved grid-set equivalence"}}}
        point = {"schema": worker.SCHEMA, "operation": "point", "status": "POINT_COMPUTED", "case": self.case,
                 "source_worker": worker.ref(FROZEN_V6), "direction": {"type": "mode", "index": mode_index},
                 "step_bohr": self.h if sign else 0, "sign": sign, "energy_hartree": float(energy),
                 "scf_converged": True, "scf_qualified": True, "settings": self.settings,
                 "grid_prune": "default", "grid_semantics": "REBUILT_ATOM_CENTERED_QUADRATURE",
                 "original_frequency_result": worker.ref(self.frequency), "arrays": arrays,
                 "gradient_observations": observations}
        worker.save(directory / "result.json", point, exclusive=True)
        return worker.ref(directory / "result.json")

    def aggregate(self, points=None):
        self.output_count += 1
        output = self.directory / f"aggregate-{self.output_count}"; output.mkdir()
        state = {"schema": worker.SCHEMA, "operation": "subspace_aggregate", "source_worker": worker.ref(FROZEN_V6)}
        request = {**self.request, "points": self.points if points is None else points}
        with patch.object(worker, "load_case", return_value=self.source):
            worker.subspace_aggregate(self.args, request, state, output, lambda *a, **k: None, [])
        result_path = output / "result.json"; worker.save(result_path, state, exclusive=True)
        return state, output, result_path

    def rewrite_point(self, binding, change):
        path = worker.checked(binding); value = worker.read(path); change(value, path.parent)
        worker.save(path, value)
        return worker.ref(path)

    def combination(self, result_path):
        args = argparse.Namespace(subspace_result=result_path, subspace_result_sha256=worker.sha(result_path))
        request = {**self.point_request, "direction": {"type": "subspace_minimum", "index": 1},
                   "step_bohr": self.h, "sign": 1}
        return worker.resolve_direction(args, request, self.source, np)

    def modified_result(self, result_path, label, change):
        result = worker.read(result_path); change(result, result_path.parent)
        path = result_path.with_name(label + ".json"); worker.save(path, result, exclusive=True)
        return path

    def test_complete_source_bound_columns_produce_a_diagnostic_candidate_even_when_grid_sets_are_unproved(self):
        state, output, result_path = self.aggregate()
        self.assertEqual(state["status"], "SUBSPACE_AGGREGATE_COMPLETE")
        self.assertEqual(state["missing_columns"], [])
        np.testing.assert_allclose(np.load(worker.checked(state["arrays"]["subspace_true_D.npy"])),
                                   self.hessian @ self.columns, atol=1e-11)
        selection = state["selected_candidate"]
        self.assertEqual(selection["status"], "DIAGNOSTIC_CANDIDATE_AVAILABLE")
        self.assertFalse(selection["grid_sets_numerically_match"])
        self.assertIn("diagnostic", selection["limitation"])
        self.assertFalse(state["full_Hessian_recomputed"])
        self.assertFalse(state["complete_matrix_step_convergence_verified"])
        self.assertTrue((output / "subspace_analysis_zh.txt").is_file())
        q, metadata = self.combination(result_path)
        self.assertAlmostEqual(float(np.linalg.norm(q)), 1., places=12)
        self.assertEqual(metadata["subspace_result"]["sha256"], worker.sha(result_path))

    def test_missing_or_unqualified_columns_remain_partial_without_a_selected_candidate(self):
        state, _, _ = self.aggregate(self.points[:-1])
        self.assertEqual(state["status"], "DIAGNOSTIC_DIRECTION_UNDEFINED")
        self.assertEqual(state["subspace_data_status"], "PARTIAL")
        self.assertEqual(state["missing_columns"][0]["mode_index"], 3)
        self.assertNotIn("selected_candidate", state)
        bad = self.rewrite_point(self.points[0], lambda value, _: value.update(scf_qualified=False))
        state, _, _ = self.aggregate([bad, *self.points[1:]])
        self.assertEqual(state["status"], "DIAGNOSTIC_DIRECTION_UNDEFINED")
        self.assertEqual(state["subspace_data_status"], "PARTIAL")
        self.assertEqual(state["excluded_points"][0]["reason"], "SCF_POINT_NOT_QUALIFIED")
        self.assertNotIn("selected_candidate", state)

    def test_signed_point_geometry_is_checked_against_original_mode_and_step(self):
        def change(value, directory):
            name = directory / "wrong_geometry.npy"
            worker.save_array(name, self.geometry + .01, np)
            value["arrays"]["geometry_bohr.npy"] = worker.ref(name)
        bad = self.rewrite_point(self.points[0], change)
        with self.assertRaisesRegex(ValueError, "exact signed geometry"):
            self.aggregate([bad, *self.points[1:]])

    def test_duplicate_points_and_inconsistent_method_bindings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate subspace point"):
            self.aggregate([*self.points, self.points[0]])
        def change(value, _):
            value["settings"] = {**value["settings"], "grid_level": 5}
        bad = self.rewrite_point(self.points[0], change)
        with self.assertRaisesRegex(ValueError, "method/grid/frequency lineage"):
            self.aggregate([bad, *self.points[1:]])

    def test_combination_requires_unchanged_direction_array_hash(self):
        state, _, result_path = self.aggregate()
        path = worker.checked(state["selected_candidate"]["direction_array"])
        with path.open("wb") as handle:
            np.save(handle, np.zeros((3, 3)), allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "hash changed"):
            self.combination(result_path)

    def test_combination_checks_geometry_and_atom_order_even_with_new_valid_json_hashes(self):
        _, _, result_path = self.aggregate()
        def geometry_change(value, directory):
            path = directory / "different_source_geometry.npy"
            worker.save_array(path, self.geometry + .01, np)
            value["source_geometry_ref"] = worker.ref(path)
        wrong_geometry = self.modified_result(result_path, "wrong-geometry-result", geometry_change)
        with self.assertRaisesRegex(ValueError, "geometry or atom order"):
            self.combination(wrong_geometry)
        wrong_order = self.modified_result(result_path, "wrong-order-result",
                                           lambda value, _: value.update(atom_symbols=["O", "C", "H"]))
        with self.assertRaisesRegex(ValueError, "geometry or atom order"):
            self.combination(wrong_order)

    def test_combination_rejects_a_sign_flipped_candidate_even_when_qv_still_matches(self):
        _, _, result_path = self.aggregate()
        def phase_change(value, directory):
            selection = value["selected_candidate"]
            for key in ("direction_array", "coefficients_array"):
                original = np.load(worker.checked(selection[key]), allow_pickle=False)
                path = directory / ("negative-" + key + ".npy")
                worker.save_array(path, -original, np)
                selection[key] = worker.ref(path)
        wrong_phase = self.modified_result(result_path, "wrong-phase-result", phase_change)
        with self.assertRaisesRegex(ValueError, "canonical phase"):
            self.combination(wrong_phase)

    def test_combination_aggregate_compares_full_vector_prediction_without_overwriting_original_hessian_prediction(self):
        subspace, _, result_path = self.aggregate()
        q, resolution = self.combination(result_path)
        dc = np.load(worker.checked(subspace["selected_candidate"]["predicted_Hq_array"]), allow_pickle=False).reshape(3, 3)
        predictions = {label: np.load(worker.checked(binding), allow_pickle=False).reshape(3, 3)
                       for label, binding in subspace["selected_candidate"]["predicted_Hq_arrays"].items()}
        combinations = []
        for sign in (-1, 1):
            directory = self.directory / f"combination-{sign}"; directory.mkdir()
            displacement = sign * self.h * q.ravel()
            energy = self.energy + self.slope @ displacement + .5 * displacement @ self.hessian @ displacement
            gradient_true = (self.slope + self.hessian @ displacement).reshape(3, 3)
            gradient_false = (self.slope + self.false_hessian @ displacement).reshape(3, 3)
            values = {"direction.npy": q, "geometry_bohr.npy": self.geometry + displacement.reshape(3, 3),
                      "gradient.npy": gradient_false, "gradient_grid_response_true.npy": gradient_true,
                      "predicted_Hq.npy": 5 * q, "predicted_subspace_Hq.npy": dc,
                      "predicted_subspace_false_Hq.npy": predictions["false"],
                      "predicted_subspace_true_Hq.npy": predictions["true"]}
            arrays = {}
            for name, value in values.items():
                worker.save_array(directory / name, value, np); arrays[name] = worker.ref(directory / name)
            point = copy.deepcopy(worker.read(worker.checked(self.points[0])))
            point.update(direction={"type": "subspace_minimum", "index": 1}, direction_resolution=resolution,
                         sign=sign, step_bohr=self.h, energy_hartree=float(energy), arrays=arrays,
                         original_settings=self.source["source"]["settings"],
                         original_center_energy_hartree=self.energy,
                         gradient_controls_share_one_SCF_energy_and_density=True,
                         gradient_observations={"false": {"gradient_grid_response": False, "array": arrays["gradient.npy"]},
                                                "true": {"gradient_grid_response": True, "array": arrays["gradient_grid_response_true.npy"]}})
            worker.save(directory / "result.json", point, exclusive=True)
            combinations.append(worker.ref(directory / "result.json"))
        output = self.directory / "combination-comparison"; output.mkdir(); state = {}
        worker.aggregate({"points": [self.center, *combinations]}, state, output, lambda *a, **k: None, [])
        self.assertEqual(len(state["paired_comparisons"]), 2)
        for observed in state["paired_comparisons"]:
            self.assertLess(observed["mixed_gradient_FD_minus_Dc_norm_Eh_per_Bohr2"], 1e-10)
        row = next(row for row in state["paired_comparisons"] if row["gradient_grid_response"])
        self.assertAlmostEqual(row["analytic_directional_curvature_Eh_per_Bohr2"], 5., places=11)
        self.assertAlmostEqual(row["predicted_subspace_curvature_Eh_per_Bohr2"],
                               subspace["selected_candidate"]["projected_eigenvalue_Eh_per_Bohr2"], places=11)
        self.assertLess(row["mixed_gradient_FD_minus_Dc_norm_Eh_per_Bohr2"], 1e-10)
        self.assertLess(abs(row["gradient_FD_minus_predicted_subspace_curvature_Eh_per_Bohr2"]), 1e-10)
        self.assertLess(abs(row["energy_FD_minus_predicted_subspace_curvature_Eh_per_Bohr2"]), 1e-7)
        self.assertEqual(row["center_point_result"]["sha256"], self.center["sha256"])

    def test_false_and_true_predictions_apply_the_same_true_selected_coefficients(self):
        state, _, _ = self.aggregate()
        selection = state["selected_candidate"]
        coefficients = np.load(worker.checked(selection["coefficients_array"]), allow_pickle=False)
        false_own_coefficients = np.load(worker.checked(state["arrays"]["subspace_false_selected_coefficients.npy"]), allow_pickle=False)
        for label in ("false", "true"):
            derivative = np.load(worker.checked(state["arrays"][f"subspace_{label}_D.npy"]), allow_pickle=False)
            prediction = np.load(worker.checked(selection["predicted_Hq_arrays"][label]), allow_pickle=False).ravel()
            np.testing.assert_allclose(prediction, derivative @ coefficients, atol=1e-12)
            self.assertAlmostEqual(selection["predicted_branch_curvatures_Eh_per_Bohr2"][label],
                                   float((self.columns @ coefficients) @ prediction), places=11)
            if label == "false":
                self.assertGreater(float(np.linalg.norm(prediction - derivative @ false_own_coefficients)), .01)

    def test_mass_normalized_rectangular_mapping_uses_original_columns_and_preserves_coefficient_convention(self):
        self.source["arrays"]["frequencies_cm1.npy"] = np.array([-3., -2., 1.])
        self.request["mode_indices"] = [1, 2]
        selected_points = [binding for binding in self.points
                           if worker.read(worker.checked(binding))["direction"]["index"] <= 2]
        state, _, _ = self.aggregate(selected_points)
        self.assertEqual(state["status"], "SUBSPACE_AGGREGATE_COMPLETE")
        all_modes = self.source["arrays"]["normal_modes.npy"].reshape(3, 9).T
        expected = all_modes.T @ self.hessian @ all_modes[:, :2]
        mapped = np.load(worker.checked(state["arrays"]["subspace_true_mass_normalized_B_all_negative.npy"]), allow_pickle=False)
        self.assertEqual(mapped.shape, (3, 2))
        np.testing.assert_allclose(mapped, expected, atol=1e-11)
        self.assertTrue(state["mass_basis_mapping"]["source_mass_normalization_verified"])
        coefficient = np.load(worker.checked(state["selected_candidate"]["original_mass_normalized_mode_coefficients_array"]), allow_pickle=False)
        direction = np.load(worker.checked(state["selected_candidate"]["direction_array"]), allow_pickle=False)
        np.testing.assert_allclose(all_modes[:, :2] @ coefficient, direction.ravel(), atol=1e-12)


def cases(suite):
    for value in suite:
        if isinstance(value, unittest.TestSuite):
            yield from cases(value)
        else:
            yield value


def suite():
    result = unittest.TestSuite()
    for test in cases(prior.suite()):
        # v7 extends pairing with the frozen combination-result hash. Preserve
        # the old scientific-setting checks separately while testing the new
        # key semantics explicitly above.
        if test._testMethodName != "test_method_settings_and_qualification_are_identical_to_v5":
            result.addTest(test)
    result.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(V7CompatibilityTests))
    result.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(ProjectedSubspaceTests))
    result.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(SubspaceProvenanceTests))
    return result


if __name__ == "__main__":
    raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(suite()).wasSuccessful() else 1)
