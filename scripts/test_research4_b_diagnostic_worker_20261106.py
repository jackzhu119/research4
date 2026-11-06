"""Lightweight numerical/provenance tests; never imports or executes PySCF."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).with_name("research4_b_diagnostic_worker_20261106.py")
spec = importlib.util.spec_from_file_location("diagnostic_worker_tests", SOURCE)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class NumericalTests(unittest.TestCase):
    def test_gradient_and_energy_difference_recover_quadratic_hessian(self):
        hessian = np.array([[2., .5, 0.], [.5, -3., .2], [0., .2, 1.]])
        q = np.array([[1., 2., -1.]])
        q /= np.linalg.norm(q)
        x = np.array([[.3, -.4, .1]])
        h = .005
        gradient = lambda point: (hessian @ point.ravel()).reshape(1, 3)
        energy = lambda point: float(.5 * point.ravel() @ hessian @ point.ravel())
        minus, plus = x - h * q, x + h * q
        result = worker.central_difference(q, gradient(minus), gradient(plus), h,
                                          gradient(q), np, e_minus=energy(minus),
                                          e_plus=energy(plus), e_center=energy(x))
        expected = float(q.ravel() @ hessian @ q.ravel())
        self.assertAlmostEqual(result["gradient_FD_curvature_Eh_per_Bohr2"], expected, places=11)
        self.assertAlmostEqual(result["energy_FD_curvature_Eh_per_Bohr2"], expected, places=10)
        self.assertLess(result["gradient_FD_Hq_relative_error"], 1e-12)

    def test_wrong_hessian_sign_is_detected_independently(self):
        q = np.array([[1., 0., 0.]])
        result = worker.central_difference(q, -.01 * q, .01 * q, .01, -q, np)
        self.assertFalse(result["curvature_sign_agrees"])
        self.assertAlmostEqual(result["gradient_FD_Hq_relative_error"], 2.)

    def test_mode_is_euclidean_normalized_without_mass_rescaling(self):
        modes = np.array([[[.2, 0., 0.], [0., .7, 0.]]])
        q = worker.unit_direction({"type": "mode", "index": 1}, modes, np)
        self.assertAlmostEqual(float(np.linalg.norm(q)), 1.)
        self.assertAlmostEqual(float(q[1, 1] / q[0, 0]), 3.5)

    def test_translation_moves_all_atoms_identically_at_unit_norm(self):
        q = worker.unit_direction({"type": "translation", "axis": 2}, np.zeros((3, 17, 3)), np)
        self.assertAlmostEqual(float(np.linalg.norm(q)), 1.)
        np.testing.assert_allclose(q[:, 2], np.full(17, 1 / np.sqrt(17)))
        np.testing.assert_array_equal(q[:, :2], np.zeros((17, 2)))

    def test_finite_difference_rejects_nonunit_direction(self):
        q = np.array([[2., 0., 0.]])
        with self.assertRaisesRegex(ValueError, "unit Euclidean"):
            worker.central_difference(q, q, q, .01, q, np)

    def test_tight_scf_center_is_not_pooled_with_old_tolerance(self):
        point = {"case": "water", "settings": {"scf_conv_tol": 1e-8, "grid_level": 1},
                 "original_frequency_result": {"sha256": "a" * 64},
                 "direction": {"type": "mode", "index": 1}}
        tightened = {**point, "settings": {**point["settings"], "scf_conv_tol": 1e-10}}
        self.assertNotEqual(worker.point_group_key(point, False), worker.point_group_key(tightened, False))


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.root_patch = patch.object(worker, "ROOT", self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def test_source_hash_and_size_are_checked(self):
        p = self.root / "source.json"
        p.write_text("{}\n")
        binding = worker.ref(p)
        self.assertEqual(worker.checked(binding), p)
        p.write_text('{"changed":true}\n')
        with self.assertRaisesRegex(ValueError, "hash changed"):
            worker.checked(binding)

    def test_source_symlink_is_rejected(self):
        p = self.root / "actual.json"; p.write_text("{}")
        alias = self.root / "alias.json"; alias.symlink_to(p)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            worker.checked({"path": str(alias), "sha256": worker.sha(p)})

    def test_failed_live_proof_prevents_output_creation(self):
        request = self.root / "request.json"
        request.write_text(json.dumps({"schema": worker.REQUEST_SCHEMA, "operation": "aggregate", "threads": 2}))
        output = self.root / "reports" / "new"
        args = argparse.Namespace(request=request, request_sha256=worker.sha(request), output=output,
                                  execute=True, point_result=[], point_result_sha256=[])
        with patch.object(worker, "supervise", side_effect=ValueError("no live proof")):
            with self.assertRaisesRegex(ValueError, "no live proof"):
                worker.run(args)
        self.assertFalse(output.exists())

    def test_default_request_is_read_only(self):
        request = self.root / "request.json"
        request.write_text(json.dumps({"schema": worker.REQUEST_SCHEMA, "operation": "aggregate"}))
        output = self.root / "reports" / "new"
        args = argparse.Namespace(request=request, request_sha256=worker.sha(request), output=output,
                                  execute=False, point_result=[], point_result_sha256=[])
        with patch.object(worker, "supervise", side_effect=AssertionError("should not supervise preflight")):
            result = worker.run(args)
        self.assertEqual(result["status"], "NOT_RUN")
        self.assertFalse(output.exists())

    def test_center_and_displacement_are_unambiguous(self):
        request = {"schema": worker.REQUEST_SCHEMA, "operation": "point", "case": "water",
                   "direction": {"type": "mode", "index": 1}, "step_bohr": 0, "sign": 0,
                   "scf_conv_tol": 1e-10}
        worker.validate_request(request)
        with self.assertRaisesRegex(ValueError, "h=0/sign=0"):
            worker.validate_request({**request, "sign": 1})

    def synthetic_point(self, label, sign, step, energy, gradient, *, tightened=True):
        directory = self.root / label; directory.mkdir()
        q = np.array([[1., 0., 0.]])
        arrays = {}
        for name, value in (("direction.npy", q), ("gradient.npy", np.asarray([gradient], dtype=float)),
                            ("predicted_Hq.npy", 2 * q)):
            worker.save_array(directory / name, value, np)
            arrays[name] = worker.ref(directory / name)
        settings = {"scf_conv_tol": 1e-10 if tightened else 1e-8, "grid_level": 1, "gradient_grid_response": False}
        point = {"schema": worker.SCHEMA, "operation": "point", "status": "POINT_COMPUTED",
                 "case": "water", "direction": {"type": "mode", "index": 1},
                 "settings": settings, "original_settings": {**settings, "scf_conv_tol": 1e-8},
                 "original_frequency_result": {"sha256": "f" * 64},
                 "sign": sign, "step_bohr": step, "energy_hartree": energy,
                 "scf_converged": True, "original_center_energy_hartree": 100., "arrays": arrays}
        worker.save(directory / "result.json", point, exclusive=True)
        return worker.ref(directory / "result.json")

    def test_aggregate_uses_new_matching_center_not_loose_original_energy(self):
        points = [self.synthetic_point("minus", -1, .01, -74.9999, [-.02, 0., 0.]),
                  self.synthetic_point("plus", 1, .01, -74.9999, [.02, 0., 0.]),
                  self.synthetic_point("center", 0, 0, -75., [0., 0., 0.])]
        output = self.root / "aggregate"; output.mkdir(); state = {}
        worker.aggregate({"points": points}, state, output, lambda *a, **k: None, [])
        row = state["paired_comparisons"][0]
        self.assertEqual(row["center_energy_source"], "NEW_MATCHING_METHOD_CENTER_POINT")
        self.assertAlmostEqual(row["energy_FD_curvature_Eh_per_Bohr2"], 2., places=8)
        self.assertAlmostEqual(row["gradient_FD_curvature_Eh_per_Bohr2"], 2., places=12)
        self.assertFalse(row["same_settings_as_original_Hessian"])

    def test_aggregate_omits_energy_curvature_without_matching_tight_center(self):
        points = [self.synthetic_point("minus", -1, .01, -74.9999, [-.02, 0., 0.]),
                  self.synthetic_point("plus", 1, .01, -74.9999, [.02, 0., 0.]),
                  self.synthetic_point("loose_center", 0, 0, -75., [0., 0., 0.], tightened=False)]
        output = self.root / "aggregate"; output.mkdir(); state = {}
        worker.aggregate({"points": points}, state, output, lambda *a, **k: None, [])
        row = state["paired_comparisons"][0]
        self.assertIsNone(row["center_energy_source"])
        self.assertNotIn("energy_FD_curvature_Eh_per_Bohr2", row)

    def test_aggregate_exposes_missing_signed_point(self):
        point = self.synthetic_point("only_minus", -1, .01, -74.9999, [-.02, 0., 0.])
        output = self.root / "aggregate"; output.mkdir(); state = {}
        worker.aggregate({"points": [point]}, state, output, lambda *a, **k: None, [])
        self.assertEqual(state["paired_comparisons"], [])
        self.assertEqual(state["missing_pairs"][0]["status"], "NOT_RUN_MISSING_SIGNED_POINT")


if __name__ == "__main__":
    unittest.main()
