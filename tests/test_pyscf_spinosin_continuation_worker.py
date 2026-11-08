"""Meaningful continuation guards; no SCF/gradient/Hessian is computed here."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location("continuation_worker", Path(__file__).parents[1] / "scripts/pyscf_spinosin_continuation_worker_20261031.py")
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def fake_method(converged=True, atoms=3):
    class DF:
        def __init__(self):
            self._cderi = np.ones((7, 6))
            self.auxmol = object()
        def reset(self, mol):
            raise AssertionError("Cache release must never reset the auxiliary molecule")
    return SimpleNamespace(with_df=DF(), converged=converged, e_tot=-76., mol=SimpleNamespace(natm=atoms),
                           _eri=object(), checkpoint="preserve.chk")


class ContinuationWorkerTests(unittest.TestCase):
    def test_release_only_cache_keeps_auxmol_and_method(self):
        mf = fake_method(); auxmol, eri = mf.with_df.auxmol, mf._eri
        result = worker.release_scf_cache_before_gradient(mf)
        self.assertIsNone(mf.with_df._cderi)
        self.assertIs(mf.with_df.auxmol, auxmol)
        self.assertIs(mf._eri, eri)
        self.assertEqual(mf.checkpoint, "preserve.chk")
        self.assertEqual(result["SCF_cache_before_release"]["nbytes"], 7 * 6 * 8)
        self.assertFalse(result["with_df_reset_called"])

    def test_missing_or_disk_cache_cannot_claim_cached_scf(self):
        for cache in (None, "disk-cache.h5"):
            mf = fake_method(); mf.with_df._cderi = cache
            with self.assertRaises(RuntimeError):
                worker.release_scf_cache_before_gradient(mf)

    def test_missing_auxmol_is_refused_before_release(self):
        mf = fake_method(); mf.with_df.auxmol = None
        with self.assertRaises(RuntimeError):
            worker.release_scf_cache_before_gradient(mf)
        self.assertIsInstance(mf.with_df._cderi, np.ndarray)

    def test_hook_runs_after_scf_before_original_gradient_and_records_time(self):
        events = []
        class Parent:
            def __init__(self, base):
                self.base, self.mol = base, base.mol
            def kernel(self, marker=None):
                self.marker = marker
                self.assertion = self.base.with_df._cderi is None
                return np.ones((self.mol.natm, 3))
        Type = worker.cache_releasing_gradient_type(Parent, lambda name, details: events.append((name, details)), memory_mb=lambda: 123.)
        gradient = Type(fake_method()); result = gradient.kernel(marker="forwarded")
        self.assertTrue(gradient.assertion)
        self.assertEqual(gradient.marker, "forwarded")
        np.testing.assert_array_equal(result, np.ones((3, 3)))
        self.assertEqual([event[0] for event in events], ["gradient_started_after_cache_release", "gradient_kernel_complete"])
        self.assertGreaterEqual(events[-1][1]["gradient_cpu_seconds"], 0)
        self.assertEqual(events[0][1]["current_rss_mb_after_release"], 123.)

    def test_unconverged_scf_never_enters_or_releases_for_gradient(self):
        class Parent:
            def __init__(self, base): self.base, self.mol = base, base.mol
            def kernel(self): raise AssertionError("must not execute")
        grad = worker.cache_releasing_gradient_type(Parent, lambda *a: None)(fake_method(False))
        with self.assertRaises(RuntimeError): grad.kernel()
        self.assertIsInstance(grad.base.with_df._cderi, np.ndarray)

    def test_scanner_reset_event_precedes_original_reset(self):
        order = []
        class Parent:
            def reset(self, mol): order.append(("parent", mol)); return self
        grad = worker.cache_releasing_gradient_type(Parent, lambda name, details: order.append((name, details)))()
        self.assertIs(grad.reset("geometry"), grad)
        self.assertEqual(order[0][0], "gradient_scanner_geometry_reset")
        self.assertEqual(order[1], ("parent", "geometry"))

    def test_hook_rejects_invalid_gradient(self):
        class Parent:
            def __init__(self, base): self.base, self.mol = base, base.mol
            def kernel(self): return np.zeros((2, 3))
        grad = worker.cache_releasing_gradient_type(Parent, lambda *a: None)(fake_method())
        with self.assertRaises(RuntimeError): grad.kernel()

    def test_actual_pyscf_scanner_mro_keeps_hook_without_running_scf(self):
        worker.install_sites()
        try:
            from pyscf import dft, gto
            from pyscf.df.grad.rks import Gradients
        except ImportError:
            self.skipTest("Optional PySCF site not installed; other guards still run")
        mol = gto.M(atom="H 0 0 0; H 0 0 0.7", basis="sto-3g", verbose=0)
        mf = dft.RKS(mol).density_fit(only_dfj=True)
        Type = worker.cache_releasing_gradient_type(Gradients, lambda *a: None)
        scanner = Type(mf).as_scanner()
        self.assertIs(scanner.kernel.__func__, Type.kernel)
        self.assertIs(scanner.reset.__func__, Type.reset)
        self.assertIsNone(mf.mo_energy)

    def test_zero_and_negative_modes_fail_minimum_checks(self):
        for values in ([0, 2, 3], [-1, 2, 3]):
            info = worker.frequencies_payload(values, np.ones((3, 3, 3)), 3)
            self.assertFalse(worker.frequency_minimum_checks(info)["frequency_validation_passed"])
        info = worker.frequencies_payload([1, 2, 3], np.ones((3, 3, 3)), 3)
        self.assertTrue(worker.frequency_minimum_checks(info)["stationary_minimum_verified"])
        self.assertEqual(info["IR_intensities_status"], "NOT_COMPUTED")

    def test_mode_count_and_nan_guards(self):
        with self.assertRaises(ValueError): worker.frequencies_payload([1, 2], np.ones((2, 3, 3)), 3)
        with self.assertRaises(ValueError): worker.frequencies_payload([1, 2, np.nan], np.ones((3, 3, 3)), 3)
        info = worker.frequencies_payload([1, 2, 3], np.ones((3, 3, 3)), 3)
        self.assertAlmostEqual(info["frequencies_thz"][0], .0299792458)

    def test_symmetrization_preserves_raw_and_refuses_large_asymmetry(self):
        raw = np.eye(9).reshape(3, 3, 3, 3).transpose(0, 2, 1, 3).copy()
        raw[0, 1, 0, 0] += 1e-7
        original = raw.copy()
        used, delta = worker.prepare_hessian_for_analysis(raw, 3)
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(used, used.transpose(1, 0, 3, 2))
        self.assertAlmostEqual(delta, 1e-7)
        raw[0, 1, 0, 0] += 1e-3
        with self.assertRaises(ValueError): worker.prepare_hessian_for_analysis(raw, 3)
        with self.assertRaises(ValueError): worker.prepare_hessian_for_analysis(np.zeros((3, 3)), 3)

    def test_fullprecision_coordinates_require_sha_and_match_xyz(self):
        xyz = np.ones((3, 3))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "coordinates.npy"
            worker.atomic_npy(path, xyz + 1e-14)
            restored = worker.precise_input_coordinates(xyz, path, worker.sha(path))
            self.assertGreater(float(np.max(restored - xyz)), 0)
            with self.assertRaises(ValueError): worker.precise_input_coordinates(xyz, path)
            with self.assertRaises(ValueError): worker.precise_input_coordinates(xyz + .01, path, worker.sha(path))
            with self.assertRaises(ValueError): worker.precise_input_coordinates(xyz, path, "wrong")

    def test_frequency_gate_binds_fullprecision_and_method_not_thread_count(self):
        with tempfile.TemporaryDirectory() as temp:
            path, coords = Path(temp) / "optimized.xyz", Path(temp) / "coordinates.npy"
            path.write_text("fixture"); worker.atomic_npy(coords, np.ones((3, 3)))
            settings = dict(xc="PBE", basis="def2-svp", grid_level=1, dispersion="NONE_EXPLICIT_PBE_PILOT_NOT_D3",
                            density_fitting=True, auxbasis="def2-universal-jfit", density_fit_backend="INCORE_DFJ_RELEASE_BEFORE_GRAD_V3", threads=1)
            result = dict(status="DONE", geometry_optimized=True, optimizer_converged=True, scf_converged=True,
                          final_gradient_thresholds_passed=True, optimized_xyz_sha256=worker.sha(path),
                          optimized_coordinates_angstrom_npy_sha256=worker.sha(coords), settings=settings)
            worker.require_optimized_result(result, path, dict(settings, threads=4), coords)
            with self.assertRaises(ValueError): worker.require_optimized_result(result, path, settings)
            with self.assertRaises(ValueError): worker.require_optimized_result(result, path, dict(settings, grid_level=2), coords)
            result["optimizer_converged"] = False
            with self.assertRaises(ValueError): worker.require_optimized_result(result, path, settings, coords)


if __name__ == "__main__":
    unittest.main()
