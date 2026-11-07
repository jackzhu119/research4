"""Direct-control guards and lazy PySCF object dispatch; no quantum jobs."""
import ast
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/pyscf_spinosin_direct_worker_20261031.py"
SPEC = importlib.util.spec_from_file_location("direct_worker", SCRIPT)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def fake_method(**kwargs):
    values = dict(mol=SimpleNamespace(natm=3, incore_anyway=False), _eri=None,
                  direct_scf=True, direct_scf_tol=1e-13, converged=True, e_tot=-76.)
    values.update(kwargs)
    return SimpleNamespace(**values)


class DirectWorkerTests(unittest.TestCase):
    def settings(self, **kwargs):
        return worker.worker_settings(SimpleNamespace(grid_level=1, threads=kwargs.get("threads", 1), maxsteps=35))

    def test_settings_explicit_no_density_fitting_same_physics(self):
        settings = self.settings()
        self.assertFalse(settings["density_fitting"])
        self.assertEqual(settings["density_fit_backend"], "SCREENED_DIRECT_RKS_V4")
        self.assertIsNone(settings["auxbasis"])
        self.assertIsNone(settings["hessian_auxbasis_response"])
        self.assertEqual(settings["xc"], "PBE")
        self.assertEqual(settings["basis"], "def2-svp")
        self.assertEqual(settings["grid_level"], 1)
        self.assertEqual(settings["dispersion"], "NONE_EXPLICIT_PBE_PILOT_NOT_D3")
        self.assertEqual(settings["scf_conv_tol_hartree"], 1e-8)
        self.assertFalse(settings["global_four_center_ERI_cache_allowed"])
        self.assertEqual(settings["optimizer_thresholds"]["convergence_grms"], 3e-4)

    def test_source_contains_no_density_fit_wrapper_or_df_gradient_import(self):
        text = SCRIPT.read_text();tree = ast.parse(text)
        self.assertNotIn("_cderi", text)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or "").startswith("pyscf.df"))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotEqual(node.func.attr, "density_fit")

    def test_direct_policy_accepts_no_cache(self):
        info = worker.direct_integral_policy(fake_method())
        self.assertFalse(info["density_fitting"])
        self.assertFalse(info["global_four_center_ERI_cached"])

    def test_direct_policy_rejects_df_eri_or_incore_heuristic(self):
        for value in (fake_method(with_df=object()), fake_method(_eri=np.ones((1,))),
                      fake_method(mol=SimpleNamespace(natm=3, incore_anyway=True)),
                      fake_method(direct_scf=False), fake_method(direct_scf_tol=1e-8)):
            with self.assertRaises(RuntimeError):
                worker.direct_integral_policy(value)

    def test_get_jk_delegates_exactly_to_reviewed_direct_contraction(self):
        events = []
        class Parent:
            def __init__(self): self.__dict__.update(vars(fake_method()))
            def get_jk(self, *args, **kwargs):
                raise AssertionError("RHF incore heuristic must be bypassed")
        def contraction(self, **kwargs):
            events.append((self, kwargs));return "J", "K"
        direct = worker.screened_direct_rks_type(Parent, contraction)()
        dm = object();mol = object()
        self.assertEqual(direct.get_jk(mol, dm, 0, False, True, .2), ("J", "K"))
        self.assertIs(events[0][0], direct)
        self.assertEqual(events[0][1], dict(mol=mol, dm=dm, hermi=0, with_j=False, with_k=True, omega=.2))
        self.assertIsNone(direct._eri)

    def test_get_jk_rejects_cache_before_any_contraction(self):
        class Parent:
            def __init__(self): self.__dict__.update(vars(fake_method(_eri=object())))
        def forbidden(*args, **kwargs): raise AssertionError("must not contract")
        direct = worker.screened_direct_rks_type(Parent, forbidden)()
        with self.assertRaises(RuntimeError): direct.get_jk()

    def test_standard_gradient_hook_forwards_arguments_and_records_timing(self):
        events = []
        class Parent:
            def __init__(self, mf): self.base, self.mol = mf, mf.mol
            def kernel(self, marker=None):
                self.marker = marker;return np.ones((self.mol.natm, 3))
        gradient = worker.direct_gradient_type(Parent, lambda e, d: events.append((e, d)), lambda: 42.)(fake_method())
        np.testing.assert_array_equal(gradient.kernel(marker="forwarded"), np.ones((3, 3)))
        self.assertEqual(gradient.marker, "forwarded")
        self.assertEqual([e[0] for e in events], ["direct_gradient_started", "direct_gradient_kernel_complete"])
        self.assertEqual(events[0][1]["current_rss_mb_before_gradient"], 42.)
        self.assertGreaterEqual(events[1][1]["gradient_cpu_seconds"], 0)
        self.assertFalse(events[1][1]["direct_integral_policy"]["global_four_center_ERI_cached"])

    def test_unconverged_or_df_method_cannot_enter_gradient(self):
        class Parent:
            def __init__(self, mf): self.base, self.mol = mf, mf.mol
            def kernel(self): raise AssertionError("must not calculate")
        for mf in (fake_method(converged=False), fake_method(with_df=object())):
            gradient = worker.direct_gradient_type(Parent, lambda *a: None)(mf)
            with self.assertRaises(RuntimeError): gradient.kernel()

    def test_gradient_hook_refuses_bad_return_and_new_eri_cache(self):
        class Parent:
            def __init__(self, mf): self.base, self.mol = mf, mf.mol
            def kernel(self): return np.ones((2, 3))
        with self.assertRaises(RuntimeError):
            worker.direct_gradient_type(Parent, lambda *a: None)(fake_method()).kernel()
        class CachingParent(Parent):
            def kernel(self): self.base._eri = object();return np.ones((3, 3))
        with self.assertRaises(RuntimeError):
            worker.direct_gradient_type(CachingParent, lambda *a: None)(fake_method()).kernel()

    def test_gradient_reset_event_and_return_preserved(self):
        order = []
        class Parent:
            def reset(self, mol): order.append(("original_reset", mol));return self
        obj = worker.direct_gradient_type(Parent, lambda e, d: order.append((e, d)))()
        self.assertIs(obj.reset("next"), obj)
        self.assertEqual(order, [("gradient_scanner_geometry_reset", {}), ("original_reset", "next")])

    def test_actual_pyscf_classes_and_scanner_remain_standard_non_df_without_integrals(self):
        worker.install_sites()
        try:
            from pyscf import dft, gto
            from pyscf.scf.hf import SCF
            from pyscf.grad.rks import Gradients
            from pyscf.hessian import rks as hessian_rks
        except ImportError:
            self.skipTest("Optional PySCF site unavailable")
        mol = gto.M(atom="H 0 0 0; H 0 0 .7", basis="sto-3g", verbose=0)
        mol.incore_anyway = False
        Type = worker.screened_direct_rks_type(dft.rks.RKS, SCF.get_jk)
        mf = Type(mol);mf.direct_scf = True;mf.direct_scf_tol = 1e-13
        self.assertIsNone(mf._eri)
        self.assertIsNone(getattr(mf, "with_df", None))
        self.assertIsNone(mf.mo_energy)
        self.assertIsInstance(mf.Hessian(), hessian_rks.Hessian)
        self.assertEqual(type(mf.Hessian()).__module__, "pyscf.hessian.rks")
        Hook = worker.direct_gradient_type(Gradients, lambda *a: None)
        scanner = Hook(mf).as_scanner()
        self.assertIs(scanner.kernel.__func__, Hook.kernel)
        self.assertIs(scanner.reset.__func__, Hook.reset)
        self.assertIs(scanner.base.get_jk.__func__, Type.get_jk)
        self.assertFalse(scanner.mol.incore_anyway)
        self.assertIsNone(scanner.base.mo_energy)

    def test_modes_219_complete_positive_and_units(self):
        info = worker.frequencies_payload(np.arange(1, 220), np.ones((219, 75, 3)), 75)
        self.assertEqual(info["normal_mode_count"], 219)
        self.assertTrue(worker.frequency_minimum_checks(info)["frequency_validation_passed"])
        self.assertAlmostEqual(info["frequencies_thz"][0], .0299792458)
        self.assertEqual(info["IR_intensities_status"], "NOT_COMPUTED")

    def test_missing_complex_nonfinite_or_zero_vectors_rejected(self):
        for freq, modes in [([1, 2], np.ones((2, 3, 3))),
                            ([1, 2, np.nan], np.ones((3, 3, 3))),
                            ([1, 2, 3j], np.ones((3, 3, 3))),
                            ([1, 2, 3], np.zeros((3, 3, 3))),
                            ([1, 2, 3], np.ones((3, 3, 3)) * np.nan)]:
            with self.assertRaises(ValueError): worker.frequencies_payload(freq, modes, 3)

    def test_nonpositive_frequencies_never_claim_verified_minimum(self):
        for freq in ([0, 2, 3], [-.1, 2, 3]):
            info = worker.frequencies_payload(freq, np.ones((3, 3, 3)), 3)
            self.assertFalse(worker.frequency_minimum_checks(info)["stationary_minimum_verified"])

    def test_hessian_symmetrization_keeps_raw_and_rejects_large_asymmetry(self):
        raw = np.zeros((3, 3, 3, 3));raw[0, 1, 0, 2] = 1e-7
        original = raw.copy();used, delta = worker.prepare_hessian_for_analysis(raw, 3)
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(used, used.transpose(1, 0, 3, 2))
        self.assertEqual(delta, 1e-7)
        raw[0, 1, 0, 2] = 1e-3
        with self.assertRaises(ValueError): worker.prepare_hessian_for_analysis(raw, 3)

    def test_fullprecision_coordinate_shape_hash_and_xyz_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "coords.npy";values = np.ones((3, 3)) + 1e-14
            worker.atomic_npy(path, values)
            np.testing.assert_array_equal(worker.precise_input_coordinates(np.ones((3, 3)), path, worker.sha(path)), values)
            with self.assertRaises(ValueError): worker.precise_input_coordinates(np.ones((3, 3)), path)
            with self.assertRaises(ValueError): worker.precise_input_coordinates(np.ones((3, 3)) + .001, path, worker.sha(path))

    def test_frequency_requires_same_v4_optimization_and_fullprecision_sha(self):
        with tempfile.TemporaryDirectory() as temp:
            p, coords = Path(temp) / "optimized.xyz", Path(temp) / "coords.npy"
            p.write_text("synthetic bytes only");worker.atomic_npy(coords, np.ones((3, 3)))
            settings = self.settings()
            result = dict(status="DONE", geometry_optimized=True, optimizer_converged=True,
                          scf_converged=True, final_gradient_thresholds_passed=True,
                          optimized_xyz_sha256=worker.sha(p), optimized_coordinates_angstrom_npy_sha256=worker.sha(coords),
                          settings=settings)
            worker.require_optimized_result(result, p, self.settings(threads=4), coords)
            changes = [dict(density_fitting=True, density_fit_backend="INCORE_DFJ_RELEASE_BEFORE_GRAD_V3"),
                       dict(direct_scf_tol=1e-8), dict(grid_level=2), dict(charge=1), dict(scf_conv_tol_hartree=1e-6)]
            for delta in changes:
                with self.assertRaises(ValueError):
                    worker.require_optimized_result(dict(result, settings={**settings, **delta}), p, settings, coords)
            with self.assertRaises(ValueError): worker.require_optimized_result(result, p, settings)
            result.pop("optimized_coordinates_angstrom_npy_sha256")
            with self.assertRaises(ValueError): worker.require_optimized_result(result, p, settings, coords)

    def test_unconverged_optimization_or_fresh_gradient_cannot_enter_frequency(self):
        with tempfile.TemporaryDirectory() as temp:
            p, coords = Path(temp) / "optimized.xyz", Path(temp) / "coords.npy"
            p.write_text("synthetic");worker.atomic_npy(coords, np.ones((3, 3)))
            result = dict(status="DONE", geometry_optimized=True, optimizer_converged=True,
                          scf_converged=True, final_gradient_thresholds_passed=True,
                          optimized_xyz_sha256=worker.sha(p), optimized_coordinates_angstrom_npy_sha256=worker.sha(coords),
                          settings=self.settings())
            for key, value in [("status", "PARTIAL"), ("geometry_optimized", False),
                               ("optimizer_converged", False), ("scf_converged", False),
                               ("final_gradient_thresholds_passed", False)]:
                with self.assertRaises(ValueError):
                    worker.require_optimized_result({**result, key: value}, p, self.settings(), coords)


if __name__ == "__main__":
    unittest.main()
