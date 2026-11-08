"""Pure engineering tests with synthetic arrays; no molecule or quantum calculation.

The reference is the frozen installed helper's AST, executed only in an isolated
namespace containing NumPy and synthetic integral/core/overlap callbacks. PySCF
is neither imported nor called. These tests are not scientific validation data.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/pyscf_spinosin_pure_j_worker_20261101.py"
RHF_SOURCE = ROOT / ".runtime/quantum-upgrade-site/pyscf/hessian/rhf.py"
FROZEN_RHF_SHA = "2d986bfc01aac98447034a207df639b3d6897f67ce1ab2a013fe2d0853c85de5"
FROZEN_V8_SHA = "acfe7708748ac237c9912aa9b1d98e047d842794fd0c6f08b1a67b993f5b277d"


def load_worker():
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location("pure_j_v9_under_pure_tests", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    new_quantum_modules = {key for key in set(sys.modules)-before if key == "pyscf" or key.startswith("pyscf.")}
    return module, new_quantum_modules


worker, IMPORTED_QUANTUM_MODULES = load_worker()


class FixtureLogger:
    """Clock/timer interface only, not a PySCF object."""
    @staticmethod
    def new_logger(*_args):
        return FixtureLogger()
    @staticmethod
    def process_clock():
        return 0.
    @staticmethod
    def perf_counter():
        return 0.
    @staticmethod
    def timer_debug1(_message, *values):
        return values
    @staticmethod
    def timer(_message, *values):
        return values


class SyntheticFixture:
    """Three nonphysical atom slices of unequal sizes exercise native contracts."""
    def __init__(self):
        self.calls = []
        self.ao_slices = np.array([[0, 1, 0, 1], [1, 2, 1, 3], [2, 3, 3, 4]])
        mol = SimpleNamespace(natm=3, nbas=3, spin=0, aoslice_by_atom=lambda: self.ao_slices.copy())
        ni = SimpleNamespace(
            rsh_and_hybrid_coeff=lambda _xc, spin=0: (0., 0., 0.),
            libxc=SimpleNamespace(is_hybrid_xc=lambda _xc: False),
            _xc_type=lambda _xc: "GGA")
        self.mf = SimpleNamespace(mol=mol, xc="PBE", _numint=ni, with_df=None,
            mo_coeff=np.array([[1., .2, .1, .3], [.1, .9, .4, .2], [.3, .1, .8, .4], [.2, .4, .2, .7]]),
            mo_energy=np.array([-1.1, -.35, .2, .8]), mo_occ=np.array([2., 2., 0., 0.]))
        self.hobj = SimpleNamespace(mol=mol, base=self.mf, hcore_generator=self.core_generator)
        self.rhf = SimpleNamespace(logger=FixtureLogger, _get_jk=self.fake_get_jk,
            _make_vhfopt=lambda *_args: None, get_ovlp=self.overlap,
            _vhf=SimpleNamespace(libcvhf=SimpleNamespace()))
        self.stock = self.make_stock_reference()
        self.rhf._partial_hess_ejk = self.stock

    @staticmethod
    def patterned(shape, base):
        return base + np.arange(int(np.prod(shape)), dtype=float).reshape(shape) / 137.

    def overlap(self, _mol):
        # Nonzero, distinct derivative tensors catch omitted overlap contributions.
        return (self.patterned((3, 3, 4, 4), .014),
                self.patterned((3, 3, 4, 4), .027),
                self.patterned((3, 4, 4), .031))

    def core_generator(self, _mol):
        def derivative(atom_i, atom_j):
            # Different atom-pair and Cartesian values catch lost e1/core terms.
            return self.patterned((3, 3, 4, 4), .17*(atom_i+1)+.07*(atom_j+1))
        return derivative

    def fake_get_jk(self, _mol, intor, comp, aosym, script_dms,
                    shls_slice=None, cintopt=None, vhfopt=None):
        self.calls.append({"intor": intor, "comp": comp, "aosym": aosym,
            "scripts": list(script_dms[::2]), "densities": [np.array(dm, copy=True) for dm in script_dms[1::2]],
            "shls_slice": shls_slice, "vhfopt": vhfopt, "cintopt": cintopt})
        self_size = 4
        if shls_slice is not None:
            atom = int(shls_slice[0]); self_size = int(self.ao_slices[atom, 3]-self.ao_slices[atom, 2])
        values = []
        for script, dm in zip(script_dms[::2], script_dms[1::2]):
            if intor == "int2e_ipip1":
                shape = (9, 4, 4)
            elif intor == "int2e_ip1ip2":
                shape = (9, 4, self_size) if script == "lj->s1ki" else (9, 4, 4)
            elif intor == "int2e_ipvip1":
                shape = (9, self_size, 4) if script == "lk->s1ij" else (9, 4, 4)
            else:
                raise AssertionError("Unexpected integral name in synthetic fixture: " + intor)
            # Values depend on this binding, never on the presence/number of K outputs.
            base = sum(map(ord, intor + script))/701. + np.sum(dm)/53.
            if shls_slice is not None:
                base += .03*shls_slice[0]
            values.append(self.patterned(shape, base))
        return values

    def make_stock_reference(self):
        if hashlib.sha256(RHF_SOURCE.read_bytes()).hexdigest() != FROZEN_RHF_SHA:
            raise AssertionError("Frozen PySCF source changed; re-audit required")
        tree = ast.parse(RHF_SOURCE.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_partial_hess_ejk")
        # Preserve the original arithmetic and scripts exactly; no original imports.
        isolated = ast.Module(body=[function], type_ignores=[])
        namespace = {"numpy": np, "logger": FixtureLogger, "_get_jk": self.fake_get_jk,
                     "_make_vhfopt": lambda *_args: None, "get_ovlp": self.overlap}
        exec(compile(isolated, "<synthetic-frozen-helper-arithmetic-not-a-quantum-job>", "exec"), namespace)
        return namespace["_partial_hess_ejk"]


class PureJArithmeticTests(unittest.TestCase):
    def compare_to_frozen_stock(self, atoms=None, overrides=False):
        reference = SyntheticFixture(); candidate = SyntheticFixture()
        kwargs = {"atmlst": atoms}
        if overrides:
            kwargs.update(mo_coeff=reference.mf.mo_coeff + .13,
                mo_energy=np.array([-1.4, -.2, .5, 1.]), mo_occ=np.array([2., 0., 2., 0.]))
        stock = reference.stock(reference.hobj, with_k=False, **kwargs)
        actual = worker.pure_j_partial_body(candidate.rhf, candidate.hobj, **kwargs)
        count = 3 if atoms is None else len(atoms)
        self.assertEqual(len(actual), 3)
        for array in actual:
            self.assertEqual(array.shape, (count, count, 3, 3))
            self.assertEqual(array.dtype, np.dtype("float64"))
            self.assertTrue(np.isfinite(array).all())
        np.testing.assert_allclose(actual[0], stock[0], atol=1e-12, rtol=1e-13)
        np.testing.assert_allclose(actual[1], stock[1], atol=1e-12, rtol=1e-13)
        np.testing.assert_array_equal(actual[2], np.zeros_like(stock[2]))
        self.assertGreater(np.max(np.abs(stock[0])), 1.)
        self.assertGreater(np.max(np.abs(stock[1])), 1.)
        self.assertGreater(np.max(np.abs(stock[2])), 1.)
        self.assertEqual(len(candidate.calls), 1 + 2*count)
        self.assertEqual(len(reference.calls), len(candidate.calls))
        for original_call, new_call in zip(reference.calls, candidate.calls):
            for key in ("intor", "comp", "aosym", "shls_slice"):
                self.assertEqual(new_call[key], original_call[key])
            self.assertEqual(len(new_call["scripts"]), 1)
            self.assertEqual(new_call["scripts"][0], original_call["scripts"][0])
            np.testing.assert_array_equal(new_call["densities"][0], original_call["densities"][0])
            self.assertIsNone(new_call["vhfopt"])
        for i in range(count):
            for j in range(i):
                np.testing.assert_array_equal(actual[0][i, j], actual[0][j, i].T)
                np.testing.assert_array_equal(actual[1][i, j], actual[1][j, i].T)
        return actual

    def test_import_does_not_import_pyscf_or_run_quantum_dependencies(self):
        self.assertEqual(IMPORTED_QUANTUM_MODULES, set())

    def test_full_J_and_nonzero_core_overlap_equal_frozen_arithmetic(self):
        actual = self.compare_to_frozen_stock()
        self.assertGreater(np.max(np.abs(actual[0][0, 2])), 1.)
        self.assertGreater(np.max(np.abs(actual[1][0, 2])), 1.)

    def test_atom_reordering_retains_input_order_and_pair_transposes(self):
        self.compare_to_frozen_stock([2, 0, 1])

    def test_unequal_AO_slice_and_single_atom_subset_preserve_native_contract(self):
        self.compare_to_frozen_stock([1])

    def test_explicit_MO_energy_coeff_occupation_overrides_are_honored(self):
        self.compare_to_frozen_stock([0, 2], overrides=True)


class PureJContextTests(unittest.TestCase):
    def test_true_branch_delegates_original_objects_and_exact_return(self):
        f = SyntheticFixture(); seen = []; sentinel = object()
        def original(*args, **kwargs):
            seen.append((args, kwargs)); return sentinel
        f.rhf._partial_hess_ejk = original
        first, second, verbose = object(), object(), object()
        with worker.PureJPartialContext(f.rhf):
            result = f.rhf._partial_hess_ejk(first, second, verbose=verbose, with_k=True)
        self.assertIs(result, sentinel)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0][0][0], first)
        self.assertIs(seen[0][0][1], second)
        self.assertIs(seen[0][1]["verbose"], verbose)
        self.assertIs(seen[0][1]["with_k"], True)
        self.assertIs(f.rhf._partial_hess_ejk, original)

    def test_default_true_branch_is_unchanged(self):
        f = SyntheticFixture(); sentinel = object()
        def original(*args, **kwargs):
            self.assertEqual(kwargs, {}); return sentinel
        f.rhf._partial_hess_ejk = original
        with worker.PureJPartialContext(f.rhf):
            self.assertIs(f.rhf._partial_hess_ejk(f.hobj), sentinel)
        self.assertIs(f.rhf._partial_hess_ejk, original)

    def test_false_context_matches_body_and_restores_after_normal_return(self):
        f = SyntheticFixture(); original = f.rhf._partial_hess_ejk
        expected = worker.pure_j_partial_body(f.rhf, f.hobj)
        with worker.PureJPartialContext(f.rhf):
            actual = f.rhf._partial_hess_ejk(f.hobj, with_k=False)
        for first, second in zip(actual, expected):
            np.testing.assert_array_equal(first, second)
        self.assertIs(f.rhf._partial_hess_ejk, original)

    def test_context_restores_original_when_block_raises(self):
        f = SyntheticFixture(); original = f.rhf._partial_hess_ejk
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            with worker.PureJPartialContext(f.rhf):
                raise RuntimeError("synthetic failure")
        self.assertIs(f.rhf._partial_hess_ejk, original)

    def test_non_PBE_false_path_is_rejected_before_integral_callbacks(self):
        f = SyntheticFixture(); f.mf.xc = "B3LYP"
        with self.assertRaises((ValueError, RuntimeError)):
            with worker.PureJPartialContext(f.rhf):
                f.rhf._partial_hess_ejk(f.hobj, with_k=False)
        self.assertEqual(f.calls, [])

    def test_derivative_optimizer_nonNone_ABI_is_rejected_without_enabling_suffix(self):
        f = SyntheticFixture()
        f.rhf._vhf.libcvhf.int2e_ip1ip2 = object()
        with self.assertRaises((ValueError, RuntimeError)):
            with worker.PureJPartialContext(f.rhf):
                f.rhf._partial_hess_ejk(f.hobj, with_k=False)
        self.assertEqual(f.calls, [])

    def test_all_three_bare_guards_are_checked_but_suffixes_are_never_enabled(self):
        for name in worker.BARE_DERIVATIVE_GUARDS:
            f = SyntheticFixture()
            setattr(f.rhf._vhf.libcvhf, name, object())
            with self.assertRaises((ValueError, RuntimeError)):
                worker.pure_j_partial_body(f.rhf, f.hobj)
            self.assertEqual(f.calls, [])
        f = SyntheticFixture()
        for name in worker.BARE_DERIVATIVE_GUARDS:
            setattr(f.rhf._vhf.libcvhf, name + "_sph", object())
            setattr(f.rhf._vhf.libcvhf, name + "_cart", object())
        worker.pure_j_partial_body(f.rhf, f.hobj)
        self.assertTrue(all(call["vhfopt"] is None for call in f.calls))

    def test_hybrid_coefficients_nonGGA_and_density_fitting_are_rejected(self):
        for case in ("hybrid", "range", "alpha", "hyb", "nonGGA", "DF"):
            f = SyntheticFixture()
            if case == "hybrid":
                f.mf._numint.libxc.is_hybrid_xc = lambda _xc: True
            elif case == "range":
                f.mf._numint.rsh_and_hybrid_coeff = lambda _xc, spin=0: (.2, 0., 0.)
            elif case == "alpha":
                f.mf._numint.rsh_and_hybrid_coeff = lambda _xc, spin=0: (0., .2, 0.)
            elif case == "hyb":
                f.mf._numint.rsh_and_hybrid_coeff = lambda _xc, spin=0: (0., 0., .2)
            elif case == "nonGGA":
                f.mf._numint._xc_type = lambda _xc: "LDA"
            else:
                f.mf.with_df = object()
            with self.assertRaises((ValueError, RuntimeError)):
                worker.pure_j_partial_body(f.rhf, f.hobj)
            self.assertEqual(f.calls, [], case)

    def test_nested_context_cannot_replace_original_or_claim_two_owners(self):
        f = SyntheticFixture(); original = f.rhf._partial_hess_ejk
        with worker.PureJPartialContext(f.rhf):
            active = f.rhf._partial_hess_ejk
            with self.assertRaisesRegex(RuntimeError, "nest"):
                with worker.PureJPartialContext(f.rhf):
                    self.fail("Nested ownership must be rejected")
            self.assertIs(f.rhf._partial_hess_ejk, active)
        self.assertIs(f.rhf._partial_hess_ejk, original)

    def test_same_context_instance_reentry_does_not_corrupt_original_pointer(self):
        f = SyntheticFixture(); original = f.rhf._partial_hess_ejk
        context = worker.PureJPartialContext(f.rhf)
        with context:
            active = f.rhf._partial_hess_ejk
            with self.assertRaisesRegex(RuntimeError, "nest"):
                with context:
                    self.fail("Same active context must reject re-entry")
            self.assertIs(f.rhf._partial_hess_ejk, active)
            self.assertTrue(context.active)
        self.assertIs(f.rhf._partial_hess_ejk, original)
        self.assertFalse(context.active)
        with context:
            self.assertIsNot(f.rhf._partial_hess_ejk, original)
        self.assertIs(f.rhf._partial_hess_ejk, original)

    def test_preexisting_pure_wrapper_is_not_replaced_on_failed_entry(self):
        f = SyntheticFixture()
        def existing(*_args, **_kwargs):
            return "synthetic preexisting wrapper"
        existing._research4_pure_j = True
        f.rhf._partial_hess_ejk = existing
        with self.assertRaisesRegex(RuntimeError, "nest"):
            with worker.PureJPartialContext(f.rhf):
                self.fail("Existing pure wrapper must reject entry")
        self.assertIs(f.rhf._partial_hess_ejk, existing)

    def test_unexpected_writer_is_reported_and_original_is_restored(self):
        f = SyntheticFixture(); original = f.rhf._partial_hess_ejk
        with self.assertRaisesRegex(RuntimeError, "Another writer"):
            with worker.PureJPartialContext(f.rhf):
                f.rhf._partial_hess_ejk = lambda *_args, **_kwargs: "unreviewed synthetic writer"
        self.assertIs(f.rhf._partial_hess_ejk, original)
        with self.assertRaisesRegex(OSError, "original synthetic error"):
            with worker.PureJPartialContext(f.rhf):
                f.rhf._partial_hess_ejk = lambda *_args, **_kwargs: "unreviewed synthetic writer"
                raise OSError("original synthetic error")
        self.assertIs(f.rhf._partial_hess_ejk, original)

    def test_callback_gets_only_successfully_returned_RHF_tuple_before_XC(self):
        f = SyntheticFixture(); seen = []
        def callback(hobj, parts, record):
            seen.append((hobj, parts, record))
        with worker.PureJPartialContext(f.rhf, callback=callback):
            parts = f.rhf._partial_hess_ejk(f.hobj, with_k=False)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0][0], f.hobj)
        self.assertIs(seen[0][1], parts)
        self.assertEqual(seen[0][2]["scope"], "ONLY_RHF_PARTIAL_BEFORE_RKS_XC")
        self.assertTrue(seen[0][2]["zero_ek_contract"])
        f = SyntheticFixture(); seen.clear()
        f.rhf._get_jk = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic contraction failure"))
        with self.assertRaisesRegex(RuntimeError, "synthetic contraction failure"):
            with worker.PureJPartialContext(f.rhf, callback=callback):
                f.rhf._partial_hess_ejk(f.hobj, with_k=False)
        self.assertEqual(seen, [])

    def test_false_positional_with_k_argument_uses_original_signature(self):
        f = SyntheticFixture(); expected = worker.pure_j_partial_body(f.rhf, f.hobj)
        with worker.PureJPartialContext(f.rhf):
            actual = f.rhf._partial_hess_ejk(f.hobj, f.mf.mo_energy, f.mf.mo_coeff,
                f.mf.mo_occ, None, 4000, None, False)
        for first, second in zip(expected, actual):
            np.testing.assert_array_equal(first, second)

    def test_true_branch_never_invokes_pure_J_cache_callback(self):
        f = SyntheticFixture(); seen = []
        def original(*_args, **_kwargs):
            return "synthetic untouched true branch"
        f.rhf._partial_hess_ejk = original
        with worker.PureJPartialContext(f.rhf, callback=lambda *_args: seen.append("wrong")):
            self.assertEqual(f.rhf._partial_hess_ejk(f.hobj, with_k=True), "synthetic untouched true branch")
        self.assertEqual(seen, [])


class SyntheticCaptureEngine:
    """A tiny standalone save/binding interface, not an electronic-structure engine."""
    def __init__(self, root):
        self.ROOT = root
        self.binding = {"synthetic_nonphysical_fixture": True, "geometry_revision": 0}
        self.fail_on = self.mutate_after = None
        self.saved = []

    def hessian_cache_binding(self, _mf, _settings, _versions, _input_evidence, hobj=None):
        return json.loads(json.dumps(self.binding))

    @staticmethod
    def require_full_atoms(atoms, atom_count):
        values = list(range(atom_count)) if atoms is None else list(atoms)
        if values != list(range(atom_count)):
            raise ValueError("Full global atom order required")
        return values

    def atomic_npy(self, path, array):
        if path.name == self.fail_on:
            raise OSError("synthetic disk write failure")
        temporary = path.with_name(path.name + ".writing")
        with temporary.open("wb") as f:
            np.save(f, array, allow_pickle=False)
        temporary.replace(path)
        self.saved.append(path.name)
        if path.name == self.mutate_after:
            self.binding["geometry_revision"] += 1

    @staticmethod
    def array_fingerprint(array):
        array = np.ascontiguousarray(array)
        return {"shape": list(array.shape), "dtype": str(array.dtype),
                "payload_sha256_C_order": hashlib.sha256(memoryview(array).cast("B")).hexdigest()}


class PureJRHFCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="pure-j-cache-synthetic-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "outputs").mkdir()
        self.fixture = SyntheticFixture()
        self.engine = SyntheticCaptureEngine(self.root)
        self.counter = 0

    def make_capture(self, validate=True, **kwargs):
        self.counter += 1
        cache = self.root / "outputs" / ("cache" + str(self.counter))
        manifest = self.root / ("manifest" + str(self.counter) + ".json")
        capture = worker.RHFPartsCapture(self.engine, self.fixture.hobj, cache, manifest,
            {"synthetic": True}, {"fixture_only": True}, {"not_a_plant_sample": True}, **kwargs)
        if validate:
            capture.validate_helper_arguments(self.fixture.hobj)
        return capture, cache, manifest

    @staticmethod
    def parts():
        shape = (3, 3, 3, 3)
        return (np.arange(81, dtype=float).reshape(shape) + 1.,
                np.arange(81, dtype=float).reshape(shape) + 4., np.zeros(shape))

    @staticmethod
    def record(atoms=None):
        return {"scope": "ONLY_RHF_PARTIAL_BEFORE_RKS_XC", "with_k": False,
                "atmlst": atoms, "zero_ek_contract": True}

    def test_complete_helper_saved_before_later_XC_failure_and_never_called_full_Hessian(self):
        capture, cache, manifest_path = self.make_capture()
        with patch.object(worker.np, "load", side_effect=AssertionError("SAVE_ONLY must not load any arrays")):
            with self.assertRaisesRegex(RuntimeError, "synthetic later XC failure"):
                with worker.PureJPartialContext(self.fixture.rhf, callback=capture):
                    actual = self.fixture.rhf._partial_hess_ejk(self.fixture.hobj, with_k=False)
                    manifest = json.loads(manifest_path.read_text())
                    self.assertEqual(manifest["status"], "DONE_RHF_HELPER_ONLY_BEFORE_RKS_XC")
                    raise RuntimeError("synthetic later XC failure")
        manifest = json.loads(manifest_path.read_text())
        self.assertFalse(manifest["complete_RKS_partial_or_full_Hessian"])
        self.assertEqual(manifest["scope"], "ONLY_RHF_PARTIAL_BEFORE_RKS_XC")
        self.assertEqual(manifest["cross_process_restore"], "NOT_IMPLEMENTED_SAVE_ONLY")
        self.assertFalse(manifest["resume_command_created"])
        self.assertEqual(set(manifest["arrays"]), {"e1", "ej", "ek"})
        self.assertEqual(Path(manifest["actual_entrypoint"]["path"]), SCRIPT)
        self.assertEqual(manifest["actual_entrypoint"]["sha256"], worker.sha(SCRIPT))
        for name, value in zip(("e1", "ej", "ek"), actual):
            path = cache / (name + ".npy")
            self.assertEqual(manifest["arrays"][name]["file_sha256"], worker.sha(path))
            np.testing.assert_array_equal(np.load(path, allow_pickle=False), value)

    def test_incomplete_nonfinite_nonfloat_and_nonzero_ek_refused_before_save(self):
        for failure in ("shape", "nan", "dtype", "ek", "list"):
            capture, cache, manifest = self.make_capture()
            values = list(self.parts())
            if failure == "shape":
                values[1] = np.zeros((2, 3, 3, 3))
            elif failure == "nan":
                values[1][0, 0, 0, 0] = np.nan
            elif failure == "dtype":
                values[1] = values[1].astype(np.float32)
            elif failure == "ek":
                values[2][0, 0, 0, 0] = 1.
            values = values if failure == "list" else tuple(values)
            with self.assertRaises(ValueError, msg=failure):
                capture(self.fixture.hobj, values, self.record())
            self.assertEqual(list(cache.iterdir()), [], failure)
            self.assertNotEqual(json.loads(manifest.read_text())["status"], "DONE_RHF_HELPER_ONLY_BEFORE_RKS_XC")

    def test_changed_geometry_binding_or_wrong_hobj_refused_before_any_write(self):
        capture, cache, _manifest = self.make_capture()
        self.engine.binding["geometry_revision"] = 1
        with self.assertRaisesRegex(RuntimeError, "binding changed"):
            capture(self.fixture.hobj, self.parts(), self.record())
        self.assertEqual(list(cache.iterdir()), [])
        self.engine.binding["geometry_revision"] = 0
        with self.assertRaisesRegex(RuntimeError, "binding changed"):
            capture(SimpleNamespace(), self.parts(), self.record())
        self.assertEqual(list(cache.iterdir()), [])

    def test_non_global_or_reordered_capture_atoms_are_rejected(self):
        for atoms in ([0, 1], [1, 0, 2]):
            capture, cache, _manifest = self.make_capture()
            with self.assertRaisesRegex(ValueError, "Full global atom order"):
                capture(self.fixture.hobj, self.parts(), self.record(atoms))
            self.assertEqual(list(cache.iterdir()), [])

    def test_duplicate_capture_preserves_completed_bytes_and_is_not_resume(self):
        capture, cache, manifest = self.make_capture()
        capture(self.fixture.hobj, self.parts(), self.record())
        before = {f.name: f.read_bytes() for f in cache.iterdir()}
        with self.assertRaisesRegex(RuntimeError, "overwrite|duplicate"):
            capture(self.fixture.hobj, self.parts(), self.record())
        self.assertEqual({f.name: f.read_bytes() for f in cache.iterdir()}, before)
        with self.assertRaises(FileExistsError):
            worker.RHFPartsCapture(self.engine, self.fixture.hobj, cache, manifest,
                                  {}, {}, {})

    def test_disk_failure_retains_partial_status_without_completed_tuple(self):
        capture, cache, manifest = self.make_capture()
        self.engine.fail_on = "ej.npy"
        with self.assertRaisesRegex(OSError, "synthetic disk write failure"):
            capture(self.fixture.hobj, self.parts(), self.record())
        data = json.loads(manifest.read_text())
        self.assertEqual(data["status"], "PARTIAL_RHF_CACHE_NOT_VALIDATED")
        self.assertIsNone(capture.parts)
        self.assertEqual(set(data["arrays"]), {"e1"})
        self.assertTrue((cache / "e1.npy").is_file())
        self.assertFalse(data["complete_RKS_partial_or_full_Hessian"])

    def test_binding_change_during_save_cannot_become_complete(self):
        capture, _cache, manifest = self.make_capture()
        self.engine.mutate_after = "e1.npy"
        with self.assertRaisesRegex(RuntimeError, "binding changed while saving"):
            capture(self.fixture.hobj, self.parts(), self.record())
        self.assertEqual(json.loads(manifest.read_text())["status"], "PARTIAL_RHF_CACHE_NOT_VALIDATED")
        self.assertIsNone(capture.parts)

    def test_stock_reference_capture_can_preserve_nonzero_ek_without_claiming_pure_J(self):
        capture, _cache, manifest = self.make_capture(require_zero_ek=False)
        values = list(self.parts()); values[2][0, 0, 0, 0] = .5
        capture(self.fixture.hobj, tuple(values), self.record())
        meta = json.loads(manifest.read_text())
        self.assertFalse(meta["zero_ek_required"])
        self.assertFalse(meta["engineering_changes"]["unused_K_shaped_contraction_scripts_and_density_bindings_removed"])
        self.assertTrue(meta["engineering_changes"]["reference_case_unmodified_stock_partial"])

    def test_capture_requires_actual_helper_MO_arguments_to_be_validated(self):
        capture, cache, _manifest = self.make_capture(validate=False)
        with self.assertRaisesRegex(RuntimeError, "MO arguments must be validated"):
            capture(self.fixture.hobj, self.parts(), self.record())
        self.assertEqual(list(cache.iterdir()), [])

    def test_equal_MO_copies_are_allowed_and_recorded_as_equal_bound_MF(self):
        capture, _cache, manifest = self.make_capture(validate=False)
        mf = self.fixture.mf
        capture.validate_helper_arguments(self.fixture.hobj, mf.mo_energy.copy(), mf.mo_coeff.copy(), mf.mo_occ.copy())
        capture(self.fixture.hobj, self.parts(), self.record())
        self.assertTrue(json.loads(manifest.read_text())["actual_helper_MO_arguments_verified_equal_bound_MF"])

    def test_nonbase_MO_overrides_rejected_before_pure_or_stock_callbacks(self):
        for use_pure in (False, True):
            for field in ("mo_energy", "mo_coeff", "mo_occ"):
                capture, cache, _manifest = self.make_capture(require_zero_ek=use_pure)
                override = getattr(self.fixture.mf, field).copy(); override.flat[0] += .125
                context = (worker.PureJPartialContext(self.fixture.rhf, callback=capture) if use_pure
                           else worker.stock_partial_capture_context(self.fixture.rhf, capture))
                self.fixture.calls.clear()
                with self.assertRaisesRegex(ValueError, "MO arguments differ"):
                    with context:
                        self.fixture.rhf._partial_hess_ejk(self.fixture.hobj, with_k=False, **{field: override})
                self.assertEqual(self.fixture.calls, [], (field, use_pure))
                self.assertEqual(list(cache.iterdir()), [])

    def test_preexisting_array_bytes_are_preserved_and_cache_is_not_marked_complete(self):
        for filename in ("e1.npy", "ej.npy", "ek.npy"):
            capture, cache, manifest = self.make_capture()
            existing = cache / filename; payload = b"synthetic existing artifact; must never be overwritten"
            existing.write_bytes(payload)
            with self.assertRaisesRegex(FileExistsError, "existing RHF helper array"):
                capture(self.fixture.hobj, self.parts(), self.record())
            self.assertEqual(existing.read_bytes(), payload)
            self.assertEqual(json.loads(manifest.read_text())["status"], "PARTIAL_RHF_CACHE_NOT_VALIDATED")
            self.assertIsNone(capture.parts)

    def test_equal_numeric_MO_values_with_different_dtype_are_rejected(self):
        for use_pure in (False, True):
            capture, cache, _manifest = self.make_capture(require_zero_ek=use_pure)
            cast = self.fixture.mf.mo_occ.astype(np.float32)
            self.assertTrue(np.array_equal(cast, self.fixture.mf.mo_occ))
            self.assertNotEqual(cast.dtype, self.fixture.mf.mo_occ.dtype)
            context = (worker.PureJPartialContext(self.fixture.rhf, capture) if use_pure
                       else worker.stock_partial_capture_context(self.fixture.rhf, capture))
            self.fixture.calls.clear()
            with self.assertRaisesRegex(ValueError, "MO arguments differ"):
                with context:
                    self.fixture.rhf._partial_hess_ejk(self.fixture.hobj, mo_occ=cast, with_k=False)
            self.assertEqual(self.fixture.calls, [])
            self.assertEqual(list(cache.iterdir()), [])

    def test_outer_failure_before_helper_return_stays_partial_without_arrays(self):
        capture, cache, manifest = self.make_capture()
        capture.mark_outer_failure(RuntimeError("synthetic helper not returned"))
        meta = json.loads(manifest.read_text())
        self.assertEqual(meta["status"], "PARTIAL_RHF_HELPER_NOT_RETURNED")
        self.assertEqual(meta["outer_Hessian_status"], "PARTIAL")
        self.assertEqual(meta["outer_exception_type"], "RuntimeError")
        self.assertEqual(meta["arrays"], {})
        self.assertFalse(meta["complete_RKS_partial_or_full_Hessian"])
        self.assertEqual(list(cache.iterdir()), [])
        self.assertIsNone(capture.parts)

    def test_outer_XC_failure_preserves_completed_RHF_tuple_without_full_Hessian_claim(self):
        capture, cache, manifest = self.make_capture()
        actual = self.parts(); capture(self.fixture.hobj, actual, self.record())
        before = {f.name: f.read_bytes() for f in cache.iterdir()}
        capture.mark_outer_failure(RuntimeError("synthetic RKS XC interrupted later"))
        meta = json.loads(manifest.read_text())
        self.assertEqual(meta["status"], "DONE_RHF_HELPER_ONLY_BEFORE_RKS_XC")
        self.assertEqual(meta["outer_Hessian_status"], "PARTIAL")
        self.assertFalse(meta["complete_RKS_partial_or_full_Hessian"])
        self.assertEqual(meta["cross_process_restore"], "NOT_IMPLEMENTED_SAVE_ONLY")
        self.assertEqual({f.name: f.read_bytes() for f in cache.iterdir()}, before)
        for original, saved in zip(actual, capture.parts):
            np.testing.assert_array_equal(original, saved)


class PureJComparisonGateTests(unittest.TestCase):
    @staticmethod
    def case(atoms=9):
        shape = (atoms, atoms, 3, 3)
        return {"energy_hartree": -1., "gradient": np.zeros((atoms, 3)),
            "e1": np.zeros(shape), "ej": np.zeros(shape), "raw_hessian": np.zeros(shape),
            "frequencies_cm1": np.arange(3*atoms-6, dtype=float)+1.}

    def test_water3_and_ethanol21_actual_mode_count_shapes_are_required(self):
        for atoms, modes in ((3, 3), (9, 21)):
            ref, actual = self.case(atoms), self.case(atoms)
            comparison = worker.compare_complete_cases(ref, actual, atoms)
            self.assertTrue(comparison["numerical_equivalence_passed"])
            self.assertTrue(comparison["all_atom_pair_cross_couplings_compared"])
            self.assertEqual(comparison["all_actual_modes_compared"], modes)
            actual["frequencies_cm1"] = actual["frequencies_cm1"][:-1]
            with self.assertRaises(ValueError):
                worker.compare_complete_cases(ref, actual, atoms)

    def test_each_fixed_numeric_gate_can_fail_independently(self):
        for field, tolerance_key in (("energy_hartree", "energy_hartree"),
                ("gradient", "gradient_hartree_per_bohr"),
                ("e1", "RHF_e1_ej_hartree_per_bohr2"),
                ("ej", "RHF_e1_ej_hartree_per_bohr2"),
                ("raw_hessian", "raw_hessian_hartree_per_bohr2"),
                ("frequencies_cm1", "frequencies_cm1")):
            ref, actual = self.case(), self.case()
            value = 2*worker.TOLERANCES[tolerance_key]
            if field == "energy_hartree":
                actual[field] += value
            else:
                actual[field].flat[-1] += value
            comparison = worker.compare_complete_cases(ref, actual, 9)
            self.assertFalse(comparison["numerical_equivalence_passed"], field)
            self.assertFalse(comparison["differences"][field]["passed"], field)

    def test_cross_block_offdiagonal_failure_is_not_hidden_by_diagonal_checks(self):
        ref, actual = self.case(), self.case()
        actual["raw_hessian"][0, 8, 1, 2] = 2e-6
        comparison = worker.compare_complete_cases(ref, actual, 9)
        self.assertFalse(comparison["numerical_equivalence_passed"])
        self.assertEqual(comparison["differences"]["raw_hessian"]["max_abs_difference"], 2e-6)

    def test_negative_or_zero_frequencies_do_not_masquerade_as_positive_minima(self):
        for frequency in (-2., 0.):
            ref, actual = self.case(), self.case()
            ref["frequencies_cm1"][0] = actual["frequencies_cm1"][0] = frequency
            comparison = worker.compare_complete_cases(ref, actual, 9)
            self.assertTrue(comparison["numerical_equivalence_passed"])
            self.assertFalse(comparison["both_minima_positive"])
            self.assertEqual(comparison["positive_frequency_minima"], {"stock": False, "pure_j": False})
            self.assertEqual(comparison["IR_intensities_status"], "NOT_COMPUTED")

    def test_nonfinite_complex_and_incomplete_arrays_cannot_pass(self):
        for field in ("gradient", "e1", "ej", "raw_hessian", "frequencies_cm1"):
            for failure in ("nan", "complex", "shape"):
                ref, actual = self.case(), self.case()
                if failure == "nan":
                    actual[field].flat[0] = np.nan
                elif failure == "complex":
                    actual[field] = actual[field].astype(complex)
                else:
                    actual[field] = actual[field][:-1]
                with self.assertRaises(ValueError, msg=(field, failure)):
                    worker.compare_complete_cases(ref, actual, 9)


class PureJProvenanceAdapterTests(unittest.TestCase):
    def test_adapter_records_actual_new_entry_and_preserves_real_V8_file_identity(self):
        written = []
        engine = SimpleNamespace(__file__=str(worker.V8_PATH),
            worker_settings=lambda _args: {"xc": "PBE", "engineering_worker_version": "original_v8"},
            write=lambda path, value: written.append((path, value)),
            run_checkpointed_hessian=lambda *_args, **_kwargs: None)
        worker.configure_engine(engine)
        self.assertEqual(engine.__file__, str(worker.V8_PATH))
        settings = engine.worker_settings(SimpleNamespace())
        self.assertEqual(settings["actual_entrypoint"]["path"], str(SCRIPT))
        self.assertEqual(settings["actual_entrypoint"]["sha256"], worker.sha(SCRIPT))
        self.assertEqual(settings["original_V8_source_sha256"], FROZEN_V8_SHA)
        self.assertTrue(settings["engineering_changes"]["unused_K_shaped_contraction_scripts_and_density_bindings_removed"])
        self.assertFalse(settings["engineering_changes"]["original_PySCF_files_modified"])
        state = {"settings": settings, "phase": "synthetic_no_quantum", "script_sha256": FROZEN_V8_SHA}
        engine.write(Path("not_written_fixture.json"), state)
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0][1]["script_sha256"], FROZEN_V8_SHA)
        self.assertEqual(written[0][1]["actual_entrypoint"]["sha256"], worker.sha(SCRIPT))
        self.assertIn("NOT_THIS_ENTRYPOINT", written[0][1]["original_script_sha256_field_scope"])
        self.assertEqual(engine.verify_resource_supervision, worker.verify_resource_supervision)

    def test_wrong_V8_bytes_rejected_before_module_loading(self):
        with tempfile.TemporaryDirectory(prefix="pure-j-source-synthetic-") as path:
            file = Path(path) / "v8.py"; file.write_text("# synthetic changed source; never executed\n")
            with patch.object(worker, "V8_PATH", file), patch.object(worker, "load_module", side_effect=AssertionError("Must not load changed source")):
                with self.assertRaisesRegex(ValueError, "Frozen V8"):
                    worker.load_frozen_v8()

    def test_new_supervision_adapter_refuses_old_schema_and_source_mismatch(self):
        with tempfile.TemporaryDirectory(prefix="pure-j-v4-adapter-synthetic-") as path:
            file = Path(path) / "supervisor.py"; file.write_text("# synthetic verifier source, never executed\n")
            for bad in ("schema", "sha"):
                proof = {"schema": "research4-pyscf-actual-cpu-supervisor-proof-v4",
                         "supervisor_script_sha256": worker.sha(file)}
                if bad == "schema":
                    proof["schema"] = "research4-pyscf-resource-supervisor-proof-v3"
                else:
                    proof["supervisor_script_sha256"] = "0"*64
                returned = {"proof": proof}
                fake = SimpleNamespace(verify_resource_supervision=lambda *_args, **_kwargs: returned)
                with patch.object(worker, "SUPERVISOR_PATH", file), patch.object(worker, "load_module", return_value=fake):
                    with self.assertRaises(ValueError):
                        worker.verify_resource_supervision(Path("unread_fixture_proof.json"), expected_threads=4)

    def test_actual_V4_verifier_result_is_forwarded_without_fabrication(self):
        with tempfile.TemporaryDirectory(prefix="pure-j-v4-adapter-synthetic-") as path:
            file = Path(path) / "supervisor.py"; file.write_text("# synthetic verifier source, never executed\n")
            seen = []; sentinel = {"proof": {"schema": "research4-pyscf-actual-cpu-supervisor-proof-v4",
                "supervisor_script_sha256": worker.sha(file)}, "synthetic_verified_result": True}
            def verify(proof_path=None, *, expected_threads=None):
                seen.append((proof_path, expected_threads)); return sentinel
            with patch.object(worker, "SUPERVISOR_PATH", file), patch.object(worker, "load_module", return_value=SimpleNamespace(verify_resource_supervision=verify)):
                result = worker.verify_resource_supervision(Path("unread_fixture_proof.json"), expected_threads=4)
            self.assertIs(result, sentinel)
            self.assertEqual(seen, [(Path("unread_fixture_proof.json"), 4)])


class ClosedValidationGeometryTests(unittest.TestCase):
    """Temporary arbitrary coordinates test I/O contracts, never real geometry."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="pure-j-closed-contract-synthetic-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.counter = 0

    def make_case(self, identity="water", fallback=False):
        self.counter += 1; path = self.root / str(self.counter); path.mkdir()
        if identity == "water":
            symbols = ["O", "H", "H"]
            coordinates = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        else:
            symbols = ["C", "C", "O"] + ["H"]*6
            coordinates = np.array([[0., 0., 0.], [1.1, 0., 0.], [2.2, .2, 0.],
                [0., 1., 0.], [0., 0., 1.], [1., 1., 0.], [1., 0., 1.], [2., 1., 0.], [2., 0., 1.]])
        settings = {"xc": "PBE", "basis": "def2-svp", "grid_level": 1,
            "density_fitting": False, "charge": 0, "multiplicity": 1,
            "dispersion": "NONE_EXPLICIT_PBE_PILOT_NOT_D3"}
        old = {"synthetic_nonphysical_test_fixture": True, "status": "DONE", "identity": identity,
            "geometry_optimized": True, "optimizer_converged": True, "scf_converged": True,
            "final_gradient_thresholds_passed": True, "settings": settings,
            "input_xyz_sha256": "f"*64}
        if identity == "validation_ethanol":
            old["source_provenance"] = {"identity_verified": True, "stereochemistry_verified": True,
                "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N", "formula": "C2H6O", "atom_count": 9,
                "input_xyz_sha256": old["input_xyz_sha256"], "source_url": "https://example.invalid/synthetic-test-not-a-source"}
        case = SimpleNamespace(identity=identity, symbols=symbols, coordinates=coordinates,
            opt_path=path/"closed_result.json", xyz=path/"closed.xyz", npy=path/"closed.npy",
            record_path=path/"case_record.json", opt=old, fallback=fallback)
        self.write_coordinates(case)
        return case

    def write_coordinates(self, case):
        np.save(case.npy, case.coordinates, allow_pickle=False)
        lines = [str(len(case.symbols)), "SYNTHETIC I/O CONTRACT FIXTURE; NOT A PHYSICAL MOLECULE OR OPT RESULT"]
        lines += [symbol+" "+" ".join(f"{v:.15f}" for v in xyz) for symbol, xyz in zip(case.symbols,case.coordinates)]
        case.xyz.write_text("\n".join(lines)+"\n")
        self.rebind(case)

    def rebind(self, case):
        case.xyz_sha = worker.sha(case.xyz); case.npy_sha = worker.sha(case.npy)
        if not case.fallback:
            case.opt["optimized_coordinates_angstrom_npy_sha256"] = case.npy_sha
        case.record = {"identity": case.identity, "validation_only": True,
            "normal_mode_geometry_source_sha256": case.npy_sha, "synthetic_nonphysical_test_fixture": True}
        case.opt_path.write_text(json.dumps(case.opt))
        case.record_path.write_text(json.dumps(case.record))

    def verify(self, case, **kwargs):
        options = {"identity": case.identity, "optimization_result_path": case.opt_path,
            "xyz_path": case.xyz, "coordinates_path": case.npy,
            "expected_xyz_sha": case.xyz_sha, "expected_coordinates_sha": case.npy_sha,
            "geometry_record_path": case.record_path if case.fallback else None}
        options.update(kwargs)
        return worker.verify_closed_validation_source(**options)

    def test_closed_water_geometry_uses_bound_optimized_coordinates_no_new_Opt(self):
        case = self.make_case()
        symbols, coordinates, evidence = self.verify(case)
        self.assertEqual(symbols, case.symbols)
        np.testing.assert_array_equal(coordinates, case.coordinates)
        self.assertTrue(evidence["prior_optimization_verified"])
        self.assertFalse(evidence["new_geometry_optimization_performed"])
        self.assertTrue(evidence["validation_only"])
        self.assertTrue(evidence["not_a_plant_or_acid_jujube_sample"])
        self.assertIsNone(evidence["closed_geometry_record_sha256"])

    def test_ethanol_legacy_top_record_requires_closed_case_geometry_binding(self):
        case = self.make_case("validation_ethanol", fallback=True)
        _symbols, coordinates, evidence = self.verify(case)
        np.testing.assert_array_equal(coordinates, case.coordinates)
        self.assertEqual(evidence["closed_geometry_record_sha256"], worker.sha(case.record_path))
        self.assertEqual(evidence["original_graph_provenance"]["inchikey"], "LFQSCWFLJHTTHZ-UHFFFAOYSA-N")
        with self.assertRaisesRegex(ValueError, "SHA-binding geometry record"):
            self.verify(case, geometry_record_path=None)

    def test_initial_MMFF_provenance_is_not_an_optimized_geometry_record(self):
        case = self.make_case("validation_ethanol", fallback=True)
        case.record = {"identity": "validation_ethanol", "validation_only": True,
            "coordinates_sha256": case.npy_sha, "input_xyz_sha256": case.opt["input_xyz_sha256"],
            "geometry_origin": "INITIAL_MMFF_MODEL_NOT_OPTIMIZED"}
        case.record_path.write_text(json.dumps(case.record))
        with self.assertRaisesRegex(ValueError, "optimized geometry record"):
            self.verify(case)

    def test_closed_source_identity_and_success_flags_are_mandatory(self):
        for key in ("geometry_optimized", "optimizer_converged", "scf_converged", "final_gradient_thresholds_passed"):
            case = self.make_case(); case.opt[key] = False; case.opt_path.write_text(json.dumps(case.opt))
            with self.assertRaisesRegex(ValueError, "closed successful"):
                self.verify(case)
        case = self.make_case(); case.opt["status"] = "PARTIAL"; case.opt_path.write_text(json.dumps(case.opt))
        with self.assertRaisesRegex(ValueError, "closed successful"):
            self.verify(case)
        case = self.make_case(); case.opt["identity"] = "spinosin"; case.opt_path.write_text(json.dumps(case.opt))
        with self.assertRaisesRegex(ValueError, "identity"):
            self.verify(case)
        with self.assertRaisesRegex(ValueError, "Only sourced water"):
            self.verify(case, identity="spinosin")

    def test_any_different_physical_model_is_rejected(self):
        wrong = {"xc": "B3LYP", "basis": "sto-3g", "grid_level": 2, "density_fitting": True,
                 "charge": 1, "multiplicity": 3, "dispersion": "D3BJ"}
        for key, value in wrong.items():
            case = self.make_case(); case.opt["settings"][key] = value; case.opt_path.write_text(json.dumps(case.opt))
            with self.assertRaisesRegex(ValueError, "different physical model"):
                self.verify(case)

    def test_ethanol_requires_prior_verified_stereochemical_graph(self):
        for key, value in (("identity_verified", False), ("stereochemistry_verified", False),
                ("inchikey", "wrong"), ("formula", "C3H8O"), ("atom_count", 10), ("input_xyz_sha256", "0"*64)):
            case = self.make_case("validation_ethanol", fallback=True)
            case.opt["source_provenance"][key] = value; case.opt_path.write_text(json.dumps(case.opt))
            with self.assertRaisesRegex(ValueError, "verified original graph provenance"):
                self.verify(case)

    def test_closed_case_identity_validation_flag_and_SHA_must_match(self):
        for key, value in (("identity", "water"), ("validation_only", False),
                           ("normal_mode_geometry_source_sha256", "0"*64)):
            case = self.make_case("validation_ethanol", fallback=True)
            case.record[key] = value; case.record_path.write_text(json.dumps(case.record))
            with self.assertRaises(ValueError):
                self.verify(case)

    def test_hash_mismatch_rejected_before_loading_coordinate_array(self):
        case = self.make_case()
        for key in ("expected_xyz_sha", "expected_coordinates_sha"):
            with patch.object(worker.np, "load", side_effect=AssertionError("Must reject digest before array access")):
                with self.assertRaisesRegex(ValueError, "SHA must match"):
                    self.verify(case, **{key: "0"*64})

    def test_finite_coordinates_shape_composition_and_XYZ_precision_are_checked(self):
        for bad in ("shape", "nan", "composition", "mismatch"):
            case = self.make_case()
            if bad == "shape":
                np.save(case.npy, np.zeros((3, 2)), allow_pickle=False)
            elif bad == "nan":
                case.coordinates[0, 0] = np.nan; self.write_coordinates(case)
            elif bad == "composition":
                case.symbols[0] = "C"; self.write_coordinates(case)
            else:
                changed = case.coordinates.copy(); changed[0, 0] += .01
                np.save(case.npy, changed, allow_pickle=False)
            self.rebind(case)
            with self.assertRaisesRegex(ValueError, "composition or full-precision"):
                self.verify(case)

    def test_overlap_or_linear_closed_input_cannot_claim_nonlinear_modes(self):
        for bad in ("overlap", "linear"):
            case = self.make_case()
            if bad == "overlap":
                case.coordinates[1] = case.coordinates[0]
            else:
                case.coordinates[:] = [[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]
            self.write_coordinates(case)
            with self.assertRaisesRegex(ValueError, "nonlinear"):
                self.verify(case)


class SameMFCaseOrchestrationTests(unittest.TestCase):
    def test_same_object_and_exact_two_cases_are_persisted_without_new_Opt(self):
        mf = SimpleNamespace(geometry=0, MO=0, grid=0)
        seen, saved = [], []
        def evaluate(actual, label, enabled):
            self.assertIs(actual, mf); seen.append((label, enabled))
            return {"synthetic_case": label}
        results = worker.compare_on_same_mf(mf, evaluate, lambda m:vars(m).copy(),
            lambda label, result:saved.append((label,result)))
        self.assertEqual(seen, [("stock_partial_B8_native4",False),("pure_j_partial_B8_native4",True)])
        self.assertEqual([r[1] for r in saved], results)

    def test_geometry_MO_or_grid_drift_during_either_case_stops_persistence(self):
        for component in ("geometry", "MO", "grid"):
            for fail_case in (False, True):
                mf = SimpleNamespace(geometry=0, MO=0, grid=0); saved = []; seen = []
                def evaluate(actual, label, enabled):
                    seen.append(enabled)
                    if enabled == fail_case:setattr(actual,component,1)
                    return {"synthetic_case":label}
                with self.assertRaisesRegex(RuntimeError, "drift during case"):
                    worker.compare_on_same_mf(mf,evaluate,lambda m:vars(m).copy(),lambda *x:saved.append(x))
                self.assertEqual(len(saved), int(fail_case))
                self.assertEqual(len(seen), 1+int(fail_case))

    def test_drift_between_cases_rejected_before_second_evaluation(self):
        mf = object(); count = [0]; seen = []; saved = []
        def fingerprint(_mf):
            count[0] += 1
            return {"synthetic_state_revision": int(count[0] >= 4)}
        with self.assertRaisesRegex(RuntimeError,"drift before case"):
            worker.compare_on_same_mf(mf,lambda m,label,enabled:seen.append(enabled),fingerprint,
                                      lambda *x:saved.append(x))
        self.assertEqual(seen,[False]);self.assertEqual(len(saved),1)

    def test_failed_case_is_not_replaced_by_other_seed_or_following_case(self):
        seen = [];saved = []
        def fail(_mf,_label,enabled):
            seen.append(enabled);raise RuntimeError("synthetic actual case failed")
        with self.assertRaisesRegex(RuntimeError,"synthetic actual case failed"):
            worker.compare_on_same_mf(object(),fail,lambda m:{"unchanged":True},lambda *x:saved.append(x))
        self.assertEqual(seen,[False]);self.assertEqual(saved,[])


class CompletedEquivalenceLaunchGateTests(unittest.TestCase):
    """Numerical/source layer only; closed-execution layer is tested separately."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="pure-j-completed-gate-synthetic-")
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        supervisor=self.root/"synthetic_supervisor.py";supervisor.write_text("# synthetic identity only; never executed\n")
        patcher=patch.object(worker,"SUPERVISOR_PATH",supervisor);patcher.start();self.addCleanup(patcher.stop)
        patcher=patch.object(worker,"verify_closed_validation_execution",
            return_value={"healthy_closed_parent_verified":True,"synthetic_isolated_layer":True})
        self.closed_execution=patcher.start();self.addCleanup(patcher.stop)
        self.counter=0

    def record(self,identity):
        comparison=worker.compare_complete_cases(PureJComparisonGateTests.case(3 if identity=="water" else 9),
            PureJComparisonGateTests.case(3 if identity=="water" else 9),3 if identity=="water" else 9)
        return {"synthetic_nonphysical_test_fixture":True,"identity":identity,"status":"DONE",
            "validation_only":True,"validation_passed":True,"numerical_equivalence_passed":True,
            "stationary_minimum_verified":True,"final_gradient_thresholds_passed":True,
            "same_MF_geometry_MO_grid_retained":True,"tolerances_fixed_before_results":dict(worker.TOLERANCES),
            "actual_entrypoint":{"sha256":worker.sha(SCRIPT)},
            "source_identity":{"frozen_V8":{"sha256":FROZEN_V8_SHA},
                "actual_V4_supervisor":{"sha256":worker.sha(worker.SUPERVISOR_PATH)}},
            "resource_supervision":{"proof":{"schema":"research4-pyscf-actual-cpu-supervisor-proof-v4"}},
            "comparison":comparison}

    def files(self,water=None,ethanol=None):
        self.counter+=1
        paths=[self.root/f"water{self.counter}.json",self.root/f"ethanol{self.counter}.json"]
        for path,record in zip(paths,(water or self.record("water"),ethanol or self.record("validation_ethanol"))):
            path.write_text(json.dumps(record))
        return paths

    def test_complete_matching_water_and_ethanol_records_return_hash_bound_evidence(self):
        paths=self.files();evidence=worker.verify_completed_equivalence_records(paths)
        self.assertEqual({r["identity"] for r in evidence},{"water","validation_ethanol"})
        self.assertEqual([r["result_sha256"] for r in evidence],[worker.sha(p) for p in paths])
        self.assertEqual([r["actual_entrypoint_sha256"] for r in evidence],[worker.sha(SCRIPT)]*2)
        self.assertEqual(self.closed_execution.call_count,2)
        self.assertTrue(all(r["closed_parent_execution"]["healthy_closed_parent_verified"] for r in evidence))

    def test_failed_closed_parent_gate_cannot_be_overridden_by_scientific_success_flags(self):
        self.closed_execution.side_effect=ValueError("synthetic closed parent rejected")
        with self.assertRaisesRegex(ValueError,"synthetic closed parent rejected"):
            worker.verify_completed_equivalence_records(self.files())

    def test_missing_duplicate_wrong_identity_or_partial_records_are_rejected(self):
        for paths in (None,[],[self.root/"not_read.json"]):
            with self.assertRaises(ValueError):worker.verify_completed_equivalence_records(paths)
        paths=self.files()
        with self.assertRaises(ValueError):worker.verify_completed_equivalence_records([paths[0],paths[0]])
        for bad in ("identity","status"):
            first=self.record("water");first[bad]="spinosin" if bad=="identity" else "PARTIAL"
            with self.assertRaises(ValueError):worker.verify_completed_equivalence_records(self.files(water=first))

    def test_every_required_scientific_success_flag_is_explicitly_true(self):
        for key in ("validation_only","validation_passed","numerical_equivalence_passed","stationary_minimum_verified",
                    "final_gradient_thresholds_passed","same_MF_geometry_MO_grid_retained"):
            for bad in (False,None):
                first=self.record("water");first[key]=bad
                with self.assertRaisesRegex(ValueError,"scientific gate"):
                    worker.verify_completed_equivalence_records(self.files(water=first))

    def test_entry_V8_supervisor_hash_and_new_proof_schema_must_match(self):
        for bad in ("entry","V8","supervisor","proof"):
            first=self.record("water")
            if bad=="entry":first["actual_entrypoint"]["sha256"]="0"*64
            elif bad=="V8":first["source_identity"]["frozen_V8"]["sha256"]="0"*64
            elif bad=="supervisor":first["source_identity"]["actual_V4_supervisor"]["sha256"]="0"*64
            else:first["resource_supervision"]["proof"]["schema"]="research4-pyscf-resource-supervisor-proof-v3"
            with self.assertRaises(ValueError):worker.verify_completed_equivalence_records(self.files(water=first))

    def test_fixed_tolerances_full_cross_couplings_and_actual_mode_counts_are_required(self):
        for bad in ("tolerance","cross","mode"):
            first=self.record("validation_ethanol")
            if bad=="tolerance":first["tolerances_fixed_before_results"]["frequencies_cm1"] = .1
            elif bad=="cross":first["comparison"]["all_atom_pair_cross_couplings_compared"]=False
            else:first["comparison"]["all_actual_modes_compared"]=20
            with self.assertRaises(ValueError):worker.verify_completed_equivalence_records(self.files(ethanol=first))

    def test_measured_numeric_gates_cannot_be_replaced_by_success_booleans(self):
        tolerances={"raw_hessian":1e-6,"e1":1e-6,"ej":1e-6,"gradient":1e-8,
                    "frequencies_cm1":.01,"energy_hartree":1e-8}
        for field,tolerance in tolerances.items():
            for bad in ("above","negative","nan","bool","missing","wrong_tolerance","failed"):
                first=self.record("water");row=first["comparison"]["differences"][field]
                if bad=="above":row["max_abs_difference"]=2*tolerance
                elif bad=="negative":row["max_abs_difference"]=-tolerance
                elif bad=="nan":row["max_abs_difference"]=float("nan")
                elif bad=="bool":row["max_abs_difference"]=False
                elif bad=="missing":row.pop("max_abs_difference")
                elif bad=="wrong_tolerance":row["fixed_tolerance"]=2*tolerance
                else:row["passed"]=False
                with self.assertRaisesRegex(ValueError,"measured numerical gate"):
                    worker.verify_completed_equivalence_records(self.files(water=first))


class ClosedValidationExecutionGateTests(unittest.TestCase):
    """Temporary ledger/proof/log text only; no real ledger, process or worker run."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="pure-j-parent-closure-synthetic-")
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.counter=0
        supervisor=self.root/"synthetic_supervisor.py"
        supervisor.write_text("# synthetic supervisor source identity only; never executed\n")
        patcher=patch.object(worker,"SUPERVISOR_PATH",supervisor)
        patcher.start();self.addCleanup(patcher.stop)

    def make_case(self,identity="water"):
        self.counter+=1;path=self.root/str(self.counter);path.mkdir()
        case=SimpleNamespace(result_path=path/"result.json",ledger_path=path/"ledger.json",
            proof_path=path/"proof.json",log_path=path/"worker.log")
        case.proof={"schema":"research4-pyscf-actual-cpu-supervisor-proof-v4",
            "supervisor_script_sha256":worker.sha(worker.SUPERVISOR_PATH),
            "ledger_path":str(case.ledger_path),"phase":"synthetic_closed_validation_only",
            "child_pid":12345,"actual_cpu_guard_source":"PROC_STAT_CUMULATIVE_FROM_PROCESS_BIRTH",
            "actual_cpu_clock_ticks_per_second":100,"actual_cpu_cleanup_reserve_seconds":120,
            "actual_cpu_stop_threshold_seconds":180,"actual_cpu_closed_batch_cost_seconds":0,
            "actual_cpu_batch_cap_seconds":129600}
        case.log_path.write_text("Synthetic closure-gate text fixture only; no process was run.\n")
        case.proof_path.write_text(json.dumps(case.proof))
        case.value={"synthetic_nonphysical_test_fixture":True,"identity":identity,"status":"DONE",
            "resource_supervision":{"proof":case.proof,"proof_sha256":worker.sha(case.proof_path)}}
        case.row={"phase":case.proof["phase"],"status":"EXITED","exit_code":0,
            "ended_at_utc":"SYNTHETIC_CLOSED_TIMESTAMP_NOT_A_REAL_EXECUTION",
            "pid":case.proof["child_pid"],"actual_cpu_guard_enabled":True,
            **{key:case.proof[key] for key in ("actual_cpu_guard_source","actual_cpu_clock_ticks_per_second",
                "actual_cpu_cleanup_reserve_seconds","actual_cpu_stop_threshold_seconds",
                "actual_cpu_closed_batch_cost_seconds","actual_cpu_batch_cap_seconds")},
            "supervisor_proof_path":str(case.proof_path),"supervisor_proof_sha256":worker.sha(case.proof_path),
            "log":str(case.log_path),"log_sha256":worker.sha(case.log_path),
            "command":[str(ROOT/".runtime/bin/python"),str(SCRIPT),"--phase","pure-j-equivalence",
                "--identity",identity,"--output",str(path)]}
        case.ledger={"schema":"research4-pyscf-actual-cpu-budget-v4",
            "runner_sha256":worker.sha(worker.SUPERVISOR_PATH),"phases":[case.row]}
        self.write(case)
        return case

    @staticmethod
    def write(case):
        case.ledger_path.write_text(json.dumps(case.ledger))
        case.result_path.write_text(json.dumps(case.value))

    def verify(self,case):
        self.write(case)
        return worker.verify_closed_validation_execution(case.value,case.result_path)

    def test_healthy_closed_exit0_binds_proof_log_command_and_canonical_row(self):
        for identity in ("water","validation_ethanol"):
            case=self.make_case(identity);evidence=self.verify(case)
            self.assertTrue(evidence["healthy_closed_parent_verified"])
            self.assertEqual(evidence["exit_code"],0)
            self.assertEqual(evidence["proof_sha256"],worker.sha(case.proof_path))
            self.assertEqual(evidence["log_sha256"],worker.sha(case.log_path))
            expected=hashlib.sha256(json.dumps(case.row,sort_keys=True,allow_nan=False).encode()).hexdigest()
            self.assertEqual(evidence["closed_row_canonical_sha256"],expected)

    def test_equals_style_arguments_preserve_exact_fixture_output_binding(self):
        case=self.make_case("validation_ethanol")
        case.row["command"]=[str(ROOT/".runtime/bin/python"),str(SCRIPT),"--phase=pure-j-equivalence",
            "--identity=validation_ethanol","--output="+str(case.result_path.parent)]
        self.assertTrue(self.verify(case)["healthy_closed_parent_verified"])

    def test_self_reported_DONE_does_not_override_unclosed_or_failed_parent(self):
        for bad in ("RUNNING","PARTIAL","PLANNED"):
            case=self.make_case();case.row["status"]=bad
            with self.assertRaisesRegex(ValueError,"closed EXITED0"):
                self.verify(case)
        for code in (2,-24,None,False,"0"):
            case=self.make_case();case.row["exit_code"]=code
            with self.assertRaisesRegex(ValueError,"closed EXITED0"):
                self.verify(case)

    def test_parent_schema_unique_phase_and_child_identity_are_required(self):
        case=self.make_case();case.ledger["schema"]="research4-pyscf-monitored-budget-v3"
        with self.assertRaisesRegex(ValueError,"actual-CPU V4"):
            self.verify(case)
        for bad in ("missing","duplicate","empty_phase"):
            case=self.make_case()
            if bad=="missing":case.ledger["phases"]=[]
            elif bad=="duplicate":case.ledger["phases"]=[case.row,dict(case.row)]
            else:case.proof["phase"]=""
            with self.assertRaisesRegex(ValueError,"unique parent phase"):
                self.verify(case)
        for key,value in (("pid",54321),("actual_cpu_guard_enabled",False)):
            case=self.make_case();case.row[key]=value
            with self.assertRaisesRegex(ValueError,"PID/actual-CPU guard"):
                self.verify(case)

    def test_parent_runner_proof_source_and_closed_end_timestamp_are_required(self):
        for bad in ("runner","proof_source"):
            case=self.make_case()
            if bad=="runner":case.ledger["runner_sha256"]="0"*64
            else:case.proof["supervisor_script_sha256"]="0"*64
            with self.assertRaisesRegex(ValueError,"supervisor source identity"):
                self.verify(case)
        for bad in (None,"",123):
            case=self.make_case();case.row["ended_at_utc"]=bad
            with self.assertRaisesRegex(ValueError,"end timestamp"):
                self.verify(case)

    def test_any_resource_monitor_residual_or_containment_failure_blocks_release(self):
        flags=("wall_terminated","memory_terminated","cpu_terminated","monitor_failed","hard_killed",
            "actual_cpu_guard_TERM_sent","owned_child_pidfd_containment_requested",
            "owned_unreaped_child_direct_PID_containment_requested",
            "residual_processes_require_review_after_main_payload_exit")
        for flag in flags:
            case=self.make_case();case.row[flag]=True
            with self.assertRaisesRegex(ValueError,"resource stop/monitor/containment"):
                self.verify(case)
        for key,value in (("resource_stop_reason","synthetic CPU stop"),
                          ("actual_cpu_stop_observation",{"synthetic_cpu_seconds":181})):
            case=self.make_case();case.row[key]=value
            with self.assertRaisesRegex(ValueError,"resource stop/monitor/containment"):
                self.verify(case)

    def test_six_actual_CPU_contract_fields_must_exist_and_match_closed_proof(self):
        keys=("actual_cpu_guard_source","actual_cpu_clock_ticks_per_second",
            "actual_cpu_cleanup_reserve_seconds","actual_cpu_stop_threshold_seconds",
            "actual_cpu_closed_batch_cost_seconds","actual_cpu_batch_cap_seconds")
        for key in keys:
            case=self.make_case();case.row[key]="mismatch"
            with self.assertRaisesRegex(ValueError,"actual-CPU contract differs"):
                self.verify(case)
            case=self.make_case();case.proof.pop(key)
            with self.assertRaisesRegex(ValueError,"actual-CPU contract differs"):
                self.verify(case)

    def test_proof_and_log_bytes_and_both_proof_hash_references_must_match(self):
        for bad in ("proof_file_changed","proof_row_hash","proof_result_hash","proof_JSON_changed","log_changed","log_hash"):
            case=self.make_case()
            if bad=="proof_file_changed":case.proof_path.write_text(case.proof_path.read_text()+" ")
            elif bad=="proof_row_hash":case.row["supervisor_proof_sha256"]="0"*64
            elif bad=="proof_result_hash":case.value["resource_supervision"]["proof_sha256"]="0"*64
            elif bad=="proof_JSON_changed":
                altered=dict(case.proof);altered["different_synthetic_proof"]=True
                case.proof_path.write_text(json.dumps(altered))
                case.row["supervisor_proof_sha256"]=worker.sha(case.proof_path)
                case.value["resource_supervision"]["proof_sha256"]=worker.sha(case.proof_path)
            elif bad=="log_changed":case.log_path.write_text("different synthetic log bytes")
            else:case.row["log_sha256"]="0"*64
            with self.assertRaisesRegex(ValueError,"proof/log bytes"):
                self.verify(case)

    def test_wrong_phase_fixture_output_or_entrypoint_command_cannot_pass(self):
        for bad in ("phase","identity","output","entrypoint","missing_output","not_list","non_string"):
            case=self.make_case()
            if bad=="phase":case.row["command"][3]="spinosin-frequency"
            elif bad=="identity":case.row["command"][5]="validation_ethanol"
            elif bad=="output":case.row["command"][7]=str(self.root/"other_output")
            elif bad=="entrypoint":case.row["command"][1]=str(worker.V8_PATH)
            elif bad=="missing_output":case.row["command"]=case.row["command"][:6]
            elif bad=="not_list":case.row["command"]="not a complete argv list"
            else:case.row["command"][1]=None
            with self.assertRaisesRegex(ValueError,"command"):
                self.verify(case)


if __name__ == "__main__":
    unittest.main()
