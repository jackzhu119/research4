"""Pure synthetic engineering tests; no PySCF import, integral, SCF or gradient.

The independent reference is the frozen V9 full-global-J body.  Every sliced
mock integral is a view/contraction of ONE fixed full AO tensor, so an output-AO
partition cannot pass merely because the fixture invents matching answers.
These tests are not real molecular or cross-process full-Hessian validation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


before_import = set(sys.modules)
inc = load("incremental_rhf_pure_test", "pyscf_incremental_rhf_checkpoint_20261103.py")
reference = load("reference_v9_incremental_test", "pyscf_spinosin_pure_j_worker_20261101.py")
imported_quantum = {name for name in set(sys.modules)-before_import if name == "pyscf" or name.startswith("pyscf.")}


class SilentLogger:
    @staticmethod
    def new_logger(*args): return SilentLogger()
    @staticmethod
    def process_clock(): return 0.
    @staticmethod
    def perf_counter(): return 0.
    @staticmethod
    def timer_debug1(message, *times): return times
    @staticmethod
    def timer(message, *times): return times


class Fixture:
    """Nonphysical small arrays; all shell slices map to the same global AO axes."""
    def __init__(self):
        rng = np.random.default_rng(14052026)
        self.aoslices = np.array([[0, 1, 0, 1], [1, 2, 1, 3], [2, 3, 3, 4]])
        self.calls, self.delegations = [], []
        self.tensors = {key: rng.normal(size=(9, 4, 4, 4, 4)) for key in
                        ("int2e_ipip1", "int2e_ip1ip2", "int2e_ipvip1")}
        self.overlaps = (rng.normal(size=(3, 3, 4, 4)), rng.normal(size=(3, 3, 4, 4)), rng.normal(size=(3, 4, 4)))
        self.cores = rng.normal(size=(3, 3, 3, 3, 4, 4))
        coordinates, charges = np.zeros((3, 3)), np.array([1, 1, 6], dtype=np.int32)
        mol = SimpleNamespace(natm=3, nbas=3, spin=0, charge=0, cart=False, symmetry=False,
            nelectron=4, incore_anyway=False, _atm=np.arange(18, dtype=np.int32).reshape(3, 6),
            _bas=np.arange(24, dtype=np.int32).reshape(3, 8), _env=np.zeros(50),
            nao_nr=lambda: 4, has_ecp=lambda: False, aoslice_by_atom=lambda: self.aoslices.copy(),
            atom_coords=lambda unit="Bohr": coordinates.copy(), atom_charges=lambda: charges.copy())
        ni = SimpleNamespace(rsh_and_hybrid_coeff=lambda xc, spin=0: (0., 0., 0.),
             libxc=SimpleNamespace(is_hybrid_xc=lambda xc: False), _xc_type=lambda xc: "GGA", cutoff=1e-15, omega=None)
        grid = SimpleNamespace(coords=rng.normal(size=(17, 3)), weights=rng.normal(size=17),
             non0tab=np.ones((1, 3), dtype=np.uint8), screen_index=np.ones((1, 3), dtype=np.uint8),
             atm_idx=None, quadrature_weights=None, level=1, cutoff=1e-15)
        coeff, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        self.mf = SimpleNamespace(mol=mol, xc="PBE", _numint=ni, with_df=None, disp=None, nlc="",
             mo_coeff=coeff, mo_energy=np.array([-1.1, -.35, .2, .8]), mo_occ=np.array([2., 2., 0., 0.]),
             e_tot=-10., converged=True, max_memory=3000, direct_scf=True, direct_scf_tol=1e-13,
             _eri=None, conv_tol=1e-8, conv_tol_cpscf=1e-8, max_cycle=80, small_rho_cutoff=1e-7, grids=grid)
        self.hobj = SimpleNamespace(mol=mol, base=self.mf, max_memory=3000, grids=None,
                                   hcore_generator=lambda mol: lambda a, b: self.cores[a, b].copy())
        self.rhf = SimpleNamespace(_get_jk=self.get_jk, logger=SilentLogger,
                                   get_ovlp=lambda mol: tuple(x.copy() for x in self.overlaps),
                                   _vhf=SimpleNamespace(libcvhf=SimpleNamespace()))
        def stock(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None,
                  max_memory=4000, verbose=None, with_k=True):
            self.delegations.append((hessobj, mo_energy, mo_coeff, mo_occ, atmlst, max_memory, verbose, with_k))
            return ("original", with_k)
        self.original = self.rhf._partial_hess_ejk = stock

    def get_jk(self, mol, intor, comp, aosym, scripts, shls_slice=None, vhfopt=None):
        self.calls.append((intor, shls_slice, aosym, scripts[0], np.array(scripts[1], copy=True)))
        assert comp == 9 and vhfopt is None and len(scripts) == 2
        tensor = self.tensors[intor]
        if shls_slice is not None:
            assert shls_slice[2:] == (0, 3)*3
            shell0, shell1 = shls_slice[:2]
            start, stop = self.aoslices[shell0, 2], self.aoslices[shell1-1, 3]
            tensor = tensor[:, start:stop]
        if intor == "int2e_ip1ip2":
            assert scripts[0] == "ji->s1kl" and aosym == "s1"
            return [np.einsum("cijkl,ji->ckl", tensor, scripts[1])]
        assert scripts[0] == "lk->s1ij" and aosym == "s2kl"
        return [np.einsum("cijkl,lk->cij", tensor, scripts[1])]

    def reference(self):
        return reference.pure_j_partial_body(self.rhf, self.hobj)


class IncrementalTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "outputs").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="unit-incremental-rhf-", dir=ROOT / "outputs")
        self.cache = Path(self.temporary.name) / "cache"
        self.fixture = Fixture()
        self.patches = [patch.object(inc, "_source_identity", return_value={"source": "SYNTHETIC_ONLY"}),
                        patch.object(inc, "_check_stock_callable", return_value=None)]
        for p in self.patches: p.start()

    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.temporary.cleanup()

    def context(self, limit=6, **overrides):
        kwargs = {"rhf_module": self.fixture.rhf, "execution_evidence": {"scope": "SYNTHETIC_ONLY"}}
        kwargs.update(overrides)
        return inc.IncrementalPureJContext(self.fixture.hobj, self.cache, {"snapshot": "synthetic-fixture"}, limit, **kwargs)

    def run_segment(self, limit):
        with self.context(limit) as context:
            result = self.fixture.rhf._partial_hess_ejk(self.fixture.hobj, with_k=False)
            return context, result

    def assert_matches(self, actual, expected):
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, atol=2e-13, rtol=2e-14)
        self.assertTrue(np.array_equal(actual[2], np.zeros((3, 3, 3, 3))))

    def test_import_does_not_import_pyscf(self):
        self.assertEqual(imported_quantum, set())
        self.assertEqual(hashlib.sha256((ROOT / "scripts/pyscf_spinosin_pure_j_worker_20261101.py").read_bytes()).hexdigest(),
                         "e3215e563b4429212084cfccbcec98e6a9f482bd8c3352d53d3064402649b450")

    def test_uninterrupted_full_matrices_match_frozen_reference(self):
        expected = self.fixture.reference()
        self.fixture.calls.clear()
        context, actual = self.run_segment(6)
        self.assert_matches(actual, expected)
        self.assertTrue(context.progress()["complete_RHF_helper"])
        self.assertEqual(len(self.fixture.calls), 9)
        self.assertTrue(all(c[1] is not None for c in self.fixture.calls[:3]))
        self.assertTrue(all(c[4].shape == (4, 4) for c in self.fixture.calls[:3]))

    def test_every_unit_boundary_restores_a_continuous_exact_prefix(self):
        expected = self.fixture.reference()
        self.fixture.calls.clear()
        for done in range(1, 6):
            previous = len(self.fixture.calls)
            with self.assertRaises(inc.CheckpointSegmentComplete) as caught:
                self.run_segment(1)
            progress = caught.exception.progress
            self.assertEqual(progress["total_units_completed"], done)
            self.assertEqual(progress["new_units_this_context"], 1)
            self.assertFalse(progress["complete_RHF_helper"])
            self.assertEqual(len(self.fixture.calls)-previous, 1 if done <= 3 else 2)
            self.assertIs(self.fixture.rhf._partial_hess_ejk, self.fixture.original)
        context, actual = self.run_segment(1)
        self.assert_matches(actual, expected)
        self.assertEqual(context.progress()["total_units_completed"], 6)
        self.assertEqual(len(self.fixture.calls), 9)

    def test_loaded_matrices_are_fresh_and_cannot_corrupt_cache(self):
        _, first = self.run_segment(6)
        expected = tuple(a.copy() for a in first)
        first[0][:] += first[1]
        first[1][:] = 177.
        count = len(self.fixture.calls)
        _, second = self.run_segment(0)
        self.assertEqual(len(self.fixture.calls), count)
        self.assert_matches(second, expected)
        self.assertFalse(np.shares_memory(first[0], second[0]))
        # Deliberately nonsymmetric synthetic diagonal must not be averaged.
        self.assertFalse(np.array_equal(second[1][1, 1], second[1][1, 1].T))

    def test_with_k_true_delegates_exact_arguments(self):
        token = object()
        with self.context(0):
            result = self.fixture.rhf._partial_hess_ejk(self.fixture.hobj, verbose=token, with_k=True)
        self.assertEqual(result, ("original", True))
        self.assertIs(self.fixture.delegations[0][6], token)
        self.assertEqual(self.fixture.calls, [])

    def test_explicit_identical_mo_copies_are_accepted(self):
        with self.context(6):
            actual = self.fixture.rhf._partial_hess_ejk(self.fixture.hobj,
                 self.fixture.mf.mo_energy.copy(), self.fixture.mf.mo_coeff.copy(),
                 self.fixture.mf.mo_occ.copy(), list(range(3)), with_k=False)
        self.assert_matches(actual, self.fixture.reference())

    def test_bad_explicit_mo_or_atom_order_rejected_before_integrals(self):
        for key, value in (("mo_energy", self.fixture.mf.mo_energy+.01),
                           ("mo_coeff", self.fixture.mf.mo_coeff.astype(np.float32)),
                           ("mo_occ", np.zeros(4)), ("atmlst", [1, 0, 2])):
            with self.subTest(key=key), self.context(6):
                with self.assertRaises(ValueError):
                    self.fixture.rhf._partial_hess_ejk(self.fixture.hobj, with_k=False, **{key: value})
        self.assertEqual(self.fixture.calls, [])

    def test_runtime_mo_grid_and_snapshot_binding_changes_rejected(self):
        with self.assertRaises(inc.CheckpointSegmentComplete): self.run_segment(1)
        for array in (self.fixture.mf.mo_coeff, self.fixture.mf.grids.coords, self.fixture.mf.grids.weights):
            old = array.flat[0]
            array.flat[0] += .001
            with self.assertRaises(ValueError):
                with self.context(1): pass
            array.flat[0] = old
        with self.assertRaises(ValueError):
            with inc.IncrementalPureJContext(self.fixture.hobj, self.cache, {"snapshot": "other"}, 1,
                                            rhf_module=self.fixture.rhf): pass
        with self.context(1):
            self.fixture.mf.mo_coeff[0, 0] += 1e-8
            with self.assertRaises(ValueError):
                self.fixture.rhf._partial_hess_ejk(self.fixture.hobj, with_k=False)
            self.fixture.mf.mo_coeff[0, 0] -= 1e-8

    def test_transient_last_rinv_marker_is_only_normalized_state(self):
        first = inc.runtime_state_fingerprint(self.fixture.hobj)
        self.fixture.mf.mol._env[17] = 2.
        self.assertEqual(first, inc.runtime_state_fingerprint(self.fixture.hobj))
        self.fixture.mf.mol._env[18] = 1.
        self.assertNotEqual(first, inc.runtime_state_fingerprint(self.fixture.hobj))
        self.assertEqual(self.fixture.mf.mol._env[17], 2.)

    def test_orphan_attempt_is_preserved_and_next_attempt_is_used(self):
        with self.assertRaises(inc.CheckpointSegmentComplete): self.run_segment(1)
        orphan = self.cache / "units/000001-diag-0001/attempt-000001"
        orphan.mkdir(parents=True)
        (orphan / "jdiag.npy").write_bytes(b"interrupted-not-a-complete-npy")
        _, actual = self.run_segment(6)
        self.assertEqual((orphan / "jdiag.npy").read_bytes(), b"interrupted-not-a-complete-npy")
        self.assertTrue((orphan.parent / "attempt-000002/complete.json").is_file())
        self.assert_matches(actual, self.fixture.reference())

    def test_duplicate_complete_and_prefix_hole_rejected(self):
        self.run_segment(6)
        first = self.cache / "units/000000-diag-0000/attempt-000001"
        duplicate = first.with_name("attempt-000002")
        shutil.copytree(first, duplicate)
        with self.assertRaises(ValueError):
            with self.context(0): pass
        shutil.rmtree(duplicate)
        first.rename(first.with_name("attempt-000099-orphan"))
        with self.assertRaises(ValueError):
            with self.context(0): pass

    def test_missing_complete_is_a_prefix_hole_and_payload_changes_rejected(self):
        self.run_segment(6)
        first = self.cache / "units/000000-diag-0000/attempt-000001"
        completion = first / "complete.json"
        old = completion.read_bytes()
        completion.unlink()
        with self.assertRaises(ValueError):
            with self.context(0): pass
        completion.write_bytes(old)
        (first / "jdiag.npy").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            with self.context(0): pass

    def test_invalid_numeric_state_rejected(self):
        for name, value in (("mo_coeff", np.zeros((4, 3))), ("mo_energy", np.array([np.nan, 0., 1., 2.])),
                            ("mo_occ", np.ones(4))):
            old = getattr(self.fixture.mf, name)
            setattr(self.fixture.mf, name, value)
            with self.assertRaises(ValueError):
                with self.context(1): pass
            setattr(self.fixture.mf, name, old)

    def test_model_and_resource_contract_rejected(self):
        for owner, key, value in ((self.fixture.mf, "xc", "B3LYP"),
                                  (self.fixture.mf, "max_memory", 8000),
                                  (self.fixture.mf, "direct_scf_tol", 1e-10),
                                  (self.fixture.mf.mol, "spin", 2),
                                  (self.fixture.mf, "disp", "D3BJ")):
            old = getattr(owner, key)
            setattr(owner, key, value)
            with self.assertRaises(ValueError):
                with self.context(1): pass
            setattr(owner, key, old)

    def test_nested_contexts_and_foreign_replacement_restore_original(self):
        context = self.context(0)
        with context:
            wrapper = self.fixture.rhf._partial_hess_ejk
            with self.assertRaises(RuntimeError): context.__enter__()
            with self.assertRaises(RuntimeError):
                with self.context(0): pass
            self.assertIs(self.fixture.rhf._partial_hess_ejk, wrapper)
        self.assertIs(self.fixture.rhf._partial_hess_ejk, self.fixture.original)
        with self.assertRaises(RuntimeError):
            with self.context(0): self.fixture.rhf._partial_hess_ejk = lambda: None
        self.assertIs(self.fixture.rhf._partial_hess_ejk, self.fixture.original)

    def test_scope_and_finite_unit_limit_rejected(self):
        for bad in (-1, True, 1.5, None):
            with self.assertRaises(ValueError): self.context(bad)
        with self.assertRaises(ValueError):
            with inc.IncrementalPureJContext(self.fixture.hobj, ROOT / "outputs", {"x": 1},
                                            rhf_module=self.fixture.rhf): pass


if __name__ == "__main__":
    unittest.main()
