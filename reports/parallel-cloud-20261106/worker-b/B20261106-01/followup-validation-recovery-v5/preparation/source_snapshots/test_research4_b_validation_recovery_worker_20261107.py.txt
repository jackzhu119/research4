"""Recovery provenance/solver tests; no SCF, gradient or grid kernels execute."""
from __future__ import annotations

import ast
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
SOURCE = SCRIPTS / "research4_b_validation_recovery_worker_20261107.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load(SOURCE, "validation_recovery_worker_test_subject")
prior = load(SCRIPTS / "test_research4_b_validation_worker_20261107.py", "recovery_v4_test_cases")
prior.worker = worker
prior.SOURCE = SOURCE
prior.prior.worker = worker
prior.prior.original_tests.worker = worker
prior.prior.recovery_tests.worker = worker
prior.prior.recovery_tests.RECOVERY = SOURCE


class RecoveryInvariantTests(unittest.TestCase):
    def test_recovery_retains_strict_internal_tolerances_and_unambiguous_centers(self):
        request = {"schema": worker.REQUEST_SCHEMA, "operation": "point", "case": "water",
                   "direction": {"type": "mode", "index": 1}, "step_bohr": 0, "sign": 0,
                   "scf_conv_tol": 1e-12, "scf_conv_tol_grad": 1e-9,
                   "scf_qualification_gradient_norm": 1e-8}
        worker.validate_request(request)
        with self.assertRaisesRegex(ValueError, "h=0/sign=0"):
            worker.validate_request({**request, "sign": 1})
        for changes in ({"scf_conv_tol": 1e-10}, {"scf_conv_tol_grad": 1e-8},
                        {"scf_qualification_gradient_norm": 1e-7}):
            with self.assertRaises(ValueError):
                worker.validate_request({**request, **changes})

    def test_nonconverged_scf_cannot_qualify_even_with_zero_fresh_fock_gradient(self):
        from types import SimpleNamespace
        class MF:
            grids = SimpleNamespace(coords=np.array([[0., 0., 0.]]), weights=np.ones(1))
            converged = False
            conv_tol_grad = 1e-9
            mo_coeff, mo_occ = np.eye(2), np.array([2., 0.])
            mol = object()
            make_rdm1 = lambda self: np.diag([2., 0.])
            get_hcore = lambda self: np.eye(2)
            get_veff = lambda self, *args: np.zeros((2, 2))
            get_grad = lambda self, *args: np.zeros(1)
        mf = MF()
        record = worker.independent_scf_qualification(mf, {"scf_qualification_gradient_norm": 1e-8},
                                                       np, lambda *a, **k: None)
        self.assertFalse(record["qualified"])
        self.assertFalse(mf.converged)
        self.assertEqual(record["status"], "SCF_NOT_QUALIFIED")

    def test_finite_difference_and_response_algebra_remain_identical_to_frozen_v4(self):
        functions = lambda path: {node.name: node for node in ast.parse(path.read_text()).body
                                  if isinstance(node, ast.FunctionDef)}
        old = functions(SCRIPTS / "research4_b_validation_worker_20261107.py")
        new = functions(SOURCE)
        for name in ("unit_direction", "central_difference", "overlap_directional_derivative",
                     "response_equation_metrics", "density_directional_derivative",
                     "independent_scf_qualification", "configure_fixed_grid", "true_gradient_quadrature",
                     "initialize_platform_processor_without_fork", "supervise"):
            self.assertEqual(ast.dump(old[name], include_attributes=False),
                             ast.dump(new[name], include_attributes=False), name)


class FakeMolecule:
    natm, charge, spin, cart, nelectron, incore_anyway = 2, 0, 0, False, 2, False
    _basis, _ecp = {"H": [[0, [1., 1.]]]}, {}
    def __init__(self, coordinates=None):
        self.coordinates = np.zeros((2, 3)) if coordinates is None else np.asarray(coordinates).copy()
        self.overlap = np.eye(2)
    def nao_nr(self):
        return 2
    def atom_symbol(self, index):
        return "H"
    def atom_coords(self, unit=None):
        return self.coordinates.copy()
    def intor_symmetric(self, name):
        if name != "int1e_ovlp":
            raise AssertionError("Only a fake source overlap may be requested")
        return self.overlap.copy()


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="v5-recovery-unit-", dir=worker.ROOT / "reports")
        self.directory = Path(self.temp.name)
        self.checkpoint = self.directory / "source.chk"
        self.checkpoint.write_bytes(b"Manufactured checkpoint bytes; payload reader is mocked")
        self.program = self.directory / "source_worker.py"
        self.program.write_text("# Manufactured source fixture\n")
        self.frequency = self.directory / "frequency.json"
        self.frequency.write_text("{}\n")
        self.geometry = self.directory / "geometry_bohr.npy"
        worker.save_array(self.geometry, np.zeros((2, 3)), np)
        self.settings = {"xc": "PBE", "basis": "def2-svp", "grid_level": 3,
                         "gradient_grid_response": False, "small_rho_cutoff": 0,
                         "scf_conv_tol": 1e-12, "scf_conv_tol_grad": 1e-9,
                         "direct_scf_tol": 1e-13}
        self.observation = {"schema": worker.SCHEMA, "case": "water", "status": "PARTIAL",
            "source_worker": worker.ref(self.program), "original_frequency_result": worker.ref(self.frequency),
            "arrays": {"geometry_bohr.npy": worker.ref(self.geometry)}, "settings": self.settings,
            "checkpoint_files": [worker.ref(self.checkpoint)], "scf_converged": None}
        self.result = self.directory / "result.json"
        worker.save(self.result, self.observation, exclusive=True)
        self.request = {"case": "water", "frequency_result": worker.ref(self.frequency),
                        "warmstart_result": worker.ref(self.result), **self.settings}
        self.args = argparse.Namespace(warmstart_result=None, warmstart_result_sha256=None)
        self.payload = {"molecule": FakeMolecule(), "mo_coeff": np.eye(2),
                        "mo_energy": np.array([-.5, .2]), "mo_occ": np.array([2., 0.]),
                        "energy_hartree": -1.}
        self.mf = SimpleNamespace(mol=FakeMolecule(), converged=False,
                                  chkfile=str(self.directory / "new_output.chk"))

    def tearDown(self):
        self.temp.cleanup()

    def load(self, payload=None, request=None):
        with patch.object(worker, "read_checkpoint_payload", return_value=payload or self.payload):
            return worker.load_warmstart(self.mf, self.args, request or self.request, np)

    def unsealed_interrupted_source(self, *, status="PARTIAL", marker=-1.):
        # Match the actual interrupted v4 shape: no checkpoint entry was
        # committed to result.json, but its last progress.json was hash-bound.
        renamed = self.directory / "fresh_point_scf.chk"
        self.checkpoint.rename(renamed)
        self.checkpoint = renamed
        self.observation.pop("checkpoint_files")
        self.observation["status"] = status
        progress = self.directory / "progress.json"
        worker.save(progress, {"event": "scf_cycle", "cycle": 65, "energy_hartree": marker}, exclusive=True)
        self.observation["output_artifacts"] = [worker.ref(progress)]
        worker.save(self.result, self.observation)
        self.request.update(warmstart_result=worker.ref(self.result), warmstart_checkpoint=worker.ref(self.checkpoint))
        return progress

    def test_checkpoint_provenance_and_initial_guess_do_not_create_convergence(self):
        coeff, occupation, metadata = self.load()
        np.testing.assert_array_equal(coeff, np.eye(2))
        np.testing.assert_array_equal(occupation, [2., 0.])
        self.assertFalse(self.mf.converged)
        self.assertEqual(metadata["checkpoint"]["sha256"], worker.sha(self.checkpoint))
        self.assertEqual(metadata["source_status"], "PARTIAL")
        self.assertFalse(metadata["original_checkpoint_modified"])
        self.assertEqual(metadata["source_electron_count_Tr_DS"], 2.)

    def test_checkpoint_change_is_rejected_before_payload_is_read(self):
        self.checkpoint.write_bytes(b"Changed checkpoint bytes")
        with patch.object(worker, "read_checkpoint_payload", side_effect=AssertionError("too early")) as read_payload:
            with self.assertRaisesRegex(ValueError, "hash changed"):
                worker.load_warmstart(self.mf, self.args, self.request, np)
        read_payload.assert_not_called()

    def test_source_checkpoint_cannot_be_the_writable_output_checkpoint(self):
        self.mf.chkfile = str(self.checkpoint)
        with patch.object(worker, "read_checkpoint_payload", side_effect=AssertionError("too early")) as read_payload:
            with self.assertRaisesRegex(ValueError, "writable output checkpoint"):
                worker.load_warmstart(self.mf, self.args, self.request, np)
        read_payload.assert_not_called()

    def test_checkpoint_dimensions_occupations_and_finiteness_are_checked(self):
        for changes in ({"mo_coeff": np.eye(3)}, {"mo_occ": np.array([1., 1.])},
                        {"mo_occ": np.array([0., 0.])}, {"mo_energy": np.array([np.nan, 0.])}):
            with self.subTest(changes=list(changes)):
                with self.assertRaises(ValueError):
                    self.load({**self.payload, **changes})
        self.assertFalse(self.mf.converged)

    def test_checkpoint_geometry_basis_and_source_overlap_are_bound(self):
        changed = copy.deepcopy(self.payload)
        changed["molecule"].coordinates[0, 0] = .1
        with self.assertRaisesRegex(ValueError, "source result geometry"):
            self.load(changed)
        changed = copy.deepcopy(self.payload)
        changed["molecule"]._basis = {"H": [[0, [2., 1.]]]}
        with self.assertRaisesRegex(ValueError, "basis or ECP"):
            self.load(changed)
        changed = copy.deepcopy(self.payload)
        changed["molecule"].overlap = np.diag([2., 1.])
        with self.assertRaisesRegex(ValueError, "orthonormality/electron trace"):
            self.load(changed)

    def test_checkpoint_lineage_and_grid_must_match_while_small_target_displacement_is_an_initial_guess(self):
        with self.assertRaisesRegex(ValueError, "lineage"):
            self.load(request={**self.request, "frequency_result": {"sha256": "a" * 64}})
        with self.assertRaisesRegex(ValueError, "grid_level"):
            self.load(request={**self.request, "grid_level": 5})
        self.mf.mol.coordinates[0, 1] = .001
        _, _, metadata = self.load()
        self.assertFalse(metadata["target_geometry_identical_to_source"])
        self.assertAlmostEqual(metadata["target_geometry_max_abs_displacement_Bohr"], .001)
        self.assertFalse(self.mf.converged)
        self.mf.mol.coordinates[0, 1] = .03
        with self.assertRaisesRegex(ValueError, "finite-displacement range"):
            self.load()

    def test_unsealed_partial_checkpoint_is_bound_to_last_progress_energy(self):
        progress = self.unsealed_interrupted_source(marker=-1. + 5e-13)
        _, _, metadata = self.load()
        self.assertEqual(metadata["checkpoint_binding_origin"],
                         "NEW_EXPLICIT_MANIFEST_BINDING_OF_UNSEALED_INTERRUPTED_CHECKPOINT")
        binding = metadata["interrupted_checkpoint_energy_binding"]
        self.assertEqual(binding["progress"]["sha256"], worker.sha(progress))
        self.assertEqual(binding["cycle"], 65)
        self.assertEqual(binding["fixed_tolerance_Eh"], 1e-12)
        self.assertLessEqual(binding["absolute_checkpoint_energy_difference_Eh"], 1e-12)
        self.assertFalse(self.mf.converged)

    def test_unsealed_partial_checkpoint_energy_mismatch_is_rejected(self):
        self.unsealed_interrupted_source(marker=-1. + 2e-12)
        with self.assertRaisesRegex(ValueError, "fixed 1e-12 Eh rule"):
            self.load()
        self.assertFalse(self.mf.converged)

    def test_unsealed_partial_checkpoint_requires_unchanged_hash_bound_progress(self):
        progress = self.unsealed_interrupted_source()
        progress.write_text(json.dumps({"cycle": 65, "energy_hartree": -1., "tampered": True}))
        with self.assertRaisesRegex(ValueError, "hash changed"):
            self.load()

    def test_unrecorded_checkpoint_is_not_accepted_from_a_successful_source(self):
        self.unsealed_interrupted_source(status="POINT_COMPUTED")
        with self.assertRaisesRegex(ValueError, "explicit new manifest binding"):
            self.load()

    def test_explicit_checkpoint_cannot_replace_a_previously_recorded_checkpoint_hash(self):
        alternative = self.directory / "alternative.chk"
        alternative.write_bytes(b"Different source checkpoint with a valid new hash")
        with self.assertRaisesRegex(ValueError, "hash already recorded"):
            self.load(request={**self.request, "warmstart_checkpoint": worker.ref(alternative)})
        # Even a new explicit hash at the old path cannot replace the prior
        # source-result binding. Both legitimate-file/path cases are rejected.
        self.checkpoint.write_bytes(b"Changed bytes at the recorded checkpoint path")
        with self.assertRaisesRegex(ValueError, "hash already recorded"):
            self.load(request={**self.request, "warmstart_checkpoint": worker.ref(self.checkpoint)})


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="v5-solver-unit-", dir=worker.ROOT / "reports")
        self.checkpoint = Path(self.temp.name) / "source.chk"
        self.checkpoint.write_bytes(b"Manufactured source checkpoint: never read by a solver")

    def tearDown(self):
        self.temp.cleanup()

    def fake_mf(self, *, cdiis_converged=False, newton_converged=True):
        class MF:
            def __init__(self):
                self._eri, self.direct_scf, self.direct_scf_tol = None, True, 1e-15
                self.mol, self.grids = FakeMolecule(), object()
                self.converged, self.e_tot, self.cycles = False, -1., 0
                self.mo_coeff, self.mo_occ, self.mo_energy = np.eye(2), np.array([2., 0.]), np.array([-.5, .2])
                self.kernel_calls, self.newton_calls, self.dumps = [], [], []
            def kernel(self, **kwargs):
                self.kernel_calls.append(kwargs)
                self.converged, self.e_tot, self.cycles = cdiis_converged, -1.01, 3
                return self.e_tot
            def newton(self):
                base = self
                class Newton:
                    def __init__(self):
                        self._scf, self.mol, self.grids, self._eri = base, base.mol, base.grids, None
                    def kernel(self, **kwargs):
                        base.newton_calls.append((kwargs, self.conv_tol, self.conv_tol_grad,
                                                  self.max_cycle, self.ah_max_cycle, self.ah_lindep))
                        self.mo_coeff, self.mo_occ = np.array([[0., -1.], [1., 0.]]), np.array([2., 0.])
                        self.mo_energy, self.e_tot, self.converged = np.array([-.6, .3]), -1.02, newton_converged
                        return self.e_tot
                return Newton()
            def dump_chk(self, value):
                self.dumps.append(value.copy())
            def make_rdm1(self):
                return (self.mo_coeff * self.mo_occ) @ self.mo_coeff.T
            def get_hcore(self):
                return np.eye(2)
            def get_veff(self, *args):
                return np.zeros((2, 2))
            def energy_tot(self, *args):
                return -1.021
        return MF()

    def run_solver(self, mf, request=None):
        state, events = {}, []
        metadata = {"initial_guess_only": True, "checkpoint": worker.ref(self.checkpoint)}
        with patch.object(worker, "load_warmstart", return_value=(np.eye(2), np.array([2., 0.]), metadata)):
            energy = worker.run_warmstart_scf(mf, argparse.Namespace(), request or {}, state,
                                              lambda event, **kwargs: events.append((event, kwargs)), np)
        return energy, state, events

    def test_short_cdiis_and_real_newton_state_propagation_retain_tolerances(self):
        mf = self.fake_mf()
        energy, state, _ = self.run_solver(mf)
        self.assertEqual(len(mf.kernel_calls), 1)
        np.testing.assert_array_equal(mf.kernel_calls[0]["dm0"], np.diag([2., 0.]))
        self.assertEqual(len(mf.newton_calls), 1)
        _, energy_tol, gradient_tol, macro, ah, lindep = mf.newton_calls[0]
        self.assertEqual((energy_tol, gradient_tol, macro, ah, lindep), (1e-12, 1e-9, 12, 20, 1e-24))
        self.assertTrue(mf.converged)
        np.testing.assert_array_equal(mf.mo_coeff, [[0., -1.], [1., 0.]])
        self.assertEqual(energy, -1.021)
        self.assertEqual(len(state["SCF_solver_transitions"]), 2)
        self.assertEqual(state["SCF_final_state"]["final_fresh_Fock_qualification_threshold"], 1e-8)
        self.assertFalse(state["SCF_final_state"]["original_failure_reclassified"])

    def test_actual_unconverged_newton_state_stays_unconverged(self):
        mf = self.fake_mf(newton_converged=False)
        _, state, _ = self.run_solver(mf)
        self.assertFalse(mf.converged)
        self.assertFalse(state["SCF_final_state"]["reported_converged"])

    def test_converged_cdiis_skips_newton_and_direct_guard_fails_before_kernel(self):
        mf = self.fake_mf(cdiis_converged=True)
        _, state, _ = self.run_solver(mf)
        self.assertEqual(mf.newton_calls, [])
        self.assertEqual(len(state["SCF_solver_transitions"]), 1)
        mf = self.fake_mf(); mf._eri = np.eye(2)
        with self.assertRaisesRegex(ValueError, "screened-direct"):
            self.run_solver(mf)
        self.assertEqual(mf.kernel_calls, [])

    def test_unorthonormalized_direct_newton_mode_is_not_released(self):
        mf = self.fake_mf()
        with self.assertRaisesRegex(ValueError, "nonorthonormal"):
            self.run_solver(mf, {"scf_solver": "warmstart_newton"})
        self.assertEqual(mf.kernel_calls, [])
        self.assertEqual(mf.newton_calls, [])


class ProbeTests(unittest.TestCase):
    setUp = CheckpointTests.setUp
    tearDown = CheckpointTests.tearDown

    def test_probe_uses_two_fresh_direct_objects_and_never_claims_scf_convergence(self):
        created = []
        class Molecule(FakeMolecule):
            def set_geom_(self, geometry, unit=None):
                self.coordinates = np.asarray(geometry).copy()
        molecule = Molecule()
        class Grid:
            coords, weights = None, None
            def make_mask(self, mol, coords):
                return np.ones((1, 1), dtype=np.uint8)
        class MF:
            def __init__(self, settings):
                self.mol, self.grids = molecule, Grid()
                self.direct_scf_tol, self.direct_scf = settings["direct_scf_tol"], True
                self._eri, self.converged, self.small_rho_cutoff = None, False, 1e-7
            def kernel(self, *args, **kwargs):
                raise AssertionError("A stationarity probe must never run SCF")
            def get_hcore(self):
                return np.eye(2)
            def get_veff(self, mol, density):
                if self.grids.coords is None:
                    self.grids.coords, self.grids.weights = np.array([[1., 0., 0.]]), np.ones(1)
                return np.zeros((2, 2))
            def get_grad(self, *args):
                return np.array([1e-9 if self.direct_scf_tol == 1e-13 else 2e-9])
            def energy_tot(self, *args):
                return -1.001
        def make_scf(mol, settings, *args):
            mf = MF(settings); created.append(mf); return mf
        original = SimpleNamespace(molecule=lambda *a: molecule, make_scf=make_scf)
        source = {"arrays": {"geometry_bohr.npy": np.zeros((2, 3))}, "source": {"settings": self.settings},
                  "source_ref": worker.ref(self.frequency), "case": "water",
                  "geometry": {"atom_symbols": ["H", "H"], "coordinates_angstrom": [[0., 0., 0.], [0., 0., 0.]]}}
        request = {**self.request, "operation": "checkpoint_probe", "source_worker": worker.ref(self.program),
                   "probe_direct_scf_tolerances": [1e-13, 1e-15]}
        metadata = {"checkpoint": worker.ref(self.checkpoint), "target_geometry_identical_to_source": True,
                    "source_energy_hartree": -1.}
        output = self.directory / "probe"; output.mkdir(); state = {}
        with patch.object(worker, "original_worker", return_value=original), \
                patch.object(worker, "load_case", return_value=source), \
                patch.object(worker, "load_warmstart", return_value=(np.eye(2), np.array([2., 0.]), metadata)):
            worker.checkpoint_probe(self.args, request, state, output, lambda *a, **k: None)
        self.assertEqual(len(created), 2)
        self.assertIsNot(created[0], created[1])
        self.assertEqual([mf.direct_scf_tol for mf in created], [1e-13, 1e-15])
        self.assertTrue(all(mf.small_rho_cutoff == 0 and mf.converged is False for mf in created))
        self.assertEqual(state["status"], "CHECKPOINT_PROBE_COMPLETE")
        self.assertFalse(state["SCF_recomputed"])
        self.assertFalse(state["scf_converged"])
        self.assertFalse(state["scf_qualified"])
        self.assertFalse(state["original_failure_reclassified"])
        rows = state["checkpoint_stationarity_observations"]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["orbital_stationarity_within_final_norm_threshold"] for row in rows))
        self.assertTrue(all(row["SCF_convergence_claimed"] is False for row in rows))
        self.assertEqual(rows[0]["grid_content_sha256"], rows[1]["grid_content_sha256"])

    def test_probe_source_contains_no_scf_kernel_or_fabricated_true_convergence_assignment(self):
        node = next(node for node in ast.parse(SOURCE.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "checkpoint_probe")
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                self.assertNotIn(child.func.attr, ("kernel", "newton", "Hessian"))
            if isinstance(child, ast.Assign) and isinstance(child.value, ast.Constant) and child.value.value is True:
                for target in child.targets:
                    self.assertFalse(isinstance(target, ast.Attribute) and target.attr == "converged")


def cases(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from cases(item)
        else:
            yield item


def suite():
    result = unittest.TestSuite()
    for test in cases(prior.suite()):
        # Recovery dispatches SCF execution through a reviewed warm-start solver
        # helper, so the v4 AST assertion of a literal point-local mf.kernel
        # call is superseded by recovery-specific provenance/solver tests.
        if test._testMethodName not in ("test_point_has_one_scf_call_and_response_has_no_scf_or_full_hessian_call",
                                       "test_center_and_displacement_are_unambiguous"):
            result.addTest(test)
    result.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryInvariantTests))
    result.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(CheckpointTests))
    result.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(SolverTests))
    result.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(ProbeTests))
    return result


if __name__ == "__main__":
    raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(suite()).wasSuccessful() else 1)
