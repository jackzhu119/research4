"""Synthetic tiny Hessian orchestration fixtures only; no PySCF quantum jobs."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/pyscf_spinosin_checkpointed_direct_worker_20261031.py"
spec = importlib.util.spec_from_file_location("checkpointed_direct", SCRIPT)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class FakeMol:
    def __init__(self, atoms):
        self.natm, self.incore_anyway = atoms, False
        self.coords = np.arange(atoms*3, dtype=float).reshape(atoms, 3)
    def atom_coords(self, unit):
        assert unit == "Bohr"
        return self.coords.copy()
    def dumps(self):
        return json.dumps({"synthetic_nonphysical_test_fixture": True, "coords": self.coords.tolist()})


class FakeNativeHessian:
    """Not PySCF or physical data: prove dispatch/indexing/failure mechanics."""
    def __init__(self, atoms=5):
        self.mol = FakeMol(atoms)
        self.base = SimpleNamespace(mol=self.mol, converged=True, e_tot=-1., _eri=None,
            direct_scf=True, direct_scf_tol=1e-13, conv_tol_cpscf=1e-8,
            mo_coeff=np.eye(4), mo_energy=np.array([-1., -.5, .2, .5]), mo_occ=np.array([2., 2., 0., 0.]),
            grids=SimpleNamespace(coords=np.zeros((2, 3)), weights=np.ones(2)))
        self.grids, self.max_cycle, self.level_shift = None, 50, 0
        self.calls, self.fail_block, self.fail_kind = [], None, None
        self.provided_fx = None
        self.h1 = None
    def partial_hess_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None, max_memory=4000, verbose=None):
        self.calls.append(("partial", list(atmlst)))
        return np.zeros((self.mol.natm, self.mol.natm, 3, 3))
    def make_h1(self, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
        self.calls.append(("H1", list(atmlst)))
        self.h1 = [np.full((3, 4, 4), float(i+1)) for i in range(self.mol.natm)]
        return self.h1
    def solve_mo1(self, mo_energy, mo_coeff, mo_occ, h1ao, fx=None, atmlst=None, max_memory=4000, verbose=None):
        self.calls.append(("solve", list(atmlst), id(h1ao), fx, max_memory))
        block_index = len([c for c in self.calls if c[0] == "solve"])
        if block_index == self.fail_block:
            if self.fail_kind == "exception":
                raise RuntimeError("Synthetic native CPHF nonconvergence")
            if self.fail_kind == "missing_global_entries":
                return [], []
        first, second = [None]*self.mol.natm, [None]*self.mol.natm
        for atom in atmlst:
            first[atom] = np.full((3, 4, 2), float(atom+1))
            second[atom] = np.full((3, 2, 2), float(atom+2))
        if block_index == self.fail_block and self.fail_kind == "bad_shape":
            first[atmlst[0]] = np.zeros((2, 4, 2))
        if block_index == self.fail_block and self.fail_kind == "nan":
            first[atmlst[0]][0, 0, 0] = np.nan
        return first, second
    def kernel(self):
        self.calls.append(("kernel",))
        atoms = list(range(self.mol.natm))
        mo_energy, mo_coeff, mo_occ = self.base.mo_energy, self.base.mo_coeff, self.base.mo_occ
        partial = self.partial_hess_elec(mo_energy, mo_coeff, mo_occ, atoms, 4000, None)
        h1 = self.make_h1(mo_coeff, mo_occ, None, atoms, None)
        mo1, moe1 = self.solve_mo1(mo_energy, mo_coeff, mo_occ, h1, self.provided_fx, atoms, 4000, None)
        # Include all cross-block atom pairs in a tiny artificial assembly.
        for i in atoms:
            for j in atoms:
                partial[i, j] = h1[i].mean()*mo1[j].mean() + moe1[j].mean()
        return partial


class CheckpointedDirectTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "outputs").mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="v5-pure-fixture-", dir=ROOT / "outputs")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.factory_calls = []
    def settings(self, block=2):
        return worker.worker_settings(SimpleNamespace(grid_level=1, threads=1, maxsteps=35, hessian_block_atoms=block))
    def factory(self, mf, coeff, occ):
        self.factory_calls.append((mf, coeff, occ))
        return lambda vector: vector
    def run_capture(self, obj=None, block=2, **kwargs):
        return worker.run_checkpointed_hessian(obj or FakeNativeHessian(), self.path/"cache", self.path/"manifest.json",
            self.settings(block), block_atoms=block, versions={"fixture": "not_scientific_data"},
            native_lindep=1e-13, input_evidence={"synthetic_fixture": True},
            response_operator_factory=kwargs.get("factory", self.factory))
    def manifest(self):
        return json.loads((self.path/"manifest.json").read_text())

    def test_global_native_kernel_and_cross_block_terms_are_preserved(self):
        obj = FakeNativeHessian()
        result = self.run_capture(obj)
        self.assertEqual([x[1] for x in obj.calls if x[0] == "solve"], [[0, 1], [2, 3], [4]])
        self.assertEqual(len([x for x in obj.calls if x[0] == "kernel"]), 1)
        self.assertEqual([x[1] for x in obj.calls if x[0] in ("partial", "H1")], [list(range(5))]*2)
        self.assertTrue(all(x[2] == id(obj.h1) for x in obj.calls if x[0] == "solve"))
        for i in range(5):
            for j in range(5):
                np.testing.assert_array_equal(result[i, j], np.full((3, 3), (i+1)*(j+1)+(j+2)))
        self.assertEqual(self.manifest()["response_complete_global_atom_ids"], list(range(5)))
        self.assertTrue(self.manifest()["global_Hessian_assembly_unchanged"])
        self.assertFalse(self.manifest()["partial_kernel_tiling_used"])

    def test_one_shared_full_response_operator_or_existing_fx_is_reused(self):
        obj = FakeNativeHessian()
        self.run_capture(obj)
        self.assertEqual(len(self.factory_calls), 1)
        functions = [x[3] for x in obj.calls if x[0] == "solve"]
        self.assertTrue(all(f is functions[0] for f in functions))
        self.assertTrue(self.manifest()["response_operator"]["shared_fx_created_once"])

    def test_existing_response_operator_does_not_create_another(self):
        obj = FakeNativeHessian();obj.provided_fx = lambda x: x
        self.run_capture(obj)
        self.assertEqual(self.factory_calls, [])
        self.assertTrue(all(x[3] is obj.provided_fx for x in obj.calls if x[0] == "solve"))
        self.assertTrue(self.manifest()["response_operator"]["caller_supplied_fx_reused"])

    def test_real_fixture_arrays_are_hashed_and_only_save_no_resume(self):
        self.run_capture()
        manifest = self.manifest()
        self.assertEqual(manifest["status"], "DONE_INTERMEDIATE_CAPTURE_ONLY")
        self.assertEqual(manifest["cross_process_restore"], "NOT_IMPLEMENTED_SAVE_ONLY")
        self.assertFalse(manifest["resume_command_created"])
        self.assertFalse(manifest["frequency_validation_performed_by_this_capture"])
        records = [manifest["partial_hessian"]["array"]] + [x["array"] for x in manifest["H1AO_atoms"]]
        records += [a[k] for block in manifest["response_blocks"] for a in block["arrays"] for k in ("MO1", "MO_E1")]
        self.assertEqual(len(records), 1+5+10)
        for value in records:
            p = Path(value["path"])
            self.assertEqual(worker.sha(p), value["file_sha256"])
            self.assertEqual(p.stat().st_size, value["file_bytes"])
            self.assertEqual(worker.array_fingerprint(np.load(p))["payload_sha256_C_order"], value["payload_sha256_C_order"])

    def test_requested_tolerance_and_lindep_do_not_fabricate_residual(self):
        self.run_capture()
        blocks = self.manifest()["response_blocks"]
        self.assertEqual([x["requested_effective_CPHF_tolerance"] for x in blocks], [2e-8, 2e-8, 1e-8])
        for block in blocks:
            self.assertEqual(block["native_krylov_lindep"], 1e-13)
            self.assertEqual(block["residual_norm"], "NOT_COMPUTED")
            self.assertTrue(block["native_solver_returned_normally"])
            self.assertTrue(block["normal_return_does_not_prove_measured_residual_below_requested_tolerance"])

    def test_native_failure_preserves_completed_blocks_and_reraises(self):
        obj = FakeNativeHessian();obj.fail_block=2;obj.fail_kind="exception"
        with self.assertRaisesRegex(RuntimeError, "Synthetic native"):
            self.run_capture(obj)
        result = self.manifest()
        self.assertEqual(result["status"], "PARTIAL")
        self.assertEqual(len(result["H1AO_atoms"]), 5)
        self.assertEqual([x["global_atom_ids"] for x in result["response_blocks"]], [[0, 1]])
        self.assertTrue(Path(result["response_blocks"][0]["arrays"][0]["MO1"]["path"]).exists())
        self.assertNotIn("response_complete_global_atom_ids", result)
        self.assertIs(obj.solve_mo1.__func__, FakeNativeHessian.solve_mo1)

    def test_bad_native_response_shapes_global_indices_and_nan_are_rejected(self):
        for failure in ("missing_global_entries", "bad_shape", "nan"):
            with self.subTest(failure=failure):
                # Each synthetic attempt uses its own unique cache/output.
                self.path = Path(self.tmp.name)/failure;self.path.mkdir()
                obj = FakeNativeHessian();obj.fail_block=1;obj.fail_kind=failure
                with self.assertRaises(RuntimeError): self.run_capture(obj)
                self.assertEqual(self.manifest()["status"], "PARTIAL")
                self.assertEqual(self.manifest()["response_blocks"], [])

    def test_hooks_restore_original_instance_and_class_methods_on_success(self):
        obj = FakeNativeHessian()
        original = obj.make_h1
        obj.make_h1 = original
        self.run_capture(obj)
        self.assertIs(obj.__dict__["make_h1"], original)
        self.assertNotIn("partial_hess_elec", obj.__dict__)
        self.assertNotIn("solve_mo1", obj.__dict__)
        self.assertIs(obj.kernel.__func__, FakeNativeHessian.kernel)

    def test_orbital_gauge_change_cannot_reuse_same_energy_binding(self):
        class Rotating(FakeNativeHessian):
            def make_h1(self, *args, **kwargs):
                value = super().make_h1(*args, **kwargs)
                self.base.mo_coeff[0, 0] *= -1
                return value
        with self.assertRaisesRegex(RuntimeError, "orbital gauge"):
            self.run_capture(Rotating())
        self.assertEqual(self.manifest()["status"], "PARTIAL")
        self.assertEqual(self.manifest()["H1AO_atoms"], [])

    def test_geometry_or_grid_change_is_not_accepted(self):
        class Moving(FakeNativeHessian):
            def partial_hess_elec(self, *args, **kwargs):
                value = super().partial_hess_elec(*args, **kwargs)
                self.mol.coords[0, 0] += .01
                return value
        with self.assertRaises(RuntimeError): self.run_capture(Moving())
        self.assertIsNone(self.manifest()["partial_hessian"])

    def test_numerical_settings_cannot_change_inside_operator_factory(self):
        def bad(mf, coeff, occ):
            mf.conv_tol_cpscf=1e-6
            return lambda x: x
        with self.assertRaisesRegex(RuntimeError, "numerical settings"):
            self.run_capture(factory=bad)
        self.assertEqual(self.manifest()["response_blocks"], [])

    def test_global_subset_kernel_is_not_tiled_or_accepted(self):
        class Subset(FakeNativeHessian):
            def kernel(self):
                return self.partial_hess_elec(None, None, None, [0, 1])
        with self.assertRaisesRegex(ValueError, "every atom"):
            self.run_capture(Subset())
        self.assertEqual(self.manifest()["status"], "PARTIAL")

    def test_bad_block_size_and_changed_native_parameters_rejected(self):
        for block in (0, 5):
            with self.assertRaises(ValueError): self.run_capture(block=block)
        for attr, value in (("max_cycle", 100), ("level_shift", .1)):
            obj=FakeNativeHessian();setattr(obj, attr, value)
            with self.assertRaises(ValueError): self.run_capture(obj)
        obj=FakeNativeHessian();obj.base.conv_tol_cpscf=1e-6
        with self.assertRaises(ValueError): self.run_capture(obj)

    def test_cache_and_manifest_never_overwritten_and_no_big_report_cache(self):
        self.run_capture()
        before=(self.path/"manifest.json").read_bytes()
        with self.assertRaises(FileExistsError): self.run_capture()
        self.assertEqual((self.path/"manifest.json").read_bytes(), before)
        outside=Path(tempfile.gettempdir())/"must-not-write-v5-cache"
        with self.assertRaises(ValueError):
            worker.run_checkpointed_hessian(FakeNativeHessian(), outside, self.path/"other.json", self.settings())
        self.assertFalse(outside.exists())

    def test_disk_headroom_blocked_before_cache_is_created(self):
        with patch.object(worker.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
            with self.assertRaises(RuntimeError): self.run_capture()
        self.assertFalse((self.path/"cache").exists())

    def test_no_unbuilt_grid_unconverged_or_open_shell_scf_can_bind(self):
        for key, value in (("converged", False), ("mo_occ", np.ones(4))):
            obj=FakeNativeHessian();setattr(obj.base, key, value)
            with self.assertRaises(ValueError): self.run_capture(obj)
        obj=FakeNativeHessian();obj.base.grids.coords=None
        with self.assertRaises(ValueError): self.run_capture(obj)

    def test_v4_optimized_parent_remains_same_integral_method_for_v5(self):
        p, coords=self.path/"geometry.xyz", self.path/"geometry.npy"
        p.write_text("synthetic bytes");worker.atomic_npy(coords, np.ones((3, 3)))
        physical=self.settings()
        # V4 has no V5 orchestration fields, but every physical/numerical
        # optimization criterion is identical and must stay bound.
        for name in ("engineering_worker_version", "hessian_response_block_atoms", "hessian_conv_tol_cpscf", "hessian_max_cycle", "hessian_level_shift"):
            physical.pop(name)
        result=dict(status="DONE", geometry_optimized=True, optimizer_converged=True, scf_converged=True,
                    final_gradient_thresholds_passed=True, optimized_xyz_sha256=worker.sha(p),
                    optimized_coordinates_angstrom_npy_sha256=worker.sha(coords), settings=physical)
        worker.require_optimized_result(result, p, self.settings(4), coords)
        result["settings"]={**physical,"density_fitting":True,"density_fit_backend":"DF_V3"}
        with self.assertRaises(ValueError):worker.require_optimized_result(result,p,self.settings(),coords)

    def test_source_retains_real_standard_hessian_class_and_has_no_load_resume(self):
        text=SCRIPT.read_text();tree=ast.parse(text)
        self.assertIn('type(hobj).__module__ != "pyscf.hessian.rks"', text)
        self.assertIn('response_operator_factory=rhf_hessian.gen_vind', text)
        self.assertFalse(any(isinstance(n, ast.ClassDef) and "Hessian" in n.name and n.name!="HessianIntermediateCapture" for n in ast.walk(tree)))
        self.assertFalse(any(isinstance(n, ast.FunctionDef) and n.name in {"restore", "resume", "load_intermediates"} for n in ast.walk(tree)))

    def test_mole_identity_ignores_only_standard_rinv_bookkeeping_without_integrals(self):
        worker.install_sites()
        try:
            from pyscf import gto
        except ImportError:
            self.skipTest("Optional PySCF site unavailable")
        mol=gto.M(atom="O 0 0 .2; H 0 .7 1; H 0 -.7 1",basis="sto-3g",verbose=0)
        before=mol.dumps()
        identity=worker.molecule_cache_identity(mol)
        with mol.with_rinv_at_nucleus(1):
            self.assertNotEqual(mol.dumps(),before)
        # PySCF intentionally leaves last nucleus env[17], while restoring the
        # physical rinv origin. Canonical binding must not hide other changes.
        self.assertNotEqual(mol.dumps(),before)
        self.assertEqual(worker.molecule_cache_identity(mol),identity)
        self.assertEqual(mol._env[17],1)
        from pyscf.gto import mole
        mol._env[mole.PTR_RANGE_OMEGA]=.2
        self.assertNotEqual(worker.molecule_cache_identity(mol),identity)

    def test_ecp_bookkeeping_identity_is_explicitly_blocked(self):
        class ECPFixture:
            natm=3
            def dumps(self):return json.dumps({"_env":[0.]*20})
            def has_ecp(self):return True
        with self.assertRaises(ValueError):worker.molecule_cache_identity(ECPFixture())


if __name__ == "__main__":
    unittest.main()
