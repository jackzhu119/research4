"""V7 resource/dispatch tests only; all matrices are tiny synthetic fixtures."""
import ast
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load("bounded_hessian_v7", ROOT / "scripts/pyscf_spinosin_bounded_hessian_worker_20261031.py")
v6 = load("bounded_hessian_prior_v6", ROOT / "scripts/pyscf_spinosin_memory_bounded_direct_worker_20261031.py")
fixture = load("bounded_hessian_synthetic_fixture", ROOT / "tests/test_pyscf_spinosin_checkpointed_direct_worker.py")


class BoundedHessianTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "outputs").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="v7-pure-fixture-", dir=ROOT / "outputs")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def settings(self, block=4, native=1):
        return worker.worker_settings(SimpleNamespace(grid_level=1, threads=4, maxsteps=35,
            hessian_block_atoms=block, native_direct_driver_threads=native))

    def obj(self, atoms=9):
        obj = fixture.FakeNativeHessian(atoms)
        obj.base.max_memory = obj.max_memory = 8000
        obj.base.xc, obj.base.mol.basis = "PBE", "synthetic_basis_fixture"
        return obj

    def run_capture(self, obj, block=8, native=1):
        self.factory_calls = []
        def factory(mf, coefficients, occupations):
            self.factory_calls.append((mf, coefficients, occupations))
            return lambda vector: vector
        return worker.run_checkpointed_hessian(obj, self.path / "cache", self.path / "manifest.json",
            self.settings(block, native), block_atoms=block,
            versions={"fixture": "NOT_SCIENTIFIC_DATA"},
            input_evidence={"synthetic_nonphysical_fixture": True},
            response_operator_factory=factory)

    def test_all_one_native_thread_block_sizes_are_accepted(self):
        for block in range(1, 9):
            worker.validate_hessian_resource_policy(block, 1)

    def test_native_two_is_rejected_for_large_blocks_before_any_computation(self):
        for block in range(1, 5):
            worker.validate_hessian_resource_policy(block, 2)
        for block in range(5, 9):
            with self.assertRaisesRegex(ValueError, "above4"):
                self.settings(block, native=2)

    def test_out_of_range_and_boolean_policies_rejected(self):
        for block in (0, 9, -1, True, 4.0):
            with self.assertRaises(ValueError):
                worker.validate_hessian_resource_policy(block, 1)
        for native in (0, 3, True, 1.0):
            with self.assertRaises(ValueError):
                worker.validate_hessian_resource_policy(4, native)

    def test_original_physical_settings_and_scf_budget_are_unchanged(self):
        args = SimpleNamespace(grid_level=1, threads=4, maxsteps=35,
            hessian_block_atoms=4, native_direct_driver_threads=1)
        previous, current = v6.worker_settings(args), worker.worker_settings(args)
        for key, value in previous.items():
            if key != "engineering_worker_version":
                self.assertEqual(current[key], value, key)
        self.assertEqual(current["max_memory_mb"], 8000)
        self.assertEqual(current["hessian_max_memory_mb"], 3000)
        self.assertEqual(current["density_fit_backend"], "SCREENED_DIRECT_RKS_V4")

    def test_memory_context_only_changes_two_resource_attributes_and_restores(self):
        obj = self.obj()
        before = (obj.base.grids.coords, obj.base.grids.weights, obj.base.mo_coeff,
                  obj.base.mo_energy, obj.base.mo_occ, obj.base.mol.basis, obj.base.xc)
        events = []
        with worker.hessian_memory_context(obj, self.settings(), emit=lambda event, meta: events.append(event)) as meta:
            self.assertEqual((obj.base.max_memory, obj.max_memory), (3000, 3000))
            self.assertIs(before[0], obj.base.grids.coords)
            self.assertIs(before[1], obj.base.grids.weights)
            self.assertIs(before[2], obj.base.mo_coeff)
        after = (obj.base.grids.coords, obj.base.grids.weights, obj.base.mo_coeff,
                 obj.base.mo_energy, obj.base.mo_occ, obj.base.mol.basis, obj.base.xc)
        self.assertTrue(all(a is b for a, b in zip(before, after)))
        self.assertEqual((obj.base.max_memory, obj.max_memory), (8000, 8000))
        self.assertEqual(meta["restoration_status"], "RESTORED")
        self.assertTrue(meta["same_grid_verified_after_Hessian"])
        self.assertEqual(events, ["hessian_memory_context_entered", "hessian_memory_context_exited"])

    def test_solver_exception_and_keyboard_interrupt_restore_memory(self):
        for exception in (RuntimeError("synthetic solver failure"), KeyboardInterrupt()):
            obj = self.obj()
            with self.assertRaises(type(exception)):
                with worker.hessian_memory_context(obj, self.settings()) as meta:
                    raise exception
            self.assertEqual((obj.base.max_memory, obj.max_memory), (8000, 8000))
            self.assertEqual(meta["restoration_status"], "RESTORED")
            self.assertNotIn("same_grid_verified_after_Hessian", meta)

    def test_callback_exception_also_restores_memory(self):
        obj = self.obj()
        def emit(event, meta):
            if event.endswith("entered"):
                raise RuntimeError("synthetic callback failure")
        with self.assertRaisesRegex(RuntimeError, "callback"):
            with worker.hessian_memory_context(obj, self.settings(), emit=emit):
                self.fail("Must not enter body")
        self.assertEqual((obj.base.max_memory, obj.max_memory), (8000, 8000))

    def test_unreviewed_or_uninitialized_context_is_rejected(self):
        obj = self.obj()
        settings = self.settings()
        settings["hessian_max_memory_mb"] = 6000
        with self.assertRaises(ValueError):
            with worker.hessian_memory_context(obj, settings):
                self.fail("Must not enter body")
        obj.base.grids.coords = None
        with self.assertRaises(ValueError):
            with worker.hessian_memory_context(obj, self.settings()):
                self.fail("Must not enter body")
        self.assertEqual((obj.base.max_memory, obj.max_memory), (8000, 8000))

    def test_grid_mutation_is_detected_and_memory_still_restored(self):
        obj = self.obj()
        with self.assertRaisesRegex(RuntimeError, "quadrature"):
            with worker.hessian_memory_context(obj, self.settings()):
                obj.base.grids.weights[0] = 3.
        self.assertEqual((obj.base.max_memory, obj.max_memory), (8000, 8000))

    def test_memory_mutation_inside_context_is_detected_then_restored(self):
        obj = self.obj()
        with self.assertRaisesRegex(RuntimeError, "resource budget"):
            with worker.hessian_memory_context(obj, self.settings()):
                obj.base.max_memory = 6000
        self.assertEqual((obj.base.max_memory, obj.max_memory), (8000, 8000))

    def test_inherited_hessian_memory_attribute_is_not_materialized_after_exit(self):
        obj = self.obj()
        class InheritedHessian:
            max_memory = 8000
        inherited = InheritedHessian()
        inherited.base = obj.base
        with worker.hessian_memory_context(inherited, self.settings()):
            self.assertIn("max_memory", inherited.__dict__)
        self.assertNotIn("max_memory", inherited.__dict__)
        self.assertEqual(inherited.max_memory, 8000)

    def test_nine_atom_eight_plus_one_dispatch_keeps_all_global_couplings(self):
        obj = self.obj()
        result = self.run_capture(obj)
        self.assertEqual([c[1] for c in obj.calls if c[0] == "solve"], [list(range(8)), [8]])
        self.assertEqual(len(self.factory_calls), 1)
        self.assertTrue(all(c[2] == id(obj.h1) for c in obj.calls if c[0] == "solve"))
        for i in range(9):
            for j in range(9):
                self.assertTrue(np.all(result[i, j] == (i+1)*(j+1)+(j+2)))
        manifest = json.loads((self.path / "manifest.json").read_text())
        self.assertEqual(manifest["response_complete_global_atom_ids"], list(range(9)))
        self.assertEqual(manifest["binding"]["actual_MF_max_memory_during_Hessian_mb"], 3000)
        self.assertEqual(manifest["binding"]["actual_Hessian_max_memory_mb"], 3000)
        self.assertEqual(manifest["cross_process_restore"], "NOT_IMPLEMENTED_SAVE_ONLY")
        self.assertEqual([r["requested_effective_CPHF_tolerance"] for r in manifest["response_blocks"]], [8e-8, 1e-8])
        self.assertEqual((obj.base.max_memory, obj.max_memory), (8000, 8000))

    def test_failed_later_block_preserves_partial_cache_and_restores_hooks_and_memory(self):
        obj = self.obj()
        obj.fail_block, obj.fail_kind = 2, "exception"
        with self.assertRaisesRegex(RuntimeError, "nonconvergence"):
            self.run_capture(obj)
        manifest = json.loads((self.path / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "PARTIAL")
        self.assertEqual(len(manifest["response_blocks"]), 1)
        self.assertTrue((self.path / "cache").exists())
        self.assertFalse(any(name in obj.__dict__ for name in ("partial_hess_elec", "make_h1", "solve_mo1")))
        self.assertEqual((obj.base.max_memory, obj.max_memory), (8000, 8000))

    def test_mismatched_recorded_response_block_cannot_create_cache(self):
        obj = self.obj()
        with self.assertRaisesRegex(ValueError, "recorded settings"):
            worker.run_checkpointed_hessian(obj, self.path / "cache", self.path / "manifest.json",
                self.settings(4), block_atoms=8)
        self.assertFalse((self.path / "cache").exists())
        self.assertEqual((obj.base.max_memory, obj.max_memory), (8000, 8000))

    def test_scientific_methods_mode_guards_and_native_thread_wrapper_unchanged(self):
        def functions(path):
            return {node.name: ast.dump(node, include_attributes=False)
                    for node in ast.parse(path.read_text()).body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        old = functions(ROOT / "scripts/pyscf_spinosin_memory_bounded_direct_worker_20261031.py")
        new = functions(ROOT / "scripts/pyscf_spinosin_bounded_hessian_worker_20261031.py")
        for name in ("screened_direct_rks_type", "direct_gradient_type", "direct_integral_policy",
                     "NativeDirectDriverThreadGuard", "require_optimized_result", "stationary_checks",
                     "frequencies_payload", "prepare_hessian_for_analysis", "frequency_minimum_checks"):
            self.assertEqual(old[name], new[name], name)


if __name__ == "__main__":
    unittest.main()
