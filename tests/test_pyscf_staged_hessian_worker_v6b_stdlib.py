"""Stdlib-only V6B preparation tests. No numerical imports, arrays or QM."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/pyscf_staged_hessian_worker_20261105_v6b.py"
before = set(sys.modules)
spec = importlib.util.spec_from_file_location("staged_v6b_stdlib_only", SOURCE)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
new_imports = set(sys.modules)-before


class FakeHessian:
    def partial_hess_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None, max_memory=4000, verbose=None):
        raise AssertionError("No electronic structure in pure fixture")

    def make_h1(self, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
        raise AssertionError("No electronic structure in pure fixture")

    def hess_nuc(self, mol=None, atmlst=None):
        return 7


class StagedV6BStdlibTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="stdlib-v6b-stage-", dir=ROOT / "reports")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def physical(self):
        return {"schema": worker.PHYSICAL_SCHEMA, "snapshot_manifest_sha256": "s"*64, "fixture": "metadata only, not chemistry"}

    def manifest(self, stage="rks-partial", n=3, nao=4):
        specs = [("full_RKS_partial", [n, n, 3, 3])] if stage == "rks-partial" else [(f"H1AO_atom_{i:03d}", [3, nao, nao]) for i in range(n)]
        return {"schema": worker.CACHE_SCHEMA, "stage": stage, "complete": True,
                "physical_binding": self.physical(), "physical_binding_sha256": worker.digest(self.physical()),
                "arrays": [{"key": key, "file": key+".npy", "shape": shape, "dtype": "float64",
                            "file_sha256": "a"*64, "payload": {"fixture": True}} for key, shape in specs],
                "includes_nuclear_hessian": False, "includes_CPHF_response": False}

    def test_import_is_stdlib_only_and_never_imports_numeric_or_quantum(self):
        self.assertFalse(any(n.split(".")[0] in {"numpy", "scipy", "pyscf", "h5py"} for n in new_imports))
        tree = ast.parse(SOURCE.read_text())
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertTrue(all(a.name.split(".")[0] not in {"numpy", "scipy", "pyscf", "h5py"} for a in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or "").split(".")[0], {"numpy", "scipy", "pyscf", "h5py"})

    def test_actual_parent_verification_precedes_numerical_imports(self):
        function = next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "execute")
        gate = [n.lineno for n in ast.walk(function) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "verify_resource_supervision"]
        numeric = [n.lineno for n in ast.walk(function) if isinstance(n, ast.Import) and any(a.name == "numpy" for a in n.names)]
        self.assertEqual(len(gate), 1); self.assertEqual(len(numeric), 1)
        self.assertLess(gate[0], numeric[0])

    def test_full_atom_order_rejects_subsets_permutations_duplicates(self):
        self.assertEqual(worker.complete_atoms(None, 3), [0, 1, 2])
        self.assertEqual(worker.complete_atoms(range(3), 3), [0, 1, 2])
        for bad in ([0, 2], [2, 1, 0], [0, 1, 1]):
            with self.assertRaises(ValueError):
                worker.complete_atoms(bad, 3)

    def test_default_target_and_explicit_control_labels(self):
        self.assertEqual(worker.case_for(["O", "H", "H"], True), "water_software_control")
        with self.assertRaises(ValueError):
            worker.case_for(["O", "H", "H"], False)
        with self.assertRaises(ValueError):
            worker.case_for(["C"]*28+["H"]*32+["O"]*15, True)

    def test_complete_partial_and_all_atom_h1_metadata(self):
        for stage in ("rks-partial", "h1ao"):
            worker.validate_manifest(self.manifest(stage), stage=stage, physical_binding=self.physical(), atoms=3, nao=4)

    def test_cache_rejects_missing_atom_bad_shape_binding_double_count_terms(self):
        for change in ("missing", "shape", "state", "nuclear", "response", "schema", "duplicate", "traversal"):
            value = self.manifest("h1ao")
            if change == "missing": value["arrays"].pop()
            elif change == "shape": value["arrays"][0]["shape"] = [3, 3, 4]
            elif change == "state": value["physical_binding"] = {"wrong": True}
            elif change == "nuclear": value["includes_nuclear_hessian"] = True
            elif change == "response": value["includes_CPHF_response"] = True
            elif change == "schema": value["schema"] = "old V5 save-only native data"
            elif change == "duplicate": value["arrays"][1]["file"] = value["arrays"][0]["file"]
            else: value["arrays"][0]["file"] = "../foreign.npy"
            with self.subTest(change=change), self.assertRaises(ValueError):
                worker.validate_manifest(value, stage="h1ao", physical_binding=self.physical(), atoms=3, nao=4)

    def test_stock_inplace_assembly_cannot_contaminate_source_cache(self):
        partial, h1 = [[10]], [[[20]], [[30]], [[40]]]
        obj = FakeHessian()
        with worker.CachedStageProviders(obj, lambda: copy.deepcopy(partial), lambda: copy.deepcopy(h1), lambda *a: None) as capture:
            owned = obj.partial_hess_elec(); owned[0][0] += 5
            owned_h1 = obj.make_h1(None, None); owned_h1[0][0][0] = -1
            value = owned[0][0] + obj.hess_nuc()
            capture.require_once()
        self.assertEqual(value, 22)  # One electronic response + one nuclear term.
        self.assertEqual(partial, [[10]]); self.assertEqual(h1[0], [[20]])
        with worker.CachedStageProviders(obj, lambda: copy.deepcopy(partial), lambda: copy.deepcopy(h1), lambda *a: None):
            self.assertEqual(obj.partial_hess_elec(), [[10]])

    def test_repeated_contribution_rejected(self):
        obj = FakeHessian()
        with worker.CachedStageProviders(obj, lambda: [[1]], lambda: [], lambda *a: None):
            obj.partial_hess_elec()
            with self.assertRaisesRegex(ValueError, "repeated"):
                obj.partial_hess_elec()

    def test_provider_restores_absent_and_instance_methods_on_exception(self):
        obj = FakeHessian()
        own = types.MethodType(lambda self, mol=None, atmlst=None: 11, obj)
        obj.hess_nuc = own
        with self.assertRaisesRegex(RuntimeError, "synthetic stop"):
            with worker.CachedStageProviders(obj, lambda: [[1]], lambda: [], lambda *a: None):
                raise RuntimeError("synthetic stop")
        self.assertNotIn("partial_hess_elec", obj.__dict__)
        self.assertNotIn("make_h1", obj.__dict__)
        self.assertIs(obj.__dict__["hess_nuc"], own)

    def test_partial_provider_install_failure_restores_prior_instance_state(self):
        class InstallationFailure(FakeHessian):
            def __setattr__(self, name, value):
                if name == "make_h1":
                    raise RuntimeError("synthetic install failure")
                object.__setattr__(self, name, value)
        obj = InstallationFailure()
        before = dict(obj.__dict__)
        with self.assertRaisesRegex(RuntimeError, "install failure"):
            with worker.CachedStageProviders(obj, lambda: [[1]], lambda: [], lambda *a: None):
                self.fail("Installation must fail")
        self.assertEqual(obj.__dict__, before)

    def test_bound_default_memory_and_h1_chkfile_preserved_for_validator(self):
        calls = []
        obj = FakeHessian()
        with worker.CachedStageProviders(obj, lambda: [[1]], lambda: [], lambda kind, args: calls.append((kind, args))) as capture:
            obj.partial_hess_elec(); obj.make_h1("full MO fixture", "full occ fixture"); obj.hess_nuc()
            capture.require_once()
        self.assertEqual(calls[0][1]["max_memory"], 4000)
        self.assertEqual(calls[1][1]["chkfile"], None)
        self.assertEqual(worker.NUMERICAL_POLICY["MF_Hessian_memory_mb"], 3000)
        self.assertEqual(worker.NUMERICAL_POLICY["stock_hess_elec_max_memory_argument_mb"], 4000)

    def test_explicit_mo_memory_atom_and_chkfile_arguments_are_checked(self):
        mf = types.SimpleNamespace(mo_energy=[-1, 2], mo_coeff=[[1, 0], [0, 1]], mo_occ=[2, 0], mol=object())
        values = {"mo_energy": None, "mo_coeff": copy.deepcopy(mf.mo_coeff), "mo_occ": None,
                  "max_memory": 4000, "atmlst": [0, 1, 2]}
        same = lambda actual, expected: actual == expected
        worker.validate_cache_call("partial", values, 3, mf, same)
        for key, bad in (("mo_coeff", [[9, 0], [0, 1]]), ("mo_energy", [0, 0]),
                         ("mo_occ", [0, 2]), ("max_memory", 4001), ("atmlst", [2, 1, 0])):
            changed = copy.deepcopy(values); changed[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.validate_cache_call("partial", changed, 3, mf, same)
        with self.assertRaises(ValueError):
            worker.validate_cache_call("h1", {"mo_coeff": None, "mo_occ": None, "chkfile": "foreign.chk"}, 3, mf, same)
        with self.assertRaises(ValueError):
            worker.validate_cache_call("nuclear", {"mol": object()}, 3, mf, same)

    def test_no_quantum_or_ledger_is_initialized_by_preflight(self):
        output = self.base / "preflight"
        result = worker.preflight(output)
        self.assertEqual(result["status"], "NOT_RUN")
        self.assertFalse(result["actual_allocation_or_ledger_created"])
        self.assertFalse(result["target_arrays_or_numerical_packages_read"])
        self.assertEqual({p.name for p in output.iterdir()}, {"preflight.json"})
        with self.assertRaises(ValueError): worker.preflight(output)

    def test_guard_pending_blocks_execution_before_numerical_import(self):
        args = types.SimpleNamespace(threads=4)
        with patch.object(worker, "GUARD_SHA", "PENDING_TEST_UNFROZEN"), self.assertRaisesRegex(ValueError, "Unfrozen/changed"):
            worker.execute(args)
        self.assertNotIn("pyscf", sys.modules); self.assertNotIn("numpy", sys.modules)

    def test_running_parent_stops_before_cache_or_payload(self):
        directory = self.base / "closed"; directory.mkdir()
        state = {"schema": worker.SCHEMA, "stage": "rks-partial", "phase": "rks-partial", "status": "STAGE_COMPLETE",
                 "no_new_SCF_performed": True, "complete_RKS_partial": True, "atoms": 3, "basis_functions": 4,
                 "physical_binding_sha256": worker.digest(self.physical()), "source_worker": {}, "resource_supervision": {},
                 "snapshot_manifest_sha256": "s"*64}
        path = directory / "result.json"; path.write_text(json.dumps(state))
        def reject(*a, **kw): raise ValueError("synthetic parent RUNNING")
        guard = types.SimpleNamespace(verify_closed_phase=reject)
        with self.assertRaisesRegex(ValueError, "parent RUNNING"):
            worker.qualify_stage(worker.artifact(path), "rks-partial", self.physical(), 3, 4, guard)
        self.assertFalse((directory / "stage-cache").exists())

    def closed_text_fixture(self):
        """Hash-only stand-ins; these text bytes are explicitly NOT NPY arrays."""
        directory = self.base / "closed-text-fixture"; directory.mkdir()
        cache = directory / "stage-cache"; cache.mkdir()
        manifest = self.manifest()
        payload = cache / manifest["arrays"][0]["file"]
        payload.write_text("STDLIB_TEXT_FIXTURE_NOT_AN_ARRAY\n")
        manifest["arrays"][0]["file_sha256"] = worker.sha(payload)
        path = directory / "result.json"
        manifest["producer"] = {"result_path": str(path), "source_worker_sha256": worker.sha(SOURCE),
                                "supervisor_proof_sha256": "p"*64, "stage": "rks-partial"}
        manifest_path = cache / "manifest.json"; manifest_path.write_text(json.dumps(manifest))
        state = {"schema": worker.SCHEMA, "stage": "rks-partial", "phase": "rks-partial", "status": "STAGE_COMPLETE",
                 "no_new_SCF_performed": True, "complete_RKS_partial": True, "atoms": 3, "basis_functions": 4,
                 "physical_binding_sha256": worker.digest(self.physical()), "source_worker": {},
                 "resource_supervision": {"proof_sha256": "p"*64}, "snapshot_manifest_sha256": "s"*64,
                 "stable_MF_state_unchanged": True, "native_direct_driver_control": {"original_driver_restored": True, "restoration_failures": 0},
                 "cache_manifest_sha256": worker.sha(manifest_path)}
        path.write_text(json.dumps(state))
        guard = types.SimpleNamespace(verify_closed_phase=lambda *a, **kw: {"producer_result_sha256": worker.sha(path)})
        return path, payload, guard

    def test_healthy_metadata_qualifies_only_hashes_text_fixture_without_numeric_imports(self):
        path, _, guard = self.closed_text_fixture()
        value = worker.qualify_stage(worker.artifact(path), "rks-partial", self.physical(), 3, 4, guard)
        self.assertEqual(value[1]["stage"], "rks-partial")
        self.assertNotIn("numpy", sys.modules)

    def test_missing_or_changed_closed_payload_rejected_before_new_kernel(self):
        path, payload, guard = self.closed_text_fixture()
        payload.write_text("CORRUPT_TEXT_NOT_AN_ARRAY")
        with self.assertRaisesRegex(ValueError, "payload missing/changed"):
            worker.qualify_stage(worker.artifact(path), "rks-partial", self.physical(), 3, 4, guard)
        payload.unlink()
        with self.assertRaisesRegex(ValueError, "payload missing/changed"):
            worker.qualify_stage(worker.artifact(path), "rks-partial", self.physical(), 3, 4, guard)


if __name__ == "__main__":
    unittest.main()
