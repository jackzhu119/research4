"""Metadata/engineering boundaries only; no quantum evaluations or array reads."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import json
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/pyscf_staged_consumer_20261105_v8.py"
OLD_SOURCE = ROOT / "scripts/pyscf_staged_consumer_20261105_v7.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


m = load("v8_consumer_stdlib", SOURCE)
old = load("v7_consumer_stdlib", OLD_SOURCE)


class ShapeObject:
    def __init__(self, shape):
        self.shape = shape


class ConsumerTests(unittest.TestCase):
    def _release_fixture(self, directory, *, audit_ir="NOT_COMPUTED", cases=None):
        """Synthetic metadata only; guard verification is deliberately mocked."""
        directory = Path(directory)
        def write(name, value):
            p = directory / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(value))
            return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        legacy = write("legacy-release.json", {"synthetic_fixture": True})
        guardian = write("guard-source.py", {"synthetic_guard_source": True})
        analyzer = write("analysis-source.py", {"synthetic_analysis_source": True})
        own_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        controls = []
        for case, n in cases or [("water_software_control", 3), ("ethanol_software_control", 9)]:
            closure = {"synthetic_fixture_closure": case}
            result = write(case + "/result.json", {"case": case, "normal_mode_count": 3*n-6,
                "source_worker": {}, "resource_supervision": {}, "snapshot_manifest_sha256": "snapshot"})
            checks = {"schema": "research4-V8-independent-staged-modes-analysis-v1",
                "status": "DONE_CONTROL_NUMERICAL_REPLAY", "case": case, "normal_modes": 3*n-6,
                "all_frequencies_positive": True, "normal_mode_count_complete": True,
                "IR_intensities_status": audit_ir, "qualified_for_mechanism": False,
                "new_quantum_evaluations": 0, "actual_parent_sha256": guardian["sha256"],
                "actual_V8_worker_sha256": own_sha, "analysis_source_sha256": analyzer["sha256"],
                "actual_old_stock_control_comparison": {
                    "status": "PASS_ACTUAL_V8_VS_AUTHENTICATED_V5_STOCK_FULL_MATRIX_AND_ALL_MODES"},
                "input_hashes": {result["path"]: {"sha256": result["sha256"]}},
                "source_result_sha256": result["sha256"], "producer_closure": closure}
            audit = write(case + "/checks.json", checks)
            index = write(case + "/artifact_index.json", {"V8_worker_sha256": own_sha,
                "analysis_source_sha256": analyzer["sha256"],
                "files": [{"path": "checks.json", "sha256": audit["sha256"]}]})
            controls.append({"case": case, "global_analysis_index": index})
        release = write("release.json", {"schema": m.CONTROL_RELEASE_SCHEMA,
            "root_execution_authorized": True, "v8_actual_water_ethanol_controls_reviewed": True,
            "no_new_budget_beyond_registered_v8": True, "legacy_v7_control_release": legacy,
            "code_refs": [{"path": str(SOURCE), "sha256": own_sha}, guardian, analyzer], "controls": controls})
        return release, guardian

    def test_import_is_stdlib_only(self):
        self.assertNotIn("numpy", m.__dict__)
        self.assertNotIn("pyscf", m.__dict__)

    def test_original_source_bytes_unchanged(self):
        self.assertEqual(hashlib.sha256(OLD_SOURCE.read_bytes()).hexdigest(), m.LEGACY_CONSUMER_SHA)

    def test_physical_binding_and_numerical_policy_not_relabelled(self):
        self.assertEqual(m.PHYSICAL_SCHEMA, old.PHYSICAL_SCHEMA)
        self.assertEqual(m.NUMERICAL_POLICY, old.NUMERICAL_POLICY)
        self.assertNotEqual(m.SCHEMA, old.SCHEMA)
        self.assertNotEqual(m.RESPONSE_SCHEMA, old.RESPONSE_SCHEMA)
        self.assertNotEqual(m.RESPONSE_INDEX_SCHEMA, old.RESPONSE_INDEX_SCHEMA)

    def test_B8_complete_original_atom_coverage(self):
        for n in (3, 9, 75):
            self.assertEqual(m.response_blocks(n), old.response_blocks(n))
            self.assertEqual(sum(m.response_blocks(n), []), list(range(n)))
        self.assertEqual(len(m.response_blocks(75)), 10)
        self.assertEqual(m.response_blocks(75)[-1], [72, 73, 74])

    def test_unknown_atom_counts_rejected(self):
        for n in (0, 8, 74, 76, True, 75.0):
            with self.assertRaises(ValueError):
                m.response_blocks(n)

    def test_observer_forwards_same_argument_once_and_same_return_object(self):
        inp, out = ShapeObject((24, 121920)), ShapeObject((24, 121920))
        called, events = [], []
        def native(value):
            called.append(value)
            return out
        observer = m.NativeVindObserver(native, lambda event, details: events.append((event, details)))
        self.assertIs(observer(inp), out)
        self.assertEqual(called, [inp])
        self.assertEqual(observer.started, 1)
        self.assertEqual(observer.completed, 1)
        self.assertEqual([e for e, _ in events], ["native_vind_call_begin", "native_vind_call_end"])
        self.assertEqual(observer.summary()["residual_norm"], "NOT_COMPUTED")
        self.assertTrue(observer.summary()["calls_are_not_Krylov_iteration_counts"])

    def test_observer_preserves_native_exception_and_no_completion_claim(self):
        exc = RuntimeError("actual native error")
        events = []
        def native(value):
            raise exc
        observer = m.NativeVindObserver(native, lambda event, details: events.append(event))
        with self.assertRaises(RuntimeError) as caught:
            observer(ShapeObject((24, 121920)))
        self.assertIs(caught.exception, exc)
        self.assertEqual((observer.started, observer.completed, observer.failed), (1, 0, 1))
        self.assertEqual(events, ["native_vind_call_begin", "native_vind_call_failed"])

    def test_observer_keeps_multiple_call_indices(self):
        inp = ShapeObject((3, 18))
        events = []
        observer = m.NativeVindObserver(lambda value: value, lambda event, details: events.append((event, details)))
        self.assertIs(observer(inp), inp)
        self.assertIs(observer(inp), inp)
        self.assertEqual([d["call_index"] for e, d in events if e.endswith("begin")], [1, 2])

    def test_legacy_partial_is_qualified_by_original_guard_only(self):
        calls = []
        legacy_guard, live_guard = object(), object()
        producer = types.SimpleNamespace(SCHEMA=m.LEGACY_SCHEMA, PHYSICAL_SCHEMA=m.PHYSICAL_SCHEMA,
            NUMERICAL_POLICY=m.NUMERICAL_POLICY, qualify_partial=lambda *args: calls.append(args) or "old-A")
        with patch.object(m, "legacy_modules", return_value=(producer, legacy_guard)):
            self.assertEqual(m.qualify_partial("r", "physical", 75, 762, live_guard), "old-A")
        self.assertIs(calls[0][-1], legacy_guard)
        self.assertIsNot(calls[0][-1], live_guard)

    def test_legacy_partial_policy_mismatch_rejected(self):
        producer = types.SimpleNamespace(SCHEMA=m.LEGACY_SCHEMA, PHYSICAL_SCHEMA=m.PHYSICAL_SCHEMA,
            NUMERICAL_POLICY={"xc": "different"})
        with patch.object(m, "legacy_modules", return_value=(producer, object())):
            with self.assertRaises(ValueError):
                m.qualify_partial({}, {}, 75, 762, object())

    def test_legacy_H1_delegates_full_original_qualifier_and_guard(self):
        calls = []
        old_guard, new_guard = object(), object()
        producer = types.SimpleNamespace(PHYSICAL_SCHEMA=m.PHYSICAL_SCHEMA, NUMERICAL_POLICY=m.NUMERICAL_POLICY,
            qualify_stage=lambda *args: calls.append(args) or "original-H1")
        with patch.object(m, "checked", return_value=Path("legacy-result.json")), \
                patch.object(m, "read", return_value={"schema": m.LEGACY_SCHEMA}), \
                patch.object(m, "legacy_modules", return_value=(producer, old_guard)):
            result = m.qualify_stage("ref", "h1ao", "physical", 75, 762, new_guard)
        self.assertEqual(result, "original-H1")
        self.assertEqual(calls[0][:-1], ("ref", "h1ao", "physical", 75, 762))
        self.assertIs(calls[0][-1], old_guard)

    def test_unknown_H1_schema_cannot_be_relabelled(self):
        with patch.object(m, "checked", return_value=Path("unknown.json")), \
                patch.object(m, "read", return_value={"schema": "foreign"}):
            with self.assertRaises(ValueError):
                m.qualify_stage("ref", "h1ao", {}, 75, 762, object())

    def test_H1_entry_does_not_accept_partial_or_response_stage(self):
        with self.assertRaises(ValueError):
            m.qualify_stage({}, "response-block", {}, 75, 762, object())

    def test_new_H1_uses_own_guard_and_complete_array_coverage(self):
        new_guard = object()
        value, manifest, closure = {"complete_H1AO": True}, {"includes_nuclear_hessian": False,
            "includes_CPHF_response": False, "arrays": []}, object()
        directory = Path("new-stage")
        with patch.object(m, "checked", return_value=directory / "result.json"), \
                patch.object(m, "read", return_value={"schema": m.SCHEMA}), \
                patch.object(m, "closed_stage", return_value=(value, directory, closure)) as closed, \
                patch.object(m, "cache_manifest", return_value=(manifest, directory)) as cached, \
                patch.object(m, "verify_arrays") as arrays:
            got = m.qualify_stage("ref", "h1ao", "physical", 75, 762, new_guard)
        self.assertEqual(got, (value, manifest, directory, closure))
        self.assertIs(closed.call_args.args[-1], new_guard)
        wants = arrays.call_args.args[-1]
        self.assertEqual(len(wants), 75)
        self.assertEqual(wants[0], {"key": "H1AO_atom_000", "shape": [3, 762, 762]})
        self.assertEqual(wants[-1]["key"], "H1AO_atom_074")

    def test_legacy_release_cannot_release_new_consumer(self):
        with patch.object(m, "checked", return_value=Path("old-release.json")), \
                patch.object(m, "read", return_value={"schema": "research4-v7-actual-small-control-release-v1"}):
            with self.assertRaises(ValueError):
                m.require_control_release("old-release", "hash", object())

    def test_preparation_release_without_root_actual_controls_rejected(self):
        with patch.object(m, "checked", return_value=Path("prepared-release.json")), \
                patch.object(m, "read", return_value={"schema": m.CONTROL_RELEASE_SCHEMA,
                    "root_execution_authorized": True, "v8_actual_water_ethanol_controls_reviewed": False,
                    "no_new_budget_beyond_registered_v8": True}):
            with self.assertRaises(ValueError):
                m.require_control_release("prepared", "hash", object())

    def test_valid_synthetic_release_uses_both_new_closures_and_separate_legacy_gate(self):
        calls, legacy_calls = [], []
        def verify(evidence, **kwargs):
            case = Path(evidence["output_directory"]).name
            calls.append((evidence, kwargs))
            return {"synthetic_fixture_closure": case}
        partial = types.SimpleNamespace(require_control_release=lambda *a, **kw: legacy_calls.append((a, kw)))
        with tempfile.TemporaryDirectory() as tmp:
            release, guardian = self._release_fixture(tmp)
            with patch.object(m, "GUARD", Path(guardian["path"])), patch.object(m, "GUARD_SHA", guardian["sha256"]), \
                    patch.object(m, "partial_worker", return_value=partial):
                value = m.require_control_release(release["path"], release["sha256"],
                    types.SimpleNamespace(verify_closed_phase=verify))
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(legacy_calls), 1)
        self.assertEqual(legacy_calls[0][1]["consumer_source"]["sha256"], m.LEGACY_CONSUMER_SHA)
        self.assertEqual(value["schema"], m.CONTROL_RELEASE_SCHEMA)
        for _, kwargs in calls:
            self.assertEqual(kwargs["expected_result_schema"], m.SCHEMA)
            self.assertEqual(kwargs["expected_worker_path"], SOURCE)
            self.assertEqual(kwargs["allowed_result_statuses"], ("FREQUENCIES_COMPUTED",))

    def test_synthetic_audit_cannot_invent_IR_intensities(self):
        partial = types.SimpleNamespace(require_control_release=lambda *a, **kw: None)
        with tempfile.TemporaryDirectory() as tmp:
            release, guardian = self._release_fixture(tmp, audit_ir="COMPUTED")
            with patch.object(m, "GUARD", Path(guardian["path"])), patch.object(m, "GUARD_SHA", guardian["sha256"]), \
                    patch.object(m, "partial_worker", return_value=partial):
                with self.assertRaises(ValueError):
                    m.require_control_release(release["path"], release["sha256"], object())

    def test_synthetic_failed_new_global_guard_cannot_release_target(self):
        partial = types.SimpleNamespace(require_control_release=lambda *a, **kw: None)
        def failure(*args, **kwargs):
            raise ValueError("new producer failed closure")
        with tempfile.TemporaryDirectory() as tmp:
            release, guardian = self._release_fixture(tmp)
            with patch.object(m, "GUARD", Path(guardian["path"])), patch.object(m, "GUARD_SHA", guardian["sha256"]), \
                    patch.object(m, "partial_worker", return_value=partial):
                with self.assertRaisesRegex(ValueError, "failed closure"):
                    m.require_control_release(release["path"], release["sha256"],
                        types.SimpleNamespace(verify_closed_phase=failure))

    def test_native_numerical_function_bodies_unchanged(self):
        def functions(path):
            return {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
        left, right = functions(OLD_SOURCE), functions(SOURCE)
        for name in ("response_blocks", "bind_actual_native_modules", "verify_arrays", "cache_manifest",
                     "qualify_response", "qualify_response_index", "AssemblyProviders"):
            self.assertEqual(ast.dump(left[name], include_attributes=False),
                ast.dump(right[name], include_attributes=False), name)

    def test_physical_dictionary_and_full_native_numeric_calls_unchanged(self):
        def extract(path):
            tree = ast.parse(path.read_text())
            execute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "execute")
            wanted = {"hobj.make_h1", "hobj.solve_mo1", "rhf.gen_vind", "hobj.hess_elec", "hobj.hess_nuc",
                      "v5.publish_modes", "core.runtime_state_fingerprint"}
            calls = sorted(ast.dump(n, include_attributes=False) for n in ast.walk(execute)
                if isinstance(n, ast.Call) and ast.unparse(n.func) in wanted)
            physical = next(n.value for n in ast.walk(execute) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "physical" for t in n.targets))
            return calls, ast.dump(physical, include_attributes=False)
        self.assertEqual(extract(OLD_SOURCE), extract(SOURCE))


if __name__ == "__main__":
    unittest.main()
