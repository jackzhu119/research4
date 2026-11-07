"""Pure native-scope/proof orchestration; no QM or genuine supervisor proof."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("native4_large_block_validation", ROOT / "scripts/validate_pyscf_native4_large_block_engineering_20261031.py")
validation = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(validation)
FIXTURE_SPEC = importlib.util.spec_from_file_location("bounded_validation_test_fixtures", ROOT / "tests/test_pyscf_large_block_engineering.py")
fixtures = importlib.util.module_from_spec(FIXTURE_SPEC); FIXTURE_SPEC.loader.exec_module(fixtures)


class FakeGuard:
    def __init__(self, state, native_threads, fail_restore=False):
        self.state, self.native_threads, self.fail_restore = state, native_threads, fail_restore
        self.restored = False

    def __enter__(self):
        if self.state["active"]:
            raise RuntimeError("nested native scope")
        self.state["active"] = True
        self.state["entered"].append(self.native_threads)
        return self

    def __exit__(self, *args):
        self.state["active"] = False
        self.restored = not self.fail_restore
        return False

    def summary(self):
        return {"native_thread_limit": self.native_threads, "max_observed_native_threads": self.native_threads,
                "restoration_failures": int(self.fail_restore), "original_driver_restored": self.restored}


class Native4ValidationTests(unittest.TestCase):
    def test_same_trusted_source_contract_without_any_default_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            xyz, precise, provenance, _ = fixtures.source_fixture(Path(temp))
            symbols, coordinates, record = validation.verify_source(xyz, provenance, validation.sha(xyz), precise)
            self.assertEqual(len(symbols), 9); self.assertEqual(coordinates.shape, (9, 3))
            self.assertTrue(record["validation_only"])
            with self.assertRaises(ValueError): validation.verify_source(xyz, provenance, "wrong")

    def test_all_cases_share_mf_and_restore_sequential_native1_4_4_scopes(self):
        state = {"active": False, "entered": []}; mf = {"fingerprint": "fixed"}; seen = []; persisted = []
        def scope_factory(threads, label): return FakeGuard(state, threads)
        def evaluate_actual(actual, label, block, native_threads):
            self.assertTrue(state["active"]); seen.append((id(actual), label, block, native_threads))
            return fixtures.comparison_fixture()
        def evaluate(actual, label, block):
            return validation.evaluate_with_native_scope(actual, label, block, evaluate_actual, scope_factory)
        rows = validation.compare_same_state(mf, evaluate, lambda actual: actual["fingerprint"],
            lambda label, actual, record: persisted.append(record))
        self.assertEqual(state["entered"], [1, 4, 4]); self.assertFalse(state["active"])
        self.assertEqual([row[1] for row in seen], ["standard_full_native1", "response_block4_native4", "response_block8_native4"])
        self.assertEqual([row[2] for row in seen], [None, 4, 8])
        self.assertEqual(len({row[0] for row in seen}), 1)
        self.assertTrue(all(row["numerical_equivalence_passed"] for row in rows))
        self.assertEqual([row["native_direct_driver_threads"] for row in persisted], [1, 4, 4])

    def test_native_scope_restores_and_records_control_after_python_failure(self):
        state = {"active": False, "entered": []}; recorded = []
        def fail(*args): raise RuntimeError("synthetic Hessian error")
        with self.assertRaises(RuntimeError):
            validation.evaluate_with_native_scope(object(), "response_block8_native4", 8, fail,
                lambda count, label: FakeGuard(state, count), recorded.append)
        self.assertFalse(state["active"]); self.assertEqual(len(recorded), 1)
        self.assertTrue(recorded[0]["original_driver_restored"])
        self.assertEqual(recorded[0]["native_thread_limit"], 4)

    def test_failed_restoration_prevents_equivalence_result(self):
        state = {"active": False, "entered": []}; recorded = []
        with self.assertRaises(RuntimeError):
            validation.evaluate_with_native_scope(object(), "response_block4_native4", 4,
                lambda *args: fixtures.comparison_fixture(),
                lambda count, label: FakeGuard(state, count, fail_restore=True), recorded.append)
        self.assertEqual(recorded[0]["restoration_failures"], 1)

    def test_missing_or_invalid_parent_proof_blocks_before_quantum_site_setup(self):
        events = []
        def reject(**kwargs):
            events.append(("verify", kwargs["expected_threads"]))
            raise ValueError("synthetic missing parent proof")
        engine = SimpleNamespace(verify_resource_supervision=reject, install_sites=lambda: events.append(("install", None)))
        with self.assertRaises(ValueError): validation.prepare_supervised_runtime(4, lambda: engine)
        self.assertEqual(events, [("verify", 4)])

    def test_real_verifier_interface_precedes_runtime_setup_and_preserves_metadata(self):
        events = []; metadata = {"unit_test_only": True, "not_a_real_production_proof": True}
        def verify(**kwargs): events.append(("verify", kwargs["expected_threads"])); return metadata
        engine = SimpleNamespace(verify_resource_supervision=verify, install_sites=lambda: events.append(("install", None)))
        result, proof = validation.prepare_supervised_runtime(4, lambda: engine)
        self.assertIs(result, engine); self.assertIs(proof, metadata)
        self.assertEqual(events, [("verify", 4), ("install", None)])

    def test_summary_keeps_observed_native4_and_labels_configured_limits(self):
        summary = validation.combined_native_controls({"max_observed_native_threads": 1, "restoration_failures": 0},
            {"candidate": {"max_observed_native_threads": 4, "restoration_failures": 0}})
        self.assertEqual(summary["max_observed_native_threads"], 4)
        self.assertEqual(summary["native_thread_observation_kind"], "OMP_CONFIGURED_MAX_NOT_ACTIVE_WORKER_COUNT")

    def test_cross_block_hessian_difference_and_negative_minimum_remain_separate(self):
        first, second = fixtures.comparison_fixture(), fixtures.comparison_fixture()
        first["frequencies_cm1"][0] = second["frequencies_cm1"][0] = -3.
        self.assertTrue(validation.compare_case(first, second)["numerical_equivalence_passed"])
        self.assertFalse(validation.compare_case(first, second)["candidate_all21_frequencies_positive"])
        second["raw_hessian"][0, 8, 0, 0] += 2e-6
        self.assertFalse(validation.compare_case(first, second)["numerical_equivalence_passed"])

    def test_native4_validation_rejects_incompatible_outer_threads_before_qm(self):
        with tempfile.TemporaryDirectory() as temp:
            xyz, precise, provenance, _ = fixtures.source_fixture(Path(temp))
            args = SimpleNamespace(input_xyz=xyz, provenance_json=provenance, expected_input_sha256=validation.sha(xyz),
                                   input_coordinates_npy=precise, threads=2, maxsteps=10)
            with self.assertRaises(ValueError): validation.execute(args)


if __name__ == "__main__":
    unittest.main()
