"""V8 pure startup/engineering guards; no quantum computations or integrals."""
import ast
from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load("supervised_parallel_v8", ROOT / "scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py")
v7 = load("supervised_parallel_prior_v7", ROOT / "scripts/pyscf_spinosin_bounded_hessian_worker_20261031.py")


class ParallelHessianTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="v8-pure-supervision-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        supervisor = self.path / "scripts/run_pyscf_monitored_budget_20261031.py"
        supervisor.parent.mkdir()
        supervisor.write_text("# synthetic supervisor identity; never executed\n")
        ledger = self.path / "ledger.json"
        ledger.write_text(json.dumps({"schema": "research4-pyscf-monitored-budget-v3",
                                      "phases": [{"phase": "synthetic_no_quantum", "status": "RUNNING"}]}))
        self.proof = {"schema": "research4-pyscf-resource-supervisor-proof-v3",
            "supervisor_pid": 101, "supervisor_start_ticks": 123,
            "child_pid": 102, "child_start_ticks": 124,
            "supervisor_script_path": str(supervisor), "supervisor_script_sha256": worker.sha(supervisor),
            "ledger_path": str(ledger), "phase": "synthetic_no_quantum",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "address_space_soft_limit_bytes": 24*1024**3, "address_space_hard_limit_bytes": 24*1024**3,
            "rss_stop_bytes": 10*1024**3, "cgroup_kill_bytes": 12*1024**3,
            "expected_cgroup_memory_max_bytes": 16*1024**3, "monitor_interval_seconds": .1,
            "memory_kill_grace_seconds": 1.0, "guard_semantics": "BEST_EFFORT_RSS_GUARD_NOT_HARD_RSS_LIMIT",
            "cpu_soft_limit_seconds": 100, "cpu_hard_limit_seconds": 160,
            "wall_limit_seconds": 180, "threads": 4}
        self.runtime = {key: self.proof[key] for key in
            ("child_pid", "supervisor_pid", "child_start_ticks", "supervisor_start_ticks")}
        self.runtime.update(parent_command=[b"python", str(supervisor).encode()], parent_cwd=str(self.path),
            address_space_limits=(24*1024**3,)*2, CPU_limits=(100, 160),
            child_cgroup="0::/synthetic\n", parent_cgroup="0::/synthetic\n",
            cgroup_memory_max_bytes=16*1024**3, cgroup_memory_current_bytes=1024**3)
        self.proof_path = self.path / "proof.json"

    def verify(self):
        self.proof_path.write_text(json.dumps(self.proof))
        with patch.object(worker, "ROOT", self.path), patch.object(worker, "resource_runtime_snapshot", return_value=self.runtime):
            return worker.verify_resource_supervision(self.proof_path, expected_threads=4)

    def settings(self, block=8, native=4, outer=4):
        return worker.worker_settings(SimpleNamespace(grid_level=1, threads=outer, maxsteps=35,
            hessian_block_atoms=block, native_direct_driver_threads=native))

    def test_every_reviewed_atom_and_native_thread_combination_is_allowed(self):
        for block in range(1, 9):
            for native in range(1, 5):
                worker.validate_hessian_resource_policy(block, native)
                worker.NativeDirectDriverThreadGuard(None, None, None, native_threads=native)

    def test_invalid_counts_and_native_above_outer_rejected(self):
        for block, native in ((0, 1), (9, 1), (True, 1), (4, 0), (4, 5), (4, True)):
            with self.assertRaises(ValueError):
                worker.validate_hessian_resource_policy(block, native)
        with self.assertRaisesRegex(ValueError, "outer allocation"):
            self.settings(native=4, outer=1)

    def test_verified_parent_proof_has_honest_as_and_rss_metadata(self):
        value = self.verify()
        self.assertTrue(value["live_parent_process_and_source_verified"])
        self.assertTrue(value["AS_is_virtual_address_cap_not_RSS_limit"])
        self.assertTrue(value["RSS_guard_is_best_effort_not_a_hard_limit"])
        self.assertTrue(value["startup_proof_is_not_future_monitor_health_confirmation"])
        self.assertEqual(value["proof_sha256"], worker.sha(self.proof_path))

    def test_missing_proof_stops_execute_before_imports_and_output_creation(self):
        with patch.dict(worker.os.environ, {}, clear=True), patch.object(worker, "install_sites", side_effect=AssertionError("Must not import quantum dependencies")):
            with self.assertRaisesRegex(ValueError, "requires"):
                worker.execute(SimpleNamespace(threads=4))

    def test_live_parent_and_child_pid_reuse_guards(self):
        for key in ("supervisor_pid", "child_pid", "supervisor_start_ticks", "child_start_ticks"):
            before = self.proof[key]
            self.proof[key] = before + 1
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                self.verify()
            self.proof[key] = before

    def test_actual_parent_command_and_source_are_verified(self):
        self.runtime["parent_command"] = [b"python", b"unrelated.py"]
        with self.assertRaisesRegex(ValueError, "Actual parent command"):
            self.verify()
        self.runtime["parent_command"] = [b"python", self.proof["supervisor_script_path"].encode()]
        self.proof["supervisor_script_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "source path/SHA"):
            self.verify()

    def test_actual_address_space_and_cpu_limits_must_match(self):
        self.runtime["address_space_limits"] = (12*1024**3,)*2
        with self.assertRaisesRegex(ValueError, "RLIMIT_AS"):
            self.verify()
        self.runtime["address_space_limits"] = (24*1024**3,)*2
        self.runtime["CPU_limits"] = (100, 999)
        with self.assertRaisesRegex(ValueError, "RLIMIT_CPU"):
            self.verify()

    def test_exact_resource_guard_thresholds_and_semantics_required(self):
        for key in ("rss_stop_bytes", "cgroup_kill_bytes", "expected_cgroup_memory_max_bytes"):
            before = self.proof[key]
            self.proof[key] += 1
            with self.assertRaisesRegex(ValueError, "Unreviewed resource"):
                self.verify()
            self.proof[key] = before
        self.proof["guard_semantics"] = "HARD_RSS_LIMIT"
        with self.assertRaisesRegex(ValueError, "must not be claimed"):
            self.verify()

    def test_slow_missing_or_nonfinite_monitor_timing_rejected(self):
        for key, bad in (("monitor_interval_seconds", .2), ("monitor_interval_seconds", None),
                         ("memory_kill_grace_seconds", 5), ("memory_kill_grace_seconds", True)):
            before = self.proof[key]
            self.proof[key] = bad
            with self.assertRaisesRegex(ValueError, "monitor timing"):
                self.verify()
            self.proof[key] = before

    def test_same_cgroup_and_actual_ceiling_and_current_checked(self):
        before = dict(self.runtime)
        for key, bad in (("parent_cgroup", "0::/other\n"), ("cgroup_memory_max_bytes", 32*1024**3),
                         ("cgroup_memory_current_bytes", 12*1024**3+1)):
            self.runtime[key] = bad
            with self.assertRaises(ValueError):
                self.verify()
            self.runtime = dict(before)

    def test_expected_threads_and_running_ledger_phase_must_match(self):
        self.proof["threads"] = 1
        with self.assertRaisesRegex(ValueError, "supervised allocation"):
            self.verify()
        self.proof["threads"] = 4
        ledger = Path(self.proof["ledger_path"])
        ledger.write_text(json.dumps({"schema": "research4-pyscf-monitored-budget-v3",
            "phases": [{"phase": "synthetic_no_quantum", "status": "EXITED"}]}))
        with self.assertRaisesRegex(ValueError, "RUNNING"):
            self.verify()

    def test_physical_settings_and_3000_memory_policy_are_unchanged(self):
        args = SimpleNamespace(grid_level=1, threads=4, maxsteps=35,
            hessian_block_atoms=4, native_direct_driver_threads=1)
        prior, current = v7.worker_settings(args), worker.worker_settings(args)
        for key, value in prior.items():
            if key != "engineering_worker_version":
                self.assertEqual(current[key], value, key)
        self.assertEqual(current["hessian_max_memory_mb"], 3000)
        self.assertEqual(current["density_fit_backend"], "SCREENED_DIRECT_RKS_V4")

    def test_all_mathematical_and_capture_functions_identical_to_v7(self):
        def functions(path):
            return {node.name: ast.dump(node, include_attributes=False)
                    for node in ast.parse(path.read_text()).body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        old = functions(ROOT / "scripts/pyscf_spinosin_bounded_hessian_worker_20261031.py")
        new = functions(ROOT / "scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py")
        for name in ("screened_direct_rks_type", "direct_gradient_type", "direct_integral_policy",
                     "require_optimized_result", "stationary_checks", "frequencies_payload",
                     "prepare_hessian_for_analysis", "frequency_minimum_checks", "hessian_memory_context",
                     "hessian_cache_binding", "_run_checkpointed_hessian_in_memory_context", "run_checkpointed_hessian"):
            self.assertEqual(old[name], new[name], name)
        # The only capture class change is its engineering provenance string.
        prior = old["HessianIntermediateCapture"].replace("BOUNDED_HESSIAN_DIRECT_V7_SAVE_ONLY", "SUPERVISED_PARALLEL_DIRECT_V8_SAVE_ONLY")
        self.assertEqual(prior, new["HessianIntermediateCapture"])

    def test_native_driver_forwarding_and_restoration_are_identical(self):
        def native_methods(path):
            node = next(node for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.ClassDef) and node.name == "NativeDirectDriverThreadGuard")
            return {item.name: ast.dump(item, include_attributes=False) for item in node.body if isinstance(item, ast.FunctionDef)}
        old = native_methods(ROOT / "scripts/pyscf_spinosin_bounded_hessian_worker_20261031.py")
        new = native_methods(ROOT / "scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py")
        for name in old:
            if name != "__init__":
                self.assertEqual(old[name], new[name], name)


if __name__ == "__main__":
    unittest.main()
