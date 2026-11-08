"""Synthetic stdlib queue tests. Never launch a Q guardian or read a Q ledger."""
from argparse import Namespace
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[1] / "scripts/research4_quantum_queue_20261105.py"
SPEC = importlib.util.spec_from_file_location("prepared_quantum_queue_tests", SOURCE)
Q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.qdir = self.root / "reports/queue-fixture"
        self.qdir.mkdir(parents=True)
        self.sdir = self.root / "scripts"
        self.sdir.mkdir()
        self.worker = self.sdir / "pyscf_xc_chunk_worker_20261105_v7.py"
        self.core = self.sdir / "pyscf_xc_chunk_checkpoint_20261105_v7.py"
        self.guardfile = self.sdir / "guard.py"
        self.python = self.root / ".runtime/bin/python"
        self.python.parent.mkdir(parents=True)
        for p in (self.worker, self.core, self.guardfile, self.python):
            p.write_text("# synthetic; never executed\n")
        for name, value in (("ROOT", self.root), ("GUARD", self.guardfile), ("GUARD_SHA", Q.sha(self.guardfile)),
                ("FROZEN_A", {str(p.relative_to(self.root)): Q.sha(p) for p in (self.worker, self.core)})):
            item = patch.object(Q, name, value)
            item.start()
            self.addCleanup(item.stop)
        self.output = self.root / "reports/science/future-output"
        self.job = {"id": "block001", "phase": "actual-scientific-phase001", "worker": Q.artifact(self.worker),
            "stage": "xc-diag-chunk", "success_status": "XC_CHUNK_COMPUTED", "result_schema": "V7_FIXTURE",
            "result": str(self.output / "result.json"), "log": str(self.root / "reports/science/phase.log"),
            "threads": 2, "memory_gib": 24, "cpu_seconds": 1800, "wall_seconds": 900,
            "payload_argv": [str(self.python), str(self.worker), "--execute", "--stage", "xc-diag-chunk",
                "--output", str(self.output), "--threads", "2", "--snapshot-manifest-sha256", "f" * 64]}
        self.deadline = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
        self.config = {"schema": Q.SCHEMA, "jobs": [self.job], "monitor_interval_seconds": 1800,
            "queue_directory": str(self.qdir), "state": str(self.qdir / "state.json"), "monitor": str(self.qdir / "monitor.jsonl"),
            "ledger": str(self.root / "reports/science/ledger.json"), "original_quantum_cpu_cap_seconds": 216000,
            "original_quantum_deadline_at_utc": self.deadline, "python_executable": str(self.python),
            "pythonpath": [".runtime/quantum-pilot-site", ".runtime/quantum-upgrade-site"],
            "source_refs": [Q.artifact(p) for p in (self.guardfile, self.worker, self.core)]}
        self.config_path = self.qdir / "config.json"
        self.write(self.config_path, self.config)
        self.grant = {"schema": Q.GRANT_SCHEMA, "config_sha256": Q.sha(self.config_path), "launcher_sha256": Q.sha(SOURCE),
            "no_new_quantum_budget": True, "no_deadline_extension": True, "root_execution_approved": True,
            "root_serial_scheduling_confirmed": True, "max_invocations": 2, "launcher_cpu_seconds_per_invocation": 60,
            "launcher_rss_stop_bytes": 256 * 1024**2, "wall_seconds_per_invocation": 1800, "expires_at_utc": self.deadline}
        self.grant_path = self.qdir / "grant.json"
        self.write(self.grant_path, self.grant)
        self.ledger = {"phases": [], "consumed_cpu_seconds": 100., "remaining_quantum_cpu_seconds": 215900.,
            "quantum_cpu_cap_seconds": 216000, "run_deadline_at_utc": self.deadline}
        self.write(Path(self.config["ledger"]), self.ledger)
        self.guard = SimpleNamespace(validate_ledger=Mock(), healthy_closed_row=lambda r: r["status"] == "EXITED" and r["exit_code"] == 0,
            verify_closed_phase=Mock())
        self.args = Namespace(config=str(self.config_path), config_sha256=Q.sha(self.config_path), grant=str(self.grant_path),
            grant_sha256=Q.sha(self.grant_path), attempt="root001", invocation_deadline=Q.utc().timestamp() + 1800,
            invocation_monotonic_deadline=Q.time.monotonic() + 1800)

    def write(self, p, value):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value) + "\n")

    def materialize_result(self):
        self.output.mkdir(parents=True)
        proof = self.root / "reports/science/proof.json"
        self.write(proof, {"phase": self.job["phase"]})
        Path(self.job["log"]).write_text("actual fixture log\n")
        self.write(self.output / "input.json", {"arguments": {"stage": self.job["stage"]}})
        payload = self.output / "stage-cache/payload.npy"
        payload.parent.mkdir()
        payload.write_bytes(b"synthetic payload; not decoded as NumPy")
        self.result = {"stage": self.job["stage"], "status": self.job["success_status"], "schema": "V7_FIXTURE",
            "snapshot_manifest_sha256": "f" * 64, "source_worker": self.job["worker"],
            "resource_supervision": {"proof": {"phase": self.job["phase"]}, "proof_path": str(proof)},
            "arrays": [{"file": payload.name, "file_sha256": Q.sha(payload)}]}
        self.write(Path(self.job["result"]), self.result)
        self.ledger["phases"] = [{"phase": self.job["phase"], "status": "EXITED", "exit_code": 0,
            "command": self.job["payload_argv"], "log": self.job["log"]}]
        self.write(Path(self.config["ledger"]), self.ledger)
        return Q.verify_stage(self.job, self.guard, self.ledger)

    def run_queue(self, child=None):
        with patch.object(Q, "load_guard", return_value=self.guard), patch.object(Q, "conflicting_processes", return_value=[]), \
                patch.object(Q, "own_rss", return_value=1024), patch.object(Q.subprocess, "Popen", return_value=child) as popen, \
                redirect_stdout(io.StringIO()):
            Q.run(self.args, self.config, self.grant)
        return popen

    def test_prepared_configuration_valid_but_root_false_cannot_execute(self):
        Q.validate_config(self.config, self.grant, self.args.config_sha256, True)
        self.grant["root_execution_approved"] = False
        with self.assertRaises(ValueError): Q.validate_config(self.config, self.grant, self.args.config_sha256, True)

    def test_duplicate_outputs_and_nonmaterialized_argv_rejected(self):
        other = deepcopy(self.job); other.update(id="other", phase="other")
        self.config["jobs"].append(other)
        with self.assertRaises(ValueError): Q.validate_config(self.config, self.grant, self.args.config_sha256)
        self.config["jobs"] = [self.job]; self.job["payload_argv"][0] = "/bin/sh"
        with self.assertRaises(ValueError): Q.validate_config(self.config, self.grant, self.args.config_sha256)

    def test_scientific_or_shared_metadata_write_paths_rejected(self):
        self.config["monitor"] = self.config["ledger"]
        with self.assertRaises(ValueError): Q.validate_config(self.config, self.grant, self.args.config_sha256)
        self.config["monitor"] = self.config["state"]
        with self.assertRaises(ValueError): Q.validate_config(self.config, self.grant, self.args.config_sha256)

    def test_changed_source_and_extended_deadline_rejected(self):
        self.core.write_text("changed")
        with self.assertRaises(ValueError): Q.validate_config(self.config, self.grant, self.args.config_sha256)
        self.core.write_text("# synthetic; never executed\n")
        self.grant["expires_at_utc"] = (Q.utc() + timedelta(days=2)).isoformat()
        with self.assertRaises(ValueError): Q.validate_config(self.config, self.grant, self.args.config_sha256)

    def test_missing_phase_is_not_a_done_result(self):
        self.assertIsNone(Q.verify_stage(self.job, self.guard, self.ledger))

    def test_failed_or_running_stage_refuses_even_with_existing_file(self):
        self.materialize_result()
        for status in ("RUNNING", "PARTIAL_OR_FAILED"):
            self.ledger["phases"][0]["status"] = status
            with self.subTest(status=status), self.assertRaises(ValueError): Q.verify_stage(self.job, self.guard, self.ledger)

    def test_exact_argv_proof_stage_and_payload_hashes_required(self):
        self.materialize_result()
        self.ledger["phases"][0]["command"] = ["different"]
        with self.assertRaises(ValueError): Q.verify_stage(self.job, self.guard, self.ledger)
        self.ledger["phases"][0]["command"] = self.job["payload_argv"]
        (self.output / "stage-cache/payload.npy").write_bytes(b"changed")
        with self.assertRaises(ValueError): Q.verify_stage(self.job, self.guard, self.ledger)

    def test_no_silent_reseal_of_result_input_proof_log_or_payload(self):
        sealed = self.materialize_result()
        for ref in sealed:
            p = Path(ref["path"]); old = p.read_bytes(); p.write_bytes(old + b" ")
            with self.subTest(path=p.name), self.assertRaises(ValueError): Q.verify_existing(self.job, self.guard, self.ledger, sealed)
            p.write_bytes(old)
        self.assertEqual(Q.verify_existing(self.job, self.guard, self.ledger, sealed), sealed)
        with self.assertRaises(ValueError): Q.verify_existing(self.job, self.guard, self.ledger, None)

    def test_healthy_root_sealed_phase_is_skipped_without_spawning(self):
        self.job["existing_seal"] = self.materialize_result()
        popen = self.run_queue(); popen.assert_not_called()
        state = Q.read(self.config["state"])
        self.assertEqual(state["jobs"][self.job["id"]]["seal"], self.job["existing_seal"])
        self.assertEqual(state["invocations"][-1]["status"], "MATERIALIZED_QUEUE_CLOSED_NOT_ALL_SCIENCE_COMPLETE")
        self.assertIn("--execute", state["resume_argv"])
        self.assertTrue(state["resume_requires_environment_ready_and_no_unresolved_invocation_or_Q_phase"])

    def test_existing_healthy_without_root_or_prior_seal_blocks(self):
        self.materialize_result()
        with self.assertRaises(ValueError): self.run_queue()

    def test_unclosed_invocation_never_becomes_done_from_missing_pid(self):
        state = {"config_sha256": self.args.config_sha256, "grant_sha256": self.args.grant_sha256,
            "jobs": {}, "invocations": [{"id": "old", "status": "RUNNING", "pid": 999999999}]}
        self.write(Path(self.config["state"]), state)
        before = Path(self.config["state"]).read_bytes()
        with self.assertRaises(ValueError): self.run_queue()
        self.assertEqual(before, Path(self.config["state"]).read_bytes())

    def test_zero_guardian_exit_is_not_scientific_success(self):
        child = Mock(pid=987654); child.poll.return_value = 0; child.wait.return_value = 0
        with self.assertRaises(ValueError): self.run_queue(child)
        state = Q.read(self.config["state"])
        self.assertNotEqual(state["jobs"][self.job["id"]]["status"], "HEALTHY_CLOSED")
        child.wait.assert_called_once()

    def test_nonzero_guardian_stops_without_automatic_retry(self):
        child = Mock(pid=987654); child.poll.return_value = 7; child.wait.return_value = 7
        with self.assertRaises(ValueError): self.run_queue(child)
        state = Q.read(self.config["state"])
        self.assertEqual(state["jobs"][self.job["id"]]["actual_guardian_returncode"], 7)
        self.assertEqual(len(state["invocations"]), 1)

    def test_source_changed_after_initial_validation_is_blocked_before_launch(self):
        Q.validate_config(self.config, self.grant, self.args.config_sha256, True)
        self.worker.write_text("later source change")
        with self.assertRaises(ValueError): self.run_queue()
        state = Q.read(self.config["state"])
        self.assertEqual(state["invocations"][-1]["status"], "BLOCKED_REQUIRES_ROOT_REVIEW")

    def test_monotonic_wall_budget_blocks_even_if_UTC_clock_is_early(self):
        self.args.invocation_deadline = Q.utc().timestamp() + 7200
        self.args.invocation_monotonic_deadline = Q.time.monotonic() - 1
        with self.assertRaises(ValueError): self.run_queue()
        self.assertFalse(Path(self.job["log"] + ".queue-console.log").exists())

    def test_global_mode_and_raw_Hessian_bytes_are_sealed_without_decoding(self):
        self.materialize_result()
        self.job["stage"] = "global-assemble"
        self.job["payload_argv"][self.job["payload_argv"].index("--stage") + 1] = self.job["stage"]
        self.result["stage"] = self.job["stage"]
        files = ("normal_modes.npy", "reduced_mass_amu.npy", "frequencies.csv", "normal_mode_geometry_angstrom.npy",
                 "normal_mode_geometry_bohr.npy", "normal_mode_geometry.xyz", "hessian_hartree_per_bohr2.npy")
        for name in files: (self.output / name).write_bytes(b"not decoded; test bytes")
        self.result["raw_hessian_sha256"] = Q.sha(self.output / "hessian_hartree_per_bohr2.npy")
        self.write(Path(self.job["result"]), self.result)
        sealed = Q.verify_stage(self.job, self.guard, self.ledger)
        self.assertTrue(set(files) <= {Path(ref["path"]).name for ref in sealed})
        (self.output / "normal_modes.npy").write_bytes(b"changed")
        with self.assertRaises(ValueError): Q.verify_existing(self.job, self.guard, self.ledger, sealed)

    def test_own_timers_do_not_lower_or_inherit_guardian_resource_limits(self):
        # This is a tiny stdlib subprocess, never a scientific guard/worker.
        parent_limits = resource.getrlimit(resource.RLIMIT_CPU)
        tiny = dict(self.grant, launcher_cpu_seconds_per_invocation=10)
        with Q.self_limits(tiny):
            raw = subprocess.check_output([sys.executable, "-S", "-c",
                "import json,signal,resource;print(json.dumps([signal.getitimer(signal.ITIMER_PROF),signal.getitimer(signal.ITIMER_REAL),resource.getrlimit(resource.RLIMIT_CPU)]))"])
        values = json.loads(raw)
        self.assertEqual(values[0], [0., 0.]); self.assertEqual(values[1], [0., 0.])
        self.assertEqual(tuple(values[2]), parent_limits)
        self.assertEqual(resource.getrlimit(resource.RLIMIT_CPU), parent_limits)
        self.assertEqual(signal.getitimer(signal.ITIMER_PROF), (0., 0.))


if __name__ == "__main__":
    unittest.main()
