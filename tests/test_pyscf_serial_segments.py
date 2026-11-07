"""Pure controller protocol/serialization tests, with a non-QM fake consumer.

No PySCF import, supervisor subprocess, SCF, integral or gradient is executed.
Synthetic files exercise the exact disk archive/recovery functions.
"""
from contextlib import nullcontext, redirect_stdout
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import io
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pure_serial_controller_test", ROOT / "scripts/run_pyscf_serial_segments_20261103.py")
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class FakeConsumer:
    STATE_SHA = "synthetic-state-source"
    EXECUTION_SCHEMA = "synthetic-execution-not-QM"
    def qualify_gradient_receipt(self, path, expected, snapshot, boot):
        if controller.sha(path) != expected: raise ValueError("changed synthetic gradient receipt")
    def boot_id(self): return "SYNTHETIC_ONLY"
    def verify_cached_unit_producers(self, *args, **kwargs): return []
    def load_forensic_release(self, path, expected, snapshot):
        if path is None:
            if expected: raise ValueError("unexpected forensic SHA")
            return None
        if not expected or controller.sha(path) != expected: raise ValueError("changed forensic release")
        return {"scope": "SYNTHETIC_ONLY", "sha256": expected}
    def healthy_closed_row(self, row):
        return row.get("status") == "EXITED" and row.get("exit_code") == 0
    def verify_closed_execution(self, evidence, **kwargs):
        return {"scope": "SYNTHETIC_ONLY", "closed": True}


class SerialControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="unit-serial-controller-", dir=ROOT / "outputs")
        self.base = Path(self.temp.name)
        self.reports = ROOT / "reports" / (self.base.name+"-pure-test")
        self.reports.mkdir()
        self.cache = self.base / "cache"
        self.ledger = self.base / "ledger.json"
        self.now = datetime.now(timezone.utc)
        save(self.ledger, {"schema": "research4-pyscf-actual-cpu-budget-v4", "runner_sha256": controller.SUPERVISOR_SHA,
             "quantum_cpu_cap_seconds": 200000, "consumed_cpu_seconds": 0., "phases": [],
             "run_deadline_at_utc": (self.now+timedelta(hours=3)).isoformat()})
        snapshot = self.reports / "snapshot"
        save(snapshot / "manifest.json", {"scope": "SYNTHETIC_NO_MOLECULE"})
        gradient = self.reports / "gradient.json"
        save(gradient, {"scope": "SYNTHETIC_NO_GRADIENT"})
        evidence = self.reports / "gate-evidence.json"
        save(evidence, {"scope": "SYNTHETIC_NO_REAL_GATE"})
        artifact = {"path": str(evidence), "sha256": controller.sha(evidence)}
        gate = self.reports / "gate.json"
        save(gate, {"schema": "research4-resumable-hessian-gate-v1", "status": "PASS", "approved_molecule": "Spinosin",
             "incremental_module_sha256": controller.CORE_SHA, "consumer_worker_sha256": "synthetic-consumer-source",
             "state_worker_sha256": FakeConsumer.STATE_SHA,
             "tests": {name: {"status": "PASS", "actual_separate_process_resume": True,
                  "same_state_gradient_verified": True, "normal_mode_count": n, "full_hessian_max_abs_diff": 0.,
                  "all_modes_max_abs_diff_cm1": 0., "artifacts": [artifact]} for name, n in (("water", 3), ("ethanol", 21))},
             "independent_review": {"status": "PASS", "report": artifact}})
        self.args = SimpleNamespace(action="run", ledger=self.ledger, snapshot=snapshot,
             snapshot_manifest_sha256=controller.sha(snapshot / "manifest.json"), gradient_check_result=gradient,
             gradient_check_result_sha256=controller.sha(gradient), gate_report=gate, gate_report_sha256=controller.sha(gate),
             worker_sha256="synthetic-consumer-source", output=self.reports / "run001", rhf_cache=self.cache,
             python=Path("/synthetic-no-executable"), phase_prefix="synthetic-spinosin", max_segments=17,
             phase_cpu_seconds=7200, phase_wall_seconds=3600, dry_run=False,
             forensic_unit_release=None, forensic_unit_release_sha256=None)
        self.fake = FakeConsumer()
        self.calls, self.mode = [], "progress"
        # Do not acquire the real global QM lock while root's real gates run.
        # These tests replace locking only; no payload subprocess can be launched.
        self.lock_patch = patch.object(controller, "_closed_cache_guard", side_effect=lambda target: nullcontext())
        self.lock_patch.start()

    def tearDown(self):
        self.lock_patch.stop()
        self.temp.cleanup()
        shutil.rmtree(self.reports)

    def publish_prefix(self, total):
        self.cache.mkdir(exist_ok=True)
        header = self.cache / "header.json"
        if not header.exists():
            save(header, {"schema": "research4-incremental-pure-pbe-rhf-v10", "natm": 75, "total_units": 150,
                          "binding_sha256": "SYNTHETIC_BINDING"})
        old = len(controller.checkpoint_records(self.cache))
        for index in range(old, total):
            kind, atom = ("diag", index) if index < 75 else ("row", index-75)
            unit = self.cache / "units" / f"{index:06d}-{kind}-{atom:04d}" / "attempt-000001"
            unit.mkdir(parents=True)
            arrays = {}
            for name in (["jdiag"] if index < 75 else ["e1", "ej"]):
                file = unit / (name+".npy")
                file.write_bytes(b"SYNTHETIC_BYTES_NOT_A_NUMPY_ARRAY_"+str(index).encode())
                arrays[name] = {"file": file.name, "file_sha256": controller.sha(file)}
            save(unit / "complete.json", {"index": index, "header_sha256": controller.sha(header),
                 "binding_sha256": "SYNTHETIC_BINDING", "arrays": arrays})

    def runner(self, command, console, emit):
        self.calls.append(command)
        self.assertEqual(command[1], str(controller.SUPERVISOR))
        self.assertEqual(command[command.index("--max-new-units")+1], "10")
        phase = command[command.index("--phase")+1]
        output = Path(command[command.index("--output")+1])
        before = len(controller.checkpoint_records(self.cache))
        after = before if self.mode == "no-progress" else min(before+10, 150)
        self.publish_prefix(after)
        ledger = controller.read(self.ledger)
        good = self.mode != "failed"
        ledger["phases"].append({"phase": phase, "status": "EXITED" if good else "PARTIAL_OR_FAILED",
                                  "exit_code": 0 if good else 2, "actual_cpu_seconds": 3.})
        ledger["consumed_cpu_seconds"] += 3.
        save(self.ledger, ledger)
        done = after == 150
        result = {"source_worker": {"scope": "SYNTHETIC_ONLY"}, "resource_supervision": {"scope": "SYNTHETIC_ONLY"},
                  "phase": "hessian-segment", "status": "FREQUENCIES_COMPUTED" if done else "CHECKPOINT_SEGMENT_COMPLETE",
                  "frequencies_status": "DONE" if done else "NOT_RUN", "normal_mode_count": 219 if done else None,
                  "complete_global_Hessian": done, "frequencies_cm1": list(range(1, 220)) if done else [],
                  "frequency_validation_passed": done, "imaginary_frequency_count": 0 if done else None,
                  "incremental_after": {"total_units_completed": after, "total_units": 150}}
        save(output / "result.json", result)
        return (0 if good else 1), None

    def run_controller(self):
        with redirect_stdout(io.StringIO()):
            return controller.execute(self.args, worker_loader=lambda expected: self.fake, phase_runner=self.runner)

    def test_finite_serial_completion_and_archives(self):
        state = self.run_controller()
        self.assertEqual(state["status"], "DONE")
        self.assertEqual(len(self.calls), 15)
        self.assertEqual(state["completed_units"], 150)
        self.assertEqual(len(state["segments"]), 15)
        seals = list((self.args.output / "checkpoint-archive/seals").glob("*.json"))
        self.assertEqual(len(seals), 15)
        self.assertTrue(all(controller.read(p)["active_arrays_read"] is False for p in seals))

    def test_one_segment_can_resume_with_new_run_and_unique_phase(self):
        self.args.max_segments = 1
        first = self.run_controller()
        self.assertEqual(first["status"], "PARTIAL")
        self.assertEqual(first["stop_reason"], "FINITE_SEGMENT_COUNT_EXHAUSTED")
        self.assertEqual(first["completed_units"], 10)
        self.args.output = self.reports / "run002"
        second = self.run_controller()
        self.assertEqual(second["initial_completed_units"], 10)
        self.assertEqual(second["completed_units"], 20)
        self.assertEqual([r["phase"] for r in controller.read(self.ledger)["phases"]],
                         ["synthetic-spinosin-segment-001", "synthetic-spinosin-segment-002"])

    def test_no_progress_stops_after_one_phase(self):
        self.mode = "no-progress"
        state = self.run_controller()
        self.assertEqual(state["status"], "PARTIAL")
        self.assertIn("no bounded progress", state["stop_reason"])
        self.assertEqual(len(self.calls), 1)

    def test_failed_supervisor_stops_without_sealing_or_retry(self):
        self.mode = "failed"
        state = self.run_controller()
        self.assertEqual(state["status"], "PARTIAL")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(state["segments"], [])
        self.assertFalse((self.args.output / "checkpoint-archive").exists())

    def test_insufficient_budget_does_not_launch(self):
        ledger = controller.read(self.ledger)
        ledger["quantum_cpu_cap_seconds"] = 130
        save(self.ledger, ledger)
        state = self.run_controller()
        self.assertEqual(state["status"], "PARTIAL")
        self.assertEqual(self.calls, [])

    def test_active_ledger_rejected_without_launch(self):
        ledger = controller.read(self.ledger)
        ledger["phases"] = [{"phase": "active", "status": "RUNNING"}]
        save(self.ledger, ledger)
        state = self.run_controller()
        self.assertEqual(state["status"], "PARTIAL")
        self.assertEqual(self.calls, [])

    def test_gate_hash_and_missing_real_gate_flags_rejected(self):
        self.args.gate_report_sha256 = "wrong"
        with self.assertRaises(ValueError): self.run_controller()
        gate = controller.read(self.args.gate_report)
        gate["tests"]["water"]["actual_separate_process_resume"] = False
        save(self.args.gate_report, gate)
        self.args.gate_report_sha256 = controller.sha(self.args.gate_report)
        with self.assertRaises(ValueError): self.run_controller()
        self.assertEqual(self.calls, [])

    def test_dry_run_does_not_launch(self):
        self.args.dry_run = True
        state = self.run_controller()
        self.assertEqual(state["status"], "NOT_RUN")
        self.assertEqual(self.calls, [])
        self.assertNotIn("--dry-run", state["recovery_command"])

    def test_explicit_forensic_release_is_forwarded_but_new_failure_still_stops(self):
        release = self.reports / "synthetic-forensic-release.json"
        save(release, {"scope": "SYNTHETIC_ONLY_NOT_AN_ACTUAL_RELEASE"})
        self.args.forensic_unit_release = release
        self.args.forensic_unit_release_sha256 = controller.sha(release)
        self.mode = "failed"
        state = self.run_controller()
        self.assertEqual(state["status"], "PARTIAL")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][self.calls[0].index("--forensic-unit-release")+1], str(release))
        self.assertEqual(state["segments"], [])

    def test_archived_prefix_restores_exact_bytes_and_rejects_changed_file(self):
        self.args.max_segments = 1
        state = self.run_controller()
        seal = state["segments"][0]["archive_seal"]
        expected = {str(p.relative_to(self.cache)): p.read_bytes() for p in self.cache.rglob("*") if p.is_file()}
        shutil.rmtree(self.cache)
        result = controller.restore_cache(seal["path"], seal["sha256"], self.cache)
        self.assertEqual(result["completed_units"], 10)
        self.assertEqual({str(p.relative_to(self.cache)): p.read_bytes() for p in self.cache.rglob("*") if p.is_file()}, expected)
        (self.cache / "header.json").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            controller.restore_cache(seal["path"], seal["sha256"], self.cache)

    def test_archive_old_seal_survives_later_ledger_updates(self):
        self.args.max_segments = 2
        state = self.run_controller()
        seal = state["segments"][0]["archive_seal"]
        shutil.rmtree(self.cache)
        result = controller.restore_cache(seal["path"], seal["sha256"], self.cache)
        self.assertEqual(result["completed_units"], 10)

    def test_archive_and_restore_stop_before_payload_reads_if_guard_busy(self):
        self.publish_prefix(10)
        with patch.object(controller, "_closed_cache_guard", side_effect=BlockingIOError("synthetic busy lock")), \
             patch.object(controller, "_archive_checkpoint_unlocked") as archive, \
             self.assertRaises(BlockingIOError):
            controller.archive_checkpoint(self.cache, self.reports / "archive", "synthetic",
                [], verify_closed=lambda: None, expected_units=10)
        archive.assert_not_called()
        with patch.object(controller, "_closed_cache_guard", side_effect=BlockingIOError("synthetic busy lock")), \
             patch.object(controller, "_restore_cache_unlocked") as restore, \
             self.assertRaises(BlockingIOError):
            controller.restore_cache(self.reports / "absent-seal.json", "synthetic", self.cache)
        restore.assert_not_called()

    def test_locked_archive_requires_repeated_closed_producer_qualification(self):
        self.publish_prefix(10)
        def reject(): raise ValueError("synthetic unclosed producer")
        with self.assertRaises(ValueError):
            controller.archive_checkpoint(self.cache, self.reports / "archive", "synthetic",
                [], verify_closed=reject, expected_units=10)
        self.assertFalse((self.reports / "archive").exists())

    def test_budget_math_dynamic_remaining_and_deadline(self):
        ledger = controller.read(self.ledger)
        ledger.update(quantum_cpu_cap_seconds=1000, consumed_cpu_seconds=100,
                      phases=[{"phase": "closed", "status": "EXITED", "actual_cpu_seconds": 100}])
        limits = controller.available_budget(ledger, requested_cpu=7200, requested_wall=3600, now=self.now)
        self.assertEqual(limits["cpu_seconds"], 890)
        ledger["run_deadline_at_utc"] = (self.now+timedelta(seconds=62)).isoformat()
        self.assertIsNone(controller.available_budget(ledger, requested_cpu=7200, requested_wall=3600, now=self.now))

    def test_controller_bounds_and_old_report_preserved(self):
        self.args.max_segments = 18
        with self.assertRaises(ValueError): self.run_controller()
        self.args.max_segments = 1
        self.run_controller()
        with self.assertRaises(ValueError): self.run_controller()

    def test_supervisor_watchdog_is_finite_without_launching_any_process(self):
        process = SimpleNamespace(pid=123456789, returncode=None)
        process.poll = Mock(return_value=None)
        process.terminate = Mock()
        def mark_killed(): process.returncode = -9
        process.kill = Mock(side_effect=mark_killed)
        process.wait = Mock(side_effect=[controller.subprocess.TimeoutExpired("SYNTHETIC", 3), -9])
        events = []
        with patch.object(controller.subprocess, "Popen", return_value=process) as launch, \
             patch.object(controller.time, "monotonic", side_effect=[0., 1000.]), \
             patch.object(controller.signal, "signal", return_value=None):
            code, stopped = controller.run_supervisor(["SYNTHETIC_NO_EXECUTABLE", "--wall-seconds", "61"],
                self.base / "synthetic-console.log", lambda event, **values: events.append(event))
        self.assertEqual(code, -9)
        self.assertEqual(stopped, "CONTROLLER_SUPERVISOR_WALL_WATCHDOG")
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 2)
        self.assertEqual(events, ["SUPERVISOR_WALL_WATCHDOG"])


if __name__ == "__main__":
    unittest.main()
