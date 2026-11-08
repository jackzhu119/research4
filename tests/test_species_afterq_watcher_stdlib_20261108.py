"""Synthetic boundary tests only; no worker, numerical library or Git execution."""
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("species_afterq_test", ROOT / "scripts/species_afterq_watcher_20261108.py")
w = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(w)


class ParentIdentityTests(unittest.TestCase):
    def setUp(self):
        self.expected = {"pid": 9876, "start_ticks": 123, "uid": os.getuid(),
                         "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}

    def observed(self, **kw):
        return {**self.expected, "state": "S", **kw}

    def test_exact_sleeping_parent_is_waiting_not_dead(self):
        self.assertEqual(w.parent_status(self.expected, lambda _: self.observed()), "WAITING_FOR_QC_PARENT")

    def test_active_parent_is_waiting(self):
        self.assertEqual(w.parent_status(self.expected, lambda _: self.observed(state="R")), "WAITING_FOR_QC_PARENT")

    def test_missing_pid_requires_more_evidence_not_completion(self):
        def absent(_): raise FileNotFoundError()
        self.assertIn("REQUIRES_FULL_CLOSURE_AUDIT", w.parent_status(self.expected, absent))

    def test_zombie_requires_more_evidence(self):
        self.assertIn("REQUIRES_FULL_CLOSURE_AUDIT", w.parent_status(self.expected, lambda _: self.observed(state="Z")))

    def test_pid_reuse_blocks(self):
        with self.assertRaisesRegex(ValueError, "PID_REUSED"):
            w.parent_status(self.expected, lambda _: self.observed(start_ticks=999))

    def test_uid_change_blocks(self):
        with self.assertRaisesRegex(ValueError, "IDENTITY_CHANGED"):
            w.parent_status(self.expected, lambda _: self.observed(uid=0))

    def test_new_boot_cannot_infer_previous_parent_closure(self):
        with self.assertRaisesRegex(ValueError, "BOOT_CHANGED"):
            w.parent_status({**self.expected, "boot_id": "different-boot"}, lambda _: self.observed())


class ConfigAndClosureTests(unittest.TestCase):
    def setUp(self):
        self.qc = {"workroot": str(w.ROOT / "reports/research72h-20261107-v1/autonomous-recovery-002"),
                   "jobs": [{"id": "controls"}, {"id": "grid1"}, {"id": "grid3"}], "source_refs": [], "immutable_inputs": []}
        self.expected = {"pid": 123, "start_ticks": 456, "uid": os.getuid(), "boot_id": "metadata-only"}
        self.ledger = {"run_deadline_at_utc": w.DEADLINE, "remaining_quantum_cpu_seconds": 5000, "phases": []}
        self.state = {"status": "FINITE_QUEUE_COMPLETE_WITH_ALL_RESULTS_PRESERVED", "jobs": {
            "controls": {"status": "SUCCESS"}, "grid1": {"status": "FAILED_CLOSED"}, "grid3": {"status": "SUCCESS"}}}
        self.events = []
        ledger_path = w.ROOT / "reports/research72h-20261107-v1/cpu_budget.json"
        self.guard = types.SimpleNamespace(validate_ledger=self.validate_ledger)
        self.c = {"schema": w.SCHEMA, "watcher_source_sha256": w.sha(w.SELF),
            "driver_ref": {"path": str(w.DRIVER), "sha256": w.DRIVER_SHA},
            "guard_ref": {"path": str(w.ROOT / "scripts/run_research4_research72h_budget_20261107.py"), "sha256": w.GUARD_SHA},
            "ledger_path": str(ledger_path), "deadline_utc": w.DEADLINE,
            "qc_config_ref": {"path": str(w.ROOT / "reports/research72h-20261107-v1/queue-configs/recovery-002/config_ROOT_APPROVED.json"), "sha256": w.QC_SHA},
            "qc_receipt_ref": {"path": str(w.ROOT / "reports/research72h-20261107-v1/autonomous-recovery-002/launch-002-detached-launch.json"), "sha256": "synthetic"},
            "qc_state_path": self.qc["workroot"] + "/state.json",
            "worker_ref": {"path": str(w.ROOT / "scripts/research4_species_correction_analysis_20261108.py"), "sha256": "synthetic"},
            "argv": ["python", "-B", str(w.ROOT / "scripts/research4_species_correction_analysis_20261108.py"), "--execute", "--output", "{output}"],
            "workroot": str(w.PREFIX / "synthetic-waiter"), "output": str(w.PREFIX / "synthetic-result"),
            "log": str(w.PREFIX / "synthetic-worker.log"), "phase": w.PHASE, "cpu_seconds": 2000,
            "wall_seconds": 1200, "threads": 1, "immutable_inputs": [], "success_statuses": ["DONE"]}
        self.d = types.SimpleNamespace(LEDGER=ledger_path, path=lambda x: Path(x),
            checked=lambda ref: Path(ref["path"]), guard_module=lambda _: self.guard,
            read=self.read, resume_state=self.resume, process_scan=self.scan)

    def validate_ledger(self, ledger, **kw):
        self.events.append(("ledger", kw))
        if not kw.get("allow_running", False) and any(r.get("status") == "RUNNING" for r in ledger["phases"]):
            raise ValueError("Unknown/open phase")

    def read(self, path):
        path = Path(path)
        if path == self.d.LEDGER: return self.ledger
        if path == Path(self.c["qc_config_ref"]["path"]): return self.qc
        if path == Path(self.c["qc_receipt_ref"]["path"]): return {"config_sha256": w.QC_SHA, "identity": self.expected}
        if path == Path(self.c["qc_state_path"]): return self.state
        raise AssertionError("Unexpected read: " + str(path))

    def resume(self, state, config_hash, config, guard):
        self.events.append(("authenticated_resume", config_hash))
        for entry in state["jobs"].values():
            if entry["status"] not in ("SUCCESS", "FAILED_CLOSED", "SKIPPED_DEPENDENCY", "SKIPPED_RESOURCE"):
                raise ValueError("Unknown closure")

    def scan(self, _): self.events.append(("independent_scan", True))

    def test_preflight_accepts_healthy_running_phase_without_execution(self):
        self.ledger["phases"] = [{"status": "RUNNING"}]
        w.validate(self.c, self.d, waiting=True)
        self.assertEqual(self.events[-1], ("ledger", {"allow_running": True}))

    def test_original_deadline_cannot_be_extended(self):
        self.c["deadline_utc"] = "2027-01-01T00:00:00+00:00"
        with self.assertRaises(ValueError): w.validate(self.c, self.d)

    def test_different_ledger_rejected(self):
        self.c["ledger_path"] = str(w.PREFIX / "reset-budget.json")
        with self.assertRaises(ValueError): w.validate(self.c, self.d)

    def test_different_phase_rejected(self):
        self.c["phase"] = "retry"
        with self.assertRaises(ValueError): w.validate(self.c, self.d)

    def test_larger_cpu_or_multithreading_rejected(self):
        for key, value in [("cpu_seconds", 90000), ("threads", 2)]:
            with self.subTest(key=key):
                old = self.c[key]; self.c[key] = value
                with self.assertRaises(ValueError): w.validate(self.c, self.d)
                self.c[key] = old

    def test_shell_or_unbound_output_rejected(self):
        for argv in [["bash", "-c", "echo hi"], self.c["argv"] + ["{other}"]]:
            with self.subTest(argv=argv):
                self.c["argv"] = argv
                with self.assertRaises(ValueError): w.validate(self.c, self.d)

    def test_other_worker_rejected(self):
        self.c["worker_ref"]["path"] = str(w.ROOT / "scripts/train_unrelated.py")
        with self.assertRaises(ValueError): w.validate(self.c, self.d)

    def test_outputs_cannot_touch_original_data(self):
        self.c["output"] = str(w.ROOT / "data/raw/replaced")
        with self.assertRaises(ValueError): w.validate(self.c, self.d)

    def test_result_basename_cannot_escape_to_old_results(self):
        for name in ["../old/result.json", "/old/result.json", "result.txt", "sub/result.json"]:
            with self.subTest(name=name):
                self.c["result_file"] = name
                with self.assertRaises(ValueError): w.validate(self.c, self.d)

    def test_explicit_nonempty_success_statuses_required(self):
        for statuses in [[], "DONE", [""], [1]]:
            with self.subTest(statuses=statuses):
                self.c["success_statuses"] = statuses
                with self.assertRaises(ValueError): w.validate(self.c, self.d)

    def test_worker_cannot_overwrite_waiter_metadata_directory(self):
        self.c["output"] = self.c["workroot"]
        with self.assertRaises(ValueError): w.validate(self.c, self.d)

    def test_running_ledger_blocks_closure_even_with_finished_state(self):
        self.ledger["phases"] = [{"phase": "Q", "status": "RUNNING"}]
        with self.assertRaisesRegex(ValueError, "Unknown/open"):
            w.authenticate_qc_closure(self.c, self.d, self.guard, self.qc)

    def test_absent_parent_does_not_substitute_finished_state(self):
        self.state["status"] = "RUNNING"
        with self.assertRaisesRegex(ValueError, "absence"):
            w.authenticate_qc_closure(self.c, self.d, self.guard, self.qc)

    def test_all_terminal_entries_required(self):
        del self.state["jobs"]["grid3"]
        with self.assertRaisesRegex(ValueError, "All QC queue"):
            w.authenticate_qc_closure(self.c, self.d, self.guard, self.qc)

    def test_unknown_closed_row_blocks(self):
        self.state["jobs"]["grid1"]["status"] = "BLOCKED_UNKNOWN_CLOSURE"
        with self.assertRaisesRegex(ValueError, "Unknown closure"):
            w.authenticate_qc_closure(self.c, self.d, self.guard, self.qc)

    def test_known_scientific_failure_still_allows_independent_evidence_analysis(self):
        with patch.object(w, "sha", return_value="synthetic"), patch.object(w, "utc", return_value=datetime(2026, 11, 8, tzinfo=timezone.utc)):
            result = w.authenticate_qc_closure(self.c, self.d, self.guard, self.qc)
        self.assertTrue(result["all_terminal_jobs_authenticated"])
        self.assertIn(("authenticated_resume", w.QC_SHA), self.events)
        self.assertIn(("independent_scan", True), self.events)

    def test_existing_phase_is_never_repeated(self):
        self.ledger["phases"] = [{"phase": w.PHASE, "status": "EXITED"}]
        with self.assertRaisesRegex(ValueError, "already exists"):
            w.authenticate_qc_closure(self.c, self.d, self.guard, self.qc)

    def test_insufficient_budget_or_wall_blocks(self):
        self.ledger["remaining_quantum_cpu_seconds"] = 1999
        with self.assertRaisesRegex(ValueError, "remaining registered CPU"):
            w.authenticate_qc_closure(self.c, self.d, self.guard, self.qc)
        self.ledger["remaining_quantum_cpu_seconds"] = 5000
        with patch.object(w, "utc", return_value=datetime.fromisoformat(w.DEADLINE) - timedelta(seconds=1200)):
            with self.assertRaisesRegex(ValueError, "wall window"):
                w.authenticate_qc_closure(self.c, self.d, self.guard, self.qc)


class ReportSafetyTests(unittest.TestCase):
    def test_exclusive_receipt_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "receipt.json"
            w.exclusive(p, {"first": True})
            with self.assertRaises(FileExistsError): w.exclusive(p, {"replacement": True})
            self.assertEqual(json.loads(p.read_text()), {"first": True})

    def test_atomic_reports_cannot_escape_new_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError): w.save(Path(directory) / "outside.json", {"x": 1})


if __name__ == "__main__": unittest.main()
