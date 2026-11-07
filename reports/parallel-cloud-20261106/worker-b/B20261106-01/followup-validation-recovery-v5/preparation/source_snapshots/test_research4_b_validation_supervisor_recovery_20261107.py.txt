"""V5 partial carry/epoch compatibility tests; no quantum calculations."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import unittest
from unittest import mock

from scripts import test_research4_b_validation_supervisor_20261107 as prior_fixture

SOURCE = Path(__file__).with_name("research4_b_validation_supervisor_recovery_20261107.py")


def load_source(filename, name):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class EpochAdapterTests(unittest.TestCase):
    def setUp(self):
        self.m = load_source(SOURCE, "r4_v5_epoch_test")
        self.boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()

    def test_old_boot_never_enumerates_current_numeric_session(self):
        calls = []
        guard = self.m.historical_session_guard(lambda expected: calls.append(expected), self.boot)
        self.assertEqual(guard({"boot_id": "11111111-1111-1111-1111-111111111111"}), [])
        self.assertEqual(calls, [])

    def test_missing_boot_is_not_a_cross_epoch_exemption(self):
        guard = self.m.historical_session_guard(lambda expected: [], self.boot)
        with self.assertRaisesRegex(ValueError, "explicit valid boot"):
            guard({"pid": 1})

    def test_same_boot_real_process_group_remains_visible(self):
        child = subprocess.Popen([os.sys.executable, "-c", "import time;time.sleep(10)"], start_new_session=True)
        try:
            saved = self.m.identity(child.pid)
            guard = self.m.historical_session_guard(self.m.session_members, self.boot)
            self.assertTrue(any(self.m.same(saved, observed) for observed in guard(saved)))
        finally:
            os.killpg(child.pid, signal.SIGKILL); child.wait()

    def test_exception_restores_global_import_factory(self):
        original = importlib.util.spec_from_file_location
        with self.assertRaisesRegex(RuntimeError, "test exit"):
            with self.m.boot_aware_frozen_history(self.boot):
                self.assertIsNot(importlib.util.spec_from_file_location, original)
                raise RuntimeError("test exit")
        self.assertIs(importlib.util.spec_from_file_location, original)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.prior = prior_fixture.ValidationTests(); self.prior.setUp()
        text = self.prior.worker.read_text().replace("out=Path(a.output);out.mkdir(parents=True)",
            "if 'old-failed-attempt' in a.output: raise SystemExit(9)\nout=Path(a.output);out.mkdir(parents=True)")
        self.prior.update_worker(text)
        self.prior.manifest["jobs"] = [self.prior.job("old-good"), self.prior.job("old-failed", max_attempts=1),
            self.prior.job("old-dependent", depends_on=["old-failed"])]
        completed = self.prior.execute(); self.assertEqual(completed.returncode, 0, completed.stderr)
        self.root = self.prior.root; self.python = self.prior.python
        self.source = self.root / "scripts" / SOURCE.name; shutil.copyfile(SOURCE, self.source)
        self.m = load_source(self.source, "r4_v5_recovery_test")
        self.worker = self.root / "scripts/research4_b_validation_recovery_worker_20261107.py"
        self.worker.write_text(self.prior.worker.read_text().replace(
            "from research4_b_validation_supervisor_20261107 import", "from research4_b_validation_supervisor_recovery_20261107 import"))
        self.work = self.prior.work.parent / "followup-validation-recovery-v5"; self.work.mkdir()
        old_state = self.m.read(self.prior.work / "state.json")
        self.manifest = {key: self.prior.manifest[key] for key in ("run_id", "branch", "cpu_cap_seconds", "deadline_utc",
            "threads", "memory_gib", "rss_stop_gib", "reserve_cpu_seconds", "python_executable")}
        self.manifest.update(schema=self.m.MANIFEST_SCHEMA, work_directory=str(self.work.relative_to(self.root)),
            ledger=str((self.work / "cpu_budget.json").relative_to(self.root)),
            predecessor_manifest=self.m.ref(self.prior.work / "manifest.json"),
            predecessor_ledger=self.m.ref(self.prior.work / "cpu_budget.json"), predecessor_state=self.m.ref(self.prior.work / "state.json"),
            predecessor_supervisor=self.m.ref(self.prior.source),
            predecessor_allocation=self.m.ref(self.root / ".runtime/worker-b-validation-allocation.json"),
            inherited_results=old_state["results"], recovery_boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            source_refs=[self.m.ref(self.source), self.m.ref(self.worker)], input_refs=[], jobs=[])
        receipt = self.work / "previous_return_receipt.json"; self.m.save(receipt, {"status": "TEST_PREVIOUS_RETURN_VERIFIED"})
        btime = int(next(line.split()[1] for line in Path("/proc/stat").read_text().splitlines() if line.startswith("btime ")))
        review = {"schema": "research4-b-independent-v4-closure-remote-review-v1",
            "status": "PASS_CLOSED_PARTIAL_RESULTS_AND_PREVIOUS_REMOTE_VERIFICATION",
            "v4_manifest": self.manifest["predecessor_manifest"], "v4_ledger": self.manifest["predecessor_ledger"],
            "v4_state": self.manifest["predecessor_state"], "v4_supervisor": self.manifest["predecessor_supervisor"],
            "persistent_v4_allocation": self.manifest["predecessor_allocation"], "return_receipt": self.m.ref(receipt),
            "current_boot_observation": {"current_boot_id": self.manifest["recovery_boot_id"], "kernel_btime_unix_seconds": btime}}
        review_path = self.work / "closure_review.json"; self.m.save(review_path, review)
        self.manifest["predecessor_closure_review"] = self.m.ref(review_path)

    def tearDown(self):
        self.prior.tearDown()

    def job(self, name="recovery-good", **overrides):
        job = self.prior.job(name); job["argv"][1] = str(self.worker.relative_to(self.root)); job.update(overrides)
        return job

    def execute(self, attempt="recovery-test-001", execute=True):
        target = self.work / "manifest.json"; self.m.save(target, self.manifest)
        command = [str(self.python), str(self.source), "--manifest", str(target), "--manifest-sha256", self.m.sha(target), "--attempt", attempt]
        if execute: command.append("--execute")
        return subprocess.run(command, cwd=self.root, text=True, capture_output=True, timeout=20)

    def test_partial_carry_includes_every_failed_cpu_cost(self):
        self.manifest["jobs"] = [self.job()]
        carry = self.m.validate(self.manifest, "0" * 64)
        old = self.m.read(self.prior.work / "cpu_budget.json")
        self.assertEqual(carry["actual_closed_cpu_seconds"], old["actual_closed_cpu_seconds"])
        self.assertEqual(carry["budget_charged_cpu_seconds"], old["budget_charged_cpu_seconds"])
        self.assertEqual(carry["conservative_debit_cpu_seconds"], 1000)
        self.assertEqual(carry["predecessor_failed_closed_phase_count"], 1)
        self.assertEqual(carry["inherited_success_count"], 1)

    def test_inherited_success_is_reused_and_new_dependency_runs(self):
        self.manifest["jobs"] = [self.job(depends_on=["old-good"])]
        result = self.execute(); self.assertEqual(result.returncode, 0, result.stderr)
        state = self.m.read(self.work / "state.json"); ledger = self.m.read(self.work / "cpu_budget.json")
        self.assertEqual(set(state["results"]), {"old-good", "recovery-good"})
        self.assertEqual(state["results"]["old-good"], self.manifest["inherited_results"]["old-good"])
        self.assertEqual(len(ledger["phases"]), 1)
        self.assertAlmostEqual(ledger["budget_charged_cpu_seconds"] - ledger["actual_closed_cpu_seconds"], 1000)

    def test_inherited_subset_cannot_silently_omit_a_success(self):
        self.manifest["jobs"] = [self.job()]; self.manifest["inherited_results"] = {}
        with self.assertRaisesRegex(ValueError, "exactly the frozen successful"):
            self.m.validate(self.manifest, "0" * 64)

    def test_old_failed_attempts_cannot_be_reset(self):
        self.manifest["jobs"] = [self.job("old-failed", max_attempts=1)]
        with self.assertRaisesRegex(ValueError, "cannot reset historical attempt"):
            self.m.validate(self.manifest, "0" * 64)

    def test_unclosed_v4_is_rejected_before_epoch_adapter(self):
        self.manifest["jobs"] = [self.job()]
        ledger_path = self.prior.work / "cpu_budget.json"; ledger = self.m.read(ledger_path)
        ledger["phases"][-1]["status"] = "RUNNING"; self.m.save(ledger_path, ledger)
        self.manifest["predecessor_ledger"] = self.m.ref(ledger_path)
        with mock.patch.object(self.m, "boot_aware_frozen_history", side_effect=AssertionError("adapter invoked")) as adapter:
            with self.assertRaisesRegex(ValueError, "not completely closed"):
                self.m.validate(self.manifest, "0" * 64)
            adapter.assert_not_called()

    def test_different_recovery_boot_is_rejected(self):
        self.manifest["jobs"] = [self.job()]
        self.manifest["recovery_boot_id"] = "11111111-1111-1111-1111-111111111111"
        with self.assertRaisesRegex(ValueError, "bound to this observed kernel boot"):
            self.m.validate(self.manifest, "0" * 64)

    def test_lost_recovery_ledger_cannot_reset_the_claim(self):
        self.manifest["jobs"] = [self.job()]
        result = self.execute(); self.assertEqual(result.returncode, 0, result.stderr)
        (self.work / "cpu_budget.json").unlink()
        result = self.execute("recovery-test-002")
        self.assertNotEqual(result.returncode, 0); self.assertIn("ledger is missing", result.stderr)

    def test_modified_inherited_science_remains_detected(self):
        self.manifest["jobs"] = [self.job()]
        binding = self.manifest["inherited_results"]["old-good"]
        self.m.path(binding["path"]).write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "Closed result changed"):
            self.m.validate(self.manifest, "0" * 64)


if __name__ == "__main__":
    unittest.main()
