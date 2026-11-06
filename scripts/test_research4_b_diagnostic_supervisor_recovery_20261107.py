"""Recovery budget/provenance/kernel-receipt tests; no quantum calculations."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from scripts.test_research4_b_diagnostic_supervisor_20261106 import SupervisorTests as OldFixture


SOURCE = Path(__file__).with_name("research4_b_diagnostic_supervisor_recovery_20261107.py")


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.old = OldFixture()
        self.old.setUp()
        self.root = self.old.root
        self.pending_job = self.old.job("new-good", fail=True, cpu_seconds=1000, max_attempts=2, depends_on=["inherited-success"])
        self.old.manifest["jobs"] = [self.old.job("inherited-success"), self.pending_job]
        completed = self.old.execute("old-completed")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.prior_work = self.old.work
        old_ledger_path = self.prior_work / "cpu_budget.json"
        old_state_path = self.prior_work / "state.json"
        ledger = self.old.m.read(old_ledger_path)
        failed = ledger["phases"][-1]
        ledger["consumed_cpu_seconds"] -= failed["actual_cpu_seconds"]
        ledger["remaining_cpu_seconds"] = self.old.m.CAP - ledger["consumed_cpu_seconds"]
        failed.update(status="BLOCKED_UNACCOUNTED", actual_cpu_seconds=None, actual_cpu_lower_bound_seconds=.61)
        ledger["blocked_unknown_cpu_or_process_tree"] = True
        self.old.m.save(old_ledger_path, ledger)
        state = self.old.m.read(old_state_path)
        state["invocations"][-1]["status"] = "BLOCKED_REQUIRES_FORENSICS"
        del state["jobs"]["new-good"]
        self.old.m.save(old_state_path, state)
        self.source = self.root / "scripts" / SOURCE.name
        shutil.copyfile(SOURCE, self.source)
        spec = importlib.util.spec_from_file_location("r4_recovery_test_module", self.source)
        self.m = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.m)
        self.worker = self.root / "scripts/research4_b_diagnostic_worker_recovery_20261107.py"
        text = self.old.worker.read_text().replace("from research4_b_diagnostic_supervisor_20261106 import", "from research4_b_diagnostic_supervisor_recovery_20261107 import")
        text = text.replace("import argparse,hashlib,json,os,time", "import argparse,hashlib,json,os,time,subprocess,sys")
        text = text.replace("p.add_argument('--fail',action='store_true');", "p.add_argument('--fail',action='store_true');p.add_argument('--fork',action='store_true');")
        text = text.replace("t=time.process_time()", "if a.fork:\n child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(2)']);child.wait()\nt=time.process_time()")
        text = text.replace("if a.fail: raise SystemExit(9)", "# The reviewed recovery fixture fixes its predecessor's test failure.")
        self.worker.write_text(text)
        self.work = self.old.original / "followup-diagnostics-recovery-v2"
        self.work.mkdir()
        prior_manifest_path = self.prior_work / "manifest.json"
        forensic = self.work / "forensic_resolution.json"
        preserved = self.work / "preserved_old_ledger.json"
        shutil.copyfile(old_ledger_path, preserved)
        resolution = {"approved_for_recovery": True, "processes_verified_closed": True,
                      "blocked_phase": failed["phase"], "actual_wait4_cpu_seconds": None,
                      "conservative_debit_cpu_seconds": 1000,
                      "predecessor_manifest_sha256": self.m.sha(prior_manifest_path),
                      "predecessor_ledger_sha256": self.m.sha(old_ledger_path), "predecessor_state_sha256": self.m.sha(old_state_path),
                      "preserved_files": [{"original": {**self.m.ref(old_ledger_path), "bytes": old_ledger_path.stat().st_size},
                                           "preserved_copy": {**self.m.ref(preserved), "bytes": preserved.stat().st_size}}]}
        self.m.save(forensic, resolution)
        self.manifest = {"schema": self.m.MANIFEST_SCHEMA, "run_id": self.m.RUN_ID,
                         "branch": self.m.BRANCH, "cpu_cap_seconds": self.m.CAP, "deadline_utc": self.m.DEADLINE,
                         "threads": 2, "memory_gib": 12, "rss_stop_gib": 8, "reserve_cpu_seconds": 60,
                         "python_executable": str(self.old.python.relative_to(self.root)),
                         "work_directory": str(self.work.relative_to(self.root)),
                         "ledger": str((self.work / "cpu_budget.json").relative_to(self.root)),
                         "source_refs": [self.m.ref(self.source), self.m.ref(self.worker)], "input_refs": [],
                         "carry_ledger": self.old.manifest["carry_ledger"], "carry_state": self.old.manifest["carry_state"],
                         "predecessor_manifest": self.m.ref(prior_manifest_path), "predecessor_ledger": self.m.ref(old_ledger_path),
                         "predecessor_state": self.m.ref(old_state_path), "predecessor_supervisor": self.m.ref(self.old.source),
                         "predecessor_allocation": self.m.ref(self.root / ".runtime/worker-b-diagnostic-allocation.json"),
                         "forensic_resolution": self.m.ref(forensic), "inherited_results": state["results"], "jobs": []}
        self.initial_actual = ledger["consumed_cpu_seconds"]

    def tearDown(self):
        self.old.tearDown()

    def job(self, name="new-good", fork=False):
        if fork:
            self.worker.write_text(self.worker.read_text().replace("if a.fork:", "if True:"))
            self.manifest["source_refs"] = [self.m.ref(self.source), self.m.ref(self.worker)]
        value = json.loads(json.dumps(self.pending_job)); value["argv"][1] = str(self.worker.relative_to(self.root))
        value["max_attempts"] = 1
        return value

    def write_manifest(self):
        p = self.work / "manifest.json"; self.m.save(p, self.manifest)
        return p, self.m.sha(p)

    def execute(self, attempt="recovery-test-001"):
        p, sha = self.write_manifest()
        return subprocess.run([str(self.old.python), str(self.source), "--manifest", str(p), "--manifest-sha256", sha,
                               "--attempt", attempt, "--execute"], cwd=self.root, text=True, capture_output=True, timeout=15)

    def test_actual_and_conservative_debit_stay_distinct(self):
        self.manifest["jobs"] = [self.job()]
        _, sha = self.write_manifest(); carry = self.m.validate(self.manifest, sha)
        self.assertAlmostEqual(carry["actual_closed_cpu_seconds"], self.initial_actual)
        self.assertEqual(carry["conservative_debit_cpu_seconds"], 1000)
        self.assertAlmostEqual(carry["budget_charged_cpu_seconds"], self.initial_actual + 1000)

    def test_inherited_success_is_reused_and_new_wait4_is_saved(self):
        self.manifest["jobs"] = [self.job()]
        completed = self.execute()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        state = self.m.read(self.work / "state.json"); ledger = self.m.read(self.work / "cpu_budget.json")
        self.assertEqual(state["results"]["inherited-success"], self.manifest["inherited_results"]["inherited-success"])
        self.assertEqual(len(ledger["phases"]), 1)
        self.assertEqual(ledger["conservative_debit_cpu_seconds"], 1000)
        self.assertAlmostEqual(ledger["budget_charged_cpu_seconds"] - ledger["actual_closed_cpu_seconds"], 1000)
        row = ledger["phases"][0]
        receipt = self.m.read(self.m.checked(row["wait4_receipt"]))
        self.assertEqual(receipt["observed_wait4_cpu_seconds"], row["actual_cpu_seconds"])
        self.assertFalse(receipt["unsupported_subprocess_observed"])

    def test_lost_recovery_ledger_cannot_refund_conservative_charge(self):
        self.manifest["jobs"] = [self.job()]
        completed = self.execute(); self.assertEqual(completed.returncode, 0, completed.stderr)
        (self.work / "cpu_budget.json").unlink()
        retry = self.execute("recovery-test-002")
        self.assertNotEqual(retry.returncode, 0)
        self.assertIn("allocation exists but ledger is missing", retry.stderr)

    def test_changed_inherited_success_is_rejected(self):
        self.manifest["jobs"] = [self.job()]
        _, sha = self.write_manifest()
        reference = self.manifest["inherited_results"]["inherited-success"]
        self.m.path(reference["path"]).write_text("{\"changed\":true}")
        with self.assertRaisesRegex(ValueError, "Pinned source/input changed"):
            self.m.validate(self.manifest, sha)

    def test_unreviewed_zero_debit_is_rejected(self):
        self.manifest["jobs"] = [self.job()]
        resolution = self.m.read(self.manifest["forensic_resolution"]["path"])
        resolution["conservative_debit_cpu_seconds"] = 0
        self.m.save(self.manifest["forensic_resolution"]["path"], resolution)
        self.manifest["forensic_resolution"] = self.m.ref(self.manifest["forensic_resolution"]["path"])
        _, sha = self.write_manifest()
        with self.assertRaisesRegex(ValueError, "entire reviewed failed CPU request"):
            self.m.validate(self.manifest, sha)

    def test_historical_attempt_limit_cannot_be_reset(self):
        self.manifest["jobs"] = [self.job()]
        self.manifest["jobs"][0]["max_attempts"] = 2
        _, sha = self.write_manifest()
        with self.assertRaisesRegex(ValueError, "cannot reset historical attempt"):
            self.m.validate(self.manifest, sha)

    def test_recovery_cannot_expand_resource_or_scientific_scope(self):
        self.manifest["jobs"] = [self.job()]
        self.manifest["jobs"][0]["cpu_seconds"] += 100
        _, sha = self.write_manifest()
        with self.assertRaisesRegex(ValueError, "cannot expand scientific parameters"):
            self.m.validate(self.manifest, sha)

    def test_preserved_forensic_copy_remains_hash_bound(self):
        self.manifest["jobs"] = [self.job()]
        _, sha = self.write_manifest()
        (self.work / "preserved_old_ledger.json").write_text("{\"rewritten\":true}")
        with self.assertRaisesRegex(ValueError, "Pinned source/input changed"):
            self.m.validate(self.manifest, sha)

    def test_closed_kernel_receipt_cannot_be_modified(self):
        self.manifest["jobs"] = [self.job()]
        completed = self.execute(); self.assertEqual(completed.returncode, 0, completed.stderr)
        ledger = self.m.read(self.work / "cpu_budget.json"); row = ledger["phases"][0]
        p = self.m.path(row["wait4_receipt"]["path"])
        value = self.m.read(p); value["observed_wait4_cpu_seconds"] = 0; self.m.save(p, value)
        with self.assertRaisesRegex(ValueError, "Pinned source/input changed"):
            self.m.seal_result(row, self.manifest["jobs"][0])

    def test_unsupported_child_keeps_real_kernel_rusage_receipt(self):
        self.manifest["jobs"] = [self.job(fork=True)]
        completed = self.execute()
        self.assertNotEqual(completed.returncode, 0)
        ledger = self.m.read(self.work / "cpu_budget.json"); row = ledger["phases"][0]
        self.assertTrue(ledger["blocked_unknown_cpu_or_process_tree"])
        self.assertEqual(row["status"], "BLOCKED_UNACCOUNTED")
        receipt = self.m.read(self.m.checked(row["wait4_receipt"]))
        self.assertTrue(receipt["unsupported_subprocess_observed"])
        self.assertGreater(receipt["observed_wait4_cpu_seconds"], 0)
        self.assertGreater(row["observed_wait4_cpu_seconds"], 0)
        self.assertIsNone(row["actual_cpu_seconds"])
        self.assertEqual(ledger["actual_closed_cpu_seconds"], ledger["initial_actual_closed_cpu_seconds"])


if __name__ == "__main__":
    unittest.main()
