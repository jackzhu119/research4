"""Third-round allocation/provenance/closure tests, without quantum work."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from scripts import test_research4_b_diagnostic_supervisor_recovery_20261107 as recovery_fixture


SOURCE = Path(__file__).with_name("research4_b_rootcause_supervisor_20261107.py")


class RootcauseTests(unittest.TestCase):
    def setUp(self):
        self.prior = recovery_fixture.RecoveryTests()
        self.prior.setUp()
        self.prior.manifest["jobs"] = [self.prior.job()]
        completed = self.prior.execute()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.root = self.prior.root
        self.source = self.root / "scripts" / SOURCE.name
        shutil.copyfile(SOURCE, self.source)
        spec = importlib.util.spec_from_file_location("r4_rootcause_test_module", self.source)
        self.m = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.m)
        self.worker = self.root / "scripts/research4_b_rootcause_worker_20261107.py"
        self.worker.write_text(self.prior.worker.read_text().replace(
            "from research4_b_diagnostic_supervisor_recovery_20261107 import",
            "from research4_b_rootcause_supervisor_20261107 import"))
        self.work = self.prior.old.original / "followup-rootcause-v3"
        self.work.mkdir()
        self.manifest = {"schema": self.m.MANIFEST_SCHEMA, "run_id": self.m.RUN_ID,
                         "branch": self.m.BRANCH, "cpu_cap_seconds": self.m.CAP, "deadline_utc": self.m.DEADLINE,
                         "threads": 2, "memory_gib": 12, "rss_stop_gib": 8, "reserve_cpu_seconds": 60,
                         "python_executable": str(self.prior.old.python.relative_to(self.root)),
                         "work_directory": str(self.work.relative_to(self.root)),
                         "ledger": str((self.work / "cpu_budget.json").relative_to(self.root)),
                         "source_refs": [self.m.ref(self.source), self.m.ref(self.worker)], "input_refs": [],
                         "predecessor_manifest": self.m.ref(self.prior.work / "manifest.json"),
                         "predecessor_ledger": self.m.ref(self.prior.work / "cpu_budget.json"),
                         "predecessor_state": self.m.ref(self.prior.work / "state.json"),
                         "predecessor_supervisor": self.m.ref(self.prior.source),
                         "predecessor_allocation": self.m.ref(self.root / ".runtime/worker-b-diagnostic-recovery-allocation.json"),
                         "jobs": []}

    def tearDown(self):
        self.prior.tearDown()

    def job(self, name="cause-a", **overrides):
        job = self.prior.old.job(name)
        job["argv"][1] = str(self.worker.relative_to(self.root))
        job.update(overrides)
        return job

    def write_manifest(self):
        p = self.work / "manifest.json"; self.m.save(p, self.manifest)
        return p, self.m.sha(p)

    def execute(self, attempt="rootcause-test-001", execute=True):
        p, sha = self.write_manifest()
        command = [str(self.prior.old.python), str(self.source), "--manifest", str(p),
                   "--manifest-sha256", sha, "--attempt", attempt]
        if execute:
            command.append("--execute")
        return subprocess.run(command, cwd=self.root, text=True, capture_output=True, timeout=15)

    def test_completed_v2_cost_and_permanent_debit_are_retained(self):
        self.manifest["jobs"] = [self.job()]
        _, sha = self.write_manifest(); carry = self.m.validate(self.manifest, sha)
        old = self.m.read(self.prior.work / "cpu_budget.json")
        self.assertEqual(carry["actual_closed_cpu_seconds"], old["actual_closed_cpu_seconds"])
        self.assertEqual(carry["conservative_debit_cpu_seconds"], 1000)
        self.assertEqual(carry["budget_charged_cpu_seconds"], old["budget_charged_cpu_seconds"])

    def test_default_check_does_not_claim_allocation_or_execute(self):
        self.manifest["jobs"] = [self.job()]
        completed = self.execute(execute=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "CHECKED_NOT_RUN")
        self.assertFalse((self.root / ".runtime/worker-b-rootcause-allocation.json").exists())
        self.assertFalse((self.work / "cpu_budget.json").exists())

    def test_new_queue_only_and_wait4_receipt_preserved(self):
        self.manifest["jobs"] = [self.job()]
        completed = self.execute()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        state = self.m.read(self.work / "state.json"); ledger = self.m.read(self.work / "cpu_budget.json")
        self.assertEqual(set(state["results"]), {"cause-a"})
        self.assertEqual(len(ledger["phases"]), 1)
        self.assertEqual(ledger["conservative_debit_cpu_seconds"], 1000)
        self.assertAlmostEqual(ledger["budget_charged_cpu_seconds"] - ledger["actual_closed_cpu_seconds"], 1000)
        self.assertEqual(self.m.verify_wait4_receipt(ledger["phases"][0])["exit_code"], 0)
        retry = self.execute("rootcause-test-002")
        self.assertEqual(retry.returncode, 0, retry.stderr)
        self.assertEqual(len(self.m.read(self.work / "cpu_budget.json")["phases"]), 1)

    def test_claim_with_lost_ledger_never_recreates_budget(self):
        self.manifest["jobs"] = [self.job()]
        completed = self.execute(); self.assertEqual(completed.returncode, 0, completed.stderr)
        (self.work / "cpu_budget.json").unlink()
        retry = self.execute("rootcause-test-002")
        self.assertNotEqual(retry.returncode, 0)
        self.assertIn("allocation exists but ledger is missing", retry.stderr)

    def test_claim_cannot_be_reused_for_changed_manifest(self):
        self.manifest["jobs"] = [self.job()]
        completed = self.execute(); self.assertEqual(completed.returncode, 0, completed.stderr)
        self.manifest["jobs"] = [self.job("second-budget")]
        retry = self.execute("rootcause-test-002")
        self.assertNotEqual(retry.returncode, 0)
        self.assertIn("Ledger/source binding changed", retry.stderr)

    def test_v2_success_artifact_and_full_chain_stay_pinned(self):
        self.manifest["jobs"] = [self.job()]
        _, sha = self.write_manifest()
        old = self.m.read(self.prior.work / "state.json")
        result = self.m.path(old["results"]["new-good"]["path"])
        result.write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, "Closed result changed"):
            self.m.validate(self.manifest, sha)

    def test_v3_dependencies_cannot_use_implicit_inherited_results(self):
        self.manifest["jobs"] = [self.job(depends_on=["inherited-success"])]
        _, sha = self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Dependencies must name preceding jobs"):
            self.m.validate(self.manifest, sha)

    def test_two_attempt_limit_is_manifest_enforced(self):
        self.manifest["jobs"] = [self.job(max_attempts=3)]
        _, sha = self.write_manifest()
        with self.assertRaisesRegex(ValueError, "At most2"):
            self.m.validate(self.manifest, sha)

    def test_resource_block_does_not_prevent_independent_low_cost_job(self):
        self.manifest["jobs"] = [self.job("expensive", cpu_seconds=self.m.CAP), self.job("independent")]
        completed = self.execute()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        state = self.m.read(self.work / "state.json")
        self.assertEqual(state["jobs"]["expensive"]["status"], "BLOCKED_RESOURCE_OR_DEADLINE")
        self.assertEqual(state["jobs"]["independent"]["status"], "DONE_SEALED")
        self.assertEqual(len(self.m.read(self.work / "cpu_budget.json")["phases"]), 1)

    def test_modified_wait4_receipt_is_rejected_on_resume(self):
        self.manifest["jobs"] = [self.job()]
        completed = self.execute(); self.assertEqual(completed.returncode, 0, completed.stderr)
        ledger = self.m.read(self.work / "cpu_budget.json")
        receipt = self.m.path(ledger["phases"][0]["wait4_receipt"]["path"])
        receipt.write_text('{"modified": true}')
        retry = self.execute("rootcause-test-002")
        self.assertNotEqual(retry.returncode, 0)
        self.assertIn("Pinned source/input changed", retry.stderr)


if __name__ == "__main__":
    unittest.main()
