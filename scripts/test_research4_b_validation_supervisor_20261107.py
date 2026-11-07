"""V4 carry, pair admission and evidence-based retry tests; no science."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from scripts import test_research4_b_rootcause_supervisor_20261107 as previous_fixture


SOURCE = Path(__file__).with_name("research4_b_validation_supervisor_20261107.py")
PREPARE = Path(__file__).with_name("prepare_research4_b_validation_20261107.py")


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.prior = previous_fixture.RootcauseTests(); self.prior.setUp()
        self.prior.manifest["jobs"] = [self.prior.job()]
        completed = self.prior.execute(); self.assertEqual(completed.returncode, 0, completed.stderr)
        self.root = self.prior.root; self.python = self.prior.prior.old.python
        self.source = self.root / "scripts" / SOURCE.name; shutil.copyfile(SOURCE, self.source)
        spec = importlib.util.spec_from_file_location("r4_validation_test_module", self.source)
        self.m = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.m)
        self.worker = self.root / "scripts/research4_b_validation_worker_20261107.py"
        self.worker.write_text(self.prior.worker.read_text().replace(
            "from research4_b_rootcause_supervisor_20261107 import", "from research4_b_validation_supervisor_20261107 import"))
        self.work = self.prior.prior.old.original / "followup-validation-v4"; self.work.mkdir()
        self.manifest = {"schema": self.m.MANIFEST_SCHEMA, "run_id": self.m.RUN_ID,
            "branch": self.m.BRANCH, "cpu_cap_seconds": self.m.CAP, "deadline_utc": self.m.DEADLINE,
            "threads": 2, "memory_gib": 12, "rss_stop_gib": 8, "reserve_cpu_seconds": 300,
            "python_executable": str(self.python.relative_to(self.root)), "work_directory": str(self.work.relative_to(self.root)),
            "ledger": str((self.work / "cpu_budget.json").relative_to(self.root)),
            "source_refs": [self.m.ref(self.source), self.m.ref(self.worker)], "input_refs": [],
            "predecessor_manifest": self.m.ref(self.prior.work / "manifest.json"),
            "predecessor_ledger": self.m.ref(self.prior.work / "cpu_budget.json"),
            "predecessor_state": self.m.ref(self.prior.work / "state.json"),
            "predecessor_supervisor": self.m.ref(self.prior.source),
            "predecessor_allocation": self.m.ref(self.root / ".runtime/worker-b-rootcause-allocation.json"), "jobs": []}

    def tearDown(self):
        self.prior.tearDown()

    def job(self, name="validate-a", **overrides):
        job = self.prior.job(name); job["argv"][1] = str(self.worker.relative_to(self.root)); job.update(overrides)
        return job

    def update_worker(self, text):
        self.worker.write_text(text)
        self.manifest["source_refs"] = [self.m.ref(self.source), self.m.ref(self.worker)]

    def execute(self, attempt="validation-test-001", execute=True):
        p = self.work / "manifest.json"; self.m.save(p, self.manifest)
        command = [str(self.python), str(self.source), "--manifest", str(p), "--manifest-sha256", self.m.sha(p), "--attempt", attempt]
        if execute: command.append("--execute")
        return subprocess.run(command, cwd=self.root, text=True, capture_output=True, timeout=15)

    def test_v3_actual_and_permanent_debit_are_preserved(self):
        self.manifest["jobs"] = [self.job()]
        carry = self.m.validate(self.manifest, "0" * 64)
        old = self.m.read(self.prior.work / "cpu_budget.json")
        self.assertEqual(carry["actual_closed_cpu_seconds"], old["actual_closed_cpu_seconds"])
        self.assertEqual(carry["conservative_debit_cpu_seconds"], 1000)
        self.assertEqual(carry["budget_charged_cpu_seconds"], old["budget_charged_cpu_seconds"])

    def test_default_never_claims_budget_or_launches(self):
        self.manifest["jobs"] = [self.job()]
        result = self.execute(execute=False); self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "CHECKED_NOT_RUN")
        self.assertFalse((self.root / ".runtime/worker-b-validation-allocation.json").exists())

    def test_lost_ledger_never_refunds_v4_cost(self):
        self.manifest["jobs"] = [self.job()]
        result = self.execute(); self.assertEqual(result.returncode, 0, result.stderr)
        (self.work / "cpu_budget.json").unlink()
        result = self.execute("validation-test-002")
        self.assertNotEqual(result.returncode, 0); self.assertIn("ledger is missing", result.stderr)

    def test_real_closed_operational_failure_can_retry_with_cost_retained(self):
        text = self.worker.read_text().replace("out=Path(a.output);out.mkdir(parents=True)",
            "if 'attempt001' in a.output: raise SystemExit(9)\nout=Path(a.output);out.mkdir(parents=True)")
        self.update_worker(text); self.manifest["jobs"] = [self.job(max_attempts=2)]
        result = self.execute(); self.assertEqual(result.returncode, 0, result.stderr)
        ledger = self.m.read(self.work / "cpu_budget.json"); state = self.m.read(self.work / "state.json")
        self.assertEqual(len(ledger["phases"]), 2)
        self.assertEqual(state["jobs"]["validate-a"]["status"], "DONE_SEALED")
        self.assertTrue(ledger["phases"][0]["automatic_retry_decision"]["allowed"])
        self.assertGreater(ledger["phases"][0]["actual_cpu_seconds"], 0)
        self.assertAlmostEqual(ledger["actual_closed_cpu_seconds"] - ledger["initial_actual_closed_cpu_seconds"],
                               sum(r["actual_cpu_seconds"] for r in ledger["phases"]))

    def test_scf_qualification_is_not_retried_even_in_new_invocation(self):
        self.update_worker(self.worker.read_text().replace("'status':'TEST_COMPLETED'", "'status':'SCF_NOT_QUALIFIED'"))
        self.manifest["jobs"] = [self.job(max_attempts=2)]
        result = self.execute(); self.assertEqual(result.returncode, 0, result.stderr)
        result = self.execute("validation-test-002"); self.assertEqual(result.returncode, 0, result.stderr)
        ledger = self.m.read(self.work / "cpu_budget.json")
        self.assertEqual(len(ledger["phases"]), 1)
        self.assertFalse(ledger["phases"][0]["automatic_retry_decision"]["allowed"])
        self.assertIn("SCF_QUALIFICATION", ledger["phases"][0]["automatic_retry_decision"]["reason"])

    def test_pair_requires_both_caps_before_either_point(self):
        self.manifest["jobs"] = [self.job("minus", cpu_seconds=30000, admission_group="pair"),
                                 self.job("plus", cpu_seconds=30000, admission_group="pair"), self.job("independent")]
        result = self.execute(); self.assertEqual(result.returncode, 0, result.stderr)
        state = self.m.read(self.work / "state.json"); ledger = self.m.read(self.work / "cpu_budget.json")
        self.assertEqual([r["job_id"] for r in ledger["phases"]], ["independent"])
        self.assertEqual(state["jobs"]["minus"]["status"], "BLOCKED_RESOURCE_OR_DEADLINE")
        self.assertEqual(state["jobs"]["plus"]["status"], "BLOCKED_RESOURCE_OR_DEADLINE")

    def test_pair_retry_keeps_partner_cap_reserved(self):
        self.manifest["jobs"] = [self.job("minus", admission_group="pair", max_attempts=2),
                                 self.job("plus", admission_group="pair")]
        self.assertEqual(self.m.admission_cpu_required(self.manifest, self.manifest["jobs"][0], {}), 40)
        self.assertEqual(self.m.admission_cpu_required(self.manifest, self.manifest["jobs"][1], {"minus": {}}), 20)
        text = self.worker.read_text().replace("out=Path(a.output);out.mkdir(parents=True)",
            "if 'minus-attempt001' in a.output: raise SystemExit(9)\nout=Path(a.output);out.mkdir(parents=True)")
        self.update_worker(text)
        result = self.execute(); self.assertEqual(result.returncode, 0, result.stderr)
        phases = self.m.read(self.work / "cpu_budget.json")["phases"]
        self.assertEqual([r["admission_cpu_caps_required"] for r in phases], [40, 40, 20])

    def test_unsupported_child_stops_queue_and_preserves_wait4_evidence(self):
        self.update_worker(self.worker.read_text().replace("if a.fork:", "if True:"))
        self.manifest["jobs"] = [self.job(max_attempts=2)]
        result = self.execute(); self.assertNotEqual(result.returncode, 0)
        ledger = self.m.read(self.work / "cpu_budget.json")
        self.assertEqual(len(ledger["phases"]), 1)
        self.assertTrue(ledger["blocked_unknown_cpu_or_process_tree"])
        self.assertGreater(ledger["phases"][0]["observed_wait4_cpu_seconds"], 0)

    def test_prepare_pins_spec_and_does_not_claim_or_execute(self):
        shutil.copyfile(self.prior.work / "manifest.json", self.prior.work / "manifest_ROOT_AUTHORIZED.json")
        prepare_path = self.root / "scripts" / PREPARE.name; shutil.copyfile(PREPARE, prepare_path)
        spec_path = self.work / "reviewed_spec.json"
        self.m.save(spec_path, {"schema": "research4-b-validation-spec-v4", "jobs": [self.job()],
            "source_refs": [self.m.ref(self.worker)], "input_refs": [], "upload": {"enabled": False}})
        result = subprocess.run([str(self.python), str(prepare_path), "--spec", str(spec_path),
            "--spec-sha256", self.m.sha(spec_path)], cwd=self.root, text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "PREPARED_NOT_RUN")
        self.assertFalse((self.work / "cpu_budget.json").exists())
        self.assertFalse((self.root / ".runtime/worker-b-validation-allocation.json").exists())


if __name__ == "__main__":
    unittest.main()
