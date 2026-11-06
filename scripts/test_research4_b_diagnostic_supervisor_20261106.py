"""Meaningful supervisor checks in disposable repositories; no quantum jobs."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


SOURCE = Path(__file__).with_name("research4_b_diagnostic_supervisor_20261106.py")


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="r4-b-diagnostic-supervisor-test-")
        self.root = Path(self.temp.name)
        scripts = self.root / "scripts"; scripts.mkdir()
        self.source = scripts / SOURCE.name
        shutil.copyfile(SOURCE, self.source)
        spec = importlib.util.spec_from_file_location("r4_diagnostic_test_module", self.source)
        self.m = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.m)
        self.original = self.root / self.m.ORIGINAL_WORK
        self.original.mkdir(parents=True)
        original_supervisor = scripts / "research4_parallel_b_supervisor_20261106.py"
        original_supervisor.write_text("# sealed original supervisor test fixture\n")
        original_job = {"id": "original-job", "operation": "test"}
        original_manifest = {"jobs": [original_job], "source_refs": [self.m.ref(original_supervisor)], "input_refs": []}
        self.m.save(self.original / "manifest.json", original_manifest)
        old_manifest_sha = self.m.sha(self.original / "manifest.json")
        self.m.save(self.original / "authorization.json", {"enable_execution": True, "manifest_sha256": old_manifest_sha,
                    "supervisor_sha256": self.m.sha(original_supervisor)})
        self.m.save(self.original / "prepared-launch.json", {"manifest_sha256": old_manifest_sha,
                    "authorization_sha256": self.m.sha(self.original / "authorization.json")})
        old_output = self.original / "science/original-job-attempt001"
        old_output.mkdir(parents=True)
        old_proof = self.original / "execution/original-job.proof.json"
        old_log = self.original / "execution/original-job.log"
        proof = {"phase": "original", "case_binding_sha256": self.m.digest(original_job),
                 "manifest_sha256": old_manifest_sha, "supervisor_source_sha256": self.m.sha(original_supervisor)}
        self.m.save(old_proof, proof)
        old_log.write_text("closed original fixture\n")
        self.m.save(old_output / "result.json", {"resource_supervision": {"proof": proof,
                    "proof_sha256": self.m.sha(old_proof), "live_parent_process_and_source_verified": True}})
        old_row = {"phase": "original", "status": "EXITED", "CPU_accounting": self.m.ACCOUNTING,
                   "actual_cpu_seconds": 100.25, "exit_code": 0, "case_binding_sha256": self.m.digest(original_job),
                   "proof_path": str(old_proof), "proof_sha256": self.m.sha(old_proof),
                   "log": str(old_log), "log_sha256": self.m.sha(old_log), "output_directory": str(old_output),
                   "result_sha256": self.m.sha(old_output / "result.json")}
        old_result_ref = self.m.ref(old_output / "result.json")
        old_seal = {"result": old_result_ref, "closed_row_sha256": self.m.digest(old_row),
                    "files": [{"relative_path": "result.json", "size_bytes": (old_output / "result.json").stat().st_size,
                               "sha256": self.m.sha(old_output / "result.json")} ]}
        ledger = {"run_id": self.m.RUN_ID, "cpu_cap_seconds": self.m.CAP, "deadline_utc": self.m.DEADLINE,
                  "initial_consumed_cpu_seconds": 0, "consumed_cpu_seconds": 100.25,
                  "manifest_sha256": old_manifest_sha, "supervisor_sha256": self.m.sha(original_supervisor),
                  "phases": [old_row]}
        self.m.save(self.original / "cpu_budget_B.json", ledger)
        self.m.save(self.original / "state.json", {"manifest_sha256": old_manifest_sha,
                    "invocations": [{"id": "original", "status": "CLOSED"}],
                    "results": {"original-job": old_result_ref}, "jobs": {"original-job": {"phase": "original", "status": "DONE_SEALED", "seal": old_seal}}})
        claim = {"run_id": self.m.RUN_ID, "nonce": "test-nonce-1234567890", "release_sha256": "a" * 64, "source_commit": "b" * 40}
        self.m.save(self.original / "owner_claim.json", claim)
        self.m.save(self.root / ".runtime/worker-b-ownership" / (self.m.RUN_ID + ".json"),
                    {**claim, "remote_verified": True, "branch": self.m.BRANCH, "repository_root": str(self.root)})
        self.python = self.root / ".runtime/test-venv/bin/python"
        self.python.parent.mkdir(parents=True)
        self.python.symlink_to(sys.executable)
        self.worker = scripts / "research4_b_diagnostic_test_payload.py"
        self.worker.write_text('''from pathlib import Path
import argparse,hashlib,json,os,time
from research4_b_diagnostic_supervisor_20261106 import verify_resource_supervision
p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--fail',action='store_true');a=p.parse_args()
proof=verify_resource_supervision(expected_worker_path=__file__,expected_threads=2)
t=time.process_time()
while time.process_time()-t<0.04: pass
if a.fail: raise SystemExit(9)
out=Path(a.output);out.mkdir(parents=True)
result={'schema':'test-diagnostic-result-v1','status':'TEST_COMPLETED','output_directory':str(out),'resource_supervision':proof}
(out/'result.json').write_text(json.dumps(result))
''')
        self.work = self.original / "followup-test-v1"
        self.manifest = {"schema": self.m.MANIFEST_SCHEMA, "run_id": self.m.RUN_ID,
                         "branch": self.m.BRANCH, "cpu_cap_seconds": self.m.CAP, "deadline_utc": self.m.DEADLINE,
                         "threads": 2, "memory_gib": 12, "rss_stop_gib": 8, "reserve_cpu_seconds": 60,
                         "python_executable": str(self.python.relative_to(self.root)),
                         "work_directory": str(self.work.relative_to(self.root)),
                         "ledger": str((self.work / "cpu_budget.json").relative_to(self.root)),
                         "source_refs": [self.m.ref(self.source), self.m.ref(self.worker)], "input_refs": [],
                         "carry_ledger": self.m.ref(self.original / "cpu_budget_B.json"),
                         "carry_state": self.m.ref(self.original / "state.json"), "jobs": []}
        subprocess.run(["git", "init", "-q", "-b", self.m.BRANCH, str(self.root)], check=True)

    def tearDown(self):
        self.temp.cleanup()

    def job(self, name, fail=False, **kwargs):
        return {"id": name, "argv": [str(self.python.relative_to(self.root)), str(self.worker.relative_to(self.root)), "--output", "{output}"] + (["--fail"] if fail else []),
                "cpu_seconds": 20, "wall_seconds": 61, "depends_on": [], "max_attempts": 1,
                "result_schema": "test-diagnostic-result-v1", "success_statuses": ["TEST_COMPLETED"], **kwargs}

    def write_manifest(self):
        self.work.mkdir(parents=True, exist_ok=True)
        p = self.work / "manifest.json"; self.m.save(p, self.manifest)
        return p, self.m.sha(p)

    def execute(self, attempt="test-001", detach=False):
        p, digest = self.write_manifest()
        command = [str(self.python), str(self.source), "--manifest", str(p), "--manifest-sha256", digest,
                   "--attempt", attempt, "--execute"] + (["--detach"] if detach else [])
        return subprocess.run(command, cwd=self.root, text=True, capture_output=True, timeout=15)

    def test_carry_and_ledger_cannot_refund_original_spending(self):
        self.manifest["jobs"] = [self.job("good")]
        p, digest = self.write_manifest()
        carry = self.m.validate(self.manifest, digest)
        self.assertEqual(carry["actual_closed_cpu_seconds"], 100.25)
        ledger = {"schema": self.m.LEDGER_SCHEMA, "run_id": self.m.RUN_ID, "manifest_sha256": digest,
                  "supervisor_sha256": self.m.sha(self.source), "cpu_cap_seconds": self.m.CAP,
                  "deadline_utc": self.m.DEADLINE, "carry": carry, "initial_consumed_cpu_seconds": 0,
                  "consumed_cpu_seconds": 0, "remaining_cpu_seconds": self.m.CAP, "phases": []}
        with self.assertRaisesRegex(ValueError, "Historical actual cost"):
            self.m.validate_ledger(ledger, self.manifest, digest, carry)

    def test_duplicate_allocation_manifest_is_rejected(self):
        self.manifest["jobs"] = [self.job("good")]
        _, digest = self.write_manifest(); carry = self.m.validate_carry(self.manifest)
        self.m.register_allocation(self.manifest, digest, carry)
        self.m.register_allocation(self.manifest, digest, carry)
        with self.assertRaisesRegex(ValueError, "cannot be reset/replaced"):
            self.m.register_allocation(self.manifest, "f" * 64, carry)

    def test_claim_with_lost_ledger_never_recreates_fresh_budget(self):
        self.manifest["jobs"] = [self.job("good")]
        result = self.execute()
        self.assertEqual(result.returncode, 0, result.stderr)
        (self.work / "cpu_budget.json").unlink()
        retry = self.execute("test-002")
        self.assertNotEqual(retry.returncode, 0)
        self.assertIn("allocation exists but ledger is missing", retry.stderr)
        self.assertFalse((self.work / "cpu_budget.json").exists())

    def test_sealed_original_result_tampering_is_not_re_pinned(self):
        self.manifest["jobs"] = [self.job("good")]
        _, digest = self.write_manifest()
        p = self.original / "science/original-job-attempt001/result.json"
        p.write_text("{\"silently_changed\":true}")
        with self.assertRaisesRegex(ValueError, "Pinned source/input changed"):
            self.m.validate(self.manifest, digest)

    def test_changed_original_ledger_is_rejected_before_launch(self):
        self.manifest["jobs"] = [self.job("good")]
        _, digest = self.write_manifest()
        original = self.original / "cpu_budget_B.json"
        value = self.m.read(original); value["consumed_cpu_seconds"] = 0; self.m.save(original, value)
        with self.assertRaisesRegex(ValueError, "Pinned source/input changed"):
            self.m.validate(self.manifest, digest)

    def test_failed_costs_preserved_and_independent_job_runs(self):
        self.manifest["jobs"] = [self.job("failed", fail=True), self.job("dependent", depends_on=["failed"]),
                                 self.job("too-large", cpu_seconds=self.m.CAP), self.job("independent")]
        result = self.execute()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        state = self.m.read(self.work / "state.json"); ledger = self.m.read(self.work / "cpu_budget.json")
        self.assertEqual(state["invocations"][-1]["status"], "CLOSED_WITH_BLOCKED_JOBS")
        self.assertEqual(state["jobs"]["failed"]["status"], "FAILED_CLOSED")
        self.assertEqual(state["jobs"]["dependent"]["status"], "BLOCKED_DEPENDENCY")
        self.assertEqual(state["jobs"]["too-large"]["status"], "BLOCKED_RESOURCE_OR_DEADLINE")
        self.assertEqual(state["jobs"]["independent"]["status"], "DONE_SEALED")
        self.assertGreater(ledger["phases"][0]["actual_cpu_seconds"], .03)
        self.assertAlmostEqual(ledger["consumed_cpu_seconds"], 100.25 + sum(r["actual_cpu_seconds"] for r in ledger["phases"]))
        before = json.dumps(ledger["phases"], sort_keys=True)
        resumed = self.execute("test-002")
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        self.assertEqual(json.dumps(self.m.read(self.work / "cpu_budget.json")["phases"], sort_keys=True), before)

    def test_detach_receipt_and_real_supervised_result(self):
        self.manifest["jobs"] = [self.job("good")]
        result = self.execute(detach=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        receipt = json.loads(result.stdout)
        self.assertEqual(receipt["status"], "ACTUAL_DIAGNOSTIC_PARENT_LAUNCHED_NOT_SCIENCE_COMPLETE")
        self.assertTrue(receipt["same_original_B_budget"])
        for _ in range(100):
            p = self.work / "state.json"
            if p.is_file():
                state = self.m.read(p)
                if state["invocations"][-1]["status"] != "RUNNING":
                    break
            time.sleep(.05)
        self.assertEqual(state["invocations"][-1]["status"], "CLOSED")
        self.assertEqual(state["jobs"]["good"]["status"], "DONE_SEALED")
        output = self.m.read(self.m.path(state["results"]["good"]["path"]))
        self.assertTrue(output["resource_supervision"]["live_parent_process_and_source_verified"])
        self.assertEqual(output["resource_supervision"]["proof"]["cpu_seconds"], 20)

    def test_automatic_return_only_after_real_payload_closure(self):
        helper = self.root / "scripts/research4_parallel_b_git_return_20261106.py"
        helper.write_text('''from pathlib import Path
import argparse,json
p=argparse.ArgumentParser();p.add_argument('--repo');p.add_argument('--manifest');p.add_argument('--manifest-sha256');p.add_argument('--attempt');p.add_argument('--execute',action='store_true');a=p.parse_args()
root=Path(a.repo);m=json.loads(Path(a.manifest).read_text());l=json.loads((root/m['ledger']).read_text())
assert all(r['status']!='RUNNING' and r['actual_cpu_seconds']>=0 for r in l['phases'])
out=root/m['ledger'];(out.parent/'test-return-received.json').write_text(json.dumps({'attempt':a.attempt,'all_payloads_closed':True}))
''')
        self.work.mkdir(parents=True)
        return_manifest = self.work / "return_manifest.json"
        self.m.save(return_manifest, {"role": "B", "run_id": self.m.RUN_ID,
                    "work_directory": self.m.ORIGINAL_WORK, "ledger": self.manifest["ledger"],
                    "upload": {"branch": self.m.BRANCH}})
        self.manifest["source_refs"].append(self.m.ref(helper))
        self.manifest["upload"] = {"enabled": True, "helper": self.m.ref(helper), "manifest": self.m.ref(return_manifest)}
        self.manifest["jobs"] = [self.job("good")]
        completed = self.execute()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        state = self.m.read(self.work / "state.json")
        receipt = self.m.read(self.work / "test-return-received.json")
        self.assertTrue(receipt["all_payloads_closed"])
        self.assertEqual(state["git_return_attempts"][0]["returncode"], 0)
        self.assertEqual(receipt["attempt"], "test-001-return-001")


if __name__ == "__main__":
    unittest.main()
