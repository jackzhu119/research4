"""Actual subprocess tests for finite CPU accounting, gates and interruption."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_pyscf_autonomous_90000_20261108.py"
spec = importlib.util.spec_from_file_location("budget", SCRIPT)
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=budget.ROOT / "reports", prefix="budget-test-")
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def config(self, jobs, cap=90000):
        value = {"run_dir": str(self.directory / "run"), "total_cpu_seconds": cap, "jobs": jobs}
        path = self.directory / "config.json"
        path.write_text(json.dumps(value))
        return path

    def job(self, name, code, **extra):
        return {"id": name, "argv": [sys.executable, "-c", code], "cpu_seconds": 5,
                "wall_seconds": 10, "success_statuses": ["PASSED"], **extra}

    def invoke(self, config):
        result = subprocess.run([sys.executable, str(SCRIPT), "--config", str(config), "--run"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        return json.loads((self.directory / "run/ledger.json").read_text())

    def test_success_requires_status_and_failed_dependency_skips(self):
        okay = "import os,json;json.dump({'status':'PASSED'},open(os.environ['R4_JOB_OUTPUT']+'/result.json','w'))"
        jobs = [self.job("no-result", "pass"), self.job("dependent", okay, depends_on=["no-result"]),
                self.job("independent", okay)]
        ledger = self.invoke(self.config(jobs))
        self.assertEqual([ledger["jobs"][j]["status"] for j in ["no-result", "dependent", "independent"]],
                         ["FAILED_CLOSED", "SKIPPED_DEPENDENCY", "SUCCESS"])
        self.assertGreater(ledger["payload_cpu_seconds"], 0)
        self.assertGreater(ledger["supervisor_cpu_seconds"], 0)

    def test_live_cpu_limit_records_real_failed_cost(self):
        ledger = self.invoke(self.config([self.job("burn", "while True: pass")]))
        row = ledger["jobs"]["burn"]
        self.assertEqual(row["status"], "FAILED_CLOSED")
        self.assertEqual(row["stop_reason"], "JOB_CPU_GUARD")
        self.assertGreater(row["actual_cpu_seconds"], 2.5)
        self.assertLess(row["actual_cpu_seconds"], 5)
        self.assertTrue(row["all_owned_processes_closed"])

    def test_children_are_reaped_and_cpu_is_counted(self):
        code = ("import subprocess,sys,os,json;"
                "subprocess.run([sys.executable,'-c','sum(i*i for i in range(1000000))']);"
                "json.dump({'status':'PASSED'},open(os.environ['R4_JOB_OUTPUT']+'/result.json','w'))")
        ledger = self.invoke(self.config([self.job("child", code)]))
        row = ledger["jobs"]["child"]
        self.assertEqual(row["status"], "SUCCESS")
        self.assertGreater(row["actual_cpu_seconds"], 0.04)

    def test_unknown_running_blocks_recovery(self):
        config_path = self.config([self.job("wait", "pass")])
        config, directory, digest, _ = budget.config_load(config_path)
        directory.mkdir()
        budget.save(directory / "ledger.json", {"schema": budget.SCHEMA, "config_sha256": digest,
                    "supervisor_sha256": budget.sha(SCRIPT), "jobs": {"wait": {"status": "RUNNING"}}})
        with self.assertRaisesRegex(ValueError, "Unclosed"):
            budget.Queue(config, directory, digest, config_path).run(resume=True)

    def test_frozen_source_or_input_change_blocks_execution(self):
        source = self.directory / "worker.py"
        source.write_text("pass\n")
        config = self.config([{"id": "frozen", "argv": [sys.executable, str(source)],
                              "cpu_seconds": 5, "wall_seconds": 10, "success_statuses": ["PASSED"]}])
        value = json.loads(config.read_text())
        value["frozen_refs"] = [{"path": str(source), "sha256": budget.sha(source)}]
        config.write_text(json.dumps(value))
        budget.config_load(config)
        source.write_text("raise Exception('changed')\n")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            budget.config_load(config)

    def test_invalid_result_is_closed_without_unknown_cost(self):
        code = "import os;open(os.environ['R4_JOB_OUTPUT']+'/result.json','w').write('invalid')"
        ledger = self.invoke(self.config([self.job("malformed", code)]))
        self.assertEqual(ledger["jobs"]["malformed"]["status"], "FAILED_CLOSED")
        self.assertFalse(ledger.get("accounting_blocked", False))
        self.assertEqual(ledger["preparation_cpu_seconds_reserved"], 2000)
        self.assertGreater(ledger["consumed_cpu_seconds"], 2000)

    def test_reserve_and_configuration_hash_are_enforced(self):
        config = self.config([self.job("large", "pass", cpu_seconds=88001)])
        with self.assertRaises(ValueError):
            budget.config_load(config)
        config = self.config([self.job("safe", "pass")])
        with self.assertRaisesRegex(ValueError, "hash"):
            budget.config_load(config, "0" * 64)

    def test_stop_file_contains_worker(self):
        config = self.config([self.job("wait", "import time; time.sleep(30)")])
        child = subprocess.Popen([sys.executable, str(SCRIPT), "--config", str(config), "--run"],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ledger_path = self.directory / "run/ledger.json"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if ledger_path.exists() and json.loads(ledger_path.read_text()).get("jobs", {}).get("wait", {}).get("worker_identity"):
                break
            time.sleep(.05)
        (self.directory / "run/STOP").touch()
        _, err = child.communicate(timeout=10)
        self.assertEqual(child.returncode, 0, err.decode())
        row = json.loads(ledger_path.read_text())["jobs"]["wait"]
        self.assertEqual(row["stop_reason"], "STOP_FILE")
        self.assertFalse(budget.alive(row["worker_identity"]))


if __name__ == "__main__":
    (budget.ROOT / "reports").mkdir(exist_ok=True)
    unittest.main()
