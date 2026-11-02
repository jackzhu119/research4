"""Actual cheap child-process checks; no quantum calculations in these tests."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_quantum_budget_20261031.py"


class BudgetTests(unittest.TestCase):
    def run_command(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                              capture_output=True, text=True, timeout=15)

    def test_refuses_above_total_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = Path(temp) / "budget.json"
            self.assertNotEqual(self.run_command("--ledger", ledger, "--initialize", "--total-cpu-seconds", 3301).returncode, 0)
            self.assertFalse(ledger.exists())

    def test_records_real_child_cpu_and_refuses_phase_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = root / "budget.json"
            self.assertEqual(self.run_command("--ledger", ledger, "--initialize", "--total-cpu-seconds", 20).returncode, 0)
            args = ["--ledger", ledger, "--phase", "fixture", "--cpu-seconds", 5,
                    "--wall-seconds", 5, "--log", root / "out.log", "--", sys.executable, "-c", "print('fixture')"]
            self.assertEqual(self.run_command(*args).returncode, 0)
            data = json.loads(ledger.read_text())
            self.assertGreater(data["consumed_cpu_seconds"], 0)
            self.assertEqual(data["phases"][0]["exit_code"], 0)
            self.assertIn("fixture", (root / "out.log").read_text())
            before = ledger.read_bytes()
            self.assertNotEqual(self.run_command(*args).returncode, 0)
            self.assertEqual(before, ledger.read_bytes())

    def test_actual_cpu_limit_stops_unbounded_synthetic_child(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = root / "budget.json"
            self.run_command("--ledger", ledger, "--initialize", "--total-cpu-seconds", 10)
            run = self.run_command("--ledger", ledger, "--phase", "cpu_fixture", "--cpu-seconds", 5,
                                   "--wall-seconds", 10, "--log", root / "out.log", "--", sys.executable,
                                   "-c", "while True: pass")
            self.assertNotEqual(run.returncode, 0)
            data = json.loads(ledger.read_text())
            self.assertLess(data["consumed_cpu_seconds"], 6)
            self.assertGreater(data["consumed_cpu_seconds"], 3)
            self.assertLess(data["phases"][0]["exit_code"], 0)

    def test_wall_limit_stops_sleeping_child(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = root / "budget.json"
            self.run_command("--ledger", ledger, "--initialize", "--total-cpu-seconds", 10)
            run = self.run_command("--ledger", ledger, "--phase", "wall_fixture", "--cpu-seconds", 5,
                                   "--wall-seconds", 1, "--log", root / "out.log", "--", sys.executable,
                                   "-c", "import time; time.sleep(30)")
            self.assertNotEqual(run.returncode, 0)
            row = json.loads(ledger.read_text())["phases"][0]
            self.assertTrue(row["wall_terminated"])
            self.assertLess(row["elapsed_wall_seconds"], 7)


if __name__ == "__main__":
    unittest.main()
