"""Bounded real process fixtures and metadata guards, never quantum calculations."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_pyscf_continuation_budget_20261031.py"
spec = importlib.util.spec_from_file_location("continuation_budget", SCRIPT)
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


class ContinuationBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "fixture.chk"
        self.checkpoint.write_bytes(b"Non-quantum synthetic checkpoint hash fixture")
        self.authorization = self.root / "authorization.json"
        self.auth = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                     "authorization_text": "Synthetic test fixture, not a scientific computation authorization",
                     "new_budget_is_separately_authorized": True, "prior_budget_preserved": True,
                     "source_base_commit": budget.SOURCE_BASE,
                     "source_checkpoint_sha256": budget.sha(self.checkpoint),
                     "source_checkpoint_path": str(self.checkpoint),
                     "max_quantum_cpu_seconds": 43200, "max_run_wall_seconds": 28800,
                     "max_phase_cpu_seconds": 21600, "max_phase_wall_seconds": 14400,
                     "max_threads": 4, "max_memory_gib": 12}
        self.authorization.write_text(json.dumps(self.auth))
        self.ledger = self.root / "cpu_budget.json"

    def command(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], cwd=ROOT,
                              text=True, capture_output=True, timeout=12)

    def init_args(self, **changes):
        values = {"--ledger": self.ledger, "--authorization": self.authorization,
                  "--authorization-sha256": budget.sha(self.authorization),
                  "--source-checkpoint": self.checkpoint,
                  "--source-checkpoint-sha256": budget.sha(self.checkpoint),
                  "--source-base-commit": budget.SOURCE_BASE,
                  "--total-cpu-seconds": 500, "--total-wall-seconds": 300}
        values.update(changes)
        return [item for pair in values.items() for item in pair] + ["--initialize"]

    def initialize(self):
        result = self.command(*self.init_args())
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(self.ledger.read_text())

    def phase_args(self, code="print('fixture')", phase="fixture", **changes):
        values = {"--ledger": self.ledger, "--phase": phase, "--cpu-seconds": 120,
                  "--wall-seconds": 100, "--threads": 2, "--memory-gib": 1,
                  "--log": self.root / (phase + ".log")}
        values.update(changes)
        return [item for pair in values.items() for item in pair] + ["--", sys.executable, "-c", code]

    def test_new_authorization_is_bound_and_cannot_be_reset(self):
        data = self.initialize()
        self.assertEqual(data["schema"], budget.SCHEMA)
        self.assertEqual(data["authorization_sha256"], budget.sha(self.authorization))
        self.assertEqual(data["source_checkpoint_sha256"], budget.sha(self.checkpoint))
        snapshot = self.ledger.read_bytes()
        self.assertNotEqual(self.command(*self.init_args()).returncode, 0)
        self.assertEqual(snapshot, self.ledger.read_bytes())

    def test_rejects_missing_authorization_and_wrong_hash(self):
        result = self.command("--ledger", self.ledger, "--initialize")
        self.assertNotEqual(result.returncode, 0)
        result = self.command(*self.init_args(**{"--authorization-sha256": "0" * 64}))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.ledger.exists())

    def test_rejects_wrong_source_revision_and_checkpoint(self):
        for changes in ({"--source-base-commit": "0" * 40}, {"--source-checkpoint-sha256": "0" * 64}):
            self.assertNotEqual(self.command(*self.init_args(**changes)).returncode, 0)
            self.assertFalse(self.ledger.exists())

    def test_rejects_caps_beyond_authorization(self):
        for changes in ({"--total-cpu-seconds": 43201}, {"--total-wall-seconds": 28801}):
            self.assertNotEqual(self.command(*self.init_args(**changes)).returncode, 0)
            self.assertFalse(self.ledger.exists())

    def test_rejects_mutated_authorization_or_checkpoint_before_launch(self):
        self.initialize()
        before = self.ledger.read_bytes()
        self.authorization.write_text(json.dumps({**self.auth, "authorization_text": "changed"}))
        self.assertNotEqual(self.command(*self.phase_args()).returncode, 0)
        self.assertEqual(before, self.ledger.read_bytes())
        self.authorization.write_text(json.dumps(self.auth))
        self.checkpoint.write_bytes(b"changed checkpoint")
        self.assertNotEqual(self.command(*self.phase_args()).returncode, 0)
        self.assertEqual(before, self.ledger.read_bytes())

    def test_refuses_old_schema_running_phase_and_duplicate_phase(self):
        data = self.initialize()
        self.ledger.write_text(json.dumps({**data, "schema": "old-pilot"}))
        self.assertNotEqual(self.command(*self.phase_args()).returncode, 0)
        self.ledger.write_text(json.dumps({**data, "phases": [{"phase": "other", "status": "RUNNING"}]}))
        self.assertNotEqual(self.command(*self.phase_args()).returncode, 0)
        self.ledger.write_text(json.dumps(data))
        self.assertEqual(self.command(*self.phase_args()).returncode, 0)
        snapshot = self.ledger.read_bytes()
        self.assertNotEqual(self.command(*self.phase_args()).returncode, 0)
        self.assertEqual(snapshot, self.ledger.read_bytes())

    def test_rejects_invalid_threads_memory_and_phase_caps(self):
        self.initialize()
        for changes in ({"--threads": 5}, {"--threads": 0}, {"--memory-gib": 13},
                        {"--cpu-seconds": 21601}, {"--cpu-seconds": 64},
                        {"--wall-seconds": 14401}, {"--wall-seconds": 60}):
            self.assertNotEqual(self.command(*self.phase_args(**changes)).returncode, 0)
        self.assertFalse((self.root / "fixture.log").exists())

    def test_records_actual_usage_threads_limits_and_log_hash(self):
        self.initialize()
        code = "import json,os,resource; print(json.dumps({'threads':[os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')], 'limits':resource.getrlimit(resource.RLIMIT_CPU)}))"
        result = self.command(*self.phase_args(code))
        self.assertEqual(result.returncode, 0, result.stderr)
        ledger = json.loads(self.ledger.read_text());row = ledger["phases"][0]
        payload = json.loads((self.root / "fixture.log").read_text())
        self.assertEqual(payload["threads"], ["2"] * 4)
        self.assertEqual(payload["limits"], [60, 120])
        self.assertGreater(row["actual_cpu_seconds"], 0)
        self.assertEqual(ledger["consumed_cpu_seconds"], row["actual_cpu_seconds"])
        self.assertEqual(row["log_sha256"], budget.sha(self.root / "fixture.log"))
        self.assertEqual(row["status"], "EXITED")
        self.assertGreater(row["peak_rss_kib"], 0)

    def test_real_busy_child_terminates_by_its_lower_cpu_limit(self):
        # The runner's production margin stays60CPU seconds. The synthetic child
        # lowers its own limit to1/2to exercise wait4 failure accounting cheaply.
        self.initialize()
        code = "import resource; resource.setrlimit(resource.RLIMIT_CPU,(1,2)); exec('while True: pass')"
        result = self.command(*self.phase_args(code, phase="busy"))
        self.assertNotEqual(result.returncode, 0)
        row = json.loads(self.ledger.read_text())["phases"][0]
        self.assertLess(row["exit_code"], 0)
        self.assertGreater(row["actual_cpu_seconds"], .8)
        self.assertLess(row["actual_cpu_seconds"], 3)
        self.assertEqual(row["cpu_cleanup_margin_seconds"], 60)
        self.assertEqual(row["status"], "PARTIAL_OR_FAILED")

    def test_real_wall_signal_stops_sleeping_child_with60s_cleanup_available(self):
        self.initialize()
        result = self.command(*self.phase_args("import time; time.sleep(90)", phase="sleep", **{"--wall-seconds": 61}))
        self.assertNotEqual(result.returncode, 0)
        row = json.loads(self.ledger.read_text())["phases"][0]
        self.assertTrue(row["wall_terminated"])
        self.assertFalse(row["hard_killed"])
        self.assertEqual(row["wall_cleanup_seconds"], 60)
        self.assertEqual(row["wall_SIGTERM_after_seconds"], 1)
        self.assertLess(row["elapsed_wall_seconds"], 4)
        self.assertLess(row["actual_cpu_seconds"], .5)

    def test_refuses_existing_log_without_modifying_it(self):
        self.initialize()
        p = self.root / "fixture.log";p.write_text("immutable")
        before = self.ledger.read_bytes()
        self.assertNotEqual(self.command(*self.phase_args()).returncode, 0)
        self.assertEqual(p.read_text(), "immutable")
        self.assertEqual(before, self.ledger.read_bytes())

    def test_failed_to_start_is_closed_with_zero_actual_cpu(self):
        self.initialize()
        args = self.phase_args()
        args[args.index("--") + 1] = str(self.root / "missing-executable")
        self.assertNotEqual(self.command(*args).returncode, 0)
        ledger = json.loads(self.ledger.read_text());row = ledger["phases"][0]
        self.assertEqual(row["status"], "FAILED_TO_START")
        self.assertEqual(row["actual_cpu_seconds"], 0)
        self.assertEqual(ledger["consumed_cpu_seconds"], 0)
        self.assertIsNone(row["exit_code"])

    def test_exit_signal_race_does_not_leave_a_false_running_phase(self):
        with patch.object(budget.os, "killpg", side_effect=ProcessLookupError):
            self.assertFalse(budget.signal_group(123, 15))

    def test_rejects_forged_usage_and_wall_deadline(self):
        data = self.initialize()
        for changed in ({**data, "consumed_cpu_seconds": 5},
                        {**data, "run_deadline_at_utc": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()}):
            self.ledger.write_text(json.dumps(changed))
            self.assertNotEqual(self.command(*self.phase_args()).returncode, 0)

    def test_phase_grants_are_clipped_without_budget_enlargement(self):
        ledger = self.initialize()
        now = datetime.now(timezone.utc)
        ledger["consumed_cpu_seconds"] = 400
        ledger["run_deadline_at_utc"] = (now + timedelta(seconds=90)).isoformat()
        args = argparse.Namespace(cpu_seconds=1000, wall_seconds=1000, threads=4, memory_gib=12)
        value = budget.phase_limits(ledger, self.auth, args, now)
        self.assertEqual(value["granted_cpu_seconds"], 90)
        self.assertEqual(value["cpu_boundary_reserve_seconds"], 10)
        self.assertEqual(value["wall_limit_seconds"], 88)
        self.assertGreaterEqual(value["cpu_cleanup_margin_seconds"], 60)

    def test_insufficient_cpu_or_overall_wall_is_blocked(self):
        ledger = self.initialize();now = datetime.now(timezone.utc)
        args = argparse.Namespace(cpu_seconds=120, wall_seconds=100, threads=4, memory_gib=12)
        for cpu, remaining_wall in [(440, 100), (0, 62)]:
            changed = {**ledger, "consumed_cpu_seconds": cpu,
                       "run_deadline_at_utc": (now + timedelta(seconds=remaining_wall)).isoformat()}
            with self.assertRaises(ValueError):
                budget.phase_limits(changed, self.auth, args, now)

    def test_parent_legacy_paths_are_refused_before_any_write(self):
        # No actual file is created in the pilot; root guard runs first.
        forbidden = budget.OLD_PILOT / "must_not_create_budget.json"
        self.assertFalse(forbidden.exists())
        result = self.command("--ledger", forbidden, "--initialize")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(forbidden.exists())


if __name__ == "__main__":
    unittest.main()
