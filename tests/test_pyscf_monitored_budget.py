"""V3 budget metadata and tiny ordinary-process tests; never quantum work."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_pyscf_monitored_budget_20261031.py"


def load_budget():
    spec = importlib.util.spec_from_file_location("monitored_budget_v3", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stat_fixture(pid=100, pgrp=7001, rss_pages=17, start_ticks=999, comm="name with (nested) parentheses"):
    fields = ["0"] * 50
    fields[0] = "R"
    for index, value in {2: pgrp, 11: 101, 12: 3, 13: 2, 14: 4, 19: start_ticks, 21: rss_pages}.items():
        fields[index] = str(value)
    return f"{pid} ({comm}) " + " ".join(fields)


class MonitoredBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.budget = load_budget()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "synthetic.chk"
        self.checkpoint.write_bytes(b"Synthetic non-quantum checkpoint fixture only")
        (self.root / "prior-v2").mkdir()
        self.prior = self.root / "prior-v2/closed_prior_v2.json"
        self.prior.write_text(json.dumps({
            "schema": "research4-pyscf-continuation-budget-v2",
            "consumed_cpu_seconds": .25,
            "phases": [{"phase": "synthetic_closed", "status": "EXITED", "actual_cpu_seconds": .25}],
        }))
        self.authorization = self.root / "authorization.json"
        self.auth = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "authorization_text": "Synthetic ordinary-process test fixture; not a quantum authorization",
            "new_budget_is_separately_authorized": True,
            "prior_budget_preserved": True,
            "source_base_commit": self.budget.SOURCE_BASE,
            "source_checkpoint_sha256": self.budget.sha(self.checkpoint),
            "source_checkpoint_path": str(self.checkpoint),
            "prior_closed_ledger_path": str(self.prior),
            "prior_closed_ledger_sha256": self.budget.sha(self.prior),
            "prior_consumed_cpu_seconds": .25,
            "max_quantum_cpu_seconds": 172800,
            "max_run_wall_seconds": 86400,
            "max_phase_cpu_seconds": 86400,
            "max_phase_wall_seconds": 43200,
            "max_threads": 4,
            "max_memory_gib": 24,
            "guard_semantics": "BEST_EFFORT_RSS_GUARD_NOT_HARD_RSS_LIMIT",
            "rss_stop_bytes": 10 * 1024**3,
            "cgroup_kill_bytes": 12 * 1024**3,
            "monitor_interval_seconds": .1,
            "memory_kill_grace_seconds": 1.,
            "expected_cgroup_memory_max_bytes": 16 * 1024**3,
        }
        self.write_auth()
        self.ledger = self.root / "new_v3_ledger.json"
        self.addCleanup(patch.stopall)
        self.cgroup = patch.object(self.budget, "read_cgroup_memory", return_value={
            "current_bytes": 2 * 1024**3, "max_bytes": 16 * 1024**3,
            "source": "SYNTHETIC_CGROUP_METADATA_NOT_A_REAL_RESOURCE_CHANGE"})
        self.cgroup.start()

    def write_auth(self):
        self.authorization.write_text(json.dumps(self.auth))

    def init_args(self, **changes):
        values = dict(ledger=self.ledger, authorization=self.authorization,
            authorization_sha256=self.budget.sha(self.authorization),
            source_checkpoint=self.checkpoint,
            source_checkpoint_sha256=self.budget.sha(self.checkpoint),
            source_base_commit=self.budget.SOURCE_BASE,
            total_cpu_seconds=500, total_wall_seconds=300)
        values.update(changes)
        return argparse.Namespace(**values)

    def initialize(self, **changes):
        return self.budget.initialize(self.init_args(**changes))

    def phase_args(self, code="print('ordinary fixture')", phase="ordinary", **changes):
        values = dict(ledger=self.ledger, phase=phase, cpu_seconds=120,
            wall_seconds=100, threads=1, memory_gib=24,
            log=self.root / (phase + ".log"), command=[sys.executable, "-c", code])
        values.update(changes)
        return argparse.Namespace(**values)

    def test_new_closed_prior_and_checkpoint_are_bound_without_reset(self):
        prior = self.prior.read_bytes()
        ledger = self.initialize()
        self.assertEqual(ledger["schema"], self.budget.SCHEMA)
        self.assertEqual(ledger["authorization_sha256"], self.budget.sha(self.authorization))
        self.assertEqual(ledger["source_checkpoint_sha256"], self.budget.sha(self.checkpoint))
        self.assertEqual(ledger["prior_closed_ledger"]["sha256"], self.budget.sha(self.prior))
        self.assertEqual(ledger["prior_consumed_cpu_seconds"], .25)
        self.assertEqual(ledger["consumed_cpu_seconds"], 0)
        before = self.ledger.read_bytes()
        with self.assertRaises(FileExistsError):
            self.initialize()
        self.assertEqual(self.prior.read_bytes(), prior)
        self.assertEqual(self.ledger.read_bytes(), before)

    def test_prior_running_or_inconsistent_cpu_is_rejected_even_when_hash_matches(self):
        for prior in (
            {"schema": "research4-pyscf-continuation-budget-v2", "consumed_cpu_seconds": .25,
             "phases": [{"phase": "active", "status": "RUNNING", "actual_cpu_seconds": .25}]},
            {"schema": "research4-pyscf-continuation-budget-v2", "consumed_cpu_seconds": 5.,
             "phases": [{"phase": "closed", "status": "EXITED", "actual_cpu_seconds": .25}]},
        ):
            self.prior.write_text(json.dumps(prior))
            self.auth["prior_closed_ledger_sha256"] = self.budget.sha(self.prior)
            self.write_auth()
            with self.assertRaises((ValueError, RuntimeError)):
                self.initialize()
            self.assertFalse(self.ledger.exists())

    def test_prior_hash_and_consumed_cpu_authorization_are_not_forgeable(self):
        for key, value in (("prior_closed_ledger_sha256", "0" * 64),
                           ("prior_consumed_cpu_seconds", 1.25)):
            original = self.auth[key]
            self.auth[key] = value
            self.write_auth()
            with self.assertRaises(ValueError):
                self.initialize()
            self.auth[key] = original
        self.write_auth()

    def test_modified_prior_or_source_after_initialization_prevents_launch(self):
        ledger = self.initialize()
        snapshot = self.ledger.read_bytes()
        prior_snapshot = self.prior.read_bytes()
        self.prior.write_text(self.prior.read_text() + "\n")
        with self.assertRaises(ValueError):
            self.budget.run_phase(self.phase_args(), ledger)
        self.assertEqual(self.ledger.read_bytes(), snapshot)
        self.assertFalse((self.root / "ordinary.log").exists())
        self.prior.write_bytes(prior_snapshot)
        self.checkpoint.write_bytes(b"changed synthetic checkpoint")
        with self.assertRaises(ValueError):
            self.budget.run_phase(self.phase_args(), ledger)
        self.assertEqual(self.ledger.read_bytes(), snapshot)

    def test_total_caps_remain_finite_48_cpu_hours_and24_wall_hours(self):
        self.assertEqual(self.budget.MAX_CPU, 172800)
        self.assertEqual(self.budget.MAX_WALL, 86400)
        self.assertEqual(self.budget.MAX_PHASE_CPU, 86400)
        self.assertEqual(self.budget.MAX_PHASE_WALL, 43200)
        for changes in (dict(total_cpu_seconds=172801), dict(total_wall_seconds=86401)):
            with self.assertRaises(ValueError):
                self.initialize(**changes)
            self.assertFalse(self.ledger.exists())

    def test_rejects_unreviewed_resource_guard_semantics_or_larger_thresholds(self):
        for key, value in (("guard_semantics", "HARD_RSS_LIMIT"),
                           ("rss_stop_bytes", 10 * 1024**3 + 1),
                           ("cgroup_kill_bytes", 12 * 1024**3 + 1),
                           ("monitor_interval_seconds", .5),
                           ("memory_kill_grace_seconds", 2.),
                           ("expected_cgroup_memory_max_bytes", 32 * 1024**3)):
            original = self.auth[key]
            self.auth[key] = value
            self.write_auth()
            with self.assertRaises(ValueError):
                self.initialize()
            self.auth[key] = original
        self.write_auth()

    def test_phase_threads_memory_and_cpu_wall_caps_are_bounded(self):
        ledger = self.initialize()
        for changes in (dict(threads=5), dict(threads=0), dict(memory_gib=25),
                        dict(cpu_seconds=86401), dict(cpu_seconds=64),
                        dict(wall_seconds=43201), dict(wall_seconds=60)):
            with self.assertRaises(ValueError):
                self.budget.phase_limits(ledger, self.auth, self.phase_args(**changes))

    def test_bad_authorization_or_checkpoint_sha_cannot_initialize(self):
        for changes in (dict(authorization_sha256="0" * 64),
                        dict(source_checkpoint_sha256="0" * 64)):
            with self.assertRaises(ValueError):
                self.initialize(**changes)
            self.assertFalse(self.ledger.exists())

    def test_remaining_cpu_wall_are_clipped_and_cleanup_margin_is_preserved(self):
        ledger = self.initialize()
        now = datetime.now(timezone.utc)
        ledger["consumed_cpu_seconds"] = 400
        ledger["run_deadline_at_utc"] = (now + timedelta(seconds=90)).isoformat()
        limits = self.budget.phase_limits(ledger, self.auth,
            self.phase_args(cpu_seconds=1000, wall_seconds=1000, threads=4), now)
        self.assertEqual(limits["granted_cpu_seconds"], 90)
        self.assertEqual(limits["wall_limit_seconds"], 88)
        self.assertGreaterEqual(limits["cpu_cleanup_margin_seconds"], 60)
        self.assertEqual(limits["monitor_interval_seconds"], .1)
        self.assertLessEqual(limits["memory_kill_grace_seconds"], 1.)

    def test_proc_stat_parser_handles_spaces_and_nested_comm_parentheses(self):
        info = self.budget.parse_proc_stat(stat_fixture())
        self.assertEqual(info["pid"], 100)
        self.assertEqual(info["pgrp"], 7001)
        self.assertEqual(info["rss_pages"], 17)
        self.assertEqual(info["start_ticks"], 999)
        self.assertEqual(info["user_ticks"], 101)
        self.assertEqual(info["system_ticks"], 3)
        self.assertEqual(info["child_user_ticks"], 2)
        self.assertEqual(info["child_system_ticks"], 4)

    def test_group_snapshot_sums_processes_once_and_reports_unreadable_entries(self):
        proc = self.root / "synthetic_proc"
        for pid, pgrp, pages in ((100, 7001, 17), (101, 7001, 11), (102, 7002, 999)):
            path = proc / str(pid)
            path.mkdir(parents=True)
            (path / "stat").write_text(stat_fixture(pid, pgrp, pages))
        # A thread directory is not another top-level process to double-count.
        thread = proc / "100/task/100"
        thread.mkdir(parents=True)
        (thread / "stat").write_text(stat_fixture(100, 7001, 17))
        unreadable = proc / "103"
        unreadable.mkdir()
        (unreadable / "stat").write_text("broken fixture")
        info = self.budget.process_group_snapshot(7001, proc_root=proc)
        self.assertEqual(set(info["pids"]), {100, 101})
        self.assertEqual(info["rss_bytes"], (17 + 11) * os.sysconf("SC_PAGE_SIZE"))
        self.assertIn(103, info["unreadable_pids"])

    def test_real_group_snapshot_observes_ordinary_parent_and_child(self):
        child = subprocess.Popen([sys.executable, "-c",
            "import os,time; data=bytearray(2*1024*1024); p=os.fork(); "
            "time.sleep(5)"], start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 2.
            while time.monotonic() < deadline:
                info = self.budget.process_group_snapshot(child.pid)
                if len(info["pids"]) == 2:
                    break
                time.sleep(.02)
            self.assertEqual(len(info["pids"]), 2)
            self.assertGreater(info["rss_bytes"], 2 * 1024**2)
        finally:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=3)

    def test_actual_wait4_usage_threads_and_rlimit_as_are_recorded(self):
        ledger = self.initialize()
        code = "import json,os,resource; print(json.dumps({'threads':[os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')], 'as':resource.getrlimit(resource.RLIMIT_AS)}))"
        row = self.budget.run_phase(self.phase_args(code, threads=4), ledger)
        payload = json.loads((self.root / "ordinary.log").read_text())
        self.assertEqual(payload["threads"], ["4"] * 4)
        self.assertEqual(payload["as"], [24 * 1024**3] * 2)
        self.assertEqual(row["status"], "EXITED")
        self.assertGreater(row["actual_cpu_seconds"], 0)
        self.assertEqual(row["log_sha256"], self.budget.sha(self.root / "ordinary.log"))

    def memory_fixture(self, ignore_term):
        threshold = 16 * 1024**2
        self.auth["rss_stop_bytes"] = threshold
        self.write_auth()
        with patch.object(self.budget, "RSS_STOP_BYTES", threshold):
            ledger = self.initialize()
            code = ("import signal,time; " +
                ("signal.signal(signal.SIGTERM,signal.SIG_IGN); " if ignore_term else "") +
                "data=bytearray(32*1024*1024); print('ordinary memory fixture',flush=True); time.sleep(10)")
            return self.budget.run_phase(self.phase_args(code, phase="rss_fixture"), ledger)

    def test_real_low_test_threshold_stops_ordinary_memory_child(self):
        row = self.memory_fixture(ignore_term=False)
        self.assertEqual(row["status"], "PARTIAL_OR_FAILED")
        self.assertLess(row["exit_code"], 0)
        self.assertIn("RSS", row.get("resource_stop_reason", ""))
        self.assertLess(row["elapsed_wall_seconds"], 3.)
        self.assertLess(row["actual_cpu_seconds"], .5)

    def test_memory_term_ignoring_child_is_killed_within_one_second_grace(self):
        row = self.memory_fixture(ignore_term=True)
        self.assertEqual(row["status"], "PARTIAL_OR_FAILED")
        self.assertEqual(row["exit_code"], -signal.SIGKILL)
        self.assertTrue(row["hard_killed"])
        self.assertLess(row["elapsed_wall_seconds"], 3.)
        self.assertLess(row["actual_cpu_seconds"], .5)

    def test_cgroup_emergency_requests_immediate_kill(self):
        ledger = self.initialize()
        calls = []
        def cgroup():
            calls.append(True)
            return {"current_bytes": (2 if len(calls) < 2 else 13) * 1024**3,
                    "max_bytes": 16 * 1024**3, "source": "SYNTHETIC_TEST_METADATA"}
        with patch.object(self.budget, "read_cgroup_memory", side_effect=cgroup):
            row = self.budget.run_phase(self.phase_args("import time; time.sleep(10)", phase="cgroup_fixture"), ledger)
        self.assertEqual(row["status"], "PARTIAL_OR_FAILED")
        self.assertEqual(row["exit_code"], -signal.SIGKILL)
        self.assertIn("CGROUP", row.get("resource_stop_reason", ""))
        self.assertLess(row["elapsed_wall_seconds"], 2.)
        self.assertLess(row["actual_cpu_seconds"], .5)

    def test_monitor_exception_kills_child_and_closes_actual_cpu_accounting(self):
        ledger = self.initialize()
        calls = []
        def cgroup():
            calls.append(True)
            if len(calls) >= 2:
                raise OSError("Synthetic monitor read failure; not an environment fault")
            return {"current_bytes": 2 * 1024**3, "max_bytes": 16 * 1024**3,
                    "source": "SYNTHETIC_TEST_METADATA"}
        with patch.object(self.budget, "read_cgroup_memory", side_effect=cgroup):
            row = self.budget.run_phase(self.phase_args("import time; time.sleep(10)", phase="monitor_failure"), ledger)
        self.assertEqual(row["status"], "PARTIAL_OR_FAILED")
        self.assertTrue(row["monitor_failed"])
        self.assertEqual(row["resource_stop_reason"], "MONITOR_FAILED_CLOSED")
        self.assertEqual(row["exit_code"], -signal.SIGKILL)
        self.assertGreaterEqual(row["actual_cpu_seconds"], 0)
        closed = json.loads(self.ledger.read_text())
        self.assertEqual(closed["consumed_cpu_seconds"], row["actual_cpu_seconds"])
        self.assertNotEqual(closed["phases"][0]["status"], "RUNNING")
        self.assertLess(row["elapsed_wall_seconds"], 2.)

    def test_nonblocking_dummy_usage_does_not_skip_exception_cleanup_and_reap(self):
        ledger = self.initialize()
        actual_wait4 = os.wait4
        calls = []
        def controlled_wait4(pid, flags):
            calls.append(flags)
            if len(calls) == 1:
                # Linux WNOHANG may supply an rusage object even when pid is0.
                # These deliberately impossible values must never be accounted.
                return 0, 0, SimpleNamespace(ru_utime=123., ru_stime=456., ru_maxrss=1)
            return actual_wait4(pid, flags)
        with patch.object(self.budget.os, "wait4", side_effect=controlled_wait4), \
                patch.object(self.budget.time, "sleep", side_effect=KeyboardInterrupt("Synthetic supervisor interruption")):
            row = self.budget.run_phase(self.phase_args("import time; time.sleep(10)", phase="supervisor_exception"), ledger)
        self.assertIn(0, calls)
        self.assertEqual(row["status"], "PARTIAL_OR_FAILED")
        self.assertEqual(row["resource_stop_reason"], "SUPERVISOR_EXCEPTION_CLEANUP")
        self.assertEqual(row["supervisor_exception_type"], "KeyboardInterrupt")
        self.assertEqual(row["exit_code"], -signal.SIGKILL)
        self.assertGreaterEqual(row["actual_cpu_seconds"], 0)
        self.assertLess(row["actual_cpu_seconds"], 1.)
        closed = json.loads(self.ledger.read_text())
        self.assertEqual(closed["consumed_cpu_seconds"], row["actual_cpu_seconds"])
        self.assertNotEqual(closed["phases"][0]["status"], "RUNNING")
        with self.assertRaises(ChildProcessError):
            actual_wait4(row["pid"], os.WNOHANG)

    def test_live_leader_missing_from_snapshot_fails_closed_instead_of_zero_rss(self):
        ledger = self.initialize()
        missing = {"rss_bytes": 0, "pids": [], "processes": [], "unreadable_pids": [],
                   "observed_at_utc": datetime.now(timezone.utc).isoformat()}
        with patch.object(self.budget, "process_group_snapshot", return_value=missing):
            row = self.budget.run_phase(self.phase_args("import time; time.sleep(10)", phase="missing_leader"), ledger)
        self.assertEqual(row["status"], "PARTIAL_OR_FAILED")
        self.assertTrue(row["monitor_failed"])
        self.assertEqual(row["resource_stop_reason"], "MONITOR_FAILED_CLOSED")
        self.assertIn("leader missing", row["monitor_exception"])
        self.assertEqual(row["exit_code"], -signal.SIGKILL)
        self.assertGreaterEqual(row["actual_cpu_seconds"], 0)
        self.assertLess(row["elapsed_wall_seconds"], 2.)
        closed = json.loads(self.ledger.read_text())
        self.assertEqual(closed["consumed_cpu_seconds"], row["actual_cpu_seconds"])
        self.assertNotEqual(closed["phases"][0]["status"], "RUNNING")

    def test_real_child_observes_parent_death_sigkill_and_bound_launch_proof(self):
        ledger = self.initialize()
        code = ("import ctypes,json,os; value=ctypes.c_int(); "
                "libc=ctypes.CDLL(None,use_errno=True); "
                "assert libc.prctl(2,ctypes.byref(value),0,0,0)==0; "
                "proof=json.load(open(os.environ['R4_RESOURCE_SUPERVISOR_PROOF'])); "
                "print(json.dumps({'pdeathsig':value.value,'proof':proof}))")
        row = self.budget.run_phase(self.phase_args(code, phase="parent_death_proof"), ledger)
        self.assertEqual(row["status"], "EXITED")
        payload = json.loads((self.root / "parent_death_proof.log").read_text())
        self.assertEqual(payload["pdeathsig"], signal.SIGKILL)
        self.assertEqual(payload["proof"]["parent_death_signal"], "SIGKILL")
        self.assertEqual(payload["proof"]["child_pid"], row["pid"])
        self.assertEqual(payload["proof"]["supervisor_pid"], os.getpid())
        self.assertTrue(payload["proof"]["proof_is_immutable_launch_contract_not_live_monitor_status"])
        self.assertEqual(row["supervisor_proof_sha256"], self.budget.sha(row["supervisor_proof_path"]))

    def test_existing_prior_budget_lock_refuses_cross_ledger_parallel_launch(self):
        ledger = self.initialize()
        snapshot = self.ledger.read_bytes()
        with self.prior.with_suffix(".lock").open("a") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.budget.run_phase(self.phase_args(), ledger)
        self.assertEqual(self.ledger.read_bytes(), snapshot)
        self.assertFalse((self.root / "ordinary.log").exists())

    def test_failed_start_closes_with_zero_cpu(self):
        ledger = self.initialize()
        args = self.phase_args(command=[str(self.root / "missing-executable")])
        with self.assertRaises(OSError):
            self.budget.run_phase(args, ledger)
        updated = json.loads(self.ledger.read_text())
        row = updated["phases"][0]
        self.assertEqual(row["status"], "FAILED_TO_START")
        self.assertEqual(row["actual_cpu_seconds"], 0)
        self.assertEqual(updated["consumed_cpu_seconds"], 0)
        self.assertIsNone(row["exit_code"])

    def test_existing_log_or_running_ledger_cannot_launch_another_payload(self):
        ledger = self.initialize()
        args = self.phase_args()
        args.log.write_text("immutable fixture")
        before = self.ledger.read_bytes()
        with self.assertRaises(FileExistsError):
            self.budget.run_phase(args, ledger)
        self.assertEqual(args.log.read_text(), "immutable fixture")
        self.assertEqual(self.ledger.read_bytes(), before)
        ledger["phases"] = [{"phase": "already", "status": "RUNNING"}]
        with self.assertRaises(RuntimeError):
            self.budget.run_phase(self.phase_args(phase="different"), ledger)

    def test_signal_exit_race_does_not_make_a_fake_live_phase(self):
        with patch.object(self.budget.os, "killpg", side_effect=ProcessLookupError):
            self.assertFalse(self.budget.signal_group(123, signal.SIGKILL))


if __name__ == "__main__":
    unittest.main()
