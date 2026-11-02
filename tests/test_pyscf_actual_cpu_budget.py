"""V4 actual-CPU and resource guards, pure/tiny fixtures only; never quantum work."""
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
SCRIPT = ROOT / "scripts/run_pyscf_actual_cpu_budget_20261101.py"


def load_budget():
    spec = importlib.util.spec_from_file_location("actual_cpu_budget_v4", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stat_fixture(pid=100, pgrp=7001, rss_pages=17, start_ticks=999,
                 comm="name with (nested) parentheses", session=None, state="R"):
    fields = ["0"] * 50
    fields[0] = state
    for index, value in {1: 9001, 2: pgrp, 3: pgrp if session is None else session,
                          11: 101, 12: 3, 13: 2, 14: 4, 19: start_ticks, 21: rss_pages}.items():
        fields[index] = str(value)
    return f"{pid} ({comm}) " + " ".join(fields)


class ActualCPUBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.budget = load_budget()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "synthetic.chk"
        self.checkpoint.write_bytes(b"Synthetic non-quantum checkpoint fixture only")
        (self.root / "prior-v3").mkdir()
        self.prior = self.root / "prior-v3/closed_prior_v3.json"
        self.prior.write_text(json.dumps({
            "schema": "research4-pyscf-monitored-budget-v3",
            "consumed_cpu_seconds": 111023.488913,
            "phases": [{"phase": "synthetic_closed", "status": "EXITED", "actual_cpu_seconds": 111023.488913}],
        }))
        # The frozen V3 lock already exists and must be opened read-only.
        self.prior.with_suffix(".lock").write_bytes(b"Synthetic pre-existing closed V3 lock fixture")
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
            "prior_consumed_cpu_seconds": 111023.488913,
            "max_quantum_cpu_seconds": 129600,
            "max_run_wall_seconds": 64800,
            "max_phase_cpu_seconds": 122400,
            "max_phase_wall_seconds": 43200,
            "max_threads": 4,
            "max_memory_gib": 24,
            "guard_semantics": "BEST_EFFORT_RSS_GUARD_NOT_HARD_RSS_LIMIT",
            "rss_stop_bytes": 10 * 1024**3,
            "cgroup_kill_bytes": 12 * 1024**3,
            "monitor_interval_seconds": .1,
            "memory_kill_grace_seconds": 1.,
            "expected_cgroup_memory_max_bytes": 16 * 1024**3,
            "prior_budget_is_historical_only": True,
            "actual_cpu_cleanup_reserve_seconds": 120,
            "cpu_guard_kill_grace_seconds": .8,
            "actual_cpu_guard_source": "PROC_STAT_CUMULATIVE_FROM_PROCESS_BIRTH",
        }
        self.write_auth()
        self.ledger = self.root / "new_v4_ledger.json"
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
        values = dict(ledger=self.ledger, phase=phase, cpu_seconds=150,
            wall_seconds=100, threads=1, memory_gib=24,
            log=self.root / (phase + ".log"), command=[sys.executable, "-c", code])
        values.update(changes)
        return argparse.Namespace(**values)

    def identity_fixture(self, **changes):
        values = {"pid": 7001, "start_ticks": 999, "uid": 1000,
            "uids": [1000] * 4, "pgrp": 7001, "session": 7001, "state": "S",
            "user_ticks": 100, "system_ticks": 25,
            "child_user_ticks": 10, "child_system_ticks": 15, "num_threads": 4}
        values.update(changes)
        return values

    def expected_identity(self):
        return {key: self.identity_fixture()[key] for key in ("pid", "start_ticks", "uid", "pgrp", "session")}

    def supervision_fixture(self):
        """Entirely simulated startup identity; no real process lookup or limits."""
        ticks = int(os.sysconf("SC_CLK_TCK"))
        proof_path = self.root / "synthetic_v4_supervisor_proof.json"
        proof = {"schema": self.budget.PROOF_SCHEMA,
            "guard_semantics": self.budget.GUARD_SEMANTICS,
            "supervisor_pid": 9001, "supervisor_start_ticks": 222,
            "child_pid": 7001, "child_start_ticks": 111,
            "supervisor_script_path": str(SCRIPT), "supervisor_script_sha256": self.budget.sha(SCRIPT),
            "ledger_path": str(self.ledger), "phase": "synthetic_running_proof_fixture",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "address_space_soft_limit_bytes": 24 * 1024**3,
            "address_space_hard_limit_bytes": 24 * 1024**3,
            "rss_stop_bytes": 10 * 1024**3, "cgroup_kill_bytes": 12 * 1024**3,
            "expected_cgroup_memory_max_bytes": 16 * 1024**3,
            "monitor_interval_seconds": .1, "memory_kill_grace_seconds": 1.,
            "cpu_soft_limit_seconds": 90, "cpu_hard_limit_seconds": 150,
            "wall_limit_seconds": 100, "threads": 4,
            "target_uid": 1000, "target_pgid": 7001, "target_session_id": 7001,
            "parent_death_signal": "SIGKILL", "actual_cpu_guard_enabled": True,
            "actual_cpu_guard_source": "PROC_STAT_CUMULATIVE_FROM_PROCESS_BIRTH",
            "actual_cpu_clock_ticks_per_second": ticks,
            "actual_cpu_cleanup_reserve_seconds": 120,
            "actual_cpu_stop_threshold_seconds": 30,
            "actual_cpu_closed_batch_cost_seconds": 0., "actual_cpu_batch_cap_seconds": 500,
            "cpu_guard_kill_grace_seconds": .8}
        row = {**proof, "phase": proof["phase"], "status": "RUNNING", "pid": proof["child_pid"],
               "granted_cpu_seconds": 150, "cpu_soft_limit_seconds": 90,
               "cpu_hard_limit_seconds": 150, "supervisor_proof_path": str(proof_path)}
        ledger = {"schema": self.budget.SCHEMA, "phases": [row],
            "runner_sha256": self.budget.sha(SCRIPT), "quantum_cpu_cap_seconds": 500,
            "consumed_cpu_seconds": 0., "prior_budget_is_historical_only": True}
        actual = {"child_pid": 7001, "child_start_ticks": 111,
            "supervisor_pid": 9001, "supervisor_start_ticks": 222,
            "child_uid": 1000, "child_pgid": 7001, "child_session_id": 7001,
            "parent_command": [sys.executable.encode(), str(SCRIPT).encode()], "parent_cwd": str(ROOT),
            "address_space_limits": (24 * 1024**3,) * 2, "CPU_limits": (90, 150),
            "child_cgroup": "0::/synthetic-unit-test-only", "parent_cgroup": "0::/synthetic-unit-test-only",
            "cgroup_memory_max_bytes": 16 * 1024**3, "cgroup_memory_current_bytes": 2 * 1024**3,
            "clock_ticks_per_second": ticks}
        proof_path.write_text(json.dumps(proof))
        row["supervisor_proof_sha256"] = self.budget.sha(proof_path)
        self.ledger.write_text(json.dumps(ledger))
        return proof_path, proof, ledger, actual

    def test_worker_startup_proof_verifies_actual_cpu_guard_from_mock_runtime_only(self):
        path, proof, ledger, actual = self.supervision_fixture()
        with patch.object(self.budget, "resource_runtime_snapshot", return_value=actual):
            verified = self.budget.verify_resource_supervision(path, expected_threads=4)
        self.assertTrue(verified["live_parent_process_and_source_verified"])
        self.assertTrue(verified["actual_CPU_sampling_is_best_effort"])
        self.assertEqual(verified["proof_sha256"], self.budget.sha(path))
        self.assertEqual(verified["proof"]["actual_cpu_stop_threshold_seconds"], 30)

    def test_old_v3_or_disabled_actual_cpu_proof_is_rejected(self):
        path, proof, ledger, actual = self.supervision_fixture()
        for key, value in (("schema", "research4-pyscf-resource-supervisor-proof-v3"),
                           ("actual_cpu_guard_enabled", False),
                           ("actual_cpu_guard_source", "WALL_SECONDS_TIMES_THREADS")):
            path.write_text(json.dumps({**proof, key: value}))
            with patch.object(self.budget, "resource_runtime_snapshot", return_value=actual):
                with self.assertRaises(ValueError):
                    self.budget.verify_resource_supervision(path, expected_threads=4)

    def test_startup_proof_rejects_wrong_cpu_math_clock_or_live_uid_session(self):
        path, proof, ledger, actual = self.supervision_fixture()
        for key, value in (("actual_cpu_stop_threshold_seconds", 31),
                           ("actual_cpu_clock_ticks_per_second", actual["clock_ticks_per_second"] + 1),
                           ("target_uid", 1001), ("target_session_id", 7002),
                           ("actual_cpu_cleanup_reserve_seconds", 119)):
            path.write_text(json.dumps({**proof, key: value}))
            with patch.object(self.budget, "resource_runtime_snapshot", return_value=actual):
                with self.assertRaises(ValueError):
                    self.budget.verify_resource_supervision(path, expected_threads=4)

    def test_startup_proof_cannot_bind_different_running_cpu_row_or_old_pool(self):
        path, proof, ledger, actual = self.supervision_fixture()
        ledger["phases"][0]["actual_cpu_stop_threshold_seconds"] = 29
        self.ledger.write_text(json.dumps(ledger))
        with patch.object(self.budget, "resource_runtime_snapshot", return_value=actual):
            with self.assertRaises(ValueError):
                self.budget.verify_resource_supervision(path, expected_threads=4)
        ledger["phases"][0]["actual_cpu_stop_threshold_seconds"] = 30
        ledger["prior_budget_is_historical_only"] = False
        self.ledger.write_text(json.dumps(ledger))
        with patch.object(self.budget, "resource_runtime_snapshot", return_value=actual):
            with self.assertRaises(ValueError):
                self.budget.verify_resource_supervision(path, expected_threads=4)

    def test_startup_proof_rejects_changed_rlimit_parent_command_or_worker_threads(self):
        path, proof, ledger, actual = self.supervision_fixture()
        for key, value in (("CPU_limits", (90, 151)),
                           ("parent_command", [b"unrelated-parent-not-a-supervisor"]),
                           ("address_space_limits", (12 * 1024**3,) * 2)):
            with patch.object(self.budget, "resource_runtime_snapshot", return_value={**actual, key: value}):
                with self.assertRaises(ValueError):
                    self.budget.verify_resource_supervision(path, expected_threads=4)
        with patch.object(self.budget, "resource_runtime_snapshot", return_value=actual):
            with self.assertRaises(ValueError):
                self.budget.verify_resource_supervision(path, expected_threads=3)

    def test_exact_process_identity_includes_session_uid_and_live_state(self):
        expected, actual = self.expected_identity(), self.identity_fixture()
        self.assertTrue(self.budget.identity_matches(expected, actual))
        for key, value in (("pid", 7002), ("start_ticks", 1000), ("uid", 1001),
                           ("pgrp", 7002), ("session", 7002), ("state", "Z"), ("state", "X")):
            changed = {**actual, key: value}
            self.assertFalse(self.budget.identity_matches(expected, changed), key)
        self.assertFalse(self.budget.identity_matches(expected, {**actual, "uids": [1000, 0, 1000, 1000]}))
        self.assertFalse(self.budget.identity_matches({**expected, "session": 7002}, actual))

    def test_process_identity_reads_real_uid_columns_from_synthetic_proc_only(self):
        proc = self.root / "identity_proc"
        directory = proc / "7001"
        directory.mkdir(parents=True)
        (directory / "stat").write_text(stat_fixture(pid=7001))
        (directory / "status").write_text("Name:\tSynthetic parser fixture\nUid:\t1000\t1000\t1000\t1000\n")
        actual = self.budget.read_process_identity(7001, proc_root=proc)
        self.assertEqual(actual["uid"], 1000)
        self.assertEqual(actual["uids"], [1000] * 4)
        self.assertEqual(actual["session"], 7001)

    def test_identity_start_change_between_proc_reads_is_rejected(self):
        proc = self.root / "identity_race_proc"
        directory = proc / "7001"
        directory.mkdir(parents=True)
        (directory / "stat").write_text("Synthetic parser input, parse result mocked")
        (directory / "status").write_text("Uid:\t1000\t1000\t1000\t1000\n")
        with patch.object(self.budget, "parse_proc_stat", side_effect=[self.identity_fixture(), self.identity_fixture(start_ticks=1000)]):
            with self.assertRaises(ValueError):
                self.budget.read_process_identity(7001, proc_root=proc)

    def test_each_signal_rechecks_identity_and_pid_reuse_prevents_later_kill(self):
        expected = self.expected_identity()
        snapshots = iter([self.identity_fixture(), self.identity_fixture(start_ticks=1001)])
        sent = []
        reader = lambda pid: next(snapshots)
        first = self.budget.signal_bound_group(expected, signal.SIGTERM, reader=reader,
            sender=lambda pgrp, signum: sent.append((pgrp, signum)))
        second = self.budget.signal_bound_group(expected, signal.SIGKILL, reader=reader,
            sender=lambda pgrp, signum: sent.append((pgrp, signum)))
        self.assertTrue(first["sent"])
        self.assertFalse(second["sent"])
        self.assertEqual(sent, [(7001, signal.SIGTERM)])

    def test_identity_read_failure_never_sends_an_unbound_signal(self):
        sent = []
        def reader(pid):
            raise OSError("Synthetic identity observation failure")
        result = self.budget.signal_bound_group(self.expected_identity(), signal.SIGKILL,
            reader=reader, sender=lambda *values: sent.append(values))
        self.assertFalse(result["sent"])
        self.assertEqual(sent, [])

    def test_self_parent_and_supervisor_group_signals_are_refused_without_reads(self):
        expected = self.expected_identity()
        reads, sends = [], []
        for pid, ppid, pgrp in ((7001, 9002, 9003), (9001, 7001, 9003), (9001, 9002, 7001)):
            with patch.object(self.budget.os, "getpid", return_value=pid), \
                    patch.object(self.budget.os, "getppid", return_value=ppid), \
                    patch.object(self.budget.os, "getpgrp", return_value=pgrp):
                result = self.budget.signal_bound_group(expected, signal.SIGKILL,
                    reader=lambda value: reads.append(value), sender=lambda *values: sends.append(values))
            self.assertFalse(result["sent"])
        self.assertEqual(reads, [])
        self.assertEqual(sends, [])

    def test_sampled_cpu_uses_complete_first_sample_and_no_thread_multiplier(self):
        snapshot = {"processes": [self.identity_fixture()]}
        arguments = (snapshot, self.expected_identity(), 100, 7.25, 200, 150, 20)
        first = self.budget.actual_cpu_observation(*arguments)
        again = self.budget.actual_cpu_observation(*arguments)
        self.assertEqual(first["cumulative_phase_cpu_seconds_sampled"], 1.5)
        self.assertEqual(first["batch_cpu_seconds_sampled"], 8.75)
        self.assertTrue(first["no_initial_sample_subtracted"])
        self.assertEqual(first, again)
        self.assertFalse(first["should_stop"])

    def test_cpu_tck_is_not_assumed_hz_or_multiplied_by_cores(self):
        snapshot = {"processes": [self.identity_fixture(user_ticks=250, system_ticks=250,
            child_user_ticks=0, child_system_ticks=0)]}
        result = self.budget.actual_cpu_observation(snapshot, self.expected_identity(),
            250, 0, 129600, 122400, 120)
        self.assertEqual(result["cumulative_phase_cpu_seconds_sampled"], 2.)
        self.assertEqual(result["clock_ticks_per_second"], 250)

    def test_waited_child_cpu_counts_but_an_extra_live_tgid_is_refused(self):
        snapshot = {"processes": [self.identity_fixture(), self.identity_fixture(pid=7002,
            start_ticks=1000, user_ticks=50, system_ticks=10, child_user_ticks=0, child_system_ticks=0)]}
        with self.assertRaisesRegex(ValueError, "Single quantum payload"):
            self.budget.actual_cpu_observation(snapshot, self.expected_identity(),
                100, 7.25, 200, 150, 20)
        result = self.budget.actual_cpu_observation({"processes": snapshot["processes"][:1]},
            self.expected_identity(), 100, 7.25, 200, 150, 20)
        self.assertEqual(result["cumulative_phase_cpu_seconds_sampled"], 1.5)

    def test_actual_cpu_trigger_uses_lower_phase_or_remaining_batch_boundary(self):
        leader = self.identity_fixture(user_ticks=12228000, system_ticks=0,
            child_user_ticks=0, child_system_ticks=0)
        result = self.budget.actual_cpu_observation({"processes": [leader]}, self.expected_identity(),
            100, 2000, 129600, 122400, 120)
        self.assertTrue(result["should_stop"])
        self.assertEqual(result["stop_threshold_phase_cpu_seconds"], 122280)
        leader["user_ticks"] = 2948000
        result = self.budget.actual_cpu_observation({"processes": [leader]}, self.expected_identity(),
            100, 100000, 129600, 122400, 120)
        self.assertTrue(result["should_stop"])
        self.assertEqual(result["stop_threshold_phase_cpu_seconds"], 29480)
        leader["user_ticks"] -= 1
        below = self.budget.actual_cpu_observation({"processes": [leader]}, self.expected_identity(),
            100, 100000, 129600, 122400, 120)
        self.assertFalse(below["should_stop"])

    def test_cpu_read_or_membership_failure_is_not_silently_zero(self):
        expected = self.expected_identity()
        for snapshot in ({"processes": []},
                         {"processes": [self.identity_fixture(uid=1001)]},
                         {"processes": [self.identity_fixture(user_ticks=-1)]},
                         {"processes": [self.identity_fixture(), self.identity_fixture(pid=7002, session=9000)]}):
            with self.assertRaises(ValueError):
                self.budget.actual_cpu_observation(snapshot, expected, 100, 0, 129600, 122400, 120)
        with self.assertRaises(ValueError):
            self.budget.actual_cpu_observation({"processes": [self.identity_fixture()]},
                expected, 0, 0, 129600, 122400, 120)

    def test_new_closed_prior_and_checkpoint_are_bound_without_reset(self):
        prior = self.prior.read_bytes()
        lock_before = self.prior.with_suffix(".lock").read_bytes()
        ledger = self.initialize()
        self.assertEqual(ledger["schema"], self.budget.SCHEMA)
        self.assertEqual(ledger["authorization_sha256"], self.budget.sha(self.authorization))
        self.assertEqual(ledger["source_checkpoint_sha256"], self.budget.sha(self.checkpoint))
        self.assertEqual(ledger["prior_closed_ledger"]["sha256"], self.budget.sha(self.prior))
        self.assertEqual(ledger["prior_consumed_cpu_seconds"], 111023.488913)
        self.assertEqual(ledger["consumed_cpu_seconds"], 0)
        self.assertEqual(ledger["remaining_quantum_cpu_seconds"], 500)
        before = self.ledger.read_bytes()
        with self.assertRaises(FileExistsError):
            self.initialize()
        self.assertEqual(self.prior.read_bytes(), prior)
        self.assertEqual(self.prior.with_suffix(".lock").read_bytes(), lock_before)
        self.assertEqual(self.ledger.read_bytes(), before)

    def test_prior_running_or_inconsistent_cpu_is_rejected_even_when_hash_matches(self):
        for prior in (
            {"schema": "research4-pyscf-monitored-budget-v3", "consumed_cpu_seconds": 111023.488913,
             "phases": [{"phase": "active", "status": "RUNNING", "actual_cpu_seconds": 111023.488913}]},
            {"schema": "research4-pyscf-monitored-budget-v3", "consumed_cpu_seconds": 5.,
             "phases": [{"phase": "closed", "status": "EXITED", "actual_cpu_seconds": 111023.488913}]},
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

    def test_total_caps_remain_finite36_cpu_hours_and18_wall_hours(self):
        self.assertEqual(self.budget.MAX_CPU, 129600)
        self.assertEqual(self.budget.MAX_WALL, 64800)
        self.assertEqual(self.budget.MAX_PHASE_CPU, 122400)
        self.assertEqual(self.budget.MAX_PHASE_WALL, 43200)
        for changes in (dict(total_cpu_seconds=129601), dict(total_wall_seconds=64801)):
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

    def test_new_actual_cpu_authorization_cannot_reset_time_or_relax_guard(self):
        for key, value in (("prior_budget_is_historical_only", False),
                           ("actual_cpu_cleanup_reserve_seconds", 119),
                           ("actual_cpu_cleanup_reserve_seconds", 601),
                           ("cpu_guard_kill_grace_seconds", 1.1),
                           ("actual_cpu_guard_source", "WALL_SECONDS_TIMES_THREADS")):
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
                        dict(cpu_seconds=122401), dict(cpu_seconds=124),
                        dict(wall_seconds=43201), dict(wall_seconds=60)):
            with self.assertRaises(ValueError):
                self.budget.phase_limits(ledger, self.auth, self.phase_args(**changes))

    def test_bad_authorization_or_checkpoint_sha_cannot_initialize(self):
        for changes in (dict(authorization_sha256="0" * 64),
                        dict(source_checkpoint_sha256="0" * 64)):
            with self.assertRaises(ValueError):
                self.initialize(**changes)
            self.assertFalse(self.ledger.exists())

    def test_new_authorized_source_revision_is_bound_without_reusing_pilot_default(self):
        self.auth["source_base_commit"] = "f" * 40
        self.write_auth()
        ledger = self.initialize(source_base_commit="f" * 40)
        self.assertEqual(ledger["source_base_commit"], "f" * 40)
        self.assertEqual(ledger["prior_consumed_cpu_seconds"], 111023.488913)
        self.assertEqual(ledger["consumed_cpu_seconds"], 0)

    def test_remaining_cpu_wall_are_clipped_and_cleanup_margin_is_preserved(self):
        ledger = self.initialize()
        now = datetime.now(timezone.utc)
        ledger["consumed_cpu_seconds"] = 300
        ledger["run_deadline_at_utc"] = (now + timedelta(seconds=90)).isoformat()
        limits = self.budget.phase_limits(ledger, self.auth,
            self.phase_args(cpu_seconds=1000, wall_seconds=1000, threads=4), now)
        self.assertEqual(limits["granted_cpu_seconds"], 190)
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

    def test_mock_full_phase_cpu_triggers_short_cleanup_but_accounts_real_wait4(self):
        ledger = self.initialize()
        args = self.phase_args("import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            "print('ordinary CPU-guard fixture ready',flush=True); time.sleep(10)", phase="sampled_cpu_fixture")
        original_snapshot = self.budget.process_group_snapshot
        ticks_per_second = os.sysconf("SC_CLK_TCK")
        def sampled(pgid, *values, **kwargs):
            snapshot = original_snapshot(pgid, *values, **kwargs)
            # Wait for the ordinary child's TERM handler to be armed. This is a
            # mocked31CPU-second observation, not CPU work or a scientific result.
            ready = args.log.exists() and "fixture ready" in args.log.read_text()
            if ready:
                for process in snapshot["processes"]:
                    if process["pid"] == pgid:
                        process.update(user_ticks=31 * ticks_per_second, system_ticks=0,
                            child_user_ticks=0, child_system_ticks=0)
            return snapshot
        with patch.object(self.budget, "process_group_snapshot", side_effect=sampled):
            row = self.budget.run_phase(args, ledger)
        self.assertEqual(row["status"], "PARTIAL_OR_FAILED")
        self.assertTrue(row["cpu_terminated"])
        self.assertEqual(row["resource_stop_reason"], "ACTUAL_CPU_GUARD")
        self.assertEqual(row["exit_code"], -signal.SIGKILL)
        self.assertGreaterEqual(row["peak_actual_cpu_seconds_sampled"], 31)
        self.assertEqual(row["actual_cpu_stop_observation"]["stop_threshold_phase_cpu_seconds"], 30)
        self.assertEqual(len([event for event in row["resource_stop_events"] if event["reason"] == "ACTUAL_CPU_GUARD"]), 1)
        self.assertLess(row["elapsed_wall_seconds"], 2.)
        self.assertLess(row["actual_cpu_seconds"], .5)
        closed = json.loads(self.ledger.read_text())
        self.assertEqual(closed["consumed_cpu_seconds"], row["actual_cpu_seconds"])
        self.assertNotEqual(closed["consumed_cpu_seconds"], 31)

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

    def test_pidfd_and_identity_read_failure_contains_only_owned_unreaped_child(self):
        ledger = self.initialize()
        original_kill = os.kill
        sent = []
        def owned_kill(pid, signum):
            sent.append((pid, signum))
            return original_kill(pid, signum)
        with patch.object(self.budget.os, "pidfd_open", side_effect=OSError("Synthetic pidfd unavailable")), \
                patch.object(self.budget, "read_process_identity", side_effect=OSError("Synthetic identity read failure")), \
                patch.object(self.budget.os, "killpg", side_effect=AssertionError("No unbound group signal is allowed")) as group_signal, \
                patch.object(self.budget.os, "kill", side_effect=owned_kill):
            row = self.budget.run_phase(self.phase_args("import time; time.sleep(10)", phase="owned_child_containment"), ledger)
        group_signal.assert_not_called()
        self.assertEqual(sent, [(row["pid"], signal.SIGKILL)])
        self.assertEqual(row["status"], "PARTIAL_OR_FAILED")
        self.assertTrue(row["owned_unreaped_child_direct_PID_containment_requested"])
        self.assertIn("Synthetic pidfd unavailable", row["owned_child_pidfd_containment_error"])
        self.assertEqual(row["exit_code"], -signal.SIGKILL)
        self.assertGreaterEqual(row["actual_cpu_seconds"], 0)
        self.assertLess(row["actual_cpu_seconds"], .5)
        closed = json.loads(self.ledger.read_text())
        self.assertEqual(closed["consumed_cpu_seconds"], row["actual_cpu_seconds"])
        self.assertNotEqual(closed["phases"][0]["status"], "RUNNING")
        with self.assertRaises(ChildProcessError):
            os.wait4(row["pid"], os.WNOHANG)

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

    def test_existing_prior_v3_lock_refuses_cross_ledger_parallel_launch(self):
        ledger = self.initialize()
        snapshot = self.ledger.read_bytes()
        with self.prior.with_suffix(".lock").open("a") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.budget.run_phase(self.phase_args(), ledger)
        self.assertEqual(self.ledger.read_bytes(), snapshot)
        self.assertFalse((self.root / "ordinary.log").exists())

    def test_prior_v3_lock_is_opened_read_only_without_mutating_legacy_bytes(self):
        lock = self.prior.with_suffix(".lock")
        original_open = Path.open
        modes = []
        before = lock.read_bytes()
        def observed(path, *args, **kwargs):
            if path == lock:
                modes.append(args[0] if args else kwargs.get("mode", "r"))
            return original_open(path, *args, **kwargs)
        with patch.object(Path, "open", observed):
            with self.budget.prior_budget_lock(self.auth):
                pass
        self.assertEqual(modes, ["rb"])
        self.assertEqual(lock.read_bytes(), before)

    def test_missing_preexisting_prior_lock_is_refused_without_creating_it(self):
        lock = self.prior.with_suffix(".lock")
        lock.unlink()
        with self.assertRaises(FileNotFoundError):
            self.initialize()
        self.assertFalse(lock.exists())
        self.assertFalse(self.ledger.exists())

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
