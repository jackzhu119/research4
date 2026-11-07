"""Pure exit-race and metadata-migration simulations; no quantum payloads."""
from argparse import Namespace
from copy import deepcopy
from contextlib import nullcontext
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_pyscf_actual_cpu_budget_20261104_v5.py"


class ActualCpuV5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("v5_actual_cpu_pure_tests", SCRIPT)
        cls.r = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.r)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.expected = {"pid": 6789, "start_ticks": 100, "uid": 1000, "pgrp": 6789, "session": 6789}

    def identity(self, state="Z", **changes):
        return {**self.expected, "uids": [1000]*4, "state": state,
                "user_ticks": 100, "system_ticks": 20, "child_user_ticks": 0, "child_system_ticks": 0, **changes}

    def snapshot(self, state="Z"):
        rows = [] if state is None else [self.identity(state)]
        return {"processes": rows, "pids": [row["pid"] for row in rows], "rss_bytes": 0}

    def test_zombie_delayed_wait4_then_reaped(self):
        grace = {}
        waiter = Mock(side_effect=[(0, 0, None), (6789, 0, "actual_usage_fixture")])
        first = self.r.terminal_reap_observation(self.snapshot(), self.expected, grace, 100., reader=lambda pid: self.identity(), waiter=waiter)
        self.assertEqual(first["state"], "PENDING")
        self.assertEqual(first["terminal_cpu_ticks"], 120)
        second = self.r.terminal_reap_observation(self.snapshot(), self.expected, grace, 100.1, reader=lambda pid: self.identity(), waiter=waiter)
        self.assertEqual(second["state"], "REAPED")
        self.assertEqual(second["wait4"][2], "actual_usage_fixture")

    def test_gone_delayed_wait4_keeps_unknown_cpu_not_zero(self):
        first = self.r.terminal_reap_observation(self.snapshot(None), self.expected, {}, 0., reader=Mock(side_effect=FileNotFoundError), waiter=lambda pid: (0,0,None))
        self.assertEqual(first["state"], "PENDING")
        self.assertIsNone(first["terminal_cpu_ticks"])
        self.assertTrue(first["grace"]["observations"][0]["RSS_is_not_replaced_by_zero"])

    def test_every_terminal_state_supported(self):
        for state in ("Z", "X", "x"):
            with self.subTest(state=state):
                result = self.r.terminal_reap_observation(self.snapshot(state), self.expected, {}, 0., reader=lambda pid: self.identity(state), waiter=lambda pid: (0,0,None))
                self.assertEqual(result["state"], "PENDING")

    def test_live_leader_missing_from_scan_is_real_failure(self):
        for state in ("R", "S", "D"):
            with self.subTest(state=state), self.assertRaisesRegex(RuntimeError, "Live payload leader"):
                self.r.terminal_reap_observation(self.snapshot(None), self.expected, {}, 0., reader=lambda pid: self.identity(state), waiter=lambda pid: (0,0,None))

    def test_permission_error_is_not_exit_grace(self):
        with self.assertRaises(PermissionError):
            self.r.terminal_reap_observation(self.snapshot(None), self.expected, {}, 0., reader=Mock(side_effect=PermissionError), waiter=lambda pid: (0,0,None))

    def test_malformed_identity_is_not_exit_grace(self):
        with self.assertRaises(ValueError):
            self.r.terminal_reap_observation(self.snapshot(None), self.expected, {}, 0., reader=Mock(side_effect=ValueError("malformed")), waiter=lambda pid: (0,0,None))

    def test_changed_pid_identity_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            self.r.terminal_reap_observation(self.snapshot(), self.expected, {}, 0., reader=lambda pid: self.identity(start_ticks=101), waiter=lambda pid: (0,0,None))

    def test_extra_group_member_rejected(self):
        snapshot = self.snapshot()
        snapshot["processes"].append(self.identity(pid=6790))
        with self.assertRaisesRegex(RuntimeError, "Unexpected process group"):
            self.r.terminal_reap_observation(snapshot, self.expected, {}, 0., reader=lambda pid: self.identity(), waiter=lambda pid: (0,0,None))

    def test_deadline_point_five_is_failure(self):
        grace = {}
        self.r.terminal_reap_observation(self.snapshot(), self.expected, grace, 10., reader=lambda pid: self.identity(), waiter=lambda pid: (0,0,None))
        with self.assertRaisesRegex(RuntimeError, "GRACE_EXPIRED"):
            self.r.terminal_reap_observation(self.snapshot(), self.expected, grace, 10.5, reader=lambda pid: self.identity(), waiter=lambda pid: (0,0,None))
        self.assertEqual(grace["observations"][-1]["elapsed_grace_seconds"], .5)

    def test_grace_cannot_be_restarted_on_each_poll(self):
        grace = {}
        for now in (1., 1.1, 1.2, 1.4):
            self.r.terminal_reap_observation(self.snapshot(), self.expected, grace, now, reader=lambda pid: self.identity(), waiter=lambda pid: (0,0,None))
        self.assertEqual(grace["started_monotonic"], 1.)
        with self.assertRaises(RuntimeError):
            self.r.terminal_reap_observation(self.snapshot(), self.expected, grace, 1.6, reader=lambda pid: self.identity(), waiter=lambda pid: (0,0,None))

    def test_unrelated_wait_pid_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "unrelated PID"):
            self.r.terminal_reap_observation(self.snapshot(), self.expected, {}, 0., waiter=lambda pid: (9999,0,None))

    def test_bounded_cleanup_success(self):
        values = Mock(side_effect=[(0,0,None), (6789,0,"usage")])
        self.assertEqual(self.r.bounded_owned_reap(6789, waiter=values, clock=lambda:0., sleeper=lambda t:None)[2], "usage")

    def test_bounded_cleanup_never_loops_forever_even_bad_clock(self):
        waiter = Mock(return_value=(0,0,None))
        result = self.r.bounded_owned_reap(6789, waiter=waiter, clock=lambda:0., sleeper=lambda t:None)
        self.assertIsNone(result)
        self.assertEqual(waiter.call_count,102)

    def test_bounded_cleanup_actual_deadline(self):
        clock = Mock(side_effect=[0., 0., .1, .2, .3, .4, .5])
        sleeps=[]
        result = self.r.bounded_owned_reap(6789, waiter=lambda pid:(0,0,None), clock=clock, sleeper=sleeps.append)
        self.assertIsNone(result)
        self.assertLessEqual(sum(sleeps),.5)

    def test_cleanup_bound_cannot_expand(self):
        with self.assertRaises(ValueError):
            self.r.bounded_owned_reap(6789, maximum_seconds=1.)

    def old_pool(self):
        return {"schema":"research4-pyscf-actual-cpu-budget-v4", "runner_sha256":self.r.V4_SHA,
                "created_at_utc":"2026-11-04T07:20:50.391742+00:00", "run_deadline_at_utc":"2026-11-05T01:20:50.391742+00:00",
                "quantum_cpu_cap_seconds":129600,"run_wall_cap_seconds":64800,"consumed_cpu_seconds":12.5,
                "remaining_quantum_cpu_seconds":129587.5,
                "phases":[{"phase":"old_success","status":"EXITED","ended_at_utc":"2026-11-04T08:00:00+00:00","actual_cpu_seconds":2.5},
                          {"phase":"old_exit_race_failure","status":"PARTIAL_OR_FAILED","ended_at_utc":"2026-11-04T08:01:00+00:00","actual_cpu_seconds":10.,"monitor_failed":True,"exit_code":0}],
                "source_checkpoint_path":str(self.path/'source.chk'),"source_checkpoint_sha256":"c"*64,
                "source_base_commit":"d"*40,"authorization_path":str(self.path/'old_auth.json'),"authorization_sha256":"e"*64}

    def migration_auth(self,old):
        return {"original_created_at_utc":old["created_at_utc"],"original_run_deadline_at_utc":old["run_deadline_at_utc"],
                "original_quantum_cpu_cap_seconds":old["quantum_cpu_cap_seconds"],"original_run_wall_cap_seconds":old["run_wall_cap_seconds"],
                "original_consumed_cpu_seconds":old["consumed_cpu_seconds"]}

    def test_migration_preserves_known_failure_costs(self):
        old=self.old_pool();original=deepcopy(old)
        self.assertEqual(self.r.migration_invariants(old,self.migration_auth(old)),12.5)
        self.assertEqual(old,original)

    def test_migration_running_phase_rejected(self):
        old=self.old_pool();auth=self.migration_auth(old);old['phases'][0]['status']='RUNNING'
        with self.assertRaisesRegex(ValueError,'already be closed'):self.r.migration_invariants(old,auth)

    def test_migration_unknown_cpu_rejected(self):
        old=self.old_pool();auth=self.migration_auth(old);old['phases'][0]['actual_cpu_seconds']=None
        with self.assertRaisesRegex(ValueError,'known actual CPU'):self.r.migration_invariants(old,auth)

    def test_migration_no_deadline_or_cap_reset(self):
        old=self.old_pool()
        for key in ('original_created_at_utc','original_run_deadline_at_utc','original_quantum_cpu_cap_seconds','original_run_wall_cap_seconds','original_consumed_cpu_seconds'):
            auth=self.migration_auth(old);auth[key]='altered'
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'original caps'):self.r.migration_invariants(old,auth)

    def test_migration_cpu_sum_must_match(self):
        old=self.old_pool();auth=self.migration_auth(old);old['phases'][1]['actual_cpu_seconds']=9.
        with self.assertRaisesRegex(ValueError,'cumulative total'):self.r.migration_invariants(old,auth)

    def test_fresh_initialization_forbidden(self):
        with self.assertRaisesRegex(ValueError,'forbids fresh'):self.r.initialize(Namespace())

    def test_actual_closed_v4_read_only_invariants(self):
        source=ROOT/'reports/pyscf-sdf-continuation-20261103-v1/cpu_budget.json'
        old=json.loads(source.read_text());before=source.read_bytes()
        self.assertAlmostEqual(self.r.migration_invariants(old,self.migration_auth(old)),3605.273693)
        self.assertEqual(len(old['phases']),13)
        self.assertEqual(source.read_bytes(),before)

    def run_monitor_fixture(self, unreadable=False):
        """Run the actual monitor loop with a fake Popen; no process is launched."""
        r=self.r
        log=self.path/'execution/fixture.log';ledger_path=self.path/'ledger.json'
        prior=self.path/'prior-v3';prior.mkdir()
        child=SimpleNamespace(pid=6789,returncode=None)
        phase_limits={"wall_limit_seconds":900,"expected_cgroup_memory_max_bytes":16*1024**3,
                      "cgroup_kill_bytes":12*1024**3,"rss_stop_bytes":10*1024**3,
                      "monitor_interval_seconds":.1,"memory_kill_grace_seconds":1.,"cpu_guard_kill_grace_seconds":1.,
                      "cpu_soft_limit_seconds":1080,"cpu_hard_limit_seconds":1200,"wall_SIGTERM_after_seconds":840,
                      "actual_cpu_clock_ticks_per_second":100,"actual_cpu_closed_batch_cost_seconds":12.5,
                      "actual_cpu_batch_cap_seconds":129600,"granted_cpu_seconds":1200,
                      "actual_cpu_cleanup_reserve_seconds":120,"actual_cpu_stop_threshold_seconds":1080,
                      "actual_cpu_guard_enabled":True,"actual_cpu_guard_source":r.ACTUAL_CPU_GUARD_SOURCE}
        args=Namespace(phase='synthetic_no_payload',command=['dummy_never_executed'],log=log,ledger=ledger_path,threads=4,memory_gib=24)
        ledger={"phases":[],"consumed_cpu_seconds":12.5,"quantum_cpu_cap_seconds":129600,
                "run_deadline_at_utc":"2099-01-01T00:00:00+00:00"}
        usage=SimpleNamespace(ru_utime=1.2,ru_stime=.3,ru_maxrss=10)
        def popen(*a,**kw):
            proof={"child_start_ticks":100,"target_uid":1000,"target_pgid":6789,"target_session_id":6789}
            Path(str(log)+'.supervisor-proof.json').write_text(json.dumps(proof))
            return child
        waited=Mock(side_effect=[(0,0,None),(0,0,None),(6789,0,usage)] if unreadable else [(0,0,None),(0,0,None),(0,0,None),(6789,0,usage)])
        memory=Mock(return_value={"max_bytes":16*1024**3,"current_bytes":1024**3})
        scans=Mock(side_effect=[self.snapshot(),{"pids":[]}] if unreadable else [self.snapshot(),self.snapshot(),{"pids":[]}])
        signal_owned=Mock()
        patches=[patch.object(r,'validate_ledger',return_value={"prior_closed_ledger_path":str(prior/'ledger.json')}),
                 patch.object(r,'phase_limits',return_value=phase_limits),patch.object(r,'read_cgroup_memory',memory),
                 patch.object(r.subprocess,'Popen',side_effect=popen),patch.object(r.os,'wait4',waited),
                 patch.object(r,'process_group_snapshot',scans),patch.object(r,'read_process_identity',side_effect=PermissionError if unreadable else lambda pid:self.identity()),
                 patch.object(r.os,'pidfd_open',return_value=50),patch.object(r.os,'close'),
                 patch.object(r.signal,'pidfd_send_signal',signal_owned),patch.object(r,'signal_bound_group',return_value={"sent":False,"reason":"SYNTHETIC_ONLY"}),
                 patch.object(r.time,'sleep',return_value=None)]
        from contextlib import ExitStack
        with ExitStack() as stack:
            for context in patches:stack.enter_context(context)
            row=r._run_phase_with_prior_lock(args,ledger)
        return row,ledger,memory,signal_owned,waited

    def test_full_monitor_race_is_healthy_and_cost_not_lost(self):
        row,ledger,memory,signal_owned,waited=self.run_monitor_fixture()
        self.assertEqual(row['status'],'EXITED')
        self.assertFalse(row['monitor_failed'])
        self.assertEqual(ledger['consumed_cpu_seconds'],14.)
        self.assertEqual(row['actual_cpu_seconds'],1.5)
        self.assertEqual(row['last_CPU_watermark_retained_during_exit_reap_seconds'],1.2)
        self.assertGreaterEqual(memory.call_count,3)
        signal_owned.assert_not_called()
        self.assertTrue(all(call.args[1]==r for call in waited.call_args_list for r in [self.r.os.WNOHANG]))

    def test_full_monitor_live_unreadable_remains_failed_closed(self):
        row,ledger,memory,signal_owned,_=self.run_monitor_fixture(unreadable=True)
        self.assertEqual(row['status'],'PARTIAL_OR_FAILED')
        self.assertTrue(row['monitor_failed'])
        self.assertEqual(ledger['consumed_cpu_seconds'],14.)
        self.assertGreaterEqual(signal_owned.call_count,1)
        self.assertTrue(all(call.args[0]==50 for call in signal_owned.call_args_list))


if __name__ == '__main__':
    unittest.main()
