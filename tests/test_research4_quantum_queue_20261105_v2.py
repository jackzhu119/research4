"""Synthetic stdlib queue tests. Never launch a Q guardian or read a Q ledger."""
from argparse import Namespace
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[1] / "scripts/research4_quantum_queue_20261105_v2.py"
SPEC = importlib.util.spec_from_file_location("prepared_quantum_queue_tests", SOURCE)
Q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.qdir = self.root / "reports/queue-fixture"
        self.qdir.mkdir(parents=True)
        self.sdir = self.root / "scripts"
        self.sdir.mkdir()
        self.worker = self.sdir / "pyscf_xc_chunk_worker_20261105_v7.py"
        self.core = self.sdir / "pyscf_xc_chunk_checkpoint_20261105_v7.py"
        self.guardfile = self.sdir / "guard.py"
        self.python = self.root / ".runtime/bin/python"
        self.python.parent.mkdir(parents=True)
        for p in (self.worker, self.core, self.guardfile, self.python):
            p.write_text("# synthetic; never executed\n")
        for name, value in (("ROOT", self.root), ("GUARD", self.guardfile), ("GUARD_SHA", Q.sha(self.guardfile)),
                ("FROZEN_A", {str(p.relative_to(self.root)): Q.sha(p) for p in (self.worker, self.core)})):
            item = patch.object(Q, name, value)
            item.start()
            self.addCleanup(item.stop)
        self.output = self.root / "reports/science/future-output"
        self.job = {"id": "block001", "phase": "actual-scientific-phase001", "worker": Q.artifact(self.worker),
            "stage": "xc-diag-chunk", "success_status": "XC_CHUNK_COMPUTED", "result_schema": "V7_FIXTURE",
            "result": str(self.output / "result.json"), "log": str(self.root / "reports/science/phase.log"),
            "threads": 2, "memory_gib": 24, "cpu_seconds": 1800, "wall_seconds": 900,
            "payload_argv": [str(self.python), str(self.worker), "--execute", "--stage", "xc-diag-chunk",
                "--output", str(self.output), "--threads", "2", "--snapshot-manifest-sha256", "f" * 64]}
        self.deadline = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
        self.config = {"schema": Q.SCHEMA, "jobs": [self.job], "monitor_interval_seconds": 1800,
            "queue_directory": str(self.qdir), "state": str(self.qdir / "state.json"), "monitor": str(self.qdir / "monitor.jsonl"),
            "ledger": str(self.root / "reports/science/ledger.json"), "original_quantum_cpu_cap_seconds": 216000,
            "original_quantum_deadline_at_utc": self.deadline, "python_executable": str(self.python),
            "pythonpath": [".runtime/quantum-pilot-site", ".runtime/quantum-upgrade-site"],
            "source_refs": [Q.artifact(p) for p in (self.guardfile, self.worker, self.core)]}
        self.config_path = self.qdir / "config.json"
        self.write(self.config_path, self.config)
        self.grant = {"schema": Q.GRANT_SCHEMA, "config_sha256": Q.sha(self.config_path), "launcher_sha256": Q.sha(SOURCE),
            "no_new_quantum_budget": True, "no_deadline_extension": True, "root_execution_approved": True,
            "root_serial_scheduling_confirmed": True, "max_invocations": 2, "launcher_cpu_seconds_per_invocation": 60,
            "launcher_rss_stop_bytes": 256 * 1024**2, "wall_seconds_per_invocation": 1800, "expires_at_utc": self.deadline}
        self.grant_path = self.qdir / "grant.json"
        self.write(self.grant_path, self.grant)
        self.ledger = {"phases": [], "consumed_cpu_seconds": 100., "remaining_quantum_cpu_seconds": 215900.,
            "quantum_cpu_cap_seconds": 216000, "run_deadline_at_utc": self.deadline}
        self.write(Path(self.config["ledger"]), self.ledger)
        self.guard = SimpleNamespace(validate_ledger=Mock(), healthy_closed_row=lambda r: r["status"] == "EXITED" and r["exit_code"] == 0,
            verify_closed_phase=Mock())
        self.args = Namespace(config=str(self.config_path), config_sha256=Q.sha(self.config_path), grant=str(self.grant_path),
            grant_sha256=Q.sha(self.grant_path), attempt="root001", invocation_deadline=Q.utc().timestamp() + 1800,
            invocation_monotonic_deadline=Q.time.monotonic() + 1800)

    def write(self, p, value):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value) + "\n")

    def materialize_result(self):
        self.output.mkdir(parents=True)
        proof = self.root / "reports/science/proof.json"
        self.write(proof, {"phase": self.job["phase"]})
        Path(self.job["log"]).write_text("actual fixture log\n")
        self.write(self.output / "input.json", {"arguments": {"stage": self.job["stage"]}})
        payload = self.output / "stage-cache/payload.npy"
        payload.parent.mkdir()
        payload.write_bytes(b"synthetic payload; not decoded as NumPy")
        self.result = {"stage": self.job["stage"], "status": self.job["success_status"], "schema": "V7_FIXTURE",
            "snapshot_manifest_sha256": "f" * 64, "source_worker": self.job["worker"],
            "resource_supervision": {"proof": {"phase": self.job["phase"]}, "proof_path": str(proof)},
            "arrays": [{"file": payload.name, "file_sha256": Q.sha(payload)}]}
        self.write(Path(self.job["result"]), self.result)
        self.ledger["phases"] = [{"phase": self.job["phase"], "status": "EXITED", "exit_code": 0,
            "command": self.job["payload_argv"], "log": self.job["log"]}]
        self.write(Path(self.config["ledger"]), self.ledger)
        return Q.verify_stage(self.job, self.guard, self.ledger)

    def run_queue(self, child=None):
        with patch.object(Q, "load_guard", return_value=self.guard), patch.object(Q, "process_observation", return_value={"records": [], "conflicts": [], "read_only_git": []}), \
                patch.object(Q, "own_rss", return_value=1024), patch.object(Q.subprocess, "Popen", return_value=child) as popen, \
                redirect_stdout(io.StringIO()):
            Q.run(self.args, self.config, self.grant)
        return popen

    def test_prepared_configuration_valid_but_root_false_cannot_execute(self):
        Q.validate_config(self.config, self.grant, self.args.config_sha256, True)
        self.grant["root_execution_approved"] = False
        with self.assertRaises(ValueError): Q.validate_config(self.config, self.grant, self.args.config_sha256, True)

    def test_duplicate_outputs_and_nonmaterialized_argv_rejected(self):
        other = deepcopy(self.job); other.update(id="other", phase="other")
        self.config["jobs"].append(other)
        with self.assertRaises(ValueError): Q.validate_config(self.config, self.grant, self.args.config_sha256)
        self.config["jobs"] = [self.job]; self.job["payload_argv"][0] = "/bin/sh"
        with self.assertRaises(ValueError): Q.validate_config(self.config, self.grant, self.args.config_sha256)

    def test_scientific_or_shared_metadata_write_paths_rejected(self):
        self.config["monitor"] = self.config["ledger"]
        with self.assertRaises(ValueError): Q.validate_config(self.config, self.grant, self.args.config_sha256)
        self.config["monitor"] = self.config["state"]
        with self.assertRaises(ValueError): Q.validate_config(self.config, self.grant, self.args.config_sha256)

    def test_changed_source_and_extended_deadline_rejected(self):
        self.core.write_text("changed")
        with self.assertRaises(ValueError): Q.validate_config(self.config, self.grant, self.args.config_sha256)
        self.core.write_text("# synthetic; never executed\n")
        self.grant["expires_at_utc"] = (Q.utc() + timedelta(days=2)).isoformat()
        with self.assertRaises(ValueError): Q.validate_config(self.config, self.grant, self.args.config_sha256)

    def test_missing_phase_is_not_a_done_result(self):
        self.assertIsNone(Q.verify_stage(self.job, self.guard, self.ledger))

    def test_failed_or_running_stage_refuses_even_with_existing_file(self):
        self.materialize_result()
        for status in ("RUNNING", "PARTIAL_OR_FAILED"):
            self.ledger["phases"][0]["status"] = status
            with self.subTest(status=status), self.assertRaises(ValueError): Q.verify_stage(self.job, self.guard, self.ledger)

    def test_exact_argv_proof_stage_and_payload_hashes_required(self):
        self.materialize_result()
        self.ledger["phases"][0]["command"] = ["different"]
        with self.assertRaises(ValueError): Q.verify_stage(self.job, self.guard, self.ledger)
        self.ledger["phases"][0]["command"] = self.job["payload_argv"]
        (self.output / "stage-cache/payload.npy").write_bytes(b"changed")
        with self.assertRaises(ValueError): Q.verify_stage(self.job, self.guard, self.ledger)

    def test_no_silent_reseal_of_result_input_proof_log_or_payload(self):
        sealed = self.materialize_result()
        for ref in sealed:
            p = Path(ref["path"]); old = p.read_bytes(); p.write_bytes(old + b" ")
            with self.subTest(path=p.name), self.assertRaises(ValueError): Q.verify_existing(self.job, self.guard, self.ledger, sealed)
            p.write_bytes(old)
        self.assertEqual(Q.verify_existing(self.job, self.guard, self.ledger, sealed), sealed)
        with self.assertRaises(ValueError): Q.verify_existing(self.job, self.guard, self.ledger, None)

    def test_healthy_root_sealed_phase_is_skipped_without_spawning(self):
        self.job["existing_seal"] = self.materialize_result()
        popen = self.run_queue(); popen.assert_not_called()
        state = Q.read(self.config["state"])
        self.assertEqual(state["jobs"][self.job["id"]]["seal"], self.job["existing_seal"])
        self.assertEqual(state["invocations"][-1]["status"], "MATERIALIZED_QUEUE_CLOSED_NOT_ALL_SCIENCE_COMPLETE")
        self.assertIn("--execute", state["resume_argv"])
        self.assertTrue(state["resume_requires_environment_ready_and_no_unresolved_invocation_or_Q_phase"])

    def test_existing_healthy_without_root_or_prior_seal_blocks(self):
        self.materialize_result()
        with self.assertRaises(ValueError): self.run_queue()

    def test_unclosed_invocation_never_becomes_done_from_missing_pid(self):
        state = {"config_sha256": self.args.config_sha256, "grant_sha256": self.args.grant_sha256,
            "jobs": {}, "invocations": [{"id": "old", "status": "RUNNING", "pid": 999999999}]}
        self.write(Path(self.config["state"]), state)
        before = Path(self.config["state"]).read_bytes()
        with self.assertRaises(ValueError): self.run_queue()
        self.assertEqual(before, Path(self.config["state"]).read_bytes())

    def test_zero_guardian_exit_is_not_scientific_success(self):
        child = Mock(pid=987654); child.poll.return_value = 0; child.wait.return_value = 0
        with self.assertRaises(ValueError): self.run_queue(child)
        state = Q.read(self.config["state"])
        self.assertNotEqual(state["jobs"][self.job["id"]]["status"], "HEALTHY_CLOSED")
        child.wait.assert_called_once()

    def test_nonzero_guardian_stops_without_automatic_retry(self):
        child = Mock(pid=987654); child.poll.return_value = 7; child.wait.return_value = 7
        with self.assertRaises(ValueError): self.run_queue(child)
        state = Q.read(self.config["state"])
        self.assertEqual(state["jobs"][self.job["id"]]["actual_guardian_returncode"], 7)
        self.assertEqual(len(state["invocations"]), 1)

    def test_source_changed_after_initial_validation_is_blocked_before_launch(self):
        Q.validate_config(self.config, self.grant, self.args.config_sha256, True)
        self.worker.write_text("later source change")
        with self.assertRaises(ValueError): self.run_queue()
        state = Q.read(self.config["state"])
        self.assertEqual(state["invocations"][-1]["status"], "BLOCKED_REQUIRES_ROOT_REVIEW")

    def test_monotonic_wall_budget_blocks_even_if_UTC_clock_is_early(self):
        self.args.invocation_deadline = Q.utc().timestamp() + 7200
        self.args.invocation_monotonic_deadline = Q.time.monotonic() - 1
        with self.assertRaises(ValueError): self.run_queue()
        self.assertFalse(Path(self.job["log"] + ".queue-console.log").exists())

    def test_global_mode_and_raw_Hessian_bytes_are_sealed_without_decoding(self):
        self.materialize_result()
        self.job["stage"] = "global-assemble"
        self.job["payload_argv"][self.job["payload_argv"].index("--stage") + 1] = self.job["stage"]
        self.result["stage"] = self.job["stage"]
        files = ("normal_modes.npy", "reduced_mass_amu.npy", "frequencies.csv", "normal_mode_geometry_angstrom.npy",
                 "normal_mode_geometry_bohr.npy", "normal_mode_geometry.xyz", "hessian_hartree_per_bohr2.npy")
        for name in files: (self.output / name).write_bytes(b"not decoded; test bytes")
        self.result["raw_hessian_sha256"] = Q.sha(self.output / "hessian_hartree_per_bohr2.npy")
        self.write(Path(self.job["result"]), self.result)
        sealed = Q.verify_stage(self.job, self.guard, self.ledger)
        self.assertTrue(set(files) <= {Path(ref["path"]).name for ref in sealed})
        (self.output / "normal_modes.npy").write_bytes(b"changed")
        with self.assertRaises(ValueError): Q.verify_existing(self.job, self.guard, self.ledger, sealed)

    def test_own_timers_do_not_lower_or_inherit_guardian_resource_limits(self):
        # This is a tiny stdlib subprocess, never a scientific guard/worker.
        parent_limits = resource.getrlimit(resource.RLIMIT_CPU)
        tiny = dict(self.grant, launcher_cpu_seconds_per_invocation=10)
        with Q.self_limits(tiny):
            raw = subprocess.check_output([sys.executable, "-S", "-c",
                "import json,signal,resource;print(json.dumps([signal.getitimer(signal.ITIMER_PROF),signal.getitimer(signal.ITIMER_REAL),resource.getrlimit(resource.RLIMIT_CPU)]))"])
        values = json.loads(raw)
        self.assertEqual(values[0], [0., 0.]); self.assertEqual(values[1], [0., 0.])
        self.assertEqual(tuple(values[2]), parent_limits)
        self.assertEqual(resource.getrlimit(resource.RLIMIT_CPU), parent_limits)
        self.assertEqual(signal.getitimer(signal.ITIMER_PROF), (0., 0.))

    def identity(self, pid=987654, start=123):
        return {"pid": pid, "start_ticks": start, "uid": 1000, "pgrp": pid,
            "session": pid, "state": "S", "comm": "python", "boot_id": "test-boot"}

    def observation(self, category=None):
        records = [] if category is None else [{**self.identity(987), "category": category}]
        return {"records": records, "conflicts": records if category != "GIT_READ_ONLY" else [],
            "read_only_git": records if category == "GIT_READ_ONLY" else [], "visibility_errors": []}

    def run_active_fixture(self, category, add_second=False):
        if add_second:
            second = deepcopy(self.job)
            second.update(id="second", phase="second-phase", result=str(self.root / "reports/science/second/result.json"),
                log=str(self.root / "reports/science/second.log"))
            second["payload_argv"][second["payload_argv"].index("--output")+1] = str(self.root / "reports/science/second")
            self.config["jobs"].append(second)
            self.write(self.config_path, self.config)
            self.args.config_sha256 = Q.sha(self.config_path)
            self.grant["config_sha256"] = self.args.config_sha256
            self.write(self.grant_path, self.grant)
            self.args.grant_sha256 = Q.sha(self.grant_path)
        child = Mock(pid=987654)
        child.poll.return_value = None
        child.wait.return_value = 0
        def finish(_seconds):
            self.materialize_result()
            child.poll.return_value = 0
        with patch.object(Q, "load_guard", return_value=self.guard), \
             patch.object(Q, "process_observation", side_effect=[self.observation(), self.observation(category)]), \
             patch.object(Q, "proc_identity", return_value=self.identity()), \
             patch.object(Q, "own_rss", return_value=1024), \
             patch.object(Q.time, "sleep", side_effect=finish), \
             patch.object(Q.subprocess, "Popen", return_value=child) as popen, redirect_stdout(io.StringIO()):
            Q.run(self.args, self.config, self.grant)
        child.terminate.assert_not_called()
        self.assertEqual(popen.call_count, 1)
        return Q.read(self.config["state"])

    def test_actual_external_readonly_git_argv_does_not_cancel_active_guardian(self):
        state = self.run_active_fixture("GIT_READ_ONLY")
        self.assertEqual(state["invocations"][-1]["status"], "MATERIALIZED_QUEUE_CLOSED_NOT_ALL_SCIENCE_COMPLETE")
        self.assertEqual(state["jobs"][self.job["id"]]["status"], "HEALTHY_CLOSED")

    def test_external_write_git_finishes_current_then_defers_second_job(self):
        state = self.run_active_fixture("GIT_WRITE_OR_UNKNOWN", True)
        self.assertEqual(state["invocations"][-1]["status"], "DEFERRED_AFTER_HEALTHY_STAGE_EXTERNAL_CONFLICT")
        self.assertNotIn("second", state["jobs"])
        monitor = Path(self.config["monitor"]).read_text()
        self.assertIn("EXTERNAL_CONFLICT_ACTIVE_STAGE_CONTINUES_NEXT_LAUNCH_DEFERRED", monitor)

    def test_external_ML_finishes_current_then_defers_second_job(self):
        state = self.run_active_fixture("ML", True)
        self.assertEqual(state["invocations"][-1]["status"], "DEFERRED_AFTER_HEALTHY_STAGE_EXTERNAL_CONFLICT")
        self.assertNotIn("second", state["jobs"])

    def test_prelaunch_write_conflict_is_bounded_defer_with_no_child(self):
        with patch.object(Q, "load_guard", return_value=self.guard), \
             patch.object(Q, "process_observation", return_value=self.observation("GIT_WRITE_OR_UNKNOWN")), \
             patch.object(Q, "own_rss", return_value=1024), patch.object(Q.subprocess, "Popen") as popen, \
             redirect_stdout(io.StringIO()):
            Q.run(self.args, self.config, self.grant)
        popen.assert_not_called()
        self.assertEqual(Q.read(self.config["state"])["invocations"][-1]["status"], "DEFERRED_BEFORE_LAUNCH_EXTERNAL_CONFLICT")

    def test_nonzero_guardian_without_launcher_call_records_unknown_external(self):
        child = Mock(pid=987654); child.poll.return_value=7; child.wait.return_value=7
        with self.assertRaises(ValueError): self.run_queue(child)
        last = Q.read(self.config["state"])["invocations"][-1]
        self.assertEqual(last["guardian_exit_observation"]["SIGTERM_sender"], "UNKNOWN_EXTERNAL_TO_LAUNCHER")
        child.terminate.assert_not_called()
        self.assertIn("GUARDIAN_EXITED_WITHOUT_LAUNCHER_STOP_REQUEST", Path(self.config["monitor"]).read_text())

    def test_stop_sample_records_exact_boundary_reasons(self):
        with patch.object(Q.time,"time",return_value=100),patch.object(Q.time,"monotonic",return_value=200), \
             patch.object(Q.time,"process_time",return_value=56),patch.object(Q,"own_rss",return_value=1234):
            sample=Q.resource_stop_sample([{"signal":15,"observed_at_monotonic_seconds":199}],150,250,60,1200)
        self.assertEqual(set(sample["reasons"]),{"EXPLICIT_LAUNCHER_SIGNAL","LAUNCHER_UTC_CLEANUP_BOUNDARY", 
            "LAUNCHER_MONOTONIC_CLEANUP_BOUNDARY","LAUNCHER_CPU_CLEANUP_BOUNDARY","LAUNCHER_RSS_STOP_BOUNDARY"})
        self.assertEqual(sample["launcher_RSS_bytes"],1234)
        self.assertEqual(sample["signals_received"][0]["signal"],15)

    def test_stop_decision_persisted_before_owned_signal_and_call_result_after(self):
        child=Mock(pid=987654);child.poll.return_value=None;events=[]
        child.terminate.side_effect=lambda:events.append("signal")
        with patch.object(Q,"proc_identity",return_value=self.identity()):
            self.assertTrue(Q.signal_owned_guardian(child,self.identity(),{"reasons":["CPU"]},lambda r:events.append(r["action"])))
        self.assertEqual(events,["REQUEST_BEFORE_SIGNAL_DELIVERY","signal","SIGTERM_CALL_RETURNED"])

    def test_PID_reuse_or_boot_UID_change_never_signals_unrelated_process(self):
        for field,value in (("start_ticks",999),("boot_id","different"),("uid",0),("pgrp",123)):
            child=Mock(pid=987654);child.poll.return_value=None;observed=dict(self.identity(),**{field:value});events=[]
            with patch.object(Q,"proc_identity",return_value=observed):
                self.assertFalse(Q.signal_owned_guardian(child,self.identity(),{},events.append))
            child.terminate.assert_not_called()
            self.assertEqual(events[-1]["action"],"NO_SIGNAL_IDENTITY_CHANGED_OR_TERMINAL")

    def test_already_reaped_or_terminal_guardian_never_receives_signal(self):
        child=Mock(pid=987654);child.poll.return_value=0
        self.assertFalse(Q.signal_owned_guardian(child,self.identity(),{},lambda _r:None));child.terminate.assert_not_called()
        child.poll.return_value=None
        with patch.object(Q,"proc_identity",return_value=dict(self.identity(),state="Z")):
            self.assertFalse(Q.signal_owned_guardian(child,self.identity(),{},lambda _r:None))
        child.terminate.assert_not_called()

    def test_classifier_recognizes_readonly_git_global_options_without_logging_secrets(self):
        for command in ("status","diff","show","rev-parse","ls-files","for-each-ref"):
            argv=["/usr/bin/git","-C","/external/repo","-c","http.extraHeader=Authorization:SECRET",command]
            self.assertEqual(Q.classify_command(argv),("GIT_READ_ONLY",command))
        self.assertEqual(Q.classify_command(["git","diff","--output=secret.txt"])[0],"GIT_WRITE_OR_UNKNOWN")
        self.assertEqual(Q.classify_command(["git","push","https://secret@host/repo"]),("GIT_WRITE_OR_UNKNOWN","push"))
        self.assertEqual(Q.classify_command(["git","unknown-alias"]),("GIT_WRITE_OR_UNKNOWN","UNKNOWN"))

    def test_ML_classifier_only_actual_interpreter_script_position(self):
        self.assertEqual(Q.classify_command(["python3.11","-u","/work/train_models.py"])[0],"ML")
        self.assertEqual(Q.classify_command(["python3","-m","torch.distributed.run"])[0],"ML")
        self.assertEqual(Q.classify_command(["bash","-lc","rg train_models.py"])[0],"OTHER")
        self.assertEqual(Q.classify_command(["python","inspect.py","train_models.py"])[0],"OTHER")
        self.assertEqual(Q.classify_command(["python","-c","print('train_models.py')"])[0],"OTHER")

    def fake_proc(self, tokens, exe="git", pid=987):
        root=self.root/"fake-proc"; (root/"sys/kernel/random").mkdir(parents=True,exist_ok=True)
        (root/"sys/kernel/random/boot_id").write_text("boot-fixture")
        d=root/str(pid);d.mkdir()
        fields=["0"]*50;fields[0]="S";fields[2]=str(pid);fields[3]=str(pid);fields[19]="12345"
        (d/"stat").write_text(str(pid)+" ("+exe+") "+" ".join(fields))
        (d/"status").write_text("Uid:\t1000 1000 1000 1000\n")
        (d/"cmdline").write_bytes(b"\0".join(t.encode() for t in tokens)+b"\0")
        (d/"cwd").symlink_to("/external/another-repository")
        (d/"exe").symlink_to("/usr/bin/"+exe)
        return root

    def test_recorded_external_git_proc_fixture_includes_identity_cwd_but_no_argv_secret(self):
        root=self.fake_proc(["/usr/bin/git","-c","http.extraHeader=SECRET_TOKEN","status","--porcelain"])
        result=Q.process_observation(root)
        self.assertFalse(result["conflicts"]);self.assertEqual(len(result["read_only_git"]),1)
        rec=result["read_only_git"][0]
        self.assertEqual((rec["pid"],rec["start_ticks"],rec["uid"],rec["boot_id"]),(987,12345,1000,"boot-fixture"))
        self.assertEqual(rec["cwd"],"/external/another-repository")
        self.assertNotIn("SECRET_TOKEN",json.dumps(result));self.assertFalse(rec["full_argv_logged"])

    def test_actual_executable_mismatch_is_unknown_and_prelaunch_conflict(self):
        root=self.fake_proc(["/usr/bin/git","status"],exe="bash")
        self.assertEqual(Q.process_observation(root)["conflicts"][0]["category"],"EXECUTABLE_ARGV_IDENTITY_UNKNOWN")

    def test_unreadable_candidate_is_explicit_visibility_conflict(self):
        root=self.fake_proc(["git","status"])
        with patch.object(Q,"proc_identity",side_effect=PermissionError("forbidden")):
            result=Q.process_observation(root)
        self.assertEqual(result["conflicts"][0]["category"],"OPAQUE_PROCESS")
        self.assertEqual(result["visibility_errors"][0]["error_type"],"PermissionError")

    def test_double_read_starttick_change_is_a_conflict_not_reused_identity(self):
        root=self.fake_proc(["git","status"]);first=self.identity(987);second=dict(first,start_ticks=124)
        with patch.object(Q,"proc_identity",side_effect=[first,second]):
            result=Q.process_observation(root)
        self.assertEqual(result["conflicts"][0]["category"],"UNSTABLE_PROCESS_IDENTITY")

    def test_argv0_disguised_actual_git_is_not_missed(self):
        root=self.fake_proc(["reader","status"],exe="git")
        result=Q.process_observation(root)
        self.assertEqual(result["conflicts"][0]["category"],"EXECUTABLE_ARGV_IDENTITY_UNKNOWN")

    def test_candidate_cwd_unavailable_is_explicit_conflict(self):
        root=self.fake_proc(["git","status"])
        (root/"987/cwd").unlink()
        result=Q.process_observation(root)
        self.assertEqual(result["conflicts"][0]["category"],"PROCESS_CWD_UNREADABLE")
        self.assertEqual(result["visibility_errors"][0]["cwd_visibility"],"FileNotFoundError")

    def test_only_exact_root_grant_daemon_exception_can_acknowledge_opaque_identity(self):
        root=self.fake_proc(["dockerd"],exe="dockerd")
        identity=dict(self.identity(987),uid=0,comm="dockerd")
        with patch.object(Q,"proc_identity",return_value=identity),patch.object(Q.os,"readlink",side_effect=PermissionError()):
            unknown=Q.process_observation(root)
            self.assertTrue(unknown["conflicts"])
            approved=Q.process_observation(root,opaque_exceptions=[identity])
            self.assertFalse(approved["conflicts"])
            self.assertEqual(approved["visibility_errors"][0]["category"],"ROOT_ACKNOWLEDGED_OPAQUE_DAEMON")
            refused=Q.process_observation(root,opaque_exceptions=[dict(identity,start_ticks=999)])
            self.assertTrue(refused["conflicts"])

    def test_grant_cannot_hide_arbitrary_unreadable_training_process(self):
        self.grant["opaque_process_identity_exceptions"]=[dict(self.identity(),uid=1000,comm="python")]
        with self.assertRaises(ValueError):Q.validate_config(self.config,self.grant,self.args.config_sha256)

    def test_scan_root_or_boot_identity_failure_is_a_defer_conflict_not_exception(self):
        root=self.fake_proc(["git","status"])
        (root/"sys/kernel/random/boot_id").unlink()
        self.assertEqual(Q.process_observation(root)["conflicts"][0]["category"],"PROC_ROOT_OR_BOOT_ID_UNREADABLE")
        (root/"sys/kernel/random/boot_id").write_text("boot-fixture")
        with patch.object(Q.Path,"iterdir",side_effect=PermissionError()):
            self.assertEqual(Q.process_observation(root)["conflicts"][0]["category"],"PROC_ROOT_OR_BOOT_ID_UNREADABLE")

    def test_opaque_daemon_exception_requires_stable_double_read(self):
        root=self.fake_proc(["dockerd"],exe="dockerd");identity=dict(self.identity(987),uid=0,comm="dockerd")
        for changed in (dict(identity,start_ticks=999),dict(identity,uid=1000),dict(identity,comm="python")):
            with patch.object(Q,"proc_identity",side_effect=[identity,changed]),patch.object(Q.os,"readlink",side_effect=PermissionError()):
                result=Q.process_observation(root,opaque_exceptions=[identity])
            self.assertTrue(result["conflicts"]);self.assertFalse(result["visibility_errors"][0]["identity_double_read_stable"])

    def test_active_visibility_failure_defers_without_cancelling_guardian(self):
        state=self.run_active_fixture("PROC_ROOT_OR_BOOT_ID_UNREADABLE",True)
        self.assertEqual(state["invocations"][-1]["status"],"DEFERRED_AFTER_HEALTHY_STAGE_EXTERNAL_CONFLICT")

    def test_Python_attached_source_module_and_hash_mode_respect_real_script_position(self):
        self.assertEqual(Q.classify_command(["python","-cprint(1)","train_models.py"])[0],"OTHER")
        self.assertEqual(Q.classify_command(["python","-mtrain_models","parameter"])[0],"ML")
        self.assertEqual(Q.classify_command(["python","--check-hash-based-pycs","always","train_models.py"])[0],"ML")

    def test_missing_executable_of_still_live_PID_is_opaque_but_true_PID_gone_is_skipped(self):
        root=self.fake_proc(["git","status"])
        (root/"987/exe").unlink()
        result=Q.process_observation(root)
        self.assertEqual(result["conflicts"][0]["category"],"OPAQUE_PROCESS")
        self.assertEqual(result["conflicts"][0]["cause"],"LIVE_PROCESS_COMPONENT_UNREADABLE")
        (root/"987/stat").unlink()
        self.assertFalse(Q.process_observation(root)["conflicts"])


if __name__ == "__main__":
    unittest.main()
