"""No quantum, training or Git: synthetic orchestration/state safety tests."""
from contextlib import nullcontext
from datetime import timedelta
import importlib.util
import gzip
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("autonomous72h_driver_test", Path(__file__).resolve().parents[1]/"scripts/run_research4_autonomous72h_20261107.py")
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


class Autonomous72hTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prefix = self.root/"reports/research72h-20261107-v1"
        self.prefix.mkdir(parents=True)
        self.driver = self.root/"driver.py"; self.driver.write_text("# driver fixture\n")
        self.guard = self.root/"guard.py"; self.guard.write_text("# guardian fixture\n")
        self.worker = self.root/"worker.py"; self.worker.write_text("# worker fixture\n")
        self.auth = self.prefix/"authorization.json"
        deadline = (M.utc()+timedelta(hours=1)).isoformat()
        self.auth.write_text(json.dumps({"run_deadline_at_utc": deadline, "authorized_additional_cpu_seconds": 900000}))
        self.ledger = self.prefix/"cpu_budget.json"
        self.ledger.write_text(json.dumps({"run_deadline_at_utc": deadline, "remaining_quantum_cpu_seconds": 10000, "phases": []}))
        patcher = patch.multiple(M, ROOT=self.root, PREFIX=self.prefix, SELF=self.driver, GUARD=self.guard,
                                 AUTH=self.auth, AUTH_SHA=M.sha(self.auth), LEDGER=self.ledger, GIT_DEADLINE=None)
        patcher.start(); self.addCleanup(patcher.stop)
        self.config = {"schema": M.SCHEMA, "workroot": str(self.prefix/"autonomous"), "ledger": str(self.ledger),
                       "authref": self.ref(self.auth), "guardref": self.ref(self.guard), "source_refs": [self.ref(self.driver)],
                       "immutable_inputs": [], "deadline_utc": deadline, "jobs": [self.job("first")]}
        self.fake_guard = SimpleNamespace(validate_ledger=lambda x: None)

    @staticmethod
    def ref(file):
        return {"path": str(file), "sha256": M.sha(file)}

    def job(self, name, deps=None):
        return {"id": name, "kind": "analysis", "workerref": self.ref(self.worker),
                "argv": [sys.executable, str(self.worker), "--output", "{output}"],
                "cpu_seconds": 200, "wall_seconds": 90, "threads": 1, "success_statuses": ["OK"],
                "required_result_fields": {"new_model_fits": 0}, "depends_on": deps or []}

    def test_config_accepts_finite_pinned_jobs_and_does_not_create_workroot(self):
        work, ledger = M.validate_config(self.config, self.fake_guard)
        self.assertFalse(work.exists())
        self.assertEqual(ledger["remaining_quantum_cpu_seconds"], 10000)

    def test_changed_source_and_forward_dependency_are_rejected(self):
        self.config["jobs"][0]["depends_on"] = ["future"]
        with self.assertRaisesRegex(ValueError, "Dependencies"):
            M.validate_config(self.config, self.fake_guard)
        self.config["jobs"][0]["depends_on"] = []
        self.worker.write_text("# changed\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            M.validate_config(self.config, self.fake_guard)

    def test_q_cannot_auto_retry_and_all_jobs_require_guard_bounds(self):
        for key, value in (("max_attempts", 2), ("cpu_seconds", 60001), ("threads", 3), ("wall_seconds", 60)):
            job = self.job("q"); job["kind"] = "quantum"; job[key] = value
            self.config["jobs"] = [job]
            with self.assertRaises(ValueError):
                M.validate_config(self.config, self.fake_guard)

    def test_argv_binding_and_placeholders_cannot_escape_dependencies(self):
        invalid = ([sys.executable, str(self.worker), "--output"],
                   [sys.executable, str(self.worker), "--output", "{output}", "{result:unknown}"],
                   [sys.executable, "-c", str(self.worker), "--output", "{output}"])
        for argv in invalid:
            self.config["jobs"][0]["argv"] = argv
            with self.assertRaises(ValueError):
                M.validate_config(self.config, self.fake_guard)
        other = self.root/"other.py"; other.write_text("# not the pinned entry\n")
        self.config["jobs"][0]["argv"] = [sys.executable, str(other), str(self.worker), "--output", "{output}"]
        with self.assertRaisesRegex(ValueError, "directly execute"):
            M.validate_config(self.config, self.fake_guard)

    def test_dependency_result_and_hash_are_materialized_only_after_recheck(self):
        file = self.prefix/"result.json"; file.write_text('{"status":"OK"}')
        reference = self.ref(file)
        actual = M.materialize(["{output}", "{result:first}", "{sha256:first}"], self.prefix/"new", {"first": reference})
        self.assertEqual(actual, [str(self.prefix/"new"), str(file), reference["sha256"]])
        file.write_text("changed")
        with self.assertRaisesRegex(ValueError, "Sealed dependency"):
            M.materialize(["{result:first}"], self.prefix/"new", {"first": reference})

    def test_result_gates_preserve_boolean_numeric_distinction(self):
        M.qualify(self.config["jobs"][0], {"status": "OK", "new_model_fits": 0})
        for result in ({"status": "FAILED", "new_model_fits": 0}, {"status": "OK", "new_model_fits": False}):
            with self.assertRaises(ValueError):
                M.qualify(self.config["jobs"][0], result)

    def test_resume_refuses_unknown_running_or_orphan_without_pid_guess(self):
        for status in ("RUNNING", "STARTING", "BLOCKED_UNKNOWN_CLOSURE"):
            state = {"config_sha256": "locked", "jobs": {"first": {"status": status}}}
            with self.assertRaisesRegex(ValueError, "forensics"):
                M.resume_state(state, "locked", self.config, self.fake_guard)

    def test_resume_closed_failure_is_preserved_not_retried(self):
        closure = {"actual_cpu_seconds": 2.3, "scientific_success": False}
        state = {"config_sha256": "locked", "jobs": {"first": {"status": "FAILED_CLOSED", "closure": closure}}}
        with patch.object(M, "authenticate", return_value=closure) as auth:
            self.assertEqual(M.resume_state(state, "locked", self.config, self.fake_guard), state)
            auth.assert_called_once()
        with patch.object(M, "authenticate", return_value={**closure, "actual_cpu_seconds": 0}):
            with self.assertRaisesRegex(ValueError, "closure changed"):
                M.resume_state(state, "locked", self.config, self.fake_guard)

    def test_git_is_never_called_when_live_science_is_detected(self):
        work = self.prefix/"autonomous"; work.mkdir()
        with patch.object(M, "lock", return_value=nullcontext()), patch.object(M, "process_scan", side_effect=ValueError("LIVE Q")), patch.object(M, "git") as git:
            result = M.upload(self.config, self.fake_guard, work, "return-001")
            git.assert_not_called()
            self.assertEqual(result["status"], "UPLOAD_FAILED_LOCAL_SCIENCE_PRESERVED")
            self.assertTrue((work/"git-return/return-001-failure.json").exists())

    def test_return_allowlist_excludes_models_and_large_arrays_without_deleting(self):
        (self.prefix/"a.json").write_text("{}")
        (self.prefix/"native.log").write_bytes(b"raw \t \n\n")
        (self.prefix/"small.npy").write_bytes(b"small")
        (self.prefix/"large.npy").write_bytes(b"x"*(1048576+1))
        (self.prefix/"state.chk").write_bytes(b"chk")
        (self.prefix/"weights").mkdir(); (self.prefix/"weights/model.json").write_text("{}")
        selected, excluded = M.return_files()
        names = {f.name for f in selected}
        self.assertIn("native.log", names); self.assertIn("small.npy", names)
        self.assertNotIn("large.npy", names); self.assertNotIn("state.chk", names); self.assertNotIn("model.json", names)
        self.assertEqual(len(excluded), 1)
        self.assertEqual((self.prefix/"native.log").read_bytes(), b"raw \t \n\n")
        self.assertTrue((self.prefix/"large.npy").exists())

    def test_return_accepts_only_genuine_small_csv_gzip_and_hash_metadata(self):
        with gzip.open(self.prefix/"predictions.csv.gz", "wt", encoding="utf-8") as handle:
            handle.write("record_id,y_true,y_predicted\ns1,A,A\ns2,B,A\n")
        with gzip.open(self.prefix/"renamed_model.csv.gz", "wb") as handle:
            handle.write(b"\x80\x04\x00not CSV")
        with gzip.open(self.prefix/"raw.gz", "wt") as handle:
            handle.write("a,b\n1,2\n")
        (self.prefix/"protocol.sha256").write_text("a"*64+"\n")
        (self.prefix/"bad.sha256").write_text("not a SHA\n")
        selected, excluded = M.return_files()
        names = {file.name for file in selected}
        self.assertIn("predictions.csv.gz", names); self.assertIn("protocol.sha256", names)
        self.assertNotIn("renamed_model.csv.gz", names); self.assertNotIn("raw.gz", names); self.assertNotIn("bad.sha256", names)
        self.assertEqual(len(excluded), 2)

    def test_gzip_csv_must_be_rectangular_and_contain_records(self):
        for text in ("a,b\n", "a,b\n1,2,3\n", "one column\nvalue\n"):
            file = self.prefix/"candidate.csv.gz"
            with gzip.open(file, "wt") as handle:
                handle.write(text)
            self.assertFalse(M.genuine_csv_gzip(file))

    def test_same_pid_different_birth_or_boot_is_not_same_process(self):
        first = {"pid": 1, "start_ticks": 4, "uid": 3, "boot_id": "a"}
        for second in ({**first, "start_ticks": 5}, {**first, "boot_id": "b"}, {**first, "uid": 4}):
            self.assertFalse(M.same_identity(first, second))

    def test_false_closed_ledger_does_not_hide_live_owned_nonpython_payload(self):
        owned = {"pid": 101, "start_ticks": 99, "uid": 7}
        value = json.loads(self.ledger.read_text())
        value["phases"] = [{"status": "EXITED", "boot_id": "current", "child_identity": owned}]
        self.ledger.write_text(json.dumps(value))
        process = SimpleNamespace(name="101")
        class FakeDirectory:
            name = "101"
            def __truediv__(self, name):
                return SimpleNamespace(read_bytes=lambda: b"/usr/bin/custom_payload\x00")
        def paths(value):
            return SimpleNamespace(iterdir=lambda: [FakeDirectory()]) if str(value) == "/proc" else Path(value)
        observed = {**owned, "boot_id": "current", "state": "R", "comm": "custom"}
        with patch.object(M, "Path", side_effect=paths), patch.object(M, "identity", return_value=observed), \
             patch.object(M, "proc_argv", return_value=["/usr/bin/custom_payload"]):
            with self.assertRaisesRegex(ValueError, "LEDGER_BOUND_OWNED_CHILD"):
                M.process_scan(self.config)

    def test_unconfigured_full_native_worker_family_still_blocks_git(self):
        directory = SimpleNamespace(name="101")
        def paths(value):
            return SimpleNamespace(iterdir=lambda: [directory]) if str(value) == "/proc" else Path(value)
        observed = {"pid": 101, "start_ticks": 99, "uid": os.getuid(), "boot_id": "current", "state": "R", "comm": "python"}
        with patch.object(M, "Path", side_effect=paths), patch.object(M, "identity", return_value=observed), \
             patch.object(M, "proc_argv", return_value=["python", "/different/research4_full_native_checkpoint_worker_20261107.py"]):
            with self.assertRaisesRegex(ValueError, "Live scientific"):
                M.process_scan(self.config)

    def detach_receipt(self):
        expected = {"pid": 101, "start_ticks": 99, "uid": os.getuid(), "boot_id": "current", "state": "R"}
        return {"identity": expected, "source_sha256": M.sha(self.driver), "config_path": str(self.prefix/"config.json"), "config_sha256": "locked"}

    def test_exact_detach_parent_may_exit_before_ordinary_process_scan(self):
        receipt = self.detach_receipt()
        argv = [sys.executable, "driver.py", "--execute", "--detach", "--config", receipt["config_path"], "--config-sha256", "locked"]
        with patch.dict(M.os.environ, {M.DETACH_PARENT_ENV: json.dumps(receipt)}), \
             patch.object(M.os, "getppid", return_value=101), patch.object(M, "proc_argv", return_value=argv), \
             patch.object(M, "identity", side_effect=[receipt["identity"], FileNotFoundError()]), patch.object(M.time, "sleep"):
            M.wait_for_detacher(receipt["config_path"], "locked")
            self.assertNotIn(M.DETACH_PARENT_ENV, M.os.environ)

    def test_detach_binding_never_exempts_another_driver_or_live_launcher(self):
        receipt = self.detach_receipt()
        argv = [sys.executable, str(self.driver), "--execute", "--detach", "--config", receipt["config_path"], "--config-sha256", "different"]
        with patch.dict(M.os.environ, {M.DETACH_PARENT_ENV: json.dumps(receipt)}), \
             patch.object(M.os, "getppid", return_value=101), patch.object(M, "proc_argv", return_value=argv), \
             patch.object(M, "identity", return_value=receipt["identity"]):
            with self.assertRaisesRegex(ValueError, "exact bound"):
                M.wait_for_detacher(receipt["config_path"], "locked")
        argv[-1] = "locked"
        with patch.dict(M.os.environ, {M.DETACH_PARENT_ENV: json.dumps(receipt)}), \
             patch.object(M.os, "getppid", return_value=101), patch.object(M, "proc_argv", return_value=argv), \
             patch.object(M, "identity", return_value=receipt["identity"]), patch.object(M.time, "monotonic", side_effect=[0, 2]):
            with self.assertRaisesRegex(ValueError, "within1s"):
                M.wait_for_detacher(receipt["config_path"], "locked")

    def test_reused_detacher_pid_is_not_treated_as_a_live_exemption(self):
        receipt = self.detach_receipt()
        replacement = {**receipt["identity"], "start_ticks": 100}
        with patch.dict(M.os.environ, {M.DETACH_PARENT_ENV: json.dumps(receipt)}), \
             patch.object(M, "identity", return_value=replacement), patch.object(M, "proc_argv") as argv:
            M.wait_for_detacher(receipt["config_path"], "locked")
            argv.assert_not_called()

    def test_git_blob_mismatch_blocks_push_and_records_local_failure(self):
        work = self.prefix/"autonomous"; work.mkdir()
        staged = 0
        def fake_git(*args):
            nonlocal staged
            if args[:2] == ("branch", "--show-current"):
                return M.BRANCH
            if args[:2] == ("remote", "get-url"):
                return "https://github.com/jackzhu119/research4.git"
            if args[:3] == ("diff", "--cached", "--name-only"):
                staged += 1
                return "" if staged == 1 else "\n".join(str(p.relative_to(self.root)) for p in M.return_files()[0])
            if args == ("rev-parse", "HEAD"):
                return "abc"
            return ""
        with patch.object(M, "lock", return_value=nullcontext()), patch.object(M, "process_scan", return_value={}), \
             patch.object(M, "git", side_effect=fake_git) as git, \
             patch.object(M.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=b"wrong committed bytes")):
            result = M.upload(self.config, self.fake_guard, work, "return-001")
            self.assertEqual(result["status"], "UPLOAD_FAILED_LOCAL_SCIENCE_PRESERVED")
            self.assertFalse(any(call.args[0] == "push" for call in git.call_args_list))

    def test_actual_closed_failure_skips_dependents_continues_independent_jobs_and_upload_failure(self):
        self.config["jobs"] = [self.job("first"), self.job("dependent", ["first"]), self.job("independent")]
        file = self.prefix/"config.json"; file.write_text(json.dumps(self.config))
        launched = []
        def popen(command, **kwargs):
            phase = command[command.index("--phase")+1]; launched.append(phase)
            out = Path(command[command.index("--output")+1]); out.mkdir()
            (out/"result.json").write_text(json.dumps({"status": "OK", "new_model_fits": 0}))
            code = 1 if "first" in phase else 0
            return SimpleNamespace(pid=1234, returncode=code, poll=lambda: code)
        def closed(guard, config, job, entry):
            return {"actual_cpu_seconds": 1.2, "scientific_success": job["id"] != "first"}
        with patch.object(M, "process_scan", return_value={}), patch.object(M, "identity", return_value={"pid": 1234, "start_ticks": 1, "uid": 1, "boot_id": "test"}), \
             patch.object(M.subprocess, "Popen", side_effect=popen), patch.object(M, "authenticate", side_effect=closed), \
             patch.object(M, "upload", return_value={"status": "UPLOAD_FAILED_LOCAL_SCIENCE_PRESERVED"}) as upload:
            state = M.execute(self.config, M.sha(file), self.fake_guard, "test-001", False, file)
            self.assertEqual(launched, ["r72-first-001", "r72-independent-001"])
            self.assertEqual(state["jobs"]["first"]["status"], "FAILED_CLOSED")
            self.assertEqual(state["jobs"]["dependent"]["status"], "SKIPPED_DEPENDENCY")
            self.assertEqual(state["jobs"]["independent"]["status"], "SUCCESS")
            self.assertEqual(upload.call_count, 3)
            self.assertEqual(state["jobs"]["first"]["closure"]["actual_cpu_seconds"], 1.2)

    def test_unknown_guard_closure_aborts_before_other_jobs_or_git(self):
        self.config["jobs"] = [self.job("first"), self.job("independent")]
        file = self.prefix/"config.json"; file.write_text(json.dumps(self.config))
        with patch.object(M, "process_scan", return_value={}), patch.object(M, "identity", return_value={"pid": 1}), \
             patch.object(M.subprocess, "Popen", return_value=SimpleNamespace(pid=1234, returncode=1, poll=lambda: 1)), \
             patch.object(M, "authenticate", side_effect=ValueError("unknown owned child")), patch.object(M, "upload") as upload:
            with self.assertRaisesRegex(ValueError, "unknown owned child"):
                M.execute(self.config, M.sha(file), self.fake_guard, "test-001", False, file)
            upload.assert_not_called()
            state = json.loads((self.prefix/"autonomous/state.json").read_text())
            self.assertEqual(state["jobs"]["first"]["status"], "BLOCKED_UNKNOWN_CLOSURE")
            self.assertNotIn("independent", state["jobs"])


if __name__ == "__main__":
    unittest.main()
