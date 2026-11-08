"""Finite finisher metadata/Git mocks only; no analysis, quantum or Git mutation."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/finalize_pyscf_monitored_pipeline_20261031.py"
BRANCH = "research/pyscf-spinosin-mechanism-20261031"
SOURCE_NAMES = (
    "scripts/finalize_pyscf_monitored_pipeline_20261031.py",
    "scripts/analyze_pyscf_monitored_results_20261031.py",
    "scripts/run_pyscf_v3_pipeline_20261031.py",
    "scripts/run_pyscf_monitored_budget_20261031.py",
)


class FakeClock:
    def __init__(self, now):
        self.now, self.sleeps = now, []
    def __call__(self):
        return self.now
    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += timedelta(seconds=seconds)


class GitFixture:
    """Exact argument mocks only; never delegates to a Git executable."""
    def __init__(self, paths, branches=None, initial_staged=None, concurrent_after_commit=False,
                 commit_blob_mismatch=False):
        self.paths, self.calls = list(paths), []
        self.branches = iter(branches or [])
        self.staged = set(initial_staged or [])
        self.head, self.committed = "a" * 40, False
        self.concurrent_after_commit = concurrent_after_commit
        self.commit_blob_mismatch = commit_blob_mismatch
    def __call__(self, arguments, deadline):
        args = list(arguments)
        self.calls.append(args)
        if args[:2] == ["branch", "--show-current"]:
            return (next(self.branches, BRANCH) + "\n").encode()
        if args == ["diff", "--cached", "--name-only", "-z"]:
            return b"\0".join(name.encode() for name in sorted(self.staged))
        if args == ["rev-parse", "HEAD"]:
            return (self.head + "\n").encode()
        if args[:3] == ["remote", "get-url", "origin"]:
            return b"https://github.com/jackzhu119/research4.git\n"
        if args[:2] == ["ls-files", "--others"]:
            return b"\0".join(name.encode() for name in self.paths)
        if args[:2] == ["diff", "--name-only"]:
            return b""
        if args[:2] == ["add", "--"]:
            self.staged = set(args[2:])
            return b""
        if args[0] == "hash-object" or (args[0] == "rev-parse" and args[1].startswith(":")):
            return b"matching_mock_blob_id\n"
        if args[0] == "rev-parse" and ":" in args[1]:
            return b"different_mock_commit_blob\n" if self.commit_blob_mismatch else b"matching_mock_blob_id\n"
        if "commit" in args:
            if "--only" not in args:
                raise AssertionError("Synthetic Git fixture requires scoped --only commit")
            self.committed, self.head = True, "b" * 40
            self.staged = {"another_researchers_work.txt"} if self.concurrent_after_commit else set()
            return b"mock commit, no Git was run\n"
        if args[0] == "diff-tree":
            return b"\0".join(name.encode() for name in self.paths)
        if args[0] == "push":
            return b"mock successful push; no network or Git operation occurred\n"
        raise AssertionError("Unexpected mocked Git command: " + repr(args))


class SyntheticBudgetLoader:
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        module.validate_ledger = lambda ledger: {"synthetic_budget_validator_mock": True}


class MonitoredFinisherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("monitored_finisher_fixture", SCRIPT)
        cls.engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.engine)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.run_root = self.repo / "reports/pyscf-spinosin-continuation-20261031-v3"
        self.pipeline = self.run_root / "pipeline-native4-001"
        self.pipeline.mkdir(parents=True)
        self.doc = self.repo / "docs/PYSCF_SPINOSIN_MONITORED_RESULTS_20261031.md"
        self.doc.parent.mkdir()
        self.doc.write_text("Synthetic finisher metadata fixture; not a scientific report.\n")
        for relative in SOURCE_NAMES:
            target = self.repo / relative
            target.parent.mkdir(exist_ok=True)
            target.write_bytes((ROOT / relative).read_bytes())
        self.addCleanup(patch.stopall)
        patch.object(self.engine, "ROOT", self.repo).start()
        patch.object(self.engine.subprocess, "run", side_effect=AssertionError(
            "This test must never execute real Git, analysis or any external command")).start()
        # Path constants are patched only if exposed; no real repository is used.
        for name, value in {
            "RUN_ROOT": self.run_root,
            "ANALYZER": self.repo / SOURCE_NAMES[1],
            "CONTROLLER": self.repo / SOURCE_NAMES[2],
            "SUPERVISOR": self.repo / SOURCE_NAMES[3],
            "FINISHER": self.repo / SOURCE_NAMES[0],
        }.items():
            if hasattr(self.engine, name):
                patch.object(self.engine, name, value).start()
        self.authorization = self.run_root / "authorization.json"
        self.authorization.write_text(json.dumps({"fixture_only": True}))
        self.ledger_path = self.run_root / "cpu_budget.json"
        self.ledger = {
            "schema": "research4-pyscf-monitored-budget-v3",
            "authorization_path": str(self.authorization),
            "authorization_sha256": self.engine.sha(self.authorization),
            "run_deadline_at_utc": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "phases": [{"phase": "ordinary_fixture", "status": "EXITED", "actual_cpu_seconds": .125,
                        "ended_at_utc": datetime.now(timezone.utc).isoformat()}],
            "consumed_cpu_seconds": .125,
        }
        self.write_ledger()
        self.state_path = self.pipeline / "state.json"
        self.state = {"status": "DONE", "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                      "ledger_sha256_at_end": self.engine.sha(self.ledger_path),
                      "synthetic_fixture_not_quantum": True}
        self.write_state()
        self.plan = self.pipeline / "plan.json"
        self.plan.write_text(json.dumps({"synthetic_plan_fixture": True}))
        self.state["plan_sha256"] = self.engine.sha(self.plan)
        self.write_state()
        self.permit_path = self.run_root / "finisher_permit.json"
        self.permit = {
            "schema": "research4-pyscf-conditional-finisher-permit-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "allow_conditional_finalization": True,
            "run_root": str(self.run_root), "pipeline_state_path": str(self.state_path),
            "ledger_path": str(self.ledger_path),
            "analysis_output_path": str(self.run_root / "final-analysis"),
            "documentation_path": str(self.doc),
            "authorization_sha256": self.engine.sha(self.authorization),
            "run_deadline_at_utc": self.ledger["run_deadline_at_utc"],
            "pipeline_plan_sha256": self.engine.sha(self.plan),
            "source_sha256": {relative: self.engine.sha(self.repo / relative) for relative in SOURCE_NAMES},
            "expected_branch": BRANCH, "allow_git_push": False,
            "max_poll_iterations": 3, "poll_interval_seconds": 15,
        }
        self.write_permit()

    def write_ledger(self):
        self.ledger_path.write_text(json.dumps(self.ledger))

    def write_state(self):
        self.state_path.write_text(json.dumps(self.state))

    def write_permit(self):
        self.permit_path.write_text(json.dumps(self.permit))

    def validated(self, push=False):
        return self.engine.validate_permit(self.permit_path, self.engine.sha(self.permit_path), push=push)

    def push_permit(self):
        self.permit["allow_git_push"] = True
        self.write_permit()
        return self.validated(push=True)

    def git_fixture(self, **changes):
        ordinary = self.run_root / "unit-result.json"
        ordinary.write_text('{"synthetic_non_scientific_fixture": true}\n')
        paths = [str(path.relative_to(self.repo)) for path in (ordinary, self.doc)]
        return GitFixture(paths, **changes)

    def run_args(self, push=False):
        return argparse.Namespace(permit=self.permit_path,
            permit_sha256=self.engine.sha(self.permit_path), push=push,
            delivery_dir=self.repo / "outputs/unit-finisher-delivery")

    def mock_budget_import(self):
        spec = importlib.util.spec_from_loader("synthetic_budget_validation_only", SyntheticBudgetLoader())
        return patch.object(self.engine.importlib.util, "spec_from_file_location", return_value=spec)

    def test_permit_binds_expected_sources_plan_authorization_and_locations(self):
        permit = self.validated()
        self.assertEqual(permit["_permit_sha256"], self.engine.sha(self.permit_path))
        self.assertEqual(permit["_run_root"], self.run_root)
        self.assertEqual(permit["_state_path"], self.state_path)
        self.assertEqual(permit["_ledger_path"], self.ledger_path)
        self.assertEqual(permit["_analysis_output"], self.run_root / "final-analysis")

    def test_missing_root_permission_or_push_permission_is_rejected(self):
        self.permit["allow_conditional_finalization"] = False
        self.write_permit()
        with self.assertRaises(ValueError):
            self.validated()
        self.permit["allow_conditional_finalization"] = True
        self.write_permit()
        with self.assertRaises(ValueError):
            self.validated(push=True)

    def test_permit_and_bound_source_changes_are_rejected(self):
        expected = self.engine.sha(self.permit_path)
        self.permit_path.write_text(self.permit_path.read_text() + "\n")
        with self.assertRaises(ValueError):
            self.engine.validate_permit(self.permit_path, expected)
        self.write_permit()
        source = self.repo / SOURCE_NAMES[2]
        source.write_text(source.read_text() + "\n# Synthetic changed source fixture\n")
        with self.assertRaises(ValueError):
            self.validated()

    def test_dangling_fixed_document_symlink_cannot_escape_or_be_overwritten(self):
        self.doc.unlink()
        outside = self.repo / "outside_missing_document.md"
        self.doc.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.validated()
        self.assertTrue(self.doc.is_symlink())
        self.assertFalse(outside.exists())

    def test_stage_gap_with_no_running_ledger_but_running_pipeline_never_closes(self):
        self.state["status"] = "RUNNING"
        self.state.pop("ended_at_utc")
        self.write_state()
        self.assertIsNone(self.engine.closed_pipeline_snapshot(self.run_root, self.state_path, self.ledger_path))

    def test_live_ledger_or_missing_terminal_timestamp_does_not_close(self):
        self.ledger["phases"][0]["status"] = "RUNNING"
        self.write_ledger()
        self.state["ledger_sha256_at_end"] = self.engine.sha(self.ledger_path)
        self.write_state()
        self.assertIsNone(self.engine.closed_pipeline_snapshot(self.run_root, self.state_path, self.ledger_path))
        self.ledger["phases"][0]["status"] = "EXITED"
        self.write_ledger()
        self.state["ledger_sha256_at_end"] = self.engine.sha(self.ledger_path)
        self.state.pop("ended_at_utc")
        self.write_state()
        self.assertIsNone(self.engine.closed_pipeline_snapshot(self.run_root, self.state_path, self.ledger_path))

    def test_terminal_done_and_partial_snapshot_bind_real_final_bytes(self):
        for status in ("DONE", "PARTIAL"):
            self.state["status"] = status
            self.write_state()
            snapshot = self.engine.closed_pipeline_snapshot(self.run_root, self.state_path, self.ledger_path)
            self.assertEqual(snapshot["ledger_sha256"], self.engine.sha(self.ledger_path))
            self.assertEqual(snapshot["pipeline_state_sha256"], self.engine.sha(self.state_path))

    def test_stale_terminal_ledger_hash_and_budget_violation_are_rejected(self):
        self.state["ledger_sha256_at_end"] = "0" * 64
        self.write_state()
        with self.assertRaises(ValueError):
            self.engine.closed_pipeline_snapshot(self.run_root, self.state_path, self.ledger_path)
        self.ledger["phases"][0]["status"] = "BUDGET_VIOLATION_REQUIRES_REVIEW"
        self.write_ledger()
        self.state["ledger_sha256_at_end"] = self.engine.sha(self.ledger_path)
        self.write_state()
        with self.assertRaises((ValueError, RuntimeError)):
            self.engine.closed_pipeline_snapshot(self.run_root, self.state_path, self.ledger_path)

    def test_wait_poll_count_is_finite_with_original_deadline(self):
        permit = self.validated()
        self.state["status"] = "RUNNING"
        self.write_state()
        clock = FakeClock(datetime.now(timezone.utc))
        with self.assertRaises(TimeoutError):
            self.engine.wait_for_closure(permit, clock=clock, sleep=clock.sleep)
        self.assertLessEqual(len(clock.sleeps), self.permit["max_poll_iterations"])
        self.assertTrue(all(value == 15 for value in clock.sleeps))

    def test_wait_does_not_sleep_after_deadline_or_extend_it(self):
        permit = self.validated()
        clock = FakeClock(permit["_deadline"] + timedelta(seconds=1))
        with self.assertRaises(TimeoutError):
            self.engine.wait_for_closure(permit, clock=clock, sleep=clock.sleep)
        self.assertEqual(clock.sleeps, [])

    def test_stage_gap_waits_until_actual_controller_terminal_marker(self):
        permit = self.validated()
        self.state["status"] = "RUNNING"
        self.state.pop("ended_at_utc")
        self.write_state()
        clock = FakeClock(datetime.now(timezone.utc))
        def sleep(seconds):
            clock.sleep(seconds)
            self.state.update(status="PARTIAL", ended_at_utc=clock().isoformat())
            self.write_state()
        snapshot = self.engine.wait_for_closure(permit, clock=clock, sleep=sleep)
        self.assertEqual(clock.sleeps, [15])
        self.assertEqual(snapshot["pipeline_state"]["status"], "PARTIAL")

    def test_release_binds_final_ledger_state_and_root_permit(self):
        permit = self.validated()
        snapshot = self.engine.closed_pipeline_snapshot(self.run_root, self.state_path, self.ledger_path)
        release = self.engine.build_release(permit, snapshot)
        rendered = json.dumps(release)
        for value in (self.engine.sha(self.ledger_path), self.engine.sha(self.state_path), self.engine.sha(self.permit_path)):
            self.assertIn(value, rendered)
        self.assertTrue(release["allow_analysis_export"])

    def test_candidate_file_rejects_symlink_deleted_and_out_of_scope(self):
        ordinary = self.run_root / "ordinary.json"
        ordinary.write_text("{}")
        records = self.engine.candidate_delivery_files(self.run_root, self.doc, [ordinary, self.doc])
        self.assertEqual(len(records), 2)
        symlink = self.run_root / "linked.json"
        symlink.symlink_to(ordinary)
        outside = self.repo / "outside.txt"
        outside.write_text("Synthetic outside fixture")
        for path in (symlink, self.run_root / "deleted.json", outside):
            with self.assertRaises((ValueError, FileNotFoundError)):
                self.engine.candidate_delivery_files(self.run_root, self.doc, [path])

    def test_candidate_size_caps_use_real_file_sizes_without_loading_giant_arrays(self):
        giant = self.run_root / "too_big.bin"
        with giant.open("wb") as stream:
            stream.truncate(25 * 1024**2 + 1)
        with self.assertRaises(ValueError):
            self.engine.candidate_delivery_files(self.run_root, self.doc, [giant])
        paths = []
        for number in range(7):
            path = self.run_root / f"sparse_fixture_{number}.bin"
            with path.open("wb") as stream:
                stream.truncate(23 * 1024**2)
            paths.append(path)
        with self.assertRaises(ValueError):
            self.engine.candidate_delivery_files(self.run_root, self.doc, paths)

    def test_existing_staged_work_stops_git_without_add_reset_or_push(self):
        permit = self.push_permit()
        git = self.git_fixture(initial_staged=["somebody_elses.txt"])
        with patch.object(self.engine, "git_output", side_effect=git):
            with self.assertRaisesRegex(RuntimeError, "Existing staged"):
                self.engine.publish_results(permit, permit["_deadline"])
        self.assertFalse(any(command[0] in ("add", "reset", "push") or "commit" in command for command in git.calls))

    def test_branch_change_before_add_preserves_shared_index(self):
        permit = self.push_permit()
        git = self.git_fixture(branches=[BRANCH, "main"])
        with patch.object(self.engine, "git_output", side_effect=git):
            with self.assertRaisesRegex(RuntimeError, "branch changed"):
                self.engine.publish_results(permit, permit["_deadline"])
        self.assertFalse(any(command[0] in ("add", "reset", "push") or "commit" in command for command in git.calls))

    def test_mock_git_delivery_is_scoped_only_commit_and_fixed_branch_push(self):
        permit = self.push_permit()
        git = self.git_fixture()
        deliveries = []
        with patch.object(self.engine, "git_output", side_effect=git):
            result = self.engine.publish_results(permit, permit["_deadline"],
                delivery_callback=lambda value: deliveries.append(dict(value)))
        self.assertEqual(result["status"], "PUSHED")
        commits = [command for command in git.calls if "commit" in command]
        self.assertEqual(len(commits), 1)
        self.assertIn("--only", commits[0])
        self.assertEqual(set(commits[0][commits[0].index("--")+1:]), set(git.paths))
        pushes = [command for command in git.calls if command[0] == "push"]
        self.assertEqual(pushes, [["push", "--porcelain", "origin", result["commit_sha"] + ":refs/heads/" + BRANCH]])
        self.assertFalse(any(command[0] in ("reset", "merge", "checkout", "switch") or "--force" in command for command in git.calls))
        self.assertEqual(deliveries[-1]["status"], "PUSHED")

    def test_changed_blob_in_actual_commit_stops_before_any_push(self):
        permit = self.push_permit()
        git = self.git_fixture(commit_blob_mismatch=True)
        with patch.object(self.engine, "git_output", side_effect=git):
            with self.assertRaisesRegex(RuntimeError, "Actual commit bytes differ"):
                self.engine.publish_results(permit, permit["_deadline"])
        self.assertTrue(git.committed)
        self.assertTrue(any(command[0] == "rev-parse" and command[1].startswith("b"*40 + ":") for command in git.calls))
        self.assertFalse(any(command[0] in ("push", "reset") for command in git.calls))

    def test_head_race_after_final_guard_still_pushes_only_audited_commit_sha(self):
        permit = self.push_permit()
        git = self.git_fixture()
        def after_last_guard(record):
            if record["push_attempted"]:
                # Model a checkout/commit immediately after the final HEAD check.
                git.head = "c" * 40
        with patch.object(self.engine, "git_output", side_effect=git):
            result = self.engine.publish_results(permit, permit["_deadline"], delivery_callback=after_last_guard)
        self.assertEqual(git.head, "c" * 40)
        self.assertEqual(result["commit_sha"], "b" * 40)
        self.assertEqual([command for command in git.calls if command[0] == "push"], [
            ["push", "--porcelain", "origin", "b" * 40 + ":refs/heads/" + BRANCH]])

    def test_concurrent_staged_work_after_scoped_commit_stops_before_push(self):
        permit = self.push_permit()
        git = self.git_fixture(concurrent_after_commit=True)
        with patch.object(self.engine, "git_output", side_effect=git):
            with self.assertRaisesRegex(RuntimeError, "staged files after commit"):
                self.engine.publish_results(permit, permit["_deadline"])
        self.assertTrue(git.committed)
        self.assertEqual(git.staged, {"another_researchers_work.txt"})
        self.assertFalse(any(command[0] in ("push", "reset") for command in git.calls))

    def test_failed_analysis_is_called_once_and_fresh_delivery_prevents_retry(self):
        self.doc.unlink()
        args = self.run_args()
        execute = Mock(return_value=subprocess.CompletedProcess(["synthetic_analyzer_not_executed"], 1,
            stdout=b"Synthetic analysis failure fixture only"))
        with self.mock_budget_import(), patch.object(self.engine, "finalization_locks", return_value=nullcontext()):
            result = self.engine.run(args, analysis_executor=execute)
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(result["analysis_attempts"], 1)
        self.assertEqual(result["status"], "PARTIAL_FINALIZATION")
        self.assertIn("no automatic retry", result["stop_reason"])
        self.assertFalse(self.doc.exists())
        with self.assertRaises(ValueError):
            self.engine.run(args, analysis_executor=execute)
        self.assertEqual(execute.call_count, 1)

    def test_no_root_go_cannot_call_analysis_or_git(self):
        self.permit["allow_conditional_finalization"] = False
        self.write_permit()
        execute = Mock()
        with self.assertRaises(ValueError):
            self.engine.run(self.run_args(), analysis_executor=execute)
        execute.assert_not_called()

    def test_post_push_delivery_record_stays_in_ignored_outputs_scope(self):
        self.doc.unlink()
        self.push_permit()
        args = self.run_args(push=True)
        def execute(command, **kwargs):
            output = self.run_root / "final-analysis"
            output.mkdir()
            (output / "summary.json").write_text('{"synthetic_non_scientific_summary_fixture": true}')
            return subprocess.CompletedProcess(command, 0, stdout=b"Synthetic analyzer mock, not executed")
        def document(path, *unused):
            path.write_text("Synthetic document dispatch fixture only.\n")
        def publication(permit, deadline, delivery_callback=None):
            record = {"status": "PUSHED", "push_attempted": True,
                      "commit_sha": "f" * 40, "synthetic_no_Git_operation": True}
            delivery_callback(record)
            return record
        with self.mock_budget_import(), patch.object(self.engine, "finalization_locks", return_value=nullcontext()), \
                patch.object(self.engine, "write_scientific_document", side_effect=document), \
                patch.object(self.engine, "publish_results", side_effect=publication):
            result = self.engine.run(args, analysis_executor=execute)
        self.assertEqual(result["status"], "SCIENTIFIC_REPORT_DONE")
        record = args.delivery_dir / "delivery.json"
        self.assertTrue(record.is_relative_to(self.repo / "outputs"))
        self.assertEqual(json.loads(record.read_text())["Git_delivery"]["status"], "PUSHED")
        self.assertFalse((self.run_root / "delivery.json").exists())
        with self.assertRaises(ValueError):
            self.engine.candidate_delivery_files(self.run_root, self.doc, [record])

    def test_document_rejects_claims_outside_analysis_only_scope(self):
        self.doc.unlink()
        snapshot = self.engine.closed_pipeline_snapshot(self.run_root, self.state_path, self.ledger_path)
        permit = self.validated()
        release = self.engine.build_release(permit, snapshot)
        with self.assertRaises(ValueError):
            self.engine.write_scientific_document(self.doc, {
                "analysis_only": True, "new_SCF_gradient_Hessian_evaluations": 0,
                "IR_intensities_status": "COMPUTED", "chemical_attribution_confirmed": False,
                "classification_modified": False}, snapshot, release)
        self.assertFalse(self.doc.exists())


if __name__ == "__main__":
    unittest.main()
