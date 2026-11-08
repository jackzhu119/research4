"""No real Git, network, numerical libraries, fitting or inference."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "scripts/species_relaunch_backup_20261108.py"
SPEC = importlib.util.spec_from_file_location("backup_worker", SOURCE)
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.root = self.repo / M.BASE; self.root.mkdir(parents=True)
        self.output = self.root / "backups/001"; self.output.mkdir(parents=True)
        self.source = self.repo / "scripts/backup.py"; self.source.parent.mkdir(); self.source.write_text("frozen")
        self.patch_repo = patch.object(M, "REPO", self.repo); self.patch_repo.start(); self.addCleanup(self.patch_repo.stop)
        self.patch_file = patch.object(M, "__file__", str(self.source)); self.patch_file.start(); self.addCleanup(self.patch_file.stop)
        # An accidental real subprocess call in any test fails immediately.
        self.no_network = patch.object(M.subprocess, "run", side_effect=AssertionError("Real process prohibited"))
        self.no_network.start(); self.addCleanup(self.no_network.stop)
        for name in ("train_only_features.npz", "train_only_split.csv"):
            (self.root / name).write_text("TRAIN_ONLY")
        M.write(self.root / "input_lock.json", {"training_feature_sha256": M.sha(self.root / "train_only_features.npz"),
                "training_split_sha256": M.sha(self.root / "train_only_split.csv")})
        M.write(self.root / "protocol.json", {"input_lock_sha256": M.sha(self.root / "input_lock.json"),
                "worker_code_sha256": {"scripts/backup.py": M.sha(self.source)}})
        self.run = self.root / "runs/done"; self.run.mkdir(parents=True)
        M.write(self.run / "summary.json", {"metrics": {"accuracy": 0.5}})
        (self.run / "weights.pt").write_bytes(b"whole-weights")
        M.write(self.run / "complete.json", {"status": "complete", "artifacts":
                {p.name: M.sha(p) for p in self.run.iterdir()}})
        self.log = self.root / "control/logs/job.log"; self.log.parent.mkdir(parents=True); self.log.write_text("closed")
        self.job = {"id": "done", "kind": "classical", "cwd": str(self.repo),
                    "command": ["python", "worker.py"], "inputs": [str(self.source)],
                    "outputs": [str(self.run / "summary.json"), str(self.run / "complete.json")], "max_retries": 1}
        self.backup = {"id": "backup", "kind": "backup", "cwd": str(self.repo), "command": ["python"],
                       "outputs": [str(self.output / "receipt.json"), str(self.output / "complete.json")], "max_retries": 0}
        self.manifest = {"protocol_sha256": M.sha(self.root / "protocol.json"), "jobs": [self.job, self.backup]}
        M.write(self.root / "jobs.json", self.manifest)
        receipt = {"job_sha256": M.objhash(self.job), "input_hashes": {str(self.source): M.sha(self.source)},
                   "output_hashes": {p: M.sha(p) for p in self.job["outputs"]},
                   "log_path": str(self.log), "log_sha256": M.sha(self.log)}
        self.receipt = self.root / "control/receipts/done.json"; self.receipt.parent.mkdir(); M.write(self.receipt, receipt)
        self.state = {"manifest_sha256": M.objhash({"manifest": self.manifest, "normalized_jobs": self.manifest["jobs"],
                "budget_cpu_seconds": 88000.}), "active": {"job_id": "backup"}, "spent_cpu_seconds": 8.5,
                "jobs": {"done": {"status": "completed", "receipt_sha256": M.sha(self.receipt)}, "backup": {"status": "running"}},
                "input_hashes": {"done": receipt["input_hashes"]}}
        M.write(self.root / "control/state.json", self.state)
        M.write(self.root / "preparation_cpu.json", {"consumed_cpu_seconds": 26.6})

    def save_state(self):
        M.write(self.root / "control/state.json", self.state)

    def test_sealed_candidates_include_full_weights_not_inputs_or_live_state(self):
        chosen = M.candidates(self.root, self.output)
        names = "\n".join(chosen)
        self.assertIn("weights.pt", names); self.assertNotIn(".npz", names)
        self.assertNotIn("control/state.json", names)
        projection = M.read(self.output / "budget_progress_snapshot.json")
        self.assertEqual(projection["state"]["spent_cpu_seconds"], 8.5)
        self.assertNotIn("input_hashes", projection["state"])

    def test_changed_closed_artifact_blocks(self):
        (self.run / "weights.pt").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            M.candidates(self.root, self.output)

    def test_different_running_job_blocks(self):
        self.state["active"]["job_id"] = "other"; self.save_state()
        with self.assertRaisesRegex(ValueError, "Another queue job"):
            M.candidates(self.root, self.output)

    def test_unsealed_job_is_not_published(self):
        self.state["jobs"]["done"]["status"] = "failed"; self.save_state()
        self.assertNotIn(str(self.run.relative_to(self.repo)) + "/weights.pt", M.candidates(self.root, self.output))

    def test_changed_train_pack_blocks(self):
        (self.root / "train_only_features.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "TRAIN input drift"):
            M.candidates(self.root, self.output)

    def test_path_escape_blocks(self):
        with self.assertRaises(ValueError):
            M.inside(self.repo / "elsewhere", self.root)

    def test_raw_and_partial_extensions_rejected(self):
        for name in ("x.xlsx", "features.npz", "file.partial.json", "x.h5", "file.tmp.3.json"):
            self.assertFalse(M.permitted(Path(name)))
        for name in ("model.joblib", "weights.pt", "predictions.csv.gz"):
            self.assertTrue(M.permitted(Path(name)))

    def test_remote_exact_repo_only(self):
        self.assertTrue(M.remote_allowed("https://github.com/jackzhu119/research4.git"))
        self.assertTrue(M.remote_allowed("git@github.com:jackzhu119/research4.git"))
        for url in ("https://github.com.evil/jackzhu119/research4", "https://github.com/other/research4", "/tmp/local"):
            self.assertFalse(M.remote_allowed(url))

    def fake_git(self, *, branch=None, staged=b"", remote="new", failure=None):
        calls = []
        def run(*args):
            calls.append(args)
            if failure and args[0] == failure:
                raise ValueError("Git operation failed: " + args[0])
            if args == ("branch", "--show-current"): return (branch or M.BRANCH).encode()
            if args[:2] == ("remote", "get-url"): return b"https://github.com/jackzhu119/research4.git"
            if args[:2] == ("diff", "--cached"): return staged
            if args == ("rev-parse", "--show-object-format"): return b"sha1"
            if args == ("rev-parse", "HEAD"): return b"new"
            if args[0] == "ls-tree": return b""
            if args[0] == "ls-remote": return (remote + "\trefs/heads/" + M.BRANCH).encode()
            if args[0] in {"push", "add", "commit"}: return b""
            raise AssertionError(args)
        return run, calls

    def test_empty_success_still_verifies_remote_head(self):
        runner, calls = self.fake_git()
        with patch.object(M, "git", side_effect=runner): result = M.publish({}, True)
        self.assertEqual(result["status"], "DONE")
        self.assertTrue(any(c[0] == "ls-remote" for c in calls))
        self.assertFalse(any(c[0] in {"add", "commit"} for c in calls))

    def test_main_and_unknown_index_never_mutated(self):
        for options in ({"branch": "main"}, {"staged": b"other.txt\0"}):
            runner, calls = self.fake_git(**options)
            with patch.object(M, "git", side_effect=runner), self.assertRaises(ValueError): M.publish({}, True)
            self.assertFalse(any(c[0] in {"add", "commit", "push", "reset"} for c in calls))

    def test_remote_mismatch_does_not_claim_success(self):
        runner, calls = self.fake_git(remote="other")
        with patch.object(M, "git", side_effect=runner), self.assertRaisesRegex(ValueError, "Remote head"):
            M.publish({}, True)

    def test_network_failure_bounded_no_science_retry(self):
        runner, calls = self.fake_git(failure="push")
        with patch.object(M, "git", side_effect=runner), self.assertRaisesRegex(ValueError, "Git operation failed: push"):
            M.publish({}, True)
        self.assertEqual(sum(c[0] == "push" for c in calls), 1)

    def test_oversize_omitted_with_hash(self):
        selected = {M.BASE + "/cache/huge.pt": {"bytes": M.FILE_CAP + 1, "sha256": "local-sha", "eligible": False}}
        runner, _ = self.fake_git()
        with patch.object(M, "git", side_effect=runner): result = M.publish(selected, False)
        self.assertEqual(next(iter(result["not_backed_up"].values()))["sha256"], "local-sha")
        self.assertEqual(result["status"], "NOT_RUN")

    def test_git_subprocess_shell_false_timeout_redacts_stderr(self):
        result = subprocess.CompletedProcess(["git"], 1, b"", b"secret-token-in-url")
        with patch.object(M.subprocess, "run", return_value=result) as mocked:
            with self.assertRaisesRegex(ValueError, "^Git operation failed: push$"): M.git("push", "origin")
        self.assertFalse(mocked.call_args.kwargs["shell"])
        self.assertLessEqual(mocked.call_args.kwargs["timeout"], 60)


if __name__ == "__main__":
    unittest.main()
