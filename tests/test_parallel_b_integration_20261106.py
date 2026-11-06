"""Isolated bare-Git integration tests of the portable result-return helper.

No quantum processes or current-repository Git mutations occur.  Successful
return tests patch only the expected transport destination and the no-Q gate
to represent a separate, idle B machine; branch/staging/file/commit/push logic
then executes against real temporary Git repositories.  Process-gate tests use
a synthetic /proc tree and never launch a fake scientific process.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/research4_parallel_b_git_return_20261106.py"
spec = importlib.util.spec_from_file_location("portable_b_git_return_under_test", SCRIPT)
HELPER = importlib.util.module_from_spec(spec)
spec.loader.exec_module(HELPER)


def run_git(repo, *args, bare=False):
    command = ["git", "--git-dir=" + str(repo)] if bare else ["git", "-C", str(repo)]
    result = subprocess.run(command + list(args), text=True, capture_output=True, timeout=20)
    if result.returncode:
        raise AssertionError(f"Fixture git operation failed: {args[0]} ({result.returncode})")
    return result.stdout.strip()


class PortableBReturnIntegration(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="research4-portable-b-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.remote = self.base / "remote.git"
        self.repo = self.base / "checkout"
        subprocess.run(["git", "init", "--bare", str(self.remote)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["git", "init", "--initial-branch=main", str(self.repo)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        run_git(self.repo, "config", "user.name", "Portable integration fixture")
        run_git(self.repo, "config", "user.email", "fixture@example.invalid")
        (self.repo / "README.md").write_text("fixture base\n")
        run_git(self.repo, "add", "README.md")
        run_git(self.repo, "commit", "-m", "Isolated fixture base")
        run_git(self.repo, "remote", "add", "origin", str(self.remote))
        run_git(self.repo, "push", "origin", "main")
        self.original_main = run_git(self.remote, "rev-parse", "refs/heads/main", bare=True)
        self.run_id = "integration-001"
        self.work_rel = "reports/parallel-cloud-20261106/worker-b/" + self.run_id
        self.work = self.repo / self.work_rel
        self.work.mkdir(parents=True)
        self.branch = "research/parallel-worker-b-20261106-" + self.run_id
        run_git(self.repo, "switch", "-c", self.branch)
        self.manifest = {
            "role": "B", "run_id": self.run_id, "work_directory": self.work_rel,
            "ledger": self.work_rel + "/cpu_budget.json",
            "upload": {"branch": self.branch, "allowlist": ["result.json"]}}
        self.manifest_path = self.repo / "manifest.json"
        self.save_manifest()
        (self.work / "cpu_budget.json").write_text(json.dumps({"phases": []}))
        (self.work / "result.json").write_text(json.dumps({"status": "ACTUAL_TEST_FIXTURE_NOT_QUANTUM_RESULT"}))

    def save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest, sort_keys=True))
        self.manifest_sha = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()

    def production_return_with_local_transport(self, attempt="return-test-001"):
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(HELPER, "EXPECTED_ORIGIN", str(self.remote)))
            # This models a separate idle B host.  No live A scan is suppressed
            # outside this temporary helper invocation.
            idle = stack.enter_context(mock.patch.object(HELPER, "require_no_active_quantum", return_value=None))
            result = HELPER.execute_return(self.repo, self.manifest, self.manifest_path,
                                           self.manifest_sha, attempt=attempt)
            self.assertGreaterEqual(idle.call_count, 3)
            return result

    def assert_remote_main_preserved(self):
        self.assertEqual(run_git(self.remote, "rev-parse", "refs/heads/main", bare=True), self.original_main)

    def test_real_two_commit_push_and_remote_verification_only_allowlisted_files(self):
        other = self.repo / "reports/parallel-cloud-20261106/worker-b/other-run/result.json"
        other.parent.mkdir(parents=True); other.write_text("other account run\n")
        (self.work / "state.chk").write_text("Unselected checkpoint fixture\n")
        original_git = HELPER.git; commands = []
        def traced_git(repo, *args):
            commands.append(args)
            return original_git(repo, *args)
        with mock.patch.object(HELPER, "git", side_effect=traced_git):
            result = self.production_return_with_local_transport()
        self.assertEqual(result["actual_result_upload"], "DONE")
        self.assertEqual(result["status"], "DONE_ACTUAL_PUSH_AND_REMOTE_REF_VERIFIED")
        head = run_git(self.remote, "rev-parse", "refs/heads/" + self.branch, bare=True)
        self.assertEqual(head, result["final_metadata_commit"])
        self.assertEqual(run_git(self.repo, "rev-list", "--count", self.original_main + "..HEAD"), "2")
        remote_files = set(run_git(self.remote, "ls-tree", "-r", "--name-only", head, bare=True).splitlines())
        self.assertEqual(remote_files, {
            "README.md", self.work_rel + "/result.json",
            self.work_rel + "/git-return/return-test-001/return_receipt.json"})
        self.assertTrue((self.work / "state.chk").exists())
        self.assertTrue(other.exists())
        receipt = json.loads(run_git(self.remote, "show", head + ":" + self.work_rel +
                                     "/git-return/return-test-001/return_receipt.json", bare=True))
        self.assertEqual(receipt["science_commit"], result["science_commit"])
        self.assertEqual(receipt["remote_download_of_content"], "NOT_RUN")
        for args in commands:
            self.assertFalse(any(str(a).startswith("--force") for a in args))
            if args[0] == "push":
                self.assertEqual(args, ("push", "origin", "HEAD:refs/heads/" + self.branch))
        self.assert_remote_main_preserved()

    def test_current_main_branch_is_rejected_before_any_staging(self):
        run_git(self.repo, "switch", "main")
        with self.assertRaises(HELPER.SafetyError):
            self.production_return_with_local_transport()
        self.assertFalse(run_git(self.repo, "diff", "--cached", "--name-only"))
        self.assert_remote_main_preserved()

    def test_declared_main_or_cross_role_branch_is_rejected(self):
        for branch in ["main", HELPER.BRANCH_A, HELPER.SOURCE_BRANCH_B, self.branch + "-different"]:
            with self.subTest(branch=branch):
                self.manifest["upload"]["branch"] = branch
                with self.assertRaises(HELPER.SafetyError):
                    HELPER.collect_return_files(self.repo, self.manifest)

    def test_preexisting_other_staged_work_is_preserved_and_rejected(self):
        (self.repo / "README.md").write_text("Other staged work must survive\n")
        run_git(self.repo, "add", "README.md")
        before = run_git(self.repo, "diff", "--cached")
        with self.assertRaises(HELPER.SafetyError):
            self.production_return_with_local_transport()
        self.assertEqual(run_git(self.repo, "diff", "--cached"), before)
        self.assertFalse((self.work / "git-return").exists())
        self.assert_remote_main_preserved()

    def test_hash_changed_manifest_is_rejected(self):
        self.manifest_path.write_text("{}")
        with self.assertRaises(HELPER.SafetyError):
            self.production_return_with_local_transport()
        self.assertFalse(run_git(self.repo, "diff", "--cached", "--name-only"))

    def test_other_run_and_A_output_prefix_cannot_be_selected(self):
        for work in ["reports/parallel-cloud-20261106/worker-b/other-run",
                     "reports/cloud-followup-20261106-v1/queue-host-a-001",
                     "data/raw", "../outside"]:
            with self.subTest(work=work):
                self.manifest["work_directory"] = work
                with self.assertRaises(HELPER.SafetyError):
                    HELPER.collect_return_files(self.repo, self.manifest)

    def test_absolute_parent_and_hidden_allowlist_patterns_rejected(self):
        for pattern in ["../result.json", "/tmp/*.json", ".env", "../other-run/*.json"]:
            with self.subTest(pattern=pattern):
                self.manifest["upload"]["allowlist"] = [pattern]
                with self.assertRaises(HELPER.SafetyError):
                    HELPER.collect_return_files(self.repo, self.manifest)

    def test_explicit_secret_checkpoint_runtime_and_binary_selection_rejected(self):
        forbidden = ["secret.json", "credentials.txt", "access_token.json", "private_key.txt",
                     "api_key.txt", "apikey.json", "password.json", "passwd.txt", "id_rsa.txt", "id_ed25519.txt",
                     "state.chk", "state.h5", "model.joblib", "key.pem", "model.pt", "weights.pkl",
                     "cache/result.json", ".runtime/result.json", "venv/result.json",
                     ".ssh/config.json", ".aws/config.json", ".azure/config.json"]
        for name in forbidden:
            with self.subTest(name=name):
                p = self.work / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("synthetic fixture\n")
                self.manifest["upload"]["allowlist"] = [name]
                with self.assertRaises(HELPER.SafetyError):
                    HELPER.collect_return_files(self.repo, self.manifest)

    def test_symlink_file_and_directory_escape_rejected(self):
        outside = self.base / "outside.json"; outside.write_text("synthetic outside fixture\n")
        link = self.work / "link.json"; link.symlink_to(outside)
        self.manifest["upload"]["allowlist"] = ["link.json"]
        with self.assertRaises(HELPER.SafetyError):
            HELPER.collect_return_files(self.repo, self.manifest)
        link.unlink()
        external_dir = self.base / "external"; external_dir.mkdir(); (external_dir / "result.json").write_text("outside\n")
        (self.work / "linked").symlink_to(external_dir, target_is_directory=True)
        self.manifest["upload"]["allowlist"] = ["linked/result.json"]
        with self.assertRaises(HELPER.SafetyError):
            HELPER.collect_return_files(self.repo, self.manifest)

    def test_single_large_file_and_aggregate_limit_rejected(self):
        large = self.work / "large.npy"
        with large.open("wb") as f: f.truncate(HELPER.MAX_FILE + 1)
        self.manifest["upload"]["allowlist"] = ["large.npy"]
        with self.assertRaises(HELPER.SafetyError):
            HELPER.collect_return_files(self.repo, self.manifest)
        large.unlink()
        for i in range(5):
            with (self.work / f"part-{i}.npy").open("wb") as f: f.truncate(HELPER.MAX_FILE)
        self.manifest["upload"]["allowlist"] = ["part-*.npy"]
        with self.assertRaises(HELPER.SafetyError):
            HELPER.collect_return_files(self.repo, self.manifest)

    def test_unresolved_ledger_prevents_git(self):
        (self.work / "cpu_budget.json").write_text(json.dumps({"phases":[{"status":"RUNNING"}]}))
        with self.assertRaises(HELPER.SafetyError):
            HELPER.require_no_active_quantum(self.repo, self.manifest)

    def test_actual_B_quantum_name_is_rejected_by_synthetic_proc_scan(self):
        proc_root = self.base / "synthetic-proc"; proc = proc_root / "99123"; proc.mkdir(parents=True)
        (proc / "stat").write_text("99123 (python) S 1 1 1\n")
        (proc / "cmdline").write_bytes(b"python\0scripts/research4_parallel_b_quantum_20261106.py\0--case\0water\0")
        real_path = Path
        def path_with_synthetic_proc(value):
            return proc_root if str(value) == "/proc" else real_path(value)
        with mock.patch.object(HELPER, "Path", side_effect=path_with_synthetic_proc):
            with self.assertRaises(HELPER.SafetyError):
                HELPER.require_no_active_quantum(self.repo, self.manifest)

    def test_prior_attempt_receipt_is_not_overwritten(self):
        (self.work / "git-return/return-test-001").mkdir(parents=True)
        marker = self.work / "git-return/return-test-001/old.json"; marker.write_text("immutable old attempt\n")
        with self.assertRaises(HELPER.SafetyError):
            self.production_return_with_local_transport()
        self.assertEqual(marker.read_text(), "immutable old attempt\n")
        self.assert_remote_main_preserved()

    def test_failed_first_push_preserved_then_fresh_attempt_reuses_science_commit(self):
        original_git = HELPER.git
        failed = []
        def once_failed_push(repo, *args):
            if args[0] == "push" and not failed:
                failed.append(args)
                raise HELPER.SafetyError("Synthetic first push transport failure")
            return original_git(repo, *args)
        with mock.patch.object(HELPER, "git", side_effect=once_failed_push):
            with self.assertRaises(HELPER.SafetyError):
                self.production_return_with_local_transport(attempt="return-failed-001")
        self.assertEqual(len(failed), 1)
        failed_receipt = self.work / "git-return/return-failed-001/failure_receipt.json"
        original_failure_bytes = failed_receipt.read_bytes()
        old_science = run_git(self.repo, "rev-parse", "HEAD")
        self.assertEqual(run_git(self.repo, "rev-list", "--count", self.original_main + "..HEAD"), "1")
        self.assertFalse(run_git(self.repo, "diff", "--cached", "--name-only"))
        result = self.production_return_with_local_transport(attempt="return-recovery-002")
        self.assertTrue(result["reused_existing_local_science_commit"])
        self.assertEqual(result["science_commit"], old_science)
        self.assertEqual(result["actual_result_upload"], "DONE")
        self.assertEqual(failed_receipt.read_bytes(), original_failure_bytes)
        self.assertEqual(run_git(self.repo, "rev-list", "--count", self.original_main + "..HEAD"), "2")
        self.assertEqual(run_git(self.remote, "rev-parse", "refs/heads/" + self.branch, bare=True),
                         result["final_metadata_commit"])
        self.assert_remote_main_preserved()

    def test_real_CRLF_scientific_CSV_returns_without_byte_or_hash_change(self):
        content = b"mode,frequency_cm1,IR_status\r\n1,37.02736962390627,NOT_COMPUTED\r\n"
        csv_path = self.work / "frequencies.csv"; csv_path.write_bytes(content)
        before = hashlib.sha256(content).hexdigest()
        self.manifest["upload"]["allowlist"] = ["result.json", "frequencies.csv"]
        self.save_manifest()
        result = self.production_return_with_local_transport()
        relative = self.work_rel + "/frequencies.csv"
        blob = subprocess.run(["git", "--git-dir=" + str(self.remote), "cat-file", "blob",
                               result["science_commit"] + ":" + relative],
                              capture_output=True, check=True, timeout=20).stdout
        self.assertEqual(csv_path.read_bytes(), content)
        self.assertEqual(blob, content)
        self.assertEqual(hashlib.sha256(blob).hexdigest(), before)
        self.assertEqual(result["artifact_hashes"][relative]["sha256"], before)
        self.assert_remote_main_preserved()

    def test_B_gate_uses_own_ledger_not_archived_A_running_snapshot(self):
        old_a = self.repo / "reports/pyscf-spinosin-96h-continuation-20261105-v8/cpu_budget.json"
        old_a.parent.mkdir(parents=True)
        old_a.write_text(json.dumps({"phases":[{"status":"RUNNING"}]}))
        empty_proc = self.base / "empty-synthetic-proc"; empty_proc.mkdir()
        real_path = Path
        def path_with_synthetic_proc(value):
            return empty_proc if str(value) == "/proc" else real_path(value)
        with mock.patch.object(HELPER, "Path", side_effect=path_with_synthetic_proc):
            HELPER.require_no_active_quantum(self.repo, self.manifest)
        role_a = {"role":"A", "run_id":"host-a-001",
                  "work_directory":"reports/cloud-followup-20261106-v1/queue-host-a-001",
                  "ledger":str(old_a.relative_to(self.repo)),
                  "upload":{"branch":HELPER.BRANCH_A,"allowlist":["*.json"]}}
        with self.assertRaises(HELPER.SafetyError):
            HELPER.require_no_active_quantum(self.repo, role_a)

    def opaque_proc_fixture(self, comm, uid):
        root = self.base / f"opaque-{comm}-{uid}"; proc = root / "99124"; proc.mkdir(parents=True)
        fields = ["S"] + ["0"] * 23; fields[19] = "7654321"
        (proc / "stat").write_text("99124 (" + comm + ") " + " ".join(fields) + "\n")
        (proc / "status").write_text("Uid:\t" + "\t".join([str(uid)]*4) + "\n")
        (proc / "cmdline").write_bytes(b"opaque\0")
        real_path = Path; real_read = Path.read_bytes
        def path_factory(value):
            return root if str(value) == "/proc" else real_path(value)
        def inaccessible_cmdline(path):
            if path == proc / "cmdline":
                raise PermissionError("synthetic managed daemon visibility boundary")
            return real_read(path)
        stack = ExitStack()
        stack.enter_context(mock.patch.object(HELPER, "Path", side_effect=path_factory))
        stack.enter_context(mock.patch.object(Path, "read_bytes", inaccessible_cmdline))
        return stack

    def test_opaque_root_python_or_user_daemon_cannot_bypass_gate(self):
        for comm, uid in [("python", 0), ("dockerd", 1000)]:
            with self.subTest(comm=comm, uid=uid), self.opaque_proc_fixture(comm, uid):
                with self.assertRaises(HELPER.SafetyError):
                    HELPER.require_no_active_quantum(self.repo, self.manifest)

    def test_only_exact_stable_root_daemon_visibility_exception_is_accepted(self):
        with self.opaque_proc_fixture("dockerd", 0):
            HELPER.require_no_active_quantum(self.repo, self.manifest)

    def test_daemon_birth_change_between_identity_reads_is_rejected(self):
        first = {"pid":99124,"start_ticks":1,"uid":0,"comm":"dockerd","boot_id":"fixtureboot"}
        second = {**first, "start_ticks":2}
        with self.opaque_proc_fixture("containerd", 0):
            with mock.patch.object(HELPER, "proc_identity", side_effect=[first, second]):
                with self.assertRaises(HELPER.SafetyError):
                    HELPER.require_no_active_quantum(self.repo, self.manifest)

    def prepare_remote_claim_fixture(self):
        # These are temporary test files, never current Research4 data.
        shutil.rmtree(self.work)
        self.manifest_path.unlink()
        run_git(self.repo, "switch", "-c", HELPER.SOURCE_BRANCH_B)
        run_git(self.repo, "branch", "-D", self.branch)
        copied_script = self.repo / "scripts/research4_parallel_b_git_return_20261106.py"
        copied_script.parent.mkdir(); copied_script.write_bytes(SCRIPT.read_bytes())
        release = self.repo / "data/research/parallel_cloud_20261106/worker_b/release_ROOT_APPROVED.json"
        release.parent.mkdir(parents=True)
        release.write_text(json.dumps({
            "schema":"research4-parallel-b-release-v1", "release_enabled":True,
            "remote_run_id":self.run_id,
            "deadline_utc":(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
            "source_refs":[{"path":str(copied_script.relative_to(self.repo)),
                            "sha256":hashlib.sha256(copied_script.read_bytes()).hexdigest()}]}))
        run_git(self.repo, "add", "--", str(copied_script.relative_to(self.repo)), str(release.relative_to(self.repo)))
        run_git(self.repo, "commit", "-m", "Temporary ROOT-release fixture, not actual authorization")
        run_git(self.repo, "push", "origin", HELPER.SOURCE_BRANCH_B)
        return release

    def test_two_independent_clones_cannot_claim_same_remote_B_allocation(self):
        release = self.prepare_remote_claim_fixture()
        second = self.base / "independent-second-checkout"
        subprocess.run(["git", "clone", "--branch", HELPER.SOURCE_BRANCH_B,
                        str(self.remote), str(second)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        run_git(second, "config", "user.name", "Second fixture host")
        run_git(second, "config", "user.email", "second@example.invalid")
        second_release = second / release.relative_to(self.repo)
        original_git = HELPER.git; second_commands=[]
        def traced(repo,*args):
            if Path(repo)==second:second_commands.append(args)
            return original_git(repo,*args)
        with mock.patch.object(HELPER, "EXPECTED_ORIGIN", str(self.remote)), \
             mock.patch.object(HELPER, "require_no_live_quantum_processes", return_value=None), \
             mock.patch.object(HELPER, "git", side_effect=traced):
            winner=HELPER.claim_result_branch(self.repo,self.run_id,release)
            self.assertEqual(winner['status'],'DONE_ATOMIC_REMOTE_BRANCH_CLAIM_NO_Q')
            with self.assertRaisesRegex(HELPER.SafetyError,'BLOCK_ALREADY_CLAIMED'):
                HELPER.claim_result_branch(second,self.run_id,second_release)
            # Even fetching the public claim does not supply a private local
            # ownership marker or authorize a second CPU ledger.
            run_git(second,'fetch','origin',self.branch+':'+self.branch)
            run_git(second,'switch',self.branch)
            self.assertTrue((second/self.work_rel/'owner_claim.json').is_file())
            with self.assertRaisesRegex(HELPER.SafetyError,'BLOCK_ALREADY_CLAIMED'):
                HELPER.claim_result_branch(second,self.run_id,second_release)
            self.assertFalse(any(args[0]=='push' for args in second_commands))
            self.assertFalse((second/'.runtime/worker-b-ownership'/f'{self.run_id}.json').exists())
            self.assertFalse((second/self.work_rel/'cpu_budget_B.json').exists())
            marker=Path(winner['local_marker']); before=marker.read_bytes()
            resumed=HELPER.claim_result_branch(self.repo,self.run_id,release)
            self.assertEqual(resumed['status'],'EXISTING_LOCAL_OWNER_VERIFIED_NO_NEW_ALLOCATION')
            self.assertEqual(marker.read_bytes(),before)
        self.assertEqual(run_git(self.remote,'rev-parse','refs/heads/'+self.branch,bare=True),winner['claim_commit'])
        self.assertFalse((self.work/'cpu_budget_B.json').exists())
        self.assert_remote_main_preserved()

    def test_failed_claim_transport_reuses_same_nonce_and_commit_without_new_allocation(self):
        release=self.prepare_remote_claim_fixture();original_git=HELPER.git;failed=[]
        def fail_first_actual_claim_push(repo,*args):
            if args[0]=='push' and '--dry-run' not in args and not failed:
                failed.append(args);raise HELPER.SafetyError('Synthetic claim transport failure')
            return original_git(repo,*args)
        with mock.patch.object(HELPER,'EXPECTED_ORIGIN',str(self.remote)), \
             mock.patch.object(HELPER,'require_no_live_quantum_processes',return_value=None):
            with mock.patch.object(HELPER,'git',side_effect=fail_first_actual_claim_push):
                with self.assertRaises(HELPER.SafetyError):
                    HELPER.claim_result_branch(self.repo,self.run_id,release)
            marker=self.repo/'.runtime/worker-b-ownership'/f'{self.run_id}.json'
            previous=json.loads(marker.read_text());old_head=run_git(self.repo,'rev-parse','HEAD')
            self.assertFalse(previous['remote_verified'])
            result=HELPER.claim_result_branch(self.repo,self.run_id,release)
            current=json.loads(marker.read_text())
            self.assertEqual(result['claim_commit'],old_head)
            self.assertEqual(current['nonce'],previous['nonce'])
            self.assertEqual(current['source_commit'],previous['source_commit'])
            self.assertTrue(current['remote_verified'])
            self.assertFalse((self.work/'cpu_budget_B.json').exists())
        self.assert_remote_main_preserved()

    def test_untracked_fabricated_release_cannot_claim_CPU_allocation(self):
        release = self.prepare_remote_claim_fixture()
        fake = release.with_name("untracked_release.json"); fake.write_bytes(release.read_bytes())
        before = run_git(self.repo, "rev-parse", "HEAD")
        with mock.patch.object(HELPER, "EXPECTED_ORIGIN", str(self.remote)), \
             mock.patch.object(HELPER, "require_no_live_quantum_processes", return_value=None):
            with self.assertRaises(HELPER.SafetyError):
                HELPER.claim_result_branch(self.repo, self.run_id, fake)
        self.assertEqual(run_git(self.repo, "rev-parse", "HEAD"), before)
        self.assertFalse(self.work.exists())
        self.assert_remote_main_preserved()

    def test_runtime_ownership_symlink_cannot_write_outside_checkout(self):
        release = self.prepare_remote_claim_fixture()
        outside = self.base / "outside-marker"; outside.mkdir()
        (self.repo / ".runtime").mkdir()
        (self.repo / ".runtime/worker-b-ownership").symlink_to(outside, target_is_directory=True)
        with mock.patch.object(HELPER, "EXPECTED_ORIGIN", str(self.remote)), \
             mock.patch.object(HELPER, "require_no_live_quantum_processes", return_value=None):
            with self.assertRaises(HELPER.SafetyError):
                HELPER.claim_result_branch(self.repo, self.run_id, release)
        self.assertFalse(list(outside.iterdir()))
        self.assert_remote_main_preserved()


if __name__ == "__main__":
    unittest.main()
