"""V4 finisher control-flow/Git mocks only; no QM, analyzer or real Git mutation.

Every fixture lives in a temporary fake repository. No actual ledger, index,
supervisor completion marker or quantum array is read by these tests.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/finalize_pyscf_pure_j_20261101.py"
BRANCH = "research/pyscf-spinosin-mechanism-20261031"
FIXED_EMAIL = "195995062+jackzhu119@users.noreply.github.com"


class FakeClock:
    def __init__(self, now):
        self.now, self.sleeps = now, []
    def __call__(self):
        return self.now
    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += timedelta(seconds=seconds)


class GitExecutorFixture:
    """Subprocess-shaped argument mock; never invokes Git or any executable."""
    def __init__(self, paths, *, branches=None, initial_staged=None, modified=None,
                 unrelated_untracked=None, origin=None, concurrent_after_commit=False,
                 commit_blob_mismatch=False, index_blob_mismatch=False, commit_scope_extra=None,
                 deleted=None, commit_parent_mismatch=False, changed_origin_after_commit=None,
                 commit_sha=None):
        self.paths = list(paths)
        self.calls = []
        self.branches = iter(branches or [])
        self.staged = set(initial_staged or [])
        self.modified = set(modified or [])
        self.unrelated_untracked = set(unrelated_untracked or [])
        self.deleted=set(deleted or [])
        self.origin = origin or "https://github.com/jackzhu119/research4.git"
        self.concurrent_after_commit = concurrent_after_commit
        self.commit_blob_mismatch, self.index_blob_mismatch = commit_blob_mismatch,index_blob_mismatch
        self.commit_scope_extra = commit_scope_extra
        self.commit_parent_mismatch=commit_parent_mismatch
        self.changed_origin_after_commit=changed_origin_after_commit
        self.commit_sha=commit_sha or "b"*40
        self.head, self.committed = "a"*40,False

    @staticmethod
    def encoded(names):
        return b"\0".join(os.fsencode(name) for name in sorted(names)) + (b"\0" if names else b"")

    def __call__(self, argv, **kwargs):
        args = list(argv)
        if not args or args[0] != "git":
            raise AssertionError("Fixture cannot execute analyzer, QM or external commands")
        args = args[1:]
        self.calls.append({"arguments":args,"kwargs":dict(kwargs)})
        stdout = self.answer(args)
        return subprocess.CompletedProcess(argv,0,stdout=stdout,stderr=b"")

    def answer(self,args):
        if args[:2] == ["branch","--show-current"]:
            return (next(self.branches,BRANCH)+"\n").encode()
        if args == ["diff","--cached","--name-only","-z"]:
            return self.encoded(self.staged)
        if args == ["rev-parse","HEAD"]:
            return (self.head+"\n").encode()
        if args[:3] == ["remote","get-url","origin"]:
            value=self.changed_origin_after_commit if self.committed and self.changed_origin_after_commit else self.origin
            return (value+"\n").encode()
        if args and args[0]=="rev-parse" and args[1].endswith("^"):
            return (("c"*40 if self.commit_parent_mismatch else "a"*40)+"\n").encode()
        if args and args[0] == "status":
            rows = ["A  "+name for name in sorted(self.staged)]
            rows += [" M "+name for name in sorted(self.modified-self.staged)]
            untracked = self.unrelated_untracked | (set() if self.committed else set(self.paths))
            rows += ["?? "+name for name in sorted(untracked-self.staged)]
            return b"\0".join(os.fsencode(row) for row in rows) + (b"\0" if rows else b"")
        if args[:2] == ["ls-files","--others"]:
            return self.encoded(self.unrelated_untracked | (set() if self.committed else set(self.paths)))
        if args[:2] == ["diff","--name-only"]:
            return self.encoded(self.modified|self.deleted)
        if args[:2] == ["diff","--diff-filter=D"]:
            return self.encoded(self.deleted)
        if args[:2] == ["add","--"]:
            self.staged = set(args[2:]);return b""
        if args and args[0] == "hash-object":
            return ("synthetic_blob_"+args[-1]+"\n").encode()
        if args and args[0] == "rev-parse" and args[1].startswith(":"):
            return ("synthetic_wrong_index_blob\n" if self.index_blob_mismatch
                    else "synthetic_blob_"+args[1][1:]+"\n").encode()
        if args and args[0] == "rev-parse" and ":" in args[1]:
            return ("synthetic_wrong_commit_blob\n" if self.commit_blob_mismatch
                    else "synthetic_blob_"+args[1].split(":",1)[1]+"\n").encode()
        if "commit" in args:
            if "--only" not in args:
                raise AssertionError("Only an explicit scoped commit is permitted")
            self.committed,self.head = True,self.commit_sha
            self.staged = {"unrelated_after_commit.txt"} if self.concurrent_after_commit else set()
            self.modified.clear()
            return b"synthetic commit only; no Git was run\n"
        if args and args[0] == "diff-tree":
            return self.encoded(set(self.paths) | ({self.commit_scope_extra} if self.commit_scope_extra else set()))
        if args and args[0] == "push":
            return b"synthetic push only; no network or Git was run\n"
        raise AssertionError("Unexpected mocked Git command: "+repr(args))


def load_engine():
    spec = importlib.util.spec_from_file_location("pure_j_finisher_under_mock_tests",SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PureJFinisherFileAndGitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine=load_engine()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="v4-finisher-tiny-repo-")
        self.addCleanup(self.tmp.cleanup);self.repo=Path(self.tmp.name)
        self.run_root=self.repo/getattr(self.engine,"RUN_REL",Path("reports/pyscf-spinosin-pure-j-20261101-v4"))
        self.run_root.mkdir(parents=True)
        self.doc=self.repo/getattr(self.engine,"DOC_REL",Path("docs/PYSCF_SPINOSIN_PURE_J_RESULTS_20261101.md"))
        self.doc.parent.mkdir(parents=True)
        self.doc.write_text("Synthetic finisher dispatch document, not a scientific report.\n")
        self.ordinary=self.run_root/"synthetic-result.json"
        self.ordinary.write_text('{"synthetic_not_scientific_data":true}\n')
        patcher=patch.object(self.engine,"ROOT",self.repo);patcher.start();self.addCleanup(patcher.stop)
        patcher=patch.object(self.engine.subprocess,"run",side_effect=AssertionError("No real Git, analyzer or subprocess is allowed"))
        patcher.start();self.addCleanup(patcher.stop)
        self.deadline=datetime.now(timezone.utc)+timedelta(hours=1)
        self.permit={"allow_git_push":True,"_run_root":self.run_root,"_document_path":self.doc,
                     "_deadline":self.deadline,"expected_branch":BRANCH,"expected_code_commit":"a"*40}

    def git(self,**kwargs):
        return GitExecutorFixture([str(p.relative_to(self.repo)) for p in (self.ordinary,self.doc)],**kwargs)

    def publish(self,git,callback=None):
        return self.engine.publish_results(self.permit,self.deadline,delivery_callback=callback,executor=git)

    @staticmethod
    def assert_no_mutating_git(testcase,git):
        commands=[c["arguments"] for c in git.calls]
        testcase.assertFalse(any(c and (c[0] in ("add","reset","push","checkout","switch","merge","clean")
                              or "commit" in c) for c in commands),commands)

    def test_only_V4_and_new_document_candidates_are_allowed_and_hashed(self):
        records=self.engine.candidate_delivery_files(self.run_root,self.doc,[self.ordinary,self.doc,self.ordinary])
        self.assertEqual({r["path"] for r in records},{str(p.relative_to(self.repo)) for p in (self.ordinary,self.doc)})
        self.assertEqual(len(records),2)
        for r in records:self.assertEqual(r["sha256"],self.engine.sha(self.repo/r["path"]))

    def test_outside_old_report_delivery_output_deleted_and_symlink_files_rejected(self):
        outside=self.repo/"other.txt";outside.write_text("synthetic unrelated work")
        old=self.repo/"reports/pyscf-spinosin-continuation-20261031-v3/old.json"
        old.parent.mkdir(parents=True);old.write_text("{}")
        delivery=self.repo/"outputs/synthetic-delivery.json";delivery.parent.mkdir();delivery.write_text("{}")
        symlink=self.run_root/"symlink.json";symlink.symlink_to(self.ordinary)
        for p in (outside,old,delivery,symlink,self.run_root/"deleted.json"):
            with self.assertRaises((ValueError,FileNotFoundError)):
                self.engine.candidate_delivery_files(self.run_root,self.doc,[p])

    def test_size_caps_use_tiny_actual_files_without_reading_giant_artifacts(self):
        first=self.run_root/"size-a.txt";second=self.run_root/"size-b.txt"
        first.write_bytes(b"1"*9);second.write_bytes(b"2"*7)
        with patch.object(self.engine,"MAX_FILE_BYTES",8):
            with self.assertRaises(ValueError):self.engine.candidate_delivery_files(self.run_root,self.doc,[first])
        with patch.object(self.engine,"MAX_FILE_BYTES",10),patch.object(self.engine,"MAX_TOTAL_BYTES",15):
            with self.assertRaises(ValueError):self.engine.candidate_delivery_files(self.run_root,self.doc,[first,second])

    def checkpoint_fixture(self):
        path=self.run_root/self.engine.TARGET_PHASE/"calculation.chk"
        path.parent.mkdir();path.write_bytes(b"Synthetic opaque checkpoint bytes; never HDF parsed.")
        binding={"path":str(path),"checkpoint_present":True,"checkpoint_sha256":self.engine.sha(path),
            "source_result_sha256":"1"*64,"closed_ledger_sha256":"2"*64,
            "closed_evidence_verified":True,"role":"SCF_WARMSTART_ONLY_NOT_HESSIAN_RESTORATION"}
        return path,binding

    def test_one_small_closed_SCF_checkpoint_has_hash_and_explicit_warmstart_only_evidence(self):
        path,binding=self.checkpoint_fixture()
        records=self.engine.candidate_delivery_files(self.run_root,self.doc,[path],checkpoint_binding=binding)
        self.assertEqual(len(records),1)
        self.assertEqual(records[0]["sha256"],binding["checkpoint_sha256"])
        self.assertEqual(records[0]["checkpoint_evidence"],binding)
        self.assertEqual(records[0]["path"],str(path.relative_to(self.repo)))

    def test_checkpoint_missing_or_wrong_closed_binding_and_modified_bytes_are_rejected(self):
        path,binding=self.checkpoint_fixture()
        for changes in (None,{"checkpoint_present":False},{"closed_evidence_verified":False},
                {"role":"HESSIAN_RESTORED"},{"path":str(self.run_root/"other.chk")},
                {"source_result_sha256":"bad"},{"closed_ledger_sha256":"bad"},
                {"checkpoint_sha256":"0"*64}):
            candidate=None if changes is None else dict(binding,**changes)
            with self.assertRaises(ValueError):
                self.engine.candidate_delivery_files(self.run_root,self.doc,[path],checkpoint_binding=candidate)
        path.write_bytes(path.read_bytes()+b"changed after binding")
        with self.assertRaisesRegex(ValueError,"differs from its actual worker result SHA"):
            self.engine.candidate_delivery_files(self.run_root,self.doc,[path],checkpoint_binding=binding)

    def test_unknown_checkpoint_HDF_cache_and_large_checkpoint_fail_before_reading_bytes(self):
        path,binding=self.checkpoint_fixture()
        other=self.run_root/"other.chk";other.write_bytes(b"opaque synthetic data")
        for candidate in (other,self.run_root/"cache.h5",self.run_root/"cache.hdf5"):
            if not candidate.exists():candidate.write_bytes(b"opaque synthetic data")
            with self.assertRaises(ValueError):
                self.engine.candidate_delivery_files(self.run_root,self.doc,[candidate],checkpoint_binding=binding)
        with patch.object(self.engine,"MAX_CHECKPOINT_BYTES",len(path.read_bytes())-1),\
             patch.object(self.engine,"sha",side_effect=AssertionError("Oversize checkpoint must not be opened")):
            with self.assertRaisesRegex(ValueError,"exceeds16MiB"):
                self.engine.candidate_delivery_files(self.run_root,self.doc,[path],checkpoint_binding=binding)

    def test_publish_checks_checkpoint_closure_only_when_checkpoint_is_a_candidate(self):
        with patch.object(self.engine,"closed_checkpoint_binding",side_effect=AssertionError("No checkpoint candidate")):
            self.assertEqual(self.publish(self.git())["status"],"PUSHED")
        path,binding=self.checkpoint_fixture()
        names=[str(p.relative_to(self.repo)) for p in (self.ordinary,self.doc,path)]
        with patch.object(self.engine,"closed_checkpoint_binding",return_value=binding) as closed:
            result=self.publish(GitExecutorFixture(names))
        closed.assert_called_once_with(self.permit)
        self.assertEqual(result["status"],"PUSHED")

    def test_missing_push_permission_does_not_execute_any_git(self):
        self.permit["allow_git_push"]=False;git=self.git()
        with self.assertRaises(ValueError):self.publish(git)
        self.assertEqual(git.calls,[])

    def test_existing_mixed_index_stops_without_reset_or_stage_mutation(self):
        git=self.git(initial_staged=["unrelated_staged.py",str(self.doc.relative_to(self.repo))])
        with self.assertRaises(RuntimeError):self.publish(git)
        self.assert_no_mutating_git(self,git)
        self.assertEqual(git.staged,{"unrelated_staged.py",str(self.doc.relative_to(self.repo))})

    def test_global_unrelated_modified_untracked_or_mixed_work_blocks_publish(self):
        for changes in ({"modified":["scripts/unrelated.py"]},
                        {"unrelated_untracked":["unrelated-not-for-paper.txt"]},
                        {"modified":["scripts/unrelated.py"],"unrelated_untracked":["unrelated-not-for-paper.txt"]}):
            git=self.git(**changes)
            with self.assertRaises(RuntimeError):self.publish(git)
            self.assert_no_mutating_git(self,git)

    def test_wrong_origin_or_branch_and_branch_race_stop_before_add(self):
        for changes in ({"origin":"https://github.com/other/repo.git"},
                        {"branches":["main"]},{"branches":[BRANCH,"main"]}):
            git=self.git(**changes)
            with self.assertRaises(RuntimeError):self.publish(git)
            self.assert_no_mutating_git(self,git)

    def test_mock_publish_scoped_commit_fixed_both_emails_and_exact_branch_SHA_push(self):
        git=self.git();deliveries=[]
        with patch.dict(os.environ,{"GIT_AUTHOR_EMAIL":"untrusted-author@example.invalid",
                                   "GIT_COMMITTER_EMAIL":"untrusted-committer@example.invalid"}):
            result=self.publish(git,lambda value:deliveries.append(dict(value)))
        self.assertEqual(result["status"],"PUSHED")
        commits=[c for c in git.calls if "commit" in c["arguments"]]
        self.assertEqual(len(commits),1)
        args=commits[0]["arguments"]
        self.assertIn("--only",args)
        self.assertEqual(set(args[args.index("--")+1:]),set(git.paths))
        self.assertEqual(commits[0]["kwargs"]["env"]["GIT_AUTHOR_EMAIL"],FIXED_EMAIL)
        self.assertEqual(commits[0]["kwargs"]["env"]["GIT_COMMITTER_EMAIL"],FIXED_EMAIL)
        pushes=[c["arguments"] for c in git.calls if c["arguments"] and c["arguments"][0]=="push"]
        self.assertEqual(pushes,[["push","--porcelain","origin",result["commit_sha"]+":refs/heads/"+BRANCH]])
        self.assertEqual(deliveries[-1]["status"],"PUSHED")
        self.assertFalse(any(any(arg.startswith("--force") for arg in c["arguments"]) for c in git.calls))
        self.assertFalse(any(c["arguments"] and c["arguments"][0] in ("reset","merge","checkout","switch","clean") for c in git.calls))

    def test_index_or_actual_commit_blob_mismatch_stops_before_push(self):
        for changes in ({"index_blob_mismatch":True},{"commit_blob_mismatch":True},
                        {"commit_scope_extra":"outside_commit_scope.py"}):
            git=self.git(**changes)
            with self.assertRaises(RuntimeError):self.publish(git)
            self.assertFalse(any(c["arguments"] and c["arguments"][0] in ("push","reset") for c in git.calls))

    def test_commit_parent_SHA_and_late_origin_races_block_push(self):
        for changes in ({"commit_parent_mismatch":True},{"commit_sha":"not-a-valid-commit"},
                        {"changed_origin_after_commit":"https://github.com/other/changed.git"}):
            git=self.git(**changes)
            with self.assertRaises(RuntimeError):self.publish(git)
            self.assertTrue(git.committed)
            self.assertFalse(any(c["arguments"] and c["arguments"][0] in ("push","reset") for c in git.calls))

    def test_concurrent_staged_work_after_scoped_commit_preserved_and_not_pushed(self):
        git=self.git(concurrent_after_commit=True)
        with self.assertRaises(RuntimeError):self.publish(git)
        self.assertTrue(git.committed)
        self.assertEqual(git.staged,{"unrelated_after_commit.txt"})
        self.assertFalse(any(c["arguments"] and c["arguments"][0] in ("push","reset") for c in git.calls))

    def test_last_instant_HEAD_race_still_pushes_only_audited_commit_SHA(self):
        git=self.git()
        def callback(value):
            if value.get("push_attempted"):git.head="c"*40
        result=self.publish(git,callback)
        self.assertEqual(git.head,"c"*40)
        self.assertEqual(result["commit_sha"],"b"*40)
        pushes=[c["arguments"] for c in git.calls if c["arguments"] and c["arguments"][0]=="push"]
        self.assertEqual(pushes,[["push","--porcelain","origin","b"*40+":refs/heads/"+BRANCH]])


class PureJFinisherPermitAndClosureTests(unittest.TestCase):
    """All source copies, permits, ledgers and locks are temporary test fixtures."""
    @classmethod
    def setUpClass(cls):
        cls.engine=load_engine()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="v4-finisher-metadata-fixture-")
        self.addCleanup(self.tmp.cleanup);self.repo=Path(self.tmp.name)
        patcher=patch.object(self.engine,"ROOT",self.repo);patcher.start();self.addCleanup(patcher.stop)
        patcher=patch.object(self.engine.subprocess,"run",side_effect=AssertionError("No real subprocess/Git/analyzer/QM in pure tests"))
        patcher.start();self.addCleanup(patcher.stop)
        start=self.engine.parse_time(self.engine.TARGET_STARTED_UTC)
        self.now=max(datetime.now(timezone.utc),start+timedelta(hours=1))
        patcher=patch.object(self.engine,"utc",side_effect=lambda:self.now);patcher.start();self.addCleanup(patcher.stop)
        self.run_root=self.repo/self.engine.RUN_REL;self.run_root.mkdir(parents=True)
        self.doc=self.repo/self.engine.DOC_REL;self.doc.parent.mkdir(parents=True)
        self.sources=(self.engine.FINISHER_REL,self.engine.FINISHER_TEST_REL,self.engine.ANALYZER_REL,
            self.engine.ANALYZER_TEST_REL,self.engine.SUPERVISOR_REL,self.engine.ENTRY_REL,self.engine.V8_REL)
        for relative in self.sources:
            target=self.repo/relative;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/relative).read_bytes())
        self.old_root=self.repo/"reports/synthetic-closed-prior-v3";self.old_root.mkdir(parents=True)
        self.prior_path=self.old_root/"cpu_budget.json"
        self.prior={"synthetic_non_scientific_fixture":True,"phases":[{
            "phase":"synthetic_prior","status":"EXITED","ended_at_utc":(start-timedelta(minutes=2)).isoformat()}]}
        self.prior_path.write_text(json.dumps(self.prior))
        self.authorization_path=self.run_root/"authorization.json"
        self.authorization={"synthetic_non_scientific_fixture":True,"prior_closed_ledger_path":str(self.prior_path),
            "prior_closed_ledger_sha256":self.engine.sha(self.prior_path)}
        self.authorization_path.write_text(json.dumps(self.authorization))
        self.execution=self.run_root/"execution";self.execution.mkdir()
        self.proof_path=self.execution/(self.engine.TARGET_PHASE+".log.supervisor-proof.json")
        self.proof={"synthetic_no_execution_fixture":True,"schema":"research4-pyscf-actual-cpu-supervisor-proof-v4",
            "phase":self.engine.TARGET_PHASE,"child_pid":self.engine.TARGET_PID,
            "supervisor_pid":self.engine.TARGET_SUPERVISOR_PID}
        self.proof_path.write_text(json.dumps(self.proof))
        self.deployed_proof_pin=self.engine.TARGET_PROOF_SHA256
        self.deployed_lock_pin=self.engine.EXECUTION_LOCK_SHA256
        patcher=patch.object(self.engine,"TARGET_PROOF_SHA256",self.engine.sha(self.proof_path))
        patcher.start();self.addCleanup(patcher.stop)
        json_input=self.run_root/"synthetic_input.json";json_input.write_text('{"synthetic_input_only":true}')
        self.binary_input=self.run_root/"DO_NOT_READ_synthetic.npy"
        self.binary_input.write_bytes(b"Not an NPY array; passive observer must never read this input.")
        phase_commands={name:[str(self.repo/".runtime/bin/python"),str(self.repo/self.engine.ENTRY_REL),
            "--phase","spinosin-frequency" if name==self.engine.TARGET_PHASE else "pure-j-equivalence",
            "--synthetic-fixture",name] for name in self.engine.PHASES}
        self.lock={"synthetic_no_execution_fixture":True,"status":"SOURCE_AND_COMMANDS_FROZEN_BEFORE_QUANTUM",
            "environment_overrides":{"PYTHONPATH":"synthetic-locked-sites-never-executed",
                "OMP_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"4","MKL_NUM_THREADS":"4",
                "NUMEXPR_NUM_THREADS":"4"},
            "phases":{name:{"payload_argv":phase_commands[name]} for name in self.engine.PHASES},
            "source_sha256":{str(p):self.engine.sha(self.repo/p) for p in
                (self.engine.ENTRY_REL,self.engine.SUPERVISOR_REL,self.engine.V8_REL)},
            "input_sha256":{str(json_input.relative_to(self.repo)):self.engine.sha(json_input),
                str(self.binary_input.relative_to(self.repo)):"0"*64}}
        self.lock_path=self.run_root/"execution_lock.json";self.lock_path.write_text(json.dumps(self.lock))
        patcher=patch.object(self.engine,"EXECUTION_LOCK_SHA256",self.engine.sha(self.lock_path))
        patcher.start();self.addCleanup(patcher.stop)
        self.rows=[]
        for index,name in enumerate(self.engine.PHASES):
            target=name==self.engine.TARGET_PHASE
            self.rows.append({"phase":name,"status":"EXITED","exit_code":0,
                "pid":self.engine.TARGET_PID if target else 100+index,
                "started_at_utc":self.engine.TARGET_STARTED_UTC if target else (start-timedelta(minutes=2)).isoformat(),
                "ended_at_utc":(start+timedelta(minutes=1) if target else start-timedelta(seconds=40-index*10)).isoformat(),
                "command":phase_commands[name],"synthetic_no_execution_fixture":True,
                **({"supervisor_proof_sha256":self.engine.sha(self.proof_path),
                    "supervisor_proof_path":str(self.proof_path)} if target else {})})
        self.ledger_path=self.run_root/"cpu_budget.json"
        self.ledger={"synthetic_no_execution_fixture":True,"schema":"research4-pyscf-actual-cpu-budget-v4",
            "runner_sha256":self.engine.sha(self.repo/self.engine.SUPERVISOR_REL),
            "authorization_path":str(self.authorization_path),"authorization_sha256":self.engine.sha(self.authorization_path),
            "run_deadline_at_utc":(self.now+timedelta(hours=1)).isoformat(),"phases":self.rows}
        self.write_ledger()
        self.completion_path=self.execution/(self.engine.TARGET_PHASE+".supervisor-completion.json")
        self.completion={"synthetic_no_execution_fixture":True,"status":"SUPERVISOR_RETURNED",
            "exit_code":0,"utc":(start+timedelta(minutes=1,seconds=1)).isoformat()}
        self.write_completion()
        self.permit_path=self.run_root/"synthetic_root_permit.json"
        self.permit={"schema":self.engine.SCHEMA,"created_at_utc":self.now.isoformat(),
            "allow_conditional_finalization":True,"allow_git_push":False,
            "run_root":str(self.run_root),"ledger_path":str(self.ledger_path),
            "analysis_output_path":str(self.run_root/"final-analysis"),"documentation_path":str(self.doc),
            "root_supervisor_completion_path":str(self.completion_path),
            "expected_branch":BRANCH,"expected_code_commit":"a"*40,
            "poll_interval_seconds":15,"max_poll_iterations":3,
            "target_phase":self.engine.TARGET_PHASE,"target_pid":self.engine.TARGET_PID,
            "target_supervisor_pid":self.engine.TARGET_SUPERVISOR_PID,
            "target_started_at_utc":self.engine.TARGET_STARTED_UTC,"target_proof_sha256":self.engine.sha(self.proof_path),
            "execution_lock_sha256":self.engine.sha(self.lock_path),
            "source_sha256":{str(p):self.engine.sha(self.repo/p) for p in self.sources},
            "authorization_sha256":self.engine.sha(self.authorization_path),
            "prior_closed_ledger_path":str(self.prior_path),"prior_closed_ledger_sha256":self.engine.sha(self.prior_path),
            "run_deadline_at_utc":self.ledger["run_deadline_at_utc"],
            "allow_disclosed_budget_violation_analysis":False}
        self.write_permit()

    def write_ledger(self):self.ledger_path.write_text(json.dumps(self.ledger))
    def write_permit(self):self.permit_path.write_text(json.dumps(self.permit))
    def write_completion(self):self.completion_path.write_text(json.dumps(self.completion))
    def validated(self,push=False):
        return self.engine.validate_permit(self.permit_path,self.engine.sha(self.permit_path),push=push)

    def run_args(self,name="synthetic-delivery",push=False):
        return argparse.Namespace(permit=self.permit_path,permit_sha256=self.engine.sha(self.permit_path),
            delivery_dir=self.repo/"outputs"/name,push=push)

    @staticmethod
    def summary_fixture(**changes):
        """Only dispatch-contract metadata; these zero counts are not QM results."""
        value={"analysis_only":True,"new_SCF_gradient_Hessian_evaluations":0,
            "IR_intensities_status":"NOT_COMPUTED","chemical_attribution_confirmed":False,
            "classification_modified":False,"cache_restoration":"NOT_IMPLEMENTED_SAVE_ONLY",
            "Spinosin_frequency_status":"SYNTHETIC_DISPATCH_TEST_NO_QUANTUM_RESULTS",
            "computed_Spinosin_frequency_rows_including_unvalidated":0,
            "validated_spinosin_frequency_rows":0,"validated_spinosin_low_band_modes":0,
            "failed_or_partial_phase_count":0}
        return dict(value,**changes)

    def successful_analysis_fixture(self,command,**kwargs):
        """Mock executor creates only temporary synthetic JSON/doc, never runs code."""
        self.assertEqual(command[1],str(self.repo/self.engine.ANALYZER_REL))
        self.assertEqual(kwargs["cwd"],self.repo)
        self.assertTrue(kwargs["close_fds"])
        for key,value in self.lock["environment_overrides"].items():
            self.assertEqual(kwargs["env"][key],value)
        for path in (self.ledger_path.with_suffix(".lock"),self.repo/".runtime/pyscf-v4-global.lock",
                     self.prior_path.with_suffix(".lock")):
            with path.open("a") as handle:
                with self.assertRaises(BlockingIOError):
                    self.engine.fcntl.flock(handle,self.engine.fcntl.LOCK_EX|self.engine.fcntl.LOCK_NB)
        output=Path(command[command.index("--output")+1]);output.mkdir()
        (output/"summary.json").write_text(json.dumps(self.summary_fixture()))
        self.doc.write_text("Synthetic dispatch test only; no scientific calculation.\n")
        return subprocess.CompletedProcess(command,0,stdout=b"synthetic analyzer executor; no QM\n")

    def test_real_deployment_phase_and_full_pins_are_not_templates_or_prefixes(self):
        self.assertEqual(self.deployed_lock_pin,"3af7acecce03964892c91fcc5d32d9ccd9e5c7abfd29e84b1fc1595bd96a9dab")
        self.assertEqual(self.deployed_proof_pin,"250007306069747141d82f5085706097349beb12d9a5052cbf1c24a69ea54afd")
        self.assertEqual(self.engine.TARGET_PID,18565)
        self.assertEqual(self.engine.TARGET_STARTED_UTC,"2026-11-01T17:06:14.692572+00:00")

    def test_root_permit_binds_seven_sources_lock_auth_prior_paths_and_finite_deadline(self):
        permit=self.validated()
        self.assertEqual(permit["_permit_sha256"],self.engine.sha(self.permit_path))
        self.assertEqual(permit["_run_root"],self.run_root)
        self.assertEqual(permit["_proof_path"],self.proof_path)
        self.assertEqual(permit["_completion_path"],self.completion_path)
        self.assertEqual(permit["_prior_ledger_path"],self.prior_path)
        self.assertEqual(permit["_deadline"],self.engine.parse_time(self.ledger["run_deadline_at_utc"]))
        self.assertEqual(set(permit["source_sha256"]),set(map(str,self.sources)))

    def test_template_missing_root_permission_or_push_flag_cannot_authorize_work(self):
        for field,value in (("schema","TEMPLATE_NOT_AUTHORIZATION"),("allow_conditional_finalization",False)):
            original=self.permit[field];self.permit[field]=value;self.write_permit()
            with self.assertRaises(ValueError):self.validated()
            self.permit[field]=original
        self.write_permit()
        with self.assertRaises(ValueError):self.validated(push=True)

    def test_permit_source_and_lock_byte_changes_are_rejected_without_rewriting_them(self):
        expected=self.engine.sha(self.permit_path);self.permit_path.write_text(self.permit_path.read_text()+" ")
        with self.assertRaises(ValueError):self.engine.validate_permit(self.permit_path,expected)
        self.write_permit()
        source=self.repo/self.engine.FINISHER_TEST_REL;source.write_text(source.read_text()+"\n# changed synthetic source\n")
        with self.assertRaises(ValueError):self.validated()
        self.permit["source_sha256"][str(self.engine.FINISHER_TEST_REL)]=self.engine.sha(source);self.write_permit()
        self.validated()
        self.lock_path.write_text(self.lock_path.read_text()+" ")
        with self.assertRaises(ValueError):self.validated()

    def test_any_missing_required_source_or_exact_target_identity_change_is_rejected(self):
        for relative in self.sources:
            saved=self.permit["source_sha256"].pop(str(relative));self.write_permit()
            with self.assertRaises(ValueError):self.validated()
            self.permit["source_sha256"][str(relative)]=saved
        for key in ("target_phase","target_pid","target_supervisor_pid","target_started_at_utc","target_proof_sha256"):
            saved=self.permit[key];self.permit[key]="different";self.write_permit()
            with self.assertRaises(ValueError):self.validated()
            self.permit[key]=saved

    def test_deadline_poll_bounds_scope_and_symlink_cannot_be_loosened_by_permit(self):
        for key,value in (("run_deadline_at_utc",(self.now+timedelta(hours=2)).isoformat()),
                ("max_poll_iterations",0),("max_poll_iterations",self.engine.MAX_ITERATIONS+1),
                ("poll_interval_seconds",1),("documentation_path",str(self.repo/"docs/OTHER.md"))):
            saved=self.permit[key];self.permit[key]=value;self.write_permit()
            with self.assertRaises(ValueError):self.validated()
            self.permit[key]=saved
        self.write_permit();self.doc.symlink_to(self.repo/"missing_external_document.md")
        with self.assertRaisesRegex(ValueError,"symlink"):self.validated()

    def test_passive_metadata_validation_never_hashes_or_opens_NPY_inputs(self):
        original=self.engine.sha
        def safe_sha(path):
            if Path(path).suffix.lower() in (".npy",".chk",".h5",".hdf5"):
                raise AssertionError("Passive observer must not read molecular arrays/checkpoints")
            return original(path)
        with patch.object(self.engine,"sha",side_effect=safe_sha):
            permit=self.validated()
            self.assertIsNotNone(self.engine.closed_stage_snapshot(permit))

    def test_auth_prior_source_and_frozen_payload_commands_are_rechecked(self):
        permit=self.validated();self.rows[-1]["command"]=["unreviewed_payload"]
        self.write_ledger()
        with self.assertRaisesRegex(ValueError,"payload differs"):
            self.engine.closed_stage_snapshot(permit)
        self.rows[-1]["command"]=self.lock["phases"][self.engine.TARGET_PHASE]["payload_argv"]
        self.write_ledger();self.prior_path.write_text(self.prior_path.read_text()+" ")
        with self.assertRaisesRegex(ValueError,"historical V3 ledger"):
            self.engine.closed_stage_snapshot(permit)

    def test_stage_gap_after_only_water_or_ethanol_never_releases_even_without_running_row(self):
        permit=self.validated()
        for count in (0,1,2):
            self.ledger["phases"]=self.rows[:count];self.write_ledger()
            self.assertIsNone(self.engine.closed_stage_snapshot(permit))

    def test_target_or_other_running_missing_end_or_missing_wrapper_never_closes(self):
        permit=self.validated()
        for index in (0,2):
            saved=self.rows[index]["status"];self.rows[index]["status"]="RUNNING";self.write_ledger()
            self.assertIsNone(self.engine.closed_stage_snapshot(permit));self.rows[index]["status"]=saved
        saved=self.rows[-1].pop("ended_at_utc");self.write_ledger()
        self.assertIsNone(self.engine.closed_stage_snapshot(permit));self.rows[-1]["ended_at_utc"]=saved;self.write_ledger()
        self.completion_path.unlink()
        self.assertIsNone(self.engine.closed_stage_snapshot(permit))

    def test_duplicate_unknown_or_reordered_phase_and_wrong_target_are_rejected(self):
        permit=self.validated()
        for rows in (self.rows+[dict(self.rows[-1])],[self.rows[1],self.rows[0],self.rows[2]],
                     [dict(self.rows[0],phase="unreviewed"),*self.rows[1:]]):
            self.ledger["phases"]=rows;self.write_ledger()
            with self.assertRaises(ValueError):self.engine.closed_stage_snapshot(permit)
        self.ledger["phases"]=self.rows
        for key,value in (("pid",1),("started_at_utc","2026-11-01T00:00:00+00:00")):
            saved=self.rows[-1][key];self.rows[-1][key]=value;self.write_ledger()
            with self.assertRaises(ValueError):self.engine.closed_stage_snapshot(permit)
            self.rows[-1][key]=saved

    def test_wrong_target_proof_bytes_or_process_identity_cannot_close(self):
        permit=self.validated();self.proof_path.write_text(self.proof_path.read_text()+" ")
        with self.assertRaisesRegex(ValueError,"proof bytes"):
            self.engine.closed_stage_snapshot(permit)

    def test_wrapper_and_payload_exit_codes_are_separate_for_real_failed_analysis_release(self):
        permit=self.validated()
        self.rows[-1].update(status="PARTIAL_OR_FAILED",exit_code=2);self.write_ledger()
        self.completion["exit_code"]=0;self.write_completion()
        with self.assertRaisesRegex(ValueError,"supervisor return semantics"):
            self.engine.closed_stage_snapshot(permit)
        self.completion["exit_code"]=1;self.write_completion()
        snapshot=self.engine.closed_stage_snapshot(permit)
        self.assertEqual(snapshot["target_payload_exit_code"],2)
        self.assertEqual(snapshot["supervisor_wrapper_exit_code"],1)

    def test_future_early_or_prestart_terminal_timestamps_are_rejected(self):
        permit=self.validated();original=self.completion["utc"]
        for timestamp in (self.now+timedelta(seconds=1),self.engine.parse_time(self.rows[-1]["ended_at_utc"])-timedelta(seconds=1)):
            self.completion["utc"]=timestamp.isoformat();self.write_completion()
            with self.assertRaises(ValueError):self.engine.closed_stage_snapshot(permit)
        self.completion["utc"]=original;self.write_completion()
        self.rows[-1]["ended_at_utc"]=(self.engine.parse_time(self.engine.TARGET_STARTED_UTC)-timedelta(seconds=1)).isoformat()
        self.write_ledger()
        with self.assertRaises(ValueError):self.engine.closed_stage_snapshot(permit)

    def test_successful_closed_snapshot_and_release_bind_exact_final_JSONs_and_sources(self):
        permit=self.validated();snapshot=self.engine.closed_stage_snapshot(permit)
        for key,path in (("ledger_sha256",self.ledger_path),("execution_lock_sha256",self.lock_path),
                         ("target_proof_sha256",self.proof_path),("root_completion_sha256",self.completion_path)):
            self.assertEqual(snapshot[key],self.engine.sha(path))
        release=self.engine.build_release(permit,snapshot)
        self.assertTrue(release["allow_analysis_export"])
        self.assertTrue(release["scientific_results_are_not_created_by_this_release"])
        self.assertEqual(release["root_permit_sha256"],self.engine.sha(self.permit_path))
        self.assertEqual(release["analyzer_source_sha256"],self.engine.sha(self.repo/self.engine.ANALYZER_REL))
        self.assertIn("ANALYZER_DOES_NOT_REACQUIRE",release["locking_contract"])

    def test_disclosed_budget_violation_requires_explicit_root_permission(self):
        permit=self.validated();self.rows[0]["status"]="BUDGET_VIOLATION_REQUIRES_REVIEW";self.write_ledger()
        with self.assertRaisesRegex(ValueError,"Budget violation"):
            self.engine.closed_stage_snapshot(permit)
        permit["allow_disclosed_budget_violation_analysis"]=True
        self.assertTrue(self.engine.closed_stage_snapshot(permit)["budget_violation_disclosed"])

    def test_polling_is_finite_deadline_bound_and_stage_gap_waits_for_real_marker(self):
        permit=self.validated();self.completion_path.unlink()
        clock=FakeClock(self.now);polls=[]
        with self.assertRaises(TimeoutError):
            self.engine.wait_for_closure(permit,clock,clock.sleep,lambda *a:polls.append(a))
        self.assertEqual(len(polls),3);self.assertEqual(clock.sleeps,[15,15])
        clock=FakeClock(permit["_deadline"])
        with self.assertRaises(TimeoutError):self.engine.wait_for_closure(permit,clock,clock.sleep)
        self.assertEqual(clock.sleeps,[])
        clock=FakeClock(self.now)
        def sleep(seconds):clock.sleep(seconds);self.write_completion()
        snapshot=self.engine.wait_for_closure(permit,clock,sleep)
        self.assertEqual(clock.sleeps,[15]);self.assertIsNotNone(snapshot)

    def test_partial_completion_write_stays_pending_then_valid_marker_closes(self):
        permit=self.validated()
        for content in ("",'{"status":"SUPERVISOR_RETURNED",'):
            self.completion_path.write_text(content)
            self.assertIsNone(self.engine.closed_stage_snapshot(permit))
        clock=FakeClock(self.now);polls=[]
        def sleep(seconds):clock.sleep(seconds);self.write_completion()
        snapshot=self.engine.wait_for_closure(permit,clock,sleep,lambda *a:polls.append(a))
        self.assertEqual(polls,[(1,False),(2,True)])
        self.assertIsNotNone(snapshot)
        self.completion_path.write_text('{}')
        with self.assertRaises(ValueError):self.engine.closed_stage_snapshot(permit)

    def closed_checkpoint_result_fixture(self):
        path=self.run_root/self.engine.TARGET_PHASE/"result.json";path.parent.mkdir()
        result={"identity":"spinosin","checkpoint_present":True,"checkpoint_sha256":"4"*64,
            "actual_entrypoint":{"sha256":self.engine.ENTRY_SHA256},
            "resource_supervision":{"proof_sha256":self.engine.TARGET_PROOF_SHA256,"proof":dict(self.proof)},
            "completed_at_utc":self.rows[-1]["ended_at_utc"],"synthetic_no_execution_fixture":True}
        path.write_text(json.dumps(result));return path,result

    def test_checkpoint_binding_requires_closed_real_result_source_proof_and_completion(self):
        permit=self.validated();path,result=self.closed_checkpoint_result_fixture()
        binding=self.engine.closed_checkpoint_binding(permit)
        self.assertEqual(binding["checkpoint_sha256"],result["checkpoint_sha256"])
        self.assertEqual(binding["source_result_sha256"],self.engine.sha(path))
        self.assertEqual(binding["closed_ledger_sha256"],self.engine.sha(self.ledger_path))
        self.assertEqual(binding["role"],"SCF_WARMSTART_ONLY_NOT_HESSIAN_RESTORATION")
        self.assertTrue(binding["closed_evidence_verified"])
        mutations=({"identity":"ethanol"},{"checkpoint_present":False},{"checkpoint_sha256":"bad"},
            {"actual_entrypoint":{"sha256":"0"*64}},
            {"resource_supervision":{"proof_sha256":"0"*64,"proof":dict(self.proof)}},
            {"resource_supervision":{"proof_sha256":self.engine.TARGET_PROOF_SHA256,"proof":{}}},
            {"completed_at_utc":(self.engine.parse_time(self.rows[-1]["ended_at_utc"])+timedelta(seconds=1)).isoformat()},
            {"completed_at_utc":(self.engine.parse_time(self.engine.TARGET_STARTED_UTC)-timedelta(seconds=1)).isoformat()})
        for changes in mutations:
            path.write_text(json.dumps(dict(result,**changes)))
            with self.assertRaises(ValueError):self.engine.closed_checkpoint_binding(permit)

    def test_checkpoint_binding_before_closure_never_opens_worker_result_or_binary(self):
        permit=self.validated();path,_=self.closed_checkpoint_result_fixture()
        original=self.engine.read
        def forbid_result_read(candidate):
            if Path(candidate)==path:raise AssertionError("Active worker result must never be inspected")
            return original(candidate)
        for incomplete in ("RUNNING","MISSING_WRAPPER","PARTIAL_WRAPPER_WRITE"):
            if incomplete=="RUNNING":self.rows[-1]["status"]="RUNNING";self.write_ledger()
            else:
                self.rows[-1]["status"]="EXITED";self.write_ledger()
                if incomplete=="MISSING_WRAPPER":self.completion_path.unlink()
                else:self.completion_path.write_text('{')
            with patch.object(self.engine,"read",side_effect=forbid_result_read):
                with self.assertRaisesRegex(ValueError,"before exact target/wrapper closure"):
                    self.engine.closed_checkpoint_binding(permit)

    def test_corrupt_ledger_and_lock_do_not_get_completion_marker_retry_exception(self):
        permit=self.validated();ledger_bytes=self.ledger_path.read_bytes()
        self.ledger_path.write_text('{')
        with self.assertRaises(json.JSONDecodeError):self.engine.closed_stage_snapshot(permit)
        self.ledger_path.write_bytes(ledger_bytes);self.lock_path.write_text('{')
        with self.assertRaises(json.JSONDecodeError):self.engine.closed_stage_snapshot(permit)

    def test_run_holds_three_temporary_locks_and_uses_frozen_environment_once(self):
        args=self.run_args();analysis=Mock(side_effect=self.successful_analysis_fixture)
        git=GitExecutorFixture([])
        ambient={key:"untrusted-ambient-value" for key in self.lock["environment_overrides"]}
        with patch.dict(os.environ,ambient):
            result=self.engine.run(args,analysis,git,FakeClock(self.now),lambda _:self.fail("Closed fixture must not sleep"))
        self.assertEqual(result["status"],"SCIENTIFIC_EXPORT_DONE")
        self.assertEqual(result["analysis_attempts"],1);analysis.assert_called_once()
        self.assertEqual(result["new_quantum_jobs"],0)
        self.assertEqual(result["active_molecular_array_reads"],0)
        self.assertEqual(result["Git_delivery"]["status"],"NOT_ATTEMPTED")
        self.assertTrue((args.delivery_dir/"analysis.log").is_file())
        self.assertTrue((args.delivery_dir/"RECOVERY.md").is_file())
        self.assertEqual(json.loads((args.delivery_dir/"delivery.json").read_text())["status"],result["status"])
        self.assertFalse((self.run_root/"delivery.json").exists())
        self.assertFalse(any(c["arguments"][0] in ("add","push","commit","reset","clean") for c in git.calls))
        for path in (self.ledger_path.with_suffix(".lock"),self.repo/".runtime/pyscf-v4-global.lock",
                     self.prior_path.with_suffix(".lock")):
            with path.open("a") as handle:
                self.engine.fcntl.flock(handle,self.engine.fcntl.LOCK_EX|self.engine.fcntl.LOCK_NB)

    def test_failed_analyzer_has_one_attempt_and_release_blocks_blind_retry(self):
        args=self.run_args();analysis=Mock(return_value=subprocess.CompletedProcess([],2,stdout=b"synthetic failure"))
        result=self.engine.run(args,analysis,GitExecutorFixture([]),FakeClock(self.now),lambda _:None)
        self.assertEqual(result["status"],"PARTIAL_FINALIZATION")
        self.assertEqual(result["analysis_attempts"],1);analysis.assert_called_once()
        self.assertIn("no automatic retry",result["stop_reason"])
        self.assertEqual((args.delivery_dir/"analysis.log").read_bytes(),b"synthetic failure")
        self.assertTrue((self.run_root/"final_analysis_release.json").is_file())
        with self.assertRaises(FileExistsError):
            self.engine.run(args,analysis,GitExecutorFixture([]),FakeClock(self.now),lambda _:None)
        second=self.engine.run(self.run_args("new-synthetic-delivery"),analysis,GitExecutorFixture([]),FakeClock(self.now),lambda _:None)
        self.assertEqual(second["status"],"PARTIAL_FINALIZATION")
        self.assertEqual(second["analysis_attempts"],0);analysis.assert_called_once()

    def test_run_cannot_bypass_root_permission_or_global_unrelated_changes(self):
        self.permit["allow_conditional_finalization"]=False;self.write_permit()
        analysis=Mock(side_effect=AssertionError("Analyzer must not run"));git=GitExecutorFixture([])
        result=self.engine.run(self.run_args("bad-permit"),analysis,git,FakeClock(self.now),lambda _:None)
        self.assertEqual(result["status"],"PARTIAL_FINALIZATION");self.assertEqual(git.calls,[])
        self.permit["allow_conditional_finalization"]=True;self.write_permit()
        for index,changes in enumerate(({"modified":{"scripts/unrelated.py"}},
                                       {"unrelated_untracked":{"notes/unrelated.txt"}},
                                       {"initial_staged":{"docs/unrelated.md"}})):
            result=self.engine.run(self.run_args("mixed-state-"+str(index)),analysis,
                GitExecutorFixture([],**changes),FakeClock(self.now),lambda _:None)
            self.assertEqual(result["status"],"PARTIAL_FINALIZATION")
        analysis.assert_not_called();self.assertFalse((self.run_root/"final_analysis_release.json").exists())

    def test_existing_release_output_or_document_is_preserved_without_analysis(self):
        analysis=Mock(side_effect=AssertionError("No automatic rerun"))
        for index,path in enumerate((self.run_root/"final_analysis_release.json",
                self.run_root/"final-analysis",self.doc)):
            path.write_text("preserved synthetic prior output")
            result=self.engine.run(self.run_args("existing-"+str(index)),analysis,
                GitExecutorFixture([]),FakeClock(self.now),lambda _:None)
            self.assertEqual(result["status"],"PARTIAL_FINALIZATION")
            self.assertEqual(path.read_text(),"preserved synthetic prior output")
            path.unlink()
        analysis.assert_not_called()

    def test_metadata_change_between_passive_snapshot_and_lock_recheck_stops_analysis(self):
        actual_context=self.engine.finalization_locks
        @contextmanager
        def race_context(permit):
            with actual_context(permit):
                self.rows[0]["synthetic_concurrent_edit"]=True;self.write_ledger()
                yield
        analysis=Mock(side_effect=AssertionError("Changed evidence must not be analyzed"))
        with patch.object(self.engine,"finalization_locks",side_effect=race_context):
            result=self.engine.run(self.run_args(),analysis,GitExecutorFixture([]),FakeClock(self.now),lambda _:None)
        self.assertEqual(result["status"],"PARTIAL_FINALIZATION")
        self.assertIn("changed before acquiring",result["stop_reason"])
        analysis.assert_not_called()

    def test_partial_wrapper_write_in_run_polls_before_single_analysis(self):
        self.completion_path.write_text('{"status":')
        analysis=Mock(side_effect=self.successful_analysis_fixture);clock=FakeClock(self.now)
        def sleep(seconds):
            analysis.assert_not_called();clock.sleep(seconds);self.write_completion()
        result=self.engine.run(self.run_args(),analysis,GitExecutorFixture([]),clock,sleep)
        self.assertEqual(result["status"],"SCIENTIFIC_EXPORT_DONE")
        self.assertEqual(clock.sleeps,[15]);analysis.assert_called_once()

    def test_inconsistent_scientific_claims_and_noninteger_counts_are_rejected(self):
        for changes in ({"analysis_only":False},{"new_SCF_gradient_Hessian_evaluations":1},
                {"IR_intensities_status":"COMPUTED"},{"chemical_attribution_confirmed":True},
                {"classification_modified":True},{"cache_restoration":"RESUMED"},
                {"computed_Spinosin_frequency_rows_including_unvalidated":True},
                {"failed_or_partial_phase_count":-1},{"validated_spinosin_low_band_modes":1},
                {"validated_spinosin_frequency_rows":1}):
            with self.assertRaises(ValueError):self.engine.validate_summary(self.summary_fixture(**changes))
        summary=self.engine.validate_summary(self.summary_fixture())
        self.assertEqual(summary["Spinosin_frequency_status"],"SYNTHETIC_DISPATCH_TEST_NO_QUANTUM_RESULTS")
        self.assertEqual(summary["qualified_rows"],0)

    def test_zero_new_git_files_is_not_reported_as_a_push(self):
        self.permit["allow_git_push"]=True;self.write_permit()
        publication={"status":"NO_NEW_SCOPED_FILES","push_attempted":False}
        with patch.object(self.engine,"publish_results",return_value=publication) as publish:
            result=self.engine.run(self.run_args(push=True),Mock(side_effect=self.successful_analysis_fixture),
                GitExecutorFixture([]),FakeClock(self.now),lambda _:None)
        publish.assert_called_once()
        self.assertEqual(result["status"],"SCIENTIFIC_EXPORT_DONE_GIT_NO_NEW_FILES")
        self.assertFalse(result["Git_delivery"]["push_attempted"])

    def test_publication_failure_preserves_successful_scientific_export_and_recovery(self):
        self.permit["allow_git_push"]=True;self.write_permit()
        with patch.object(self.engine,"publish_results",side_effect=RuntimeError("synthetic mixed shared index")):
            result=self.engine.run(self.run_args(push=True),Mock(side_effect=self.successful_analysis_fixture),
                GitExecutorFixture([]),FakeClock(self.now),lambda _:None)
        self.assertEqual(result["status"],"SCIENTIFIC_EXPORT_DONE_GIT_BLOCKED_OR_FAILED")
        self.assertEqual(result["Git_delivery"]["status"],"BLOCKED_OR_FAILED_PRESERVE_SHARED_INDEX")
        self.assertTrue(self.doc.is_file())

