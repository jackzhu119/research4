"""Root-permitted one-shot finalization after the exact V4 target really ends.

Passive JSON observation only; never launches quantum work or parses molecular
arrays. The frozen analyzer reads closed arrays after release. Git delivery is
restricted to the new V4 report tree and its single new document.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN_REL = Path("reports/pyscf-spinosin-pure-j-20261101-v4")
DOC_REL = Path("docs/PYSCF_SPINOSIN_PURE_J_RESULTS_20261101.md")
FINISHER_REL = Path("scripts/finalize_pyscf_pure_j_20261101.py")
FINISHER_TEST_REL = Path("tests/test_pyscf_pure_j_finisher.py")
ANALYZER_REL = Path("scripts/analyze_pyscf_pure_j_results_20261101.py")
ANALYZER_TEST_REL = Path("tests/test_pyscf_pure_j_results.py")
SUPERVISOR_REL = Path("scripts/run_pyscf_actual_cpu_budget_20261101.py")
ENTRY_REL = Path("scripts/pyscf_spinosin_pure_j_worker_20261101.py")
V8_REL = Path("scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py")
BRANCH = "research/pyscf-spinosin-mechanism-20261031"
SCHEMA = "research4-pyscf-pure-j-passive-finisher-permit-v1"
PHASES = ("water-pure-j-equivalence-001", "ethanol-pure-j-equivalence-001", "spinosin-pure-j-frequency-001")
TARGET_PHASE = PHASES[-1]
TARGET_PID, TARGET_SUPERVISOR_PID = 18565, 18564
TARGET_STARTED_UTC = "2026-11-01T17:06:14.692572+00:00"
TARGET_PROOF_SHA256 = "250007306069747141d82f5085706097349beb12d9a5052cbf1c24a69ea54afd"
EXECUTION_LOCK_SHA256 = "3af7acecce03964892c91fcc5d32d9ccd9e5c7abfd29e84b1fc1595bd96a9dab"
ENTRY_SHA256 = "e3215e563b4429212084cfccbcec98e6a9f482bd8c3352d53d3064402649b450"
SUPERVISOR_SHA256 = "354ff51b5c2af2810033656d9972948262cfebb4dabb3978fca2d021f97e2b8f"
V8_SHA256 = "acfe7708748ac237c9912aa9b1d98e047d842794fd0c6f08b1a67b993f5b277d"
ANALYZER_SHA256 = "5d9544a11f17a2e45d8b4ffed221da97e99c1aa157c8c7aa59ee1d809f13aa3d"
GITHUB_EMAIL = "195995062+jackzhu119@users.noreply.github.com"
GITHUB_NAME = "jackzhu119"
MAX_FILE_BYTES, MAX_TOTAL_BYTES = 25*1024**2, 150*1024**2
MAX_CHECKPOINT_BYTES = 16*1024**2
POLL_SECONDS, MAX_ITERATIONS = 15, 4321


def utc():
    return datetime.now(timezone.utc)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024**2),b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write("\n")


def resolve(path):
    path=Path(path)
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def parse_time(value):
    if not isinstance(value,str):raise ValueError("Timezone-aware ISO timestamp required")
    result=datetime.fromisoformat(value.replace("Z","+00:00"))
    if result.tzinfo is None:raise ValueError("Timezone-aware ISO timestamp required")
    return result


def safe_path(path,within=None):
    raw=Path(path);raw=raw if raw.is_absolute() else ROOT/raw
    if raw.is_symlink() or any(p.is_symlink() for p in raw.parents if p!=ROOT):
        raise ValueError("Do not traverse symlinked finisher paths")
    actual=raw.resolve()
    if within is not None and not actual.is_relative_to(Path(within).resolve()):
        raise ValueError("Finisher path escapes its fixed scope")
    return actual


def validate_metadata_bindings(run_root,ledger,lock,permit):
    """Read only source/JSON inputs; closed binary input auditing belongs to analyzer."""
    if ledger.get("schema")!="research4-pyscf-actual-cpu-budget-v4":
        raise ValueError("Only actual-CPU V4 ledger may be finalized")
    if ledger.get("runner_sha256")!=sha(ROOT/SUPERVISOR_REL):
        raise ValueError("Ledger supervisor source changed")
    authorization=safe_path(ledger.get("authorization_path",""),ROOT)
    if sha(authorization)!=ledger.get("authorization_sha256") or sha(authorization)!=permit.get("authorization_sha256"):
        raise ValueError("Original V4 authorization bytes changed")
    auth=read(authorization)
    prior=safe_path(auth.get("prior_closed_ledger_path",""),ROOT)
    if (resolve(permit.get("prior_closed_ledger_path",""))!=prior
            or permit.get("prior_closed_ledger_sha256")!=auth.get("prior_closed_ledger_sha256")
            or sha(prior)!=auth.get("prior_closed_ledger_sha256")):
        raise ValueError("Bound historical V3 ledger changed")
    old=read(prior)
    if any(row.get("status")=="RUNNING" or not row.get("ended_at_utc") for row in old.get("phases",[])):
        raise ValueError("Historical ledger is not closed")
    if list(lock.get("phases",{}))!=list(PHASES) or lock.get("status")!="SOURCE_AND_COMMANDS_FROZEN_BEFORE_QUANTUM":
        raise ValueError("Frozen three-phase execution plan differs")
    for name,expected in lock.get("source_sha256",{}).items():
        if sha(safe_path(name,ROOT))!=expected:raise ValueError("Frozen source changed: "+name)
    # No NPY/CHK/HDF/binary input is opened by this observer. Their exact hashes
    # are already predeclared in the immutable lock and rechecked by analyzer
    # after the real closed-only release, before its molecular reads/exports.
    for name,expected in lock.get("input_sha256",{}).items():
        path=safe_path(name,ROOT)
        if path.suffix.lower()==".json" and sha(path)!=expected:
            raise ValueError("Preauthorized JSON input changed: "+name)
    return authorization,auth,prior


def validate_permit(path,expected_sha,push=False):
    path=safe_path(path,ROOT/RUN_REL)
    if not expected_sha or sha(path)!=expected_sha:
        raise ValueError("Explicit root permit SHA must match real immutable bytes")
    value=read(path)
    if value.get("schema")!=SCHEMA or value.get("allow_conditional_finalization") is not True:
        raise ValueError("A real future root permit is required; a template cannot authorize execution")
    parse_time(value.get("created_at_utc"))
    run_root=safe_path(ROOT/RUN_REL,ROOT)
    paths={"run_root":run_root,"ledger_path":run_root/"cpu_budget.json",
        "analysis_output_path":run_root/"final-analysis","documentation_path":ROOT/DOC_REL,
        "root_supervisor_completion_path":run_root/"execution"/(TARGET_PHASE+".supervisor-completion.json")}
    for key,expected in paths.items():
        if safe_path(value.get(key,""),ROOT)!=expected.resolve():raise ValueError("Unreviewed finisher path: "+key)
    if (value.get("expected_branch")!=BRANCH or not re.fullmatch(r"[0-9a-f]{40}",value.get("expected_code_commit",""))
            or type(value.get("allow_git_push")) is not bool):
        raise ValueError("Fixed research branch, code commit and explicit Git flag required")
    if push and not value["allow_git_push"]:raise ValueError("--push requires explicit root permit Git authorization")
    if value.get("poll_interval_seconds")!=POLL_SECONDS or type(value.get("max_poll_iterations")) is not int or not 1<=value["max_poll_iterations"]<=MAX_ITERATIONS:
        raise ValueError("Passive polling must be finite at15-second intervals")
    target={"target_phase":TARGET_PHASE,"target_pid":TARGET_PID,"target_supervisor_pid":TARGET_SUPERVISOR_PID,
        "target_started_at_utc":TARGET_STARTED_UTC,"target_proof_sha256":TARGET_PROOF_SHA256}
    if any(value.get(k)!=v for k,v in target.items()):raise ValueError("Permit does not bind this exact already-started Spinosin phase")
    lock_path=run_root/"execution_lock.json"
    if value.get("execution_lock_sha256")!=EXECUTION_LOCK_SHA256 or sha(lock_path)!=EXECUTION_LOCK_SHA256:
        raise ValueError("Pre-quantum execution lock changed")
    sources=value.get("source_sha256",{})
    for relative in (FINISHER_REL,FINISHER_TEST_REL,ANALYZER_REL,ANALYZER_TEST_REL,
                     SUPERVISOR_REL,ENTRY_REL,V8_REL):
        if sources.get(str(relative))!=sha(ROOT/relative):raise ValueError("Root permit source binding differs: "+str(relative))
    for relative,expected in sources.items():
        if sha(safe_path(relative,ROOT))!=expected:raise ValueError("Root-permitted source changed: "+relative)
    for relative,expected in ((ENTRY_REL,ENTRY_SHA256),(SUPERVISOR_REL,SUPERVISOR_SHA256),
                              (V8_REL,V8_SHA256),(ANALYZER_REL,ANALYZER_SHA256)):
        if sha(ROOT/relative)!=expected:raise ValueError("Active calculation frozen source changed")
    ledger=read(paths["ledger_path"]);lock=read(lock_path)
    authorization,auth,prior=validate_metadata_bindings(run_root,ledger,lock,value)
    if value.get("run_deadline_at_utc")!=ledger.get("run_deadline_at_utc"):
        raise ValueError("Finisher cannot extend this V4 deadline")
    deadline=parse_time(value["run_deadline_at_utc"])
    return {**value,"_permit_path":path,"_permit_sha256":expected_sha,"_run_root":run_root,
        "_ledger_path":paths["ledger_path"],"_lock_path":lock_path,
        "_analysis_output":paths["analysis_output_path"],"_document_path":paths["documentation_path"],
        "_completion_path":paths["root_supervisor_completion_path"],
        "_proof_path":run_root/"execution"/(TARGET_PHASE+".log.supervisor-proof.json"),
        "_authorization_path":authorization,"_authorization":auth,"_prior_ledger_path":prior,"_deadline":deadline}


def closed_stage_snapshot(permit):
    """All controls ending is insufficient: the exact target and wrapper must end."""
    ledger=read(permit["_ledger_path"]);rows=ledger.get("phases",[])
    names=[row.get("phase") for row in rows]
    if len(set(names))!=len(names) or any(name not in PHASES for name in names):
        raise ValueError("Unexpected or duplicate phase in this one-shot epoch")
    if names!=list(PHASES[:len(names)]):raise ValueError("Frozen control-before-target phase order changed")
    if names!=list(PHASES):return None
    target=rows[-1]
    if target.get("pid")!=TARGET_PID or target.get("started_at_utc")!=TARGET_STARTED_UTC:
        raise ValueError("Observed target is not the root-permitted running phase")
    if (not permit["_proof_path"].is_file() or sha(permit["_proof_path"])!=TARGET_PROOF_SHA256
            or target.get("supervisor_proof_sha256")!=TARGET_PROOF_SHA256
            or resolve(target.get("supervisor_proof_path",""))!=permit["_proof_path"]):
        raise ValueError("Actual target proof bytes differ")
    proof=read(permit["_proof_path"])
    if (proof.get("phase")!=TARGET_PHASE or proof.get("child_pid")!=TARGET_PID
            or proof.get("supervisor_pid")!=TARGET_SUPERVISOR_PID
            or proof.get("schema")!="research4-pyscf-actual-cpu-supervisor-proof-v4"):
        raise ValueError("Target proof process identity differs")
    if any(row.get("status")=="RUNNING" or not row.get("ended_at_utc") for row in rows):return None
    terminal=("EXITED","PARTIAL_OR_FAILED","FAILED_TO_START","BUDGET_VIOLATION_REQUIRES_REVIEW")
    if any(row.get("status") not in terminal for row in rows):raise ValueError("Unreviewed terminal phase state")
    for row in rows:parse_time(row["ended_at_utc"])
    if not permit["_completion_path"].is_file():return None
    try:
        completion=read(permit["_completion_path"])
    except (json.JSONDecodeError,FileNotFoundError):
        # The root wrapper may create then write this small marker directly.
        # A partial write is still "not closed", never a release or retry.
        return None
    # The frozen supervisor CLI maps its audited EXITED state to0 and every
    # other terminal state to1. Keep that wrapper code separate from the
    # payload's own exit code (which can be2 or a negative signal code).
    expected_wrapper_exit=0 if target["status"]=="EXITED" else 1
    if (completion.get("status")!="SUPERVISOR_RETURNED"
            or type(completion.get("exit_code")) is not int or completion["exit_code"]!=expected_wrapper_exit):
        raise ValueError("Root wrapper completion does not agree with real supervisor return semantics")
    completion_time=parse_time(completion.get("utc"))
    if not (parse_time(TARGET_STARTED_UTC)<=parse_time(target["ended_at_utc"])
            <=completion_time<=utc()):
        raise ValueError("Root wrapper completion must follow the target end and cannot be future-dated")
    if any(parse_time(row["ended_at_utc"])>completion_time for row in rows):
        raise ValueError("Root wrapper completion predates a supposedly closed phase")
    lock=read(permit["_lock_path"])
    validate_metadata_bindings(permit["_run_root"],ledger,lock,permit)
    for row in rows:
        if row.get("command")!=lock["phases"][row["phase"]]["payload_argv"]:
            raise ValueError("Actual payload differs from preauthorized frozen inputs/command")
    violation=any(row["status"]=="BUDGET_VIOLATION_REQUIRES_REVIEW" for row in rows)
    if violation and permit.get("allow_disclosed_budget_violation_analysis") is not True:
        raise ValueError("Budget violation requires explicit root disclosed-failure analysis permission")
    return {"ledger":ledger,"ledger_sha256":sha(permit["_ledger_path"]),
        "execution_lock_sha256":sha(permit["_lock_path"]),"prior_ledger_sha256":sha(permit["_prior_ledger_path"]),
        "target_proof_sha256":sha(permit["_proof_path"]),"root_completion_sha256":sha(permit["_completion_path"]),
        "target_payload_exit_code":target.get("exit_code"),"supervisor_wrapper_exit_code":completion["exit_code"],
        "budget_violation_disclosed":violation}


def wait_for_closure(permit,clock=utc,sleep=time.sleep,on_poll=None):
    remaining=(permit["_deadline"]-clock()).total_seconds()
    count=min(permit["max_poll_iterations"],max(1,math.ceil(max(0,remaining)/POLL_SECONDS)+1))
    for iteration in range(1,count+1):
        if clock()>=permit["_deadline"]:raise TimeoutError("Finite V4 observer deadline reached; no automatic extension")
        snapshot=closed_stage_snapshot(permit)
        if on_poll is not None:on_poll(iteration,snapshot is not None)
        if snapshot is not None:return snapshot
        if iteration<count:sleep(min(POLL_SECONDS,max(0,(permit["_deadline"]-clock()).total_seconds())))
    raise TimeoutError("Root-permitted finite polling limit reached")


@contextmanager
def finalization_locks(permit):
    """Supervisor order; analyzer takes no flock and must not reacquire these."""
    paths=[permit["_ledger_path"].with_suffix(".lock"),ROOT/".runtime/pyscf-v4-global.lock",
           permit["_prior_ledger_path"].with_suffix(".lock")]
    with ExitStack() as stack:
        for path in paths:
            path=safe_path(path,ROOT);path.parent.mkdir(parents=True,exist_ok=True)
            handle=stack.enter_context(path.open("a"));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def build_release(permit,snapshot):
    return {"allow_analysis_export":True,"created_at_utc":utc().isoformat(),"run_root":str(permit["_run_root"]),
        "cpu_budget_sha256":snapshot["ledger_sha256"],"execution_lock_sha256":snapshot["execution_lock_sha256"],
        "analyzer_source_sha256":permit["source_sha256"][str(ANALYZER_REL)],
        "entry_source_sha256":permit["source_sha256"][str(ENTRY_REL)],
        "supervisor_source_sha256":permit["source_sha256"][str(SUPERVISOR_REL)],
        "finisher_source_sha256":permit["source_sha256"][str(FINISHER_REL)],
        "root_permit_path":str(permit["_permit_path"]),"root_permit_sha256":permit["_permit_sha256"],
        "target_phase":TARGET_PHASE,"target_pid":TARGET_PID,"target_started_at_utc":TARGET_STARTED_UTC,
        "target_proof_sha256":snapshot["target_proof_sha256"],"root_supervisor_completion_sha256":snapshot["root_completion_sha256"],
        "allow_disclosed_budget_violation_analysis":permit.get("allow_disclosed_budget_violation_analysis") is True,
        "locking_contract":"FINISHER_HOLDS_V4_GLOBAL_PRIOR_LOCKS_ANALYZER_DOES_NOT_REACQUIRE",
        "scientific_results_are_not_created_by_this_release":True}


def remaining_seconds(deadline,cap=300):
    value=(deadline-utc()).total_seconds()
    if value<=0:raise TimeoutError("Finite finalization deadline reached")
    return min(float(cap),value)


def redact(text):
    return re.sub(r"(https?://)[^/\s@]+@",r"\1[REDACTED]@",text)


def git_environment():
    value={k:v for k,v in os.environ.items() if not k.startswith("GIT_TRACE") and k!="GIT_CURL_VERBOSE"}
    value.update(GIT_AUTHOR_NAME=GITHUB_NAME,GIT_AUTHOR_EMAIL=GITHUB_EMAIL,
        GIT_COMMITTER_NAME=GITHUB_NAME,GIT_COMMITTER_EMAIL=GITHUB_EMAIL,GIT_TERMINAL_PROMPT="0")
    return value


def git_output(arguments,deadline,executor=None):
    result=(executor or subprocess.run)(["git",*arguments],cwd=ROOT,stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,timeout=remaining_seconds(deadline,120),env=git_environment())
    if result.returncode:raise RuntimeError("Git command failed: "+redact(result.stderr.decode(errors="replace")))
    return result.stdout


def allowed_scope(path,run_root,doc_path):
    path=resolve(path)
    return path.is_relative_to(run_root) or path==doc_path


def guard_git_state(permit,deadline,executor=None,*,require_original_head=False):
    output=lambda args:git_output(args,deadline,executor)
    if output(["branch","--show-current"]).decode().strip()!=BRANCH:
        raise RuntimeError("Research branch changed; preserve shared index")
    if output(["diff","--cached","--name-only","-z"]):
        raise RuntimeError("Mixed/shared index is not empty; never reset or unstage it")
    if require_original_head and output(["rev-parse","HEAD"]).decode().strip()!=permit["expected_code_commit"]:
        raise RuntimeError("Root-permitted code commit changed before finalization")
    paths=[]
    for args in (["diff","--name-only","--no-renames","-z"],
                 ["ls-files","--others","--exclude-standard","-z"]):
        paths.extend(os.fsdecode(raw) for raw in output(args).split(b"\0") if raw)
    for path in paths:
        if not allowed_scope(path,permit["_run_root"],permit["_document_path"]):
            raise RuntimeError("Non-whitelisted modified/new file; preserve shared index: "+path)
    if output(["diff","--diff-filter=D","--name-only","-z"]):
        raise RuntimeError("Do not stage any deletion")
    return paths


def closed_checkpoint_binding(permit):
    """Bind one small SCF warm-start checkpoint; never parse its HDF content."""
    snapshot=closed_stage_snapshot(permit)
    if snapshot is None:raise ValueError("Never inspect a checkpoint before exact target/wrapper closure")
    result_path=safe_path(permit["_run_root"]/TARGET_PHASE/"result.json",permit["_run_root"])
    result=read(result_path)
    resource=result.get("resource_supervision",{})
    if (result.get("identity")!="spinosin" or result.get("checkpoint_present") is not True
            or not re.fullmatch(r"[0-9a-f]{64}",str(result.get("checkpoint_sha256","")))
            or result.get("actual_entrypoint",{}).get("sha256")!=ENTRY_SHA256
            or resource.get("proof_sha256")!=TARGET_PROOF_SHA256
            or resource.get("proof")!=read(permit["_proof_path"])):
        raise ValueError("Small checkpoint requires its real closed target worker result and exact source/proof binding")
    if not (parse_time(TARGET_STARTED_UTC)<=parse_time(result.get("completed_at_utc"))
            <=parse_time(snapshot["ledger"]["phases"][-1]["ended_at_utc"])):
        raise ValueError("Closed target worker result completion time differs from its ledger")
    return {"path":str(permit["_run_root"]/TARGET_PHASE/"calculation.chk"),
        "checkpoint_present":True,"checkpoint_sha256":result["checkpoint_sha256"],
        "source_result_sha256":sha(result_path),"closed_ledger_sha256":snapshot["ledger_sha256"],
        "closed_evidence_verified":True,"role":"SCF_WARMSTART_ONLY_NOT_HESSIAN_RESTORATION"}


def candidate_delivery_files(run_root,doc_path,candidates,*,checkpoint_binding=None):
    run_root,doc_path=resolve(run_root),resolve(doc_path)
    result=[];seen=set();total=0
    for candidate in sorted(map(Path,candidates),key=str):
        path=safe_path(candidate,ROOT)
        if path in seen:continue
        seen.add(path)
        if not allowed_scope(path,run_root,doc_path) or not path.is_file():
            raise ValueError("Non-file/deleted/out-of-scope Git candidate")
        size=path.stat().st_size
        if path.suffix.lower() in (".h5",".hdf5"):
            raise ValueError("HDF caches are not Git delivery artifacts")
        if path.suffix.lower()==".chk":
            expected=run_root/TARGET_PHASE/"calculation.chk"
            binding=checkpoint_binding or {}
            if (path!=expected or resolve(binding.get("path",""))!=expected
                    or binding.get("checkpoint_present") is not True
                    or binding.get("closed_evidence_verified") is not True
                    or binding.get("role")!="SCF_WARMSTART_ONLY_NOT_HESSIAN_RESTORATION"
                    or not re.fullmatch(r"[0-9a-f]{64}",str(binding.get("source_result_sha256","")))
                    or not re.fullmatch(r"[0-9a-f]{64}",str(binding.get("closed_ledger_sha256","")))):
                raise ValueError("Only the exact closed target SCF checkpoint may be delivered")
            if size>MAX_CHECKPOINT_BYTES:
                raise ValueError("Closed scientific checkpoint exceeds16MiB; preserve it for root review")
            if sha(path)!=binding.get("checkpoint_sha256"):
                raise ValueError("Closed checkpoint differs from its actual worker result SHA")
        if size>MAX_FILE_BYTES:raise ValueError("Closed file exceeds25MiB; do not auto-delete or upload: "+str(path))
        total+=size
        if total>MAX_TOTAL_BYTES:raise ValueError("Closed Git delivery exceeds150MiB")
        row={"path":str(path.relative_to(ROOT)),"bytes":size,"sha256":sha(path)}
        if path.suffix.lower()==".chk":row["checkpoint_evidence"]=dict(checkpoint_binding)
        result.append(row)
    return result


def publish_results(permit,deadline,delivery_callback=None,executor=None):
    if permit.get("allow_git_push") is not True:raise ValueError("Root permit does not authorize Git publication")
    output=lambda args:git_output(args,deadline,executor)
    paths=guard_git_state(permit,deadline,executor,require_original_head=True)
    origin=output(["remote","get-url","origin"]).decode().strip()
    if not re.fullmatch(r"(?:https://(?:[^/@]+@)?github\.com/|git@github\.com:)jackzhu119/research4(?:\.git)?/?",origin):
        raise RuntimeError("Origin is not the authorized research4 repository")
    checkpoint_binding=None
    if any(Path(path).suffix.lower()==".chk" for path in paths):
        checkpoint_binding=closed_checkpoint_binding(permit)
    files=candidate_delivery_files(permit["_run_root"],permit["_document_path"],[ROOT/p for p in paths],
        checkpoint_binding=checkpoint_binding)
    if not files:return {"status":"NO_NEW_SCOPED_FILES","push_attempted":False}
    names=[row["path"] for row in files]
    guard_git_state(permit,deadline,executor,require_original_head=True)
    output(["add","--",*names])
    staged={os.fsdecode(raw) for raw in output(["diff","--cached","--name-only","-z"]).split(b"\0") if raw}
    if staged!=set(names):raise RuntimeError("Concurrent/mixed staging; stop without resetting shared index")
    if output(["rev-parse","HEAD"]).decode().strip()!=permit["expected_code_commit"]:
        raise RuntimeError("HEAD changed after staging; preserve shared index")
    blobs={}
    for row in files:
        name=row["path"]
        if sha(ROOT/name)!=row["sha256"]:raise RuntimeError("Closed file changed before commit")
        blob=output(["hash-object","--",name]).strip()
        if output(["rev-parse",":"+name]).strip()!=blob:raise RuntimeError("Index blob differs from audited bytes")
        blobs[name]=blob
    output(["-c","core.hooksPath=/dev/null","commit","--only","-m",
            "research: finalize closed pure-J Spinosin evidence","--",*names])
    commit=output(["rev-parse","HEAD"]).decode().strip()
    result={"status":"COMMITTED_NOT_YET_PUSHED","commit_sha":commit,"branch":BRANCH,
        "files":files,"push_attempted":False,"force_push":False,"main_modified":False,
        "author_email":GITHUB_EMAIL,"committer_email":GITHUB_EMAIL}
    if delivery_callback:delivery_callback(dict(result))
    if (not re.fullmatch(r"[0-9a-f]{40}",commit)
            or output(["rev-parse",commit+"^"]).decode().strip()!=permit["expected_code_commit"]):
        raise RuntimeError("Actual commit parent differs from the root-permitted code commit; stop before push")
    actual={os.fsdecode(raw) for raw in output(["diff-tree","--no-commit-id","--name-only","-r","-z",commit]).split(b"\0") if raw}
    if actual!=set(names):raise RuntimeError("Actual commit scope differs; stop before push")
    for name,blob in blobs.items():
        if output(["rev-parse",commit+":"+name]).strip()!=blob:
            raise RuntimeError("Actual committed blob differs from audited data")
    if output(["branch","--show-current"]).decode().strip()!=BRANCH or output(["rev-parse","HEAD"]).decode().strip()!=commit:
        raise RuntimeError("HEAD/branch changed before push")
    if output(["diff","--cached","--name-only","-z"]):raise RuntimeError("Concurrent shared-index changes; do not push")
    # Recheck all remaining global changes, not merely our report tree.
    guard_git_state(permit,deadline,executor)
    if output(["remote","get-url","origin"]).decode().strip()!=origin:
        raise RuntimeError("Authorized origin changed before push")
    result["push_attempted"]=True
    if delivery_callback:delivery_callback(dict(result))
    pushed=output(["push","--porcelain","origin",commit+":refs/heads/"+BRANCH])
    result.update(status="PUSHED",push_output=redact(pushed.decode(errors="replace")),completed_at_utc=utc().isoformat())
    if delivery_callback:delivery_callback(dict(result))
    return result


def validate_summary(summary):
    if (summary.get("analysis_only") is not True or summary.get("new_SCF_gradient_Hessian_evaluations")!=0
            or summary.get("IR_intensities_status")!="NOT_COMPUTED"
            or summary.get("chemical_attribution_confirmed") is not False
            or summary.get("classification_modified") is not False
            or summary.get("cache_restoration")!="NOT_IMPLEMENTED_SAVE_ONLY"):
        raise ValueError("Analyzer output exceeds the permitted evidence-only scientific scope")
    for name in ("computed_Spinosin_frequency_rows_including_unvalidated","validated_spinosin_frequency_rows",
                 "validated_spinosin_low_band_modes","failed_or_partial_phase_count"):
        if type(summary.get(name)) is not int or summary[name]<0:raise ValueError("Actual analyzer counts required: "+name)
    if not (summary["validated_spinosin_low_band_modes"]<=summary["validated_spinosin_frequency_rows"]
            <=summary["computed_Spinosin_frequency_rows_including_unvalidated"]):
        raise ValueError("Qualified/scoped scientific counts are inconsistent")
    return {"Spinosin_frequency_status":summary.get("Spinosin_frequency_status"),
        "computed_signed_rows":summary["computed_Spinosin_frequency_rows_including_unvalidated"],
        "qualified_rows":summary["validated_spinosin_frequency_rows"],
        "failed_or_partial_phase_count":summary["failed_or_partial_phase_count"],
        "IR_intensities_status":"NOT_COMPUTED","classification_modified":False}


def run(args,analysis_executor=None,git_executor=None,clock=utc,sleep=time.sleep):
    delivery=safe_path(args.delivery_dir,ROOT/"outputs")
    if delivery.exists():raise FileExistsError("Preserve previous delivery logs; use a new ignored delivery directory")
    delivery.mkdir(parents=True)
    result={"status":"INITIALIZING_PASSIVE_FINISHER","started_at_utc":clock().isoformat(),
        "analysis_attempts":0,"new_quantum_jobs":0,"active_molecular_array_reads":0,
        "Git_delivery":{"status":"NOT_ATTEMPTED"}}
    def event(name,details=None):
        with (delivery/"events.jsonl").open("a") as f:
            f.write(json.dumps({"utc":clock().isoformat(),"event":name,"details":details},ensure_ascii=False,allow_nan=False)+"\n")
    try:
        permit=validate_permit(args.permit,args.permit_sha256,push=args.push)
        guard_git_state(permit,permit["_deadline"],git_executor,require_original_head=True)
        result["status"]="WAITING_FOR_EXACT_TARGET_AND_WRAPPER_CLOSURE"
        snapshot=wait_for_closure(permit,clock,sleep,lambda i,ready:event("passive_JSON_poll",{"iteration":i,"ready":ready}))
        with finalization_locks(permit):
            current=closed_stage_snapshot(permit)
            if current is None or current!=snapshot:raise ValueError("Closed evidence changed before acquiring finalization locks")
            validate_permit(args.permit,args.permit_sha256,push=args.push)
            guard_git_state(permit,permit["_deadline"],git_executor,require_original_head=True)
            release_path=permit["_run_root"]/"final_analysis_release.json"
            if release_path.exists() or permit["_analysis_output"].exists() or permit["_document_path"].exists():
                raise FileExistsError("Preserve prior release/analysis/document; no automatic analysis retry")
            release=build_release(permit,snapshot);write_new(release_path,release)
            command=[str(ROOT/".runtime/bin/python"),str(ROOT/ANALYZER_REL),"--run-root",str(permit["_run_root"]),
                "--output",str(permit["_analysis_output"]),"--release",str(release_path),"--report",str(permit["_document_path"])]
            result["analysis_attempts"]=1
            write_new(delivery/"analysis_attempt.json",{"command":command,"release_sha256":sha(release_path),
                "one_shot":True,"locks_owned_by_finisher_analyzer_does_not_reacquire":True})
            event("analysis_requested_once",{"release_sha256":sha(release_path)})
            analysis_environment=dict(os.environ)
            # Reuse the already frozen dependency/thread environment, rather
            # than depend on the observer launcher's ambient PYTHONPATH.
            analysis_environment.update(read(permit["_lock_path"])["environment_overrides"])
            completed=(analysis_executor or subprocess.run)(command,cwd=ROOT,stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,timeout=remaining_seconds(permit["_deadline"],600),close_fds=True,
                env=analysis_environment)
            (delivery/"analysis.log").write_bytes(completed.stdout)
            if completed.returncode:raise RuntimeError("Frozen closed-only analyzer failed; no automatic retry")
            if closed_stage_snapshot(permit)!=snapshot:raise ValueError("Closed source evidence changed during analysis")
            validate_permit(args.permit,args.permit_sha256,push=args.push)
            summary=read(permit["_analysis_output"]/"summary.json")
            scientific=validate_summary(summary)
            if not permit["_document_path"].is_file():raise ValueError("Actual analyzer did not create the fixed new document")
            result.update(status="SCIENTIFIC_EXPORT_DONE",scientific_status=scientific,
                final_ledger_sha256=snapshot["ledger_sha256"],release_path=str(release_path),
                release_sha256=sha(release_path),summary_path=str(permit["_analysis_output"]/"summary.json"),
                document_path=str(permit["_document_path"]),document_sha256=sha(permit["_document_path"]))
            if args.push:
                try:
                    def save_git(value):
                        result["Git_delivery"]=value;event("Git_delivery",value)
                    result["Git_delivery"]=publish_results(permit,permit["_deadline"],save_git,git_executor)
                    result["status"]=("SCIENTIFIC_EXPORT_DONE_GIT_PUSHED"
                        if result["Git_delivery"].get("status")=="PUSHED"
                        else "SCIENTIFIC_EXPORT_DONE_GIT_NO_NEW_FILES")
                except Exception as exc:
                    result.update(status="SCIENTIFIC_EXPORT_DONE_GIT_BLOCKED_OR_FAILED")
                    result["Git_delivery"]={**result["Git_delivery"],"status":"BLOCKED_OR_FAILED_PRESERVE_SHARED_INDEX","reason":redact(str(exc))}
    except Exception as exc:
        result.update(status="PARTIAL_FINALIZATION",exception_type=type(exc).__name__,stop_reason=redact(str(exc)))
    finally:
        result["ended_at_utc"]=clock().isoformat()
        write_new(delivery/"delivery.json",result)
        (delivery/"RECOVERY.md").write_text("# Delivery recovery\n\nInspect delivery.json, events.jsonl and analysis.log before any action. No automatic retry, force-push, reset, clean, merge or index repair was performed. Existing release/output/document files remain intact. If analysis failed or stopped, root must review partial evidence and explicitly authorize a new output/permit or manual continuation; do not run the same analyzer twice blindly. If a commit exists but push failed, verify its exact scope/blob hashes and the research branch, then root may push that fixed commit with the recorded GitHub noreply author/committer identity. Preserve unrelated staged work. No quantum job is launched by this finisher.\n")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permit",required=True,type=Path)
    parser.add_argument("--permit-sha256",required=True)
    parser.add_argument("--delivery-dir",required=True,type=Path)
    parser.add_argument("--push",action="store_true")
    args=parser.parse_args();value=run(args);print(json.dumps(value,ensure_ascii=False))
    raise SystemExit(0 if value["status"] in ("SCIENTIFIC_EXPORT_DONE","SCIENTIFIC_EXPORT_DONE_GIT_PUSHED",
        "SCIENTIFIC_EXPORT_DONE_GIT_NO_NEW_FILES") else 2)


if __name__=="__main__":main()
