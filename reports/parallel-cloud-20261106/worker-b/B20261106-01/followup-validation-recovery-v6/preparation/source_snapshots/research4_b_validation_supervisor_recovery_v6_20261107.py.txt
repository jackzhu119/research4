#!/usr/bin/env python3
"""Finite smoke-gated validation recovery carrying all prior costs plus the
user-authorized single-use ceiling extension.

No implicit execution, budget reset, unknown-orphan recovery, Git mutation or
scientific success inference. Frozen initial-run files are read-only evidence.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
RUN_ID = "B20261106-01"
BRANCH = "research/parallel-worker-b-20261106-" + RUN_ID
ORIGINAL_WORK = "reports/parallel-cloud-20261106/worker-b/" + RUN_ID
DEADLINE = "2026-11-09T08:25:17.730193+00:00"
PREDECESSOR_CAP = 58800
ADDITIONAL_CPU_SECONDS = 31200
CAP = 90000
EXTENSION_CLAIM = ".runtime/worker-b-cpu-extension-20261107.json"
MANIFEST_SCHEMA = "research4-b-validation-recovery-manifest-v6"
LEDGER_SCHEMA = "research4-b-validation-recovery-cpu-ledger-v6"
PROOF_SCHEMA = "research4-b-validation-recovery-resource-proof-v6"
PREDECESSOR_SUPERVISOR_SHA256 = "7e20292017b5d9d7a43dd24969053fb40361f572e1f52b28eae929ccb4eb310a"
ACCOUNTING = "ACTUAL_WAIT4_SELF_AND_WAITED_CHILDREN_SINGLE_TGID_ENFORCED"
SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
TOKEN = re.compile(r"\{(result|sha256):([A-Za-z0-9_-]+)\}\Z")


def require(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(file):
    h = hashlib.sha256()
    with Path(file).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def path(value):
    p = Path(value)
    lexical = Path(os.path.abspath(p if p.is_absolute() else ROOT / p))
    require(lexical.is_relative_to(ROOT), "Path leaves this checkout")
    require(not any(x.is_symlink() for x in (lexical, *lexical.parents) if x.is_relative_to(ROOT)), "Symlink metadata/output rejected")
    return lexical


def python_path(value):
    p = Path(value)
    p = Path(os.path.abspath(p if p.is_absolute() else ROOT / p))
    require(p.is_relative_to(ROOT / ".runtime"), "Dedicated B interpreter selector required")
    # A venv selector normally is a symlink; it is deliberately not a data path.
    return p


def read(file, expected=None):
    p = path(file)
    require(p.stat().st_size <= 4 * 1024 ** 2, "Metadata larger than4MiB")
    if expected:
        require(sha(p) == expected, "Pinned bytes changed: " + str(p))
    return json.loads(p.read_text())


def save(file, value, exclusive=False):
    p = path(file)
    p.parent.mkdir(parents=True, exist_ok=True)
    target = p if exclusive else p.with_suffix(p.suffix + ".writing")
    with target.open("x" if exclusive else "w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if not exclusive:
        target.replace(p)


def ref(file):
    p = path(file)
    return {"path": str(p.relative_to(ROOT)), "sha256": sha(p)}


def checked(value):
    p = path(value["path"])
    require(sha(p) == value["sha256"], "Pinned source/input changed: " + str(p))
    return p


def identity(pid):
    base = Path("/proc") / str(pid)
    fields = (base / "stat").read_text().rsplit(")", 1)[1].split()
    uids = next(x for x in (base / "status").read_text().splitlines() if x.startswith("Uid:")).split()[1:]
    again = (base / "stat").read_text().rsplit(")", 1)[1].split()
    require(fields[19] == again[19] and len(uids) == 4 and len(set(uids)) == 1, "Unstable process identity")
    return {"pid": int(pid), "start_ticks": int(fields[19]), "uid": int(uids[0]),
            "pgrp": int(fields[2]), "session": int(fields[3]), "state": fields[0],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "CPU_seconds": sum(int(fields[k]) for k in (11, 12, 13, 14)) / os.sysconf("SC_CLK_TCK"),
            "rss_bytes": max(0, int(fields[21])) * os.sysconf("SC_PAGE_SIZE")}


def same(a, b):
    return all(a.get(k) == b.get(k) for k in ("pid", "start_ticks", "uid", "pgrp", "session", "boot_id"))


def is_live(saved):
    try:
        observed = identity(saved["pid"])
        return same(saved, observed) and observed["state"] not in ("Z", "X", "x")
    except (FileNotFoundError, ProcessLookupError):
        return False


def session_members(expected):
    found = []
    for base in Path("/proc").iterdir():
        if not base.name.isdigit():
            continue
        try:
            fields = (base / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) != expected["pgrp"] and int(fields[3]) != expected["session"]:
                continue
            observed = identity(int(base.name))
            if observed["state"] not in ("Z", "X", "x"):
                found.append(observed)
        except (FileNotFoundError, ProcessLookupError):
            pass
    return found


def no_other_workers(args):
    """Do not compete with another scientific parent/worker on this host."""
    for base in Path("/proc").iterdir():
        if not base.name.isdigit() or int(base.name) == os.getpid():
            continue
        try:
            observed = identity(int(base.name))
            if observed["uid"] != os.getuid() or observed["state"] in ("Z", "X", "x"):
                continue
            tokens = [v.decode(errors="replace") for v in (base / "cmdline").read_bytes().split(b"\0") if v]
            if not tokens or not Path(tokens[0]).name.startswith(("python", "pypy")):
                continue
            scripts = [Path(t).name for t in tokens[1:] if t.endswith(".py")]
            if not any(any(k in name for k in ("pyscf", "quantum", "parallel_b_supervisor", "b_diagnostic", "b_rootcause", "b_validation", "cloud_orbital_stability")) for name in scripts):
                continue
            own_detacher = (observed["pid"] == os.getppid() and SELF.name in scripts and "--detach" in tokens
                            and "--manifest-sha256" in tokens
                            and tokens[tokens.index("--manifest-sha256") + 1] == args.manifest_sha256)
            require(own_detacher, "Existing scientific parent/worker blocks duplicate launch; pid=" + str(observed["pid"]))
        except (FileNotFoundError, ProcessLookupError):
            pass


@contextmanager
def exclusive_lock():
    p = path(".runtime/parallel-b-exclusive.lock")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def validate_carry(manifest):
    """Keep every prior CPU cost and extend its ceiling exactly once by grant."""
    prior_path = checked(manifest["predecessor_manifest"])
    ledger_path = checked(manifest["predecessor_ledger"])
    state_path = checked(manifest["predecessor_state"])
    source = checked(manifest["predecessor_supervisor"])
    allocation_path = checked(manifest["predecessor_allocation"])
    require(source == path("scripts/research4_b_validation_supervisor_recovery_20261107.py")
            and sha(source) == PREDECESSOR_SUPERVISOR_SHA256
            and allocation_path == path(".runtime/worker-b-validation-recovery-allocation.json"),
            "Exact closed v5 supervisor/allocation required")
    prior = read(prior_path); old = read(ledger_path); state = read(state_path); allocation = read(allocation_path)
    require(prior.get("schema") == "research4-b-validation-recovery-manifest-v5"
            and old.get("schema") == "research4-b-validation-recovery-cpu-ledger-v5"
            and old.get("run_id") == RUN_ID and old.get("cpu_cap_seconds") == PREDECESSOR_CAP
            and prior.get("cpu_cap_seconds") == PREDECESSOR_CAP
            and old.get("deadline_utc") == DEADLINE, "V5 historical ceiling must remain58800")
    require(old["manifest_sha256"] == manifest["predecessor_manifest"]["sha256"] == state["manifest_sha256"]
            and old["supervisor_sha256"] == manifest["predecessor_supervisor"]["sha256"], "V5 source/configuration differs")
    require(path(prior["ledger"]) == ledger_path and path(prior["work_directory"]) == ledger_path.parent
            and state_path == ledger_path.parent / "state.json", "V5 ledger/state paths differ")
    require(allocation["manifest_sha256"] == old["manifest_sha256"] and path(allocation["ledger"]) == ledger_path
            and path(allocation["work_directory"]) == path(prior["work_directory"])
            and allocation["cpu_cap_seconds"] == PREDECESSOR_CAP and allocation["carry"] == old["carry"],
            "Persistent v5 allocation differs")
    require(old["phases"] and all(row["status"] in ("EXITED", "FAILED_CLOSED") for row in old["phases"])
            and not old.get("blocked_unknown_cpu_or_process_tree"), "V5 CPU/process history must be completely closed")
    spec = importlib.util.spec_from_file_location("r4_b_frozen_closed_v5", source)
    predecessor = importlib.util.module_from_spec(spec); spec.loader.exec_module(predecessor)
    predecessor.validate_ledger(old, prior, old["manifest_sha256"], old["carry"])
    old_carry = predecessor.validate(prior, old["manifest_sha256"])
    predecessor.validate_ledger(old, prior, old["manifest_sha256"], old_carry)
    require(state.get("invocations") and all(i["status"] in ("CLOSED", "CLOSED_WITH_BLOCKED_JOBS") for i in state["invocations"]),
            "V5 parent is not closed")
    require(set(state["jobs"]) == {j["id"] for j in prior["jobs"]}
            and all(saved["status"] in ("DONE_SEALED", "FAILED_CLOSED", "FAILED_RESULT_VALIDATION_CLOSED", "PARTIAL_SCIENTIFIC_CLOSED",
                                       "BLOCKED_DEPENDENCY", "BLOCKED_RESOURCE_OR_DEADLINE") for saved in state["jobs"].values()),
            "V5 has unknown or unclosed jobs")
    previous_inherited = prior["inherited_results"]
    new_success_ids = {key for key, saved in state["jobs"].items() if saved["status"] == "DONE_SEALED"}
    require(not set(previous_inherited).intersection(new_success_ids), "Inherited/new v5 success collision")
    new_refs = {key: state["results"][key] for key in new_success_ids}
    require(state["results"] == {**previous_inherited, **new_refs}
            and manifest.get("inherited_results") == state["results"], "V6 must inherit all prior successes exactly")
    current_boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    require(manifest.get("recovery_boot_id") == prior["recovery_boot_id"] == current_boot,
            "This recovery must bind the explicitly reviewed current boot")
    for row in old["phases"]:
        require(not is_live(row["child_identity"]) and not session_members(row["child_identity"]), "V5 child/group remains live")
    for invocation in state["invocations"]:
        require(not is_live(invocation["identity"]), "V5 parent remains live")
    for job in prior["jobs"]:
        if job["id"] not in new_success_ids:
            continue
        saved = state["jobs"][job["id"]]
        rows = [row for row in old["phases"] if row["phase"] == saved["phase"]]
        require(len(rows) == 1, "V5 success lacks unique closed phase")
        _, seal = predecessor.seal_result(rows[0], job)
        require(seal == saved["seal"] and seal["result"] == state["results"][job["id"]], "V5 historical seal differs")
    review = read(checked(manifest["predecessor_closure_review"]))
    require(review.get("schema") == "research4-b-independent-v5-closure-remote-review-v1"
            and review.get("status") == "PASS_CLOSED_PARTIAL_RESULTS_AND_PREVIOUS_REMOTE_VERIFICATION", "V5 closure review required")
    for key, binding in (("v5_manifest", manifest["predecessor_manifest"]), ("v5_ledger", manifest["predecessor_ledger"]),
                         ("v5_state", manifest["predecessor_state"]), ("v5_supervisor", manifest["predecessor_supervisor"]),
                         ("persistent_v5_allocation", manifest["predecessor_allocation"])):
        require(checked(review[key]) == checked(binding) and review[key]["sha256"] == binding["sha256"], "V5 review binding differs: " + key)
    require(review["current_boot_observation"]["current_boot_id"] == current_boot, "V5 review boot differs")
    checked(review["return_receipt"])
    actual = old["actual_closed_cpu_seconds"]; debit = old["conservative_debit_cpu_seconds"]; charged = old["budget_charged_cpu_seconds"]
    require(all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in (actual, debit, charged))
            and debit == 1000 and math.isclose(actual + debit, charged, abs_tol=1e-6) and charged <= PREDECESSOR_CAP,
            "Prior actual costs or permanent conservative debit cannot be refunded")
    grant = read(checked(manifest["extension_authorization"]))
    require(grant.get("schema") == "research4-b-cpu-extension-authorization-v1" and grant.get("run_id") == RUN_ID
            and grant.get("approved") is True and grant.get("single_use") is True and SAFE.fullmatch(grant["grant_id"]), "Explicit single-use CPU grant required")
    require(grant.get("original_cpu_cap_seconds") == PREDECESSOR_CAP
            and grant.get("additional_cpu_seconds") == ADDITIONAL_CPU_SECONDS
            and grant.get("effective_cpu_cap_seconds") == grant.get("user_specified_numeric_total") == CAP
            and grant.get("deadline_utc") == DEADLINE and grant.get("branch") == BRANCH
            and grant.get("recovery_boot_id") == current_boot, "Grant must explicitly authorize cumulative90000, not a refund or repeated increment")
    for key in ("predecessor_manifest", "predecessor_ledger", "predecessor_state", "predecessor_allocation"):
        require(checked(grant[key]) == checked(manifest[key]) and grant[key]["sha256"] == manifest[key]["sha256"], "Grant predecessor binding differs")
    require(math.isclose(grant["initial_actual_closed_cpu_seconds"], actual, abs_tol=1e-6)
            and grant["permanent_conservative_debit_cpu_seconds"] == debit
            and math.isclose(grant["initial_budget_charged_cpu_seconds"], charged, abs_tol=1e-6)
            and math.isclose(grant["new_remaining_before_execution_cpu_seconds"], CAP - charged, abs_tol=1e-6), "Grant may only extend the ceiling; all existing charges remain")
    return {"predecessor_manifest": manifest["predecessor_manifest"], "predecessor_ledger": manifest["predecessor_ledger"],
            "predecessor_state": manifest["predecessor_state"], "predecessor_supervisor": manifest["predecessor_supervisor"],
            "predecessor_allocation": manifest["predecessor_allocation"], "predecessor_closure_review": manifest["predecessor_closure_review"],
            "actual_closed_cpu_seconds": actual, "conservative_debit_cpu_seconds": debit, "budget_charged_cpu_seconds": charged,
            "conservative_debit_is_not_actual_wait4": True, "recovery_boot_id": current_boot,
            "original_cpu_cap_seconds": PREDECESSOR_CAP, "additional_cpu_authorized_seconds": ADDITIONAL_CPU_SECONDS,
            "effective_cpu_cap_seconds": CAP, "extension_is_not_actual_cpu_consumption": True,
            "extension_authorization": manifest["extension_authorization"], "extension_grant_id": grant["grant_id"],
            "inherited_results_sha256": digest(state["results"]), "inherited_success_count": len(state["results"]),
            "older_inherited_success_count": len(previous_inherited), "v5_new_sealed_success_count": len(new_success_ids),
            "v5_new_seals_sha256": digest({key: state["jobs"][key]["seal"] for key in sorted(new_success_ids)}),
            "predecessor_all_closed_phase_count": len(old["phases"]),
            "predecessor_failed_closed_phase_count": sum(row["status"] == "FAILED_CLOSED" for row in old["phases"])}


def validate(manifest, manifest_sha):
    require(manifest.get("schema") == MANIFEST_SCHEMA and manifest.get("run_id") == RUN_ID, "Exact B validation manifest required")
    require(manifest.get("branch") == BRANCH and manifest.get("cpu_cap_seconds") == CAP, "Same branch and aggregate cap required")
    deadline = datetime.fromisoformat(manifest["deadline_utc"])
    require(deadline.tzinfo is not None and deadline <= datetime.fromisoformat(DEADLINE), "Diagnostic deadline cannot extend original grant")
    require(manifest.get("threads") == 2 and manifest.get("memory_gib") == 12 and manifest.get("rss_stop_gib") == 8, "Original B thread/memory bounds required")
    require(isinstance(manifest.get("reserve_cpu_seconds"), (int, float)) and manifest["reserve_cpu_seconds"] >= 60, "Explicit residual CPU reserve required")
    work = path(manifest["work_directory"])
    require(work.parent == path(ORIGINAL_WORK) and work.name == "followup-validation-recovery-v6", "Reviewed validation work directory is fixed under original B run")
    require(path(manifest["ledger"]) == work / "cpu_budget.json", "Unique diagnostic ledger required")
    require(python_path(manifest["python_executable"]).is_file(), "Dedicated B interpreter absent")
    refs = manifest.get("source_refs", [])
    require(any(checked(r) == SELF for r in refs), "Actual diagnostic supervisor source must be pinned")
    for r in refs + manifest.get("input_refs", []):
        checked(r)
    upload = manifest.get("upload", {})
    if upload.get("enabled"):
        helper = checked(upload["helper"])
        require(helper == path("scripts/research4_parallel_b_git_return_20261106.py") and any(checked(r) == helper for r in refs), "Only pinned original B return helper is supported")
        return_manifest = read(checked(upload["manifest"]))
        require(return_manifest.get("run_id") == RUN_ID and return_manifest.get("role", "B") == "B"
                and path(return_manifest["work_directory"]) == path(ORIGINAL_WORK)
                and path(return_manifest["ledger"]) == path(manifest["ledger"])
                and return_manifest.get("upload", {}).get("branch") == BRANCH, "Return manifest must bind this same follow-up ledger/branch")
    ids = []
    inherited_ids = set(manifest.get("inherited_results", {}))
    for job in manifest["jobs"]:
        require(SAFE.fullmatch(job["id"]) and job["id"] not in ids and job["id"] not in inherited_ids, "Unique new diagnostic job IDs required")
        require(set(job.get("depends_on", [])).issubset(set(ids) | inherited_ids), "Dependencies must precede their job or be sealed inherited v4 results")
        ids.append(job["id"])
        require(isinstance(job["cpu_seconds"], int) and 20 <= job["cpu_seconds"] <= CAP, "Bounded integer job CPU required")
        require(isinstance(job["wall_seconds"], int) and 61 <= job["wall_seconds"] <= 72 * 3600, "Bounded job wall time required")
        require(isinstance(job.get("max_attempts", 1), int) and 1 <= job.get("max_attempts", 1) <= 2, "At most2 preserved attempts per validation job")
        require(isinstance(job["argv"], list) and len(job["argv"]) >= 2 and all(isinstance(x, str) for x in job["argv"]), "Explicit worker argv required")
        require(python_path(job["argv"][0]) == python_path(manifest["python_executable"]), "Worker must use qualified B Python")
        worker = path(job["argv"][1])
        require(worker.suffix == ".py" and any(checked(r) == worker for r in refs), "Actual worker must be a pinned Python source")
        require("{output}" in job["argv"], "Worker must receive its unique output directory")
        if job.get("worker"):
            require(checked(job["worker"]) == worker, "Worker ref differs from argv")
        if job.get("admission_group"):
            require(isinstance(job["admission_group"], str) and SAFE.fullmatch(job["admission_group"]), "Safe admission group required")
    for group in {j["admission_group"] for j in manifest["jobs"] if j.get("admission_group")}:
        members = [i for i, j in enumerate(manifest["jobs"]) if j.get("admission_group") == group]
        require(len(members) == 2 and members[1] == members[0] + 1,
                "Each admission group is one consecutive two-point pair")
    require(ids and len(ids) <= 100, "Finite nonempty diagnostic queue required")
    quantum_ids = [job["id"] for job in manifest["jobs"] if job["id"].endswith(("-minus", "-plus"))]
    require(quantum_ids and manifest.get("fail_fast_quantum_job_ids") == quantum_ids
            and manifest.get("smoke_job_id") == quantum_ids[0] == ids[0]
            and manifest["jobs"][0].get("smoke_success_gate") is True,
            "The first pending point must be the explicit complete smoke success gate")
    by_id = {job["id"]: job for job in manifest["jobs"]}
    for previous, current in zip(quantum_ids, quantum_ids[1:]):
        require({quantum_ids[0], previous}.issubset(by_id[current].get("depends_on", [])),
                "Every subsequent quantum point must depend on the smoke and preceding quantum success")
    prior_jobs = {job["id"]: job for job in read(checked(manifest["predecessor_manifest"]))["jobs"]}
    prior_phases = read(checked(manifest["predecessor_ledger"]))["phases"]
    for job in manifest["jobs"]:
        if job["id"] in prior_jobs:
            spent_attempts = sum(row["job_id"] == job["id"] for row in prior_phases)
            require(job.get("max_attempts", 1) <= prior_jobs[job["id"]].get("max_attempts", 1) - spent_attempts,
                    "Recovery cannot reset historical attempt limits")
    require(re.fullmatch(r"[0-9a-f]{64}", manifest_sha), "Pinned manifest SHA-256 required")
    return validate_carry(manifest)


def validate_ledger(ledger, manifest, manifest_sha, carry):
    require(ledger.get("schema") == LEDGER_SCHEMA and ledger.get("manifest_sha256") == manifest_sha and ledger.get("supervisor_sha256") == sha(SELF), "Ledger/source binding changed")
    require(ledger.get("run_id") == RUN_ID and ledger.get("cpu_cap_seconds") == CAP and ledger.get("deadline_utc") == manifest["deadline_utc"], "Ledger allocation changed")
    require(ledger.get("original_cpu_cap_seconds") == PREDECESSOR_CAP
            and ledger.get("additional_cpu_authorized_seconds") == ADDITIONAL_CPU_SECONDS
            and ledger.get("extension_authorization") == manifest["extension_authorization"],
            "The user extension only raises the cumulative ceiling once")
    require(ledger.get("carry") == carry and ledger.get("initial_actual_closed_cpu_seconds") == carry["actual_closed_cpu_seconds"]
            and ledger.get("initial_conservative_debit_cpu_seconds") == carry["conservative_debit_cpu_seconds"]
            and ledger.get("initial_budget_charged_cpu_seconds") == carry["budget_charged_cpu_seconds"]
            and ledger.get("conservative_debit_cpu_seconds") == carry["conservative_debit_cpu_seconds"], "Historical actual cost and conservative debit cannot be reset")
    phases = ledger["phases"]
    require(len({r["phase"] for r in phases}) == len(phases) and sum(r["status"] == "RUNNING" for r in phases) <= 1, "Duplicate/multiple live phase")
    require(not ledger.get("blocked_unknown_cpu_or_process_tree"), "Unknown actual CPU/closure requires forensic repair")
    require(all(r["status"] in ("RUNNING", "EXITED", "FAILED_CLOSED") for r in phases), "Unresolved phase state")
    closed = [r for r in phases if r["status"] != "RUNNING"]
    require(all(r.get("CPU_accounting") == ACCOUNTING and isinstance(r.get("actual_cpu_seconds"), (int, float)) and math.isfinite(r["actual_cpu_seconds"]) and r["actual_cpu_seconds"] >= 0 for r in closed), "Unreconciled closed phase cost")
    for row in closed:
        verify_wait4_receipt(row)
    actual = carry["actual_closed_cpu_seconds"] + sum(r["actual_cpu_seconds"] for r in closed)
    used = actual + carry["conservative_debit_cpu_seconds"]
    require(math.isclose(actual, ledger["actual_closed_cpu_seconds"], abs_tol=1e-6)
            and math.isclose(used, ledger["budget_charged_cpu_seconds"], abs_tol=1e-6)
            and math.isclose(used, ledger["consumed_cpu_seconds"], abs_tol=1e-6)
            and math.isclose(max(0, CAP - used), ledger["remaining_cpu_seconds"], abs_tol=1e-6), "CPU totals must retain actual costs and separate conservative debit")
    require(used <= CAP, "Actual cap overrun retained; no continuation")
    return ledger


def materialize(argv, output, results):
    values = []
    for index, token in enumerate(argv):
        if index == 0:
            values.append(str(python_path(token)))
        elif token == "{output}":
            values.append(str(output))
        elif TOKEN.fullmatch(token):
            match = TOKEN.fullmatch(token)
            value = results[match[2]]
            checked(value)
            values.append(str(path(value["path"])) if match[1] == "result" else value["sha256"])
        else:
            values.append(str(path(token)) if index == 1 else token)
    return values


def verify_resource_supervision(*, expected_worker_path, expected_threads=2, expected_job_id=None):
    """Workers call this before numerical work; return value belongs in result.json."""
    supplied = os.environ.get("R4_B_DIAGNOSTIC_RESOURCE_PROOF")
    require(supplied, "Actual diagnostic supervisor proof required")
    proof = read(supplied)
    require(proof.get("schema") == PROOF_SCHEMA and proof.get("supervisor_sha256") == sha(SELF), "Wrong/source-changed diagnostic proof")
    manifest = read(proof["manifest_path"], proof["manifest_sha256"])
    carry = validate(manifest, proof["manifest_sha256"])
    me = identity(os.getpid()); parent = identity(os.getppid())
    require(same(me, proof["child_identity"]) and same(parent, proof["supervisor_identity"]), "Actual child/parent birth/UID/boot mismatch")
    require(me["pid"] == me["pgrp"] == me["session"], "Dedicated single-TGID session required")
    tokens = [v.decode(errors="replace") for v in (Path("/proc") / str(os.getppid()) / "cmdline").read_bytes().split(b"\0") if v]
    require(str(SELF) in tokens and "--execute" in tokens, "Actual parent is not this pinned supervisor")
    require(Path(os.path.abspath(sys.executable)) == python_path(manifest["python_executable"]), "Wrong worker Python selector")
    require(checked(proof["worker"]) == path(expected_worker_path), "Wrong actual worker source")
    require(expected_threads == proof["threads"] == manifest["threads"], "Thread limit differs")
    if expected_job_id is not None:
        require(proof["job_id"] == expected_job_id, "Wrong diagnostic job")
    ledger = validate_ledger(read(proof["ledger_path"]), manifest, proof["manifest_sha256"], carry)
    require(path(proof["ledger_path"]) == path(manifest["ledger"]), "Proof names another ledger")
    require("--manifest" in tokens and path(tokens[tokens.index("--manifest") + 1]) == path(proof["manifest_path"])
            and "--manifest-sha256" in tokens and tokens[tokens.index("--manifest-sha256") + 1] == proof["manifest_sha256"], "Actual parent argv differs from proof manifest")
    rows = [r for r in ledger["phases"] if r["status"] == "RUNNING"]
    require(len(rows) == 1 and rows[0]["phase"] == proof["phase"] and rows[0]["job_sha256"] == proof["job_sha256"], "Unique running phase binding absent")
    row = rows[0]
    job = next((j for j in manifest["jobs"] if j["id"] == proof["job_id"]), None)
    require(job is not None and row["job_id"] == proof["job_id"] and digest(job) == proof["job_sha256"], "Proof differs from pinned job")
    # The parent records identities immediately after Popen; allow only this
    # short metadata race before numerical work, never an unbound worker.
    for _ in range(20):
        if row.get("child_identity") and row.get("supervisor_identity"):
            break
        time.sleep(.01)
        ledger = validate_ledger(read(proof["ledger_path"]), manifest, proof["manifest_sha256"], carry)
        row = next(r for r in ledger["phases"] if r["phase"] == proof["phase"])
    require(same(row.get("child_identity", {}), me) and same(row.get("supervisor_identity", {}), parent), "Running row has different actual child/parent")
    require(proof["cpu_seconds"] == job["cpu_seconds"]
            and proof["cpu_soft_limit_seconds"] == job["cpu_seconds"] - min(60, max(5, job["cpu_seconds"] // 10))
            and proof["address_space_bytes"] == manifest["memory_gib"] * 1024 ** 3
            and proof["rss_stop_bytes"] == manifest["rss_stop_gib"] * 1024 ** 3
            and proof["deadline_utc"] == manifest["deadline_utc"]
            and proof["wall_seconds"] <= job["wall_seconds"]
            and proof["original_cpu_cap_seconds"] == PREDECESSOR_CAP
            and proof["additional_cpu_authorized_seconds"] == ADDITIONAL_CPU_SECONDS
            and proof["effective_cpu_cap_seconds"] == CAP
            and proof["extension_authorization_sha256"] == manifest["extension_authorization"]["sha256"],
            "Proof resource bounds or explicit CPU extension differ from pinned job/allocation")
    require(tuple(resource.getrlimit(resource.RLIMIT_CPU)) == (proof["cpu_soft_limit_seconds"], proof["cpu_seconds"]), "Actual CPU limits differ")
    require(tuple(resource.getrlimit(resource.RLIMIT_AS)) == (proof["address_space_bytes"],) * 2, "Actual address-space limits differ")
    require(time.time() < datetime.fromisoformat(proof["deadline_utc"]).timestamp(), "Original deadline expired")
    return {"proof_path": str(path(supplied)), "proof_sha256": sha(path(supplied)), "proof": proof,
            "live_parent_process_and_source_verified": True, "verified_at_utc": utc(), "same_cumulative_B_history_with_explicit_user_extension": True,
            "original_cpu_cap_seconds": PREDECESSOR_CAP, "additional_cpu_authorized_seconds": ADDITIONAL_CPU_SECONDS,
            "effective_cpu_cap_seconds": CAP, "extension_authorization_sha256": manifest["extension_authorization"]["sha256"]}


def bound_signal(expected, sig):
    if is_live(expected):
        os.killpg(expected["pgrp"], sig)
        return True
    return False


def verify_wait4_receipt(row):
    receipt = read(checked(row["wait4_receipt"]))
    require(receipt["schema"] == "research4-b-diagnostic-kernel-wait4-receipt-v1"
            and receipt["phase"] == row["phase"] and same(receipt["child_identity"], row["child_identity"])
            and same(receipt["supervisor_identity"], row["supervisor_identity"]), "Kernel wait4 receipt phase/process binding differs")
    actual = receipt["ru_utime_seconds"] + receipt["ru_stime_seconds"]
    require(all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in (receipt["ru_utime_seconds"], receipt["ru_stime_seconds"]))
            and math.isclose(actual, receipt["observed_wait4_cpu_seconds"], abs_tol=1e-9)
            and math.isclose(actual, row["actual_cpu_seconds"], abs_tol=1e-9)
            and math.isclose(actual, row["observed_wait4_cpu_seconds"], abs_tol=1e-9), "Kernel wait4 receipt actual CPU differs")
    require(receipt["exit_code"] == row["exit_code"] == os.waitstatus_to_exitcode(receipt["raw_wait_status"])
            and receipt["unsupported_subprocess_observed"] is False
            and receipt["observed_wait4_is_not_complete_subtree_proof"] is False, "Kernel wait4 receipt is not complete supported closure")
    return receipt


def seal_result(row, job):
    output = path(row["output_directory"])
    result_path = output / "result.json"
    result = read(result_path)
    require(row["status"] == "EXITED" and row["exit_code"] == 0, "Real successful closure required")
    verify_wait4_receipt(row)
    require(row["result_sha256"] == sha(result_path), "Closed result changed")
    if job.get("result_schema"):
        require(result.get("schema") == job["result_schema"], "Diagnostic result schema differs")
    if result.get("output_directory") is not None:
        require(path(result["output_directory"]) == output, "Diagnostic output binding differs")
    proof = read(row["proof_path"])
    require(sha(path(row["proof_path"])) == row["proof_sha256"] and sha(path(row["log"])) == row["log_sha256"], "Closed proof/log changed")
    require(proof["phase"] == row["phase"] and proof["job_sha256"] == row["job_sha256"] == digest(job), "Closed job binding differs")
    supervision = result.get("resource_supervision", {})
    require(supervision.get("proof") == proof and supervision.get("proof_sha256") == row["proof_sha256"] and supervision.get("live_parent_process_and_source_verified") is True, "Worker did not verify real supervision")
    if result.get("source_worker"):
        require(checked(result["source_worker"]) == path(job["argv"][1]), "Result worker binding differs")
    if job.get("smoke_success_gate"):
        qualification = result.get("scf_qualification", {})
        norm = qualification.get("orbital_gradient_norm_Eh")
        require(result.get("status") == "POINT_COMPUTED" and result.get("operation") in ("point", "fixed_grid_point")
                and result.get("step_bohr", 0) > 0 and result.get("sign") in (-1, 1)
                and result.get("scf_converged") is True and result.get("scf_qualified") is True
                and qualification.get("qualified") is True and qualification.get("orbital_gradient_finite") is True
                and type(norm) in (int, float) and math.isfinite(norm) and 0 <= norm <= 1e-8
                and qualification.get("qualification_norm_threshold_Eh") == 1e-8,
                "Smoke requires a converged displaced SCF and independent fresh-Fock qualification")
        for name in ("gradient.npy", "geometry_bohr.npy", "direction.npy", "predicted_Hq.npy"):
            artifact = checked(result["arrays"][name])
            require(artifact.parent == output and artifact.name == name and artifact.stat().st_size > 128,
                    "Smoke requires its complete source-verified full gradient/geometry/direction artifacts")
    files = []
    for p in sorted(output.rglob("*")):
        require(not p.is_symlink(), "Symlink diagnostic artifact rejected")
        if p.is_file():
            size = p.stat().st_size
            files.append({"relative_path": str(p.relative_to(output)), "sha256": sha(p), "size_bytes": size,
                          "upload_eligible": p.suffix.lower() not in (".chk", ".h5", ".hdf5", ".npz", ".pyc") and size <= 5 * 1024 ** 2})
    return result, {"result": ref(result_path), "phase": row["phase"], "closed_row_sha256": digest(row),
                    "files": files, "actual_cpu_seconds": row["actual_cpu_seconds"], "CPU_accounting": ACCOUNTING}


def run_payload(args, manifest, ledger, job, attempt, results, publish):
    no_other_workers(args)
    work = path(manifest["work_directory"])
    label = job["id"] + f"-attempt{attempt:03d}"
    output = work / "science" / label
    log = work / "execution" / (label + ".log")
    proof_path = work / "execution" / (label + ".proof.json")
    require(not any(p.exists() for p in (output, log, proof_path)), "Preserve previous attempt artifacts")
    cpu = job["cpu_seconds"]
    required_cpu = admission_cpu_required(manifest, job, results)
    require(ledger["remaining_cpu_seconds"] >= required_cpu + manifest["reserve_cpu_seconds"], "Insufficient original remaining allocation for complete point pair")
    deadline = datetime.fromisoformat(manifest["deadline_utc"]).timestamp()
    wall = min(job["wall_seconds"], math.floor(deadline - time.time()) - 2)
    require(wall >= 61, "No safe wall time remains")
    argv = materialize(job["argv"], output, results)
    supervisor = identity(os.getpid())
    phase = "validation-recovery-v6-" + label
    margin = min(60, max(5, cpu // 10))
    asbytes = manifest["memory_gib"] * 1024 ** 3
    row = {"phase": phase, "job_id": job["id"], "job_sha256": digest(job), "attempt": attempt, "status": "RUNNING",
           "started_at_utc": utc(), "command": argv, "output_directory": str(output), "log": str(log),
           "proof_path": str(proof_path), "cpu_seconds": cpu, "wall_seconds": wall, "actual_cpu_seconds": None,
           "admission_group": job.get("admission_group"), "admission_cpu_caps_required": required_cpu,
           "remaining_cpu_seconds_at_admission": ledger["remaining_cpu_seconds"], "reserve_cpu_seconds_at_admission": manifest["reserve_cpu_seconds"]}
    ledger["phases"].append(row)
    save(manifest["ledger"], ledger)
    libc = ctypes.CDLL(None, use_errno=True)

    def limits():
        require(libc.prctl(1, signal.SIGKILL, 0, 0, 0) == 0 and os.getppid() == supervisor["pid"], "Cannot bind worker lifetime to parent")
        resource.setrlimit(resource.RLIMIT_CPU, (cpu - margin, cpu))
        resource.setrlimit(resource.RLIMIT_AS, (asbytes,) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        proof = {"schema": PROOF_SCHEMA, "supervisor_sha256": sha(SELF), "supervisor_identity": supervisor,
                 "child_identity": identity(os.getpid()), "manifest_path": str(path(args.manifest)),
                 "manifest_sha256": args.manifest_sha256, "ledger_path": str(path(manifest["ledger"])),
                 "phase": phase, "job_id": job["id"], "job_sha256": digest(job), "worker": ref(job["argv"][1]),
                 "threads": manifest["threads"], "cpu_seconds": cpu, "cpu_soft_limit_seconds": cpu - margin,
                 "actual_cpu_stop_threshold_seconds": cpu - margin, "address_space_bytes": asbytes,
                 "rss_stop_bytes": manifest["rss_stop_gib"] * 1024 ** 3, "wall_seconds": wall,
                 "deadline_utc": manifest["deadline_utc"], "same_cumulative_B_history_with_explicit_user_extension": True,
                 "original_cpu_cap_seconds": PREDECESSOR_CAP, "additional_cpu_authorized_seconds": ADDITIONAL_CPU_SECONDS,
                 "effective_cpu_cap_seconds": CAP, "extension_authorization_sha256": manifest["extension_authorization"]["sha256"],
                 "single_TGID_native_threads_only": True}
        save(proof_path, proof, exclusive=True)

    env = dict(os.environ, R4_B_DIAGNOSTIC_RESOURCE_PROOF=str(proof_path), PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None); env.pop("PYTHONHOME", None)
    env.update({key: str(manifest["threads"]) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); waited = None; stop_reason = None; stop_at = None
    unsupported = False; peakcpu = 0.; peakrss = 0; last_save = started
    try:
        with log.open("x") as handle:
            child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                     start_new_session=True, preexec_fn=limits)
            expected = identity(child.pid)
            row.update(child_identity=expected, supervisor_identity=supervisor)
            save(manifest["ledger"], ledger)
            publish("VALIDATION_RECOVERY_V6_PAYLOAD_STARTED", {"phase": phase, "identity": expected})
            while True:
                observed_wait = os.wait4(child.pid, os.WNOHANG)
                if observed_wait[0]:
                    waited = observed_wait
                    child.returncode = os.waitstatus_to_exitcode(waited[1])
                    # Preserve kernel evidence *before* any subtree/accounting
                    # decision. A blocked tree must not discard real rusage.
                    receipt_path = work / "execution" / (label + ".wait4.json")
                    receipt = {"schema": "research4-b-diagnostic-kernel-wait4-receipt-v1", "phase": phase,
                               "observed_at_utc": utc(), "supervisor_identity": supervisor, "child_identity": expected,
                               "raw_wait_status": waited[1], "exit_code": child.returncode,
                               "ru_utime_seconds": waited[2].ru_utime, "ru_stime_seconds": waited[2].ru_stime,
                               "observed_wait4_cpu_seconds": waited[2].ru_utime + waited[2].ru_stime,
                               "ru_maxrss_kib": waited[2].ru_maxrss,
                               "unsupported_subprocess_observed": unsupported,
                               "observed_wait4_is_not_complete_subtree_proof": unsupported}
                    save(receipt_path, receipt, exclusive=True)
                    row.update(wait4_receipt=ref(receipt_path), observed_wait4_cpu_seconds=receipt["observed_wait4_cpu_seconds"])
                    save(manifest["ledger"], ledger)
                    break
                try:
                    observed = identity(child.pid)
                    require(same(expected, observed), "Worker identity changed")
                    extra_members = [p for p in session_members(expected) if not same(expected, p)]
                    if extra_members:
                        unsupported = True; stop_reason = stop_reason or "UNSUPPORTED_SUBPROCESS_TREE"
                        if "unexpected_group_members" not in row:
                            row["unexpected_group_members"] = [{"identity": member, "observed_at_utc": utc()} for member in extra_members]
                            save(manifest["ledger"], ledger)
                    peakcpu = max(peakcpu, observed["CPU_seconds"])
                    peakrss = max(peakrss, observed["rss_bytes"])
                    if observed["CPU_seconds"] >= cpu - margin:
                        stop_reason = stop_reason or "ACTUAL_CPU_GUARD"
                    if observed["rss_bytes"] >= manifest["rss_stop_gib"] * 1024 ** 3:
                        stop_reason = stop_reason or "RSS_GUARD"
                    if time.monotonic() - started >= wall - 30 or time.time() >= deadline - 30:
                        stop_reason = stop_reason or "WALL_GUARD"
                    if time.process_time() > 600:
                        stop_reason = stop_reason or "SUPERVISOR_CPU_GUARD"
                except (OSError, ValueError):
                    stop_reason = stop_reason or "PROCESS_OBSERVATION_UNCERTAIN"
                if stop_reason and stop_at is None:
                    stop_at = time.monotonic()
                    row.update(stop_reason=stop_reason, TERM_sent=bound_signal(expected, signal.SIGTERM))
                    save(manifest["ledger"], ledger)
                if stop_at is not None and time.monotonic() - stop_at >= 1:
                    row["KILL_sent"] = bound_signal(expected, signal.SIGKILL)
                    if time.monotonic() - stop_at > 3:
                        unsupported = True
                        break
                row.update(latest_CPU_seconds=peakcpu, peak_RSS_bytes=peakrss)
                if time.monotonic() - last_save >= 30:
                    save(manifest["ledger"], ledger); last_save = time.monotonic()
                time.sleep(.1)
    except BaseException:
        # A failed Popen/preexec or parent observation is never a zero-cost run.
        ledger["blocked_unknown_cpu_or_process_tree"] = True
        row.update(status="BLOCKED_UNACCOUNTED", actual_cpu_lower_bound_seconds=peakcpu,
                   CPU_accounting="UNRESOLVED_NOT_ZERO", ended_at_utc=utc())
        save(manifest["ledger"], ledger)
        raise
    if waited is None or session_members(expected):
        unsupported = True
    if unsupported:
        ledger["blocked_unknown_cpu_or_process_tree"] = True
        row.update(status="BLOCKED_UNACCOUNTED", actual_cpu_lower_bound_seconds=peakcpu,
                   CPU_accounting="UNRESOLVED_NOT_ZERO", ended_at_utc=utc())
        save(manifest["ledger"], ledger)
        raise RuntimeError("No trustworthy wait4/single-TGID closure; automatic restart forbidden")
    actual = waited[2].ru_utime + waited[2].ru_stime
    require(math.isfinite(actual) and actual >= 0, "Invalid real wait4 CPU")
    row.update(status="EXITED" if child.returncode == 0 and stop_reason is None else "FAILED_CLOSED",
               exit_code=child.returncode, stop_reason=stop_reason, actual_cpu_seconds=actual,
               CPU_accounting=ACCOUNTING, ended_at_utc=utc(), elapsed_wall_seconds=time.monotonic() - started,
               peak_RSS_bytes=peakrss, log_sha256=sha(log), proof_sha256=sha(proof_path))
    if (output / "result.json").is_file():
        row["result_sha256"] = sha(output / "result.json")
    ledger["consumed_cpu_seconds"] += actual
    ledger["actual_closed_cpu_seconds"] += actual
    ledger["budget_charged_cpu_seconds"] += actual
    ledger["remaining_cpu_seconds"] = max(0, CAP - ledger["consumed_cpu_seconds"])
    save(manifest["ledger"], ledger)
    require(ledger["consumed_cpu_seconds"] <= CAP, "Actual cap overrun retained")
    publish("VALIDATION_RECOVERY_V6_PAYLOAD_CLOSED", {"phase": phase, "status": row["status"], "actual_cpu_seconds": actual})
    return row


def register_allocation(manifest, manifest_sha, carry):
    p = path(".runtime/worker-b-validation-recovery-v6-allocation.json")
    value = {"run_id": RUN_ID, "manifest_sha256": manifest_sha, "ledger": manifest["ledger"],
             "cpu_cap_seconds": CAP, "carry": carry, "work_directory": manifest["work_directory"],
             "predecessor_allocation": manifest["predecessor_allocation"],
             "predecessor_ledger": manifest["predecessor_ledger"], "predecessor_state": manifest["predecessor_state"],
             "recovery_boot_id": manifest["recovery_boot_id"], "inherited_results_sha256": digest(manifest["inherited_results"])}
    extension_path = path(EXTENSION_CLAIM)
    extension = {"schema": "research4-b-single-use-cpu-extension-claim-v1", "run_id": RUN_ID,
        "grant_id": carry["extension_grant_id"], "extension_authorization": manifest["extension_authorization"],
        "original_cpu_cap_seconds": PREDECESSOR_CAP, "additional_cpu_authorized_seconds": ADDITIONAL_CPU_SECONDS,
        "effective_cpu_cap_seconds": CAP, "extension_is_not_actual_cpu_consumption": True,
        "predecessor_allocation": manifest["predecessor_allocation"], "manifest_sha256": manifest_sha,
        "ledger": manifest["ledger"], "work_directory": manifest["work_directory"], "recovery_boot_id": manifest["recovery_boot_id"]}
    # Check both existing durable claims before creating either new marker.
    if p.exists():
        require(read(p) == value, "Existing recovery allocation cannot be reset/replaced")
    if extension_path.exists():
        require(read(extension_path) == extension, "Single-use user CPU extension cannot be claimed again by another manifest/ledger")
    if not extension_path.exists():
        save(extension_path, extension, exclusive=True)
    if not p.exists():
        save(p, value, exclusive=True)


def closed_retry_decision(row, job):
    """Permit only evidenced operational retries; preserve scientific outcomes."""
    require(row["status"] in ("EXITED", "FAILED_CLOSED"), "Retry requires real closed phase")
    verify_wait4_receipt(row)
    decision = {"allowed": False, "reason": "SCIENTIFIC_OR_UNCLASSIFIED_OUTCOME_NOT_RETRIED",
                "decided_at_utc": utc(), "wait4_receipt": row["wait4_receipt"]}
    if row.get("stop_reason"):
        decision["reason"] = "CLOSED_RESOURCE_OR_OBSERVATION_GUARD_NOT_RETRIED"
        return decision
    result_path = path(row["output_directory"]) / "result.json"
    if not result_path.exists():
        decision.update(allowed=True, reason="CLOSED_PROCESS_WITHOUT_RESULT_OPERATIONAL_RETRY")
        return decision
    require(sha(result_path) == row.get("result_sha256"), "Retry result changed after closure")
    try:
        result = read(result_path)
    except (ValueError, json.JSONDecodeError):
        decision["reason"] = "INVALID_RESULT_REQUIRES_REVIEW_NOT_AUTOMATIC_RETRY"
        return decision
    status = result.get("status")
    decision.update(result=ref(result_path), observed_status=status, exception_type=result.get("exception_type"))
    if status == "SCF_NOT_QUALIFIED":
        decision["reason"] = "SCIENTIFIC_SCF_QUALIFICATION_RECORDED_NOT_RETRIED"
    elif status in job.get("success_statuses", []):
        decision["reason"] = "SCIENTIFIC_COMPUTATION_COMPLETE_NOT_RETRIED"
    elif status == "PARTIAL" and (result.get("quantum_payload_entered") is False
            or result.get("exception_type") in ("OSError", "IOError", "MemoryError", "TimeoutError", "ImportError", "ModuleNotFoundError")):
        decision.update(allowed=True, reason="CLOSED_OPERATIONAL_EXCEPTION_RETRY")
    return decision


def admission_cpu_required(manifest, job, results):
    """Reserve both unsatisfied signs before starting or retrying a pair."""
    group = job.get("admission_group")
    if not group:
        return job["cpu_seconds"]
    return sum(j["cpu_seconds"] for j in manifest["jobs"]
               if j.get("admission_group") == group and j["id"] not in results)


def block_admission_group(manifest, state, job):
    group = job.get("admission_group")
    members = [j for j in manifest["jobs"] if group and j.get("admission_group") == group] if group else [job]
    for member in members:
        if member["id"] not in state["results"]:
            state["jobs"].setdefault(member["id"], {"status": "BLOCKED_RESOURCE_OR_DEADLINE"}).update(
                admission_group=group, admission_blocked="COMPLETE_PAIR_CAPS_PLUS_RESERVE_REQUIRED" if group else "RESOURCE_OR_DEADLINE")


def return_results(args, manifest, state, work):
    upload = manifest.get("upload", {})
    if not upload.get("enabled"):
        return
    ledger = read(manifest["ledger"])
    require(not any(r["status"] == "RUNNING" for r in ledger["phases"]), "No Git return while a diagnostic phase is active")
    no_other_workers(args)
    helper = checked(upload["helper"])
    return_manifest = checked(upload["manifest"])
    for index in range(1, 4):
        attempt = args.attempt + f"-return-{index:03d}"
        command = [str(python_path(manifest["python_executable"])), str(helper), "--repo", str(ROOT),
                   "--manifest", str(return_manifest), "--manifest-sha256", upload["manifest"]["sha256"],
                   "--attempt", attempt, "--execute"]
        try:
            completed = subprocess.run(command, cwd=ROOT, timeout=300, check=False)
            value = {"status": "HELPER_EXITED0_VERIFY_HELPER_RECEIPT" if completed.returncode == 0 else "BLOCKED_UPLOAD_FAILED_SCIENCE_RETAINED",
                     "returncode": completed.returncode, "attempt": attempt, "attempted_at_utc": utc()}
        except subprocess.TimeoutExpired:
            value = {"status": "BLOCKED_UPLOAD_TIMEOUT_SCIENCE_RETAINED", "attempt": attempt, "attempted_at_utc": utc()}
        state.setdefault("git_return_attempts", []).append(value)
        save(work / "state.json", state)
        if value.get("returncode") == 0:
            break
        if index < 3:
            time.sleep(2)


def run(args, manifest):
    work = path(manifest["work_directory"])
    work.mkdir(parents=True, exist_ok=True)
    with exclusive_lock():
        no_other_workers(args)
        carry = validate(manifest, args.manifest_sha256)
        ledger_path = path(manifest["ledger"])
        allocation_exists = (path(".runtime/worker-b-validation-recovery-v6-allocation.json").exists()
                             or path(EXTENSION_CLAIM).exists())
        # A persistent claim is evidence that diagnostics may already have
        # spent CPU. A lost ledger can never be recreated from the old carry.
        require(not allocation_exists or ledger_path.is_file(), "Diagnostic allocation exists but ledger is missing; forensic recovery required")
        if ledger_path.exists():
            ledger = validate_ledger(read(ledger_path), manifest, args.manifest_sha256, carry)
        else:
            ledger = {"schema": LEDGER_SCHEMA, "run_id": RUN_ID, "manifest_sha256": args.manifest_sha256,
                      "supervisor_sha256": sha(SELF), "cpu_cap_seconds": CAP, "deadline_utc": manifest["deadline_utc"],
                      "original_cpu_cap_seconds": PREDECESSOR_CAP, "additional_cpu_authorized_seconds": ADDITIONAL_CPU_SECONDS,
                      "extension_authorization": manifest["extension_authorization"],
                      "same_cumulative_B_history_with_explicit_user_extension": True,
                      "carry": carry, "initial_actual_closed_cpu_seconds": carry["actual_closed_cpu_seconds"],
                      "initial_conservative_debit_cpu_seconds": carry["conservative_debit_cpu_seconds"],
                      "initial_budget_charged_cpu_seconds": carry["budget_charged_cpu_seconds"],
                      "actual_closed_cpu_seconds": carry["actual_closed_cpu_seconds"],
                      "conservative_debit_cpu_seconds": carry["conservative_debit_cpu_seconds"],
                      "budget_charged_cpu_seconds": carry["budget_charged_cpu_seconds"],
                      "consumed_cpu_seconds": carry["budget_charged_cpu_seconds"],
                      "consumed_cpu_seconds_is_budget_charge_not_actual_wait4": True,
                      "remaining_cpu_seconds": CAP - carry["budget_charged_cpu_seconds"], "phases": [], "created_at_utc": utc()}
            save(ledger_path, ledger, exclusive=True)
        register_allocation(manifest, args.manifest_sha256, carry)
        require(not any(r["status"] == "RUNNING" for r in ledger["phases"]), "Missing PID is not phase closure; perform forensics")
        state_path = work / "state.json"
        state = read(state_path) if state_path.exists() else {"schema": "research4-b-validation-recovery-state-v6", "manifest_sha256": args.manifest_sha256,
                    "jobs": {}, "results": dict(manifest["inherited_results"]), "invocations": []}
        require(state["manifest_sha256"] == args.manifest_sha256 and len(state["invocations"]) < 8, "Fixed manifest and at most8 preserved invocations")
        require(all(state["results"].get(key) == value for key, value in manifest["inherited_results"].items()),
                "Inherited v4 success references cannot be removed or replaced")
        require(all(i["status"] in ("CLOSED", "CLOSED_WITH_BLOCKED_JOBS") for i in state["invocations"]), "Unresolved prior parent requires forensics")
        require(args.attempt not in [i["id"] for i in state["invocations"]], "Fresh invocation ID required")
        invocation = {"id": args.attempt, "identity": identity(os.getpid()), "status": "RUNNING", "started_at_utc": utc()}
        state["invocations"].append(invocation)

        def publish(event, details):
            state.update(last_event=event, updated_at_utc=utc())
            save(state_path, state)
            with (work / "monitor.jsonl").open("a") as handle:
                handle.write(json.dumps({"utc": utc(), "event": event, "details": details}, allow_nan=False) + "\n")
                handle.flush(); os.fsync(handle.fileno())

        publish("VALIDATION_RECOVERY_V6_QUEUE_STARTED_SAME_B_BUDGET", {})
        try:
            for job in manifest["jobs"]:
                carry = validate(manifest, args.manifest_sha256)
                validate_ledger(ledger, manifest, args.manifest_sha256, carry)
                previous = state["jobs"].get(job["id"], {})
                if previous.get("status") == "DONE_SEALED":
                    rows = [r for r in ledger["phases"] if r["phase"] == previous["phase"]]
                    require(len(rows) == 1, "Successful diagnostic lacks unique actual phase")
                    _, sealed = seal_result(rows[0], job)
                    require(sealed == previous["seal"], "Sealed diagnostic checkpoint changed")
                    checked(state["results"][job["id"]])
                    continue
                if not set(job.get("depends_on", [])).issubset(state["results"]):
                    state["jobs"][job["id"]] = {"status": "BLOCKED_DEPENDENCY"}
                    publish("VALIDATION_RECOVERY_V6_DEPENDENCY_BLOCKED", {"job_id": job["id"]}); continue
                if ledger["remaining_cpu_seconds"] < admission_cpu_required(manifest, job, state["results"]) + manifest["reserve_cpu_seconds"] or time.time() + 61 >= datetime.fromisoformat(manifest["deadline_utc"]).timestamp() or shutil.disk_usage(ROOT).free < 200 * 1024 ** 2:
                    block_admission_group(manifest, state, job)
                    publish("VALIDATION_RECOVERY_V6_RESOURCE_BLOCKED", {"job_id": job["id"], "admission_group": job.get("admission_group")}); continue
                attempts = [r for r in ledger["phases"] if r["job_id"] == job["id"]]
                require(all(r["status"] in ("EXITED", "FAILED_CLOSED") for r in attempts), "Unclosed diagnostic attempt")
                maximum = job.get("max_attempts", 1)
                if len(attempts) >= maximum:
                    publish("VALIDATION_RECOVERY_V6_ATTEMPT_LIMIT_RETAINED", {"job_id": job["id"]}); continue
                if attempts:
                    previous_decision = closed_retry_decision(attempts[-1], job)
                    if attempts[-1].get("automatic_retry_decision", {}).get("allowed") is not True or previous_decision["allowed"] is not True:
                        publish("VALIDATION_RECOVERY_V6_PRIOR_OUTCOME_NOT_RETRIED", {"job_id": job["id"], "decision": previous_decision})
                        continue
                while len(attempts) < maximum:
                    # A retry can only spend the remainder of this same ledger
                    # after the prior kernel wait4 receipt has been reverified.
                    validate_ledger(ledger, manifest, args.manifest_sha256, carry)
                    require(all(r["status"] in ("EXITED", "FAILED_CLOSED") for r in attempts), "Unclosed validation retry forbidden")
                    if ledger["remaining_cpu_seconds"] < admission_cpu_required(manifest, job, state["results"]) + manifest["reserve_cpu_seconds"] or time.time() + 61 >= datetime.fromisoformat(manifest["deadline_utc"]).timestamp() or shutil.disk_usage(ROOT).free < 200 * 1024 ** 2:
                        state["jobs"].setdefault(job["id"], {"status": "BLOCKED_RESOURCE_OR_DEADLINE"}).update(retry_blocked="RESOURCE_OR_DEADLINE")
                        publish("VALIDATION_RECOVERY_V6_RETRY_RESOURCE_BLOCKED", {"job_id": job["id"]}); break
                    row = run_payload(args, manifest, ledger, job, len(attempts) + 1, state["results"], publish)
                    row["automatic_retry_decision"] = closed_retry_decision(row, job)
                    save(manifest["ledger"], ledger)
                    current = {"status": row["status"], "phase": row["phase"], "actual_cpu_seconds": row["actual_cpu_seconds"]}
                    state["jobs"][job["id"]] = current
                    if row["status"] == "EXITED":
                        try:
                            result, seal = seal_result(row, job)
                            good = not job.get("success_statuses") or result.get("status") in job["success_statuses"]
                            current.update(status="DONE_SEALED" if good else "PARTIAL_SCIENTIFIC_CLOSED", seal=seal,
                                           actual_scientific_status=result.get("status"))
                            if good:
                                state["results"][job["id"]] = seal["result"]
                        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                            # Missing/invalid result is a known *spent*, closed job.
                            current.update(status="FAILED_RESULT_VALIDATION_CLOSED", reason=str(error))
                    publish("VALIDATION_RECOVERY_V6_JOB_CLOSED", {"job_id": job["id"], "status": current["status"]})
                    attempts.append(row)
                    if current["status"] == "DONE_SEALED":
                        break
                    if not row["automatic_retry_decision"]["allowed"]:
                        publish("VALIDATION_RECOVERY_V6_SCIENTIFIC_OR_UNCLASSIFIED_OUTCOME_NOT_RETRIED", {"job_id": job["id"], "decision": row["automatic_retry_decision"]})
                        break
                    if len(attempts) < maximum:
                        publish("VALIDATION_RECOVERY_V6_CLOSED_FAILURE_RETRY", {"job_id": job["id"], "next_attempt": len(attempts) + 1})
            invocation["status"] = "CLOSED" if len(state["jobs"]) == len(manifest["jobs"]) and all(j["status"] == "DONE_SEALED" for j in state["jobs"].values()) else "CLOSED_WITH_BLOCKED_JOBS"
            publish("VALIDATION_RECOVERY_V6_FINITE_QUEUE_CLOSED", {"status": invocation["status"]})
        except BaseException as error:
            invocation.update(status="BLOCKED_REQUIRES_FORENSICS", error_type=type(error).__name__, reason=str(error))
            publish("VALIDATION_RECOVERY_V6_QUEUE_BLOCKED_PRESERVED", {"reason": str(error)})
            raise
        finally:
            invocation["ended_at_utc"] = utc()
            publish("VALIDATION_RECOVERY_V6_INVOCATION_ENDED", {})
        return_results(args, manifest, state, work)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--attempt", default="validation-recovery-v6-invocation-001")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    require(SAFE.fullmatch(args.attempt), "Safe invocation ID required")
    manifest = read(args.manifest, args.manifest_sha256)
    carry = validate(manifest, args.manifest_sha256)
    if not args.execute or args.check_only:
        print(json.dumps({"status": "CHECKED_NOT_RUN", "jobs": len(manifest["jobs"]), "aggregate_B_cap_seconds": CAP,
                          "original_cpu_cap_seconds": PREDECESSOR_CAP, "additional_cpu_authorized_seconds": ADDITIONAL_CPU_SECONDS,
                          "effective_cpu_cap_seconds": CAP, "extension_authorization_sha256": manifest["extension_authorization"]["sha256"],
                          "historical_carry_cpu_seconds": carry["actual_closed_cpu_seconds"],
                          "conservative_debit_cpu_seconds": carry["conservative_debit_cpu_seconds"],
                          "budget_charged_before_validation_recovery_cpu_seconds": carry["budget_charged_cpu_seconds"],
                          "remaining_before_validation_recovery_cpu_seconds": CAP - carry["budget_charged_cpu_seconds"],
                          "deadline_utc": manifest["deadline_utc"], "new_quantum_evaluations": 0}))
        return
    require(time.time() < datetime.fromisoformat(manifest["deadline_utc"]).timestamp(), "Original B deadline expired")
    branch = subprocess.check_output(["git", "-C", str(ROOT), "branch", "--show-current"], text=True).strip()
    require(branch == BRANCH, "Only original B result branch may execute diagnostics")
    if args.detach:
        # The detached child rechecks under the same allocation lock.
        work = path(manifest["work_directory"])
        work.mkdir(parents=True, exist_ok=True)
        receipt = work / ("detached-" + args.attempt + ".json")
        console = work / ("detached-" + args.attempt + ".console.log")
        require(not receipt.exists() and not console.exists(), "Preserve previous launch receipts")
        command = [str(python_path(manifest["python_executable"])), str(SELF), "--manifest", str(path(args.manifest)),
                   "--manifest-sha256", args.manifest_sha256, "--attempt", args.attempt, "--execute"]
        with console.open("x") as handle:
            child = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        value = {"status": "ACTUAL_VALIDATION_RECOVERY_V6_PARENT_LAUNCHED_NOT_SCIENCE_COMPLETE", "identity": identity(child.pid),
                 "manifest_sha256": args.manifest_sha256, "supervisor_sha256": sha(SELF), "launched_at_utc": utc(),
                 "same_cumulative_B_history_with_explicit_user_extension": True, "machine_shutdown_survival_not_guaranteed": True}
        value.update(original_cpu_cap_seconds=PREDECESSOR_CAP, additional_cpu_authorized_seconds=ADDITIONAL_CPU_SECONDS,
                     effective_cpu_cap_seconds=CAP, extension_authorization_sha256=manifest["extension_authorization"]["sha256"])
        save(receipt, value, exclusive=True); save(work / "current-launch.json", value)
        print(json.dumps(value))
        return
    run(args, manifest)


if __name__ == "__main__":
    main()
