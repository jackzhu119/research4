#!/usr/bin/env python3
"""Reviewed recovery of diagnostic follow-up; retain real costs and conservative holds.

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
CAP = 58800
MANIFEST_SCHEMA = "research4-b-diagnostic-recovery-manifest-v2"
LEDGER_SCHEMA = "research4-b-diagnostic-recovery-cpu-ledger-v2"
PROOF_SCHEMA = "research4-b-diagnostic-recovery-resource-proof-v2"
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
            if not any(any(k in name for k in ("pyscf", "quantum", "parallel_b_supervisor", "b_diagnostic", "cloud_orbital_stability")) for name in scripts):
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


def validate_original_carry(manifest):
    original_ledger = checked(manifest["carry_ledger"])
    original_state = checked(manifest["carry_state"])
    require(original_ledger == path(ORIGINAL_WORK + "/cpu_budget_B.json") and original_state == path(ORIGINAL_WORK + "/state.json"), "Carry must name the actual original allocation")
    old = read(original_ledger)
    state = read(original_state)
    require(old.get("run_id") == RUN_ID and old.get("cpu_cap_seconds") == CAP, "Carry allocation identity changed")
    require(old.get("deadline_utc") == DEADLINE and not old.get("blocked_unknown_cpu_or_process_tree"), "Original deadline/accounting unresolved")
    require(old.get("initial_consumed_cpu_seconds", 0) == 0, "Unexpected original allocation carry")
    phases = old["phases"]
    require(phases and len({r["phase"] for r in phases}) == len(phases), "Invalid original phase history")
    require(all(r.get("status") in ("EXITED", "FAILED_CLOSED") and r.get("CPU_accounting") == ACCOUNTING
                and isinstance(r.get("actual_cpu_seconds"), (int, float)) and math.isfinite(r["actual_cpu_seconds"])
                and r["actual_cpu_seconds"] >= 0 for r in phases), "Original actual costs are not completely closed")
    actual = sum(r["actual_cpu_seconds"] for r in phases)
    require(math.isclose(actual, old["consumed_cpu_seconds"], abs_tol=1e-6) and actual <= CAP, "Original carry does not reconcile")
    require(state.get("invocations") and all(i["status"] in ("CLOSED", "CLOSED_WITH_BLOCKED_JOBS") for i in state["invocations"]), "Original parent not closed")
    for row in phases:
        if row.get("child_identity"):
            require(not is_live(row["child_identity"]) and not session_members(row["child_identity"]), "Original quantum process still active")
    for invocation in state["invocations"]:
        if invocation.get("identity"):
            require(not is_live(invocation["identity"]), "Original parent still active")
    # The carry is a sealed scientific history, not just arithmetic in JSON.
    original_work = path(ORIGINAL_WORK)
    prior_manifest = read(original_work / "manifest.json", old["manifest_sha256"])
    require(state.get("manifest_sha256") == old["manifest_sha256"], "Original state/ledger manifest binding differs")
    prepared = read(original_work / "prepared-launch.json")
    require(prepared["manifest_sha256"] == old["manifest_sha256"], "Original prepared manifest differs")
    authorization = read(original_work / "authorization.json", prepared["authorization_sha256"])
    require(authorization.get("enable_execution") is True and authorization["manifest_sha256"] == old["manifest_sha256"]
            and authorization["supervisor_sha256"] == old["supervisor_sha256"], "Original authorization/source differs")
    original_supervisor = path("scripts/research4_parallel_b_supervisor_20261106.py")
    require(sha(original_supervisor) == old["supervisor_sha256"], "Original supervisor source changed")
    for source in prior_manifest.get("source_refs", []) + prior_manifest.get("input_refs", []):
        checked(source)
    jobs = {j["id"]: j for j in prior_manifest["jobs"]}
    for job_id, reference in state.get("results", {}).items():
        result_file = checked(reference)
        saved = state["jobs"][job_id]
        require(saved["status"] == "DONE_SEALED" and saved["seal"]["result"] == reference, "Original successful result lost its seal")
        matching = [r for r in phases if r["phase"] == saved["phase"]]
        require(len(matching) == 1, "Original result lacks unique actual closed row")
        row = matching[0]; seal = saved["seal"]
        require(row["status"] == "EXITED" and row["exit_code"] == 0 and row["result_sha256"] == reference["sha256"], "Original result/real exit binding differs")
        require(seal["closed_row_sha256"] == digest(row) and row["case_binding_sha256"] == digest(jobs[job_id]), "Original closed row/configuration seal differs")
        proof = read(row["proof_path"], row["proof_sha256"])
        require(sha(path(row["log"])) == row["log_sha256"], "Original closed log differs")
        require(proof["phase"] == row["phase"] and proof["case_binding_sha256"] == row["case_binding_sha256"]
                and proof["manifest_sha256"] == old["manifest_sha256"] and proof["supervisor_source_sha256"] == old["supervisor_sha256"], "Original producer proof/configuration differs")
        result = read(result_file)
        actual_supervision = result.get("resource_supervision", {})
        require(actual_supervision.get("proof") == proof and actual_supervision.get("proof_sha256") == row["proof_sha256"]
                and actual_supervision.get("live_parent_process_and_source_verified") is True, "Original result producer proof differs")
        output = path(row["output_directory"])
        require(output.is_relative_to(original_work / "science") and result_file == output / "result.json", "Original scientific output escaped its stage")
        sealed_names = set()
        for artifact in seal["files"]:
            artifact_file = path(output / artifact["relative_path"])
            require(artifact_file.is_relative_to(output), "Original sealed artifact escaped its stage")
            require(artifact_file.stat().st_size == artifact["size_bytes"] and sha(artifact_file) == artifact["sha256"], "Original sealed scientific artifact changed: " + str(artifact_file))
            sealed_names.add(artifact_file)
        require(sealed_names == {p for p in output.rglob("*") if p.is_file()}, "Original sealed scientific file set differs")
    claim = read(ORIGINAL_WORK + "/owner_claim.json")
    owner = read(".runtime/worker-b-ownership/" + RUN_ID + ".json")
    require(owner.get("remote_verified") is True and owner.get("branch") == BRANCH and owner.get("repository_root") == str(ROOT), "Verified original local B owner required")
    require(all(owner.get(k) == claim.get(k) and claim.get(k) is not None for k in ("run_id", "nonce", "release_sha256", "source_commit")), "Original ownership proof differs")
    return {"original_ledger": manifest["carry_ledger"], "original_state": manifest["carry_state"], "actual_closed_cpu_seconds": actual}


def validate_carry(manifest):
    """Import frozen successful science; debit the unresolved attempt separately."""
    original = validate_original_carry(manifest)
    prior_manifest_path = checked(manifest["predecessor_manifest"])
    prior_ledger_path = checked(manifest["predecessor_ledger"])
    prior_state_path = checked(manifest["predecessor_state"])
    prior_source = checked(manifest["predecessor_supervisor"])
    prior_allocation_path = checked(manifest["predecessor_allocation"])
    resolution_path = checked(manifest["forensic_resolution"])
    prior = read(prior_manifest_path)
    old = read(prior_ledger_path)
    state = read(prior_state_path)
    allocation = read(prior_allocation_path)
    resolution = read(resolution_path)
    require(prior_source == path("scripts/research4_b_diagnostic_supervisor_20261106.py")
            and prior_allocation_path == path(".runtime/worker-b-diagnostic-allocation.json"), "Exact frozen predecessor source/allocation required")
    require(old["schema"] == "research4-b-diagnostic-cpu-ledger-v1" and old["run_id"] == RUN_ID
            and old["cpu_cap_seconds"] == CAP and old["deadline_utc"] == DEADLINE, "Predecessor allocation differs")
    require(old["manifest_sha256"] == manifest["predecessor_manifest"]["sha256"] == state["manifest_sha256"]
            and old["supervisor_sha256"] == manifest["predecessor_supervisor"]["sha256"], "Predecessor source/configuration binding differs")
    require(path(old.get("carry", {}).get("original_ledger", {}).get("path", "")) == path(manifest["carry_ledger"]["path"])
            and old["carry"]["original_ledger"]["sha256"] == manifest["carry_ledger"]["sha256"]
            and old["carry"]["original_state"]["sha256"] == manifest["carry_state"]["sha256"]
            and math.isclose(old["initial_consumed_cpu_seconds"], original["actual_closed_cpu_seconds"], abs_tol=1e-6), "Predecessor historical carry differs")
    require(allocation["manifest_sha256"] == old["manifest_sha256"] and path(allocation["ledger"]) == prior_ledger_path
            and path(allocation["work_directory"]) == path(prior["work_directory"])
            and allocation["cpu_cap_seconds"] == CAP and allocation["carry"] == old["carry"], "Persistent predecessor allocation claim differs")
    for value in prior.get("source_refs", []) + prior.get("input_refs", []):
        checked(value)
    phases = old["phases"]
    require(len({r["phase"] for r in phases}) == len(phases), "Duplicate predecessor phases")
    unresolved = [r for r in phases if r["status"] not in ("EXITED", "FAILED_CLOSED")]
    require(len(unresolved) == 1 and unresolved[0]["status"] == "BLOCKED_UNACCOUNTED"
            and unresolved[0].get("actual_cpu_seconds") is None and old.get("blocked_unknown_cpu_or_process_tree") is True,
            "Recovery is limited to the one reviewed unresolved predecessor attempt")
    failed = unresolved[0]
    require(resolution.get("approved_for_recovery") is True and resolution.get("processes_verified_closed") is True
            and resolution["blocked_phase"] == failed["phase"] and resolution.get("actual_wait4_cpu_seconds") is None,
            "Explicit forensic closure and non-fabricated CPU record required")
    for key, reference in (("predecessor_manifest_sha256", "predecessor_manifest"),
                           ("predecessor_ledger_sha256", "predecessor_ledger"), ("predecessor_state_sha256", "predecessor_state")):
        require(resolution[key] == manifest[reference]["sha256"], "Forensic resolution binds another predecessor")
    preserved = resolution.get("preserved_files")
    require(isinstance(preserved, list) and preserved, "Byte-preserved predecessor forensic evidence required")
    for entry in preserved:
        first = checked(entry["original"]); copy = checked(entry["preserved_copy"])
        require(entry["original"]["sha256"] == entry["preserved_copy"]["sha256"]
                and first.stat().st_size == entry["original"]["bytes"]
                and copy.stat().st_size == entry["preserved_copy"]["bytes"]
                and first.stat().st_size == copy.stat().st_size, "Original and byte-preserved forensic evidence differ")
    debit = resolution["conservative_debit_cpu_seconds"]
    require(isinstance(debit, (int, float)) and math.isfinite(debit) and debit == failed["cpu_seconds"] == 1000,
            "The entire reviewed failed CPU request remains permanently debited")
    closed = [r for r in phases if r["status"] in ("EXITED", "FAILED_CLOSED")]
    require(all(r.get("CPU_accounting") == ACCOUNTING and isinstance(r.get("actual_cpu_seconds"), (int, float))
                and math.isfinite(r["actual_cpu_seconds"]) and r["actual_cpu_seconds"] >= 0 for r in closed), "Predecessor actual closed costs unresolved")
    actual = original["actual_closed_cpu_seconds"] + sum(r["actual_cpu_seconds"] for r in closed)
    require(math.isclose(actual, old["consumed_cpu_seconds"], abs_tol=1e-6), "Predecessor closed costs do not reconcile")
    for row in phases:
        if row.get("child_identity"):
            require(not is_live(row["child_identity"]) and not session_members(row["child_identity"]), "Predecessor process/group remains live")
    require(state.get("invocations") and all(i["status"] != "RUNNING" for i in state["invocations"]), "Predecessor invocation still running")
    for invocation in state["invocations"]:
        require(not is_live(invocation["identity"]), "Predecessor parent remains live")
    require(manifest.get("inherited_results") == state["results"], "Inherited results must exactly retain all predecessor successful references")
    # Call the immutable predecessor's own sealing implementation, then compare
    # its historical seal; never create a fresh seal for changed results.
    spec = importlib.util.spec_from_file_location("r4_b_frozen_diagnostic_predecessor", prior_source)
    predecessor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predecessor)
    jobs = {j["id"]: j for j in prior["jobs"]}
    for job_id, reference in state["results"].items():
        checked(reference)
        saved = state["jobs"][job_id]
        require(saved["status"] == "DONE_SEALED" and saved["seal"]["result"] == reference, "Predecessor success lacks its historical seal")
        rows = [r for r in closed if r["phase"] == saved["phase"]]
        require(len(rows) == 1, "Predecessor success lacks unique actual closed phase")
        _, seal = predecessor.seal_result(rows[0], jobs[job_id])
        require(seal == saved["seal"], "Frozen predecessor scientific seal differs")
    pending = [j for j in prior["jobs"] if j["id"] not in state["results"]]
    require([j["id"] for j in manifest["jobs"]] == [j["id"] for j in pending], "Recovery must retain exactly the uncompleted predecessor job scope/order")
    for job in manifest["jobs"]:
        prior_job = jobs[job["id"]]
        used_attempts = sum(r["job_id"] == job["id"] for r in phases)
        require(job.get("max_attempts", 1) <= prior_job.get("max_attempts", 1) - used_attempts,
                "Recovery cannot reset historical attempt limits")
        require(path(job["argv"][1]) == path("scripts/research4_b_diagnostic_worker_recovery_20261107.py")
                and job["argv"][0] == prior_job["argv"][0] and job["argv"][2:] == prior_job["argv"][2:],
                "Only the reviewed worker implementation may change; requests/arguments stay fixed")
        require({k: v for k, v in job.items() if k not in ("argv", "worker", "max_attempts")}
                == {k: v for k, v in prior_job.items() if k not in ("argv", "worker", "max_attempts")},
                "Recovery cannot expand scientific parameters, dependencies, resource limits or success criteria")
    charged = actual + debit
    require(charged <= CAP, "Reviewed conservative debit exhausts allocation")
    return {"original_carry": original, "predecessor_manifest": manifest["predecessor_manifest"],
            "predecessor_ledger": manifest["predecessor_ledger"], "predecessor_state": manifest["predecessor_state"],
            "predecessor_supervisor": manifest["predecessor_supervisor"], "predecessor_allocation": manifest["predecessor_allocation"],
            "forensic_resolution": manifest["forensic_resolution"], "actual_closed_cpu_seconds": actual,
            "conservative_debit_cpu_seconds": debit, "budget_charged_cpu_seconds": charged,
            "conservative_debit_is_not_actual_wait4": True}


def validate(manifest, manifest_sha):
    require(manifest.get("schema") == MANIFEST_SCHEMA and manifest.get("run_id") == RUN_ID, "Exact B diagnostic manifest required")
    require(manifest.get("branch") == BRANCH and manifest.get("cpu_cap_seconds") == CAP, "Same branch and aggregate cap required")
    deadline = datetime.fromisoformat(manifest["deadline_utc"])
    require(deadline.tzinfo is not None and deadline <= datetime.fromisoformat(DEADLINE), "Diagnostic deadline cannot extend original grant")
    require(manifest.get("threads") == 2 and manifest.get("memory_gib") == 12 and manifest.get("rss_stop_gib") == 8, "Original B thread/memory bounds required")
    require(isinstance(manifest.get("reserve_cpu_seconds"), (int, float)) and manifest["reserve_cpu_seconds"] >= 60, "Explicit residual CPU reserve required")
    work = path(manifest["work_directory"])
    require(work.parent == path(ORIGINAL_WORK) and work.name == "followup-diagnostics-recovery-v2", "Reviewed recovery work directory is fixed under original B run")
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
    inherited = set(manifest.get("inherited_results", {}))
    for job in manifest["jobs"]:
        require(SAFE.fullmatch(job["id"]) and job["id"] not in ids and job["id"] not in inherited, "Unique new diagnostic job IDs required")
        require(set(job.get("depends_on", [])).issubset(set(ids) | inherited), "Dependencies must precede their job or be sealed predecessor results")
        ids.append(job["id"])
        require(isinstance(job["cpu_seconds"], int) and 20 <= job["cpu_seconds"] <= CAP, "Bounded integer job CPU required")
        require(isinstance(job["wall_seconds"], int) and 61 <= job["wall_seconds"] <= 72 * 3600, "Bounded job wall time required")
        require(isinstance(job.get("max_attempts", 1), int) and 1 <= job.get("max_attempts", 1) <= 3, "At most3 preserved attempts per diagnostic")
        require(isinstance(job["argv"], list) and len(job["argv"]) >= 2 and all(isinstance(x, str) for x in job["argv"]), "Explicit worker argv required")
        require(python_path(job["argv"][0]) == python_path(manifest["python_executable"]), "Worker must use qualified B Python")
        worker = path(job["argv"][1])
        require(worker.suffix == ".py" and any(checked(r) == worker for r in refs), "Actual worker must be a pinned Python source")
        require("{output}" in job["argv"], "Worker must receive its unique output directory")
        if job.get("worker"):
            require(checked(job["worker"]) == worker, "Worker ref differs from argv")
    require(ids and len(ids) <= 100, "Finite nonempty diagnostic queue required")
    require(re.fullmatch(r"[0-9a-f]{64}", manifest_sha), "Pinned manifest SHA-256 required")
    return validate_carry(manifest)


def validate_ledger(ledger, manifest, manifest_sha, carry):
    require(ledger.get("schema") == LEDGER_SCHEMA and ledger.get("manifest_sha256") == manifest_sha and ledger.get("supervisor_sha256") == sha(SELF), "Ledger/source binding changed")
    require(ledger.get("run_id") == RUN_ID and ledger.get("cpu_cap_seconds") == CAP and ledger.get("deadline_utc") == manifest["deadline_utc"], "Ledger allocation changed")
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
            and proof["wall_seconds"] <= job["wall_seconds"], "Proof resource bounds differ from pinned job/allocation")
    require(tuple(resource.getrlimit(resource.RLIMIT_CPU)) == (proof["cpu_soft_limit_seconds"], proof["cpu_seconds"]), "Actual CPU limits differ")
    require(tuple(resource.getrlimit(resource.RLIMIT_AS)) == (proof["address_space_bytes"],) * 2, "Actual address-space limits differ")
    require(time.time() < datetime.fromisoformat(proof["deadline_utc"]).timestamp(), "Original deadline expired")
    return {"proof_path": str(path(supplied)), "proof_sha256": sha(path(supplied)), "proof": proof,
            "live_parent_process_and_source_verified": True, "verified_at_utc": utc(), "same_original_B_budget": True}


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
    require(ledger["remaining_cpu_seconds"] >= cpu + manifest["reserve_cpu_seconds"], "Insufficient original remaining allocation")
    deadline = datetime.fromisoformat(manifest["deadline_utc"]).timestamp()
    wall = min(job["wall_seconds"], math.floor(deadline - time.time()) - 2)
    require(wall >= 61, "No safe wall time remains")
    argv = materialize(job["argv"], output, results)
    supervisor = identity(os.getpid())
    phase = "diagnostic-" + label
    margin = min(60, max(5, cpu // 10))
    asbytes = manifest["memory_gib"] * 1024 ** 3
    row = {"phase": phase, "job_id": job["id"], "job_sha256": digest(job), "attempt": attempt, "status": "RUNNING",
           "started_at_utc": utc(), "command": argv, "output_directory": str(output), "log": str(log),
           "proof_path": str(proof_path), "cpu_seconds": cpu, "wall_seconds": wall, "actual_cpu_seconds": None}
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
                 "deadline_utc": manifest["deadline_utc"], "same_original_B_budget": True,
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
            publish("DIAGNOSTIC_PAYLOAD_STARTED", {"phase": phase, "identity": expected})
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
    publish("DIAGNOSTIC_PAYLOAD_CLOSED", {"phase": phase, "status": row["status"], "actual_cpu_seconds": actual})
    return row


def register_allocation(manifest, manifest_sha, carry):
    p = path(".runtime/worker-b-diagnostic-recovery-allocation.json")
    value = {"run_id": RUN_ID, "manifest_sha256": manifest_sha, "ledger": manifest["ledger"],
             "cpu_cap_seconds": CAP, "carry": carry, "work_directory": manifest["work_directory"],
             "predecessor_allocation": manifest["predecessor_allocation"],
             "predecessor_ledger": manifest["predecessor_ledger"], "predecessor_state": manifest["predecessor_state"],
             "forensic_resolution": manifest["forensic_resolution"]}
    if p.exists():
        require(read(p) == value, "Existing diagnostic allocation cannot be reset/replaced")
    else:
        save(p, value, exclusive=True)


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
        allocation_exists = path(".runtime/worker-b-diagnostic-recovery-allocation.json").exists()
        # A persistent claim is evidence that diagnostics may already have
        # spent CPU. A lost ledger can never be recreated from the old carry.
        require(not allocation_exists or ledger_path.is_file(), "Diagnostic allocation exists but ledger is missing; forensic recovery required")
        if ledger_path.exists():
            ledger = validate_ledger(read(ledger_path), manifest, args.manifest_sha256, carry)
        else:
            ledger = {"schema": LEDGER_SCHEMA, "run_id": RUN_ID, "manifest_sha256": args.manifest_sha256,
                      "supervisor_sha256": sha(SELF), "cpu_cap_seconds": CAP, "deadline_utc": manifest["deadline_utc"],
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
        state = read(state_path) if state_path.exists() else {"schema": "research4-b-diagnostic-recovery-state-v2", "manifest_sha256": args.manifest_sha256,
                    "jobs": {}, "results": dict(manifest["inherited_results"]), "inherited_results": dict(manifest["inherited_results"]), "invocations": []}
        require(state["manifest_sha256"] == args.manifest_sha256 and len(state["invocations"]) < 8, "Fixed manifest and at most8 preserved invocations")
        require(state.get("inherited_results") == manifest["inherited_results"]
                and all(state["results"].get(k) == v for k, v in manifest["inherited_results"].items()), "Inherited successful results changed")
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

        publish("DIAGNOSTIC_QUEUE_STARTED_SAME_B_BUDGET", {})
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
                    publish("DIAGNOSTIC_DEPENDENCY_BLOCKED", {"job_id": job["id"]}); continue
                if ledger["remaining_cpu_seconds"] < job["cpu_seconds"] + manifest["reserve_cpu_seconds"] or time.time() + 61 >= datetime.fromisoformat(manifest["deadline_utc"]).timestamp() or shutil.disk_usage(ROOT).free < 200 * 1024 ** 2:
                    state["jobs"][job["id"]] = {"status": "BLOCKED_RESOURCE_OR_DEADLINE"}
                    publish("DIAGNOSTIC_RESOURCE_BLOCKED", {"job_id": job["id"]}); continue
                attempts = [r for r in ledger["phases"] if r["job_id"] == job["id"]]
                require(all(r["status"] in ("EXITED", "FAILED_CLOSED") for r in attempts), "Unclosed diagnostic attempt")
                maximum = job.get("max_attempts", 1)
                if len(attempts) >= maximum:
                    publish("DIAGNOSTIC_ATTEMPT_LIMIT_RETAINED", {"job_id": job["id"]}); continue
                # At most one new attempt for each job per queue invocation.
                row = run_payload(args, manifest, ledger, job, len(attempts) + 1, state["results"], publish)
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
                publish("DIAGNOSTIC_JOB_CLOSED", {"job_id": job["id"], "status": current["status"]})
            invocation["status"] = "CLOSED" if len(state["jobs"]) == len(manifest["jobs"]) and all(j["status"] == "DONE_SEALED" for j in state["jobs"].values()) else "CLOSED_WITH_BLOCKED_JOBS"
            publish("DIAGNOSTIC_FINITE_QUEUE_CLOSED", {"status": invocation["status"]})
        except BaseException as error:
            invocation.update(status="BLOCKED_REQUIRES_FORENSICS", error_type=type(error).__name__, reason=str(error))
            publish("DIAGNOSTIC_QUEUE_BLOCKED_PRESERVED", {"reason": str(error)})
            raise
        finally:
            invocation["ended_at_utc"] = utc()
            publish("DIAGNOSTIC_INVOCATION_ENDED", {})
        return_results(args, manifest, state, work)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--attempt", default="diagnostic-invocation-001")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    require(SAFE.fullmatch(args.attempt), "Safe invocation ID required")
    manifest = read(args.manifest, args.manifest_sha256)
    carry = validate(manifest, args.manifest_sha256)
    if not args.execute or args.check_only:
        print(json.dumps({"status": "CHECKED_NOT_RUN", "jobs": len(manifest["jobs"]), "aggregate_B_cap_seconds": CAP,
                          "historical_carry_cpu_seconds": carry["actual_closed_cpu_seconds"],
                          "conservative_debit_cpu_seconds": carry["conservative_debit_cpu_seconds"],
                          "budget_charged_before_recovery_cpu_seconds": carry["budget_charged_cpu_seconds"],
                          "remaining_before_diagnostics_cpu_seconds": CAP - carry["budget_charged_cpu_seconds"],
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
        value = {"status": "ACTUAL_DIAGNOSTIC_PARENT_LAUNCHED_NOT_SCIENCE_COMPLETE", "identity": identity(child.pid),
                 "manifest_sha256": args.manifest_sha256, "supervisor_sha256": sha(SELF), "launched_at_utc": utc(),
                 "same_original_B_budget": True, "machine_shutdown_survival_not_guaranteed": True}
        save(receipt, value, exclusive=True); save(work / "current-launch.json", value)
        print(json.dumps(value))
        return
    run(args, manifest)


if __name__ == "__main__":
    main()
