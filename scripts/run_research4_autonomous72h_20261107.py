#!/usr/bin/env python3
"""Finite serial background research, with authenticated closure before reuse/Git.

Default: read-only preflight. Root releases an exact config hash to execute.
Every scientific/statistical job runs once through the new 72h guardian.
Detachment survives terminal disconnect, never promises host availability.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import gzip
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / "reports/research72h-20261107-v1"
SELF = Path(__file__).resolve()
GUARD = ROOT / "scripts/run_research4_research72h_budget_20261107.py"
AUTH = PREFIX / "resource-protocol/user_authorization_observed-001.json"
AUTH_SHA = "058c24d076567b14054f78467c001a7858ef2601a070f3c5b97c8b074bc9981d"
LEDGER = PREFIX / "cpu_budget.json"
BRANCH = "research/pyscf-spinosin-mechanism-20261031"
SCHEMA = "research4-autonomous72h-config-v1"
SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,69}\Z")
TOKEN = re.compile(r"\{(result|sha256):([A-Za-z0-9_-]+)\}\Z")
FORMATS = {".json", ".jsonl", ".csv", ".md", ".log", ".png", ".pdf", ".xyz", ".npy"}
BLOCKED_PARTS = {"cache", "caches", "stage-cache", "models", "weights", "raw", "originals", ".git", "__pycache__"}
GIT_DEADLINE = None
DETACH_PARENT_ENV = "R4_AUTONOMOUS72H_DETACH_PARENT"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc)


def sha(file):
    digest = hashlib.sha256()
    with Path(file).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def path(value):
    raw = Path(value)
    candidate = raw if raw.is_absolute() else ROOT / raw
    require(".." not in raw.parts and not any(p.is_symlink() for p in (candidate, *candidate.parents)), "Symlink/traversal path rejected")
    candidate = candidate.resolve()
    require(candidate.is_relative_to(ROOT), "Path escapes repository")
    return candidate


def read(file):
    file = path(file)
    require(file.stat().st_size <= 8*1048576, "Oversized metadata; use a small manifest")
    return json.loads(file.read_text())


def atomic(file, value):
    file = path(file)
    require(file.is_relative_to(PREFIX), "Only new report prefix may be written")
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(file.name+".writing")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(file)


def checked(ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "Explicit path/hash reference required")
    file = path(ref["path"])
    require(file.is_file() and sha(file) == ref["sha256"], "Frozen input/source changed: "+str(file.relative_to(ROOT)))
    return file


def guard_module(ref):
    file = checked(ref)
    require(file == GUARD, "Only the new registered 72h guardian may launch jobs")
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("autonomous72h_frozen_guard", file)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@contextmanager
def lock(file):
    file = path(file); file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def identity(pid):
    directory = Path("/proc") / str(pid)
    value = (directory/"stat").read_text(); fields = value[value.rfind(")")+2:].split()
    uid = next(v for v in (directory/"status").read_text().splitlines() if v.startswith("Uid:")).split()[1]
    return {"pid": int(pid), "start_ticks": int(fields[19]), "uid": int(uid), "state": fields[0],
            "comm": value[value.find("(")+1:value.rfind(")")],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def same_identity(first, second):
    return all(first[key] == second[key] for key in ("pid", "start_ticks", "uid", "boot_id"))


def proc_argv(pid):
    return [v.decode(errors="replace") for v in (Path("/proc")/str(pid)/"cmdline").read_bytes().split(b"\0") if v]


def wait_for_detacher(config_path, config_hash):
    """Wait <=1s only for the exact non-scientific launcher; exempt nobody."""
    raw = os.environ.pop(DETACH_PARENT_ENV, None)
    if raw is None:
        return
    require(len(raw) <= 4096, "Oversized detach identity receipt")
    receipt = json.loads(raw); expected = receipt["identity"]
    require(receipt["source_sha256"] == sha(SELF) and path(receipt["config_path"]) == path(config_path)
            and receipt["config_sha256"] == config_hash and expected["uid"] == os.getuid(), "Detach source/config/UID binding differs")
    deadline = time.monotonic()+1.0
    while True:
        try:
            actual = identity(expected["pid"])
        except (FileNotFoundError, ProcessLookupError):
            return
        if not same_identity(expected, actual) or actual["state"] in ("Z", "X", "x"):
            return  # Original launcher ended; ordinary scanning still checks any replacement.
        require(os.getppid() == expected["pid"], "Detach identity is not the actual parent launcher")
        argv = proc_argv(expected["pid"])
        require(any(v.endswith(".py") and path(v) == SELF for v in argv) and "--execute" in argv and "--detach" in argv
                and argv.count("--config") == argv.count("--config-sha256") == 1
                and path(argv[argv.index("--config")+1]) == path(config_path)
                and argv[argv.index("--config-sha256")+1] == config_hash,
                "Detach parent is not the exact bound non-scientific launcher")
        require(time.monotonic() < deadline, "Identified detach launcher did not exit within1s; fail closed")
        time.sleep(.02)


def process_scan(config):
    """Independent live check; a quiet ledger alone is never sufficient."""
    names = {path(j["workerref"]["path"]).name for j in config["jobs"]} | {GUARD.name}
    owned = []
    if LEDGER.exists():
        for row in read(LEDGER).get("phases", []):
            child = row.get("child_identity", {})
            if all(key in child for key in ("pid", "start_ticks", "uid")) and row.get("boot_id"):
                owned.append({**child, "boot_id": row["boot_id"]})
    active = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            first = identity(int(directory.name))
            if first["state"] in ("Z", "X", "x"):
                continue
            tokens = proc_argv(int(directory.name))
            scripts = [Path(v).name for v in tokens[1:] if v.endswith(".py")]
            relevant = [s for s in scripts if s in names or any(k in s for k in
                        ("pyscf", "quantum", "fixed_grid", "fixed_quadrature", "full_native_checkpoint", "run_", "train_", "cloud_followup_worker"))]
            if any(same_identity(first, child) for child in owned):
                relevant.append("LEDGER_BOUND_OWNED_CHILD")
            if relevant or (tokens and Path(tokens[0]).name == "git"):
                require(same_identity(first, identity(int(directory.name))), "Unstable process identity")
                active.append({**first, "script_basenames": relevant, "argv_not_recorded": True})
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            first, second = identity(int(directory.name)), identity(int(directory.name))
            require(same_identity(first, second) and first["uid"] == 0 and first["comm"] in ("dockerd", "containerd"),
                    "Process visibility insufficient to prove scientific closure")
    require(not active, "Live scientific/guard/Git processes prevent launch or Git return: "+json.dumps(active))
    return {"observed_at_utc": utc().isoformat(), "active_scientific_processes": []}


def validate_config(config, guard):
    require(config.get("schema") == SCHEMA, "Unknown driver config schema")
    require(checked(config["guardref"]) == GUARD, "Changed frozen guardian source")
    work = path(config["workroot"])
    require(work.is_relative_to(PREFIX) and work != PREFIX, "Dedicated child workroot required")
    require(path(config["ledger"]) == LEDGER and checked(config["authref"]) == AUTH
            and config["authref"]["sha256"] == AUTH_SHA, "Exact observed authorization and new ledger required")
    refs = config.get("source_refs", []) + config.get("immutable_inputs", [])
    require(any(path(r["path"]) == SELF and r["sha256"] == sha(SELF) for r in config.get("source_refs", [])), "Driver source must be pinned")
    for ref in refs:
        checked(ref)
    authorization = read(AUTH)
    require(config.get("deadline_utc") == authorization["run_deadline_at_utc"], "Fixed observed 72h deadline required")
    jobs = config.get("jobs")
    require(isinstance(jobs, list) and 1 <= len(jobs) <= 64, "Finite 1..64 job list required")
    seen = set()
    for job in jobs:
        name = job["id"]
        require(isinstance(name, str) and SAFE.fullmatch(name) and name not in seen, "Unique safe job ID required")
        require(job["kind"] in ("quantum", "ml", "analysis"), "Explicit job kind required")
        worker = checked(job["workerref"])
        require(worker.suffix == ".py", "Pinned Python worker required")
        deps = job.get("depends_on", [])
        require(isinstance(deps, list) and len(set(deps)) == len(deps) and set(deps).issubset(seen), "Dependencies must refer to earlier jobs")
        argv = job["argv"]
        require(isinstance(argv, list) and argv and all(isinstance(t, str) and "\0" not in t for t in argv), "Explicit shell-free argv required")
        require(argv.count("{output}") == 1 and argv.count("--output") == 1 and argv.index("--output")+1 < len(argv)
                and argv[argv.index("--output")+1] == "{output}", "Worker must own one fresh --output directory")
        scripts = [(i, path(v)) for i, v in enumerate(argv) if v.endswith(".py")]
        require(scripts and scripts[0][1] == worker and all(v in ("-B", "-u") for v in argv[1:scripts[0][0]]), "Argv must directly execute the pinned worker")
        require(Path(argv[0]).name.startswith(("python", "pypy")) and not any(v in argv for v in ("-c", "-m")), "Direct Python script invocation required")
        for token in argv:
            match = TOKEN.fullmatch(token)
            require("{" not in token or token == "{output}" or (match and match[2] in deps), "Invalid/unbound argv placeholder")
        require(type(job["cpu_seconds"]) is int and 125 <= job["cpu_seconds"] <= 60000, "Finite guardian CPU grant 125..60000 required")
        require(type(job["wall_seconds"]) is int and 61 <= job["wall_seconds"] <= 28800, "Finite guardian wall grant 61..28800 required")
        require(type(job["threads"]) is int and 1 <= job["threads"] <= 2, "One or two threads required")
        require(isinstance(job.get("success_statuses"), list) and job["success_statuses"], "Explicit scientific success statuses required")
        require(isinstance(job.get("required_result_fields", {}), dict), "Required result fields must be a dictionary")
        result_file = job.get("result_file", "result.json")
        require(Path(result_file).name == result_file and result_file.endswith(".json"), "Safe JSON result basename required")
        require(job.get("max_attempts", 1) == 1, "No automatic scientific retries")
        seen.add(name)
    require(sum(j["cpu_seconds"] for j in jobs) <= authorization["authorized_additional_cpu_seconds"], "Finite list exceeds fresh authorization")
    ledger = read(LEDGER); guard.validate_ledger(ledger)
    require(ledger["run_deadline_at_utc"] == config["deadline_utc"], "Ledger deadline differs")
    return work, ledger


def materialize(argv, output, results):
    actual = []
    for token in argv:
        if token == "{output}":
            actual.append(str(output)); continue
        match = TOKEN.fullmatch(token)
        if match:
            value = results[match[2]]
            require(sha(path(value["path"])) == value["sha256"], "Sealed dependency result changed")
            actual.append(value["path"] if match[1] == "result" else value["sha256"])
        else:
            actual.append(token)
    return actual


def qualify(job, result):
    require(result.get("status") in job["success_statuses"], "Scientific result did not qualify")
    for dotted, expected in job.get("required_result_fields", {}).items():
        value = result
        for key in dotted.split("."):
            require(isinstance(value, dict) and key in value, "Missing required result field: "+dotted)
            value = value[key]
        require(type(value) is type(expected) and value == expected, "Scientific gate failed: "+dotted)


def authenticate(guard, config, job, entry):
    closure = guard.verify_closed_row(path(config["ledger"]), entry["phase"], expected_worker_path=path(job["workerref"]["path"]),
        expected_worker_sha256=job["workerref"]["sha256"], expected_output=path(entry["output"]), allow_failure=True)
    ledger = read(LEDGER)
    row = next(r for r in ledger["phases"] if r["phase"] == entry["phase"])
    require(row["command"] == entry["argv"], "Actual guarded argv differs from the frozen materialization")
    return closure


def resume_state(state, config_hash, config, guard):
    require(state.get("config_sha256") == config_hash, "Resume config differs")
    known = {job["id"]: job for job in config["jobs"]}
    require(set(state.get("jobs", {})).issubset(known), "State contains unknown jobs")
    for name, entry in state.get("jobs", {}).items():
        require(entry["status"] in ("SUCCESS", "FAILED_CLOSED", "SKIPPED_DEPENDENCY", "SKIPPED_RESOURCE"), "Unknown/live/orphan phase requires explicit root forensics; no automatic resume")
        if entry["status"] in ("SUCCESS", "FAILED_CLOSED"):
            closure = authenticate(guard, config, known[name], entry)
            require(closure == entry["closure"], "Sealed actual closure changed")
            if entry["status"] == "SUCCESS":
                require(sha(path(entry["result"]["path"])) == entry["result"]["sha256"], "Sealed result changed")
                qualify(known[name], read(entry["result"]["path"]))
            if "observed_result" in entry:
                require(sha(path(entry["observed_result"]["path"])) == entry["observed_result"]["sha256"], "Observed successful/failed result changed")
    return state


def git(*args):
    timeout = min(120, (GIT_DEADLINE-utc()).total_seconds()) if GIT_DEADLINE else 120
    require(timeout > 0, "72h deadline reached; preserve local result for later authorized return")
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, timeout=timeout)
    require(result.returncode == 0, "Git operation failed: "+args[0]+"; preserve local artifacts/index")
    return result.stdout.decode().strip()


def return_files():
    selected, excluded = [], []
    for file in sorted(PREFIX.rglob("*")):
        if not file.is_file() or file.is_symlink():
            continue
        require(path(file) == file.resolve(), "Symlink output ancestor rejected")
        rel = file.relative_to(PREFIX)
        compressed_csv = file.name.lower().endswith(".csv.gz")
        hash_metadata = file.suffix.lower() == ".sha256"
        if any(p.lower() in BLOCKED_PARTS for p in rel.parts) or (file.suffix.lower() not in FORMATS and not compressed_csv and not hash_metadata):
            continue
        maximum = 65536 if hash_metadata else (1048576 if file.suffix.lower() == ".npy" else 5*1048576)
        if file.stat().st_size > maximum:
            excluded.append({"path": str(file.relative_to(ROOT)), "bytes": file.stat().st_size, "sha256": sha(file), "reason": "SIZE_LIMIT"})
        elif (compressed_csv and not genuine_csv_gzip(file)) or (hash_metadata and not genuine_hash_metadata(file)):
            excluded.append({"path": str(file.relative_to(ROOT)), "bytes": file.stat().st_size, "sha256": sha(file), "reason": "NOT_VALID_SMALL_CSV_OR_HASH_METADATA"})
        else:
            selected.append(file)
    require(sum(p.stat().st_size for p in selected) <= 40*1048576, "Small-artifact return exceeds 40MiB; preserve locally")
    return selected, excluded


def genuine_csv_gzip(file):
    """Accept bounded UTF-8 rectangular CSV, never arbitrary renamed gzip."""
    try:
        with gzip.open(file, "rb") as handle:
            raw = handle.read(32*1048576+1)
        if len(raw) > 32*1048576:
            return False
        rows = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
        header = next(rows, [])
        if len(header) < 2:
            return False
        observed = 0
        for row in rows:
            if len(row) != len(header):
                return False
            observed += 1
        return observed > 0
    except (OSError, EOFError, UnicodeError, csv.Error):
        return False


def genuine_hash_metadata(file):
    try:
        lines = file.read_text(encoding="ascii").splitlines()
        return bool(lines) and all(re.fullmatch(r"[0-9a-fA-F]{64}(?:[ \t]+\*?[^\x00-\x1f]+)?", line) for line in lines)
    except (OSError, UnicodeError):
        return False


def upload(config, guard, work, attempt):
    """One finite upload attempt; a transport failure never reruns science."""
    global GIT_DEADLINE
    GIT_DEADLINE = datetime.fromisoformat(config["deadline_utc"])
    receipt = work/"git-return"/(attempt+".json")
    value = {"status": "NOT_PUSHED", "at_utc": utc().isoformat(), "branch": BRANCH,
             "force_push": False, "no_main_merge": True, "raw_logs_preserved_verbatim": True}
    snapshots = {}
    try:
        with lock(ROOT/".runtime/pyscf-v4-global.lock"):
            guard.validate_ledger(read(LEDGER)); process_scan(config)
            require(git("branch", "--show-current") == BRANCH and not git("diff", "--cached", "--name-only"), "Exact research branch and initially empty staged index required")
            remote = git("remote", "get-url", "origin")
            remote = "https://github.com/"+remote.split(":", 1)[1] if remote.startswith("git@github.com:") else remote
            parsed = urlsplit(remote)
            require(parsed.hostname == "github.com" and parsed.path.removesuffix(".git") == "/jackzhu119/research4", "Expected GitHub repository required")
            files, excluded = return_files()
            snapshots = {str(f.relative_to(ROOT)): {"sha256": sha(f), "bytes": f.stat().st_size} for f in files}
            value.update(artifact_hashes=snapshots, excluded_files_retained_locally=excluded)
            if snapshots:
                process_scan(config); git("add", "--sparse", "--", *snapshots)
                staged = set(git("diff", "--cached", "--name-only").splitlines())
                require(staged.issubset(snapshots), "Unexpected staged paths")
                require(all(sha(ROOT/p) == s["sha256"] for p, s in snapshots.items()), "Artifacts changed during return")
                if staged:
                    process_scan(config)
                    git("-c", "user.email=195995062+jackzhu119@users.noreply.github.com", "commit", "-m", "Save bounded research72h actual closed results")
                commit = git("rev-parse", "HEAD")
                for relative, saved in snapshots.items():
                    blob = subprocess.run(["git", "-C", str(ROOT), "cat-file", "blob", commit+":"+relative], capture_output=True, timeout=30)
                    require(blob.returncode == 0 and hashlib.sha256(blob.stdout).hexdigest() == saved["sha256"], "Committed blob bytes differ")
                process_scan(config); git("push", "origin", "HEAD:refs/heads/"+BRANCH)
                require(any(line.split() == [commit, "refs/heads/"+BRANCH] for line in git("ls-remote", "--heads", "origin", BRANCH).splitlines()), "Remote branch did not verify")
                value.update(status="ACTUAL_PUSH_REMOTE_HEAD_VERIFIED", scientific_commit=commit, scientific_blobs_verified=True)
                atomic(receipt, value)
                process_scan(config); git("add", "--sparse", "--", str(receipt.relative_to(ROOT)))
                require(set(git("diff", "--cached", "--name-only").splitlines()) == {str(receipt.relative_to(ROOT))}, "Unexpected receipt staging")
                git("-c", "user.email=195995062+jackzhu119@users.noreply.github.com", "commit", "-m", "Record actual research72h result push receipt")
                process_scan(config); git("push", "origin", "HEAD:refs/heads/"+BRANCH)
                final_commit = git("rev-parse", "HEAD")
                require(any(line.split() == [final_commit, "refs/heads/"+BRANCH] for line in git("ls-remote", "--heads", "origin", BRANCH).splitlines()), "Receipt remote head did not verify")
                return {**value, "receipt_remote_head": final_commit}
    except Exception as error:
        value.update(status="UPLOAD_FAILED_LOCAL_SCIENCE_PRESERVED", error_type=type(error).__name__, error=str(error))
        atomic(receipt.with_name(receipt.stem+"-failure.json"), value)
    return value


def execute(config, config_hash, guard, attempt, resume, config_path):
    work, _ = validate_config(config, guard)
    with lock(ROOT/".runtime/research4-autonomous72h-driver.lock"):
        state_file = work/"state.json"
        require(resume == state_file.exists(), "Use --resume only for an existing, fully closed state")
        state = resume_state(read(state_file), config_hash, config, guard) if resume else {
            "schema": "research4-autonomous72h-state-v1", "config_sha256": config_hash, "jobs": {}, "uploads": []}
        work.mkdir(parents=True, exist_ok=True)
        receipt = work/"launches"/(attempt+".json")
        require(SAFE.fullmatch(attempt) and not receipt.exists(), "Fresh safe launch attempt required")
        live = {"identity": identity(os.getpid()), "config_sha256": config_hash, "source_sha256": sha(SELF),
                "started_at_utc": utc().isoformat(), "detachment_does_not_survive_host_shutdown": True,
                "effective_pythonpath": os.environ.get("PYTHONPATH", ""), "python_executable": sys.executable}
        atomic(receipt, live)
        def publish(event, details=None):
            state["updated_at_utc"] = utc().isoformat(); state["driver_identity"] = live["identity"]
            atomic(state_file, state)
            with (work/"events.jsonl").open("a") as handle:
                handle.write(json.dumps({"at_utc": utc().isoformat(), "event": event, **(details or {})}, allow_nan=False)+"\n")
                handle.flush(); os.fsync(handle.fileno())
        publish("DRIVER_STARTED", {"attempt": attempt, "resume": resume})
        for job in config["jobs"]:
            if job["id"] in state["jobs"]:
                continue
            require(sha(path(config_path)) == config_hash, "Frozen config changed between jobs")
            _, ledger = validate_config(config, guard); process_scan(config)
            if any(state["jobs"].get(d, {}).get("status") != "SUCCESS" for d in job.get("depends_on", [])):
                state["jobs"][job["id"]] = {"status": "SKIPPED_DEPENDENCY"}; publish("JOB_SKIPPED_DEPENDENCY", {"job": job["id"]}); continue
            remaining = (datetime.fromisoformat(config["deadline_utc"])-utc()).total_seconds()
            if ledger["remaining_quantum_cpu_seconds"] < job["cpu_seconds"]+2*job["threads"]+2 or remaining < job["wall_seconds"]+2:
                state["jobs"][job["id"]] = {"status": "SKIPPED_RESOURCE"}; publish("JOB_SKIPPED_RESOURCE", {"job": job["id"]}); continue
            output = work/"jobs"/job["id"]/"attempt-001"
            require(not output.exists(), "Do not overwrite or retry an earlier scientific output")
            output.parent.mkdir(parents=True, exist_ok=True)
            results = {name: value["result"] for name, value in state["jobs"].items() if value["status"] == "SUCCESS"}
            actual = materialize(job["argv"], output, results)
            phase = "r72-"+job["id"]+"-001"
            log = work/"logs"/(phase+".log")
            entry = {"status": "STARTING", "phase": phase, "output": str(output), "argv": actual}
            state["jobs"][job["id"]] = entry; publish("JOB_STARTING", {"job": job["id"]})
            command = [sys.executable, "-B", str(GUARD), "run", "--ledger", str(LEDGER), "--phase", phase,
                       "--log", str(log), "--cpu-seconds", str(job["cpu_seconds"]), "--wall-seconds", str(job["wall_seconds"]),
                       "--threads", str(job["threads"]), "--memory-gib", "24", "--", *actual]
            try:
                console = work/"logs"/(phase+"-guardian.log"); console.parent.mkdir(parents=True, exist_ok=True)
                with console.open("x") as handle:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                    entry.update(status="RUNNING", guardian_identity=identity(child.pid)); publish("JOB_RUNNING", {"job": job["id"]})
                    while child.poll() is None:
                        time.sleep(2)
                entry["guard_returncode"] = child.returncode
                entry["closure"] = authenticate(guard, config, job, entry)
                process_scan(config)
            except BaseException as error:
                entry.update(status="BLOCKED_UNKNOWN_CLOSURE", error_type=type(error).__name__, error=str(error))
                publish("UNRESOLVED_CLOSURE_STOP", {"job": job["id"]}); raise
            try:
                result_path = output/job.get("result_file", "result.json")
                if result_path.exists():
                    entry["observed_result"] = {"path": str(result_path), "sha256": sha(result_path)}
                require(entry["guard_returncode"] == 0 and entry["closure"].get("scientific_success") is True, "Actual guarded execution failed")
                result = read(result_path)
                qualify(job, result)
                if "output_directory" in result:
                    require(path(result["output_directory"]) == output, "Worker result output binding differs")
                entry.update(status="SUCCESS", result={"path": str(result_path), "sha256": sha(result_path)})
            except Exception as error:
                entry.update(status="FAILED_CLOSED", error_type=type(error).__name__, error=str(error))
            publish("JOB_ACTUALLY_CLOSED", {"job": job["id"], "status": entry["status"], "actual_cpu_seconds": entry["closure"]["actual_cpu_seconds"]})
            state["uploads"].append(upload(config, guard, work, attempt+"-"+job["id"])); publish("RETURN_ATTEMPT_FINISHED")
        state["status"] = "FINITE_QUEUE_COMPLETE_WITH_ALL_RESULTS_PRESERVED"; publish("DRIVER_FINISHED")
        state["uploads"].append(upload(config, guard, work, attempt+"-final")); publish("FINAL_RETURN_ATTEMPT_FINISHED")
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--attempt", default="launch-001"); parser.add_argument("--execute", action="store_true")
    parser.add_argument("--detach", action="store_true"); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    require(SAFE.fullmatch(args.attempt), "Safe launch attempt required")
    config_path = checked({"path": str(args.config), "sha256": args.config_sha256})
    wait_for_detacher(config_path, args.config_sha256)
    config = read(config_path); guard = guard_module(config["guardref"])
    work, ledger = validate_config(config, guard); process_scan(config)
    if not args.execute:
        require(not args.detach, "--detach requires explicit --execute")
        print(json.dumps({"status": "READ_ONLY_PREFLIGHT_NO_LAUNCH", "jobs": len(config["jobs"]),
                          "remaining_cpu_seconds": ledger["remaining_quantum_cpu_seconds"], "deadline": config["deadline_utc"]})); return
    if args.detach:
        require(not (work/"state.json").exists() or args.resume, "Existing state requires explicit sealed resume")
        work.mkdir(parents=True, exist_ok=True)
        receipt = work/(args.attempt+"-detached-launch.json")
        require(not receipt.exists(), "Preserve earlier detach receipt")
        argv = [sys.executable, "-B", str(SELF), "--execute", "--config", str(config_path), "--config-sha256", args.config_sha256,
                "--attempt", args.attempt] + (["--resume"] if args.resume else [])
        with (work/(args.attempt+"-driver.log")).open("x") as handle:
            env = dict(os.environ)
            env[DETACH_PARENT_ENV] = json.dumps({"identity": identity(os.getpid()), "source_sha256": sha(SELF),
                                               "config_path": str(config_path), "config_sha256": args.config_sha256})
            child = subprocess.Popen(argv, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True, stdin=subprocess.DEVNULL, env=env)
        actual = {"status": "DETACHED_PROCESS_OBSERVED_NOT_A_FUTURE_AVAILABILITY_GUARANTEE", "identity": identity(child.pid),
                  "config_sha256": args.config_sha256, "observed_at_utc": utc().isoformat(), "no_host_shutdown_survival_claim": True,
                  "effective_pythonpath": env.get("PYTHONPATH", ""), "python_executable": sys.executable}
        atomic(receipt, actual); print(json.dumps(actual)); return
    value = execute(config, args.config_sha256, guard, args.attempt, args.resume, config_path)
    print(json.dumps({"status": value["status"], "jobs": {k: v["status"] for k, v in value["jobs"].items()}}))


if __name__ == "__main__":
    main()
