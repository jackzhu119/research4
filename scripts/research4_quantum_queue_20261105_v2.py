"""Prepared, bounded queue for already materialized Research4 quantum jobs.

V2 records exact resource/signal stop reasons before signalling owned guardians.
External Git/ML observations defer new launches and never kill active science.
Default mode only checks the configuration. This is not an operating-system
startup hook or a user-notification service. Root must invoke the saved resume
command after environment recovery. No scientific index or unknown SHA is made.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "scripts/run_pyscf_continuation_budget_20261105_v6b.py"
GUARD_SHA = "b985d6d6e20f5e4a48387094461f19b133dae761338664e88836c62051f993a5"
SCHEMA = "research4-materialized-quantum-queue-v2"
GRANT_SCHEMA = "research4-materialized-quantum-queue-grant-v2"
MAX_JOBS, MAX_INVOCATIONS, MAX_SELF_CPU, MAX_WALL = 128, 3, 600, 86400
INTERVAL = 1800
WORKERS = {"pyscf_xc_chunk_worker_20261105_v7.py", "pyscf_staged_consumer_20261105_v7.py"}
FROZEN_A = {
    "scripts/pyscf_xc_chunk_checkpoint_20261105_v7.py": "da1e2b0267c48e748a2bcb7d3f2ee3110d8e2288aa340d23dab7a12c994ddf8e",
    "scripts/pyscf_xc_chunk_worker_20261105_v7.py": "00e710eb0d637dbd073939bea5d4496993cb68f29efad1439322df3972415f02"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc)


def aware(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, "An actual timezone-aware deadline is required")
    return result


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path(value):
    supplied = Path(value)
    require(not supplied.is_symlink(), "Symlink queue artifact rejected")
    result = supplied.resolve() if supplied.is_absolute() else (ROOT / supplied).resolve()
    require(result.is_relative_to(ROOT), "Queue files must stay inside this repository")
    return result


def read(file, expected=None, maximum=4 * 1024**2):
    file = path(file)
    require(file.is_file() and file.stat().st_size <= maximum, "Missing/oversized queue metadata")
    require(expected is None or sha(file) == expected, "Bound metadata SHA changed: " + str(file))
    return json.loads(file.read_text())


def save(file, value):
    file = path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(file.name + ".writing-" + str(os.getpid()))
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, file)
    descriptor = os.open(file.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def checked(ref):
    file = path(ref["path"])
    require(file.is_file() and sha(file) == ref["sha256"], "Bound source/payload changed: " + str(file))
    require("bytes" not in ref or file.stat().st_size == ref["bytes"], "Bound artifact size changed")
    return file


def artifact(file):
    file = path(file)
    return {"path": str(file), "sha256": sha(file), "bytes": file.stat().st_size}


def own_rss():
    line = next(v for v in Path("/proc/self/status").read_text().splitlines() if v.startswith("VmRSS:"))
    return int(line.split()[1]) * 1024


@contextmanager
def self_limits(grant):
    """Per-process timers are NOT inherited by forked guardian processes.

    Never lower an inherited RLIMIT_CPU hard cap: that would also cap the
    guardian, which must be able to supervise a much longer quantum job.
    A second CPU/wall expiry has default terminate action if cleanup stalls.
    """
    cpu = grant["launcher_cpu_seconds_per_invocation"]
    wall = min(grant["wall_seconds_per_invocation"], aware(grant["expires_at_utc"]).timestamp() - time.time())
    require(type(cpu) is int and 10 <= cpu <= MAX_SELF_CPU and 0 < wall <= MAX_WALL, "Invalid own CPU/wall envelope")
    require(all(signal.getitimer(timer) == (0., 0.) for timer in (signal.ITIMER_PROF, signal.ITIMER_REAL)), "Do not replace an existing process timer")
    originals = {sig: signal.getsignal(sig) for sig in (signal.SIGPROF, signal.SIGALRM)}
    def expire(sig, _frame):
        timer = signal.ITIMER_PROF if sig == signal.SIGPROF else signal.ITIMER_REAL
        signal.signal(sig, signal.SIG_DFL)
        signal.setitimer(timer, 2.)
        raise TimeoutError("Launcher own CPU/wall limit reached; no success inferred")
    try:
        signal.signal(signal.SIGPROF, expire)
        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_PROF, max(.01, cpu - time.process_time() - 2))
        signal.setitimer(signal.ITIMER_REAL, wall)
        yield
    finally:
        for timer in (signal.ITIMER_PROF, signal.ITIMER_REAL):
            signal.setitimer(timer, 0.)
        for sig, handler in originals.items():
            signal.signal(sig, handler)


def option(command, name):
    require(command.count(name) == 1 and command.index(name) + 1 < len(command), "Unique option required: " + name)
    return command[command.index(name) + 1]


def load_guard():
    require(sha(GUARD) == GUARD_SHA, "Frozen cumulative budget guard changed")
    spec = importlib.util.spec_from_file_location("queue_original_budget_guard", GUARD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_config(config, grant, config_sha, execute=False):
    require(config.get("schema") == SCHEMA and grant.get("schema") == GRANT_SCHEMA, "Unsupported queue/grant")
    require(grant.get("config_sha256") == config_sha and grant.get("launcher_sha256") == sha(__file__), "Source/config-bound root grant required")
    require(grant.get("no_new_quantum_budget") is True and grant.get("no_deadline_extension") is True,
            "A queue cannot allocate resources or reset the research clock")
    require(config.get("monitor_interval_seconds") == INTERVAL, "Progress is fixed at1800seconds")
    require(type(grant.get("max_invocations")) is int and 1 <= grant["max_invocations"] <= MAX_INVOCATIONS, "Invocation count must be bounded")
    require(type(grant.get("launcher_cpu_seconds_per_invocation")) is int and 10 <= grant["launcher_cpu_seconds_per_invocation"] <= MAX_SELF_CPU, "Launcher own CPU limit invalid")
    require(type(grant.get("wall_seconds_per_invocation")) is int and 180 <= grant["wall_seconds_per_invocation"] <= MAX_WALL, "Launcher wall limit invalid")
    require(type(grant.get("launcher_rss_stop_bytes")) is int and 16 * 1024**2 <= grant["launcher_rss_stop_bytes"] <= 512 * 1024**2, "Launcher own best-effort RSS guard invalid")
    require(aware(grant["expires_at_utc"]) <= aware(config["original_quantum_deadline_at_utc"]), "Queue cannot extend Q deadline")
    require(config["original_quantum_cpu_cap_seconds"] == 216000, "Original cumulative CPU cap must be retained")
    exceptions = grant.get("opaque_process_identity_exceptions", [])
    require(isinstance(exceptions, list) and len(exceptions) <= 2, "Only at most two explicit root daemon visibility exceptions")
    require(len({e.get("pid") for e in exceptions}) == len(exceptions), "Duplicate process visibility exception")
    for e in exceptions:
        require(e.get("uid") == 0 and e.get("comm") in ("dockerd", "containerd")
                and type(e.get("pid")) is int and e["pid"] > 0
                and type(e.get("start_ticks")) is int and e["start_ticks"] > 0
                and isinstance(e.get("boot_id"), str) and len(e["boot_id"]) >= 8,
                "Root exception must bind exact daemon pid/start/UID/comm/boot")
    jobs = config["jobs"]
    require(isinstance(jobs, list) and 0 < len(jobs) <= MAX_JOBS, "A finite materialized job list is required")
    require(len({j["id"] for j in jobs}) == len(jobs) and len({j["phase"] for j in jobs}) == len(jobs), "Duplicate job/phase")
    refs = {str(checked(r)): r["sha256"] for r in config["source_refs"]}
    for filename in refs:
        if Path(filename).suffix == ".py":
            for statement in ast.parse(Path(filename).read_text()).body:
                if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Constant) and statement.value.value == "NOT_FROZEN":
                    require(not any(isinstance(t, ast.Name) and t.id.endswith("_SHA") for t in statement.targets), "A downstream source pin is still NOT_FROZEN")
    require(len(refs) == len(config["source_refs"]), "Duplicate source ref")
    require(refs.get(str(GUARD)) == GUARD_SHA, "Original guard must be explicitly source-bound")
    require(all(refs.get(str(ROOT / name)) == expected for name, expected in FROZEN_A.items()), "Reviewed A/core sources required")
    require(config.get("pythonpath") == [".runtime/quantum-pilot-site", ".runtime/quantum-upgrade-site"], "Keep the reviewed Python site order")
    queue_directory = path(config["queue_directory"])
    require(queue_directory.is_relative_to(ROOT / "reports") and queue_directory != ROOT / "reports",
            "Queue metadata needs its own separate report directory")
    require(path(config["state"]) == queue_directory / "state.json"
            and path(config["monitor"]) == queue_directory / "monitor.jsonl", "Only the two reserved queue checkpoint paths may be written")
    protected = [path(config["ledger"]), *[path(r["path"]) for r in config["source_refs"]]]
    for job in jobs:
        protected += [path(job["result"]), path(job["result"]).parent / "input.json", path(job["log"]),
                      path(job["log"] + ".queue-console.log")]
        protected += [path(r["path"]) for r in job.get("inputs", [])]
    require(all(not file.is_relative_to(queue_directory) for file in protected), "Queue writes must be isolated from scientific inputs/results/logs and sources")
    require(len({str(path(j["result"])) for j in jobs}) == len(jobs)
            and len({str(path(j["log"])) for j in jobs}) == len(jobs), "Duplicate output/log target")
    for job in jobs:
        command = job["payload_argv"]
        worker = path(job["worker"]["path"])
        require(worker.name in WORKERS and refs.get(str(worker)) == job["worker"]["sha256"], "Only explicitly pinned V7 Q workers supported")
        require(isinstance(command, list) and len(command) >= 4 and all(isinstance(t, str) for t in command), "Arguments must be materialized strings; no shell")
        require(path(command[0]) == path(config["python_executable"]) and path(command[1]) == worker, "Exact Python/worker argv required")
        require(command.count("--execute") == 1 and option(command, "--stage") == job["stage"]
                and path(option(command, "--output")) == path(job["result"]).parent
                and int(option(command, "--threads")) == job["threads"] == 2, "Job argv/identity differs")
        require(type(job["cpu_seconds"]) is int and 125 <= job["cpu_seconds"] <= 122400
                and type(job["wall_seconds"]) is int and 61 <= job["wall_seconds"] <= 43200, "Invalid bounded job request")
        require(job["memory_gib"] == 24 and path(job["result"]).name == "result.json", "Original resource/result contract required")
        path(job["log"])
    for key in ("ledger", "state", "monitor"):
        path(config[key])
    if execute:
        require(grant.get("root_execution_approved") is True and grant.get("root_serial_scheduling_confirmed") is True,
                "Prepared queue is disabled until root explicitly approves serial execution")
        require(utc() < aware(grant["expires_at_utc"]), "Original grant expired")
    return config


def verify_stage(job, guard, ledger):
    """No PID absence or output-file existence is a success criterion."""
    rows = [r for r in ledger["phases"] if r.get("phase") == job["phase"]]
    if not rows:
        return None
    require(len(rows) == 1 and guard.healthy_closed_row(rows[0]), "FAILED/RUNNING/orphan stage requires forensic closure and a fresh root attempt")
    row = rows[0]
    require(row["command"] == job["payload_argv"] and path(row["log"]) == path(job["log"]), "Actual guardian payload argv/log changed")
    result_path = path(job["result"])
    result = read(result_path)
    require(result.get("stage") == job["stage"] and result.get("status") == job["success_status"], "Scientific stage is incomplete")
    proof_phase = result["resource_supervision"]["proof"]["phase"]
    require(proof_phase == job["phase"], "Result belongs to another actual phase")
    evidence = {"source_worker": result["source_worker"], "resource_supervision": result["resource_supervision"],
                "output_directory": str(result_path.parent), "snapshot_manifest_sha256": result["snapshot_manifest_sha256"], "stage": job["stage"]}
    guard.verify_closed_phase(evidence, expected_worker_path=path(job["worker"]["path"]),
        expected_worker_sha256=job["worker"]["sha256"], expected_result_schema=job["result_schema"],
        allowed_result_statuses=(job["success_status"],), expected_output=result_path.parent,
        expected_snapshot_sha256=option(job["payload_argv"], "--snapshot-manifest-sha256"))
    input_path = result_path.parent / "input.json"
    read(input_path)
    sealed = [artifact(result_path), artifact(input_path), artifact(path(result["resource_supervision"]["proof_path"])), artifact(path(row["log"]))]
    for key in ("plans_artifact", "partial_differential"):
        if key in result:
            sealed.append(artifact(checked(result[key])))
    for key, name in (("raw_hessian_sha256", "hessian_hartree_per_bohr2.npy"),
                      ("analysis_hessian_sha256", "hessian_used_for_analysis.npy"),
                      ("global_assembly_capture_sha256", "global_assembly_capture.json")):
        if key in result:
            sealed.append(artifact(checked({"path": str(result_path.parent / name), "sha256": result[key]})))
    if job["stage"] == "global-assemble":
        for name in ("normal_modes.npy", "reduced_mass_amu.npy", "frequencies.csv", "normal_mode_geometry_angstrom.npy",
                     "normal_mode_geometry_bohr.npy", "normal_mode_geometry.xyz"):
            sealed.append(artifact(result_path.parent / name))
    for ref in job.get("inputs", []):
        checked(ref)
    arrays = result.get("arrays", [])
    if "cache_manifest_sha256" in result:
        manifest_path = result_path.parent / "stage-cache/manifest.json"
        manifest = read(manifest_path, result["cache_manifest_sha256"])
        sealed.append(artifact(manifest_path))
        arrays = [*arrays, *manifest.get("arrays", [])]
    for item in arrays:
        require(Path(item["file"]).name == item["file"], "Unsafe payload filename")
        sealed.append(artifact(checked({"path": str(result_path.parent / "stage-cache" / item["file"]), "sha256": item["file_sha256"]})))
    unique = {r["path"]: r for r in sealed}
    return [unique[key] for key in sorted(unique)]


def verify_existing(job, guard, ledger, expected):
    require(expected, "Existing phase needs a persisted or root-config-bound seal before it can be skipped")
    for ref in expected:
        checked(ref)
    actual = verify_stage(job, guard, ledger)
    require(actual == expected, "Do not silently re-seal a changed historical result/input/proof/payload")
    return expected


# Orchestration-only policy. Resource supervision remains the frozen V6b guard.
PROCESS_SCAN_INTERVAL = 10.
MAX_PROCESS_RECORDS = 128
READ_ONLY_GIT = frozenset({"status", "diff", "show", "rev-parse", "ls-files",
    "for-each-ref", "log", "shortlog", "ls-tree", "cat-file", "name-rev",
    "describe", "merge-base", "rev-list", "check-ignore", "check-attr",
    "diff-tree", "diff-files", "diff-index", "ls-remote", "show-ref"})
ML_PREFIXES = ("train_", "run_supplemental_", "run_cross_plant_",
    "run_calibration_", "run_benchmark_")


def proc_identity(pid, proc_root=Path("/proc"), boot_id=None):
    directory = proc_root / str(pid)
    text = (directory / "stat").read_text()
    close = text.rfind(")")
    fields = text[close + 2:].split()
    uid_line = next(line for line in (directory / "status").read_text().splitlines()
                    if line.startswith("Uid:"))
    return {"pid": int(pid), "start_ticks": int(fields[19]), "uid": int(uid_line.split()[1]),
        "pgrp": int(fields[2]), "session": int(fields[3]), "state": fields[0],
        "comm": text[text.find("(") + 1:close][:80],
        "boot_id": boot_id if boot_id is not None else (proc_root / "sys/kernel/random/boot_id").read_text().strip()}


def same_identity(a, b):
    return all(a.get(k) == b.get(k) for k in ("pid", "start_ticks", "uid", "pgrp", "session", "boot_id"))


def classify_command(tokens):
    """Never return arbitrary arguments; tokens are used only in memory."""
    executable = Path(tokens[0]).name if tokens else "UNKNOWN"
    if executable == "git":
        i = 1
        while i < len(tokens):
            token = tokens[i]
            if token in ("-C", "-c", "--git-dir", "--work-tree", "--namespace"):
                i += 2; continue
            if token.startswith(("--git-dir=", "--work-tree=", "--namespace=", "--config-env=")):
                i += 1; continue
            if token in ("--no-pager", "--paginate", "--no-optional-locks", "--literal-pathspecs",
                         "--no-literal-pathspecs", "--no-replace-objects", "--no-lazy-fetch"):
                i += 1; continue
            if token.startswith("-"):
                return "GIT_WRITE_OR_UNKNOWN", "UNKNOWN"
            command = token
            # --output can write arbitrarily; do not whitelist it as read-only.
            writes_output = any(t == "--output" or t.startswith("--output=") for t in tokens[i+1:])
            return ("GIT_READ_ONLY" if command in READ_ONLY_GIT and not writes_output
                    else "GIT_WRITE_OR_UNKNOWN"), (command if command in READ_ONLY_GIT or command in
                    {"add", "commit", "push", "fetch", "pull", "checkout", "switch", "reset", "clean",
                     "merge", "rebase", "update-ref", "gc", "repack", "worktree", "config", "branch"} else "UNKNOWN")
        return "GIT_WRITE_OR_UNKNOWN", "UNKNOWN"
    if executable.startswith(("git-", "git_")):
        return "GIT_WRITE_OR_UNKNOWN", "HELPER_OR_UNKNOWN"
    if executable in ("torchrun", "accelerate"):
        return "ML", None
    if executable.startswith(("python", "pypy")):
        i = 1
        while i < len(tokens):
            token = tokens[i]
            if token.startswith("-c"): return "OTHER", None  # Do not inspect arbitrary source strings.
            if token.startswith("-m"):
                module = (tokens[i+1] if i+1 < len(tokens) else "") if token == "-m" else token[2:]
                return ("ML", None) if module.startswith(ML_PREFIXES + ("torch.distributed",)) else ("OTHER", None)
            if token in ("-X", "-W", "--check-hash-based-pycs"):
                i += 2; continue
            if token.startswith("-"):
                i += 1; continue
            # Only the actual Python script position is a training candidate.
            name = Path(token).name
            return ("ML", None) if name.endswith(".py") and name.startswith(ML_PREFIXES) else ("OTHER", None)
    return "OTHER", None


def process_observation(proc_root=Path("/proc"), opaque_exceptions=()):
    """Bounded redacted evidence. Unreadable/unstable identities are explicit.

    External read-only Git is informational. Write/unknown Git, recognized ML,
    and opaque process observations defer NEW work, never terminate active Q.
    No command is executed and no full command line is published.
    """
    result = {"observed_at_utc": utc().isoformat(), "records": [], "conflicts": [],
        "read_only_git": [], "visibility_errors": [], "records_truncated": False}
    try:
        boot_id = (proc_root / "sys/kernel/random/boot_id").read_text().strip()
        require(bool(boot_id), "Missing process boot identity")
        directories = sorted(proc_root.iterdir(), key=lambda p: p.name)
    except (OSError, ValueError) as error:
        record = {"category": "PROC_ROOT_OR_BOOT_ID_UNREADABLE", "error_type": type(error).__name__, "full_argv_logged": False}
        result["records"].append(record); result["conflicts"].append(record); result["visibility_errors"].append(record)
        return result
    for directory in directories:
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        first = None
        cwd_error = False
        try:
            first = proc_identity(int(directory.name), proc_root, boot_id=boot_id)
            if first["state"] in ("Z", "X", "x"):
                continue
            with (directory / "cmdline").open("rb") as handle:
                raw = handle.read(65537)
            truncated = len(raw) > 65536
            raw = raw[:65536]
            tokens = [t.decode(errors="replace") for t in raw.split(b"\0") if t]
            category, git_subcommand = classify_command(tokens)
            # Kernel threads can legitimately have no argv; comm=git remains suspect.
            if not tokens and first["comm"].startswith("git"):
                category = "GIT_WRITE_OR_UNKNOWN"
            executable = Path(os.readlink(directory / "exe")).name
            actual_category, actual_subcommand = classify_command([executable, *tokens[1:]])
            if actual_category != category:
                category = "EXECUTABLE_ARGV_IDENTITY_UNKNOWN"
            else:
                category, git_subcommand = actual_category, actual_subcommand
            if category == "OTHER" and not truncated:
                continue
            try:
                cwd = os.readlink(directory / "cwd")
                cwd_record = {"cwd": cwd[:512], "cwd_truncated": len(cwd) > 512}
            except (FileNotFoundError, PermissionError, ProcessLookupError) as error:
                cwd_record = {"cwd": None, "cwd_visibility": type(error).__name__}
                category = "PROCESS_CWD_UNREADABLE"
                cwd_error = True
            second = proc_identity(int(directory.name), proc_root, boot_id=boot_id)
            if not same_identity(first, second):
                category = "UNSTABLE_PROCESS_IDENTITY"
            if truncated:
                category = "TRUNCATED_COMMAND_UNKNOWN"
            if category == "OTHER":
                continue
            record = {**first, **cwd_record, "category": category, "git_subcommand": git_subcommand,
                "executable_basename": (executable[:80] if executable else "UNAVAILABLE"),
                "argv_prefix_sha256": hashlib.sha256(raw).hexdigest(), "argv_bytes_truncated": truncated,
                "full_argv_logged": False}
        except (FileNotFoundError, ProcessLookupError) as error:
            # A vanished component (exe/cmdline/cwd/status) is not proof that
            # its PID vanished. Only a missing/terminal PID can be skipped.
            try:
                live_stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
                if live_stat[0] in ("Z", "X", "x"):
                    continue
                record = {"pid": int(directory.name), "start_ticks": int(live_stat[19]),
                    "state": live_stat[0], "boot_id": boot_id, "category": "OPAQUE_PROCESS",
                    "error_type": type(error).__name__, "cause": "LIVE_PROCESS_COMPONENT_UNREADABLE",
                    "full_argv_logged": False}
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (OSError, ValueError, IndexError) as followup:
                record = {"pid": int(directory.name), "category": "OPAQUE_PROCESS",
                    "error_type": type(followup).__name__, "cause": "PID_PRESENCE_UNREADABLE",
                    "full_argv_logged": False}
            result["visibility_errors"].append(record)
        except (PermissionError, OSError, ValueError, StopIteration, IndexError) as error:
            try:
                second = proc_identity(int(directory.name), proc_root, boot_id=boot_id)
                identity_stable = first is not None and same_identity(first, second) and first["comm"] == second["comm"]
            except (OSError, ValueError, StopIteration, IndexError):
                identity_stable = False
            acknowledged = identity_stable and any(
                all(second.get(k) == e.get(k) for k in ("pid", "start_ticks", "uid", "comm", "boot_id"))
                and e.get("uid") == 0 and e.get("comm") in ("dockerd", "containerd")
                for e in opaque_exceptions)
            record = {**(first or {"pid": int(directory.name)}),
                "category": "ROOT_ACKNOWLEDGED_OPAQUE_DAEMON" if acknowledged else "OPAQUE_PROCESS",
                "error_type": type(error).__name__, "full_argv_logged": False,
                "visibility_exception_root_grant_bound": acknowledged, "identity_double_read_stable": identity_stable}
            result["visibility_errors"].append(record)
        if len(result["records"]) >= MAX_PROCESS_RECORDS:
            result["records_truncated"] = True
            break
        result["records"].append(record)
        if cwd_error:
            result["visibility_errors"].append(record)
        if record["category"] == "GIT_READ_ONLY":
            result["read_only_git"].append(record)
        elif record["category"] != "ROOT_ACKNOWLEDGED_OPAQUE_DAEMON":
            result["conflicts"].append(record)
    if result["records_truncated"]:
        result["conflicts"].append({"category": "PROCESS_RECORD_LIMIT", "limit": MAX_PROCESS_RECORDS})
    return result


def conflicting_processes():
    return process_observation()["conflicts"]


def resource_stop_sample(signals_received, end, monotonic_end, own_cpu, rss_limit):
    now, mono, cpu, rss = time.time(), time.monotonic(), time.process_time(), own_rss()
    reasons = []
    if signals_received: reasons.append("EXPLICIT_LAUNCHER_SIGNAL")
    if now >= end - 60: reasons.append("LAUNCHER_UTC_CLEANUP_BOUNDARY")
    if mono >= monotonic_end - 60: reasons.append("LAUNCHER_MONOTONIC_CLEANUP_BOUNDARY")
    if cpu >= own_cpu - 5: reasons.append("LAUNCHER_CPU_CLEANUP_BOUNDARY")
    if rss >= rss_limit: reasons.append("LAUNCHER_RSS_STOP_BOUNDARY")
    return {"observed_at_utc": utc().isoformat(), "reasons": reasons,
        "signals_received": list(signals_received), "wall_remaining_seconds": end-now,
        "monotonic_remaining_seconds": monotonic_end-mono, "launcher_CPU_seconds": cpu,
        "launcher_CPU_limit_seconds": own_cpu, "launcher_RSS_bytes": rss,
        "launcher_RSS_stop_bytes": rss_limit}


def signal_owned_guardian(child, expected_identity, sample, publish):
    """Only the unreaped Popen-owned same-identity guardian may receive SIGTERM.

    The complete stop request is published before signal delivery. An exited or
    PID-reused process is never signalled. This does not signal arbitrary groups.
    """
    record = {"event": "GUARDIAN_SIGTERM_REQUEST", "requested_signal": "SIGTERM",
        "expected_guardian_identity": expected_identity, "stop_sample": sample,
        "observed_at_utc": utc().isoformat()}
    if child.poll() is not None:
        record["action"] = "NO_SIGNAL_ALREADY_EXITED"; publish(record); return False
    try:
        current = proc_identity(child.pid)
    except (FileNotFoundError, ProcessLookupError):
        record["action"] = "NO_SIGNAL_PROCESS_GONE"; publish(record); return False
    record["observed_guardian_identity"] = current
    if not same_identity(expected_identity, current) or current["state"] in ("Z", "X", "x"):
        record["action"] = "NO_SIGNAL_IDENTITY_CHANGED_OR_TERMINAL"; publish(record); return False
    record["action"] = "REQUEST_BEFORE_SIGNAL_DELIVERY"
    publish(record)
    # A direct child cannot have its PID reused while still owned and unreaped.
    if child.poll() is not None:
        publish({**record, "action": "NO_SIGNAL_EXITED_AFTER_PUBLICATION"}); return False
    try:
        final = proc_identity(child.pid)
    except (FileNotFoundError, ProcessLookupError):
        publish({**record, "action": "NO_SIGNAL_GONE_AFTER_PUBLICATION"}); return False
    if not same_identity(expected_identity, final) or final["state"] in ("Z", "X", "x"):
        publish({**record, "action": "NO_SIGNAL_IDENTITY_CHANGED_AFTER_PUBLICATION"}); return False
    try:
        child.terminate()
    except BaseException as error:
        publish({**record, "action": "SIGTERM_CALL_FAILED", "error_type": type(error).__name__})
        raise
    publish({**record, "action": "SIGTERM_CALL_RETURNED"})
    return True


@contextmanager
def queue_lock():
    file = ROOT / ".runtime/research4-materialized-quantum-queue.lock"
    file.parent.mkdir(exist_ok=True)
    require(not file.is_symlink(), "Symlink launcher lock rejected")
    with file.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def run(args, config, grant):
    guard = load_guard()
    with queue_lock():
        ledger = read(config["ledger"])
        guard.validate_ledger(ledger)
        require(ledger["quantum_cpu_cap_seconds"] == config["original_quantum_cpu_cap_seconds"]
                and ledger["run_deadline_at_utc"] == config["original_quantum_deadline_at_utc"], "Original quantum envelope changed")
        state_path = path(config["state"])
        state = read(state_path) if state_path.exists() else {"schema": SCHEMA, "config_sha256": args.config_sha256,
            "grant_sha256": args.grant_sha256, "invocations": [], "jobs": {}}
        require(state["config_sha256"] == args.config_sha256 and state["grant_sha256"] == args.grant_sha256, "Existing state belongs to another immutable queue/grant")
        require(all(r["status"] in ("MATERIALIZED_QUEUE_CLOSED_NOT_ALL_SCIENCE_COMPLETE", "DEFERRED_BEFORE_LAUNCH_EXTERNAL_CONFLICT", "DEFERRED_AFTER_HEALTHY_STAGE_EXTERNAL_CONFLICT") for r in state["invocations"]), "Unclosed/blocked launcher invocation requires root forensic review and a fresh approved configuration; never infer success from a missing PID")
        require(len(state["invocations"]) < grant["max_invocations"] and args.attempt not in {r["id"] for r in state["invocations"]}, "Fresh bounded root invocation required")
        start = time.monotonic()
        end = min(aware(grant["expires_at_utc"]).timestamp(), args.invocation_deadline)
        monotonic_end = args.invocation_monotonic_deadline
        own_cpu = grant["launcher_cpu_seconds_per_invocation"]
        invocation = {"id": args.attempt, "status": "RUNNING", "pid": os.getpid(), "started_at_utc": utc().isoformat(),
            "per_invocation_self_CPU_cap_seconds": own_cpu, "wall_deadline_utc": datetime.fromtimestamp(end, timezone.utc).isoformat(),
            "own_CPU_envelope_not_quantum_CPU_or_observed_actual_cost": True}
        invocation["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        invocation["start_ticks"] = int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
        state["resume_argv"] = [str(path(config["python_executable"])), str(Path(__file__).resolve()),
            "--config", str(path(args.config)), "--config-sha256", args.config_sha256, "--grant", str(path(args.grant)),
            "--grant-sha256", args.grant_sha256, "--attempt", args.attempt + "-resume", "--execute"]
        state["resume_requires_environment_ready_and_no_unresolved_invocation_or_Q_phase"] = True
        state["invocations"].append(invocation)
        child = None
        interrupted = []
        guardian_identity = None
        def receive_signal(sig, _frame):
            if len(interrupted) < 8:
                interrupted.append({"signal": int(sig), "observed_at_monotonic_seconds": time.monotonic(), "sender": "UNKNOWN"})
        handlers = {s: signal.signal(s, receive_signal) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGXCPU)}
        next_progress = time.monotonic() + INTERVAL
        def persist(event):
            nonlocal next_progress
            current = read(config["ledger"])
            invocation["self_CPU_observed_seconds"] = time.process_time()
            state["next_stage"] = next((j["id"] for j in config["jobs"] if state["jobs"].get(j["id"], {}).get("status") != "HEALTHY_CLOSED"), None)
            state["last_event"] = event
            save(state_path, state)
            record = {"utc": utc().isoformat(), "event": event, "next_stage": state["next_stage"],
                "closed_quantum_CPU_seconds": current["consumed_cpu_seconds"], "remaining_quantum_CPU_before_open_phase": current["remaining_quantum_cpu_seconds"],
                "open_phases": [r["phase"] for r in current["phases"] if r.get("status") == "RUNNING"],
                "launcher_self_CPU_observed_seconds": invocation["self_CPU_observed_seconds"],
                "launcher_RSS_observed_bytes": own_rss(), "platform_push_service_created": False}
            monitor = path(config["monitor"])
            monitor.parent.mkdir(parents=True, exist_ok=True)
            with monitor.open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
            print(json.dumps(record), flush=True)
            if event == "PROGRESS_1800_SECONDS":
                while next_progress <= time.monotonic():
                    next_progress += INTERVAL
        try:
            persist("QUEUE_STARTED")
            for job in config["jobs"]:
                ledger = read(config["ledger"])
                guard.validate_ledger(ledger)
                existing = [r for r in ledger["phases"] if r.get("phase") == job["phase"]]
                if existing:
                    expected = state["jobs"].get(job["id"], {}).get("seal") or job.get("existing_seal")
                    verify_existing(job, guard, ledger, expected)
                    state["jobs"][job["id"]] = {"status": "HEALTHY_CLOSED", "skipped_after_full_reverification": True, "seal": expected}
                    continue
                require(not interrupted and min(end - time.time(), monotonic_end - time.monotonic()) >= 181
                        and time.process_time() < own_cpu - 5 and own_rss() < grant["launcher_rss_stop_bytes"], "Launcher resource/stop boundary reached")
                observed = process_observation(opaque_exceptions=grant.get("opaque_process_identity_exceptions", []))
                invocation["prelaunch_process_observation"] = observed
                if observed["conflicts"]:
                    invocation["status"] = "DEFERRED_BEFORE_LAUNCH_EXTERNAL_CONFLICT"
                    persist("NEXT_LAUNCH_DEFERRED_EXTERNAL_CONFLICT")
                    break
                require(sha(path(args.config)) == args.config_sha256 and sha(path(args.grant)) == args.grant_sha256
                        and sha(__file__) == grant["launcher_sha256"], "Queue/grant/launcher bytes changed before a new job")
                for ref in [*config["source_refs"], *job.get("inputs", [])]:
                    checked(ref)
                require(not path(job["log"]).exists() and not path(job["result"]).parent.exists(), "Unclosed output/log already exists; preserve and create a fresh root attempt")
                wall = min(job["wall_seconds"], int(min(end - time.time(), monotonic_end - time.monotonic())) - 120)
                command = [str(path(config["python_executable"])), str(GUARD), "run", "--ledger", str(path(config["ledger"])),
                    "--phase", job["phase"], "--cpu-seconds", str(job["cpu_seconds"]), "--wall-seconds", str(wall),
                    "--threads", "2", "--memory-gib", "24", "--log", str(path(job["log"])), "--", *job["payload_argv"]]
                state["jobs"][job["id"]] = {"status": "STARTING", "actual_guardian_argv": command}
                persist("GUARDED_STAGE_STARTING")
                console = path(job["log"] + ".queue-console.log")
                console.parent.mkdir(parents=True, exist_ok=True)
                with console.open("x") as handle:
                    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(str(path(p)) for p in config["pythonpath"]))
                    child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        guardian_identity = proc_identity(child.pid)
                    except (FileNotFoundError, ProcessLookupError):
                        require(child.poll() is not None, "Live guardian identity unreadable; do not infer ownership")
                        guardian_identity = None
                    state["jobs"][job["id"]].update(status="RUNNING", guardian_pid=child.pid, guardian_identity=guardian_identity)
                    save(state_path, state)
                    stopping = False
                    next_scan = time.monotonic()
                    last_external_signature = None
                    def publish_stop(record):
                        history = invocation.setdefault("guardian_stop_requests", [])
                        require(len(history) < 8, "Bounded guardian stop history exhausted")
                        history.append({**record, "job_id": job["id"], "phase": job["phase"]})
                        persist("GUARDIAN_STOP_REASON_BEFORE_SIGNAL")
                    while child.poll() is None:
                        if time.monotonic() >= next_progress:
                            persist("PROGRESS_1800_SECONDS")
                        if time.monotonic() >= next_scan:
                            observed = process_observation(opaque_exceptions=grant.get("opaque_process_identity_exceptions", []))
                            next_scan = time.monotonic() + PROCESS_SCAN_INTERVAL
                            signature = hashlib.sha256(json.dumps(observed["records"], sort_keys=True).encode()).hexdigest()
                            invocation["last_active_process_observation"] = observed
                            if observed["conflicts"]:
                                invocation["defer_next_launch"] = True
                                if signature != last_external_signature:
                                    invocation["external_conflict_observations_count"] = invocation.get("external_conflict_observations_count", 0) + 1
                                    if invocation["external_conflict_observations_count"] <= 64:
                                        persist("EXTERNAL_CONFLICT_ACTIVE_STAGE_CONTINUES_NEXT_LAUNCH_DEFERRED")
                            last_external_signature = signature
                        sample = resource_stop_sample(interrupted, end, monotonic_end, own_cpu, grant["launcher_rss_stop_bytes"])
                        if sample["reasons"] and not stopping:
                            stopping = signal_owned_guardian(child, guardian_identity, sample, publish_stop)
                        require(time.time() < end and time.monotonic() < monotonic_end, "Guardian not closed by outer wall deadline; retain unresolved state and do not retry")
                        time.sleep(1)
                    returncode = child.wait()
                child = None
                state["jobs"][job["id"]].update(status="GUARDIAN_EXITED_UNVERIFIED", actual_guardian_returncode=returncode)
                if returncode != 0:
                    sent = any(r.get("action") == "SIGTERM_CALL_RETURNED" and r.get("job_id") == job["id"]
                        and guardian_identity is not None and same_identity(r.get("expected_guardian_identity", {}), guardian_identity)
                        for r in invocation.get("guardian_stop_requests", []))
                    invocation["guardian_exit_observation"] = {"returncode": returncode,
                        "launcher_stop_request_published": sent, "SIGTERM_sender": "UNKNOWN_UNLESS_SEPARATELY_AUTHENTICATED" if sent else "UNKNOWN_EXTERNAL_TO_LAUNCHER",
                        "observed_at_utc": utc().isoformat()}
                    persist("GUARDIAN_EXITED_AFTER_LAUNCHER_STOP_REQUEST" if sent else "GUARDIAN_EXITED_WITHOUT_LAUNCHER_STOP_REQUEST")
                require(returncode == 0, "Actual guardian failed; keep failed cost and stop without retry")
                ledger = read(config["ledger"])
                guard.validate_ledger(ledger)
                sealed = verify_stage(job, guard, ledger)
                require(sealed, "Guardian exit alone is insufficient; scientific stage verification failed")
                state["jobs"][job["id"]] = {"status": "HEALTHY_CLOSED", "actual_guardian_returncode": returncode, "seal": sealed}
                persist("GUARDED_STAGE_HEALTHY_CLOSED")
                if invocation.get("defer_next_launch"):
                    invocation["status"] = "DEFERRED_AFTER_HEALTHY_STAGE_EXTERNAL_CONFLICT"
                    persist("NEXT_LAUNCH_DEFERRED_AFTER_HEALTHY_STAGE")
                    break
            else:
                invocation["status"] = "MATERIALIZED_QUEUE_CLOSED_NOT_ALL_SCIENCE_COMPLETE"
        except BaseException as error:
            invocation.update(status="BLOCKED_REQUIRES_ROOT_REVIEW", reason=type(error).__name__ + ": " + str(error))
            if child is not None and child.poll() is None:
                sample = resource_stop_sample(interrupted, end, monotonic_end, own_cpu, grant["launcher_rss_stop_bytes"])
                sample["reasons"].append("EXCEPTION_CLEANUP_" + type(error).__name__)
                def publish_cleanup(record):
                    invocation.setdefault("guardian_stop_requests", []).append({**record, "job_id": job["id"], "phase": job["phase"]})
                    persist("GUARDIAN_EXCEPTION_STOP_REASON_BEFORE_SIGNAL")
                if guardian_identity is not None:
                    signal_owned_guardian(child, guardian_identity, sample, publish_cleanup)
                else:
                    invocation["cleanup_signal_refused"] = "Guardian identity not captured; no arbitrary PID signal"
                    persist("GUARDIAN_CLEANUP_IDENTITY_UNAVAILABLE")
                try:
                    code = child.wait(timeout=max(.01, min(10., end - time.time(), monotonic_end - time.monotonic())))
                    invocation["exception_cleanup_actual_guardian_returncode"] = code
                except subprocess.TimeoutExpired:
                    invocation.update(status="BLOCKED_GUARDIAN_UNRESOLVED_NO_AUTOMATIC_RESUME", unresolved_guardian_pid=child.pid)
            raise
        finally:
            invocation.update(ended_observed_at_utc=utc().isoformat(), self_CPU_observed_seconds=time.process_time(), elapsed_wall_observed_seconds=time.monotonic() - start)
            persist("QUEUE_INVOCATION_CLOSED_OR_BLOCKED")
            for sig, old in handlers.items():
                signal.signal(sig, old)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--grant", required=True)
    parser.add_argument("--grant-sha256", required=True)
    parser.add_argument("--attempt", default="PREPARATION_ONLY")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config, grant = read(args.config, args.config_sha256), read(args.grant, args.grant_sha256)
    require(not {path(args.config), path(args.grant)} & {path(config["state"]), path(config["monitor"])}, "Queue checkpoint cannot overwrite its configuration or root grant")
    if not args.execute:
        validate_config(config, grant, args.config_sha256, False)
        print(json.dumps({"status": "NOT_RUN_CONFIGURATION_CHECKED", "jobs": len(config["jobs"]), "background_start_hook_created": False}))
        return
    require(args.attempt != "PREPARATION_ONLY", "Root must provide a fresh invocation identity")
    require(grant.get("root_execution_approved") is True and grant.get("root_serial_scheduling_confirmed") is True,
            "Root execution grant required before activating even the own-process timers")
    args.invocation_deadline = min(aware(grant["expires_at_utc"]).timestamp(), time.time() + grant["wall_seconds_per_invocation"])
    args.invocation_monotonic_deadline = time.monotonic() + grant["wall_seconds_per_invocation"]
    with self_limits(grant):
        validate_config(config, grant, args.config_sha256, True)
        run(args, config, grant)


if __name__ == "__main__":
    main()
