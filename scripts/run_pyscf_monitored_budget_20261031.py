"""Monitor one PySCF process group under a separately authorized V3 budget.

The earlier pilot ledger is immutable. This runner binds the separate continuation
authorization, source checkpoint and source revision, and never resets a ledger.
CPU means actual wait4 user+system seconds, not wall seconds multiplied by threads.
RSS polling is BEST_EFFORT_RSS_GUARD_NOT_HARD_RSS_LIMIT. A0.1s polling interval
cannot prevent all transient overshoot or guarantee6GiB of physical headroom.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
OLD_PILOT = ROOT / "reports/pyscf-spinosin-pilot-20261031"
SOURCE_BASE = "aef64c65295bee03764da1dd72c21d32ed3e6c24"
SCHEMA = "research4-pyscf-monitored-budget-v3"
MAX_CPU, MAX_WALL = 172800, 86400
MAX_PHASE_CPU, MAX_PHASE_WALL = 86400, 43200
RSS_STOP_BYTES = 10 * 1024**3
CGROUP_KILL_BYTES = 12 * 1024**3
MONITOR_INTERVAL_SECONDS = .1
MEMORY_KILL_GRACE_SECONDS = 1.0
CGROUP_MEMORY_MAX_BYTES = 16 * 1024**3
GUARD_SEMANTICS = "BEST_EFFORT_RSS_GUARD_NOT_HARD_RSS_LIMIT"
PROOF_SCHEMA = "research4-pyscf-resource-supervisor-proof-v3"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def utc():
    return datetime.now(timezone.utc)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def root_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def signal_group(pid, signum):
    try:
        os.killpg(pid, signum)
        return True
    except ProcessLookupError:
        # A payload may exit between nonblocking wait4 and the signal request.
        # Reap it on the next iteration rather than leave a false RUNNING phase.
        return False


def parse_proc_stat(value):
    """Linux/proc/PID/stat, preserving spaces and ')' characters in comm."""
    left, right = value.find("("), value.rfind(")")
    if left < 1 or right <= left:
        raise ValueError("Malformed process stat comm field")
    fields = value[right+1:].split()
    if len(fields) < 22:
        raise ValueError("Incomplete process stat")
    return {"pid": int(value[:left].strip()), "pgrp": int(fields[2]),
            "user_ticks": int(fields[11]), "system_ticks": int(fields[12]),
            "child_user_ticks": int(fields[13]), "child_system_ticks": int(fields[14]),
            "start_ticks": int(fields[19]), "vsize_bytes": int(fields[20]),
            "rss_pages": max(0, int(fields[21]))}


def process_group_snapshot(pgid, proc_root=Path("/proc")):
    """Sum process RSS, never/task thread RSS; shared pages may be overcounted."""
    processes, unreadable = [], []
    page_bytes = os.sysconf("SC_PAGE_SIZE")
    for directory in Path(proc_root).iterdir():
        if not directory.name.isdigit():
            continue
        try:
            value = parse_proc_stat((directory / "stat").read_text())
            if value["pgrp"] == pgid:
                processes.append({**value, "rss_bytes": value["rss_pages"] * page_bytes})
        except (OSError, ValueError):
            unreadable.append(int(directory.name))
    processes.sort(key=lambda row: row["pid"])
    return {"observed_at_utc": utc().isoformat(),
            "rss_bytes": sum(row["rss_bytes"] for row in processes),
            "virtual_address_bytes_sum": sum(row["vsize_bytes"] for row in processes),
            "pids": [row["pid"] for row in processes], "processes": processes,
            "unreadable_pids": sorted(unreadable),
            "shared_mapping_double_count_is_conservative": True}


def read_cgroup_memory():
    root = Path("/sys/fs/cgroup")
    current = int((root / "memory.current").read_text().strip())
    maximum = (root / "memory.max").read_text().strip()
    return {"current_bytes": current, "max_bytes": None if maximum == "max" else int(maximum),
            "source": str(root), "observed_at_utc": utc().isoformat()}


def validate_prior_closed_ledger(authorization):
    path = root_path(authorization.get("prior_closed_ledger_path", ""))
    expected_sha = authorization.get("prior_closed_ledger_sha256")
    if not path.is_file() or not expected_sha or sha(path) != expected_sha:
        raise ValueError("Prior closed ledger bytes differ from explicit authorization")
    prior = read(path)
    if prior.get("schema") != "research4-pyscf-continuation-budget-v2":
        raise ValueError("New budget must bind the specifically closed V2 ledger")
    phases = prior.get("phases")
    if not isinstance(phases, list) or any(row.get("status") == "RUNNING" for row in phases):
        raise RuntimeError("Prior V2 phase is unresolved; forbid parallel budgets")
    if any(row.get("status") == "BUDGET_VIOLATION_REQUIRES_REVIEW" for row in phases):
        raise RuntimeError("Prior budget violation requires explicit resolution before new budget")
    used = [float(row["actual_cpu_seconds"]) for row in phases]
    if any(not math.isfinite(v) or v < 0 for v in used):
        raise ValueError("Prior ledger CPU values are invalid")
    total = float(prior["consumed_cpu_seconds"])
    authorized_total = float(authorization.get("prior_consumed_cpu_seconds", float("nan")))
    if (not math.isclose(sum(used), total, rel_tol=1e-12, abs_tol=1e-7)
            or not math.isclose(total, authorized_total, rel_tol=1e-12, abs_tol=1e-7)):
        raise ValueError("Prior actual CPU totals differ from bound closed-ledger usage")
    return {"path": str(path), "sha256": expected_sha, "actual_cpu_seconds": total,
            "phase_count": len(phases), "no_RUNNING_phases": True}


def validate_authorization(path, expected_sha, checkpoint, checkpoint_sha, source_base):
    path, checkpoint = Path(path).resolve(), Path(checkpoint).resolve()
    if sha(path) != expected_sha:
        raise ValueError("Authorization bytes do not match explicit SHA-256")
    value = read(path)
    if not value.get("created_at_utc") or not value.get("authorization_text"):
        raise ValueError("Dated, explicit user authorization is required")
    if value.get("new_budget_is_separately_authorized") is not True or value.get("prior_budget_preserved") is not True:
        raise ValueError("A separately authorized budget must preserve the earlier pilot")
    if source_base != SOURCE_BASE or value.get("source_base_commit") != source_base:
        raise ValueError("Continuation source revision must be the recorded aef64c6 pilot")
    if value.get("source_checkpoint_sha256") != checkpoint_sha or sha(checkpoint) != checkpoint_sha:
        raise ValueError("Source checkpoint differs from the authorized immutable snapshot")
    if root_path(value.get("source_checkpoint_path", "")) != checkpoint:
        raise ValueError("Source checkpoint path differs from authorization")
    limits = {"max_quantum_cpu_seconds": MAX_CPU, "max_run_wall_seconds": MAX_WALL,
              "max_phase_cpu_seconds": MAX_PHASE_CPU, "max_phase_wall_seconds": MAX_PHASE_WALL,
              "max_threads": 4, "max_memory_gib": 24}
    for name, maximum in limits.items():
        if type(value.get(name)) is not int or not 0 < value[name] <= maximum:
            raise ValueError("Invalid authorized bounded limit: " + name)
    if value.get("guard_semantics") != GUARD_SEMANTICS:
        raise ValueError("Authorization must acknowledge best-effort rather than hard RSS monitoring")
    for name, maximum in (("rss_stop_bytes", 10*1024**3), ("cgroup_kill_bytes", 12*1024**3),
                          ("monitor_interval_seconds", .1), ("memory_kill_grace_seconds", 1.0)):
        limit = value.get(name)
        if type(limit) not in (int, float) or not math.isfinite(limit) or not 0 < limit <= maximum:
            raise ValueError("Invalid resource guard authorization: " + name)
    if value.get("expected_cgroup_memory_max_bytes") != CGROUP_MEMORY_MAX_BYTES:
        raise ValueError("Explicit original16GiB cgroup envelope must be bound")
    validate_prior_closed_ledger(value)
    return value


@contextmanager
def prior_budget_lock(authorization):
    """Use exactly the old V2 runner's lock, blocking concurrent old phases."""
    path = root_path(authorization["prior_closed_ledger_path"]).with_suffix(".lock")
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def initialize(args):
    if args.ledger.exists():
        raise FileExistsError("Never reset or overwrite an earlier budget ledger")
    if not args.authorization:
        raise ValueError("Explicit authorization file required")
    authorization = read(args.authorization)
    with prior_budget_lock(authorization):
        return _initialize_with_prior_lock(args)


def _initialize_with_prior_lock(args):
    if args.ledger.exists():
        raise FileExistsError("Never reset or overwrite an earlier budget ledger")
    required = (args.authorization, args.authorization_sha256, args.source_checkpoint,
                args.source_checkpoint_sha256, args.source_base_commit)
    if not all(required):
        raise ValueError("Initialization requires authorization and all source bindings")
    auth = validate_authorization(*required)
    prior = validate_prior_closed_ledger(auth)
    if args.ledger.resolve().is_relative_to(Path(prior["path"]).parent):
        raise ValueError("New ledger must remain outside the immutable V2 report tree")
    if not 65 <= args.total_cpu_seconds <= auth["max_quantum_cpu_seconds"]:
        raise ValueError("New total CPU budget must be65..authorized maximum")
    if not 61 <= args.total_wall_seconds <= auth["max_run_wall_seconds"]:
        raise ValueError("New total wall budget must be61..authorized maximum")
    started = utc()
    ledger = {"schema": SCHEMA, "created_at_utc": started.isoformat(),
              "run_deadline_at_utc": (started + timedelta(seconds=args.total_wall_seconds)).isoformat(),
              "quantum_cpu_cap_seconds": args.total_cpu_seconds,
              "run_wall_cap_seconds": args.total_wall_seconds,
              "consumed_cpu_seconds": 0.0, "remaining_quantum_cpu_seconds": float(args.total_cpu_seconds),
              "phases": [], "cpu_accounting": "Actual wait4 child user+system CPU; one payload process,1..4threads",
              "authorization_path": str(args.authorization.resolve()),
              "authorization_sha256": args.authorization_sha256,
              "source_checkpoint_path": str(args.source_checkpoint.resolve()),
              "source_checkpoint_sha256": args.source_checkpoint_sha256,
              "source_base_commit": args.source_base_commit,
              "prior_closed_ledger": prior,
              "prior_consumed_cpu_seconds": prior["actual_cpu_seconds"],
              "guard_semantics": GUARD_SEMANTICS,
              "not_a_hard_RSS_limit": True,
              "prior_pilot_ledger_modified": False,
              "runner_sha256": sha(__file__)}
    save(args.ledger, ledger)
    return ledger


def validate_ledger(ledger):
    if ledger.get("schema") != SCHEMA:
        raise ValueError("Not a continuation ledger; old pilot budgets cannot be repurposed")
    if ledger.get("runner_sha256") != sha(__file__):
        raise ValueError("Runner source changed after budget initialization")
    auth = validate_authorization(ledger["authorization_path"], ledger["authorization_sha256"],
                                  ledger["source_checkpoint_path"], ledger["source_checkpoint_sha256"],
                                  ledger["source_base_commit"])
    if validate_prior_closed_ledger(auth) != ledger.get("prior_closed_ledger"):
        raise ValueError("Closed prior budget binding changed")
    if not 65 <= ledger["quantum_cpu_cap_seconds"] <= auth["max_quantum_cpu_seconds"] or not 61 <= ledger["run_wall_cap_seconds"] <= auth["max_run_wall_seconds"]:
        raise ValueError("Ledger caps exceed original authorization")
    if any(row.get("status") == "RUNNING" for row in ledger["phases"]):
        raise RuntimeError("A prior phase is unresolved; do not double-launch")
    used = [float(row["actual_cpu_seconds"]) for row in ledger["phases"]]
    if any(not math.isfinite(v) or v < 0 for v in used) or not math.isclose(sum(used), ledger["consumed_cpu_seconds"], rel_tol=1e-12, abs_tol=1e-7):
        raise ValueError("Actual completed CPU totals differ from ledger")
    started = datetime.fromisoformat(ledger["created_at_utc"])
    expected_deadline = started + timedelta(seconds=ledger["run_wall_cap_seconds"])
    if datetime.fromisoformat(ledger["run_deadline_at_utc"]) != expected_deadline:
        raise ValueError("Overall wall deadline was altered")
    return auth


def phase_limits(ledger, auth, args, now=None):
    if type(args.threads) is not int or not 1 <= args.threads <= auth["max_threads"]:
        raise ValueError("Threads must remain within1..4and authorization")
    if not 1 <= args.memory_gib <= auth["max_memory_gib"]:
        raise ValueError("Address space exceeds authorized memory cap")
    if not args.cpu_seconds or not 65 <= args.cpu_seconds <= auth["max_phase_cpu_seconds"]:
        raise ValueError("Phase CPU request must be65..86400seconds and within authorization")
    if not args.wall_seconds or not 61 <= args.wall_seconds <= auth["max_phase_wall_seconds"]:
        raise ValueError("Phase wall request must allow60s cleanup and be<=43200seconds")
    remaining_cpu = ledger["quantum_cpu_cap_seconds"] - ledger["consumed_cpu_seconds"]
    # Linux per-process limits can be observed a little late with several threads.
    # Keep an extra CPU boundary allowance rather than grant the final fraction.
    reserve = 2 * args.threads + 2
    grant = min(args.cpu_seconds, math.floor(remaining_cpu) - reserve)
    remaining_wall = (datetime.fromisoformat(ledger["run_deadline_at_utc"]) - (now or utc())).total_seconds()
    wall = min(args.wall_seconds, math.floor(remaining_wall) - 2)
    if grant < 65 or wall < 61:
        raise ValueError("Insufficient original CPU/wall budget for a safe bounded phase")
    cpu_margin = min(120, max(60, grant // 10))
    return {"granted_cpu_seconds": grant, "cpu_soft_limit_seconds": grant - cpu_margin,
            "cpu_hard_limit_seconds": grant, "cpu_cleanup_margin_seconds": cpu_margin,
            "cpu_boundary_reserve_seconds": reserve, "wall_limit_seconds": wall,
            "wall_cleanup_seconds": 60, "wall_SIGTERM_after_seconds": wall - 60,
            "address_space_limit_gib": args.memory_gib, "threads": args.threads,
            "rss_stop_bytes": min(RSS_STOP_BYTES, auth["rss_stop_bytes"]),
            "cgroup_kill_bytes": min(CGROUP_KILL_BYTES, auth["cgroup_kill_bytes"]),
            "monitor_interval_seconds": min(MONITOR_INTERVAL_SECONDS, auth["monitor_interval_seconds"]),
            "memory_kill_grace_seconds": min(MEMORY_KILL_GRACE_SECONDS, auth["memory_kill_grace_seconds"]),
            "expected_cgroup_memory_max_bytes": auth["expected_cgroup_memory_max_bytes"],
            "guard_semantics": GUARD_SEMANTICS, "not_a_hard_RSS_limit": True}


def run_phase(args, ledger):
    auth = validate_ledger(ledger)
    with prior_budget_lock(auth):
        return _run_phase_with_prior_lock(args, ledger)


def _run_phase_with_prior_lock(args, ledger):
    auth = validate_ledger(ledger)
    if not args.phase or args.phase in {row["phase"] for row in ledger["phases"]} or args.log is None:
        raise ValueError("Fresh phase name and output log required")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("Explicit single-payload command required")
    limits = phase_limits(ledger, auth, args)
    args.log = args.log.resolve()
    prior_tree = root_path(auth["prior_closed_ledger_path"]).parent
    if (args.log.exists() or args.log == args.ledger.resolve() or args.log.is_relative_to(OLD_PILOT)
            or args.log.is_relative_to(prior_tree)):
        raise FileExistsError("Preserve all earlier logs and pilot paths")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.update({name: str(args.threads) for name in THREAD_VARIABLES})
    proof_path = Path(str(args.log) + ".supervisor-proof.json")
    if proof_path.exists():
        raise FileExistsError("Preserve prior immutable launch proof")
    env["R4_RESOURCE_SUPERVISOR_PROOF"] = str(proof_path)
    supervisor_pid = os.getpid()
    supervisor_info = parse_proc_stat(Path("/proc/self/stat").read_text())
    source_sha = sha(__file__)
    native_libc = ctypes.CDLL(None, use_errno=True)
    cgroup_start = read_cgroup_memory()
    if cgroup_start["max_bytes"] != limits["expected_cgroup_memory_max_bytes"]:
        raise RuntimeError("Runtime cgroup differs from the authorized16GiB envelope")
    if cgroup_start["current_bytes"] > limits["cgroup_kill_bytes"]:
        raise RuntimeError("Cgroup already exceeds resource guard; do not launch")
    def child_limits():
        if native_libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "Cannot arm Linux parent-death SIGKILL")
        if os.getppid() != supervisor_pid:
            raise RuntimeError("Supervisor exited before child resource initialization")
        resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu_soft_limit_seconds"], limits["cpu_hard_limit_seconds"]))
        resource.setrlimit(resource.RLIMIT_AS, (args.memory_gib * 1024**3,) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        child_info = parse_proc_stat(Path("/proc/self/stat").read_text())
        soft_as, hard_as = resource.getrlimit(resource.RLIMIT_AS)
        proof = {"schema": PROOF_SCHEMA, "guard_semantics": GUARD_SEMANTICS,
                 "supervisor_pid": supervisor_pid, "supervisor_start_ticks": supervisor_info["start_ticks"],
                 "child_pid": os.getpid(), "child_start_ticks": child_info["start_ticks"],
                 "supervisor_script_path": str(Path(__file__).resolve()), "supervisor_script_sha256": source_sha,
                 "ledger_path": str(args.ledger.resolve()), "phase": args.phase, "started_at_utc": utc().isoformat(),
                 "address_space_soft_limit_bytes": soft_as, "address_space_hard_limit_bytes": hard_as,
                 "rss_stop_bytes": limits["rss_stop_bytes"], "cgroup_kill_bytes": limits["cgroup_kill_bytes"],
                 "expected_cgroup_memory_max_bytes": limits["expected_cgroup_memory_max_bytes"],
                 "monitor_interval_seconds": limits["monitor_interval_seconds"],
                 "memory_kill_grace_seconds": limits["memory_kill_grace_seconds"],
                 "cpu_soft_limit_seconds": limits["cpu_soft_limit_seconds"],
                 "cpu_hard_limit_seconds": limits["cpu_hard_limit_seconds"],
                 "wall_limit_seconds": limits["wall_limit_seconds"], "threads": args.threads,
                 "cgroup_membership": Path("/proc/self/cgroup").read_text().strip(),
                 "parent_death_signal": "SIGKILL",
                 "proof_is_immutable_launch_contract_not_live_monitor_status": True}
        data = (json.dumps(proof, indent=2, allow_nan=False) + "\n").encode()
        fd = os.open(proof_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            # Preserve any incomplete proof as evidence; never start without it.
            raise
    row = {"phase": args.phase, "started_at_utc": utc().isoformat(), "status": "RUNNING",
           "command": command, "log": str(args.log), **limits,
           "supervisor_proof_path": str(proof_path),
           "prior_V2_exclusive_lock_held": True,
           "cgroup_memory_before_launch": cgroup_start,
           "environment_thread_limits": {name: env[name] for name in THREAD_VARIABLES}}
    ledger["phases"].append(row)
    save(args.ledger, ledger)
    start = time.monotonic()
    stop_request = {"signal": None}
    def supervisor_stop(signum, _frame):
        stop_request["signal"] = signum
    old_handlers = {sig: signal.signal(sig, supervisor_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    wall_terminated = False
    hard_killed = False
    cleanup_deadline = limits["wall_limit_seconds"]
    memory_terminated, monitor_failed = False, False
    peak_rss, peak_cgroup, samples = 0, cgroup_start["current_bytes"], 0
    usage = None
    child = None
    try:
        with args.log.open("xb") as handle:
            try:
                child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env,
                                         preexec_fn=child_limits, start_new_session=True)
            except Exception as exc:
                row.update(status="FAILED_TO_START", ended_at_utc=utc().isoformat(), actual_cpu_seconds=0.0,
                           elapsed_wall_seconds=time.monotonic() - start, exit_code=None, failure=repr(exc),
                           log_sha256=sha(args.log))
                save(args.ledger, ledger)
                raise
            row["pid"] = child.pid
            row["supervisor_proof_sha256"] = sha(proof_path)
            save(args.ledger, ledger)
            while True:
                pid, status, usage = os.wait4(child.pid, os.WNOHANG)
                if pid:
                    child.returncode = os.waitstatus_to_exitcode(status)
                    break
                elapsed = time.monotonic() - start
                try:
                    snapshot = process_group_snapshot(child.pid)
                    if child.pid not in snapshot["pids"]:
                        # It may have exited during/proc traversal. Reap that
                        # legitimate race; otherwise never interpret missing
                        # leader RSS as zero for a live high-AS payload.
                        observed_pid, observed_status, observed_usage = os.wait4(child.pid, os.WNOHANG)
                        if observed_pid:
                            usage = observed_usage
                            child.returncode = os.waitstatus_to_exitcode(observed_status)
                            break
                        raise RuntimeError("Live payload leader missing from RSS observation")
                    cgroup = read_cgroup_memory()
                    samples += 1
                    peak_rss = max(peak_rss, snapshot["rss_bytes"])
                    peak_cgroup = max(peak_cgroup, cgroup["current_bytes"])
                    if cgroup["max_bytes"] != limits["expected_cgroup_memory_max_bytes"]:
                        raise RuntimeError("Runtime cgroup envelope changed")
                    if cgroup["current_bytes"] > limits["cgroup_kill_bytes"] and not hard_killed:
                        hard_killed = signal_group(child.pid, signal.SIGKILL)
                        memory_terminated = True
                        row.update(resource_stop_reason="CGROUP_MEMORY_IMMEDIATE_KILL",
                                   memory_stop_snapshot=snapshot, memory_stop_cgroup=cgroup,
                                   SIGKILL_requested_at_utc=utc().isoformat())
                        save(args.ledger, ledger)
                    elif snapshot["rss_bytes"] > limits["rss_stop_bytes"] and not memory_terminated:
                        memory_terminated = signal_group(child.pid, signal.SIGTERM)
                        # Leave one polling interval so the nominal KILL is within
                        # the authorized1s grace; OS scheduling can still delay it.
                        cleanup_deadline = min(cleanup_deadline, elapsed + max(0, limits["memory_kill_grace_seconds"] - limits["monitor_interval_seconds"]))
                        row.update(resource_stop_reason="PROCESS_GROUP_RSS_GUARD",
                                   memory_stop_snapshot=snapshot, memory_stop_cgroup=cgroup,
                                   SIGTERM_requested_at_utc=utc().isoformat())
                        save(args.ledger, ledger)
                except Exception as exc:
                    monitor_failed = True
                    hard_killed = signal_group(child.pid, signal.SIGKILL) or hard_killed
                    row.update(resource_stop_reason="MONITOR_FAILED_CLOSED", monitor_exception=repr(exc),
                               SIGKILL_requested_at_utc=utc().isoformat())
                    save(args.ledger, ledger)
                if not memory_terminated and not hard_killed and not wall_terminated and (elapsed >= limits["wall_SIGTERM_after_seconds"] or stop_request["signal"] is not None):
                    if signal_group(child.pid, signal.SIGTERM):
                        wall_terminated = True
                        cleanup_deadline = min(limits["wall_limit_seconds"], elapsed + 60)
                        row.update(SIGTERM_requested_at_utc=utc().isoformat(), supervisor_signal=stop_request["signal"],
                                   resource_stop_reason="SUPERVISOR_INTERRUPTED" if stop_request["signal"] else "WALL_CLEANUP_WINDOW")
                        save(args.ledger, ledger)
                if elapsed >= cleanup_deadline and not hard_killed:
                    hard_killed = signal_group(child.pid, signal.SIGKILL)
                    row["SIGKILL_requested_at_utc"] = utc().isoformat()
                time.sleep(limits["monitor_interval_seconds"])
    except BaseException as exc:
        if child is None:
            # FAILED_TO_START is already persisted by the launch handler.
            raise
        monitor_failed = True
        row.update(resource_stop_reason="SUPERVISOR_EXCEPTION_CLEANUP",
                   supervisor_exception_type=type(exc).__name__, supervisor_exception=repr(exc))
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        if child is not None and child.returncode is None:
            hard_killed = signal_group(child.pid, signal.SIGKILL) or hard_killed
            _pid, status, usage = os.wait4(child.pid, 0)
            child.returncode = os.waitstatus_to_exitcode(status)
    cpu = usage.ru_utime + usage.ru_stime
    try:
        residual = process_group_snapshot(child.pid)
    except Exception as exc:
        monitor_failed = True
        signal_group(child.pid, signal.SIGKILL)
        residual = {"pids": []}
        row["post_exit_process_audit_failure"] = repr(exc)
    if residual["pids"]:
        signal_group(child.pid, signal.SIGKILL)
        row["residual_processes_killed_after_main_payload_exit"] = residual["pids"]
    row.update(status="EXITED" if child.returncode == 0 and not any((wall_terminated, memory_terminated, monitor_failed, hard_killed, bool(residual["pids"]))) else "PARTIAL_OR_FAILED",
               ended_at_utc=utc().isoformat(), actual_cpu_seconds=cpu,
               elapsed_wall_seconds=time.monotonic() - start, peak_rss_kib=usage.ru_maxrss,
               exit_code=child.returncode, wall_terminated=wall_terminated, hard_killed=hard_killed,
               memory_terminated=memory_terminated, monitor_failed=monitor_failed,
               memory_monitor_samples=samples, peak_group_rss_bytes_sampled=peak_rss,
               peak_cgroup_memory_bytes_sampled=peak_cgroup,
               log_sha256=sha(args.log))
    ledger["consumed_cpu_seconds"] += cpu
    ledger["remaining_quantum_cpu_seconds"] = max(0.0, ledger["quantum_cpu_cap_seconds"] - ledger["consumed_cpu_seconds"])
    if ledger["consumed_cpu_seconds"] > ledger["quantum_cpu_cap_seconds"]:
        row["status"] = "BUDGET_VIOLATION_REQUIRES_REVIEW"
    save(args.ledger, ledger)
    return row


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--initialize", action="store_true")
    p.add_argument("--authorization", type=Path)
    p.add_argument("--authorization-sha256")
    p.add_argument("--source-checkpoint", type=Path)
    p.add_argument("--source-checkpoint-sha256")
    p.add_argument("--source-base-commit")
    p.add_argument("--total-cpu-seconds", type=int, default=MAX_CPU)
    p.add_argument("--total-wall-seconds", type=int, default=MAX_WALL)
    p.add_argument("--phase")
    p.add_argument("--cpu-seconds", type=int)
    p.add_argument("--wall-seconds", type=int)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--memory-gib", type=int, default=24)
    p.add_argument("--log", type=Path)
    p.add_argument("command", nargs=argparse.REMAINDER)
    return p


def main():
    args = parser().parse_args()
    args.ledger = args.ledger.resolve()
    if args.ledger.is_relative_to(OLD_PILOT):
        raise ValueError("Never write a budget inside the immutable pilot directory")
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    with args.ledger.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        global_lock = ROOT / ".runtime/pyscf-v3-global.lock"
        global_lock.parent.mkdir(parents=True, exist_ok=True)
        with global_lock.open("a") as global_handle:
            fcntl.flock(global_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.initialize:
                initialize(args)
                print(json.dumps({"status": "NEW_AUTHORIZED_LEDGER_INITIALIZED", "ledger": str(args.ledger)}))
                return
            row = run_phase(args, read(args.ledger))
            print(json.dumps(row), flush=True)
            raise SystemExit(0 if row["status"] == "EXITED" else 1)


if __name__ == "__main__":
    main()
