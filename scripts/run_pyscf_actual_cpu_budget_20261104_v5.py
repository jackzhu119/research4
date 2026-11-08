"""Continue a closed V4 pool with V5 bounded exit-reaping transition handling.

The earlier pilot ledger is immutable. This runner binds the separate continuation
authorization, source checkpoint and source revision, and never resets a ledger.
Migration preserves every old phase, actual cost, cap and original deadline.
CPU means actual wait4 user+system seconds, not wall seconds multiplied by threads.
CPU and RSS polling are best effort, not exact kernel hard guarantees. The
independent sampled CPU guard uses the whole process history since birth and
new-batch completed actual cost, without resetting the first observed sample.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import re
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
OLD_PILOT = ROOT / "reports/pyscf-spinosin-pilot-20261031"
SOURCE_BASE = "0629d4c6a1496dc96b6852c07755b61c42efadea"
SCHEMA = "research4-pyscf-actual-cpu-budget-v5"
MAX_CPU, MAX_WALL = 129600, 64800
MAX_PHASE_CPU, MAX_PHASE_WALL = 122400, 43200
RSS_STOP_BYTES = 10 * 1024**3
CGROUP_KILL_BYTES = 12 * 1024**3
MONITOR_INTERVAL_SECONDS = .1
MEMORY_KILL_GRACE_SECONDS = 1.0
CGROUP_MEMORY_MAX_BYTES = 16 * 1024**3
GUARD_SEMANTICS = "BEST_EFFORT_RSS_GUARD_NOT_HARD_RSS_LIMIT"
PROOF_SCHEMA = "research4-pyscf-actual-cpu-supervisor-proof-v5"
ACTUAL_CPU_RESERVE_SECONDS = 120
ACTUAL_CPU_GUARD_SOURCE = "PROC_STAT_CUMULATIVE_FROM_PROCESS_BIRTH"
CPU_GUARD_KILL_GRACE_SECONDS = 1.0
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
EXIT_REAP_GRACE_SECONDS = .5
V4_SOURCE = ROOT / "scripts/run_pyscf_actual_cpu_budget_20261101.py"
V4_SHA = "354ff51b5c2af2810033656d9972948262cfebb4dabb3978fca2d021f97e2b8f"
MIGRATION_AUTH_SCHEMA = "research4-v5-budget-migration-authorization-v1"


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
    return {"pid": int(value[:left].strip()), "state": fields[0],
            "ppid": int(fields[1]), "pgrp": int(fields[2]), "session": int(fields[3]),
            "user_ticks": int(fields[11]), "system_ticks": int(fields[12]),
            "child_user_ticks": int(fields[13]), "child_system_ticks": int(fields[14]),
            "start_ticks": int(fields[19]), "vsize_bytes": int(fields[20]),
            "rss_pages": max(0, int(fields[21]))}


def read_process_identity(pid, proc_root=Path("/proc")):
    directory = Path(proc_root)/str(pid)
    first = parse_proc_stat((directory/"stat").read_text())
    uid_line = next(line for line in (directory/"status").read_text().splitlines() if line.startswith("Uid:"))
    uids = [int(value) for value in uid_line.split()[1:]]
    second = parse_proc_stat((directory/"stat").read_text())
    if len(uids) != 4 or first["start_ticks"] != second["start_ticks"]:
        raise ValueError("Unstable process identity observation")
    return {**second, "uid": uids[0], "uids": uids}


def identity_matches(expected, actual):
    keys = ("pid", "start_ticks", "uid", "pgrp", "session")
    return (all(type(expected.get(key)) is int and actual.get(key) == expected[key] for key in keys)
            and expected["pid"] == expected["pgrp"] == expected["session"]
            and actual.get("state") not in ("Z", "X", "x")
            and all(uid == expected["uid"] for uid in actual.get("uids", [actual.get("uid")]*4)))


def owned_identity_matches(expected, actual):
    """Same original child identity, allowing only its terminal-state transition."""
    keys = ("pid", "start_ticks", "uid", "pgrp", "session")
    return (all(type(expected.get(key)) is int and actual.get(key) == expected[key] for key in keys)
            and expected["pid"] == expected["pgrp"] == expected["session"]
            and all(uid == expected["uid"] for uid in actual.get("uids", [actual.get("uid")]*4)))


def terminal_reap_observation(snapshot, expected, grace, now, *, reader=None, waiter=None):
    """Bound the exit-to-wait4 race; never reinterpret an unreadable live child.

    Called only when the group scan missed the leader or observed Z/X/x. It
    retries wait4, then independently checks the original owned child. The
    caller continues cgroup, CPU-watermark and wall guards during PENDING.
    """
    reader = reader or read_process_identity
    waiter = waiter or (lambda pid: os.wait4(pid, os.WNOHANG))
    rows = [r for r in snapshot["processes"] if r["pid"] == expected["pid"]]
    if len(rows) > 1 or any(r["pid"] != expected["pid"] for r in snapshot["processes"]):
        raise RuntimeError("Unexpected process group members during terminal transition")
    waited = waiter(expected["pid"])
    if waited[0]:
        if waited[0] != expected["pid"]:
            raise RuntimeError("wait4 returned an unrelated PID")
        return {"state": "REAPED", "wait4": waited, "grace": grace}
    try:
        observed = reader(expected["pid"])
    except (FileNotFoundError, ProcessLookupError):
        observed = None
    # PermissionError, malformed/unstable identity and arbitrary I/O failures are
    # intentionally not converted to a terminal transition: fail closed.
    if observed is not None:
        if not owned_identity_matches(expected, observed):
            raise RuntimeError("Owned child identity changed during reaping transition")
        if observed.get("state") not in ("Z", "X", "x"):
            raise RuntimeError("Live payload leader missing from RSS observation")
    for value in rows:
        if not owned_identity_matches(expected, value) or value.get("state") not in ("Z", "X", "x"):
            raise RuntimeError("Group scan and owned terminal identity disagree")
    if grace.get("started_monotonic") is None:
        grace["started_monotonic"] = float(now)
        grace["started_at_utc"] = utc().isoformat()
        grace["maximum_seconds"] = EXIT_REAP_GRACE_SECONDS
    elapsed = now - grace["started_monotonic"]
    event = {"observed_at_utc": utc().isoformat(), "elapsed_grace_seconds": elapsed,
             "owned_leader_state": "GONE" if observed is None else observed["state"],
             "group_leader_states": [r.get("state") for r in rows],
             "RSS_is_not_replaced_by_zero": True}
    grace.setdefault("observations", []).append(event)
    if elapsed < 0 or elapsed >= EXIT_REAP_GRACE_SECONDS:
        raise RuntimeError("EXIT_REAP_GRACE_EXPIRED: original terminal child still not wait4-ready")
    ticks = None
    if observed is not None:
        values = [observed[key] for key in ("user_ticks", "system_ticks", "child_user_ticks", "child_system_ticks")]
        if any(type(v) is not int or v < 0 for v in values):
            raise ValueError("Invalid terminal cumulative CPU ticks")
        ticks = sum(values)
    return {"state": "PENDING", "grace": grace, "terminal_cpu_ticks": ticks,
            "remaining_grace_seconds": EXIT_REAP_GRACE_SECONDS - elapsed}


def bounded_owned_reap(pid, *, waiter=None, clock=None, sleeper=None, maximum_seconds=.5):
    """Cleanup does not wait forever after an actual monitor failure."""
    if not 0 < maximum_seconds <= EXIT_REAP_GRACE_SECONDS:
        raise ValueError("Bounded cleanup reap maximum is0..0.5seconds")
    waiter = waiter or (lambda target: os.wait4(target, os.WNOHANG))
    clock, sleeper = clock or time.monotonic, sleeper or time.sleep
    started = clock()
    for _ in range(102):  # Also finite under a broken/nonadvancing test clock.
        value = waiter(pid)
        if value[0]:
            if value[0] != pid:
                raise RuntimeError("Cleanup wait4 returned an unrelated PID")
            return value
        remaining = maximum_seconds - (clock()-started)
        if remaining <= 0:
            return None
        sleeper(min(.01, remaining))
    return None


def signal_bound_group(expected, signum, reader=None, sender=None):
    """Recheck identity before group signaling; no atomic killpg identity claim."""
    reader, sender = reader or read_process_identity, sender or os.killpg
    if expected["pid"] in (os.getpid(), os.getppid()) or expected["pgrp"] == os.getpgrp():
        return {"sent": False, "reason": "REFUSE_SUPERVISOR_OR_PARENT_GROUP"}
    try:
        actual = reader(expected["pid"])
        if not identity_matches(expected, actual):
            return {"sent": False, "reason": "PROCESS_IDENTITY_CHANGED_OR_ENDED"}
        sender(expected["pgrp"], signum)
        return {"sent": True, "reason": "IDENTITY_RECHECKED_GROUP_SIGNAL_SENT"}
    except (OSError, ValueError, StopIteration) as exc:
        return {"sent": False, "reason": "PROCESS_OBSERVATION_OR_SIGNAL_FAILED", "error": repr(exc)}


def actual_cpu_observation(snapshot, expected, clock_ticks, closed_cpu, batch_cap, phase_grant, reserve):
    if type(clock_ticks) is not int or clock_ticks <= 0 or not 0 < reserve < min(phase_grant, batch_cap-closed_cpu):
        raise ValueError("Invalid actual CPU guard clock/reserve")
    matches = [row for row in snapshot["processes"] if row["pid"] == expected["pid"]]
    if len(matches) != 1 or not identity_matches(expected, matches[0]):
        raise ValueError("Live phase leader identity differs from actual CPU guard")
    if len(snapshot["processes"]) != 1:
        raise ValueError("Single quantum payload required; unexpected extra process-group member")
    ticks = 0
    for process in snapshot["processes"]:
        if process["pgrp"] != expected["pgrp"] or process["session"] != expected["session"] or process["uid"] != expected["uid"]:
            raise ValueError("Process group membership differs from bound session/UID")
        values = [process[key] for key in ("user_ticks", "system_ticks", "child_user_ticks", "child_system_ticks")]
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("Invalid cumulative process CPU ticks")
        ticks += sum(values)
    phase_cpu = ticks/clock_ticks
    threshold = min(phase_grant, batch_cap-closed_cpu)-reserve
    return {"clock_ticks_per_second": clock_ticks, "cumulative_phase_cpu_seconds_sampled": phase_cpu,
            "closed_batch_actual_cpu_seconds": closed_cpu,
            "batch_cpu_seconds_sampled": closed_cpu+phase_cpu,
            "stop_threshold_phase_cpu_seconds": threshold, "cleanup_reserve_seconds": reserve,
            "phase_boundary_near": phase_cpu+reserve >= phase_grant,
            "batch_boundary_near": closed_cpu+phase_cpu+reserve >= batch_cap,
            "should_stop": phase_cpu >= threshold, "clock_source": ACTUAL_CPU_GUARD_SOURCE,
            "no_initial_sample_subtracted": True,
            "scope": "Single TGID self plus waited-child CPU; native threads not enumerated; unexpected live group members rejected"}


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
                processes.append({**value, "uid": directory.stat().st_uid,
                                  "rss_bytes": value["rss_pages"] * page_bytes})
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


def resource_runtime_snapshot():
    child, parent = os.getpid(), os.getppid()
    child_identity = read_process_identity(child)
    parent_identity = read_process_identity(parent)
    cgroup = read_cgroup_memory()
    return {"child_pid": child, "supervisor_pid": parent,
        "child_start_ticks": child_identity["start_ticks"], "supervisor_start_ticks": parent_identity["start_ticks"],
        "child_uid": child_identity["uid"], "child_pgid": child_identity["pgrp"],
        "child_session_id": child_identity["session"],
        "parent_command": Path(f"/proc/{parent}/cmdline").read_bytes().split(b"\0"),
        "parent_cwd": str(Path(f"/proc/{parent}/cwd").resolve()),
        "address_space_limits": resource.getrlimit(resource.RLIMIT_AS), "CPU_limits": resource.getrlimit(resource.RLIMIT_CPU),
        "child_cgroup": Path(f"/proc/{child}/cgroup").read_text(), "parent_cgroup": Path(f"/proc/{parent}/cgroup").read_text(),
        "cgroup_memory_max_bytes": cgroup["max_bytes"], "cgroup_memory_current_bytes": cgroup["current_bytes"],
        "clock_ticks_per_second": int(os.sysconf("SC_CLK_TCK"))}


def verify_resource_supervision(proof_path=None, *, expected_threads=None):
    """V5 worker entry gate; verifies a real new parent, not renamed V3 proof."""
    supplied = proof_path or os.environ.get("R4_RESOURCE_SUPERVISOR_PROOF")
    if not supplied:
        raise ValueError("Actual-CPU V5 supervision proof required before quantum work")
    path = Path(supplied).resolve()
    raw = path.read_bytes()
    if len(raw) > 65536:
        raise ValueError("Unexpectedly large supervision proof")
    proof = json.loads(raw)
    if proof.get("schema") != PROOF_SCHEMA or proof.get("guard_semantics") != GUARD_SEMANTICS:
        raise ValueError("A real independent V5 actual-CPU proof is required")
    if proof.get("terminal_exit_reap_grace_seconds") != EXIT_REAP_GRACE_SECONDS:
        raise ValueError("V5 requires its explicit bounded terminal reaping policy")
    observed = resource_runtime_snapshot()
    for key in ("supervisor_pid", "supervisor_start_ticks", "child_pid", "child_start_ticks"):
        if type(proof.get(key)) is not int or proof[key] != observed[key]:
            raise ValueError("Live supervisor/child identity mismatch: " + key)
    source = ROOT/"scripts/run_pyscf_actual_cpu_budget_20261104_v5.py"
    if root_path(proof.get("supervisor_script_path", "")) != source or proof.get("supervisor_script_sha256") != sha(source):
        raise ValueError("Actual V5 supervisor source path/SHA differs")
    commands = [Path(os.fsdecode(token)) for token in observed["parent_command"] if token]
    if not any((token if token.is_absolute() else Path(observed["parent_cwd"])/token).resolve() == source for token in commands):
        raise ValueError("Actual parent command is not the V5 supervisor")
    expected_values = {"address_space_soft_limit_bytes": 24*1024**3, "address_space_hard_limit_bytes": 24*1024**3,
        "rss_stop_bytes": 10*1024**3, "cgroup_kill_bytes": 12*1024**3,
        "expected_cgroup_memory_max_bytes": CGROUP_MEMORY_MAX_BYTES}
    if any(type(proof.get(key)) is not int or proof[key] != value for key,value in expected_values.items()):
        raise ValueError("Unreviewed V5 resource policy")
    if tuple(observed["address_space_limits"]) != (24*1024**3,)*2:
        raise ValueError("Actual RLIMIT_AS differs from proof")
    cpu = [proof.get("cpu_soft_limit_seconds"), proof.get("cpu_hard_limit_seconds")]
    if (any(type(value) is not int or value <= 0 for value in cpu) or cpu[0] >= cpu[1]
            or cpu[1] > MAX_PHASE_CPU or tuple(cpu) != tuple(observed["CPU_limits"])):
        raise ValueError("Actual supplementary profiling RLIMIT_CPU differs")
    if (proof.get("target_uid") != observed["child_uid"] or proof.get("target_pgid") != observed["child_pgid"]
            or proof.get("target_session_id") != observed["child_session_id"]
            or observed["child_pid"] != observed["child_pgid"] or observed["child_pid"] != observed["child_session_id"]):
        raise ValueError("Actual worker UID/PGID/session differs")
    if expected_threads is not None and proof.get("threads") != expected_threads:
        raise ValueError("Worker threads differ from supervised allocation")
    if type(proof.get("threads")) is not int or not 1 <= proof["threads"] <= 4:
        raise ValueError("Maximum four existing cores")
    if (observed["child_cgroup"] != observed["parent_cgroup"]
            or observed["cgroup_memory_max_bytes"] != CGROUP_MEMORY_MAX_BYTES
            or observed["cgroup_memory_current_bytes"] >= CGROUP_KILL_BYTES):
        raise ValueError("Actual cgroup envelope differs or is already above stop threshold")
    for key, maximum in (("monitor_interval_seconds", .1), ("memory_kill_grace_seconds", 1), ("cpu_guard_kill_grace_seconds", 1)):
        value = proof.get(key)
        if type(value) not in (int,float) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError("Unreviewed actual CPU/RSS guard timing")
    reserve, closed, cap = (proof.get(key) for key in ("actual_cpu_cleanup_reserve_seconds", "actual_cpu_closed_batch_cost_seconds", "actual_cpu_batch_cap_seconds"))
    if (proof.get("actual_cpu_guard_enabled") is not True or proof.get("actual_cpu_guard_source") != ACTUAL_CPU_GUARD_SOURCE
            or type(reserve) is not int or not 120 <= reserve <= 600
            or type(cap) is not int or not 0 < cap <= MAX_CPU
            or type(closed) not in (int,float) or not math.isfinite(closed) or closed < 0
            or proof.get("actual_cpu_clock_ticks_per_second") != observed["clock_ticks_per_second"]
            or proof.get("actual_cpu_stop_threshold_seconds") != min(cpu[1],cap-closed)-reserve):
        raise ValueError("Actual cumulative CPU guard parameters differ")
    if type(proof.get("wall_limit_seconds")) is not int or not 61 <= proof["wall_limit_seconds"] <= MAX_PHASE_WALL:
        raise ValueError("Finite unchanged V5 phase wall bound required")
    ledger = read(root_path(proof.get("ledger_path", "")))
    phases = [row for row in ledger.get("phases",[]) if row.get("phase") == proof.get("phase")]
    if (ledger.get("schema") != SCHEMA or len(phases) != 1 or phases[0].get("status") != "RUNNING"
            or sum(row.get("status") == "RUNNING" for row in ledger["phases"]) != 1
            or ledger.get("runner_sha256") != sha(source)):
        raise ValueError("Actual new V5 ledger must contain one matching live phase")
    if ledger.get("quantum_cpu_cap_seconds") != cap or ledger.get("consumed_cpu_seconds") != closed or ledger.get("prior_budget_is_historical_only") is not True:
        raise ValueError("Live independent V5 pool differs from proof; old costs are historical only")
    for key in ("actual_cpu_stop_threshold_seconds", "actual_cpu_clock_ticks_per_second", "actual_cpu_cleanup_reserve_seconds",
                "actual_cpu_closed_batch_cost_seconds", "actual_cpu_batch_cap_seconds", "actual_cpu_guard_source"):
        if phases[0].get(key) != proof[key]:
            raise ValueError("CPU guard proof differs from actual running row: " + key)
    return {"verified_at_utc": utc().isoformat(), "proof_path": str(path), "proof_sha256": hashlib.sha256(raw).hexdigest(),
        "proof": proof, "observed_CPU_limits_seconds": list(observed["CPU_limits"]),
        "observed_address_space_limits_bytes": list(observed["address_space_limits"]),
        "observed_cgroup_memory_max_bytes": observed["cgroup_memory_max_bytes"],
        "observed_cgroup_memory_current_bytes": observed["cgroup_memory_current_bytes"],
        "observed_child_pid": observed["child_pid"], "observed_supervisor_pid": observed["supervisor_pid"],
        "AS_is_virtual_address_cap_not_RSS_limit": True, "RSS_guard_is_best_effort_not_a_hard_limit": True,
        "live_parent_process_and_source_verified": True, "actual_CPU_sampling_is_best_effort": True,
        "profiling_RLIMIT_is_supplementary_not_exact_actual_CPU_guarantee": True,
        "startup_proof_is_not_future_monitor_health_confirmation": True}


def validate_prior_closed_ledger(authorization):
    path = root_path(authorization.get("prior_closed_ledger_path", ""))
    expected_sha = authorization.get("prior_closed_ledger_sha256")
    if not path.is_file() or not expected_sha or sha(path) != expected_sha:
        raise ValueError("Prior closed V3 ledger bytes differ from explicit authorization")
    prior = read(path)
    if prior.get("schema") != "research4-pyscf-monitored-budget-v3":
        raise ValueError("New budget must bind the specifically closed V3 ledger")
    phases = prior.get("phases")
    if not isinstance(phases, list) or any(row.get("status") == "RUNNING" for row in phases):
        raise RuntimeError("Prior V3 phase is unresolved; forbid parallel budgets")
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
    if (value.get("schema") != MIGRATION_AUTH_SCHEMA or value.get("migration_preserves_existing_budget") is not True
            or value.get("new_budget_is_separately_authorized") is not False
            or value.get("prior_budget_preserved") is not True
            or value.get("v5_supervisor_source_sha256") != sha(__file__)):
        raise ValueError("V5 must explicitly migrate the existing pool, not create a new budget")
    if not isinstance(source_base, str) or not re.fullmatch(r"[0-9a-f]{40}", source_base) or value.get("source_base_commit") != source_base:
        raise ValueError("New batch must bind its explicit full source-base revision")
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
    if value.get("prior_budget_is_historical_only") is not True:
        raise ValueError("Prior V3 costs must be historical; do not silently reuse its48hour pool")
    reserve = value.get("actual_cpu_cleanup_reserve_seconds")
    if type(reserve) is not int or not ACTUAL_CPU_RESERVE_SECONDS <= reserve <= 600:
        raise ValueError("Actual CPU reserve must be120..600seconds")
    grace = value.get("cpu_guard_kill_grace_seconds")
    if type(grace) not in (int, float) or not math.isfinite(grace) or not 0 < grace <= 1:
        raise ValueError("Actual CPU guard cleanup must be<=1second")
    if value.get("actual_cpu_guard_source") != ACTUAL_CPU_GUARD_SOURCE:
        raise ValueError("Full-birth proc CPU observation must be explicitly authorized")
    validate_prior_closed_ledger(value)
    return value


@contextmanager
def prior_budget_lock(authorization):
    """Use exactly the old V3 runner's lock, blocking concurrent old phases."""
    path = root_path(authorization["prior_closed_ledger_path"]).with_suffix(".lock")
    with path.open("rb") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def initialize(args):
    raise ValueError("V5 forbids fresh initialization; use --migrate-from-closed-v4 and exact SHA")


def frozen_v4_module():
    if sha(V4_SOURCE) != V4_SHA:
        raise ValueError("Immutable V4 source changed")
    spec = importlib.util.spec_from_file_location("research4_frozen_v4_migration_verifier", V4_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def migration_invariants(old, authorization):
    """Pure invariant checks; copying never resolves/relabels old failures."""
    if old.get("schema") != "research4-pyscf-actual-cpu-budget-v4" or old.get("runner_sha256") != V4_SHA:
        raise ValueError("Migration source must be the original closed V4 pool")
    phases = old.get("phases")
    if not isinstance(phases, list) or any(row.get("status") == "RUNNING" or not row.get("ended_at_utc") for row in phases):
        raise ValueError("Every V4 phase must already be closed")
    costs = [row.get("actual_cpu_seconds") for row in phases]
    if any(type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0 for cost in costs):
        raise ValueError("Every migrated phase requires a known actual CPU cost")
    total = sum(costs)
    if not math.isclose(total, old.get("consumed_cpu_seconds", float("nan")), rel_tol=1e-12, abs_tol=1e-7):
        raise ValueError("V4 per-phase costs differ from the cumulative total")
    if old.get("quantum_cpu_cap_seconds", MAX_CPU+1) > MAX_CPU or old.get("run_wall_cap_seconds", MAX_WALL+1) > MAX_WALL:
        raise ValueError("Old pool exceeds bounded V5 envelope")
    expected = {"original_created_at_utc": old["created_at_utc"],
                "original_run_deadline_at_utc": old["run_deadline_at_utc"],
                "original_quantum_cpu_cap_seconds": old["quantum_cpu_cap_seconds"],
                "original_run_wall_cap_seconds": old["run_wall_cap_seconds"],
                "original_consumed_cpu_seconds": old["consumed_cpu_seconds"]}
    if any(authorization.get(key) != value for key, value in expected.items()):
        raise ValueError("Authorization does not bind original caps, deadline and consumed cost")
    if (datetime.fromisoformat(old["created_at_utc"]) + timedelta(seconds=old["run_wall_cap_seconds"])
            != datetime.fromisoformat(old["run_deadline_at_utc"])):
        raise ValueError("Original V4 deadline is inconsistent")
    return total


@contextmanager
def migrated_v4_lock(authorization):
    path = root_path(authorization["migrate_from_closed_v4_path"]).with_suffix(".lock")
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def migrate_from_closed_v4(args):
    if args.ledger.exists():
        raise FileExistsError("Never overwrite/reset a V5 or historical ledger")
    required = (args.authorization, args.authorization_sha256, args.source_checkpoint,
                args.source_checkpoint_sha256, args.source_base_commit)
    if not all(required) or not args.migrate_from_closed_v4 or not args.migrate_from_sha256:
        raise ValueError("Explicit closed V4 migration and authorization/source hashes required")
    auth = validate_authorization(*required)
    source = args.migrate_from_closed_v4.resolve()
    if (source == args.ledger.resolve() or root_path(auth.get("migrate_from_closed_v4_path", "")) != source
            or auth.get("migrate_from_closed_v4_sha256") != args.migrate_from_sha256):
        raise ValueError("Migration source differs from explicit authorization")
    with migrated_v4_lock(auth), prior_budget_lock(auth):
        if sha(source) != args.migrate_from_sha256:
            raise ValueError("Closed V4 ledger bytes changed")
        old = read(source)
        frozen_v4_module().validate_ledger(old)
        migration_invariants(old, auth)
        if old["source_checkpoint_sha256"] != args.source_checkpoint_sha256 or root_path(old["source_checkpoint_path"]) != args.source_checkpoint.resolve():
            raise ValueError("Migration cannot change the original source checkpoint")
        if args.ledger.resolve().is_relative_to(root_path(auth["prior_closed_ledger_path"]).parent):
            raise ValueError("Do not write V5 inside immutable V3 reports")
        ledger = json.loads(json.dumps(old))
        ledger.update(schema=SCHEMA, runner_sha256=sha(__file__),
            authorization_path=str(args.authorization.resolve()), authorization_sha256=args.authorization_sha256,
            source_base_commit=args.source_base_commit, migrated_at_utc=utc().isoformat(),
            migration={"source_v4_path": str(source), "source_v4_sha256": args.migrate_from_sha256,
                       "original_runner_sha256": V4_SHA, "new_runner_sha256": sha(__file__),
                       "original_authorization_path": old["authorization_path"],
                       "original_authorization_sha256": old["authorization_sha256"],
                       "original_source_base_commit": old["source_base_commit"],
                       "original_phase_count": len(old["phases"]),
                       "original_phase_array_sha256": hashlib.sha256(json.dumps(old["phases"],sort_keys=True).encode()).hexdigest(),
                       "new_budget_created": False, "prior_phase_statuses_rewritten": False,
                       "prior_actual_CPU_preserved": old["consumed_cpu_seconds"]})
        validate_ledger(ledger)
        if args.ledger.exists():
            raise FileExistsError("V5 ledger appeared during migration")
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
    migration = ledger.get("migration", {})
    source = root_path(auth["migrate_from_closed_v4_path"])
    if (migration.get("source_v4_path") != str(source) or migration.get("source_v4_sha256") != auth["migrate_from_closed_v4_sha256"]
            or sha(source) != auth["migrate_from_closed_v4_sha256"]):
        raise ValueError("Migrated immutable V4 source ledger changed")
    old = read(source)
    migration_invariants(old, auth)
    count = migration.get("original_phase_count")
    if type(count) is not int or count != len(old["phases"]) or ledger.get("phases", [])[:count] != old["phases"]:
        raise ValueError("Migrated historical V4 phases were modified or omitted")
    for key in ("created_at_utc", "run_deadline_at_utc", "quantum_cpu_cap_seconds", "run_wall_cap_seconds"):
        if ledger.get(key) != old[key]:
            raise ValueError("Migration enlarged/reset the original budget: " + key)
    if validate_prior_closed_ledger(auth) != ledger.get("prior_closed_ledger"):
        raise ValueError("Closed prior budget binding changed")
    if not auth["actual_cpu_cleanup_reserve_seconds"]+5 <= ledger["quantum_cpu_cap_seconds"] <= auth["max_quantum_cpu_seconds"] or not 61 <= ledger["run_wall_cap_seconds"] <= auth["max_run_wall_seconds"]:
        raise ValueError("Ledger caps exceed original authorization")
    if any(row.get("status") == "RUNNING" for row in ledger["phases"]):
        raise RuntimeError("A prior phase is unresolved; do not double-launch")
    if any(row.get("residual_processes_require_review_after_main_payload_exit") for row in ledger["phases"]):
        raise RuntimeError("Residual process-group members require explicit review before another phase")
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
    minimum_cpu = auth["actual_cpu_cleanup_reserve_seconds"]+5
    if not args.cpu_seconds or not minimum_cpu <= args.cpu_seconds <= auth["max_phase_cpu_seconds"]:
        raise ValueError("Phase CPU request must be125..122400seconds and within authorization")
    if not args.wall_seconds or not 61 <= args.wall_seconds <= auth["max_phase_wall_seconds"]:
        raise ValueError("Phase wall request must allow60s cleanup and be<=43200seconds")
    remaining_cpu = ledger["quantum_cpu_cap_seconds"] - ledger["consumed_cpu_seconds"]
    # Linux per-process limits can be observed a little late with several threads.
    # Keep an extra CPU boundary allowance rather than grant the final fraction.
    reserve = 2 * args.threads + 2
    grant = min(args.cpu_seconds, math.floor(remaining_cpu) - reserve)
    remaining_wall = (datetime.fromisoformat(ledger["run_deadline_at_utc"]) - (now or utc())).total_seconds()
    wall = min(args.wall_seconds, math.floor(remaining_wall) - 2)
    if grant < minimum_cpu or wall < 61:
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
            "guard_semantics": GUARD_SEMANTICS, "not_a_hard_RSS_limit": True,
            "actual_cpu_guard_enabled": True, "actual_cpu_guard_source": ACTUAL_CPU_GUARD_SOURCE,
            "actual_cpu_clock_ticks_per_second": int(os.sysconf("SC_CLK_TCK")),
            "actual_cpu_cleanup_reserve_seconds": auth["actual_cpu_cleanup_reserve_seconds"],
            "actual_cpu_stop_threshold_seconds": grant-auth["actual_cpu_cleanup_reserve_seconds"],
            "actual_cpu_closed_batch_cost_seconds": float(ledger["consumed_cpu_seconds"]),
            "actual_cpu_batch_cap_seconds": ledger["quantum_cpu_cap_seconds"],
            "cpu_guard_kill_grace_seconds": auth["cpu_guard_kill_grace_seconds"],
            "profiling_RLIMIT_is_supplementary_not_exact_actual_CPU_guarantee": True}


def run_phase(args, ledger):
    auth = validate_ledger(ledger)
    with migrated_v4_lock(auth), prior_budget_lock(auth):
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
    if cgroup_start["current_bytes"] >= limits["cgroup_kill_bytes"]:
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
                 "target_uid": os.getuid(), "target_pgid": os.getpgrp(), "target_session_id": os.getsid(0),
                 "cgroup_membership": Path("/proc/self/cgroup").read_text().strip(),
                 "parent_death_signal": "SIGKILL",
                 "terminal_exit_reap_grace_seconds": EXIT_REAP_GRACE_SECONDS,
                 "terminal_exit_reap_policy": "ONLY_CONFIRMED_OWNED_Z_X_OR_GONE_NO_LIVE_RSS_ZERO_SUBSTITUTION",
                 "proof_is_immutable_launch_contract_not_live_monitor_status": True}
        proof.update({key: limits[key] for key in (
            "actual_cpu_guard_enabled", "actual_cpu_guard_source", "actual_cpu_clock_ticks_per_second",
            "actual_cpu_cleanup_reserve_seconds", "actual_cpu_stop_threshold_seconds",
            "actual_cpu_closed_batch_cost_seconds", "actual_cpu_batch_cap_seconds", "cpu_guard_kill_grace_seconds")})
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
           "prior_V3_exclusive_lock_held": True,
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
    cpu_terminated = False
    peak_actual_cpu = 0.0
    exit_reap_grace = {}
    expected_identity = None
    child_pidfd = None
    peak_rss, peak_cgroup, samples = 0, cgroup_start["current_bytes"], 0
    usage = None
    child = None
    hard_kill_cleanup_started = None
    def bound_signal(signum):
        if expected_identity is None:
            return False
        event = signal_bound_group(expected_identity, signum)
        row.setdefault("identity_checked_signal_attempts", []).append({"utc": utc().isoformat(), "signal": signum, **event})
        return event["sent"]

    def contain_owned_child():
        nonlocal child_pidfd
        if child is None or child.returncode is not None:
            return False
        try:
            if child_pidfd is None:
                child_pidfd = os.pidfd_open(child.pid)
            signal.pidfd_send_signal(child_pidfd, signal.SIGKILL)
            row["owned_child_pidfd_containment_requested"] = True
            return True
        except ProcessLookupError:
            return False
        except OSError as exc:
            # Before wait4 reaps this original Popen child, its PID cannot be
            # reassigned. This restricted fallback never signals a group or a
            # discovered/unrelated PID, and records the pidfd failure explicitly.
            row["owned_child_pidfd_containment_error"] = repr(exc)
            try:
                os.kill(child.pid, signal.SIGKILL)
                row["owned_unreaped_child_direct_PID_containment_requested"] = True
                return True
            except ProcessLookupError:
                return False
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
            proof = read(proof_path)
            expected_identity = {"pid": child.pid, "start_ticks": proof["child_start_ticks"],
                "uid": proof["target_uid"], "pgrp": proof["target_pgid"], "session": proof["target_session_id"]}
            try:
                child_pidfd = os.pidfd_open(child.pid)
            except ProcessLookupError:
                pass  # A fast completed ordinary payload is reaped by wait4.
            save(args.ledger, ledger)
            while True:
                pid, status, usage = os.wait4(child.pid, os.WNOHANG)
                if pid:
                    child.returncode = os.waitstatus_to_exitcode(status)
                    break
                elapsed = time.monotonic() - start
                terminal_pending = False
                remaining_reap_grace = None
                try:
                    snapshot = process_group_snapshot(child.pid)
                    cgroup = read_cgroup_memory()
                    samples += 1
                    peak_cgroup = max(peak_cgroup, cgroup["current_bytes"])
                    if cgroup["max_bytes"] != limits["expected_cgroup_memory_max_bytes"]:
                        raise RuntimeError("Runtime cgroup envelope changed")
                    leader_rows = [row for row in snapshot["processes"] if row["pid"] == child.pid]
                    if child.pid not in snapshot["pids"] or (leader_rows and leader_rows[0].get("state") in ("Z","X","x")):
                        transition = terminal_reap_observation(snapshot, expected_identity, exit_reap_grace,
                                                                time.monotonic())
                        row["exit_reap_transition"] = exit_reap_grace
                        if transition["state"] == "REAPED":
                            _pid, observed_status, usage = transition["wait4"]
                            child.returncode = os.waitstatus_to_exitcode(observed_status)
                            if cgroup["current_bytes"] >= limits["cgroup_kill_bytes"]:
                                memory_terminated = True
                                row.update(resource_stop_reason="CGROUP_MEMORY_VIOLATION_OBSERVED_AT_REAP",
                                           memory_stop_cgroup=cgroup, memory_stop_snapshot=snapshot)
                            break
                        terminal_pending = True
                        remaining_reap_grace = transition["remaining_grace_seconds"]
                        if transition["terminal_cpu_ticks"] is not None:
                            peak_actual_cpu = max(peak_actual_cpu, transition["terminal_cpu_ticks"] / limits["actual_cpu_clock_ticks_per_second"])
                        cpu_observation = {"observation": "TERMINAL_REAP_PENDING_NO_ZERO_RSS_ASSUMPTION",
                                           "cumulative_phase_cpu_seconds_sampled": peak_actual_cpu,
                                           "clock_source": ACTUAL_CPU_GUARD_SOURCE}
                        row["last_CPU_watermark_retained_during_exit_reap_seconds"] = peak_actual_cpu
                        row["last_exit_reap_cgroup_observation"] = cgroup
                        save(args.ledger, ledger)
                    else:
                        if exit_reap_grace.get("started_monotonic") is not None:
                            raise RuntimeError("Owned terminal leader unexpectedly returned to live state")
                        peak_rss = max(peak_rss, snapshot["rss_bytes"])
                        cpu_observation = actual_cpu_observation(snapshot, expected_identity,
                            limits["actual_cpu_clock_ticks_per_second"], limits["actual_cpu_closed_batch_cost_seconds"],
                            limits["actual_cpu_batch_cap_seconds"], limits["granted_cpu_seconds"],
                            limits["actual_cpu_cleanup_reserve_seconds"])
                        peak_actual_cpu = max(peak_actual_cpu, cpu_observation["cumulative_phase_cpu_seconds_sampled"])
                    cpu_observation["cumulative_phase_cpu_high_water_seconds"] = peak_actual_cpu
                    cpu_stop = peak_actual_cpu >= limits["actual_cpu_stop_threshold_seconds"]
                    if cgroup["current_bytes"] >= limits["cgroup_kill_bytes"] and not hard_killed:
                        hard_killed = bound_signal(signal.SIGKILL) or contain_owned_child()
                        memory_terminated = True
                        row.update(resource_stop_reason="CGROUP_MEMORY_IMMEDIATE_KILL",
                                   memory_stop_snapshot=snapshot, memory_stop_cgroup=cgroup,
                                   SIGKILL_requested_at_utc=utc().isoformat())
                        save(args.ledger, ledger)
                    elif not terminal_pending and snapshot["rss_bytes"] > limits["rss_stop_bytes"] and not memory_terminated:
                        memory_terminated = bound_signal(signal.SIGTERM)
                        # Leave one polling interval so the nominal KILL is within
                        # the authorized1s grace; OS scheduling can still delay it.
                        cleanup_deadline = min(cleanup_deadline, elapsed + max(0, limits["memory_kill_grace_seconds"] - limits["monitor_interval_seconds"]))
                        row.update(resource_stop_reason="PROCESS_GROUP_RSS_GUARD",
                                   memory_stop_snapshot=snapshot, memory_stop_cgroup=cgroup,
                                   SIGTERM_requested_at_utc=utc().isoformat())
                        save(args.ledger, ledger)
                    elif cpu_stop and not cpu_terminated and not memory_terminated and not hard_killed:
                        cpu_terminated = True
                        sent = bound_signal(signal.SIGTERM)
                        if not sent:
                            hard_killed = contain_owned_child()
                        # Set once; later polls cannot extend a resource cleanup.
                        cleanup_deadline = min(cleanup_deadline, elapsed + max(0,
                            limits["cpu_guard_kill_grace_seconds"] - 2*limits["monitor_interval_seconds"]))
                        row.update(resource_stop_reason="ACTUAL_CPU_GUARD",
                            actual_cpu_stop_observation=cpu_observation,
                            SIGTERM_requested_at_utc=utc().isoformat(), actual_cpu_guard_TERM_sent=sent)
                        row.setdefault("resource_stop_events", []).append({"utc": utc().isoformat(),
                            "reason": "ACTUAL_CPU_GUARD", "observation": cpu_observation,
                            "TERM_sent": sent, "KILL_after_elapsed_seconds": cleanup_deadline})
                        save(args.ledger, ledger)
                except Exception as exc:
                    monitor_failed = True
                    hard_killed = bound_signal(signal.SIGKILL) or contain_owned_child() or hard_killed
                    row.update(resource_stop_reason="MONITOR_FAILED_CLOSED", monitor_exception=repr(exc),
                               SIGKILL_requested_at_utc=utc().isoformat())
                    save(args.ledger, ledger)
                    # A real observation failure or grace expiry is fail-closed;
                    # leave for bounded cleanup rather than an unbounded retry.
                    break
                if terminal_pending and (elapsed >= limits["wall_limit_seconds"] or utc() >= datetime.fromisoformat(ledger["run_deadline_at_utc"])):
                    wall_terminated = True
                    hard_killed = contain_owned_child() or hard_killed
                    row.update(resource_stop_reason="WALL_DEADLINE_DURING_EXIT_REAP", SIGKILL_requested_at_utc=utc().isoformat())
                    save(args.ledger, ledger)
                if not cpu_terminated and not memory_terminated and not hard_killed and not wall_terminated and (elapsed >= limits["wall_SIGTERM_after_seconds"] or stop_request["signal"] is not None):
                    if bound_signal(signal.SIGTERM):
                        wall_terminated = True
                        cleanup_deadline = min(limits["wall_limit_seconds"], elapsed + 60)
                        row.update(SIGTERM_requested_at_utc=utc().isoformat(), supervisor_signal=stop_request["signal"],
                                   resource_stop_reason="SUPERVISOR_INTERRUPTED" if stop_request["signal"] else "WALL_CLEANUP_WINDOW")
                        save(args.ledger, ledger)
                if elapsed >= cleanup_deadline and not hard_killed:
                    hard_killed = bound_signal(signal.SIGKILL) or contain_owned_child()
                    row["SIGKILL_requested_at_utc"] = utc().isoformat()
                if hard_killed:
                    if hard_kill_cleanup_started is None:
                        hard_kill_cleanup_started = time.monotonic()
                    elif time.monotonic()-hard_kill_cleanup_started >= EXIT_REAP_GRACE_SECONDS:
                        row["owned_child_post_KILL_reap_poll_limit_seconds"] = EXIT_REAP_GRACE_SECONDS
                        break
                time.sleep(min(limits["monitor_interval_seconds"], remaining_reap_grace)
                           if remaining_reap_grace is not None else limits["monitor_interval_seconds"])
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
            hard_killed = bound_signal(signal.SIGKILL) or contain_owned_child() or hard_killed
            reaped = bounded_owned_reap(child.pid)
            if reaped is None:
                monitor_failed = True
                row.update(status="UNRESOLVED_WAIT4_REAP_FAILURE", ended_at_utc=utc().isoformat(),
                    actual_cpu_seconds=None, actual_CPU_is_unknown_do_not_resume_budget=True,
                    final_actual_CPU_lower_bound_seconds=peak_actual_cpu,
                    resource_stop_reason="BOUNDED_OWNED_CHILD_REAP_FAILED", monitor_failed=True,
                    hard_killed=hard_killed, log_sha256=sha(args.log))
                save(args.ledger, ledger)
            else:
                _pid, status, usage = reaped
                child.returncode = os.waitstatus_to_exitcode(status)
        if child_pidfd is not None:
            os.close(child_pidfd)
    if child.returncode is None or usage is None:
        raise RuntimeError("Unresolved owned child wait4/CPU; ledger blocked for explicit forensic review")
    cpu = usage.ru_utime + usage.ru_stime
    try:
        residual = process_group_snapshot(child.pid)
    except Exception as exc:
        monitor_failed = True
        # Payload exited: do not group-signal after its leader identity ended.
        residual = {"pids": []}
        row["post_exit_process_audit_failure"] = repr(exc)
    if residual["pids"]:
        row["residual_processes_require_review_after_main_payload_exit"] = residual["pids"]
    row.update(status="EXITED" if child.returncode == 0 and not any((wall_terminated, memory_terminated, cpu_terminated, monitor_failed, hard_killed, bool(residual["pids"]))) else "PARTIAL_OR_FAILED",
               ended_at_utc=utc().isoformat(), actual_cpu_seconds=cpu,
               elapsed_wall_seconds=time.monotonic() - start, peak_rss_kib=usage.ru_maxrss,
               exit_code=child.returncode, wall_terminated=wall_terminated, hard_killed=hard_killed,
               memory_terminated=memory_terminated, monitor_failed=monitor_failed,
               cpu_terminated=cpu_terminated, peak_actual_cpu_seconds_sampled=peak_actual_cpu,
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
    p.add_argument("--migrate-from-closed-v4", type=Path)
    p.add_argument("--migrate-from-sha256")
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
        global_lock = ROOT / ".runtime/pyscf-v4-global.lock"
        global_lock.parent.mkdir(parents=True, exist_ok=True)
        with global_lock.open("a") as global_handle:
            fcntl.flock(global_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.initialize:
                initialize(args)
                print(json.dumps({"status": "NEW_AUTHORIZED_LEDGER_INITIALIZED", "ledger": str(args.ledger)}))
                return
            if args.migrate_from_closed_v4 is not None:
                ledger = migrate_from_closed_v4(args)
                print(json.dumps({"status": "EXISTING_V4_POOL_MIGRATED_NO_NEW_BUDGET", "ledger": str(args.ledger),
                                  "consumed_cpu_seconds": ledger["consumed_cpu_seconds"],
                                  "run_deadline_at_utc": ledger["run_deadline_at_utc"]}))
                return
            row = run_phase(args, read(args.ledger))
            print(json.dumps(row), flush=True)
            raise SystemExit(0 if row["status"] == "EXITED" else 1)


if __name__ == "__main__":
    main()
