"""NOT_RUN contingency: explicit V6b continuation of an exact closed V5 pool.

The earlier pilot ledger is immutable. This runner binds the separate continuation
authorization, source checkpoint and source revision, and never resets a ledger.
Registration carries every old phase and actual cost. Only root-authorized
additional CPU/deadline increments are accepted; the original V5 stays immutable.
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
SCHEMA = "research4-pyscf-continuation-budget-v6b"
MAX_CPU, MAX_WALL = 216000, 151200
MAX_ADDITIONAL_CPU = 86400
MAX_DEADLINE_EXTENSION = 86400
MAX_PHASE_CPU, MAX_PHASE_WALL = 122400, 43200
RSS_STOP_BYTES = 10 * 1024**3
CGROUP_KILL_BYTES = 12 * 1024**3
MONITOR_INTERVAL_SECONDS = .1
MEMORY_KILL_GRACE_SECONDS = 1.0
CGROUP_MEMORY_MAX_BYTES = 16 * 1024**3
GUARD_SEMANTICS = "BEST_EFFORT_RSS_GUARD_NOT_HARD_RSS_LIMIT"
PROOF_SCHEMA = "research4-pyscf-continuation-supervisor-proof-v6b"
ACTUAL_CPU_RESERVE_SECONDS = 120
ACTUAL_CPU_GUARD_SOURCE = "PROC_STAT_CUMULATIVE_FROM_PROCESS_BIRTH"
CPU_GUARD_KILL_GRACE_SECONDS = 1.0
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
EXIT_REAP_GRACE_SECONDS = .5
V5_SOURCE = ROOT / "scripts/run_pyscf_actual_cpu_budget_20261104_v5.py"
V5_SHA = "33f1dc3db5280f16c9898c561e345143614730dc052b171b4f11fbbb079cdeee"
AUTH_SCHEMA = "research4-v6b-continuation-authorization-v1"
CONTINUATION_REGISTRY = ROOT / ".runtime/pyscf-v6-continuation-registry"


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
    """Verify the real V6b parent, cumulative allocation and unchanged V5 source."""
    supplied = proof_path or os.environ.get("R4_RESOURCE_SUPERVISOR_PROOF")
    if not supplied:
        raise ValueError("Actual-CPU V6b supervision proof required before quantum work")
    path = Path(supplied).resolve()
    raw = path.read_bytes()
    if len(raw) > 65536:
        raise ValueError("Unexpectedly large supervision proof")
    proof = json.loads(raw)
    if proof.get("schema") != PROOF_SCHEMA or proof.get("guard_semantics") != GUARD_SEMANTICS:
        raise ValueError("A real independent V6b actual-CPU proof is required")
    if proof.get("terminal_exit_reap_grace_seconds") != EXIT_REAP_GRACE_SECONDS:
        raise ValueError("V6b requires its explicit bounded terminal reaping policy")
    observed = resource_runtime_snapshot()
    for key in ("supervisor_pid", "supervisor_start_ticks", "child_pid", "child_start_ticks"):
        if type(proof.get(key)) is not int or proof[key] != observed[key]:
            raise ValueError("Live supervisor/child identity mismatch: " + key)
    source = ROOT/"scripts/run_pyscf_continuation_budget_20261105_v6b.py"
    if root_path(proof.get("supervisor_script_path", "")) != source or proof.get("supervisor_script_sha256") != sha(source):
        raise ValueError("Actual V6b supervisor source path/SHA differs")
    commands = [Path(os.fsdecode(token)) for token in observed["parent_command"] if token]
    if not any((token if token.is_absolute() else Path(observed["parent_cwd"])/token).resolve() == source for token in commands):
        raise ValueError("Actual parent command is not the V6b supervisor")
    expected_values = {"address_space_soft_limit_bytes": 24*1024**3, "address_space_hard_limit_bytes": 24*1024**3,
        "rss_stop_bytes": 10*1024**3, "cgroup_kill_bytes": 12*1024**3,
        "expected_cgroup_memory_max_bytes": CGROUP_MEMORY_MAX_BYTES}
    if any(type(proof.get(key)) is not int or proof[key] != value for key,value in expected_values.items()):
        raise ValueError("Unreviewed V6b resource policy")
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
        raise ValueError("Finite unchanged V6b phase wall bound required")
    ledger = read(root_path(proof.get("ledger_path", "")))
    validate_ledger(ledger, allow_running=True)
    if root_path(proof["ledger_path"]) != root_path(ledger["continuation_ledger_path"]):
        raise ValueError("Actual proof must use the sole registered V6b ledger path")
    phases = [row for row in ledger.get("phases",[]) if row.get("phase") == proof.get("phase")]
    if (ledger.get("schema") != SCHEMA or len(phases) != 1 or phases[0].get("status") != "RUNNING"
            or sum(row.get("status") == "RUNNING" for row in ledger["phases"]) != 1
            or ledger.get("runner_sha256") != sha(source)):
        raise ValueError("Actual new V6b ledger must contain one matching live phase")
    if ledger.get("quantum_cpu_cap_seconds") != cap or ledger.get("consumed_cpu_seconds") != closed or ledger.get("prior_budget_is_historical_only") is not True:
        raise ValueError("Live cumulative V6b pool differs from proof; closed V5 costs must be carried")
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
        "startup_proof_is_not_future_monitor_health_confirmation": True,
        "closed_V6b_lineage_and_explicit_increment_verified": True}


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


def frozen_v5_module():
    if sha(V5_SOURCE) != V5_SHA:
        raise ValueError("Frozen V5 guard source changed")
    spec = importlib.util.spec_from_file_location("research4_closed_v5_continuation_source", V5_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def aware_time(value, field):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Timezone-aware recorded timestamp required: " + field)
    return result


def closed_costs(phases):
    if not isinstance(phases, list):
        raise ValueError("Complete phase list required")
    values = []
    names = set()
    for row in phases:
        if row.get("status") == "RUNNING" or not row.get("ended_at_utc"):
            raise ValueError("Source phase is unresolved; wait for actual closure")
        name = row.get("phase")
        if not name or name in names:
            raise ValueError("Phase identities must be unique")
        names.add(name)
        cost = row.get("actual_cpu_seconds")
        if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
            raise ValueError("Unknown or invalid actual phase CPU; no continuation/reset permitted")
        if row.get("residual_processes_require_review_after_main_payload_exit"):
            raise ValueError("Residual process ownership unresolved")
        values.append(cost)
    return sum(values)


def validate_source_v5(path, expected_sha):
    supplied_path = Path(path)
    if supplied_path.is_symlink():
        raise ValueError("Symlink V5 ledger is not an immutable continuation source")
    path = root_path(path)
    if not expected_sha or not path.is_file() or path.is_symlink() or sha(path) != expected_sha:
        raise ValueError("Exact immutable closed-V5 ledger SHA required")
    source = read(path)
    if source.get("schema") != "research4-pyscf-actual-cpu-budget-v5" or source.get("runner_sha256") != V5_SHA:
        raise ValueError("Only the actual reviewed V5 ledger can be continued")
    cost = closed_costs(source.get("phases"))
    if (type(source.get("consumed_cpu_seconds")) not in (int, float)
            or not math.isclose(cost, source["consumed_cpu_seconds"], rel_tol=1e-12, abs_tol=1e-7)):
        raise ValueError("ALL closed V5 phase costs, including failures, must equal its cumulative cost")
    old_auth = frozen_v5_module().validate_ledger(source)
    return source, old_auth


def validate_authorization(path, expected_sha, source_path=None, source_sha=None, source_base=None):
    path = root_path(path)
    if not expected_sha or sha(path) != expected_sha:
        raise ValueError("Explicit V6b authorization SHA required")
    value = read(path)
    if (value.get("schema") != AUTH_SCHEMA or value.get("root_approved_for_execution") is not True
            or value.get("no_budget_reset") is not True or value.get("no_automatic_budget_growth") is not True
            or value.get("v6b_supervisor_source_sha256") != sha(__file__)
            or not value.get("authorization_text") or not value.get("research_authorization_evidence")):
        raise ValueError("An explicit root decision with user authorization evidence is required; no allocation is enabled by default")
    if not isinstance(value.get("continuation_ledger_path"), str) or not value["continuation_ledger_path"]:
        raise ValueError("Authorization must bind the sole V6b continuation ledger path")
    if value["continuation_ledger_path"] != str(root_path(value["continuation_ledger_path"])):
        raise ValueError("Authorized continuation ledger path must be canonical and absolute")
    source_path = root_path(source_path or value.get("source_v5_ledger_path", ""))
    source_sha = source_sha or value.get("source_v5_ledger_sha256")
    if (root_path(value.get("source_v5_ledger_path", "")) != source_path
            or value.get("source_v5_ledger_sha256") != source_sha
            or value.get("source_v5_runner_sha256") != V5_SHA):
        raise ValueError("Authorization/source-V5 binding differs")
    source, old_auth = validate_source_v5(source_path, source_sha)
    expected = {"source_v5_consumed_cpu_seconds": source["consumed_cpu_seconds"],
                "original_v5_created_at_utc": source["created_at_utc"],
                "original_v5_run_deadline_at_utc": source["run_deadline_at_utc"],
                "original_v5_quantum_cpu_cap_seconds": source["quantum_cpu_cap_seconds"],
                "original_v5_run_wall_cap_seconds": source["run_wall_cap_seconds"],
                "original_v5_authorization_path": source["authorization_path"],
                "original_v5_authorization_sha256": source["authorization_sha256"]}
    if any(value.get(key) != actual for key, actual in expected.items()):
        raise ValueError("Original V5 history/cost/cap/deadline/authorization was not preserved exactly")
    revision = source_base or value.get("source_base_commit")
    if (not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)
            or value.get("source_base_commit") != revision):
        raise ValueError("Explicit full reviewed source-base commit required")
    increment, extension = value.get("additional_cpu_seconds"), value.get("deadline_extension_seconds")
    if type(increment) is not int or not 0 <= increment <= MAX_ADDITIONAL_CPU:
        raise ValueError("Explicit additional CPU must be0..86400seconds")
    if type(extension) is not int or not 0 <= extension <= MAX_DEADLINE_EXTENSION:
        raise ValueError("Explicit deadline extension must be0..86400seconds")
    research_end = aware_time(value.get("research_authorization_deadline", ""), "research_authorization_deadline")
    auth_created = aware_time(value.get("created_at_utc", ""), "created_at_utc")
    original_created = aware_time(source["created_at_utc"], "source V5 created")
    original_deadline = aware_time(source["run_deadline_at_utc"], "source V5 deadline")
    effective = original_deadline + timedelta(seconds=extension)
    if not original_created <= auth_created <= research_end or effective > research_end:
        raise ValueError("Continuation exceeds the explicitly supplied research authorization deadline")
    if auth_created > utc():
        raise ValueError("Root decision timestamp cannot be invented in the future")
    # Never derive a fresh72h window from register time. The root-provided
    # conservative endpoint and verbatim/evidence record are immutable inputs.
    record = value.get("research_authorization_record")
    if record is not None:
        record_path = root_path(record["path"])
        if sha(record_path) != record["sha256"]:
            raise ValueError("Research authorization evidence bytes changed")
    if not 0 < source["quantum_cpu_cap_seconds"] <= 129600 or not 0 < source["run_wall_cap_seconds"] <= 64800:
        raise ValueError("Source V5 caps differ from the reviewed bounded pool")
    cap = source["quantum_cpu_cap_seconds"] + increment
    wall_cap = source["run_wall_cap_seconds"] + extension
    if cap > MAX_CPU or wall_cap > MAX_WALL:
        raise ValueError("Explicit V6b cumulative resource envelope exceeded")
    if value.get("max_quantum_cpu_seconds") != cap or value.get("max_run_wall_seconds") != wall_cap:
        raise ValueError("Cumulative caps must equal original caps plus explicit increments")
    for key, maximum in (("max_phase_cpu_seconds", MAX_PHASE_CPU), ("max_phase_wall_seconds", MAX_PHASE_WALL),
                         ("max_threads", 4), ("max_memory_gib", 24)):
        if type(value.get(key)) is not int or not 0 < value[key] <= maximum:
            raise ValueError("Invalid explicit V6b phase resource limit: " + key)
    if value["max_memory_gib"] != 24:
        raise ValueError("Keep the reviewed24GiB virtual-address envelope")
    for key in ("guard_semantics", "rss_stop_bytes", "cgroup_kill_bytes", "monitor_interval_seconds",
                "memory_kill_grace_seconds", "expected_cgroup_memory_max_bytes", "actual_cpu_cleanup_reserve_seconds",
                "cpu_guard_kill_grace_seconds", "actual_cpu_guard_source"):
        if value.get(key) != old_auth.get(key):
            raise ValueError("V6b must preserve reviewed V5 runtime guard semantics: " + key)
    for key in ("prior_closed_ledger_path", "prior_closed_ledger_sha256", "prior_consumed_cpu_seconds", "prior_budget_is_historical_only"):
        if value.get(key) != old_auth.get(key):
            raise ValueError("Prior V3 provenance bank changed: " + key)
    validate_prior_closed_ledger(value)
    return value


@contextmanager
def lineage_locks(authorization):
    """Same global lock is held by main; also bind V5/V4/V3 ledger locks."""
    from contextlib import ExitStack
    source_path = root_path(authorization["source_v5_ledger_path"])
    source = read(source_path)
    paths = [source_path.with_suffix(".lock"),
             root_path(source["migration"]["source_v4_path"]).with_suffix(".lock"),
             root_path(authorization["prior_closed_ledger_path"]).with_suffix(".lock")]
    with ExitStack() as stack:
        for path in dict.fromkeys(paths):
            if path.is_symlink():
                raise ValueError("Symlink lineage lock rejected")
            handle = stack.enter_context(path.open("a"))
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def save_new_ledger(path, ledger):
    """Atomically publish a new ledger without replacing any existing file."""
    path = Path(path)
    temporary = path.with_name(path.name+".registering."+str(os.getpid()))
    try:
        with temporary.open("x") as handle:
            handle.write(json.dumps(ledger, indent=2, allow_nan=False)+"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def continuation_claim(source, source_path, source_sha, auth_path, auth_sha, ledger_path):
    """One allocation per original budget, even across later/annotated copies."""
    canonical = json.dumps(source, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    lineage = {"schema": "research4-original-v5-budget-lineage-v1",
        "original_created_at_utc": source["created_at_utc"],
        "original_v5_authorization_sha256": source["authorization_sha256"],
        "source_checkpoint_sha256": source["source_checkpoint_sha256"],
        "source_v4_sha256": source["migration"]["source_v4_sha256"]}
    key = hashlib.sha256(json.dumps(lineage, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = CONTINUATION_REGISTRY / (key + ".json")
    value = {"schema": "research4-single-v5-continuation-claim-v1", "budget_lineage": lineage,
        "budget_lineage_sha256": key, "canonical_v5_payload_sha256": hashlib.sha256(canonical).hexdigest(),
        "source_v5_path": str(root_path(source_path)), "source_v5_sha256": source_sha,
        "continuation_ledger_path": str(root_path(ledger_path)),
        "authorization_path": str(root_path(auth_path)), "authorization_sha256": auth_sha,
        "v6b_supervisor_source_sha256": sha(__file__), "single_allocation_no_re_registration": True}
    return path, value


def verify_continuation_claim(ledger, source, authorization):
    path, expected = continuation_claim(source, authorization["source_v5_ledger_path"],
        authorization["source_v5_ledger_sha256"], ledger["authorization_path"], ledger["authorization_sha256"],
        authorization["continuation_ledger_path"])
    recorded = ledger.get("continuation_registry_claim", {})
    if (recorded.get("path") != str(path) or path.is_symlink() or not path.is_file()
            or sha(path) != recorded.get("sha256") or read(path) != expected):
        raise ValueError("Sole immutable continuation lineage claim is absent or differs")
    if ledger.get("continuation_ledger_path") != authorization["continuation_ledger_path"]:
        raise ValueError("Registered V6b ledger path differs from the sole authorized allocation")


def register_continuation(args):
    if args.ledger.exists():
        raise FileExistsError("Never overwrite/reset an existing continuation ledger")
    if not all((args.authorization, args.authorization_sha256, args.source_v5_ledger,
                args.source_v5_sha256, args.source_base_commit)):
        raise ValueError("Register requires explicit auth, closed V5 and full source bindings")
    auth = validate_authorization(args.authorization, args.authorization_sha256,
        args.source_v5_ledger, args.source_v5_sha256, args.source_base_commit)
    if args.ledger.resolve() != root_path(auth["continuation_ledger_path"]):
        raise ValueError("Register target differs from the explicitly authorized sole ledger path")
    source_path = root_path(args.source_v5_ledger)
    if args.ledger.resolve() == source_path or args.ledger.resolve().is_relative_to(OLD_PILOT):
        raise ValueError("New V6b ledger must not replace an immutable historical ledger")
    with lineage_locks(auth):
        auth = validate_authorization(args.authorization, args.authorization_sha256,
            source_path, args.source_v5_sha256, args.source_base_commit)
        source = read(source_path)
        effective_deadline = aware_time(source["run_deadline_at_utc"], "source deadline") + timedelta(seconds=auth["deadline_extension_seconds"])
        if effective_deadline <= utc():
            raise ValueError("Explicit continuation deadline is already expired")
        ledger = json.loads(json.dumps(source))
        old_count = len(source["phases"])
        ledger.update(schema=SCHEMA, runner_sha256=sha(__file__),
            authorization_path=str(root_path(args.authorization)), authorization_sha256=args.authorization_sha256,
            continuation_ledger_path=str(args.ledger.resolve()),
            source_base_commit=args.source_base_commit,
            original_v5_created_at_utc=source["created_at_utc"],
            original_v5_run_deadline_at_utc=source["run_deadline_at_utc"],
            continuation_registered_at_utc=utc().isoformat(),
            quantum_cpu_cap_seconds=source["quantum_cpu_cap_seconds"]+auth["additional_cpu_seconds"],
            run_wall_cap_seconds=source["run_wall_cap_seconds"]+auth["deadline_extension_seconds"],
            run_deadline_at_utc=effective_deadline.isoformat(),
            effective_run_deadline_at_utc=effective_deadline.isoformat(),
            continuation_source={"schema": "research4-exact-closed-v5-lineage-v1", "source_v5_path": str(source_path),
                "source_v5_sha256": args.source_v5_sha256, "source_v5_runner_sha256": V5_SHA,
                "source_v5_authorization_path": source["authorization_path"], "source_v5_authorization_sha256": source["authorization_sha256"],
                "source_v5_source_base_commit": source["source_base_commit"], "original_phase_count": old_count,
                "original_phase_array_sha256": hashlib.sha256(json.dumps(source["phases"],sort_keys=True).encode()).hexdigest(),
                "source_consumed_cpu_seconds": source["consumed_cpu_seconds"], "original_created_at_utc": source["created_at_utc"],
                "original_cap_seconds": source["quantum_cpu_cap_seconds"], "original_run_wall_cap_seconds": source["run_wall_cap_seconds"],
                "original_deadline_at_utc": source["run_deadline_at_utc"], "failed_costs_carried": True,
                "prior_proof_schemas_not_relabelled": True},
            continuation_allocation={"additional_cpu_seconds": auth["additional_cpu_seconds"],
                "deadline_extension_seconds": auth["deadline_extension_seconds"],
                "research_authorization_deadline": auth["research_authorization_deadline"],
                "research_authorization_evidence": auth["research_authorization_evidence"],
                "authorization_sha256": args.authorization_sha256, "new_72h_clock_started": False})
        # Keep the exact actual cumulative cost, including every failed source
        # phase. No source phase/proof is relabelled as a V6b calculation.
        ledger["consumed_cpu_seconds"] = source["consumed_cpu_seconds"]
        ledger["remaining_quantum_cpu_seconds"] = max(0., ledger["quantum_cpu_cap_seconds"]-ledger["consumed_cpu_seconds"])
        claim_path, claim = continuation_claim(source, source_path, args.source_v5_sha256,
            args.authorization, args.authorization_sha256, args.ledger)
        if CONTINUATION_REGISTRY.is_symlink():
            raise ValueError("Continuation registry cannot be a symlink")
        CONTINUATION_REGISTRY.mkdir(parents=True, exist_ok=True)
        # Publish the no-replace claim FIRST. A crash afterwards blocks any
        # fresh allocation, rather than allowing a second pool or silent reset.
        # Recovery requires a separate explicit review, never automatic reuse.
        save_new_ledger(claim_path, claim)
        ledger["continuation_registry_claim"] = {"path": str(claim_path), "sha256": sha(claim_path)}
        validate_ledger(ledger)
        save_new_ledger(args.ledger, ledger)
        return ledger


def validate_ledger(ledger, *, allow_running=False):
    if ledger.get("schema") != SCHEMA or ledger.get("runner_sha256") != sha(__file__):
        raise ValueError("Not the explicitly registered frozen V6b continuation")
    auth = validate_authorization(ledger["authorization_path"], ledger["authorization_sha256"],
        source_base=ledger["source_base_commit"])
    source_path = root_path(auth["source_v5_ledger_path"])
    source = read(source_path)
    verify_continuation_claim(ledger, source, auth)
    lineage = ledger.get("continuation_source", {})
    count = len(source["phases"])
    expected_lineage = {"source_v5_path": str(source_path), "source_v5_sha256": auth["source_v5_ledger_sha256"],
        "schema": "research4-exact-closed-v5-lineage-v1",
        "source_v5_runner_sha256": V5_SHA, "source_consumed_cpu_seconds": source["consumed_cpu_seconds"],
        "source_v5_authorization_path": source["authorization_path"],
        "source_v5_authorization_sha256": source["authorization_sha256"],
        "source_v5_source_base_commit": source["source_base_commit"],
        "original_phase_count": count, "original_created_at_utc": source["created_at_utc"],
        "original_cap_seconds": source["quantum_cpu_cap_seconds"], "original_deadline_at_utc": source["run_deadline_at_utc"],
        "original_run_wall_cap_seconds": source["run_wall_cap_seconds"],
        "original_phase_array_sha256": hashlib.sha256(json.dumps(source["phases"],sort_keys=True).encode()).hexdigest(),
        "failed_costs_carried": True, "prior_proof_schemas_not_relabelled": True}
    if any(lineage.get(k) != v for k,v in expected_lineage.items()) or ledger.get("phases", [])[:count] != source["phases"]:
        raise ValueError("Original full V5 phase prefix or lineage was modified/omitted")
    if (ledger.get("created_at_utc") != source["created_at_utc"]
            or ledger.get("original_v5_created_at_utc") != source["created_at_utc"]
            or ledger.get("original_v5_run_deadline_at_utc") != source["run_deadline_at_utc"]):
        raise ValueError("Original creation/deadline history was reset")
    effective = aware_time(source["run_deadline_at_utc"], "source deadline") + timedelta(seconds=auth["deadline_extension_seconds"])
    if (ledger.get("quantum_cpu_cap_seconds") != source["quantum_cpu_cap_seconds"]+auth["additional_cpu_seconds"]
            or ledger.get("run_wall_cap_seconds") != source["run_wall_cap_seconds"]+auth["deadline_extension_seconds"]
            or ledger.get("run_deadline_at_utc") != effective.isoformat()
            or ledger.get("effective_run_deadline_at_utc") != effective.isoformat()):
        raise ValueError("Continuation caps/deadline differ from the explicit increments")
    allocation = ledger.get("continuation_allocation", {})
    if allocation.get("authorization_sha256") != ledger["authorization_sha256"]:
        raise ValueError("Explicit allocation authorization SHA changed")
    for name in ("additional_cpu_seconds", "deadline_extension_seconds", "research_authorization_deadline", "research_authorization_evidence"):
        if allocation.get(name) != auth[name]:
            raise ValueError("Continuation allocation differs from root decision: " + name)
    if allocation.get("new_72h_clock_started") is not False:
        raise ValueError("Never restart the user's research authorization clock")
    for key in ("source_checkpoint_path", "source_checkpoint_sha256", "prior_closed_ledger", "prior_consumed_cpu_seconds", "prior_budget_is_historical_only", "migration"):
        if ledger.get(key) != source.get(key):
            raise ValueError("Historical source/checkpoint/prior-bank provenance changed: " + key)
    phases = ledger.get("phases", [])
    running = [r for r in phases if r.get("status") == "RUNNING"]
    if running and (not allow_running or len(running) != 1):
        raise RuntimeError("A quantum phase remains unresolved; forbid another launch")
    used = closed_costs([r for r in phases if r.get("status") != "RUNNING"])
    names = [r.get("phase") for r in phases]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate phase identity")
    if not math.isclose(used, ledger.get("consumed_cpu_seconds", float("nan")), rel_tol=1e-12, abs_tol=1e-7):
        raise ValueError("Every actual closed V5+V6b CPU cost must be retained")
    remaining = max(0.,ledger["quantum_cpu_cap_seconds"]-used)
    if not math.isclose(remaining,ledger.get("remaining_quantum_cpu_seconds",float("nan")),rel_tol=1e-12,abs_tol=1e-7):
        raise ValueError("Remaining budget is not cumulative cap minus all actual cost")
    return auth


def healthy_closed_row(row):
    return (row.get("status") == "EXITED" and row.get("exit_code") == 0 and bool(row.get("ended_at_utc"))
        and all(row.get(k) is False for k in ("wall_terminated","memory_terminated","cpu_terminated","monitor_failed","hard_killed"))
        and not row.get("residual_processes_require_review_after_main_payload_exit")
        and type(row.get("actual_cpu_seconds")) in (int,float)
        and math.isfinite(row["actual_cpu_seconds"]) and row["actual_cpu_seconds"] >= 0)


def verify_closed_phase(evidence, *, expected_worker_path, expected_worker_sha256,
                        expected_result_schema, allowed_result_statuses,
                        expected_output=None, expected_snapshot_sha256=None):
    """Stdlib producer-closure gate; no numerical payload is ever loaded here."""
    if not allowed_result_statuses:
        raise ValueError("Caller must explicitly list successful scientific stage statuses")
    worker = root_path(expected_worker_path)
    source = evidence.get("source_worker", {})
    if (root_path(source.get("path", "")) != worker or source.get("sha256") != expected_worker_sha256
            or sha(worker) != expected_worker_sha256):
        raise ValueError("Unexpected frozen V6b stage entry source")
    if expected_snapshot_sha256 and evidence.get("snapshot_manifest_sha256") != expected_snapshot_sha256:
        raise ValueError("Stage source snapshot differs")
    supervision = evidence.get("resource_supervision", {})
    proof_path = root_path(supervision.get("proof_path", ""))
    if sha(proof_path) != supervision.get("proof_sha256"):
        raise ValueError("Stage producer proof bytes changed")
    proof = read(proof_path)
    if (proof != supervision.get("proof") or proof.get("schema") != PROOF_SCHEMA
            or proof.get("supervisor_script_sha256") != sha(__file__)
            or root_path(proof.get("supervisor_script_path", "")) != Path(__file__).resolve()
            or supervision.get("live_parent_process_and_source_verified") is not True):
        raise ValueError("Actual V6b parent proof required, never a relabelled old proof")
    ledger = read(root_path(proof["ledger_path"]))
    validate_ledger(ledger,allow_running=True)
    if root_path(proof["ledger_path"]) != root_path(ledger["continuation_ledger_path"]):
        raise ValueError("Producer proof uses an unregistered ledger alias")
    rows = [r for r in ledger["phases"] if r.get("phase") == proof.get("phase")]
    if len(rows) != 1 or not healthy_closed_row(rows[0]):
        raise ValueError("Source stage parent is not a healthy actual EXITED/0")
    row = rows[0]
    if row.get("supervisor_proof_sha256") != sha(proof_path) or root_path(row["supervisor_proof_path"]) != proof_path:
        raise ValueError("Stage proof and closed ledger row differ")
    log = root_path(row["log"])
    if sha(log) != row.get("log_sha256"):
        raise ValueError("Closed source stage log changed")
    output = root_path(evidence["output_directory"])
    if expected_output and output != root_path(expected_output):
        raise ValueError("Source stage output path differs")
    result_path = output/"result.json"
    result = read(result_path)
    if (result.get("schema") != expected_result_schema or result.get("status") not in allowed_result_statuses
            or result.get("source_worker") != source or result.get("resource_supervision") != supervision
            or result.get("stable_MF_state_unchanged") is not True
            or (expected_snapshot_sha256 and result.get("snapshot_manifest_sha256") != expected_snapshot_sha256)
            or (evidence.get("stage") is not None and result.get("stage") != evidence["stage"])):
        raise ValueError("Source stage lacks a matching qualified result")
    control = result.get("native_direct_driver_control", {})
    if control.get("restoration_failures") != 0 or control.get("original_driver_restored") is not True:
        raise ValueError("Source stage native driver restoration failed")
    command = row.get("command", [])
    if (str(worker) not in [str(root_path(t)) for t in command if isinstance(t,str) and t.endswith('.py')]
            or '--output' not in command or root_path(command[command.index('--output')+1]) != output):
        raise ValueError("Actual producer command does not bind this source/output")
    return {"phase":row["phase"],"closed_row_sha256":hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest(),
        "proof_sha256":sha(proof_path),"producer_result_sha256":sha(result_path),
        "log_sha256":sha(log),"actual_cpu_seconds":row["actual_cpu_seconds"]}

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
    if args.ledger.resolve() != root_path(auth["continuation_ledger_path"]):
        raise ValueError("Run must use the sole registered continuation ledger, never a copied alias")
    with lineage_locks(auth):
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
    sub = p.add_subparsers(dest="action",required=True)
    register = sub.add_parser("register",help="Register an explicitly authorized closed-V5 continuation; never launch Q")
    register.add_argument("--ledger",type=Path,required=True)
    register.add_argument("--source-v5-ledger",type=Path,required=True)
    register.add_argument("--source-v5-sha256",required=True)
    register.add_argument("--authorization",type=Path,required=True)
    register.add_argument("--authorization-sha256",required=True)
    register.add_argument("--source-base-commit",required=True)
    run = sub.add_parser("run",help="Run one bounded payload in an already registered V6b ledger")
    run.add_argument("--ledger",type=Path,required=True)
    run.add_argument("--phase",required=True)
    run.add_argument("--cpu-seconds",type=int,required=True)
    run.add_argument("--wall-seconds",type=int,required=True)
    run.add_argument("--threads",type=int,default=1)
    run.add_argument("--memory-gib",type=int,default=24)
    run.add_argument("--log",type=Path,required=True)
    run.add_argument("command",nargs=argparse.REMAINDER)
    return p


def main():
    args=parser().parse_args()
    args.ledger=args.ledger.resolve()
    if args.ledger.is_relative_to(OLD_PILOT):
        raise ValueError("Never write a continuation inside the immutable pilot")
    args.ledger.parent.mkdir(parents=True,exist_ok=True)
    with args.ledger.with_suffix(".lock").open("a") as local:
        fcntl.flock(local,fcntl.LOCK_EX|fcntl.LOCK_NB)
        shared=ROOT/".runtime/pyscf-v4-global.lock"
        shared.parent.mkdir(parents=True,exist_ok=True)
        with shared.open("a") as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.action=="register":
                ledger=register_continuation(args)
                print(json.dumps({"status":"EXPLICIT_CLOSED_V5_CONTINUATION_REGISTERED_NO_PAYLOAD_LAUNCHED",
                    "ledger":str(args.ledger),"consumed_cpu_seconds":ledger["consumed_cpu_seconds"],
                    "original_v5_deadline":ledger["original_v5_run_deadline_at_utc"],
                    "effective_deadline":ledger["run_deadline_at_utc"]}),flush=True)
                return
            row=run_phase(args,read(args.ledger))
            print(json.dumps(row),flush=True)
            raise SystemExit(0 if row["status"]=="EXITED" else 1)


if __name__=="__main__":
    main()
