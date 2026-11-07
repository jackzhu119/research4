"""Prospective user-authorized 72h allocation; all older ledgers remain immutable.

The frozen V8 monitor is retained with explicit boot/identity/CPU heartbeats and
a disk guard. Old failed costs are carried once; old unused credit is retired.
No process or budget recovery is inferred from missing PIDs or host reboot.
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
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OLD_PILOT = ROOT / "reports/pyscf-spinosin-pilot-20261031"
OLD_GUARD = ROOT / "scripts/run_pyscf_continuation_budget_20261105_v8.py"
OLD_GUARD_SHA = "7bc773076ecc5e0f90fb58b81bf0641948dd09c14b2816138a0469f16e511c65"
OLD_SCHEMA = "research4-pyscf-continuation-budget-v8"
OLD_PROOF_SCHEMA = "research4-pyscf-continuation-supervisor-proof-v8"
SCHEMA = "research4-research72h-cpu-budget-v1"
PROOF_SCHEMA = "research4-research72h-supervisor-proof-v1"
AUTH_SCHEMA = "research4-user-72h-renewal-authorization-v1"
EXPECTED_AUTH_SHA = "058c24d076567b14054f78467c001a7858ef2601a070f3c5b97c8b074bc9981d"
MAX_CPU, MAX_WALL = 1288677.533795, 259200
MAX_ADDITIONAL_CPU = 900000
MAX_PHASE_CPU, MAX_PHASE_WALL = 60000, 28800
RSS_STOP_BYTES = 10 * 1024**3
CGROUP_KILL_BYTES = 12 * 1024**3
MONITOR_INTERVAL_SECONDS = .1
MEMORY_KILL_GRACE_SECONDS = 1.0
CGROUP_MEMORY_MAX_BYTES = 16 * 1024**3
GUARD_SEMANTICS = "BEST_EFFORT_RSS_GUARD_NOT_HARD_RSS_LIMIT"
ACTUAL_CPU_RESERVE_SECONDS = 120
ACTUAL_CPU_GUARD_SOURCE = "PROC_STAT_CUMULATIVE_FROM_PROCESS_BIRTH"
CPU_GUARD_KILL_GRACE_SECONDS = 1.0
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
EXIT_REAP_GRACE_SECONDS = .5
CONTINUATION_REGISTRY = ROOT / ".runtime/research4-research72h-authorization-claims"

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


def healthy_closed_row(row):
    return (row.get("status") == "EXITED" and row.get("exit_code") == 0 and bool(row.get("ended_at_utc"))
        and all(row.get(k) is False for k in ("wall_terminated","memory_terminated","cpu_terminated","monitor_failed","hard_killed"))
        and not row.get("residual_processes_require_review_after_main_payload_exit")
        and type(row.get("actual_cpu_seconds")) in (int,float)
        and math.isfinite(row["actual_cpu_seconds"]) and row["actual_cpu_seconds"] >= 0)



def old_guard():
    if sha(OLD_GUARD) != OLD_GUARD_SHA:
        raise ValueError("Frozen read-only V8 dependency changed")
    spec = importlib.util.spec_from_file_location("research72h_readonly_v8",OLD_GUARD)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def checked_json(path, expected):
    path = root_path(path)
    if path.is_symlink() or not path.is_relative_to(ROOT) or not path.is_file() or path.stat().st_size > 32*1024**2 or sha(path) != expected:
        raise ValueError("Bounded repository JSON with exact SHA required")
    return read(path)


def validate_authorization(path, expected_sha):
    if expected_sha != EXPECTED_AUTH_SHA:
        raise ValueError("This source supports only the observed first prospective authorization")
    auth = checked_json(path,expected_sha)
    if auth.get("schema") != AUTH_SCHEMA or auth.get("authorization_source") != "ACTUAL_USER_MESSAGE_CURRENT_CHAT" or not auth.get("user_text"):
        raise ValueError("Actual user72h authorization required")
    start,deadline = (aware_time(auth[k],k) for k in ("observed_at_utc","run_deadline_at_utc"))
    if not start <= utc() or (deadline-start).total_seconds() != MAX_WALL:
        raise ValueError("The observed72h window cannot be reset from initialization time")
    if (auth.get("authorized_additional_cpu_seconds") != MAX_ADDITIONAL_CPU
            or auth.get("maximum_worker_threads") != 2 or auth.get("phase_memory_RSS_softstop_GiB") != 10
            or auth.get("host_cgroup_memory_GiB") != 16 or auth.get("minimum_workspace_free_bytes") != 64*1024**2
            or auth.get("source_ledger_must_remain_byte_identical") is not True):
        raise ValueError("Explicit finite allocation or physical resource policy differs")
    source = checked_json(auth["source_ledger"],auth["source_ledger_sha256"])
    old_guard().validate_ledger(source,allow_running=False)
    if (source["schema"] != OLD_SCHEMA or source["consumed_cpu_seconds"] != auth["source_consumed_cpu_seconds"]
            or source["remaining_quantum_cpu_seconds"] != auth["source_unused_cpu_seconds_retired_not_recredited"]
            or not math.isclose(auth["new_cumulative_cap_cpu_seconds"],source["consumed_cpu_seconds"]+MAX_ADDITIONAL_CPU,rel_tol=0,abs_tol=1e-7)):
        raise ValueError("Carry exact actual failed+successful old costs once; never recredit old unused allocation")
    target = root_path(auth["new_ledger_path"])
    if target == root_path(auth["source_ledger"]) or not target.is_relative_to(ROOT/"reports/research72h-20261107-v1"):
        raise ValueError("New sole ledger must preserve all previous report trees")
    # Policy aliases keep the reviewed native monitoring API unchanged.
    return {**auth,"continuation_ledger_path":str(target),"prior_closed_ledger_path":auth["source_ledger"],
        "max_threads":2,"max_memory_gib":24,"max_phase_cpu_seconds":MAX_PHASE_CPU,"max_phase_wall_seconds":MAX_PHASE_WALL,
        "rss_stop_bytes":RSS_STOP_BYTES,"cgroup_kill_bytes":CGROUP_KILL_BYTES,
        "monitor_interval_seconds":MONITOR_INTERVAL_SECONDS,"memory_kill_grace_seconds":MEMORY_KILL_GRACE_SECONDS,
        "expected_cgroup_memory_max_bytes":CGROUP_MEMORY_MAX_BYTES,"actual_cpu_cleanup_reserve_seconds":ACTUAL_CPU_RESERVE_SECONDS,
        "cpu_guard_kill_grace_seconds":CPU_GUARD_KILL_GRACE_SECONDS}


@contextmanager
def lineage_locks(auth):
    source = root_path(auth["source_ledger"])
    if source.with_suffix(".lock").is_symlink():
        raise ValueError("Symlink legacy ledger lock rejected")
    with source.with_suffix(".lock").open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior = checked_json(source,auth["source_ledger_sha256"])
        prior_auth = old_guard().validate_ledger(prior)
        with old_guard().lineage_locks(prior_auth):
            yield
        checked_json(source,auth["source_ledger_sha256"])


def claim_value(auth, auth_path, auth_sha, ledger_path):
    return {"schema":"research4-single-observed72h-authorization-claim-v1",
        "authorization_path":str(root_path(auth_path)),"authorization_sha256":auth_sha,
        "source_ledger":auth["source_ledger"],"source_ledger_sha256":auth["source_ledger_sha256"],
        "new_ledger_path":str(root_path(ledger_path)),"runner_sha256":sha(__file__),
        "authorized_additional_cpu_seconds":MAX_ADDITIONAL_CPU,"deadline":auth["run_deadline_at_utc"]}


def initialize(args):
    if args.ledger.exists():
        raise FileExistsError("Never overwrite or reset an existing ledger")
    auth = validate_authorization(args.authorization,args.authorization_sha256)
    if args.ledger.resolve() != root_path(auth["new_ledger_path"]) or utc() >= aware_time(auth["run_deadline_at_utc"],"deadline"):
        raise ValueError("Wrong sole ledger or expired prospective authorization")
    with lineage_locks(auth):
        source = checked_json(auth["source_ledger"],auth["source_ledger_sha256"])
        claim = CONTINUATION_REGISTRY/(args.authorization_sha256+".json")
        claim.parent.mkdir(parents=True,exist_ok=True)
        if any(p.is_symlink() for p in (claim,*claim.parents)):
            raise ValueError("Symlink authorization claim or parent rejected")
        value = claim_value(auth,args.authorization,args.authorization_sha256,args.ledger)
        save_new_ledger(claim,value)
        ledger = {"schema":SCHEMA,"runner_sha256":sha(__file__),"created_at_utc":utc().isoformat(),
            "run_deadline_at_utc":auth["run_deadline_at_utc"],"effective_run_deadline_at_utc":auth["run_deadline_at_utc"],
            "authorization_observed_at_utc":auth["observed_at_utc"],"run_wall_cap_seconds":MAX_WALL,
            "authorization_path":str(args.authorization.resolve()),"authorization_sha256":args.authorization_sha256,
            "continuation_ledger_path":str(args.ledger.resolve()),"authorization_claim":{"path":str(claim),"sha256":sha(claim)},
            "source_ledger":{"path":auth["source_ledger"],"sha256":auth["source_ledger_sha256"],
                "phase_count":len(source["phases"]),"actual_closed_CPU_seconds":source["consumed_cpu_seconds"]},
            "source_phase_prefix_sha256":hashlib.sha256(json.dumps(source["phases"],sort_keys=True).encode()).hexdigest(),
            "quantum_cpu_cap_seconds":auth["new_cumulative_cap_cpu_seconds"],"consumed_cpu_seconds":source["consumed_cpu_seconds"],
            "remaining_quantum_cpu_seconds":float(MAX_ADDITIONAL_CPU),"phases":source["phases"],
            "prior_budget_is_historical_only":True,"old_unused_CPU_seconds_retired_not_recredited":source["remaining_quantum_cpu_seconds"],
            "new_allocation_CPU_seconds":MAX_ADDITIONAL_CPU,"no_automatic_budget_growth":True,
            "guard_semantics":GUARD_SEMANTICS,"not_a_hard_RSS_limit":True,"no_inferred_orphan_recovery":True}
        save_new_ledger(args.ledger,ledger)
        validate_ledger(ledger)
        return ledger


def validate_ledger(ledger, *, allow_running=False):
    if ledger.get("schema") == OLD_SCHEMA:
        return old_guard().validate_ledger(ledger,allow_running=allow_running)
    if ledger.get("schema") != SCHEMA or ledger.get("runner_sha256") != sha(__file__):
        raise ValueError("Unknown new ledger or changed frozen guard source")
    auth = validate_authorization(ledger["authorization_path"],ledger["authorization_sha256"])
    source = checked_json(auth["source_ledger"],auth["source_ledger_sha256"])
    claim = ledger.get("authorization_claim",{})
    expected_claim = claim_value(auth,ledger["authorization_path"],ledger["authorization_sha256"],auth["new_ledger_path"])
    if checked_json(claim.get("path",""),claim.get("sha256","")) != expected_claim:
        raise ValueError("Sole authorization claim differs")
    count=len(source["phases"]);rows=ledger.get("phases",[])
    if (rows[:count] != source["phases"] or ledger.get("source_phase_prefix_sha256") != hashlib.sha256(json.dumps(source["phases"],sort_keys=True).encode()).hexdigest()
            or ledger.get("source_ledger") != {"path":auth["source_ledger"],"sha256":auth["source_ledger_sha256"],
                "phase_count":count,"actual_closed_CPU_seconds":source["consumed_cpu_seconds"]}
            or ledger.get("continuation_ledger_path") != auth["new_ledger_path"]):
        raise ValueError("Full old closed phase prefix, costs or sole ledger identity changed")
    for key,value in {"quantum_cpu_cap_seconds":auth["new_cumulative_cap_cpu_seconds"],"run_deadline_at_utc":auth["run_deadline_at_utc"],
        "effective_run_deadline_at_utc":auth["run_deadline_at_utc"],"authorization_observed_at_utc":auth["observed_at_utc"],
        "run_wall_cap_seconds":MAX_WALL,"new_allocation_CPU_seconds":MAX_ADDITIONAL_CPU,
        "old_unused_CPU_seconds_retired_not_recredited":source["remaining_quantum_cpu_seconds"],
        "no_automatic_budget_growth":True,"prior_budget_is_historical_only":True}.items():
        if ledger.get(key) != value:
            raise ValueError("Immutable allocation field differs: "+key)
    running=[r for r in rows if r.get("status")=="RUNNING"]
    if running and (not allow_running or len(running)!=1):
        raise RuntimeError("Unknown/open phase blocks recovery; never infer closure from missing PID or reboot")
    if any(r.get("post_exit_process_audit_failure") or r.get("actual_CPU_is_unknown_do_not_resume_budget")
           for r in rows[count:] if r.get("status")!="RUNNING"):
        raise RuntimeError("Unresolved owned-process audit or actual CPU requires explicit forensic recovery")
    used=closed_costs([r for r in rows if r.get("status")!="RUNNING"])
    names=[r.get("phase") for r in rows]
    if len(names)!=len(set(names)) or not math.isclose(used,ledger.get("consumed_cpu_seconds",float("nan")),rel_tol=1e-12,abs_tol=1e-7):
        raise ValueError("Unique phase identity or exact actual CPU cost differs")
    if not math.isclose(max(0.,ledger["quantum_cpu_cap_seconds"]-used),ledger.get("remaining_quantum_cpu_seconds",float("nan")),rel_tol=1e-12,abs_tol=1e-7):
        raise ValueError("Remaining CPU must subtract all actual costs")
    return auth

def await_live_ledger_binding(ledger_path, proof, proof_sha, *, reader=None, clock=None, sleeper=None, maximum_seconds=1.0):
    """Bounded launch handshake after live parent verification; never orphan recovery."""
    reader,clock,sleeper=reader or read,clock or time.monotonic,sleeper or time.sleep
    deadline=clock()+maximum_seconds
    identity={"pid":proof["child_pid"],"start_ticks":proof["child_start_ticks"],"uid":proof["target_uid"],
              "pgrp":proof["target_pgid"],"session":proof["target_session_id"]}
    while True:
        ledger=reader(root_path(ledger_path));rows=[r for r in ledger.get("phases",[]) if r.get("phase")==proof["phase"]]
        validate_ledger(ledger,allow_running=True)
        if (len(rows)!=1 or rows[0].get("status")!="RUNNING" or ledger.get("schema")!=SCHEMA
                or root_path(ledger.get("continuation_ledger_path",""))!=root_path(ledger_path)):
            raise ValueError("Startup handshake requires this exact sole registered RUNNING phase")
        row=rows[0]
        for key in ("boot_id","supervisor_pid","supervisor_start_ticks","actual_cpu_stop_threshold_seconds",
                    "actual_cpu_clock_ticks_per_second","actual_cpu_cleanup_reserve_seconds",
                    "actual_cpu_closed_batch_cost_seconds","actual_cpu_batch_cap_seconds","actual_cpu_guard_source"):
            if row.get(key)!=proof.get(key):
                raise ValueError("Startup guard/counter/parent binding changed: "+key)
        if (("child_identity" in row and row["child_identity"]!=identity)
                or ("pid" in row and row["pid"]!=proof["child_pid"])
                or ("supervisor_proof_sha256" in row and row["supervisor_proof_sha256"]!=proof_sha)):
            raise ValueError("Startup child identity or immutable proof changed")
        if row.get("child_identity")==identity and row.get("supervisor_proof_sha256")==proof_sha:
            return ledger
        if clock()>=deadline:
            raise ValueError("Live parent did not persist this exact owned child/proof within the bounded startup handshake")
        sleeper(.01)


def verify_resource_supervision(proof_path=None, *, expected_threads=None):
    """Verify the real V8 parent, cumulative allocation and unchanged V6b source."""
    supplied = proof_path or os.environ.get("R4_RESOURCE_SUPERVISOR_PROOF")
    if not supplied:
        raise ValueError("Actual-CPU V8 supervision proof required before quantum work")
    path = Path(supplied).resolve()
    raw = path.read_bytes()
    if len(raw) > 65536:
        raise ValueError("Unexpectedly large supervision proof")
    proof = json.loads(raw)
    if proof.get("schema") != PROOF_SCHEMA or proof.get("guard_semantics") != GUARD_SEMANTICS:
        raise ValueError("A real independent V8 actual-CPU proof is required")
    if proof.get("terminal_exit_reap_grace_seconds") != EXIT_REAP_GRACE_SECONDS:
        raise ValueError("V8 requires its explicit bounded terminal reaping policy")
    if proof.get("boot_id") != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
        raise ValueError("Proof belongs to a different boot; do not infer old child ownership")
    observed = resource_runtime_snapshot()
    for key in ("supervisor_pid", "supervisor_start_ticks", "child_pid", "child_start_ticks"):
        if type(proof.get(key)) is not int or proof[key] != observed[key]:
            raise ValueError("Live supervisor/child identity mismatch: " + key)
    source = Path(__file__).resolve()
    if root_path(proof.get("supervisor_script_path", "")) != source or proof.get("supervisor_script_sha256") != sha(source):
        raise ValueError("Actual V8 supervisor source path/SHA differs")
    commands = [Path(os.fsdecode(token)) for token in observed["parent_command"] if token]
    if not any((token if token.is_absolute() else Path(observed["parent_cwd"])/token).resolve() == source for token in commands):
        raise ValueError("Actual parent command is not the V8 supervisor")
    expected_values = {"address_space_soft_limit_bytes": 24*1024**3, "address_space_hard_limit_bytes": 24*1024**3,
        "rss_stop_bytes": 10*1024**3, "cgroup_kill_bytes": 12*1024**3,
        "expected_cgroup_memory_max_bytes": CGROUP_MEMORY_MAX_BYTES}
    if any(type(proof.get(key)) is not int or proof[key] != value for key,value in expected_values.items()):
        raise ValueError("Unreviewed V8 resource policy")
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
    if type(proof.get("threads")) is not int or not 1 <= proof["threads"] <= 2:
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
            or type(cap) not in (int,float) or not math.isfinite(cap) or not 0 < cap <= MAX_CPU
            or type(closed) not in (int,float) or not math.isfinite(closed) or closed < 0
            or proof.get("actual_cpu_clock_ticks_per_second") != observed["clock_ticks_per_second"]
            or proof.get("actual_cpu_stop_threshold_seconds") != min(cpu[1],cap-closed)-reserve):
        raise ValueError("Actual cumulative CPU guard parameters differ")
    if type(proof.get("wall_limit_seconds")) is not int or not 61 <= proof["wall_limit_seconds"] <= MAX_PHASE_WALL:
        raise ValueError("Finite unchanged V8 phase wall bound required")
    ledger = await_live_ledger_binding(proof.get("ledger_path",""),proof,hashlib.sha256(raw).hexdigest())
    validate_ledger(ledger, allow_running=True)
    if root_path(proof["ledger_path"]) != root_path(ledger["continuation_ledger_path"]):
        raise ValueError("Actual proof must use the sole registered V8 ledger path")
    phases = [row for row in ledger.get("phases",[]) if row.get("phase") == proof.get("phase")]
    if (ledger.get("schema") != SCHEMA or len(phases) != 1 or phases[0].get("status") != "RUNNING"
            or sum(row.get("status") == "RUNNING" for row in ledger["phases"]) != 1
            or ledger.get("runner_sha256") != sha(source)):
        raise ValueError("Actual new V8 ledger must contain one matching live phase")
    if ledger.get("quantum_cpu_cap_seconds") != cap or ledger.get("consumed_cpu_seconds") != closed or ledger.get("prior_budget_is_historical_only") is not True:
        raise ValueError("Live cumulative V8 pool differs from proof; closed V5 costs must be carried")
    for key in ("actual_cpu_stop_threshold_seconds", "actual_cpu_clock_ticks_per_second", "actual_cpu_cleanup_reserve_seconds",
                "actual_cpu_closed_batch_cost_seconds", "actual_cpu_batch_cap_seconds", "actual_cpu_guard_source"):
        if phases[0].get(key) != proof[key]:
            raise ValueError("CPU guard proof differs from actual running row: " + key)
    if phases[0].get("child_identity") != {"pid":proof["child_pid"],"start_ticks":proof["child_start_ticks"],"uid":proof["target_uid"],"pgrp":proof["target_pgid"],"session":proof["target_session_id"]}:
        raise ValueError("Persisted owned child identity differs from actual proof")
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
        "closed_V8_lineage_and_new72h_explicit_increment_verified": True}


def verify_closed_phase(evidence, *, expected_worker_path, expected_worker_sha256,
                        expected_result_schema=None, allowed_result_statuses=(),
                        expected_output=None, expected_snapshot_sha256=None, allow_failure=False):
    """Stdlib producer-closure gate; no numerical payload is ever loaded here."""
    if expected_result_schema is None:
        proof=evidence.get("resource_supervision",{}).get("proof",{})
        return verify_closed_row(evidence.get("ledger_path",proof.get("ledger_path")),
            evidence.get("phase",proof.get("phase")),expected_worker_path=expected_worker_path,
            expected_worker_sha256=expected_worker_sha256,expected_output=expected_output,allow_failure=allow_failure)
    if not allowed_result_statuses:
        raise ValueError("Caller must explicitly list successful scientific stage statuses")
    legacy_supervision=evidence.get("resource_supervision",{})
    raw_path=root_path(legacy_supervision.get("proof_path",""))
    if not raw_path.is_file() or sha(raw_path)!=legacy_supervision.get("proof_sha256"):
        raise ValueError("Closed proof changed before era dispatch")
    incoming=read(raw_path)
    if incoming.get("schema")==OLD_PROOF_SCHEMA:
        if (root_path(incoming.get("supervisor_script_path",""))!=OLD_GUARD
                or incoming.get("supervisor_script_sha256")!=OLD_GUARD_SHA):
            raise ValueError("Old proof source/path mismatch; never relabel it asV8")
        return old_guard().verify_closed_phase(evidence,expected_worker_path=expected_worker_path,
            expected_worker_sha256=expected_worker_sha256,expected_result_schema=expected_result_schema,
            allowed_result_statuses=allowed_result_statuses,expected_output=expected_output,
            expected_snapshot_sha256=expected_snapshot_sha256)
    if incoming.get("schema")!=PROOF_SCHEMA:
        raise ValueError("Unknown closed proof schema; no implicit compatibility")
    worker = root_path(expected_worker_path)
    source = evidence.get("source_worker", {})
    if (root_path(source.get("path", "")) != worker or source.get("sha256") != expected_worker_sha256
            or sha(worker) != expected_worker_sha256):
        raise ValueError("Unexpected frozen V8 stage entry source")
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
        raise ValueError("Actual V8 parent proof required, never a relabelled old proof")
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
    if ledger.get("schema") != SCHEMA:
        raise ValueError("Live launches require a newly registered V8 ledger; legacy is read-only")
    auth=validate_ledger(ledger)
    if args.ledger.resolve()!=root_path(auth["continuation_ledger_path"]):
        raise ValueError("Never launch from an unregistered ledger alias")
    with lineage_locks(auth):
        return _run_phase_with_prior_lock(args,ledger)


def _run_phase_with_prior_lock(args, ledger):
    auth = validate_ledger(ledger)
    if not args.phase or args.phase in {row["phase"] for row in ledger["phases"]} or args.log is None:
        raise ValueError("Fresh phase name and output log required")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("Explicit single-payload command required")
    limits = phase_limits(ledger, auth, args)
    if shutil.disk_usage(ROOT).free < auth["minimum_workspace_free_bytes"]:
        raise RuntimeError("Insufficient workspace disk; preserve old files and do not launch")
    args.log = args.log.resolve()
    prior_tree = root_path(auth["prior_closed_ledger_path"]).parent
    if (not args.log.is_relative_to(ROOT/"reports/research72h-20261107-v1")
            or args.log.exists() or args.log == args.ledger.resolve() or args.log.is_relative_to(OLD_PILOT)
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
                 "boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                 "immutable_legacy_monitor_source_sha256":OLD_GUARD_SHA,
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
    row = {"boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
           "supervisor_pid":supervisor_pid,"supervisor_start_ticks":supervisor_info["start_ticks"],
           "phase": args.phase, "started_at_utc": utc().isoformat(), "status": "RUNNING",
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
    last_heartbeat = 0.0
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
                row.update(status="UNRESOLVED_START_FAILURE", ended_at_utc=utc().isoformat(), actual_cpu_seconds=None,
                           actual_CPU_is_unknown_do_not_resume_budget=True,
                           elapsed_wall_seconds=time.monotonic() - start, exit_code=None, failure=repr(exc),
                           log_sha256=sha(args.log))
                save(args.ledger, ledger)
                raise
            row["pid"] = child.pid
            row["supervisor_proof_sha256"] = sha(proof_path)
            proof = read(proof_path)
            expected_identity = {"pid": child.pid, "start_ticks": proof["child_start_ticks"],
                "uid": proof["target_uid"], "pgrp": proof["target_pgid"], "session": proof["target_session_id"]}
            row["child_identity"] = dict(expected_identity)
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
                    free_bytes = shutil.disk_usage(ROOT).free
                    if free_bytes < auth["minimum_workspace_free_bytes"]:
                        raise RuntimeError("DISK_FREE_GUARD_FAILED_CLOSED_PRESERVE_OLD_FILES")
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
                    if elapsed-last_heartbeat >= 1.0:
                        row.update(last_monitor_at_utc=utc().isoformat(),last_monitor_elapsed_wall_seconds=elapsed,
                            last_monitor_CPU_observation=cpu_observation,last_monitor_cgroup=cgroup,
                            last_monitor_workspace_free_bytes=free_bytes,peak_actual_cpu_seconds_sampled=peak_actual_cpu)
                        save(args.ledger,ledger)
                        last_heartbeat = elapsed
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
               owned_process_closure_confirmed=not residual["pids"] and not row.get("post_exit_process_audit_failure"),
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



def verify_closed_row(ledger_path, phase, *, expected_worker_path, expected_worker_sha256, expected_output=None, allow_failure=False):
    """Authenticate actual guardian closure even when an ML payload has no proof field."""
    ledger=read(root_path(ledger_path));validate_ledger(ledger,allow_running=True)
    if ledger.get("schema")!=SCHEMA:
        raise ValueError("Generic closure requires the new registered ledger")
    rows=[r for r in ledger["phases"] if r.get("phase")==phase]
    closed_failed = (len(rows)==1 and allow_failure and rows[0].get("status")=="PARTIAL_OR_FAILED"
        and bool(rows[0].get("ended_at_utc")) and type(rows[0].get("exit_code")) is int
        and type(rows[0].get("actual_cpu_seconds")) in (int,float)
        and math.isfinite(rows[0]["actual_cpu_seconds"]) and rows[0]["actual_cpu_seconds"]>=0)
    if len(rows)!=1 or (not healthy_closed_row(rows[0]) and not closed_failed):
        raise ValueError("Actual phase is not a healthy owned-process EXITED/0 closure")
    row=rows[0];path=root_path(row["supervisor_proof_path"]);proof=read(path)
    if (row.get("owned_process_closure_confirmed") is not True or row.get("post_exit_process_audit_failure")
            or row.get("residual_processes_require_review_after_main_payload_exit")
            or row.get("actual_CPU_is_unknown_do_not_resume_budget")):
        raise ValueError("Closed payload process ownership or actual CPU remains unresolved")
    worker=root_path(expected_worker_path)
    if (sha(path)!=row.get("supervisor_proof_sha256") or proof.get("schema")!=PROOF_SCHEMA
            or proof.get("supervisor_script_sha256")!=sha(__file__)
            or root_path(proof.get("supervisor_script_path",""))!=Path(__file__).resolve()
            or root_path(proof.get("ledger_path",""))!=root_path(ledger_path) or proof.get("phase")!=phase
            or proof.get("boot_id")!=row.get("boot_id")
            or row.get("child_identity")!={"pid":proof.get("child_pid"),"start_ticks":proof.get("child_start_ticks"),
                "uid":proof.get("target_uid"),"pgrp":proof.get("target_pgid"),"session":proof.get("target_session_id")}
            or sha(worker)!=expected_worker_sha256 or sha(root_path(row["log"]))!=row.get("log_sha256")):
        raise ValueError("Closed phase source, proof, log or ledger identity differs")
    command=row.get("command",[])
    if worker not in [root_path(t) for t in command if isinstance(t,str) and t.endswith(".py")]:
        raise ValueError("Closed phase command does not bind expected worker")
    if expected_output is not None and ("--output" not in command
            or root_path(command[command.index("--output")+1])!=root_path(expected_output)):
        raise ValueError("Closed phase output differs")
    return {"phase":phase,"closed_row_sha256":hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest(),
        "proof_sha256":sha(path),"log_sha256":sha(root_path(row["log"])),
        "actual_cpu_seconds":row["actual_cpu_seconds"],"all_owned_processes_closed":True,
        "payload_result_json_not_required":True,"scientific_success":healthy_closed_row(row),
        "actual_exit_code":row["exit_code"],"actual_status":row["status"]}


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="action",required=True)
    init=sub.add_parser("init",help="Register the one observed prospective72h allocation; launch no payload")
    init.add_argument("--ledger",type=Path,required=True)
    init.add_argument("--authorization",type=Path,required=True);init.add_argument("--authorization-sha256",required=True)
    check=sub.add_parser("validate",help="Read-only validation; RUNNING/unknown phases block ordinary resume")
    check.add_argument("--ledger",type=Path,required=True)
    run=sub.add_parser("run",help="One bounded serial payload under the new allocation")
    for key in ("ledger","log"):run.add_argument("--"+key,type=Path,required=True)
    run.add_argument("--phase",required=True);run.add_argument("--cpu-seconds",type=int,required=True)
    run.add_argument("--wall-seconds",type=int,required=True);run.add_argument("--threads",type=int,default=2)
    run.add_argument("--memory-gib",type=int,default=24);run.add_argument("command",nargs=argparse.REMAINDER)
    return p


def main():
    args=parser().parse_args();args.ledger=args.ledger.resolve()
    if not args.ledger.is_relative_to(ROOT/"reports/research72h-20261107-v1"):
        raise ValueError("Only the new research72h ledger can be mutated or launched")
    if args.action=="validate":
        value=read(args.ledger);validate_ledger(value)
        print(json.dumps({"status":"ALL_PHASES_CLOSED_VALIDATED","ledger":str(args.ledger),
            "consumed_cpu_seconds":value["consumed_cpu_seconds"],"remaining_quantum_cpu_seconds":value["remaining_quantum_cpu_seconds"]}));return
    args.ledger.parent.mkdir(parents=True,exist_ok=True)
    local_path=args.ledger.with_suffix(".lock")
    shared=ROOT/".runtime/pyscf-v4-global.lock";shared.parent.mkdir(parents=True,exist_ok=True)
    if local_path.is_symlink() or shared.is_symlink():
        raise ValueError("Symlink ledger/global lock rejected")
    with local_path.open("a") as local,shared.open("a") as global_lock:
        fcntl.flock(local,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=="init":
            ledger=initialize(args)
            print(json.dumps({"status":"NEW_OBSERVED72H_REGISTERED_NO_PAYLOAD","ledger":str(args.ledger),
                "consumed_cpu_seconds":ledger["consumed_cpu_seconds"],"quantum_cpu_cap_seconds":ledger["quantum_cpu_cap_seconds"],
                "deadline":ledger["run_deadline_at_utc"]}),flush=True);return
        row=run_phase(args,read(args.ledger));print(json.dumps(row),flush=True)
        raise SystemExit(0 if row["status"]=="EXITED" else 1)


if __name__=="__main__":main()
