#!/usr/bin/env python3
"""Detach, supervise, verify and publish the frozen B-mechanism round 3 experiment.

The worker is independent of the invoking shell.  It is not independent of the
cloud machine's lifetime: a suspended or destroyed machine cannot execute code.
Restart with --resume on a surviving checkout to reuse the worker's checkpoints.
There are no score, wall-time, CPU-budget or retry-count stopping thresholds.
Exit code 42 is reserved for integrity errors and is never retried blindly.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/b_mechanism_round3_20261110"
RUNTIME = OUTPUT / ".background"
STATE = RUNTIME / "state.json"
RECEIPT = OUTPUT / "background_finalization.json"
BRANCH = "research/b-mechanism-round3-20261110"
BASE = "research/b-mechanism-followup-20261110"
INTEGRITY_ERROR = 42
PROTOCOL = OUTPUT / "frozen_protocol.json"
SINGLE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
)
TASK_SCRIPTS = (
    "run_b_mechanism_round3_20261110.py",
    "summarize_b_mechanism_round3_20261110.py",
    "b_round3_common_20261110.py",
    "b_round3_regularization_20261110.py",
    "b_round3_labels_20261110.py",
    "b_round3_margin_20261110.py",
    "background_b_mechanism_round3_20261110.py",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def planned_counts(required=True):
    """Read the frozen finite matrix; never guess scientific fit counts."""
    protocol = read_json(PROTOCOL)
    if protocol is None:
        if not required:
            return None
        raise ValueError("Integrity error: prepare the frozen round 3 protocol before launching")
    design = protocol.get("design")
    if not isinstance(design, list) or not design:
        raise ValueError("Integrity error: frozen protocol has no task design")
    identifiers = [item.get("job_id") for item in design if isinstance(item, dict)]
    if len(identifiers) != len(design) or any(not isinstance(item, str) or not item for item in identifiers):
        raise ValueError("Integrity error: frozen task identities are incomplete")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Integrity error: frozen task identities are duplicated")
    if not isinstance(protocol.get("input_hash"), str) or not protocol["input_hash"]:
        raise ValueError("Integrity error: frozen protocol has no scientific input hash")
    counts = {}
    for name in ("planned_new_model_fits", "planned_new_adapter_fits"):
        value = protocol.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Integrity error: invalid frozen count " + name)
        counts[name] = value
    return {
        "planned_task_bundles": len(design), **counts,
        "frozen_protocol_sha256": sha256(PROTOCOL),
        "frozen_input_hash": protocol.get("input_hash"),
    }


def ensure_frozen_protocol(state):
    counts = planned_counts()
    for name, actual in counts.items():
        if state.get(name) != actual:
            raise ValueError("Integrity error: frozen protocol changed after supervisor startup: " + name)
    return counts


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def safe_message(message):
    result = str(message)
    for key in ("GH_TOKEN", "GITHUB_TOKEN"):
        value = os.environ.get(key)
        if value and len(value) >= 8:
            result = result.replace(value, "[REDACTED]")
    return result[-8000:]


def git(*arguments):
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, text=True, capture_output=True,
        stdin=subprocess.DEVNULL, check=False,
    )


def require_branch():
    result = git("branch", "--show-current")
    if result.returncode or result.stdout.strip() != BRANCH:
        raise ValueError("Integrity error: the checkout must be on " + BRANCH)


def process_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ValueError, TypeError, ProcessLookupError, PermissionError):
        return False


def supervisor_alive(state):
    if not state or not process_alive(state.get("supervisor_pid")):
        return False
    try:
        command = Path(f"/proc/{state['supervisor_pid']}/cmdline").read_bytes()
        return (
            Path(__file__).name.encode() in command
            and b"--supervise" in command
            and state.get("instance", "").encode() in command
        )
    except (OSError, KeyError):
        return False


def status():
    state = read_json(STATE)
    result = {"state": state, "supervisor_alive": supervisor_alive(state)}
    if state:
        result["worker_alive"] = process_alive(state.get("worker_pid"))
        try:
            result["observed_supervisor_sid"] = os.getsid(state["supervisor_pid"])
        except (OSError, TypeError, KeyError):
            result["observed_supervisor_sid"] = None
    result["finalization_receipt"] = read_json(RECEIPT)
    result["cloud_lifetime_guaranteed"] = False
    return result


def phases():
    return [
        ("training", [sys.executable, str(ROOT / "scripts" / TASK_SCRIPTS[0]), "--resume"]),
        ("summarizing", [sys.executable, str(ROOT / "scripts" / TASK_SCRIPTS[1]), "--resume"]),
        ("verifying_summary", [sys.executable, str(ROOT / "scripts" / TASK_SCRIPTS[1]), "--verify"]),
        ("verifying_artifacts", [sys.executable, str(ROOT / "scripts" / TASK_SCRIPTS[0]), "--verify"]),
    ]


def eligible_files():
    candidates = [ROOT / "scripts" / name for name in TASK_SCRIPTS]
    candidates += [ROOT / "tests/test_b_mechanism_round3_20261110.py"]
    candidates += list((ROOT / "docs").glob("B_MECHANISM_ROUND3*"))
    if OUTPUT.exists():
        candidates += list(OUTPUT.rglob("*"))
    selected = []
    for path in candidates:
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(ROOT)
        if path == OUTPUT / ".gitignore":
            selected.append(relative.as_posix())
            continue
        if any(
            part.startswith(".") or part in {"__pycache__", "cache", "features", "feature_cache"}
            for part in relative.parts
        ):
            continue
        if path.name.endswith((".pid", ".lock", ".log", ".tmp", ".partial", ".pyc")):
            continue
        if path.name == "features.npz" or path.name.startswith("features_"):
            continue
        selected.append(relative.as_posix())
    return sorted(set(selected))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preserved_sessions(previous):
    """Archive per-instance counters, never an already accumulated total twice."""
    if not previous:
        return []
    unique = {
        item["instance"]: item for item in previous.get("session_history", [])
        if isinstance(item, dict) and item.get("instance")
    }
    instance = previous.get("instance")
    if instance:
        interrupted = previous.get("active_phase_attempt")
        # Also recognize states written by the original supervisor version.
        if not interrupted and previous.get("worker_pid"):
            interrupted = {
                "phase": previous.get("phase"),
                "attempt": previous.get("phase_attempt"),
                "worker_pid": previous["worker_pid"],
            }
        unique[instance] = {
            "instance": instance,
            "started_at_utc": previous.get("started_at_utc"),
            "last_heartbeat_at_utc": previous.get("heartbeat_at_utc"),
            "status_at_resume": previous.get("status"),
            "measured_child_cpu_seconds": previous.get(
                "session_child_cpu_seconds", previous.get("total_child_cpu_seconds", 0.0)),
            "measured_supervisor_cpu_seconds": previous.get(
                "session_supervisor_cpu_seconds", previous.get("total_supervisor_cpu_seconds", 0.0)),
            "measured_phase_child_cpu_seconds": previous.get(
                "session_phase_child_cpu_seconds", previous.get("total_phase_child_cpu_seconds", 0.0)),
            "phase_attempt_costs": previous.get("phase_attempt_costs", []),
            "unmeasured_interrupted_attempt": (
                {**interrupted, "cpu_seconds": None,
                 "measurement_status": "UNMEASURED_INTERRUPTED",
                 "reason": "The previous supervisor did not persist a post-wait CPU measurement; unknown CPU is not counted as zero."}
                if interrupted else None
            ),
        }
    return list(unique.values())


def compute_cost_views(state):
    """Expose overlapping measurements rather than adding them into a total."""
    engine_results = [read_json(path) for path in (OUTPUT / "jobs").glob("*/result.json")]
    engine_results = [value for value in engine_results if value]
    execution = read_json(OUTPUT / "execution_state.json") or {}
    attempts = execution.get("attempts", [])
    planned = ensure_frozen_protocol(state)
    unmeasured = [
        {"job_id": item.get("job_id"), "started_at_utc": item.get("started_at_utc"),
         "status": item.get("status"), "cpu_seconds": None,
         "measurement_status": "UNMEASURED_INTERRUPTED"}
        for item in attempts if item.get("attempt_cpu_seconds") is None
    ]
    return {
        "measured_at_utc": state["heartbeat_at_utc"],
        "engine_job_cpu_seconds": sum(item.get("cpu_seconds", 0.0) for item in engine_results),
        "engine_completed_jobs": len(engine_results),
        "actual_new_classifier_fits": sum(item.get("new_model_fits", 0) for item in engine_results),
        "actual_new_adapter_fits": sum(item.get("new_adapter_fits", 0) for item in engine_results),
        **planned,
        "planned_new_classifier_fits": planned["planned_new_model_fits"],
        "cost_scope": "Only newly executed round 3 work; reused parent models and predictions are provenance references and add zero new fitting cost.",
        "engine_execution_attempt_cpu_seconds": sum(
            item.get("attempt_cpu_seconds", 0.0) for item in attempts
            if item.get("attempt_cpu_seconds") is not None),
        "engine_execution_attempt_scope": "Measured execution_state.json attempts, including the separately executed pilot and failed attempts; excludes unmeasured interrupted attempts.",
        "engine_unmeasured_interrupted_attempts": unmeasured,
        "supervisor_total_child_cpu_seconds": state["total_child_cpu_seconds"],
        "supervisor_total_self_cpu_seconds": state["total_supervisor_cpu_seconds"],
        "supervisor_total_phase_child_cpu_seconds": state["total_phase_child_cpu_seconds"],
        "supervisor_session_history": state.get("session_history", []),
        "supervisor_current_phase_attempt_costs": state["phase_attempt_costs"],
        "supervisor_unmeasured_interrupted_attempts": [
            {"instance": item["instance"], **item["unmeasured_interrupted_attempt"]}
            for item in state.get("session_history", []) if item.get("unmeasured_interrupted_attempt")
        ],
        "supervisor_scope": "RUSAGE_CHILDREN for completed phases across unique supervisor sessions; excludes the pilot executed before supervision. Final Git publication overhead is recorded subsequently in runtime state.",
        "accounting_rule": "OVERLAPPING_VIEWS_DO_NOT_ADD: engine job CPU is contained in execution-attempt CPU; supervised execution-attempt CPU is contained in supervisor child-phase CPU. The separately run pilot is absent from supervisor measurements but present in the engine ledger. These views must never be summed as independent costs.",
        "unknown_cpu_policy": "CPU of interrupted attempts without a persisted measurement remains unknown; reported measured totals are lower bounds when such attempts exist.",
        "stopping_cap": None,
    }


def publish(state, update):
    """Retry publication without rerunning successful experiments."""
    planned = ensure_frozen_protocol(state)
    files = eligible_files()
    tracked_before = git("rev-parse", "HEAD")
    if tracked_before.returncode:
        raise ValueError("Integrity error: cannot identify the current source commit")
    update(phase="preparing_publication")
    receipt = {
        "schema_version": 1,
        "status": "FINALIZED_LOCAL",
        "finalized_at_utc": utc_now(),
        "branch": BRANCH,
        "base_branch": BASE,
        "source_head_before_finalization": tracked_before.stdout.strip(),
        "validations": state["completed_phases"],
        **planned,
        "actual_compute_cost": compute_cost_views(state),
        "artifact_sha256": {name: sha256(ROOT / name) for name in files if ROOT / name != RECEIPT},
        "publication": "The receipt and listed artifacts are committed together, then pushed without force.",
        "runtime_state_path": str(STATE.relative_to(ROOT)),
        "cloud_lifetime_guaranteed": False,
    }
    atomic_json(RECEIPT, receipt)
    files = eligible_files()
    allowed = set(files)
    staged = git("diff", "--cached", "--name-only", "-z")
    if staged.returncode:
        raise ValueError("Integrity error: cannot inspect the Git index")
    foreign = [name for name in staged.stdout.split("\0") if name and name not in allowed]
    if foreign:
        raise ValueError("Integrity error: unrelated staged files: " + ", ".join(foreign))
    attempt = 0
    committed = False
    while True:
        require_branch()
        attempt += 1
        update(phase="publishing", publication_attempt=attempt, worker_pid=None)
        if not committed:
            added = git("add", "--", *files)
            if added.returncode:
                update(last_error=safe_message(added.stderr), retry_after_seconds=30)
                time.sleep(30)
                continue
            changed = git("diff", "--cached", "--quiet")
            if changed.returncode not in (0, 1):
                update(last_error=safe_message(changed.stderr), retry_after_seconds=30)
                time.sleep(30)
                continue
            if changed.returncode == 1:
                result = git("commit", "-m", "Finalize controlled B mechanism round 3 experiments and evidence")
                if result.returncode:
                    update(last_error=safe_message(result.stderr), retry_after_seconds=30)
                    time.sleep(30)
                    continue
            committed = True
            head = git("rev-parse", "HEAD")
            update(final_commit_sha=head.stdout.strip())
        pushed = git("push", "origin", "HEAD:refs/heads/" + BRANCH)
        if pushed.returncode == 0:
            update(publication="PUSHED", published_at_utc=utc_now(), last_error=None)
            return
        update(
            publication="RETRYING_PUSH", last_error=safe_message(pushed.stderr),
            retry_after_seconds=30,
        )
        time.sleep(30)


def supervise(instance):
    initial_self_cpu = time.process_time()
    initial_child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    require_branch()
    planned = planned_counts()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    lock = (RUNTIME / "supervisor.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("An independent supervisor already holds the experiment lock.", flush=True)
        return 0
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    previous = read_json(STATE)
    if previous and previous.get("frozen_protocol_sha256") not in (None, planned["frozen_protocol_sha256"]):
        raise ValueError("Integrity error: previous runtime state belongs to a different frozen protocol")
    history = preserved_sessions(previous)
    prior_child_cpu = sum(item["measured_child_cpu_seconds"] for item in history)
    prior_supervisor_cpu = sum(item["measured_supervisor_cpu_seconds"] for item in history)
    prior_phase_cpu = sum(item["measured_phase_child_cpu_seconds"] for item in history)
    state = {
        "schema_version": 1, "instance": instance,
        "status": "RUNNING", "supervisor_pid": os.getpid(),
        "supervisor_sid": os.getsid(0), "supervisor_ppid_at_start": os.getppid(),
        "started_at_utc": utc_now(), "completed_phases": [],
        **planned,
        "session_history": history,
        "phase_attempt_costs": [], "session_phase_child_cpu_seconds": 0.0,
        "active_phase_attempt": None,
        "quality_runtime_cpu_cutoffs": None, "retry_count_limit": None,
        "integrity_error_exit_code": INTEGRITY_ERROR,
        "cloud_lifetime_guaranteed": False,
    }

    def update(**values):
        state.update(values)
        state["heartbeat_at_utc"] = utc_now()
        child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        state["session_child_cpu_seconds"] = (
            child_usage.ru_utime + child_usage.ru_stime
            - initial_child_usage.ru_utime - initial_child_usage.ru_stime
        )
        state["session_supervisor_cpu_seconds"] = time.process_time() - initial_self_cpu
        state["total_child_cpu_seconds"] = prior_child_cpu + state["session_child_cpu_seconds"]
        state["total_supervisor_cpu_seconds"] = prior_supervisor_cpu + state["session_supervisor_cpu_seconds"]
        state["total_phase_child_cpu_seconds"] = prior_phase_cpu + state["session_phase_child_cpu_seconds"]
        atomic_json(STATE, state)

    update(phase="starting", worker_pid=None)
    env = os.environ.copy()
    env.update({name: "1" for name in SINGLE_THREAD_VARIABLES})
    env["PYTHONUNBUFFERED"] = "1"
    try:
        for name, command in phases():
            attempt = 0
            while True:
                require_branch()
                ensure_frozen_protocol(state)
                attempt += 1
                update(phase=name, phase_attempt=attempt, command=command, last_error=None)
                print(json.dumps({"at": utc_now(), "phase": name, "attempt": attempt}), flush=True)
                phase_usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
                phase_wall_before = time.monotonic()
                update(active_phase_attempt={
                    "instance": instance, "phase": name, "attempt": attempt,
                    "started_at_utc": utc_now(), "cpu_seconds": None,
                    "measurement_status": "PENDING_POST_WAIT_MEASUREMENT",
                })
                worker = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL)
                update(worker_pid=worker.pid)
                while worker.poll() is None:
                    update(worker_pid=worker.pid)
                    time.sleep(5)
                code = worker.wait()
                phase_usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
                user_cpu = phase_usage_after.ru_utime - phase_usage_before.ru_utime
                system_cpu = phase_usage_after.ru_stime - phase_usage_before.ru_stime
                state["phase_attempt_costs"].append({
                    "instance": instance, "phase": name, "attempt": attempt, "exit_code": code,
                    "user_cpu_seconds": user_cpu, "system_cpu_seconds": system_cpu,
                    "cpu_seconds": user_cpu + system_cpu,
                    "wall_seconds": time.monotonic() - phase_wall_before,
                    "finished_at_utc": utc_now(),
                    "measurement": "resource.getrusage(RUSAGE_CHILDREN) difference after wait",
                })
                state["session_phase_child_cpu_seconds"] += user_cpu + system_cpu
                update(worker_pid=None, active_phase_attempt=None, last_exit_code=code)
                if code == 0:
                    state["completed_phases"].append({"phase": name, "finished_at_utc": utc_now(), "attempts": attempt})
                    update()
                    break
                if code == INTEGRITY_ERROR:
                    update(status="BLOCKED_INTEGRITY", last_error=f"{name} reported an integrity mismatch; unsafe resume was refused")
                    return INTEGRITY_ERROR
                delay = min(60, 5 * (2 ** min(attempt - 1, 4)))
                update(last_error=f"{name} exited {code}; resuming its complete artifacts", retry_after_seconds=delay)
                time.sleep(delay)
        publish(state, update)
        update(status="DONE", phase="completed", worker_pid=None, finished_at_utc=utc_now())
        return 0
    except ValueError as error:
        update(status="BLOCKED_INTEGRITY", worker_pid=None, last_error=safe_message(error))
        return INTEGRITY_ERROR
    except Exception as error:
        update(status="SUPERVISOR_ERROR", last_error=safe_message(repr(error)))
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def launch():
    require_branch()
    planned_counts()
    existing = read_json(STATE)
    if supervisor_alive(existing):
        print(json.dumps(status(), ensure_ascii=False, indent=2))
        return 0
    RUNTIME.mkdir(parents=True, exist_ok=True)
    instance = uuid.uuid4().hex
    command = [sys.executable, str(Path(__file__).resolve()), "--supervise", "--instance", instance]
    with (RUNTIME / "supervisor.log").open("ab", buffering=0) as log:
        child = subprocess.Popen(
            command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
        )
    # This bounds the launch handshake only; it neither bounds nor kills training.
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        snapshot = read_json(STATE)
        if snapshot and snapshot.get("instance") == instance:
            if snapshot.get("status") in {"BLOCKED_INTEGRITY", "SUPERVISOR_ERROR"}:
                print(json.dumps(status(), ensure_ascii=False, indent=2))
                return INTEGRITY_ERROR
            if snapshot.get("worker_pid") and snapshot.get("supervisor_sid") == child.pid:
                print(json.dumps(status(), ensure_ascii=False, indent=2))
                return 0
        if child.poll() is not None:
            print(json.dumps(status(), ensure_ascii=False, indent=2))
            return child.returncode
        time.sleep(0.2)
    print(json.dumps({"launch_handshake": "PENDING", "pid": child.pid, "state_path": str(STATE), "training_terminated": False}))
    return 0


def verify():
    require_branch()
    current = status()
    snapshot = current["state"]
    if not snapshot:
        print("No supervisor state exists.")
        return 1
    valid = snapshot.get("schema_version") == 1 and snapshot.get("quality_runtime_cpu_cutoffs") is None
    if current["supervisor_alive"]:
        valid = valid and snapshot.get("supervisor_pid") == current["observed_supervisor_sid"]
    else:
        receipt = current["finalization_receipt"]
        valid = valid and snapshot.get("status") == "DONE" and bool(receipt)
        if receipt:
            for name, expected in receipt.get("artifact_sha256", {}).items():
                path = ROOT / name
                if not path.is_file() or sha256(path) != expected:
                    valid = False
    print(json.dumps({"verified": bool(valid), **current}, ensure_ascii=False, indent=2))
    return 0 if valid else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true", help="launch an independent session and return")
    parser.add_argument("--supervise", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--instance", default="", help=argparse.SUPPRESS)
    parser.add_argument("--status", action="store_true", help="inspect durable runtime state")
    parser.add_argument("--dry-run", action="store_true", help="show the planned commands without writes")
    parser.add_argument("--resume", action="store_true", help="launch the idempotent resumable worker")
    parser.add_argument("--verify", action="store_true", help="check isolation or finalized artifact hashes")
    args = parser.parse_args()
    try:
        if args.dry_run:
            print(json.dumps({
                "branch": BRANCH, "phases": phases(),
                "publication": "commit only task files; push the study branch without force",
                "cutoffs": None, "cloud_lifetime_guaranteed": False,
                "runtime_directory": str(RUNTIME),
                "planned_counts": planned_counts(required=False),
                "protocol_required_for_launch": True,
            }, ensure_ascii=False, indent=2))
            return 0
        if args.status:
            print(json.dumps(status(), ensure_ascii=False, indent=2))
            return 0
        if args.verify:
            return verify()
        if args.supervise:
            return supervise(args.instance or uuid.uuid4().hex)
        if args.launch or args.resume:
            return launch()
        parser.error("choose --launch, --resume, --status, --verify or --dry-run")
    except ValueError as error:
        print(safe_message(error), file=sys.stderr)
        return INTEGRITY_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
