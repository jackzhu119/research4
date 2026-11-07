"""Run one threaded PySCF payload under a new, explicitly authorized budget.

The earlier pilot ledger is immutable. This runner binds the separate continuation
authorization, source checkpoint and source revision, and never resets a ledger.
CPU means actual wait4 user+system seconds, not wall seconds multiplied by threads.
"""
from __future__ import annotations

import argparse
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
SCHEMA = "research4-pyscf-continuation-budget-v2"
MAX_CPU, MAX_WALL = 43200, 28800
MAX_PHASE_CPU, MAX_PHASE_WALL = 21600, 14400
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
              "max_threads": 4, "max_memory_gib": 12}
    for name, maximum in limits.items():
        if type(value.get(name)) is not int or not 0 < value[name] <= maximum:
            raise ValueError("Invalid authorized bounded limit: " + name)
    return value


def initialize(args):
    if args.ledger.exists():
        raise FileExistsError("Never reset or overwrite an earlier budget ledger")
    required = (args.authorization, args.authorization_sha256, args.source_checkpoint,
                args.source_checkpoint_sha256, args.source_base_commit)
    if not all(required):
        raise ValueError("Initialization requires authorization and all source bindings")
    auth = validate_authorization(*required)
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
        raise ValueError("Phase CPU request must be65..21600seconds and within authorization")
    if not args.wall_seconds or not 61 <= args.wall_seconds <= auth["max_phase_wall_seconds"]:
        raise ValueError("Phase wall request must allow60s cleanup and be<=14400seconds")
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
            "address_space_limit_gib": args.memory_gib, "threads": args.threads}


def run_phase(args, ledger):
    auth = validate_ledger(ledger)
    if not args.phase or args.phase in {row["phase"] for row in ledger["phases"]} or args.log is None:
        raise ValueError("Fresh phase name and output log required")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("Explicit single-payload command required")
    limits = phase_limits(ledger, auth, args)
    args.log = args.log.resolve()
    if args.log.exists() or args.log == args.ledger.resolve() or args.log.is_relative_to(OLD_PILOT):
        raise FileExistsError("Preserve all earlier logs and pilot paths")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.update({name: str(args.threads) for name in THREAD_VARIABLES})
    def child_limits():
        resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu_soft_limit_seconds"], limits["cpu_hard_limit_seconds"]))
        resource.setrlimit(resource.RLIMIT_AS, (args.memory_gib * 1024**3,) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    row = {"phase": args.phase, "started_at_utc": utc().isoformat(), "status": "RUNNING",
           "command": command, "log": str(args.log), **limits, "environment_thread_limits": {name: env[name] for name in THREAD_VARIABLES}}
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
            save(args.ledger, ledger)
            while True:
                pid, status, usage = os.wait4(child.pid, os.WNOHANG)
                if pid:
                    child.returncode = os.waitstatus_to_exitcode(status)
                    break
                elapsed = time.monotonic() - start
                if not wall_terminated and (elapsed >= limits["wall_SIGTERM_after_seconds"] or stop_request["signal"] is not None):
                    if signal_group(child.pid, signal.SIGTERM):
                        wall_terminated = True
                        cleanup_deadline = min(limits["wall_limit_seconds"], elapsed + 60)
                        row.update(SIGTERM_requested_at_utc=utc().isoformat(), supervisor_signal=stop_request["signal"],
                                   resource_stop_reason="SUPERVISOR_INTERRUPTED" if stop_request["signal"] else "WALL_CLEANUP_WINDOW")
                        save(args.ledger, ledger)
                if elapsed >= cleanup_deadline and not hard_killed:
                    hard_killed = signal_group(child.pid, signal.SIGKILL)
                time.sleep(.1)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    cpu = usage.ru_utime + usage.ru_stime
    row.update(status="EXITED" if child.returncode == 0 and not wall_terminated else "PARTIAL_OR_FAILED",
               ended_at_utc=utc().isoformat(), actual_cpu_seconds=cpu,
               elapsed_wall_seconds=time.monotonic() - start, peak_rss_kib=usage.ru_maxrss,
               exit_code=child.returncode, wall_terminated=wall_terminated, hard_killed=hard_killed,
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
    p.add_argument("--memory-gib", type=int, default=12)
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
        if args.initialize:
            initialize(args)
            print(json.dumps({"status": "NEW_AUTHORIZED_LEDGER_INITIALIZED", "ledger": str(args.ledger)}))
            return
        row = run_phase(args, read(args.ledger))
        print(json.dumps(row), flush=True)
        raise SystemExit(0 if row["status"] == "EXITED" else 1)


if __name__ == "__main__":
    main()
