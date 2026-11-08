"""Run one quantum worker under a cumulative CPU, memory and wall budget."""
import argparse
from datetime import datetime, timezone
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


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--initialize", action="store_true")
    p.add_argument("--total-cpu-seconds", type=int, default=3300)
    p.add_argument("--phase")
    p.add_argument("--cpu-seconds", type=int)
    p.add_argument("--wall-seconds", type=int)
    p.add_argument("--memory-gib", type=int, default=12)
    p.add_argument("--log", type=Path)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    a.ledger.parent.mkdir(parents=True, exist_ok=True)
    with a.ledger.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.initialize:
            if a.ledger.exists() or not 0 < a.total_cpu_seconds <= 3300:
                raise ValueError("Fresh ledger required; quantum cap must be <=3300 CPU seconds")
            save(a.ledger, {"created_at_utc": utc(), "quantum_cpu_cap_seconds": a.total_cpu_seconds,
                           "user_total_cpu_cap_seconds": 3600, "non_quantum_reserve_seconds": 300,
                           "cpu_accounting": "sum of child user+system CPU via wait4; all workers one thread",
                           "consumed_cpu_seconds": 0.0, "phases": []})
            return
        ledger = json.loads(a.ledger.read_text())
        if any(r.get("status") == "RUNNING" for r in ledger["phases"]):
            raise RuntimeError("Unresolved prior phase: inspect evidence; do not double-launch")
        if not a.phase or a.phase in [r["phase"] for r in ledger["phases"]] or not a.log:
            raise ValueError("A fresh phase name and output log are required")
        command = a.command[1:] if a.command[:1] == ["--"] else a.command
        if not command:
            raise ValueError("Explicit worker command required")
        remaining = ledger["quantum_cpu_cap_seconds"] - ledger["consumed_cpu_seconds"]
        grant = min(a.cpu_seconds or 0, math.floor(remaining) - 2)
        if grant < 5 or not a.wall_seconds or a.wall_seconds > 7200 or not 1 <= a.memory_gib <= 12:
            raise ValueError("Insufficient remaining CPU or invalid bounded wall/memory request")
        env = os.environ.copy()
        for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
            env[name] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        a.log.parent.mkdir(parents=True, exist_ok=True)
        if a.log.exists():
            raise FileExistsError("Preserve prior logs")
        def limits():
            resource.setrlimit(resource.RLIMIT_CPU, (grant - 1, grant))
            resource.setrlimit(resource.RLIMIT_AS, (a.memory_gib * 1024**3,) * 2)
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        row = {"phase": a.phase, "started_at_utc": utc(), "status": "RUNNING", "command": command,
               "granted_cpu_seconds": grant, "wall_limit_seconds": a.wall_seconds,
               "address_space_limit_gib": a.memory_gib, "threads": 1, "log": str(a.log)}
        ledger["phases"].append(row)
        save(a.ledger, ledger)
        start = time.monotonic()
        wall_terminated = False
        with a.log.open("xb") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True, preexec_fn=limits)
            row["pid"] = child.pid
            save(a.ledger, ledger)
            while True:
                pid, status, usage = os.wait4(child.pid, os.WNOHANG)
                if pid:
                    child.returncode = os.waitstatus_to_exitcode(status)
                    break
                elapsed = time.monotonic() - start
                if elapsed > a.wall_seconds and not wall_terminated:
                    os.killpg(child.pid, signal.SIGTERM)
                    wall_terminated = True
                if elapsed > a.wall_seconds + 5:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                time.sleep(.25)
        cpu = usage.ru_utime + usage.ru_stime
        row.update(ended_at_utc=utc(), elapsed_wall_seconds=time.monotonic() - start,
                   actual_cpu_seconds=cpu, peak_rss_kib=usage.ru_maxrss,
                   exit_code=child.returncode, status="EXITED" if child.returncode == 0 else "PARTIAL_OR_FAILED",
                   wall_terminated=wall_terminated, log_sha256=hashlib.sha256(a.log.read_bytes()).hexdigest())
        ledger["consumed_cpu_seconds"] += cpu
        ledger["remaining_quantum_cpu_seconds"] = max(0.0, ledger["quantum_cpu_cap_seconds"] - ledger["consumed_cpu_seconds"])
        save(a.ledger, ledger)
        print(json.dumps(row), flush=True)
        raise SystemExit(0 if child.returncode == 0 else 1)


if __name__ == "__main__":
    main()
